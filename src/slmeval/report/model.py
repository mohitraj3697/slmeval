"""The scorecard as data, built from stored rows and nothing else.

Pure functions from rows to a value object, with no Rich and no printing. Two
things follow from that, and both are the reason the module exists.

``slmeval report --run-id X`` reproduces the exact table from a stored run with
no re-execution. If anything in the headline numbers were computed from live
objects during the run, that promise would be a lie the first time someone
asked to see yesterday's results.

And every view is snapshot-testable. The renderers take one of these and return
a renderable; a column that changes shape turns a test red instead of turning
up in a screenshot.

The aggregation rules that are not obvious:

- **Rates are weighted by their own denominator.** A candidate's schema
  adherence is ``sum(valid) / sum(calls)`` across executions, not the mean of
  per-execution rates: an execution that made one call would otherwise count
  as much as one that made twenty.
- **Unjudgeable executions leave the denominator.** ``harness_error`` and
  ``uncovered`` are counted and surfaced as warnings, never scored.
- **Latency is omitted, not estimated, in throughput mode.** The run records
  which scheduling mode produced it, and a p95 from concurrent requests to one
  GPU describes our scheduler. ``None`` prints as a dash and a footnote.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from statistics import median
from typing import Any, Final

from slmeval.eval.cost import PRICES_AS_OF
from slmeval.eval.stats import Aggregate, Consistency, aggregate, wilson_interval
from slmeval.storage.db import RunStore
from slmeval.storage.models import (
    NOT_THE_MODELS_FAULT,
    BlameRow,
    CandidateRecord,
    CaseRollupRow,
    ExecutionRecord,
    MetricSample,
    NodeSummaryRow,
    Outcome,
    RunRecord,
)

LATENCY_DISCLAIMER: Final = (
    "latency not measured in this mode - requests ran concurrently, so a p95 "
    "would describe the scheduler rather than the model (re-run with "
    "--serial-latency)"
)


@dataclass(frozen=True, slots=True)
class Incident:
    """A forbidden tool call. Printed above the table, never in a column."""

    candidate_id: str
    case_id: str
    tools: tuple[str, ...]
    exec_id: str = ""
    executions: int = 1
    """How many executions made this same call.

    Aggregated rather than one line per execution: three repeats of one case
    calling ``issue_refund`` is one finding seen three times, and printing it
    three times above the table pushes the other warnings off the screen -
    which is the opposite of what putting it above the table was for.
    """

    def describe(self) -> str:
        where = f"{self.case_id} ({self.candidate_id})"
        times = f", {self.executions}x" if self.executions > 1 else ""
        return f"FORBIDDEN tool call: {', '.join(self.tools)} in {where}{times}"


@dataclass(frozen=True, slots=True)
class CandidateReport:
    """One row of the scorecard."""

    candidate_id: str
    role: str = "candidate"
    provider: str = ""
    model: str = ""
    ordinal: int = 0

    stats: Aggregate = field(default_factory=Aggregate)
    rollups: dict[str, Consistency] = field(default_factory=dict)
    """Per-case consistency, keyed by ``case_id``.

    Keyed rather than ordered: the per-case table is read next to the
    scorecard to find *which* cases a candidate failed, and matching rows by
    position would mis-attribute every one of them the moment two candidates
    ran different case subsets."""
    outcomes: dict[Outcome, int] = field(default_factory=dict)

    schema_calls: int = 0
    schema_valid: float = 0.0
    schema_repaired: float = 0.0
    tool_f1: float | None = None
    tool_precision: float | None = None
    tool_recall: float | None = None
    trajectory_distance: float | None = None
    redundancy: float | None = None
    mean_steps: float | None = None
    loops: int = 0

    durations_ms: tuple[float, ...] = ()
    cost_total: float | None = None
    cost_cases: int = 0
    tokens_estimated: bool = False
    unpriced_models: tuple[str, ...] = ()

    unsafe_actions: int = 0
    incidents: tuple[Incident, ...] = ()
    uncovered_cases: tuple[str, ...] = ()
    uncovered_executions: int = 0
    """Executions, not cases. Three repeats of one uncovered case is three
    executions missing from ``n`` and one gap in the recordings, and the
    warning has to be able to say both without conflating them."""
    harness_errors: int = 0

    faults_met: int = 0
    recovered: int = 0
    structured_modes: dict[str, int] = field(default_factory=dict)
    json_stages: dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------ rates

    @property
    def baseline(self) -> bool:
        return self.role == "baseline"

    @property
    def schema_valid_rate(self) -> float | None:
        if not self.schema_calls:
            return None
        return self.schema_valid / self.schema_calls

    @property
    def schema_repaired_rate(self) -> float | None:
        if not self.schema_calls:
            return None
        return self.schema_repaired / self.schema_calls

    @property
    def p50_ms(self) -> float | None:
        return median(self.durations_ms) if self.durations_ms else None

    @property
    def p95_ms(self) -> float | None:
        """The 95th percentile by nearest rank.

        Nearest rank, not interpolation: with ``n = 30`` an interpolated p95 is
        a number that no execution actually took, and a benchmark should report
        an observation rather than a construction.
        """
        if not self.durations_ms:
            return None
        ordered = sorted(self.durations_ms)
        index = max(0, min(len(ordered) - 1, round(0.95 * len(ordered)) - 1))
        return ordered[index]

    @property
    def cost_per_case(self) -> float | None:
        if self.cost_total is None or not self.cost_cases:
            return None
        return self.cost_total / self.cost_cases

    @property
    def recovery_rate(self) -> float | None:
        """Passes among executions that actually met an injected fault.

        ``None`` with no faults met - not 100%. A candidate that never
        encountered a fault has not demonstrated recovery, and a full marks
        column for an untested property would be the most flattering possible
        reading.
        """
        if not self.faults_met:
            return None
        return self.recovered / self.faults_met

    @property
    def pass_hat_k(self) -> float | None:
        return self.stats.pass_hat_k

    @property
    def wilson(self) -> tuple[float, float] | None:
        if self.stats.interval is None:
            return None
        return (self.stats.interval.lo, self.stats.interval.hi)

    def warnings(self) -> tuple[str, ...]:
        """The lines that go above the table for this candidate."""
        out: list[str] = []
        mode = _dominant(self.structured_modes)
        if mode and mode not in {"tools", "json_schema"} and self.schema_calls:
            share = self.structured_modes.get(mode, 0) / max(1, self.schema_calls)
            out.append(
                f"{self.candidate_id} needed {mode} on {share:.0%} of structured calls"
            )
        if self.schema_repaired_rate:
            out.append(
                f"{self.candidate_id} needed prompt-repair or a fallback rung on "
                f"{self.schema_repaired_rate:.0%} of structured calls"
            )
        if self.loops:
            out.append(
                f"{self.candidate_id} looped in {self.loops} execution(s)"
            )
        if self.unpriced_models:
            out.append(
                f"no price for {', '.join(self.unpriced_models)} - cost left blank "
                "rather than counted as zero (set it in prices.toml)"
            )
        return tuple(out)


@dataclass(frozen=True, slots=True)
class Scorecard:
    """A whole run, ready to render in any format."""

    run: RunRecord
    rows: tuple[CandidateReport, ...] = ()
    nodes: tuple[NodeSummaryRow, ...] = ()
    blame: tuple[BlameRow, ...] = ()
    cases: tuple[str, ...] = ()
    repeats: int = 1
    prices_as_of: str = PRICES_AS_OF

    @property
    def baseline(self) -> CandidateReport | None:
        return next((r for r in self.rows if r.baseline), None)

    @property
    def candidates(self) -> tuple[CandidateReport, ...]:
        return tuple(r for r in self.rows if not r.baseline)

    @property
    def latency_reportable(self) -> bool:
        """Whether a latency column may be printed at all."""
        return self.run.scheduling == "latency"

    @property
    def executions(self) -> int:
        return sum(r.stats.executions + r.stats.excluded for r in self.rows)

    @property
    def harness_errors(self) -> int:
        return sum(r.harness_errors for r in self.rows)

    @property
    def incidents(self) -> tuple[Incident, ...]:
        return tuple(i for r in self.rows for i in r.incidents)

    @property
    def uncovered(self) -> tuple[tuple[str, str], ...]:
        """``(candidate_id, case_id)`` pairs the cassettes did not cover."""
        return tuple(
            (r.candidate_id, case) for r in self.rows for case in r.uncovered_cases
        )

    def delta_cost(self, row: CandidateReport) -> float | None:
        """Fractional change in cost per case against the baseline."""
        base = self.baseline
        if base is None or row is base:
            return None
        reference = base.cost_per_case
        mine = row.cost_per_case
        if reference is None or reference == 0 or mine is None:
            # A baseline that cost nothing has no ratio to express - a scripted
            # or local baseline would otherwise divide by zero or report an
            # infinite saving.
            return None
        return (mine - reference) / reference

    def delta_pass(self, row: CandidateReport) -> float | None:
        base = self.baseline
        if base is None or row is base:
            return None
        if base.pass_hat_k is None or row.pass_hat_k is None:
            return None
        return row.pass_hat_k - base.pass_hat_k

    def warnings(self) -> tuple[str, ...]:
        """Everything that must appear above the table, worst first.

        Order matters: a forbidden tool call must never be below a note about
        token estimates, and a harness error must be visible before anyone
        reads a single rate - it is the one warning that calls the whole run
        into question.
        """
        out: list[str] = []
        for incident in self.incidents:
            out.append(incident.describe())
        if self.harness_errors:
            out.append(
                f"{self.harness_errors} execution(s) failed inside slmeval itself - "
                "excluded from every rate below, and a bug in the harness, not in "
                "any model"
            )
        if self.uncovered:
            cases = ", ".join(sorted({case for _, case in self.uncovered}))
            executions = sum(r.uncovered_executions for r in self.rows)
            out.append(
                f"{executions} execution(s) across {len(self.uncovered)} "
                f"candidate/case pair(s) explored paths the cassettes do not cover "
                f"({cases}) - excluded from pass rates rather than counted against "
                "the candidate; re-run `slmeval record --fill-gaps`"
            )
        for row in self.rows:
            out.extend(row.warnings())
        if any(r.tokens_estimated for r in self.rows):
            out.append(
                "some token counts were estimated, not reported by the provider - "
                "figures built from them are marked ~"
            )
        if not self.latency_reportable:
            out.append(LATENCY_DISCLAIMER)
        return tuple(out)


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #


def build_scorecard(
    store: RunStore, run_id: str, *, prices_as_of: str = PRICES_AS_OF
) -> Scorecard:
    """Assemble a scorecard from the store. No re-execution, ever."""
    run = store.require_run(run_id)
    candidates = store.candidates(run_id)
    executions = store.executions(run_id)
    rollups = store.rollups(run_id)
    samples = store.metrics(run_id)

    return Scorecard(
        run=run,
        rows=tuple(
            candidate_report(
                candidate,
                executions=[e for e in executions if e.candidate_id == candidate.candidate_id],
                rollups=[r for r in rollups if r.candidate_id == candidate.candidate_id],
                samples=[s for s in samples if s.candidate_id == candidate.candidate_id],
                repeats=run.runs_per_case,
            )
            for candidate in sorted(candidates, key=lambda c: (c.role != "baseline", c.ordinal))
        ),
        nodes=tuple(store.node_summary(run_id)),
        blame=tuple(store.blame(run_id)),
        cases=tuple(dict.fromkeys(e.case_id for e in executions)),
        repeats=run.runs_per_case,
        prices_as_of=prices_as_of,
    )


def candidate_report(
    candidate: CandidateRecord,
    *,
    executions: Sequence[ExecutionRecord],
    rollups: Sequence[CaseRollupRow],
    samples: Sequence[MetricSample],
    repeats: int = 1,
) -> CandidateReport:
    """One scorecard row, from that candidate's rows alone."""
    by_metric = _group(samples)
    judgeable = [e for e in executions if e.outcome_class not in NOT_THE_MODELS_FAULT]

    schema_calls = _sum(by_metric.get("schema_calls", ()))
    calls = by_metric.get("schema_calls", ())
    valid_rate = _weighted(by_metric.get("schema_valid_rate", ()), calls)
    repaired_rate = _weighted(by_metric.get("schema_repaired_rate", ()), calls)

    incidents = _incidents(candidate.candidate_id, by_metric.get("unsafe_action", ()))

    return CandidateReport(
        candidate_id=candidate.candidate_id,
        role=candidate.role,
        provider=candidate.provider,
        model=candidate.model,
        ordinal=candidate.ordinal,
        stats=aggregate([_as_consistency(r, repeats) for r in rollups]),
        rollups={r.case_id: _as_consistency(r, repeats) for r in rollups},
        outcomes=_outcomes(executions),
        schema_calls=int(schema_calls),
        schema_valid=(valid_rate or 0.0) * schema_calls,
        schema_repaired=(repaired_rate or 0.0) * schema_calls,
        tool_f1=_mean(by_metric.get("tool_f1", ())),
        tool_precision=_mean(by_metric.get("tool_precision", ())),
        tool_recall=_mean(by_metric.get("tool_recall", ())),
        trajectory_distance=_mean(by_metric.get("trajectory_distance", ())),
        redundancy=_mean(by_metric.get("tool_redundancy", ())),
        mean_steps=_mean(by_metric.get("steps", ())),
        loops=len(by_metric.get("loop_detected", ())),
        durations_ms=tuple(e.duration_ms for e in judgeable),
        cost_total=_cost_total(judgeable),
        cost_cases=sum(1 for e in judgeable if e.cost_usd is not None),
        tokens_estimated=any(e.tokens_estimated for e in executions),
        unpriced_models=() if _cost_total(judgeable) is not None else (candidate.model,),
        unsafe_actions=int(_sum(by_metric.get("unsafe_action", ()))),
        incidents=incidents,
        uncovered_cases=tuple(
            dict.fromkeys(
                e.case_id for e in executions if e.outcome_class is Outcome.UNCOVERED
            )
        ),
        uncovered_executions=sum(
            1 for e in executions if e.outcome_class is Outcome.UNCOVERED
        ),
        harness_errors=sum(
            1 for e in executions if e.outcome_class is Outcome.HARNESS_ERROR
        ),
        faults_met=len(by_metric.get("recovered", ())),
        recovered=int(_sum(by_metric.get("recovered", ()))),
        structured_modes=_histogram(by_metric.get("structured_mode_used", ())),
        json_stages=_histogram(by_metric.get("json_extraction_stage", ())),
    )


def _incidents(
    candidate_id: str, samples: Sequence[MetricSample]
) -> tuple[Incident, ...]:
    """One incident per (case, tools), carrying how many executions hit it."""
    grouped: dict[tuple[str, tuple[str, ...]], Incident] = {}
    for sample in samples:
        tools = tuple(t for t in (sample.value_text or "").split(", ") if t)
        key = (sample.case_id, tools)
        found = grouped.get(key)
        if found is None:
            grouped[key] = Incident(
                candidate_id=candidate_id,
                case_id=sample.case_id,
                tools=tools,
                exec_id=sample.exec_id,
            )
        else:
            grouped[key] = replace(found, executions=found.executions + 1)
    return tuple(grouped.values())


def _as_consistency(row: CaseRollupRow, repeats: int) -> Consistency:
    """Rehydrate a stored rollup without recomputing it.

    The stored numbers win. A report that recomputed ``pass^k`` could disagree
    with the run that printed it - and after an estimator changes, it would
    silently restate last month's published results.
    """
    interval = (
        None
        if row.wilson_lo is None or row.wilson_hi is None
        else wilson_interval(row.n, row.c)
    )
    return Consistency(
        n=row.n,
        c=row.c,
        k=row.n or repeats,
        pass_k=row.pass_k,
        pass_hat_k=row.pass_hat_k,
        interval=interval,
    )


def _group(samples: Sequence[MetricSample]) -> dict[str, tuple[MetricSample, ...]]:
    out: dict[str, list[MetricSample]] = {}
    for sample in samples:
        out.setdefault(sample.metric, []).append(sample)
    return {k: tuple(v) for k, v in out.items()}


def _sum(samples: Sequence[MetricSample]) -> float:
    return sum(s.value_num or 0.0 for s in samples)


def _mean(samples: Sequence[MetricSample]) -> float | None:
    values = [s.value_num for s in samples if s.value_num is not None]
    return sum(values) / len(values) if values else None


def _weighted(
    rates: Sequence[MetricSample], weights: Sequence[MetricSample]
) -> float | None:
    """A rate weighted by its own denominator, matched up per execution.

    An execution that made one structured call must not count as much as one
    that made twenty - that is the difference between an average of rates and a
    rate.
    """
    by_exec = {s.exec_id: (s.value_num or 0.0) for s in weights}
    numerator = 0.0
    denominator = 0.0
    for rate in rates:
        if rate.value_num is None:
            continue
        weight = by_exec.get(rate.exec_id, 1.0)
        numerator += rate.value_num * weight
        denominator += weight
    return numerator / denominator if denominator else None


def _histogram(samples: Sequence[MetricSample]) -> dict[str, int]:
    out: dict[str, int] = {}
    for sample in samples:
        if sample.value_text:
            out[sample.value_text] = out.get(sample.value_text, 0) + 1
    return out


def _cost_total(executions: Sequence[ExecutionRecord]) -> float | None:
    """Total measured cost, or ``None`` if any execution was unpriced.

    ``None``, never a partial sum. A total that quietly omitted the expensive
    executions would read as a cheap run, and "cheap" is the most persuasive
    cell in the table.
    """
    if not executions:
        return 0.0
    if any(e.cost_usd is None for e in executions):
        return None
    return sum(e.cost_usd or 0.0 for e in executions)


def _outcomes(executions: Sequence[ExecutionRecord]) -> dict[Outcome, int]:
    out: dict[Outcome, int] = {}
    for execution in executions:
        out[execution.outcome_class] = out.get(execution.outcome_class, 0) + 1
    return out


def _dominant(histogram: Mapping[str, int]) -> str:
    return max(sorted(histogram), key=lambda k: (histogram[k], k)) if histogram else ""


def as_dict(card: Scorecard) -> dict[str, Any]:
    """The stable, versioned JSON shape. See :mod:`slmeval.report.export`."""
    from slmeval.report.export import to_payload

    return to_payload(card)


__all__ = [
    "LATENCY_DISCLAIMER",
    "CandidateReport",
    "Incident",
    "Scorecard",
    "as_dict",
    "build_scorecard",
    "candidate_report",
]
