"""The matrix runner.

This is what turns "I can run an agent once" into "I ran 1,500 trajectories and
lost none of them". Everything here is shaped by one assumption: the run takes
twenty minutes, something will go wrong in the middle of it, and the data that
existed before that moment has to survive.

**The unit of work is one execution** - ``(run_id, candidate_id, case_id,
repeat_idx)`` - and it is persisted the moment it finishes. Never 1,500 results
held in memory and written at the end, because a ``KeyboardInterrupt`` at 95%
would take all of them. The exec id is derived from the tuple, so a resumed run
recomputes the same id and a double-write replaces a row instead of doubling a
candidate's ``n``.

**Resumability is not a nicety.** A benchmark on local hardware takes tens of
minutes; one crash without resume makes the tool unusable in practice.
``--resume`` subtracts what the store already holds in a terminal state from
the planned matrix, and refuses to resume a run whose config or dataset has
changed underneath it.

**Two scheduling modes, and the distinction is the product's credibility.**
Throughput mode runs high concurrency and its latency numbers are not
reportable - requests queue inside the provider and a local model reads several
times slower than it is. Latency mode runs one request at a time per endpoint
after a warm-up call. The run records which mode it was, and the report refuses
to print a p95 column for a throughput run rather than printing a misleading
one.

**Per-endpoint semaphores, not one global.** Keyed on ``(provider, base_url)``,
because the thing being protected is the server: two candidates pointed at the
same Ollama share one GPU, and giving each its own limit would put twice the
load on it as either asked for.

**Crash isolation is absolute.** Every execution's body is wrapped in
``except BaseException``, classified and recorded, and the task returns
normally. One malformed case must never abort the matrix. The single exception
is cancellation, which is re-raised untouched: a ``Ctrl-C`` has to stop the run,
and the units it interrupted must be left with no row at all so ``--resume``
sees them as work rather than as results.

Nothing in here prints. Progress goes to an :class:`Observer`, so the runner is
testable without a terminal and the Rich progress bar is replaceable.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from slmeval import __version__
from slmeval.config.models import EvalConfig, ModelCfg
from slmeval.core.adapter import Target, execute_case
from slmeval.core.context import RunContext, benchmark_running, no_context
from slmeval.core.outcome import ABORTING, Classification, classify
from slmeval.core.router import ModelRouter
from slmeval.core.verdict import Evaluator, Verdict, carry_classification
from slmeval.dataset.loader import Dataset
from slmeval.dataset.models import Case
from slmeval.errors import ConfigError
from slmeval.eval.cost import Priced, PriceTable
from slmeval.eval.cost import total as total_cost
from slmeval.providers.registry import is_local, resolve_spec
from slmeval.sandbox.faults import FaultSchedule, fault_schedule
from slmeval.sandbox.store import CassetteStore
from slmeval.storage import codec, ingest
from slmeval.storage.db import RunStore, make_exec_id
from slmeval.storage.models import (
    CandidateRecord,
    ExecStatus,
    Outcome,
    RunRecord,
    RunStatus,
)
from slmeval.storage.provenance import Provenance, divergences, new_run_id

THROUGHPUT: Final = "throughput"
LATENCY: Final = "latency"

DEFAULT_LOCAL_CONCURRENCY: Final = 2
"""Throughput mode's cap on a local endpoint.

The provider layer defaults a local model to 1, which is right for measuring
latency and wasteful for measuring correctness - a 90-execution replay matrix
would run strictly serially. Two keeps a laptop GPU busy without queueing
deeply enough to distort the wall clock, and throughput-mode latency is not
reportable anyway, which is the whole reason this is allowed to differ.
"""

FAIL_FAST_STREAK: Final = 3
"""Consecutive non-passing executions before ``--fail-fast`` gives up.

Not one. Benchmarks are *expected* to contain failures - that is the output -
so aborting on the first would make the flag useless. Three in a row from one
candidate is the signal it exists for: a model that is not merely worse but
broken, where the remaining nineteen minutes buy nothing.
"""

WARMUP_PROMPT: Final = "Reply with the single word: ok"


# --------------------------------------------------------------------------- #
# The matrix
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True, order=True)
class WorkUnit:
    """One cell of the matrix: the unit of work, of persistence and of resume."""

    candidate_id: str
    case_id: str
    repeat_idx: int

    @property
    def key(self) -> tuple[str, str, int]:
        """What :meth:`RunStore.completed_units` returns, for set arithmetic."""
        return (self.candidate_id, self.case_id, self.repeat_idx)

    def exec_id(self, run_id: str) -> str:
        return make_exec_id(run_id, self.candidate_id, self.case_id, self.repeat_idx)

    def __str__(self) -> str:
        return f"{self.candidate_id}/{self.case_id}#{self.repeat_idx}"


@dataclass(frozen=True, slots=True)
class Matrix:
    """The planned work, in execution order, and what resume took out."""

    units: tuple[WorkUnit, ...]
    resumed: tuple[WorkUnit, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    case_ids: tuple[str, ...] = ()
    repeats: int = 1
    baseline_id: str | None = None

    def __len__(self) -> int:
        return len(self.units)

    def __iter__(self) -> Iterator[WorkUnit]:
        return iter(self.units)

    @property
    def planned(self) -> int:
        """Every cell, including the ones resume skipped. The denominator the
        progress bar has to use, or a resumed run reports 40/40 and looks like
        it did the whole matrix."""
        return len(self.units) + len(self.resumed)

    def for_candidate(self, candidate_id: str) -> tuple[WorkUnit, ...]:
        return tuple(u for u in self.units if u.candidate_id == candidate_id)

    def phases(self) -> tuple[tuple[WorkUnit, ...], tuple[WorkUnit, ...]]:
        """Baseline work, then everything else.

        Two phases rather than one ordered queue because the baseline does more
        than go first: in ``record`` and ``auto`` modes it is what *populates*
        the cassettes every candidate then replays. A candidate starting before
        the baseline has recorded its case would miss on a path that is covered
        a second later, and the miss would be reported as the model's doing.
        """
        if self.baseline_id is None:
            return (self.units, ())
        first = tuple(u for u in self.units if u.candidate_id == self.baseline_id)
        return first, tuple(u for u in self.units if u.candidate_id != self.baseline_id)


def plan_matrix(
    *,
    candidate_ids: Sequence[str],
    case_ids: Sequence[str],
    repeats: int,
    baseline_id: str | None = None,
    completed: Iterable[tuple[str, str, int]] = (),
) -> Matrix:
    """Build the cross product in the order it should run.

    The order is not cosmetic. The baseline goes first, whole: it populates the
    cassettes and gives every later comparison its reference. After that,
    candidates are **interleaved case by case** rather than run one candidate at
    a time, so a matrix abandoned at 30% still has partial results for *every*
    candidate - three rows with ``n=3`` beats one complete row and two empty
    ones, because the first is a comparison and the second is not.

    ``completed`` is subtracted, which is all ``--resume`` is.
    """
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}")
    done = {tuple(k) for k in completed}

    ordered = list(dict.fromkeys(candidate_ids))
    if baseline_id is not None and baseline_id not in ordered:
        raise ValueError(f"baseline {baseline_id!r} is not among the candidates")
    others = [c for c in ordered if c != baseline_id]

    planned: list[WorkUnit] = []
    if baseline_id is not None:
        planned += [
            WorkUnit(baseline_id, case_id, r)
            for case_id in case_ids
            for r in range(repeats)
        ]
    planned += [
        WorkUnit(candidate_id, case_id, r)
        for case_id in case_ids
        for r in range(repeats)
        for candidate_id in others
    ]

    todo = tuple(u for u in planned if u.key not in done)
    skipped = tuple(u for u in planned if u.key in done)
    return Matrix(
        units=todo,
        resumed=skipped,
        candidate_ids=tuple(ordered),
        case_ids=tuple(dict.fromkeys(case_ids)),
        repeats=repeats,
        baseline_id=baseline_id,
    )


# --------------------------------------------------------------------------- #
# Options, limits, budget
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RunOptions:
    """Everything about *how* a matrix runs, separate from what it measures."""

    repeats: int = 1
    tool_mode: str = "replay"
    fault_rate: float = 0.0
    seed: int = 42
    timeout_s: float = 120.0
    max_steps: int = 25

    concurrency: int = 4
    """Global ceiling on executions in flight. Per-endpoint semaphores narrow
    it further; neither one can widen the other."""

    local_concurrency: int = DEFAULT_LOCAL_CONCURRENCY
    serial_latency: bool = False
    warmup: bool = True
    """Make one throwaway call per local model before timing it. Weights coming
    off disk on the first call can be 60 seconds, and charging that to the
    first case would be a lie about the model."""

    max_spend_usd: float | None = None
    fail_fast: bool = False
    fail_fast_streak: int = FAIL_FAST_STREAK
    case_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    evaluate: bool = True
    gpu_hourly_usd: float | None = None

    @property
    def scheduling(self) -> str:
        return LATENCY if self.serial_latency else THROUGHPUT

    @property
    def latency_reportable(self) -> bool:
        """Whether this run's timings mean anything.

        The one question the report asks of the scheduling mode, and the reason
        it is stored rather than inferred later.
        """
        return self.serial_latency

    @classmethod
    def from_config(cls, config: EvalConfig, **overrides: Any) -> RunOptions:
        """Defaults from ``eval.toml``, overridden by command-line flags.

        ``None`` overrides are dropped rather than applied, so an unset Typer
        option cannot quietly reset a configured value to nothing.
        """
        bench = config.benchmark
        base = cls(
            repeats=bench.runs_per_case,
            tool_mode=bench.tool_mode,
            fault_rate=bench.fault_injection_rate,
            seed=bench.seed,
            timeout_s=bench.timeout_s,
            max_steps=bench.max_steps,
            concurrency=bench.concurrency,
        )
        return replace(base, **{k: v for k, v in overrides.items() if v is not None})


@dataclass(frozen=True, slots=True)
class Endpoint:
    """One model server, as the limiter sees it."""

    provider: str
    base_url: str
    local: bool

    @property
    def key(self) -> str:
        return f"{self.provider}|{self.base_url}"

    def __str__(self) -> str:
        return self.base_url or self.provider


class ProviderLimits:
    """One semaphore per model *endpoint*.

    Per endpoint, not per candidate and not per provider name: the resource
    being protected is the server. Two candidates on the same Ollama share a
    GPU, and a limit each would put twice the load on it as either asked for.
    Conversely a local llama.cpp and a cloud gateway are both
    ``openai_compatible`` and must not throttle each other.
    """

    def __init__(self, limits: Mapping[str, int], endpoints: Mapping[str, Endpoint]) -> None:
        self._limits = dict(limits)
        self._endpoints = dict(endpoints)
        self._semaphores: dict[str, asyncio.Semaphore] = {}

    @classmethod
    def for_run(cls, models: Sequence[ModelCfg], options: RunOptions) -> ProviderLimits:
        limits: dict[str, int] = {}
        endpoints: dict[str, Endpoint] = {}
        for cfg in models:
            spec = resolve_spec(cfg, timeout_s=options.timeout_s)
            endpoint = Endpoint(
                provider=spec.provider,
                base_url=spec.base_url or "",
                local=is_local(spec.provider, spec.base_url),
            )
            allowed = _endpoint_limit(spec.max_concurrency, endpoint, options)
            # The tightest requirement wins. A user who pins one candidate to
            # a lower concurrency means it for the server, not for that row.
            limits[endpoint.key] = min(limits.get(endpoint.key, allowed), allowed)
            endpoints[endpoint.key] = endpoint
        return cls(limits, endpoints)

    def limit_for(self, cfg: ModelCfg, *, timeout_s: float | None = None) -> int:
        spec = resolve_spec(cfg, timeout_s=timeout_s)
        return self._limits.get(f"{spec.provider}|{spec.base_url or ''}", 1)

    def key_for(self, cfg: ModelCfg, *, timeout_s: float | None = None) -> str:
        spec = resolve_spec(cfg, timeout_s=timeout_s)
        return f"{spec.provider}|{spec.base_url or ''}"

    @property
    def limits(self) -> Mapping[str, int]:
        return dict(self._limits)

    def describe(self) -> str:
        return ", ".join(
            f"{self._endpoints[k]}={v}" for k, v in sorted(self._limits.items())
        )

    @contextlib.asynccontextmanager
    async def slot(self, key: str) -> Any:
        """Hold a slot on one endpoint for the duration of an execution.

        Created lazily and on the running loop, never in ``__init__``: a
        semaphore built before the loop exists binds to the wrong one, which
        shows up as a deadlock on the first concurrent run and nowhere in a
        unit test.
        """
        semaphore = self._semaphores.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(max(1, self._limits.get(key, 1)))
            self._semaphores[key] = semaphore
        async with semaphore:
            yield


def _endpoint_limit(spec_limit: int, endpoint: Endpoint, options: RunOptions) -> int:
    """How many requests this endpoint may have in flight.

    Latency mode pins every endpoint to one, which is the entire point of the
    mode: a p95 taken from four concurrent requests to one GPU describes our
    scheduler, not the model.
    """
    if options.serial_latency:
        return 1
    if endpoint.local:
        # Deliberately above the provider layer's default of 1, and safe only
        # because throughput-mode latency is marked unreportable.
        return max(1, min(options.local_concurrency, options.concurrency))
    return max(1, min(spec_limit, options.concurrency))


class SpendBudget:
    """A ceiling on what a run may cost, checked as results land.

    After each execution rather than before: the cost of a call is not knowable
    until the tokens come back. So the guarantee is "stops soon after
    crossing", not "never exceeds" - and it is written that way in the message,
    because a budget that claimed to be a hard cap would be trusted as one.

    A tool that can silently spend $400 on a judge model gets uninstalled, so
    the default is no ceiling and the flag is cheap to set.
    """

    def __init__(self, limit_usd: float | None) -> None:
        self.limit = limit_usd
        self._spent = 0.0
        self._unpriced: list[str] = []

    @property
    def spent(self) -> float:
        return self._spent

    @property
    def unpriced(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self._unpriced))

    def add(self, priced: Priced) -> None:
        if priced.usd is not None:
            self._spent += priced.usd
        self._unpriced.extend(priced.unpriced)

    @property
    def exceeded(self) -> bool:
        return self.limit is not None and self._spent >= self.limit

    def reason(self) -> str:
        return (
            f"spend budget reached: ${self._spent:.4f} of ${self.limit:.2f} "
            "(checked after each execution, so the final figure can exceed the limit "
            "by at most one execution)"
        )


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class UnitResult:
    """One finished execution, as the progress UI and the summary see it.

    The row is already in SQLite by the time this exists. This is a report of
    what was written, not the only copy of it.
    """

    unit: WorkUnit
    exec_id: str
    status: ExecStatus
    outcome: Outcome
    duration_ms: float
    steps: int = 0
    cost: Priced = field(default_factory=Priced)
    error_class: str | None = None
    error_message: str | None = None
    warnings: tuple[str, ...] = ()
    verdict: Verdict | None = None

    @property
    def candidate_id(self) -> str:
        return self.unit.candidate_id

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASSED

    @property
    def harness_error(self) -> bool:
        return self.outcome is Outcome.HARNESS_ERROR

    @property
    def judgeable(self) -> bool:
        """Whether this may enter a pass rate. See
        :data:`~slmeval.storage.models.NOT_THE_MODELS_FAULT`."""
        return self.outcome not in {Outcome.HARNESS_ERROR, Outcome.UNCOVERED}


@dataclass(frozen=True, slots=True)
class RunSummary:
    """What a finished (or abandoned) run amounts to."""

    run_id: str
    status: RunStatus
    planned: int
    resumed: int
    results: tuple[UnitResult, ...] = ()
    spend_usd: float = 0.0
    unpriced_models: tuple[str, ...] = ()
    scheduling: str = THROUGHPUT
    aborted_reason: str | None = None
    started_at: str = ""
    finished_at: str = ""

    @property
    def executed(self) -> int:
        return len(self.results)

    @property
    def outcomes(self) -> dict[Outcome, int]:
        counts: dict[Outcome, int] = {}
        for r in self.results:
            counts[r.outcome] = counts.get(r.outcome, 0) + 1
        return counts

    @property
    def harness_errors(self) -> int:
        """Counted, printed, and never folded into a candidate's score."""
        return sum(1 for r in self.results if r.harness_error)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def complete(self) -> bool:
        return self.executed + self.resumed >= self.planned and self.aborted_reason is None


# --------------------------------------------------------------------------- #
# Progress
# --------------------------------------------------------------------------- #


class Observer(Protocol):
    """Where progress goes.

    The runner never prints. A twenty-minute run has to look alive, but a
    ``rich.Progress`` wired into the orchestration loop would make the runner
    untestable without a terminal and unusable from a library.
    """

    def on_plan(self, plan: RunPlan) -> None: ...
    def on_unit_start(self, unit: WorkUnit) -> None: ...
    def on_unit_done(self, result: UnitResult) -> None: ...
    def on_note(self, message: str) -> None: ...
    def on_finish(self, summary: RunSummary) -> None: ...


class NullObserver:
    """Does nothing, for library use and for tests."""

    def on_plan(self, plan: RunPlan) -> None:
        return None

    def on_unit_start(self, unit: WorkUnit) -> None:
        return None

    def on_unit_done(self, result: UnitResult) -> None:
        return None

    def on_note(self, message: str) -> None:
        return None

    def on_finish(self, summary: RunSummary) -> None:
        return None


@dataclass(frozen=True, slots=True)
class RunPlan:
    """The run as it is about to be attempted. Handed to the observer so a
    progress bar can size itself before the first result lands."""

    run_id: str
    name: str
    matrix: Matrix
    tool_mode: str
    scheduling: str
    fault_rate: float
    seed: int
    limits: str = ""

    @property
    def total(self) -> int:
        return self.matrix.planned


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #


class MatrixRunner:
    """Executes ``candidates x cases x repeats`` and persists every cell."""

    def __init__(
        self,
        *,
        config: EvalConfig,
        dataset: Dataset,
        target: Target,
        store: RunStore,
        options: RunOptions,
        cassettes: CassetteStore | None = None,
        observer: Observer | None = None,
        evaluator: Evaluator | None = None,
        prices: PriceTable | None = None,
        root: Path | None = None,
        name: str | None = None,
    ) -> None:
        self.config = config
        self.dataset = dataset
        self.target = target
        self.store = store
        self.options = options
        self.cassettes = cassettes
        self.observer: Observer = observer or NullObserver()
        self.evaluator: Evaluator = evaluator or carry_classification
        self.prices = prices or PriceTable.load(root, gpu_hourly_usd=options.gpu_hourly_usd)
        self.root = root
        self.name = name or config.benchmark.name

        self._models: dict[str, ModelCfg] = {m.id: m for m in config.all_models}
        self._cases: dict[str, Case] = {}
        self._routers: dict[str, ModelRouter] = {}
        self._faults: dict[str, FaultSchedule] = {}
        self._budget = SpendBudget(options.max_spend_usd)
        self._results: list[UnitResult] = []
        self._streaks: dict[str, int] = {}
        self._aborted: str | None = None
        self._limits: ProviderLimits | None = None
        self._run_id: str | None = None

    @property
    def last_run_id(self) -> str | None:
        """The run id in progress, or the last one attempted.

        Exposed because of ``Ctrl-C``: the most valuable thing a user has at
        that moment is the id to pass to ``--resume``, and it is unreachable
        from a summary that never got returned.
        """
        return self._run_id

    # ------------------------------------------------------------- selection

    def _selected_cases(self) -> list[Case]:
        wanted = self.options.case_ids
        cases = list(self.dataset.cases)
        if not wanted:
            return cases
        by_id = {c.case_id: c for c in cases}
        missing = [c for c in wanted if c not in by_id]
        if missing:
            raise ConfigError(
                f"no such case(s) in {self.dataset.path.name}: {', '.join(missing)}",
                hint=f"available: {', '.join(sorted(by_id)[:10])}",
            )
        return [by_id[c] for c in wanted]

    def _selected_models(self) -> list[ModelCfg]:
        wanted = self.options.candidate_ids
        if not wanted:
            return list(self.config.all_models)
        missing = [c for c in wanted if c not in self._models]
        if missing:
            raise ConfigError(
                f"no such candidate(s) in eval.toml: {', '.join(missing)}",
                hint=f"available: {', '.join(self._models)}",
            )
        chosen = [self._models[c] for c in wanted]
        baseline = self.config.baseline
        # The baseline comes along whether or not it was asked for: it records
        # the cassettes the candidates replay, and every delta column in the
        # report is computed against it. A `--candidate` run without it would
        # produce a table with nothing to compare to.
        if baseline.id not in wanted:
            chosen.insert(0, baseline)
        return chosen

    # ------------------------------------------------------------------- plan

    def plan(self, *, run_id: str | None = None, resume: str | None = None) -> RunPlan:
        """Decide what to run, without running any of it.

        Separate from :meth:`run` so ``--dry-run`` and the progress bar's total
        cost nothing, and so the planning rules are testable without an event
        loop.
        """
        cases = self._selected_cases()
        models = self._selected_models()
        self._cases = {c.case_id: c for c in cases}

        completed: set[tuple[str, str, int]] = set()
        if resume is not None:
            run_id = self._prepare_resume(resume)
            completed = self.store.completed_units(run_id)
        run_id = run_id or new_run_id()
        self._run_id = run_id

        chosen = {m.id for m in models}
        matrix = plan_matrix(
            candidate_ids=[m.id for m in models],
            case_ids=[c.case_id for c in cases],
            repeats=self.options.repeats,
            baseline_id=(
                self.config.baseline.id if self.config.baseline.id in chosen else None
            ),
            completed=completed,
        )
        self._limits = ProviderLimits.for_run(models, self.options)
        return RunPlan(
            run_id=run_id,
            name=self.name,
            matrix=matrix,
            tool_mode=self.options.tool_mode,
            scheduling=self.options.scheduling,
            fault_rate=self.options.fault_rate,
            seed=self.options.seed,
            limits=self._limits.describe(),
        )

    def _prepare_resume(self, fragment: str) -> str:
        """Resolve a run to resume, and refuse if its premises have moved.

        Resuming a run whose dataset or config changed would stitch half a
        matrix measured one way onto half measured another and print the result
        as one table. The refusal is the feature; ``--force`` is not offered
        here, because the cheap alternative is simply starting a new run.
        """
        run_id = self.store.resolve_run_id(fragment)
        stored = self.store.require_run(run_id)
        pending = replace(
            stored,
            config_hash=self.config.config_hash(),
            dataset_hash=self.dataset.content_hash,
            mode=self.options.tool_mode,
            seed=self.options.seed,
            slmeval_version=__version__,
        )
        blocking = [d for d in divergences(stored, pending) if d.blocking]
        if blocking:
            detail = "; ".join(f"{d.field}: {d.reason}" for d in blocking)
            raise ConfigError(
                f"cannot resume {run_id}: {detail}",
                hint=(
                    "the finished half of that run was measured under different "
                    "premises, and joining the halves would manufacture a "
                    "difference. Start a fresh run instead."
                ),
            )
        return run_id

    # -------------------------------------------------------------------- run

    async def run(self, *, run_id: str | None = None, resume: str | None = None) -> RunSummary:
        """Execute the matrix. Returns once every unit has a row or the run aborts."""
        plan = self.plan(run_id=run_id, resume=resume)
        started_at = datetime.now(UTC).isoformat()

        if resume is None:
            self._write_header(plan, started_at)
        self._write_candidates(plan)
        self._build_routers()
        self._build_faults(plan)

        self.observer.on_plan(plan)

        # `benchmark_running` is what lets a tool call with no context tell
        # "production, calling the real API is correct" from "context lost
        # crossing a thread boundary during a benchmark", which is a warning
        # the user must see.
        try:
            with benchmark_running():
                first, rest = plan.matrix.phases()
                await self._warm_up(plan)
                await self._drain(plan, first, serial=self._record_serially())
                if self._aborted is None:
                    await self._drain(plan, rest, serial=self.options.serial_latency)
        finally:
            await self._close_routers()

        summary = self._summarize(plan, started_at)
        self.store.finish_run(
            plan.run_id,
            status=summary.status,
            error_message=summary.aborted_reason,
            finished_at=summary.finished_at,
        )
        self.observer.on_finish(summary)
        return summary

    def _record_serially(self) -> bool:
        """Whether the baseline phase runs one at a time.

        It does while recording. Two executions recording the same case into
        one cassette store at once is a write race whose loser's payloads
        simply are not there later - and a missing recording surfaces as a
        cassette miss, which reads as a model failure.
        """
        return self.options.serial_latency or self.options.tool_mode in {"record", "auto"}

    def _write_header(self, plan: RunPlan, started_at: str) -> None:
        provenance = Provenance.collect(
            config_hash=self.config.config_hash(),
            dataset_hash=self.dataset.content_hash,
            dataset_path=self.dataset.path,
            root=self.root,
        )
        self.store.save_run(
            RunRecord(
                run_id=plan.run_id,
                name=self.name,
                started_at=started_at,
                config_json=self.config.model_dump_json(),
                config_hash=provenance.config_hash,
                dataset_hash=provenance.dataset_hash,
                dataset_path=provenance.dataset_path,
                slmeval_version=provenance.slmeval_version,
                git_sha=provenance.git_sha,
                host_info=provenance.host,
                mode=self.options.tool_mode,
                seed=self.options.seed,
                scheduling=self.options.scheduling,
                fault_rate=self.options.fault_rate,
                runs_per_case=self.options.repeats,
                status=RunStatus.RUNNING,
            )
        )

    def _write_candidates(self, plan: RunPlan) -> None:
        baseline_id = self.config.baseline.id
        self.store.save_candidates(
            CandidateRecord(
                run_id=plan.run_id,
                candidate_id=cid,
                role="baseline" if cid == baseline_id else "candidate",
                ordinal=index,
                provider=self._models[cid].provider,
                model=self._models[cid].model,
                # The whole resolved config, not a pointer at the file:
                # eval.toml will have been edited by the time anyone reads this
                # row, and a report that re-read it would describe a run that
                # never happened.
                model_cfg_json=self._models[cid].model_dump_json(),
            )
            for index, cid in enumerate(plan.matrix.candidate_ids)
        )

    def _build_routers(self) -> None:
        for candidate_id in self._models:
            with contextlib.suppress(KeyError):
                self._routers[candidate_id] = ModelRouter.for_candidate(
                    self.config, candidate_id
                )

    def _build_faults(self, plan: RunPlan) -> None:
        """One schedule per case, built before any candidate runs it.

        Per case and shared, never per execution: every candidate has to meet
        the same experiment, and the only way to guarantee that is for there to
        be one object to hand out.
        """
        if self.options.fault_rate <= 0.0:
            return
        kwargs = self.config.benchmark.faults.to_kwargs(rate=self.options.fault_rate)
        for case_id in plan.matrix.case_ids:
            self._faults[case_id] = fault_schedule(self.options.seed, case_id, **kwargs)

    async def _close_routers(self) -> None:
        for router in self._routers.values():
            with contextlib.suppress(Exception):
                await router.aclose()

    # ------------------------------------------------------------- warm-up

    async def _warm_up(self, plan: RunPlan) -> None:
        """One throwaway call per local model, before anything is timed.

        Only local models, and only in latency mode. The first call to an
        unloaded Ollama model pays for weights coming off disk - tens of
        seconds - and charging that to the first case would publish a number
        about our scheduling rather than about the model. Cloud endpoints have
        no cold start worth warming, and the scripted provider has no server at
        all.

        A warm-up failure is a note, not an error: the real executions will
        report the provider problem with far better context than a warm-up can.
        """
        if not (self.options.warmup and self.options.serial_latency):
            return
        for candidate_id in plan.matrix.candidate_ids:
            router = self._routers.get(candidate_id)
            if router is None:
                continue
            for client in router.clients():
                spec = client.spec
                if not is_local(spec.provider, spec.base_url):
                    continue
                self.observer.on_note(f"warming up {spec.model} on {spec.base_url}")
                try:
                    # No context: a warm-up call must not land in any
                    # execution's trace buffer and inflate its token count.
                    with no_context():
                        await client.complete(WARMUP_PROMPT)
                except BaseException as exc:
                    if isinstance(exc, ABORTING):
                        raise
                    self.observer.on_note(
                        f"warm-up of {spec.model} failed ({type(exc).__name__}); "
                        "continuing - the first timed call will carry the cold start"
                    )

    # ------------------------------------------------------------- scheduling

    async def _drain(
        self, plan: RunPlan, units: Sequence[WorkUnit], *, serial: bool
    ) -> None:
        if not units or self._aborted is not None:
            return
        queue: asyncio.Queue[WorkUnit] = asyncio.Queue()
        for unit in units:
            queue.put_nowait(unit)

        workers = 1 if serial else max(1, min(self.options.concurrency, len(units)))
        # TaskGroup, so a cancellation anywhere takes the whole phase down
        # together and no worker is left running after `run` returns. Every
        # unit body swallows its own failures, so the group only ever unwinds
        # on a real abort.
        try:
            async with asyncio.TaskGroup() as group:
                for _ in range(workers):
                    group.create_task(self._worker(plan, queue))
        except BaseExceptionGroup as grouped:
            # A TaskGroup wraps whatever its children raised, and the only
            # thing ours ever raise is an abort. Unwrapped here so a caller can
            # write `except KeyboardInterrupt` and have it mean what it says -
            # an ExceptionGroup leaking out would make every caller of this
            # runner re-implement this unwrapping, and the one that forgot
            # would report a Ctrl-C as a crash.
            raise _first_abort(grouped) from None

    async def _worker(self, plan: RunPlan, queue: asyncio.Queue[WorkUnit]) -> None:
        while True:
            if self._aborted is not None:
                return
            try:
                unit = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            result = await self._run_unit(plan, unit)
            self._record(result)

    def _record(self, result: UnitResult) -> None:
        self._results.append(result)
        self._budget.add(result.cost)
        self.observer.on_unit_done(result)

        if self._budget.exceeded:
            self._abort(self._budget.reason())
            return

        if not self.options.fail_fast:
            return
        # Streak per candidate, reset by any pass. A candidate that is merely
        # worse keeps passing occasionally; one that is broken does not.
        if result.passed or not result.judgeable:
            self._streaks[result.candidate_id] = 0
            return
        streak = self._streaks.get(result.candidate_id, 0) + 1
        self._streaks[result.candidate_id] = streak
        if streak >= self.options.fail_fast_streak:
            self._abort(
                f"--fail-fast: {result.candidate_id} failed {streak} executions in a row "
                f"(last: {result.outcome.value})"
            )

    def _abort(self, reason: str) -> None:
        if self._aborted is None:
            self._aborted = reason
            self.observer.on_note(reason)

    # ------------------------------------------------------- one execution

    async def _run_unit(self, plan: RunPlan, unit: WorkUnit) -> UnitResult:
        """Run one cell, classify however it ends, and persist it.

        Returns normally for every outcome except cancellation. That is the
        crash-isolation guarantee: one case engineered to raise inside the
        harness is reported as ``harness_error`` and the other 89 rows are
        unaffected.
        """
        case = self._cases[unit.case_id]
        ctx = self._context(plan, unit)
        started_at = datetime.now(UTC).isoformat()
        clock = time.perf_counter()
        state: dict[str, Any] | None = None
        error: BaseException | None = None

        self.observer.on_unit_start(unit)
        try:
            key = self._limits.key_for(  # type: ignore[union-attr]
                self._models[unit.candidate_id], timeout_s=self.options.timeout_s
            )
            async with self._limits.slot(key):  # type: ignore[union-attr]
                # wait_for, so a hung agent costs one cell instead of the run.
                # The context is installed *inside* the coroutine it wraps, by
                # execute_case, which is what keeps 89 sibling executions from
                # seeing each other's traces.
                state = await asyncio.wait_for(
                    execute_case(ctx, self.target, case.input),
                    timeout=self.options.timeout_s or None,
                )
        except BaseException as exc:
            if isinstance(exc, ABORTING):
                # Never recorded, always re-raised. A unit with no row is work
                # that `--resume` will pick up; a unit recorded as cancelled
                # would be skipped and silently lost from the matrix.
                raise
            error = exc

        duration_ms = (time.perf_counter() - clock) * 1000.0
        classification = classify(error)
        verdict = self._evaluate(ctx, case, state, classification)
        cost = self._price(ctx, duration_ms)

        for message in verdict.warnings:
            ctx.warn(message)

        exec_id = unit.exec_id(plan.run_id)
        ingest.save_context(
            self.store,
            ctx,
            status=classification.status,
            outcome=verdict.outcome,
            duration_ms=duration_ms,
            started_at=started_at,
            cost_usd=cost.usd,
            error=error,
            output=_final_answer(state),
            state=state,
            evaluations=verdict.rows(exec_id),
        )
        return UnitResult(
            unit=unit,
            exec_id=exec_id,
            status=classification.status,
            outcome=verdict.outcome,
            duration_ms=duration_ms,
            steps=ctx.steps,
            cost=cost,
            error_class=classification.error_class,
            error_message=classification.message,
            warnings=tuple(ctx.warnings),
            verdict=verdict,
        )

    def _context(self, plan: RunPlan, unit: WorkUnit) -> RunContext:
        return RunContext(
            run_id=plan.run_id,
            candidate_id=unit.candidate_id,
            case_id=unit.case_id,
            attempt=unit.repeat_idx,
            seed=self.options.seed,
            tool_mode=self.options.tool_mode,
            max_steps=self.options.max_steps,
            router=self._routers.get(unit.candidate_id),
            cassettes=self.cassettes,
            faults=self._faults.get(unit.case_id),
        )

    def _evaluate(
        self,
        ctx: RunContext,
        case: Case,
        state: dict[str, Any] | None,
        classification: Classification,
    ) -> Verdict:
        """Score the execution, and treat an evaluator crash as *our* bug.

        An exception from an evaluator must never be attributed to the
        candidate. Scoring our own failure as a model failure is the specific
        way a measurement instrument becomes worthless.
        """
        if not self.options.evaluate:
            return Verdict(outcome=classification.outcome)
        try:
            return self.evaluator(
                ctx=ctx, case=case, state=state, classification=classification
            )
        except BaseException as exc:
            if isinstance(exc, ABORTING):
                raise
            ctx.warn(f"evaluator raised {type(exc).__name__}: {exc}")
            return Verdict(outcome=Outcome.HARNESS_ERROR)

    def _price(self, ctx: RunContext, duration_ms: float) -> Priced:
        """Cost this execution, model call by model call.

        Per call, not per execution, because a node-scoped candidate's
        executions are billed partly to the candidate and partly to the
        baseline - which is the entire economics of "downsize one node".
        """
        costs: list[Priced] = []
        for call in ctx.llm_calls:
            cfg = self._models.get(call.model_id)
            if cfg is None:
                continue
            costs.append(
                self.prices.cost_of(
                    cfg,
                    prompt_tokens=call.prompt_tokens,
                    completion_tokens=call.completion_tokens,
                    estimated=call.tokens_estimated,
                    wall_seconds=call.latency_ms / 1000.0,
                )
            )
        if not costs:
            return Priced(usd=0.0)
        return total_cost(costs)

    # -------------------------------------------------------------- summary

    def _summarize(self, plan: RunPlan, started_at: str) -> RunSummary:
        if self._aborted is not None:
            status = RunStatus.CANCELLED
        elif len(self._results) + len(plan.matrix.resumed) >= plan.matrix.planned:
            status = RunStatus.COMPLETED
        else:
            status = RunStatus.FAILED
        return RunSummary(
            run_id=plan.run_id,
            status=status,
            planned=plan.matrix.planned,
            resumed=len(plan.matrix.resumed),
            results=tuple(self._results),
            spend_usd=self._budget.spent,
            unpriced_models=self._budget.unpriced,
            scheduling=plan.scheduling,
            aborted_reason=self._aborted,
            started_at=started_at,
            finished_at=datetime.now(UTC).isoformat(),
        )


def _first_abort(grouped: BaseExceptionGroup[BaseException]) -> BaseException:
    """The abort inside a TaskGroup's exception group, flattened.

    Returns the group itself if it somehow holds something else - which would
    be a harness bug, and swapping it for a plausible-looking
    ``KeyboardInterrupt`` would hide it.
    """
    stack: list[BaseException] = [grouped]
    while stack:
        current = stack.pop()
        if isinstance(current, BaseExceptionGroup):
            stack.extend(current.exceptions)
        elif isinstance(current, ABORTING):
            return current
    return grouped


def _final_answer(state: Mapping[str, Any] | None) -> Any:
    """The answer the drill-down shows, pulled out of the final state.

    Stored alongside the whole state rather than instead of it: the diff view
    leads with what the agent *said*, and digging two levels into a state dict
    to find it would make the most-used view the slowest to read.
    """
    if not isinstance(state, Mapping):
        return None
    for key in ("final_answer", "answer", "output", "response"):
        if key in state:
            return state[key]
    return None


def encode_plan(plan: RunPlan) -> str:
    """The plan as JSON, for ``--dry-run`` and for tests."""
    return codec.dumps(
        {
            "run_id": plan.run_id,
            "name": plan.name,
            "tool_mode": plan.tool_mode,
            "scheduling": plan.scheduling,
            "seed": plan.seed,
            "fault_rate": plan.fault_rate,
            "candidates": list(plan.matrix.candidate_ids),
            "cases": list(plan.matrix.case_ids),
            "repeats": plan.matrix.repeats,
            "planned": plan.matrix.planned,
            "to_run": len(plan.matrix.units),
            "resumed": len(plan.matrix.resumed),
            "limits": plan.limits,
        }
    )


__all__ = [
    "DEFAULT_LOCAL_CONCURRENCY",
    "FAIL_FAST_STREAK",
    "LATENCY",
    "THROUGHPUT",
    "Endpoint",
    "Matrix",
    "MatrixRunner",
    "NullObserver",
    "Observer",
    "ProviderLimits",
    "RunOptions",
    "RunPlan",
    "RunSummary",
    "SpendBudget",
    "UnitResult",
    "WorkUnit",
    "encode_plan",
    "plan_matrix",
]
