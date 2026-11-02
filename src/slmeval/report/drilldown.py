"""The failure drill-down: one execution, and two of them side by side.

The scorecard says a candidate lost four points. This says *where*. A user who
cannot get from a red cell to the exact step that diverged will not believe the
red cell for long, and a benchmark nobody audits is a benchmark nobody trusts -
so the drill-down is a first-class view, not a debug dump.

Three things it renders, all of them from stored rows:

:func:`render_run`
    The run as a list of cells: which executions exist, how each ended, which
    are worth opening.
:func:`render_execution`
    One execution's timeline - trajectory, model calls, tool calls, metrics,
    final state.
:func:`render_diff`
    Baseline on the left, candidate on the right, divergence marked, with the
    cassette-miss diagnosis block inline at the step that missed.

**Why the diff aligns three sequences rather than one timeline.** Model calls
and tool calls are numbered by two independent counters (see
:class:`~slmeval.core.context.RunContext`), so there is no recorded global
order to interleave them by, and inventing one would put a tool call before the
model call that asked for it as often as not. Instead each kind of sequence is
aligned against its own counterpart: the trajectory against the trajectory, the
tool calls against the tool calls, the per-node model behaviour against itself.
Every row in the result is then a comparison of two things that are genuinely
comparable.

**Alignment, not zipping.** Two trajectories that differ by one extra node
visit are not "different from step 3 onwards" - they are the same path with one
insertion, and a view that zipped them would report four divergences where
there is one. :func:`align` uses :class:`difflib.SequenceMatcher`, so an
inserted step reads as an insertion and the rest of the path still lines up.

**The diagnosis block is rebuilt, not re-derived.** The verdict stored at the
moment of the miss wins: it was computed with the live tool registry, which a
report running without the user's agent loaded does not have. The nearest
recorded call and the per-argument diff are recovered from the cassette store
for display, and if the rebuilt verdict disagrees with the stored one, the
stored one is what gets printed.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import Any, Final

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import Symbols, render_miss, symbols
from slmeval.cli.render import dash as dash_for
from slmeval.sandbox.diagnosis import MissDiagnosis, Verdict, diagnose
from slmeval.sandbox.store import CassetteStore
from slmeval.storage import codec
from slmeval.storage.models import (
    NOT_THE_MODELS_FAULT,
    CandidateRecord,
    ExecutionDetail,
    ExecutionRecord,
    NodeTraceRow,
    Outcome,
    RunRecord,
)

EQUAL: Final = "equal"
REPLACE: Final = "replace"
INSERT: Final = "insert"
"""Present on the right (the candidate) and not on the left."""
DELETE: Final = "delete"
"""Present on the left (the baseline) and not on the right."""

OUTCOME_STYLE: Final = {
    Outcome.PASSED: "green",
    Outcome.UNCLASSIFIED: "dim",
    Outcome.UNCOVERED: "cyan",
    Outcome.HARNESS_ERROR: "bold magenta",
    Outcome.FORBIDDEN_TOOL: "bold red",
    Outcome.PROVIDER_ERROR: "yellow",
}
"""Colour per outcome, defaulting to red.

``uncovered`` is cyan and ``harness_error`` magenta for the same reason they
are not red in the scorecard: neither is a model failure, and colouring them
like one would undo the distinction the classifier exists to draw.
"""

ARG_WIDTH: Final = 60
"""How much of a tool call's arguments a timeline row shows.

A tool that takes a 4KB document would otherwise push every other column off
the screen. The full payload is one ``show exec`` away, and the alignment key
uses the whole thing regardless of what is printed.
"""


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pair:
    """One row of a side-by-side view: a left item, a right item, or both."""

    op: str
    left: Any = None
    right: Any = None
    left_index: int | None = None
    right_index: int | None = None

    @property
    def equal(self) -> bool:
        return self.op == EQUAL

    @property
    def glyph(self) -> str:
        """The marker in the gutter. ``~`` for a substitution, not ``!``.

        A replaced step is one thing done differently; an inserted or deleted
        one is a step that exists on one side only. Those are different
        findings and they get different marks.
        """
        return {EQUAL: " ", REPLACE: "~", INSERT: "+", DELETE: "-"}[self.op]


@dataclass(frozen=True, slots=True)
class Alignment:
    """Two sequences, lined up.

    What the rows were matched *on* and what they carry are deliberately
    different things: :func:`align` takes a ``key`` function, so the tool-call
    diff lines up on ``name(args)`` while every row still carries the whole
    step and can print outcome, match tier and fault beside it.
    """

    rows: tuple[Pair, ...] = ()

    @property
    def identical(self) -> bool:
        return all(row.equal for row in self.rows)

    @property
    def diverged_at(self) -> int | None:
        """Index of the first row that is not equal, or ``None``.

        The single most useful number in the whole view: "they agreed for three
        steps and parted at the fourth" is the sentence a user is looking for.
        """
        return next((i for i, row in enumerate(self.rows) if not row.equal), None)

    @property
    def differences(self) -> int:
        return sum(1 for row in self.rows if not row.equal)


def align(left: Sequence[Any], right: Sequence[Any], *, key: Any = None) -> Alignment:
    """Line two sequences up with :class:`difflib.SequenceMatcher`.

    ``key`` maps an item to the hashable value the match is computed on, so a
    caller can align rich objects by the part of them that counts as "the same
    step". It defaults to the item itself.

    ``autojunk`` is off. It is a heuristic for prose that discards elements
    appearing in more than 1% of a long sequence - which, in a trajectory where
    one node is visited on every step, is exactly the element the diff is
    about.
    """
    to_key = key if key is not None else (lambda item: item)
    left_keys = [to_key(item) for item in left]
    right_keys = [to_key(item) for item in right]
    matcher = SequenceMatcher(a=left_keys, b=right_keys, autojunk=False)

    rows: list[Pair] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for offset in range(i2 - i1):
                rows.append(
                    Pair(
                        op=EQUAL,
                        left=left[i1 + offset],
                        right=right[j1 + offset],
                        left_index=i1 + offset,
                        right_index=j1 + offset,
                    )
                )
            continue
        if op == "replace":
            # Pair them up as far as both sides go, then spill the remainder as
            # insertions or deletions: a 2-for-3 substitution is two changed
            # steps and one extra one, and flattening it to three of either
            # would misreport what happened.
            overlap = min(i2 - i1, j2 - j1)
            for offset in range(overlap):
                rows.append(
                    Pair(
                        op=REPLACE,
                        left=left[i1 + offset],
                        right=right[j1 + offset],
                        left_index=i1 + offset,
                        right_index=j1 + offset,
                    )
                )
            for index in range(i1 + overlap, i2):
                rows.append(Pair(op=DELETE, left=left[index], left_index=index))
            for index in range(j1 + overlap, j2):
                rows.append(Pair(op=INSERT, right=right[index], right_index=index))
            continue
        if op == "delete":
            for index in range(i1, i2):
                rows.append(Pair(op=DELETE, left=left[index], left_index=index))
            continue
        for index in range(j1, j2):
            rows.append(Pair(op=INSERT, right=right[index], right_index=index))
    return Alignment(rows=tuple(rows))


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ToolStep:
    """One tool call from a stored trace, with its arguments resolved."""

    seq: int
    name: str
    arguments: Any = None
    node: str = ""
    outcome: str = "live"
    match_tier: str | None = None
    fault: str | None = None
    miss_verdict: str | None = None
    error: str | None = None
    latency_ms: float = 0.0

    @property
    def key(self) -> str:
        """What the diff aligns on: the call, not how it resolved.

        Arguments are part of the key, because "called ``fetch_order`` with the
        wrong id" is the finding the diff exists to surface, and a key of the
        tool name alone would call that an identical step. Outcome is *not*:
        the same call replayed by one candidate and missed by another is the
        same step with two different results, which is the comparison.
        """
        return f"{self.name}({codec.dumps(self.arguments)})"

    @property
    def missed(self) -> bool:
        return self.outcome == "miss"

    @property
    def uncovered(self) -> bool:
        return self.miss_verdict == Verdict.UNRECORDED_VALID_PATH.value

    def render(self, *, width: int = ARG_WIDTH) -> str:
        return f"{self.name}({_short(codec.dumps(self.arguments), width)})"


@dataclass(frozen=True, slots=True)
class NodeStep:
    """One model call from a stored trace."""

    seq: int
    node: str
    model: str = ""
    schema_state: str = "not_requested"
    structured_mode: str | None = None
    repair_attempts: int = 0
    json_stage: str | None = None
    retries: int = 0
    latency_ms: float = 0.0
    error: str | None = None

    @property
    def key(self) -> str:
        """Aligned on the node, not on how the call went.

        The model behaviour at a node is what the right-hand columns show; the
        sequence being aligned is which nodes asked a model and in what order.
        """
        return self.node

    @property
    def clean(self) -> bool:
        return (
            self.error is None
            and not self.repair_attempts
            and self.schema_state in {"valid", "not_requested"}
        )

    def render(self) -> str:
        bits = [self.node]
        if self.schema_state not in {"valid", "not_requested"}:
            bits.append(self.schema_state)
        if self.structured_mode and self.structured_mode not in {"tools", "json_schema"}:
            bits.append(self.structured_mode)
        if self.repair_attempts:
            bits.append(f"{self.repair_attempts} repair")
        if self.json_stage and self.json_stage != "direct":
            bits.append(self.json_stage)
        if self.error:
            bits.append(self.error)
        return " · ".join(bits)


def tool_steps(detail: ExecutionDetail) -> tuple[ToolStep, ...]:
    """The tool calls of one execution, in recorded order."""
    return tuple(
        ToolStep(
            seq=row.seq,
            name=row.tool_name,
            arguments=detail.tool_arguments.get(row.seq),
            node=row.node_name,
            outcome=row.outcome,
            match_tier=row.match_tier,
            fault=row.fault,
            miss_verdict=row.miss_verdict,
            error=row.error_message,
            latency_ms=row.latency_ms,
        )
        for row in detail.tool_traces
    )


def node_steps(detail: ExecutionDetail) -> tuple[NodeStep, ...]:
    """The model calls of one execution, in recorded order."""
    return tuple(_node_step(row) for row in detail.node_traces)


def _node_step(row: NodeTraceRow) -> NodeStep:
    return NodeStep(
        seq=row.seq,
        node=row.node_name,
        model=row.model,
        schema_state=row.schema_state,
        structured_mode=row.structured_mode,
        repair_attempts=row.repair_attempts,
        json_stage=row.json_stage,
        retries=row.retries,
        latency_ms=row.latency_ms,
        error=row.error_class,
    )


def trajectory_of(detail: ExecutionDetail) -> tuple[str, ...]:
    """Node visits as ``parent > child`` strings, in order."""
    return tuple(" > ".join(path) for path in detail.trajectory)


# --------------------------------------------------------------------------- #
# The diagnosis block, rebuilt for display
# --------------------------------------------------------------------------- #


def known_tool_names(
    details: Iterable[ExecutionDetail], store: CassetteStore | None
) -> tuple[str, ...]:
    """Tool names this project is known to have, without importing the agent.

    ``show`` deliberately does not load the user's module - a checkout whose
    agent has since been refactored still has valid results - so the live
    ``@sandbox_tool`` registry is empty here. Without this, every rebuilt
    diagnosis would classify a perfectly real tool as invented.
    """
    names = {step.name for detail in details for step in tool_steps(detail)}
    if store is not None:
        # Rebuilding a diagnosis is a display concern. A cassette store that
        # cannot be read degrades the block rather than failing the command the
        # user ran to understand a failure.
        with contextlib.suppress(Exception):
            names.update(store.tool_names())
    return tuple(sorted(names))


def rebuild_diagnosis(
    step: ToolStep,
    *,
    case_id: str,
    candidate_id: str,
    cassettes: CassetteStore | None,
    known_tools: Sequence[str] = (),
) -> MissDiagnosis | None:
    """Recover the diagnosis block for a stored miss.

    The stored verdict wins over the recomputed one. It was decided at the
    moment of the miss with the live tool registry and the store as it was
    then; this runs later, possibly on another machine, and its job is to
    recover the *detail* - nearest recorded call, per-argument diff - not to
    re-judge the candidate. A report that silently re-classified last month's
    failures would make the stored outcome column a lie.
    """
    if not step.missed:
        return None
    arguments = step.arguments if isinstance(step.arguments, Mapping) else {}
    finding = diagnose(
        case_id=case_id,
        tool_name=step.name,
        arguments=arguments,
        store=cassettes,
        known_tools=known_tools,
        candidate_id=candidate_id,
        node=step.node or None,
    )
    stored = step.miss_verdict
    if stored and stored != finding.verdict.value:
        try:
            verdict = Verdict(stored)
        except ValueError:  # pragma: no cover - a verdict from a newer version
            return finding
        finding = replace(
            finding,
            verdict=verdict,
            rationale=(
                f"{finding.rationale} (recorded verdict: {stored} - kept, because it "
                "was decided with the tool registry live)"
            ),
        )
    return finding


# --------------------------------------------------------------------------- #
# The diff
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Diff:
    """One candidate execution against the baseline's, for the same case."""

    case_id: str
    candidate_id: str
    baseline_id: str
    candidate: ExecutionDetail
    baseline: ExecutionDetail | None = None
    trajectory: Alignment = Alignment()
    tools: Alignment = Alignment()
    nodes: Alignment = Alignment()
    diagnoses: tuple[MissDiagnosis, ...] = ()
    fault_parity: bool = True
    """Whether both executions met the same faults.

    False means the two columns were not the same experiment, and the view says
    so above the tables - a divergence explained by a fault one side met and
    the other did not is not a finding about the model.
    """

    @property
    def comparable(self) -> bool:
        return self.baseline is not None

    @property
    def identical(self) -> bool:
        return self.trajectory.identical and self.tools.identical

    def summary(self, arrow: str = "->") -> str:
        """One line: did they agree, and if not, where did they part."""
        if not self.comparable:
            return f"no baseline execution stored for {self.case_id}"
        if self.identical:
            return "same trajectory, same tool calls"
        parts: list[str] = []
        step = self.trajectory.diverged_at
        if step is not None:
            row = self.trajectory.rows[step]
            parts.append(
                f"trajectory diverged at step {step + 1}: "
                f"{row.left or '(none)'} {arrow} {row.right or '(none)'}"
            )
        tool = self.tools.diverged_at
        if tool is not None:
            row = self.tools.rows[tool]
            left = row.left.render() if row.left else "(none)"
            right = row.right.render() if row.right else "(none)"
            parts.append(f"tool call {tool + 1}: {left} {arrow} {right}")
        return "; ".join(parts) or "same trajectory, same tool calls"


def build_diff(
    *,
    case_id: str,
    candidate: ExecutionDetail,
    baseline: ExecutionDetail | None,
    candidate_id: str = "",
    baseline_id: str = "baseline",
    cassettes: CassetteStore | None = None,
) -> Diff:
    """Align one candidate execution against the baseline's, from rows alone."""
    left_tools = tool_steps(baseline) if baseline else ()
    right_tools = tool_steps(candidate)
    known = known_tool_names(
        [d for d in (baseline, candidate) if d is not None], cassettes
    )
    diagnoses = tuple(
        finding
        for step in right_tools
        if (
            finding := rebuild_diagnosis(
                step,
                case_id=case_id,
                candidate_id=candidate_id,
                cassettes=cassettes,
                known_tools=known,
            )
        )
        is not None
    )
    return Diff(
        case_id=case_id,
        candidate_id=candidate_id or candidate.execution.candidate_id,
        baseline_id=baseline_id,
        candidate=candidate,
        baseline=baseline,
        trajectory=align(
            trajectory_of(baseline) if baseline else (), trajectory_of(candidate)
        ),
        tools=align(left_tools, right_tools, key=lambda step: step.key),
        nodes=align(
            node_steps(baseline) if baseline else (),
            node_steps(candidate),
            key=lambda step: step.key,
        ),
        diagnoses=diagnoses,
        fault_parity=(
            baseline is None
            or baseline.execution.fault_digest == candidate.execution.fault_digest
        ),
    )


# --------------------------------------------------------------------------- #
# Uncovered paths
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Gap:
    """One place the cassettes did not cover, and who found it.

    Aggregated per ``(case, tool, arguments)`` rather than per execution: three
    repeats of one case reaching the same unrecorded call is one gap to fill,
    and printing it three times would make a two-line fix look like a
    six-execution catastrophe.
    """

    case_id: str
    tool_name: str
    arguments: Any = None
    verdict: str = Verdict.UNRECORDED_VALID_PATH.value
    candidates: tuple[str, ...] = ()
    executions: int = 0
    exec_id: str = ""

    @property
    def uncovered(self) -> bool:
        """Whether this is our gap rather than the model's mistake."""
        return self.verdict == Verdict.UNRECORDED_VALID_PATH.value

    def render(self, *, width: int = ARG_WIDTH) -> str:
        return f"{self.tool_name}({_short(codec.dumps(self.arguments), width)})"


def gaps(
    details: Iterable[tuple[str, ExecutionDetail]], *, uncovered_only: bool = False
) -> tuple[Gap, ...]:
    """Every cassette miss across a run, grouped.

    Takes ``(candidate_id, detail)`` pairs so the caller decides which
    executions to load - a full matrix is thousands of them, and the command
    that prints this asks for the ones whose outcome was a miss.

    ``uncovered_only`` keeps the gaps that are ours to fill and drops the
    hallucinations, which is what ``slmeval show uncovered`` wants: the list
    that feeds ``record --fill-gaps``.
    """
    grouped: dict[tuple[str, str, str], Gap] = {}
    for candidate_id, detail in details:
        case_id = detail.execution.case_id
        for step in tool_steps(detail):
            if not step.missed:
                continue
            if uncovered_only and not step.uncovered:
                continue
            key = (case_id, step.name, codec.dumps(step.arguments))
            found = grouped.get(key)
            if found is None:
                grouped[key] = Gap(
                    case_id=case_id,
                    tool_name=step.name,
                    arguments=step.arguments,
                    verdict=step.miss_verdict or Verdict.UNRECORDED_VALID_PATH.value,
                    candidates=(candidate_id,),
                    executions=1,
                    exec_id=detail.execution.exec_id,
                )
                continue
            grouped[key] = replace(
                found,
                candidates=tuple(dict.fromkeys((*found.candidates, candidate_id))),
                executions=found.executions + 1,
            )
    return tuple(
        sorted(grouped.values(), key=lambda gap: (gap.case_id, gap.tool_name))
    )


# --------------------------------------------------------------------------- #
# Rendering: one execution
# --------------------------------------------------------------------------- #


def render_execution(
    detail: ExecutionDetail,
    *,
    ascii_mode: bool | None = None,
    diagnoses: Sequence[MissDiagnosis] = (),
    show_payloads: bool = True,
) -> RenderableType:
    """One execution, end to end: what it did and how it was judged."""
    sym = symbols(ascii_mode)
    execution = detail.execution
    parts: list[RenderableType] = [
        Text(""),
        _execution_header(execution, sym),
        Text(""),
    ]

    trajectory = trajectory_of(detail)
    if trajectory:
        parts += [
            Text("  trajectory", style="bold dim"),
            Text(f"    {f' {sym.arrow} '.join(trajectory)}"),
            Text(""),
        ]

    calls = node_steps(detail)
    if calls:
        parts += [_model_calls_table(calls, ascii_mode=ascii_mode), Text("")]

    tools = tool_steps(detail)
    if tools:
        parts += [_tool_calls_table(tools, ascii_mode=ascii_mode), Text("")]

    for finding in diagnoses:
        parts += [render_miss(finding, ascii_mode=ascii_mode), Text("")]

    if detail.evaluations:
        parts += [_metrics_table(detail, ascii_mode=ascii_mode), Text("")]

    warnings = codec.decode_strings(execution.warnings_json)
    if warnings:
        block = Text()
        for index, warning in enumerate(warnings):
            block.append(f"  {sym.warn} {warning}", style="yellow")
            if index < len(warnings) - 1:
                block.append("\n")
        parts += [block, Text("")]

    # Both, and labelled apart: the runner stores the final answer and the
    # whole final state separately, and an assertion can fail on either. A view
    # that printed only one of them would leave half the failures unexplainable.
    if show_payloads and detail.output is not None:
        parts += [
            Text("  final answer", style="bold dim"),
            Text(f"    {_short(codec.dumps(detail.output), 600)}", style="dim"),
            Text(""),
        ]
    if show_payloads and detail.state is not None:
        parts += [
            Text("  final state", style="bold dim"),
            Text(f"    {_short(codec.dumps(detail.state), 2000)}", style="dim"),
            Text(""),
        ]
    return Group(*parts)


def _execution_header(execution: ExecutionRecord, sym: Symbols) -> Text:
    out = Text("  ")
    out.append(execution.exec_id, style="bold")
    out.append("\n  ")
    out.append(
        f"{execution.candidate_id} · {execution.case_id} · repeat "
        f"{execution.repeat_idx} · ",
        style="dim",
    )
    out.append(
        execution.outcome_class.value,
        style=OUTCOME_STYLE.get(execution.outcome_class, "red"),
    )
    out.append(
        f" · {execution.status.value} · {execution.steps} steps · "
        f"{execution.duration_ms:,.0f} ms",
        style="dim",
    )
    if execution.error_class:
        out.append(f"\n  {sym.fail} ", style="red")
        out.append(
            f"{execution.error_class}: {execution.error_message or ''}".strip(": "),
            style="red",
        )
    return out


def _model_calls_table(
    calls: Sequence[NodeStep], *, ascii_mode: bool | None = None
) -> Table:
    dash = dash_for(ascii_mode)
    grid = _grid("model calls")
    grid.add_column("#", justify="right", style="dim", no_wrap=True)
    grid.add_column("node", no_wrap=True)
    grid.add_column("schema", no_wrap=True)
    grid.add_column("mode", no_wrap=True)
    grid.add_column("repairs", justify="right", no_wrap=True)
    grid.add_column("json", no_wrap=True)
    grid.add_column("ms", justify="right", no_wrap=True)
    for call in calls:
        grid.add_row(
            str(call.seq),
            Text(call.node or dash),
            _schema_cell(call.schema_state, dash),
            Text(call.structured_mode or dash, style=_mode_style(call.structured_mode)),
            Text(str(call.repair_attempts) if call.repair_attempts else dash,
                 style="yellow" if call.repair_attempts else "dim"),
            Text(call.json_stage or dash, style=_stage_style(call.json_stage)),
            Text(f"{call.latency_ms:,.0f}", style="dim"),
        )
    return grid


def _tool_calls_table(
    tools: Sequence[ToolStep], *, ascii_mode: bool | None = None
) -> Table:
    dash = dash_for(ascii_mode)
    grid = _grid("tool calls")
    grid.add_column("#", justify="right", style="dim", no_wrap=True)
    grid.add_column("tool", overflow="fold")
    grid.add_column("outcome", no_wrap=True)
    grid.add_column("tier", no_wrap=True)
    grid.add_column("fault", no_wrap=True)
    grid.add_column("node", no_wrap=True)
    for step in tools:
        grid.add_row(
            str(step.seq),
            Text(step.render(width=70)),
            Text(step.outcome, style=_tool_outcome_style(step)),
            # A normalized or subset hit is a finding, not a detail: the model
            # reached the recorded call, but not with the recorded arguments.
            Text(step.match_tier or dash,
                 style="yellow" if step.match_tier not in {None, "exact"} else "dim"),
            Text(step.fault or dash, style="magenta" if step.fault else "dim"),
            Text(step.node or dash, style="dim"),
        )
    return grid


def _metrics_table(
    detail: ExecutionDetail, *, ascii_mode: bool | None = None
) -> Table:
    dash = dash_for(ascii_mode)
    grid = _grid("metrics")
    grid.add_column("tier", justify="right", style="dim", no_wrap=True)
    grid.add_column("metric", no_wrap=True)
    grid.add_column("value", justify="right", no_wrap=True)
    grid.add_column("detail", overflow="fold", style="dim")
    for row in detail.evaluations:
        value = dash if row.value_num is None else f"{row.value_num:g}"
        grid.add_row(
            str(row.tier),
            row.metric,
            Text(value),
            Text(_short(row.value_text or row.detail_json or "", 70)),
        )
    return grid


# --------------------------------------------------------------------------- #
# Rendering: the side-by-side diff
# --------------------------------------------------------------------------- #


def render_diff(diff: Diff, *, ascii_mode: bool | None = None) -> RenderableType:
    """Baseline on the left, candidate on the right, divergence marked."""
    sym = symbols(ascii_mode)
    parts: list[RenderableType] = [Text(""), _diff_header(diff, sym), Text("")]

    if not diff.comparable:
        parts += [
            Text(
                f"  {sym.warn} no baseline execution stored for {diff.case_id} - "
                "nothing to diff against; run the baseline over this case first",
                style="yellow",
            ),
            Text(""),
            render_execution(diff.candidate, ascii_mode=ascii_mode,
                             diagnoses=diff.diagnoses),
        ]
        return Group(*parts)

    if not diff.fault_parity:
        parts += [
            Text(
                f"  {sym.warn} the two executions met different faults, so part of "
                "any divergence below is the fault schedule rather than the model "
                "(re-run with --fault-rate 0 to compare cleanly)",
                style="yellow",
            ),
            Text(""),
        ]

    parts += [
        _alignment_table(
            diff.trajectory,
            title="trajectory",
            left=diff.baseline_id,
            right=diff.candidate_id,
            render=lambda value: str(value),
            ascii_mode=ascii_mode,
        ),
        Text(""),
    ]
    if diff.tools.rows:
        parts += [
            _alignment_table(
                diff.tools,
                title="tool calls",
                left=diff.baseline_id,
                right=diff.candidate_id,
                render=lambda step: step.render(),
                ascii_mode=ascii_mode,
                style=_tool_step_style,
            ),
            Text(""),
        ]
    if not diff.nodes.identical:
        parts += [
            _alignment_table(
                diff.nodes,
                title="model calls",
                left=diff.baseline_id,
                right=diff.candidate_id,
                render=lambda step: step.render(),
                ascii_mode=ascii_mode,
                style=lambda step: "" if step.clean else "yellow",
            ),
            Text(""),
        ]

    # Inline, under the tables rather than in a column: the block is the output
    # people screenshot, and a cell five columns wide would destroy it.
    for finding in diff.diagnoses:
        parts += [render_miss(finding, ascii_mode=ascii_mode), Text("")]

    parts.append(_diff_footer(diff, sym))
    return Group(*parts)


def _diff_header(diff: Diff, sym: Symbols) -> Text:
    out = Text("  ")
    out.append(diff.case_id, style="bold")
    out.append(f"  {diff.baseline_id} {sym.arrow} {diff.candidate_id}", style="dim")
    if diff.baseline is not None:
        out.append("\n  ")
        out.append(
            f"{diff.baseline.execution.outcome_class.value}",
            style=OUTCOME_STYLE.get(diff.baseline.execution.outcome_class, "red"),
        )
        out.append(f" {sym.arrow} ", style="dim")
        out.append(
            f"{diff.candidate.execution.outcome_class.value}",
            style=OUTCOME_STYLE.get(diff.candidate.execution.outcome_class, "red"),
        )
        out.append(
            f" · {diff.baseline.execution.steps} {sym.arrow} "
            f"{diff.candidate.execution.steps} steps",
            style="dim",
        )
    return out


def _diff_footer(diff: Diff, sym: Symbols) -> Text:
    out = Text()
    out.append(f"  {diff.summary(sym.arrow)}", style="cyan" if not diff.identical else "dim")
    if diff.candidate.execution.outcome_class in NOT_THE_MODELS_FAULT:
        out.append(
            f"\n  {sym.skip} this execution is excluded from every rate in the "
            "scorecard - it is our gap or our bug, not the candidate's failure",
            style="dim",
        )
    return out


def _alignment_table(
    alignment: Alignment,
    *,
    title: str,
    left: str,
    right: str,
    render: Any,
    ascii_mode: bool | None = None,
    style: Any = None,
) -> Table:
    """Two columns and a gutter, one row per aligned step."""
    dash = dash_for(ascii_mode)
    grid = _grid(title)
    grid.add_column("#", justify="right", style="dim", no_wrap=True)
    grid.add_column(left, overflow="fold")
    grid.add_column("", justify="center", no_wrap=True)
    grid.add_column(right, overflow="fold")

    for index, row in enumerate(alignment.rows):
        accent = _op_style(row.op)
        grid.add_row(
            str(index + 1),
            _side(row.left, render, dash, accent, style),
            Text(row.glyph, style=accent or "dim"),
            _side(row.right, render, dash, accent, style),
        )
    return grid


def _side(value: Any, render: Any, dash: str, accent: str, style: Any) -> Text:
    if value is None:
        return Text(dash, style="dim")
    own = style(value) if style is not None else ""
    return Text(render(value), style=accent or own)


def _op_style(op: str) -> str:
    return {EQUAL: "", REPLACE: "red", INSERT: "red", DELETE: "cyan"}[op]


# --------------------------------------------------------------------------- #
# Rendering: a run, and the coverage gaps
# --------------------------------------------------------------------------- #


def render_run(
    run: RunRecord,
    candidates: Sequence[CandidateRecord],
    executions: Sequence[ExecutionRecord],
    *,
    ascii_mode: bool | None = None,
    limit: int = 0,
) -> RenderableType:
    """The run as a list of cells, worst first.

    Worst first, because the reason anyone opens this view is to find the next
    execution to drill into - and sorting by case id would put a page of passes
    above the one failure they are looking for.

    The duration column appears only for a run scheduled in latency mode, for
    the same reason the scorecard refuses a p95 column there: under concurrency
    these numbers are the scheduler's queueing, and a table that invites a
    reader to compare them row by row would be inviting them to compare noise.
    A single execution's own wall time is still shown by
    :func:`render_execution`, where nothing is being compared.
    """
    sym = symbols(ascii_mode)
    dash = dash_for(ascii_mode)
    parts: list[RenderableType] = [Text(""), _run_header(run, candidates), Text("")]

    ordered = sorted(executions, key=_severity)
    shown = ordered[:limit] if limit else ordered

    grid = _grid("executions")
    grid.add_column("exec", overflow="fold")
    grid.add_column("outcome", no_wrap=True)
    grid.add_column("steps", justify="right", no_wrap=True)
    timed = run.scheduling == "latency"
    if timed:
        grid.add_column("ms", justify="right", no_wrap=True)
    # One line per execution: this table exists to be skimmed for the next
    # thing to open, and a wrapped error message turns ten rows into a page.
    grid.add_column("error", overflow="ellipsis", no_wrap=True, style="dim")
    for execution in shown:
        cells: list[RenderableType] = [
            Text(_readable_exec_id(execution)),
            Text(
                execution.outcome_class.value,
                style=OUTCOME_STYLE.get(execution.outcome_class, "red"),
            ),
            Text(str(execution.steps), style="dim"),
        ]
        if timed:
            cells.append(Text(f"{execution.duration_ms:,.0f}", style="dim"))
        cells.append(Text(_error_summary(execution) or dash))
        grid.add_row(*cells)
    parts.append(grid)
    if limit and len(ordered) > limit:
        parts += [
            Text(""),
            Text(
                f"  {sym.skip} {len(ordered) - limit} more - pass --limit 0 for all",
                style="dim",
            ),
        ]
    parts += [
        Text(""),
        Text(
            f"  {sym.arrow} slmeval show exec <id>   one execution's timeline\n"
            f"  {sym.arrow} slmeval show case <id> --candidate <id>   side by side "
            "against the baseline",
            style="dim",
        ),
    ]
    return Group(*parts)


def _run_header(run: RunRecord, candidates: Sequence[CandidateRecord]) -> Text:
    out = Text("  ")
    out.append(run.name, style="bold")
    out.append(
        f"  {run.run_id} · {run.status.value} · {run.mode} · seed {run.seed} · "
        f"{run.runs_per_case} runs/case",
        style="dim",
    )
    out.append("\n  ")
    out.append(
        " · ".join(
            f"{c.candidate_id}={c.provider}/{c.model}"
            + (" (baseline)" if c.role == "baseline" else "")
            for c in candidates
        ),
        style="dim",
    )
    out.append(f"\n  dataset {run.dataset_hash} · config {run.config_hash}", style="dim")
    if run.git_sha:
        out.append(f" · {run.git_sha}", style="dim")
    return out


def _error_summary(execution: ExecutionRecord) -> str:
    """One short line about how an execution failed.

    A :class:`~slmeval.sandbox.tool.CassetteMiss` carries the whole diagnosis
    block as its message, so its first line is that block's header - case,
    candidate and node, every one of which is already a column here. For those
    the class is shown and the block is left to ``show exec``, which has the
    room to print it properly.
    """
    message = execution.error_message or ""
    if "CASSETTE_MISS" in message:
        return execution.error_class or "cassette miss"
    if execution.error_class and message:
        return _short(f"{execution.error_class}: {message}", 60)
    return _short(execution.error_class or message, 60)


def _severity(execution: ExecutionRecord) -> tuple[int, str, str, int]:
    rank = {
        Outcome.HARNESS_ERROR: 0,
        Outcome.FORBIDDEN_TOOL: 1,
        Outcome.CASSETTE_MISS: 2,
        Outcome.SCHEMA_FAILURE: 3,
        Outcome.FAILED_ASSERTION: 4,
        Outcome.STEP_BUDGET: 5,
        Outcome.TOOL_ERROR: 6,
        Outcome.TIMEOUT: 7,
        Outcome.CRASHED: 8,
        Outcome.PROVIDER_ERROR: 9,
        Outcome.UNCOVERED: 10,
        Outcome.UNCLASSIFIED: 11,
        Outcome.PASSED: 12,
    }
    return (
        rank.get(execution.outcome_class, 11),
        execution.candidate_id,
        execution.case_id,
        execution.repeat_idx,
    )


def _readable_exec_id(execution: ExecutionRecord) -> str:
    """The part of an exec id a user types back at us.

    ``resolve_exec_id`` matches on a suffix for exactly this reason: the run id
    prefix is noise once you are already looking at one run's table.
    """
    run_prefix, _, rest = execution.exec_id.partition(".")
    return rest or run_prefix


def render_gaps(
    found: Sequence[Gap], *, ascii_mode: bool | None = None, run_id: str = ""
) -> RenderableType:
    """What the cassettes do not cover, and the command that fixes it."""
    sym = symbols(ascii_mode)
    if not found:
        return Group(
            Text(""),
            Text(
                f"  {sym.ok} every tool call in this run was covered by a recording",
                style="green",
            ),
            Text(""),
        )

    ours = sum(1 for gap in found if gap.uncovered)
    theirs = len(found) - ours
    # The title names what is in the table. A list that includes hallucinated
    # calls must not be headed "uncovered", or a user reads the whole thing as
    # recordings to go and make - and re-records a mangled order id as ground
    # truth.
    grid = _grid("cassette misses" if theirs else "uncovered tool calls")
    grid.add_column("case", no_wrap=True)
    grid.add_column("call", overflow="fold")
    grid.add_column("verdict", no_wrap=True)
    grid.add_column("found by", overflow="fold", style="dim")
    grid.add_column("execs", justify="right", style="dim", no_wrap=True)
    for gap in found:
        grid.add_row(
            Text(gap.case_id),
            Text(gap.render(width=70)),
            # Cyan, not red: an uncovered path is a hole in our recordings. The
            # three hallucination verdicts are the model's doing and are the
            # only ones coloured as failures.
            Text(gap.verdict, style="cyan" if gap.uncovered else "red"),
            Text(", ".join(gap.candidates)),
            Text(str(gap.executions)),
        )

    tail = Text()
    tail.append(
        f"  {ours} coverage gap(s) of ours, {theirs} hallucinated call(s)\n", style="dim"
    )
    if ours:
        tail.append(
            f"  {sym.arrow} slmeval record --fill-gaps"
            + (f" --run-id {run_id}" if run_id else ""),
            style="cyan",
        )
    return Group(Text(""), grid, Text(""), tail, Text(""))


# --------------------------------------------------------------------------- #
# JSON
# --------------------------------------------------------------------------- #


def diff_payload(diff: Diff) -> dict[str, Any]:
    """The diff as data, for ``--json``.

    Same shape discipline as :mod:`slmeval.report.export`: nothing is inferred,
    a missing side is ``null`` rather than an empty string, and the op names
    are the vocabulary ``difflib`` uses so a consumer can look them up.
    """
    return {
        "case_id": diff.case_id,
        "baseline_id": diff.baseline_id,
        "candidate_id": diff.candidate_id,
        "comparable": diff.comparable,
        "identical": diff.identical,
        "fault_parity": diff.fault_parity,
        "summary": diff.summary(),
        "baseline_exec_id": diff.baseline.execution.exec_id if diff.baseline else None,
        "candidate_exec_id": diff.candidate.execution.exec_id,
        "baseline_outcome": (
            diff.baseline.execution.outcome_class.value if diff.baseline else None
        ),
        "candidate_outcome": diff.candidate.execution.outcome_class.value,
        "trajectory": [
            {"op": row.op, "baseline": row.left, "candidate": row.right}
            for row in diff.trajectory.rows
        ],
        "tool_calls": [
            {
                "op": row.op,
                "baseline": _tool_payload(row.left),
                "candidate": _tool_payload(row.right),
            }
            for row in diff.tools.rows
        ],
        "diagnoses": [
            {
                "tool": finding.tool_name,
                "arguments": dict(finding.arguments),
                "verdict": finding.verdict.value,
                "uncovered": finding.uncovered,
                "rationale": finding.rationale,
                "nearest": (
                    dict(finding.nearest_arguments)
                    if finding.nearest_arguments is not None
                    else None
                ),
                "diffs": [
                    {
                        "argument": d.name,
                        "recorded": None if d.recorded_missing else d.recorded,
                        "called": None if d.called_missing else d.called,
                        "note": d.note,
                    }
                    for d in finding.diffs
                ],
                "block": finding.render(ascii_mode=True),
            }
            for finding in diff.diagnoses
        ],
    }


def _tool_payload(step: ToolStep | None) -> dict[str, Any] | None:
    if step is None:
        return None
    return {
        "seq": step.seq,
        "tool": step.name,
        "arguments": step.arguments,
        "node": step.node or None,
        "outcome": step.outcome,
        "match_tier": step.match_tier,
        "fault": step.fault,
        "miss_verdict": step.miss_verdict,
    }


def execution_payload(detail: ExecutionDetail) -> dict[str, Any]:
    """One execution as data, payloads resolved."""
    execution = detail.execution
    return {
        "exec_id": execution.exec_id,
        "run_id": execution.run_id,
        "candidate_id": execution.candidate_id,
        "case_id": execution.case_id,
        "repeat_idx": execution.repeat_idx,
        "attempt": execution.attempt,
        "status": execution.status.value,
        "outcome": execution.outcome_class.value,
        "judgeable": execution.outcome_class not in NOT_THE_MODELS_FAULT,
        "steps": execution.steps,
        "duration_ms": execution.duration_ms,
        "cost_usd": execution.cost_usd,
        "tokens_estimated": execution.tokens_estimated,
        "fault_digest": execution.fault_digest,
        "fault_vector": codec.loads(execution.fault_vector_json, []),
        "trajectory": list(trajectory_of(detail)),
        "warnings": list(codec.decode_strings(execution.warnings_json)),
        "error_class": execution.error_class,
        "error_message": execution.error_message,
        "model_calls": [
            {
                "seq": call.seq,
                "node": call.node,
                "model": call.model,
                "schema_state": call.schema_state,
                "structured_mode": call.structured_mode,
                "repair_attempts": call.repair_attempts,
                "json_stage": call.json_stage,
                "retries": call.retries,
                "latency_ms": call.latency_ms,
                "error": call.error,
            }
            for call in node_steps(detail)
        ],
        "tool_calls": [_tool_payload(step) for step in tool_steps(detail)],
        "metrics": [
            {
                "tier": row.tier,
                "metric": row.metric,
                "value": row.value_num,
                "text": row.value_text,
                "detail": codec.loads(row.detail_json, None),
            }
            for row in detail.evaluations
        ],
        "output": detail.output,
        "state": detail.state,
    }


def gaps_payload(found: Sequence[Gap]) -> dict[str, Any]:
    return {
        "gaps": [
            {
                "case_id": gap.case_id,
                "tool": gap.tool_name,
                "arguments": gap.arguments,
                "verdict": gap.verdict,
                "uncovered": gap.uncovered,
                "candidates": list(gap.candidates),
                "executions": gap.executions,
                "exec_id": gap.exec_id,
            }
            for gap in found
        ],
        "coverage_gaps": sum(1 for gap in found if gap.uncovered),
        "hallucinated_calls": sum(1 for gap in found if not gap.uncovered),
    }


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


def _grid(title: str) -> Table:
    return Table(
        box=None,
        pad_edge=False,
        padding=(0, 2),
        header_style="bold",
        show_edge=False,
        title=title,
        title_style="bold dim",
        title_justify="left",
    )


def _schema_cell(state: str, dash: str) -> Text:
    style = {
        "valid": "green",
        "invalid": "red",
        "error": "red",
        "not_requested": "dim",
    }.get(state, "")
    return Text(dash if state == "not_requested" else state, style=style)


def _mode_style(mode: str | None) -> str:
    """A fallback rung is a finding, so it is not printed in the same colour as
    the mechanism the model was asked for."""
    if mode is None:
        return "dim"
    return "" if mode in {"tools", "json_schema"} else "yellow"


def _stage_style(stage: str | None) -> str:
    if stage is None or stage == "direct":
        return "dim"
    return "yellow"


def _tool_outcome_style(step: ToolStep) -> str:
    if step.missed:
        return "cyan" if step.uncovered else "red"
    return {"replayed": "green", "recorded": "dim", "fault": "magenta"}.get(
        step.outcome, ""
    )


def _tool_step_style(step: ToolStep) -> str:
    return _tool_outcome_style(step)


def _short(text: str, width: int) -> str:
    """Truncate with a marker, never silently.

    An elided payload that looked complete would have a user comparing two
    calls that differ only past the cut.
    """
    if not text:
        return ""
    if len(text) <= width:
        return text
    # ASCII dots rather than an ellipsis glyph: this lands in table cells that
    # a legacy Windows codepage has to be able to encode, and `--ascii` exists
    # for the box drawing, not for every string the tool prints.
    return text[: max(0, width - 3)] + "..."


__all__ = [
    "ARG_WIDTH",
    "DELETE",
    "EQUAL",
    "INSERT",
    "OUTCOME_STYLE",
    "REPLACE",
    "Alignment",
    "Diff",
    "Gap",
    "NodeStep",
    "Pair",
    "ToolStep",
    "align",
    "build_diff",
    "diff_payload",
    "execution_payload",
    "gaps",
    "gaps_payload",
    "known_tool_names",
    "node_steps",
    "rebuild_diagnosis",
    "render_diff",
    "render_execution",
    "render_gaps",
    "render_run",
    "tool_steps",
    "trajectory_of",
]
