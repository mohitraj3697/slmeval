"""Tier 2: trajectory and state, per execution, $0, under 10ms.

Tier 1 says whether the answer was right. Tier 2 says whether the agent got
there the same way - which is the part that tells you *where* to intervene, and
it is the difference between "gemma3:4b scores 83%" and "gemma3:4b skips the
policy check on refund cases".

Four decisions worth defending.

**The reference is a set of accepted paths, not one golden path.** Agents
legitimately have several valid routes - an order that needs no lookup skips
two nodes - and scoring against a single path manufactures failures for
behaviour nobody thinks is wrong. Distance is the minimum over the accepted
set.

**The edit script is the output, not the distance.** ``0.33`` is a number;
``policy_check -> (missing)`` is a finding. The normalized distance is what a
table sorts on and the script is what makes a failure explainable, so both are
stored.

**Tool matching is on ``(name, arg_match_tier)``, not on name alone.** A
candidate that called ``fetch_order`` with arguments that only matched a
recording after normalization did not make the same call the baseline made. It
reached the right tool with the wrong arguments, and precision that ignored
that would hide the most common small-model failure there is.

**A loop is reported as suspected, not proven.** The plan's definition is "a
repeated node cycle with no change in the asserted state keys", and we hold
only the *final* state - there are no per-step snapshots to compare. So a
repeated cycle whose tool calls are redundant is flagged, and the flag says
what it is based on. A metric that overstated its evidence would be worse than
one that admits its limit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from slmeval.core.trace import ToolCallTrace
from slmeval.core.verdict import Metric
from slmeval.sandbox.keys import MatchTier, canonicalize

TIER: Final = 2

EXACT: Final = MatchTier.EXACT.value

MATCH = "match"
SUBSTITUTE = "substitute"
INSERT = "insert"
DELETE = "delete"


# --------------------------------------------------------------------------- #
# Trajectory distance
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Edit:
    """One step of the alignment between a reference path and what happened."""

    op: str
    index: int
    reference: str | None = None
    candidate: str | None = None

    @property
    def divergent(self) -> bool:
        return self.op != MATCH

    def render(self, arrow: str = "->") -> str:
        if self.op == MATCH:
            return f"  {self.reference}"
        if self.op == SUBSTITUTE:
            return f"~ {self.reference} {arrow} {self.candidate}"
        if self.op == INSERT:
            return f"+ (nothing) {arrow} {self.candidate}"
        return f"- {self.reference} {arrow} (missing)"

    def as_dict(self) -> dict[str, Any]:
        return {
            "op": self.op,
            "index": self.index,
            "reference": self.reference,
            "candidate": self.candidate,
        }


@dataclass(frozen=True, slots=True)
class Alignment:
    """How far a trajectory is from a reference, and exactly where."""

    distance: int
    reference: tuple[str, ...] = ()
    candidate: tuple[str, ...] = ()
    script: tuple[Edit, ...] = ()

    @property
    def normalized(self) -> float:
        """Distance over the longer sequence, so 0.0 is identical and 1.0 is
        "nothing in common". Dividing by the reference alone would let a
        candidate that visited forty nodes score better than one that visited
        five."""
        span = max(len(self.reference), len(self.candidate))
        return self.distance / span if span else 0.0

    @property
    def identical(self) -> bool:
        return self.distance == 0

    @property
    def divergences(self) -> tuple[Edit, ...]:
        return tuple(e for e in self.script if e.divergent)

    def render(self, arrow: str = "->") -> str:
        return "\n".join(e.render(arrow) for e in self.script)

    def as_dict(self) -> dict[str, Any]:
        return {
            "distance": self.distance,
            "normalized": round(self.normalized, 4),
            "reference": list(self.reference),
            "candidate": list(self.candidate),
            "script": [e.as_dict() for e in self.divergences],
        }


def align(reference: Sequence[str], candidate: Sequence[str]) -> Alignment:
    """Levenshtein distance with the edit script that produced it.

    The full matrix rather than the two-row trick, because the backtrace is the
    point: without it this returns a number nobody can act on. Trajectories are
    tens of nodes long, so the memory is irrelevant.

    Ties are resolved substitute, then delete, then insert, which keeps the
    script stable across runs - a snapshot test of a rendered diff has to be
    reproducible.
    """
    ref = tuple(reference)
    cand = tuple(candidate)
    rows, cols = len(ref) + 1, len(cand) + 1

    cost = [[0] * cols for _ in range(rows)]
    for i in range(1, rows):
        cost[i][0] = i
    for j in range(1, cols):
        cost[0][j] = j
    for i in range(1, rows):
        for j in range(1, cols):
            if ref[i - 1] == cand[j - 1]:
                cost[i][j] = cost[i - 1][j - 1]
            else:
                cost[i][j] = 1 + min(
                    cost[i - 1][j - 1],  # substitute
                    cost[i - 1][j],  # delete from reference
                    cost[i][j - 1],  # insert into candidate
                )

    script: list[Edit] = []
    i, j = len(ref), len(cand)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref[i - 1] == cand[j - 1]:
            script.append(Edit(MATCH, i - 1, ref[i - 1], cand[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and cost[i][j] == cost[i - 1][j - 1] + 1:
            script.append(Edit(SUBSTITUTE, i - 1, ref[i - 1], cand[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and cost[i][j] == cost[i - 1][j] + 1:
            script.append(Edit(DELETE, i - 1, ref[i - 1], None))
            i -= 1
        else:
            script.append(Edit(INSERT, max(i - 1, 0), None, cand[j - 1]))
            j -= 1

    script.reverse()
    return Alignment(
        distance=cost[len(ref)][len(cand)],
        reference=ref,
        candidate=cand,
        script=tuple(script),
    )


def closest(
    candidate: Sequence[str], accepted: Sequence[Sequence[str]]
) -> Alignment | None:
    """The best alignment over every accepted reference path.

    ``None`` when the case declares no reference trajectory. Not a distance of
    zero and not a distance of one: with nothing to compare against there is no
    measurement, and inventing either value would put a fabricated number in a
    column users read as measured.
    """
    options = [tuple(path) for path in accepted if path]
    if not options:
        return None
    return min((align(path, candidate) for path in options), key=lambda a: a.normalized)


# --------------------------------------------------------------------------- #
# Tool-call comparison
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Reference:
    """What the baseline did on one case, as the yardstick for candidates.

    Taken from an actual baseline execution rather than hand-written: the
    question a candidate answers is "did you do what the model we trust did",
    and a reference typed out by hand drifts from the agent the first time
    anybody edits it.
    """

    case_id: str
    calls: tuple[tuple[str, str], ...] = ()
    """``(tool_name, match_tier)`` in order."""
    trajectory: tuple[str, ...] = ()
    steps: int = 0
    tool_calls: int = 0

    @classmethod
    def from_trace(
        cls,
        case_id: str,
        *,
        tool_calls: Sequence[ToolCallTrace],
        trajectory: Sequence[str],
        steps: int = 0,
    ) -> Reference:
        return cls(
            case_id=case_id,
            calls=tuple(signature_of(t) for t in tool_calls),
            trajectory=tuple(trajectory),
            steps=steps,
            tool_calls=len(tool_calls),
        )

    @property
    def empty(self) -> bool:
        return not self.calls and not self.trajectory


def signature_of(trace: ToolCallTrace) -> tuple[str, str]:
    """The identity a tool call is compared on.

    ``match_tier`` is part of it. A recorded or live call is ``exact`` by
    definition - it *is* the reference - and a replayed call carries the tier
    at which its arguments matched. So a candidate whose arguments only matched
    after a normalizer compares unequal to the baseline's exact call, which is
    the honest answer: it reached the right tool with the wrong arguments.
    """
    return (trace.name, trace.match_tier or EXACT)


@dataclass(frozen=True, slots=True)
class ToolOverlap:
    """Precision, recall and F1 over the multiset of tool calls."""

    matched: int = 0
    candidate_calls: int = 0
    reference_calls: int = 0
    extra: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()

    @property
    def precision(self) -> float | None:
        if not self.candidate_calls:
            return None
        return self.matched / self.candidate_calls

    @property
    def recall(self) -> float | None:
        if not self.reference_calls:
            return None
        return self.matched / self.reference_calls

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if p is None or r is None or p + r == 0:
            return None
        return 2 * p * r / (p + r)

    def as_dict(self) -> dict[str, Any]:
        return {
            "matched": self.matched,
            "candidate_calls": self.candidate_calls,
            "reference_calls": self.reference_calls,
            "extra": list(self.extra),
            "missing": list(self.missing),
        }


def overlap(
    candidate: Sequence[tuple[str, str]], reference: Sequence[tuple[str, str]]
) -> ToolOverlap:
    """Multiset intersection, so repeated calls count the number of times made.

    A multiset and not a set: an agent that calls ``fetch_order`` three times
    where the baseline called it once has done something worth seeing, and set
    semantics would score that as a perfect match.
    """
    cand = Counter(candidate)
    ref = Counter(reference)
    shared = cand & ref
    return ToolOverlap(
        matched=sum(shared.values()),
        candidate_calls=len(candidate),
        reference_calls=len(reference),
        extra=tuple(_labels(cand - ref)),
        missing=tuple(_labels(ref - cand)),
    )


def _labels(counter: Counter[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    for (name, tier), count in sorted(counter.items()):
        label = name if tier == EXACT else f"{name}[{tier}]"
        out.append(label if count == 1 else f"{label} x{count}")
    return out


# --------------------------------------------------------------------------- #
# Redundancy and loops
# --------------------------------------------------------------------------- #


def redundancy(tool_calls: Sequence[ToolCallTrace]) -> float:
    """Fraction of calls that repeat a call already made, by tool and arguments.

    "Identical" is decided by :func:`~slmeval.sandbox.keys.canonicalize`, the
    same function that decides whether two calls share a cassette. Two
    definitions of identical would eventually disagree, and then a run could be
    simultaneously fully cached and highly redundant.
    """
    if not tool_calls:
        return 0.0
    seen: set[tuple[str, str]] = set()
    repeats = 0
    for trace in tool_calls:
        key = (trace.name, canonicalize(trace.arguments or {}, tool_name=trace.name))
        if key in seen:
            repeats += 1
        else:
            seen.add(key)
    return repeats / len(tool_calls)


@dataclass(frozen=True, slots=True)
class Loop:
    """A repeated node cycle, and what it is evidence of."""

    cycle: tuple[str, ...]
    repeats: int
    redundant_tools: bool

    @property
    def suspected_stall(self) -> bool:
        """A cycle that also made no new tool calls.

        The strongest claim the available data supports. Proving a stall needs
        per-step state snapshots, which we do not keep; a cycle that re-made
        calls it had already made is the state-free evidence that it achieved
        nothing on the second pass.
        """
        return self.redundant_tools

    def as_dict(self) -> dict[str, Any]:
        return {
            "cycle": list(self.cycle),
            "repeats": self.repeats,
            "redundant_tools": self.redundant_tools,
            "suspected_stall": self.suspected_stall,
        }


def find_loop(trajectory: Sequence[str], *, redundant: bool = False) -> Loop | None:
    """The shortest node cycle that repeats back to back, if any.

    Shortest and adjacent: ``a b c b c`` is a ``b c`` loop, and reporting the
    whole trajectory as "repeated" would be true and useless. An agent that
    revisits a node much later is branching, not looping, so only immediate
    repetition counts.
    """
    nodes = list(trajectory)
    n = len(nodes)
    for size in range(1, n // 2 + 1):
        for start in range(0, n - 2 * size + 1):
            window = nodes[start : start + size]
            repeats = 1
            cursor = start + size
            while nodes[cursor : cursor + size] == window:
                repeats += 1
                cursor += size
            if repeats >= 2:
                return Loop(
                    cycle=tuple(window), repeats=repeats, redundant_tools=redundant
                )
    return None


# --------------------------------------------------------------------------- #
# The tier
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Tier2:
    """Trajectory and state findings for one execution."""

    alignment: Alignment | None = None
    tools: ToolOverlap = field(default_factory=ToolOverlap)
    redundancy: float = 0.0
    loop: Loop | None = None
    steps: int = 0
    reference_steps: int | None = None
    faults_met: int = 0
    faults_applied: int = 0
    trajectory: tuple[str, ...] = ()

    @property
    def trajectory_distance(self) -> float | None:
        return self.alignment.normalized if self.alignment else None

    @property
    def step_delta(self) -> int | None:
        if self.reference_steps is None:
            return None
        return self.steps - self.reference_steps

    @property
    def diverged(self) -> bool:
        return self.alignment is not None and not self.alignment.identical

    def metrics(self) -> tuple[Metric, ...]:
        out: list[Metric] = [
            Metric(TIER, "steps", float(self.steps)),
            Metric(TIER, "tool_redundancy", self.redundancy),
        ]
        if self.alignment is not None:
            out.append(
                Metric(
                    TIER,
                    "trajectory_distance",
                    self.alignment.normalized,
                    # The rendered script, stored with the number: the raw edit
                    # script is what makes a failure explainable, and a distance
                    # without it is a number nobody can act on.
                    text=", ".join(e.render() for e in self.alignment.divergences) or None,
                    detail=self.alignment.as_dict(),
                )
            )
        # The baseline's first run of a case sets the reference, so it has nothing
        # to be compared with. Scoring it against an empty reference would give
        # the run everything else is measured by a tool precision of 0.
        compared = self.reference_steps is not None
        for name, value in (
            ("tool_precision", self.tools.precision),
            ("tool_recall", self.tools.recall),
            ("tool_f1", self.tools.f1),
        ):
            if compared and value is not None:
                out.append(Metric(TIER, name, value))
        if compared and (self.tools.reference_calls or self.tools.candidate_calls):
            out.append(
                Metric(
                    TIER,
                    "tool_overlap",
                    None,
                    text=_overlap_text(self.tools),
                    detail=self.tools.as_dict(),
                )
            )
        if self.step_delta is not None:
            out.append(Metric(TIER, "step_delta", float(self.step_delta)))
        if self.loop is not None:
            out.append(
                Metric(
                    TIER,
                    "loop_detected",
                    1.0,
                    text=f"{' > '.join(self.loop.cycle)} x{self.loop.repeats}",
                    detail=self.loop.as_dict(),
                )
            )
        if self.faults_met:
            # The denominator of the recovery rate. Per execution, because a
            # candidate that stopped after two calls never met the fault
            # scheduled for the fourth - the rate has to be over faults
            # actually encountered.
            out.append(Metric(TIER, "faults_met", float(self.faults_met)))
        return tuple(out)

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.loop is not None and self.loop.suspected_stall:
            out.append(
                f"suspected stall: {' > '.join(self.loop.cycle)} repeated "
                f"{self.loop.repeats}x with no new tool calls"
            )
        if self.redundancy >= 0.5 and self.tools.candidate_calls > 1:
            out.append(
                f"{self.redundancy:.0%} of tool calls repeated a call already made"
            )
        return tuple(out)


def tier2(
    *,
    tool_calls: Sequence[ToolCallTrace],
    trajectory: Sequence[str],
    steps: int = 0,
    accepted_trajectories: Sequence[Sequence[str]] = (),
    reference: Reference | None = None,
    faults_met: int = 0,
    faults_applied: int = 0,
) -> Tier2:
    """Compute every tier-2 finding for one execution.

    The accepted trajectories come from the dataset and the :class:`Reference`
    from the baseline's own run. Both are optional, and a missing one produces
    ``None`` rather than a default: a trajectory distance of zero against a
    reference that does not exist would be the most flattering possible
    fabrication.
    """
    accepted = [tuple(path) for path in accepted_trajectories if path]
    if not accepted and reference is not None and reference.trajectory:
        # No declared paths, but the baseline took one. Using it is the whole
        # idea of a baseline - and it means a user gets trajectory distance
        # without hand-writing reference paths for every case.
        accepted = [reference.trajectory]

    rate = redundancy(tool_calls)
    return Tier2(
        alignment=closest(tuple(trajectory), accepted),
        tools=overlap(
            [signature_of(t) for t in tool_calls],
            list(reference.calls) if reference else [],
        ),
        redundancy=rate,
        loop=find_loop(trajectory, redundant=rate > 0.0),
        steps=steps,
        reference_steps=reference.steps if reference else None,
        faults_met=faults_met,
        faults_applied=faults_applied,
        trajectory=tuple(trajectory),
    )


def _overlap_text(tools: ToolOverlap) -> str | None:
    parts: list[str] = []
    if tools.missing:
        parts.append(f"missing: {', '.join(tools.missing)}")
    if tools.extra:
        parts.append(f"extra: {', '.join(tools.extra)}")
    return " · ".join(parts) or None


__all__ = [
    "DELETE",
    "INSERT",
    "MATCH",
    "SUBSTITUTE",
    "TIER",
    "Alignment",
    "Edit",
    "Loop",
    "Reference",
    "Tier2",
    "ToolOverlap",
    "align",
    "closest",
    "find_loop",
    "overlap",
    "redundancy",
    "signature_of",
    "tier2",
]
