"""A cassette miss, turned into a diagnosis.

A strict-replay miss raised as an exception string answers the wrong question.
The user knows *that* the tool call was not recorded - they got an exception.
What they need to know is which of four very different things happened, because
each one asks something different of them:

==========================  ==================================================
``hallucinated_parameter``  the model mangled an argument
``hallucinated_tool``       the model invented a tool
``wrong_tool_for_step``     a real tool, at the wrong moment
``unrecorded_valid_path``   plausible, but never recorded
==========================  ==================================================

The first three are findings about the model. The fourth is a gap in *our*
coverage, and keeping it separate is the whole point of this module. A harness
that reports "the SLM did something wrong" when the truth is "the SLM did
something new that we have no ground truth for" inflates its own failure
numbers, and anyone who checks one of those failures by hand will stop trusting
every other number in the table. So the uncovered verdict is the *safe
default*: a miss is called a hallucination only when the evidence for one is
positive - a recognizable mangle of a call we did record - and otherwise it is
reported as uncovered, with ``slmeval record --fill-gaps`` as the fix.

The block this produces is also, bluntly, the most marketable output the tool
has::

    ✗ refund_104 · gemma3:4b · node=order_lookup · CASSETTE_MISS
      tool: fetch_order_from_erp
      called with: {"order_id": "9921"}
      nearest recorded: {"order_id": "ORD-9921"}
      diff: order_id: "ORD-9921" → "9921"   (dropped prefix)
      verdict: hallucinated_parameter

It is what gets screenshotted into a pull request, so it is built as a
first-class view - structured fields, a nearest-neighbour search, a per-argument
diff with a plain-English reason - and not as an error path.

The nearest-neighbour search is a heuristic and is documented as one. It ranks
the recorded calls for this ``(case, tool)`` by argument distance and shows the
closest. It can be wrong about *which* recorded call the model was reaching
for; it cannot be wrong about what was called and what exists, which is the
part the user acts on.
"""

from __future__ import annotations

import json
import sys
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from difflib import SequenceMatcher, get_close_matches
from enum import StrEnum
from typing import Any, Final

from slmeval.sandbox.store import Cassette, CassetteStore

NEAREST_LIMIT: Final = 50
"""How many recorded calls the nearest search considers.

A case with more than fifty recordings of one tool is a loop, and the fiftieth
is not going to be the one the model was reaching for. Bounded so a failure
path cannot become slow on a pathological case.
"""

MANGLE_DISTANCE: Final = 0.15
"""Above this average per-argument distance, a miss is reported as uncovered
rather than as a hallucination.

This is the backstop for a corruption :data:`MANGLES` has no name for - a
transposition, a one-character typo - so it is deliberately tight. It has to
be: identifiers in one system share their prefix, so ``"ORD-7733"`` against a
recorded ``"ORD-9921"`` scores 50% similar while being a perfectly ordinary
*different order*. At anything near 0.5 the harness accuses a model of
hallucinating every time it legitimately looks up a second record, which is
both wrong and the most damaging way to be wrong.

0.15 means the arguments must be ~85% similar before the backstop fires, which
in practice is a typo rather than a different value. Erring tight costs a false
"uncovered", which reads as "record more"; erring loose puts a hallucination in
the report that was never there.
"""

_MIN_STRING_DISTANCE: Final = 0.05
"""Two unequal strings are never distance zero, however similar. Without a
floor, a one-character mangle of a long identifier would score as a perfect
match and the nearest search would have nothing to rank by."""


class Verdict(StrEnum):
    """What a miss means. See the module docstring for why there are four."""

    HALLUCINATED_PARAMETER = "hallucinated_parameter"
    """The tool is real and was recorded for this case; the arguments are a
    recognizable corruption of a call we have. The model's mistake."""

    HALLUCINATED_TOOL = "hallucinated_tool"
    """The tool name is not a sandboxed tool and has never been recorded. The
    model invented the name."""

    WRONG_TOOL_FOR_STEP = "wrong_tool_for_step"
    """A real tool, but nothing in this case was ever recorded for it while
    other tools were. The model reached for the wrong instrument."""

    UNRECORDED_VALID_PATH = "unrecorded_valid_path"
    """Tool and arguments are plausible and nothing says otherwise. **Not a
    model failure** - a hole in our coverage, reported separately and fixed by
    re-recording rather than by blaming the candidate."""

    @property
    def uncovered(self) -> bool:
        """True when this verdict is about our coverage, not the model.

        The gate reads this: an uncovered miss must not count against a
        candidate's pass rate, or the scorecard punishes a model for going
        somewhere the baseline never went.
        """
        return self is Verdict.UNRECORDED_VALID_PATH


class _Missing:
    """Sentinel type for "this argument was not passed at all".

    A class rather than ``object()`` so the type checker can tell it from a
    legitimate value, and distinct from ``None`` because a model that passes
    ``None`` explicitly has made a different mistake from one that omitted the
    argument entirely.
    """

    def __repr__(self) -> str:
        return "(not passed)"


_MISSING: Final = _Missing()

MANGLES: Final = frozenset(
    {
        "dropped prefix",
        "added prefix",
        "dropped suffix",
        "added suffix",
        "case differs",
        "whitespace differs",
        "argument not passed",
        "unexpected argument",
        "digits only",
    }
)
"""Notes that count as positive evidence of a mangled argument.

Deliberately *not* in the set: "different value" and "type changed". A
different value is exactly what a model exploring a legitimately new path
produces, and a type change is as often a tool-signature mismatch on our side
as a model error. Treating either as proof of hallucination is how a harness
starts inventing findings.
"""


@dataclass(frozen=True, slots=True)
class ArgDiff:
    """One argument that differs between the call and the nearest recording.

    ``note`` is the plain-English reason, and it is what makes the block worth
    reading: ``order_id: "ORD-9921" → "9921"`` tells a user something is wrong,
    while ``(dropped prefix)`` tells them what it was.
    """

    name: str
    recorded: Any
    called: Any
    note: str
    recorded_missing: bool = False
    called_missing: bool = False

    @property
    def mangle(self) -> bool:
        """True when this difference corrupts the recorded value rather than
        simply being a different value.

        Positive evidence of a hallucination: one of these is enough to call a
        miss a parameter hallucination regardless of the distance threshold.
        """
        return self.note in MANGLES

    def render(self, arrow: str = "→") -> str:
        left = "(not passed)" if self.recorded_missing else _pretty(self.recorded)
        right = "(not passed)" if self.called_missing else _pretty(self.called)
        return f"{self.name}: {left} {arrow} {right}"


@dataclass(frozen=True, slots=True)
class MissDiagnosis:
    """A structured cassette miss: what was called, what exists, what it means.

    Built by :func:`diagnose` at the moment of the miss, while the store is
    open and the node is known. Carried by
    :class:`~slmeval.sandbox.tool.CassetteMiss` so the runner can persist it
    rather than re-deriving it from an exception string.
    """

    case_id: str
    tool_name: str
    arguments: Mapping[str, Any]
    verdict: Verdict
    rationale: str
    """Why this verdict and not another, in one sentence. In the block, so a
    user who disagrees with a classification can see what drove it instead of
    guessing at our thresholds."""

    nearest: Cassette | None = None
    nearest_arguments: Mapping[str, Any] | None = None
    """Decoded once here rather than per render: the envelope decode is the
    expensive part, and the block may be rendered to a terminal and to a log."""
    distance: float | None = None
    diffs: tuple[ArgDiff, ...] = ()
    recorded_calls: int = 0
    """Exact-tier recordings of this tool in this case. Zero is itself the
    finding for :attr:`Verdict.WRONG_TOOL_FOR_STEP`."""
    candidate_id: str | None = None
    node: str | None = None
    node_path: tuple[str, ...] = ()
    suggestions: tuple[str, ...] = ()
    """Close tool names, for an invented one. A model that emitted
    ``fetch_order_from_erps`` should not send a user hunting."""
    also_recorded: tuple[str, ...] = ()
    """Other tools recorded for this case. Shown for a wrong-tool verdict,
    because "you called X; we recorded Y and Z here" is the whole diagnosis."""

    @property
    def uncovered(self) -> bool:
        return self.verdict.uncovered

    def render(self, *, ascii_mode: bool | None = None) -> str:
        """The diagnosis block, as plain text.

        Plain text rather than a Rich renderable on purpose: this string is
        also the message of the raised
        :class:`~slmeval.sandbox.tool.CassetteMiss`, so it has to be readable
        in a log, a pytest failure and a traceback - none of which render
        markup. :func:`slmeval.cli.render.render_miss` colours these same lines
        for the terminal.
        """
        cross, arrow = glyphs(ascii_mode)
        where = " · ".join(
            part
            for part in (
                self.case_id,
                self.candidate_id,
                f"node={self.node}" if self.node else None,
            )
            if part
        )
        lines = [
            f"{cross} {where} · CASSETTE_MISS",
            f"  tool: {self.tool_name}",
            f"  called with: {_pretty_args(self.arguments)}",
        ]

        if self.nearest_arguments is not None:
            lines.append(f"  nearest recorded: {_pretty_args(self.nearest_arguments)}")
        for index, diff in enumerate(self.diffs):
            label = "  diff: " if index == 0 else "        "
            note = f"   ({diff.note})" if diff.note else ""
            lines.append(f"{label}{diff.render(arrow)}{note}")

        if self.suggestions:
            lines.append(f"  did you mean: {', '.join(self.suggestions)}")
        if self.verdict is Verdict.WRONG_TOOL_FOR_STEP and self.also_recorded:
            lines.append(f"  recorded here instead: {', '.join(self.also_recorded)}")

        lines.append(f"  verdict: {self.verdict.value}")
        lines.append(f"  {arrow} {self.rationale}")
        if self.uncovered:
            # The fix for an uncovered path is ours, not the model's, so the
            # block says so rather than leaving the user to infer it from a
            # verdict name.
            lines.append(f"  {arrow} {FILL_GAPS}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.render()


FILL_GAPS: Final = (
    "no ground truth here, so this is a coverage gap rather than a model "
    "failure - `slmeval record --fill-gaps` extends coverage"
)


# --------------------------------------------------------------------------- #
# Argument distance
# --------------------------------------------------------------------------- #


def _value_distance(called: Any, recorded: Any) -> float:
    """How far apart two argument values are, in ``[0, 1]``.

    Total and type-tolerant, like the normalizers and for the same reason: this
    runs on whatever the model emitted, including values of entirely the wrong
    type, and it runs on a path that is already a failure. Raising here would
    replace the user's finding with a harness crash.
    """
    if called is _MISSING or recorded is _MISSING:
        return 1.0
    if type(called) is type(recorded) and called == recorded:
        return 0.0
    if isinstance(called, str) and isinstance(recorded, str):
        ratio = SequenceMatcher(None, _fold(called), _fold(recorded)).ratio()
        return max(_MIN_STRING_DISTANCE, 1.0 - ratio)
    if isinstance(called, bool) or isinstance(recorded, bool):
        # Before the numeric branch: `True == 1` in Python, and a model passing
        # `True` where `1` was recorded has made a type mistake, not a tiny
        # numeric one.
        return 0.0 if called is recorded else 1.0
    if isinstance(called, int | float) and isinstance(recorded, int | float):
        scale = max(abs(float(called)), abs(float(recorded)), 1.0)
        return min(1.0, abs(float(called) - float(recorded)) / scale)
    if isinstance(called, Mapping) and isinstance(recorded, Mapping):
        names = set(called) | set(recorded)
        return sum(
            _value_distance(called.get(n, _MISSING), recorded.get(n, _MISSING)) for n in names
        ) / max(1, len(names))
    if _is_seq(called) and _is_seq(recorded):
        left, right = list(called), list(recorded)
        longest = max(len(left), len(right), 1)
        pairs = list(zip(left, right, strict=False))
        matched = sum(_value_distance(a, b) for a, b in pairs)
        # A length mismatch is a full penalty per unpaired element, so [1, 2]
        # against [1, 2, 3] is not scored as a perfect match.
        return min(1.0, (matched + (longest - len(pairs))) / longest)
    return 1.0


def _is_seq(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, str | bytes)


def _fold(text: str) -> str:
    """NFC, for the same reason the cassette key normalizes: a composed and a
    decomposed "café" are the same text and would otherwise rank as
    different."""
    return unicodedata.normalize("NFC", text)


def argument_distance(called: Mapping[str, Any], recorded: Mapping[str, Any]) -> float:
    """Average per-argument distance between a call and a recording.

    Averaged rather than summed, so a tool with eight arguments and one mangled
    one does not rank worse than a tool with one argument that is entirely
    different. The ranking has to be comparable across tools because the block
    shows exactly one nearest neighbour.
    """
    names = set(called) | set(recorded)
    if not names:
        return 0.0
    total = sum(
        _value_distance(called.get(name, _MISSING), recorded.get(name, _MISSING))
        for name in names
    )
    return total / len(names)


# --------------------------------------------------------------------------- #
# Why two values differ
# --------------------------------------------------------------------------- #


def _string_note(called: str, recorded: str) -> str:
    """The plain-English reason two strings differ.

    Ordered most-specific first. ``"ORD-9921"`` to ``"9921"`` is reported as a
    dropped prefix rather than as a case or whitespace difference, because the
    first thing a user wants is the shape of the mistake, not its most generic
    description.
    """
    if called == recorded:
        return ""
    if called and recorded:
        if recorded.endswith(called):
            return "dropped prefix"
        if called.endswith(recorded):
            return "added prefix"
        if recorded.startswith(called):
            return "dropped suffix"
        if called.startswith(recorded):
            return "added suffix"
    if called.strip() == recorded.strip():
        return "whitespace differs"
    if called.lower() == recorded.lower():
        return "case differs"
    if " ".join(called.split()) == " ".join(recorded.split()):
        return "whitespace differs"
    digits = "".join(c for c in recorded if c.isdigit())
    if digits and called == digits:
        # The classic SLM failure: it read the number out of an identifier and
        # dropped everything that made it an identifier.
        return "digits only"
    return "different value"


def diff_arguments(called: Mapping[str, Any], recorded: Mapping[str, Any]) -> tuple[ArgDiff, ...]:
    """Per-argument differences, in a stable order.

    Only the arguments that actually differ: a block that re-listed the eight
    arguments the model got right would bury the one it got wrong.
    """
    _, arrow = glyphs()
    diffs: list[ArgDiff] = []
    for name in sorted(set(called) | set(recorded)):
        has_called = name in called
        has_recorded = name in recorded
        left = recorded.get(name, _MISSING)
        right = called.get(name, _MISSING)
        if has_called and has_recorded and type(left) is type(right) and left == right:
            continue

        if not has_called:
            note = "argument not passed"
        elif not has_recorded:
            note = "unexpected argument"
        elif isinstance(right, str) and isinstance(left, str):
            note = _string_note(right, left)
        elif type(right) is not type(left):
            note = f"type changed: {type(left).__name__} {arrow} {type(right).__name__}"
        else:
            note = "different value"

        diffs.append(
            ArgDiff(
                name=name,
                recorded=left if has_recorded else None,
                called=right if has_called else None,
                note=note,
                recorded_missing=not has_recorded,
                called_missing=not has_called,
            )
        )
    return tuple(diffs)


# --------------------------------------------------------------------------- #
# Nearest neighbour
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Nearest:
    """The recorded call a miss is compared against."""

    cassette: Cassette
    arguments: Mapping[str, Any]
    distance: float


def nearest_recorded(
    arguments: Mapping[str, Any], candidates: Iterable[Cassette]
) -> Nearest | None:
    """The recorded call closest to ``arguments``.

    Ties keep the earliest recording, because ``neighbours()`` returns them in
    record order and a stable choice means the block is reproducible - the
    screenshot a user posts has to match what they see when they re-run.
    """
    best: Nearest | None = None
    for cassette in candidates:
        try:
            recorded = cassette.arguments()
        except Exception:
            # A cassette whose envelope no longer decodes (a renamed Pydantic
            # model, a deleted class) must not turn the diagnosis into a second
            # failure. Skip it and rank the rest.
            continue
        if not isinstance(recorded, Mapping):
            continue
        distance = argument_distance(arguments, recorded)
        if best is None or distance < best.distance:
            best = Nearest(cassette=cassette, arguments=recorded, distance=distance)
    return best


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def _classify(
    *,
    tool_name: str,
    known_tools: Iterable[str],
    recorded_anywhere: Iterable[str],
    recorded_here: int,
    nearest: Nearest | None,
    diffs: tuple[ArgDiff, ...],
) -> tuple[Verdict, str]:
    """Pick a verdict, and say in one sentence why.

    The order of these tests is the design. Tool existence is checked before
    anything about arguments, because arguments are meaningless for a tool that
    does not exist; and the parameter-hallucination test demands positive
    evidence, so everything that falls through lands on the uncovered verdict
    rather than on an accusation.
    """
    known = set(known_tools)
    anywhere = set(recorded_anywhere)

    if tool_name not in known and tool_name not in anywhere:
        return (
            Verdict.HALLUCINATED_TOOL,
            f"{tool_name!r} is not a sandboxed tool in this process and has never been "
            "recorded, so the model invented the name",
        )

    if not recorded_here:
        others = sorted(anywhere - {tool_name})
        if others:
            return (
                Verdict.WRONG_TOOL_FOR_STEP,
                f"{tool_name} is a real tool, but nothing in this case was recorded for "
                f"it while {len(others)} other tool(s) were",
            )
        return (
            Verdict.UNRECORDED_VALID_PATH,
            "nothing at all is recorded for this case, so there is nothing to compare "
            "against - record it before reading anything into this miss",
        )

    if nearest is None:
        return (
            Verdict.UNRECORDED_VALID_PATH,
            f"{recorded_here} call(s) to {tool_name} are recorded here, but none of them "
            "could be decoded for comparison",
        )

    mangled = [d for d in diffs if d.mangle]
    if mangled:
        return (
            Verdict.HALLUCINATED_PARAMETER,
            f"{', '.join(d.name for d in mangled)} differs from the nearest recorded call "
            f"in a way that corrupts it ({mangled[0].note}), which is a model error "
            "rather than a new path",
        )
    if nearest.distance <= MANGLE_DISTANCE:
        return (
            Verdict.HALLUCINATED_PARAMETER,
            f"the arguments are {(1.0 - nearest.distance) * 100:.0f}% similar to a "
            "recorded call, which reads as a corruption of it rather than a different "
            "call",
        )
    return (
        Verdict.UNRECORDED_VALID_PATH,
        f"{tool_name} is real and these arguments are not a recognizable corruption of "
        "anything recorded, so this is a path the baseline never took",
    )


def diagnose(
    *,
    case_id: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    store: CassetteStore | None = None,
    known_tools: Iterable[str] | None = None,
    candidate_id: str | None = None,
    node: str | None = None,
    node_path: tuple[str, ...] = (),
) -> MissDiagnosis:
    """Diagnose a replay miss.

    Runs on the failure path, so every lookup here is defensive: a store that
    cannot be read, a cassette that cannot be decoded, a tool registry that was
    never populated - each degrades the diagnosis rather than replacing the
    user's finding with one of ours.

    ``known_tools`` defaults to the live ``@sandbox_tool`` registry. It is a
    parameter so the classification is testable without declaring tools, and so
    a future out-of-process runner can pass the registry it recorded with.
    """
    if known_tools is None:
        # Imported here, not at module scope: `tool` imports this module to
        # build its exception, so a top-level import would be circular.
        from slmeval.sandbox.tool import registered_tools

        known_tools = tuple(registered_tools())

    neighbours: list[Cassette] = []
    recorded_anywhere: list[str] = []
    if store is not None:
        try:
            neighbours = store.neighbours(case_id, tool_name, limit=NEAREST_LIMIT)
            recorded_anywhere = store.tool_names(case_id)
        except Exception:
            # The store is already in trouble if this fails. The miss itself is
            # still a real finding and still has to be reported.
            neighbours = []
            recorded_anywhere = []

    nearest = nearest_recorded(arguments, neighbours)
    diffs = diff_arguments(arguments, nearest.arguments) if nearest else ()
    verdict, rationale = _classify(
        tool_name=tool_name,
        known_tools=known_tools,
        recorded_anywhere=recorded_anywhere,
        recorded_here=len(neighbours),
        nearest=nearest,
        diffs=diffs,
    )

    suggestions: tuple[str, ...] = ()
    if verdict is Verdict.HALLUCINATED_TOOL:
        pool = sorted(set(known_tools) | set(recorded_anywhere))
        suggestions = tuple(get_close_matches(tool_name, pool, n=2, cutoff=0.6))

    return MissDiagnosis(
        case_id=case_id,
        tool_name=tool_name,
        arguments=dict(arguments),
        verdict=verdict,
        rationale=rationale,
        nearest=nearest.cassette if nearest else None,
        nearest_arguments=nearest.arguments if nearest else None,
        distance=nearest.distance if nearest else None,
        diffs=diffs,
        recorded_calls=len(neighbours),
        candidate_id=candidate_id,
        node=node,
        node_path=node_path,
        suggestions=suggestions,
        also_recorded=tuple(n for n in sorted(recorded_anywhere) if n != tool_name),
    )


# --------------------------------------------------------------------------- #
# Rendering helpers
# --------------------------------------------------------------------------- #


def glyphs(ascii_mode: bool | None = None) -> tuple[str, str]:
    """``(cross, arrow)``, degraded on a console that cannot encode them.

    Deliberately not importing :mod:`slmeval.cli.render`: the sandbox runs
    inside the user's agent, and making it drag in the CLI layer is a worse
    trade than duplicating two glyphs.
    """
    if ascii_mode is None:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        try:
            "✗→".encode(encoding)
        except (LookupError, UnicodeEncodeError):
            ascii_mode = True
        else:
            ascii_mode = False
    return ("XX", "->") if ascii_mode else ("✗", "→")


def _pretty(value: Any) -> str:
    """One value, as JSON where possible and as ``repr`` where not.

    JSON because the block shows a tool call and that is how a tool call is
    written everywhere else the user sees one; ``repr`` as a fallback because a
    renderer on a failure path is not allowed to raise.
    """
    try:
        return json.dumps(value, ensure_ascii=False, default=repr)
    except (TypeError, ValueError):
        return repr(value)


def _pretty_args(arguments: Mapping[str, Any]) -> str:
    try:
        return json.dumps(dict(arguments), ensure_ascii=False, sort_keys=True, default=repr)
    except (TypeError, ValueError):
        return repr(dict(arguments))


__all__ = [
    "FILL_GAPS",
    "MANGLES",
    "MANGLE_DISTANCE",
    "NEAREST_LIMIT",
    "ArgDiff",
    "MissDiagnosis",
    "Nearest",
    "Verdict",
    "argument_distance",
    "diagnose",
    "diff_arguments",
    "glyphs",
    "nearest_recorded",
]
