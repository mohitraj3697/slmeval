"""Tier 1: deterministic, per execution, $0, under 2ms.

Roughly 80% of the signal, computed from the trace with no model in the loop.
Everything here is a count or a ratio over what already happened, which is why
it can be recomputed from stored rows without re-running anything.

The headline pair is :attr:`Tier1.schema_valid_rate` and
:attr:`Tier1.schema_repaired_rate`, and they are separate on purpose. A small
model that "passes" because our JSON cascade dug an object out of a prose
preamble and then re-asked once has not demonstrated schema adherence - it has
demonstrated that our parser is forgiving. Folding the two together would make
every candidate look better than it is, and would erase the clearest
small-model-versus-frontier signal the tool produces:

    gemma3:4b-q4 needed json_mode on 100% of calls, prompt-repair on 12%

Three rules the rest of the module follows:

- **A call that never produced a response is not a schema failure.** A 500 or a
  timeout counts against the execution and is excluded from the schema
  denominator, because an adherence rate diluted with unrelated crashes means
  nothing.
- **Leniency is recorded, never silent.** Every rung past the first and every
  extraction stage past ``direct`` is counted and surfaced.
- **A forbidden tool call is its own number.** Not a failed assertion among
  others - a separate ``unsafe_action`` count the scorecard prints above the
  table.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, get_args, get_origin

from slmeval.core.trace import JsonStage, LLMCallTrace, SchemaStatus, ToolCallTrace
from slmeval.core.verdict import Metric
from slmeval.dataset.models import Assertions
from slmeval.eval.assertions import AssertionResult, check_assertions
from slmeval.sandbox.keys import MatchTier
from slmeval.sandbox.tool import SandboxedTool, registered_tools

TIER: Final = 1

JUDGEABLE_SCHEMA: Final = frozenset({SchemaStatus.VALID, SchemaStatus.INVALID})
"""Schema states that belong in the adherence denominator.

``NOT_REQUESTED`` had no schema to adhere to, and ``ERROR`` never produced a
candidate response. Counting either would make the rate a function of how
often the provider fell over.
"""

#: Annotations we are willing to type-check, by the string the module's
#: ``from __future__ import annotations`` leaves behind. Deliberately short:
#: a wrong "hallucinated argument type" finding is worse than no finding, so
#: anything not listed here is simply not checked and says so in ``arg_checks``.
_SIMPLE_TYPES: Final[dict[str, type | tuple[type, ...]]] = {
    "str": str,
    "int": int,
    "float": (int, float),
    "bool": bool,
    "bytes": bytes,
    "dict": dict,
    "list": list,
    "tuple": tuple,
    "set": set,
}


@dataclass(frozen=True, slots=True)
class Tier1:
    """Deterministic findings about one execution."""

    # --- structured output -------------------------------------------------
    schema_calls: int = 0
    """Calls that asked for a schema and got a response. The denominator."""
    schema_first_try: int = 0
    schema_repaired: int = 0
    schema_invalid: int = 0
    schema_errors: int = 0
    """Calls that never produced a response. Counted, excluded from the rates."""

    rung_histogram: dict[str, int] = field(default_factory=dict)
    stage_histogram: dict[str, int] = field(default_factory=dict)
    repair_attempts: int = 0

    # --- tool use ----------------------------------------------------------
    tool_calls: int = 0
    tool_name_invalid: int = 0
    required_args_missing: int = 0
    arg_type_invalid: int = 0
    extra_args_hallucinated: int = 0
    arg_checks: int = 0
    """Arguments whose type we could actually verify. Printed with the rate, so
    "100% valid" cannot be read as a strong claim when nothing was checkable."""
    degraded_matches: int = 0
    """Cassette hits that needed a normalizer or a subset fallback. The model
    reached the recorded call, but not with the recorded arguments."""
    missed_calls: int = 0

    # --- the dataset -------------------------------------------------------
    assertions: AssertionResult = field(default_factory=AssertionResult)
    tools_called: tuple[str, ...] = ()
    problems: tuple[str, ...] = ()

    # ------------------------------------------------------------------ rates

    @property
    def schema_valid_rate(self) -> float | None:
        """Fraction that validated on the **first attempt at the first rung**.

        ``None`` when no call asked for a schema - not zero. A run with no
        structured output has no adherence rate, and printing 0% would read as
        total failure.
        """
        return self._rate(self.schema_first_try)

    @property
    def schema_repaired_rate(self) -> float | None:
        """Fraction that validated only with help: a lower rung, a lenient
        extraction stage, or a repair re-ask."""
        return self._rate(self.schema_repaired)

    @property
    def schema_failure_rate(self) -> float | None:
        return self._rate(self.schema_invalid)

    def _rate(self, count: int) -> float | None:
        if not self.schema_calls:
            return None
        return count / self.schema_calls

    @property
    def tool_name_valid_rate(self) -> float | None:
        if not self.tool_calls:
            return None
        return (self.tool_calls - self.tool_name_invalid) / self.tool_calls

    @property
    def required_args_present_rate(self) -> float | None:
        if not self.tool_calls:
            return None
        return (self.tool_calls - self.required_args_missing) / self.tool_calls

    @property
    def arg_type_valid_rate(self) -> float | None:
        if not self.arg_checks:
            return None
        return (self.arg_checks - self.arg_type_invalid) / self.arg_checks

    @property
    def degraded_match_rate(self) -> float | None:
        if not self.tool_calls:
            return None
        return self.degraded_matches / self.tool_calls

    @property
    def unsafe_actions(self) -> int:
        return len(self.assertions.forbidden_called)

    @property
    def passed_assertions(self) -> bool:
        return self.assertions.ok

    @property
    def lenient(self) -> bool:
        """Whether anything about this execution leaned on our forgiveness."""
        return bool(self.schema_repaired or self.degraded_matches or self.repair_attempts)

    # ---------------------------------------------------------------- output

    def metrics(self) -> tuple[Metric, ...]:
        """The rows this tier contributes to ``evaluations``.

        Counts *and* rates. The rate is what a table prints; the count is what
        makes a rate from three calls distinguishable from a rate from three
        hundred, and anyone aggregating across executions needs the count.
        """
        out: list[Metric] = [
            Metric(TIER, "schema_calls", float(self.schema_calls)),
            Metric(TIER, "tool_calls", float(self.tool_calls)),
            Metric(TIER, "repair_attempts", float(self.repair_attempts)),
        ]
        for name, value in (
            ("schema_valid_rate", self.schema_valid_rate),
            ("schema_repaired_rate", self.schema_repaired_rate),
            ("schema_failure_rate", self.schema_failure_rate),
            ("tool_name_valid", self.tool_name_valid_rate),
            ("required_args_present", self.required_args_present_rate),
            ("arg_type_valid", self.arg_type_valid_rate),
            ("cassette_degraded_rate", self.degraded_match_rate),
        ):
            if value is not None:
                out.append(Metric(TIER, name, value))

        out.append(
            Metric(TIER, "arg_checks", float(self.arg_checks))
        )
        out.append(
            Metric(
                TIER,
                "extra_args_hallucinated",
                float(self.extra_args_hallucinated),
            )
        )
        if self.schema_errors:
            out.append(Metric(TIER, "schema_call_errors", float(self.schema_errors)))
        if self.stage_histogram:
            out.append(
                Metric(
                    TIER,
                    "json_extraction_stage",
                    None,
                    text=_dominant(self.stage_histogram),
                    detail=dict(self.stage_histogram),
                )
            )
        if self.rung_histogram:
            out.append(
                Metric(
                    TIER,
                    "structured_mode_used",
                    None,
                    text=_dominant(self.rung_histogram),
                    detail=dict(self.rung_histogram),
                )
            )
        out.append(
            Metric(
                TIER,
                "assertions_passed",
                1.0 if self.assertions.ok else 0.0,
                text=None if self.assertions.ok else "; ".join(self.assertions.describe()),
                detail=self.assertions.as_dict() if not self.assertions.ok else None,
            )
        )
        if self.unsafe_actions:
            out.append(
                Metric(
                    TIER,
                    "unsafe_action",
                    float(self.unsafe_actions),
                    text=", ".join(self.assertions.forbidden_called),
                    detail={"tools": list(self.assertions.forbidden_called)},
                )
            )
        if self.missed_calls:
            out.append(Metric(TIER, "cassette_misses", float(self.missed_calls)))
        return tuple(out)

    def warnings(self) -> tuple[str, ...]:
        """What the user must see even if this execution passed."""
        out: list[str] = []
        if self.schema_repaired:
            out.append(
                f"{self.schema_repaired}/{self.schema_calls} structured call(s) only "
                "validated after a fallback rung or a repair re-ask"
            )
        if self.degraded_matches:
            out.append(
                f"{self.degraded_matches} tool call(s) matched a cassette only after "
                "normalization - the arguments were not what was recorded"
            )
        if self.assertions.empty:
            out.append("this case asserts nothing, so it cannot fail")
        out.extend(self.problems)
        return tuple(out)


def tier1(
    *,
    assertions: Assertions,
    llm_calls: Sequence[LLMCallTrace],
    tool_calls: Sequence[ToolCallTrace],
    state: Mapping[str, Any] | None,
    tools: Mapping[str, SandboxedTool] | None = None,
) -> Tier1:
    """Compute every tier-1 finding for one execution."""
    registry = registered_tools() if tools is None else tools

    first_try = repaired = invalid = errors = judgeable = 0
    rungs: dict[str, int] = {}
    stages: dict[str, int] = {}
    repairs = 0

    for call in llm_calls:
        repairs += call.repair_attempts
        if call.schema_status is SchemaStatus.NOT_REQUESTED:
            continue
        if call.schema_status is SchemaStatus.ERROR:
            errors += 1
            continue
        judgeable += 1
        if call.structured_mode_used:
            rungs[call.structured_mode_used] = rungs.get(call.structured_mode_used, 0) + 1
        if call.json_stage is not None:
            stages[call.json_stage.value] = stages.get(call.json_stage.value, 0) + 1

        if call.schema_status is SchemaStatus.INVALID:
            invalid += 1
        elif _clean_first_attempt(call):
            first_try += 1
        else:
            repaired += 1

    names = tuple(t.name for t in tool_calls)
    inspected = [_inspect_call(t, registry) for t in tool_calls]

    return Tier1(
        schema_calls=judgeable,
        schema_first_try=first_try,
        schema_repaired=repaired,
        schema_invalid=invalid,
        schema_errors=errors,
        rung_histogram=rungs,
        stage_histogram=stages,
        repair_attempts=repairs,
        tool_calls=len(tool_calls),
        tool_name_invalid=sum(1 for c in inspected if not c.name_valid),
        required_args_missing=sum(1 for c in inspected if c.missing_required),
        arg_type_invalid=sum(c.bad_types for c in inspected),
        extra_args_hallucinated=sum(c.extra_args for c in inspected),
        arg_checks=sum(c.checked for c in inspected),
        degraded_matches=sum(1 for t in tool_calls if _degraded(t)),
        missed_calls=sum(1 for t in tool_calls if t.outcome == "miss"),
        assertions=check_assertions(assertions, state=state, tools_called=names),
        tools_called=tuple(dict.fromkeys(names)),
        problems=tuple(
            f"{c.tool_name}: {note}" for c in inspected for note in c.notes
        ),
    )


def _clean_first_attempt(call: LLMCallTrace) -> bool:
    """Whether a valid response needed no help at all.

    All four conditions, because each one is a different way a model can fail
    to hold up its end while still ending in a valid object: a lower rung, a
    re-ask, a preamble our parser had to dig through, or a response our repair
    prompt rescued. Any of them and the model did not adhere to the schema - we
    did.
    """
    return (
        call.schema_status is SchemaStatus.VALID
        and call.repair_attempts == 0
        and len(call.rungs_tried) <= 1
        and (call.json_stage is None or call.json_stage is JsonStage.DIRECT)
    )


def _degraded(trace: ToolCallTrace) -> bool:
    return trace.match_tier is not None and trace.match_tier != MatchTier.EXACT.value


@dataclass(frozen=True, slots=True)
class _CallCheck:
    tool_name: str
    name_valid: bool = True
    missing_required: bool = False
    bad_types: int = 0
    extra_args: int = 0
    checked: int = 0
    notes: tuple[str, ...] = ()


def _inspect_call(
    trace: ToolCallTrace, registry: Mapping[str, SandboxedTool]
) -> _CallCheck:
    """Check one tool call against the tool's own signature.

    The signature is the only ground truth available for free: it says which
    arguments exist, which are required, and what they should be. A model that
    invents a parameter or passes a string where an int belongs has made a
    specific, nameable mistake, and naming it is worth far more to the user
    than one more point off an accuracy score.
    """
    tool = registry.get(trace.name)
    if tool is None:
        # Reached a name that is not a tool at all. The sandbox only traces
        # decorated functions, so in practice this comes from a miss diagnosed
        # as a hallucinated tool.
        return _CallCheck(
            tool_name=trace.name,
            name_valid=False,
            notes=(f"{trace.name!r} is not a registered tool",),
        )
    if trace.miss_verdict == "hallucinated_tool":
        return _CallCheck(tool_name=trace.name, name_valid=False)
    if tool.signature is None:
        return _CallCheck(tool_name=trace.name)

    arguments = trace.arguments or {}
    required: list[str] = []
    accepts_extra = False
    known: set[str] = set()
    bad = 0
    checked = 0
    notes: list[str] = []

    for name, parameter in tool.signature.parameters.items():
        if name in {"self", "cls"}:
            continue
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            accepts_extra = True
            continue
        known.add(name)
        if parameter.default is inspect.Parameter.empty:
            required.append(name)
        if name not in arguments:
            continue
        expected = _annotation_type(parameter.annotation)
        if expected is None:
            continue
        checked += 1
        if not _type_ok(arguments[name], expected):
            bad += 1
            notes.append(
                f"{name}={arguments[name]!r} is a {type(arguments[name]).__name__}, "
                f"not {_annotation_name(parameter.annotation)}"
            )

    missing = [name for name in required if name not in arguments]
    if missing:
        notes.append(f"missing required argument(s): {', '.join(missing)}")
    extra = [] if accepts_extra else [name for name in arguments if name not in known]
    if extra:
        notes.append(f"argument(s) the tool does not accept: {', '.join(extra)}")

    return _CallCheck(
        tool_name=trace.name,
        missing_required=bool(missing),
        bad_types=bad,
        extra_args=len(extra),
        checked=checked,
        notes=tuple(notes),
    )


def _annotation_type(annotation: Any) -> type | tuple[type, ...] | None:
    """What to check an argument against, or ``None`` for "do not check".

    Conservative by design. The tool modules use
    ``from __future__ import annotations``, so annotations arrive as strings,
    and resolving them properly would mean importing the user's module
    namespace at scoring time. A short table of builtins covers the cases where
    a type error is a real model mistake; everything else is skipped, and
    ``arg_checks`` says how much was skipped. A wrong finding here would be
    worse than a missing one.
    """
    if annotation is inspect.Parameter.empty or annotation is Any:
        return None
    if isinstance(annotation, type):
        return annotation if annotation is not object else None
    if isinstance(annotation, str):
        return _annotation_from_text(annotation)
    origin = get_origin(annotation)
    if origin is not None:
        if origin is type(None):  # pragma: no cover - defensive
            return None
        args = get_args(annotation)
        if args and type(None) in args:
            # Optional: checking it would reject the legitimate None.
            return None
        return origin if isinstance(origin, type) else None
    return None


def _annotation_from_text(text: str) -> type | tuple[type, ...] | None:
    cleaned = text.strip()
    if "|" in cleaned or cleaned.startswith(("Optional", "Union")):
        # A union is one more thing that can be wrong in a way we cannot verify
        # cheaply, and `int | None` must accept None.
        return None
    head = cleaned.split("[", 1)[0].strip()
    return _SIMPLE_TYPES.get(head)


def _annotation_name(annotation: Any) -> str:
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", str(annotation))


def _type_ok(value: Any, expected: type | tuple[type, ...]) -> bool:
    """Whether a value matches an annotation, with ``bool``/``int`` kept apart.

    ``bool`` is a subclass of ``int`` in Python, so a plain ``isinstance``
    would accept ``True`` for an ``int`` parameter and ``1`` for a ``bool``
    one. Those are exactly the confusions a small model makes when filling a
    schema, so they are the ones worth catching.
    """
    if expected is bool:
        return isinstance(value, bool)
    if expected is int:
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == (int, float):
        return isinstance(value, int | float) and not isinstance(value, bool)
    return isinstance(value, expected)


def _dominant(histogram: Mapping[str, int]) -> str:
    """The most frequent key, ties broken by name so reports are stable."""
    return max(sorted(histogram), key=lambda k: (histogram[k], k)) if histogram else ""


__all__ = ["JUDGEABLE_SCHEMA", "TIER", "Tier1", "tier1"]
