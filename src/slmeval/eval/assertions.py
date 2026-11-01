"""The dataset's own checks: did this execution actually do the right thing?

"Did it pass?" has to be a mechanical question. Every statistic downstream -
``pass^k``, the Wilson interval, the gate's verdict - is built on this boolean,
so if it is fuzzy, none of them mean anything.

Two design decisions carry most of the weight.

**A missing key is not a ``None`` value.** An agent that never set
``refund_eligible`` and an agent that set it to ``None`` are different
failures, and a checker that cannot tell them apart will report the first as
the second and send the user looking in the wrong place. Hence the
:data:`MISSING` sentinel.

**``forbidden_tools`` is reported apart from everything else.** A 4B model that
issues a refund when asked "where is my order" is not a two-point accuracy
regression, it is a production incident. It fails the case *and* raises an
``unsafe_action`` count that the scorecard prints above the table, where it
cannot be averaged away or scrolled past.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from slmeval.dataset.models import Assertions


class _Missing:
    """Sentinel for "this path does not exist in the state".

    A class rather than ``None``, because ``None`` is a legitimate value an
    agent can set and the difference between "unset" and "set to null" is a
    different bug with a different fix.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<missing>"

    def __bool__(self) -> bool:
        return False


MISSING: Final = _Missing()

CHECK_STATE: Final = "state"
CHECK_CONTAINS: Final = "state_contains"
CHECK_ANSWER: Final = "final_answer_regex"
CHECK_REQUIRED_TOOLS: Final = "required_tools"
CHECK_FORBIDDEN_TOOLS: Final = "forbidden_tools"
CHECK_MAX_TOOL_CALLS: Final = "max_tool_calls"


@dataclass(frozen=True, slots=True)
class Failure:
    """One assertion that did not hold, in terms the user can act on."""

    check: str
    path: str = ""
    expected: Any = None
    actual: Any = None
    message: str = ""

    def describe(self) -> str:
        if self.message:
            return self.message
        where = f" at {self.path}" if self.path else ""
        return f"{self.check}{where}: expected {self.expected!r}, got {self.actual!r}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "path": self.path,
            "expected": _plain(self.expected),
            "actual": _plain(self.actual),
            "message": self.describe(),
        }


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """Everything the dataset had to say about one execution."""

    failures: tuple[Failure, ...] = ()
    checks: int = 0
    forbidden_called: tuple[str, ...] = ()
    missing_tools: tuple[str, ...] = ()
    empty: bool = False
    """The case asserted nothing, so it cannot fail. Reported rather than
    quietly passing: an all-green column produced by an empty assertion block
    is the most expensive kind of wrong."""

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def unsafe(self) -> bool:
        """A forbidden tool was called. A hard fail and its own counter."""
        return bool(self.forbidden_called)

    def describe(self) -> tuple[str, ...]:
        return tuple(f.describe() for f in self.failures)

    def as_dict(self) -> dict[str, Any]:
        return {
            "checks": self.checks,
            "failures": [f.as_dict() for f in self.failures],
            "forbidden_called": list(self.forbidden_called),
            "missing_tools": list(self.missing_tools),
            "empty": self.empty,
        }


def resolve_path(state: Any, path: str) -> Any:
    """Read a dotted path out of a final state, or :data:`MISSING`.

    Supports mapping keys and list indices - ``items.0.sku`` - because an agent
    whose state holds a list of line items should be assertable without
    restructuring it for the benchmark. A key containing a literal dot is
    matched first, so a flat state with ``"order.status"`` as an actual key
    still works.
    """
    if path in _keys(state):
        return _get(state, path)

    current = state
    for part in path.split("."):
        if isinstance(current, Mapping):
            if part not in current:
                return MISSING
            current = current[part]
            continue
        if isinstance(current, Sequence) and not isinstance(current, str | bytes):
            try:
                index = int(part)
            except ValueError:
                return MISSING
            if not -len(current) <= index < len(current):
                return MISSING
            current = current[index]
            continue
        attribute = getattr(current, part, MISSING)
        if attribute is MISSING:
            return MISSING
        current = attribute
    return current


def _keys(state: Any) -> tuple[Any, ...]:
    return tuple(state.keys()) if isinstance(state, Mapping) else ()


def _get(state: Any, key: str) -> Any:
    return state[key] if isinstance(state, Mapping) else MISSING


def contains(actual: Any, expected: Any) -> bool:
    """Whether ``expected`` is "in" ``actual``, per type.

    One operator spelled four ways, because the alternative is four assertion
    keys in the dataset schema and users having to know which one their field
    needs. Strings are substrings, collections are membership, mappings are
    subsets, and anything else falls back to equality - which is what makes
    ``state_contains = { refund_issued = false }`` mean what a reader expects.
    """
    if actual is MISSING:
        return False
    if isinstance(expected, bool) or expected is None:
        # `True in [1, 2]` is True in Python. A boolean assertion must mean the
        # value, never membership.
        return bool(actual == expected)
    if isinstance(actual, str):
        return str(expected) in actual
    if isinstance(actual, Mapping) and isinstance(expected, Mapping):
        return all(k in actual and actual[k] == v for k, v in expected.items())
    if isinstance(actual, Mapping):
        return expected in actual
    if isinstance(actual, Sequence) and not isinstance(actual, bytes):
        return expected in actual
    if isinstance(actual, set | frozenset):
        return expected in actual
    return bool(actual == expected)


def check_assertions(
    assertions: Assertions,
    *,
    state: Mapping[str, Any] | None,
    tools_called: Sequence[str] = (),
) -> AssertionResult:
    """Run every check the case declares.

    Every one of them, and the failures are collected rather than short-
    circuited: a user fixing a case wants the whole list, and the drill-down
    view shows all of them side by side with the baseline's.

    ``state is None`` means the agent never returned - it raised, or timed out.
    The state checks then fail by *absence* rather than being skipped, because
    a crashed execution that skipped its assertions would be indistinguishable
    from one with nothing to assert.
    """
    failures: list[Failure] = []
    checks = 0
    called = list(tools_called)

    for path, expected in assertions.state.items():
        checks += 1
        actual = resolve_path(state, path) if state is not None else MISSING
        if actual is MISSING or actual != expected:
            failures.append(
                Failure(check=CHECK_STATE, path=path, expected=expected, actual=actual)
            )

    for path, expected in assertions.state_contains.items():
        checks += 1
        actual = resolve_path(state, path) if state is not None else MISSING
        if not contains(actual, expected):
            failures.append(
                Failure(
                    check=CHECK_CONTAINS,
                    path=path,
                    expected=expected,
                    actual=actual,
                    message=(
                        f"state_contains at {path}: {expected!r} not found in {actual!r}"
                        if actual is not MISSING
                        else f"state_contains at {path}: the agent never set this key"
                    ),
                )
            )

    if assertions.final_answer_regex is not None:
        checks += 1
        answer = _answer_of(state)
        if answer is None or not re.search(assertions.final_answer_regex, answer):
            failures.append(
                Failure(
                    check=CHECK_ANSWER,
                    path="final_answer",
                    expected=assertions.final_answer_regex,
                    actual=answer,
                    message=(
                        "final_answer_regex: the agent produced no final answer"
                        if answer is None
                        else f"final_answer_regex {assertions.final_answer_regex!r} "
                        f"did not match {_truncate(answer)!r}"
                    ),
                )
            )

    missing_tools: tuple[str, ...] = ()
    if assertions.required_tools:
        checks += 1
        missing_tools = tuple(t for t in assertions.required_tools if t not in called)
        if missing_tools:
            failures.append(
                Failure(
                    check=CHECK_REQUIRED_TOOLS,
                    expected=list(assertions.required_tools),
                    actual=called,
                    message=(
                        f"required tool(s) never called: {', '.join(missing_tools)}"
                    ),
                )
            )

    forbidden_called: tuple[str, ...] = ()
    if assertions.forbidden_tools:
        checks += 1
        forbidden_called = tuple(
            dict.fromkeys(t for t in called if t in set(assertions.forbidden_tools))
        )
        if forbidden_called:
            failures.append(
                Failure(
                    check=CHECK_FORBIDDEN_TOOLS,
                    expected=[],
                    actual=list(forbidden_called),
                    message=(
                        f"FORBIDDEN tool called: {', '.join(forbidden_called)} - "
                        "a hard fail, counted as an unsafe action"
                    ),
                )
            )

    if assertions.max_tool_calls is not None:
        checks += 1
        if len(called) > assertions.max_tool_calls:
            failures.append(
                Failure(
                    check=CHECK_MAX_TOOL_CALLS,
                    expected=assertions.max_tool_calls,
                    actual=len(called),
                    message=(
                        f"{len(called)} tool calls, over the ceiling of "
                        f"{assertions.max_tool_calls} - usually a loop"
                    ),
                )
            )

    return AssertionResult(
        failures=tuple(failures),
        checks=checks,
        forbidden_called=forbidden_called,
        missing_tools=missing_tools,
        empty=assertions.is_empty,
    )


def _answer_of(state: Mapping[str, Any] | None) -> str | None:
    if not isinstance(state, Mapping):
        return None
    for key in ("final_answer", "answer", "output", "response"):
        value = state.get(key)
        if isinstance(value, str):
            return value
    return None


def _truncate(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else f"{text[:limit]}..."


def _plain(value: Any) -> Any:
    """JSON-safe form of a value, for the stored detail blob."""
    if value is MISSING:
        return None
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, str | bytes):
        return [_plain(v) for v in value]
    return str(value)


__all__ = [
    "CHECK_ANSWER",
    "CHECK_CONTAINS",
    "CHECK_FORBIDDEN_TOOLS",
    "CHECK_MAX_TOOL_CALLS",
    "CHECK_REQUIRED_TOOLS",
    "CHECK_STATE",
    "MISSING",
    "AssertionResult",
    "Failure",
    "check_assertions",
    "contains",
    "resolve_path",
]
