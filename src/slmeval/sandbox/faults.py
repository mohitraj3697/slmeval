"""Deterministic fault injection.

The happy path is not what breaks SLM agents in production. What breaks them is
a ``429`` on step four, an empty list where a record was expected, or a payload
that contradicts what the agent was told two steps earlier - and the model not
noticing. Testing recovery is the differentiating measurement in this tool, and
it only means anything if every candidate faces the *same* failures at the same
places.

**The alignment bug this module exists to avoid.** The obvious implementation
keys faults on a single global tool-call counter: fault the 3rd call, the 7th,
and so on. Then candidate A makes one extra call to *any* tool, every later
fault shifts by one, and A and B are no longer running the same experiment -
while the table still prints them side by side as though they were. The
comparison is silently invalid, which is worse than being obviously broken.

So the schedule is **per tool name, indexed by that tool's own Nth call**::

    schedule.fault_for("fetch_order", 0)   # first call to fetch_order
    schedule.fault_for("search_kb", 0)     # first call to search_kb

An extra call to ``search_kb`` cannot move the fault on ``fetch_order``.

**Determinism comes from the hash, not from an order of evaluation.** Every
decision is a pure function of ``(seed, case_id, tool_name, call_index)``, so
the same seed yields byte-identical faults whether the schedule is precomputed
up front or consulted lazily, in any order, from any number of threads. The
precomputed table that :meth:`FaultSchedule.table` returns is built by calling
the same lazy function, so the two can never drift apart.

**The honest limitation, stated rather than hidden.** Under divergent
trajectories perfect fault parity is impossible: if candidate B never calls
``fetch_order`` a third time, it never meets the fault that candidate A met
there. That cannot be fixed, so it is *reported*. Every run records a
:class:`FaultVector` - the ordered list of faults actually injected, plus a
digest - and the comparison table flags any candidate whose vector differs from
the baseline's. ``--require-fault-parity`` discards the non-matching pairs
instead of pretending they are comparable.

Faults apply in ``replay`` mode, and in ``auto`` mode only when a cassette
answered. They never apply while recording: a faulted payload written to the
store would poison every later replay of that call, and the poison would be
indistinguishable from a real recording.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

SCHEDULE_VERSION: Final = "f1"
"""In the draw's hash. Bumped only if the taxonomy or the draw changes shape,
because bumping it invalidates every published fault vector - and a seed that
no longer reproduces a documented result is a reproducibility claim broken."""

_UNIT: Final = "\x1f"
"""ASCII unit separator, so a case id and a tool name cannot run together in the
hashed payload and alias onto each other's draws."""

DEFAULT_MAX_CALLS: Final = 64
"""How deep :meth:`FaultSchedule.table` precomputes per tool.

Only a display and testing bound. Lookup is lazy and unbounded, so a tool
called more often than this still gets correctly-seeded faults.
"""


class Delivery(StrEnum):
    """How a fault reaches the agent.

    Configurable per fault, and the plan says why: which one you choose changes
    what you are testing. A ``429`` raised as an exception tests the agent's
    backoff and retry wrapper. The same ``429`` returned as a payload tests
    whether the model *reads* its tool output - a completely different question,
    and the one SLMs fail more often.
    """

    RAISE = "raise"
    RETURN = "return"


class FaultKind(StrEnum):
    """The taxonomy.

    ``EMPTY_RESULT`` and ``SEMANTIC_CONFLICT`` are the two worth leading with:
    they are where small models fail most interestingly - inventing data to
    fill a gap, or carrying on happily with a payload that contradicts what
    they were told - and no competing harness looks at either.
    """

    RATE_LIMIT_429 = "rate_limit_429"
    """Tests backoff and retry."""

    TIMEOUT_504 = "timeout_504"
    """Raised after a real delay, so timeout handling is actually exercised
    rather than short-circuited."""

    MALFORMED_JSON = "malformed_json"
    """Garbage tool output. Does the agent handle it, or parse-and-pray?"""

    EMPTY_RESULT = "empty_result"
    """An empty list or object where data was expected. Does the model
    hallucinate data to fill the gap?"""

    PARTIAL_SCHEMA = "partial_schema"
    """A payload missing fields the agent assumes are present."""

    SEMANTIC_CONFLICT = "semantic_conflict"
    """Data contradicting earlier state. Does the model notice?"""

    LATENCY_SPIKE = "latency_spike"
    """A correct result, late. Tests step and time budget behaviour."""


_RAISEABLE: Final = frozenset(
    {FaultKind.RATE_LIMIT_429, FaultKind.TIMEOUT_504, FaultKind.MALFORMED_JSON}
)
"""Faults that can be delivered as an exception.

The others are payload shapes - "empty", "partial", "contradictory", "late" -
and have no exception form that still tests the same thing. A raised
``empty_result`` is just an error, and an agent that handles errors would pass
it while still hallucinating data on the real fault.
"""

_RETURNABLE: Final = frozenset(FaultKind) - {FaultKind.TIMEOUT_504}
"""Everything but a timeout can be delivered as a payload. A returned timeout
is a contradiction in terms: the call completed."""

DEFAULT_DELIVERY: Final[Mapping[FaultKind, Delivery]] = {
    FaultKind.RATE_LIMIT_429: Delivery.RAISE,
    FaultKind.TIMEOUT_504: Delivery.RAISE,
    FaultKind.MALFORMED_JSON: Delivery.RETURN,
    FaultKind.EMPTY_RESULT: Delivery.RETURN,
    FaultKind.PARTIAL_SCHEMA: Delivery.RETURN,
    FaultKind.SEMANTIC_CONFLICT: Delivery.RETURN,
    FaultKind.LATENCY_SPIKE: Delivery.RETURN,
}

DEFAULT_WEIGHTS: Final[Mapping[FaultKind, float]] = {
    FaultKind.RATE_LIMIT_429: 1.0,
    FaultKind.TIMEOUT_504: 1.0,
    FaultKind.MALFORMED_JSON: 1.0,
    FaultKind.EMPTY_RESULT: 2.0,
    FaultKind.SEMANTIC_CONFLICT: 2.0,
    FaultKind.PARTIAL_SCHEMA: 1.0,
    FaultKind.LATENCY_SPIKE: 1.0,
}
"""Relative frequency of each fault.

``empty_result`` and ``semantic_conflict`` are weighted double because they are
the interesting failures and because an even split over seven kinds puts barely
any of either into a ten-case dataset.
"""

DEFAULT_DELAYS_MS: Final[Mapping[FaultKind, float]] = {
    FaultKind.TIMEOUT_504: 2_000.0,
    FaultKind.LATENCY_SPIKE: 5_000.0,
}
"""Real wall-clock delays, in milliseconds, before the fault lands.

Real because a timeout that does not take time tests nothing: the agent's own
timeout never fires and the measurement is a lie. Scaled by
:attr:`FaultSchedule.delay_scale`, which the test suite sets to zero.
"""


class FaultError(RuntimeError):
    """Base class for an injected fault delivered as an exception.

    A ``RuntimeError`` and deliberately **not** an
    :class:`~slmeval.errors.SlmevalError`: this is not a problem the user has to
    fix, it is the stimulus. The agent under test is supposed to catch it, so it
    has to look like something a tool would plausibly raise - and the runner
    classifies an uncaught one as a failed execution rather than a harness bug.
    """

    def __init__(self, message: str, *, kind: FaultKind, tool_name: str, call_index: int) -> None:
        super().__init__(message)
        self.kind = kind
        self.tool_name = tool_name
        self.call_index = call_index


class RateLimited(FaultError):
    """A synthetic ``429``."""


class ToolTimeout(FaultError):
    """A synthetic ``504``, raised after the delay has already elapsed."""


class MalformedToolOutput(FaultError):
    """The tool returned something that is not parseable."""


@dataclass(frozen=True, slots=True)
class Fault:
    """One scheduled fault: what, how, and how late.

    Immutable and fully determined by the draw, so two processes that compute
    the schedule from the same seed hold equal objects.
    """

    kind: FaultKind
    delivery: Delivery
    tool_name: str
    call_index: int
    """Which call *to this tool* this fault lands on, counting from zero."""
    delay_ms: float = 0.0

    @property
    def delay_s(self) -> float:
        return self.delay_ms / 1000.0

    def error(self) -> FaultError:
        """The exception for a raise-delivered fault."""
        kind_to_error = {
            FaultKind.RATE_LIMIT_429: (
                RateLimited,
                f"429 Too Many Requests from {self.tool_name} (injected fault)",
            ),
            FaultKind.TIMEOUT_504: (
                ToolTimeout,
                f"504 Gateway Timeout from {self.tool_name} after "
                f"{self.delay_ms:.0f}ms (injected fault)",
            ),
            FaultKind.MALFORMED_JSON: (
                MalformedToolOutput,
                f"{self.tool_name} returned unparseable output (injected fault)",
            ),
        }
        cls, message = kind_to_error.get(
            self.kind,
            (FaultError, f"{self.kind.value} from {self.tool_name} (injected fault)"),
        )
        return cls(message, kind=self.kind, tool_name=self.tool_name, call_index=self.call_index)

    def label(self) -> str:
        """Short form for a trace field and the comparison table."""
        return f"{self.kind.value}:{self.delivery.value}"


# --------------------------------------------------------------------------- #
# The draw
# --------------------------------------------------------------------------- #


def draw(seed: int, case_id: str, tool_name: str, call_index: int, *, salt: str = "") -> float:
    """A uniform value in ``[0, 1)`` for one scheduling decision.

    SHA-256 rather than :mod:`random`, because a PRNG's output depends on how
    many values were drawn before it. That would make the schedule depend on
    evaluation order - exactly the coupling this module exists to remove - and
    it would differ between a lazy lookup and a precomputed table. A hash of
    the coordinates is addressable: any decision can be recomputed, alone, in
    any order, years later, from the seed and the case id.

    ``salt`` separates independent decisions at the same coordinate, so
    "is there a fault here" and "which fault" never read the same bits.
    """
    payload = _UNIT.join(
        (SCHEDULE_VERSION, str(seed), case_id, tool_name, str(call_index), salt)
    )
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _pick(weights: Sequence[tuple[FaultKind, float]], value: float) -> FaultKind:
    """Weighted choice from a uniform ``value``.

    Walks a cumulative distribution over the kinds in a fixed order, so the
    result depends only on ``value`` and on the declared weights.
    """
    total = sum(weight for _, weight in weights)
    cursor = value * total
    for kind, weight in weights:
        cursor -= weight
        if cursor < 0:
            return kind
    # Only reachable on floating-point slack at the very top of the range.
    return weights[-1][0]


@dataclass(frozen=True, slots=True)
class FaultSchedule:
    """Which faults a case meets, and where.

    Built once per ``(case_id, seed)`` by :func:`fault_schedule` and shared by
    every candidate running that case - the schedule object itself is immutable
    and its lookups are pure, so sharing it across threads needs no lock.
    """

    seed: int
    case_id: str
    rate: float
    """Probability that any one tool call is faulted, in ``[0, 1]``."""
    kinds: tuple[FaultKind, ...]
    weights: tuple[tuple[FaultKind, float], ...]
    delivery: Mapping[FaultKind, Delivery]
    delays_ms: Mapping[FaultKind, float]
    delay_scale: float = 1.0
    """Multiplier on every delay. The test suite sets 0.0; a user who wants a
    fast smoke run can set 0.1 and lose only the realism of the wait."""

    def fault_for(self, tool_name: str, call_index: int) -> Fault | None:
        """The fault on this tool's ``call_index``-th call, if any.

        Pure, cheap and order-independent. Called on the hot path of every
        intercepted tool call, which is why it hashes rather than consulting a
        table: a dict lookup would need the table to exist first, and the table
        would need to know every tool name in advance.
        """
        if self.rate <= 0.0 or not self.kinds:
            return None
        if draw(self.seed, self.case_id, tool_name, call_index, salt="hit") >= self.rate:
            return None
        kind = _pick(
            self.weights, draw(self.seed, self.case_id, tool_name, call_index, salt="kind")
        )
        return Fault(
            kind=kind,
            delivery=self.delivery.get(kind, DEFAULT_DELIVERY[kind]),
            tool_name=tool_name,
            call_index=call_index,
            delay_ms=self.delays_ms.get(kind, 0.0) * self.delay_scale,
        )

    def table(
        self, tools: Iterable[str], *, max_calls: int = DEFAULT_MAX_CALLS
    ) -> dict[str, list[Fault | None]]:
        """The schedule as the plan describes it: per tool, a list indexed by
        that tool's own Nth call.

        For display (``slmeval show faults``) and for the parity tests. Built
        from :meth:`fault_for`, so the precomputed and lazy views cannot drift -
        a second implementation here is exactly how a determinism guarantee
        quietly stops holding.
        """
        return {
            name: [self.fault_for(name, index) for index in range(max_calls)]
            for name in sorted(set(tools))
        }

    def digest(self, tools: Iterable[str], *, max_calls: int = DEFAULT_MAX_CALLS) -> str:
        """A hash of the planned schedule, for ``slmeval doctor`` and the report
        header.

        This is the *plan*. :attr:`FaultVector.digest` is what actually
        happened, and the two differ whenever a candidate's trajectory did not
        reach every scheduled fault.
        """
        table = self.table(tools, max_calls=max_calls)
        payload = {
            name: [f.label() if f else None for f in faults] for name, faults in table.items()
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def fault_schedule(
    seed: int,
    case_id: str,
    *,
    rate: float,
    kinds: Iterable[FaultKind] | None = None,
    delivery: Mapping[FaultKind, Delivery] | None = None,
    delays_ms: Mapping[FaultKind, float] | None = None,
    weights: Mapping[FaultKind, float] | None = None,
    delay_scale: float = 1.0,
) -> FaultSchedule:
    """Build the fault schedule for one case.

    Called once per case, before any candidate runs it, so every candidate is
    handed the same object and cannot be given a different one by accident.

    Raises :class:`ValueError` for a delivery that the fault has no meaningful
    form for - a returned timeout, a raised ``empty_result``. Caught at
    construction, which is config-load time, rather than two hours into a run.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"fault rate must be in [0, 1], got {rate!r}")
    if delay_scale < 0.0:
        raise ValueError(f"delay_scale must be >= 0, got {delay_scale!r}")

    enabled = tuple(kinds) if kinds is not None else tuple(FaultKind)
    if not enabled and rate > 0.0:
        raise ValueError(
            "a fault rate above zero with no fault kinds enabled would silently "
            "inject nothing - set rate = 0 or enable a kind"
        )
    duplicates = sorted({k.value for k in enabled if enabled.count(k) > 1})
    if duplicates:
        raise ValueError(f"fault kinds listed twice: {', '.join(duplicates)}")

    chosen_delivery = {k: DEFAULT_DELIVERY[k] for k in enabled}
    for kind, how in (delivery or {}).items():
        allowed = _RAISEABLE if how is Delivery.RAISE else _RETURNABLE
        if kind not in allowed:
            raise ValueError(
                f"{kind.value} cannot be delivered as {how.value!r}: "
                + (
                    "it is a payload shape, and raising it would test error handling "
                    "instead of the thing it exists to test"
                    if how is Delivery.RAISE
                    else "a returned timeout is a contradiction - the call completed"
                )
            )
        chosen_delivery[kind] = how

    weight_map = {**DEFAULT_WEIGHTS, **(weights or {})}
    for kind, weight in weight_map.items():
        if weight < 0.0:
            raise ValueError(f"weight for {kind.value} must be >= 0, got {weight!r}")
    # Kept in FaultKind declaration order, not dict order: the weighted pick
    # walks this sequence, so its order is part of the determinism guarantee
    # and must not depend on how the user wrote their config.
    ordered = tuple((k, weight_map[k]) for k in FaultKind if k in set(enabled))
    if enabled and not sum(w for _, w in ordered):
        raise ValueError("every enabled fault kind has weight 0, so none can be chosen")

    return FaultSchedule(
        seed=seed,
        case_id=case_id,
        rate=rate,
        kinds=enabled,
        weights=ordered,
        delivery=chosen_delivery,
        delays_ms={**DEFAULT_DELAYS_MS, **(delays_ms or {})},
        delay_scale=delay_scale,
    )


# --------------------------------------------------------------------------- #
# What actually happened
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class InjectedFault:
    """One fault that was actually delivered, in run order."""

    seq: int
    """Position in this run's fault vector, not in the tool call sequence."""
    tool_name: str
    call_index: int
    kind: FaultKind
    delivery: Delivery
    node: str | None = None
    applied: bool = True
    """False when a payload fault could not be shaped onto the recorded
    response - a ``partial_schema`` on a scalar, for instance. Recorded as
    delivered-but-ineffective rather than dropped, because a vector that
    quietly omitted it would compare equal to a run that never scheduled it."""

    def key(self) -> tuple[str, int, str, str, bool]:
        """The identity the parity comparison uses. Excludes ``seq`` and
        ``node``: interleaving between different tools is a property of the
        trajectory, not of the faults, and two runs that met the same faults on
        the same calls are fault-equivalent."""
        return (self.tool_name, self.call_index, self.kind.value, self.delivery.value, self.applied)

    def label(self) -> str:
        suffix = "" if self.applied else " (not applicable)"
        return f"{self.tool_name}#{self.call_index} {self.kind.value}:{self.delivery.value}{suffix}"


@dataclass(slots=True)
class FaultVector:
    """The ordered list of faults one execution actually met.

    The schedule says what *would* happen; this says what did. They diverge the
    moment a candidate takes a different path, which is the limitation this
    module admits rather than hides: a candidate that stopped after two calls
    never met the fault scheduled for the fourth.

    So the vector, not the schedule, is what the comparison table compares. A
    candidate whose vector differs from the baseline's is flagged, and
    ``--require-fault-parity`` drops the pair instead of printing two numbers
    that were produced under different conditions.
    """

    faults: list[InjectedFault] = field(default_factory=list)

    def add(self, fault: Fault, *, node: str | None = None, applied: bool = True) -> InjectedFault:
        injected = InjectedFault(
            seq=len(self.faults),
            tool_name=fault.tool_name,
            call_index=fault.call_index,
            kind=fault.kind,
            delivery=fault.delivery,
            node=node,
            applied=applied,
        )
        self.faults.append(injected)
        return injected

    def __len__(self) -> int:
        return len(self.faults)

    def __bool__(self) -> bool:
        return bool(self.faults)

    def __iter__(self) -> Iterator[InjectedFault]:
        return iter(self.faults)

    @property
    def digest(self) -> str:
        """Stable hash of the faults met, order-independent.

        Sorted before hashing, because two candidates that met the same faults
        on the same calls are comparable even if they interleaved two
        independent tools differently. Order between tools is a trajectory
        property and belongs in the trajectory metrics, not here.
        """
        canonical = json.dumps(sorted(f.key() for f in self.faults), separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def labels(self) -> tuple[str, ...]:
        return tuple(f.label() for f in self.faults)


@dataclass(frozen=True, slots=True)
class ParityReport:
    """Whether two runs met the same faults, and what differed if not.

    What ``--require-fault-parity`` and the comparison table's warning column
    read. The listed differences are what makes the flag actionable: "fault
    vectors differ" sends a user looking, while "the candidate never reached
    fetch_order#2 empty_result" tells them why.
    """

    matched: bool
    baseline_digest: str
    candidate_digest: str
    missing: tuple[str, ...] = ()
    """Faults the baseline met and the candidate never reached."""
    extra: tuple[str, ...] = ()
    """Faults the candidate met and the baseline did not."""

    def reason(self) -> str:
        if self.matched:
            return "fault parity holds"
        parts = []
        if self.missing:
            parts.append(f"not reached by the candidate: {', '.join(self.missing)}")
        if self.extra:
            parts.append(f"met only by the candidate: {', '.join(self.extra)}")
        return "; ".join(parts) or "fault vectors differ"


def compare_vectors(baseline: FaultVector, candidate: FaultVector) -> ParityReport:
    """Fault parity between a baseline run and a candidate run.

    Perfect parity is impossible under divergent trajectories, so this reports
    the divergence rather than asserting it away. The caller decides what to do
    with a mismatch - flag it in the table, or drop the pair.
    """
    left = sorted(f.key() for f in baseline.faults)
    right = sorted(f.key() for f in candidate.faults)
    if left == right:
        return ParityReport(
            matched=True,
            baseline_digest=baseline.digest,
            candidate_digest=candidate.digest,
        )

    by_key = {f.key(): f for f in (*baseline.faults, *candidate.faults)}
    missing = sorted(set(left) - set(right))
    extra = sorted(set(right) - set(left))
    return ParityReport(
        matched=False,
        baseline_digest=baseline.digest,
        candidate_digest=candidate.digest,
        missing=tuple(by_key[k].label() for k in missing),
        extra=tuple(by_key[k].label() for k in extra),
    )


# --------------------------------------------------------------------------- #
# Shaping a payload fault
# --------------------------------------------------------------------------- #

MALFORMED_PAYLOAD: Final = '{"order_id": "ORD-9921", "status": "shi'
"""Truncated mid-string, which is what a real truncated response looks like.

A ``str`` and not a dict, on purpose: the agent asked for structured data and
got bytes it has to parse. An agent that passes this is one that checks, and
that is the whole question.
"""

_CONFLICTS: Final[Mapping[str, str]] = {
    "shipped": "cancelled",
    "delivered": "lost_in_transit",
    "pending": "completed",
    "completed": "pending",
    "open": "closed",
    "closed": "open",
    "active": "suspended",
    "paid": "refunded",
    "refunded": "paid",
    "approved": "rejected",
    "rejected": "approved",
    "true": "false",
    "false": "true",
    "yes": "no",
    "no": "yes",
}
"""Status words and their contradictions.

A table rather than a random string because the point of ``semantic_conflict``
is a payload that is *valid and wrong*. Replacing a status with noise tests
parsing; replacing ``"shipped"`` with ``"cancelled"`` tests whether the model
noticed that its tool just contradicted the conversation.
"""


def _conflict(value: Any, *, strong_only: bool = False) -> tuple[Any, bool]:
    """A contradictory version of one value, and whether one was found.

    ``strong_only`` restricts this to a *named* contradiction - a status word
    from :data:`_CONFLICTS`, or a flipped boolean. :func:`_conflict_payload`
    makes a first pass with it set, so a payload carrying both a status and a
    count has its status flipped rather than its count negated.
    """
    if isinstance(value, bool):
        return (not value, True)
    if isinstance(value, str):
        swapped = _CONFLICTS.get(value.strip().lower())
        if swapped is not None:
            # Preserve the recorded casing style, so the payload still looks
            # like something the tool would return.
            return (swapped.upper() if value.isupper() else swapped, True)
        return (value, False)
    if isinstance(value, int | float) and not strong_only:
        # Negate, or move off zero: a quantity that flipped sign contradicts
        # anything the agent was told about it. Weaker than a flipped status,
        # because a negative count also reads as a corrupt payload - which an
        # agent may reject on sight instead of acting on.
        return ((-value if value else -1), True)
    return (value, False)


def _empty_like(value: Any) -> Any:
    """The empty form of whatever the tool returned.

    Shape-preserving, because the question ``empty_result`` asks is "does the
    model invent data when the shape is right and the data is gone". Handing a
    dict-expecting agent a list would test type handling instead.
    """
    if isinstance(value, Mapping):
        return {}
    if isinstance(value, str):
        return ""
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return []
    if isinstance(value, int | float | bool) or value is None:
        return None
    return {}


def _strip_fields(payload: Mapping[str, Any]) -> tuple[dict[str, Any], bool]:
    """Drop roughly half of a mapping's keys, deterministically.

    Sorted and sliced rather than sampled, so the same recorded payload always
    loses the same fields and a ``partial_schema`` failure is reproducible from
    the report. At least one key is always dropped when there is one to drop -
    a "partial" payload identical to the original would be a silent no-op fault
    that still counted in the vector.
    """
    names = sorted(payload)
    if not names:
        return (dict(payload), False)
    keep = names[: len(names) // 2]
    return ({name: payload[name] for name in keep}, True)


def apply_fault(fault: Fault, response: Any) -> tuple[Any, bool]:
    """Shape a return-delivered fault onto a recorded response.

    Returns ``(payload, applied)``. ``applied`` is False when the fault has no
    meaningful form for this response - ``partial_schema`` on a bare integer -
    and the recorded response passes through unchanged. That is recorded in the
    fault vector rather than dropped: a vector that silently omitted an
    inapplicable fault would compare equal to a run that never scheduled one,
    and the parity check would pass on a difference that matters.
    """
    if fault.delivery is not Delivery.RETURN:
        raise ValueError(
            f"apply_fault is for return-delivered faults; {fault.kind.value} here is "
            f"{fault.delivery.value}-delivered"
        )

    match fault.kind:
        case FaultKind.LATENCY_SPIKE:
            # The result is correct; the delay already happened at the call
            # site. Counted in the vector because the step budget felt it.
            return (response, True)
        case FaultKind.MALFORMED_JSON:
            return (MALFORMED_PAYLOAD, True)
        case FaultKind.EMPTY_RESULT:
            return (_empty_like(response), True)
        case FaultKind.RATE_LIMIT_429 | FaultKind.TIMEOUT_504:
            # Error-as-payload: the shape a REST wrapper hands back when it
            # swallows the status code. Tests whether the model reads its tool
            # output at all.
            return (
                {
                    "error": fault.kind.value,
                    "status": 429 if fault.kind is FaultKind.RATE_LIMIT_429 else 504,
                    "message": f"{fault.tool_name} is unavailable (injected fault)",
                },
                True,
            )
        case FaultKind.PARTIAL_SCHEMA:
            if isinstance(response, Mapping):
                return _strip_fields(response)
            if isinstance(response, Sequence) and not isinstance(response, str | bytes):
                stripped = [
                    _strip_fields(item)[0] if isinstance(item, Mapping) else item
                    for item in response
                ]
                any_mapping = any(isinstance(item, Mapping) for item in response)
                return (stripped, any_mapping)
            return (response, False)
        case FaultKind.SEMANTIC_CONFLICT:
            return _conflict_payload(response)

    # No fallthrough on purpose: the match above covers every FaultKind, so
    # adding a kind without deciding how it is shaped onto a payload is a type
    # error here rather than a fault that silently does nothing.


def _conflict_payload(response: Any) -> tuple[Any, bool]:
    """Contradict one field, preferring a named contradiction.

    One field rather than all: a payload where everything flipped reads as a
    different record entirely, which an agent may reasonably reject on sight.
    One flipped field is the hard case - a payload that is *almost* right.

    Two passes over the fields, and the order is the design. The first pass
    takes only a named contradiction - a status word, a boolean - because
    ``"shipped"`` becoming ``"cancelled"`` is the fault that catches a model
    carrying on as though its tool had not just contradicted the conversation.
    Only if no field offers one does the second pass fall back to negating a
    number, which is a weaker test.
    """
    if isinstance(response, Mapping):
        for strong_only in (True, False):
            changed = dict(response)
            for name in sorted(response):
                value, ok = _conflict(response[name], strong_only=strong_only)
                if ok:
                    changed[name] = value
                    return (changed, True)
        return (dict(response), False)
    if isinstance(response, Sequence) and not isinstance(response, str | bytes):
        items = list(response)
        for index, item in enumerate(items):
            shaped, ok = _conflict_payload(item)
            if ok:
                items[index] = shaped
                return (items, True)
        return (items, False)
    return _conflict(response)


__all__ = [
    "DEFAULT_DELAYS_MS",
    "DEFAULT_DELIVERY",
    "DEFAULT_MAX_CALLS",
    "DEFAULT_WEIGHTS",
    "MALFORMED_PAYLOAD",
    "SCHEDULE_VERSION",
    "Delivery",
    "Fault",
    "FaultError",
    "FaultKind",
    "FaultSchedule",
    "FaultVector",
    "InjectedFault",
    "MalformedToolOutput",
    "ParityReport",
    "RateLimited",
    "ToolTimeout",
    "apply_fault",
    "compare_vectors",
    "draw",
    "fault_schedule",
]
