"""The cassette key, and the matching policy that produces it.

A cassette is a recorded tool response, looked up by a hash of the call that
produced it. Getting that hash right is the hardest design decision in the
product, because one hash has to answer two questions that pull in opposite
directions:

- ``order_id="9921"`` where the record said ``"ORD-9921"`` is a **parameter
  hallucination**. It has to miss. That miss, with its diagnosis block, is the
  single most valuable thing this tool prints.
- ``query="SELECT * FROM orders WHERE id='X'"`` against a recorded
  ``"select * from orders where id = 'X'"`` is **the same call**. A miss here is
  our bug, reported as the model's, and it poisons every number in the table.

Hashing exact canonical arguments and raising on any mismatch - the obvious
implementation - conflates the two. So matching is a **declared, per-tool
policy** with a lookup cascade:

====================  ===================================================
``exact``             every argument, as the model sent it
``normalized``        the declared normalizers applied
``subset``            only the arguments named in ``match_on``
====================  ===================================================

Lookup tries them in that order and **records which tier hit**, because the tier
is itself a finding. A ``subset`` hit means the arguments the tool keys on were
right and something else was wrong - the model got an extra argument wrong while
still reaching the right call. That is a reportable Tier-1 result, not a silent
pass, and it is only visible because the cascade is ordered rather than fuzzy.

Three things are deliberately *not* in the key:

``tool_version``
    An edited tool makes its old cassettes stale, and staleness should be
    *flagged* by ``slmeval cassette verify``, not turn into a wall of misses
    that looks like the model fell apart.
``dataset_hash``
    Same reasoning; it is a stored column, used for provenance.
``attempt``
    Every repeat of a case must replay the same recorded responses or the
    repeats measure different things, which would make ``pass^k`` meaningless.

The tier string *is* in the hash, and that matters. Without it, a subset key
over ``{"order_id": "X"}`` would equal the exact key of a different call that
happened to pass only ``order_id`` - so a model that dropped a required
argument would hit the other call's cassette and be reported as an exact match.
Paying for three rows per recorded call buys an unambiguous answer to "which
tier matched", which is the question the report is built on.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import unicodedata
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from difflib import get_close_matches
from enum import StrEnum
from typing import Any, Final

from slmeval.errors import SlmevalError
from slmeval.sandbox import envelope
from slmeval.sandbox.normalizers import Normalizer, UnknownNormalizer, resolve_all

KEY_VERSION: Final = "k1"
"""Bumped if canonicalization ever changes.

In the hash, so cassettes written under the old scheme miss cleanly and
`cassette verify` can say "re-record" instead of every call silently matching
the wrong payload.
"""

KEY_LENGTH: Final = 32
"""128 bits of SHA-256. Collision risk across every cassette anyone will ever
record is nil, and a short key keeps the store's index small."""

_UNIT = "\x1f"
"""ASCII unit separator - cannot appear in a case id or a tool name, so the
hashed fields cannot be made to run together."""


class MatchTier(StrEnum):
    """Which rung of the cascade a key belongs to."""

    EXACT = "exact"
    NORMALIZED = "normalized"
    SUBSET = "subset"


CASCADE: Final = (MatchTier.EXACT, MatchTier.NORMALIZED, MatchTier.SUBSET)


class PolicyError(SlmevalError):
    """A ``@sandbox_tool`` declaration is wrong.

    Raised when the tool is declared, which is import time - long before a
    benchmark starts. A policy typo that surfaced two hours into a matrix run
    would be unforgivable, since everything needed to catch it is available the
    moment the decorator runs.
    """


@dataclass(frozen=True, slots=True)
class KeyEntry:
    """One tier's key, plus the canonical arguments it was computed from.

    The canonical form is kept because the store writes it to a column and the
    miss diagnosis reads it back to show the user what was compared.
    """

    tier: MatchTier
    key: str
    canonical_args: str


@dataclass(frozen=True, slots=True)
class CassetteKeys:
    """Every key one tool call can be found by, in cascade order."""

    case_id: str
    tool_name: str
    entries: tuple[KeyEntry, ...]

    def __iter__(self) -> Iterator[KeyEntry]:
        """Iterate in lookup order: exact, then normalized, then subset."""
        return iter(self.entries)

    def of(self, tier: MatchTier) -> KeyEntry | None:
        return next((e for e in self.entries if e.tier is tier), None)

    @property
    def exact(self) -> str:
        """The exact-tier key. Always present; it is what a recording is filed
        under primarily."""
        return self.entries[0].key

    @property
    def normalized(self) -> str | None:
        entry = self.of(MatchTier.NORMALIZED)
        return entry.key if entry else None

    @property
    def subset(self) -> str | None:
        """None unless the tool declared ``match_on``."""
        entry = self.of(MatchTier.SUBSET)
        return entry.key if entry else None

    @property
    def keys(self) -> tuple[str, ...]:
        return tuple(e.key for e in self.entries)


def _hash(tier: MatchTier, case_id: str, tool_name: str, canonical: str) -> str:
    payload = _UNIT.join((KEY_VERSION, tier.value, case_id, tool_name, canonical))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def _nfc(value: Any) -> Any:
    """Unicode-normalize strings, recursively.

    ``"café"`` composed (NFC) and decomposed (NFD) are the same text and
    different bytes. macOS hands out NFD from its filesystem APIs while Linux
    and Windows produce NFC, so without this a cassette recorded on a laptop
    misses in CI - a false miss caused entirely by the recording machine.

    Applied to the key only, never to the recorded payload: the payload must
    stay byte-identical to what the tool returned.
    """
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, dict):
        return {_nfc(k): _nfc(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nfc(item) for item in value]
    return value


def canonicalize(arguments: Mapping[str, Any], *, tool_name: str) -> str:
    """Canonical JSON for a set of arguments.

    Sorted keys and no whitespace, so key order and formatting cannot change the
    hash. Values go through the typed envelope rather than ``default=str``, so a
    ``datetime`` argument and the string that looks like it hash differently -
    they are different calls.
    """
    jsonable = envelope.to_jsonable(
        dict(arguments), what=f"arguments to tool {tool_name!r}", path="$"
    )
    return json.dumps(
        _nfc(jsonable),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class MatchPolicy:
    """How one tool's calls are matched against its cassettes.

    Built by :func:`declare`, which validates it against the tool's signature.
    Frozen: a policy that changed mid-run would silently split one tool's
    cassettes across two key schemes.
    """

    tool_name: str
    parameters: tuple[str, ...] = ()
    """Declared parameter names, in signature order. Empty if unknown."""
    match_on: tuple[str, ...] | None = None
    ignore_args: frozenset[str] = frozenset()
    normalize: Mapping[str, str] = field(default_factory=dict)
    normalizers: tuple[tuple[str, Normalizer], ...] = ()
    defaults: tuple[tuple[str, Any], ...] = ()
    """Parameter defaults, filled in before hashing - see :meth:`effective`."""
    tool_version: str = ""
    accepts_kwargs: bool = False

    # ---------------------------------------------------------------- arguments

    def effective(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        """The arguments as the key sees them: defaults filled, ignored dropped.

        Defaults are filled on both sides of the round trip. Recording
        ``f(x=1)`` and replaying ``f(x=1, y=2)`` where ``y`` defaults to ``2``
        is the same call, and a model that spells out a default it could have
        omitted has not done anything wrong. Without this, half the arguments a
        verbose model emits would read as misses.

        Required parameters have no default, so a model that *omits* one still
        misses - which is correct, and is the finding.
        """
        merged = dict(self.defaults)
        merged.update(arguments)
        return {k: v for k, v in merged.items() if k not in self.ignore_args}

    def normalized_args(self, arguments: Mapping[str, Any]) -> dict[str, Any]:
        applied = dict(self.effective(arguments))
        for name, normalizer in self.normalizers:
            if name in applied:
                applied[name] = normalizer(applied[name])
        return applied

    def subset_args(self, arguments: Mapping[str, Any]) -> dict[str, Any] | None:
        """Only the ``match_on`` arguments, normalized. None if undeclared.

        Missing ``match_on`` arguments are left out rather than filled with a
        placeholder: a call that omits one is genuinely a different call, and it
        should miss here too.
        """
        if self.match_on is None:
            return None
        applied = self.normalized_args(arguments)
        return {name: applied[name] for name in self.match_on if name in applied}

    # -------------------------------------------------------------------- keys

    def compute(self, case_id: str, arguments: Mapping[str, Any]) -> CassetteKeys:
        """Every key this call can be found by, in cascade order.

        Record writes a row per entry; replay looks them up in order and reports
        the tier that hit.
        """
        if not case_id:
            raise PolicyError(
                f"tool {self.tool_name!r} was called with no case id, so its response "
                "cannot be filed against a case",
                hint=(
                    "This means the sandbox had no run context. Tool calls are scoped "
                    "per case so every repeat of a case replays the same responses."
                ),
            )
        shapes: list[tuple[MatchTier, Mapping[str, Any]]] = [
            (MatchTier.EXACT, self.effective(arguments)),
            (MatchTier.NORMALIZED, self.normalized_args(arguments)),
        ]
        subset = self.subset_args(arguments)
        if subset is not None:
            shapes.append((MatchTier.SUBSET, subset))

        entries = tuple(
            KeyEntry(
                tier=tier,
                key=_hash(tier, case_id, self.tool_name, canonical),
                canonical_args=canonical,
            )
            for tier, canonical in (
                (tier, canonicalize(shape, tool_name=self.tool_name)) for tier, shape in shapes
            )
        )
        return CassetteKeys(case_id=case_id, tool_name=self.tool_name, entries=entries)


def _signature_of(fn: Callable[..., Any]) -> tuple[tuple[str, ...], dict[str, Any], bool]:
    """Parameter names, defaults and whether the tool takes ``**kwargs``."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        # A C function or an exotic callable. Validation degrades to "cannot
        # check names" rather than refusing to wrap the tool at all.
        return (), {}, True

    names: list[str] = []
    defaults: dict[str, Any] = {}
    accepts_kwargs = False
    for index, parameter in enumerate(signature.parameters.values()):
        if parameter.kind is inspect.Parameter.VAR_KEYWORD:
            accepts_kwargs = True
            continue
        if parameter.kind is inspect.Parameter.VAR_POSITIONAL:
            continue
        if index == 0 and parameter.name in {"self", "cls"}:
            continue
        names.append(parameter.name)
        if parameter.default is not inspect.Parameter.empty:
            defaults[parameter.name] = parameter.default
    return tuple(names), defaults, accepts_kwargs


def _check_names(
    kind: str,
    names: Iterable[str],
    *,
    tool_name: str,
    parameters: tuple[str, ...],
    accepts_kwargs: bool,
) -> None:
    if accepts_kwargs or not parameters:
        # The tool takes **kwargs, so argument names are not knowable from the
        # signature and anything the user declares may legitimately arrive.
        return
    for name in names:
        if name in parameters:
            continue
        close = get_close_matches(name, parameters, n=1)
        raise PolicyError(
            f"tool {tool_name!r} declares {kind}={name!r}, but it has no such argument",
            hint=(
                (f"Did you mean {close[0]!r}? " if close else "")
                + f"Arguments of {tool_name}: {', '.join(parameters) or '(none)'}."
            ),
        )


def declare(
    fn: Callable[..., Any] | None = None,
    *,
    tool_name: str,
    match_on: list[str] | tuple[str, ...] | None = None,
    normalize: Mapping[str, str] | None = None,
    ignore_args: list[str] | tuple[str, ...] | frozenset[str] | None = None,
    tool_version: str = "",
) -> MatchPolicy:
    """Build and validate a :class:`MatchPolicy`.

    Every check here is one the decorator can make at import time, so the user
    finds out about a typo in half a second rather than halfway through a
    matrix run.
    """
    parameters, defaults, accepts_kwargs = _signature_of(fn) if fn is not None else ((), {}, True)

    normalize_specs = dict(normalize or {})
    ignored = frozenset(ignore_args or ())
    declared_match: tuple[str, ...] | None
    if match_on is None:
        declared_match = None
    else:
        declared_match = tuple(match_on)
        if not declared_match:
            raise PolicyError(
                f"tool {tool_name!r} declares match_on=[], which would make every call "
                "to it match the same cassette",
                hint=(
                    "Name the arguments that identify the call, or drop match_on "
                    "entirely to match on all of them."
                ),
            )
        duplicates = sorted({n for n in declared_match if declared_match.count(n) > 1})
        if duplicates:
            raise PolicyError(
                f"tool {tool_name!r} lists {', '.join(duplicates)} twice in match_on"
            )

    _check_names(
        "match_on",
        declared_match or (),
        tool_name=tool_name,
        parameters=parameters,
        accepts_kwargs=accepts_kwargs,
    )
    _check_names(
        "ignore_args",
        ignored,
        tool_name=tool_name,
        parameters=parameters,
        accepts_kwargs=accepts_kwargs,
    )
    _check_names(
        "normalize",
        tuple(normalize_specs),
        tool_name=tool_name,
        parameters=parameters,
        accepts_kwargs=accepts_kwargs,
    )

    if declared_match is not None:
        both = sorted(set(declared_match) & ignored)
        if both:
            raise PolicyError(
                f"tool {tool_name!r} both matches on and ignores {', '.join(both)}",
                hint=(
                    "match_on says an argument identifies the call; ignore_args says it "
                    "is noise. Pick one."
                ),
            )

    contradictory = sorted(set(normalize_specs) & ignored)
    if contradictory:
        raise PolicyError(
            f"tool {tool_name!r} normalizes {', '.join(contradictory)}, which it also ignores",
            hint=(
                "An ignored argument never reaches the key, so normalizing it has no "
                "effect. Remove it from one of the two lists."
            ),
        )

    try:
        resolved = resolve_all(normalize_specs)
    except UnknownNormalizer as exc:
        raise PolicyError(f"tool {tool_name!r}: {exc}") from exc

    return MatchPolicy(
        tool_name=tool_name,
        parameters=parameters,
        match_on=declared_match,
        ignore_args=ignored,
        normalize=dict(normalize_specs),
        normalizers=tuple(resolved.items()),
        defaults=tuple(defaults.items()),
        tool_version=tool_version,
        accepts_kwargs=accepts_kwargs,
    )


__all__ = [
    "CASCADE",
    "KEY_LENGTH",
    "KEY_VERSION",
    "CassetteKeys",
    "KeyEntry",
    "MatchPolicy",
    "MatchTier",
    "PolicyError",
    "canonicalize",
    "declare",
]
