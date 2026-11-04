"""The typed envelope: how a tool's return value survives a round trip.

A cassette is only useful if the agent sees on replay exactly what it saw on
record. The obvious implementation - ``json.dumps(response, default=str)`` -
quietly breaks that. A tool returning a ``datetime`` records as a string and
replays as a string, so on record the agent's next line does
``order.created.year`` and on replay it does the same thing to a ``str`` and
raises. The benchmark then reports a model failure that is entirely our fault,
and the day you spend finding it is a day spent doubting your own tool.

The same hole swallows ``Decimal`` (replays as ``str``, so arithmetic breaks or
silently loses precision), tuples (replay as lists), non-string dict keys
(``{1: "a"}`` replays as ``{"1": "a"}``), sets, and - worst of the lot -
``StrEnum`` and ``IntEnum`` members, which *are* strings and ints, so a naive
encoder flattens them without even noticing.

So every value is tagged::

    {"__t": "json", "v": {...}}
    {"__t": "datetime", "v": "2026-10-28T10:00:00+00:00"}
    {"__t": "pydantic", "cls": "myapp.models.Order", "v": {...}}

Two rules make this trustworthy:

1. **Refuse, never coerce.** A value with no faithful representation - an open
   database connection, a file handle, a socket - stops the recording with a
   message naming the tool, the path to the value and its type. A loud failure
   at record time costs a user two minutes. A silent divergence at replay time
   costs them an afternoon and costs us their trust.
2. **Nothing is escaped by accident.** A tool that legitimately returns a dict
   containing the key ``"__t"`` round-trips intact, because such dicts are
   re-encoded as explicit key/value pairs rather than left to collide with our
   own tag.
"""

from __future__ import annotations

import base64
import dataclasses
import importlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from pathlib import PurePath
from typing import Any, Final, cast
from uuid import UUID

from pydantic import BaseModel

from slmeval.errors import SlmevalError

TAG: Final = "__t"
"""The reserved key. Short because it appears once per nested value, and a
cassette store holds hundreds of thousands of them."""

VALUE: Final = "v"
CLS: Final = "cls"

SUPPORTED: Final = (
    "dict, list, tuple, set, frozenset, str, int, float, bool, None, "
    "datetime, date, time, timedelta, Decimal, UUID, bytes, bytearray, "
    "Path, Enum, Pydantic models and dataclasses"
)


class UnserializableResponse(SlmevalError):
    """A tool returned something that cannot be recorded faithfully.

    The user's to fix: their tool hands back a live object, and the harness
    refuses to pretend otherwise.
    """

    def __init__(
        self, *, what: str, path: str, python_type: str, reason: str | None = None
    ) -> None:
        detail = reason or "it has no JSON representation"
        super().__init__(
            f"cannot record {what}: the value at {path} is a {python_type}, and {detail}.",
            hint=(
                "Return something representable and slmeval will preserve its type exactly.\n"
                f"Supported: {SUPPORTED}.\n"
                "slmeval refuses to coerce it with str() on purpose: that would record a "
                "string and replay a string, so your agent would see a different type on "
                "replay than it saw on record - a divergence that looks like a model bug."
            ),
        )
        self.what = what
        self.path = path
        self.python_type = python_type


class EnvelopeDecodeError(SlmevalError):
    """A recorded envelope cannot be rebuilt.

    Almost always a code change since recording: the class a cassette names has
    been moved, renamed or deleted. That is staleness, not corruption, so the
    message points at the cassette rather than at the payload.
    """

    def __init__(self, message: str, *, hint: str | None = None) -> None:
        super().__init__(
            message,
            hint=hint
            or (
                "This cassette was recorded against code that has since changed. "
                "Re-record the affected cases with `slmeval record --case <id>`."
            ),
        )


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def _qualified(cls: type) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"


def _check_importable(cls: type, *, what: str, path: str) -> str:
    """Reject classes that could not be imported back on replay.

    A class defined in ``__main__`` or inside a function body has a name nobody
    can resolve in a fresh process. Replay happens in a fresh process - usually
    in CI - so recording one guarantees a decode failure later. Say so now.
    """
    name = _qualified(cls)
    if cls.__module__ in {"__main__", "builtins"} or "<locals>" in cls.__qualname__:
        raise UnserializableResponse(
            what=what,
            path=path,
            python_type=name,
            reason=(
                "its class cannot be imported back on replay (it is defined in "
                f"{cls.__module__!r}"
                + (" inside a function body" if "<locals>" in cls.__qualname__ else "")
                + "), and replay runs in a fresh process"
            ),
        )
    return name


def to_jsonable(value: Any, *, what: str = "value", path: str = "$") -> Any:
    """Recursively convert to a JSON-safe tree, tagging anything non-native.

    Returns a node, not a complete envelope - see :func:`encode`.
    """
    return _encode(value, what=what, path=path, seen=())


def _encode(value: Any, *, what: str, path: str, seen: tuple[int, ...]) -> Any:
    # Enum first: StrEnum and IntEnum members pass `isinstance(x, str)` and
    # `isinstance(x, int)`, so any later branch would flatten them to their
    # value and the agent would see a bare str on replay.
    if isinstance(value, Enum):
        cls = _check_importable(type(value), what=what, path=path)
        return {TAG: "enum", CLS: cls, VALUE: _encode(value.value, what=what, path=path, seen=seen)}

    if value is None or isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if math.isfinite(value):
            return value
        # json.dumps emits bare NaN/Infinity, which is not JSON and which a
        # strict reader on the other side of the store will reject.
        marker = "nan" if math.isnan(value) else ("inf" if value > 0 else "-inf")
        return {TAG: "float", VALUE: marker}

    if isinstance(value, (bytes, bytearray)):
        tag = "bytes" if isinstance(value, bytes) else "bytearray"
        return {TAG: tag, VALUE: base64.b64encode(bytes(value)).decode("ascii")}

    # datetime before date: datetime is a date subclass.
    if isinstance(value, datetime):
        return {TAG: "datetime", VALUE: value.isoformat()}
    if isinstance(value, date):
        return {TAG: "date", VALUE: value.isoformat()}
    if isinstance(value, time):
        return {TAG: "time", VALUE: value.isoformat()}
    if isinstance(value, timedelta):
        return {TAG: "timedelta", VALUE: value.total_seconds()}
    if isinstance(value, Decimal):
        # str(), not float(): Decimal exists precisely because float loses
        # digits, and a money amount is the likeliest thing a tool returns.
        return {TAG: "decimal", VALUE: str(value)}
    if isinstance(value, UUID):
        return {TAG: "uuid", VALUE: str(value)}
    if isinstance(value, PurePath):
        return {TAG: "path", VALUE: str(value)}

    if id(value) in seen:
        raise UnserializableResponse(
            what=what,
            path=path,
            python_type=_qualified(type(value)),
            reason="it contains a circular reference back to itself",
        )
    nested = (*seen, id(value))

    if isinstance(value, BaseModel):
        cls = _check_importable(type(value), what=what, path=path)
        # model_dump(mode="python") then recurse, rather than mode="json":
        # pydantic's JSON mode would str()-coerce the very types this module
        # exists to preserve, and would hide an unserializable field instead of
        # refusing it.
        return {
            TAG: "pydantic",
            CLS: cls,
            VALUE: _encode(value.model_dump(mode="python"), what=what, path=path, seen=nested),
        }

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = _check_importable(type(value), what=what, path=path)
        fields = {f.name: getattr(value, f.name) for f in dataclasses.fields(value) if f.init}
        return {
            TAG: "dataclass",
            CLS: cls,
            VALUE: {
                name: _encode(item, what=what, path=f"{path}.{name}", seen=nested)
                for name, item in fields.items()
            },
        }

    if isinstance(value, Mapping):
        return _encode_mapping(value, what=what, path=path, seen=nested)

    if isinstance(value, (set, frozenset)):
        tag = "set" if isinstance(value, set) else "frozenset"
        items = [
            _encode(item, what=what, path=f"{path}{{{index}}}", seen=nested)
            for index, item in enumerate(sorted(value, key=_stable_repr))
        ]
        return {TAG: tag, VALUE: items}

    if isinstance(value, tuple):
        # Tagged, because JSON has one array type and a tuple replaying as a
        # list breaks unpacking and any `isinstance` check downstream.
        return {
            TAG: "tuple",
            VALUE: [
                _encode(item, what=what, path=f"{path}[{index}]", seen=nested)
                for index, item in enumerate(value)
            ],
        }

    if isinstance(value, Sequence):
        return [
            _encode(item, what=what, path=f"{path}[{index}]", seen=nested)
            for index, item in enumerate(value)
        ]

    raise UnserializableResponse(what=what, path=path, python_type=_qualified(type(value)))


def _stable_repr(item: Any) -> str:
    """Ordering key for sets, so a recorded set has a deterministic layout and
    the cassette key for an identical call does not change between runs."""
    return f"{type(item).__name__}:{item!r}"


def _encode_mapping(
    value: Mapping[Any, Any], *, what: str, path: str, seen: tuple[int, ...]
) -> Any:
    plain = all(isinstance(key, str) for key in value) and TAG not in value
    if plain:
        return {
            str(key): _encode(item, what=what, path=f"{path}.{key}", seen=seen)
            for key, item in value.items()
        }
    # Non-string keys would be stringified by JSON ({1: "a"} -> {"1": "a"}), and
    # a dict that genuinely contains our tag key would be mistaken for an
    # envelope on the way back. Both become explicit pairs. Insertion order is
    # preserved: dict order is observable, so it is part of the payload.
    pairs = [
        [
            _encode(key, what=what, path=f"{path}.<key {index}>", seen=seen),
            _encode(item, what=what, path=f"{path}.<value {index}>", seen=seen),
        ]
        for index, (key, item) in enumerate(value.items())
    ]
    return {TAG: "map", VALUE: pairs}


def encode(value: Any, *, what: str = "tool return value") -> dict[str, Any]:
    """Wrap a value in a complete envelope, ready to store.

    Always a tagged dict, so the stored column has one shape whatever the tool
    returned. Raises :class:`UnserializableResponse` rather than coercing.
    """
    node = _encode(value, what=what, path="$", seen=())
    if isinstance(node, dict) and TAG in node:
        return node
    return {TAG: "json", VALUE: node}


def encode_json(value: Any, *, what: str = "tool return value") -> str:
    """Encode and serialize in one step - what the store writes."""
    return json.dumps(
        encode(value, what=what), sort_keys=False, separators=(",", ":"), allow_nan=False
    )


# --------------------------------------------------------------------------- #
# Decoding
# --------------------------------------------------------------------------- #


def _import_class(name: str, *, tag: str) -> Any:
    module_name, _, attribute = name.rpartition(".")
    if not module_name:
        raise EnvelopeDecodeError(f"{tag} envelope names {name!r}, which is not a dotted path")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise EnvelopeDecodeError(
            f"cannot import {module_name!r} to rebuild the recorded {tag} {name!r}: {exc}"
        ) from exc
    try:
        return getattr(module, attribute)
    except AttributeError:
        raise EnvelopeDecodeError(
            f"{module_name!r} no longer defines {attribute!r}, "
            f"which a cassette recorded as a {tag}"
        ) from None


_FLOATS: Final = {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}


def decode(envelope: Any) -> Any:
    """Rebuild the original value from a complete envelope."""
    if not isinstance(envelope, dict) or TAG not in envelope:
        raise EnvelopeDecodeError(
            f"expected a typed envelope with a {TAG!r} tag, got {type(envelope).__name__}",
            hint="Envelopes are written by slmeval; this payload was not.",
        )
    return _decode(envelope)


def decode_json(text: str) -> Any:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise EnvelopeDecodeError(f"stored envelope is not valid JSON: {exc}") from exc
    return decode(payload)


def _decode(node: Any) -> Any:
    if isinstance(node, list):
        return [_decode(item) for item in node]
    if not isinstance(node, dict):
        return node
    if TAG not in node:
        return {key: _decode(item) for key, item in node.items()}

    tag = node[TAG]
    raw = node.get(VALUE)

    if tag == "json":
        return _decode(raw)
    if tag == "map":
        if not isinstance(raw, list):
            raise EnvelopeDecodeError("a 'map' envelope must hold a list of key/value pairs")
        return {_hashable(_decode(pair[0])): _decode(pair[1]) for pair in raw}
    if tag == "tuple":
        return tuple(_decode(item) for item in raw or [])
    if tag == "set":
        return {_hashable(_decode(item)) for item in raw or []}
    if tag == "frozenset":
        return frozenset(_hashable(_decode(item)) for item in raw or [])
    if tag == "float":
        try:
            return _FLOATS[str(raw)]
        except KeyError:
            raise EnvelopeDecodeError(f"unknown non-finite float marker {raw!r}") from None
    if tag == "bytes":
        return base64.b64decode(str(raw))
    if tag == "bytearray":
        return bytearray(base64.b64decode(str(raw)))
    if tag == "datetime":
        return datetime.fromisoformat(str(raw))
    if tag == "date":
        return date.fromisoformat(str(raw))
    if tag == "time":
        return time.fromisoformat(str(raw))
    if tag == "timedelta":
        return timedelta(seconds=float(raw))  # type: ignore[arg-type]
    if tag == "decimal":
        return Decimal(str(raw))
    if tag == "uuid":
        return UUID(str(raw))
    if tag == "path":
        from pathlib import Path

        return Path(str(raw))
    if tag in {"pydantic", "dataclass", "enum"}:
        return _decode_class(tag, node, raw)

    raise EnvelopeDecodeError(
        f"unknown envelope tag {tag!r}",
        hint=(
            "This cassette was written by a newer slmeval than the one reading it. "
            "Upgrade slmeval, or re-record."
        ),
    )


def _decode_class(tag: str, node: dict[str, Any], raw: Any) -> Any:
    name = node.get(CLS)
    if not isinstance(name, str):
        raise EnvelopeDecodeError(f"a {tag!r} envelope must carry a {CLS!r} class path")
    cls = _import_class(name, tag=tag)

    if tag == "enum":
        member = _decode(raw)
        try:
            return cls(member)
        except (ValueError, TypeError) as exc:
            raise EnvelopeDecodeError(
                f"{name} no longer has a member with value {member!r}: {exc}"
            ) from exc

    decoded = _decode(raw)
    if not isinstance(decoded, dict):
        raise EnvelopeDecodeError(
            f"a {tag!r} envelope must hold an object, got {type(decoded).__name__}"
        )

    if tag == "pydantic":
        if not (isinstance(cls, type) and issubclass(cls, BaseModel)):
            raise EnvelopeDecodeError(
                f"{name} was recorded as a Pydantic model but no longer is one"
            )
        try:
            return cls.model_validate(decoded)
        except Exception as exc:  # pydantic ValidationError, and anything a validator raises
            raise EnvelopeDecodeError(
                f"a recorded {name} no longer validates against its current definition: {exc}"
            ) from exc

    if not dataclasses.is_dataclass(cls):
        raise EnvelopeDecodeError(f"{name} was recorded as a dataclass but no longer is one")
    known = {f.name for f in dataclasses.fields(cls) if f.init}
    unknown = sorted(set(decoded) - known)
    if unknown:
        raise EnvelopeDecodeError(
            f"a recorded {name} carries field(s) {', '.join(unknown)} "
            "that its current definition does not accept"
        )
    try:
        return cast("Any", cls)(**decoded)
    except TypeError as exc:
        raise EnvelopeDecodeError(f"cannot rebuild a recorded {name}: {exc}") from exc


def _hashable(value: Any) -> Any:
    """Dict keys and set members have to be hashable.

    A decoded list cannot be either, which means the encoder produced something
    the decoder cannot place - a harness bug, so it keeps its traceback.
    """
    if isinstance(value, (list, dict, set, bytearray)):
        raise TypeError(
            f"decoded an unhashable {type(value).__name__} where a key or set member "
            "was expected - this is an slmeval bug, please report the cassette"
        )
    return value


__all__ = [
    "CLS",
    "SUPPORTED",
    "TAG",
    "VALUE",
    "EnvelopeDecodeError",
    "UnserializableResponse",
    "decode",
    "decode_json",
    "encode",
    "encode_json",
    "to_jsonable",
]
