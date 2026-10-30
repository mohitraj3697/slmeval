"""JSON in and out of columns, one way only.

Every column that holds JSON is written through these helpers rather than with
a bare ``json.dumps``, for two reasons. Compact separators, because at ~40
JSON columns per execution the whitespace in the default separators is real
megabytes across a benchmark. And ``sort_keys``, because a column written
differently by two code paths breaks the one thing content addressing buys us:
identical payloads have to hash identically or the blob store stops deduping.

Reads are forgiving by design. A row written by an older build, or hand-edited
by a user poking at their own SQLite file, must not take down a report - a
malformed JSON column degrades to the empty value and the rest of the timeline
still prints.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import Any

_SEPARATORS = (",", ":")


def dumps(value: Any) -> str:
    """Compact, key-sorted JSON. The only writer for a JSON column."""
    return json.dumps(value, separators=_SEPARATORS, sort_keys=True, default=str)


def loads(text: str | None, default: Any = None) -> Any:
    """Parse a JSON column, falling back rather than raising."""
    if not text:
        return default
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return default


def encode_paths(paths: Iterable[Sequence[str] | str] | None) -> str:
    """Serialize node paths - a sequence of name tuples.

    A bare string is accepted as a one-element path, because a nested node's
    full path and a top-level node's name are the same thing at different
    depths and callers should not have to wrap one of them.
    """
    if not paths:
        return "[]"
    out: list[list[str]] = []
    for path in paths:
        if isinstance(path, str):
            out.append([path])
        else:
            out.append([str(part) for part in path])
    return dumps(out)


def decode_paths(text: str | None) -> tuple[tuple[str, ...], ...]:
    """Rebuild node paths. Tolerates the flat ``["a", "b"]`` shape an older
    build wrote, reading each entry as a one-element path."""
    raw = loads(text, [])
    if not isinstance(raw, list):
        return ()
    out: list[tuple[str, ...]] = []
    for item in raw:
        if isinstance(item, str):
            out.append((item,))
        elif isinstance(item, list):
            out.append(tuple(str(part) for part in item))
    return tuple(out)


def encode_strings(values: Iterable[object] | None) -> str:
    if not values:
        return "[]"
    return dumps([str(v) for v in values])


def decode_strings(text: str | None) -> tuple[str, ...]:
    raw = loads(text, [])
    if not isinstance(raw, list):
        return ()
    return tuple(str(v) for v in raw)


__all__ = [
    "decode_paths",
    "decode_strings",
    "dumps",
    "encode_paths",
    "encode_strings",
    "loads",
]
