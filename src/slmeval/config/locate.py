"""Map a Pydantic error location onto a line number in the source TOML.

`tomllib` discards position information, so after validation fails we no longer
know where a bad key lived. Rather than take a second TOML dependency just to
pretty-print errors, this does a small best-effort scan of the raw text.

Best-effort is the right contract: a wrong line number would be worse than
none, so every lookup that is not confident returns ``None`` and the caller
falls back to printing the dotted key path alone.
"""

from __future__ import annotations

import re
from typing import NamedTuple

_HEADER_RE = re.compile(r"^\s*(\[\[|\[)\s*([^\]\s]+)\s*(\]\]|\])\s*(#.*)?$")


class _Header(NamedTuple):
    line_index: int
    name: str
    is_array: bool


def _headers(lines: list[str]) -> list[_Header]:
    out: list[_Header] = []
    for i, line in enumerate(lines):
        m = _HEADER_RE.match(line)
        if m:
            out.append(_Header(line_index=i, name=m.group(2), is_array=m.group(1) == "[["))
    return out


def _key_pattern(key: str) -> re.Pattern[str]:
    k = re.escape(key)
    # bare, "quoted" or 'quoted' key, followed by '='
    return re.compile(rf"""^\s*({k}|"{k}"|'{k}')\s*=""")


def locate(text: str, loc: tuple[int | str, ...]) -> int | None:
    """Return a 1-based line number for a Pydantic error ``loc``, or None.

    Handles the shapes this schema actually produces::

        ("benchmark", "runs_per_case")   -> the key inside [benchmark]
        ("candidates", 0, "provider")    -> the key inside the 1st [[candidates]]
        ("candidates", 1)                -> the 2nd [[candidates]] header
        ("baseline",)                    -> the [baseline] header
    """
    if not loc:
        return None

    lines = text.splitlines()
    if not lines:
        return None

    names = [c for c in loc if isinstance(c, str)]
    indices = [c for c in loc if isinstance(c, int)]
    occurrence = indices[0] if indices else 0

    if not names:
        return None

    if len(names) >= 2:
        section, key = ".".join(names[:-1]), names[-1]
    else:
        section, key = names[0], None

    headers = _headers(lines)
    matching = [h for h in headers if h.name == section]

    if not matching:
        # No such table. Fall back to a whole-file search for the key, which is
        # also the top-level-key case.
        return _search(lines, key, 0, len(lines)) if key else None

    header = matching[occurrence] if occurrence < len(matching) else matching[0]
    if key is None:
        return header.line_index + 1

    # Search only between this header and the next one, so `model` inside
    # [[candidates]] #2 is never confused with `model` inside #1.
    following = [h.line_index for h in headers if h.line_index > header.line_index]
    end = following[0] if following else len(lines)
    found = _search(lines, key, header.line_index + 1, end)
    return found if found is not None else header.line_index + 1


def _search(lines: list[str], key: str | None, start: int, end: int) -> int | None:
    if key is None:
        return None
    pattern = _key_pattern(key)
    for i in range(start, min(end, len(lines))):
        if pattern.match(lines[i]):
            return i + 1
    return None
