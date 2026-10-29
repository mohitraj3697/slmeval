"""The JSON extraction cascade.

Small models do not emit clean JSON. They emit fenced blocks, prose preambles
("Sure! Here is the JSON you asked for:"), trailing commentary, single quotes,
Python literals, trailing commas, smart quotes from an over-helpful tokenizer,
and sometimes two objects where one was asked for.

A forgiving parser is necessary to get any signal at all from a 4B model. But a
*silently* forgiving parser turns a measurement instrument into a liar: the
model "passes" and the report never mentions that our regexes did the work. So
every stage past the first is recorded, counted as a leniency event, and shown
in the scorecard.

The cascade, in order:

======== =====================================================================
direct   ``json.loads`` on the raw text. The only non-event.
fenced   strip a ```json ... ``` (or bare ```) fence, then retry
balanced brace/bracket-balanced scan for the first complete value, which drops
         prose preambles and trailing commentary
repaired trailing commas, smart quotes; then ``ast.literal_eval`` for
         Python-dict output
======== =====================================================================
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any

from slmeval.core.trace import JsonStage

FENCE_RE = re.compile(
    r"```(?:json|JSON|json5|javascript|js)?[ \t]*\r?\n(?P<body>.*?)(?:\r?\n)?[ \t]*```",
    re.DOTALL,
)
TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")

# ruff: noqa: RUF001 - the whole point of this table is the ambiguous characters
SMART_QUOTES = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "〝": '"',
        "〞": '"',
        "«": '"',
        "»": '"',
    }
)

MAX_PREVIEW = 400


class JsonExtractionError(ValueError):
    """No stage of the cascade produced a JSON value.

    Carries a truncated preview of the text, because "invalid JSON" without the
    offending output is an unactionable bug report.
    """

    def __init__(self, text: str | None) -> None:
        preview = (text or "").strip()
        if len(preview) > MAX_PREVIEW:
            preview = preview[:MAX_PREVIEW] + f"... (+{len(text or '') - MAX_PREVIEW} chars)"
        super().__init__(
            "no JSON value found in the model output"
            + (f"\n  model emitted: {preview!r}" if preview else " (output was empty)")
        )
        self.text = text
        self.preview = preview


@dataclass(slots=True)
class JsonExtraction:
    value: Any
    stage: JsonStage

    @property
    def lenient(self) -> bool:
        """True when our parser, not the model, produced well-formed JSON."""
        return self.stage is not JsonStage.DIRECT


def extract_json(text: str | None) -> JsonExtraction:
    """Pull the first JSON value out of model output, reporting which stage won.

    Raises :class:`JsonExtractionError` when every stage fails.
    """
    if text is None or not text.strip():
        raise JsonExtractionError(text)

    value = _try_loads(text)
    if value is not _MISS:
        return JsonExtraction(value, JsonStage.DIRECT)

    for body in _fenced_bodies(text):
        value = _try_loads(body)
        if value is not _MISS:
            return JsonExtraction(value, JsonStage.FENCED)

    # Scan the fence bodies first: a preamble plus a fenced block is the single
    # most common SLM shape, and the body is the part the model meant.
    for candidate in [*_fenced_bodies(text), text]:
        for chunk in _balanced_chunks(candidate):
            value = _try_loads(chunk)
            if value is not _MISS:
                return JsonExtraction(value, JsonStage.BALANCED)

    for candidate in [*_fenced_bodies(text), text]:
        for chunk in [*_balanced_chunks(candidate), candidate]:
            value = _repair(chunk)
            if value is not _MISS:
                return JsonExtraction(value, JsonStage.REPAIRED)

    raise JsonExtractionError(text)


class _Miss:
    __slots__ = ()

    def __repr__(self) -> str:
        return "<miss>"


_MISS = _Miss()


def _try_loads(text: str) -> Any:
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        return _MISS


def _fenced_bodies(text: str) -> list[str]:
    """Every fenced block, in order.

    Plural because models sometimes emit a worked example and then the real
    answer; taking the first complete *parseable* one is the best guess.
    """
    return [m.group("body") for m in FENCE_RE.finditer(text)]


def _balanced_chunks(text: str) -> list[str]:
    """Brace/bracket-balanced substrings, outermost-first, in document order.

    String-aware: a ``}`` inside a quoted value does not close the object, and
    neither does an escaped quote. Without that, any model output containing a
    JSON-looking string gets truncated mid-object.
    """
    chunks: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch not in "{[":
            i += 1
            continue
        end = _scan_balanced(text, i)
        if end is None:
            # Unterminated from here on; nothing later can be complete either
            # unless it starts after this opener, so advance by one.
            i += 1
            continue
        chunks.append(text[i : end + 1])
        i = end + 1
    return chunks


def _scan_balanced(text: str, start: int) -> int | None:
    """Index of the character closing the value opened at `start`, or None."""
    stack: list[str] = []
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                return None
            opener = stack.pop()
            if (opener == "{") != (ch == "}"):
                return None
            if not stack:
                return i
    return None


def _repair(text: str) -> Any:
    """Last resort: light syntactic repair, then a literal eval.

    ``ast.literal_eval`` handles single quotes and ``True``/``False``/``None``
    in one shot, which covers models that print a Python dict. It evaluates
    literals only - no names, no calls - so there is nothing to execute. The
    result is still checked for JSON-representable types, because a tuple or a
    set would not survive the round-trip into a cassette.
    """
    candidate = text.translate(SMART_QUOTES)
    candidate = TRAILING_COMMA_RE.sub(r"\1", candidate)
    value = _try_loads(candidate)
    if value is not _MISS:
        return value

    try:
        literal = ast.literal_eval(candidate.strip())
    except (ValueError, SyntaxError, MemoryError, RecursionError):
        return _MISS
    if not _json_safe(literal):
        return _MISS
    return literal


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, list):
        return all(_json_safe(v) for v in value)
    if isinstance(value, dict):
        return all(isinstance(k, str) and _json_safe(v) for k, v in value.items())
    return False


__all__ = ["JsonExtraction", "JsonExtractionError", "extract_json"]
