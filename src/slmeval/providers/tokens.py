"""Token accounting, honestly.

Cost-per-case is the number that decides whether anyone downsizes, so it has to
be either measured or visibly marked as a guess. Some OpenAI-compatible servers
return no ``usage`` block at all; llama.cpp's server historically omitted it and
a few proxies strip it. When that happens the tokens are estimated and
``estimated=True`` rides along with them all the way to the report, where the
aggregate gets a ``~``. A fabricated number must never look precise.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from slmeval.providers.base import Msg, ToolCall

CHARS_PER_TOKEN = 3.7
"""Fallback ratio when tiktoken is not installed.

Measured against cl100k_base on English prose plus JSON, which is what agent
traffic mostly is. Wrong by 10-20% on code and non-Latin scripts - which is why
anything derived from it is marked estimated rather than quietly trusted.
"""


@dataclass(slots=True)
class Usage:
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    estimated: bool = False

    @property
    def total(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@lru_cache(maxsize=1)
def _encoder() -> Any:
    """cl100k_base if tiktoken happens to be installed, else None.

    tiktoken is deliberately not a core dependency: it is a Rust wheel and a
    download, and `pip install slmeval-cli` has to stay fast. It is a stand-in
    anyway - no local model uses the OpenAI tokenizer - so the estimate is
    marked as an estimate either way.
    """
    try:
        import tiktoken  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:  # pragma: no cover - network-less first use
        return None


def estimate_tokens(text: str) -> int:
    """Approximate token count. Never returns 0 for non-empty text."""
    if not text:
        return 0
    enc = _encoder()
    if enc is not None:
        return len(enc.encode(text, disallowed_special=()))
    return max(1, math.ceil(len(text) / CHARS_PER_TOKEN))


def estimate_messages(messages: Sequence[Msg]) -> int:
    """Prompt tokens for a message list, with per-message framing overhead.

    The +4 per message approximates the role/separator tokens every chat
    template adds. It is the same constant OpenAI documented for cl100k chat
    models and is close enough for a figure that reports as an estimate.
    """
    total = 0
    for m in messages:
        total += 4
        if m.content:
            total += estimate_tokens(m.content)
        for tc in m.tool_calls:
            total += estimate_tokens(tc.name)
            total += estimate_tokens(tc.raw_arguments or "")
    return total


def estimate_completion(text: str | None, tool_calls: Sequence[ToolCall] = ()) -> int:
    """Completion tokens for a response, whichever channel it arrived on.

    A tool-calling reply has no message content, so counting only `content`
    would score it at zero tokens - and understate the cost of every
    tool-calling candidate on any server that omits `usage`. Cost-per-case is
    the number this tool exists to produce; it cannot be quietly wrong.
    """
    total = estimate_tokens(text or "")
    for call in tool_calls:
        total += estimate_tokens(call.name)
        total += estimate_tokens(call.raw_arguments or json.dumps(call.arguments))
    return total


def usage_from_response(
    raw: dict[str, Any],
    *,
    messages: Sequence[Msg],
    output_text: str | None = None,
    output_tool_calls: Sequence[ToolCall] = (),
) -> Usage:
    """Read ``usage`` if the server sent it; estimate if it did not.

    A server that reports only one of the two counts is treated as not having
    reported usage at all. Half-measured, half-estimated totals are the kind of
    number that looks authoritative and is not.
    """
    block = raw.get("usage")
    if isinstance(block, dict):
        prompt = _as_int(block.get("prompt_tokens", block.get("input_tokens")))
        completion = _as_int(block.get("completion_tokens", block.get("output_tokens")))
        if prompt is not None and completion is not None:
            return Usage(prompt_tokens=prompt, completion_tokens=completion, estimated=False)

    return Usage(
        prompt_tokens=estimate_messages(messages),
        completion_tokens=estimate_completion(output_text, output_tool_calls),
        estimated=True,
    )


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


__all__ = [
    "CHARS_PER_TOKEN",
    "Usage",
    "estimate_completion",
    "estimate_messages",
    "estimate_tokens",
    "usage_from_response",
]
