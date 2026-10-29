"""Token accounting.

Cost-per-case is the number that decides whether anyone downsizes, so the only
unacceptable outcome here is a guess that does not announce itself.
"""

from __future__ import annotations

import pytest

from slmeval.providers.base import Msg, ToolCall
from slmeval.providers.registry import CLOUD, LOCAL, is_local, operational_for
from slmeval.providers.tokens import (
    estimate_completion,
    estimate_messages,
    estimate_tokens,
    usage_from_response,
)


def test_empty_text_is_zero_tokens() -> None:
    assert estimate_tokens("") == 0


def test_non_empty_text_is_never_zero_tokens() -> None:
    assert estimate_tokens("a") >= 1


def test_estimates_grow_with_length() -> None:
    assert estimate_tokens("hello " * 100) > estimate_tokens("hello")


def test_message_framing_is_counted() -> None:
    """Chat templates add role and separator tokens to every turn."""
    one = estimate_messages([Msg(role="user", content="hi")])
    two = estimate_messages([Msg(role="user", content="hi"), Msg(role="assistant", content="hi")])
    assert two > one * 1.5


def test_reported_usage_is_trusted() -> None:
    usage = usage_from_response(
        {"usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        messages=[Msg(role="user", content="hi")],
        output_text="ok",
    )
    assert (usage.prompt_tokens, usage.completion_tokens, usage.total) == (10, 3, 13)
    assert usage.estimated is False


def test_anthropic_style_usage_keys_are_read_too() -> None:
    usage = usage_from_response(
        {"usage": {"input_tokens": 8, "output_tokens": 2}},
        messages=[Msg(role="user", content="hi")],
        output_text="ok",
    )
    assert (usage.prompt_tokens, usage.completion_tokens) == (8, 2)
    assert usage.estimated is False


@pytest.mark.parametrize(
    "block",
    [
        None,
        {},
        {"prompt_tokens": 10},
        {"completion_tokens": 3},
        {"prompt_tokens": "ten", "completion_tokens": 3},
        "not a dict",
    ],
)
def test_anything_short_of_both_counts_is_an_estimate(block: object) -> None:
    """A half-measured total looks authoritative and is not."""
    raw = {} if block is None else {"usage": block}
    usage = usage_from_response(
        raw, messages=[Msg(role="user", content="where is my order")], output_text="ok"
    )
    assert usage.estimated is True
    assert usage.prompt_tokens is not None
    assert usage.completion_tokens is not None


def test_a_tool_calling_reply_is_not_scored_at_zero_tokens() -> None:
    """A tool call carries no message content. Counting only `content` would
    understate the cost of every tool-calling candidate on a server that omits
    usage - and cost-per-case is the number this tool exists to produce."""
    call = ToolCall(id="c0", name="Routing", raw_arguments='{"intent": "refund"}')
    assert estimate_completion(None, [call]) > 0
    assert estimate_completion(None, [call]) > estimate_completion(None, [])


def test_tool_call_arguments_are_counted_when_only_parsed_args_survive() -> None:
    call = ToolCall(id="c0", name="Routing", arguments={"intent": "refund"})
    assert estimate_completion(None, [call]) > 0


def test_content_and_tool_calls_are_both_counted() -> None:
    call = ToolCall(id="c0", name="t", raw_arguments='{"a": 1}')
    assert estimate_completion("some prose", [call]) > estimate_completion("some prose", [])


def test_a_tool_calling_response_without_usage_estimates_the_arguments() -> None:
    usage = usage_from_response(
        {},
        messages=[Msg(role="user", content="hi")],
        output_text=None,
        output_tool_calls=[ToolCall(id="c0", name="Routing", raw_arguments='{"intent": "refund"}')],
    )
    assert usage.estimated is True
    assert usage.completion_tokens is not None
    assert usage.completion_tokens > 0


def test_usage_total_is_none_when_nothing_is_known() -> None:
    from slmeval.providers.tokens import Usage

    assert Usage().total is None
    assert Usage(prompt_tokens=1).total == 1


@pytest.mark.parametrize(
    ("provider", "base_url", "expected"),
    [
        ("ollama", "http://localhost:11434/v1", True),
        ("ollama", "http://gpubox:11434/v1", True),
        ("openai_compatible", "http://127.0.0.1:8000/v1", True),
        ("openai_compatible", "http://workstation.local:8000/v1", True),
        ("openai_compatible", "https://api.together.xyz/v1", False),
        ("openai", "https://api.openai.com/v1", False),
    ],
)
def test_local_endpoints_are_recognised(provider: str, base_url: str, expected: bool) -> None:
    assert is_local(provider, base_url) is expected


def test_local_models_are_called_one_at_a_time() -> None:
    """A p95 taken from four concurrent requests to one GPU describes our
    scheduler, not the model."""
    assert operational_for("ollama", "http://localhost:11434/v1") is LOCAL
    assert LOCAL.max_concurrency == 1
    assert operational_for("openai", "https://api.openai.com/v1") is CLOUD
    assert CLOUD.max_concurrency > 1
