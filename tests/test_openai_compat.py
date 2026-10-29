"""The HTTP client, against a scripted transport.

Two things are being checked here. First, that each ladder rung puts the schema
where the OpenAI dialect expects it - get that wrong and the benchmark measures
our request bug rather than the model. Second, that a server rejecting a
mechanism is classified as "this candidate cannot do that rung" rather than as a
crash, because that classification is the finding.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from slmeval.providers.base import (
    ClientSpec,
    ProviderRequestError,
    ProviderServerError,
    ProviderUnreachable,
    RetryPolicy,
    StructuredOutputError,
)
from slmeval.providers.openai_compat import OpenAICompatClient

pytestmark = pytest.mark.asyncio

GOOD = '{"intent": "refund", "order_id": "ORD-1"}'


class Routing(BaseModel):
    intent: str
    order_id: str


def reply(content: str, *, usage: dict[str, int] | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}}],
    }
    if usage is not None:
        body["usage"] = usage
    return body


def tool_reply(name: str, arguments: str) -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_0",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                }
            }
        ]
    }


def client_for(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    structured_output: str = "auto",
    provider: str = "openai_compatible",
    **kw: Any,
) -> OpenAICompatClient:
    spec = ClientSpec(
        id="cand",
        provider=provider,
        model="test-model",
        base_url="http://localhost:9999/v1",
        structured_output=structured_output,
        retry=RetryPolicy(attempts=3, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
        **kw,
    )
    return OpenAICompatClient(spec, http=httpx.AsyncClient(transport=httpx.MockTransport(handler)))


def recorder() -> tuple[list[dict[str, Any]], Callable[[httpx.Request], httpx.Response]]:
    seen: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=reply(GOOD))

    return seen, handler


# --------------------------------------------------------------------------- #
# Rung mechanics
# --------------------------------------------------------------------------- #


async def test_the_tools_rung_forces_a_call_to_the_schema_tool() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tool_choice"] == {"type": "function", "function": {"name": "Routing"}}
        assert [t["function"]["name"] for t in body["tools"]] == ["Routing"]
        return httpx.Response(200, json=tool_reply("Routing", GOOD))

    client = client_for(handler, structured_output="tools")
    result = await client.complete("hi", schema=Routing)
    assert result.structured_mode_used == "tools"
    assert isinstance(result.parsed, Routing)
    await client.aclose()


async def test_the_json_schema_rung_sends_a_strict_response_format() -> None:
    seen, handler = recorder()
    client = client_for(handler, structured_output="json_schema")
    await client.complete("hi", schema=Routing)

    fmt = seen[0]["response_format"]
    assert fmt["type"] == "json_schema"
    assert fmt["json_schema"]["strict"] is True
    schema = fmt["json_schema"]["schema"]
    # Strict mode requires both of these on every object, or the API 400s.
    assert schema["additionalProperties"] is False
    assert sorted(schema["required"]) == ["intent", "order_id"]
    await client.aclose()


async def test_the_json_mode_rung_sends_json_object_and_the_schema_in_prompt() -> None:
    seen, handler = recorder()
    client = client_for(handler, structured_output="json_mode")
    await client.complete("where is my order", schema=Routing)

    assert seen[0]["response_format"] == {"type": "json_object"}
    # OpenAI rejects json_object mode unless the messages mention JSON.
    assert "JSON" in seen[0]["messages"][-1]["content"]
    await client.aclose()


async def test_the_prompt_rung_sends_no_response_format() -> None:
    seen, handler = recorder()
    client = client_for(handler, structured_output="prompt")
    await client.complete("hi", schema=Routing)

    assert "response_format" not in seen[0]
    assert "tools" not in seen[0]
    assert "JSON Schema" in seen[0]["messages"][-1]["content"]
    await client.aclose()


async def test_extra_body_is_passed_through() -> None:
    seen, handler = recorder()
    client = client_for(handler, structured_output="prompt", extra_body={"seed": 7, "top_p": 0.9})
    await client.complete("hi", schema=Routing)

    assert seen[0]["seed"] == 7
    assert seen[0]["top_p"] == 0.9
    await client.aclose()


async def test_ollama_says_so_when_v1_will_drop_an_option() -> None:
    """Setting num_ctx and silently getting the default means the user measured
    the wrong thing with no way to know."""
    _, handler = recorder()
    client = client_for(
        handler, provider="ollama", structured_output="prompt", extra_body={"num_ctx": 8192}
    )
    with pytest.warns(UserWarning, match="num_ctx"):
        await client.complete("hi", schema=Routing)
    await client.aclose()


# --------------------------------------------------------------------------- #
# A rejected mechanism is data
# --------------------------------------------------------------------------- #


async def test_a_400_naming_response_format_drops_the_rung_not_the_run() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "tools" in body:
            calls.append("tools")
            return httpx.Response(400, json={"error": {"message": "tools are not supported"}})
        fmt = body.get("response_format", {}).get("type")
        if fmt == "json_schema":
            calls.append("json_schema")
            return httpx.Response(
                400, json={"error": {"message": "response_format json_schema is unsupported"}}
            )
        calls.append(fmt or "prompt")
        return httpx.Response(200, json=reply(GOOD))

    client = client_for(handler)
    result = await client.complete("hi", schema=Routing)

    assert calls == ["tools", "json_schema", "json_object"]
    assert result.structured_mode_used == "json_mode"
    assert client.unsupported == frozenset({"tools", "json_schema"})
    await client.aclose()


async def test_a_rejected_rung_is_remembered_across_calls() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        attempts.append(1)
        if "tools" in body:
            return httpx.Response(400, json={"error": {"message": "tool use not supported"}})
        return httpx.Response(200, json=reply(GOOD))

    client = client_for(handler, structured_output="auto")
    await client.complete("hi", schema=Routing)
    first = len(attempts)
    await client.complete("hi", schema=Routing)

    # The second call must not pay to rediscover the same rejection.
    assert len(attempts) - first == 1
    await client.aclose()


async def test_a_500_is_a_server_error_not_an_unsupported_rung() -> None:
    client = client_for(
        lambda r: httpx.Response(500, text="internal error"), structured_output="prompt"
    )
    with pytest.raises(ProviderServerError):
        await client.complete("hi", schema=Routing)
    await client.aclose()


async def test_a_plain_400_is_not_mistaken_for_an_unsupported_rung() -> None:
    """Swallowing a real request bug as "the model can't do this" would make
    every candidate look broken for the same reason."""
    client = client_for(
        lambda r: httpx.Response(400, json={"error": {"message": "context length exceeded"}}),
        structured_output="prompt",
    )
    with pytest.raises(ProviderRequestError, match="context length"):
        await client.complete("hi", schema=Routing)
    await client.aclose()


async def test_a_401_names_the_environment_variable_to_check() -> None:
    client = client_for(
        lambda r: httpx.Response(401, text="bad key"),
        structured_output="prompt",
        api_key="secret-value",
        api_key_env="MY_KEY",
    )
    with pytest.raises(ProviderRequestError) as excinfo:
        await client.complete("hi", schema=Routing)
    assert "$MY_KEY" in str(excinfo.value)
    # And never the value itself.
    assert "secret-value" not in str(excinfo.value)
    await client.aclose()


async def test_a_404_suggests_the_base_url() -> None:
    client = client_for(lambda r: httpx.Response(404, text="nope"), structured_output="prompt")
    with pytest.raises(ProviderRequestError, match="base_url"):
        await client.complete("hi", schema=Routing)
    await client.aclose()


async def test_a_connection_error_names_the_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    client = client_for(handler, structured_output="prompt")
    with pytest.raises(ProviderUnreachable, match="localhost:9999"):
        await client.complete("hi", schema=Routing)
    await client.aclose()


async def test_rate_limits_are_retried_and_counted_separately() -> None:
    attempts: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 3:
            return httpx.Response(429, text="slow down")
        return httpx.Response(200, json=reply(GOOD))

    client = client_for(handler, structured_output="prompt")
    result = await client.complete("hi", schema=Routing)

    assert len(attempts) == 3
    assert result.retries == 2
    # Latency is the final attempt alone; a candidate needing three retries per
    # call must not hide behind a healthy-looking p95.
    assert result.latency_ms >= 0.0
    await client.aclose()


# --------------------------------------------------------------------------- #
# Usage honesty
# --------------------------------------------------------------------------- #


async def test_reported_usage_is_used_as_is() -> None:
    client = client_for(
        lambda r: httpx.Response(
            200, json=reply(GOOD, usage={"prompt_tokens": 41, "completion_tokens": 7})
        ),
        structured_output="prompt",
    )
    result = await client.complete("hi", schema=Routing)

    assert (result.prompt_tokens, result.completion_tokens) == (41, 7)
    assert result.tokens_estimated is False
    await client.aclose()


async def test_missing_usage_is_estimated_and_flagged() -> None:
    client = client_for(lambda r: httpx.Response(200, json=reply(GOOD)), structured_output="prompt")
    result = await client.complete("hi", schema=Routing)

    assert result.prompt_tokens is not None
    assert result.completion_tokens is not None
    assert result.tokens_estimated is True
    await client.aclose()


async def test_a_tool_call_with_no_usage_block_is_not_free() -> None:
    """The tools rung returns no message content; estimating from content alone
    would report zero completion tokens and a zero marginal cost."""
    client = client_for(
        lambda r: httpx.Response(200, json=tool_reply("Routing", GOOD)),
        structured_output="tools",
    )
    result = await client.complete("hi", schema=Routing)

    assert result.tokens_estimated is True
    assert result.completion_tokens is not None
    assert result.completion_tokens > 0
    await client.aclose()


async def test_half_reported_usage_counts_as_not_reported() -> None:
    """A half-measured total is the kind of number that looks authoritative
    and is not."""
    client = client_for(
        lambda r: httpx.Response(200, json=reply(GOOD, usage={"prompt_tokens": 41})),
        structured_output="prompt",
    )
    result = await client.complete("hi", schema=Routing)
    assert result.tokens_estimated is True
    await client.aclose()


# --------------------------------------------------------------------------- #
# Response parsing
# --------------------------------------------------------------------------- #


async def test_broken_tool_arguments_go_through_the_cascade() -> None:
    client = client_for(
        lambda r: httpx.Response(200, json=tool_reply("Routing", f"```json\n{GOOD}\n```")),
        structured_output="tools",
    )
    result = await client.complete("hi", schema=Routing)

    assert isinstance(result.parsed, Routing)
    assert result.schema_lenient is True
    assert result.tool_calls[0].raw_arguments is not None
    await client.aclose()


async def test_a_response_with_no_choices_is_a_request_error() -> None:
    client = client_for(lambda r: httpx.Response(200, json={"id": "x"}), structured_output="prompt")
    with pytest.raises(ProviderRequestError, match="no choices"):
        await client.complete("hi")
    await client.aclose()


async def test_non_json_from_the_server_is_reported_with_its_body() -> None:
    client = client_for(
        lambda r: httpx.Response(200, text="<html>proxy error</html>"), structured_output="prompt"
    )
    with pytest.raises(ProviderRequestError, match="proxy error"):
        await client.complete("hi")
    await client.aclose()


async def test_prose_only_output_exhausts_the_ladder() -> None:
    client = client_for(
        lambda r: httpx.Response(200, json=reply("I am unable to answer that.")),
        structured_output="prompt",
    )
    with pytest.raises(StructuredOutputError) as excinfo:
        await client.complete("hi", schema=Routing)
    assert excinfo.value.last_text == "I am unable to answer that."
    await client.aclose()


async def test_a_call_without_a_schema_passes_the_callers_tools_through() -> None:
    seen, _ = recorder()

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=tool_reply("fetch_order", '{"order_id": "A1"}'))

    client = client_for(handler)
    from slmeval.providers.base import ToolSpec

    tool = ToolSpec(name="fetch_order", parameters={"type": "object", "properties": {}})
    result = await client.complete("hi", tools=[tool], tool_choice="auto")

    assert seen[0]["tool_choice"] == "auto"
    assert result.tool_calls[0].name == "fetch_order"
    assert result.tool_calls[0].arguments == {"order_id": "A1"}
    assert result.structured_mode_used is None
    await client.aclose()


async def test_a_timeout_mentions_cold_local_models() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("too slow", request=request)

    client = client_for(handler, structured_output="prompt")
    with pytest.raises(ProviderUnreachable, match="cold local model"):
        await client.complete("hi")
    await client.aclose()


async def test_unsupported_detection_requires_both_a_keyword_and_a_marker() -> None:
    """"tools" plus "rate limit" is not an unsupported-feature signal."""
    assert OpenAICompatClient._looks_unsupported("tool calling is not supported", "tools")
    assert not OpenAICompatClient._looks_unsupported("your tool quota is exhausted", "tools")
    assert not OpenAICompatClient._looks_unsupported("something is unsupported", "tools")


async def test_a_bad_message_role_fails_before_any_request() -> None:
    called: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        called.append(1)
        return httpx.Response(200, json=reply(GOOD))

    client = client_for(handler)
    with pytest.raises(ValueError, match="role"):
        await client.complete([{"role": "robot", "content": "hi"}])
    assert called == []
    await client.aclose()


async def test_unsupported_feature_raised_during_repair_does_not_mask_the_failure() -> None:
    """The repair re-ask must not turn a model failure into a transport error."""
    state = {"calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return httpx.Response(200, json=reply('{"intent": "refund"}'))
        return httpx.Response(400, json={"error": {"message": "response_format unsupported"}})

    client = client_for(handler, structured_output="json_mode")
    with pytest.raises(StructuredOutputError):
        await client.complete("hi", schema=Routing)
    await client.aclose()
