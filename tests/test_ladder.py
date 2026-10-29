"""The structured-output ladder.

The product claim is that slmeval can say *which mechanism a candidate actually
needed*. These tests are that claim, pinned: for each shape of model behaviour,
`structured_mode_used`, `rungs_tried`, `json_stage` and `repair_attempts` must
describe what really happened - not what we hoped happened.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from slmeval.core.trace import JsonStage
from slmeval.providers.base import ClientSpec, StructuredOutputError, ToolSpec, UnsupportedFeature
from slmeval.providers.fake import FakeClient, FakeTurn

pytestmark = pytest.mark.asyncio

GOOD = '{"intent": "refund", "order_id": "ORD-1"}'


class Routing(BaseModel):
    intent: str
    order_id: str


def spec(structured_output: str = "auto") -> ClientSpec:
    return ClientSpec(id="cand", provider="fake", model="m", structured_output=structured_output)


async def test_a_capable_model_uses_the_top_rung() -> None:
    client = FakeClient(spec(), script=[FakeTurn(text=GOOD)])
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "tools"
    assert result.rungs_tried == ("tools",)
    assert result.repair_attempts == 0
    assert result.json_stage is JsonStage.DIRECT
    assert isinstance(result.parsed, Routing)
    assert result.parsed.order_id == "ORD-1"
    assert client.calls == 1


async def test_a_small_model_lands_on_json_mode_with_a_lenient_parse() -> None:
    """The headline finding: 'needed json_mode, and our parser had to help'."""
    client = FakeClient(
        spec(),
        script=[FakeTurn(text=f"Sure! Here is the JSON:\n```json\n{GOOD}\n```")],
        unsupported=["tools", "json_schema"],
    )
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "json_mode"
    assert result.rungs_tried == ("json_mode",)
    assert result.json_stage is JsonStage.FENCED
    assert result.schema_lenient is True


async def test_the_ladder_walks_down_past_rejected_rungs() -> None:
    client = FakeClient(
        spec(),
        script=[FakeTurn(text=GOOD, unsupported=frozenset({"tools", "json_schema", "json_mode"}))],
    )
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "prompt"
    # Every rung attempted is reported, including the ones the server refused:
    # "this server cannot do json_schema" is itself a result.
    assert result.rungs_tried == ("tools", "json_schema", "json_mode", "prompt")


async def test_a_rejected_rung_is_only_discovered_once() -> None:
    """900 calls must not each pay a 400 to learn the same fact."""
    client = FakeClient(
        spec(),
        script=[FakeTurn(text=GOOD, unsupported=frozenset({"tools"}))],
    )
    first = await client.complete("hi", schema=Routing)
    assert first.rungs_tried == ("tools", "json_schema")
    assert client.unsupported == frozenset({"tools"})

    second = await client.complete("hi", schema=Routing)
    assert second.rungs_tried == ("json_schema",)


async def test_pinning_a_rung_stops_the_harness_flattering_the_model() -> None:
    client = FakeClient(spec("prompt"), script=[FakeTurn(text=GOOD)])
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "prompt"
    assert result.rungs_tried == ("prompt",)
    assert client.seen[0]["rung"] == "prompt"


async def test_a_pinned_rung_the_server_refuses_fails_loudly() -> None:
    client = FakeClient(
        spec("json_schema"), script=[FakeTurn(text=GOOD)], unsupported=["json_schema"]
    )
    with pytest.raises(UnsupportedFeature, match="pinned to 'json_schema'"):
        await client.complete("hi", schema=Routing)


async def test_one_repair_re_ask_and_it_is_counted() -> None:
    client = FakeClient(
        spec("prompt"),
        script=[FakeTurn(text='{"intent": "refund"}'), FakeTurn(text=GOOD)],
    )
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "prompt"
    assert result.repair_attempts == 1
    assert client.calls == 2


async def test_the_repair_prompt_carries_the_validation_error() -> None:
    """Re-asking without showing the error gets the same invalid output back."""
    client = FakeClient(
        spec("prompt"),
        script=[FakeTurn(text='{"intent": "refund"}'), FakeTurn(text=GOOD)],
    )
    await client.complete("hi", schema=Routing)

    repair_prompt = client.seen[1]["messages"][-1]["content"]
    assert "order_id" in repair_prompt
    assert "did not validate" in repair_prompt


async def test_repair_is_not_a_loop() -> None:
    """Three repairs would turn an unusable model into a passing one."""
    client = FakeClient(spec("prompt"), script=[FakeTurn(text='{"intent": "refund"}')])
    with pytest.raises(StructuredOutputError):
        await client.complete("hi", schema=Routing)
    assert client.calls == 2


async def test_exhaustion_reports_the_diagnosis() -> None:
    client = FakeClient(spec(), script=[FakeTurn(text="I cannot help with that.")])
    with pytest.raises(StructuredOutputError) as excinfo:
        await client.complete("hi", schema=Routing)

    error = excinfo.value
    assert error.schema_name == "Routing"
    assert error.rungs_tried == ("tools", "json_schema", "json_mode", "prompt")
    assert error.repair_attempts == 1
    assert error.last_text == "I cannot help with that."
    assert "Routing" in str(error)


async def test_the_tools_rung_fails_when_the_model_answers_in_prose() -> None:
    """A model that ignores a forced tool call is a finding, not a parse bug."""
    client = FakeClient(
        spec("tools"),
        script=[FakeTurn(text="I'll look that up for you.", refuse_tool_call=True)],
    )
    with pytest.raises(StructuredOutputError) as excinfo:
        await client.complete("hi", schema=Routing)
    assert "did not call the forced tool" in (excinfo.value.validation_error or "")


async def test_the_schema_instruction_goes_on_the_last_user_turn() -> None:
    """A trailing system message is silently dropped by several chat templates."""
    client = FakeClient(spec("prompt"), script=[FakeTurn(text=GOOD)])
    await client.complete(
        [
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "where is my order"},
        ],
        schema=Routing,
    )
    messages = client.seen[0]["messages"]
    assert len(messages) == 2
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("where is my order")
    assert "JSON Schema" in messages[-1]["content"]


async def test_the_instruction_is_appended_when_the_last_turn_is_not_a_user_turn() -> None:
    client = FakeClient(spec("prompt"), script=[FakeTurn(text=GOOD)])
    await client.complete([{"role": "assistant", "content": "thinking"}], schema=Routing)
    messages = client.seen[0]["messages"]
    assert [m["role"] for m in messages] == ["assistant", "user"]


async def test_callers_own_tools_disable_the_tools_rung() -> None:
    """Forcing a schema tool would drop the agent's tools and measure a
    different run, so the rung is skipped rather than silently substituted."""
    client = FakeClient(spec(), script=[FakeTurn(text=GOOD)])
    tool = ToolSpec(name="fetch_order", parameters={"type": "object", "properties": {}})
    result = await client.complete("hi", schema=Routing, tools=[tool])

    assert result.structured_mode_used == "json_schema"
    assert "tools" not in result.rungs_tried
    assert client.seen[0]["tools"] == ["fetch_order"]


async def test_no_schema_means_no_ladder() -> None:
    client = FakeClient(spec(), script=[FakeTurn(text="plain answer")])
    result = await client.complete("hi")

    assert result.text == "plain answer"
    assert result.parsed is None
    assert result.structured_mode_used is None
    assert result.rungs_tried == ()
    assert client.seen[0]["rung"] is None


async def test_tool_call_arguments_go_through_the_same_cascade() -> None:
    """Small models emit broken JSON inside `arguments` too."""
    client = FakeClient(spec("tools"), script=[FakeTurn(text=f"```json\n{GOOD}\n```")])
    result = await client.complete("hi", schema=Routing)

    assert result.structured_mode_used == "tools"
    assert result.json_stage is JsonStage.FENCED
    assert result.schema_lenient is True
