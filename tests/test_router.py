"""Node routing and per-call tracing.

Routing is the experiment: "replace the model everywhere" and "replace it only
on the router node" are different questions, and the scorecard can only tell
them apart if `resolve` is right.

Tracing is the measurement. The rule under test is that *every* path through a
model call leaves exactly one trace, with a schema status that distinguishes
"the model produced bad JSON" from "the HTTP call failed". Conflating those two
inflates the one metric the tool exists to report.
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from slmeval.config import EvalConfig
from slmeval.core.context import RunContext, node, use_context
from slmeval.core.router import ModelRouter, chat_model
from slmeval.core.trace import JsonStage, SchemaStatus
from slmeval.providers.base import ProviderServerError, StructuredOutputError
from slmeval.providers.fake import FakeTurn, clear_scripts, register_script

GOOD = '{"intent": "refund", "order_id": "ORD-1"}'


class Routing(BaseModel):
    intent: str
    order_id: str


@pytest.fixture(autouse=True)
def clean_scripts() -> None:
    clear_scripts()


def config_for(candidates: list[dict[str, object]]) -> EvalConfig:
    return EvalConfig.model_validate(
        {
            "benchmark": {"name": "t", "dataset": "cases.jsonl"},
            "target": {"entrypoint": "mod:run_case"},
            "baseline": {"provider": "fake", "model": "strong"},
            "candidates": candidates,
        }
    )


# --------------------------------------------------------------------------- #
# Routing
# --------------------------------------------------------------------------- #


def test_a_candidate_with_no_target_nodes_replaces_the_model_everywhere() -> None:
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")

    assert router.resolve("intent_router").id == "small"
    assert router.resolve("order_lookup").id == "small"
    assert router.resolve(None).id == "small"


def test_a_node_scoped_candidate_leaves_the_rest_to_the_baseline() -> None:
    """The downsize-the-boring-80% experiment."""
    config = config_for(
        [
            {
                "id": "small-router",
                "provider": "fake",
                "model": "weak",
                "target_nodes": ["intent_router"],
            }
        ]
    )
    router = ModelRouter.for_candidate(config, "small-router")

    assert router.resolve("intent_router").id == "small-router"
    assert router.resolve("order_lookup").id == "baseline"


def test_unlabelled_calls_fall_to_the_baseline_for_a_scoped_candidate() -> None:
    """Crediting a router-only swap with unlabelled calls would make it look
    cheaper than it is."""
    config = config_for(
        [{"id": "scoped", "provider": "fake", "model": "weak", "target_nodes": ["a"]}]
    )
    router = ModelRouter.for_candidate(config, "scoped")
    assert router.resolve(None).id == "baseline"


def test_the_baseline_row_routes_to_itself_everywhere() -> None:
    config = config_for([])
    router = ModelRouter.for_candidate(config, "baseline")

    assert router.baseline is None
    assert router.resolve("anything").id == "baseline"
    assert router.models == [config.baseline]


def test_an_unknown_candidate_lists_the_known_ones() -> None:
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    with pytest.raises(KeyError, match="baseline, small"):
        ModelRouter.for_candidate(config, "typo")


def test_clients_are_reused_per_model() -> None:
    """One client per model, so the per-provider concurrency limit is real."""
    config = config_for(
        [{"id": "scoped", "provider": "fake", "model": "weak", "target_nodes": ["a"]}]
    )
    router = ModelRouter.for_candidate(config, "scoped")

    assert router.client_for("a") is router.client_for("a")
    assert router.client_for("a") is not router.client_for("b")


# --------------------------------------------------------------------------- #
# Tracing
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_successful_call_is_traced_with_the_rung_it_needed() -> None:
    register_script("weak", [FakeTurn(text=f"Sure:\n```json\n{GOOD}\n```")])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx), node("intent_router"):
        result = await chat_model().complete("where is my order", schema=Routing)

    assert isinstance(result.parsed, Routing)
    (trace,) = ctx.llm_calls
    assert trace.node == "intent_router"
    assert trace.node_path == ("intent_router",)
    assert trace.model_id == "small"
    assert trace.schema_name == "Routing"
    assert trace.schema_status is SchemaStatus.VALID
    assert trace.structured_mode_used == "tools"
    assert trace.json_stage is JsonStage.FENCED
    assert trace.schema_lenient is True
    assert trace.tokens_estimated is True


@pytest.mark.asyncio
async def test_a_call_with_no_schema_is_excluded_from_adherence() -> None:
    register_script("weak", [FakeTurn(text="just prose")])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx):
        await chat_model().complete("hi")

    assert ctx.llm_calls[0].schema_status is SchemaStatus.NOT_REQUESTED


@pytest.mark.asyncio
async def test_a_model_that_cannot_produce_the_schema_is_recorded_as_invalid() -> None:
    register_script("weak", [FakeTurn(text="I cannot help with that.")])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx), node("intent_router"), pytest.raises(StructuredOutputError):
        await chat_model().complete("hi", schema=Routing)

    (trace,) = ctx.llm_calls
    assert trace.schema_status is SchemaStatus.INVALID
    assert trace.rungs_tried == ("tools", "json_schema", "json_mode", "prompt")
    assert trace.repair_attempts == 1
    assert trace.error_type == "StructuredOutputError"


@pytest.mark.asyncio
async def test_a_transport_failure_is_an_error_not_a_schema_failure() -> None:
    """The blueprint's bug: an HTTP crash filed as schema_valid=True dilutes
    the adherence rate with problems the model never caused."""
    register_script("weak", [FakeTurn(status=500)])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx), pytest.raises(ProviderServerError):
        await chat_model().complete("hi", schema=Routing)

    (trace,) = ctx.llm_calls
    assert trace.schema_status is SchemaStatus.ERROR
    assert trace.error_type == "ProviderServerError"


@pytest.mark.asyncio
async def test_every_path_leaves_exactly_one_trace() -> None:
    """Recorded in `finally`, not per-except: otherwise the denominator of
    every rate in the report is wrong."""
    register_script("weak", [FakeTurn(text=GOOD), FakeTurn(status=500)])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx):
        llm = chat_model()
        await llm.complete("hi", schema=Routing)
        with pytest.raises(ProviderServerError):
            await llm.complete("hi", schema=Routing)

    assert len(ctx.llm_calls) == 2
    assert [t.seq for t in ctx.llm_calls] == [0, 1]
    assert ctx.warnings  # the failure is surfaced, not swallowed


@pytest.mark.asyncio
async def test_nested_nodes_are_attributed_to_the_innermost_one() -> None:
    register_script("weak", [FakeTurn(text=GOOD)])
    config = config_for([{"id": "small", "provider": "fake", "model": "weak"}])
    router = ModelRouter.for_candidate(config, "small")
    ctx = RunContext(run_id="r", candidate_id="small", case_id="c1", router=router)

    with use_context(ctx), node("router"), node("classify"):
        await chat_model().complete("hi", schema=Routing)

    (trace,) = ctx.llm_calls
    assert trace.node == "classify"
    assert trace.node_path == ("router", "classify")


@pytest.mark.asyncio
async def test_a_pinned_node_overrides_the_ambient_one() -> None:
    register_script("weak", [FakeTurn(text=GOOD)])
    config = config_for(
        [{"id": "scoped", "provider": "fake", "model": "weak", "target_nodes": ["pinned"]}]
    )
    router = ModelRouter.for_candidate(config, "scoped")
    ctx = RunContext(run_id="r", candidate_id="scoped", case_id="c1", router=router)

    with use_context(ctx), node("somewhere_else"):
        await chat_model(node="pinned").complete("hi", schema=Routing)

    assert ctx.llm_calls[0].model_id == "scoped"
    assert ctx.llm_calls[0].node == "pinned"


# --------------------------------------------------------------------------- #
# Production passthrough
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_chat_model_outside_a_run_says_what_to_do() -> None:
    with pytest.raises(LookupError, match="provider="):
        await chat_model().complete("hi")


@pytest.mark.asyncio
async def test_chat_model_falls_back_to_the_production_model_outside_a_run() -> None:
    """The line has to be safe to leave in shipped code."""
    register_script("prod-model", [FakeTurn(text=GOOD)])
    llm = chat_model(provider="fake", model="prod-model")
    result = await llm.complete("hi", schema=Routing)

    assert isinstance(result.parsed, Routing)
    await llm.aclose()


def test_chat_model_needs_both_provider_and_model_or_neither() -> None:
    with pytest.raises(ValueError, match="both"):
        chat_model(provider="fake")
    with pytest.raises(ValueError, match="both"):
        chat_model(model="x")


@pytest.mark.asyncio
async def test_a_context_without_a_router_is_a_passthrough() -> None:
    """A user calling their agent directly, with slmeval imported but idle."""
    register_script("prod-model", [FakeTurn(text=GOOD)])
    ctx = RunContext(run_id="r", candidate_id="x", case_id="c1")
    llm = chat_model(provider="fake", model="prod-model")

    with use_context(ctx):
        await llm.complete("hi", schema=Routing)

    assert ctx.llm_calls == []
    await llm.aclose()
