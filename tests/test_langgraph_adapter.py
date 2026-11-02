"""The LangGraph adapter, and the parity claim it has to make true.

Step 7's acceptance criterion is one sentence: *the example agent runs under
both adapters and produces identical node traces*. That is the whole point of
having an adapter at all - if instrumenting a graph produced a different
trajectory from marking it by hand, every comparison between a LangGraph user's
numbers and anyone else's would be meaningless, and the Tier 2 metrics computed
from the trajectory would be measuring the adapter.

LangGraph is an optional extra, so the parity test is written against a stub
builder with the same public shape: ``add_node(name, func)`` and a ``nodes``
dict. That is not a convenience - it is the contract this adapter is
deliberately limited to, because the executor that *calls* those nodes has
changed across LangGraph versions and the shape of the builder has not. The
real-library test below runs the identical assertions when langgraph is
installed, and skips when it is not.

The other thing pinned here is what the wrapper records and what it refuses to
record. A node that raised leaves a trace with an error type - not a missing
trace, which is how the blueprint's wrapper silently shrank the denominator of
every per-node rate - and never a schema verdict, because a `KeyError` in a
node says nothing about whether the model's JSON validated.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from slmeval.adapters.instrument import ERROR_CHARS, instrument_node, state_keys
from slmeval.adapters.langgraph import MARK, instrument_langgraph
from slmeval.core.context import RunContext, no_context, node, use_context
from slmeval.errors import ConfigError


class StubSpec:
    """What a builder keeps for a node: a name's worth of callable."""

    def __init__(self, runnable: Any) -> None:
        self.runnable = runnable


class StubRunnable:
    """LangGraph's own wrapper shape: a `func` and/or an `afunc`."""

    def __init__(self, func: Any = None, afunc: Any = None) -> None:
        self.func = func
        self.afunc = afunc


class StubBuilder:
    """`add_node(name, func)` plus a `nodes` dict. The adapter's whole world."""

    def __init__(self) -> None:
        self.nodes: dict[str, StubSpec] = {}

    def add_node(self, node: Any = None, action: Any = None, **kwargs: Any) -> StubBuilder:
        if action is None and callable(node):
            name, func = node.__name__, node
        else:
            name, func = node, action
        self.nodes[name] = StubSpec(StubRunnable(afunc=func))
        return self

    async def run(self, state: dict[str, Any]) -> dict[str, Any]:
        """Call every node in insertion order, threading the state."""
        for spec in self.nodes.values():
            state = await spec.runnable.afunc(state)
        return state


def fresh_context() -> RunContext:
    return RunContext(run_id="r", candidate_id="c", case_id="case", attempt=0)


async def route(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "intent": "order_status"}


async def lookup(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "order": "ORD-9921"}


async def compose(state: dict[str, Any]) -> dict[str, Any]:
    return {**state, "final_answer": "shipped"}


# --------------------------------------------------------------------------- #
# Parity: the acceptance criterion
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_both_adapters_produce_the_same_trajectory() -> None:
    """Instrumented graph, hand-marked coroutine, identical node traces."""

    async def by_hand(state: dict[str, Any]) -> dict[str, Any]:
        with node("intent_router"):
            state = await route(state)
        with node("order_lookup"):
            state = await lookup(state)
        with node("compose_reply"):
            state = await compose(state)
        return state

    builder = instrument_langgraph(StubBuilder())
    builder.add_node("intent_router", route)
    builder.add_node("order_lookup", lookup)
    builder.add_node("compose_reply", compose)

    graph_ctx = fresh_context()
    with use_context(graph_ctx):
        graph_state = await builder.run({"user_query": "where is ORD-9921"})

    hand_ctx = fresh_context()
    with use_context(hand_ctx):
        hand_state = await by_hand({"user_query": "where is ORD-9921"})

    assert graph_state == hand_state
    assert graph_ctx.node_names == hand_ctx.node_names == [
        "intent_router",
        "order_lookup",
        "compose_reply",
    ]
    assert [v.path for v in graph_ctx.trajectory] == [v.path for v in hand_ctx.trajectory]
    assert [v.seq for v in graph_ctx.trajectory] == [0, 1, 2]


@pytest.mark.asyncio
async def test_instrumenting_after_the_nodes_are_added_also_works() -> None:
    """The other order. Supported, and held to the same trajectory."""
    builder = StubBuilder()
    builder.add_node("intent_router", route)
    builder.add_node("order_lookup", lookup)
    instrument_langgraph(builder)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({})
    assert ctx.node_names == ["intent_router", "order_lookup"]


@pytest.mark.asyncio
async def test_a_second_call_does_not_double_every_node() -> None:
    """Two calls - a test fixture and the graph's own factory - would otherwise
    enter each node twice, and Tier 2 would read that as the agent looping."""
    builder = instrument_langgraph(StubBuilder())
    builder.add_node("intent_router", route)
    instrument_langgraph(builder)
    instrument_langgraph(builder)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({})
    assert ctx.node_names == ["intent_router"]


@pytest.mark.asyncio
async def test_a_prefix_keeps_two_composed_graphs_apart() -> None:
    """Two subgraphs with a `plan` node each would otherwise share a row, and
    "which plan node is failing" would have no answer."""
    builder = instrument_langgraph(StubBuilder(), prefix="research.")
    builder.add_node("plan", route)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({})
    assert ctx.node_names == ["research.plan"]


@pytest.mark.asyncio
async def test_add_node_still_accepts_a_bare_function() -> None:
    """`add_node(func)` takes the node's name from the function. A wrapper that
    handled only `add_node("name", func)` would break every graph written in
    the other style - and `functools.wraps` is what keeps `__name__` intact."""
    builder = instrument_langgraph(StubBuilder())
    builder.add_node(route)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({})
    assert ctx.node_names == ["route"]


# --------------------------------------------------------------------------- #
# What a node records
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_node_records_its_latency_and_state_keys() -> None:
    builder = instrument_langgraph(StubBuilder())
    builder.add_node("intent_router", route)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({"user_query": "q"})

    visit = ctx.trajectory[0]
    assert visit.latency_ms is not None
    assert visit.input_keys == ("user_query",)
    assert visit.output_keys == ("intent", "user_query")
    # The keys the node *added*, which is what says a node wrote nothing when
    # it should have.
    assert visit.written_keys == ("intent",)
    assert not visit.failed


@pytest.mark.asyncio
async def test_a_node_that_raises_leaves_a_trace_rather_than_a_gap() -> None:
    """The blueprint's sync wrapper lost generic exceptions entirely, so the
    denominator of every per-node rate silently shrank."""

    async def explodes(state: dict[str, Any]) -> dict[str, Any]:
        raise KeyError("customer_id")

    builder = instrument_langgraph(StubBuilder())
    builder.add_node("order_lookup", explodes)

    ctx = fresh_context()
    with use_context(ctx), pytest.raises(KeyError):
        await builder.run({"user_query": "q"})

    assert len(ctx.trajectory) == 1
    visit = ctx.trajectory[0]
    assert visit.error_type == "KeyError"
    assert "customer_id" in (visit.error or "")
    assert visit.latency_ms is not None
    # No output keys: the node did not return one. Not an empty state - a
    # state that never existed.
    assert visit.output_keys == ()


@pytest.mark.asyncio
async def test_a_long_exception_message_is_truncated() -> None:
    async def explodes(state: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError("x" * 5000)

    builder = instrument_langgraph(StubBuilder())
    builder.add_node("n", explodes)

    ctx = fresh_context()
    with use_context(ctx), pytest.raises(RuntimeError):
        await builder.run({})
    assert len(ctx.trajectory[0].error or "") == ERROR_CHARS


@pytest.mark.asyncio
async def test_nodes_nest_and_the_path_survives() -> None:
    """A node that calls a node. `reset(token)` on the way out is what keeps
    the second entry from swallowing everything after it."""

    async def outer(state: dict[str, Any]) -> dict[str, Any]:
        with node("classify"):
            pass
        return state

    builder = instrument_langgraph(StubBuilder())
    builder.add_node("intent_router", outer)
    builder.add_node("order_lookup", lookup)

    ctx = fresh_context()
    with use_context(ctx):
        await builder.run({})

    assert [v.path for v in ctx.trajectory] == [
        ("intent_router",),
        ("intent_router", "classify"),
        ("order_lookup",),
    ]


def test_a_sync_node_stays_sync() -> None:
    """A framework that calls its nodes synchronously must not be handed back
    a coroutine it never awaits."""

    def plain(state: dict[str, Any]) -> dict[str, Any]:
        return {**state, "done": True}

    wrapped = instrument_node("plain", plain)
    assert not asyncio.iscoroutinefunction(wrapped)

    ctx = fresh_context()
    with use_context(ctx):
        assert wrapped({"a": 1}) == {"a": 1, "done": True}
    assert ctx.node_names == ["plain"]


@pytest.mark.asyncio
async def test_an_instrumented_graph_is_a_no_op_in_production() -> None:
    """No context, no trajectory, no error - which is what makes it safe to
    leave `instrument_langgraph` in shipped code."""
    builder = instrument_langgraph(StubBuilder())
    builder.add_node("intent_router", route)
    with no_context():
        assert await builder.run({"q": "x"}) == {"q": "x", "intent": "order_status"}


# --------------------------------------------------------------------------- #
# Shapes and refusals
# --------------------------------------------------------------------------- #


def test_state_keys_reads_keys_and_never_values() -> None:
    """A trace store must not quietly become a copy of someone's state."""
    from dataclasses import dataclass

    from pydantic import BaseModel

    class Model(BaseModel):
        intent: str = ""
        order_id: str = ""

    @dataclass
    class Plain:
        intent: str = ""

    assert state_keys({"b": 2, "a": 1}) == ("a", "b")
    assert state_keys(Model()) == ("intent", "order_id")
    assert state_keys(Plain()) == ("intent",)
    assert state_keys("a string") == ()
    assert state_keys(None) == ()


def test_the_compiled_graph_is_refused_by_name() -> None:
    """Wrapping a compiled graph would not change what runs, so saying so is
    worth more than failing later with an empty scorecard."""
    with pytest.raises(ConfigError) as caught:
        instrument_langgraph(object())
    assert "StateGraph" in (caught.value.hint or "")


def test_an_unreachable_node_callable_says_what_to_do_instead() -> None:
    """A LangGraph version this adapter has not seen. It must not wrap nothing
    and return quietly: a scorecard with one row reads as a finding."""

    class Opaque:
        __slots__ = ()

    builder = StubBuilder()
    builder.nodes["mystery"] = Opaque()  # type: ignore[assignment]

    with pytest.raises(ConfigError) as caught:
        instrument_langgraph(builder)
    hint = caught.value.hint or ""
    assert "adding nodes" in hint
    assert "generic" in hint


def test_wrapping_marks_what_it_touched() -> None:
    builder = instrument_langgraph(StubBuilder())
    builder.add_node("n", route)
    assert getattr(builder.nodes["n"].runnable.afunc, MARK, False)


# --------------------------------------------------------------------------- #
# The real library, when it is installed
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_example_agent_runs_identically_under_both_adapters(
    tmp_path: Any,
) -> None:
    """Step 7's acceptance criterion, against the shipped example.

    ``support_agent.agent`` marks its four nodes by hand;
    ``support_agent.graph`` is the same agent as a LangGraph graph with one
    ``instrument_langgraph`` call and no node markers. Same tools in the same
    order, same trajectory, same final state - otherwise a LangGraph user's
    scorecard is not comparable with anybody else's.
    """
    pytest.importorskip("langgraph")
    import sys
    from pathlib import Path

    from slmeval.config.models import ModelCfg
    from slmeval.core.adapter import Target, execute_case
    from slmeval.core.router import ModelRouter
    from slmeval.providers.fake import FakeTurn, clear_scripts, register_script
    from slmeval.sandbox.store import CassetteStore

    examples = str(Path(__file__).resolve().parent.parent / "examples")
    sys.path.insert(0, examples)
    register_script(
        "scripted-baseline",
        [
            FakeTurn(
                text='{"intent": "order_status", "order_id": "ORD-9921", '
                '"wants_confirmation": false}'
            ),
            FakeTurn(text='{"final_answer": "Your order is with DHL."}'),
        ],
    )

    def context(store: CassetteStore, case_id: str) -> RunContext:
        return RunContext(
            run_id="parity",
            candidate_id="baseline",
            case_id=case_id,
            tool_mode="record",
            max_steps=12,
            cassettes=store,
            router=ModelRouter(
                candidate=ModelCfg(id="baseline", provider="fake", model="scripted-baseline")
            ),
        )

    case_input = {"user_query": "Where is my order ORD-9921?"}
    try:
        results = {}
        for label, entrypoint in (
            ("hand", "support_agent:run_case"),
            ("graph", "support_agent.graph:run_case"),
        ):
            with CassetteStore(tmp_path / f"{label}.sqlite") as store:
                ctx = context(store, "status_shipped")
                state = await execute_case(ctx, Target.load(entrypoint, root=examples), case_input)
                results[label] = (
                    state,
                    ctx.node_names,
                    [t.name for t in ctx.tool_calls],
                )
    finally:
        clear_scripts()
        if examples in sys.path:
            sys.path.remove(examples)
        for name in [m for m in sys.modules if m.startswith("support_agent")]:
            del sys.modules[name]

    hand_state, hand_nodes, hand_tools = results["hand"]
    graph_state, graph_nodes, graph_tools = results["graph"]

    assert hand_nodes == graph_nodes == [
        "intent_router",
        "order_lookup",
        "policy_check",
        "compose_reply",
    ]
    assert hand_tools == graph_tools
    # Same keys and same values for everything the dataset asserts on.
    for key in ("intent", "order_id", "refund_issued", "final_answer"):
        assert hand_state[key] == graph_state[key], key


@pytest.mark.asyncio
async def test_a_real_langgraph_graph_produces_the_same_trajectory() -> None:
    """The same assertions as the stub, against the library itself.

    Skipped when the extra is not installed, which is the normal state of the
    offline suite - `pip install slmeval` must not pull in a model framework.
    CI installs the extra for this one test.
    """
    pytest.importorskip("langgraph")
    from typing import TypedDict

    from langgraph.graph import END, START, StateGraph

    class State(TypedDict, total=False):
        user_query: str
        intent: str
        order: str

    builder = instrument_langgraph(StateGraph(State))
    builder.add_node("intent_router", route)
    builder.add_node("order_lookup", lookup)
    builder.add_edge(START, "intent_router")
    builder.add_edge("intent_router", "order_lookup")
    builder.add_edge("order_lookup", END)
    graph = builder.compile()

    ctx = fresh_context()
    with use_context(ctx):
        state = await graph.ainvoke({"user_query": "where is ORD-9921"})

    assert state["intent"] == "order_status"
    assert ctx.node_names == ["intent_router", "order_lookup"]
    assert all(v.latency_ms is not None for v in ctx.trajectory)
