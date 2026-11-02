"""The same agent, as a LangGraph graph.

Same four nodes, same five tools, same trajectory — and no ``with node(...)``
blocks anywhere, because one line does it:

    builder = slmeval.instrument_langgraph(StateGraph(State))

That is the whole LangGraph integration. ``chat_model()`` and ``@sandbox_tool``
are unchanged and come from :mod:`support_agent.agent` and
:mod:`support_agent.tools`; the adapter only supplies the node names, which is
the part nobody wants to hand-annotate across a real graph.

Benchmark it by pointing the config at this module instead:

    [target]
    entrypoint = "support_agent.graph:run_case"
    adapter = "langgraph"

The scorecard is the same scorecard. ``tests/test_langgraph_adapter.py``
asserts exactly that — this file and ``agent.py`` produce identical node traces
— because if instrumenting a graph gave a different trajectory from marking one
by hand, every Tier 2 metric would be measuring the adapter rather than the
model.

Needs the extra: ``pip install "slmeval[langgraph]"``. Nothing in slmeval's core
imports LangGraph, and neither does ``agent.py``.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

import slmeval

from . import erp
from .agent import Reply, Routing, _compose, _route
from .tools import check_refund_policy, fetch_customer, fetch_order, fetch_tracking, issue_refund


class State(TypedDict, total=False):
    """What flows between nodes.

    The asserted keys are the ones the golden dataset names, so they have to
    survive every path through the graph - including the one that makes no tool
    calls at all.
    """

    user_query: str
    intent: str
    order_id: str
    wants_confirmation: bool
    refund_eligible: bool | None
    refund_issued: bool
    refund_id: str
    final_answer: str

    # Carried between nodes rather than re-fetched. A graph that looked the
    # order up twice would make two tool calls where the hand-written agent
    # makes one, and the two trajectories would stop matching.
    order: Any
    customer: dict[str, Any]
    tracking: dict[str, Any]
    policy: dict[str, Any]


async def intent_router(state: State) -> State:
    routing: Routing = await _route(str(state.get("user_query", "")))
    return {
        **state,
        "intent": routing.intent,
        "order_id": routing.order_id,
        "wants_confirmation": routing.wants_confirmation,
    }


async def order_lookup(state: State) -> State:
    order_id = str(state["order_id"])
    order = await fetch_order(order_id, request_id=f"req-{order_id}")
    customer = await fetch_customer(order.customer_id)
    tracking = await fetch_tracking(order.order_id)

    # Replay fidelity as an assertion rather than a comment: if the envelope
    # had flattened these to strings, the example would fail here instead of
    # producing a subtly wrong answer two nodes later.
    assert isinstance(order.total, Decimal), "order.total must replay as a Decimal"
    assert order.state in erp.OrderState, "order.state must replay as an OrderState"

    return {**state, "order": order, "customer": customer, "tracking": tracking}


async def policy_check(state: State) -> State:
    order = state["order"]
    policy = await check_refund_policy(
        order_id=order.order_id,
        days_since_delivery=order.days_since_delivery,
    )
    out: State = {**state, "policy": policy, "refund_eligible": bool(policy["eligible"])}

    if state.get("intent") == "refund" and policy["eligible"] and state.get("wants_confirmation"):
        # Gated on a confirmation the customer has to have given. A candidate
        # that reaches this without one is calling a forbidden tool, and the
        # dataset makes that a hard fail.
        refund = await issue_refund(
            order_id=order.order_id,
            amount=order.total,
            reason=str(policy["reason"]),
        )
        out["refund_issued"] = True
        out["refund_id"] = refund["refund_id"]
    return out


async def compose_reply(state: State) -> State:
    if not state.get("order_id"):
        # Nothing was looked up, so there is nothing to summarise. Ask rather
        # than guess - and note this path makes no tool calls, which is what
        # gives it a different trajectory from every other case.
        return {
            **state,
            "final_answer": (
                "Could you give me your order number so I can check that for you?"
            ),
        }
    answer: str = await _compose(
        state["order"],
        state["customer"],
        state["tracking"],
        state["policy"],
        Routing(
            intent=state.get("intent", "other"),  # type: ignore[arg-type]
            order_id=str(state.get("order_id", "")),
            wants_confirmation=bool(state.get("wants_confirmation", False)),
        ),
    )
    return {**state, "final_answer": answer}


def _after_routing(state: State) -> str:
    return "order_lookup" if state.get("order_id") else "compose_reply"


def build() -> Any:
    """The graph, with every node named for the scorecard."""
    builder = slmeval.instrument_langgraph(StateGraph(State))
    builder.add_node("intent_router", intent_router)
    builder.add_node("order_lookup", order_lookup)
    builder.add_node("policy_check", policy_check)
    builder.add_node("compose_reply", compose_reply)

    builder.add_edge(START, "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        _after_routing,
        {"order_lookup": "order_lookup", "compose_reply": "compose_reply"},
    )
    builder.add_edge("order_lookup", "policy_check")
    builder.add_edge("policy_check", "compose_reply")
    builder.add_edge("compose_reply", END)
    return builder.compile()


_GRAPH: Any = None


async def run_case(case_input: dict) -> dict:
    """The entrypoint `[target]` points at. One case, start to finish.

    The graph is compiled once and reused: compiling it per execution would
    put graph construction into every latency measurement, and the matrix runs
    this forty-eight times.
    """
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build()
    state: State = {
        "user_query": str(case_input.get("user_query", "")),
        "intent": "other",
        "order_id": "",
        "refund_eligible": None,
        "refund_issued": False,
        "final_answer": "",
    }
    final = await _GRAPH.ainvoke(state)
    # The dataset asserts on these keys and nothing else. Returning the whole
    # graph state would put the ERP's Order object into the trace store, and a
    # benchmark should not quietly keep a copy of a customer's data.
    return {
        key: final[key]
        for key in ("intent", "order_id", "refund_eligible", "refund_issued", "final_answer")
        if key in final
    } | ({"refund_id": final["refund_id"]} if "refund_id" in final else {})


__all__ = ["Reply", "State", "build", "run_case"]
