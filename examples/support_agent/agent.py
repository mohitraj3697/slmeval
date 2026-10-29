"""The example agent: four nodes, five tool calls, no framework.

This is what the `generic` adapter expects and what `[target].entrypoint` points
at - one coroutine taking a case's `input` dict and returning the final state.
There is no LangGraph, no LangChain and no slmeval base class to inherit, which
is the point: the two touch points are `node()` and `chat_model()`, and both are
no-ops when no benchmark is running.

The node split is the thing to copy. It is not cosmetic - it is the unit of
attribution, and it decides which questions the scorecard can answer:

``intent_router``
    Classify the request and pull out an order id. Short input, rigid output,
    no reasoning. **This is the node a 4B model should be able to take**, and
    the reason the example exists.
``order_lookup``
    Fan out to the ERP, the CRM and the carrier. Mostly tool calls.
``policy_check``
    Decide eligibility from what came back.
``compose_reply``
    Write the customer-facing answer. Still worth keeping on the big model.

Point a candidate at `target_nodes = ["intent_router"]` and the scorecard tells
you what swapping *just that node* costs - which is the realistic downsizing
question, and almost always a better trade than replacing the model everywhere.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, Field

from slmeval import chat_model, node

from . import erp
from .tools import (
    check_refund_policy,
    fetch_customer,
    fetch_order,
    fetch_tracking,
    issue_refund,
)

Intent = Literal["order_status", "refund", "cancel", "other"]


class Routing(BaseModel):
    """What the router node has to produce.

    Small and rigid on purpose. A schema with eleven optional fields is a schema
    a 4B model fails at for reasons that have nothing to do with the task, and
    then you have measured your schema rather than the model.
    """

    intent: Intent
    order_id: str = Field("", description="Order id if the user named one, else empty")
    wants_confirmation: bool = Field(
        False, description="True if the user has already confirmed a destructive action"
    )


class Reply(BaseModel):
    final_answer: str


async def run_case(case_input: dict) -> dict:
    """Handle one customer message end to end."""
    query = str(case_input.get("user_query", ""))
    state: dict[str, Any] = {
        "intent": "other",
        "order_id": "",
        "refund_eligible": None,
        "refund_issued": False,
        "final_answer": "",
    }

    with node("intent_router"):
        routing = await _route(query)
    state["intent"] = routing.intent
    state["order_id"] = routing.order_id

    if not routing.order_id:
        # Nothing to look up. Ask rather than guess - and note that this path
        # makes *no* tool calls, so a case that lands here and a case that does
        # not have different trajectories. That is what trajectory distance
        # measures.
        with node("compose_reply"):
            state["final_answer"] = (
                "Could you give me your order number so I can check that for you?"
            )
        return state

    with node("order_lookup"):
        order = await fetch_order(routing.order_id, request_id=f"req-{routing.order_id}")
        customer = await fetch_customer(order.customer_id)
        tracking = await fetch_tracking(order.order_id)

    # Replay fidelity, stated as an assertion rather than a comment: if the
    # envelope flattened these to strings, the example would fail here instead
    # of producing a subtly wrong answer four nodes later.
    assert isinstance(order.total, Decimal), "order.total must replay as a Decimal"
    assert order.state in erp.OrderState, "order.state must replay as an OrderState"

    with node("policy_check"):
        policy = await check_refund_policy(
            order_id=order.order_id,
            days_since_delivery=order.days_since_delivery,
        )
        state["refund_eligible"] = bool(policy["eligible"])

        if routing.intent == "refund" and policy["eligible"] and routing.wants_confirmation:
            # Gated on an explicit confirmation the user has to have given. A
            # candidate that reaches this without one is calling a forbidden
            # tool, and the dataset makes that a hard fail.
            refund = await issue_refund(
                order_id=order.order_id,
                amount=order.total,
                reason=str(policy["reason"]),
            )
            state["refund_issued"] = True
            state["refund_id"] = refund["refund_id"]

    with node("compose_reply"):
        state["final_answer"] = await _compose(order, customer, tracking, policy, routing)

    return state


async def _route(query: str) -> Routing:
    llm = chat_model()
    completion = await llm.complete(
        [
            {
                "role": "system",
                "content": (
                    "Classify the customer message. Return the intent, the order id "
                    "if one is mentioned, and whether the customer has already "
                    "confirmed a destructive action."
                ),
            },
            {"role": "user", "content": query},
        ],
        schema=Routing,
    )
    return completion.parsed_as(Routing)


async def _compose(
    order: erp.Order,
    customer: dict[str, Any],
    tracking: dict[str, Any],
    policy: dict[str, Any],
    routing: Routing,
) -> str:
    llm = chat_model()
    completion = await llm.complete(
        [
            {
                "role": "system",
                "content": "Answer the customer in two sentences. Be specific about dates.",
            },
            {
                "role": "user",
                "content": (
                    f"intent={routing.intent} order={order.order_id} "
                    f"state={order.state.value} total={order.total} "
                    f"tier={customer['tier']} carrier={tracking['carrier']} "
                    f"eta={tracking['eta']} refundable={policy['eligible']}"
                ),
            },
        ],
        schema=Reply,
    )
    return completion.parsed_as(Reply).final_answer


__all__ = ["Reply", "Routing", "run_case"]
