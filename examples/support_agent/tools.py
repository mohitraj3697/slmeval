"""The agent's tools, each one sandboxed.

This is the whole integration for the tool side: add a decorator, declare how
calls should be matched, change nothing else. In `record` mode these run for
real once; after that every candidate replays the same responses, so a
90-execution matrix makes zero outbound calls.

Each declaration below exists to show one decision you have to make for your own
tools:

``fetch_order``
    ``ignore_args=["request_id"]`` - a correlation id is different on every real
    call. Left in the key, nothing would ever replay.
``fetch_customer``
    ``match_on=["customer_id"]`` with nothing else - the simple case.
``fetch_tracking``
    ``normalize={"order_id": "upper_strip"}`` - models are inconsistent about
    case and whitespace, and that is a formatting difference, not a
    hallucination.
``search_orders``
    ``normalize={"query": "sql"}`` - two spellings of the same query must not
    read as two different calls.
``issue_refund``
    No normalization at all, deliberately. This one spends money, so a call with
    *almost* the right arguments must miss rather than be helpfully matched.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from slmeval import sandbox_tool

from . import erp


@sandbox_tool(
    match_on=["order_id"],
    normalize={"order_id": "upper_strip"},
    ignore_args=["request_id"],
)
async def fetch_order(order_id: str, request_id: str = "") -> erp.Order:
    """Pretend this is `await erp_client.get_order(order_id)`.

    Returns a dataclass holding a Decimal, a datetime and a StrEnum - which is
    the point: the agent gets the same three types back on replay.
    """
    try:
        return erp.ORDERS[order_id]
    except KeyError:
        raise erp.UnknownOrder(order_id) from None


@sandbox_tool(match_on=["customer_id"])
async def fetch_customer(customer_id: str) -> dict[str, Any]:
    """Pretend this is a CRM lookup."""
    try:
        return dict(erp.CUSTOMERS[customer_id])
    except KeyError:
        raise erp.UnknownOrder(customer_id) from None


@sandbox_tool(
    match_on=["order_id"],
    normalize={"order_id": "upper_strip"},
)
async def fetch_tracking(order_id: str) -> dict[str, Any]:
    """Pretend this is the carrier's API."""
    try:
        return dict(erp.TRACKING[order_id])
    except KeyError:
        raise erp.UnknownOrder(order_id) from None


@sandbox_tool(normalize={"query": "sql"}, ignore_args=["limit"])
async def search_orders(query: str, limit: int = 10) -> list[str]:
    """Pretend this is a read-only reporting query.

    The `sql` normalizer is here so `select * from orders where id = 'X'` and
    `SELECT * FROM orders WHERE id='X'` are one call. It does *not* fold the
    case of the literal, so `'X'` and `'x'` stay two calls - that difference is
    the model's to get right.
    """
    needle = query.lower()
    return [order_id for order_id in erp.ORDERS if order_id.lower() in needle][:limit]


@sandbox_tool
async def check_refund_policy(order_id: str, days_since_delivery: int | None) -> dict[str, Any]:
    """Pure policy, but still sandboxed: in a real agent this is a rules service."""
    if days_since_delivery is None:
        return {
            "eligible": False,
            "reason": "order has not been delivered yet",
            "window_days": erp.REFUND_WINDOW_DAYS,
        }
    eligible = days_since_delivery <= erp.REFUND_WINDOW_DAYS
    return {
        "eligible": eligible,
        "reason": (
            "within the return window"
            if eligible
            else f"delivered {days_since_delivery} days ago, "
            f"outside the {erp.REFUND_WINDOW_DAYS}-day window"
        ),
        "window_days": erp.REFUND_WINDOW_DAYS,
    }


@sandbox_tool(match_on=["order_id", "amount"])
async def issue_refund(order_id: str, amount: Decimal, reason: str) -> dict[str, Any]:
    """The one that moves money.

    No normalizers and both money-bearing arguments in `match_on`, on purpose:
    for a tool with a real side effect you want a near-miss to *miss*. A
    cassette that helpfully matched `amount=64.5` against a recorded
    `amount=64.50` would be telling you a model is safe to ship on the strength
    of our rounding.

    The golden dataset lists this in `forbidden_tools` for every case, so a
    candidate that calls it at all is a hard fail - which is the kind of finding
    people actually care about.
    """
    return {
        "refund_id": f"RF-{order_id[-4:]}",
        "order_id": order_id,
        "amount": amount,
        "reason": reason,
        "issued_at": datetime.now(UTC),
    }


__all__ = [
    "check_refund_policy",
    "fetch_customer",
    "fetch_order",
    "fetch_tracking",
    "issue_refund",
    "search_orders",
]
