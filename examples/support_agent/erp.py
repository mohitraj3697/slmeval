"""A stand-in for the systems a real support agent talks to.

In your own agent these functions are HTTP calls to an ERP, a payments provider
and a shipping API. Here they are an in-memory dict, so the example runs with no
credentials - but they are shaped like the real thing in the one way that
matters for the sandbox: **they return rich Python types.**

``datetime``, ``Decimal`` and an ``Enum`` are what an ERP client actually hands
back, and they are exactly what a naive ``json.dumps(..., default=str)``
recorder would flatten into strings. Replay would then feed the agent a ``str``
where record gave it a ``Decimal``, ``order.total * Decimal("0.2")`` would raise
on replay only, and the benchmark would report a model failure that was really a
harness bug. The typed envelope is what stops that, and this module is where the
example proves it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from enum import StrEnum


class OrderState(StrEnum):
    """A StrEnum member is a `str`, which is what makes it a trap: an encoder
    that checks `isinstance(x, str)` before `Enum` records "delivered" and
    replays a bare string, so `state is OrderState.DELIVERED` quietly stops
    being true."""

    PENDING = "pending"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class Order:
    order_id: str
    customer_id: str
    state: OrderState
    total: Decimal
    placed_at: datetime
    delivered_at: datetime | None = None

    @property
    def days_since_delivery(self) -> int | None:
        if self.delivered_at is None:
            return None
        return (_NOW - self.delivered_at).days


_NOW = datetime(2026, 10, 28, 12, 0, tzinfo=UTC)

ORDERS: dict[str, Order] = {
    "ORD-9921": Order(
        order_id="ORD-9921",
        customer_id="CUS-41",
        state=OrderState.SHIPPED,
        total=Decimal("64.50"),
        placed_at=_NOW - timedelta(days=4),
    ),
    "ORD-7310": Order(
        order_id="ORD-7310",
        customer_id="CUS-41",
        state=OrderState.DELIVERED,
        total=Decimal("129.99"),
        placed_at=_NOW - timedelta(days=30),
        delivered_at=_NOW - timedelta(days=21),
    ),
}

CUSTOMERS: dict[str, dict[str, object]] = {
    "CUS-41": {
        "customer_id": "CUS-41",
        "tier": "standard",
        "lifetime_value": Decimal("1240.00"),
        "since": _NOW - timedelta(days=900),
    },
}

TRACKING: dict[str, dict[str, object]] = {
    "ORD-9921": {
        "carrier": "DHL",
        "status": "in_transit",
        "last_scan_at": _NOW - timedelta(hours=9),
        "eta": _NOW + timedelta(days=2),
        "checkpoints": ("Leipzig", "Brussels"),
    },
    "ORD-7310": {
        "carrier": "DHL",
        "status": "delivered",
        "last_scan_at": _NOW - timedelta(days=21),
        "eta": None,
        "checkpoints": ("Leipzig", "Brussels", "Manchester"),
    },
}

REFUND_WINDOW_DAYS = 28


class UnknownOrder(KeyError):
    """What the real ERP raises for an id that does not exist.

    Worth having in the example: a small model that invents an order id should
    produce a cassette miss, not a tool that politely returns ``None`` and lets
    the agent carry on as if the order were real.
    """
