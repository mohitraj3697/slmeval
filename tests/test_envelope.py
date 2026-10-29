"""The typed envelope.

Two properties are under test, and they are the whole point of the module:

1. **Round-trip fidelity of types, not just values.** Every assertion checks
   ``type(...) is`` or an exact equality that a coerced value would fail. A test
   that only compared values would pass against ``json.dumps(default=str)``,
   which is the bug this module exists to prevent.
2. **Refusal, with a usable message.** When a value cannot be recorded
   faithfully, the error has to name the tool, the path and the type - the three
   things the user needs to go and fix their tool.
"""

from __future__ import annotations

import dataclasses
import math
from datetime import UTC, date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum, IntEnum, StrEnum
from pathlib import Path
from typing import Any
from uuid import UUID

import pytest
from pydantic import BaseModel

from slmeval.sandbox.envelope import (
    EnvelopeDecodeError,
    UnserializableResponse,
    decode,
    decode_json,
    encode,
    encode_json,
)


class Status(StrEnum):
    OPEN = "open"
    SHIPPED = "shipped"


class Priority(IntEnum):
    LOW = 1
    HIGH = 9


class Colour(Enum):
    RED = "red"


class Order(BaseModel):
    order_id: str
    created: datetime
    total: Decimal


@dataclasses.dataclass
class Line:
    sku: str
    quantity: int
    shipped_at: datetime | None = None


def roundtrip(value: Any) -> Any:
    """Through the full storage path - encode, serialize, read back, decode."""
    return decode_json(encode_json(value))


# --------------------------------------------------------------------------- #
# Fidelity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "value",
    [
        None,
        True,
        False,
        0,
        -17,
        3.5,
        "",
        "plain",
        "unicode: café 日本語",
        {},
        [],
        {"a": 1, "b": [1, {"c": None}]},
        [1, "two", 3.0, None, True],
    ],
)
def test_json_native_values_survive_unchanged(value: Any) -> None:
    assert roundtrip(value) == value


@pytest.mark.parametrize(
    "value",
    [
        datetime(2026, 10, 28, 10, 0, tzinfo=UTC),
        datetime(2026, 10, 28, 10, 0),
        date(2026, 10, 28),
        time(10, 30, 15),
        timedelta(days=2, seconds=90),
        Decimal("10.05"),
        Decimal("-0.000001"),
        UUID("3f2504e0-4f89-11d3-9a0c-0305e82c3301"),
        b"\x00\xff binary",
        bytearray(b"mutable"),
        Status.SHIPPED,
        Priority.HIGH,
        Colour.RED,
        ("a", "b"),
        (),
        {1, 2, 3},
        frozenset({"x"}),
    ],
)
def test_rich_types_keep_their_type(value: Any) -> None:
    restored = roundtrip(value)
    assert restored == value
    assert type(restored) is type(value)


def test_datetime_is_not_recorded_as_a_date() -> None:
    # datetime subclasses date, so a branch in the wrong order silently drops
    # the time component.
    moment = datetime(2026, 10, 28, 10, 0, tzinfo=UTC)
    restored = roundtrip(moment)
    assert restored == moment
    assert isinstance(restored, datetime)


def test_str_enum_is_not_flattened_to_a_string() -> None:
    # A StrEnum member IS a str. Any encoder that checks `isinstance(x, str)`
    # before Enum records "shipped" and replays a bare str, so the agent's
    # `status is Status.SHIPPED` stops being true on replay only.
    restored = roundtrip(Status.SHIPPED)
    assert restored is Status.SHIPPED
    assert type(restored) is Status


def test_int_enum_is_not_flattened_to_an_int() -> None:
    assert roundtrip(Priority.HIGH) is Priority.HIGH


def test_tuple_does_not_replay_as_a_list() -> None:
    restored = roundtrip({"tags": ("a", "b")})
    assert restored["tags"] == ("a", "b")
    assert isinstance(restored["tags"], tuple)


def test_non_string_dict_keys_are_preserved() -> None:
    # JSON stringifies keys: {1: "a"} would come back as {"1": "a"}, and the
    # agent's `data[order_id]` lookup would start raising KeyError on replay.
    restored = roundtrip({1: "one", 2: "two"})
    assert restored == {1: "one", 2: "two"}
    assert list(restored) == [1, 2]


def test_a_dict_containing_the_reserved_tag_round_trips() -> None:
    # A tool may legitimately return a payload with our own tag key in it.
    payload = {"__t": "not ours", "v": 1}
    assert roundtrip(payload) == payload


def test_nested_reserved_tag_round_trips() -> None:
    payload = {"rows": [{"__t": "datetime", "v": "spoofed"}]}
    assert roundtrip(payload) == payload


def test_dict_order_is_preserved() -> None:
    payload = {"z": 1, "a": 2, "m": 3}
    assert list(roundtrip(payload)) == ["z", "a", "m"]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_non_finite_floats_survive_as_valid_json(value: float) -> None:
    # json.dumps would emit bare NaN/Infinity, which is not JSON and which a
    # strict reader on the far side of the store rejects.
    text = encode_json(value)
    assert "NaN" not in text and "Infinity" not in text
    restored = decode_json(text)
    if math.isnan(value):
        assert math.isnan(restored)
    else:
        assert restored == value


def test_pydantic_model_round_trips_with_its_field_types() -> None:
    order = Order(
        order_id="ORD-9921",
        created=datetime(2026, 10, 28, 10, 0, tzinfo=UTC),
        total=Decimal("10.05"),
    )
    restored = roundtrip(order)
    assert isinstance(restored, Order)
    assert restored == order
    # The field types, not just the rendered values: pydantic's own JSON mode
    # would hand back strings here.
    assert isinstance(restored.created, datetime)
    assert isinstance(restored.total, Decimal)


def test_dataclass_round_trips() -> None:
    line = Line(sku="A-1", quantity=2, shipped_at=datetime(2026, 1, 1, tzinfo=UTC))
    restored = roundtrip(line)
    assert isinstance(restored, Line)
    assert restored == line


def test_deeply_nested_mixture() -> None:
    payload = {
        "orders": [
            Order(
                order_id="ORD-1",
                created=datetime(2026, 1, 1, tzinfo=UTC),
                total=Decimal("1.00"),
            )
        ],
        "lines": (Line(sku="A-1", quantity=1),),
        "seen": {Status.OPEN},
        "paths": [Path("a") / "b"],
        "meta": {"ratio": Decimal("0.5"), "checked": date(2026, 1, 2)},
    }
    restored = roundtrip(payload)
    assert isinstance(restored["orders"][0], Order)
    assert isinstance(restored["lines"], tuple)
    assert isinstance(restored["lines"][0], Line)
    assert restored["seen"] == {Status.OPEN}
    assert isinstance(restored["paths"][0], Path)
    assert restored["meta"]["ratio"] == Decimal("0.5")


def test_envelope_always_has_one_shape() -> None:
    # The store writes this into one column, so every payload is a tagged dict.
    for value in (1, "s", None, [1], {"a": 1}, datetime(2026, 1, 1)):
        assert "__t" in encode(value)


# --------------------------------------------------------------------------- #
# Refusal
# --------------------------------------------------------------------------- #


class Connection:
    """Stands in for a live handle: a DB connection, socket or file object."""


class Holder(BaseModel):
    """A model with a field pydantic itself will not serialize."""

    model_config = {"arbitrary_types_allowed": True}
    handle: Any


def test_unserializable_return_is_refused_loudly() -> None:
    with pytest.raises(UnserializableResponse) as caught:
        encode({"rows": [{"conn": Connection()}]}, what="tool 'query_warehouse' return value")

    message = str(caught.value)
    assert "query_warehouse" in message
    assert "Connection" in message
    assert "$.rows[0].conn" in message
    # The refusal has to explain itself, or the user's next move is to wrap the
    # value in str() and reintroduce exactly the bug we refused.
    assert "str()" in (caught.value.hint or "")


def test_the_refusal_reaches_inside_a_pydantic_field() -> None:
    # pydantic's own JSON mode would either warn and coerce this or drop it.
    # Recursing through model_dump(mode="python") means an unserializable field
    # is refused like any other.
    with pytest.raises(UnserializableResponse) as caught:
        encode(Holder(handle=Connection()))
    assert "Connection" in str(caught.value)


def test_a_class_defined_in_a_function_is_refused_at_record_time() -> None:
    # Replay runs in a fresh process - usually in CI - so a class with an
    # unimportable name would decode-fail there. Better to refuse now, where the
    # user is looking at their tool.
    @dataclasses.dataclass
    class Local:
        x: int

    with pytest.raises(UnserializableResponse) as caught:
        encode(Local(x=1))
    assert "imported back on replay" in str(caught.value)


def test_circular_references_are_refused_rather_than_hanging() -> None:
    loop: dict[str, Any] = {"name": "a"}
    loop["self"] = loop
    with pytest.raises(UnserializableResponse) as caught:
        encode(loop)
    assert "circular" in str(caught.value)


def test_refusal_is_not_a_traceback_the_user_has_to_read() -> None:
    from slmeval import exit_codes
    from slmeval.errors import SlmevalError

    error = UnserializableResponse(what="tool 't' return value", path="$", python_type="X")
    assert isinstance(error, SlmevalError)
    assert error.exit_code == exit_codes.USAGE


# --------------------------------------------------------------------------- #
# Decode failures
# --------------------------------------------------------------------------- #


def test_decoding_a_class_that_no_longer_exists_says_so() -> None:
    stale = {"__t": "pydantic", "cls": "tests.test_envelope.Deleted", "v": {}}
    with pytest.raises(EnvelopeDecodeError) as caught:
        decode(stale)
    assert "Deleted" in str(caught.value)
    assert "cassette verify" in (caught.value.hint or "")


def test_decoding_a_module_that_no_longer_exists_says_so() -> None:
    stale = {"__t": "dataclass", "cls": "gone.module.Line", "v": {}}
    with pytest.raises(EnvelopeDecodeError):
        decode(stale)


def test_a_dataclass_field_that_was_removed_is_reported() -> None:
    stale = {
        "__t": "dataclass",
        "cls": "tests.test_envelope.Line",
        "v": {"sku": "A", "quantity": 1, "discontinued_field": True},
    }
    with pytest.raises(EnvelopeDecodeError) as caught:
        decode(stale)
    assert "discontinued_field" in str(caught.value)


def test_a_newer_tag_asks_the_user_to_upgrade() -> None:
    with pytest.raises(EnvelopeDecodeError) as caught:
        decode({"__t": "quantum", "v": 1})
    assert "Upgrade slmeval" in (caught.value.hint or "")


def test_a_payload_that_is_not_an_envelope_is_rejected() -> None:
    with pytest.raises(EnvelopeDecodeError):
        decode({"order_id": "ORD-1"})
    with pytest.raises(EnvelopeDecodeError):
        decode_json("not json at all")
