"""Dataset assertions: the boolean every statistic downstream is built on.

If "did it pass?" is fuzzy, ``pass^k``, the Wilson interval and the gate's
verdict are all fuzzy, so these tests are about the edges rather than the happy
path - a missing key versus a null value, a boolean versus membership, and a
crashed execution versus one with nothing to assert.
"""

from __future__ import annotations

import pytest

from slmeval.dataset.models import Assertions
from slmeval.eval.assertions import (
    MISSING,
    check_assertions,
    contains,
    resolve_path,
)

STATE = {
    "intent": "order_status",
    "order_id": "ORD-9921",
    "refund_issued": False,
    "refund_eligible": None,
    "final_answer": "Your order is with DHL and arrives on 2026-02-03.",
    "order": {"status": "shipped", "lines": [{"sku": "A-1"}, {"sku": "B-2"}]},
    "tags": ["urgent", "vip"],
}


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #


def test_dotted_paths_walk_mappings_and_lists() -> None:
    assert resolve_path(STATE, "intent") == "order_status"
    assert resolve_path(STATE, "order.status") == "shipped"
    assert resolve_path(STATE, "order.lines.1.sku") == "B-2"
    assert resolve_path(STATE, "order.lines.-1.sku") == "B-2"


def test_a_missing_key_is_not_a_none_value() -> None:
    """The distinction is a different bug with a different fix.

    An agent that never set ``refund_eligible`` and one that set it to null
    have failed differently, and a checker that could not tell them apart would
    report the first as the second.
    """
    assert resolve_path(STATE, "refund_eligible") is None
    assert resolve_path(STATE, "nope") is MISSING
    assert resolve_path(STATE, "order.nope") is MISSING
    assert resolve_path(STATE, "order.lines.9.sku") is MISSING
    assert not MISSING


def test_a_literal_dotted_key_wins_over_traversal() -> None:
    """A flat state that really has an "order.status" key still works."""
    assert resolve_path({"order.status": "flat"}, "order.status") == "flat"


def test_resolving_against_nothing_is_missing_not_an_error() -> None:
    assert resolve_path(None, "intent") is MISSING
    assert resolve_path("a string", "intent") is MISSING


# --------------------------------------------------------------------------- #
# contains
# --------------------------------------------------------------------------- #


def test_contains_is_substring_for_strings() -> None:
    assert contains("Your order is with DHL", "DHL")
    assert not contains("Your order is with DHL", "UPS")


def test_contains_is_membership_for_sequences() -> None:
    assert contains(["urgent", "vip"], "vip")
    assert not contains(["urgent"], "vip")


def test_contains_is_a_subset_check_for_mappings() -> None:
    assert contains({"status": "shipped", "carrier": "DHL"}, {"status": "shipped"})
    assert not contains({"status": "shipped"}, {"status": "lost"})


def test_contains_on_a_boolean_means_the_value_not_membership() -> None:
    """`True in [1, 2]` is True in Python, which would make
    `state_contains = { refund_issued = false }` mean something nobody wrote."""
    assert contains(False, False)
    assert not contains(True, False)
    assert not contains([1, 2], True)


def test_contains_falls_back_to_equality() -> None:
    assert contains(3, 3)
    assert not contains(3, 4)
    assert not contains(MISSING, "anything")


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #


def test_a_fully_satisfied_case_passes() -> None:
    result = check_assertions(
        Assertions(
            state_contains={"intent": "order_status", "order_id": "ORD-9921"},
            final_answer_regex="(?i)dhl",
            required_tools=["fetch_order"],
            forbidden_tools=["issue_refund"],
            max_tool_calls=5,
        ),
        state=STATE,
        tools_called=["fetch_order", "fetch_tracking"],
    )
    assert result.ok
    assert not result.unsafe
    # Two state_contains entries plus regex, required, forbidden and the ceiling.
    assert result.checks == 6


def test_every_failure_is_collected_not_short_circuited() -> None:
    """A user fixing a case wants the whole list, not the first problem."""
    result = check_assertions(
        Assertions(
            state={"intent": "refund"},
            state_contains={"order_id": "ORD-0000"},
            final_answer_regex="never going to match",
            required_tools=["issue_refund"],
            max_tool_calls=1,
        ),
        state=STATE,
        tools_called=["fetch_order", "fetch_tracking"],
    )
    assert len(result.failures) == 5
    checks = {f.check for f in result.failures}
    assert checks == {
        "state",
        "state_contains",
        "final_answer_regex",
        "required_tools",
        "max_tool_calls",
    }


def test_a_forbidden_tool_is_a_hard_fail_and_its_own_counter() -> None:
    """Not a two-point accuracy regression - a production incident.

    It fails the case *and* raises an unsafe-action count the scorecard prints
    above the table, where it cannot be averaged away.
    """
    result = check_assertions(
        Assertions(forbidden_tools=["issue_refund"]),
        state=STATE,
        tools_called=["fetch_order", "issue_refund", "issue_refund"],
    )
    assert not result.ok
    assert result.unsafe
    # Deduplicated: one forbidden tool named twice is one tool, not two.
    assert result.forbidden_called == ("issue_refund",)
    assert "FORBIDDEN" in result.describe()[0]


def test_a_missing_required_tool_names_which_one() -> None:
    result = check_assertions(
        Assertions(required_tools=["fetch_order", "check_refund_policy"]),
        state=STATE,
        tools_called=["fetch_order"],
    )
    assert result.missing_tools == ("check_refund_policy",)
    assert "check_refund_policy" in result.describe()[0]


def test_state_checks_fail_by_absence_when_the_agent_never_returned() -> None:
    """A crashed execution must not look like one with nothing to assert."""
    result = check_assertions(
        Assertions(state_contains={"intent": "order_status"}, final_answer_regex="dhl"),
        state=None,
        tools_called=[],
    )
    assert not result.ok
    assert len(result.failures) == 2
    assert "never set this key" in result.describe()[0]
    assert "no final answer" in result.describe()[1]


def test_an_empty_assertion_block_is_reported_rather_than_passed() -> None:
    """A green row produced by an empty assert block is the most expensive
    kind of wrong."""
    result = check_assertions(Assertions(), state=STATE, tools_called=[])
    assert result.ok
    assert result.empty
    assert result.checks == 0


def test_exact_state_equality_compares_values_not_shapes() -> None:
    assert check_assertions(Assertions(state={"refund_issued": False}), state=STATE).ok
    assert not check_assertions(
        Assertions(state={"refund_issued": True}), state=STATE
    ).ok
    # `0 == False` in Python, so an exact-state assertion written with 0 does
    # match a False. Documented rather than worked around: `state` is equality,
    # and `state_contains` is the check that treats booleans specially.
    assert check_assertions(
        Assertions(state={"refund_issued": 0}), state={"refund_issued": False}
    ).ok


def test_max_tool_calls_counts_calls_not_distinct_tools() -> None:
    result = check_assertions(
        Assertions(max_tool_calls=2),
        state=STATE,
        tools_called=["fetch_order", "fetch_order", "fetch_order"],
    )
    assert not result.ok
    assert "usually a loop" in result.describe()[0]


def test_the_detail_blob_is_json_safe() -> None:
    """It goes into a SQLite column, so a sentinel must not leak into it."""
    result = check_assertions(
        Assertions(state_contains={"nope": "x"}), state=STATE, tools_called=[]
    )
    payload = result.as_dict()
    assert payload["failures"][0]["actual"] is None
    assert payload["failures"][0]["expected"] == "x"


@pytest.mark.parametrize(
    ("pattern", "expected"),
    [("(?i)order number", False), ("(?i)dhl", True), ("2026-02-03", True)],
)
def test_final_answer_regex_searches_rather_than_matches(
    pattern: str, expected: bool
) -> None:
    result = check_assertions(
        Assertions(final_answer_regex=pattern), state=STATE, tools_called=[]
    )
    assert result.ok is expected
