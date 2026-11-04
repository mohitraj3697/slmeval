"""Tier 2 against hand-computed fixtures.

Day 12's acceptance criterion is "metrics match hand-computed fixtures", so the
expected values here are worked out in comments rather than captured from a
previous run. A trajectory distance regression test seeded from the
implementation's own output would lock in whatever it did on the day, including
the bug.
"""

from __future__ import annotations

import pytest

from slmeval.core.trace import ToolCallTrace
from slmeval.eval.tier2 import (
    DELETE,
    INSERT,
    MATCH,
    SUBSTITUTE,
    Reference,
    align,
    closest,
    find_loop,
    overlap,
    redundancy,
    signature_of,
    tier2,
)


def call(
    name: str,
    *,
    tier: str | None = None,
    args: dict | None = None,
    node: str = "",
) -> ToolCallTrace:
    return ToolCallTrace(
        seq=0,
        name=name,
        node=node or None,
        arguments=args or {},
        match_tier=tier,
        outcome="replayed" if tier else "recorded",
    )


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def test_identical_trajectories_have_distance_zero() -> None:
    result = align(["a", "b", "c"], ["a", "b", "c"])
    assert result.distance == 0
    assert result.normalized == 0.0
    assert result.identical
    assert [e.op for e in result.script] == [MATCH, MATCH, MATCH]
    assert result.divergences == ()


def test_a_missing_node_is_one_delete() -> None:
    # a b c  ->  a c : one deletion, normalized 1/3.
    result = align(["a", "b", "c"], ["a", "c"])
    assert result.distance == 1
    assert result.normalized == pytest.approx(1 / 3)
    assert [e.op for e in result.divergences] == [DELETE]
    assert result.divergences[0].reference == "b"
    assert result.divergences[0].candidate is None
    assert "b -> (missing)" in result.render()


def test_an_extra_node_is_one_insert() -> None:
    result = align(["a", "c"], ["a", "b", "c"])
    assert result.distance == 1
    assert result.normalized == pytest.approx(1 / 3)
    assert [e.op for e in result.divergences] == [INSERT]
    assert result.divergences[0].candidate == "b"


def test_a_wrong_node_is_one_substitution() -> None:
    result = align(["a", "b", "c"], ["a", "x", "c"])
    assert result.distance == 1
    assert [e.op for e in result.divergences] == [SUBSTITUTE]
    assert "b -> x" in result.render()


def test_nothing_in_common_normalizes_to_one() -> None:
    assert align(["a", "b"], ["x", "y"]).normalized == 1.0


def test_normalization_divides_by_the_longer_sequence() -> None:
    """Dividing by the reference alone would let a 40-node wander score better
    than a 5-node one."""
    short = align(["a", "b", "c"], ["a"])
    long = align(["a", "b", "c"], ["a", *["z"] * 37])
    assert short.normalized < long.normalized


def test_distance_matches_levenshtein_by_hand() -> None:
    # kitten -> sitting is the textbook distance of 3.
    assert align(list("kitten"), list("sitting")).distance == 3


def test_an_empty_candidate_costs_the_whole_reference() -> None:
    result = align(["a", "b", "c"], [])
    assert result.distance == 3
    assert result.normalized == 1.0


def test_closest_takes_the_minimum_over_accepted_paths() -> None:
    """Agents legitimately have several valid routes, so scoring against one
    golden path manufactures failures."""
    paths = [["a", "b", "c", "d"], ["a", "d"]]
    result = closest(["a", "d"], paths)
    assert result is not None
    assert result.distance == 0
    assert result.reference == ("a", "d")


def test_no_reference_path_means_no_measurement() -> None:
    """Not zero and not one: with nothing to compare against, a number would
    be a fabrication in a column people read as measured."""
    assert closest(["a"], []) is None
    assert closest(["a"], [[]]) is None


# --------------------------------------------------------------------------- #
# Tool overlap
# --------------------------------------------------------------------------- #


def test_the_signature_includes_the_match_tier() -> None:
    """A call whose arguments only matched after normalization is not the call
    the baseline made - it reached the right tool with the wrong arguments."""
    assert signature_of(call("fetch_order")) == ("fetch_order", "exact")
    assert signature_of(call("fetch_order", tier="normalized")) == (
        "fetch_order",
        "normalized",
    )


def test_precision_and_recall_by_hand() -> None:
    # Candidate made 4 calls, reference 3, and 2 are shared.
    #   precision = 2/4, recall = 2/3, f1 = 2*(0.5*0.667)/(0.5+0.667)
    candidate = [("a", "exact"), ("a", "exact"), ("b", "exact"), ("c", "exact")]
    reference = [("a", "exact"), ("b", "exact"), ("d", "exact")]
    result = overlap(candidate, reference)

    assert result.matched == 2
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(2 / 3)
    assert result.f1 == pytest.approx(2 * (0.5 * (2 / 3)) / (0.5 + 2 / 3))
    assert result.missing == ("d",)
    assert result.extra == ("a", "c")


def test_overlap_is_a_multiset_not_a_set() -> None:
    """Three calls where the baseline made one is a finding, and set semantics
    would score it as perfect."""
    result = overlap([("a", "exact")] * 3, [("a", "exact")])
    assert result.matched == 1
    assert result.precision == pytest.approx(1 / 3)
    assert result.recall == 1.0
    assert result.extra == ("a x2",)


def test_a_degraded_match_does_not_count_as_the_reference_call() -> None:
    result = overlap([("a", "normalized")], [("a", "exact")])
    assert result.matched == 0
    assert result.extra == ("a[normalized]",)
    assert result.missing == ("a",)


def test_overlap_with_no_reference_has_no_recall() -> None:
    """Recall over an empty reference is undefined, so it is None rather than
    zero. Precision is still measurable: the candidate made a call that the
    reference did not, and nothing it made was matched."""
    result = overlap([("a", "exact")], [])
    assert result.recall is None
    assert result.f1 is None
    assert result.precision == 0.0


# --------------------------------------------------------------------------- #
# Redundancy and loops
# --------------------------------------------------------------------------- #


def test_redundancy_counts_repeats_of_the_same_call() -> None:
    calls = [
        call("fetch_order", args={"order_id": "A"}),
        call("fetch_order", args={"order_id": "A"}),
        call("fetch_order", args={"order_id": "B"}),
        call("fetch_tracking", args={"order_id": "A"}),
    ]
    # One of four calls repeats an earlier one.
    assert redundancy(calls) == pytest.approx(0.25)


def test_different_arguments_are_not_redundant() -> None:
    calls = [
        call("fetch_order", args={"order_id": "A"}),
        call("fetch_order", args={"order_id": "B"}),
    ]
    assert redundancy(calls) == 0.0


def test_redundancy_of_nothing_is_zero() -> None:
    assert redundancy([]) == 0.0


def test_the_shortest_adjacent_cycle_is_the_loop() -> None:
    """`a b c b c` is a `b c` loop; reporting the whole trajectory as repeated
    would be true and useless."""
    loop = find_loop(["a", "b", "c", "b", "c"])
    assert loop is not None
    assert loop.cycle == ("b", "c")
    assert loop.repeats == 2


def test_a_single_node_repeating_is_a_loop() -> None:
    loop = find_loop(["a", "b", "b", "b"])
    assert loop is not None
    assert loop.cycle == ("b",)
    assert loop.repeats == 3


def test_revisiting_a_node_much_later_is_branching_not_looping() -> None:
    assert find_loop(["a", "b", "c", "d", "a"]) is None
    assert find_loop(["a", "b", "c"]) is None


def test_a_stall_is_only_suspected_when_the_tool_calls_repeated_too() -> None:
    """We hold only the final state, so "no change in the asserted keys" is not
    observable. A cycle whose calls were redundant is the strongest claim the
    data supports, and the flag says so.
    """
    without = find_loop(["b", "b"], redundant=False)
    with_repeats = find_loop(["b", "b"], redundant=True)
    assert without is not None and not without.suspected_stall
    assert with_repeats is not None and with_repeats.suspected_stall


# --------------------------------------------------------------------------- #
# The tier
# --------------------------------------------------------------------------- #


def test_tier2_puts_it_together() -> None:
    reference = Reference.from_trace(
        "c1",
        tool_calls=[call("fetch_order"), call("fetch_tracking")],
        trajectory=["router", "lookup", "reply"],
        steps=5,
    )
    result = tier2(
        tool_calls=[call("fetch_order"), call("fetch_order")],
        trajectory=["router", "lookup", "lookup", "reply"],
        steps=7,
        accepted_trajectories=[["router", "lookup", "reply"]],
        reference=reference,
        faults_met=1,
    )

    assert result.trajectory_distance == pytest.approx(0.25)  # one insert of 4
    assert result.tools.matched == 1
    assert result.tools.precision == pytest.approx(0.5)
    assert result.tools.recall == pytest.approx(0.5)
    assert result.redundancy == pytest.approx(0.5)
    assert result.loop is not None and result.loop.cycle == ("lookup",)
    assert result.step_delta == 2
    assert result.faults_met == 1

    names = {m.name for m in result.metrics()}
    assert {"trajectory_distance", "tool_f1", "step_delta", "loop_detected", "faults_met"} <= names
    assert any("suspected stall" in w for w in result.warnings())


def test_tier2_falls_back_to_the_baselines_own_path() -> None:
    """So a user gets trajectory distance without hand-writing reference paths
    for every case."""
    reference = Reference.from_trace(
        "c1", tool_calls=[], trajectory=["router", "reply"], steps=2
    )
    result = tier2(
        tool_calls=[],
        trajectory=["router", "reply"],
        accepted_trajectories=[],
        reference=reference,
    )
    assert result.trajectory_distance == 0.0


def test_tier2_without_any_reference_reports_no_distance() -> None:
    result = tier2(tool_calls=[], trajectory=["router"], accepted_trajectories=[])
    assert result.trajectory_distance is None
    assert result.step_delta is None
    assert not any(m.name == "trajectory_distance" for m in result.metrics())


def test_a_run_with_no_reference_gets_no_tool_comparison() -> None:
    """The baseline's first run sets the reference, so it has nothing to match."""
    result = tier2(
        tool_calls=[],
        trajectory=["a"],
        reference=None,
    )
    names = {metric.name for metric in result.metrics()}
    assert not names & {"tool_precision", "tool_recall", "tool_f1", "tool_overlap"}
