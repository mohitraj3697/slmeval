"""Deterministic fault injection.

The acceptance criterion for Day 7 is "seeds reproduce fault vectors", and the
bug it is guarding against is subtle enough to be worth restating here: key
faults on a single global tool-call counter and one extra call to *any* tool
shifts every later fault onto a different step, so two candidates stop running
the same experiment while the table still prints them side by side. The
isolation tests below are the ones that matter most - a regression there would
not fail loudly, it would quietly invalidate every comparison the tool makes.
"""

from __future__ import annotations

import pytest

from slmeval.sandbox.faults import (
    DEFAULT_DELIVERY,
    MALFORMED_PAYLOAD,
    Delivery,
    Fault,
    FaultKind,
    FaultVector,
    RateLimited,
    ToolTimeout,
    apply_fault,
    compare_vectors,
    draw,
    fault_schedule,
)

TOOLS = ("fetch_order", "search_kb", "issue_refund")


def schedule(seed: int = 42, case_id: str = "refund_104", rate: float = 0.5, **kwargs: object):
    return fault_schedule(seed, case_id, rate=rate, delay_scale=0.0, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #


def test_the_same_seed_gives_a_byte_identical_schedule() -> None:
    left = schedule().digest(TOOLS)
    right = schedule().digest(TOOLS)
    assert left == right


def test_a_different_seed_gives_a_different_schedule() -> None:
    assert schedule(seed=42).digest(TOOLS) != schedule(seed=43).digest(TOOLS)


def test_a_different_case_gives_a_different_schedule() -> None:
    # Faults are per case, so two cases in one run do not meet the same
    # pattern and a single unlucky draw cannot dominate the dataset.
    assert schedule(case_id="a").digest(TOOLS) != schedule(case_id="b").digest(TOOLS)


def test_the_lazy_lookup_and_the_precomputed_table_agree() -> None:
    # The table is built from `fault_for`, and this asserts it stays that way.
    # A second implementation of the draw is exactly how a determinism
    # guarantee quietly stops holding.
    sched = schedule()
    table = sched.table(TOOLS, max_calls=12)
    for name, row in table.items():
        for index, planned in enumerate(row):
            assert planned == sched.fault_for(name, index)


def test_a_decision_does_not_depend_on_evaluation_order() -> None:
    # The reason this hashes instead of using `random`: a PRNG's output
    # depends on how many values were drawn before it, which would make the
    # schedule depend on the order tools happen to be called in.
    sched = schedule()
    forward = [sched.fault_for("fetch_order", i) for i in range(20)]
    backward = [sched.fault_for("fetch_order", i) for i in reversed(range(20))]
    assert forward == list(reversed(backward))


def test_draws_at_one_coordinate_are_independent() -> None:
    # "Is there a fault here" and "which fault" must not read the same bits,
    # or the kind would be a function of how close the hit draw was to the
    # rate - and changing the rate would change the kinds.
    assert draw(42, "c", "t", 0, salt="hit") != draw(42, "c", "t", 0, salt="kind")


def test_draws_are_uniform_enough_to_honour_the_rate() -> None:
    sched = schedule(rate=0.2)
    hits = sum(1 for i in range(4000) if sched.fault_for("fetch_order", i) is not None)
    assert 0.17 < hits / 4000 < 0.23


# --------------------------------------------------------------------------- #
# Per-tool isolation: the alignment bug this module exists to avoid
# --------------------------------------------------------------------------- #


def test_each_tool_is_indexed_by_its_own_calls() -> None:
    # The whole correctness argument for fault parity. An extra call to
    # `search_kb` must not move the fault on `fetch_order`.
    sched = schedule(rate=1.0)
    before = [sched.fault_for("fetch_order", i) for i in range(10)]
    _ = [sched.fault_for("search_kb", i) for i in range(5)]
    after = [sched.fault_for("fetch_order", i) for i in range(10)]
    assert before == after


def test_two_tools_do_not_share_a_pattern() -> None:
    sched = schedule(rate=0.5)
    a = [sched.fault_for("fetch_order", i) is not None for i in range(40)]
    b = [sched.fault_for("search_kb", i) is not None for i in range(40)]
    assert a != b


def test_a_tool_called_past_the_precomputed_depth_still_gets_faults() -> None:
    # The table's depth is a display bound, not a limit on injection: a tool
    # in a loop must not silently stop being faulted.
    sched = schedule(rate=1.0)
    assert sched.fault_for("fetch_order", 10_000) is not None


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


def test_a_zero_rate_injects_nothing() -> None:
    sched = schedule(rate=0.0)
    assert all(sched.fault_for("fetch_order", i) is None for i in range(50))


def test_only_the_enabled_kinds_are_drawn() -> None:
    sched = schedule(rate=1.0, kinds=[FaultKind.EMPTY_RESULT, FaultKind.RATE_LIMIT_429])
    kinds = {sched.fault_for("fetch_order", i).kind for i in range(60)}  # type: ignore[union-attr]
    assert kinds == {FaultKind.EMPTY_RESULT, FaultKind.RATE_LIMIT_429}


def test_delivery_is_configurable() -> None:
    # Which one you choose changes what you are testing: a 429 raised tests
    # the retry wrapper, the same 429 returned tests whether the model reads
    # its tool output at all.
    sched = schedule(
        rate=1.0,
        kinds=[FaultKind.RATE_LIMIT_429],
        delivery={FaultKind.RATE_LIMIT_429: Delivery.RETURN},
    )
    fault = sched.fault_for("fetch_order", 0)
    assert fault is not None
    assert fault.delivery is Delivery.RETURN


def test_a_returned_timeout_is_refused() -> None:
    with pytest.raises(ValueError, match="contradiction"):
        schedule(delivery={FaultKind.TIMEOUT_504: Delivery.RETURN})


def test_a_raised_payload_fault_is_refused() -> None:
    # Raising `empty_result` tests error handling, not whether the model
    # invents data to fill a gap - a different question entirely.
    with pytest.raises(ValueError, match="payload shape"):
        schedule(delivery={FaultKind.EMPTY_RESULT: Delivery.RAISE})


def test_a_rate_above_zero_with_no_kinds_is_refused() -> None:
    # Otherwise it injects nothing and the run reads as "faults change
    # nothing", which is the opposite of the truth.
    with pytest.raises(ValueError, match="silently inject nothing"):
        fault_schedule(42, "c", rate=0.5, kinds=[])


def test_an_impossible_rate_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        fault_schedule(42, "c", rate=1.5)


def test_all_weights_zero_is_refused() -> None:
    with pytest.raises(ValueError, match="weight 0"):
        schedule(rate=1.0, kinds=[FaultKind.EMPTY_RESULT], weights={FaultKind.EMPTY_RESULT: 0.0})


def test_weights_shift_the_distribution() -> None:
    sched = schedule(
        rate=1.0,
        kinds=[FaultKind.EMPTY_RESULT, FaultKind.LATENCY_SPIKE],
        weights={FaultKind.EMPTY_RESULT: 9.0, FaultKind.LATENCY_SPIKE: 1.0},
    )
    drawn = [sched.fault_for("fetch_order", i).kind for i in range(400)]  # type: ignore[union-attr]
    assert drawn.count(FaultKind.EMPTY_RESULT) > drawn.count(FaultKind.LATENCY_SPIKE) * 4


def test_the_kind_order_does_not_depend_on_how_the_user_wrote_it() -> None:
    # The weighted pick walks a sequence, so its order is part of the
    # determinism guarantee and must come from the enum, not from the config.
    forward = schedule(rate=1.0, kinds=[FaultKind.EMPTY_RESULT, FaultKind.TIMEOUT_504])
    reverse = schedule(rate=1.0, kinds=[FaultKind.TIMEOUT_504, FaultKind.EMPTY_RESULT])
    assert forward.digest(TOOLS) == reverse.digest(TOOLS)


def test_delay_scale_zero_removes_the_waits() -> None:
    sched = schedule(rate=1.0, kinds=[FaultKind.LATENCY_SPIKE])
    fault = sched.fault_for("fetch_order", 0)
    assert fault is not None
    assert fault.delay_ms == 0.0


def test_delays_are_real_by_default() -> None:
    # A timeout that takes no time never trips the agent's own timeout, so the
    # default has to be a real wait even though the test suite turns it off.
    sched = fault_schedule(42, "c", rate=1.0, kinds=[FaultKind.TIMEOUT_504])
    fault = sched.fault_for("fetch_order", 0)
    assert fault is not None
    assert fault.delay_s > 0.0


def test_every_kind_has_a_default_delivery() -> None:
    assert set(DEFAULT_DELIVERY) == set(FaultKind)


# --------------------------------------------------------------------------- #
# Delivery
# --------------------------------------------------------------------------- #


def fault(kind: FaultKind, delivery: Delivery = Delivery.RETURN) -> Fault:
    return Fault(kind=kind, delivery=delivery, tool_name="fetch_order", call_index=0)


def test_a_raised_fault_looks_like_a_real_api_error() -> None:
    # The agent under test is supposed to catch this, so it has to look like
    # something a tool would plausibly raise.
    assert isinstance(fault(FaultKind.RATE_LIMIT_429, Delivery.RAISE).error(), RateLimited)
    assert isinstance(fault(FaultKind.TIMEOUT_504, Delivery.RAISE).error(), ToolTimeout)
    assert "429" in str(fault(FaultKind.RATE_LIMIT_429, Delivery.RAISE).error())


def test_empty_result_preserves_the_shape() -> None:
    # The question it asks is "does the model invent data when the shape is
    # right and the data is gone". Handing a dict-expecting agent a list would
    # test type handling instead.
    assert apply_fault(fault(FaultKind.EMPTY_RESULT), {"a": 1}) == ({}, True)
    assert apply_fault(fault(FaultKind.EMPTY_RESULT), [1, 2]) == ([], True)
    assert apply_fault(fault(FaultKind.EMPTY_RESULT), "text") == ("", True)
    assert apply_fault(fault(FaultKind.EMPTY_RESULT), 7) == (None, True)


def test_malformed_json_returns_text_the_agent_has_to_parse() -> None:
    payload, applied = apply_fault(fault(FaultKind.MALFORMED_JSON), {"a": 1})
    assert applied
    assert payload == MALFORMED_PAYLOAD
    assert isinstance(payload, str)


def test_partial_schema_always_drops_something() -> None:
    # A "partial" payload identical to the original would be a silent no-op
    # fault that still counted in the vector.
    payload, applied = apply_fault(
        fault(FaultKind.PARTIAL_SCHEMA), {"a": 1, "b": 2, "c": 3, "d": 4}
    )
    assert applied
    assert len(payload) < 4


def test_partial_schema_is_reproducible() -> None:
    recorded = {"order_id": "x", "status": "shipped", "total": 1.0}
    first = apply_fault(fault(FaultKind.PARTIAL_SCHEMA), recorded)
    second = apply_fault(fault(FaultKind.PARTIAL_SCHEMA), recorded)
    assert first == second


def test_partial_schema_on_a_scalar_is_reported_as_inapplicable() -> None:
    # Reported rather than dropped: a vector that quietly omitted it would
    # compare equal to a run that never scheduled one.
    assert apply_fault(fault(FaultKind.PARTIAL_SCHEMA), 42) == (42, False)


def test_semantic_conflict_prefers_a_named_contradiction() -> None:
    # `"shipped"` becoming `"cancelled"` is the fault that catches a model
    # carrying on as though nothing happened. Negating the line count is a
    # weaker test, and reads more like a corrupt payload.
    payload, applied = apply_fault(
        fault(FaultKind.SEMANTIC_CONFLICT), {"status": "shipped", "lines": 3}
    )
    assert applied
    assert payload == {"status": "cancelled", "lines": 3}


def test_semantic_conflict_falls_back_to_a_number() -> None:
    payload, applied = apply_fault(fault(FaultKind.SEMANTIC_CONFLICT), {"lines": 3})
    assert applied
    assert payload == {"lines": -3}


def test_semantic_conflict_flips_exactly_one_field() -> None:
    # A payload where everything flipped reads as a different record entirely,
    # which an agent may reasonably reject on sight. One flipped field is the
    # hard case: a payload that is almost right.
    recorded = {"status": "shipped", "paid": "paid", "open": "open"}
    payload, _ = apply_fault(fault(FaultKind.SEMANTIC_CONFLICT), recorded)
    assert sum(1 for k in recorded if payload[k] != recorded[k]) == 1


def test_semantic_conflict_reaches_into_a_list() -> None:
    payload, applied = apply_fault(
        fault(FaultKind.SEMANTIC_CONFLICT), [{"id": "a"}, {"status": "open"}]
    )
    assert applied
    assert payload == [{"id": "a"}, {"status": "closed"}]


def test_semantic_conflict_on_untouchable_data_is_inapplicable() -> None:
    assert apply_fault(fault(FaultKind.SEMANTIC_CONFLICT), {"note": "free text"}) == (
        {"note": "free text"},
        False,
    )


def test_latency_spike_returns_the_correct_result() -> None:
    recorded = {"status": "shipped"}
    assert apply_fault(fault(FaultKind.LATENCY_SPIKE), recorded) == (recorded, True)


def test_an_error_as_payload_carries_the_status() -> None:
    payload, applied = apply_fault(fault(FaultKind.RATE_LIMIT_429), {"a": 1})
    assert applied
    assert payload["status"] == 429
    assert payload["error"] == "rate_limit_429"


def test_apply_fault_refuses_a_raise_delivered_fault() -> None:
    with pytest.raises(ValueError, match="return-delivered"):
        apply_fault(fault(FaultKind.RATE_LIMIT_429, Delivery.RAISE), {})


# --------------------------------------------------------------------------- #
# The fault vector and parity
# --------------------------------------------------------------------------- #


def vector(*specs: tuple[str, int, FaultKind]) -> FaultVector:
    v = FaultVector()
    for name, index, kind in specs:
        v.add(
            Fault(
                kind=kind,
                delivery=DEFAULT_DELIVERY[kind],
                tool_name=name,
                call_index=index,
            )
        )
    return v


def test_an_empty_vector_is_falsey() -> None:
    assert not FaultVector()
    assert len(FaultVector()) == 0


def test_identical_vectors_have_parity() -> None:
    left = vector(("fetch_order", 0, FaultKind.EMPTY_RESULT))
    right = vector(("fetch_order", 0, FaultKind.EMPTY_RESULT))
    report = compare_vectors(left, right)
    assert report.matched
    assert left.digest == right.digest
    assert report.reason() == "fault parity holds"


def test_interleaving_between_tools_does_not_break_parity() -> None:
    # Order between independent tools is a trajectory property and belongs in
    # the trajectory metrics, not in the fault comparison.
    left = vector(
        ("fetch_order", 0, FaultKind.EMPTY_RESULT), ("search_kb", 0, FaultKind.TIMEOUT_504)
    )
    right = vector(
        ("search_kb", 0, FaultKind.TIMEOUT_504), ("fetch_order", 0, FaultKind.EMPTY_RESULT)
    )
    assert compare_vectors(left, right).matched


def test_a_shorter_trajectory_breaks_parity_and_says_why() -> None:
    # The honest limitation: a candidate that stopped early never met the
    # fault scheduled for a later call. Reported, not papered over.
    baseline = vector(
        ("fetch_order", 0, FaultKind.EMPTY_RESULT), ("fetch_order", 1, FaultKind.TIMEOUT_504)
    )
    candidate = vector(("fetch_order", 0, FaultKind.EMPTY_RESULT))
    report = compare_vectors(baseline, candidate)
    assert not report.matched
    assert report.missing == ("fetch_order#1 timeout_504:raise",)
    assert "not reached by the candidate" in report.reason()


def test_an_extra_call_is_reported_as_extra() -> None:
    baseline = vector(("fetch_order", 0, FaultKind.EMPTY_RESULT))
    candidate = vector(
        ("fetch_order", 0, FaultKind.EMPTY_RESULT), ("search_kb", 0, FaultKind.EMPTY_RESULT)
    )
    report = compare_vectors(baseline, candidate)
    assert not report.matched
    assert report.extra == ("search_kb#0 empty_result:return",)


def test_an_inapplicable_fault_still_counts_in_the_vector() -> None:
    # Otherwise a run where the fault could not be shaped would compare equal
    # to a run that never scheduled one, and the parity check would pass on a
    # difference that matters.
    applied = FaultVector()
    applied.add(
        Fault(
            kind=FaultKind.PARTIAL_SCHEMA,
            delivery=Delivery.RETURN,
            tool_name="fetch_order",
            call_index=0,
        )
    )
    skipped = FaultVector()
    skipped.add(
        Fault(
            kind=FaultKind.PARTIAL_SCHEMA,
            delivery=Delivery.RETURN,
            tool_name="fetch_order",
            call_index=0,
        ),
        applied=False,
    )
    assert not compare_vectors(applied, skipped).matched
    assert "not applicable" in skipped.labels()[0]


def test_the_plan_digest_and_the_met_digest_are_different_things() -> None:
    # The schedule says what would happen; the vector says what did. Confusing
    # the two is how fault parity would appear to hold when it does not.
    sched = schedule(rate=1.0)
    met = vector(("fetch_order", 0, FaultKind.EMPTY_RESULT))
    assert sched.digest(TOOLS) != met.digest
