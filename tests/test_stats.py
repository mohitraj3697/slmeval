"""Consistency math, against hand-computed values.

Day 13's acceptance criterion is "a fixture reproduces the values exactly", so
every expected number in this file is either worked out by hand in a comment or
written as the arithmetic itself. A test that asserted
``wilson_interval(30, 29) == wilson_interval(30, 29)`` would pass forever and
protect nothing.

The two headline fixtures are the ones from the implementation plan:
``29/30`` has to give ``[83%, 99%]``, and ``3/3`` has to give a lower bound far
below 100% - because gating on the point estimate from three samples is the
mistake this whole module exists to prevent.
"""

from __future__ import annotations

from math import comb, isclose

import pytest

from slmeval.eval.stats import (
    Z_95,
    aggregate,
    clopper_pearson,
    consistency,
    pass_at_k,
    pass_hat_k,
    wilson_interval,
    z_for,
)

# --------------------------------------------------------------------------- #
# pass@k
# --------------------------------------------------------------------------- #


def test_pass_at_k_matches_the_codex_estimator() -> None:
    # 1 - C(n-c, k)/C(n, k), worked out longhand.
    assert pass_at_k(3, 1, 2) == 1 - comb(2, 2) / comb(3, 2)
    assert pass_at_k(5, 2, 2) == 1 - comb(3, 2) / comb(5, 2)
    assert pass_at_k(10, 3, 5) == 1 - comb(7, 5) / comb(10, 5)


def test_pass_at_k_is_one_when_too_few_failures_remain_to_fill_k() -> None:
    """With 2 failures out of 3, any 2 samples must include a pass."""
    assert pass_at_k(3, 1, 3) == 1.0
    assert pass_at_k(3, 2, 2) == 1.0


def test_pass_at_k_is_zero_when_nothing_passed() -> None:
    assert pass_at_k(3, 0, 1) == 0.0
    assert pass_at_k(3, 0, 3) == 0.0


def test_pass_at_k_is_undefined_past_the_samples_we_have() -> None:
    """k > n cannot be estimated. Returning a number would be inventing data."""
    assert pass_at_k(3, 3, 4) is None
    assert pass_at_k(0, 0, 1) is None


def test_pass_at_k_is_not_the_naive_form() -> None:
    """The biased estimator understates a retry-tolerant system at small n."""
    naive = 1 - (1 - 1 / 3) ** 2
    assert pass_at_k(3, 1, 2) != pytest.approx(naive)
    assert pass_at_k(3, 1, 2) > naive


# --------------------------------------------------------------------------- #
# pass^k
# --------------------------------------------------------------------------- #


def test_pass_hat_k_is_exact_without_replacement() -> None:
    assert pass_hat_k(3, 3, 3) == 1.0
    assert pass_hat_k(3, 2, 3) == 0.0
    assert pass_hat_k(5, 4, 2) == comb(4, 2) / comb(5, 2)
    assert pass_hat_k(10, 9, 3) == comb(9, 3) / comb(10, 3)


def test_pass_hat_k_plugin_differs_and_is_opt_in() -> None:
    """(c/n)^k is biased high at 3/3 - exactly the trap - so it is never default."""
    assert pass_hat_k(3, 2, 3, plugin=True) == pytest.approx((2 / 3) ** 3)
    assert pass_hat_k(3, 2, 3) == 0.0
    assert pass_hat_k(3, 3, 3, plugin=True) == 1.0


def test_pass_hat_k_is_stricter_than_pass_at_k() -> None:
    """"all k pass" can never be likelier than "at least one passes"."""
    for n in range(1, 8):
        for c in range(n + 1):
            for k in range(1, n + 1):
                strict = pass_hat_k(n, c, k)
                lenient = pass_at_k(n, c, k)
                assert strict is not None and lenient is not None
                assert strict <= lenient + 1e-12


# --------------------------------------------------------------------------- #
# Wilson
# --------------------------------------------------------------------------- #


def test_z_95_is_the_two_sided_normal_quantile() -> None:
    assert z_for(0.95) == Z_95
    assert isclose(Z_95, 1.959963984540054, rel_tol=1e-12)


def test_wilson_reproduces_the_plans_headline_fixture() -> None:
    """29/30 -> [83%, 99%]. The interval printed in the plan's own scorecard."""
    interval = wilson_interval(30, 29)
    assert interval is not None
    assert round(interval.lo, 4) == 0.8333
    assert round(interval.hi, 4) == 0.9941
    assert interval.render() == "[83%, 99%]"


def test_wilson_on_three_of_three_is_nowhere_near_certainty() -> None:
    """The whole reason the gate reads the lower bound.

    A point estimate of 100% from three samples has a 95% lower bound of about
    44%. A gate set at 90% that read the point estimate would approve a model
    failing one run in four.
    """
    interval = wilson_interval(3, 3)
    assert interval is not None
    assert interval.hi == 1.0
    assert round(interval.lo, 4) == 0.4385
    assert interval.lo < 0.5


def test_wilson_matches_the_formula_written_out() -> None:
    from math import sqrt

    n, c = 17, 11
    p = c / n
    z2 = Z_95**2
    centre = p + z2 / (2 * n)
    spread = Z_95 * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denominator = 1 + z2 / n

    interval = wilson_interval(n, c)
    assert interval is not None
    assert interval.lo == pytest.approx((centre - spread) / denominator)
    assert interval.hi == pytest.approx((centre + spread) / denominator)


def test_wilson_stays_inside_zero_and_one() -> None:
    for n in range(1, 40):
        for c in range(n + 1):
            interval = wilson_interval(n, c)
            assert interval is not None
            assert 0.0 <= interval.lo <= interval.hi <= 1.0


def test_wilson_narrows_as_n_grows() -> None:
    widths = [wilson_interval(n, n // 2) for n in (4, 16, 64, 256)]
    assert all(w is not None for w in widths)
    spans = [w.width for w in widths if w is not None]
    assert spans == sorted(spans, reverse=True)


def test_no_observations_has_no_interval() -> None:
    """Not [0, 1] - an interval over nothing is undefined, and a printed
    [0%, 100%] would read as a measurement."""
    assert wilson_interval(0, 0) is None
    assert clopper_pearson(0, 0) is None


def test_c_greater_than_n_is_refused() -> None:
    with pytest.raises(ValueError, match="c must be in"):
        wilson_interval(3, 4)


# --------------------------------------------------------------------------- #
# Clopper-Pearson
# --------------------------------------------------------------------------- #


def test_clopper_pearson_on_three_of_three_is_the_plans_29_percent() -> None:
    """0.025 ** (1/3) = 0.2924, the exact two-sided lower bound.

    The plan's prose cites this figure while specifying the Wilson formula; the
    two disagree by 15 points at n=3, which is itself the lesson. Both are
    implemented, Wilson is the default because it is what the formula says, and
    this is the conservative alternative.
    """
    interval = clopper_pearson(3, 3)
    assert interval is not None
    assert interval.lo == pytest.approx(0.025 ** (1 / 3), abs=1e-9)
    assert round(interval.lo, 2) == 0.29


def test_clopper_pearson_is_more_conservative_than_wilson() -> None:
    for n in (3, 5, 10, 30):
        for c in (0, 1, n - 1, n):
            exact = clopper_pearson(n, c)
            score = wilson_interval(n, c)
            assert exact is not None and score is not None
            assert exact.lo <= score.lo + 1e-9
            assert exact.hi >= score.hi - 1e-9


def test_clopper_pearson_endpoints_are_degenerate_where_they_should_be() -> None:
    low = clopper_pearson(5, 0)
    high = clopper_pearson(5, 5)
    assert low is not None and high is not None
    assert low.lo == 0.0
    assert high.hi == 1.0


# --------------------------------------------------------------------------- #
# Consistency and aggregation
# --------------------------------------------------------------------------- #


def test_consistency_defaults_k_to_n() -> None:
    stats = consistency(3, 3)
    assert stats.k == 3
    assert stats.pass_hat_k == 1.0
    assert stats.point == 1.0
    assert stats.all_passed
    assert stats.lower is not None and stats.lower < 0.5
    assert "3/3" in stats.render()
    assert "[44%, 100%]" in stats.render()


def test_consistency_can_use_the_exact_interval() -> None:
    stats = consistency(3, 3, method="clopper-pearson")
    assert stats.interval is not None
    assert stats.interval.method == "clopper-pearson"
    assert stats.lower == pytest.approx(0.025 ** (1 / 3), abs=1e-9)


def test_consistency_carries_what_was_excluded() -> None:
    """n has to be believable: a case measured once is not a case measured
    three times, however good the single result was."""
    stats = consistency(1, 1, excluded=2)
    assert stats.n == 1
    assert stats.excluded == 2
    assert stats.point == 1.0


def test_aggregate_counts_cases_and_executions_separately() -> None:
    # Ten cases, 3 repeats each. Nine pass all three; one passes twice.
    rollups = [consistency(3, 3) for _ in range(9)] + [consistency(3, 2)]
    rolled = aggregate(rollups)

    assert rolled.cases == 10
    assert rolled.cases_all_pass == 9
    assert rolled.pass_hat_k == 0.9
    # Pooled executions: 29 of 30, which is the plan's headline fixture.
    assert rolled.executions == 30
    assert rolled.passes == 29
    assert rolled.run_rate == pytest.approx(29 / 30)
    assert rolled.run_interval is not None
    assert rolled.run_interval.render() == "[83%, 99%]"
    # The headline interval is on *cases*, matching the headline rate - so the
    # label and the interval can never describe different things.
    assert rolled.interval is not None
    assert rolled.interval.render() == wilson_interval(10, 9).render()  # type: ignore[union-attr]


def test_aggregate_ignores_cases_with_nothing_judgeable() -> None:
    rolled = aggregate([consistency(3, 3), consistency(0, 0, excluded=3)])
    assert rolled.cases == 1
    assert rolled.pass_hat_k == 1.0
    assert rolled.excluded == 3


def test_aggregate_of_nothing_is_empty_not_zero() -> None:
    rolled = aggregate([])
    assert rolled.cases == 0
    assert rolled.pass_hat_k is None
    assert rolled.run_rate is None
    assert rolled.interval is None
