"""Consistency math: what ``c`` passes out of ``n`` repeats actually tells you.

This is the most important statistical decision in the product, and it is a
small amount of code guarding against one specific, very easy mistake:
**gating on a point estimate from three samples**.

With ``runs_per_case = 3``, three passes out of three is a point estimate of
100%. The 95% Wilson lower bound on that is about **44%** - and the exact
Clopper-Pearson bound is about **29%**. A gate set at "90% or better" that
reads the point estimate will cheerfully approve a model that fails one run in
four. So the gate reads the lower bound by default, and every table prints
``n`` next to the rate. A rate without its ``n`` is a number pretending to be
evidence.

Three estimators, for three different questions:

``pass@k``
    "If we sampled k times, would at least one pass?" The unbiased Codex
    estimator, ``1 - C(n-c, k) / C(n, k)``. The right question for a system
    that can retry.

``pass^k``
    "Would all k pass?" ``C(c, k) / C(n, k)`` exactly, without replacement.
    **This is the production-relevant one**: an agent you cannot retry has to
    pass every time, and ``pass@k`` on a flaky model looks wonderful.

Wilson and Clopper-Pearson
    The interval around ``c/n``. Wilson is the default because it is the method
    the implementation plan names, it behaves sanely at the boundaries, and it
    needs no special functions. Clopper-Pearson is also offered, computed by
    bisection on the binomial tail, for users who want the strictly
    conservative bound - it is what the "about 29%" figure above refers to, and
    the two disagreeing by 15 points at ``n = 3`` is itself a useful lesson
    about three samples.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import comb, isclose, sqrt
from statistics import NormalDist
from typing import Final

DEFAULT_CONFIDENCE: Final = 0.95

Z_95: Final = 1.959963984540054
"""Two-sided 95% normal quantile, spelled out.

Pinned as a literal as well as derived, so a test can assert the constant and a
future change to how it is computed cannot silently move every published
interval.
"""


def z_for(confidence: float = DEFAULT_CONFIDENCE) -> float:
    """Two-sided normal quantile for a confidence level."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence!r}")
    if isclose(confidence, DEFAULT_CONFIDENCE):
        return Z_95
    return NormalDist().inv_cdf(1.0 - (1.0 - confidence) / 2.0)


def _validate(n: int, c: int) -> None:
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if not 0 <= c <= n:
        raise ValueError(f"c must be in [0, n], got c={c}, n={n}")


def pass_at_k(n: int, c: int, k: int) -> float | None:
    """``1 - C(n-c, k) / C(n, k)``: at least one of k samples passes.

    The unbiased estimator, not ``1 - (1 - c/n)^k``. The naive form is biased
    low for small ``n``, which here means systematically understating a
    retry-tolerant system - the opposite of the conservative direction.

    ``None`` when ``k > n``: you cannot estimate the behaviour of five samples
    from three. Returning something anyway would be the tool inventing data.
    """
    _validate(n, c)
    if n == 0 or k <= 0 or k > n:
        return None
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def pass_hat_k(n: int, c: int, k: int, *, plugin: bool = False) -> float | None:
    """``C(c, k) / C(n, k)``: all k samples pass. The production-relevant one.

    Exact and without replacement by default - it is the probability that a
    random subset of k of the runs we actually observed were all passes.

    ``plugin=True`` gives ``(c/n)^k``, which is what to use when extrapolating
    past the number of repeats observed. It is biased high at small ``n``
    (``3/3`` gives exactly 1.0), which is the mistake this module exists to
    stop, so it is opt-in and never the default.
    """
    _validate(n, c)
    if n == 0 or k <= 0:
        return None
    if plugin:
        return (c / n) ** k
    if k > n:
        return None
    if c < k:
        return 0.0
    return comb(c, k) / comb(n, k)


@dataclass(frozen=True, slots=True)
class Interval:
    """A confidence interval, with the method that produced it named."""

    lo: float
    hi: float
    method: str = "wilson"
    confidence: float = DEFAULT_CONFIDENCE

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def render(self, *, percent: bool = True) -> str:
        if percent:
            return f"[{self.lo:.0%}, {self.hi:.0%}]"
        return f"[{self.lo:.3f}, {self.hi:.3f}]"


def wilson_interval(
    n: int, c: int, confidence: float = DEFAULT_CONFIDENCE
) -> Interval | None:
    """The Wilson score interval on ``c/n``.

    ``(p + z^2/2n +- z * sqrt(p(1-p)/n + z^2/4n^2)) / (1 + z^2/n)``

    Written out in that form deliberately: it is the formula in the
    implementation plan, and a reader checking our arithmetic against a
    textbook has to be able to see it line up.

    ``None`` for ``n = 0``. An interval over no observations is not wide, it is
    undefined, and ``[0, 1]`` would read as a measurement.
    """
    _validate(n, c)
    if n == 0:
        return None
    z = z_for(confidence)
    p = c / n
    z2 = z * z
    centre = p + z2 / (2 * n)
    spread = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n))
    denominator = 1 + z2 / n
    return Interval(
        lo=max(0.0, (centre - spread) / denominator),
        hi=min(1.0, (centre + spread) / denominator),
        method="wilson",
        confidence=confidence,
    )


def clopper_pearson(
    n: int, c: int, confidence: float = DEFAULT_CONFIDENCE, *, tolerance: float = 1e-12
) -> Interval | None:
    """The exact (Clopper-Pearson) interval, by bisection on the binomial tail.

    Bisection rather than a Beta quantile so this needs nothing beyond the
    standard library - scipy is not a dependency a CLI that must install in
    seconds can take on. The binomial CDF is monotone in ``p``, so bisection
    converges to machine precision in about forty iterations.

    Strictly more conservative than Wilson, and the source of the "3/3 gives a
    lower bound near 29%" figure: ``0.025 ** (1/3) = 0.292``.
    """
    _validate(n, c)
    if n == 0:
        return None
    alpha = (1.0 - confidence) / 2.0

    # Both roots are of functions increasing in p, so one bisection serves
    # both ends: the lower bound solves P(X >= c) = alpha, and the upper solves
    # P(X <= c) = alpha with the sign flipped.
    lo = 0.0 if c == 0 else _solve(lambda p: _tail_at_least(n, c, p) - alpha, tolerance)
    hi = 1.0 if c == n else _solve(lambda p: alpha - _tail_at_most(n, c, p), tolerance)
    return Interval(lo=lo, hi=hi, method="clopper-pearson", confidence=confidence)


def _tail_at_least(n: int, c: int, p: float) -> float:
    """``P(X >= c)`` for ``X ~ Binomial(n, p)``."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(c, n + 1))


def _tail_at_most(n: int, c: int, p: float) -> float:
    """``P(X <= c)``."""
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(0, c + 1))


def _solve(f: Callable[[float], float], tolerance: float) -> float:
    """Bisect a monotone increasing function to its root in ``[0, 1]``."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if hi - lo < tolerance:
            return mid
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True, slots=True)
class Consistency:
    """``c`` passes out of ``n`` repeats, with every derived figure.

    Stored as a row per ``(candidate, case)`` rather than recomputed at report
    time: a later change to an estimator must not silently restate last
    month's published results, and a report that disagreed with the run that
    printed it would be worse than either number alone.
    """

    n: int
    c: int
    k: int
    pass_k: float | None = None
    pass_hat_k: float | None = None
    interval: Interval | None = None
    excluded: int = 0
    """Repeats dropped as unjudgeable - a harness error, or a path we never
    recorded. Counted so ``n`` can be believed: a case whose ``n`` quietly fell
    from 3 to 1 is a different measurement, and the report says so."""

    @property
    def point(self) -> float | None:
        """``c/n``. Never gate on this. Printed beside the interval so the gap
        between the two is visible at a glance."""
        return self.c / self.n if self.n else None

    @property
    def lower(self) -> float | None:
        """The number to gate on."""
        return self.interval.lo if self.interval else None

    @property
    def upper(self) -> float | None:
        return self.interval.hi if self.interval else None

    @property
    def all_passed(self) -> bool:
        """Every judged repeat passed. The strict, retry-free view of a case."""
        return self.n > 0 and self.c == self.n

    def render(self) -> str:
        if not self.n:
            return "n=0"
        interval = f" {self.interval.render()}" if self.interval else ""
        return f"{self.point:.0%} ({self.c}/{self.n}){interval}"


def consistency(
    n: int,
    c: int,
    *,
    k: int | None = None,
    confidence: float = DEFAULT_CONFIDENCE,
    excluded: int = 0,
    method: str = "wilson",
) -> Consistency:
    """Everything derivable from ``c`` passes out of ``n``.

    ``k`` defaults to ``n``, which makes ``pass^k`` read as "every repeat we
    ran passed" - the number the scorecard's headline column reports.
    """
    _validate(n, c)
    k = n if k is None else k
    interval = (
        clopper_pearson(n, c, confidence)
        if method == "clopper-pearson"
        else wilson_interval(n, c, confidence)
    )
    return Consistency(
        n=n,
        c=c,
        k=k,
        pass_k=pass_at_k(n, c, k),
        pass_hat_k=pass_hat_k(n, c, k),
        interval=interval,
        excluded=excluded,
    )


@dataclass(frozen=True, slots=True)
class Aggregate:
    """A candidate's consistency across every case.

    Two different rates, kept apart because they answer different questions and
    conflating them is how a table ends up with a column whose label does not
    describe its contents:

    :attr:`pass_hat_k` is over **cases** - what fraction of cases passed every
    repeat. It is the headline, and its interval is computed on the same
    numerator and denominator, so the rate and the interval always agree about
    what they are describing.

    :attr:`run_rate` is over **executions**, pooled. Useful, finer-grained, and
    not the same thing: nine of ten cases passing all three runs and all ten
    passing twenty-nine of thirty runs are different products.
    """

    cases: int = 0
    cases_all_pass: int = 0
    executions: int = 0
    passes: int = 0
    excluded: int = 0
    k: int = 1
    interval: Interval | None = None
    run_interval: Interval | None = None
    pass_at_k_mean: float | None = None

    @property
    def pass_hat_k(self) -> float | None:
        """Fraction of cases where every judged repeat passed."""
        return self.cases_all_pass / self.cases if self.cases else None

    @property
    def run_rate(self) -> float | None:
        """Pooled per-execution pass rate."""
        return self.passes / self.executions if self.executions else None

    @property
    def lower(self) -> float | None:
        return self.interval.lo if self.interval else None

    @property
    def upper(self) -> float | None:
        return self.interval.hi if self.interval else None


def aggregate(
    rollups: Sequence[Consistency], *, confidence: float = DEFAULT_CONFIDENCE
) -> Aggregate:
    """Roll per-case consistency up to one candidate's row in the scorecard."""
    judged = [r for r in rollups if r.n > 0]
    cases = len(judged)
    all_pass = sum(1 for r in judged if r.all_passed)
    executions = sum(r.n for r in judged)
    passes = sum(r.c for r in judged)
    at_k = [r.pass_k for r in judged if r.pass_k is not None]
    return Aggregate(
        cases=cases,
        cases_all_pass=all_pass,
        executions=executions,
        passes=passes,
        excluded=sum(r.excluded for r in rollups),
        k=max((r.k for r in judged), default=1),
        interval=wilson_interval(cases, all_pass, confidence),
        run_interval=wilson_interval(executions, passes, confidence),
        pass_at_k_mean=sum(at_k) / len(at_k) if at_k else None,
    )


__all__ = [
    "DEFAULT_CONFIDENCE",
    "Z_95",
    "Aggregate",
    "Consistency",
    "Interval",
    "aggregate",
    "clopper_pearson",
    "consistency",
    "pass_at_k",
    "pass_hat_k",
    "wilson_interval",
    "z_for",
]
