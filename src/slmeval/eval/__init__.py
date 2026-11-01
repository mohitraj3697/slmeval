"""Evaluators: turning a trajectory into numbers somebody can act on.

Tiered, and the tiering is what makes the economics work. Roughly 80% of the
signal is computable for nothing in microseconds, so a judge model is only ever
paid for the remainder - and never for grading a run that already failed
deterministically.

:mod:`slmeval.eval.assertions`
    The dataset's own checks: exact state, ``state_contains``,
    ``final_answer_regex``, required and forbidden tools, call ceilings.
:mod:`slmeval.eval.tier1`
    Deterministic, per execution, $0, under 2ms. Schema adherence on the
    *first* attempt versus after repair - the gap between those two numbers is
    the clearest small-model-versus-frontier signal the tool produces.
:mod:`slmeval.eval.tier2`
    Trajectory and state: tool precision/recall, trajectory distance with its
    edit script, redundancy, loops, recovery from injected faults.
:mod:`slmeval.eval.stats`
    The consistency math across repeats - ``pass@k``, ``pass^k`` and the
    Wilson interval. Gating on a point estimate from three samples is the
    single easiest way to be confidently wrong, so this module exists to make
    the interval impossible to avoid.
:mod:`slmeval.eval.cost`
    Measured cloud spend, and clearly-labelled modelled cost for local models.
:mod:`slmeval.eval.pipeline`
    The :class:`~slmeval.core.verdict.Evaluator` the runner actually calls:
    tier 1, then tier 2, then one outcome.

Tier 3 - the judge - lands in week 3, behind a gate that only lets it run when
tiers 1 and 2 passed and the case carries a rubric.
"""

from __future__ import annotations

from slmeval.eval.assertions import AssertionResult, check_assertions
from slmeval.eval.cost import Priced, PriceTable
from slmeval.eval.pipeline import evaluate, evaluator_for
from slmeval.eval.stats import (
    Consistency,
    consistency,
    pass_at_k,
    pass_hat_k,
    wilson_interval,
)
from slmeval.eval.tier1 import Tier1, tier1
from slmeval.eval.tier2 import Tier2, tier2

__all__ = [
    "AssertionResult",
    "Consistency",
    "PriceTable",
    "Priced",
    "Tier1",
    "Tier2",
    "check_assertions",
    "consistency",
    "evaluate",
    "evaluator_for",
    "pass_at_k",
    "pass_hat_k",
    "tier1",
    "tier2",
    "wilson_interval",
]
