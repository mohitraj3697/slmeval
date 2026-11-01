"""The evaluator the runner actually calls, and the one outcome it produces.

Tier 1, then tier 2, then a single verdict. The ordering rules here are the
whole of the scoring policy, and three of them are load-bearing.

**A forbidden tool call outranks everything.** It is decided before the
execution's own ending is even consulted. An agent that issued a refund and
*then* crashed still issued the refund; reporting that as ``crashed`` would
bury the only part anyone cares about.

**A schema failure the agent recovered from does not fail the execution.** If
the model needed a lower rung and a repair re-ask but still produced the right
answer, it passed - with a warning, and with ``schema_repaired_rate`` carrying
the finding. Downgrading the outcome too would double-count one event and make
the pass rate a function of our own leniency.

**An unjudgeable execution is never scored.** A harness error or a path we
never recorded leaves the outcome as it was classified and stays out of every
rate. Counting our own gaps against a candidate would be a regression the tool
invented, which is the one failure mode a measurement instrument cannot
survive.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from slmeval.config.models import EvalConfig
from slmeval.core.context import RunContext
from slmeval.core.outcome import OK, Classification
from slmeval.core.verdict import Metric, Verdict
from slmeval.dataset.models import Case
from slmeval.eval.stats import Consistency, consistency
from slmeval.eval.tier1 import Tier1, tier1
from slmeval.eval.tier2 import Reference, Tier2, tier2
from slmeval.storage.models import (
    NOT_THE_MODELS_FAULT,
    CaseRollupRow,
    ExecutionRecord,
    Outcome,
)


@dataclass(frozen=True, slots=True)
class Scored:
    """Both tiers plus the verdict, for callers that want the detail."""

    verdict: Verdict
    tier_1: Tier1
    tier_2: Tier2

    @property
    def outcome(self) -> Outcome:
        return self.verdict.outcome


def decide(
    classification: Classification, first: Tier1, second: Tier2
) -> Outcome:
    """The one outcome for an execution. See the module docstring for the order."""
    # Before anything else, including how the execution ended.
    if first.assertions.forbidden_called:
        return Outcome.FORBIDDEN_TOOL

    if not classification.judgeable:
        # Our bug or our missing recording. Leave it exactly as classified so
        # nothing downstream can mistake it for a quality signal.
        return classification.outcome

    if not classification.ok:
        # The agent did not return. The classifier already named why, and it
        # knows more about the exception than the evaluators do.
        return classification.outcome

    if first.assertions.empty:
        # Nothing was asserted, so nothing can have failed. Reported as
        # unclassified rather than as a pass: a green row produced by an empty
        # assertion block is the most expensive kind of wrong, and the warning
        # from tier 1 rides along to say so.
        return Outcome.UNCLASSIFIED

    if not first.assertions.ok:
        return Outcome.FAILED_ASSERTION

    return Outcome.PASSED


def evaluate(
    *,
    ctx: RunContext,
    case: Case,
    state: Mapping[str, Any] | None,
    classification: Classification = OK,
    reference: Reference | None = None,
) -> Scored:
    """Score one execution from its live context."""
    trajectory = ctx.node_names
    first = tier1(
        assertions=case.assertions,
        llm_calls=ctx.llm_calls,
        tool_calls=ctx.tool_calls,
        state=state,
    )
    second = tier2(
        tool_calls=ctx.tool_calls,
        trajectory=trajectory,
        steps=ctx.steps,
        accepted_trajectories=case.assertions.reference_trajectory,
        reference=reference,
        faults_met=len(ctx.fault_vector),
        faults_applied=sum(1 for f in ctx.fault_vector if f.applied),
    )
    outcome = decide(classification, first, second)
    metrics: tuple[Metric, ...] = (*first.metrics(), *second.metrics())
    if second.faults_met:
        # Recovery is only meaningful for an execution that actually met a
        # fault, and only once the outcome is known - so it is recorded here
        # rather than inside tier 2, which runs before the verdict exists.
        metrics = (
            *metrics,
            Metric(2, "recovered", 1.0 if outcome is Outcome.PASSED else 0.0),
        )
    return Scored(
        verdict=Verdict(
            outcome=outcome,
            metrics=metrics,
            warnings=(*first.warnings(), *second.warnings()),
        ),
        tier_1=first,
        tier_2=second,
    )


@dataclass
class EvaluationPipeline:
    """Stateful evaluator: remembers the baseline's run of each case.

    The state is the reason this is an object. Tool precision, recall and
    trajectory distance are all measured *against the baseline*, so the
    baseline's own trajectory has to be captured as it happens - which is
    exactly why the runner runs the baseline phase first and whole.

    One instance per run. Reference capture is a plain dict write with no
    ``await`` in between, so concurrent executions on one event loop cannot
    interleave inside it.
    """

    baseline_id: str
    references: dict[str, Reference] = field(default_factory=dict)

    def __call__(
        self,
        *,
        ctx: RunContext,
        case: Case,
        state: Mapping[str, Any] | None,
        classification: Classification = OK,
        reference: Sequence[Any] = (),
    ) -> Verdict:
        scored = evaluate(
            ctx=ctx,
            case=case,
            state=state,
            classification=classification,
            reference=self.references.get(ctx.case_id),
        )
        self._maybe_capture(ctx, classification, scored)
        return scored.verdict

    def _maybe_capture(
        self, ctx: RunContext, classification: Classification, scored: Scored
    ) -> None:
        """Keep the baseline's first clean run of a case as the reference.

        First clean run, not the last and not an average. A reference taken
        from a baseline execution that itself failed would measure candidates
        against a mistake, and comparing every candidate to a *different*
        baseline repeat would make the comparison depend on scheduling order.
        """
        if ctx.candidate_id != self.baseline_id:
            return
        if ctx.case_id in self.references:
            return
        if not classification.ok or scored.outcome is Outcome.FORBIDDEN_TOOL:
            return
        self.references[ctx.case_id] = Reference.from_trace(
            ctx.case_id,
            tool_calls=ctx.tool_calls,
            trajectory=ctx.node_names,
            steps=ctx.steps,
        )


def evaluator_for(config: EvalConfig) -> EvaluationPipeline:
    """The evaluator for one run of one config."""
    return EvaluationPipeline(baseline_id=config.baseline.id)


# --------------------------------------------------------------------------- #
# Rollups
# --------------------------------------------------------------------------- #


def consistency_of(
    executions: Sequence[ExecutionRecord], *, k: int | None = None
) -> Consistency:
    """``c`` passes out of ``n`` judgeable repeats of one case.

    Unjudgeable repeats are removed from ``n`` rather than counted as failures,
    and the number removed is carried on the result. A case whose ``n`` quietly
    fell from 3 to 1 is a weaker measurement than one measured three times, and
    a report that printed both as "100%" would be hiding the difference.
    """
    judged = [e for e in executions if e.outcome_class not in NOT_THE_MODELS_FAULT]
    unjudged = [e for e in executions if e.outcome_class in NOT_THE_MODELS_FAULT]
    scorable = [e for e in judged if e.outcome_class is not Outcome.UNCLASSIFIED]
    passes = sum(1 for e in scorable if e.outcome_class is Outcome.PASSED)
    return consistency(
        len(scorable),
        passes,
        k=k,
        excluded=len(unjudged) + (len(judged) - len(scorable)),
    )


def case_rollups(
    run_id: str,
    executions: Sequence[ExecutionRecord],
    *,
    k: int | None = None,
) -> list[CaseRollupRow]:
    """One stored rollup per ``(candidate, case)``.

    Derived from ``executions`` and stored anyway: ``pass^k`` and the Wilson
    interval are the headline numbers, and a report that recomputed them could
    disagree with the run that printed them.
    """
    grouped: dict[tuple[str, str], list[ExecutionRecord]] = {}
    for execution in executions:
        grouped.setdefault((execution.candidate_id, execution.case_id), []).append(
            execution
        )

    rows: list[CaseRollupRow] = []
    for (candidate_id, case_id), group in sorted(grouped.items()):
        stats = consistency_of(group, k=k)
        rows.append(
            CaseRollupRow(
                run_id=run_id,
                candidate_id=candidate_id,
                case_id=case_id,
                n=stats.n,
                c=stats.c,
                pass_k=stats.pass_k,
                pass_hat_k=stats.pass_hat_k,
                wilson_lo=stats.lower,
                wilson_hi=stats.upper,
            )
        )
    return rows


__all__ = [
    "EvaluationPipeline",
    "Scored",
    "case_rollups",
    "consistency_of",
    "decide",
    "evaluate",
    "evaluator_for",
]
