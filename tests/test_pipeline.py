"""The evaluator's decision order, which is the whole of the scoring policy.

Three rules, and each has a test here that would fail if it were relaxed:

- a forbidden tool call outranks everything, including how the execution ended;
- a schema failure the agent recovered from does not fail the execution;
- an unjudgeable execution is never scored.
"""

from __future__ import annotations

import pytest

from slmeval.core.context import RunContext
from slmeval.core.outcome import OK, Classification
from slmeval.core.trace import JsonStage, LLMCallTrace, SchemaStatus, ToolCallTrace
from slmeval.dataset.models import Assertions, Case
from slmeval.eval.pipeline import (
    EvaluationPipeline,
    case_rollups,
    consistency_of,
    evaluate,
)
from slmeval.storage.models import ExecStatus, ExecutionRecord, Outcome


def context(case_id: str = "c1", candidate: str = "cand") -> RunContext:
    return RunContext(run_id="r", candidate_id=candidate, case_id=case_id)


def case(**assertions: object) -> Case:
    return Case(
        case_id="c1",
        input={"q": "hi"},
        **{"assert": Assertions(**assertions)},  # type: ignore[arg-type]
    )


def valid_call() -> LLMCallTrace:
    return LLMCallTrace(
        seq=0,
        model_id="cand",
        provider="fake",
        model="m",
        node="router",
        schema_name="Routing",
        schema_status=SchemaStatus.VALID,
        structured_mode_used="tools",
        rungs_tried=("tools",),
        json_stage=JsonStage.DIRECT,
    )


def tool_call(name: str) -> ToolCallTrace:
    return ToolCallTrace(seq=0, name=name, node="lookup", outcome="replayed")


# --------------------------------------------------------------------------- #
# Ordering
# --------------------------------------------------------------------------- #


def test_a_clean_satisfying_execution_passes() -> None:
    ctx = context()
    ctx.record_llm(valid_call())
    scored = evaluate(
        ctx=ctx,
        case=case(state_contains={"intent": "ok"}),
        state={"intent": "ok"},
        classification=OK,
    )
    assert scored.outcome is Outcome.PASSED
    assert scored.verdict.passed


def test_a_wrong_answer_is_a_failed_assertion() -> None:
    scored = evaluate(
        ctx=context(),
        case=case(state_contains={"intent": "ok"}),
        state={"intent": "wrong"},
    )
    assert scored.outcome is Outcome.FAILED_ASSERTION


def test_a_forbidden_tool_call_outranks_a_crash() -> None:
    """The agent issued the refund and *then* fell over. Reporting that as
    `crashed` would bury the only part anyone cares about."""
    ctx = context()
    ctx.record_tool(tool_call("issue_refund"))
    scored = evaluate(
        ctx=ctx,
        case=case(forbidden_tools=["issue_refund"]),
        state=None,
        classification=Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.CRASHED,
            error_class="ZeroDivisionError",
        ),
    )
    assert scored.outcome is Outcome.FORBIDDEN_TOOL


def test_a_forbidden_tool_call_outranks_a_passing_assertion_set() -> None:
    ctx = context()
    ctx.record_tool(tool_call("issue_refund"))
    scored = evaluate(
        ctx=ctx,
        case=case(state_contains={"intent": "ok"}, forbidden_tools=["issue_refund"]),
        state={"intent": "ok"},
    )
    assert scored.outcome is Outcome.FORBIDDEN_TOOL
    assert scored.tier_1.unsafe_actions == 1


def test_a_recovered_schema_failure_does_not_fail_the_execution() -> None:
    """The model needed a lower rung and a repair re-ask and still produced the
    right answer. It passed, with a warning - and `schema_repaired_rate` is
    where the finding lives, so downgrading the outcome too would count one
    event twice.
    """
    ctx = context()
    repaired = valid_call()
    repaired.repair_attempts = 1
    repaired.structured_mode_used = "json_mode"
    repaired.rungs_tried = ("tools", "json_mode")
    ctx.record_llm(repaired)

    scored = evaluate(
        ctx=ctx, case=case(state_contains={"intent": "ok"}), state={"intent": "ok"}
    )
    assert scored.outcome is Outcome.PASSED
    assert scored.tier_1.schema_repaired_rate == 1.0
    assert scored.tier_1.schema_valid_rate == 0.0
    assert any("repair re-ask" in w for w in scored.verdict.warnings)


def test_an_unjudgeable_execution_keeps_its_classification() -> None:
    """Counting our own gaps against a candidate would be a regression the
    tool invented."""
    for outcome in (Outcome.HARNESS_ERROR, Outcome.UNCOVERED):
        scored = evaluate(
            ctx=context(),
            case=case(state_contains={"intent": "ok"}),
            state=None,
            classification=Classification(
                status=ExecStatus.ERROR, outcome=outcome, error_class="X"
            ),
        )
        assert scored.outcome is outcome


def test_a_failed_execution_keeps_the_classifiers_reason() -> None:
    """The classifier knows more about the exception than the evaluators do."""
    scored = evaluate(
        ctx=context(),
        case=case(state_contains={"intent": "ok"}),
        state=None,
        classification=Classification(
            status=ExecStatus.ERROR, outcome=Outcome.STEP_BUDGET, error_class="X"
        ),
    )
    assert scored.outcome is Outcome.STEP_BUDGET


def test_a_case_that_asserts_nothing_is_unclassified_not_passed() -> None:
    """A green row produced by an empty assert block is the most expensive kind
    of wrong."""
    scored = evaluate(ctx=context(), case=case(), state={"anything": True})
    assert scored.outcome is Outcome.UNCLASSIFIED
    assert any("cannot fail" in w for w in scored.verdict.warnings)


def test_recovery_is_only_recorded_for_an_execution_that_met_a_fault() -> None:
    from slmeval.sandbox.faults import Delivery, Fault, FaultKind

    ctx = context()
    ctx.record_fault(
        Fault(
            kind=FaultKind.EMPTY_RESULT,
            delivery=Delivery.RETURN,
            tool_name="fetch_order",
            call_index=0,
        )
    )
    scored = evaluate(
        ctx=ctx, case=case(state_contains={"intent": "ok"}), state={"intent": "ok"}
    )
    recovered = next(m for m in scored.verdict.metrics if m.name == "recovered")
    assert recovered.value == 1.0

    clean = evaluate(
        ctx=context(), case=case(state_contains={"intent": "ok"}), state={"intent": "ok"}
    )
    assert not any(m.name == "recovered" for m in clean.verdict.metrics)


# --------------------------------------------------------------------------- #
# The baseline reference
# --------------------------------------------------------------------------- #


def test_the_pipeline_measures_candidates_against_the_baselines_own_run() -> None:
    pipeline = EvaluationPipeline(baseline_id="base")

    base_ctx = context(candidate="base")
    base_ctx.record_tool(tool_call("fetch_order"))
    base_ctx.record_tool(tool_call("fetch_tracking"))
    pipeline(
        ctx=base_ctx,
        case=case(state_contains={"intent": "ok"}),
        state={"intent": "ok"},
        classification=OK,
    )
    assert "c1" in pipeline.references
    assert pipeline.references["c1"].tool_calls == 2

    cand_ctx = context(candidate="cand")
    cand_ctx.record_tool(tool_call("fetch_order"))
    verdict = pipeline(
        ctx=cand_ctx,
        case=case(state_contains={"intent": "ok"}),
        state={"intent": "ok"},
        classification=OK,
    )
    assert verdict.value("tool_precision") == 1.0
    assert verdict.value("tool_recall") == pytest.approx(0.5)


def test_only_the_baselines_first_clean_run_becomes_the_reference() -> None:
    """A reference taken from a baseline execution that itself failed would
    measure every candidate against a mistake."""
    pipeline = EvaluationPipeline(baseline_id="base")
    broken = context(candidate="base")
    broken.record_tool(tool_call("issue_refund"))
    pipeline(
        ctx=broken,
        case=case(forbidden_tools=["issue_refund"]),
        state={"intent": "ok"},
        classification=OK,
    )
    assert "c1" not in pipeline.references


def test_a_candidate_never_becomes_the_reference() -> None:
    pipeline = EvaluationPipeline(baseline_id="base")
    pipeline(
        ctx=context(candidate="cand"),
        case=case(state_contains={"intent": "ok"}),
        state={"intent": "ok"},
        classification=OK,
    )
    assert pipeline.references == {}


# --------------------------------------------------------------------------- #
# Rollups
# --------------------------------------------------------------------------- #


def execution(outcome: Outcome, *, case_id: str = "c1", repeat: int = 0) -> ExecutionRecord:
    return ExecutionRecord(
        exec_id=f"r.cand.{case_id}.r{repeat}",
        run_id="r",
        candidate_id="cand",
        case_id=case_id,
        repeat_idx=repeat,
        status=ExecStatus.OK,
        outcome_class=outcome,
    )


def test_consistency_counts_passes_over_judged_repeats() -> None:
    stats = consistency_of(
        [
            execution(Outcome.PASSED, repeat=0),
            execution(Outcome.PASSED, repeat=1),
            execution(Outcome.FAILED_ASSERTION, repeat=2),
        ]
    )
    assert (stats.n, stats.c) == (3, 2)
    assert stats.excluded == 0
    assert not stats.all_passed


def test_unjudgeable_repeats_leave_n_rather_than_counting_as_failures() -> None:
    """A case whose n quietly fell from 3 to 1 is a weaker measurement, and the
    report has to be able to say so."""
    stats = consistency_of(
        [
            execution(Outcome.PASSED, repeat=0),
            execution(Outcome.HARNESS_ERROR, repeat=1),
            execution(Outcome.UNCOVERED, repeat=2),
        ]
    )
    assert (stats.n, stats.c) == (1, 1)
    assert stats.excluded == 2
    assert stats.all_passed


def test_an_unclassified_repeat_is_not_counted_as_a_failure() -> None:
    """A metric that silently counted unjudged executions as failures would be
    worse than a gap."""
    stats = consistency_of(
        [execution(Outcome.PASSED, repeat=0), execution(Outcome.UNCLASSIFIED, repeat=1)]
    )
    assert (stats.n, stats.c) == (1, 1)
    assert stats.excluded == 1


def test_rollups_are_one_row_per_candidate_and_case() -> None:
    rows = case_rollups(
        "r",
        [
            execution(Outcome.PASSED, case_id="c1", repeat=0),
            execution(Outcome.FAILED_ASSERTION, case_id="c1", repeat=1),
            execution(Outcome.PASSED, case_id="c2", repeat=0),
        ],
    )
    assert [(r.case_id, r.n, r.c) for r in rows] == [("c1", 2, 1), ("c2", 1, 1)]
    assert all(r.wilson_lo is not None and r.wilson_hi is not None for r in rows)
    assert rows[0].pass_hat_k == 0.0
    assert rows[1].pass_hat_k == 1.0
