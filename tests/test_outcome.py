"""Classification: keeping our own bugs out of the models' scores.

The one that matters most is the last section. ``harness_error`` is counted
separately and never attributed to a model or to the agent, and the way an
unrecognised exception gets there is the *deepest* frame of its traceback. Every
execution passes through slmeval code on the way in, so "slmeval appears in the
traceback" is true of literally every failure - it has to be where the
exception was raised, or a bug in a user's agent would be filed as ours and
dropped from their results.
"""

from __future__ import annotations

import asyncio

import pytest

from slmeval.core.context import StepBudgetExceeded
from slmeval.core.outcome import classify, raised_inside_slmeval
from slmeval.errors import ConfigError
from slmeval.providers.base import ProviderUnreachable, StructuredOutputError
from slmeval.sandbox.diagnosis import MissDiagnosis, Verdict
from slmeval.sandbox.faults import Delivery, Fault, FaultKind
from slmeval.sandbox.keys import CassetteKeys, KeyEntry, MatchTier
from slmeval.sandbox.tool import CassetteMiss, SandboxNotReady
from slmeval.storage.models import ExecStatus, Outcome


def keys() -> CassetteKeys:
    return CassetteKeys(
        case_id="c1",
        tool_name="fetch_order",
        entries=(KeyEntry(tier=MatchTier.EXACT, key="k", canonical_args="{}"),),
    )


def miss(verdict: Verdict) -> CassetteMiss:
    return CassetteMiss(
        case_id="c1",
        tool_name="fetch_order",
        arguments={"order_id": "9921"},
        keys=keys(),
        diagnosis=MissDiagnosis(
            case_id="c1",
            tool_name="fetch_order",
            arguments={"order_id": "9921"},
            verdict=verdict,
            rationale="engineered for this test",
            recorded_calls=1,
        ),
    )


def raised(exc: BaseException) -> BaseException:
    """Give an exception a real traceback, which is what attribution reads."""
    try:
        raise exc
    except BaseException as caught:
        return caught


# --------------------------------------------------------------------------- #
# Nothing went wrong
# --------------------------------------------------------------------------- #


def test_no_exception_leaves_the_verdict_open() -> None:
    """"The agent returned" is not "the agent was right" - that is the
    evaluators' call, and a runner that could decide it would make them
    optional."""
    result = classify(None)
    assert result.status is ExecStatus.OK
    assert result.outcome is Outcome.UNCLASSIFIED
    assert result.ok
    assert result.needs_evaluation


# --------------------------------------------------------------------------- #
# Findings about the model
# --------------------------------------------------------------------------- #


def test_a_hallucinated_parameter_is_the_models_failure() -> None:
    result = classify(raised(miss(Verdict.HALLUCINATED_PARAMETER)))
    assert result.outcome is Outcome.CASSETTE_MISS
    assert result.judgeable
    assert result.detail == "hallucinated_parameter"


def test_an_unrecorded_valid_path_is_our_gap_not_the_models() -> None:
    """Marking a model down for exploring a route the baseline never took
    would be our own coverage hole reported as its failure."""
    result = classify(raised(miss(Verdict.UNRECORDED_VALID_PATH)))
    assert result.outcome is Outcome.UNCOVERED
    assert not result.judgeable


def test_the_miss_message_is_one_line_not_the_whole_diagnosis_block() -> None:
    """The full block belongs in a terminal; a database column and a table cell
    need a summary."""
    result = classify(raised(miss(Verdict.HALLUCINATED_TOOL)))
    assert result.message is not None
    assert "\n" not in result.message


def test_a_runaway_loop_is_a_result_not_a_crash() -> None:
    result = classify(raised(StepBudgetExceeded(26, 25)))
    assert result.outcome is Outcome.STEP_BUDGET
    assert result.detail == "26/25 steps"
    assert result.judgeable


def test_a_schema_failure_keeps_the_rungs_it_tried() -> None:
    exc = StructuredOutputError(
        schema_name="Routing",
        rungs_tried=["tools", "json_mode"],
        last_text="nope",
        validation_error="intent missing",
    )
    result = classify(raised(exc))
    assert result.outcome is Outcome.SCHEMA_FAILURE
    assert result.detail == "tools,json_mode"


def test_a_timeout_is_its_own_status() -> None:
    result = classify(raised(TimeoutError()))
    assert result.status is ExecStatus.TIMEOUT
    assert result.outcome is Outcome.TIMEOUT


def test_an_injected_fault_the_agent_never_recovered_from_is_a_tool_error() -> None:
    """Raised by us, and still a finding: it is the point of injecting it, and
    the numerator of the recovery rate."""
    fault = Fault(
        kind=FaultKind.RATE_LIMIT_429,
        delivery=Delivery.RAISE,
        tool_name="fetch_order",
        call_index=2,
    )
    result = classify(raised(fault.error()))
    assert result.outcome is Outcome.TOOL_ERROR
    assert result.detail == "rate_limit_429 on fetch_order#2"
    assert result.judgeable


def test_an_unreachable_provider_is_not_a_quality_signal() -> None:
    """"Ollama was not running" must never be reported as a regression."""
    result = classify(raised(ProviderUnreachable("connection refused")))
    assert result.outcome is Outcome.PROVIDER_ERROR
    assert result.outcome is not Outcome.CRASHED


# --------------------------------------------------------------------------- #
# The agent's own bugs
# --------------------------------------------------------------------------- #


def test_a_user_fixable_error_mid_run_is_the_agents_crash() -> None:
    """Most often: the entrypoint returned something that is not a final-state
    mapping. Theirs to fix, and not the model's doing either."""
    result = classify(raised(ConfigError("run_case returned str, not a mapping")))
    assert result.outcome is Outcome.CRASHED
    assert not result.harness_error


def test_an_exception_from_the_users_own_code_is_not_ours() -> None:
    def user_agent() -> None:
        raise ZeroDivisionError("division by zero")

    try:
        user_agent()
    except ZeroDivisionError as exc:
        result = classify(exc)

    assert result.outcome is Outcome.CRASHED
    assert not result.harness_error
    assert result.judgeable


# --------------------------------------------------------------------------- #
# Our bugs
# --------------------------------------------------------------------------- #


def test_a_sandbox_that_cannot_do_its_job_is_our_bug() -> None:
    result = classify(raised(SandboxNotReady("no cassette store")))
    assert result.outcome is Outcome.HARNESS_ERROR
    assert result.harness_error
    assert not result.judgeable


def test_a_lost_context_is_our_bug() -> None:
    result = classify(raised(LookupError("replay requires a benchmark run context")))
    assert result.outcome is Outcome.HARNESS_ERROR


def test_an_exception_raised_inside_slmeval_is_attributed_to_us() -> None:
    """The acceptance criterion for crash isolation: something that blows up
    inside the harness is reported as a harness error, not as a model failure.

    The exception here is raised by real slmeval code - a guard in the stats
    module, reached with an impossible (n, c), which is the shape an actual
    harness bug would take.
    """
    from slmeval.eval.stats import wilson_interval

    try:
        wilson_interval(3, 4)
    except ValueError as exc:
        assert raised_inside_slmeval(exc.__traceback__)
        result = classify(exc)

    assert result.outcome is Outcome.HARNESS_ERROR
    assert result.harness_error
    assert not result.judgeable
    assert result.detail == "raised inside slmeval"


def test_attribution_reads_the_deepest_frame_not_any_frame() -> None:
    """Every execution passes through our code on the way in, so "slmeval is in
    the traceback" is true of every failure there has ever been."""

    def user_code() -> None:
        raise ValueError("mine")

    def slmeval_calls_it() -> None:
        # Stands in for the harness calling the user's agent: the outer frame
        # is ours, the raising frame is theirs.
        user_code()

    try:
        slmeval_calls_it()
    except ValueError as exc:
        assert not raised_inside_slmeval(exc.__traceback__)


def test_an_exception_with_no_traceback_is_not_guessed_at() -> None:
    """Constructed rather than raised. Guessing "ours" would let a stray
    `raise ValueError` in a user's agent be dropped from their results."""
    assert not raised_inside_slmeval(None)
    assert classify(ValueError("never raised")).outcome is Outcome.CRASHED


# --------------------------------------------------------------------------- #
# Cancellation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "exc", [asyncio.CancelledError(), KeyboardInterrupt(), SystemExit()]
)
def test_cancellation_is_re_raised_never_classified(exc: BaseException) -> None:
    """A unit with no row is work that --resume picks up. A unit recorded as
    "cancelled" would be skipped on resume and silently lost from the matrix,
    so this path must be impossible to swallow by accident.
    """
    with pytest.raises(type(exc)):
        classify(exc)
