"""Every way an execution can end, driven by a scripted model.

This is Step 15's load-bearing test. The harness classifies nine kinds of
failure and the whole value of the scorecard rests on the differences between
them: a cassette miss caused by a hallucinated order id is a finding about the
model, a miss on a route the baseline never took is a hole in our recordings,
and a `ZeroDivisionError` inside slmeval is our bug. Printed in one column
labelled "failed", those three numbers add up to something that looks like
evidence and is not.

So each outcome gets a candidate whose script provokes it, and the run has to
put every one in its own bucket. Mutate the classifier and this file goes red.

One agent, one dataset, no network, no keys - about a second for the lot.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slmeval.storage.models import NOT_THE_MODELS_FAULT, ExecStatus, Outcome
from tests.offline import (
    ONE_CASE,
    behaves,
    candidate,
    forget_the_agent_module,  # noqa: F401 - autouse fixture
    offline_project,
    replies,
    routes,
)

# Each candidate is one scripted model. The turns are consumed in order and the
# last one repeats, so two turns describe "routes, then composes a reply".
CANDIDATES = "".join(
    [
        # Emits a prose preamble and a fenced block. The extraction cascade digs
        # the object out, so this candidate *passes* - and the scorecard says it
        # only passed with help.
        candidate(
            "fenced",
            behaves("fenced_json", '{"intent": "order_status", "order_id": "ORD-9921"}'),
            behaves("fenced_json", '{"final_answer": "with DHL"}'),
            extra='extra_body.unsupported = ["tools", "json_schema"]',
        ),
        # Right shape, wrong answer: the ordinary failure.
        candidate("wrong-route", routes("refund_request"), replies()),
        # Cannot produce the schema at all, by two different routes.
        candidate("prose-only", behaves("not_json")),
        candidate("off-schema", behaves("invalid_schema")),
        # The model server is down, and a 429 that never clears.
        candidate("dead", behaves("unreachable")),
        candidate("throttled", behaves("rate_limit")),
        # Drops the ORD- prefix - the single most common small-model failure on
        # this task. The ERP was never asked for order 9921, so this is a miss,
        # and the diagnosis says which argument diverged.
        candidate("sloppy-id", routes("order_status", "9921"), replies()),
        # Never stops asking. A runaway loop is a finding, not a hang.
        candidate("spinner", routes("spin")),
        # The agent itself falls over on this answer.
        candidate("exploder", routes("boom")),
    ]
)


@pytest.fixture
def matrix(tmp_path: Path):  # type: ignore[no-untyped-def]
    """One recorded baseline, then every candidate replayed against it."""
    with offline_project(tmp_path, candidates=CANDIDATES, cases=ONE_CASE) as runs:
        runs.record()
        yield runs, runs.run(tool_mode="replay")


def test_a_scripted_model_that_works_passes_every_case(matrix) -> None:  # type: ignore[no-untyped-def]
    """The reference row. Without this the rest of the file could all be
    passing for the wrong reason."""
    project, summary = matrix
    baseline = project.config.baseline.id
    assert project.outcomes(summary.run_id)[baseline] == {Outcome.PASSED.value}


def test_a_dirty_envelope_still_passes(matrix) -> None:  # type: ignore[no-untyped-def]
    project, summary = matrix
    assert project.outcomes(summary.run_id)["fenced"] == {Outcome.PASSED.value}


@pytest.mark.parametrize(
    ("model_id", "outcome"),
    [
        ("wrong-route", Outcome.FAILED_ASSERTION),
        ("prose-only", Outcome.SCHEMA_FAILURE),
        ("off-schema", Outcome.SCHEMA_FAILURE),
        ("dead", Outcome.PROVIDER_ERROR),
        ("throttled", Outcome.PROVIDER_ERROR),
        ("sloppy-id", Outcome.CASSETTE_MISS),
        ("spinner", Outcome.STEP_BUDGET),
        ("exploder", Outcome.CRASHED),
    ],
)
def test_each_failure_lands_in_its_own_bucket(  # type: ignore[no-untyped-def]
    matrix, model_id: str, outcome: Outcome
) -> None:
    project, summary = matrix
    assert project.outcomes(summary.run_id)[model_id] == {outcome.value}


def test_the_schema_failure_says_which_rungs_were_tried(matrix) -> None:  # type: ignore[no-untyped-def]
    """"This model cannot do structured output" is a claim about four
    mechanisms, not about one attempt."""
    project, summary = matrix
    row = project.executions(summary.run_id, "prose-only")[0]
    assert row.status is ExecStatus.ERROR
    assert row.error_class == "StructuredOutputError"
    assert "Routing" in (row.error_message or "")


def test_the_miss_carries_a_verdict_a_user_can_act_on(matrix) -> None:  # type: ignore[no-untyped-def]
    project, summary = matrix
    row = project.executions(summary.run_id, "sloppy-id")[0]
    assert row.error_class == "CassetteMiss"
    assert "fetch_order" in (row.error_message or "")


def test_a_runaway_is_stopped_at_the_step_budget_not_by_a_timeout(matrix) -> None:  # type: ignore[no-untyped-def]
    project, summary = matrix
    row = project.executions(summary.run_id, "spinner")[0]
    assert row.steps == 9  # max_steps is 8, plus the call that tripped it
    assert "max_steps" in (row.error_message or "")


def test_only_our_own_failures_leave_the_denominator() -> None:
    """Pinned as a set, because every rate in the report is computed against
    it. `harness_error` is our bug and `uncovered` is our missing recording;
    a candidate marked down for either would be a regression the tool
    invented. `provider_error` is deliberately *not* here - a dead server is
    not a quality finding either, but it fails the whole run with exit code 3
    instead of being quietly dropped from one row (see test_run_cmd)."""
    assert frozenset({Outcome.HARNESS_ERROR, Outcome.UNCOVERED}) == NOT_THE_MODELS_FAULT
    assert Outcome.SCHEMA_FAILURE not in NOT_THE_MODELS_FAULT


def test_the_agents_own_crash_is_not_filed_as_ours(matrix) -> None:  # type: ignore[no-untyped-def]
    """`RuntimeError` raised in the user's agent is theirs; the classifier
    attributes by the deepest frame, so a change to that logic shows up here."""
    project, summary = matrix
    row = project.executions(summary.run_id, "exploder")[0]
    assert row.error_class == "RuntimeError"
    assert row.outcome_class is Outcome.CRASHED


def test_one_candidates_failure_does_not_touch_the_others(matrix) -> None:  # type: ignore[no-untyped-def]
    """Crash isolation, stated as a property: nine candidates, nine verdicts,
    and the run completes."""
    project, summary = matrix
    outcomes = project.outcomes(summary.run_id)

    assert len(outcomes) == 10
    assert summary.executed == summary.planned
    assert all(len(seen) == 1 for seen in outcomes.values())


# --------------------------------------------------------------------------- #
# The two that need a different mode
# --------------------------------------------------------------------------- #


def test_a_forbidden_tool_call_is_its_own_outcome(tmp_path: Path) -> None:
    """Recorded, not replayed: in replay the call would miss the cassette
    first, and `forbidden_tool` is the finding that must not be averaged away."""
    with offline_project(
        tmp_path, candidates=candidate("refunder", routes("refund"), replies())
    ) as runs:
        summary = runs.run(tool_mode="record")
        assert runs.outcomes(summary.run_id)["refunder"] == {Outcome.FORBIDDEN_TOOL.value}


def test_an_injected_fault_the_agent_ignores_is_a_tool_error(tmp_path: Path) -> None:
    """Raised by slmeval, and still a finding about the agent: it is the
    numerator of the recovery rate, not a harness failure."""
    with offline_project(
        tmp_path,
        fault_rate=1.0,
        cases=ONE_CASE,
        extra_benchmark=(
            "[benchmark.faults]\n"
            'kinds = ["rate_limit_429"]\n'
            'delivery = { rate_limit_429 = "raise" }\n'
            "delay_scale = 0.0\n"
        ),
    ) as runs:
        # Recorded first, then replayed: a fault is never injected while
        # recording, because the faulted payload would go into the cassette.
        runs.record()
        summary = runs.run(tool_mode="replay")

        assert runs.outcomes(summary.run_id)[runs.config.baseline.id] == {
            Outcome.TOOL_ERROR.value
        }


def test_an_execution_that_outlives_its_budget_is_a_timeout(tmp_path: Path) -> None:
    """A timeout is its own status, because an agent that hangs and an agent
    that answers wrongly are not the same problem."""
    with offline_project(
        tmp_path,
        candidates=candidate("staller", routes("stall"), replies()),
        timeout_s=0.2,
    ) as runs:
        summary = runs.run(tool_mode="record")
        rows = runs.executions(summary.run_id, "staller")

        assert {r.status for r in rows} == {ExecStatus.TIMEOUT}
        assert {r.outcome_class for r in rows} == {Outcome.TIMEOUT}
