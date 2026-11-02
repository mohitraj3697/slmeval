"""The drill-down: alignment, the rebuilt diagnosis, and the gap list.

The alignment tests are the load-bearing ones. A diff that reported four
divergences where one step was inserted would send a user hunting through a
trajectory that is almost entirely fine, and that is exactly what zipping two
sequences does - so the behaviour is pinned on hand-written sequences first and
on a real run second.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from slmeval.report.drilldown import (
    DELETE,
    EQUAL,
    INSERT,
    REPLACE,
    align,
    build_diff,
    diff_payload,
    execution_payload,
    gaps,
    rebuild_diagnosis,
    tool_steps,
    trajectory_of,
)
from slmeval.sandbox.diagnosis import Verdict
from slmeval.storage.models import Outcome
from tests.offline import (
    behaves,
    build_and_run,
    candidate,
    clone_project,
    forget_the_agent_module,  # noqa: F401 - autouse fixture
    offline_project,
    open_built,
    replies,
    routes,
)


def ops(left: list[str], right: list[str]) -> list[str]:
    return [row.op for row in align(left, right).rows]


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #


def test_identical_sequences_align_entirely() -> None:
    alignment = align(["a", "b", "c"], ["a", "b", "c"])
    assert alignment.identical
    assert alignment.diverged_at is None
    assert alignment.differences == 0


def test_an_inserted_step_is_one_difference_not_a_shifted_tail() -> None:
    """The whole reason this uses SequenceMatcher rather than zip.

    A candidate that visits one node twice has made one mistake. Zipping would
    report every later step as different too, and a user would go looking for
    three bugs.
    """
    alignment = align(["a", "b", "c"], ["a", "b", "b", "c"])
    assert alignment.differences == 1
    assert [row.op for row in alignment.rows] == [EQUAL, EQUAL, INSERT, EQUAL]
    assert alignment.diverged_at == 2


def test_a_missing_step_reads_as_a_deletion() -> None:
    assert ops(["a", "b", "c"], ["a", "c"]) == [EQUAL, DELETE, EQUAL]


def test_a_substitution_pairs_the_two_sides_up() -> None:
    rows = align(["a", "x", "c"], ["a", "y", "c"]).rows
    assert [r.op for r in rows] == [EQUAL, REPLACE, EQUAL]
    assert rows[1].left == "x"
    assert rows[1].right == "y"


def test_an_uneven_substitution_spills_the_remainder() -> None:
    """Two steps replaced by three is two changes and one insertion.

    Flattening it to three of either would misreport what happened, and the
    count above the table is the number a user acts on.
    """
    rows = align(["a", "x", "y", "d"], ["a", "p", "q", "r", "d"]).rows
    assert [r.op for r in rows] == [EQUAL, REPLACE, REPLACE, INSERT, EQUAL]


def test_an_empty_left_side_is_all_insertions() -> None:
    assert ops([], ["a", "b"]) == [INSERT, INSERT]


def test_a_repeated_element_is_not_treated_as_junk() -> None:
    """``autojunk`` would discard the node every step visits.

    SequenceMatcher's default heuristic drops elements appearing in more than
    1% of a sequence over 200 long - in a trajectory where one node runs on
    every step, that is the element the diff is about.
    """
    left = ["loop"] * 300
    right = ["loop"] * 299 + ["done"]
    alignment = align(left, right)
    assert alignment.differences == 1


def test_the_gutter_glyph_distinguishes_the_three_kinds() -> None:
    rows = align(["a", "b", "c"], ["a", "x", "c", "d"]).rows
    assert [r.glyph for r in rows] == [" ", "~", " ", "+"]


# --------------------------------------------------------------------------- #
# A real run
# --------------------------------------------------------------------------- #


SLOPPY = candidate("sloppy-router", routes("order_status", "9921"), replies())

_TEMPLATE: dict[str, Path] = {}


@pytest.fixture
def run(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """One recorded baseline and a candidate that mangles the order id.

    Recorded and run once for the module, then cloned and reopened per test.
    Nothing below writes to the store, but each test needs its own copy anyway -
    SQLite files and a chdir are not shareable - and a clone costs a thousandth
    of what re-running the matrix does.
    """
    if "sloppy" not in _TEMPLATE:
        _TEMPLATE["sloppy"] = build_and_run(
            tmp_path_factory.mktemp("drilldown"), candidates=SLOPPY
        )
    root = clone_project(_TEMPLATE["sloppy"], tmp_path / "project")
    with open_built(root) as runs:
        latest = runs.store.latest_run()
        assert latest is not None
        yield runs, latest.run_id


def build(  # type: ignore[no-untyped-def]
    runs,
    run_id: str,
    *,
    case_id: str = "status_shipped",
    candidate_id: str = "sloppy-router",
):
    executions = runs.store.executions(run_id, case_id=case_id)
    mine = next(e for e in executions if e.candidate_id == candidate_id)
    reference = next(e for e in executions if e.candidate_id == "baseline")
    return build_diff(
        case_id=case_id,
        candidate=runs.store.reconstruct(mine.exec_id),
        baseline=runs.store.reconstruct(reference.exec_id),
        candidate_id=candidate_id,
        cassettes=runs.project.cassettes,
    )


def test_a_mangled_argument_shows_as_a_replaced_tool_call(run) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = run
    diff = build(runs, run_id)

    assert diff.comparable
    assert not diff.identical
    assert diff.tools.rows[0].op == REPLACE
    assert "ORD-9921" in diff.tools.rows[0].left.render()
    assert '"9921"' in diff.tools.rows[0].right.render()


def test_the_diagnosis_block_comes_back_with_the_diff(run) -> None:  # type: ignore[no-untyped-def]
    """The block is the product's voice, and the drill-down prints it inline.

    Rebuilt from the cassette store rather than from the exception string,
    which only ever held its first line.
    """
    runs, run_id = run
    diff = build(runs, run_id)

    assert len(diff.diagnoses) == 1
    finding = diff.diagnoses[0]
    assert finding.verdict is Verdict.HALLUCINATED_PARAMETER
    assert finding.nearest_arguments == {"order_id": "ORD-9921"}
    assert [d.note for d in finding.diffs] == ["dropped prefix"]


def test_the_stored_verdict_wins_over_a_rebuilt_one(run) -> None:  # type: ignore[no-untyped-def]
    """A report must not re-judge a candidate.

    The verdict was decided with the live tool registry; this runs later and
    without the agent loaded. Recovering the *detail* is the job - deciding the
    outcome again is not, or the stored outcome column becomes a lie.
    """
    runs, run_id = run
    detail = next(
        runs.store.reconstruct(e.exec_id)
        for e in runs.store.executions(run_id)
        if e.candidate_id == "sloppy-router" and e.outcome_class is Outcome.CASSETTE_MISS
    )
    step = next(s for s in tool_steps(detail) if s.missed)

    # Pretend the recording said something else. Nothing about the live store
    # changed, so a rebuild left to itself would disagree.
    pinned = replace(step, miss_verdict=Verdict.WRONG_TOOL_FOR_STEP.value)
    finding = rebuild_diagnosis(
        pinned,
        case_id="status_shipped",
        candidate_id="sloppy-router",
        cassettes=runs.project.cassettes,
        known_tools=["fetch_order", "issue_refund"],
    )
    assert finding is not None
    assert finding.verdict is Verdict.WRONG_TOOL_FOR_STEP
    assert "recorded verdict" in finding.rationale


def test_a_real_tool_is_not_called_invented_without_the_agent_loaded(run) -> None:  # type: ignore[no-untyped-def]
    """``show`` never imports the user's module, so the live registry is empty.

    Without the names recovered from the traces and the cassettes, every
    rebuilt diagnosis would read ``hallucinated_tool`` - a finding the harness
    invented out of its own loading strategy.
    """
    runs, run_id = run
    diff = build(runs, run_id)
    assert diff.diagnoses[0].verdict is not Verdict.HALLUCINATED_TOOL


def test_fault_parity_is_reported_not_assumed(run) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = run
    assert build(runs, run_id).fault_parity is True


def test_a_candidate_with_no_baseline_execution_still_renders(run) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = run
    mine = next(
        e for e in runs.store.executions(run_id) if e.candidate_id == "sloppy-router"
    )
    diff = build_diff(
        case_id=mine.case_id,
        candidate=runs.store.reconstruct(mine.exec_id),
        baseline=None,
        candidate_id="sloppy-router",
    )
    assert not diff.comparable
    assert "no baseline execution" in diff.summary()


def test_the_trajectory_renders_nested_paths(run) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = run
    detail = runs.store.reconstruct(
        next(
            e.exec_id
            for e in runs.store.executions(run_id)
            if e.candidate_id == "baseline"
        )
    )
    assert trajectory_of(detail)[0] == "intent_router"


# --------------------------------------------------------------------------- #
# Gaps
# --------------------------------------------------------------------------- #


def test_repeats_of_one_gap_are_one_row(tmp_path: Path) -> None:
    """Three repeats reaching the same unrecorded call is one recording to make.

    Printed per execution it would read as three separate holes, and a user
    sizing the work would see six where there are two.
    """
    with offline_project(
        tmp_path,
        candidates=candidate(
            "explorer", routes("order_status", "ORD-NEW"), replies()
        ),
        repeats=3,
    ) as runs:
        runs.record()
        summary = runs.run(tool_mode="replay")
        details = [
            (e.candidate_id, runs.store.reconstruct(e.exec_id))
            for e in runs.store.executions(summary.run_id)
            if e.candidate_id == "explorer"
        ]

    found = gaps(details)
    uncovered = [g for g in found if g.case_id == "status_shipped"]
    assert len(uncovered) == 1
    assert uncovered[0].executions == 3
    assert uncovered[0].candidates == ("explorer",)


def test_uncovered_only_drops_the_hallucinations(run) -> None:  # type: ignore[no-untyped-def]
    """The list that feeds ``record --fill-gaps`` must not contain a mangled id.

    Re-recording a hallucinated order id would write the model's mistake into
    the cassettes as ground truth, and every later run would then grade against
    it.
    """
    runs, run_id = run
    details = [
        (e.candidate_id, runs.store.reconstruct(e.exec_id))
        for e in runs.store.executions(run_id)
    ]
    every = gaps(details)
    mangled = [g for g in every if g.verdict == Verdict.HALLUCINATED_PARAMETER.value]
    assert mangled, "the candidate mangled an order id on at least one case"

    to_record = gaps(details, uncovered_only=True)
    assert all(g.uncovered for g in to_record)
    assert mangled[0] not in to_record


# --------------------------------------------------------------------------- #
# Payloads
# --------------------------------------------------------------------------- #


def test_the_diff_payload_keeps_a_missing_side_null(run) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = run
    payload = diff_payload(build(runs, run_id))

    assert payload["identical"] is False
    assert payload["candidate_outcome"] == Outcome.CASSETTE_MISS.value
    changed = [row for row in payload["tool_calls"] if row["op"] == REPLACE]
    assert changed[0]["baseline"]["arguments"] == {"order_id": "ORD-9921"}
    assert changed[0]["candidate"]["arguments"] == {"order_id": "9921"}
    assert changed[0]["candidate"]["outcome"] == "miss"
    assert payload["diagnoses"][0]["verdict"] == "hallucinated_parameter"
    assert "CASSETTE_MISS" in payload["diagnoses"][0]["block"]


def test_the_execution_payload_says_whether_it_was_judgeable(tmp_path: Path) -> None:
    """``uncovered`` and ``harness_error`` leave every rate, and the drill-down
    has to say so - otherwise a user reading one execution has no way to know
    it was excluded from the table they came from."""
    with offline_project(
        tmp_path, candidates=candidate("prose-only", behaves("not_json"))
    ) as runs:
        summary = runs.run(tool_mode="record")
        detail = runs.store.reconstruct(
            next(
                e.exec_id
                for e in runs.store.executions(summary.run_id)
                if e.candidate_id == "prose-only"
            )
        )

    payload = execution_payload(detail)
    assert payload["judgeable"] is True
    assert payload["outcome"] == Outcome.SCHEMA_FAILURE.value
    assert payload["model_calls"]
