"""Every table the tool prints, pinned character for character.

The scorecard *is* the product. A column that silently changes width, a label
that loses its caveat, a delta that stops being signed - none of those break a
behavioural test, and all of them change what a user believes. So each rendered
table has a golden file under ``tests/snapshots`` and a diff here is a decision
somebody has to make rather than a surprise in a screenshot.

The run underneath is scripted, so these are comparisons rather than
tolerances: the fake provider's token counts, costs and verdicts are the same
on every machine. The two things that genuinely cannot be - the run id and the
clock - are replaced before comparing, and nothing else is.

Regenerate with ``SLMEVAL_UPDATE_SNAPSHOTS=1 pytest tests/test_snapshots.py``
and read the diff.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from rich.console import Console

from slmeval.cli.commands.gate_cmd import render_gate, to_markdown
from slmeval.cli.render import render_miss
from slmeval.gate import Thresholds, evaluate
from slmeval.report.attribution import render_attribution
from slmeval.report.drilldown import (
    build_diff,
    gaps,
    render_diff,
    render_execution,
    render_gaps,
    render_run,
)
from slmeval.report.model import build_scorecard
from slmeval.report.scorecard import (
    render_rollups,
    render_scorecard,
    render_summary_line,
    table,
    warnings_block,
)
from slmeval.sandbox.diagnosis import diagnose
from slmeval.storage.models import Outcome
from tests.conftest import assert_snapshot
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

CANDIDATES = "".join(
    [
        # The three rows worth looking at: one that passes with help, one that
        # hallucinates an order id, one that cannot emit the schema at all.
        candidate(
            "fenced-json",
            behaves("fenced_json", '{"intent": "order_status", "order_id": "ORD-9921"}'),
            behaves("fenced_json", '{"final_answer": "Your order is with DHL."}'),
            extra='extra_body.unsupported = ["tools", "json_schema"]',
        ),
        candidate("sloppy-router", routes("order_status", "9921"), replies("found it")),
        candidate("prose-only", behaves("not_json")),
    ]
)

RUN_ID = re.compile(r"\d{8}-\d{6}-[0-9a-f]{6}")
TIMESTAMP = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?")
DURATION = re.compile(r"\b\d+(\.\d+)?\s?(ms|s)\b")
VERSION = re.compile(r"\bv?\d+\.\d+\.\d+\b")


def render(renderable: object, *, width: int = 110) -> str:
    """Render to text the way a snapshot needs: no colour, fixed width.

    Width is pinned because wrapping is part of what is being asserted - the
    table has to stay readable in an 80-to-120 column terminal, which is what
    CI logs are.
    """
    console = Console(
        record=True,
        width=width,
        no_color=True,
        highlight=False,
        emoji=False,
        legacy_windows=False,
    )
    console.print(renderable)
    return console.export_text()


def without_the_clock(text: str) -> str:
    """Blank out the three things that cannot be the same twice."""
    text = RUN_ID.sub("<run-id>", text)
    text = TIMESTAMP.sub("<timestamp>", text)
    text = VERSION.sub("<version>", text)
    return DURATION.sub("<duration>", text)


@pytest.fixture
def card(tmp_path: Path):  # type: ignore[no-untyped-def]
    """One stored run: three scripted candidates, two cases, two repeats."""
    with offline_project(tmp_path, candidates=CANDIDATES, repeats=2) as runs:
        runs.record()
        summary = runs.run(tool_mode="replay")
        yield build_scorecard(runs.store, summary.run_id)


@pytest.fixture
def unsafe_card(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A run where a candidate calls the tool that moves money."""
    with offline_project(
        tmp_path, candidates=candidate("refunder", routes("refund"), replies())
    ) as runs:
        summary = runs.run(tool_mode="record")
        yield build_scorecard(runs.store, summary.run_id)


# --------------------------------------------------------------------------- #
# The scorecard
# --------------------------------------------------------------------------- #


def test_the_table(card) -> None:  # type: ignore[no-untyped-def]
    assert_snapshot("scorecard_table", render(table(card, ascii_mode=True)))


def test_the_whole_scorecard(card) -> None:  # type: ignore[no-untyped-def]
    """Header, warnings, table and footer together - the thing a user actually
    sees, including which caveats appear and in what order."""
    rendered = render(render_scorecard(card, ascii_mode=True, show_nodes=True))
    assert_snapshot("scorecard_full", without_the_clock(rendered))


def test_the_per_case_table(card) -> None:  # type: ignore[no-untyped-def]
    assert_snapshot("per_case", render(render_rollups(card, ascii_mode=True)))


def test_the_node_attribution_table(card) -> None:  # type: ignore[no-untyped-def]
    assert_snapshot("nodes", render(render_attribution(card, ascii_mode=True)))


def test_the_warnings_above_the_table(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    notes = warnings_block(unsafe_card, ascii_mode=True)
    assert notes is not None
    assert_snapshot("warnings", render(notes))


def test_the_one_line_verdict(tmp_path: Path) -> None:
    """Only offered for a single candidate, which is why it gets its own run."""
    with offline_project(
        tmp_path,
        candidates=candidate("sloppy-router", routes("order_status", "9921"), replies()),
        repeats=2,
    ) as runs:
        runs.record()
        summary = runs.run(tool_mode="replay")
        card = build_scorecard(runs.store, summary.run_id)

    assert_snapshot("summary_line", render(render_summary_line(card, ascii_mode=True)))


# --------------------------------------------------------------------------- #
# The diagnosis block
# --------------------------------------------------------------------------- #


@pytest.fixture
def recorded(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A project whose cassettes hold the baseline's real calls.

    Diagnosed against a populated store, because the interesting half of the
    block - "recorded: ORD-9921, called with: 9921, one argument diverged" -
    only exists when there is something to compare against.
    """
    with offline_project(
        tmp_path, candidates=candidate("sloppy-router", routes("order_status", "9921"))
    ) as runs:
        runs.record()
        yield runs.project.cassettes


def test_the_cassette_miss_block(recorded) -> None:  # type: ignore[no-untyped-def]
    """The block that has to be good enough to paste into a bug report: what
    was asked for, what was recorded, which argument diverged, what to do."""
    finding = diagnose(
        case_id="status_shipped",
        tool_name="fetch_order",
        arguments={"order_id": "9921"},
        store=recorded,
        known_tools=["fetch_order", "issue_refund"],
        candidate_id="sloppy-router",
        node="order_lookup",
    )
    assert_snapshot("miss_diagnosis", render(render_miss(finding, ascii_mode=True)))


def test_the_hallucinated_tool_block(recorded) -> None:  # type: ignore[no-untyped-def]
    """A different verdict, and a different thing to do about it, so it is a
    different block rather than the same one with a word changed."""
    finding = diagnose(
        case_id="status_shipped",
        tool_name="fetch_ordr",
        arguments={"order_id": "ORD-9921"},
        store=recorded,
        known_tools=["fetch_order", "issue_refund"],
        candidate_id="sloppy-router",
        node="order_lookup",
    )
    assert_snapshot("miss_unknown_tool", render(render_miss(finding, ascii_mode=True)))


# --------------------------------------------------------------------------- #
# The drill-down
# --------------------------------------------------------------------------- #


_DRILLDOWN: dict[str, Path] = {}


@pytest.fixture
def drilldown(tmp_path: Path, tmp_path_factory: pytest.TempPathFactory):  # type: ignore[no-untyped-def]
    """A stored run plus the store it came from, for the per-execution views.

    Its own fixture rather than a reuse of ``card``: the drill-down needs the
    open project - executions are reconstructed one at a time and the cassette
    store is what the diagnosis block is rebuilt from - and the scorecard
    fixture has closed both by the time it yields.

    Run once for the module and cloned per test, because four snapshots of one
    run do not need four runs.
    """
    if "drilldown" not in _DRILLDOWN:
        _DRILLDOWN["drilldown"] = build_and_run(
            tmp_path_factory.mktemp("snapshot-drilldown"),
            candidates=candidate(
                "sloppy-router", routes("order_status", "9921"), replies()
            ),
            repeats=2,
        )
    root = clone_project(_DRILLDOWN["drilldown"], tmp_path / "project")
    with open_built(root) as runs:
        latest = runs.store.latest_run()
        assert latest is not None
        yield runs, latest.run_id


def test_the_run_listing(drilldown) -> None:  # type: ignore[no-untyped-def]
    """Worst first, one line per cell. The index a user drills down from."""
    runs, run_id = drilldown
    rendered = render(
        render_run(
            runs.store.require_run(run_id),
            runs.store.candidates(run_id),
            runs.store.executions(run_id),
            ascii_mode=True,
        )
    )
    assert_snapshot("show_run", without_the_clock(rendered))


def test_the_side_by_side_diff(drilldown) -> None:  # type: ignore[no-untyped-def]
    """The view the plan calls the failure drill-down: baseline left, candidate
    right, divergence marked, diagnosis block inline."""
    runs, run_id = drilldown
    case_id = "status_shipped"
    executions = runs.store.executions(run_id, case_id=case_id)
    mine = next(e for e in executions if e.candidate_id == "sloppy-router")
    reference = next(e for e in executions if e.candidate_id == "baseline")
    diff = build_diff(
        case_id=case_id,
        candidate=runs.store.reconstruct(mine.exec_id),
        baseline=runs.store.reconstruct(reference.exec_id),
        candidate_id="sloppy-router",
        cassettes=runs.project.cassettes,
    )
    assert_snapshot("show_diff", without_the_clock(render(render_diff(diff, ascii_mode=True))))


def test_one_execution_timeline(drilldown) -> None:  # type: ignore[no-untyped-def]
    runs, run_id = drilldown
    reference = next(
        e for e in runs.store.executions(run_id) if e.candidate_id == "baseline"
    )
    rendered = render(
        render_execution(runs.store.reconstruct(reference.exec_id), ascii_mode=True)
    )
    assert_snapshot("show_exec", without_the_clock(rendered))


def test_the_uncovered_listing(drilldown) -> None:  # type: ignore[no-untyped-def]
    """What `record --fill-gaps` is pointed at, with the hallucinations kept
    out of it - see the command's own docstring for why that matters."""
    runs, run_id = drilldown
    details = [
        (e.candidate_id, runs.store.reconstruct(e.exec_id))
        for e in runs.store.executions(run_id)
        if e.outcome_class in {Outcome.CASSETTE_MISS, Outcome.UNCOVERED}
    ]
    rendered = render(render_gaps(gaps(details), ascii_mode=True, run_id=run_id))
    assert_snapshot("show_uncovered", without_the_clock(rendered))


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #


def test_the_gate_verdict(card) -> None:  # type: ignore[no-untyped-def]
    """The block a CI log shows, and the one people will paste into an issue
    when they disagree with a verdict. Verdict first, then why."""
    report = evaluate(
        card,
        Thresholds(min_pass_k=0.9, max_cost_ratio=0.5, max_p95_ratio=2.0),
    )
    assert_snapshot(
        "gate_failed", without_the_clock(render(render_gate(report, ascii_mode=True)))
    )


def test_the_gate_markdown(card) -> None:  # type: ignore[no-untyped-def]
    """The pull-request comment. Pinned because this is where most people meet
    the tool, and because a lost caveat here is a misread merge."""
    report = evaluate(card, Thresholds(min_pass_k=0.9))
    assert_snapshot("gate_markdown", without_the_clock(to_markdown(report)))
