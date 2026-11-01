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

from slmeval.cli.render import render_miss
from slmeval.report.attribution import render_attribution
from slmeval.report.model import build_scorecard
from slmeval.report.scorecard import (
    render_rollups,
    render_scorecard,
    render_summary_line,
    table,
    warnings_block,
)
from slmeval.sandbox.diagnosis import diagnose
from tests.conftest import assert_snapshot
from tests.offline import (
    behaves,
    candidate,
    forget_the_agent_module,  # noqa: F401 - autouse fixture
    offline_project,
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
