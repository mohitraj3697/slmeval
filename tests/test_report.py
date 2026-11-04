"""The scorecard, rendered from stored rows only.

Day 14's acceptance criterion has two halves and both are tested here: the
scorecard renders from stored data with nothing re-executed, and a snapshot
test fails if a column changes.

The snapshot is the header row asserted character for character. A whole-table
golden string would break on every latency number; the header is exactly the
part that must not change without somebody deciding to change it.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import pytest
from rich.console import Console

from slmeval.eval.pipeline import case_rollups
from slmeval.report.attribution import node_cells, node_order, recommendation
from slmeval.report.export import (
    SCHEMA_VERSION,
    to_csv,
    to_json,
    to_junit,
    to_markdown,
)
from slmeval.report.model import LATENCY_DISCLAIMER, _histogram, build_scorecard
from slmeval.report.scorecard import (
    render_rollups,
    render_scorecard,
    render_summary_line,
    table,
)
from slmeval.storage.db import RunStore
from slmeval.storage.models import MetricSample, Outcome
from tests.test_runner import MODULE, build

HEADER = (
    "Candidate    pass^2         95% CI    runs    Schema    Tool F1    "
    "Steps      $/case    d Cost"
)


def render(renderable: object, *, width: int = 200) -> str:
    """Render to text the way a snapshot test needs: no colour, fixed width."""
    console = Console(
        record=True, width=width, no_color=True, highlight=False, emoji=False, legacy_windows=False
    )
    console.print(renderable)
    return console.export_text()


@pytest.fixture(autouse=True)
def forget_the_agent_module() -> Iterator[None]:
    yield
    sys.modules.pop(MODULE, None)


@pytest.fixture
def card(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A real stored run: two candidates, two cases, two repeats."""
    fixture = build(tmp_path, cases=2, repeats=2, candidates=(("weak", "bad"),))
    try:
        summary = fixture.go()
        fixture.store.save_rollups(
            case_rollups(summary.run_id, fixture.store.executions(summary.run_id))
        )
        # Rebuilt from a *fresh* store, so nothing in the card can have come
        # from an object the run left in memory.
        with RunStore.for_project(tmp_path) as reopened:
            yield build_scorecard(reopened, summary.run_id)
    finally:
        fixture.store.close()


@pytest.fixture
def unsafe_card(tmp_path: Path):  # type: ignore[no-untyped-def]
    """A run where a candidate calls a forbidden tool."""
    fixture = build(tmp_path, cases=1, repeats=2, candidates=(("weak", "bad"),))
    (tmp_path / "cases.jsonl").write_text(
        '{"case_id": "c0", "input": {"q": "q0"}, "assert": '
        '{"state_contains": {"intent": "ok"}, "forbidden_tools": ["danger"]}}\n',
        encoding="utf-8",
    )
    (tmp_path / f"{MODULE}.py").write_text(
        (tmp_path / f"{MODULE}.py").read_text(encoding="utf-8").replace(
            "from slmeval import chat_model, node",
            "from slmeval import chat_model, node, sandbox_tool\n\n\n"
            "@sandbox_tool\n"
            "async def danger(reason: str) -> dict:\n"
            '    return {"did": reason}',
        ).replace(
            "    routing = completion.parsed_as(Routing)",
            "    routing = completion.parsed_as(Routing)\n"
            '    if routing.intent != "ok":\n'
            '        await danger("oops")',
        ),
        encoding="utf-8",
    )
    sys.modules.pop(MODULE, None)
    from slmeval.core.adapter import load_target
    from slmeval.dataset.loader import load_dataset

    fixture.dataset = load_dataset(tmp_path / "cases.jsonl")
    fixture.target = load_target(f"{MODULE}:run_case", root=tmp_path)
    try:
        summary = fixture.go(tool_mode="record")
        fixture.store.save_rollups(
            case_rollups(summary.run_id, fixture.store.executions(summary.run_id))
        )
        yield build_scorecard(fixture.store, summary.run_id)
    finally:
        fixture.store.close()


# --------------------------------------------------------------------------- #
# Built from storage alone
# --------------------------------------------------------------------------- #


def test_the_scorecard_is_rebuilt_from_rows_with_nothing_re_executed(card) -> None:  # type: ignore[no-untyped-def]
    assert card.run.status.value == "completed"
    assert [r.candidate_id for r in card.rows] == ["baseline", "weak"]
    assert card.baseline is not None and card.baseline.candidate_id == "baseline"
    assert [r.candidate_id for r in card.candidates] == ["weak"]
    assert set(card.cases) == {"c0", "c1"}
    assert card.repeats == 2
    assert card.executions == 8


def test_the_baseline_row_comes_first_whatever_the_query_returns(card) -> None:  # type: ignore[no-untyped-def]
    assert card.rows[0].baseline
    assert not card.rows[1].baseline


def test_pass_hat_k_and_its_interval_describe_the_same_thing(card) -> None:  # type: ignore[no-untyped-def]
    """The label and the interval must never be computed over different
    denominators."""
    baseline = card.baseline
    assert baseline is not None
    assert baseline.pass_hat_k == 1.0
    assert baseline.stats.cases == 2
    assert baseline.stats.cases_all_pass == 2
    assert baseline.wilson is not None
    lo, hi = baseline.wilson
    # Two cases, both passing: nowhere near certainty, which is the point.
    assert lo < 0.4
    assert hi == 1.0


def test_the_weak_candidate_fails_every_case(card) -> None:  # type: ignore[no-untyped-def]
    weak = card.candidates[0]
    assert weak.pass_hat_k == 0.0
    assert weak.outcomes[Outcome.FAILED_ASSERTION] == 4


def test_rates_are_weighted_by_their_own_denominator(card) -> None:  # type: ignore[no-untyped-def]
    """An execution that made one structured call must not count as much as
    one that made twenty."""
    baseline = card.baseline
    assert baseline is not None
    assert baseline.schema_calls == 4
    assert baseline.schema_valid_rate == 1.0
    assert baseline.schema_repaired_rate == 0.0


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #


def test_the_column_header_is_pinned(card) -> None:  # type: ignore[no-untyped-def]
    """The snapshot. If a column is added, renamed or reordered, this fails -
    which is the only thing standing between a thought-through table and one
    that drifted."""
    header = render(table(card, ascii_mode=True)).splitlines()[0].rstrip()
    assert header == HEADER


def test_a_throughput_run_has_no_latency_column(card) -> None:  # type: ignore[no-untyped-def]
    """Publishing local-model latency measured under concurrency would be the
    fastest way to lose credibility, so the column is absent rather than
    filled in."""
    assert not card.latency_reportable
    assert "p95" not in render(table(card, ascii_mode=True))
    assert LATENCY_DISCLAIMER in card.warnings()


def test_a_latency_run_gets_the_column_and_loses_the_disclaimer(
    tmp_path: Path,
) -> None:
    fixture = build(tmp_path, cases=1, repeats=1)
    try:
        summary = fixture.go(serial_latency=True)
        fixture.store.save_rollups(
            case_rollups(summary.run_id, fixture.store.executions(summary.run_id))
        )
        card = build_scorecard(fixture.store, summary.run_id)
        assert card.run.scheduling == "latency"
        assert card.latency_reportable
        assert "p95 ms" in render(table(card, ascii_mode=True))
        assert LATENCY_DISCLAIMER not in card.warnings()
        assert card.rows[0].p95_ms is not None
    finally:
        fixture.store.close()


def test_n_is_always_printed(card) -> None:  # type: ignore[no-untyped-def]
    """A rate without its sample size is a number pretending to be evidence."""
    text = render(table(card, ascii_mode=True))
    assert "4/4" in text  # baseline: four executions, four passes
    assert "0/4" in text
    assert "runs" in text


def test_the_interval_is_printed_next_to_the_point_estimate(card) -> None:  # type: ignore[no-untyped-def]
    text = render(table(card, ascii_mode=True))
    assert "100.0%" in text
    assert "[" in text and "%]" in text


def test_the_whole_scorecard_renders_with_header_and_footer(card) -> None:  # type: ignore[no-untyped-def]
    text = render(render_scorecard(card, ascii_mode=True, show_nodes=True))
    assert "tiny" in text
    assert card.run.run_id in text
    assert f"dataset {card.run.dataset_hash}" in text
    assert "per-node attribution" in text
    assert "gate on the lower bound, never the point estimate" in text
    assert f"prices as of {card.prices_as_of}" in text


def test_the_per_case_table_says_which_cases_failed(card) -> None:  # type: ignore[no-untyped-def]
    text = render(render_rollups(card, ascii_mode=True))
    assert "c0" in text and "c1" in text
    assert "2/2" in text  # baseline passed both repeats
    assert "0/2" in text  # the weak candidate passed neither


def test_the_one_line_verdict_is_only_offered_for_a_single_candidate(card) -> None:  # type: ignore[no-untyped-def]
    line = str(render_summary_line(card, ascii_mode=True))
    assert "pass^2" in line
    assert "100.0%" in line


# --------------------------------------------------------------------------- #
# Warnings above the table
# --------------------------------------------------------------------------- #


def test_a_forbidden_tool_call_is_reported_above_the_table(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    """Day 11's acceptance criterion, seen from the report: a hard fail with a
    loud warning that cannot be averaged away or scrolled past."""
    warnings = unsafe_card.warnings()
    assert warnings
    assert "FORBIDDEN tool call" in warnings[0]
    assert "danger" in warnings[0]

    text = render(render_scorecard(unsafe_card, ascii_mode=True))
    table_at = text.index("Candidate")
    assert text.index("FORBIDDEN") < table_at


def test_repeated_incidents_are_counted_not_repeated(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    """Three repeats of one case calling the same tool is one finding seen
    three times, and printing it three times pushes the other warnings off the
    screen."""
    incidents = unsafe_card.incidents
    assert len(incidents) == 1
    assert incidents[0].executions == 2
    assert "2x" in incidents[0].describe()


def test_a_forbidden_call_is_the_candidates_outcome(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    weak = unsafe_card.candidates[0]
    assert weak.unsafe_actions == 2
    assert weak.outcomes[Outcome.FORBIDDEN_TOOL] == 2


# --------------------------------------------------------------------------- #
# Per-node attribution
# --------------------------------------------------------------------------- #


def test_nodes_are_ordered_worst_first_not_busiest_first(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    """The busiest node is the busiest on every run and says nothing; failures
    are what is scarce."""
    cells = node_cells(unsafe_card)
    order = node_order(unsafe_card, cells)
    failures = [
        sum(c.failures for (_, n), c in cells.items() if n == node) for node in order
    ]
    assert failures == sorted(failures, reverse=True)


def test_the_recommendation_names_the_node_to_keep_on_the_baseline(card) -> None:  # type: ignore[no-untyped-def]
    """The consulting wedge in one sentence - offered only when the data
    supports it."""
    cells = node_cells(card)
    advice = recommendation(card, cells, "->")
    if advice is not None:
        assert "keep baseline on" in str(advice)


def test_a_candidate_that_fails_nowhere_gets_no_advice(tmp_path: Path) -> None:
    """A candidate with no failing node has nothing to recommend, and
    inventing a sentence for it would be the tool talking past its evidence."""
    fixture = build(tmp_path, cases=1, repeats=1, candidates=(("twin", "ok"),))
    try:
        summary = fixture.go()
        fixture.store.save_rollups(
            case_rollups(summary.run_id, fixture.store.executions(summary.run_id))
        )
        card = build_scorecard(fixture.store, summary.run_id)
        assert recommendation(card, node_cells(card)) is None
    finally:
        fixture.store.close()


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #


def test_the_json_export_is_a_versioned_contract(card) -> None:  # type: ignore[no-untyped-def]
    payload = json.loads(to_json(card))
    assert payload["schema"] == SCHEMA_VERSION

    run = payload["run"]
    # Provenance is part of the contract: without these a consumer cannot tell
    # whether two payloads are comparable.
    assert run["config_hash"] and run["dataset_hash"]
    assert run["scheduling"] == "throughput"
    assert run["latency_reportable"] is False
    assert run["cases"] == list(card.cases)

    baseline = payload["candidates"][0]
    assert baseline["candidate_id"] == "baseline"
    assert baseline["role"] == "baseline"
    assert baseline["pass_hat_k"] == 1.0
    assert baseline["wilson_lo"] is not None
    assert len(baseline["cases_detail"]) == 2
    assert payload["nodes"]


def test_the_json_export_keeps_unmeasured_values_null(card) -> None:  # type: ignore[no-untyped-def]
    """A consumer that charted a throughput run's p95 would publish our
    queueing as the model's latency."""
    payload = json.loads(to_json(card))
    assert all(c["p95_ms"] is None for c in payload["candidates"])
    assert all(c["p50_ms"] is None for c in payload["candidates"])


def test_the_markdown_export_leads_with_the_warnings(unsafe_card) -> None:  # type: ignore[no-untyped-def]
    text = to_markdown(unsafe_card)
    assert text.index("FORBIDDEN") < text.index("| Candidate")
    assert "|---|" in text
    assert "Gate on the lower bound" in text


def test_the_markdown_export_carries_the_recovery_rate(card) -> None:  # type: ignore[no-untyped-def]
    """The markdown table is what gets pasted into a pull request.

    A faulted run has no recovery *column* - it is a footnote, as in the
    terminal - and it is the only thing in the report that explains why the
    pass rates dropped. Without it a reviewer reads a faulted table as the
    models having got worse, when what happened is that the agent fell over
    on the first empty payload a tool handed it.
    """
    faulted = replace(
        card,
        rows=tuple(replace(row, faults_met=4, recovered=1) for row in card.rows),
    )
    text = to_markdown(faulted)
    assert "Recovery from injected faults:" in text
    assert "25%" in text
    assert "8 fault(s) met" in text

    # And absent, not zero, when no fault was injected: a line claiming 0%
    # recovery on a run with no faults in it would be a fabricated finding.
    assert "Recovery from injected faults" not in to_markdown(card)


def test_the_csv_export_is_long_and_narrow(card) -> None:  # type: ignore[no-untyped-def]
    """Narrow, so adding a metric in week 3 appends rows instead of changing
    every consumer's columns."""
    lines = to_csv(card).strip().splitlines()
    assert lines[0] == "run_id,candidate_id,role,provider,model,metric,value,n"
    metrics = {line.split(",")[5] for line in lines[1:]}
    assert {"pass_hat_2", "wilson_lo", "schema_valid_rate", "tool_f1"} <= metrics
    # Every row carries its n, because a rate without one does not belong in a
    # spreadsheet.
    assert all(line.split(",")[7] != "" for line in lines[1:])


def test_the_junit_export_has_one_test_case_per_candidate_per_case(card) -> None:  # type: ignore[no-untyped-def]
    """The format that puts this in a dashboard the team already reads."""
    tree = ElementTree.fromstring(to_junit(card))
    suites = tree.findall("testsuite")
    name = card.run.name
    assert {s.get("name") for s in suites} == {
        f"slmeval.{name}.baseline",
        f"slmeval.{name}.weak",
    }

    by_name = {s.get("name"): s for s in suites}
    baseline = by_name[f"slmeval.{name}.baseline"]
    assert len(baseline.findall("testcase")) == len(card.cases)
    assert baseline.findall("failure") == []
    assert baseline.get("failures") == "0"

    # The weak candidate fails every case, and the message carries c/n and the
    # interval: "2/3" and "2/3 [21%, 94%]" lead to different decisions.
    weak = by_name[f"slmeval.{name}.weak"]
    failures = weak.findall("testcase/failure")
    assert len(failures) == len(card.cases)
    assert all("0/2 repeat(s) passed" in (f.get("message") or "") for f in failures)
    assert all("[" in (f.get("message") or "") for f in failures)

    # Provenance on the suite, so a dashboard row can be traced back to a run.
    properties = {
        p.get("name"): p.get("value") for p in baseline.findall("properties/property")
    }
    assert properties["run_id"] == card.run.run_id
    assert properties["dataset_hash"] == card.run.dataset_hash
    assert properties["runs_per_case"] == str(card.repeats)


def test_the_junit_export_skips_what_it_could_not_judge(card) -> None:  # type: ignore[no-untyped-def]
    """Our own recording gap must not reach a dashboard as a model failure."""
    row = card.rows[0]
    missing = dict(row.rollups)
    missing.pop(card.cases[0])
    stripped = replace(card, rows=(replace(row, rollups=missing), *card.rows[1:]))

    suite = ElementTree.fromstring(to_junit(stripped)).findall("testsuite")[0]
    skipped = suite.findall("testcase/skipped")
    assert len(skipped) == 1
    assert "record --fill-gaps" in (skipped[0].get("message") or "")
    assert suite.get("skipped") == "1"


def test_the_junit_export_reports_no_latency(card) -> None:  # type: ignore[no-untyped-def]
    """Per-case wall time only means something on a --serial-latency run, and a
    dashboard plotting concurrency noise as a trend is worse than no trend."""
    assert "time=" not in to_junit(card)


def test_an_unknown_export_format_is_refused(card) -> None:  # type: ignore[no-untyped-def]
    from slmeval.report.export import export

    with pytest.raises(ValueError, match="unknown export format"):
        export(card, "yaml")


def test_structured_modes_count_calls_not_executions() -> None:
    """Warnings divide these counts by the number of calls, so they must be calls.

    Counted per execution, a model using json_mode on all 21 of its calls across
    12 executions was reported as needing it on 57%.
    """
    samples = [
        MetricSample(
            candidate_id="c",
            case_id=f"case{i}",
            repeat_idx=0,
            exec_id=f"e{i}",
            tier=1,
            metric="structured_mode_used",
            value_text="json_mode",
            detail_json='{"json_mode": 2}',
        )
        for i in range(3)
    ]
    assert _histogram(samples) == {"json_mode": 6}
