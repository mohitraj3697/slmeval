"""Reporting: the terminal table, the per-node breakdown, and the exports.

For a CLI tool the table *is* the product. It is what gets screenshotted,
posted and pasted into a pull request, so this package is marketing as much as
it is presentation - and it is held to the same standard as the measurement
code, because an honest number rendered misleadingly is still misleading.

Everything in here is a pure function from stored data to a renderable or a
string. Nothing prints, nothing queries a model, and nothing re-executes
anything:

:mod:`slmeval.report.model`
    The scorecard as data, built from SQLite rows alone. This is what makes
    ``slmeval report --run-id X`` reproduce an old table exactly.
:mod:`slmeval.report.scorecard`
    The headline table, warnings above it, provenance below it.
:mod:`slmeval.report.attribution`
    Per-node attribution - which node each candidate broke in, and therefore
    which node to leave on the bigger model.
:mod:`slmeval.report.drilldown`
    The failure drill-down: one execution's timeline, and two of them side by
    side with the divergence marked and the cassette-miss block inline.
:mod:`slmeval.report.baseline`
    ``slmeval.baseline.json``: the committed reference a gate compares
    against, metrics plus the provenance that says whether comparing is even
    valid.
:mod:`slmeval.report.export`
    ``json`` (a versioned contract), ``md`` (for PR comments), ``csv``, and
    ``junit`` (per-case rows for a CI dashboard that already exists).
:mod:`slmeval.report.progress`
    The live run view, as an observer the runner knows nothing about.
"""

from __future__ import annotations

from slmeval.report.attribution import NodeCell, node_cells, render_attribution
from slmeval.report.baseline import Baseline, CandidateBaseline, baseline_path
from slmeval.report.drilldown import (
    Alignment,
    Diff,
    Gap,
    Pair,
    align,
    build_diff,
    diff_payload,
    execution_payload,
    gaps,
    render_diff,
    render_execution,
    render_gaps,
    render_run,
)
from slmeval.report.export import (
    FORMATS,
    SCHEMA_VERSION,
    export,
    to_csv,
    to_json,
    to_junit,
    to_markdown,
    to_payload,
)
from slmeval.report.model import (
    CandidateReport,
    Incident,
    Scorecard,
    build_scorecard,
    candidate_report,
)
from slmeval.report.progress import ProgressObserver, Tally
from slmeval.report.scorecard import (
    render_rollups,
    render_scorecard,
    render_summary_line,
)

__all__ = [
    "FORMATS",
    "SCHEMA_VERSION",
    "Alignment",
    "Baseline",
    "CandidateBaseline",
    "CandidateReport",
    "Diff",
    "Gap",
    "Incident",
    "NodeCell",
    "Pair",
    "ProgressObserver",
    "Scorecard",
    "Tally",
    "align",
    "baseline_path",
    "build_diff",
    "build_scorecard",
    "candidate_report",
    "diff_payload",
    "execution_payload",
    "export",
    "gaps",
    "node_cells",
    "render_attribution",
    "render_diff",
    "render_execution",
    "render_gaps",
    "render_rollups",
    "render_run",
    "render_scorecard",
    "render_summary_line",
    "to_csv",
    "to_json",
    "to_junit",
    "to_markdown",
    "to_payload",
]
