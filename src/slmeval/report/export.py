"""Exports: JSON, Markdown, CSV.

The JSON shape is a **public, versioned contract**. It is what the future
hosted product ingests and what users will automate against, so it carries a
``schema`` version and changes additively. Renaming a key here breaks somebody's
dashboard silently, which is why :data:`SCHEMA_VERSION` exists and why every
field is spelled out rather than produced by dumping dataclasses.

Markdown is for pull-request comments, which is where most people will first
see this tool's output. CSV is for the spreadsheet somebody will inevitably
build, and it is deliberately long and narrow - one row per candidate per
metric - because a wide CSV becomes a migration every time a metric is added.
JUnit XML is for the CI dashboard that already exists: one test case per
``(candidate, case)``, so a team gets per-case history in the tool they already
read without learning anything about this one.

The same honesty rules apply in every format. ``null`` stays ``null``: an
unpriced model must not export as ``0``, a latency that was not measured must
not export as a number, and a rate over no observations must not export as
zero. A consumer that sees a figure has to be able to trust that it was
measured.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Iterable
from typing import Any, Final
from xml.etree import ElementTree

from slmeval.report.model import CandidateReport, Scorecard
from slmeval.storage import codec

SCHEMA_VERSION: Final = "slmeval.report/1"
"""Versioned on purpose. Consumers branch on it; it changes additively."""

XML_HEADER: Final = '<?xml version="1.0" encoding="UTF-8"?>\n'


def to_payload(card: Scorecard) -> dict[str, Any]:
    """The documented JSON structure, as a plain dict."""
    run = card.run
    return {
        "schema": SCHEMA_VERSION,
        "run": {
            "run_id": run.run_id,
            "name": run.name,
            "started_at": run.started_at,
            "finished_at": run.finished_at,
            "status": run.status.value,
            "slmeval_version": run.slmeval_version,
            # Provenance is part of the contract, not metadata: without these
            # two hashes a consumer cannot tell whether two payloads are
            # comparable, and comparing them anyway is how a dataset edit gets
            # published as a regression.
            "config_hash": run.config_hash,
            "dataset_hash": run.dataset_hash,
            "dataset_path": run.dataset_path,
            "git_sha": run.git_sha,
            "mode": run.mode,
            "scheduling": run.scheduling,
            "latency_reportable": card.latency_reportable,
            "seed": run.seed,
            "fault_rate": run.fault_rate,
            "runs_per_case": run.runs_per_case,
            "cases": list(card.cases),
            "host_info": codec.loads(run.host_info, {}),
        },
        "prices_as_of": card.prices_as_of,
        "warnings": list(card.warnings()),
        "candidates": [_candidate(card, row) for row in card.rows],
        "nodes": [
            {
                "candidate_id": node.candidate_id,
                "node": node.node_name,
                "provider": node.provider,
                "model": node.model,
                "calls": node.calls,
                "errors": node.errors,
                "schema_valid_rate": node.schema_valid_rate,
                "repair_attempts": node.repair_attempts,
                "mean_latency_ms": node.mean_latency_ms,
                "prompt_tokens": node.prompt_tokens,
                "completion_tokens": node.completion_tokens,
            }
            for node in card.nodes
        ],
        "blame": [
            {
                "candidate_id": row.candidate_id,
                "node": row.node_name,
                "kind": row.kind,
                "error": row.error_class,
                "count": row.count,
            }
            for row in card.blame
        ],
        "incidents": [
            {
                "candidate_id": i.candidate_id,
                "case_id": i.case_id,
                "tools": list(i.tools),
                "exec_id": i.exec_id,
            }
            for i in card.incidents
        ],
    }


def _candidate(card: Scorecard, row: CandidateReport) -> dict[str, Any]:
    return {
        "candidate_id": row.candidate_id,
        "role": row.role,
        "provider": row.provider,
        "model": row.model,
        "cases": row.stats.cases,
        "cases_all_pass": row.stats.cases_all_pass,
        "executions": row.stats.executions,
        "passes": row.stats.passes,
        "excluded": row.stats.excluded,
        f"pass_hat_{card.repeats}": row.pass_hat_k,
        "pass_hat_k": row.pass_hat_k,
        "pass_at_k_mean": row.stats.pass_at_k_mean,
        "wilson_lo": row.stats.lower,
        "wilson_hi": row.stats.upper,
        "run_rate": row.stats.run_rate,
        "schema_calls": row.schema_calls,
        "schema_valid_rate": row.schema_valid_rate,
        "schema_repaired_rate": row.schema_repaired_rate,
        "tool_precision": row.tool_precision,
        "tool_recall": row.tool_recall,
        "tool_f1": row.tool_f1,
        "trajectory_distance": row.trajectory_distance,
        "tool_redundancy": row.redundancy,
        "mean_steps": row.mean_steps,
        "loops": row.loops,
        # Explicitly null in throughput mode. A consumer that charted these
        # would otherwise publish our scheduler's queueing as the model's
        # latency.
        "p50_ms": row.p50_ms if card.latency_reportable else None,
        "p95_ms": row.p95_ms if card.latency_reportable else None,
        "cost_usd_total": row.cost_total,
        "cost_usd_per_case": row.cost_per_case,
        "cost_delta": card.delta_cost(row),
        "tokens_estimated": row.tokens_estimated,
        "unpriced_models": list(row.unpriced_models),
        "unsafe_actions": row.unsafe_actions,
        "harness_errors": row.harness_errors,
        "uncovered_cases": list(row.uncovered_cases),
        "faults_met": row.faults_met,
        "recovery_rate": row.recovery_rate,
        "structured_modes": dict(row.structured_modes),
        "json_stages": dict(row.json_stages),
        "outcomes": {k.value: v for k, v in sorted(row.outcomes.items())},
        "cases_detail": [
            {
                "case_id": case_id,
                "n": stats.n,
                "c": stats.c,
                "pass_k": stats.pass_k,
                "pass_hat_k": stats.pass_hat_k,
                "wilson_lo": stats.lower,
                "wilson_hi": stats.upper,
                "excluded": stats.excluded,
            }
            for case_id, stats in sorted(row.rollups.items())
        ],
    }


def to_json(card: Scorecard, *, indent: int = 2) -> str:
    """Pretty-printed by default: this gets committed and diffed."""
    import json

    return json.dumps(to_payload(card), indent=indent, sort_keys=False, default=str)


def to_markdown(card: Scorecard) -> str:
    """A table for a pull-request comment.

    Warnings go above it in bold, same as in the terminal. A forbidden tool
    call must be the first thing a reviewer sees, not something they find by
    widening a column.
    """
    run = card.run
    lines: list[str] = [
        f"### {run.name}",
        "",
        f"`{run.run_id}` · {len(card.cases)} cases x {card.repeats} runs · "
        f"{run.mode} · seed {run.seed} · dataset `{run.dataset_hash}`",
        "",
    ]
    notes = card.warnings()
    if notes:
        lines += [f"- **{note}**" for note in notes] + [""]

    headers = [
        "Candidate",
        f"pass^{card.repeats}",
        "95% CI",
        "runs",
        "Schema",
        "Tool F1",
        "Steps",
    ]
    if card.latency_reportable:
        headers.append("p95 ms")
    headers += ["$/case", "ΔCost"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")

    for row in card.rows:
        cells = [
            f"{'**' if row.baseline else ''}{row.candidate_id}"
            f"{'**' if row.baseline else ''}",
            _pct(row.pass_hat_k),
            row.stats.interval.render() if row.stats.interval else "—",
            f"{row.stats.passes}/{row.stats.executions}" if row.stats.executions else "—",
            _pct(row.schema_valid_rate),
            f"{row.tool_f1:.2f}" if row.tool_f1 is not None else "—",
            f"{row.mean_steps:.1f}" if row.mean_steps is not None else "—",
        ]
        if card.latency_reportable:
            cells.append(f"{row.p95_ms:,.0f}" if row.p95_ms is not None else "—")
        cells.append(
            f"${row.cost_per_case:.4f}" if row.cost_per_case is not None else "—"
        )
        delta = card.delta_cost(row)
        cells.append(f"{delta:+.0%}" if delta is not None else "—")
        lines.append("| " + " | ".join(cells) + " |")

    lines += [
        "",
        f"`pass^{card.repeats}` is the fraction of cases passing every repeat; "
        "the interval is the Wilson 95% bound on it. Gate on the lower bound, "
        "not the point estimate.",
        f"Prices as of {card.prices_as_of}.",
    ]
    return "\n".join(lines)


CSV_COLUMNS: Final = (
    "run_id",
    "candidate_id",
    "role",
    "provider",
    "model",
    "metric",
    "value",
    "n",
)


def to_csv(card: Scorecard) -> str:
    """Long and narrow: one row per candidate per metric.

    Narrow rather than wide so adding a metric in week 3 appends rows instead
    of changing every consumer's column layout. ``n`` rides on every row,
    because a rate without its sample size is not a number anyone should put in
    a spreadsheet.
    """
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for row in card.rows:
        for metric, value, n in _csv_metrics(card, row):
            writer.writerow(
                [
                    card.run.run_id,
                    row.candidate_id,
                    row.role,
                    row.provider,
                    row.model,
                    metric,
                    "" if value is None else value,
                    n,
                ]
            )
    return buffer.getvalue()


def _csv_metrics(
    card: Scorecard, row: CandidateReport
) -> Iterable[tuple[str, float | None, int]]:
    cases = row.stats.cases
    executions = row.stats.executions
    yield (f"pass_hat_{card.repeats}", row.pass_hat_k, cases)
    yield ("wilson_lo", row.stats.lower, cases)
    yield ("wilson_hi", row.stats.upper, cases)
    yield ("run_rate", row.stats.run_rate, executions)
    yield ("schema_valid_rate", row.schema_valid_rate, row.schema_calls)
    yield ("schema_repaired_rate", row.schema_repaired_rate, row.schema_calls)
    yield ("tool_precision", row.tool_precision, executions)
    yield ("tool_recall", row.tool_recall, executions)
    yield ("tool_f1", row.tool_f1, executions)
    yield ("trajectory_distance", row.trajectory_distance, executions)
    yield ("mean_steps", row.mean_steps, executions)
    yield ("unsafe_actions", float(row.unsafe_actions), executions)
    yield ("harness_errors", float(row.harness_errors), executions)
    yield ("recovery_rate", row.recovery_rate, row.faults_met)
    yield ("cost_usd_per_case", row.cost_per_case, row.cost_cases)
    if card.latency_reportable:
        yield ("p50_ms", row.p50_ms, executions)
        yield ("p95_ms", row.p95_ms, executions)


def to_junit(card: Scorecard) -> str:
    """JUnit XML: one test case per ``(candidate, case)``.

    This is the format that gets slmeval into a team's existing dashboard, so
    the mapping onto JUnit's three states is a judgement worth stating:

    ``failure``
        the case did not pass every repeat. The message carries ``c/n`` and the
        Wilson interval, because "2/3" and "2/3 [21%, 94%]" lead to different
        decisions and only one of them is honest.
    ``skipped``
        nothing judgeable was measured - every repeat was a harness error or an
        uncovered path. Neither green nor red is exactly right, and JUnit's
        vocabulary already has the word for it. A dashboard that showed our own
        recording gap as a model failure would be the tool inventing a
        regression.
    passing
        every judged repeat passed.

    The baseline gets its own suite rather than being dropped: a dashboard
    whose history shows the frontier model failing a case is how a team finds
    out the *case* is wrong, not the candidate.

    No ``time`` attribute. Per-case wall time is only meaningful on a
    ``--serial-latency`` run, and a dashboard that plotted concurrency noise as
    a latency trend would be worse than a dashboard with no trend at all.
    """
    run = card.run
    suites = ElementTree.Element("testsuites", {"name": f"slmeval.{run.name}"})
    totals = {"tests": 0, "failures": 0, "skipped": 0}

    for row in card.rows:
        cases = [(case_id, row.rollups.get(case_id)) for case_id in card.cases]
        failures = [c for _, c in cases if c is not None and c.n and not c.all_passed]
        skipped = [case_id for case_id, c in cases if c is None or not c.n]
        suite = ElementTree.SubElement(
            suites,
            "testsuite",
            {
                "name": f"slmeval.{run.name}.{row.candidate_id}",
                "tests": str(len(cases)),
                "failures": str(len(failures)),
                "skipped": str(len(skipped)),
                "errors": "0",
            },
        )
        properties = ElementTree.SubElement(suite, "properties")
        for name, value in (
            ("run_id", run.run_id),
            ("role", row.role),
            ("provider", row.provider),
            ("model", row.model),
            ("runs_per_case", str(card.repeats)),
            ("dataset_hash", run.dataset_hash),
            ("config_hash", run.config_hash),
            ("slmeval_version", run.slmeval_version),
        ):
            ElementTree.SubElement(properties, "property", {"name": name, "value": value})

        for case_id, stats in cases:
            case = ElementTree.SubElement(
                suite,
                "testcase",
                {"classname": f"slmeval.{run.name}.{row.candidate_id}", "name": case_id},
            )
            totals["tests"] += 1
            if stats is None or not stats.n:
                reason = (
                    "no judgeable repeat: every run was a harness error or took a "
                    "path the cassettes do not cover (`slmeval record --fill-gaps`)"
                )
                ElementTree.SubElement(case, "skipped", {"message": reason})
                totals["skipped"] += 1
                continue
            if stats.all_passed:
                continue
            interval = f" {stats.interval.render()}" if stats.interval else ""
            message = (
                f"{stats.c}/{stats.n} repeat(s) passed{interval}"
                + (f", {stats.excluded} not judgeable" if stats.excluded else "")
            )
            ElementTree.SubElement(
                case, "failure", {"message": message, "type": "pass_hat_k"}
            ).text = message
            totals["failures"] += 1

    suites.set("tests", str(totals["tests"]))
    suites.set("failures", str(totals["failures"]))
    suites.set("skipped", str(totals["skipped"]))
    return (
        XML_HEADER + ElementTree.tostring(suites, encoding="unicode") + "\n"
    )


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "—"


FORMATS: Final = ("table", "json", "md", "csv", "junit")


def export(card: Scorecard, fmt: str) -> str:
    """Render a scorecard in any non-terminal format."""
    if fmt == "json":
        return to_json(card)
    if fmt == "md":
        return to_markdown(card)
    if fmt == "csv":
        return to_csv(card)
    if fmt == "junit":
        return to_junit(card)
    raise ValueError(f"unknown export format {fmt!r}; expected one of {', '.join(FORMATS)}")


__all__ = [
    "CSV_COLUMNS",
    "FORMATS",
    "SCHEMA_VERSION",
    "export",
    "to_csv",
    "to_json",
    "to_junit",
    "to_markdown",
    "to_payload",
]
