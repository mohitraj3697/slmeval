"""``slmeval gate`` - the command that blocks a merge.

The exit code is the product here. Everything printed is for the human reading
the CI log afterwards; the number is what the pipeline branches on, and it is a
documented contract (:mod:`slmeval.exit_codes`), frozen from 0.1.0:

====  ==========================================================
``0`` every threshold met
``1`` a threshold was not met. A real quality finding.
``2`` bad invocation - no candidate, no threshold, unreadable baseline
``3`` the model server was unreachable. **Infrastructure, not a regression.**
``4`` not enough recorded ground truth to judge the run
====  ==========================================================

Distinguishing 1 from 3 is the whole reason the codes exist. A job that reports
"Ollama was not running" as "the model got worse" teaches a team to ignore the
gate, and after that the gate is worth nothing.

Nothing here runs the matrix. ``gate`` reads a stored run, which means CI can
run ``slmeval run --tool-mode replay`` once and gate the result, and means a
failing gate can be re-examined locally with ``slmeval show`` without spending
another token.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as ElementTree
from pathlib import Path
from typing import Annotated, Any, Final

import typer
from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from slmeval.cli.loading import LATEST, open_project, resolve_run
from slmeval.cli.render import Symbols, console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.errors import ConfigError
from slmeval.gate import (
    Check,
    GateReport,
    Severity,
    Thresholds,
    blocking_divergences,
    evaluate,
)
from slmeval.report.baseline import Baseline, baseline_path
from slmeval.report.model import build_scorecard

FORMATS: Final = ("text", "md", "json", "junit")

_SEVERITY_STYLE: Final = {
    Severity.GATE: "bold red",
    Severity.ENVIRONMENT: "bold yellow",
    Severity.COVERAGE: "bold cyan",
}


@handle_errors
def gate(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
    ] = None,
    run_id: Annotated[
        str, typer.Option("--run-id", help="A run id or prefix, or 'latest'.")
    ] = LATEST,
    min_pass_k: Annotated[
        float | None,
        typer.Option(
            "--min-pass-k",
            min=0.0,
            max=1.0,
            help="Floor on pass^k, read on the Wilson lower bound.",
        ),
    ] = None,
    use_wilson_lower: Annotated[
        bool | None,
        typer.Option(
            "--use-wilson-lower/--use-point-estimate",
            help="Gate on the lower bound (default) or on the point estimate. "
            "Three passes out of three is not 100%.",
        ),
    ] = None,
    max_cost_ratio: Annotated[
        float | None,
        typer.Option(
            "--max-cost-ratio",
            min=0.0,
            help="Candidate cost per case as a fraction of the baseline's.",
        ),
    ] = None,
    max_p95_ratio: Annotated[
        float | None,
        typer.Option(
            "--max-p95-ratio",
            min=0.0,
            help="Candidate p95 as a multiple of the baseline's. Needs a "
            "--serial-latency run; skipped loudly otherwise.",
        ),
    ] = None,
    no_forbidden_tools: Annotated[
        bool | None,
        typer.Option(
            "--no-forbidden-tools/--allow-forbidden-tools",
            help="Fail on any call to a tool a case forbids. On by default.",
        ),
    ] = None,
    max_regression: Annotated[
        float | None,
        typer.Option(
            "--max-regression",
            min=0.0,
            max=1.0,
            help="Allowed drop in pass^k against the committed baseline file.",
        ),
    ] = None,
    max_uncovered: Annotated[
        float | None,
        typer.Option(
            "--max-uncovered",
            min=0.0,
            max=1.0,
            help="Share of executions that may take unrecorded paths before the "
            "run is called unjudgeable (exit 4).",
        ),
    ] = None,
    max_provider_errors: Annotated[
        float | None,
        typer.Option(
            "--max-provider-errors",
            min=0.0,
            max=1.0,
            help="Share of executions that may fail to reach the model server "
            "before the job is called broken (exit 3).",
        ),
    ] = None,
    require_fault_parity: Annotated[
        bool | None,
        typer.Option(
            "--require-fault-parity/--no-fault-parity",
            help="Refuse to compare candidates that met different faults.",
        ),
    ] = None,
    baseline_file: Annotated[
        Path | None,
        typer.Option(
            "--baseline",
            help="The committed baseline to compare against. Defaults to "
            "slmeval.baseline.json beside eval.toml.",
        ),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(
            "--force",
            help="Compare against the baseline file even when the dataset or "
            "config it was built on has changed.",
        ),
    ] = False,
    fmt: Annotated[
        str, typer.Option("--format", "-f", help=f"One of: {', '.join(FORMATS)}.")
    ] = "text",
    out_path: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write the report to this file as well."),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show every check, not only the failures. Worth running once "
            "when you set the thresholds up, to see what they actually enforce.",
        ),
    ] = False,
    ascii_mode: Annotated[
        bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
    ] = False,
) -> None:
    """Fail the build when a candidate regresses."""
    if fmt not in FORMATS:
        raise typer.BadParameter(f"format must be one of: {', '.join(FORMATS)}")

    out = console()
    sym = symbols(ascii_mode or None)

    with open_project(config, load_agent=False) as project:
        resolved = resolve_run(project.store, run_id)
        card = build_scorecard(project.store, resolved, prices_as_of=project.prices.as_of)
        thresholds = Thresholds.from_config(
            project.config.gate,
            min_pass_k=min_pass_k,
            use_wilson_lower=use_wilson_lower,
            max_cost_ratio=max_cost_ratio,
            max_p95_ratio=max_p95_ratio,
            no_forbidden_tools=no_forbidden_tools,
            max_regression=max_regression,
            max_uncovered=max_uncovered,
            max_provider_error_rate=max_provider_errors,
            require_fault_parity=require_fault_parity,
        )

        # Only when a regression check was actually asked for. A committed
        # baseline sitting in the repository is not a reason to refuse a gate
        # that was never going to read it - and refusing one because that file
        # predates a dataset edit would block a pass-rate check that is
        # perfectly valid.
        if baseline_file is not None and thresholds.max_regression is None:
            # They named a file to compare against and set no allowance, so
            # nothing would read it. Refusing beats running a gate that
            # silently ignores half of what was asked for.
            raise ConfigError(
                f"--baseline {baseline_file} was given with no --max-regression, "
                "so nothing would be compared against it",
                hint="add --max-regression 0.05 (or [gate].max_regression)",
            )
        stored = None
        if thresholds.max_regression is not None:
            path = baseline_path(project.root, baseline_file)
            stored = _load_baseline(path, explicit=baseline_file is not None)
            if stored is not None:
                _check_provenance(card, stored, force=force, path=path)

        report = evaluate(card, thresholds, baseline=stored)

        if fmt == "text":
            out.print(
                render_gate(report, ascii_mode=ascii_mode or None, verbose=verbose)
            )
        else:
            text = export(report, fmt)
            print(text)
        if out_path is not None:
            out_path.write_text(
                export(report, "md" if fmt == "text" else fmt), encoding="utf-8"
            )
            if fmt == "text":
                out.print(Text(f"  {sym.ok} wrote {out_path}", style="dim"))
                out.print()

        raise typer.Exit(report.exit_code)


def _load_baseline(path: Path, *, explicit: bool) -> Baseline | None:
    """The committed baseline, or ``None`` when there is none.

    A missing default file is not an error - most runs do not gate on a
    regression, and demanding a baseline before the first gate would make the
    command unusable on day one. A missing file the user *named* is an error,
    because they clearly meant to compare against something.
    """
    if not path.is_file():
        if explicit:
            raise ConfigError(
                f"no baseline file at {path}",
                hint="`slmeval baseline save` writes one from a stored run",
            )
        return None
    return Baseline.load(path)


def _check_provenance(
    card: Any, stored: Baseline, *, force: bool, path: Path
) -> None:
    """Refuse to compare a run against a baseline built on different inputs.

    This is the refusal the provenance hashes exist for. A dataset edit
    compared against an old baseline shows up as a regression nobody caused,
    and a tool that publishes one of those loses the right to be believed -
    once, permanently. ``--force`` exists for the user who knows exactly what
    they changed.
    """
    blocking = blocking_divergences(card, stored)
    if not blocking or force:
        return
    reasons = "; ".join(f"{d.reason} ({d.field})" for d in blocking)
    raise ConfigError(
        f"{path.name} was built on different inputs: {reasons}",
        source=path,
        hint=(
            "re-save it with `slmeval baseline save` if the change was intended, "
            "or pass --force to compare anyway and read the numbers knowing why "
            "they moved"
        ),
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def render_gate(
    report: GateReport, *, ascii_mode: bool | None = None, verbose: bool = False
) -> RenderableType:
    """The verdict first, then why, then what was not checked.

    Verdict first because a CI log is read from the top and usually only from
    the top. The failures that follow name the candidate and the case, because
    "pass^3 below floor" is not something anybody can act on.

    Only the failures and the skips, unless ``verbose``: a passing gate in a CI
    log should be one line, and nine green rows nobody reads are nine rows of
    scrollback between a reviewer and the thing that did fail.
    """
    sym = symbols(ascii_mode)
    parts: list[RenderableType] = [Text(""), _verdict(report, sym), Text("")]

    if report.usage_error:
        parts += [Text(f"  {report.usage_error}", style="yellow"), Text("")]
        return Group(*parts)

    if verbose:
        passing = tuple(c for c in report.checks if c.passed and not c.skipped)
        if passing:
            parts += [
                Text(f"  {sym.ok} met:", style="dim"),
                _checks_table(passing, sym, failed=False),
                Text(""),
            ]
    if report.failures:
        parts += [_checks_table(report.failures, sym, failed=True), Text("")]
    if report.skipped:
        parts += [
            Text(
                f"  {sym.skip} not checked - the figure was never measured, so "
                "these neither passed nor failed:",
                style="dim",
            ),
            _checks_table(report.skipped, sym, failed=False),
            Text(""),
        ]
    for note in report.warnings:
        parts.append(Text(f"  {sym.warn} {note}", style="yellow"))
    if report.warnings:
        parts.append(Text(""))
    parts += [_summary(report, sym), Text("")]
    return Group(*parts)


def _verdict(report: GateReport, sym: Symbols) -> Text:
    out = Text("  ")
    if report.usage_error:
        out.append(f"{sym.warn} GATE NOT RUN", style="bold yellow")
        out.append(f"  (exit {report.exit_code})", style="dim")
        return out
    if report.passed:
        out.append(f"{sym.ok} GATE PASSED", style="bold green")
    else:
        severity = report.severity
        label = {
            Severity.GATE: "GATE FAILED",
            # Named so nobody reads it as a regression. This is the line the
            # whole exit-code contract exists to make possible.
            Severity.ENVIRONMENT: "INFRASTRUCTURE PROBLEM - not a regression",
            Severity.COVERAGE: "NOT ENOUGH COVERAGE TO JUDGE - not a regression",
        }[severity] if severity is not None else "GATE FAILED"
        glyph = sym.fail if severity is Severity.GATE else sym.warn
        out.append(f"{glyph} {label}", style=_SEVERITY_STYLE.get(severity or Severity.GATE))
    run = report.card.run
    out.append(f"  (exit {report.exit_code})", style="dim")
    out.append(
        f"\n  {run.name} · {run.run_id} · {len(report.card.cases)} cases x "
        f"{report.card.repeats} runs · {run.mode} · seed {run.seed}",
        style="dim",
    )
    estimator = (
        "Wilson lower bound" if report.thresholds.use_wilson_lower else "point estimate"
    )
    out.append(f"\n  gating on the {estimator}", style="dim")
    if report.baseline is not None:
        out.append(
            f" · baseline {report.baseline.run_id} saved {report.baseline.saved_at[:10]}",
            style="dim",
        )
    return out


def _checks_table(checks: tuple[Check, ...], sym: Symbols, *, failed: bool) -> Table:
    grid = Table(box=None, pad_edge=False, padding=(0, 2), show_edge=False, show_header=False)
    grid.add_column(width=2, no_wrap=True)
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(no_wrap=True)
    grid.add_column(overflow="fold")
    for check in checks:
        if check.passed and not check.skipped:
            glyph, style = sym.ok, "green"
        elif check.skipped:
            glyph, style = sym.skip, "dim"
        else:
            glyph = sym.fail if check.severity is Severity.GATE else sym.warn
            style = _SEVERITY_STYLE.get(check.severity, "red")
        grid.add_row(
            Text(glyph, style=style),
            Text(check.candidate_id, style="dim" if check.skipped else ""),
            Text(check.name, style=style),
            Text(check.detail, style="dim" if check.skipped else ""),
        )
    return grid


def _summary(report: GateReport, sym: Symbols) -> Text:
    checked = [c for c in report.checks if not c.skipped]
    passed = [c for c in checked if c.passed]
    out = Text()
    out.append(
        f"  {len(passed)}/{len(checked)} check(s) passed"
        f"{f', {len(report.skipped)} not measured' if report.skipped else ''}\n",
        style="dim",
    )
    out.append(
        f"  {sym.arrow} exit codes: 0 pass · 1 gate failed · 2 usage · "
        "3 environment · 4 coverage",
        style="dim",
    )
    return out


# --------------------------------------------------------------------------- #
# Exports
# --------------------------------------------------------------------------- #


def to_payload(report: GateReport) -> dict[str, Any]:
    """The gate as data. Stable keys; a CI script may read any of them."""
    thresholds = report.thresholds
    return {
        "schema": "slmeval.gate/1",
        "passed": report.passed,
        "exit_code": report.exit_code,
        "severity": report.severity.value if report.severity else None,
        "usage_error": report.usage_error or None,
        "run_id": report.card.run.run_id,
        "name": report.card.run.name,
        "cases": len(report.card.cases),
        "runs_per_case": report.card.repeats,
        "baseline_run_id": report.baseline.run_id if report.baseline else None,
        "thresholds": {
            "min_pass_k": thresholds.min_pass_k,
            "use_wilson_lower": thresholds.use_wilson_lower,
            "max_cost_ratio": thresholds.max_cost_ratio,
            "max_p95_ratio": thresholds.max_p95_ratio,
            "no_forbidden_tools": thresholds.no_forbidden_tools,
            "max_regression": thresholds.max_regression,
            "max_uncovered": thresholds.max_uncovered,
            "max_provider_error_rate": thresholds.max_provider_error_rate,
            "require_fault_parity": thresholds.require_fault_parity,
        },
        "checks": [
            {
                "name": check.name,
                "candidate_id": check.candidate_id,
                "passed": check.passed,
                "skipped": check.skipped,
                "absolute": check.absolute,
                "severity": check.severity.value,
                "observed": check.observed,
                "limit": check.limit,
                "detail": check.detail,
            }
            for check in report.checks
        ],
        "warnings": list(report.warnings),
    }


def to_markdown(report: GateReport) -> str:
    """For a pull-request comment, which is where most people meet the gate."""
    # The headline word matches the exit code, because a comment that said
    # "FAILED" over an unreachable model server would be read as a regression
    # by everyone who did not get as far as the blockquote.
    verdict = {
        None: "**PASSED**",
        Severity.GATE: "**FAILED**",
        Severity.ENVIRONMENT: "**INFRASTRUCTURE PROBLEM** - not a regression",
        Severity.COVERAGE: "**NOT ENOUGH COVERAGE TO JUDGE** - not a regression",
    }[report.severity]
    if report.usage_error:
        verdict = "**GATE NOT RUN**"
    run = report.card.run
    lines = [
        f"### slmeval gate · {verdict} (exit {report.exit_code})",
        "",
        f"`{run.run_id}` · {len(report.card.cases)} cases x {report.card.repeats} "
        f"runs · {run.mode} · seed {run.seed} · dataset `{run.dataset_hash}`",
        "",
    ]
    if report.usage_error:
        return "\n".join([*lines, report.usage_error, ""])
    if report.severity is Severity.ENVIRONMENT:
        lines += [
            "> This is an infrastructure failure, **not a model regression** - "
            "the model server was not reachable.",
            "",
        ]
    if report.severity is Severity.COVERAGE:
        lines += [
            "> Not enough recorded ground truth to judge this run. Run "
            "`slmeval record --fill-gaps`; this is **not a model regression**.",
            "",
        ]
    lines += ["| | Candidate | Check | Detail |", "|---|---|---|---|"]
    for check in report.checks:
        mark = "-" if check.skipped else ("✅" if check.passed else "❌")
        lines.append(
            f"| {mark} | `{check.candidate_id}` | {check.name} | {check.detail} |"
        )
    estimator = (
        "the Wilson lower bound" if report.thresholds.use_wilson_lower else "the point estimate"
    )
    lines += [
        "",
        f"Gated on {estimator} of `pass^{report.card.repeats}`. "
        "A rate without its interval is a number pretending to be evidence.",
    ]
    for note in report.warnings:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def to_junit(report: GateReport) -> str:
    """JUnit XML, so an existing CI dashboard picks up the checks for free.

    One test case per check, named ``candidate.check``. A skipped check is a
    JUnit ``skipped``, which is exactly the right vocabulary: the dashboard
    shows it as neither green nor red, which is what it is.
    """
    checks = report.checks
    suite = ElementTree.Element(
        "testsuite",
        {
            "name": f"slmeval.gate.{report.card.run.name}",
            "tests": str(len(checks)),
            "failures": str(len(report.failures)),
            "skipped": str(len(report.skipped)),
            "errors": "0",
        },
    )
    if report.usage_error:
        case = ElementTree.SubElement(
            suite, "testcase", {"classname": "slmeval.gate", "name": "configured"}
        )
        ElementTree.SubElement(
            case, "failure", {"message": report.usage_error, "type": "usage"}
        ).text = report.usage_error
    for check in checks:
        case = ElementTree.SubElement(
            suite,
            "testcase",
            {"classname": f"slmeval.gate.{check.candidate_id}", "name": check.name},
        )
        if check.skipped:
            ElementTree.SubElement(case, "skipped", {"message": check.detail})
        elif not check.passed:
            ElementTree.SubElement(
                case,
                "failure",
                {"message": check.detail, "type": check.severity.value},
            ).text = check.detail
    root = ElementTree.Element(
        "testsuites",
        {
            "name": "slmeval",
            "tests": str(len(checks)),
            "failures": str(len(report.failures)),
        },
    )
    root.append(suite)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        + ElementTree.tostring(root, encoding="unicode")
        + "\n"
    )


def export(report: GateReport, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(to_payload(report), indent=2, default=str)
    if fmt == "md":
        return to_markdown(report)
    if fmt == "junit":
        return to_junit(report)
    raise ValueError(f"unknown gate format {fmt!r}; expected one of {', '.join(FORMATS)}")


def register(app: typer.Typer) -> None:
    app.command(name="gate", help="Fail the build when a candidate regresses.")(gate)


__all__ = [
    "FORMATS",
    "export",
    "gate",
    "register",
    "render_gate",
    "to_junit",
    "to_markdown",
    "to_payload",
]
