"""``slmeval run`` - execute the matrix and print the scorecard.

This is the command the whole tool exists for, so the flag surface is wide. The
defaults are the important part: ``replay`` tool mode, throughput scheduling,
no spend ceiling, and resume available by run id. A first run needs none of
them.

Two behaviours worth calling out because they are easy to get wrong:

**Rollups are written after the matrix, not during it.** ``pass^k`` and the
Wilson interval are properties of a case's whole set of repeats, and computing
them incrementally would mean storing a number that was briefly wrong - which
somebody would eventually read.

**A ``Ctrl-C`` is not an error.** It prints what was finished, names the run id
to resume, and exits. Treating an interrupt as a crash would throw away the
most valuable thing the user has at that moment: the id of the half-finished
run.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.loading import Project, open_project
from slmeval.cli.render import Symbols, console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.core.runner import MatrixRunner, RunOptions, RunSummary, encode_plan
from slmeval.eval.pipeline import case_rollups, evaluator_for
from slmeval.report.export import export
from slmeval.report.model import build_scorecard
from slmeval.report.progress import ProgressObserver
from slmeval.report.scorecard import render_scorecard, render_summary_line
from slmeval.storage.models import Outcome, RunStatus

_TOOL_MODES = ("replay", "auto", "record", "live")


@handle_errors
def run(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
    ] = None,
    candidate: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate",
            help="Only these candidates. The baseline is always included - every "
            "delta is measured against it.",
        ),
    ] = None,
    case: Annotated[
        list[str] | None,
        typer.Option("--case", help="Only these case ids."),
    ] = None,
    runs: Annotated[
        int | None,
        typer.Option("--runs", min=1, max=50, help="Repeats per case. Overrides runs_per_case."),
    ] = None,
    tool_mode: Annotated[
        str | None,
        typer.Option(
            "--tool-mode",
            help="replay | auto | record | live. replay needs no network and no keys.",
        ),
    ] = None,
    fault_rate: Annotated[
        float | None,
        typer.Option("--fault-rate", min=0.0, max=1.0, help="Chance any tool call is faulted."),
    ] = None,
    concurrency: Annotated[
        int | None,
        typer.Option("--concurrency", min=1, max=256, help="Executions in flight, at most."),
    ] = None,
    serial_latency: Annotated[
        bool,
        typer.Option(
            "--serial-latency",
            help="One request at a time per endpoint, after a warm-up, so latency "
            "is reportable. Much slower, and the only mode whose p95 means anything.",
        ),
    ] = False,
    max_spend: Annotated[
        float | None,
        typer.Option("--max-spend", min=0.0, help="Abort once estimated spend passes this (USD)."),
    ] = None,
    resume: Annotated[
        str | None,
        typer.Option("--resume", help="Finish an interrupted run. Takes a run id or a prefix."),
    ] = None,
    fail_fast: Annotated[
        bool,
        typer.Option("--fail-fast", help="Stop once a candidate fails three executions in a row."),
    ] = False,
    gpu_hourly: Annotated[
        float | None,
        typer.Option(
            "--gpu-hourly",
            min=0.0,
            help="Your $/hour for GPU time, used to model local cost. Reported as "
            "modelled, never as measured.",
        ),
    ] = None,
    json_out: Annotated[
        Path | None,
        typer.Option("--json", help="Also write the versioned JSON report here."),
    ] = None,
    nodes: Annotated[
        bool, typer.Option("--nodes/--no-nodes", help="Show the per-node attribution table.")
    ] = True,
    ascii_mode: Annotated[
        bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Print the plan as JSON and execute nothing.")
    ] = False,
    quiet: Annotated[
        bool, typer.Option("--quiet", "-q", help="No progress bars; one line per failure.")
    ] = False,
) -> None:
    """Run the candidate matrix and print the scorecard."""
    if tool_mode is not None and tool_mode not in _TOOL_MODES:
        raise typer.BadParameter(f"tool mode must be one of: {', '.join(_TOOL_MODES)}")

    out = console()
    sym = symbols(ascii_mode or None)

    with open_project(config, gpu_hourly_usd=gpu_hourly) as project:
        options = RunOptions.from_config(
            project.config,
            repeats=runs,
            tool_mode=tool_mode,
            fault_rate=fault_rate,
            concurrency=concurrency,
            serial_latency=serial_latency or None,
            max_spend_usd=max_spend,
            fail_fast=fail_fast or None,
            case_ids=tuple(case) if case else None,
            candidate_ids=tuple(candidate) if candidate else None,
            gpu_hourly_usd=gpu_hourly,
        )
        runner = MatrixRunner(
            config=project.config,
            dataset=project.dataset,
            target=project.target,
            store=project.store,
            options=options,
            cassettes=project.cassettes,
            evaluator=evaluator_for(project.config),
            prices=project.prices,
            root=project.root,
        )

        if dry_run:
            # print, not rich-print: this is JSON on its way into a pipe, and
            # Rich would wrap it at the terminal width into something no parser
            # can read.
            print(encode_plan(runner.plan(resume=resume)))
            return

        observer = ProgressObserver(console=out, ascii_mode=ascii_mode or None, quiet=quiet)
        runner.observer = observer
        try:
            summary = asyncio.run(runner.run(resume=resume))
        except KeyboardInterrupt:
            # Everything finished is already in SQLite - that is the point of
            # persisting per unit - so the only thing at risk here is the user
            # knowing how to pick it back up.
            observer.close()
            _interrupted(out, sym, runner)
            raise typer.Exit(exit_codes.USAGE) from None

        _save_rollups(project, summary)
        card = build_scorecard(
            project.store, summary.run_id, prices_as_of=project.prices.as_of
        )
        out.print(render_scorecard(card, ascii_mode=ascii_mode or None, show_nodes=nodes))
        line = render_summary_line(card, ascii_mode=ascii_mode or None)
        if str(line).strip():
            out.print(line)
            out.print()

        if json_out is not None:
            json_out.write_text(export(card, "json"), encoding="utf-8")
            out.print(Text(f"  {sym.ok} wrote {json_out}", style="dim"))
            out.print()

        raise typer.Exit(_exit_code(summary))


def _save_rollups(project: Project, summary: RunSummary) -> None:
    """Compute and store the consistency rollups for the whole run.

    After the matrix, and from the *stored* executions rather than from the
    results in memory: a resumed run's earlier half exists only in the
    database, and rolling up what this process happened to execute would report
    ``n=1`` for every case the first half already covered.
    """
    executions = project.store.executions(summary.run_id)
    project.store.save_rollups(case_rollups(summary.run_id, executions))


def _interrupted(out: Console, sym: Symbols, runner: MatrixRunner) -> None:
    out.print()
    out.print(
        Text(
            f"  {sym.warn} interrupted - everything finished is already stored",
            style="yellow",
        )
    )
    if runner.last_run_id:
        out.print(
            Text(
                f"    resume with: slmeval run --resume {runner.last_run_id}",
                style="cyan",
            )
        )
    out.print()


def _exit_code(summary: RunSummary) -> int:
    """What the shell sees.

    An unreachable provider exits 3, not 1. That distinction is the whole
    reason the codes are a documented contract: a CI job that reports "Ollama
    was not running" as a model regression teaches people to ignore the gate,
    and once they do, the gate is worth nothing.
    """
    if summary.results and all(
        r.outcome is Outcome.PROVIDER_ERROR for r in summary.results
    ):
        return exit_codes.ENVIRONMENT
    if summary.status is RunStatus.CANCELLED:
        return exit_codes.GATE_FAILED
    return exit_codes.OK


def register(app: typer.Typer) -> None:
    app.command(name="run", help="Run the candidate matrix and print the scorecard.")(run)
