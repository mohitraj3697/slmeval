"""``slmeval record`` - execute the real tools once, so every later run is free.

The matrix runner does the work; this is a narrow front door onto it with three
settings fixed, and each of them prevents a specific way recording goes wrong.

**The baseline only.** Cassettes are what candidates replay, so they have to be
recorded by the model you trust. Letting a 4B candidate record them would bake
its mistakes into the reference every other candidate is measured against.

**One repeat per case.** Cassettes are scoped per case, not per repeat - every
repeat of a case must replay the same tool responses, or the repeats measure
different things. Recording three times would just overwrite the same rows
twice while tripling the calls to someone's real API.

**Serially.** Two executions recording the same case into one store at once is
a write race, and the loser's payloads simply are not there later - surfacing
as a cassette miss, which reads as a model failure.

``--fill-gaps`` switches to ``auto``: replay what is recorded, record what is
not. That is what to run after adding a case, and it is the one mode where an
existing recording is never touched.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.loading import open_project
from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.core.runner import MatrixRunner, RunOptions
from slmeval.eval.pipeline import evaluator_for
from slmeval.report.progress import ProgressObserver
from slmeval.storage.models import Outcome


@handle_errors
def record(
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
    ] = None,
    case: Annotated[
        list[str] | None,
        typer.Option("--case", help="Only these case ids."),
    ] = None,
    fill_gaps: Annotated[
        bool,
        typer.Option(
            "--fill-gaps",
            help="Replay what is already recorded and record only what is missing. "
            "Run this after adding a case.",
        ),
    ] = False,
    ascii_mode: Annotated[
        bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
    ] = False,
    quiet: Annotated[bool, typer.Option("--quiet", "-q", help="No progress bars.")] = False,
) -> None:
    """Execute real tools once and cache them for replay."""
    out = console()
    sym = symbols(ascii_mode or None)

    with open_project(config) as project:
        options = RunOptions.from_config(
            project.config,
            repeats=1,
            tool_mode="auto" if fill_gaps else "record",
            # Recording a faulted payload would poison every later replay of
            # that call, and the poison would be indistinguishable from a real
            # recording. The sandbox refuses to inject while recording; this
            # makes the intent explicit at the front door too.
            fault_rate=0.0,
            candidate_ids=(project.config.baseline.id,),
            case_ids=tuple(case) if case else None,
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
            name=f"{project.config.benchmark.name} (record)",
        )
        observer = ProgressObserver(console=out, ascii_mode=ascii_mode or None, quiet=quiet)
        runner.observer = observer
        try:
            summary = asyncio.run(runner.run())
        except KeyboardInterrupt:
            observer.close()
            out.print()
            out.print(
                Text(
                    f"  {sym.warn} interrupted - cassettes recorded so far are kept",
                    style="yellow",
                )
            )
            out.print()
            raise typer.Exit(exit_codes.USAGE) from None

        stats = project.cassettes.stats()
        failed = [r for r in summary.results if r.outcome is not Outcome.PASSED]
        out.print(
            Text(
                f"  {sym.ok} recorded {stats.cassettes} cassette(s) across "
                f"{len(summary.results)} execution(s)",
                style="bold green",
            )
        )
        out.print(
            Text(
                "    every later run can now use --tool-mode replay: no network, "
                "no API keys, no GPU",
                style="dim",
            )
        )
        if failed:
            # A case that could not be recorded has no coverage, so every
            # candidate will miss on it and read as a model failure. Exit 4 -
            # insufficient coverage - rather than letting that happen quietly.
            out.print()
            for result in failed:
                out.print(
                    Text(
                        f"  {sym.fail} {result.unit.case_id}: {result.outcome.value}"
                        + (f" - {result.error_message}" if result.error_message else ""),
                        style="red",
                    )
                )
            out.print()
            out.print(
                Text(
                    f"  {len(failed)} case(s) were not recorded, so candidates will "
                    "miss on them. Fix those before benchmarking.",
                    style="yellow",
                )
            )
            out.print()
            raise typer.Exit(exit_codes.COVERAGE)
        out.print()


def register(app: typer.Typer) -> None:
    app.command(name="record", help="Execute real tools once and cache them for replay.")(record)
