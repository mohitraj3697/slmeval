"""``slmeval report`` - re-render a stored run, with nothing re-executed.

The acceptance criterion for this command is narrow and worth stating: the
table it prints for a finished run has to be *identical* to the one that run
printed when it finished. Nothing is recomputed from live objects, nothing is
re-estimated, and nothing is re-asked of a model. If that ever stops being
true, the trace store has stopped being the asset and the tool has become a
thing you have to re-run to look at twice.

It deliberately does not need the user's agent to import. A checkout whose
agent has since been refactored still has valid results from last Tuesday, and
refusing to print them would be the tool losing data it already holds.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from slmeval.cli.loading import open_project
from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.errors import ConfigError
from slmeval.report.export import FORMATS, export
from slmeval.report.model import build_scorecard
from slmeval.report.scorecard import (
    render_rollups,
    render_scorecard,
    render_summary_line,
)
from slmeval.storage.db import RunStore

LATEST = "latest"


@handle_errors
def report(
    run_id: Annotated[
        str,
        typer.Option("--run-id", help="A run id or prefix, or 'latest'."),
    ] = LATEST,
    fmt: Annotated[
        str,
        typer.Option("--format", "-f", help=f"One of: {', '.join(FORMATS)}."),
    ] = "table",
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
    ] = None,
    nodes: Annotated[
        bool, typer.Option("--nodes/--no-nodes", help="Show the per-node attribution table.")
    ] = True,
    cases: Annotated[
        bool, typer.Option("--cases", help="Also show the per-case pass table.")
    ] = False,
    out_path: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Write to this file instead of stdout."),
    ] = None,
    ascii_mode: Annotated[
        bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
    ] = False,
) -> None:
    """Re-render the scorecard from a stored run."""
    if fmt not in FORMATS:
        raise typer.BadParameter(f"format must be one of: {', '.join(FORMATS)}")

    out = console()
    sym = symbols(ascii_mode or None)

    with open_project(config, load_agent=False) as project:
        resolved = _resolve(project.store, run_id)
        card = build_scorecard(project.store, resolved, prices_as_of=project.prices.as_of)

        if fmt != "table":
            text = export(card, fmt)
            if out_path is not None:
                out_path.write_text(text, encoding="utf-8")
                out.print(Text(f"  {sym.ok} wrote {out_path}", style="dim"))
                return
            # print, not rich-print: a JSON or CSV payload is data on its way
            # into a pipe, and Rich would wrap and style it into something no
            # parser can read.
            print(text)
            return

        out.print(render_scorecard(card, ascii_mode=ascii_mode or None, show_nodes=nodes))
        if cases:
            out.print()
            out.print(render_rollups(card, ascii_mode=ascii_mode or None))
        line = render_summary_line(card, ascii_mode=ascii_mode or None)
        if str(line).strip():
            out.print()
            out.print(line)
        out.print()


def _resolve(store: RunStore, run_id: str) -> str:
    if run_id == LATEST:
        latest = store.latest_run()
        if latest is None:
            raise ConfigError(
                "no runs stored yet",
                hint="run `slmeval run` first, then `slmeval report`",
            )
        return latest.run_id
    return store.resolve_run_id(run_id)


def register(app: typer.Typer) -> None:
    app.command(name="report", help="Re-render the scorecard from a stored run.")(report)
