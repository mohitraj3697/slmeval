"""Commands that are planned but not built yet.

They appear in `--help` with the step that delivers them, and they accept
whatever arguments you throw at them so a scripted invocation gets a useful
message instead of a Click usage error.
"""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.render import console, symbols

PLANNED: list[tuple[str, str, str]] = [
    ("record", "Execute real tools once and cache them for replay.", "step 8, week 1"),
    ("run", "Run the candidate matrix and print the scorecard.", "step 10, week 2"),
    ("report", "Re-render the scorecard from a stored run.", "step 13, week 2"),
    ("compare", "Put two runs side by side.", "step 13, week 2"),
    ("show", "Drill into a run, execution or case.", "step 13, week 3"),
    ("gate", "Fail the build when a candidate regresses.", "step 14, week 3"),
    ("cassette", "Inspect, verify and prune recorded tool responses.", "step 8, week 1"),
    ("cache", "Trace store statistics and cleanup.", "step 12, week 2"),
]

_CONTEXT = {"ignore_unknown_options": True, "allow_extra_args": True}


def _make(name: str, summary: str, when: str) -> Callable[..., None]:
    def command(ctx: typer.Context) -> None:
        out = console(stderr=True)
        sym = symbols()
        out.print()
        out.print(Text(f"  {sym.skip} `slmeval {name}` is not implemented yet.", style="yellow"))
        out.print(Text(f"    {summary}", style="dim"))
        out.print(Text(f"    Lands in {when}.", style="dim"))
        out.print()
        out.print(
            Text(
                "  Available now: init, doctor, validate, dataset validate",
                style="dim",
            )
        )
        out.print()
        raise typer.Exit(exit_codes.USAGE)

    command.__name__ = name
    command.__doc__ = summary
    return command


def register(app: typer.Typer) -> None:
    for name, summary, when in PLANNED:
        app.command(
            name=name,
            help=f"{summary}  [dim](not yet: {when})[/dim]",
            context_settings=_CONTEXT,
        )(_make(name, summary, when))
