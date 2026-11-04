"""Commands that are planned but not built yet.

They are hidden from `--help`, so nobody is offered a command that does not work.
They still accept whatever arguments you throw at them, so a scripted invocation
gets a useful message instead of a Click usage error.
"""

from __future__ import annotations

from collections.abc import Callable

import typer
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.render import console, symbols

PLANNED: list[tuple[str, str]] = [
    ("compare", "Put two runs side by side."),
    ("cassette", "Inspect, verify and prune recorded tool responses."),
]

_CONTEXT = {"ignore_unknown_options": True, "allow_extra_args": True}


def _make(name: str, summary: str) -> Callable[..., None]:
    def command(ctx: typer.Context) -> None:
        out = console(stderr=True)
        sym = symbols()
        out.print()
        out.print(Text(f"  {sym.skip} `slmeval {name}` is not implemented yet.", style="yellow"))
        out.print(Text(f"    {summary}", style="dim"))
        out.print(Text("    It is not part of this release.", style="dim"))
        out.print()
        out.print(
            Text(
                "  Available now: init, doctor, validate, dataset validate, "
                "record, run, report, show, gate, baseline save, cache stats",
                style="dim",
            )
        )
        out.print()
        raise typer.Exit(exit_codes.USAGE)

    command.__name__ = name
    command.__doc__ = summary
    return command


def register(app: typer.Typer) -> None:
    for name, summary in PLANNED:
        app.command(
            name=name,
            help=summary,
            hidden=True,
            context_settings=_CONTEXT,
        )(_make(name, summary))
