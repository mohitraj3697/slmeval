"""CLI entrypoint.

Commands are registered in the order they should appear in `--help`. The planned
ones come last and stay hidden from it.
"""

from __future__ import annotations

from typing import Annotated

import typer

from slmeval import __version__
from slmeval.cli.commands import (
    baseline_cmd,
    cache_cmd,
    dataset_cmd,
    doctor_cmd,
    gate_cmd,
    init_cmd,
    judge_cmd,
    record_cmd,
    report_cmd,
    run_cmd,
    show_cmd,
    stubs,
    validate_cmd,
)
from slmeval.cli.render import console

app = typer.Typer(
    name="slmeval",
    no_args_is_help=True,
    add_completion=True,
    rich_markup_mode="rich",
    # Locals can hold a resolved API key, so they never go into a traceback.
    pretty_exceptions_show_locals=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        console().print(f"slmeval {__version__}")
        raise typer.Exit()


@app.callback()
def cli(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            "-V",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Benchmark agent workflows across small language models.

    Record your agent's tool calls once, replay them deterministically against
    every candidate model, and see what each swap costs you in reliability.

    Start with [cyan]slmeval init[/cyan], then [cyan]slmeval doctor[/cyan].
    """


init_cmd.register(app)
doctor_cmd.register(app)
validate_cmd.register(app)
dataset_cmd.register(app)
record_cmd.register(app)
run_cmd.register(app)
report_cmd.register(app)
show_cmd.register(app)
gate_cmd.register(app)
baseline_cmd.register(app)
judge_cmd.register(app)
cache_cmd.register(app)
stubs.register(app)


def main() -> None:
    """Console-script entrypoint."""
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
