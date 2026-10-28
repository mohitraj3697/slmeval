"""`slmeval dataset` — work with golden case files."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.dataset import load_dataset

dataset_app = typer.Typer(
    no_args_is_help=True,
    help="Inspect and validate golden datasets.",
)


@dataset_app.command("validate")
@handle_errors
def validate_dataset(
    path: Annotated[Path, typer.Argument(help="Path to a .jsonl dataset.")],
) -> None:
    """Check a dataset file: JSON, schema, unique ids, compilable regexes."""
    out = console()
    sym = symbols()
    dataset = load_dataset(path)

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("file", str(dataset.path))
    table.add_row("cases", str(len(dataset)))
    table.add_row("hash", dataset.content_hash)
    if dataset.tags:
        table.add_row("tags", ", ".join(dataset.tags))

    safety = sum(1 for c in dataset.cases if c.assertions.forbidden_tools)
    table.add_row("safety cases", f"{safety} declare forbidden_tools")

    out.print()
    out.print(table)
    out.print()
    out.print(Text(f"  {sym.ok} valid", style="bold green"))
    out.print()


def register(app: typer.Typer) -> None:
    app.add_typer(dataset_app, name="dataset")
