"""`slmeval init` — scaffold a benchmark in the current project."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.config import templates
from slmeval.errors import SlmevalError

Outcome = Literal["created", "skipped", "appended"]


@dataclass
class Written:
    path: Path
    outcome: Outcome
    note: str = ""


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "agent"


def default_benchmark_name(directory: Path) -> str:
    return f"{slugify(directory.resolve().name)}-slm-downsizing"


def scaffold(directory: Path, *, name: str | None = None, force: bool = False) -> list[Written]:
    """Write the starter files. Never overwrites without ``force``."""
    directory = Path(directory)
    if directory.exists() and not directory.is_dir():
        raise SlmevalError(f"{directory} exists and is not a directory")
    directory.mkdir(parents=True, exist_ok=True)

    benchmark_name = name or default_benchmark_name(directory)
    plan: list[tuple[Path, str]] = [
        (directory / "eval.toml", templates.eval_toml(benchmark_name)),
        (directory / "datasets" / "golden_cases.jsonl", templates.DATASET_JSONL),
        (directory / "slmeval_target.py", templates.TARGET_PY),
    ]

    written: list[Written] = []
    for path, content in plan:
        if path.exists() and not force:
            written.append(Written(path, "skipped", "already exists; --force overwrites"))
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        # newline="\n" so the file hashes identically on Windows and in CI.
        path.write_text(content, encoding="utf-8", newline="\n")
        written.append(Written(path, "created"))

    written.append(_ensure_gitignore(directory))
    return written


def _ensure_gitignore(directory: Path) -> Written:
    """Keep the trace cache out of git without clobbering an existing file."""
    path = directory / ".gitignore"
    entry = templates.GITIGNORE_ENTRY
    if not path.exists():
        path.write_text(entry + "\n", encoding="utf-8", newline="\n")
        return Written(path, "created")

    existing = path.read_text(encoding="utf-8-sig").splitlines()
    if any(line.strip().rstrip("/") == entry.rstrip("/") for line in existing):
        return Written(path, "skipped", f"already ignores {entry}")

    with path.open("a", encoding="utf-8", newline="\n") as fh:
        if existing and existing[-1].strip():
            fh.write("\n")
        fh.write(f"# slmeval trace cache\n{entry}\n")
    return Written(path, "appended", f"added {entry}")


def register(app: typer.Typer) -> None:
    @app.command()
    @handle_errors
    def init(
        directory: Annotated[
            Path,
            typer.Argument(help="Where to scaffold. Defaults to the current directory."),
        ] = Path(),
        name: Annotated[
            str | None,
            typer.Option("--name", "-n", help="Benchmark name written into eval.toml."),
        ] = None,
        force: Annotated[
            bool,
            typer.Option("--force", help="Overwrite files that already exist."),
        ] = False,
    ) -> None:
        """Create eval.toml, a starter dataset, and a target stub."""
        out = console()
        sym = symbols()
        written = scaffold(directory, name=name, force=force)

        table = Table.grid(padding=(0, 2))
        table.add_column(width=9)
        table.add_column(style="bold", no_wrap=True)
        table.add_column(style="dim", overflow="fold")
        for item in written:
            style = {"created": "green", "appended": "green", "skipped": "yellow"}[item.outcome]
            table.add_row(
                Text(item.outcome, style=style),
                str(item.path),
                item.note,
            )

        out.print()
        out.print(table)
        out.print()

        if any(w.outcome == "skipped" and "already exists" in w.note for w in written):
            out.print(
                Text(
                    f"  {sym.warn} some files were left alone - pass --force to overwrite",
                    style="yellow",
                )
            )
            out.print()

        out.print(Text("  Next:", style="bold"))
        out.print(Text("    slmeval doctor      ", style="cyan"), end="")
        out.print(Text("check this machine can run the benchmark", style="dim"))
        out.print(Text("    slmeval validate    ", style="cyan"), end="")
        out.print(Text("check eval.toml and the dataset", style="dim"))
        out.print()
        out.print(
            Text(
                "  Then point [target].entrypoint at your own agent and edit the dataset.",
                style="dim",
            )
        )
        out.print()
