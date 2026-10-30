"""`slmeval cache` — what the trace store holds, and how to reclaim it.

Benchmark traces grow to gigabytes. Give users the broom before they go looking
for one, because the alternative is someone deleting ``.slmeval/`` by hand and
losing the baselines their gate compares against.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.config.loader import CACHE_DIR_NAME, DEFAULT_CONFIG_NAME
from slmeval.errors import ConfigError
from slmeval.storage import RunStore

cache_app = typer.Typer(
    no_args_is_help=True,
    help="Trace store statistics and cleanup.",
)

_AGE_RE = re.compile(r"^\s*(\d+)\s*([smhdw])\s*$", re.IGNORECASE)
_UNITS = {
    "s": timedelta(seconds=1),
    "m": timedelta(minutes=1),
    "h": timedelta(hours=1),
    "d": timedelta(days=1),
    "w": timedelta(weeks=1),
}


def parse_age(text: str) -> timedelta:
    """``30d``, ``12h``, ``2w``. No bare numbers.

    A bare ``30`` could mean seconds, days or runs, and the command it feeds
    deletes things. Refusing to guess costs the user one retyped character; a
    wrong guess costs them their benchmark history.
    """
    match = _AGE_RE.match(text)
    if not match:
        raise ConfigError(
            f"cannot read {text!r} as an age",
            hint="use a number and a unit: 30d, 12h, 2w",
        )
    return int(match.group(1)) * _UNITS[match.group(2).lower()]


def human_bytes(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def project_root(explicit: Path | None) -> Path:
    """The directory whose ``.slmeval`` we operate on.

    Walks up for ``eval.toml`` like the other commands, but never parses it:
    the whole point of ``cache stats`` is to work when something else is broken,
    and a config too invalid to load must not stop a user inspecting their own
    disk usage.
    """
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        if (directory / DEFAULT_CONFIG_NAME).is_file() or (directory / CACHE_DIR_NAME).is_dir():
            return directory
    raise ConfigError(
        f"no {DEFAULT_CONFIG_NAME} or {CACHE_DIR_NAME}/ found in {here} or any parent",
        hint="run this from a project directory, or pass --path",
    )


_PathOpt = Annotated[
    Path | None,
    typer.Option("--path", help="Project directory to inspect. Defaults to the one you are in."),
]


@cache_app.command("stats")
@handle_errors
def cache_stats(
    path: _PathOpt = None,
    verbose: Annotated[
        bool, typer.Option("--verbose", "-v", help="Also show the schema migration history.")
    ] = False,
) -> None:
    """Show what the trace store holds and what it costs on disk."""
    root = project_root(path)
    out = console()
    sym = symbols()

    with RunStore.for_project(root) as store:
        stats = store.stats()
        cassettes = root / CACHE_DIR_NAME / "cassettes.sqlite"
        cassette_bytes = cassettes.stat().st_size if cassettes.is_file() else 0

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        table.add_row("store", str(store.path))
        table.add_row("schema", f"v{store.schema_version}")
        table.add_row("runs", str(stats.runs))
        if stats.running:
            table.add_row("in progress", Text(str(stats.running), style="yellow"))
        table.add_row("executions", str(stats.executions))
        table.add_row("model calls", str(stats.node_traces))
        table.add_row("tool calls", str(stats.tool_traces))
        table.add_row("evaluations", str(stats.evaluations))
        if stats.oldest_run:
            table.add_row("oldest run", stats.oldest_run[:19])
            table.add_row("newest run", (stats.newest_run or "")[:19])
        table.add_row("database", human_bytes(stats.db_bytes))
        table.add_row("payloads", f"{stats.blobs} files, {human_bytes(stats.blob_bytes)}")
        if cassette_bytes:
            table.add_row("cassettes", human_bytes(cassette_bytes))
        table.add_row(
            "total",
            human_bytes(stats.db_bytes + stats.blob_bytes + cassette_bytes),
        )

        out.print()
        out.print(table)
        if verbose:
            out.print()
            out.print(Text("  migrations", style="dim"))
            for version, name, at, built_by in store.migration_history():
                out.print(
                    Text(
                        f"    v{version} {name} · {at[:19]} · slmeval {built_by}",
                        style="dim",
                    )
                )
        out.print()
        if stats.runs == 0:
            out.print(Text(f"  {sym.skip} no runs stored yet", style="dim"))
            out.print()


@cache_app.command("prune")
@handle_errors
def cache_prune(
    older_than: Annotated[
        str,
        typer.Option("--older-than", help="Delete runs older than this: 30d, 12h, 2w."),
    ] = "30d",
    keep_last: Annotated[
        int,
        typer.Option(
            "--keep-last",
            min=0,
            help="Always keep this many of the most recent runs, whatever their age.",
        ),
    ] = 3,
    path: _PathOpt = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would go, delete nothing.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip the confirmation.")] = False,
    force: Annotated[
        bool,
        typer.Option("--force", help="Prune even with runs marked in progress (crash leftovers)."),
    ] = False,
) -> None:
    """Delete old runs and the payloads nothing references any more."""
    root = project_root(path)
    age = parse_age(older_than)
    out = console()
    sym = symbols()

    with RunStore.for_project(root) as store:
        cutoff = datetime.now(UTC) - age
        stale = store.runs_older_than(cutoff, keep_last=keep_last)

        out.print()
        if not stale:
            out.print(Text(f"  {sym.ok} nothing older than {older_than} to prune", style="green"))
            out.print()
            # Payloads can still be orphaned by a crash between a blob write
            # and its commit, so sweep anyway - just quietly.
            swept = store.blobs.sweep(store.referenced_blobs(), dry_run=dry_run)
            if swept.deleted:
                out.print(
                    Text(
                        f"  {sym.ok} freed {human_bytes(swept.freed_bytes)} "
                        f"of unreferenced payloads",
                        style="green",
                    )
                )
                out.print()
            return

        table = Table.grid(padding=(0, 2))
        table.add_column(style="dim", no_wrap=True)
        table.add_column(overflow="fold")
        for run in stale[:10]:
            table.add_row(run.run_id, f"{run.name} · {run.started_at[:19]}")
        if len(stale) > 10:
            table.add_row("", Text(f"... and {len(stale) - 10} more", style="dim"))
        out.print(Text(f"  {len(stale)} run(s) older than {older_than}:", style="bold"))
        out.print()
        out.print(table)
        out.print()

        if dry_run:
            out.print(Text(f"  {sym.skip} dry run, nothing deleted", style="dim"))
            out.print()
            return
        # Deleting benchmark history is not undoable, and the thing being
        # deleted is what the next gate compares against. Ask.
        if not yes and not typer.confirm("  Delete these runs and their traces?"):
            out.print()
            out.print(Text(f"  {sym.skip} cancelled", style="dim"))
            out.print()
            raise typer.Exit()

        deleted, swept = store.prune(older_than=age, keep_last=keep_last, force=force)
        out.print()
        out.print(
            Text(
                f"  {sym.ok} deleted {len(deleted)} run(s), freed "
                f"{human_bytes(swept.freed_bytes)} of payloads",
                style="bold green",
            )
        )
        if swept.kept_recent:
            out.print(
                Text(
                    f"    kept {swept.kept_recent} recent unreferenced payload(s) - "
                    "a sweep spares anything a run may still be committing",
                    style="dim",
                )
            )
        out.print()


def register(app: typer.Typer) -> None:
    app.add_typer(cache_app, name="cache")
