"""``slmeval baseline`` - write and inspect the committed reference.

``slmeval.baseline.json`` is the file ``slmeval gate --max-regression`` compares
against, and it belongs in the repository. That is the whole point:
gating against "the last run in the local database" gives different verdicts on
different machines and cannot be reviewed, while a committed file turns "we
accepted a two-point drop" into a diff somebody approved.

So this command is deliberately boring. It takes a stored run, writes the
metrics and the provenance to a JSON file, and tells the user to commit it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from slmeval.cli.loading import LATEST, open_project, resolve_run
from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.errors import ConfigError
from slmeval.report.baseline import Baseline, baseline_path
from slmeval.report.model import build_scorecard

baseline_app = typer.Typer(
    no_args_is_help=True,
    help="The committed reference a gate compares against.",
)

_ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
]
_OutOpt = Annotated[
    Path | None,
    typer.Option("--out", "-o", help="Where to write it. Defaults to slmeval.baseline.json."),
]


@baseline_app.command("save")
@handle_errors
def baseline_save(
    config: _ConfigOpt = None,
    run_id: Annotated[
        str, typer.Option("--run-id", help="A run id or prefix, or 'latest'.")
    ] = LATEST,
    out_path: _OutOpt = None,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing baseline file."),
    ] = False,
) -> None:
    """Write a stored run's metrics to the committed baseline file."""
    out = console()
    sym = symbols()

    with open_project(config, load_agent=False) as project:
        resolved = resolve_run(project.store, run_id)
        card = build_scorecard(
            project.store, resolved, prices_as_of=project.prices.as_of
        )
        path = baseline_path(project.root, out_path)
        previous = Baseline.load(path) if path.is_file() else None
        if previous is not None and not force:
            raise ConfigError(
                f"{path.name} already holds the baseline from run {previous.run_id}",
                source=path,
                hint=(
                    "pass --force to replace it - and commit the diff, because "
                    "moving a baseline is a decision somebody should review"
                ),
            )

        baseline = Baseline.from_scorecard(card)
        baseline.save(path)

        out.print()
        out.print(Text(f"  {sym.ok} wrote {path.name}", style="bold green"))
        out.print()
        out.print(_table(baseline, previous))
        out.print()
        out.print(
            Text(
                f"  {sym.arrow} commit it: a gate that compares against an "
                "uncommitted file gives a different answer on every machine",
                style="cyan",
            )
        )
        if not card.latency_reportable:
            out.print(
                Text(
                    f"  {sym.skip} no p95 recorded - this run was scheduled for "
                    "throughput, so its latency describes the scheduler. Re-run "
                    "with --serial-latency to gate on p95.",
                    style="dim",
                )
            )
        out.print()


@baseline_app.command("show")
@handle_errors
def baseline_show(
    config: _ConfigOpt = None,
    path_opt: _OutOpt = None,
) -> None:
    """Print the committed baseline, so a failing gate can be read against it."""
    out = console()
    with open_project(config, load_agent=False) as project:
        path = baseline_path(project.root, path_opt)
        baseline = Baseline.load(path)

    out.print()
    out.print(
        Text(
            f"  {path.name}  {baseline.run_id} · saved {baseline.saved_at[:19]} · "
            f"slmeval {baseline.slmeval_version}",
            style="dim",
        )
    )
    out.print(
        Text(
            f"  dataset {baseline.dataset_hash} · config {baseline.config_hash} · "
            f"{baseline.mode} · seed {baseline.seed} · "
            f"{baseline.runs_per_case} runs/case",
            style="dim",
        )
    )
    out.print()
    out.print(_table(baseline, None))
    out.print()


def _table(baseline: Baseline, previous: Baseline | None) -> Table:
    """The rows, with the previous file's numbers beside them when replacing.

    Shown side by side on a ``--force`` overwrite because that is the moment a
    user is deciding whether to accept a change, and the diff they are about to
    commit is the evidence.
    """
    grid = Table(box=None, pad_edge=False, padding=(0, 2), header_style="bold", show_edge=False)
    grid.add_column("Candidate", no_wrap=True)
    grid.add_column("pass^k", justify="right", no_wrap=True)
    grid.add_column("wilson lo", justify="right", no_wrap=True)
    grid.add_column("cases", justify="right", no_wrap=True)
    grid.add_column("$/case", justify="right", no_wrap=True)
    if previous is not None:
        grid.add_column("was", justify="right", no_wrap=True)

    for candidate_id, row in sorted(
        baseline.candidates.items(), key=lambda kv: (kv[1].role != "baseline", kv[0])
    ):
        cells = [
            Text(candidate_id, style="bold" if row.role == "baseline" else ""),
            Text(_pct(row.pass_hat_k)),
            Text(_pct(row.wilson_lo), style="dim"),
            Text(f"{row.cases_all_pass}/{row.cases}", style="dim"),
            Text(
                f"${row.cost_per_case:.4f}" if row.cost_per_case is not None else "-",
                style="dim",
            ),
        ]
        if previous is not None:
            before = previous.candidate(candidate_id)
            cells.append(
                Text(
                    _pct(before.pass_hat_k) if before else "new",
                    style=_delta_style(
                        before.pass_hat_k if before else None, row.pass_hat_k
                    ),
                )
            )
        grid.add_row(*cells)
    return grid


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "-"


def _delta_style(before: float | None, now: float | None) -> str:
    """Colour the old number by which way the new one moved.

    One percentage point of slack, same as the scorecard: floating-point noise
    must not flicker a committed baseline green and red between saves of the
    same run.
    """
    if before is None or now is None or abs(now - before) < 0.01:
        return "dim"
    return "green" if now > before else "red"


def register(app: typer.Typer) -> None:
    app.add_typer(baseline_app, name="baseline")


__all__ = ["baseline_app", "register"]
