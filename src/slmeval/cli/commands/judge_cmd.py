"""``slmeval judge`` - measure the judge before trusting it.

An LLM judge is the only evaluator here that costs money and the only one that
can be confidently wrong, so it does not get to produce a number until this
command has said what its agreement with human labels is. That is the whole
reason `judge calibrate` exists before Tier 3 is wired into a run: a judge
whose agreement nobody has measured is a random number generator with a
confident tone.

The output is three things, in this order, because that is the order a reader
needs them:

1. the raw agreement **with its Wilson interval**, because thirty labels is a
   small sample and a point estimate from one is exactly the mistake the rest
   of this tool exists to avoid;
2. Cohen's **kappa** beside it, because on a set where nine pairs in ten pass,
   a judge that says "pass" to everything scores 90% and knows nothing;
3. the **disagreements**, listed. The deliverable of a calibration run is not
   a score, it is the handful of pairs where the judge and the human parted
   company - that is what tells you whether the rubric is ambiguous, the judge
   is lenient, or the human was wrong.

It writes `slmeval.judge.json`, which is committed, for the same reason the
baseline file is: "we accepted a judge that agrees with us 71% of the time"
should be a diff somebody approved.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Annotated

import typer
from rich.table import Table
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.loading import open_project
from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.config.models import JudgeCfg
from slmeval.errors import ConfigError
from slmeval.eval.calibration import (
    MIN_LABELS,
    Agreement,
    Label,
    agreement,
    calibration_path,
    load_labels,
    verdict_counts,
)
from slmeval.eval.tier3 import Judge, Judgement, JudgementCache
from slmeval.providers.registry import build_client

judge_app = typer.Typer(
    no_args_is_help=True,
    help="Measure and inspect the Tier 3 judge.",
)

_ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
]


@judge_app.command("calibrate")
@handle_errors
def judge_calibrate(
    labels_path: Annotated[
        Path,
        typer.Argument(
            metavar="LABELS",
            help="A JSONL file of human-labelled pairs: case_id, rubric, "
            "reference, candidate, verdict.",
        ),
    ],
    config: _ConfigOpt = None,
    out_path: Annotated[
        Path | None,
        typer.Option("--out", "-o", help="Where to write it. Default: slmeval.judge.json."),
    ] = None,
    min_agreement: Annotated[
        float | None,
        typer.Option(
            "--min-agreement",
            min=0.0,
            max=1.0,
            help="Exit 1 if the Wilson lower bound falls below this. "
            "Defaults to [judge].min_agreement.",
        ),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Emit the record as JSON instead of a table.")
    ] = False,
    ascii_mode: Annotated[
        bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
    ] = False,
) -> None:
    """Run the judge over human-labelled pairs and report its agreement."""
    out = console()
    sym = symbols(ascii_mode or None)

    labels = load_labels(labels_path)

    with open_project(config, load_agent=False) as project:
        cfg = project.config.judge
        if not cfg.enabled:
            raise ConfigError(
                "no judge model configured",
                hint=(
                    'Add a [judge] section to eval.toml: provider = "openai", '
                    'model = "gpt-4o-mini". Calibration needs the model it is '
                    "about to measure."
                ),
            )
        floor = min_agreement if min_agreement is not None else cfg.min_agreement
        result = asyncio.run(_calibrate(cfg, labels, project.store.connection()))
        path = calibration_path(project.root, out_path)
        path.write_text(
            json.dumps(result.payload(), indent=2) + "\n", encoding="utf-8"
        )

    if as_json:
        print(json.dumps(result.payload(), indent=2))
    else:
        out.print()
        out.print(_summary(result, labels, sym.arrow))
        out.print()
        out.print(_confusion(result))
        if result.disagreements:
            out.print()
            out.print(_disagreements(result))
        for note in result.warnings():
            out.print(Text(f"  {sym.warn} {note}", style="yellow"))
        out.print()
        out.print(Text(f"  {sym.ok} wrote {path.name} - commit it", style="dim"))
        out.print()

    lower = result.lower
    if not result.usable:
        # Not a gate failure: nothing was measured, so there is nothing to
        # fail. Exit 2 says "this invocation did not answer the question",
        # which is a usage problem and not a verdict about the judge.
        raise typer.Exit(exit_codes.USAGE)
    if lower is not None and lower < floor:
        out.print(
            Text(
                f"  {sym.fail} agreement lower bound {lower:.0%} is below the "
                f"{floor:.0%} floor - this judge is not ready to grade anything",
                style="bold red",
            )
        )
        out.print()
        raise typer.Exit(exit_codes.GATE_FAILED)


async def _calibrate(
    cfg: JudgeCfg, labels: list[Label], conn: sqlite3.Connection
) -> Agreement:
    """Judge every labelled pair, then compare.

    Cached like any other judgement, so re-running a calibration after fixing
    one label costs one call rather than thirty. ``uncalibrated_ok`` is true
    here by definition: this is the run that produces the calibration.
    """
    client = build_client(cfg)
    judge = Judge(client=client, cache=JudgementCache(conn), uncalibrated_ok=True)
    pairs: list[tuple[Label, Judgement]] = []
    try:
        for label in labels:
            judgement = await judge.judge(
                rubric=label.rubric,
                reference=label.reference,
                candidate=label.candidate,
                case_id=label.case_id,
                question=label.question,
            )
            pairs.append((label, judgement))
    finally:
        await client.aclose()
    return agreement(pairs, judge_model=judge.model)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _summary(result: Agreement, labels: list[Label], arrow: str) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(overflow="fold")

    grid.add_row("judge", result.judge_model)
    grid.add_row("labels", f"{result.n} ({_counts(verdict_counts(labels))})")
    rate = f"{result.rate:.1%}" if result.rate is not None else "-"
    interval = result.interval.render() if result.interval else "-"
    grid.add_row("agreement", f"{rate}  {interval}")
    grid.add_row(
        "kappa",
        f"{result.kappa:.2f}" if result.kappa is not None else "undefined",
    )
    grid.add_row("verdict", _verdict_line(result))
    grid.add_row(
        "",
        f"{arrow} gate on the lower bound, not the rate: {result.n} labels is a "
        "small sample and the interval says how small",
    )
    return grid


def _verdict_line(result: Agreement) -> str:
    """Whether this is a calibration, and if not, which of the two reasons.

    Naming the reason matters: "you need more labels" and "these are not your
    labels" are fixed by completely different work, and a single message for
    both sends half the readers to the wrong one.
    """
    if result.usable:
        return "usable as a calibration"
    if result.illustrative:
        return "NOT a calibration - these labels are not human labels"
    return f"NOT a calibration - {MIN_LABELS} human labels is the floor, this has {result.n}"


def _confusion(result: Agreement) -> Table:
    """Human down the side, judge across the top.

    Worth printing even at n=30: the shape says *how* a judge is wrong, and
    "lenient" and "harsh" need opposite fixes.
    """
    labels = ("pass", "fail", "unclear")
    table = Table(
        title="human (rows) vs judge (columns)",
        title_style="dim",
        box=None,
        pad_edge=False,
    )
    table.add_column("", style="dim")
    for name in labels:
        table.add_column(name, justify="right")
    for human in labels:
        cells = [
            str(result.confusion.get((human, judge), 0)) for judge in labels
        ]
        table.add_row(human, *cells)
    return table


def _disagreements(result: Agreement) -> Table:
    table = Table(
        title="where the judge and the human parted company",
        title_style="dim",
        box=None,
        pad_edge=False,
    )
    table.add_column("case", style="cyan", no_wrap=True)
    table.add_column("human", no_wrap=True)
    table.add_column("judge", no_wrap=True)
    table.add_column("score", justify="right", no_wrap=True)
    table.add_column("the judge's reason", overflow="fold")
    for item in result.disagreements:
        table.add_row(
            item.case_id,
            item.human,
            item.judge,
            f"{item.score:.2f}",
            item.reasoning,
        )
    return table


def _counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"


def register(app: typer.Typer) -> None:
    app.add_typer(judge_app, name="judge")


__all__ = ["judge_app", "judge_calibrate", "register"]
