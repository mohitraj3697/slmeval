"""The scorecard. For a CLI tool, the terminal table *is* the product.

This is what gets screenshotted, posted, and pasted into a pull request, so it
gets the same care as the measurement code. Pure functions from a
:class:`~slmeval.report.model.Scorecard` to a Rich renderable - nothing here
prints, which is what makes every view snapshot-testable.

Four rules it follows, all of them about not being misleading:

**Always print ``n``.** A rate without its sample size is a number pretending
to be evidence, and the whole product's credibility rests on nobody
over-reading three samples.

**Always print the interval, next to the point estimate.** Seeing ``100%`` and
``[44%, 100%]`` side by side teaches the reader more about ``n = 3`` than any
documentation will.

**Colour by delta, not by absolute value.** 83% is not red; 83% where the
baseline got 97% is. Colouring absolutes would paint a hard benchmark red
everywhere and a trivial one green, telling the reader nothing either way.

**Warnings go above the table.** A forbidden tool call must never be a column
somebody scrolls past. It appears before the first number, and so does a
harness error, because that one calls the whole run into question.
"""

from __future__ import annotations

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import Symbols, ascii_only, symbols
from slmeval.cli.render import dash as dash_for
from slmeval.report.model import CandidateReport, Scorecard

GOOD = "green"
BAD = "red"
MEH = "yellow"
DIM = "dim"


def render_scorecard(
    card: Scorecard,
    *,
    ascii_mode: bool | None = None,
    show_nodes: bool = False,
) -> RenderableType:
    """The whole scorecard: header, warnings, table, footer."""
    sym = symbols(ascii_mode)
    parts: list[RenderableType] = [
        Text(""),
        header(card),
        Text(""),
    ]
    notes = warnings_block(card, ascii_mode=ascii_mode)
    if notes is not None:
        parts += [notes, Text("")]
    parts.append(table(card, ascii_mode=ascii_mode))
    if show_nodes:
        from slmeval.report.attribution import render_attribution

        parts += [Text(""), render_attribution(card, ascii_mode=ascii_mode)]
    parts += [Text(""), footer(card, sym)]
    return Group(*parts)


def header(card: Scorecard) -> Text:
    """One line naming everything that would invalidate a comparison.

    The seed, the tool mode and the fault rate are in the headline rather than
    a footnote because they are what someone re-running this needs, and because
    two tables that differ in any of them are not comparable however similar
    they look.
    """
    run = card.run
    out = Text("  ")
    out.append(run.name, style="bold")
    bits = [
        f"{len(card.cases)} cases x {card.repeats} runs",
        run.mode,
        f"faults {run.fault_rate:.0%}" if run.fault_rate else "no faults",
        f"seed {run.seed}",
        run.scheduling,
    ]
    out.append(" · ", style=DIM)
    out.append(" · ".join(bits), style=DIM)
    out.append("\n  ")
    out.append(f"{run.run_id}", style=DIM)
    if run.git_sha:
        out.append(f" · {run.git_sha}", style=DIM)
    out.append(f" · dataset {run.dataset_hash}", style=DIM)
    return out


def warnings_block(
    card: Scorecard, *, ascii_mode: bool | None = None
) -> Text | None:
    """The lines above the table, worst first."""
    notes = card.warnings()
    if not notes:
        return None
    sym = symbols(ascii_mode)
    out = Text()
    for index, note in enumerate(notes):
        glyph, style = _note_style(note, sym)
        out.append(f"  {glyph} ", style=style)
        out.append(note, style=style)
        if index < len(notes) - 1:
            out.append("\n")
    return out


def _note_style(note: str, sym: Symbols) -> tuple[str, str]:
    lowered = note.lower()
    if "forbidden" in lowered:
        return sym.fail, "bold red"
    if "inside slmeval" in lowered:
        return sym.fail, "bold magenta"
    # Not red, and that is the point: an uncovered path is a gap in our
    # recordings, not a model failure, and colouring it like one would undo the
    # distinction the diagnosis exists to draw.
    if "do not cover" in lowered:
        return sym.warn, "cyan"
    if "latency not measured" in lowered:
        return sym.skip, DIM
    return sym.warn, MEH


def table(card: Scorecard, *, ascii_mode: bool | None = None) -> Table:
    """The table itself."""
    dash = dash_for(ascii_mode)
    grid = Table(
        box=None,
        pad_edge=False,
        padding=(0, 2),
        header_style="bold",
        show_edge=False,
    )
    grid.add_column("Candidate", no_wrap=True)
    grid.add_column(f"pass^{card.repeats}", justify="right", no_wrap=True)
    grid.add_column("95% CI", justify="right", no_wrap=True)
    grid.add_column("runs", justify="right", no_wrap=True)
    grid.add_column("Schema", justify="right", no_wrap=True)
    grid.add_column("Tool F1", justify="right", no_wrap=True)
    grid.add_column("Steps", justify="right", no_wrap=True)
    if card.latency_reportable:
        grid.add_column("p95 ms", justify="right", no_wrap=True)
    grid.add_column("$/case", justify="right", no_wrap=True)
    grid.add_column(_delta_header(ascii_mode), justify="right", no_wrap=True)

    for row in card.rows:
        grid.add_row(*_cells(card, row, dash=dash))
    return grid


def _delta_header(ascii_mode: bool | None) -> str:
    """``d Cost`` on a legacy console, which cannot encode a delta glyph."""
    return "d Cost" if ascii_only(ascii_mode) else "Δ Cost"


def _cells(card: Scorecard, row: CandidateReport, *, dash: str) -> list[RenderableType]:
    estimated = "~" if row.tokens_estimated else ""
    cells: list[RenderableType] = [
        _name(row),
        _pass_cell(card, row, dash),
        _interval_cell(row, dash),
        Text(
            f"{row.stats.passes}/{row.stats.executions}"
            if row.stats.executions
            else dash,
            style=DIM,
        ),
        _rate_cell(card, row, row.schema_valid_rate, "schema", dash),
        _rate_cell(card, row, row.tool_f1, "f1", dash, digits=2),
        Text(f"{row.mean_steps:.1f}" if row.mean_steps is not None else dash),
    ]
    if card.latency_reportable:
        cells.append(
            Text(f"{row.p95_ms:,.0f}" if row.p95_ms is not None else dash)
        )
    cells.append(
        Text(
            f"{estimated}${row.cost_per_case:.4f}"
            if row.cost_per_case is not None
            else dash,
            style=DIM if row.cost_per_case == 0 else "",
        )
    )
    cells.append(_delta_cost_cell(card, row, dash))
    return cells


def _name(row: CandidateReport) -> Text:
    out = Text()
    if row.baseline and row.candidate_id != "baseline":
        out.append("baseline ", style=DIM)
    out.append(row.candidate_id, style="bold" if row.baseline else "")
    return out


def _pass_cell(card: Scorecard, row: CandidateReport, dash: str) -> Text:
    value = row.pass_hat_k
    if value is None:
        return Text(dash, style=DIM)
    delta = card.delta_pass(row)
    return Text(f"{value:.1%}", style=_delta_style(delta, bigger_is_better=True))


def _interval_cell(row: CandidateReport, dash: str) -> Text:
    interval = row.stats.interval
    if interval is None:
        return Text(dash, style=DIM)
    return Text(interval.render(), style=DIM)


def _rate_cell(
    card: Scorecard,
    row: CandidateReport,
    value: float | None,
    which: str,
    dash: str,
    *,
    digits: int = 0,
) -> Text:
    """One rate, coloured by how it compares with the baseline's own.

    ``which`` names the baseline figure to compare against rather than taking
    the number directly, because the baseline row has to be able to render
    itself - and it has no delta against itself to colour by.
    """
    if value is None:
        return Text(dash, style=DIM)
    reference = _reference_value(card, which)
    delta = None if reference is None or row.baseline else value - reference
    text = f"{value:.{digits}f}" if digits else f"{value:.0%}"
    return Text(text, style=_delta_style(delta, bigger_is_better=True))


def _delta_cost_cell(card: Scorecard, row: CandidateReport, dash: str) -> Text:
    delta = card.delta_cost(row)
    if delta is None:
        return Text(dash, style=DIM)
    # Cheaper is better, so the sign convention is inverted against the
    # quality columns. Spelled out here rather than left to the reader.
    return Text(f"{delta:+.0%}", style=_delta_style(delta, bigger_is_better=False))


def _reference_value(card: Scorecard, which: str) -> float | None:
    base = card.baseline
    if base is None:
        return None
    return {"schema": base.schema_valid_rate, "f1": base.tool_f1}.get(which)


def _delta_style(delta: float | None, *, bigger_is_better: bool) -> str:
    """Colour from the change, never the level.

    A threshold of one percentage point, so floating-point noise and a genuinely
    identical result both render neutral instead of flickering green and red
    between runs of the same config.
    """
    if delta is None or abs(delta) < 0.01:
        return ""
    improved = delta > 0 if bigger_is_better else delta < 0
    return GOOD if improved else BAD


def footer(card: Scorecard, sym: Symbols) -> Text:
    """Where the numbers came from, in one dim block.

    The estimator and the price date belong on the artefact, not in the
    documentation: this table gets screenshotted away from everything that
    would otherwise explain it.
    """
    out = Text()
    out.append(
        f"  pass^{card.repeats} = fraction of cases passing every repeat; "
        "95% CI is the Wilson interval on that fraction\n",
        style=DIM,
    )
    out.append(
        f"  gate on the lower bound, never the point estimate "
        f"(n={len(card.cases)} cases)\n",
        style=DIM,
    )
    out.append(f"  prices as of {card.prices_as_of}", style=DIM)
    if any(r.recovery_rate is not None for r in card.rows):
        out.append("\n  ", style=DIM)
        out.append(
            " · ".join(
                f"{r.candidate_id} recovered from {r.recovery_rate:.0%} of faults"
                for r in card.rows
                if r.recovery_rate is not None
            ),
            style=DIM,
        )
    return out


def render_summary_line(card: Scorecard, *, ascii_mode: bool | None = None) -> Text:
    """The one-line verdict the whole product exists to print.

        intent_router: gpt-4o -> gemma3:4b - cost -78%, pass^3 96.4% -> 91.2%

    Only for the single-candidate case. With several candidates there is no
    single recommendation to make, and inventing one would be the tool
    editorialising over the user's own trade-off.
    """
    sym = symbols(ascii_mode)
    base = card.baseline
    others = card.candidates
    if base is None or len(others) != 1:
        return Text("")
    row = others[0]
    cost = card.delta_cost(row)
    out = Text("  ")
    out.append(f"{base.model} {sym.arrow} {row.model}", style="bold")
    if cost is not None:
        out.append(f" - cost {cost:+.0%}", style=GOOD if cost < 0 else BAD)
    if base.pass_hat_k is not None and row.pass_hat_k is not None:
        interval = row.stats.interval
        tail = f" (CI {interval.render()})" if interval else ""
        out.append(
            f", pass^{card.repeats} {base.pass_hat_k:.1%} {sym.arrow} "
            f"{row.pass_hat_k:.1%}{tail}"
        )
    return out


def render_rollups(
    card: Scorecard, *, ascii_mode: bool | None = None, limit: int = 0
) -> Table:
    """Per-case detail: which cases each candidate actually failed.

    The scorecard says a candidate lost four points; this says which four cases
    and how consistently. A user cannot act on the first without the second.
    """
    dash = dash_for(ascii_mode)
    grid = Table(box=None, pad_edge=False, padding=(0, 2), header_style="bold", show_edge=False)
    grid.add_column("Case", no_wrap=True)
    for row in card.rows:
        grid.add_column(row.candidate_id, justify="right", no_wrap=True)

    shown = card.cases[:limit] if limit else card.cases
    for case_id in shown:
        cells: list[RenderableType] = [Text(case_id)]
        for row in card.rows:
            found = row.rollups.get(case_id)
            if found is None or not found.n:
                cells.append(Text(dash, style=DIM))
                continue
            style = GOOD if found.all_passed else (MEH if found.c else BAD)
            cells.append(Text(f"{found.c}/{found.n}", style=style))
        grid.add_row(*cells)
    return grid


__all__ = [
    "footer",
    "header",
    "render_rollups",
    "render_scorecard",
    "render_summary_line",
    "table",
    "warnings_block",
]
