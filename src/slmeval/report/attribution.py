"""Per-node attribution: which node each candidate broke in.

This is the table that turns a benchmark into a recommendation. "gemma3:4b is
thirteen points worse" is a result; "gemma3:4b is fine on ``intent_router`` and
fails in ``policy_check``" is an action - keep the small model on the router,
leave the big one on policy. That single sentence is what makes "downsize node
3 only" a thing a user can do on Monday, and it is a column, not an essay.

Two things it deliberately does not do.

It does not rank nodes by traffic. The busiest node is almost always the one
with the most calls, and sorting by volume would put it first on every run
while saying nothing. Failures are what is scarce, so failures set the order.

It does not attribute a harness error to a node. Our own crash has a node name
in its trace like anything else, and letting it colour a node red would send
the user to rewrite a prompt over a bug in our code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rich.console import Group, RenderableType
from rich.table import Table
from rich.text import Text

from slmeval.cli.render import dash as dash_for
from slmeval.cli.render import symbols
from slmeval.report.model import Scorecard
from slmeval.storage.models import BlameRow


@dataclass(frozen=True, slots=True)
class NodeCell:
    """One candidate's behaviour at one node."""

    candidate_id: str
    node_name: str
    calls: int = 0
    failures: int = 0
    schema_valid_rate: float | None = None
    mean_latency_ms: float | None = None
    repair_attempts: int = 0
    model: str = ""

    @property
    def failure_rate(self) -> float | None:
        return self.failures / self.calls if self.calls else None


def node_cells(card: Scorecard) -> dict[tuple[str, str], NodeCell]:
    """``(candidate_id, node_name) -> cell``, from the stored aggregates."""
    blame = _blame_by_node(card.blame)
    out: dict[tuple[str, str], NodeCell] = {}
    for row in card.nodes:
        key = (row.candidate_id, row.node_name)
        out[key] = NodeCell(
            candidate_id=row.candidate_id,
            node_name=row.node_name,
            calls=row.calls,
            failures=row.errors + blame.get(key, 0),
            schema_valid_rate=row.schema_valid_rate,
            mean_latency_ms=row.mean_latency_ms,
            repair_attempts=row.repair_attempts,
            model=row.model,
        )
    # A node can fail purely through its tools and never log a model error, so
    # blame rows can name a node that has no model calls at all. Dropping those
    # would hide exactly the failures this table exists to locate.
    for (candidate_id, node_name), count in blame.items():
        key = (candidate_id, node_name)
        if key not in out:
            out[key] = NodeCell(
                candidate_id=candidate_id, node_name=node_name, failures=count
            )
    return out


def _blame_by_node(rows: Sequence[BlameRow]) -> dict[tuple[str, str], int]:
    out: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row.candidate_id, row.node_name)
        out[key] = out.get(key, 0) + row.count
    return out


def node_order(card: Scorecard, cells: dict[tuple[str, str], NodeCell]) -> list[str]:
    """Nodes worst-first: most failures, then most repairs, then by name.

    By name last so the order is stable across runs - a snapshot test of this
    table has to be reproducible, and a user comparing two reports should not
    have to re-find a row.
    """
    names = {node for _, node in cells}
    return sorted(
        names,
        key=lambda node: (
            -sum(c.failures for (_, n), c in cells.items() if n == node),
            -sum(c.repair_attempts for (_, n), c in cells.items() if n == node),
            node,
        ),
    )


def render_attribution(
    card: Scorecard, *, ascii_mode: bool | None = None
) -> RenderableType:
    """The per-node table, plus the one-line recommendation it supports."""
    cells = node_cells(card)
    if not cells:
        return Text("")
    sym = symbols(ascii_mode)
    dash = dash_for(ascii_mode)

    grid = Table(
        box=None,
        pad_edge=False,
        padding=(0, 2),
        header_style="bold",
        show_edge=False,
        title="per-node attribution",
        title_style="bold dim",
        title_justify="left",
    )
    grid.add_column("Node", no_wrap=True)
    for row in card.rows:
        grid.add_column(row.candidate_id, justify="right", no_wrap=True)

    for node in node_order(card, cells):
        line: list[RenderableType] = [Text(node)]
        for row in card.rows:
            cell = cells.get((row.candidate_id, node))
            line.append(_cell_text(cell, dash))
        grid.add_row(*line)

    advice = recommendation(card, cells, sym.arrow)
    return Group(grid, Text(""), advice) if advice else grid


def _cell_text(cell: NodeCell | None, dash: str) -> Text:
    if cell is None or (not cell.calls and not cell.failures):
        return Text(dash, style="dim")
    if cell.failures:
        return Text(
            f"{cell.failures} fail" + (f"/{cell.calls}" if cell.calls else ""),
            style="red",
        )
    if cell.repair_attempts:
        return Text(f"{cell.calls} ok ~{cell.repair_attempts}", style="yellow")
    return Text(f"{cell.calls} ok", style="green")


def recommendation(
    card: Scorecard, cells: dict[tuple[str, str], NodeCell], arrow: str = "->"
) -> Text | None:
    """The actionable sentence, when the data actually supports one.

    It is offered only when a candidate is clean on at least one node and
    failing on at least one other - the "downsize this node, not that one"
    shape. A candidate that fails everywhere has no node worth keeping, and a
    candidate that fails nowhere needs no advice; inventing a recommendation
    for either would be the tool talking past its own evidence.
    """
    base = card.baseline
    if base is None:
        return None
    lines: list[str] = []
    for row in card.candidates:
        mine = {n: c for (cid, n), c in cells.items() if cid == row.candidate_id}
        clean = sorted(n for n, c in mine.items() if c.calls and not c.failures)
        broken = sorted(n for n, c in mine.items() if c.failures)
        if not clean or not broken:
            continue
        lines.append(
            f"{row.candidate_id} held up on {', '.join(clean)} and broke in "
            f"{', '.join(broken)} {arrow} keep {base.candidate_id} on "
            f"{', '.join(broken)}"
        )
    if not lines:
        return None
    out = Text()
    for index, line in enumerate(lines):
        out.append("  " + line, style="cyan")
        if index < len(lines) - 1:
            out.append("\n")
    return out


__all__ = [
    "NodeCell",
    "node_cells",
    "node_order",
    "recommendation",
    "render_attribution",
]
