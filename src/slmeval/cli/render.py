"""Terminal output.

Rendering lives in pure functions that take data and return Rich renderables,
so every view is snapshot-testable with ``Console(record=True)`` and nothing
deep in the engine ever calls ``print``.

The terminal table *is* the product for a CLI tool — it is what gets
screenshotted and pasted into a pull request — so this module gets the same
care as the measurement code.
"""

from __future__ import annotations

import contextlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from slmeval.doctor import Check, Status
from slmeval.errors import SlmevalError


@dataclass(frozen=True)
class Symbols:
    ok: str
    warn: str
    fail: str
    skip: str
    arrow: str
    bar: str


UNICODE = Symbols(ok="✓", warn="!", fail="✗", skip="·", arrow="→", bar="│")
ASCII = Symbols(ok="OK", warn="!!", fail="XX", skip="--", arrow="->", bar="|")

STATUS_STYLE = {
    Status.OK: "green",
    Status.WARN: "yellow",
    Status.FAIL: "red",
    Status.SKIP: "dim",
}


def supports_unicode(stream: object | None = None) -> bool:
    """Legacy Windows consoles use cp1252 and cannot encode box glyphs."""
    encoding = getattr(stream or sys.stdout, "encoding", None) or ""
    try:
        UNICODE.ok.encode(encoding or "ascii")
    except (LookupError, UnicodeEncodeError):
        return False
    return True


def symbols(ascii_mode: bool | None = None) -> Symbols:
    if ascii_mode is None:
        ascii_mode = not supports_unicode()
    return ASCII if ascii_mode else UNICODE


def make_encoding_safe(stream: object | None) -> None:
    """Never let printing crash the tool.

    On a legacy Windows codepage an unencodable character raises
    UnicodeEncodeError from deep inside the writer. Our own messages are ASCII,
    but the text we echo is not always ours: `render_error` prints the user's
    own config line, and a case id or model name can contain anything. Degrading
    one glyph to '?' is always better than a traceback over a success message.
    """
    if stream is None or supports_unicode(stream):
        return
    reconfigure = getattr(stream, "reconfigure", None)
    if reconfigure is None:
        return
    # A detached or exotic stream may refuse; degrading is the whole point here.
    with contextlib.suppress(OSError, ValueError):
        reconfigure(errors="replace")


def console(*, stderr: bool = False, ascii_mode: bool | None = None) -> Console:
    """Build a Console that behaves when piped.

    Rich already honours NO_COLOR and non-tty detection; the explicit
    ``no_color`` keeps behaviour identical when a caller hands us a recording
    console in tests.
    """
    make_encoding_safe(sys.stderr if stderr else sys.stdout)
    return Console(
        stderr=stderr,
        no_color=bool(os.environ.get("NO_COLOR")),
        soft_wrap=False,
        highlight=False,
        emoji=False,
    )


def status_cell(status: Status, sym: Symbols) -> Text:
    glyph = {
        Status.OK: sym.ok,
        Status.WARN: sym.warn,
        Status.FAIL: sym.fail,
        Status.SKIP: sym.skip,
    }[status]
    return Text(glyph, style=STATUS_STYLE[status])


def render_checks(checks: list[Check], *, ascii_mode: bool | None = None) -> Table:
    """The doctor report: one line per check, fix hints indented beneath."""
    sym = symbols(ascii_mode)
    grid = Table.grid(padding=(0, 2))
    grid.add_column(width=2, justify="left")
    grid.add_column(style="bold", no_wrap=True)
    grid.add_column(overflow="fold")

    for check in checks:
        grid.add_row(status_cell(check.status, sym), check.name, check.detail)
        if check.hint and check.status is not Status.OK:
            grid.add_row("", "", Text(f"{sym.arrow} {check.hint}", style="dim"))
    return grid


def checks_summary(checks: list[Check]) -> Text:
    counts = {s: sum(1 for c in checks if c.status is s) for s in Status}
    parts = []
    if counts[Status.FAIL]:
        parts.append(Text(f"{counts[Status.FAIL]} failed", style="bold red"))
    if counts[Status.WARN]:
        parts.append(Text(f"{counts[Status.WARN]} warning", style="yellow"))
    if counts[Status.SKIP]:
        parts.append(Text(f"{counts[Status.SKIP]} skipped", style="dim"))
    parts.append(Text(f"{counts[Status.OK]} ok", style="green"))
    out = Text()
    for i, part in enumerate(parts):
        if i:
            out.append(", ", style="dim")
        out.append(part)
    return out


def render_error(err: SlmevalError, *, ascii_mode: bool | None = None) -> None:
    """Print a user-fixable error: what is wrong, where, and the source line."""
    sym = symbols(ascii_mode)
    out = console(stderr=True, ascii_mode=ascii_mode)

    out.print()
    out.print(Text(f"{sym.fail} {err.message}", style="bold red"))

    source_lines = _source_lines(err.source)
    label = err.source.name if err.source else ""

    for problem in err.problems:
        out.print()
        out.print(Text(f"  {problem.path}", style="bold"), end="")
        out.print(Text(f"  {sym.arrow}  {problem.message}"))
        if problem.line and problem.line <= len(source_lines):
            text = source_lines[problem.line - 1].rstrip()
            prefix = f"    {label}:{problem.line} {sym.bar} "
            out.print(Text(prefix, style="dim"), end="")
            out.print(Text(text, style="yellow"))
        elif problem.value is not None:
            out.print(Text(f"    got {problem.value}", style="dim"))

    if err.hint:
        out.print()
        out.print(Text(f"  hint: {err.hint}", style="cyan"))
    out.print()


def _source_lines(path: Path | None) -> list[str]:
    if path is None or not path.exists():
        return []
    try:
        return path.read_text(encoding="utf-8-sig").splitlines()
    except OSError:
        return []
