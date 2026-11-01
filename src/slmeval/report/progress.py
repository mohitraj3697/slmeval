"""The live progress view, as an :class:`~slmeval.core.runner.Observer`.

A matrix run takes twenty minutes. The user has to be able to see it working
and form an opinion before it ends - a silent terminal for twenty minutes is
indistinguishable from a hang, and the first thing anyone does about a hang is
``Ctrl-C``.

It lives here rather than in the runner because the runner must not print. That
keeps the orchestration testable without a terminal, and it means this whole
file can be swapped for a JSON-lines observer in CI without touching the
scheduler.

Three details that matter more than they look:

- **One bar per candidate.** The interesting question mid-run is not "how far
  along" but "is this candidate collapsing", and a single aggregate bar cannot
  answer it.
- **Live pass / fail / error counters, with errors separate.** A harness error
  must be visibly not a model failure even while the run is still going.
- **Non-TTY degrades to plain lines.** Piped into a CI log, a repainting
  progress bar becomes thousands of escape sequences; the same information goes
  out one line at a time instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import TracebackType

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.text import Text

from slmeval.cli.render import console as make_console
from slmeval.cli.render import symbols
from slmeval.core.runner import RunPlan, RunSummary, UnitResult, WorkUnit
from slmeval.storage.models import Outcome

_QUIET_OUTCOMES = frozenset({Outcome.PASSED, Outcome.UNCLASSIFIED})


@dataclass
class Tally:
    """Live counters. ``errors`` is ours, and stays apart from ``failed``."""

    passed: int = 0
    failed: int = 0
    errors: int = 0
    uncovered: int = 0
    unsafe: int = 0

    def add(self, result: UnitResult) -> None:
        if result.outcome is Outcome.PASSED:
            self.passed += 1
        elif result.outcome is Outcome.HARNESS_ERROR:
            self.errors += 1
        elif result.outcome is Outcome.UNCOVERED:
            self.uncovered += 1
        else:
            self.failed += 1
        if result.outcome is Outcome.FORBIDDEN_TOOL:
            self.unsafe += 1

    def render(self) -> str:
        parts = [f"{self.passed} pass", f"{self.failed} fail"]
        if self.errors:
            parts.append(f"{self.errors} harness")
        if self.uncovered:
            parts.append(f"{self.uncovered} uncovered")
        if self.unsafe:
            parts.append(f"{self.unsafe} UNSAFE")
        return " · ".join(parts)


@dataclass
class ProgressObserver:
    """Renders a run as it happens. Use as a context manager."""

    console: Console = field(default_factory=make_console)
    ascii_mode: bool | None = None
    quiet: bool = False

    _progress: Progress | None = field(default=None, init=False, repr=False)
    _tasks: dict[str, TaskID] = field(default_factory=dict, init=False, repr=False)
    _overall: TaskID | None = field(default=None, init=False, repr=False)
    _tally: Tally = field(default_factory=Tally, init=False, repr=False)
    _plain: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        # Rich's own terminal detection, not an environment guess: it already
        # accounts for NO_COLOR, dumb terminals and redirected output.
        self._plain = self.quiet or not self.console.is_terminal

    # ------------------------------------------------------------- lifecycle

    def __enter__(self) -> ProgressObserver:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    # -------------------------------------------------------------- observer

    def on_plan(self, plan: RunPlan) -> None:
        self.console.print()
        self.console.print(
            Text(f"  {plan.name}", style="bold"),
            Text(
                f" · {len(plan.matrix.candidate_ids)} candidates"
                f" · {len(plan.matrix.case_ids)} cases"
                f" x {plan.matrix.repeats} runs"
                f" · {plan.matrix.planned} executions",
                style="dim",
            ),
        )
        detail = f"  {plan.run_id} · {plan.tool_mode} · {plan.scheduling}"
        if plan.limits:
            detail += f" · limits {plan.limits}"
        if plan.matrix.resumed:
            detail += f" · resuming, {len(plan.matrix.resumed)} already done"
        self.console.print(Text(detail, style="dim"))
        self.console.print()

        if self._plain:
            return
        self._progress = Progress(
            SpinnerColumn(style="dim"),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn("{task.fields[note]}"),
            console=self.console,
            transient=False,
        )
        self._progress.start()
        for candidate_id in plan.matrix.candidate_ids:
            total = len(plan.matrix.for_candidate(candidate_id))
            if not total:
                continue
            self._tasks[candidate_id] = self._progress.add_task(
                f"  {candidate_id}", total=total, note=""
            )
        self._overall = self._progress.add_task(
            "  all", total=len(plan.matrix.units), note=""
        )

    def on_unit_start(self, unit: WorkUnit) -> None:
        if self._progress is None:
            return
        task = self._tasks.get(unit.candidate_id)
        if task is not None:
            self._progress.update(task, note=f"[dim]{unit.case_id}#{unit.repeat_idx}[/dim]")

    def on_unit_done(self, result: UnitResult) -> None:
        self._tally.add(result)
        if self._progress is None:
            self._print_line(result)
            return
        task = self._tasks.get(result.candidate_id)
        if task is not None:
            self._progress.advance(task)
        if self._overall is not None:
            self._progress.update(
                self._overall, advance=1, note=f"[dim]{self._tally.render()}[/dim]"
            )
        # A forbidden tool call is not something to leave in a progress field
        # that scrolls away. It is printed as its own line, in red, while the
        # run continues.
        if result.outcome is Outcome.FORBIDDEN_TOOL:
            self._progress.console.print(
                Text(
                    f"  {symbols(self.ascii_mode).fail} FORBIDDEN TOOL in "
                    f"{result.unit.case_id} ({result.candidate_id})",
                    style="bold red",
                )
            )
        elif result.outcome is Outcome.HARNESS_ERROR:
            self._progress.console.print(
                Text(
                    f"  {symbols(self.ascii_mode).fail} harness error in "
                    f"{result.unit}: {result.error_class} - this is our bug, not "
                    "the model's",
                    style="bold magenta",
                )
            )

    def _print_line(self, result: UnitResult) -> None:
        """One line per execution, for a log rather than a terminal.

        Failures and anything unsafe always print. Passes do not: a 1,500-row
        CI log of "ok" is noise that hides the twelve lines that matter.
        """
        if result.outcome in _QUIET_OUTCOMES:
            return
        sym = symbols(self.ascii_mode)
        self.console.print(
            Text(
                f"  {sym.fail} {result.unit} {sym.arrow} {result.outcome.value}"
                + (f": {result.error_message}" if result.error_message else ""),
                style="red" if result.outcome is not Outcome.UNCOVERED else "cyan",
            )
        )

    def on_note(self, message: str) -> None:
        target = self._progress.console if self._progress else self.console
        target.print(Text(f"  {symbols(self.ascii_mode).skip} {message}", style="dim"))

    def on_finish(self, summary: RunSummary) -> None:
        self.close()
        sym = symbols(self.ascii_mode)
        self.console.print()
        style = "bold green" if summary.complete else "bold yellow"
        line = (
            f"  {sym.ok if summary.complete else sym.warn} "
            f"{summary.executed} executed"
        )
        if summary.resumed:
            line += f", {summary.resumed} resumed"
        line += f" of {summary.planned} · {self._tally.render()}"
        if summary.spend_usd:
            line += f" · ${summary.spend_usd:.4f}"
        self.console.print(Text(line, style=style))
        if summary.aborted_reason:
            self.console.print(
                Text(f"  {sym.warn} {summary.aborted_reason}", style="yellow")
            )
        self.console.print()

    @property
    def tally(self) -> Tally:
        return self._tally


__all__ = ["ProgressObserver", "Tally"]
