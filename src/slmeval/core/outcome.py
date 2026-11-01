"""What a failed execution *means*.

One execution can end badly in nine different ways and they are not
interchangeable. A cassette miss caused by a hallucinated order id is a model
finding. A cassette miss on a route the baseline never took is a hole in our
own recordings. A ``ZeroDivisionError`` inside the harness is our bug. Printed
in one column labelled "failed", those three numbers add up to something that
looks like evidence and is not.

So every exception that escapes an execution lands here, and the mapping is
explicit rather than inferred:

- Known failure types are classified by type, in order. ``CassetteMiss``,
  ``StepBudgetExceeded`` and the fault errors are all raised *by slmeval* and
  are all findings about the agent or the model, so type beats origin.
- Anything unrecognised is attributed by **where it was raised**: the deepest
  frame of its traceback. Inside our package and it is ours
  (:attr:`Outcome.HARNESS_ERROR`); anywhere else and it is the agent under test
  (:attr:`Outcome.CRASHED`).

:attr:`Outcome.HARNESS_ERROR` is the load-bearing one.
:data:`~slmeval.storage.models.NOT_THE_MODELS_FAULT` keeps it out of every rate
in the report, and the scorecard prints it above the table as a caveat on the
whole run. A harness bug quietly counted as a model failure is how a
measurement instrument stops being one.

Cancellation is deliberately *not* classified. ``Ctrl-C`` at 50% must leave the
remaining units with no row at all, because ``--resume`` picks up exactly what
has no terminal row - a unit recorded as failed-by-cancellation would be
skipped on resume and silently lost from the matrix.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Final

from slmeval.core.context import StepBudgetExceeded
from slmeval.errors import SlmevalError
from slmeval.providers.base import ProviderError, StructuredOutputError
from slmeval.sandbox.faults import FaultError
from slmeval.sandbox.tool import CassetteMiss, SandboxNotReady
from slmeval.storage.models import ExecStatus, Outcome

_PACKAGE_ROOT: Final = Path(__file__).resolve().parent.parent
"""The ``slmeval/`` directory. Frames under here are ours."""

ABORTING: Final = (asyncio.CancelledError, KeyboardInterrupt, SystemExit)
"""Never classified, always re-raised. See the module docstring."""


@dataclass(frozen=True, slots=True)
class Classification:
    """How one execution ended, in the vocabulary the store and report share."""

    status: ExecStatus
    outcome: Outcome
    error_class: str | None = None
    message: str | None = None

    detail: str | None = None
    """Extra context worth persisting: a miss verdict, a fault label. Kept out
    of ``message`` so the message stays the exception's own words."""

    @property
    def ok(self) -> bool:
        return self.status is ExecStatus.OK

    @property
    def harness_error(self) -> bool:
        """Our bug. Never attributed to the model or to the agent."""
        return self.outcome is Outcome.HARNESS_ERROR

    @property
    def judgeable(self) -> bool:
        """Whether this execution may enter a pass rate at all.

        ``harness_error`` and ``uncovered`` may not: one is our code failing,
        the other our recording missing. Counting either against a candidate
        would be a regression the harness invented.
        """
        return self.outcome not in {Outcome.HARNESS_ERROR, Outcome.UNCOVERED}

    @property
    def needs_evaluation(self) -> bool:
        """True when the verdict is still open - the agent returned, and
        whether what it returned is any good is the evaluators' call."""
        return self.outcome is Outcome.UNCLASSIFIED


OK: Final = Classification(status=ExecStatus.OK, outcome=Outcome.UNCLASSIFIED)
"""The agent returned. Nothing is known yet about whether it was right."""


def classify(exc: BaseException | None) -> Classification:
    """Turn whatever escaped an execution into a stored verdict.

    Raises on cancellation rather than classifying it, so a caller cannot
    accidentally swallow a ``Ctrl-C`` by routing it through here.
    """
    if exc is None:
        return OK
    if isinstance(exc, ABORTING):
        raise exc

    name = type(exc).__name__

    # A miss is two different findings wearing one exception, and the
    # difference is the whole reason the diagnosis exists. `uncovered` means
    # the model took a route we never recorded: our gap, excluded from its
    # pass rate rather than counted against it.
    if isinstance(exc, CassetteMiss):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.UNCOVERED if exc.uncovered else Outcome.CASSETTE_MISS,
            error_class=name,
            message=_first_line(exc),
            detail=exc.verdict,
        )

    # A runaway loop is a result, not a hang. Reported as a finding about the
    # candidate - usually the most interesting one in the table.
    if isinstance(exc, StepBudgetExceeded):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.STEP_BUDGET,
            error_class=name,
            message=str(exc),
            detail=f"{exc.used}/{exc.limit} steps",
        )

    if isinstance(exc, StructuredOutputError):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.SCHEMA_FAILURE,
            error_class=name,
            message=_first_line(exc),
            detail=",".join(exc.rungs_tried) or None,
        )

    if isinstance(exc, TimeoutError):
        # Covers asyncio.TimeoutError, which is TimeoutError on 3.11+.
        return Classification(
            status=ExecStatus.TIMEOUT,
            outcome=Outcome.TIMEOUT,
            error_class=name,
            message=str(exc) or "execution exceeded timeout_s",
        )

    # An injected fault the agent did not recover from. The point of injecting
    # it, and the numerator of the recovery rate - not a harness failure, even
    # though the harness raised it.
    if isinstance(exc, FaultError):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.TOOL_ERROR,
            error_class=name,
            message=str(exc),
            detail=f"{exc.kind.value} on {exc.tool_name}#{exc.call_index}",
        )

    # The provider never produced a response: the server was down, the key was
    # rejected, the model was not pulled. Kept out of the model's quality
    # numbers - "Ollama was not running" must never read as a regression.
    if isinstance(exc, ProviderError):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.PROVIDER_ERROR,
            error_class=name,
            message=_first_line(exc),
        )

    # The sandbox could not do its job while a benchmark was running, or the
    # context was lost crossing a thread boundary. Both are ours.
    if isinstance(exc, SandboxNotReady | LookupError):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.HARNESS_ERROR,
            error_class=name,
            message=_first_line(exc),
        )

    # A user-fixable error raised mid-execution is the agent's own contract
    # being broken - most often returning something that is not a final-state
    # mapping. Theirs to fix, so it is not a harness error, but it is also not
    # the model's doing.
    if isinstance(exc, SlmevalError):
        return Classification(
            status=ExecStatus.ERROR,
            outcome=Outcome.CRASHED,
            error_class=name,
            message=_first_line(exc),
        )

    ours = raised_inside_slmeval(exc.__traceback__)
    return Classification(
        status=ExecStatus.ERROR,
        outcome=Outcome.HARNESS_ERROR if ours else Outcome.CRASHED,
        error_class=name,
        message=_first_line(exc) or name,
        detail="raised inside slmeval" if ours else None,
    )


def raised_inside_slmeval(tb: TracebackType | None) -> bool:
    """Whether the deepest frame of a traceback is in our own package.

    The deepest frame, not any frame: every execution passes through our code
    on the way in, so "slmeval appears in the traceback" is true of literally
    every failure. Where it was actually raised is the only part that
    attributes blame.

    An exception with no traceback - constructed and passed in rather than
    raised - is not ours. Guessing in that direction would let a stray
    ``raise ValueError`` in a user's agent be filed as a harness bug and
    dropped from their results.
    """
    if tb is None:
        return False
    deepest = tb
    while deepest.tb_next is not None:
        deepest = deepest.tb_next
    try:
        filename = Path(deepest.tb_frame.f_code.co_filename).resolve()
    except (OSError, ValueError):  # pragma: no cover - exotic frames
        return False
    return _PACKAGE_ROOT in filename.parents


def _first_line(exc: BaseException) -> str:
    """The first line of an exception's message.

    ``CassetteMiss`` carries a whole diagnosis block as its message, which is
    right for a terminal and wrong for a database column and a table cell. The
    full block is reachable from the stored trace; this is the summary.
    """
    text = str(exc).strip()
    if not text:
        return ""
    return text.split("\n", 1)[0].strip()


__all__ = [
    "ABORTING",
    "OK",
    "Classification",
    "classify",
    "raised_inside_slmeval",
]
