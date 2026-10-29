"""The runtime context layer.

A benchmark run is K candidates x N cases x R repeats, executing concurrently in
one process. Every one of those executions needs its own answer to "which model,
which cassette scope, which trace buffer" - and it needs it without a single
global, because a global is how candidate A's traces end up in candidate B's
scorecard.

``contextvars`` is the right primitive: a value set inside a task is visible to
everything that task awaits and invisible to its siblings. Three rules follow,
and all three are load-bearing:

1. **Every mutation goes through a context manager that calls ``reset(token)``.**
   Not ``set(old_value)`` - that is not the contextvars API and it corrupts node
   attribution the first time nodes nest.
2. **The context is set inside the task, never before it.** A ContextVar set in
   the parent before ``create_task`` is copied into the child; set after, it is
   invisible. So the runner creates the task and the first thing the task's
   coroutine does is :func:`set_context`.
3. **A missing context is loud.** Context propagates through ``asyncio.to_thread``
   but not through a bare ``Thread`` or ``ThreadPoolExecutor.submit``. If the
   sandbox cannot find a context it must say so, because the alternative is
   passing a tool call through to the live API. Silently charging someone's
   account during a benchmark is the worst bug this tool could have.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from slmeval.core.trace import LLMCallTrace, NodeVisit, ToolCallTrace

if TYPE_CHECKING:
    from slmeval.core.router import ModelRouter


class SlmevalContextWarning(UserWarning):
    """Emitted when harness context is missing somewhere it was needed."""


class StepBudgetExceeded(RuntimeError):
    """The agent exceeded `benchmark.max_steps`.

    A runaway loop is a result, not a crash: the runner classifies it as a
    failed execution rather than reporting it as a harness bug.
    """

    def __init__(self, used: int, limit: int) -> None:
        super().__init__(
            f"agent took {used} steps, exceeding max_steps={limit} - "
            "treated as a failed execution, not a harness error"
        )
        self.used = used
        self.limit = limit


@dataclass(slots=True)
class RunContext:
    """Everything one execution of the user's agent needs, plus its telemetry.

    Mutable and mutated constantly. One instance per (candidate, case, attempt).
    """

    run_id: str
    candidate_id: str
    case_id: str
    attempt: int = 0
    seed: int = 42
    tool_mode: str = "replay"
    max_steps: int = 25

    router: ModelRouter | None = None
    """Resolves node -> model. None outside a benchmark (production passthrough)."""

    llm_calls: list[LLMCallTrace] = field(default_factory=list)
    tool_calls: list[ToolCallTrace] = field(default_factory=list)
    trajectory: list[NodeVisit] = field(default_factory=list)
    """Node entries in order. Tier 2 trajectory distance is computed from this."""

    steps: int = 0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scope(self) -> str:
        """Cassette and trace scope. Case-level, not attempt-level, on purpose:
        every repeat of a case must replay the same recorded tool responses, or
        the repeats are measuring different things."""
        return self.case_id

    @property
    def node_names(self) -> list[str]:
        return [v.name for v in self.trajectory]

    def next_llm_seq(self) -> int:
        return len(self.llm_calls)

    def next_tool_seq(self) -> int:
        return len(self.tool_calls)

    def record_llm(self, trace: LLMCallTrace) -> None:
        self.llm_calls.append(trace)

    def record_tool(self, trace: ToolCallTrace) -> None:
        self.tool_calls.append(trace)

    def note_step(self) -> None:
        """Count one agent step and enforce the budget."""
        self.steps += 1
        if self.steps > self.max_steps:
            raise StepBudgetExceeded(self.steps, self.max_steps)

    def warn(self, message: str) -> None:
        """Attach a warning to this execution. Surfaced in the scorecard, so a
        degraded-but-passing run never looks clean."""
        if message not in self.warnings:
            self.warnings.append(message)


_context: ContextVar[RunContext | None] = ContextVar("slmeval_context", default=None)
_node_path: ContextVar[tuple[str, ...]] = ContextVar("slmeval_node_path", default=())

_warned: set[str] = set()

_NO_CONTEXT_HINT = (
    "slmeval found no run context. Either you are running your agent outside "
    "`slmeval run` (fine - this is a passthrough), or the context was lost "
    "crossing a thread boundary. contextvars propagate through "
    "`asyncio.to_thread` but NOT through a bare `threading.Thread` or "
    "`ThreadPoolExecutor.submit`. If your agent uses either, re-install the "
    "context inside the worker with `slmeval.core.context.use_context(ctx)`."
)


def set_context(ctx: RunContext) -> None:
    """Install the context for the current task.

    Call this as the first statement of a task coroutine. There is no matching
    reset: the task's context copy dies with the task. Use :func:`use_context`
    wherever a scope has to be restored afterwards.
    """
    _context.set(ctx)


@contextmanager
def use_context(ctx: RunContext) -> Iterator[RunContext]:
    """Install a context for the duration of a block, then restore the previous."""
    token = _context.set(ctx)
    try:
        yield ctx
    finally:
        _context.reset(token)


@contextmanager
def no_context() -> Iterator[None]:
    """Detach the context.

    Used by tests, and by the harness's own model calls: a Tier 3 judge call
    must not land in the agent's trace buffer and inflate its token count.
    """
    token = _context.set(None)
    try:
        yield
    finally:
        _context.reset(token)


def get_context() -> RunContext | None:
    """The active context, or None outside a benchmark run."""
    return _context.get()


def require_context(feature: str) -> RunContext:
    """The active context, or fail with the thread-loss diagnosis.

    For paths that cannot safely degrade - replaying a cassette, for instance.
    """
    ctx = _context.get()
    if ctx is None:
        raise LookupError(f"{feature} requires a benchmark run context.\n\n{_NO_CONTEXT_HINT}")
    return ctx


def warn_missing_context(feature: str) -> None:
    """Warn once per feature that no context was found.

    Once, not per call: a 10-step agent would otherwise emit hundreds of
    identical warnings and the user would stop reading them.
    """
    if feature in _warned:
        return
    _warned.add(feature)
    warnings.warn(
        f"{feature} ran without a benchmark context, so it was NOT intercepted."
        f"\n\n{_NO_CONTEXT_HINT}",
        SlmevalContextWarning,
        stacklevel=3,
    )


def reset_warnings() -> None:
    """Clear the warn-once memo. For tests only."""
    _warned.clear()


@contextmanager
def node(name: str) -> Iterator[tuple[str, ...]]:
    """Name the agent step inside this block.

    Nests: inside ``node("router")``, a ``node("classify")`` block has the path
    ``("router", "classify")``, so per-node attribution stays correct however
    you slice it. Entering a node appends to the trajectory, which is what the
    Tier 2 trajectory metrics are computed from.

        with node("intent_router"):
            decision = await chat_model().complete(msgs, schema=Routing)

    A no-op outside a benchmark run, so it is safe to leave in production code.
    """
    cleaned = name.strip()
    if not cleaned:
        raise ValueError("node() needs a non-empty name - it becomes a scorecard row")
    path = (*_node_path.get(), cleaned)
    token = _node_path.set(path)
    ctx = _context.get()
    if ctx is not None:
        ctx.trajectory.append(NodeVisit(path=path, seq=len(ctx.trajectory)))
    try:
        yield path
    finally:
        _node_path.reset(token)


def current_node() -> str | None:
    """The innermost node name, or None outside any node."""
    path = _node_path.get()
    return path[-1] if path else None


def current_node_path() -> tuple[str, ...]:
    """The full nesting path, outermost first."""
    return _node_path.get()


__all__ = [
    "RunContext",
    "SlmevalContextWarning",
    "StepBudgetExceeded",
    "current_node",
    "current_node_path",
    "get_context",
    "no_context",
    "node",
    "require_context",
    "reset_warnings",
    "set_context",
    "use_context",
    "warn_missing_context",
]
