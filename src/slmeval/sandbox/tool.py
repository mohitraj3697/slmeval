"""``@sandbox_tool`` and the four interception modes.

This decorator is the first of the two touch points a user adds to their agent,
and the one that makes the whole benchmark possible: a recorded tool call is
comparable across models, fast, and safe to run 500 times, while a live one is
none of those things.

The modes:

========  ====================================================================
record    call the real tool, store what it returned
replay    serve from the store; a miss raises, it never falls through
auto      replay when recorded, record when not
live      no interception at all - the escape hatch, with a red banner
========  ====================================================================

The single most important rule here is that **no path falls back to the real
tool by accident**. In ``replay`` mode a miss is a finding and must surface as
one; silently calling the customer's live Stripe instead would spend their money
and invalidate the run, and it would do it without telling anyone. So every
degraded situation - no store, no case id, a lost context during a benchmark -
either raises or warns, and the only silent passthrough is the one case where
passing through is correct: there is no benchmark running at all, because the
user is in production.

Two things happen here that are the whole of Day 7:

**A miss is a diagnosis, not an exception string.** On a strict-replay miss the
store is searched for the nearest recorded call and the result is classified -
mangled argument, invented tool, wrong tool for the step, or a path nobody ever
recorded. The last of those is *our* gap, not the model's failure, and keeping
it separate is what stops the scorecard inflating itself. See
:mod:`slmeval.sandbox.diagnosis`.

**A fault is delivered after the response resolves, never before.** A cassette
miss is a finding about the model; a fault is a stimulus we chose. Resolving
first means an injected ``429`` can never mask a hallucinated argument - the
miss surfaces, and the fault is simply not reached. See
:mod:`slmeval.sandbox.faults`.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import os
import sys
import textwrap
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeVar, cast

from slmeval.core.context import (
    RunContext,
    benchmark_running_now,
    current_node,
    current_node_path,
    get_context,
    warn_missing_context,
)
from slmeval.core.trace import ToolCallTrace
from slmeval.sandbox.diagnosis import MissDiagnosis, diagnose
from slmeval.sandbox.faults import Delivery, Fault, apply_fault
from slmeval.sandbox.keys import CassetteKeys, MatchPolicy, MatchTier, PolicyError, declare
from slmeval.sandbox.store import CassetteStore, Hit

F = TypeVar("F", bound=Callable[..., Any])


class CassetteMiss(Exception):
    """A tool was called in a way that was never recorded.

    Not an :class:`~slmeval.errors.SlmevalError`, and deliberately not: in a
    benchmark this is a *result*. The runner catches it, classifies the
    execution as failed and keeps going, exactly as it would for any other way
    an agent can fail. Aborting the process on it would throw away the other 89
    executions and the finding itself.

    Its message **is** the diagnosis block from
    :mod:`slmeval.sandbox.diagnosis` - nearest recorded call, per-argument
    diff, verdict - so one finding reads the same in a terminal, a log, a
    pytest failure and a traceback, with nothing to re-parse out of a string.
    :attr:`diagnosis` carries the same thing structured, for the runner to
    persist and the scorecard to aggregate.
    """

    def __init__(
        self,
        *,
        case_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
        keys: CassetteKeys,
        diagnosis: MissDiagnosis | None = None,
        recorded_calls: int = 0,
        node: str | None = None,
        candidate_id: str | None = None,
    ) -> None:
        if diagnosis is not None:
            super().__init__(diagnosis.render())
            recorded_calls = diagnosis.recorded_calls
        else:
            # Only reachable when something builds this by hand: diagnosing
            # needs a store to search, and a miss is still a miss without one.
            where = " · ".join(
                part for part in (case_id, candidate_id, f"node={node}" if node else None) if part
            )
            super().__init__(
                f"{where} · CASSETTE_MISS\n"
                f"  tool: {tool_name}\n"
                f"  called with: {dict(arguments)!r}\n"
                "  no cassette store was available, so this miss could not be diagnosed"
            )
        self.case_id = case_id
        self.tool_name = tool_name
        self.arguments = dict(arguments)
        self.keys = keys
        self.diagnosis = diagnosis
        self.recorded_calls = recorded_calls
        self.node = node
        self.candidate_id = candidate_id

    @property
    def verdict(self) -> str | None:
        """``hallucinated_parameter``, ``unrecorded_valid_path``, and so on."""
        return self.diagnosis.verdict.value if self.diagnosis else None

    @property
    def uncovered(self) -> bool:
        """True when this miss is a hole in our coverage rather than a model
        failure, so the gate can keep it out of the candidate's pass rate."""
        return self.diagnosis is not None and self.diagnosis.uncovered


class SandboxNotReady(RuntimeError):
    """A benchmark is running but the sandbox cannot do its job.

    A harness bug, so it keeps its traceback. It raises rather than degrading
    because every degradation available here is worse than stopping.
    """


@dataclass(slots=True)
class SandboxedTool:
    """The registry entry for one decorated tool."""

    name: str
    policy: MatchPolicy
    version: str
    qualname: str
    is_async: bool
    func: Callable[..., Any] = field(repr=False)
    signature: inspect.Signature | None = field(default=None, repr=False)

    def bind(self, args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> dict[str, Any]:
        """Positional arguments, resolved to names.

        The key is computed from named arguments, so ``f("ORD-1")`` and
        ``f(order_id="ORD-1")`` must be the same call. Defaults are *not*
        applied here - the policy fills them, and the trace should show what the
        caller actually passed.
        """
        if self.signature is None:
            return dict(kwargs)
        # A TypeError here is the caller invoking the tool wrongly, which the
        # real tool would raise too. Let it through unchanged.
        bound = self.signature.bind(*args, **kwargs)
        return {
            name: value
            for name, value in bound.arguments.items()
            if name not in {"self", "cls"}
        }


REGISTRY: dict[str, SandboxedTool] = {}


def registered_tools() -> Mapping[str, SandboxedTool]:
    """Every sandboxed tool known to this process.

    The miss diagnosis needs it to tell `hallucinated_tool` (a name that is not
    a tool at all) from `unrecorded_valid_path` (a real tool the baseline never
    called). Only the decorator can populate it, so it lives here.
    """
    return REGISTRY


def clear_registry() -> None:
    """For tests. A leaked registration would leak across test modules."""
    REGISTRY.clear()


def _register(tool: SandboxedTool) -> None:
    existing = REGISTRY.get(tool.name)
    if existing is not None and existing.qualname != tool.qualname:
        raise PolicyError(
            f"two different tools are both named {tool.name!r}: "
            f"{existing.qualname} and {tool.qualname}",
            hint=(
                "Cassettes are keyed by tool name, so these two would share - and "
                "answer - each other's recordings. Pass name= to one of them."
            ),
        )
    REGISTRY[tool.name] = tool


def _source_version(fn: Callable[..., Any]) -> str:
    """A hash of the tool's source, used to flag stale cassettes.

    Not part of the cassette key: an edited tool should be *reported* as stale
    by a warning on replay, not turn into a wall of misses that reads
    like the model fell apart. Unavailable source (a C function, a REPL) yields
    an empty version, which disables staleness checking rather than guessing.
    """
    try:
        source = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return ""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


_LIVE_BANNER_SHOWN = False


def _live_banner(tool_name: str) -> None:
    """Say out loud that real APIs are being hit.

    ``live`` mode exists because some tool genuinely cannot be recorded, but a
    user who reaches for it by accident - or inherits it from a committed
    ``eval.toml`` - must not discover it from their cloud bill. Printed once,
    to stderr, so it cannot be mistaken for part of the scorecard.
    """
    global _LIVE_BANNER_SHOWN
    if _LIVE_BANNER_SHOWN:
        return
    _LIVE_BANNER_SHOWN = True
    message = (
        f"  tool_mode = \"live\": {tool_name} and every other sandboxed tool will hit "
        "your REAL APIs.\n"
        "  Nothing is recorded, nothing is replayed, and side effects are real."
    )
    colour = not os.environ.get("NO_COLOR") and sys.stderr.isatty()
    if colour:
        message = f"\033[1;31m{message}\033[0m"
    print(f"\n{message}\n", file=sys.stderr)


def reset_live_banner() -> None:
    """For tests, and for a second run in one process."""
    global _LIVE_BANNER_SHOWN
    _LIVE_BANNER_SHOWN = False


def _store_for(ctx: RunContext, tool_name: str) -> CassetteStore:
    if ctx.cassettes is None:
        raise SandboxNotReady(
            f"tool {tool_name!r} was called in tool_mode={ctx.tool_mode!r} but no "
            "cassette store was attached to the run context.\n"
            "  The sandbox will not fall through to the real tool to cover for "
            "this: in a benchmark that means spending real money on an API call "
            "whose result is then thrown away."
        )
    return ctx.cassettes


def _trace(
    ctx: RunContext,
    tool: SandboxedTool,
    arguments: Mapping[str, Any],
) -> ToolCallTrace:
    return ToolCallTrace(
        seq=ctx.next_tool_seq(),
        name=tool.name,
        node=current_node(),
        node_path=current_node_path(),
        arguments=dict(arguments),
        tool_version=tool.version or None,
    )


def _note_degraded(ctx: RunContext, tool: SandboxedTool, hit: Hit) -> None:
    """Attach a warning for anything but an exact match.

    A run whose numbers depend on our normalizers covering for the model must
    not look clean in the scorecard.
    """
    if hit.tier is MatchTier.NORMALIZED:
        ctx.warn(
            f"{tool.name}: matched a cassette only after normalizing arguments "
            "(formatting differed from the recording)"
        )
    elif hit.tier is MatchTier.SUBSET:
        ctx.warn(
            f"{tool.name}: matched only on {', '.join(tool.policy.match_on or ())} - "
            "the model got the other arguments wrong"
        )
    if (
        tool.version
        and hit.cassette.tool_version
        and tool.version != hit.cassette.tool_version
    ):
        ctx.warn(
            f"{tool.name}: replayed a cassette recorded for a different version of "
            "this tool - re-record it with `slmeval record`"
        )


def _replay(
    ctx: RunContext,
    tool: SandboxedTool,
    arguments: Mapping[str, Any],
    trace: ToolCallTrace,
) -> tuple[bool, Any]:
    """Try the store. Returns ``(hit, response)``.

    Raises :class:`CassetteMiss` only in strict ``replay`` mode; ``auto`` wants
    the miss back so it can record.
    """
    store = _store_for(ctx, tool.name)
    keys = tool.policy.compute(ctx.scope, arguments)
    hit = store.lookup(keys)
    if hit is not None:
        trace.cassette_key = hit.cassette.key
        trace.match_tier = hit.tier.value
        trace.outcome = "replayed"
        _note_degraded(ctx, tool, hit)
        return True, hit.response

    if ctx.tool_mode == "replay":
        trace.outcome = "miss"
        trace.cassette_key = keys.exact
        trace.error = "CASSETTE_MISS"
        finding = diagnose(
            case_id=ctx.scope,
            tool_name=tool.name,
            arguments=arguments,
            store=store,
            candidate_id=ctx.candidate_id,
            node=current_node(),
            node_path=current_node_path(),
        )
        trace.miss_verdict = finding.verdict.value
        if finding.uncovered:
            # An uncovered path is a coverage gap, so it is surfaced as a
            # warning on the execution as well as in the block. Without this a
            # user reading only the scorecard would see a failed case and
            # assume the model broke.
            ctx.warn(
                f"{tool.name}: {finding.verdict.value} - the baseline never recorded this "
                "call, so it is uncovered rather than wrong "
                "(`slmeval record --fill-gaps`)"
            )
        raise CassetteMiss(
            case_id=ctx.scope,
            tool_name=tool.name,
            arguments=arguments,
            keys=keys,
            diagnosis=finding,
            node=current_node(),
            candidate_id=ctx.candidate_id,
        )
    return False, None


def _after_real_call(
    ctx: RunContext,
    tool: SandboxedTool,
    arguments: Mapping[str, Any],
    trace: ToolCallTrace,
    result: Any,
) -> Any:
    """Record a real call's result, in ``record`` and ``auto`` modes."""
    store = _store_for(ctx, tool.name)
    keys = tool.policy.compute(ctx.scope, arguments)
    # An unserializable return raises out of here, which fails the run loudly at
    # record time. That is the point: coercing it would replay a different type
    # than it recorded, and the user would spend a day on it instead of a minute.
    hit = store.record(
        keys,
        arguments=arguments,
        response=result,
        tool_version=tool.version,
        dataset_hash=str(ctx.metadata.get("dataset_hash", "")),
    )
    trace.cassette_key = hit.cassette.key
    trace.match_tier = MatchTier.EXACT.value
    trace.outcome = "recorded"
    return result


def _deliver_fault(
    ctx: RunContext,
    tool: SandboxedTool,
    fault: Fault,
    response: Any,
    trace: ToolCallTrace,
) -> Any:
    """Deliver one fault: return a mangled payload, or raise.

    Called *after* the response has been resolved, so an injected fault can
    never mask a cassette miss - the miss is a finding about the model and the
    fault is a stimulus we chose, and when both apply to one call the finding
    wins.

    The delay happened at the call site, because sleeping correctly is the one
    thing this function cannot do for both the sync and the async wrapper.
    """
    trace.fault = fault.label()
    trace.outcome = "fault"
    node = current_node()

    if fault.delivery is Delivery.RAISE:
        error = fault.error()
        ctx.record_fault(fault, node=node)
        # Set before raising, so the wrapper's `except BaseException` sees an
        # error already attributed to the fault and does not overwrite it with
        # a generic one.
        trace.error = f"{type(error).__name__}: {error}"
        raise error

    payload, applied = apply_fault(fault, response)
    ctx.record_fault(fault, node=node, applied=applied)
    if not applied:
        # Recorded as delivered-but-ineffective rather than dropped: a vector
        # that quietly omitted it would compare equal to a run that never
        # scheduled it, and the parity check would pass on a real difference.
        ctx.warn(
            f"{tool.name}: {fault.kind.value} could not be shaped onto a "
            f"{type(response).__name__} response, so the call was counted as faulted "
            "but the payload is unchanged"
        )
    return payload


async def _fault_async(
    ctx: RunContext,
    tool: SandboxedTool,
    fault: Fault,
    response: Any,
    trace: ToolCallTrace,
) -> Any:
    if fault.delay_s:
        # A real wait, because a timeout that takes no time never trips the
        # agent's own timeout and the measurement would be a lie. Scaled by
        # the schedule's `delay_scale`, which the test suite sets to zero.
        await asyncio.sleep(fault.delay_s)
    return _deliver_fault(ctx, tool, fault, response, trace)


def _fault_sync(
    ctx: RunContext,
    tool: SandboxedTool,
    fault: Fault,
    response: Any,
    trace: ToolCallTrace,
) -> Any:
    if fault.delay_s:
        time.sleep(fault.delay_s)
    return _deliver_fault(ctx, tool, fault, response, trace)


def sandbox_tool(
    fn: F | None = None,
    *,
    match_on: list[str] | tuple[str, ...] | None = None,
    normalize: Mapping[str, str] | None = None,
    ignore_args: list[str] | tuple[str, ...] | None = None,
    version: str | None = None,
    name: str | None = None,
) -> Any:
    """Record this tool once, then replay it forever.

        @sandbox_tool(
            match_on=["order_id"],
            normalize={"order_id": "upper_strip", "query": "sql"},
            ignore_args=["request_id", "timestamp"],
        )
        async def fetch_order_from_erp(order_id: str, request_id: str) -> dict:
            ...

    ``match_on``
        The arguments that identify the call. They form the last rung of the
        lookup cascade, so a call that gets *these* right and something else
        wrong is reported as such instead of passing or missing outright.
    ``normalize``
        Named normalizers, per argument, so a formatting difference is not
        reported as a hallucination. See
        :mod:`slmeval.sandbox.normalizers`.
    ``ignore_args``
        Arguments that never identify a call - a request id, a timestamp. Left
        in, they would make every call unique and every replay a miss.
    ``version``
        Declare it when you want to invalidate cassettes deliberately.
        Otherwise a hash of the tool's source is used to *flag* staleness.

    A no-op outside a benchmark run: with no run in progress the real tool is
    called, untouched, so this is safe to leave in production code.
    """

    def decorate(func: F) -> F:
        tool_name = name or func.__name__
        policy = declare(
            func,
            tool_name=tool_name,
            match_on=match_on,
            normalize=normalize,
            ignore_args=ignore_args,
            tool_version=version or "",
        )
        try:
            signature: inspect.Signature | None = inspect.signature(func)
        except (TypeError, ValueError):
            signature = None

        tool = SandboxedTool(
            name=tool_name,
            policy=policy,
            version=version or _source_version(func),
            qualname=f"{func.__module__}.{func.__qualname__}",
            is_async=inspect.iscoroutinefunction(func),
            func=func,
            signature=signature,
        )
        _register(tool)

        if tool.is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_context()
                if ctx is None:
                    _warn_if_benchmarking(tool_name)
                    return await func(*args, **kwargs)
                arguments = tool.bind(args, kwargs)
                ctx.note_step()
                if ctx.tool_mode == "live":
                    _live_banner(tool_name)
                    return await _run_traced(ctx, tool, arguments, func(*args, **kwargs))

                trace = _trace(ctx, tool, arguments)
                # This tool's own call index, not a global counter, and the
                # fault is looked up before the call resolves so a raise-
                # delivered one can be applied to a replayed response without
                # a second trip through the schedule.
                fault = ctx.fault_for(tool.name, ctx.next_tool_index(tool.name))
                try:
                    if ctx.tool_mode != "record":
                        # `record` does not consult the store: it executes the
                        # real tool and overwrites. Looking first would make
                        # `record` silently behave like `auto`, so re-recording
                        # a case after changing a fixture would appear to do
                        # nothing.
                        replayed, response = _replay(ctx, tool, arguments, trace)
                        if replayed:
                            if fault is not None:
                                return await _fault_async(ctx, tool, fault, response, trace)
                            return response
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    if trace.error is None:
                        trace.outcome = "error"
                        trace.error = f"{type(exc).__name__}: {exc}"
                    raise
                else:
                    recorded = _after_real_call(ctx, tool, arguments, trace, result)
                    if fault is not None:
                        # The cassette already holds the clean response - it was
                        # written above - so `auto` mode faults what the agent
                        # sees without poisoning what the store keeps.
                        return await _fault_async(ctx, tool, fault, recorded, trace)
                    return recorded
                finally:
                    # In `finally`: every path leaves exactly one trace, or the
                    # denominator of every per-tool rate in the report is wrong.
                    ctx.record_tool(trace)

            wrapper: Callable[..., Any] = async_wrapper
        else:

            @functools.wraps(func)
            def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                ctx = get_context()
                if ctx is None:
                    _warn_if_benchmarking(tool_name)
                    return func(*args, **kwargs)
                arguments = tool.bind(args, kwargs)
                ctx.note_step()
                if ctx.tool_mode == "live":
                    _live_banner(tool_name)
                    return _run_traced_sync(ctx, tool, arguments, func, args, kwargs)

                trace = _trace(ctx, tool, arguments)
                fault = ctx.fault_for(tool.name, ctx.next_tool_index(tool.name))
                try:
                    if ctx.tool_mode != "record":
                        replayed, response = _replay(ctx, tool, arguments, trace)
                        if replayed:
                            if fault is not None:
                                return _fault_sync(ctx, tool, fault, response, trace)
                            return response
                    result = func(*args, **kwargs)
                except BaseException as exc:
                    if trace.error is None:
                        trace.outcome = "error"
                        trace.error = f"{type(exc).__name__}: {exc}"
                    raise
                else:
                    recorded = _after_real_call(ctx, tool, arguments, trace, result)
                    if fault is not None:
                        return _fault_sync(ctx, tool, fault, recorded, trace)
                    return recorded
                finally:
                    ctx.record_tool(trace)

            wrapper = sync_wrapper

        wrapper.slmeval_tool = tool  # type: ignore[attr-defined]
        return cast("F", wrapper)

    if fn is not None:
        return decorate(fn)
    return decorate


def _warn_if_benchmarking(tool_name: str) -> None:
    """No context: production passthrough, or a context lost crossing a thread?

    Only the second is a problem, and only it gets a warning - a warning on
    every production tool call is a warning nobody reads.
    """
    if benchmark_running_now():
        warn_missing_context(f"@sandbox_tool {tool_name!r}")


async def _run_traced(
    ctx: RunContext,
    tool: SandboxedTool,
    arguments: Mapping[str, Any],
    awaitable: Any,
) -> Any:
    """`live` mode: no interception, but still traced and still counted."""
    trace = _trace(ctx, tool, arguments)
    trace.outcome = "live"
    try:
        return await awaitable
    except BaseException as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ctx.record_tool(trace)


def _run_traced_sync(
    ctx: RunContext,
    tool: SandboxedTool,
    arguments: Mapping[str, Any],
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    trace = _trace(ctx, tool, arguments)
    trace.outcome = "live"
    try:
        return func(*args, **dict(kwargs))
    except BaseException as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ctx.record_tool(trace)


__all__ = [
    "REGISTRY",
    "CassetteMiss",
    "MissDiagnosis",
    "SandboxNotReady",
    "SandboxedTool",
    "clear_registry",
    "registered_tools",
    "reset_live_banner",
    "sandbox_tool",
]
