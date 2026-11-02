"""Turning one callable into an attributed node.

This is the piece every framework adapter is built from, and it is where the
blueprint's node wrapper had three bugs worth fixing by name, because each one
corrupts a number the scorecard prints.

**Exit through ``reset(token)``.** The blueprint restored the contextvar with
``set(token.old_value)``, which is not the contextvars API and goes wrong the
first time nodes nest - which is the first thing any real graph does. Entry and
exit both go through :func:`slmeval.core.context.entering_node` here, so there
is exactly one place that could get it wrong.

**Record in ``finally``, not per-``except``.** The blueprint's sync wrapper
caught the schema error and lost every other exception: a node that raised
``KeyError`` left no trace at all, so the denominator of every per-node rate
silently shrank. One ``finally``, one record, whatever happened - including
cancellation, which is how a benchmark's own ``timeout_s`` arrives.

**A crash is not a schema failure.** The blueprint recorded
``schema_valid=True`` on the generic-exception path, which quietly mixed
unrelated crashes into the one metric this tool exists to report. Schema
validity is tri-state, it lives on the model call
(:class:`~slmeval.core.trace.SchemaStatus`), and nothing here touches it: a
node that raised records an *error*, which is a different column.

Outside a benchmark the wrapper is a thin pass-through - it enters a node whose
trajectory nobody is collecting and calls the original. That is what makes it
safe to leave an instrumented graph in shipped code.
"""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable, Mapping
from functools import wraps
from typing import Any

from slmeval.core.context import entering_node
from slmeval.core.trace import NodeVisit

ERROR_CHARS = 300
"""How much of an exception message is kept.

Enough to recognise, not enough to be a liability: one of these is stored per
node per execution, and a 4 kB traceback times ninety executions is a trace
store that costs more than the thing it measures. The *type* is what
aggregates anyway.
"""


def state_keys(value: Any) -> tuple[str, ...]:
    """The keys of a state object, sorted, or ``()`` if it has none.

    Keys and never values. A trace store must not quietly become a copy of a
    customer's conversation state, and the shape is what carries the finding
    anyway: a node that was supposed to write ``refund_eligible`` and returned
    an empty dict is visible from the keys alone.
    """
    if isinstance(value, Mapping):
        return tuple(sorted(str(k) for k in value))
    # Off the class, not the instance: Pydantic deprecated the instance
    # attribute, and a benchmark that printed a DeprecationWarning per node per
    # execution would teach people to filter our warnings.
    owner = value if isinstance(value, type) else type(value)
    fields = getattr(owner, "model_fields", None) or getattr(owner, "__dataclass_fields__", None)
    if isinstance(fields, Mapping):
        return tuple(sorted(str(k) for k in fields))
    return ()


def _record(
    visit: NodeVisit | None, started: float, *, error: BaseException | None
) -> None:
    if visit is None:
        return
    visit.latency_ms = (time.perf_counter() - started) * 1000.0
    if error is not None:
        visit.error_type = type(error).__name__
        visit.error = str(error)[:ERROR_CHARS] or None


def instrument_node(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    """Wrap ``func`` so that calling it enters the node ``name``.

    Async-ness is preserved: a framework that awaits its nodes has to get a
    coroutine function back, and one that calls them synchronously must not be
    handed a coroutine it never awaits.
    """
    if inspect.iscoroutinefunction(func):

        @wraps(func)
        async def async_node(*args: Any, **kwargs: Any) -> Any:
            with entering_node(name) as visit:
                if visit is not None and args:
                    visit.input_keys = state_keys(args[0])
                started = time.perf_counter()
                failure: BaseException | None = None
                try:
                    result = await func(*args, **kwargs)
                except BaseException as exc:
                    failure = exc
                    raise
                else:
                    if visit is not None:
                        visit.output_keys = state_keys(result)
                    return result
                finally:
                    _record(visit, started, error=failure)

        return async_node

    @wraps(func)
    def sync_node(*args: Any, **kwargs: Any) -> Any:
        with entering_node(name) as visit:
            if visit is not None and args:
                visit.input_keys = state_keys(args[0])
            started = time.perf_counter()
            failure: BaseException | None = None
            try:
                result = func(*args, **kwargs)
            except BaseException as exc:
                failure = exc
                raise
            else:
                if visit is not None:
                    visit.output_keys = state_keys(result)
                return result
            finally:
                _record(visit, started, error=failure)

    return sync_node


__all__ = ["ERROR_CHARS", "instrument_node", "state_keys"]
