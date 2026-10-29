"""Loading and running the user's agent.

The harness knows nothing about the user's framework. It knows one coroutine:
takes the ``input`` dict from a golden case, returns the final state. Everything
else - which model answers which node, which tool call is replayed - arrives by
ambient context, which is why this file is about 200 lines and not an
integration per framework.

The ``generic`` adapter is this: import ``module:callable``, call it, hand back
the state. Node attribution comes from the user's own ``node()`` blocks, so a
user who marks none still gets a working benchmark - just one row in the
per-node table instead of five. That gradient matters for adoption: the tool has
to be useful before it is fully wired in.

The ``langgraph`` adapter, which wraps a compiled graph's nodes so the user
marks nothing at all, lands in week 3.

Two rules the context layer imposes on anything in this file:

1. **Set the context inside the task, never before it.** A ContextVar set before
   ``create_task`` is copied into the child task; set after, it is invisible.
   :func:`execute_case` is what a worker task runs first, so the context is
   installed there.
2. **A sync agent goes through ``asyncio.to_thread``,** not a direct call.
   Contextvars propagate through ``to_thread``, so the sandbox still sees the
   run - and a blocking agent stops stalling every other execution in the
   matrix.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import importlib
import inspect
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slmeval.config.models import ENTRYPOINT_RE
from slmeval.core.context import RunContext, set_context
from slmeval.errors import ConfigError


@dataclass(slots=True)
class Target:
    """The user's agent, loaded and ready to call."""

    entrypoint: str
    func: Callable[..., Any]
    is_async: bool

    @classmethod
    def load(cls, entrypoint: str, *, root: Path | str | None = None) -> Target:
        """Import ``module.path:callable``.

        ``root`` goes on ``sys.path`` first, because the entrypoint in
        ``eval.toml`` is written relative to the project - ``slmeval_target:run_case``
        means "the file next to eval.toml", and a user running ``slmeval run``
        from a parent directory should not have to think about ``PYTHONPATH``.
        """
        if not ENTRYPOINT_RE.match(entrypoint):
            raise ConfigError(
                f"target entrypoint {entrypoint!r} is not 'module.path:callable_name'",
                hint="example: entrypoint = \"slmeval_target:run_case\"",
            )
        module_name, _, attribute = entrypoint.partition(":")

        with _path_including(root):
            try:
                module = importlib.import_module(module_name)
            except ImportError as exc:
                raise ConfigError(
                    f"cannot import {module_name!r} for target {entrypoint!r}: {exc}",
                    hint=(
                        "paths are resolved against "
                        f"{Path(root).resolve() if root else Path.cwd()}. Check the module "
                        "name, and that any package it imports is installed in the same "
                        "environment as slmeval."
                    ),
                ) from exc
            except BaseException as exc:
                # The module imported but blew up on the way. That is the user's
                # own code failing, and conflating it with "module not found"
                # would send them looking in the wrong place.
                raise ConfigError(
                    f"importing {module_name!r} raised {type(exc).__name__}: {exc}",
                    hint="the target module itself failed to import; run it directly to see why",
                ) from exc

        func = getattr(module, attribute, None)
        if func is None:
            raise ConfigError(
                f"{module_name!r} has no {attribute!r}",
                hint=_suggest_callables(module, attribute),
            )
        if not callable(func):
            raise ConfigError(
                f"{entrypoint} is a {type(func).__name__}, not a callable",
                hint="the entrypoint takes the case input dict and returns the final state",
            )
        return cls(
            entrypoint=entrypoint,
            func=func,
            is_async=inspect.iscoroutinefunction(func),
        )

    async def __call__(self, case_input: Mapping[str, Any]) -> dict[str, Any]:
        """Run the agent once and normalize what it gives back."""
        payload = dict(case_input)
        if self.is_async:
            result = await self.func(payload)
        else:
            # to_thread, not a direct call: contextvars propagate through it, so
            # a synchronous agent's sandboxed tools still see the run - and one
            # blocking agent does not stall the whole matrix.
            result = await asyncio.to_thread(self.func, payload)
        return as_state(result, entrypoint=self.entrypoint)


def as_state(result: Any, *, entrypoint: str) -> dict[str, Any]:
    """Coerce an agent's return value into the final-state dict.

    Mappings, Pydantic models and dataclasses are all accepted, because an agent
    that already has a typed state object should not have to unwrap it to be
    measurable. Anything else is refused by name: the dataset's assertions are
    written against state *keys*, so a bare string would silently fail every
    one of them and look like a model failure.
    """
    if isinstance(result, BaseModel):
        return result.model_dump()
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return dataclasses.asdict(result)
    if isinstance(result, Mapping):
        return dict(result)
    raise ConfigError(
        f"{entrypoint} returned {type(result).__name__}, not a final-state mapping",
        hint=(
            "Return a dict (or a Pydantic model or dataclass) of the agent's final "
            "state. The dataset asserts on its keys."
        ),
    )


async def execute_case(
    ctx: RunContext,
    target: Target,
    case_input: Mapping[str, Any],
) -> dict[str, Any]:
    """Run one (candidate, case, attempt) with its context installed.

    The first statement of the worker task, by design: a ContextVar set before
    ``create_task`` is copied into the child, but one set *after* is invisible
    to it. So the task installs its own context, and sibling executions - the
    other 89 in the matrix - cannot see or corrupt it.
    """
    set_context(ctx)
    return await target(case_input)


def load_target(
    entrypoint: str,
    *,
    adapter: str = "generic",
    root: Path | str | None = None,
) -> Target:
    """Load the agent named by ``[target]`` in eval.toml."""
    if adapter == "generic":
        return Target.load(entrypoint, root=root)
    if adapter == "langgraph":
        raise ConfigError(
            "the langgraph adapter is not implemented yet (step 7, week 3)",
            hint=(
                'Use adapter = "generic" today: call slmeval.chat_model() inside your '
                "graph's nodes and wrap each one in `with node(\"name\")`. That gives "
                "the same per-node attribution, by hand."
            ),
        )
    raise ConfigError(
        f"unknown target adapter {adapter!r}",
        hint='adapter must be "generic" or "langgraph"',
    )


class _path_including:
    """Put ``root`` at the front of ``sys.path`` for the duration of an import.

    Restored afterwards, and only if we were the ones who added it: leaving a
    user's project directory on ``sys.path`` for the rest of the process would
    let it shadow real packages in ways that are maddening to debug.
    """

    def __init__(self, root: Path | str | None) -> None:
        self._entry = str(Path(root).resolve()) if root is not None else None
        self._added = False

    def __enter__(self) -> None:
        if self._entry is None or self._entry in sys.path:
            return
        sys.path.insert(0, self._entry)
        self._added = True

    def __exit__(self, *exc: object) -> None:
        if self._added:
            with contextlib.suppress(ValueError):
                sys.path.remove(self._entry)  # type: ignore[arg-type]
            self._added = False


def _suggest_callables(module: Any, wanted: str) -> str:
    from difflib import get_close_matches

    names = [
        name
        for name, value in vars(module).items()
        if callable(value) and not name.startswith("_")
    ]
    close = get_close_matches(wanted, names, n=1)
    if close:
        return f"did you mean {close[0]!r}?"
    if names:
        return f"callables in that module: {', '.join(sorted(names)[:8])}"
    return "that module defines no public callables"


__all__ = ["Target", "as_state", "execute_case", "load_target"]
