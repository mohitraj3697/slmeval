"""The LangGraph adapter: one line, and every node is attributed.

    import slmeval
    from langgraph.graph import StateGraph

    builder = slmeval.instrument_langgraph(StateGraph(State))
    builder.add_node("intent_router", route)
    builder.add_node("order_lookup", lookup)
    graph = builder.compile()

Nothing else changes. The user writes no ``with node(...)`` blocks and the
scorecard still has a row per node - which is the whole value of a framework
adapter, because per-node attribution is the part nobody will hand-annotate
across a thirty-node graph.

**Instrument the builder, then add the nodes.** That ordering is the supported
path and the reason this adapter is boring: it replaces the builder's own
``add_node``, so each node's callable is wrapped *before* LangGraph ever sees
it, and from that point on the library's internals are none of our business.
Calling it after the nodes are added also works - the registered callables are
wrapped in place - but that path has to know how a given LangGraph version
stores a node, and it says so loudly rather than wrapping nothing when it does
not recognise one. An adapter that quietly attributed every call to ``None``
would print a flat scorecard, which reads as a finding rather than as a bug.

LangGraph itself is never imported here. It is an optional extra
(``pip install "slmeval-cli[langgraph]"``), the core must install in seconds
without a model framework, and the only thing this file needs to know is that
a builder has named nodes holding callables.

The adapter adds attribution and nothing else. Model routing still comes from
the ambient context and tool interception from ``@sandbox_tool``, both of which
work with no adapter at all - so an instrumented graph and a hand-marked one
produce the same trajectory. ``examples/support_agent`` is benchmarked both
ways to keep that true.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from slmeval.adapters.instrument import instrument_node
from slmeval.errors import ConfigError

MARK = "_slmeval_instrumented"
"""Set on anything this module has wrapped, so a second call is a no-op.

``instrument_langgraph(builder)`` twice - in a test fixture and again in the
graph's own factory, say - would otherwise enter every node twice, and Tier 2
would read the doubled trajectory as the agent looping.
"""

#: Where a node's callable lives, in the versions this has been built against.
#: ``func``/``afunc`` are LangGraph's own ``RunnableCallable``; ``invoke``/
#: ``ainvoke`` are the generic Runnable surface, wrapped only as a fallback.
CALLABLE_ATTRS = ("func", "afunc", "invoke", "ainvoke")

#: Where a builder keeps the runnable for a node.
NODE_ATTRS = ("runnable", "action", "bound")


def instrument_langgraph(builder: Any, *, prefix: str = "") -> Any:
    """Wrap every node of a LangGraph builder so each one names itself.

    Returns the same builder, so it reads as a pipeline step either way:

        builder = slmeval.instrument_langgraph(StateGraph(State))
        ...
        graph = builder.compile()

    ``prefix`` namespaces the node names, for an agent that composes two graphs
    with a ``plan`` node in each. Without it the two share one scorecard row
    and "which plan node is failing" has no answer.
    """
    if not hasattr(builder, "add_node"):
        raise ConfigError(
            f"instrument_langgraph() needs a graph builder, not {type(builder).__name__}",
            hint=(
                "Pass the StateGraph, not the compiled graph: a compiled "
                "graph's nodes are already bound, and wrapping them would not "
                "change what runs."
            ),
        )

    _patch_add_node(builder, prefix)
    existing = _existing_nodes(builder)
    for name, holder in existing:
        _wrap_node(_named(name, prefix), holder)

    # The failure worth reporting is "nothing here was even a callable", not
    # "nothing was wrapped": a second call finds every node already wrapped,
    # which is success.
    if existing and not any(_reachable(holder) for _, holder in existing):
        raise ConfigError(
            f"instrument_langgraph() could not reach the callable behind "
            f"{len(existing)} already-registered node(s)",
            hint=(
                "Instrument the builder *before* adding nodes - "
                "`builder = slmeval.instrument_langgraph(StateGraph(State))` - "
                "which needs to know nothing about how your LangGraph version "
                'stores a node. Or use adapter = "generic": call '
                "slmeval.chat_model() in your node functions and wrap each in "
                '`with node("name")`, for the same attribution by hand.'
            ),
        )
    return builder


# --------------------------------------------------------------------------- #
# Before the nodes are added: replace add_node
# --------------------------------------------------------------------------- #


def _patch_add_node(builder: Any, prefix: str) -> None:
    """Wrap each node's callable on its way in.

    The supported path, and the one that needs no knowledge of LangGraph's
    internals: ``add_node(name, action)`` is the public API, and the action is
    still the user's own function when it arrives here.
    """
    original = builder.add_node
    if getattr(original, MARK, False):
        return

    def add_node(node: Any = None, action: Any = None, **kwargs: Any) -> Any:
        name, func, rest = _split_add_node_args(node, action)
        if func is not None:
            name = name or getattr(func, "__name__", None)
            if name:
                func = _instrument(_named(str(name), prefix), func)
        if rest is None:
            return original(name, func, **kwargs) if name else original(func, **kwargs)
        return original(*rest, **kwargs)

    add_node._slmeval_instrumented = True  # type: ignore[attr-defined]
    builder.add_node = add_node


def _split_add_node_args(
    node: Any, action: Any
) -> tuple[str | None, Callable[..., Any] | None, tuple[Any, ...] | None]:
    """Make sense of ``add_node``'s two accepted shapes.

    ``add_node("name", func)`` and ``add_node(func)`` - where the node takes
    the function's own name - are both public, and a wrapper that handled only
    the first would break every graph written in the second style. Anything
    else is passed through untouched rather than guessed at.
    """
    if action is None and callable(node):
        return None, node, None
    if isinstance(node, str) and callable(action):
        return node, action, None
    return None, None, (node, action)


# --------------------------------------------------------------------------- #
# After the nodes are added: wrap what is registered
# --------------------------------------------------------------------------- #


def _existing_nodes(builder: Any) -> list[tuple[str, Any]]:
    nodes = getattr(builder, "nodes", None)
    if not isinstance(nodes, dict):
        return []
    return [(str(name), holder) for name, holder in nodes.items()]


def _wrap_node(name: str, holder: Any) -> bool:
    """Instrument one registered node in place. True if anything was wrapped.

    The holder is a node spec whose ``runnable`` holds the callable, or - in
    the simplest builders, and in the stub this is tested against - the
    callable itself. The runnable is wrapped *in place* rather than replaced,
    because LangGraph's executor calls methods on it that a bare function does
    not have.
    """
    for attribute in NODE_ATTRS:
        runnable = getattr(holder, attribute, None)
        if runnable is not None and _wrap_in_place(name, runnable):
            return True
    if callable(holder):
        return _wrap_in_place(name, holder)
    return False


def _wrap_in_place(name: str, runnable: Any) -> bool:
    if getattr(runnable, MARK, False):
        return False
    wrapped = False
    for attribute in CALLABLE_ATTRS:
        current = getattr(runnable, attribute, None)
        if current is None or not callable(current):
            continue
        if _set(runnable, attribute, _instrument(name, current)):
            wrapped = True
    if wrapped:
        _set(runnable, MARK, True)
    return wrapped


def _instrument(name: str, func: Callable[..., Any]) -> Callable[..., Any]:
    if getattr(func, MARK, False):
        return func
    wrapped = instrument_node(name, func)
    _set(wrapped, MARK, True)
    return wrapped


def _set(obj: Any, attribute: str, value: Any) -> bool:
    """Assign, through whatever the object allows. False if it allows nothing."""
    try:
        setattr(obj, attribute, value)
    except (AttributeError, TypeError, ValueError):
        try:
            object.__setattr__(obj, attribute, value)
        except (AttributeError, TypeError, ValueError):
            return False
    return True


def _reachable(holder: Any) -> bool:
    """Whether this node holds a callable the adapter could reach at all."""
    for attribute in NODE_ATTRS:
        runnable = getattr(holder, attribute, None)
        if runnable is None:
            continue
        if any(callable(getattr(runnable, a, None)) for a in CALLABLE_ATTRS):
            return True
    return callable(holder)


def _named(name: str, prefix: str) -> str:
    return f"{prefix}{name}" if prefix else name


__all__ = ["CALLABLE_ATTRS", "MARK", "NODE_ATTRS", "instrument_langgraph"]
