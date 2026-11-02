"""Framework adapters: the same benchmark, without the user marking anything.

There are two, and the gradient between them is deliberate.

:mod:`slmeval.adapters.instrument` is the generic machinery - a wrapper that
turns one callable into a timed, attributed, traced node. It knows nothing
about any framework and is what the others are built from.

:mod:`slmeval.adapters.langgraph` wraps every node of a LangGraph builder, so
a user with a graph writes one line and marks nothing by hand.

The adapter is never where the measurement happens. Model routing comes from
the ambient context, tool interception from ``@sandbox_tool``, and both work
with no adapter at all - so an adapter can only ever *add* attribution. That is
what keeps a framework integration from becoming a second implementation of the
harness, which is the way integrations usually rot.
"""

from __future__ import annotations

from slmeval.adapters.instrument import instrument_node, state_keys
from slmeval.adapters.langgraph import instrument_langgraph

__all__ = ["instrument_langgraph", "instrument_node", "state_keys"]
