"""A realistic support agent, wired for slmeval in two lines of its own code.

Four nodes, five sandboxed tools, no framework and no credentials. It exists to
be benchmarked, and to be the thing you read when you are deciding how to wire
up your own agent:

``erp.py``
    The systems the agent talks to, in memory. They return ``datetime``,
    ``Decimal`` and ``StrEnum`` values, because that is what makes them a real
    test of the cassette envelope.
``tools.py``
    Those calls behind ``@sandbox_tool``, with one matching decision per tool
    and a comment on why.
``agent.py``
    ``run_case(case_input) -> dict``, the contract the ``generic`` adapter
    expects. ``node()`` marks the four steps; ``chat_model()`` lets the harness
    decide which model answers each one.

What works today:

    slmeval validate --config examples/support_agent/eval.toml

`slmeval record` and `slmeval run` need the matrix runner, which lands in week 2;
until then the agent is driven from `tests/test_example_agent.py`, which records
its tool calls once and then replays the whole thing 200 times offline with
sockets blocked. That test is the honest demonstration, and it is also where to
look for how to drive the harness from your own code.
"""

from __future__ import annotations

from .agent import Reply, Routing, run_case

__all__ = ["Reply", "Routing", "run_case"]
