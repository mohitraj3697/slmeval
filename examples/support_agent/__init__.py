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

Run it, offline, with no API key and no GPU:

    cd examples/support_agent
    slmeval record      # execute the real tools once, cache them
    slmeval run         # replay them against every candidate

`record` touches the in-memory systems in `erp.py`, not anything real. After it,
`run` makes no outbound calls at all.

The candidates in `eval.toml` are scripted `fake` models chosen to show the two
failure modes worth knowing about: one drops the `ORD-` prefix and misses the
cassette with a per-argument diff, and one gets the right answer but can only
emit JSON inside a prose preamble, so it passes *with help* - which the
scorecard says out loud.

`tests/test_example_agent.py` drives the same agent as a library, and is where
to look for how to run the harness from your own code.
"""

from __future__ import annotations

from .agent import Reply, Routing, run_case

__all__ = ["Reply", "Routing", "run_case"]
