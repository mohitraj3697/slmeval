"""slmeval — benchmark agent workflows across small language models.

The public API is intentionally tiny. Integration is two touch points:

    from slmeval import chat_model, node, sandbox_tool

    async def run_case(case_input: dict) -> dict:
        with node("intent_router"):               # names the row in the scorecard
            llm = chat_model()                    # routed per candidate, per node
            decision = await llm.complete(msgs, schema=Routing)

`node` is a no-op outside a benchmark run, and `chat_model` falls through to
whichever model you name for production, so this is safe to leave in shipped
code:

    llm = chat_model(provider="openai", model="gpt-4o-mini")

The second touch point is `@sandbox_tool`, which records a tool call once and
then replays it for every candidate, so a 90-execution matrix never touches your
real APIs:

    @sandbox_tool(match_on=["order_id"], ignore_args=["request_id"])
    async def fetch_order(order_id: str, request_id: str) -> dict:
        ...

It is a no-op outside a benchmark too, so in production your tools do exactly
what they always did.
"""

from __future__ import annotations

__version__ = "0.1.0"

from slmeval.core.context import RunContext, node
from slmeval.core.router import ChatModel, chat_model
from slmeval.providers.base import (
    Completion,
    Msg,
    ProviderError,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
)
from slmeval.sandbox.tool import CassetteMiss, sandbox_tool

__all__ = [
    "CassetteMiss",
    "ChatModel",
    "Completion",
    "Msg",
    "ProviderError",
    "RunContext",
    "StructuredOutputError",
    "ToolCall",
    "ToolSpec",
    "__version__",
    "chat_model",
    "node",
    "sandbox_tool",
]
