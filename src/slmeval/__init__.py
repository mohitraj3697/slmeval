"""slmeval — benchmark agent workflows across small language models.

The public API is intentionally tiny. Integration is two touch points, and one
of them is built:

    from slmeval import chat_model, node

    async def run_case(case_input: dict) -> dict:
        with node("intent_router"):               # names the row in the scorecard
            llm = chat_model()                    # routed per candidate, per node
            decision = await llm.complete(msgs, schema=Routing)

`node` is a no-op outside a benchmark run, and `chat_model` falls through to
whichever model you name for production, so this is safe to leave in shipped
code:

    llm = chat_model(provider="openai", model="gpt-4o-mini")

The second touch point, `@sandbox_tool`, arrives with the cassette store in
Step 8. Until then tool calls run live and are not recorded.
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

__all__ = [
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
]
