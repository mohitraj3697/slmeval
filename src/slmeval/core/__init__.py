"""Runtime machinery: context, routing, telemetry.

Nothing in here imports the CLI, so the user's agent can depend on it without
pulling in Typer or Rich.
"""

from __future__ import annotations

from slmeval.core.context import (
    RunContext,
    SlmevalContextWarning,
    StepBudgetExceeded,
    current_node,
    current_node_path,
    get_context,
    no_context,
    node,
    require_context,
    set_context,
    use_context,
    warn_missing_context,
)
from slmeval.core.router import ChatModel, ModelRouter, chat_model
from slmeval.core.trace import (
    JsonStage,
    LLMCallTrace,
    NodeVisit,
    SchemaStatus,
    ToolCallTrace,
)

__all__ = [
    "ChatModel",
    "JsonStage",
    "LLMCallTrace",
    "ModelRouter",
    "NodeVisit",
    "RunContext",
    "SchemaStatus",
    "SlmevalContextWarning",
    "StepBudgetExceeded",
    "ToolCallTrace",
    "chat_model",
    "current_node",
    "current_node_path",
    "get_context",
    "no_context",
    "node",
    "require_context",
    "set_context",
    "use_context",
    "warn_missing_context",
]
