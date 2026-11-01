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
from slmeval.core.outcome import Classification, classify
from slmeval.core.router import ChatModel, ModelRouter, chat_model
from slmeval.core.runner import (
    MatrixRunner,
    Observer,
    RunOptions,
    RunPlan,
    RunSummary,
    UnitResult,
    WorkUnit,
    plan_matrix,
)
from slmeval.core.trace import (
    JsonStage,
    LLMCallTrace,
    NodeVisit,
    SchemaStatus,
    ToolCallTrace,
)
from slmeval.core.verdict import Evaluator, Metric, Verdict

__all__ = [
    "ChatModel",
    "Classification",
    "Evaluator",
    "JsonStage",
    "LLMCallTrace",
    "MatrixRunner",
    "Metric",
    "ModelRouter",
    "NodeVisit",
    "Observer",
    "RunContext",
    "RunOptions",
    "RunPlan",
    "RunSummary",
    "SchemaStatus",
    "SlmevalContextWarning",
    "StepBudgetExceeded",
    "ToolCallTrace",
    "UnitResult",
    "Verdict",
    "WorkUnit",
    "chat_model",
    "classify",
    "current_node",
    "current_node_path",
    "get_context",
    "no_context",
    "node",
    "plan_matrix",
    "require_context",
    "set_context",
    "use_context",
    "warn_missing_context",
]
