"""The provider gateway.

One protocol, one HTTP client covering every server that speaks
``/v1/chat/completions``, and the structured-output ladder on top of both. No
model framework is imported anywhere in here: wrapping one would mean measuring
the framework instead of the model.
"""

from __future__ import annotations

from slmeval.providers.base import (
    LADDER,
    ChatClient,
    ClientSpec,
    Completion,
    Msg,
    ProviderError,
    ProviderRequestError,
    ProviderServerError,
    ProviderUnreachable,
    RateLimited,
    RetryPolicy,
    Rung,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
    UnsupportedFeature,
    as_messages,
)
from slmeval.providers.fake import FakeClient, FakeTurn, register_script
from slmeval.providers.jsonx import JsonExtraction, JsonExtractionError, extract_json
from slmeval.providers.ladder import LadderClient, RungResult
from slmeval.providers.openai_compat import OpenAICompatClient
from slmeval.providers.registry import build_client, is_local, resolve_spec
from slmeval.providers.tokens import Usage, estimate_tokens

__all__ = [
    "LADDER",
    "ChatClient",
    "ClientSpec",
    "Completion",
    "FakeClient",
    "FakeTurn",
    "JsonExtraction",
    "JsonExtractionError",
    "LadderClient",
    "Msg",
    "OpenAICompatClient",
    "ProviderError",
    "ProviderRequestError",
    "ProviderServerError",
    "ProviderUnreachable",
    "RateLimited",
    "RetryPolicy",
    "Rung",
    "RungResult",
    "StructuredOutputError",
    "ToolCall",
    "ToolSpec",
    "UnsupportedFeature",
    "Usage",
    "as_messages",
    "build_client",
    "estimate_tokens",
    "extract_json",
    "is_local",
    "register_script",
    "resolve_spec",
]
