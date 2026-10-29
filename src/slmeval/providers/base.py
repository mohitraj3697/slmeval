"""Provider-facing types and the one protocol every backend implements.

The gateway talks HTTP directly instead of going through a model framework, for
three reasons that all matter to the product:

- A provider quirk becomes a three-line fix here rather than a wait for an
  upstream release. SLM structured-output behaviour is *made* of quirks.
- ``raw`` stays available, so a failure report can show what the model actually
  emitted. That block is the output people share.
- The core install stays at four dependencies. Wrapping a framework would mean
  measuring the framework, not the model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from slmeval.core.trace import JsonStage

ModelT = TypeVar("ModelT", bound=BaseModel)

Role = Literal["system", "user", "assistant", "tool"]

Rung = Literal["tools", "json_schema", "json_mode", "prompt"]

LADDER: tuple[Rung, ...] = ("tools", "json_schema", "json_mode", "prompt")
"""The structured-output ladder, strongest mechanism first.

Walked in order when ``structured_output = "auto"``. A user can pin a single
rung so that a published benchmark is honest and comparable - a model that only
succeeds at the `prompt` rung should not be flattered by a harness that silently
tried native tool-calling first.
"""


@dataclass(slots=True)
class ToolCall:
    """A tool call the model asked for."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_arguments: str | None = None
    """Exactly what the model emitted, before parsing. SLMs produce broken JSON
    in tool arguments too, and the failure report needs the original."""
    arguments_stage: JsonStage | None = None
    """Which extraction rung parsed the arguments. Anything past DIRECT is a
    leniency event, counted the same way as a lenient schema parse."""


@dataclass(slots=True)
class Msg:
    """One chat message.

    A dataclass rather than a dict so a typo in ``role`` fails here instead of
    as an opaque 400 from a provider forty minutes into a run.
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            out["content"] = self.content
        elif not self.tool_calls:
            # Some servers reject a message with neither content nor tool_calls.
            out["content"] = ""
        if self.tool_calls:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.raw_arguments
                        if tc.raw_arguments is not None
                        else json.dumps(tc.arguments),
                    },
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            out["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            out["name"] = self.name
        return out


MessageLike = Msg | Mapping[str, Any]


def as_messages(messages: Sequence[MessageLike] | str) -> list[Msg]:
    """Accept a prompt string, dicts, or Msg objects.

    Users arrive from every framework; making them convert their message list
    before they can try the tool is friction for no gain.
    """
    if isinstance(messages, str):
        return [Msg(role="user", content=messages)]
    out: list[Msg] = []
    for m in messages:
        if isinstance(m, Msg):
            out.append(m)
            continue
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            raise ValueError(f"message role must be system/user/assistant/tool, got {role!r}")
        raw_calls = m.get("tool_calls") or []
        calls: list[ToolCall] = []
        for c in raw_calls:
            if isinstance(c, ToolCall):
                calls.append(c)
                continue
            fn = c.get("function", c)
            args = fn.get("arguments")
            parsed = args if isinstance(args, dict) else {}
            calls.append(
                ToolCall(
                    id=str(c.get("id", f"call_{len(calls)}")),
                    name=str(fn.get("name", "")),
                    arguments=parsed,
                    raw_arguments=args if isinstance(args, str) else None,
                )
            )
        out.append(
            Msg(
                role=role,
                content=m.get("content"),
                tool_calls=calls,
                tool_call_id=m.get("tool_call_id"),
                name=m.get("name"),
            )
        )
    return out


@dataclass(slots=True)
class ToolSpec:
    """A tool offered to the model, as a JSON schema."""

    name: str
    parameters: dict[str, Any]
    description: str = ""

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_model(
        cls, model: type[BaseModel], *, name: str | None = None, description: str = ""
    ) -> ToolSpec:
        from slmeval.providers.schema import json_schema_for

        return cls(
            name=name or model.__name__,
            parameters=json_schema_for(model),
            description=description or (model.__doc__ or "").strip(),
        )


@dataclass(slots=True)
class Completion:
    """One model response, with the measurement metadata the scorecard needs.

    ``structured_mode_used`` and ``repair_attempts`` are not diagnostics - they
    are the headline finding. "gemma3:4b needed json_mode on 100% of calls and
    prompt-repair on 12%" is the line a generic tracing tool cannot produce.
    """

    text: str | None = None
    parsed: BaseModel | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_estimated: bool = False
    """True when the provider returned no usage block. Any aggregate built from
    one of these is marked `~` in reports; a fabricated number must never look
    precise."""

    latency_ms: float = 0.0
    """Final attempt only. Retry time lives in `retry_ms`, because a candidate
    that needs three retries per call is unreliable even when the last call
    looks fast."""
    ttft_ms: float | None = None
    retries: int = 0
    retry_ms: float = 0.0

    raw: dict[str, Any] = field(default_factory=dict)
    structured_mode_used: str | None = None
    rungs_tried: tuple[str, ...] = ()
    repair_attempts: int = 0
    json_stage: JsonStage | None = None

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)

    @property
    def schema_lenient(self) -> bool:
        return self.json_stage is not None and self.json_stage is not JsonStage.DIRECT

    def parsed_as(self, schema: type[ModelT]) -> ModelT:
        """The validated object, typed as what you asked for.

        ``complete(schema=X)`` either returns a validated ``X`` or raises
        :class:`StructuredOutputError`, so in agent code this never legitimately
        returns ``None``. But ``parsed`` has to be typed ``BaseModel | None``,
        because a completion made without a schema has nothing parsed - which
        would otherwise leave every agent writing the same ``assert`` to get its
        own type back. This narrows it once, here.
        """
        if self.parsed is None:
            raise ValueError(
                "this completion has no parsed object - it was made without schema=. "
                "Pass schema= to complete() to get a validated model back."
            )
        if not isinstance(self.parsed, schema):
            raise TypeError(
                f"this completion parsed as {type(self.parsed).__name__}, "
                f"not {schema.__name__}"
            )
        return self.parsed


@runtime_checkable
class ChatClient(Protocol):
    """The entire provider interface.

    One method. Everything a candidate needs to be measured - which rung worked,
    how many repairs, whether usage was estimated - comes back on the
    :class:`Completion`.
    """

    spec: ClientSpec

    async def complete(
        self,
        messages: Sequence[MessageLike] | str,
        *,
        schema: type[BaseModel] | None = None,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> Completion: ...

    async def aclose(self) -> None: ...


# --------------------------------------------------------------------------- #
# Operational settings
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class RetryPolicy:
    """Retries are per provider, because a 4B model on a laptop GPU and a cloud
    frontier model have nothing in common operationally.

    Only transient failures are retried: 429, 5xx, and connection errors. A 400
    is a bug in our request or an unsupported feature, and retrying it three
    times just makes the bug slower to find.
    """

    attempts: int = 3
    base_delay_s: float = 0.5
    max_delay_s: float = 8.0
    jitter: float = 0.3


@dataclass(slots=True)
class ClientSpec:
    """Everything needed to talk to one model, resolved from config.

    The API key is resolved here, once, from the env var *name* in eval.toml.
    It never appears in a config file, a log line, or a traceback.
    """

    id: str
    provider: str
    model: str
    base_url: str | None = None
    api_key: str | None = None
    api_key_env: str | None = None
    """The variable *name* the key came from. Safe to print, and the only useful
    thing to say in a 401: "check $OPENAI_API_KEY"."""
    temperature: float = 0.0
    max_tokens: int = 1024
    structured_output: str = "auto"
    extra_body: dict[str, Any] = field(default_factory=dict)

    connect_timeout_s: float = 10.0
    read_timeout_s: float = 120.0
    max_concurrency: int = 4
    """Local default is 1. Publishing local-model latency measured under
    concurrency would be the fastest possible way to lose credibility."""
    retry: RetryPolicy = field(default_factory=RetryPolicy)

    def __repr__(self) -> str:
        # Explicit, so an api_key can never reach a traceback or a log.
        return (
            f"ClientSpec(id={self.id!r}, provider={self.provider!r}, "
            f"model={self.model!r}, base_url={self.base_url!r}, "
            f"api_key_env={self.api_key_env!r}, "
            f"api_key={'<set>' if self.api_key else None})"
        )


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ProviderError(Exception):
    """A provider call failed. The runner classifies these into outcomes."""

    transient: bool = False

    def __init__(self, message: str, *, status: int | None = None, body: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.body = body


class ProviderUnreachable(ProviderError):
    """Connection refused, DNS failure, timeout on connect."""

    transient = True


class RateLimited(ProviderError):
    transient = True


class ProviderServerError(ProviderError):
    """5xx. Worth retrying; not worth hiding."""

    transient = True


class ProviderRequestError(ProviderError):
    """4xx that is not a rate limit. Our request, or the user's config."""


class UnsupportedFeature(ProviderError):
    """The server rejected a structured-output mechanism.

    Not fatal: the ladder records the rung as unsupported, stops trying it for
    this client, and moves down. This is exactly the measurement we want - a
    server that cannot do `json_schema` is a fact about that candidate.
    """

    def __init__(self, rung: str, detail: str) -> None:
        super().__init__(f"{rung} not supported by this server: {detail}")
        self.rung = rung
        self.detail = detail


class StructuredOutputError(ProviderError):
    """Every rung failed, including the repair re-ask.

    Carries the diagnosis rather than just a message, because "the model could
    not produce your schema" is useless without seeing what it did produce.
    """

    def __init__(
        self,
        *,
        schema_name: str,
        rungs_tried: Sequence[str],
        last_text: str | None,
        validation_error: str | None,
        repair_attempts: int = 0,
    ) -> None:
        tried = ", ".join(rungs_tried) or "none"
        super().__init__(
            f"could not get valid {schema_name} out of the model "
            f"(rungs tried: {tried}; repairs: {repair_attempts})"
        )
        self.schema_name = schema_name
        self.rungs_tried = tuple(rungs_tried)
        self.last_text = last_text
        self.validation_error = validation_error
        self.repair_attempts = repair_attempts


__all__ = [
    "LADDER",
    "ChatClient",
    "ClientSpec",
    "Completion",
    "MessageLike",
    "ModelT",
    "Msg",
    "ProviderError",
    "ProviderRequestError",
    "ProviderServerError",
    "ProviderUnreachable",
    "RateLimited",
    "RetryPolicy",
    "Role",
    "Rung",
    "StructuredOutputError",
    "ToolCall",
    "ToolSpec",
    "UnsupportedFeature",
    "as_messages",
]
