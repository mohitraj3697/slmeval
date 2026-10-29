"""Node-aware model routing, and the `chat_model()` the user calls.

This is the second half of the two-line integration. The user replaces their LLM
construction with ``chat_model()`` and the harness decides, per call, which
model answers it - because that is the whole experiment.

The routing rule comes straight from ``target_nodes``:

- a candidate with no ``target_nodes`` replaces the model **everywhere**: the
  "can a 4B model run this whole agent" question;
- a candidate with ``target_nodes = ["intent_router"]`` replaces the model on
  those nodes only, and the baseline answers the rest: the "downsize the boring
  80%" question, which is the one most people actually ship.

Both are measured the same way, so a scorecard can say *swapping only the router
costs you 1.2 points and 61% of the bill*.

Every call is traced, and the trace is appended in a ``finally``. That detail
matters: recording per-``except`` branch is how the blueprint ended up filing
HTTP crashes as schema-valid responses, which quietly inflates the one metric
the tool exists to report.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel

from slmeval.config import EvalConfig, ModelCfg
from slmeval.core.context import current_node, current_node_path, get_context
from slmeval.core.trace import LLMCallTrace, SchemaStatus
from slmeval.providers.base import (
    ChatClient,
    Completion,
    MessageLike,
    ProviderError,
    StructuredOutputError,
    ToolSpec,
)
from slmeval.providers.registry import build_client


@dataclass
class ModelRouter:
    """Owns the clients for one candidate and answers "who serves this node".

    One router per candidate per run, shared by every concurrent execution of
    that candidate - so the clients, and therefore the per-provider concurrency
    limits, are shared too.
    """

    candidate: ModelCfg
    baseline: ModelCfg | None = None
    timeout_s: float | None = None
    http: httpx.AsyncClient | None = None
    _clients: dict[str, ChatClient] = field(default_factory=dict, repr=False)

    @classmethod
    def for_candidate(cls, config: EvalConfig, candidate_id: str) -> ModelRouter:
        model = config.candidate(candidate_id)
        if model is None:
            known = ", ".join(m.id for m in config.all_models)
            raise KeyError(f"no candidate {candidate_id!r} in this config (have: {known})")
        return cls(
            candidate=model,
            baseline=config.baseline if model.id != config.baseline.id else None,
            timeout_s=config.benchmark.timeout_s,
        )

    def resolve(self, node: str | None) -> ModelCfg:
        """Which model configuration answers a call at `node`.

        A call made outside any ``node()`` block falls to the baseline whenever
        the candidate is node-scoped: attributing unlabelled calls to a
        router-only swap would credit it with work it never did.
        """
        if self.candidate.replaces_everything or self.baseline is None:
            return self.candidate
        if node is not None and node in self.candidate.target_nodes:
            return self.candidate
        return self.baseline

    def client_for(self, node: str | None) -> ChatClient:
        model = self.resolve(node)
        client = self._clients.get(model.id)
        if client is None:
            client = build_client(model, timeout_s=self.timeout_s, http=self.http)
            self._clients[model.id] = client
        return client

    @property
    def models(self) -> list[ModelCfg]:
        """Every configuration this router can route to, baseline last."""
        if self.baseline is None:
            return [self.candidate]
        return [self.candidate, self.baseline]

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()


class ChatModel:
    """What :func:`chat_model` returns: a model handle that resolves late.

    Late on purpose. The same user code runs for the baseline and for every
    candidate, so "which model" cannot be decided when the handle is built - it
    is decided per call, from the ambient context and the current node.
    """

    def __init__(
        self,
        *,
        node: str | None = None,
        fallback: ModelCfg | None = None,
    ) -> None:
        self._pinned_node = node
        self._fallback = fallback
        self._fallback_client: ChatClient | None = None

    def __repr__(self) -> str:
        where = self._pinned_node or "<current node>"
        return f"ChatModel(node={where!r})"

    async def complete(
        self,
        messages: Sequence[MessageLike] | str,
        *,
        schema: type[BaseModel] | None = None,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
        node: str | None = None,
    ) -> Completion:
        """Call the model that this run has assigned to this node."""
        ctx = get_context()
        where = node or self._pinned_node or current_node()

        if ctx is None or ctx.router is None:
            return await self._complete_untraced(
                messages, schema=schema, tools=tools, tool_choice=tool_choice
            )

        client = ctx.router.client_for(where)
        spec = client.spec
        trace = LLMCallTrace(
            seq=ctx.next_llm_seq(),
            model_id=spec.id,
            provider=spec.provider,
            model=spec.model,
            node=where,
            node_path=current_node_path(),
            schema_name=schema.__name__ if schema else None,
            structured_requested=spec.structured_output,
            schema_status=SchemaStatus.NOT_REQUESTED if schema is None else SchemaStatus.ERROR,
        )
        try:
            completion = await client.complete(
                messages, schema=schema, tools=tools, tool_choice=tool_choice
            )
        except StructuredOutputError as exc:
            # The model answered; it just never produced the schema. That is a
            # model failure and belongs in the adherence rate.
            trace.schema_status = SchemaStatus.INVALID
            trace.rungs_tried = exc.rungs_tried
            trace.repair_attempts = exc.repair_attempts
            trace.error = exc.validation_error or exc.message
            trace.error_type = type(exc).__name__
            raise
        except ProviderError as exc:
            # The call never produced a candidate response. A failure, but not
            # one that says anything about the model's schema adherence.
            trace.schema_status = (
                SchemaStatus.NOT_REQUESTED if schema is None else SchemaStatus.ERROR
            )
            trace.error = exc.message
            trace.error_type = type(exc).__name__
            raise
        except BaseException as exc:
            trace.schema_status = (
                SchemaStatus.NOT_REQUESTED if schema is None else SchemaStatus.ERROR
            )
            trace.error = str(exc)
            trace.error_type = type(exc).__name__
            raise
        else:
            _fill_trace(trace, completion, schema is not None)
            return completion
        finally:
            # In `finally`, not per-branch: every path has to leave exactly one
            # trace, or the denominator of every rate in the report is wrong.
            ctx.record_llm(trace)
            if trace.error is not None:
                ctx.warn(f"{spec.id}: {trace.error_type} on {where or 'unnamed node'}")

    async def _complete_untraced(
        self,
        messages: Sequence[MessageLike] | str,
        *,
        schema: type[BaseModel] | None,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> Completion:
        """Production passthrough: no benchmark running, so no interception.

        This is the path that makes `chat_model()` safe to leave in shipped
        code, but it only works if the call site said which model to use.
        """
        if self._fallback is None:
            raise LookupError(
                "chat_model() was called outside a benchmark run and has no "
                "model to fall back on.\n"
                "  Inside `slmeval run` the harness supplies the model.\n"
                "  Everywhere else, say which one: "
                "chat_model(provider='openai', model='gpt-4o-mini')."
            )
        if self._fallback_client is None:
            self._fallback_client = build_client(self._fallback)
        return await self._fallback_client.complete(
            messages, schema=schema, tools=tools, tool_choice=tool_choice
        )

    async def aclose(self) -> None:
        """Only closes a client this handle owns; the router owns its own."""
        if self._fallback_client is not None:
            await self._fallback_client.aclose()
            self._fallback_client = None


def chat_model(
    *,
    node: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    api_key_env: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    structured_output: str = "auto",
    extra_body: dict[str, Any] | None = None,
) -> ChatModel:
    """The model handle to use inside your agent.

        with node("intent_router"):
            llm = chat_model()
            decision = await llm.complete(messages, schema=Routing)

    Inside ``slmeval run`` the harness decides which model answers, per node,
    per candidate. The keyword arguments are your *production* model - the one
    used when no benchmark is running - so this line is safe to ship:

        llm = chat_model(provider="openai", model="gpt-4o-mini")

    `node` pins the attribution for calls made outside a ``node()`` block.
    """
    fallback: ModelCfg | None = None
    if provider or model:
        if not (provider and model):
            raise ValueError("chat_model() needs both provider= and model=, or neither")
        fallback = ModelCfg(
            id="production",
            provider=provider,  # type: ignore[arg-type]
            model=model,
            base_url=base_url,
            api_key_env=api_key_env,
            temperature=temperature,
            max_tokens=max_tokens,
            structured_output=structured_output,  # type: ignore[arg-type]
            extra_body=extra_body or {},
        )
    return ChatModel(node=node, fallback=fallback)


def _fill_trace(trace: LLMCallTrace, completion: Completion, schema_requested: bool) -> None:
    trace.schema_status = SchemaStatus.VALID if schema_requested else SchemaStatus.NOT_REQUESTED
    trace.structured_mode_used = completion.structured_mode_used
    trace.rungs_tried = completion.rungs_tried
    trace.repair_attempts = completion.repair_attempts
    trace.json_stage = completion.json_stage
    trace.prompt_tokens = completion.prompt_tokens
    trace.completion_tokens = completion.completion_tokens
    trace.tokens_estimated = completion.tokens_estimated
    trace.latency_ms = completion.latency_ms
    trace.ttft_ms = completion.ttft_ms
    trace.retries = completion.retries
    trace.retry_ms = completion.retry_ms


__all__ = ["ChatModel", "ModelRouter", "chat_model"]
