"""The structured-output ladder.

This is where small models actually fail, so this is where the measurement
lives. The failure is almost never "the model is dumb" - it is "the model's
structured-output path on this provider silently degrades". A framework
abstraction papers over exactly that, which is why the ladder is ours.

For every request with a schema, the rungs are tried strongest-first and the one
that *worked* is recorded:

========== =================================================================
tools      native function-calling, with the schema as a forced tool
json_schema OpenAI-style strict ``response_format``
json_mode  ``response_format={"type":"json_object"}`` plus the schema in-prompt
prompt     schema in the prompt only, parse whatever comes back
repair     one re-ask carrying the validation error
========== =================================================================

Which produces the line that no generic tracing tool can: *gpt-4o used native
tools throughout; gemma3:4b needed json_mode on 100% of calls and prompt-repair
on 12%.*

Two deliberate constraints:

- ``structured_output = "auto"`` walks the ladder; pinning a rung forces the
  comparison to be honest. A model that only succeeds at ``prompt`` should not
  be flattered by a harness that quietly tried three stronger mechanisms first.
- A rung the server rejects outright is remembered per client, so a benchmark
  does not pay a 400 on every one of 900 calls to learn the same fact twice.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError

from slmeval.core.trace import JsonStage
from slmeval.providers.base import (
    LADDER,
    ClientSpec,
    Completion,
    MessageLike,
    Msg,
    Rung,
    StructuredOutputError,
    ToolCall,
    ToolSpec,
    UnsupportedFeature,
    as_messages,
)
from slmeval.providers.jsonx import JsonExtractionError, extract_json
from slmeval.providers.schema import (
    format_validation_error,
    repair_instruction,
    schema_instruction,
    validate_into,
)
from slmeval.providers.tokens import Usage

MAX_REPAIRS = 1
"""One re-ask, not a loop. Three repairs would turn an unusable model into a
passing one and hide the finding that matters."""


@dataclass(slots=True)
class RungResult:
    """One HTTP round trip, before any schema interpretation."""

    text: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    retries: int = 0
    retry_ms: float = 0.0


@dataclass(slots=True)
class _Parsed:
    model: BaseModel
    stage: JsonStage


class LadderClient(ABC):
    """Base for any client that can attempt a single rung.

    Subclasses implement one method - :meth:`attempt` - and inherit the whole
    ladder, the repair re-ask, and the bookkeeping.
    """

    def __init__(self, spec: ClientSpec) -> None:
        self.spec = spec
        self._unsupported: set[str] = set()

    # -- to implement ------------------------------------------------------ #

    @abstractmethod
    async def attempt(
        self,
        rung: Rung | None,
        messages: Sequence[Msg],
        *,
        schema: type[BaseModel] | None,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> RungResult:
        """One round trip using `rung`'s mechanism.

        Raise :class:`UnsupportedFeature` when the server rejects the mechanism
        itself, so the ladder can drop that rung and move down.
        """

    async def aclose(self) -> None:  # noqa: B027 - optional, not abstract
        """Release connections. Safe to call twice.

        Deliberately concrete and empty: a client with nothing to release (the
        fake one) should not have to implement a no-op to satisfy the base.
        """

    def unavailable_rungs(self) -> frozenset[str]:
        """Rungs this backend cannot offer at all, as opposed to ones a server
        turned out to reject at runtime."""
        return frozenset()

    # -- the ladder -------------------------------------------------------- #

    @property
    def unsupported(self) -> frozenset[str]:
        """Rungs this server rejected during the run. A reportable fact."""
        return frozenset(self._unsupported)

    def rungs(
        self, *, schema: type[BaseModel] | None, tools: Sequence[ToolSpec] | None
    ) -> list[Rung]:
        """The rungs to try, in order, for this request."""
        if schema is None:
            return []
        configured = self.spec.structured_output
        order: list[Rung] = list(LADDER) if configured == "auto" else [configured]  # type: ignore[list-item]

        blocked = self._unsupported | set(self.unavailable_rungs())
        if tools:
            # The `tools` rung forces a call to the schema tool. It cannot
            # coexist with the caller's own tools, so skip it rather than
            # quietly dropping the agent's tools and measuring a different run.
            blocked.add("tools")
        usable = [r for r in order if r not in blocked]

        if not usable:
            pinned = "" if configured == "auto" else f" (pinned to {configured!r})"
            raise UnsupportedFeature(
                configured,
                f"no structured-output mechanism left to try{pinned}; "
                f"ruled out: {', '.join(sorted(blocked)) or 'none'}",
            )
        return usable

    async def complete(
        self,
        messages: Sequence[MessageLike] | str,
        *,
        schema: type[BaseModel] | None = None,
        tools: Sequence[ToolSpec] | None = None,
        tool_choice: str | None = None,
    ) -> Completion:
        msgs = as_messages(messages)

        if schema is None:
            result = await self.attempt(
                None, msgs, schema=None, tools=tools, tool_choice=tool_choice
            )
            return self._completion(result, rungs_tried=(), rung=None, parsed=None)

        tried: list[str] = []
        last: RungResult | None = None
        last_rung: Rung | None = None
        last_error: str | None = None

        for rung in self.rungs(schema=schema, tools=tools):
            tried.append(rung)
            try:
                result = await self.attempt(
                    rung, msgs, schema=schema, tools=tools, tool_choice=None
                )
            except UnsupportedFeature as exc:
                # Remember, so 900 calls do not each pay to learn this.
                self._unsupported.add(rung)
                last_error = exc.detail
                continue

            parsed, error = self._interpret(rung, result, schema)
            if parsed is not None:
                return self._completion(result, rungs_tried=tuple(tried), rung=rung, parsed=parsed)
            last, last_rung, last_error = result, rung, error

        if last is not None and last_rung is not None:
            repaired = await self._repair(
                msgs, last, last_rung, schema=schema, tools=tools, error=last_error or ""
            )
            if repaired is not None:
                result, parsed = repaired
                return self._completion(
                    result,
                    rungs_tried=tuple(tried),
                    rung=last_rung,
                    parsed=parsed,
                    repair_attempts=MAX_REPAIRS,
                )

        raise StructuredOutputError(
            schema_name=schema.__name__,
            rungs_tried=tried,
            last_text=last.text if last else None,
            validation_error=last_error,
            repair_attempts=MAX_REPAIRS if last is not None else 0,
        )

    async def _repair(
        self,
        msgs: list[Msg],
        failed: RungResult,
        rung: Rung,
        *,
        schema: type[BaseModel],
        tools: Sequence[ToolSpec] | None,
        error: str,
    ) -> tuple[RungResult, _Parsed] | None:
        """One re-ask on the rung that got closest, carrying the exact failure."""
        emitted = failed.text
        if not emitted and failed.tool_calls:
            emitted = failed.tool_calls[0].raw_arguments
        conversation = [
            *msgs,
            Msg(role="assistant", content=emitted or ""),
            Msg(role="user", content=repair_instruction(schema, emitted, error)),
        ]
        try:
            result = await self.attempt(
                rung, conversation, schema=schema, tools=tools, tool_choice=None
            )
        except UnsupportedFeature:
            return None
        parsed, _ = self._interpret(rung, result, schema)
        if parsed is None:
            return None
        return result, parsed

    def _interpret(
        self, rung: Rung, result: RungResult, schema: type[BaseModel]
    ) -> tuple[_Parsed | None, str | None]:
        """Turn a round trip into a validated model, or an error to report."""
        if rung == "tools":
            call = _pick_schema_call(result.tool_calls, schema)
            if call is None:
                return None, (
                    "the model did not call the forced tool "
                    f"{schema.__name__!r}; it replied with free text instead"
                )
            value: Any = call.arguments
            stage = call.arguments_stage or JsonStage.DIRECT
        else:
            try:
                extraction = extract_json(result.text)
            except JsonExtractionError as exc:
                return None, str(exc)
            value, stage = extraction.value, extraction.stage

        try:
            model = validate_into(schema, value)
        except ValidationError as exc:
            return None, format_validation_error(exc)
        return _Parsed(model=model, stage=stage), None

    def _completion(
        self,
        result: RungResult,
        *,
        rungs_tried: tuple[str, ...],
        rung: Rung | None,
        parsed: _Parsed | None,
        repair_attempts: int = 0,
    ) -> Completion:
        return Completion(
            text=result.text,
            parsed=parsed.model if parsed else None,
            tool_calls=result.tool_calls,
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            tokens_estimated=result.usage.estimated,
            latency_ms=result.latency_ms,
            ttft_ms=result.ttft_ms,
            retries=result.retries,
            retry_ms=result.retry_ms,
            raw=result.raw,
            structured_mode_used=rung,
            rungs_tried=rungs_tried,
            repair_attempts=repair_attempts,
            json_stage=parsed.stage if parsed else None,
        )


def with_schema_instruction(messages: Sequence[Msg], schema: type[BaseModel]) -> list[Msg]:
    """Append the schema to the conversation for the json_mode/prompt rungs.

    It goes on the *last user turn*, merged in if there is one. A trailing
    system message is tempting and wrong: several chat templates render only the
    first system message, so the instruction would silently vanish and the rung
    would measure nothing.
    """
    instruction = schema_instruction(schema)
    out = list(messages)
    if out and out[-1].role == "user" and out[-1].content:
        last = out[-1]
        out[-1] = Msg(
            role="user",
            content=f"{last.content}\n\n{instruction}",
            tool_calls=last.tool_calls,
            tool_call_id=last.tool_call_id,
            name=last.name,
        )
        return out
    out.append(Msg(role="user", content=instruction))
    return out


def _pick_schema_call(calls: Sequence[ToolCall], schema: type[BaseModel]) -> ToolCall | None:
    """The forced schema call, by name, falling back to the only call made."""
    for call in calls:
        if call.name == schema.__name__:
            return call
    return calls[0] if len(calls) == 1 else None


__all__ = ["MAX_REPAIRS", "LadderClient", "RungResult", "with_schema_instruction"]
