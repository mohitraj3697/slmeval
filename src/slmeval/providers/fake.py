"""A scripted provider, so the test suite needs no GPU and no API keys.

This is not a toy. Most of what makes slmeval worth using is behaviour under
*badly behaved* models, and you cannot write a reliable regression test for "the
model emitted a prose preamble then a fenced JSON block with a trailing comma"
against a real 4B model - it will do something slightly different tomorrow. The
fake provider lets every one of those shapes be pinned in a test that runs
offline in milliseconds.

It is also the quickest honest smoke test for a user's own wiring: point
``provider = "fake"`` at their config and `slmeval run` exercises the whole
harness without spending a cent.

    register_script("preamble", [FakeTurn(text='Sure!\\n```json\\n{"a": 1}\\n```')])

    [[candidates]]
    id = "offline"
    provider = "fake"
    model = "preamble"
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from slmeval.core.context import get_context
from slmeval.providers.base import (
    ClientSpec,
    Msg,
    ProviderError,
    ProviderRequestError,
    ProviderServerError,
    RateLimited,
    Rung,
    ToolCall,
    ToolSpec,
    UnsupportedFeature,
)
from slmeval.providers.jsonx import JsonExtractionError, extract_json
from slmeval.providers.ladder import LadderClient, RungResult, with_schema_instruction
from slmeval.providers.schema import json_schema_for
from slmeval.providers.tokens import Usage, usage_from_response


@dataclass(slots=True)
class FakeTurn:
    """What the scripted server does for one call."""

    text: str | None = None
    """Message content. None plus no tool_calls means "synthesize something
    schema-valid", which is what makes `fake` usable as a stand-in baseline."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    refuse_tool_call: bool = False
    """Deliver `text` as message content even on the `tools` rung - i.e. the
    model ignored the forced tool and answered in prose. Extremely common with
    small models, and a different failure from "it called the tool badly"."""
    unsupported: frozenset[str] = frozenset()
    """Rungs this turn rejects, as a server would with a 400."""
    status: int | None = None
    """Simulate an HTTP failure. 429 and 5xx are raised as transient, so the
    retry path is testable without waiting on a real rate limit."""
    usage: tuple[int, int] | None = None
    """(prompt, completion) when the server reports usage. None means it does
    not, which exercises the estimate-and-flag path."""
    latency_ms: float = 0.0


SCRIPTS: dict[str, list[FakeTurn]] = {}


SCRIPTS_BY_CASE: dict[tuple[str, str], list[FakeTurn]] = {}


def register_script(
    name: str, turns: Sequence[FakeTurn], *, case_id: str | None = None
) -> None:
    """Make a script addressable as ``model = "<name>"``.

    ``case_id`` scopes it to one golden case. A benchmark is many cases and one
    model, and a single script cannot answer them all: the routing response for
    "where is my order ORD-9921" is the wrong answer to "my package never
    turned up", so a shared script would have the scripted baseline fail most
    of its own dataset. Per-case scripts are what make `fake` usable as an
    offline reference rather than a one-case toy.
    """
    if case_id is None:
        SCRIPTS[name] = list(turns)
    else:
        SCRIPTS_BY_CASE[(name, case_id)] = list(turns)


def clear_scripts() -> None:
    SCRIPTS.clear()
    SCRIPTS_BY_CASE.clear()


class FakeClient(LadderClient):
    """Replays a script of turns. Never touches the network.

    Turns are consumed in order; the last one repeats once the script runs out,
    so a one-turn script describes a model that always behaves that way and a
    two-turn script describes "fails, then succeeds on the re-ask".
    """

    def __init__(
        self,
        spec: ClientSpec,
        *,
        script: Sequence[FakeTurn] | None = None,
        unsupported: Sequence[str] | None = None,
    ) -> None:
        super().__init__(spec)
        self._script = list(script) if script is not None else _resolve_script(spec)
        self._by_case = {} if script is not None else _resolve_case_scripts(spec)
        self._always_unsupported = frozenset(
            unsupported if unsupported is not None else _unsupported_from_config(spec)
        )
        self._calls = 0
        self.seen: list[dict[str, Any]] = []
        """Every request made, for assertions: rung, messages, schema name."""

    @property
    def calls(self) -> int:
        return self._calls

    def unavailable_rungs(self) -> frozenset[str]:
        return self._always_unsupported

    async def attempt(
        self,
        rung: Rung | None,
        messages: Sequence[Msg],
        *,
        schema: type[BaseModel] | None,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> RungResult:
        wire_messages = list(messages)
        if rung in ("json_mode", "prompt") and schema is not None:
            wire_messages = with_schema_instruction(wire_messages, schema)

        turn = self._next_turn()
        self._calls += 1
        self._advance()
        self.seen.append(
            {
                "rung": rung,
                "schema": schema.__name__ if schema else None,
                "messages": [m.to_wire() for m in wire_messages],
                "tools": [t.name for t in tools or []],
            }
        )

        if rung is not None and rung in turn.unsupported:
            raise UnsupportedFeature(rung, "scripted rejection")
        if turn.status is not None:
            raise _http_error(turn.status)

        started = time.perf_counter()
        text, calls = self._render(turn, rung, schema)
        elapsed = (time.perf_counter() - started) * 1000.0

        if turn.usage is not None:
            usage = Usage(
                prompt_tokens=turn.usage[0], completion_tokens=turn.usage[1], estimated=False
            )
        else:
            usage = usage_from_response(
                {}, messages=wire_messages, output_text=text, output_tool_calls=calls
            )

        return RungResult(
            text=text,
            tool_calls=calls,
            raw={"fake": True, "rung": rung, "model": self.spec.model},
            usage=usage,
            latency_ms=turn.latency_ms or elapsed,
            ttft_ms=None,
        )

    def _next_turn(self) -> FakeTurn:
        script = self._script_for_case()
        if not script:
            return FakeTurn()
        index = min(self._position(), len(script) - 1)
        return script[index]

    def _script_for_case(self) -> list[FakeTurn]:
        """This case's script, falling back to the model-wide one.

        Falling back rather than erroring: a model-wide script plus overrides
        for the two cases that need different answers is how these are
        actually written, and demanding an entry per case would make the
        config longer than the dataset.
        """
        ctx = get_context()
        if ctx is not None:
            scoped = self._by_case.get(ctx.case_id)
            if scoped:
                return scoped
        return self._script

    # ----------------------------------------------------------- script position
    #
    # A script describes how a model behaves, so every execution has to start
    # at turn 0. The position therefore lives on the *execution*, not on the
    # client: the matrix runner shares one client per model across all 90
    # executions - it has to, because that is what shares the per-endpoint
    # concurrency limit - and a per-client counter would have execution 2 start
    # at the compose turn and fail its routing schema. Which is a failure the
    # harness invented, in the one provider whose whole job is to be
    # predictable.
    #
    # Outside a run there is no execution to key on, and the client's own
    # counter is exactly right: a test driving the client directly means
    # "fails, then succeeds on the re-ask".

    def _key(self) -> str:
        return f"fake_turn:{self.spec.id}:{self.spec.model}"

    def _position(self) -> int:
        ctx = get_context()
        if ctx is None:
            return self._calls
        return int(ctx.metadata.get(self._key(), 0))

    def _advance(self) -> None:
        ctx = get_context()
        if ctx is None:
            return
        ctx.metadata[self._key()] = self._position() + 1

    def _render(
        self, turn: FakeTurn, rung: Rung | None, schema: type[BaseModel] | None
    ) -> tuple[str | None, list[ToolCall]]:
        """Put the scripted payload where this rung would really deliver it.

        On the `tools` rung a real server returns tool-call arguments rather
        than message content, so a script written as text has to arrive that
        way - otherwise every tools-rung test would need its own script.
        """
        text = turn.text
        calls = list(turn.tool_calls)

        if text is None and not calls and schema is not None:
            text = _synthesize(schema)

        if rung == "tools" and schema is not None and not calls and not turn.refuse_tool_call:
            payload = text or "{}"
            args: dict[str, Any] = {}
            stage = None
            try:
                extraction = extract_json(payload)
            except JsonExtractionError:
                pass
            else:
                if isinstance(extraction.value, dict):
                    args, stage = extraction.value, extraction.stage
            return None, [
                ToolCall(
                    id="call_0",
                    name=schema.__name__,
                    arguments=args,
                    raw_arguments=payload,
                    arguments_stage=stage,
                )
            ]
        return text, calls


def _resolve_script(spec: ClientSpec) -> list[FakeTurn]:
    """A registered script by model name, or one declared inline in eval.toml."""
    if spec.model in SCRIPTS:
        return list(SCRIPTS[spec.model])
    return _turns_from_config(spec)


def _turns_from_config(spec: ClientSpec) -> list[FakeTurn]:
    """Read ``extra_body.script`` so an offline benchmark is expressible in TOML.

        [[candidates]]
        provider = "fake"
        model = "offline"
        extra_body.unsupported = ["tools", "json_schema"]
        extra_body.script = [{ text = '{"intent": "refund"}' }]
    """
    return _turns_from_list(spec.extra_body.get("script"))


def _turns_from_list(raw: object) -> list[FakeTurn]:
    """One script, as a list of strings or tables."""
    if not isinstance(raw, list):
        return []
    turns: list[FakeTurn] = []
    for item in raw:
        if isinstance(item, str):
            turns.append(FakeTurn(text=item))
            continue
        if not isinstance(item, dict):
            raise ValueError(
                f"fake script entries must be strings or tables, got {type(item).__name__}"
            )
        usage = item.get("usage")
        turns.append(
            FakeTurn(
                text=item.get("text"),
                refuse_tool_call=bool(item.get("refuse_tool_call", False)),
                unsupported=frozenset(item.get("unsupported", ())),
                status=item.get("status"),
                usage=(int(usage[0]), int(usage[1])) if isinstance(usage, (list, tuple)) else None,
                latency_ms=float(item.get("latency_ms", 0.0)),
            )
        )
    return turns


def _resolve_case_scripts(spec: ClientSpec) -> dict[str, list[FakeTurn]]:
    """Per-case scripts, from the registry and from ``extra_body.script_by_case``.

        [baseline]
        provider = "fake"
        model = "scripted"
        extra_body.script = [{ text = '{"intent": "other"}' }]
        extra_body.script_by_case.refund_207 = [{ text = '{"intent": "refund"}' }]
    """
    out: dict[str, list[FakeTurn]] = {
        case_id: list(turns)
        for (model, case_id), turns in SCRIPTS_BY_CASE.items()
        if model == spec.model
    }
    raw = spec.extra_body.get("script_by_case")
    if raw is None:
        return out
    if not isinstance(raw, dict):
        raise ValueError(
            "extra_body.script_by_case must be a table of case_id -> script"
        )
    for case_id, turns in raw.items():
        out[str(case_id)] = _turns_from_list(turns)
    return out


def _unsupported_from_config(spec: ClientSpec) -> list[str]:
    raw = spec.extra_body.get("unsupported")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("extra_body.unsupported must be a list of rung names")
    return [str(r) for r in raw]


def _http_error(status: int) -> ProviderError:
    if status == 429:
        return RateLimited("scripted rate limit", status=status)
    if status >= 500:
        return ProviderServerError(f"scripted {status}", status=status)
    return ProviderRequestError(f"scripted {status}", status=status)


def _synthesize(schema: type[BaseModel]) -> str:
    """A minimal schema-valid JSON object, built from the JSON schema.

    Deliberately boring values - "", 0, false, [] - because a fake that emitted
    plausible-looking content would make it easy to mistake a smoke test for a
    real result.
    """
    root = json_schema_for(schema)
    return json.dumps(_fill(root, root.get("$defs") or {}))


def _fill(node: dict[str, Any], defs: dict[str, Any], depth: int = 0) -> Any:
    ref = node.get("$ref")
    if isinstance(ref, str) and depth < 8:
        # Nested models arrive as "#/$defs/Name"; resolving them is what keeps
        # the synthesized object actually valid against the schema.
        name = ref.rsplit("/", 1)[-1]
        target = defs.get(name)
        if isinstance(target, dict):
            return _fill(target, defs, depth + 1)
        return None

    if "const" in node:
        return node["const"]
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    if "default" in node:
        return node["default"]

    for key in ("anyOf", "oneOf"):
        options = node.get(key)
        if isinstance(options, list) and options:
            # Prefer a non-null branch: `str | None` should produce "", not
            # null, or every optional field would come back empty.
            chosen = next(
                (o for o in options if isinstance(o, dict) and o.get("type") != "null"),
                options[0],
            )
            return _fill(chosen, defs, depth + 1) if isinstance(chosen, dict) else None

    kind = node.get("type")
    if isinstance(kind, list):
        kind = next((k for k in kind if k != "null"), kind[0] if kind else None)
    if kind == "object" or "properties" in node:
        props = node.get("properties") or {}
        required = node.get("required") or list(props)
        return {
            name: _fill(props[name], defs, depth + 1) for name in required if name in props
        }
    if kind == "array":
        return []
    if kind == "integer":
        return 0
    if kind == "number":
        return 0.0
    if kind == "boolean":
        return False
    if kind == "null":
        return None
    return ""


__all__ = ["SCRIPTS", "FakeClient", "FakeTurn", "clear_scripts", "register_script"]
