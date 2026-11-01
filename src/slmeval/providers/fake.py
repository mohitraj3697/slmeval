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
import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from slmeval.core.context import get_context
from slmeval.providers.base import (
    ClientSpec,
    Msg,
    ProviderError,
    ProviderRequestError,
    ProviderServerError,
    ProviderUnreachable,
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
    """Simulate an HTTP failure. 429 and 5xx keep their own transient types and
    400 keeps its permanent one, because the retry policy and the outcome
    classifier both read the type rather than the number."""
    usage: tuple[int, int] | None = None
    """(prompt, completion) when the server reports usage. None means it does
    not, which exercises the estimate-and-flag path."""
    latency_ms: float = 0.0
    """Reported as this call's latency; **not** slept. A suite that really
    waited would stop being run, and the thing under test is what the report
    does with a slow model, not our ability to call `sleep`. For an execution
    that genuinely outlives `timeout_s`, make the agent wait."""
    error: str | None = None
    """A transport failure with no HTTP status: ``"unreachable"`` or
    ``"timeout"``.

    A model server that is down is the most common way a benchmark fails, and
    the exit code for it - 3, environment, never 1 - is a documented contract
    that CI scripts branch on. Asserting that against a real socket costs two
    seconds per attempt on Windows, where a refused connect to a local port
    sits in SYN retry; scripting it here costs nothing and classifies
    identically, because both arrive as :class:`ProviderUnreachable`."""


_OK_PAYLOAD = '{"ok": true}'
"""Stand-in payload for the shapes that are about *form*, not content."""


def _with_trailing_comma(payload: str) -> str:
    body = payload.rstrip()
    if body.endswith(("}", "]")):
        return f"{body[:-1]},{body[-1]}"
    return f"{body},"


MISBEHAVIOURS: dict[str, Callable[[str | None], FakeTurn]] = {
    # The happy path, for contrast.
    "valid_json": lambda p: FakeTurn(text=p or _OK_PAYLOAD),
    # Extraction-cascade territory: the content is right, the envelope is not.
    "fenced_json": lambda p: FakeTurn(
        text=f"Sure! Here is the result:\n```json\n{p or _OK_PAYLOAD}\n```"
    ),
    "prose_json": lambda p: FakeTurn(
        text=f"Of course - the answer is {p or _OK_PAYLOAD}. Let me know if you need more."
    ),
    "trailing_comma": lambda p: FakeTurn(text=_with_trailing_comma(p or _OK_PAYLOAD)),
    # Parses as JSON, is not the schema we asked for. A different failure from
    # "not JSON at all", and the two must not share a metric.
    "invalid_schema": lambda p: FakeTurn(text=p or '{"vibes": "immaculate"}'),
    "not_json": lambda p: FakeTurn(text=p or "I think they want a refund, probably."),
    "empty": lambda p: FakeTurn(text=p or ""),
    # Answered in prose on the tools rung: the model ignored a forced tool
    # call. Very common on small models, and not the same as calling it badly.
    "refusal": lambda p: FakeTurn(
        text=p or "Sorry, I cannot help with that.", refuse_tool_call=True
    ),
    "hallucinated_tool": lambda p: FakeTurn(
        tool_calls=[
            ToolCall(
                id="call_0",
                name=p or "frobnicate_order",
                arguments={},
                raw_arguments="{}",
            )
        ]
    ),
    "hallucinated_args": lambda p: FakeTurn(
        tool_calls=[
            ToolCall(
                id="call_0",
                name=p or "fetch_order",
                arguments={"ordr_id": "9921", "urgent": True},
                raw_arguments='{"ordr_id": "9921", "urgent": true}',
            )
        ]
    ),
    "timeout": lambda p: FakeTurn(error="timeout"),
    "unreachable": lambda p: FakeTurn(error="unreachable"),
    "rate_limit": lambda p: FakeTurn(status=429),
    "server_error": lambda p: FakeTurn(status=503),
    "bad_request": lambda p: FakeTurn(status=400),
    # The last turn of a script repeats, so a one-turn script *is* a model
    # that answers this way forever. Named, because "the agent loops until the
    # step budget stops it" is a scripted scenario rather than an accident.
    "loop": lambda p: FakeTurn(text=p or _OK_PAYLOAD),
}
"""Named model misbehaviours, each as one turn.

Every one of these is something a 4B model does in the wild and none of them
can be pinned in a regression test against a real one - it will do something
slightly different tomorrow. Naming them is what lets a test say *which*
misbehaviour it covers, instead of embedding a hand-written prose preamble the
next reader has to decode. The same names work from TOML (``kind = ...``), so
an offline benchmark of badly behaved models needs no Python at all.
"""


def misbehaviour(kind: str, payload: str | None = None, **overrides: Any) -> FakeTurn:
    """One entry of :data:`MISBEHAVIOURS`, with ``payload`` as its content.

    ``overrides`` are :class:`FakeTurn` fields applied on top, so
    ``misbehaviour("fenced_json", body, latency_ms=800)`` is a slow model that
    also cannot emit bare JSON.
    """
    build = MISBEHAVIOURS.get(kind)
    if build is None:
        raise ValueError(
            f"unknown fake behaviour {kind!r}; known: {', '.join(sorted(MISBEHAVIOURS))}"
        )
    turn = build(payload)
    return replace(turn, **overrides) if overrides else turn


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
        declared = _declared(spec)
        self._script = list(script) if script is not None else declared.script
        self._by_case = {} if script is not None else declared.by_case
        self._always_unsupported = frozenset(
            unsupported if unsupported is not None else declared.unsupported
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
        if turn.error is not None:
            raise _transport_error(turn.error, self.spec)
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


EXTRA_BODY_KEYS = frozenset({"script", "script_by_case", "script_file", "unsupported"})
"""Everything `fake` reads from ``extra_body``.

Checked rather than ignored: ``extra_body`` is provider passthrough, so a
``scripts = [...]`` typo would otherwise leave a fake that silently
synthesizes schema-valid blanks - a run that looks like a pass and measures
nothing.
"""


@dataclass(frozen=True, slots=True)
class _Declared:
    """What a config says this fake does, before any per-execution state."""

    script: list[FakeTurn] = field(default_factory=list)
    by_case: dict[str, list[FakeTurn]] = field(default_factory=dict)
    unsupported: list[str] = field(default_factory=list)


def validate_extra_body(extra: Mapping[str, Any]) -> None:
    """Parse everything `fake` reads from one ``extra_body``, and discard it.

    Called by the config loader so a typo in a script fails at load with a
    line number. The alternative is a ``ValueError`` raised inside the provider
    thirty executions into a matrix, where the outcome classifier attributes
    anything raised inside slmeval to slmeval - printing a user's typo as a
    harness bug, which is the one classification we must never get wrong.
    """
    _declared_from(extra, model="")


def _declared(spec: ClientSpec) -> _Declared:
    return _declared_from(spec.extra_body, spec.model)


def _declared_from(extra: Mapping[str, Any], model: str) -> _Declared:
    """Merge the three places a script can come from, in precedence order.

    A registry entry (``register_script``) beats the config, because that is a
    test or a harness pinning behaviour deliberately; inline ``extra_body``
    beats a fixture file, because an override belongs next to the candidate it
    applies to. Per-case entries merge across all three rather than replacing
    one another: a fixture file carrying the whole dataset plus one inline
    override for the case under test is the shape these actually take.
    """
    unknown = sorted(set(extra) - EXTRA_BODY_KEYS)
    if unknown:
        raise ValueError(
            f"provider 'fake' does not understand extra_body key(s) "
            f"{', '.join(unknown)} (known: {', '.join(sorted(EXTRA_BODY_KEYS))})"
        )
    if "script" in extra and "script_file" in extra:
        raise ValueError(
            "set extra_body.script or extra_body.script_file, not both - "
            "two scripts for one model has no defined order"
        )

    fixture = _load_fixture(extra["script_file"]) if "script_file" in extra else _Declared()

    script = (
        list(SCRIPTS[model])
        if model in SCRIPTS
        else _turns_from_list(extra.get("script")) or fixture.script
    )

    by_case: dict[str, list[FakeTurn]] = dict(fixture.by_case)
    by_case.update(_case_scripts_from(extra.get("script_by_case")))
    by_case.update(
        {
            case_id: list(turns)
            for (registered, case_id), turns in SCRIPTS_BY_CASE.items()
            if registered == model
        }
    )

    unsupported = _unsupported_from(extra.get("unsupported")) or fixture.unsupported
    return _Declared(script=script, by_case=by_case, unsupported=unsupported)


def _load_fixture(raw: object) -> _Declared:
    """Read a script fixture off disk.

        [[candidates]]
        provider = "fake"
        model = "weak-4b"
        extra_body.script_file = "./fakes/weak-4b.toml"

    TOML or JSON, both in the standard library. Deliberately not YAML: TOML is
    already this tool's config language, and a fixture format that needs a
    third-party parser in the *core* install - which the test suite runs
    against - is a dependency bought for nothing.

    A fixture holds the same three keys a candidate can declare inline::

        unsupported = ["tools", "json_schema"]

        [[script]]
        kind = "fenced_json"
        text = '{"intent": "order_status"}'

        [script_by_case]
        refund_207 = [{ kind = "invalid_schema" }]

    Relative paths resolve against the directory holding ``eval.toml`` - the
    config loader rewrites them before the provider sees them - so a benchmark
    stays runnable from any working directory.
    """
    if not isinstance(raw, str):
        raise ValueError("extra_body.script_file must be a path")
    data = load_script_file(raw)
    return _Declared(
        script=_turns_from_list(data.get("script")),
        by_case=_case_scripts_from(data.get("script_by_case")),
        unsupported=_unsupported_from(data.get("unsupported")),
    )


def load_script_file(raw: str | Path) -> dict[str, Any]:
    """Read one fixture and return its table, validated.

    The config loader calls this and inlines the result into the candidate's
    ``extra_body``, which is what keeps the fixture's *content* rather than its
    path in the benchmark's config hash: the same fixture in two checkouts is
    the same benchmark, and an edited fixture is a different one.
    """
    path = Path(raw)
    try:
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ValueError(f"could not read fake script file {path}: {exc}") from exc

    try:
        data = json.loads(text) if path.suffix.lower() == ".json" else tomllib.loads(text)
    except (tomllib.TOMLDecodeError, json.JSONDecodeError) as exc:
        kind = path.suffix.lstrip(".").lower() or "toml"
        raise ValueError(f"{path} is not valid {kind}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path} must hold a table with a 'script' key, not a bare list")
    unknown = sorted(set(data) - EXTRA_BODY_KEYS)
    if unknown:
        raise ValueError(f"unknown key(s) in {path}: {', '.join(unknown)}")
    if "script_file" in data:
        raise ValueError(f"{path} cannot point at another script_file")

    # Building the turns *is* the validation; they are rebuilt per client.
    _turns_from_list(data.get("script"))
    _case_scripts_from(data.get("script_by_case"))
    _unsupported_from(data.get("unsupported"))
    return {key: value for key, value in data.items() if key in EXTRA_BODY_KEYS}


def _turns_from_list(raw: object) -> list[FakeTurn]:
    """One script, as a list of strings or tables."""
    if not isinstance(raw, list):
        return []
    return [_turn_from_entry(item) for item in raw]


_ENTRY_KEYS = frozenset(
    {
        "kind",
        "text",
        "tool_calls",
        "refuse_tool_call",
        "unsupported",
        "status",
        "error",
        "usage",
        "latency_ms",
    }
)


def _turn_from_entry(item: object) -> FakeTurn:
    """One turn, as written in TOML or JSON.

    ``kind`` names an entry of :data:`MISBEHAVIOURS` and every other key is an
    override on it, so the common cases are one word:

        script = [
            { kind = "fenced_json", text = '{"intent": "refund"}' },
            { kind = "rate_limit" },
            { kind = "valid_json", text = '{"done": true}', latency_ms = 250 },
        ]
    """
    if isinstance(item, str):
        return FakeTurn(text=item)
    if not isinstance(item, dict):
        raise ValueError(
            f"fake script entries must be strings or tables, got {type(item).__name__}"
        )
    unknown = sorted(set(item) - _ENTRY_KEYS)
    if unknown:
        raise ValueError(
            f"unknown key(s) in a fake script entry: {', '.join(unknown)} "
            f"(known: {', '.join(sorted(_ENTRY_KEYS))})"
        )

    usage = item.get("usage")
    fields: dict[str, Any] = {
        "refuse_tool_call": bool(item.get("refuse_tool_call", False)),
        "unsupported": frozenset(item.get("unsupported", ())),
        "status": item.get("status"),
        "error": item.get("error"),
        "usage": (int(usage[0]), int(usage[1])) if isinstance(usage, (list, tuple)) else None,
        "latency_ms": float(item.get("latency_ms", 0.0)),
    }
    if "tool_calls" in item:
        fields["tool_calls"] = _tool_calls_from_list(item["tool_calls"])

    kind = item.get("kind")
    if kind is None:
        return FakeTurn(text=item.get("text"), **fields)

    # `kind` builds the turn and the rest overrides it, so `text` means "the
    # payload this shape wraps" rather than "the literal message content".
    base = misbehaviour(str(kind), item.get("text"))
    explicit = {name: value for name, value in fields.items() if name in item}
    return replace(base, **explicit) if explicit else base


def _tool_calls_from_list(raw: object) -> list[ToolCall]:
    """Tool calls the scripted model requests, for hallucinated names and args.

        tool_calls = [{ name = "issue_refund", arguments = { order_id = "ORD-1" } }]
    """
    if not isinstance(raw, list):
        raise ValueError("tool_calls must be a list of tables")
    calls: list[ToolCall] = []
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping) or "name" not in item:
            raise ValueError("each tool_calls entry must be a table with a 'name'")
        arguments = item.get("arguments") or {}
        if not isinstance(arguments, Mapping):
            raise ValueError(f"tool_calls[{index}].arguments must be a table")
        calls.append(
            ToolCall(
                id=str(item.get("id") or f"call_{index}"),
                name=str(item["name"]),
                arguments=dict(arguments),
                raw_arguments=json.dumps(arguments),
            )
        )
    return calls


def _case_scripts_from(raw: object) -> dict[str, list[FakeTurn]]:
    """Per-case scripts, as a table of case id -> script.

        [baseline]
        provider = "fake"
        model = "scripted"
        extra_body.script = [{ text = '{"intent": "other"}' }]
        extra_body.script_by_case.refund_207 = [{ text = '{"intent": "refund"}' }]
    """
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("script_by_case must be a table of case_id -> script")
    return {str(case_id): _turns_from_list(turns) for case_id, turns in raw.items()}


def _unsupported_from(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("unsupported must be a list of rung names")
    return [str(r) for r in raw]


def _transport_error(kind: str, spec: ClientSpec) -> ProviderError:
    """A failure with no HTTP response, shaped like the real client's.

    Both land as :class:`ProviderUnreachable` because that is what the
    OpenAI-compatible client raises for a refused connect and for a read
    timeout, and the classification the harness owes the user - "the server
    was down, this is not a model regression" - follows from the type.
    """
    if kind == "timeout":
        return ProviderUnreachable(
            f"{spec.model} did not respond within {spec.read_timeout_s:g}s (scripted timeout)"
        )
    if kind == "unreachable":
        return ProviderUnreachable(
            f"cannot reach {spec.base_url or 'the scripted endpoint'} (scripted)"
        )
    raise ValueError(f"unknown fake error {kind!r}; expected 'timeout' or 'unreachable'")


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


__all__ = [
    "EXTRA_BODY_KEYS",
    "MISBEHAVIOURS",
    "SCRIPTS",
    "FakeClient",
    "FakeTurn",
    "clear_scripts",
    "load_script_file",
    "misbehaviour",
    "register_script",
    "validate_extra_body",
]
