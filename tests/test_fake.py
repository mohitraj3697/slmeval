"""The `fake` provider - the instrument we measure the instrument with.

Step 15's premise: a measurement tool cannot be tested against real models.
They are slow, they cost money, and - the part that actually matters - they are
not reproducible. "The model emitted a prose preamble, then a fenced block with
a trailing comma" is a real 4B failure and the single most useful one to have a
regression test for, and it is also the one thing a real 4B model will not do
again tomorrow.

So every shape of misbehaviour is scripted here, by name, and this file pins
what the harness *reports* for each one - which rung it landed on, whether the
parser had to help, how the failure is classified. If the fake drifts, every
number in the scorecard is unanchored, so the fake gets its own suite rather
than being tested incidentally through the runner.

Offline, no keys, no sockets: the whole file runs in well under a second.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import BaseModel

from slmeval.config import ModelCfg
from slmeval.config.loader import load_config
from slmeval.core.context import RunContext, use_context
from slmeval.core.trace import JsonStage
from slmeval.errors import ConfigError
from slmeval.providers.base import (
    ClientSpec,
    ProviderRequestError,
    ProviderServerError,
    ProviderUnreachable,
    RateLimited,
    StructuredOutputError,
    ToolSpec,
    UnsupportedFeature,
)
from slmeval.providers.fake import (
    MISBEHAVIOURS,
    FakeClient,
    FakeTurn,
    clear_scripts,
    misbehaviour,
    register_script,
)
from slmeval.providers.registry import build_client
from tests.conftest import write_config

ROUTING = '{"intent": "refund", "order_id": "ORD-1"}'


class Routing(BaseModel):
    intent: str
    order_id: str


def spec(**over: object) -> ClientSpec:
    return ClientSpec(id="cand", provider="fake", model="m", **over)  # type: ignore[arg-type]


def client(kind: str, payload: str | None = None, **over: object) -> FakeClient:
    """A client whose model always behaves one scripted way."""
    return FakeClient(spec(), script=[misbehaviour(kind, payload, **over)])


# --------------------------------------------------------------------------- #
# The catalogue is a contract
# --------------------------------------------------------------------------- #

# Step 15 names the shapes the harness has to be testable against. Pinned as a
# list, because these names appear in configs and in other test files: dropping
# or renaming one silently removes a row from this file's coverage.
REQUIRED = (
    "valid_json",
    "fenced_json",
    "prose_json",
    "invalid_schema",
    "hallucinated_tool",
    "hallucinated_args",
    "timeout",
    "rate_limit",
    "loop",
)


def test_every_shape_the_plan_names_has_a_name_here() -> None:
    assert set(REQUIRED) <= set(MISBEHAVIOURS)


def test_an_unknown_behaviour_names_the_ones_that_exist() -> None:
    with pytest.raises(ValueError, match="unknown fake behaviour 'fenced'"):
        misbehaviour("fenced")


def test_overrides_compose_with_a_shape() -> None:
    """A model can be two things at once: slow *and* unable to emit bare JSON."""
    turn = misbehaviour("fenced_json", ROUTING, latency_ms=800.0)
    assert turn.latency_ms == 800.0
    assert "```json" in (turn.text or "")


# --------------------------------------------------------------------------- #
# JSON shapes: what the extraction cascade reports
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_clean_json_is_reported_as_clean() -> None:
    result = await client("valid_json", ROUTING).complete("hi", schema=Routing)

    assert result.json_stage is JsonStage.DIRECT
    assert result.schema_lenient is False
    assert result.repair_attempts == 0
    assert result.parsed_as(Routing).order_id == "ORD-1"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "stage"),
    [
        ("fenced_json", JsonStage.FENCED),
        ("prose_json", JsonStage.BALANCED),
        ("trailing_comma", JsonStage.REPAIRED),
    ],
)
async def test_a_dirty_envelope_parses_and_is_flagged(kind: str, stage: JsonStage) -> None:
    """Each of these *passes*, and the scorecard has to say it only passed with
    help - that gap between schema_valid and schema_repaired is the clearest
    small-model signal the tool produces."""
    fake = FakeClient(spec(), script=[misbehaviour(kind, ROUTING)], unsupported=["tools"])
    result = await fake.complete("hi", schema=Routing)

    assert result.json_stage is stage
    assert result.schema_lenient is True
    assert result.parsed_as(Routing).intent == "refund"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["invalid_schema", "not_json", "empty"])
async def test_a_model_that_cannot_answer_the_schema_fails_loudly(kind: str) -> None:
    fake = client(kind)
    with pytest.raises(StructuredOutputError) as caught:
        await fake.complete("hi", schema=Routing)

    # Every rung was tried and then repairs were attempted, which is what makes
    # "this model cannot do structured output" a defensible claim rather than a
    # first-attempt verdict.
    assert caught.value.rungs_tried == ("tools", "json_schema", "json_mode", "prompt")
    assert caught.value.repair_attempts > 0
    assert fake.calls > len(caught.value.rungs_tried)


@pytest.mark.asyncio
async def test_a_refusal_arrives_as_prose_on_the_tools_rung() -> None:
    """The model ignored a forced tool call and answered in words. Common on
    small models, and a different failure from calling the tool badly."""
    fake = client("refusal")
    with pytest.raises(StructuredOutputError):
        await fake.complete("hi", schema=Routing)

    assert fake.seen[0]["rung"] == "tools"


# --------------------------------------------------------------------------- #
# Tool calls: hallucinated names and arguments
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_hallucinated_tool_name_reaches_the_caller_unchanged() -> None:
    """Never corrected, never dropped: tier 1 counts it, and a provider that
    quietly fixed the name would delete the finding."""
    fake = client("hallucinated_tool", "frobnicate_order")
    result = await fake.complete("hi", tools=[ToolSpec(name="fetch_order", parameters={})])

    assert [c.name for c in result.tool_calls] == ["frobnicate_order"]


@pytest.mark.asyncio
async def test_hallucinated_arguments_survive_with_their_raw_text() -> None:
    result = await client("hallucinated_args", "fetch_order").complete("hi")
    call = result.tool_calls[0]

    assert call.name == "fetch_order"
    assert "ordr_id" in call.arguments
    assert json.loads(call.raw_arguments or "{}") == call.arguments


@pytest.mark.asyncio
async def test_tool_calls_are_declarable_in_toml() -> None:
    """So a benchmark of a tool-hallucinating model needs no Python."""
    cfg = ModelCfg(
        id="h",
        provider="fake",
        model="h",
        extra_body={
            "script": [
                {"tool_calls": [{"name": "issue_refund", "arguments": {"order_id": "ORD-1"}}]}
            ]
        },
    )
    result = await build_client(cfg).complete("hi")

    assert [c.name for c in result.tool_calls] == ["issue_refund"]
    assert result.tool_calls[0].arguments == {"order_id": "ORD-1"}


# --------------------------------------------------------------------------- #
# Transport and HTTP failures
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["timeout", "unreachable"])
async def test_a_dead_endpoint_is_unreachable_not_a_model_failure(kind: str) -> None:
    """Both land as ProviderUnreachable, exactly as the real client's connect
    and read-timeout paths do, which is what puts the run on exit code 3
    instead of reporting "the server was down" as a regression."""
    with pytest.raises(ProviderUnreachable):
        await client(kind).complete("hi", schema=Routing)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("kind", "error"),
    [
        ("rate_limit", RateLimited),
        ("server_error", ProviderServerError),
        ("bad_request", ProviderRequestError),
    ],
)
async def test_http_failures_keep_their_own_types(kind: str, error: type[Exception]) -> None:
    """A 429 and a 400 are not the same event: one is worth retrying and one
    means our request was wrong. The fake has to preserve that distinction,
    because the retry policy is what reads it."""
    with pytest.raises(error):
        await client(kind).complete("hi", schema=Routing)


@pytest.mark.asyncio
async def test_a_scripted_error_name_is_refused_rather_than_ignored() -> None:
    with pytest.raises(ValueError, match="unknown fake error"):
        await FakeClient(spec(), script=[FakeTurn(error="flaky")]).complete("hi")


# --------------------------------------------------------------------------- #
# Rungs, usage, latency
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_server_that_rejects_a_rung_is_only_asked_once() -> None:
    fake = FakeClient(
        spec(),
        script=[FakeTurn(text=ROUTING, unsupported=frozenset({"tools", "json_schema"}))],
    )
    first = await fake.complete("hi", schema=Routing)
    await fake.complete("hi", schema=Routing)

    assert first.structured_mode_used == "json_mode"
    assert first.rungs_tried == ("tools", "json_schema", "json_mode")
    # The second call skips what the first learned: 900 calls must not each pay
    # two rejections to find out.
    assert [s["rung"] for s in fake.seen[-1:]] == ["json_mode"]


@pytest.mark.asyncio
async def test_rungs_can_be_declared_unavailable_for_the_whole_run() -> None:
    fake = FakeClient(spec(), script=[FakeTurn(text=ROUTING)], unsupported=["tools"])
    result = await fake.complete("hi", schema=Routing)

    assert "tools" not in result.rungs_tried
    assert fake.unavailable_rungs() == frozenset({"tools"})


@pytest.mark.asyncio
async def test_a_pinned_mode_with_nothing_left_to_try_is_refused() -> None:
    fake = FakeClient(
        spec(structured_output="tools"), script=[FakeTurn(text=ROUTING)], unsupported=["tools"]
    )
    with pytest.raises(UnsupportedFeature, match="pinned to 'tools'"):
        await fake.complete("hi", schema=Routing)


@pytest.mark.asyncio
async def test_reported_usage_is_used_and_absent_usage_is_flagged() -> None:
    told = FakeClient(spec(), script=[FakeTurn(text=ROUTING, usage=(11, 7))])
    silent = FakeClient(spec(), script=[FakeTurn(text=ROUTING)])

    counted = await told.complete("hi", schema=Routing)
    estimated = await silent.complete("hi", schema=Routing)

    assert (counted.prompt_tokens, counted.completion_tokens) == (11, 7)
    assert counted.tokens_estimated is False
    # Not zero and not silently exact: a fabricated number must never look
    # measured, or a $/case column built on it would read as evidence.
    assert estimated.tokens_estimated is True
    assert (estimated.total_tokens or 0) > 0


@pytest.mark.asyncio
async def test_a_scripted_latency_is_what_the_completion_reports() -> None:
    result = await client("valid_json", ROUTING, latency_ms=250.0).complete("hi", schema=Routing)
    assert result.latency_ms == 250.0


# --------------------------------------------------------------------------- #
# Script position
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_last_turn_repeats_so_one_turn_means_always() -> None:
    """Which is also what `loop` is: a model that answers this way forever,
    until the step budget stops the agent."""
    fake = client("loop", ROUTING)
    for _ in range(5):
        await fake.complete("hi", schema=Routing)

    assert fake.calls == 5
    assert {s["rung"] for s in fake.seen} == {"tools"}


@pytest.mark.asyncio
async def test_a_two_turn_script_means_fails_then_recovers() -> None:
    fake = FakeClient(
        spec(), script=[misbehaviour("not_json"), misbehaviour("valid_json", ROUTING)]
    )
    result = await fake.complete("hi", schema=Routing)

    # The first turn failed the `tools` rung, the second answered `json_schema`.
    assert result.rungs_tried[:2] == ("tools", "json_schema")
    assert result.parsed_as(Routing).intent == "refund"


@pytest.mark.asyncio
async def test_every_execution_starts_at_turn_zero() -> None:
    """The runner shares one client per model across the whole matrix - it has
    to, because that is what shares the per-endpoint concurrency limit. If the
    position lived on the client, execution 2 would start at the second turn
    and fail a schema the model answers correctly: a failure the harness
    invented, in the one provider whose job is to be predictable."""
    fake = FakeClient(
        spec(), script=[misbehaviour("valid_json", ROUTING), misbehaviour("not_json")]
    )

    for _ in range(3):
        ctx = RunContext(run_id="r", candidate_id="cand", case_id="c")
        with use_context(ctx):
            result = await fake.complete("hi", schema=Routing)
        assert result.json_stage is JsonStage.DIRECT

    assert fake.calls == 3


# --------------------------------------------------------------------------- #
# Where a script comes from
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_registered_script_is_addressable_as_a_model_name() -> None:
    register_script("pinned", [misbehaviour("valid_json", ROUTING)])
    try:
        result = await build_client(
            ModelCfg(id="c", provider="fake", model="pinned")
        ).complete("hi", schema=Routing)
        assert result.parsed_as(Routing).intent == "refund"
    finally:
        clear_scripts()


@pytest.mark.asyncio
async def test_a_per_case_script_answers_that_case_and_the_rest_fall_back() -> None:
    """One model has to answer a whole dataset, and a shared script would have
    the scripted baseline fail most of its own cases: the routing answer for
    "where is my order" is the wrong answer to "my package never turned up"."""
    cfg = ModelCfg(
        id="c",
        provider="fake",
        model="scripted",
        extra_body={
            "script": [{"text": '{"intent": "other", "order_id": ""}'}],
            "script_by_case": {"refund_207": [{"text": ROUTING}]},
        },
    )
    fake = build_client(cfg)

    answers = {}
    for case_id in ("refund_207", "something_else"):
        ctx = RunContext(run_id="r", candidate_id="c", case_id=case_id)
        with use_context(ctx):
            answers[case_id] = (await fake.complete("hi", schema=Routing)).parsed_as(Routing)

    assert answers["refund_207"].order_id == "ORD-1"
    assert answers["something_else"].intent == "other"


@pytest.mark.asyncio
async def test_a_kind_in_toml_builds_the_same_turn_as_in_python() -> None:
    cfg = ModelCfg(
        id="c",
        provider="fake",
        model="weak",
        extra_body={
            "unsupported": ["tools", "json_schema"],
            "script": [{"kind": "fenced_json", "text": ROUTING}],
        },
    )
    result = await build_client(cfg).complete("hi", schema=Routing)

    assert result.structured_mode_used == "json_mode"
    assert result.json_stage is JsonStage.FENCED


def test_an_unknown_extra_body_key_is_refused() -> None:
    """`extra_body` is provider passthrough, so a typo would otherwise leave a
    fake that synthesizes schema-valid blanks - a run that looks like a pass
    and measures nothing."""
    with pytest.raises(ValueError, match="does not understand extra_body key"):
        build_client(ModelCfg(id="c", provider="fake", model="m", extra_body={"scripts": []}))


def test_an_unknown_key_inside_a_turn_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown key"):
        build_client(
            ModelCfg(id="c", provider="fake", model="m", extra_body={"script": [{"txt": "x"}]})
        )


# --------------------------------------------------------------------------- #
# Script fixture files
# --------------------------------------------------------------------------- #

FIXTURE = """\
unsupported = ["tools", "json_schema"]

[[script]]
kind = "fenced_json"
text = '{"intent": "refund", "order_id": "ORD-1"}'

[script_by_case]
quiet = [{ kind = "empty" }]
"""


def problem_in(path: Path) -> str:
    """Load a config expecting it to fail, and return the problem's message.

    The top line of a ConfigError names the model; what tells the user what to
    fix is the problem under it, with the key and the line it is on.
    """
    with pytest.raises(ConfigError) as caught:
        load_config(path)
    problems = caught.value.problems
    assert len(problems) == 1, problems
    return problems[0].message


def config_with(body: str, directory: Path) -> Path:
    return write_config(
        directory,
        """\
[benchmark]
name = "t"
dataset = "./datasets/cases.jsonl"

[target]
entrypoint = "mod:run_case"

"""
        + body,
    )


@pytest.mark.asyncio
async def test_a_fixture_file_keeps_a_long_script_out_of_eval_toml(tmp_path: Path) -> None:
    (tmp_path / "fakes").mkdir()
    (tmp_path / "fakes" / "weak.toml").write_text(FIXTURE, encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./fakes/weak.toml"
""",
        tmp_path,
    )
    loaded = load_config(path)
    result = await build_client(loaded.config.baseline).complete("hi", schema=Routing)

    assert result.json_stage is JsonStage.FENCED
    assert result.structured_mode_used == "json_mode"


def test_a_fixture_is_inlined_so_the_config_hash_follows_its_content(tmp_path: Path) -> None:
    """The path must not reach the hash: the same benchmark in two checkouts is
    the same benchmark. Its content must: an edited fixture is a different one."""
    (tmp_path / "fakes").mkdir()
    fixture = tmp_path / "fakes" / "weak.toml"
    fixture.write_text(FIXTURE, encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./fakes/weak.toml"
""",
        tmp_path,
    )

    first = load_config(path)
    assert "script_file" not in first.config.baseline.extra_body
    assert first.config.baseline.extra_body["unsupported"] == ["tools", "json_schema"]

    fixture.write_text(FIXTURE.replace("refund", "order_status"), encoding="utf-8")
    assert load_config(path).config.config_hash() != first.config.config_hash()


def test_a_fixture_resolves_against_eval_toml_not_the_shell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "fakes").mkdir()
    (tmp_path / "fakes" / "weak.toml").write_text(FIXTURE, encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./fakes/weak.toml"
""",
        tmp_path,
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_config(path).config.config_hash()


@pytest.mark.asyncio
async def test_a_json_fixture_works_too(tmp_path: Path) -> None:
    (tmp_path / "weak.json").write_text(
        json.dumps({"script": [{"kind": "valid_json", "text": ROUTING}]}), encoding="utf-8"
    )
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./weak.json"
""",
        tmp_path,
    )
    loaded = load_config(path)
    result = await build_client(loaded.config.baseline).complete("hi", schema=Routing)

    assert result.parsed_as(Routing).order_id == "ORD-1"


def test_an_inline_per_case_script_overrides_the_fixtures(tmp_path: Path) -> None:
    (tmp_path / "weak.toml").write_text(FIXTURE, encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./weak.toml"
extra_body.script_by_case.quiet = [{ kind = "valid_json", text = '{"a": 1}' }]
""",
        tmp_path,
    )
    extra = load_config(path).config.baseline.extra_body

    assert extra["script_by_case"]["quiet"] == [{"kind": "valid_json", "text": '{"a": 1}'}]


def test_two_scripts_for_one_model_is_a_config_error(tmp_path: Path) -> None:
    (tmp_path / "weak.toml").write_text(FIXTURE, encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./weak.toml"
extra_body.script = [{ kind = "empty" }]
""",
        tmp_path,
    )
    assert "not both" in problem_in(path)


def test_a_missing_fixture_fails_at_load_with_the_key_that_named_it(tmp_path: Path) -> None:
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./nope.toml"
""",
        tmp_path,
    )
    with pytest.raises(ConfigError) as caught:
        load_config(path)

    problem = caught.value.problems[0]
    assert problem.path == "baseline.extra_body.script_file"
    assert problem.line == 11
    assert "could not read" in problem.message


def test_a_bad_behaviour_name_in_a_fixture_is_a_config_error(tmp_path: Path) -> None:
    """Not an exception thirty executions into the matrix - where the outcome
    classifier would correctly, and uselessly, call a user's typo a harness
    bug."""
    (tmp_path / "weak.toml").write_text('[[script]]\nkind = "fenced"\n', encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./weak.toml"
""",
        tmp_path,
    )
    assert "unknown fake behaviour" in problem_in(path)


def test_a_malformed_fixture_says_which_file(tmp_path: Path) -> None:
    (tmp_path / "weak.toml").write_text("[[script]\nkind = 1\n", encoding="utf-8")
    path = config_with(
        """\
[baseline]
provider = "fake"
model = "weak"
extra_body.script_file = "./weak.toml"
""",
        tmp_path,
    )
    message = problem_in(path)
    assert "not valid toml" in message
    assert "weak.toml" in message
