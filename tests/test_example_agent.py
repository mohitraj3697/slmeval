"""The example agent, end to end - and the week 1 acceptance criterion.

The claim this file has to make true:

    record a real 5-step agent run once, then replay it 200 times offline, with
    no network access, in under two seconds total.

So the replay test blocks ``socket.socket`` outright rather than trusting that
nothing reaches out. "Offline" asserted by inspection is "offline" until someone
adds a telemetry call; asserted by a blocked socket, it stays true.

It also runs the whole stack together for the first time - adapter, context,
router, provider ladder, cassette store, envelope - which is the point of having
an example at all. Each piece passed its own tests on Day 5 and 6; this is where
they have to agree with each other.
"""

from __future__ import annotations

import socket
import sys
import time
from collections.abc import Iterator
from decimal import Decimal
from pathlib import Path

import pytest

from slmeval.config.models import ModelCfg
from slmeval.core.adapter import Target, execute_case
from slmeval.core.context import RunContext
from slmeval.core.router import ModelRouter
from slmeval.providers.fake import FakeTurn, clear_scripts, register_script
from slmeval.sandbox.keys import MatchTier
from slmeval.sandbox.store import CassetteStore
from slmeval.sandbox.tool import CassetteMiss

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

ROUTING_JSON = '{"intent": "order_status", "order_id": "ORD-9921", "wants_confirmation": false}'
REPLY_JSON = '{"final_answer": "Your order is with DHL and should arrive in two days."}'

#: The agent makes two model calls per execution: route, then compose.
SCRIPT = [FakeTurn(text=ROUTING_JSON), FakeTurn(text=REPLY_JSON)]

CASE_INPUT = {"user_query": "Where is my order ORD-9921?"}
CASE_ID = "status_shipped"


@pytest.fixture(autouse=True)
def example_on_path() -> Iterator[None]:
    entry = str(EXAMPLES)
    sys.path.insert(0, entry)
    register_script("scripted-baseline", SCRIPT)
    try:
        yield
    finally:
        clear_scripts()
        if entry in sys.path:
            sys.path.remove(entry)
        for name in [m for m in sys.modules if m.startswith("support_agent")]:
            del sys.modules[name]


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CassetteStore]:
    with CassetteStore(tmp_path / "cassettes.sqlite") as store:
        yield store


@pytest.fixture
def target() -> Target:
    return Target.load("support_agent:run_case", root=EXAMPLES)


def make_context(store: CassetteStore, mode: str, attempt: int = 0) -> RunContext:
    """A fresh context and router per execution, as the runner will do.

    Fresh router per execution because the scripted provider consumes its turns
    in order; sharing one across 200 executions would have the second execution
    start at the compose turn.
    """
    baseline = ModelCfg(id="baseline", provider="fake", model="scripted-baseline")
    return RunContext(
        run_id="run-1",
        candidate_id="baseline",
        case_id=CASE_ID,
        attempt=attempt,
        tool_mode=mode,
        max_steps=12,
        cassettes=store,
        router=ModelRouter(candidate=baseline),
    )


# --------------------------------------------------------------------------- #
# Record
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_recording_the_example_captures_its_whole_tool_trace(
    store: CassetteStore, target: Target
) -> None:
    ctx = make_context(store, "record")
    state = await execute_case(ctx, target, CASE_INPUT)

    assert state["intent"] == "order_status"
    assert state["order_id"] == "ORD-9921"
    assert state["refund_issued"] is False
    assert "DHL" in state["final_answer"]

    # Four tools, four nodes, no refund: the happy path the golden case asserts.
    assert [t.name for t in ctx.tool_calls] == [
        "fetch_order",
        "fetch_customer",
        "fetch_tracking",
        "check_refund_policy",
    ]
    assert {t.outcome for t in ctx.tool_calls} == {"recorded"}
    assert ctx.node_names == [
        "intent_router",
        "order_lookup",
        "policy_check",
        "compose_reply",
    ]
    # Three tiers each for the tools that declare `match_on`, two for
    # check_refund_policy, which declares none and so has no subset tier.
    assert len(store) == 3 + 3 + 3 + 2


@pytest.mark.asyncio
async def test_the_forbidden_tool_is_not_called_on_the_happy_path(
    store: CassetteStore, target: Target
) -> None:
    ctx = make_context(store, "record")
    await execute_case(ctx, target, CASE_INPUT)
    assert "issue_refund" not in {t.name for t in ctx.tool_calls}


@pytest.mark.asyncio
async def test_rich_types_round_trip_through_the_real_agent(
    store: CassetteStore, target: Target
) -> None:
    # The example's `assert isinstance(order.total, Decimal)` runs inside the
    # agent on replay, so if the envelope flattened anything this raises there.
    # Here we check it from the outside too.
    await execute_case(make_context(store, "record"), target, CASE_INPUT)

    from support_agent import tools  # type: ignore[import-not-found]

    policy = tools.fetch_order.slmeval_tool.policy
    recorded_args = {"order_id": "ORD-9921", "request_id": "req-ORD-9921"}
    hit = store.lookup(policy.compute(CASE_ID, recorded_args))
    assert hit is not None
    order = hit.response
    assert isinstance(order.total, Decimal)
    assert order.state.value == "shipped"
    assert hit.tier is MatchTier.EXACT


# --------------------------------------------------------------------------- #
# Replay - the acceptance criterion
# --------------------------------------------------------------------------- #


class _NoNetwork:
    """Make any socket creation fail for the duration of the block.

    The whole promise of replay is that a benchmark run in CI needs no GPU, no
    API keys and no network. Blocking the socket is the only way to assert the
    third part and keep it asserted.
    """

    def __enter__(self) -> None:
        self._real = socket.socket

        def blocked(*args: object, **kwargs: object) -> None:
            raise AssertionError("replay attempted to open a socket")

        socket.socket = blocked  # type: ignore[assignment, misc]

    def __exit__(self, *exc: object) -> None:
        socket.socket = self._real  # type: ignore[misc]


@pytest.mark.asyncio
async def test_record_once_then_replay_200_times_offline_under_two_seconds(
    store: CassetteStore, target: Target
) -> None:
    await execute_case(make_context(store, "record"), target, CASE_INPUT)

    started = time.perf_counter()
    with _NoNetwork():
        for attempt in range(200):
            ctx = make_context(store, "replay", attempt=attempt)
            state = await execute_case(ctx, target, CASE_INPUT)
            assert state["order_id"] == "ORD-9921"
            assert len(ctx.tool_calls) == 4
            assert {t.outcome for t in ctx.tool_calls} == {"replayed"}
    elapsed = time.perf_counter() - started

    assert elapsed < 2.0, f"200 replays took {elapsed:.2f}s"
    # Every replay matched exactly: no normalizer and no subset fallback was
    # needed, so none of this is leaning on our own leniency.
    assert store.misses == 0


@pytest.mark.asyncio
async def test_replay_is_deterministic(store: CassetteStore, target: Target) -> None:
    # Comparability is the reason replay exists: if two replays of one case can
    # differ, no two candidates can be compared.
    await execute_case(make_context(store, "record"), target, CASE_INPUT)
    first = await execute_case(make_context(store, "replay"), target, CASE_INPUT)
    second = await execute_case(make_context(store, "replay"), target, CASE_INPUT)
    assert first == second


@pytest.mark.asyncio
async def test_every_repeat_of_a_case_replays_the_same_responses(
    store: CassetteStore, target: Target
) -> None:
    # Cassettes are scoped per case, not per attempt. If attempt 2 saw different
    # tool responses than attempt 1, pass^k would be measuring the harness.
    await execute_case(make_context(store, "record", attempt=0), target, CASE_INPUT)
    keys = [
        t.cassette_key
        for t in (await _traced(store, target, attempt=1))
    ]
    again = [t.cassette_key for t in (await _traced(store, target, attempt=2))]
    assert keys == again


async def _traced(store: CassetteStore, target: Target, attempt: int) -> list:
    ctx = make_context(store, "replay", attempt=attempt)
    await execute_case(ctx, target, CASE_INPUT)
    return ctx.tool_calls


@pytest.mark.asyncio
async def test_a_hallucinated_order_id_misses_instead_of_hitting_the_erp(
    store: CassetteStore, target: Target
) -> None:
    await execute_case(make_context(store, "record"), target, CASE_INPUT)

    # The model drops the prefix - "9921" instead of "ORD-9921" - which is the
    # headline failure the tool exists to catch.
    register_script(
        "scripted-baseline",
        [
            FakeTurn(
                text='{"intent": "order_status", "order_id": "9921", '
                '"wants_confirmation": false}'
            ),
            FakeTurn(text=REPLY_JSON),
        ],
    )
    with pytest.raises(CassetteMiss) as caught:
        await execute_case(make_context(store, "replay"), target, CASE_INPUT)

    assert caught.value.tool_name == "fetch_order"
    assert caught.value.arguments["order_id"] == "9921"
    assert caught.value.recorded_calls == 1


@pytest.mark.asyncio
async def test_a_case_strict_replay_has_never_seen_is_a_miss(
    store: CassetteStore, target: Target
) -> None:
    await execute_case(make_context(store, "record"), target, CASE_INPUT)
    ctx = make_context(store, "replay")
    ctx.case_id = "a_case_nobody_recorded"
    with pytest.raises(CassetteMiss):
        await execute_case(ctx, target, CASE_INPUT)
