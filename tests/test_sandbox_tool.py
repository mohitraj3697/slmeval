"""``@sandbox_tool`` and the four modes.

The claim being tested over and over, in different shapes: **no path falls
through to the real tool by accident.** In a benchmark, silently calling a
customer's live API instead of reporting a miss would spend their money, produce
a number nobody can reproduce, and do both without saying anything. So the tests
below count real invocations rather than only checking return values - the count
is the safety property.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest

from slmeval.core.context import (
    RunContext,
    SlmevalContextWarning,
    StepBudgetExceeded,
    benchmark_running,
    node,
    reset_warnings,
    use_context,
)
from slmeval.sandbox.envelope import UnserializableResponse
from slmeval.sandbox.faults import (
    Delivery,
    FaultKind,
    RateLimited,
    compare_vectors,
    fault_schedule,
)
from slmeval.sandbox.keys import MatchTier, PolicyError
from slmeval.sandbox.store import CassetteStore
from slmeval.sandbox.tool import (
    REGISTRY,
    CassetteMiss,
    SandboxNotReady,
    reset_live_banner,
    sandbox_tool,
)


@pytest.fixture(autouse=True)
def fresh_process_state() -> Iterator[None]:
    """The warn-once and banner-once memos are process-wide, so a test that
    asserts a warning fires has to start from a clean slate. (The tool registry
    itself is restored by an autouse fixture in conftest.)"""
    reset_live_banner()
    reset_warnings()
    yield


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CassetteStore]:
    with CassetteStore(tmp_path / "cassettes.sqlite") as store:
        yield store


def context(store: CassetteStore | None, mode: str = "replay", **kwargs: Any) -> RunContext:
    return RunContext(
        run_id="run-1",
        candidate_id="gemma3-4b",
        case_id="status_shipped",
        tool_mode=mode,
        cassettes=store,
        **kwargs,
    )


class Counter:
    """Counts real invocations - the thing that must stay at zero on replay."""

    def __init__(self) -> None:
        self.calls = 0


# --------------------------------------------------------------------------- #
# record / replay
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_then_replay_never_calls_the_tool_again(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool(match_on=["order_id"], ignore_args=["request_id"])
    async def fetch_order(order_id: str, request_id: str = "") -> dict:
        counter.calls += 1
        return {"order_id": order_id, "state": "shipped"}

    with use_context(context(store, "record")):
        assert await fetch_order("ORD-9921", request_id="r1") == {
            "order_id": "ORD-9921",
            "state": "shipped",
        }
    assert counter.calls == 1

    with use_context(context(store, "replay")) as ctx:
        for _ in range(20):
            assert await fetch_order("ORD-9921", request_id="r-different") == {
                "order_id": "ORD-9921",
                "state": "shipped",
            }
    assert counter.calls == 1
    assert {t.outcome for t in ctx.tool_calls} == {"replayed"}


@pytest.mark.asyncio
async def test_positional_and_keyword_calls_are_the_same_call(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {"ok": True}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")
    with use_context(context(store, "replay")):
        await fetch_order(order_id="ORD-9921")
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_a_default_the_model_spells_out_still_replays(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool
    async def search(query: str, limit: int = 10) -> list[str]:
        counter.calls += 1
        return ["ORD-1"]

    with use_context(context(store, "record")):
        await search("orders")
    with use_context(context(store, "replay")):
        await search("orders", limit=10)
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_rich_return_types_replay_as_themselves(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_total(order_id: str) -> dict:
        return {"total": Decimal("64.50"), "at": datetime(2026, 1, 1, tzinfo=UTC)}

    with use_context(context(store, "record")):
        recorded = await fetch_total("ORD-1")
    with use_context(context(store, "replay")):
        replayed = await fetch_total("ORD-1")

    assert replayed == recorded
    assert isinstance(replayed["total"], Decimal)
    assert isinstance(replayed["at"], datetime)


@pytest.mark.asyncio
async def test_a_sync_tool_is_sandboxed_too(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool(match_on=["sku"])
    def price_of(sku: str) -> Decimal:
        counter.calls += 1
        return Decimal("9.99")

    with use_context(context(store, "record")):
        assert price_of("A-1") == Decimal("9.99")
    with use_context(context(store, "replay")):
        assert price_of("A-1") == Decimal("9.99")
    assert counter.calls == 1


# --------------------------------------------------------------------------- #
# Misses
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_strict_replay_miss_raises_and_does_not_call_the_tool(
    store: CassetteStore,
) -> None:
    counter = Counter()

    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {"ok": True}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    with use_context(context(store, "replay")) as ctx, pytest.raises(CassetteMiss) as caught:
        await fetch_order("9921")

    # The safety property: the tool was NOT called to cover for the miss.
    assert counter.calls == 1
    assert caught.value.tool_name == "fetch_order"
    assert caught.value.arguments == {"order_id": "9921"}
    assert caught.value.recorded_calls == 1
    assert ctx.tool_calls[-1].outcome == "miss"


@pytest.mark.asyncio
async def test_a_miss_carries_its_diagnosis(store: CassetteStore) -> None:
    # The miss carries the structured finding, and its message *is* the
    # rendered block - so the runner persists one object and a log, a traceback
    # and a terminal all show the same thing.
    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        return {}

    with (
        use_context(context(store, "replay")),
        node("order_lookup"),
        pytest.raises(CassetteMiss) as caught,
    ):
        await fetch_order("9921")

    miss = caught.value
    assert miss.case_id == "status_shipped"
    assert miss.candidate_id == "gemma3-4b"
    assert miss.node == "order_lookup"
    assert miss.keys.exact
    assert miss.recorded_calls == 0
    assert "CASSETTE_MISS" in str(miss)
    assert miss.diagnosis is not None
    assert str(miss) == miss.diagnosis.render()
    # The tool is registered but nothing at all was recorded for this case, so
    # there is nothing to compare against. Calling that a hallucination would
    # be the harness inventing a finding out of its own empty store.
    assert miss.verdict == "unrecorded_valid_path"
    assert miss.uncovered


@pytest.mark.asyncio
async def test_record_mode_overwrites_instead_of_replaying(store: CassetteStore) -> None:
    # `record` means "execute real, write". If it consulted the store first it
    # would silently behave like `auto`, so re-recording a case after changing a
    # fixture would appear to do nothing - and the user would be comparing
    # models against stale tool responses without being told.
    counter = Counter()

    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {"n": counter.calls}

    with use_context(context(store, "record")):
        assert await fetch_order("ORD-1") == {"n": 1}
        assert await fetch_order("ORD-1") == {"n": 2}

    assert counter.calls == 2
    with use_context(context(store, "replay")):
        assert await fetch_order("ORD-1") == {"n": 2}


@pytest.mark.asyncio
async def test_auto_records_a_miss_instead_of_raising(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {"n": counter.calls}

    with use_context(context(store, "auto")) as ctx:
        assert await fetch_order("ORD-1") == {"n": 1}
        assert await fetch_order("ORD-1") == {"n": 1}
        assert await fetch_order("ORD-2") == {"n": 2}

    assert counter.calls == 2
    assert [t.outcome for t in ctx.tool_calls] == ["recorded", "replayed", "recorded"]


# --------------------------------------------------------------------------- #
# Degraded matches are reported
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_normalized_hit_is_warned_about(store: CassetteStore) -> None:
    @sandbox_tool(match_on=["order_id"], normalize={"order_id": "upper_strip"})
    async def fetch_order(order_id: str) -> dict:
        return {"ok": True}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    with use_context(context(store, "replay")) as ctx:
        await fetch_order(" ord-9921 ")

    assert ctx.tool_calls[-1].match_tier == MatchTier.NORMALIZED.value
    # A run that only passes because our normalizers covered for the model must
    # not look clean in the scorecard.
    assert any("normalizing arguments" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_a_subset_hit_says_the_other_arguments_were_wrong(
    store: CassetteStore,
) -> None:
    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str, include_lines: bool = False) -> dict:
        return {"ok": True}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    with use_context(context(store, "replay")) as ctx:
        await fetch_order("ORD-9921", include_lines=True)

    assert ctx.tool_calls[-1].match_tier == MatchTier.SUBSET.value
    assert any("got the other arguments wrong" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_replaying_a_cassette_for_edited_code_is_flagged(store: CassetteStore) -> None:
    # Staleness is reported, not turned into a miss: a wall of misses after an
    # edit reads like the model fell apart.
    @sandbox_tool(version="v1")
    async def fetch_order(order_id: str) -> dict:
        return {"ok": True}

    with use_context(context(store, "record")):
        await fetch_order("ORD-1")

    # The user edits the tool and re-runs, so the module imports afresh and
    # registers the new version under the same name.
    REGISTRY.pop("fetch_order")

    @sandbox_tool(version="v2", name="fetch_order")
    async def fetch_order_v2(order_id: str) -> dict:
        return {"ok": True, "extra": "added in v2"}

    with use_context(context(store, "replay")) as ctx:
        # Still replays the v1 payload - the cassette is stale, not invalid.
        assert await fetch_order_v2("ORD-1") == {"ok": True}

    assert ctx.tool_calls[-1].outcome == "replayed"
    assert any("cassette verify" in w for w in ctx.warnings)


# --------------------------------------------------------------------------- #
# The modes at their edges
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_live_mode_calls_through_and_warns_once(
    store: CassetteStore, capsys: pytest.CaptureFixture[str]
) -> None:
    counter = Counter()

    @sandbox_tool
    async def charge_card(amount: Decimal) -> dict:
        counter.calls += 1
        return {"ok": True}

    with use_context(context(store, "live")) as ctx:
        await charge_card(Decimal("1"))
        await charge_card(Decimal("2"))

    assert counter.calls == 2
    assert len(store) == 0
    banner = capsys.readouterr().err
    # A user who reaches live mode by accident must not learn about it from
    # their cloud bill.
    assert "REAL APIs" in banner
    assert banner.count("REAL APIs") == 1
    assert [t.outcome for t in ctx.tool_calls] == ["live", "live"]


@pytest.mark.asyncio
async def test_no_store_in_replay_mode_raises_rather_than_calling_the_tool() -> None:
    counter = Counter()

    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {}

    with use_context(context(None, "replay")), pytest.raises(SandboxNotReady):
        await fetch_order("ORD-1")
    assert counter.calls == 0


@pytest.mark.asyncio
async def test_no_context_is_a_production_passthrough(store: CassetteStore) -> None:
    counter = Counter()

    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        counter.calls += 1
        return {"ok": True}

    # No benchmark running: calling the real tool is exactly right, and a
    # warning here would be noise a user learns to ignore.
    with _no_warning():
        assert await fetch_order("ORD-1") == {"ok": True}
    assert counter.calls == 1


@pytest.mark.asyncio
async def test_a_context_lost_during_a_benchmark_is_loud(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"ok": True}

    # Same situation from inside the tool - no context - but a benchmark IS
    # running, so this is a context lost crossing a thread boundary, and that
    # silently spends money.
    with benchmark_running(), pytest.warns(SlmevalContextWarning, match="fetch_order"):
        await fetch_order("ORD-1")


@pytest.mark.asyncio
async def test_the_context_survives_a_sync_agent_in_a_worker_thread(
    store: CassetteStore,
) -> None:
    # contextvars propagate through asyncio.to_thread but not through a bare
    # Thread. The adapter uses to_thread for exactly this reason.
    @sandbox_tool
    def fetch_order(order_id: str) -> dict:
        return {"ok": True}

    def agent() -> dict:
        return fetch_order("ORD-1")

    with use_context(context(store, "record")) as ctx:
        assert await asyncio.to_thread(agent) == {"ok": True}
    assert [t.outcome for t in ctx.tool_calls] == ["recorded"]


@pytest.mark.asyncio
async def test_a_bare_thread_loses_the_context_and_says_so(store: CassetteStore) -> None:
    @sandbox_tool
    def fetch_order(order_id: str) -> dict:
        return {"ok": True}

    captured: list[Any] = []

    def worker() -> None:
        with pytest.warns(SlmevalContextWarning):
            captured.append(fetch_order("ORD-1"))

    with use_context(context(store, "record")), benchmark_running():
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()

    assert captured == [{"ok": True}]


# --------------------------------------------------------------------------- #
# Failure paths
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_unserializable_return_refuses_to_record(store: CassetteStore) -> None:
    class Connection:
        pass

    @sandbox_tool
    async def open_db(dsn: str) -> Any:
        return {"conn": Connection()}

    with use_context(context(store, "record")), pytest.raises(UnserializableResponse):
        await open_db("postgres://")
    assert len(store) == 0


@pytest.mark.asyncio
async def test_a_tool_that_raises_is_traced_and_the_error_propagates(
    store: CassetteStore,
) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        raise KeyError(order_id)

    with use_context(context(store, "record")) as ctx, pytest.raises(KeyError):
        await fetch_order("NOPE")

    assert len(store) == 0
    assert ctx.tool_calls[-1].outcome == "error"
    assert "KeyError" in (ctx.tool_calls[-1].error or "")


@pytest.mark.asyncio
async def test_calling_a_tool_wrongly_still_raises_typeerror(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {}

    # The real tool would raise this, so the sandbox must not swallow it into a
    # cassette miss and blame the model for the wrong thing.
    with use_context(context(store, "replay")), pytest.raises(TypeError):
        await fetch_order(nope=1)  # type: ignore[call-arg]


@pytest.mark.asyncio
async def test_a_runaway_loop_hits_the_step_budget(store: CassetteStore) -> None:
    @sandbox_tool
    async def poke(n: int) -> dict:
        return {"n": n}

    with (
        use_context(context(store, "auto", max_steps=3)),
        pytest.raises(StepBudgetExceeded),
    ):
        for i in range(10):
            await poke(i)


# --------------------------------------------------------------------------- #
# Traces and the registry
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_every_call_leaves_exactly_one_trace(store: CassetteStore) -> None:
    @sandbox_tool
    async def ok(x: int) -> dict:
        return {}

    @sandbox_tool
    async def boom(x: int) -> dict:
        raise RuntimeError("no")

    with use_context(context(store, "auto")) as ctx:
        await ok(1)
        with pytest.raises(RuntimeError):
            await boom(1)
        await ok(1)

    # Three calls, three traces. Appended in `finally`, because a trace recorded
    # per-except branch is how the denominator of every per-tool rate goes wrong.
    assert len(ctx.tool_calls) == 3
    assert [t.seq for t in ctx.tool_calls] == [0, 1, 2]


@pytest.mark.asyncio
async def test_a_trace_attributes_the_call_to_its_node(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {}

    with use_context(context(store, "record")) as ctx, node("order_lookup"), node("erp"):
        await fetch_order("ORD-1")

    trace = ctx.tool_calls[-1]
    assert trace.node == "erp"
    assert trace.node_path == ("order_lookup", "erp")


def test_a_tool_is_registered_under_its_name() -> None:
    @sandbox_tool(match_on=["order_id"])
    async def fetch_order(order_id: str) -> dict:
        return {}

    assert REGISTRY["fetch_order"].policy.match_on == ("order_id",)
    assert REGISTRY["fetch_order"].is_async is True


def test_two_different_tools_cannot_share_a_name() -> None:
    # Cassettes are keyed by tool name, so a silent collision would have these
    # two answering each other's recordings.
    @sandbox_tool(name="fetch")
    async def fetch_a(x: int) -> dict:
        return {}

    with pytest.raises(PolicyError, match="both named"):

        @sandbox_tool(name="fetch")
        async def fetch_b(x: int) -> dict:
            return {}


def test_a_policy_typo_is_caught_when_the_module_imports() -> None:
    with pytest.raises(PolicyError):

        @sandbox_tool(match_on=["order_number"])
        async def fetch_order(order_id: str) -> dict:
            return {}


def test_the_decorator_preserves_the_function_it_wraps() -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        """Docstring kept."""
        return {}

    assert fetch_order.__name__ == "fetch_order"
    assert fetch_order.__doc__ == "Docstring kept."
    assert fetch_order.slmeval_tool.name == "fetch_order"


class _no_warning:
    """Assert no SlmevalContextWarning is emitted inside the block."""

    def __enter__(self) -> None:
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._log = self._ctx.__enter__()
        warnings.simplefilter("always")

    def __exit__(self, *exc: object) -> None:
        caught = [w for w in self._log if issubclass(w.category, SlmevalContextWarning)]
        self._ctx.__exit__(*exc)
        assert not caught, f"unexpected context warning: {[str(w.message) for w in caught]}"


# --------------------------------------------------------------------------- #
# Fault injection, through the decorator
# --------------------------------------------------------------------------- #


def faults(*kinds: FaultKind, rate: float = 1.0, seed: int = 42, **kwargs: Any) -> Any:
    """A schedule with the waits turned off. The delays are real by default and
    deliberately so, but a suite that sat out a 2-second timeout per fault is a
    suite people stop running."""
    return fault_schedule(
        seed, "status_shipped", rate=rate, kinds=list(kinds) or None, delay_scale=0.0, **kwargs
    )


@pytest.mark.asyncio
async def test_a_returned_fault_replaces_the_replayed_payload(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"order_id": order_id, "status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx):
        assert await fetch_order("ORD-9921") == {}

    trace = ctx.tool_calls[-1]
    assert trace.fault == "empty_result:return"
    assert trace.outcome == "fault"
    # The cassette still answered, and the trace says so: a faulted call is
    # not a miss, and the per-tool replay rate must not lose it.
    assert trace.cassette_key
    assert trace.match_tier == MatchTier.EXACT.value
    assert len(ctx.fault_vector) == 1


@pytest.mark.asyncio
async def test_a_raised_fault_reaches_the_agent_as_an_exception(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay", faults=faults(FaultKind.RATE_LIMIT_429))
    with use_context(ctx), pytest.raises(RateLimited) as caught:
        await fetch_order("ORD-9921")

    assert caught.value.kind is FaultKind.RATE_LIMIT_429
    assert caught.value.tool_name == "fetch_order"
    trace = ctx.tool_calls[-1]
    assert trace.outcome == "fault"
    # Attributed to the fault, not overwritten with a generic error by the
    # wrapper's own exception handler.
    assert "RateLimited" in (trace.error or "")


def test_faults_work_on_a_sync_tool(store: CassetteStore) -> None:
    @sandbox_tool
    def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        fetch_order("ORD-9921")

    ctx = context(store, "replay", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx):
        assert fetch_order("ORD-9921") == {}
    assert len(ctx.fault_vector) == 1


@pytest.mark.asyncio
async def test_a_cassette_miss_beats_a_fault(store: CassetteStore) -> None:
    # The response is resolved before the fault is delivered, so an injected
    # 429 can never mask a hallucinated argument. The miss is a finding about
    # the model; the fault is a stimulus we chose.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay", faults=faults(FaultKind.RATE_LIMIT_429))
    with use_context(ctx), pytest.raises(CassetteMiss):
        await fetch_order("9921")
    assert len(ctx.fault_vector) == 0


@pytest.mark.asyncio
async def test_record_mode_never_injects(store: CassetteStore) -> None:
    # A faulted payload written to the store would poison every later replay
    # of that call, and the poison would be indistinguishable from a real
    # recording.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    ctx = context(store, "record", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx):
        assert await fetch_order("ORD-9921") == {"status": "shipped"}

    assert len(ctx.fault_vector) == 0
    # And it says so, or the run reads as "faults change nothing".
    assert any("no faults were injected" in w for w in ctx.warnings)

    # The cassette is clean, so a later replay is unaffected.
    with use_context(context(store, "replay")):
        assert await fetch_order("ORD-9921") == {"status": "shipped"}


@pytest.mark.asyncio
async def test_auto_mode_faults_what_the_agent_sees_not_what_is_stored(
    store: CassetteStore,
) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    ctx = context(store, "auto", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx):
        assert await fetch_order("ORD-9921") == {}

    assert len(ctx.fault_vector) == 1
    with use_context(context(store, "replay")):
        assert await fetch_order("ORD-9921") == {"status": "shipped"}


@pytest.mark.asyncio
async def test_live_mode_is_not_faulted(store: CassetteStore) -> None:
    # `live` means no interception at all, and a fault is interception.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    ctx = context(store, "live", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx):
        assert await fetch_order("ORD-9921") == {"status": "shipped"}
    assert len(ctx.fault_vector) == 0


@pytest.mark.asyncio
async def test_the_same_seed_reproduces_the_fault_vector(store: CassetteStore) -> None:
    # The Day 7 acceptance criterion, end to end through the decorator.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped", "lines": 2}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    schedule = faults(rate=1.0)
    digests = []
    for _ in range(2):
        ctx = context(store, "replay", faults=schedule)
        with use_context(ctx):
            for _ in range(6):
                with contextlib.suppress(Exception):
                    await fetch_order("ORD-9921")
        digests.append(ctx.fault_vector.digest)

    assert digests[0] == digests[1]
    assert len(digests[0]) == 16


@pytest.mark.asyncio
async def test_two_candidates_on_the_same_path_meet_the_same_faults(
    store: CassetteStore,
) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    schedule = faults(rate=1.0)
    vectors = []
    for candidate in ("baseline", "gemma3-4b"):
        ctx = RunContext(
            run_id="run-1",
            candidate_id=candidate,
            case_id="status_shipped",
            tool_mode="replay",
            cassettes=store,
            faults=schedule,
        )
        with use_context(ctx):
            for _ in range(5):
                with contextlib.suppress(Exception):
                    await fetch_order("ORD-9921")
        vectors.append(ctx.fault_vector)

    assert compare_vectors(*vectors).matched


@pytest.mark.asyncio
async def test_an_extra_call_to_one_tool_does_not_shift_another_tools_faults(
    store: CassetteStore,
) -> None:
    # The alignment bug, through the decorator. With a single global counter,
    # the candidate's extra `search_kb` call would move every later fault on
    # `fetch_order` and the two runs would stop being comparable - while the
    # table still printed them side by side.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    @sandbox_tool
    async def search_kb(query: str) -> list:
        return ["a"]

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")
        await search_kb("refund policy")

    schedule = faults(rate=1.0)

    async def run(*, with_extra_calls: bool) -> list[tuple[int, str]]:
        ctx = context(store, "replay", faults=schedule)
        with use_context(ctx):
            for _ in range(4):
                if with_extra_calls:
                    with contextlib.suppress(Exception):
                        await search_kb("refund policy")
                with contextlib.suppress(Exception):
                    await fetch_order("ORD-9921")
        return [
            (f.call_index, f.kind.value)
            for f in ctx.fault_vector
            if f.tool_name == "fetch_order"
        ]

    assert await run(with_extra_calls=False) == await run(with_extra_calls=True)


@pytest.mark.asyncio
async def test_an_inapplicable_fault_warns_and_still_counts(store: CassetteStore) -> None:
    @sandbox_tool
    async def count_orders(customer_id: str) -> int:
        return 7

    with use_context(context(store, "record")):
        await count_orders("CUST-1")

    ctx = context(store, "replay", faults=faults(FaultKind.PARTIAL_SCHEMA))
    with use_context(ctx):
        # Unchanged: there is no partial form of a bare integer.
        assert await count_orders("CUST-1") == 7

    assert len(ctx.fault_vector) == 1
    assert not next(iter(ctx.fault_vector)).applied
    assert any("could not be shaped" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_a_fault_records_the_node_it_landed_in(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay", faults=faults(FaultKind.EMPTY_RESULT))
    with use_context(ctx), node("order_lookup"):
        await fetch_order("ORD-9921")

    assert next(iter(ctx.fault_vector)).node == "order_lookup"


@pytest.mark.asyncio
async def test_a_returned_delivery_turns_a_429_into_a_payload(store: CassetteStore) -> None:
    # Testing whether the model *reads* its tool output, rather than whether
    # the retry wrapper works.
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(
        store,
        "replay",
        faults=faults(
            FaultKind.RATE_LIMIT_429,
            delivery={FaultKind.RATE_LIMIT_429: Delivery.RETURN},
        ),
    )
    with use_context(ctx):
        got = await fetch_order("ORD-9921")
    assert got["status"] == 429


@pytest.mark.asyncio
async def test_per_tool_call_indexes_advance_independently(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    @sandbox_tool
    async def search_kb(query: str) -> list:
        return []

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")
        await search_kb("q")

    ctx = context(store, "replay")
    with use_context(ctx):
        await fetch_order("ORD-9921")
        await search_kb("q")
        await fetch_order("ORD-9921")

    assert ctx.tool_counts == {"fetch_order": 2, "search_kb": 1}


@pytest.mark.asyncio
async def test_an_uncovered_miss_is_flagged_as_coverage_not_failure(
    store: CassetteStore,
) -> None:
    # The scorecard must be able to tell "the model did something wrong" from
    # "the model did something we have no ground truth for".
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay")
    with use_context(ctx), pytest.raises(CassetteMiss) as caught:
        await fetch_order("ORD-7733")

    assert caught.value.uncovered
    assert ctx.tool_calls[-1].miss_verdict == "unrecorded_valid_path"
    assert any("fill-gaps" in w for w in ctx.warnings)


@pytest.mark.asyncio
async def test_a_mangled_argument_is_not_flagged_as_coverage(store: CassetteStore) -> None:
    @sandbox_tool
    async def fetch_order(order_id: str) -> dict:
        return {"status": "shipped"}

    with use_context(context(store, "record")):
        await fetch_order("ORD-9921")

    ctx = context(store, "replay")
    with use_context(ctx), pytest.raises(CassetteMiss) as caught:
        await fetch_order("9921")

    assert not caught.value.uncovered
    assert ctx.tool_calls[-1].miss_verdict == "hallucinated_parameter"
    assert not any("fill-gaps" in w for w in ctx.warnings)
