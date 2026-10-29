"""The context layer's one job is isolation under concurrency.

If these tests pass, candidate A's traces cannot end up in candidate B's
scorecard. If they fail, every number the tool prints is suspect, so they are
deliberately paranoid about ordering and nesting.
"""

from __future__ import annotations

import asyncio

import pytest

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
    reset_warnings,
    set_context,
    use_context,
    warn_missing_context,
)
from slmeval.core.trace import LLMCallTrace


def make_ctx(candidate: str, case: str = "c1") -> RunContext:
    return RunContext(run_id="r1", candidate_id=candidate, case_id=case)


def test_no_context_by_default() -> None:
    assert get_context() is None
    assert current_node() is None
    assert current_node_path() == ()


def test_node_is_a_noop_without_a_context() -> None:
    """The integration has to be safe to leave in production code."""
    with node("intent_router"):
        assert current_node() == "intent_router"
    assert current_node() is None


def test_nodes_nest_and_unwind() -> None:
    with node("router"):
        assert current_node_path() == ("router",)
        with node("classify"):
            assert current_node_path() == ("router", "classify")
            assert current_node() == "classify"
        # The reset(token) form restores the parent exactly; a set(old_value)
        # would leave the path corrupted from here on.
        assert current_node_path() == ("router",)
    assert current_node_path() == ()


def test_node_survives_an_exception() -> None:
    with pytest.raises(RuntimeError), node("router"):
        raise RuntimeError("boom")
    assert current_node_path() == ()


def test_node_rejects_an_empty_name() -> None:
    with pytest.raises(ValueError, match="non-empty"), node("   "):
        pass


def test_node_records_the_trajectory() -> None:
    ctx = make_ctx("a")
    with use_context(ctx):
        with node("router"), node("classify"):
            pass
        with node("lookup"):
            pass
    assert ctx.node_names == ["router", "classify", "lookup"]
    assert ctx.trajectory[1].path == ("router", "classify")


def test_use_context_restores_the_previous_one() -> None:
    outer, inner = make_ctx("outer"), make_ctx("inner")
    with use_context(outer):
        with use_context(inner):
            assert get_context() is inner
        assert get_context() is outer
    assert get_context() is None


def test_no_context_detaches() -> None:
    ctx = make_ctx("a")
    with use_context(ctx):
        with no_context():
            assert get_context() is None
        assert get_context() is ctx


def test_require_context_names_the_thread_problem() -> None:
    with pytest.raises(LookupError, match="ThreadPoolExecutor"):
        require_context("cassette replay")


def test_warn_missing_context_warns_once() -> None:
    reset_warnings()
    with pytest.warns(SlmevalContextWarning, match="NOT intercepted"):
        warn_missing_context("sandbox_tool('fetch_order')")
    # A ten-step agent would otherwise emit hundreds of identical warnings and
    # the user would stop reading them.
    import warnings as w

    with w.catch_warnings(record=True) as seen:
        w.simplefilter("always")
        warn_missing_context("sandbox_tool('fetch_order')")
    assert seen == []


def test_step_budget_is_enforced() -> None:
    ctx = make_ctx("a")
    ctx.max_steps = 3
    for _ in range(3):
        ctx.note_step()
    with pytest.raises(StepBudgetExceeded) as excinfo:
        ctx.note_step()
    assert excinfo.value.limit == 3
    assert excinfo.value.used == 4


def test_warnings_are_deduplicated_per_context() -> None:
    ctx = make_ctx("a")
    ctx.warn("json_mode fallback")
    ctx.warn("json_mode fallback")
    assert ctx.warnings == ["json_mode fallback"]


def test_scope_is_case_level_not_attempt_level() -> None:
    """Every repeat of a case must replay the same recorded tool responses."""
    first = RunContext(run_id="r", candidate_id="a", case_id="c7", attempt=0)
    second = RunContext(run_id="r", candidate_id="a", case_id="c7", attempt=2)
    assert first.scope == second.scope == "c7"


# --------------------------------------------------------------------------- #
# Concurrency: the whole reason this layer exists
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_eight_concurrent_tasks_do_not_contaminate_each_other() -> None:
    async def execution(candidate: str, calls: int) -> RunContext:
        ctx = make_ctx(candidate, case=f"case-{candidate}")
        # Set INSIDE the task. A ContextVar set in the parent before
        # create_task is copied in; set after, it is invisible - which is
        # exactly how candidate A's traces end up in candidate B's buffer.
        set_context(ctx)
        for i in range(calls):
            with node(f"{candidate}-node-{i}"):
                await asyncio.sleep(0)
                live = get_context()
                assert live is ctx
                assert current_node() == f"{candidate}-node-{i}"
                live.record_llm(
                    LLMCallTrace(
                        seq=live.next_llm_seq(),
                        model_id=candidate,
                        provider="fake",
                        model="m",
                        node=current_node(),
                        node_path=current_node_path(),
                    )
                )
                await asyncio.sleep(0)
        return ctx

    async with asyncio.TaskGroup() as tg:
        tasks = [
            tg.create_task(execution(f"cand{i}", calls=i + 1)) for i in range(8)
        ]

    for i, task in enumerate(tasks):
        ctx = task.result()
        assert len(ctx.llm_calls) == i + 1
        assert {t.model_id for t in ctx.llm_calls} == {f"cand{i}"}
        assert [t.node for t in ctx.llm_calls] == [f"cand{i}-node-{j}" for j in range(i + 1)]
        assert ctx.node_names == [f"cand{i}-node-{j}" for j in range(i + 1)]

    # And nothing leaked back out to the parent task.
    assert get_context() is None
    assert current_node_path() == ()


@pytest.mark.asyncio
async def test_nested_node_attribution_is_correct_under_concurrency() -> None:
    async def execution(name: str) -> RunContext:
        ctx = make_ctx(name)
        set_context(ctx)
        with node("outer"):
            await asyncio.sleep(0)
            with node("inner"):
                await asyncio.sleep(0)
                assert current_node_path() == ("outer", "inner")
            assert current_node_path() == ("outer",)
        return ctx

    results = await asyncio.gather(*(execution(f"c{i}") for i in range(6)))
    for ctx in results:
        assert [v.path for v in ctx.trajectory] == [("outer",), ("outer", "inner")]


@pytest.mark.asyncio
async def test_context_propagates_through_to_thread() -> None:
    """`asyncio.to_thread` copies the context; a bare Thread does not.

    This asymmetry is the documented escape hatch, so it is pinned here - if a
    Python release changed it, the sandbox's safety story would change with it.
    """
    ctx = make_ctx("a")
    set_context(ctx)

    def in_thread() -> tuple[object, str | None]:
        return get_context(), current_node()

    with node("tool_step"):
        seen_ctx, seen_node = await asyncio.to_thread(in_thread)
    assert seen_ctx is ctx
    assert seen_node == "tool_step"


@pytest.mark.asyncio
async def test_context_is_lost_across_a_bare_thread() -> None:
    import threading

    ctx = make_ctx("a")
    set_context(ctx)
    seen: list[object] = []

    thread = threading.Thread(target=lambda: seen.append(get_context()))
    thread.start()
    thread.join()
    assert seen == [None]
