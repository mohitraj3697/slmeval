"""Loading and running the user's agent.

Half of these are about error messages, and that is the right proportion. The
adapter's failure mode is "your agent did not load", which is the first wall
every new user hits, and the difference between a message that names the
directory it searched and one that says `ModuleNotFoundError` is the difference
between a two-minute fix and a bug report.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import BaseModel

from slmeval.core.adapter import Target, as_state, execute_case, load_target
from slmeval.core.context import RunContext, get_context
from slmeval.errors import ConfigError

AGENT = '''
import slmeval

async def run_case(case_input: dict) -> dict:
    with slmeval.node("intent_router"):
        pass
    return {"intent": "order_status", "echo": case_input.get("user_query", "")}

def sync_run_case(case_input: dict) -> dict:
    return {"intent": "sync", "context_seen": slmeval.core.context.get_context() is not None}

def returns_a_string(case_input: dict) -> str:
    return "nope"

NOT_CALLABLE = 42
'''


@pytest.fixture
def project(tmp_path: Path):
    (tmp_path / "my_agent.py").write_text(AGENT, encoding="utf-8")
    yield tmp_path
    sys.modules.pop("my_agent", None)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def test_an_entrypoint_loads_relative_to_the_project(project: Path) -> None:
    # `entrypoint = "my_agent:run_case"` means "the file next to eval.toml".
    # A user running slmeval from a parent directory should not have to think
    # about PYTHONPATH.
    target = Target.load("my_agent:run_case", root=project)
    assert target.is_async is True


def test_loading_does_not_leave_the_project_on_sys_path(project: Path) -> None:
    # Leaving it there lets a user's `utils.py` shadow a real package for the
    # rest of the process, which is maddening to debug.
    before = list(sys.path)
    Target.load("my_agent:run_case", root=project)
    assert sys.path == before


def test_a_missing_module_names_where_it_looked(project: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        Target.load("not_there:run_case", root=project)
    assert "not_there" in str(caught.value)
    assert str(project.resolve()) in (caught.value.hint or "")


def test_a_missing_callable_suggests_the_right_one(project: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        Target.load("my_agent:run_cases", root=project)
    assert "run_case" in (caught.value.hint or "")


def test_a_non_callable_entrypoint_is_refused(project: Path) -> None:
    with pytest.raises(ConfigError, match="not a callable"):
        Target.load("my_agent:NOT_CALLABLE", root=project)


def test_a_malformed_entrypoint_is_refused() -> None:
    with pytest.raises(ConfigError, match=r"module\.path:callable_name"):
        Target.load("my_agent.run_case")


def test_a_module_that_raises_on_import_is_not_reported_as_missing(tmp_path: Path) -> None:
    # "module not found" would send the user looking in the wrong place: the
    # module was found, it is their own code that blew up.
    (tmp_path / "broken.py").write_text("raise ValueError('boom')\n", encoding="utf-8")
    try:
        with pytest.raises(ConfigError) as caught:
            Target.load("broken:run_case", root=tmp_path)
        assert "ValueError" in str(caught.value)
        assert "failed to import" in (caught.value.hint or "")
    finally:
        sys.modules.pop("broken", None)


def test_the_langgraph_adapter_says_what_to_do_instead(project: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_target("my_agent:run_case", adapter="langgraph", root=project)
    assert "not implemented yet" in str(caught.value)
    assert "generic" in (caught.value.hint or "")


def test_an_unknown_adapter_is_refused(project: Path) -> None:
    with pytest.raises(ConfigError, match="unknown target adapter"):
        load_target("my_agent:run_case", adapter="crewai", root=project)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_an_async_agent_runs_and_returns_its_state(project: Path) -> None:
    target = Target.load("my_agent:run_case", root=project)
    state = await target({"user_query": "where is ORD-1"})
    assert state == {"intent": "order_status", "echo": "where is ORD-1"}


@pytest.mark.asyncio
async def test_a_sync_agent_runs_in_a_thread_with_the_context_intact(project: Path) -> None:
    # to_thread, not a direct call: contextvars propagate through it, so a
    # synchronous agent's sandboxed tools still see the run - and one blocking
    # agent does not stall the other 89 executions.
    target = Target.load("my_agent:sync_run_case", root=project)
    ctx = RunContext(run_id="r", candidate_id="c", case_id="case_1")
    state = await execute_case(ctx, target, {"user_query": "hi"})
    assert state == {"intent": "sync", "context_seen": True}


@pytest.mark.asyncio
async def test_execute_case_installs_the_context_inside_the_task(project: Path) -> None:
    # The context layer's rule 2: a ContextVar set before `create_task` is
    # copied into the child, but one set after is invisible to it. So the task
    # installs its own.
    seen: list[str | None] = []

    async def agent(case_input: dict) -> dict:
        ctx = get_context()
        seen.append(ctx.case_id if ctx else None)
        return {}

    target = Target(entrypoint="x:y", func=agent, is_async=True)
    ctx = RunContext(run_id="r", candidate_id="c", case_id="case_1")
    await execute_case(ctx, target, {})
    assert seen == ["case_1"]


@pytest.mark.asyncio
async def test_sibling_executions_do_not_see_each_other(project: Path) -> None:
    import asyncio

    async def agent(case_input: dict) -> dict:
        ctx = get_context()
        await asyncio.sleep(0)
        assert ctx is not None
        return {"case": ctx.case_id}

    target = Target(entrypoint="x:y", func=agent, is_async=True)
    results = await asyncio.gather(
        *(
            execute_case(
                RunContext(run_id="r", candidate_id="c", case_id=f"case_{i}"), target, {}
            )
            for i in range(8)
        )
    )
    assert [r["case"] for r in results] == [f"case_{i}" for i in range(8)]


# --------------------------------------------------------------------------- #
# Final state
# --------------------------------------------------------------------------- #


class State(BaseModel):
    intent: str


@dataclass
class DataclassState:
    intent: str


@pytest.mark.parametrize(
    "result",
    [{"intent": "x"}, State(intent="x"), DataclassState(intent="x")],
)
def test_a_typed_state_object_is_accepted(result: object) -> None:
    # An agent that already has a typed state should not have to unwrap it just
    # to be measurable.
    assert as_state(result, entrypoint="m:f") == {"intent": "x"}


def test_a_non_state_return_is_refused_by_name() -> None:
    # The dataset asserts on state keys, so a bare string would silently fail
    # every assertion and look like a model failure.
    with pytest.raises(ConfigError) as caught:
        as_state("just a string", entrypoint="my_agent:run_case")
    assert "returned str" in str(caught.value)
    assert "asserts on its keys" in (caught.value.hint or "")


@pytest.mark.asyncio
async def test_an_agent_returning_the_wrong_shape_is_refused(project: Path) -> None:
    target = Target.load("my_agent:returns_a_string", root=project)
    with pytest.raises(ConfigError):
        await target({"user_query": "hi"})
