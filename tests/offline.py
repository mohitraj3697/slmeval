"""A scripted project, for the suites that need a whole run rather than a unit.

Step 15's constraint is that the test suite must run offline, with no GPU, no
keys and no network, and still exercise the real path: config -> dataset ->
runner -> sandbox -> evaluators -> storage -> report. The `fake` provider is
what makes that possible, and this module is the project it runs in.

One agent, four nodes, two sandboxed tools, and candidates declared as TOML
blocks - so a test says "this candidate is a model that cannot emit JSON" in
one line and asserts what the harness concluded. Opening it goes through
:func:`slmeval.cli.loading.open_project`, the same function the commands use,
so these suites cannot pass against a wiring the CLI does not have.

:mod:`tests.test_runner` keeps its own smaller builder for planning and resume
mechanics; this one exists because classification, determinism and the rendered
tables all need the same richer agent, and three copies of it would drift.
"""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from slmeval.cli.loading import Project, open_project
from slmeval.core.runner import MatrixRunner, RunOptions
from slmeval.eval.pipeline import case_rollups, evaluator_for
from slmeval.storage.models import ExecutionRecord

MODULE = "offline_agent"

AGENT = '''
"""A support agent: route, look the order up, compose a reply.

Deliberately ordinary. The interesting behaviour belongs to the scripted
models, not to the agent - an agent with cleverness of its own would make it
impossible to say which of the two a finding is about.
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from slmeval import chat_model, node, sandbox_tool


class Routing(BaseModel):
    intent: str
    order_id: str


class Reply(BaseModel):
    final_answer: str


@sandbox_tool(match_on=["order_id"])
async def fetch_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped", "carrier": "DHL"}


@sandbox_tool
async def issue_refund(order_id: str) -> dict:
    """The one tool that moves money. Forbidden by every case in the dataset."""
    return {"refunded": order_id}


async def run_case(case_input: dict) -> dict:
    query = str(case_input.get("user_query", ""))

    with node("intent_router"):
        routing = (
            await chat_model().complete([{"role": "user", "content": query}], schema=Routing)
        ).parsed_as(Routing)

    state: dict = {"intent": routing.intent, "order_id": routing.order_id}

    # Each branch is reachable only by a scripted model asking for it, which is
    # how one agent covers every outcome the classifier can produce.
    if routing.intent == "boom":
        raise RuntimeError("the agent itself fell over")
    if routing.intent == "stall":
        await asyncio.sleep(5)
    while routing.intent == "spin":
        with node("spin"):
            await chat_model().complete([{"role": "user", "content": "again"}], schema=Routing)

    if routing.order_id:
        with node("order_lookup"):
            state["order"] = await fetch_order(routing.order_id)
    if routing.intent == "refund":
        with node("refund"):
            state["refund"] = await issue_refund(routing.order_id or "unknown")

    with node("compose_reply"):
        reply = (
            await chat_model().complete([{"role": "user", "content": "reply"}], schema=Reply)
        ).parsed_as(Reply)
    state["final_answer"] = reply.final_answer
    return state
'''

CASES = (
    '{"case_id": "status_shipped", "input": {"user_query": "Where is my order ORD-9921?"}, '
    '"assert": {"state_contains": {"intent": "order_status", "order_id": "ORD-9921"}, '
    '"required_tools": ["fetch_order"], "forbidden_tools": ["issue_refund"], '
    '"max_tool_calls": 3, '
    '"reference_trajectory": [["intent_router"], ["order_lookup"], ["compose_reply"]]}}\n'
    '{"case_id": "status_other", "input": {"user_query": "any update on ORD-7310?"}, '
    '"assert": {"state_contains": {"intent": "order_status", "order_id": "ORD-7310"}, '
    '"required_tools": ["fetch_order"], "forbidden_tools": ["issue_refund"], '
    '"max_tool_calls": 3}}\n'
)

#: Just the first case, for suites where every candidate has one script: a
#: cassette's scope is the case, so a model that answers both cases the same way
#: would be reported as taking a route we never recorded - true, and not what
#: those tests are about.
ONE_CASE = CASES.splitlines(keepends=True)[0]


#: What a model that works says, per case. A benchmark is many cases and one
#: model, so the reference has to answer each of them - a single script would
#: have the scripted baseline fail most of its own dataset.
BASELINE = """\
[baseline]
provider = "fake"
model = "scripted-baseline"
extra_body.script = [
    { text = '{"intent": "order_status", "order_id": "ORD-9921"}' },
    { text = '{"final_answer": "Your order ORD-9921 is with DHL."}' },
]

[baseline.extra_body.script_by_case]
status_other = [
    { text = '{"intent": "order_status", "order_id": "ORD-7310"}' },
    { text = '{"final_answer": "Your order ORD-7310 is with DHL."}' },
]
"""


def candidate(model_id: str, *script: str, extra: str = "") -> str:
    """One candidate, with a turn per entry of ``script``.

    Entries are TOML inline tables, so a test reads as the list of things the
    model does: ``candidate("weak", behaves("not_json"))``.
    """
    turns = ",\n    ".join(script)
    return f"""
[[candidates]]
id = "{model_id}"
provider = "fake"
model = "{model_id}"
extra_body.script = [
    {turns},
]
{extra}
"""


def says(payload: str) -> str:
    """A turn that emits exactly this payload."""
    return f"{{ text = '{payload}' }}"


def routes(intent: str, order_id: str = "ORD-9921") -> str:
    return says(f'{{"intent": "{intent}", "order_id": "{order_id}"}}')


def replies(answer: str = "here you go") -> str:
    return says(f'{{"final_answer": "{answer}"}}')


def behaves(kind: str, payload: str | None = None) -> str:
    """A turn built from a named misbehaviour, as a config would write it."""
    if payload is None:
        return f'{{ kind = "{kind}" }}'
    return f"{{ kind = \"{kind}\", text = '{payload}' }}"


def write_project(
    root: Path,
    *,
    candidates: str = "",
    repeats: int = 1,
    seed: int = 42,
    fault_rate: float = 0.0,
    timeout_s: float = 10.0,
    max_steps: int = 8,
    concurrency: int = 2,
    cases: str = CASES,
    extra_benchmark: str = "",
) -> Path:
    """Write agent, dataset and config into ``root``; return the config path."""
    (root / f"{MODULE}.py").write_text(AGENT, encoding="utf-8")
    (root / "cases.jsonl").write_text(cases, encoding="utf-8")
    config = root / "eval.toml"
    config.write_text(
        f"""
[benchmark]
name = "offline"
dataset = "./cases.jsonl"
runs_per_case = {repeats}
tool_mode = "replay"
fault_injection_rate = {fault_rate}
seed = {seed}
concurrency = {concurrency}
timeout_s = {timeout_s}
max_steps = {max_steps}
{extra_benchmark}

[target]
entrypoint = "{MODULE}:run_case"

{BASELINE}
{candidates}
""",
        encoding="utf-8",
    )
    # Each test writes its own agent under one module name; a stale import
    # would have the second test in a file silently benchmark the first one's.
    sys.modules.pop(MODULE, None)
    return config


@dataclass
class Runs:
    """An opened project plus the one call that runs its matrix."""

    project: Project

    @property
    def root(self) -> Path:
        return self.project.loaded.root

    @property
    def config(self) -> Any:
        return self.project.config

    @property
    def store(self) -> Any:
        return self.project.store

    def run(self, **options: Any) -> Any:
        """Run the matrix and save the rollups, exactly as `run` does."""
        runner = MatrixRunner(
            config=self.config,
            dataset=self.project.dataset,
            target=self.project.target,
            store=self.store,
            options=RunOptions.from_config(self.config, **options),
            cassettes=self.project.cassettes,
            evaluator=evaluator_for(self.config),
            prices=self.project.prices,
            root=self.root,
        )
        summary = asyncio.run(runner.run())
        self.store.save_rollups(
            case_rollups(summary.run_id, self.store.executions(summary.run_id))
        )
        return summary

    def record(self) -> Any:
        """Populate the cassettes from the baseline, as `slmeval record` does.

        The baseline only: the cassettes are the fixed environment every
        candidate is measured against, so a candidate allowed to record its own
        tool calls would be grading its own homework.
        """
        return self.run(
            tool_mode="record",
            candidate_ids=(self.config.baseline.id,),
            # Never while recording: a faulted payload written to the store
            # would poison every later replay of that call.
            fault_rate=0.0,
        )

    def executions(self, run_id: str, candidate_id: str) -> list[ExecutionRecord]:
        return [r for r in self.store.executions(run_id) if r.candidate_id == candidate_id]

    def outcomes(self, run_id: str) -> dict[str, set[str]]:
        """Candidate id -> the set of outcomes its executions ended in."""
        out: dict[str, set[str]] = {}
        for row in self.store.executions(run_id):
            out.setdefault(row.candidate_id, set()).add(row.outcome_class.value)
        return out


@contextlib.contextmanager
def offline_project(root: Path, **options: Any) -> Iterator[Runs]:
    """Write a project into ``root`` and open it the way the CLI does."""
    config = write_project(root, **options)
    with open_project(config) as project:
        yield Runs(project)


def build_and_run(
    root: Path, *, record: bool = True, tool_mode: str = "replay", **options: Any
) -> Path:
    """Leave a finished run on disk, then rewrite the project around it.

    Rewritten afterwards so the agent module is importable again - which also
    proves the point for the suites that use this: ``show``, ``report`` and
    ``gate`` read the stored rows and never load the agent, so the project they
    are pointed at is only there for its config and its store.
    """
    with offline_project(root, **options) as runs:
        if record:
            runs.record()
        runs.run(tool_mode=tool_mode)
    write_project(root, **options)
    return root


@contextlib.contextmanager
def open_built(root: Path) -> Iterator[Runs]:
    """Open a project that already holds a finished run, without touching it.

    The counterpart to :func:`build_and_run`: the store, the cassettes and the
    config are read as they are, so a test can inspect a run it did not have to
    execute.
    """
    with open_project(root / "eval.toml") as project:
        yield Runs(project)


def clone_project(template: Path, destination: Path) -> Path:
    """Copy a finished project so a test can mutate it without rebuilding it.

    Running a matrix costs about a quarter of a second and copying the finished
    store costs almost nothing, so a suite that needs one stored run per test
    builds it once and clones it. The whole suite has to stay under thirty
    seconds; past that people stop running it, and a test suite nobody runs
    protects nothing.
    """
    shutil.copytree(template, destination, dirs_exist_ok=True)
    # The clone carries its own copy of the agent module file, and a stale
    # import would have this test benchmark the template's.
    sys.modules.pop(MODULE, None)
    return destination


@pytest.fixture(autouse=True)
def forget_the_agent_module() -> Iterator[None]:
    """Drop the temp agent from ``sys.modules`` after every test."""
    yield
    sys.modules.pop(MODULE, None)
