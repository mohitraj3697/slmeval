"""The matrix runner - Day 9 and Day 10's acceptance criteria.

Three of these tests are the criteria verbatim:

- a 3-candidate x 10-case x 3-repeat matrix (90 executions) completes with
  mixed outcomes;
- an interrupt at roughly half way, then ``--resume``, finishes cleanly and
  without double-counting anything;
- a case engineered to raise inside the harness is reported as
  ``harness_error`` and the other rows are unaffected.

Everything runs against the scripted ``fake`` provider, so the whole file is
offline and takes about a second. The agent under test is written to a temp
directory per test, which is also what exercises ``[target].sys_path`` and the
import path.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from slmeval.config.loader import load_config
from slmeval.core.adapter import load_target
from slmeval.core.outcome import Classification
from slmeval.core.runner import (
    LATENCY,
    THROUGHPUT,
    Endpoint,
    MatrixRunner,
    ProviderLimits,
    RunOptions,
    SpendBudget,
    WorkUnit,
    encode_plan,
    plan_matrix,
)
from slmeval.core.verdict import Verdict
from slmeval.dataset.loader import load_dataset
from slmeval.errors import ConfigError
from slmeval.eval.cost import Priced, PriceTable
from slmeval.eval.pipeline import case_rollups, evaluator_for
from slmeval.storage.db import RunStore
from slmeval.storage.models import ExecStatus, Outcome, RunStatus

MODULE = "tiny_agent"

AGENT = '''
"""A two-node agent, small enough to reason about and real enough to measure."""

from __future__ import annotations

from pydantic import BaseModel

from slmeval import chat_model, node


class Routing(BaseModel):
    intent: str


async def run_case(case_input: dict) -> dict:
    with node("router"):
        completion = await chat_model().complete(
            [{"role": "user", "content": str(case_input.get("q", ""))}],
            schema=Routing,
        )
    routing = completion.parsed_as(Routing)
    with node("reply"):
        answer = f"intent is {routing.intent}"
    return {"intent": routing.intent, "final_answer": answer}
'''


def script(intent: str) -> str:
    return (
        "extra_body.script = [{ text = '"
        + f'{{"intent": "{intent}"}}'
        + "' }]"
    )


@dataclass
class Fixture:
    """A loaded temp project, with everything a runner needs."""

    root: Path
    store: RunStore
    config: Any
    dataset: Any
    target: Any

    def runner(self, **options: Any) -> MatrixRunner:
        return MatrixRunner(
            config=self.config,
            dataset=self.dataset,
            target=self.target,
            store=self.store,
            options=RunOptions.from_config(self.config, **options),
            evaluator=evaluator_for(self.config),
            prices=PriceTable(),
            root=self.root,
        )

    def go(self, *, resume: str | None = None, **options: Any) -> Any:
        runner = self.runner(**options)
        summary = asyncio.run(runner.run(resume=resume))
        self.store.save_rollups(
            case_rollups(summary.run_id, self.store.executions(summary.run_id))
        )
        return summary


def build(
    tmp_path: Path,
    *,
    cases: int = 2,
    repeats: int = 1,
    candidates: tuple[tuple[str, str], ...] = (),
    concurrency: int = 4,
) -> Fixture:
    (tmp_path / f"{MODULE}.py").write_text(AGENT, encoding="utf-8")

    lines = []
    for index in range(cases):
        lines.append(
            f'{{"case_id": "c{index}", "input": {{"q": "q{index}"}}, '
            '"assert": {"state_contains": {"intent": "ok"}, '
            '"max_tool_calls": 0}}'
        )
    (tmp_path / "cases.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    blocks = [
        f"""
[benchmark]
name = "tiny"
dataset = "./cases.jsonl"
runs_per_case = {repeats}
tool_mode = "replay"
seed = 42
concurrency = {concurrency}
timeout_s = 10.0
max_steps = 8

[target]
entrypoint = "{MODULE}:run_case"

[baseline]
provider = "fake"
model = "good"
{script("ok")}
"""
    ]
    for candidate_id, intent in candidates:
        blocks.append(
            f"""
[[candidates]]
id = "{candidate_id}"
provider = "fake"
model = "{candidate_id}"
{script(intent)}
"""
        )
    (tmp_path / "eval.toml").write_text("".join(blocks), encoding="utf-8")

    loaded = load_config(tmp_path / "eval.toml")
    return Fixture(
        root=tmp_path,
        store=RunStore.for_project(tmp_path),
        config=loaded.config,
        dataset=load_dataset(loaded.dataset_path),
        target=load_target(loaded.config.target.entrypoint, root=tmp_path),
    )


@pytest.fixture(autouse=True)
def forget_the_agent_module() -> Iterator[None]:
    """Drop the temp agent from ``sys.modules`` between tests.

    Each test writes its own ``tiny_agent.py`` into its own temp directory
    under the same module name, so without this the second test would silently
    benchmark the first test's agent.
    """
    yield
    sys.modules.pop(MODULE, None)


@pytest.fixture
def project(tmp_path: Path) -> Iterator[Fixture]:
    fixture = build(tmp_path, cases=2, repeats=1, candidates=(("weak", "bad"),))
    try:
        yield fixture
    finally:
        fixture.store.close()


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #


def test_the_matrix_is_the_cross_product() -> None:
    matrix = plan_matrix(
        candidate_ids=["base", "a", "b"],
        case_ids=["c1", "c2"],
        repeats=3,
        baseline_id="base",
    )
    assert len(matrix) == 3 * 2 * 3
    assert matrix.planned == 18
    assert len(matrix.for_candidate("a")) == 6


def test_the_baseline_runs_first_and_whole() -> None:
    """It populates the cassettes every candidate then replays, so a candidate
    starting first would miss on a path that is covered a second later."""
    matrix = plan_matrix(
        candidate_ids=["base", "a"], case_ids=["c1", "c2"], repeats=2, baseline_id="base"
    )
    first, rest = matrix.phases()
    assert {u.candidate_id for u in first} == {"base"}
    assert {u.candidate_id for u in rest} == {"a"}
    assert len(first) == 4


def test_candidates_are_interleaved_case_by_case() -> None:
    """A matrix abandoned at 30% should have partial results for every
    candidate: three rows with n=3 is a comparison, one complete row and two
    empty ones is not."""
    matrix = plan_matrix(
        candidate_ids=["base", "a", "b"], case_ids=["c1", "c2"], repeats=1, baseline_id="base"
    )
    _, rest = matrix.phases()
    assert [(u.candidate_id, u.case_id) for u in rest] == [
        ("a", "c1"),
        ("b", "c1"),
        ("a", "c2"),
        ("b", "c2"),
    ]


def test_completed_units_are_subtracted() -> None:
    matrix = plan_matrix(
        candidate_ids=["base"],
        case_ids=["c1", "c2"],
        repeats=2,
        baseline_id="base",
        completed=[("base", "c1", 0), ("base", "c1", 1)],
    )
    assert len(matrix.units) == 2
    assert len(matrix.resumed) == 2
    # The denominator still counts them, or a resumed run reports 2/2 and looks
    # like it did the whole matrix.
    assert matrix.planned == 4


def test_a_baseline_outside_the_candidate_list_is_refused() -> None:
    with pytest.raises(ValueError, match="not among the candidates"):
        plan_matrix(candidate_ids=["a"], case_ids=["c"], repeats=1, baseline_id="base")


def test_the_plan_can_be_inspected_without_running(project: Fixture) -> None:
    import json

    payload = json.loads(encode_plan(project.runner().plan()))
    assert payload["candidates"] == ["baseline", "weak"]
    assert payload["cases"] == ["c0", "c1"]
    assert payload["to_run"] == 4
    assert payload["scheduling"] == THROUGHPUT
    assert project.store.get_run(payload["run_id"]) is None, "planning writes nothing"


def test_selecting_a_candidate_keeps_the_baseline(project: Fixture) -> None:
    """Every delta is measured against it, so a --candidate run without it
    would produce a table with nothing to compare to."""
    plan = project.runner(candidate_ids=("weak",)).plan()
    assert plan.matrix.candidate_ids == ("baseline", "weak")


def test_an_unknown_case_is_a_usage_error(project: Fixture) -> None:
    with pytest.raises(ConfigError, match="no such case"):
        project.runner(case_ids=("nope",)).plan()


def test_an_unknown_candidate_is_a_usage_error(project: Fixture) -> None:
    with pytest.raises(ConfigError, match="no such candidate"):
        project.runner(candidate_ids=("nope",)).plan()


# --------------------------------------------------------------------------- #
# Per-endpoint limits
# --------------------------------------------------------------------------- #


def test_limits_are_per_endpoint_not_per_candidate() -> None:
    """Two candidates on one Ollama share a GPU. A limit each would put twice
    the load on it as either asked for."""
    from slmeval.config.models import ModelCfg

    models = [
        ModelCfg(id="a", provider="ollama", model="x"),
        ModelCfg(id="b", provider="ollama", model="y"),
    ]
    limits = ProviderLimits.for_run(models, RunOptions(concurrency=8))
    assert len(limits.limits) == 1
    assert limits.key_for(models[0]) == limits.key_for(models[1])


def test_a_local_and_a_cloud_endpoint_do_not_throttle_each_other() -> None:
    from slmeval.config.models import ModelCfg

    models = [
        ModelCfg(id="local", provider="ollama", model="x"),
        ModelCfg(id="cloud", provider="fake", model="y"),
    ]
    limits = ProviderLimits.for_run(models, RunOptions(concurrency=8))
    assert len(limits.limits) == 2


def test_throughput_mode_raises_the_local_limit_above_one() -> None:
    """The provider layer defaults a local model to 1, which is right for
    measuring latency and would make a 90-execution replay matrix serial."""
    from slmeval.config.models import ModelCfg

    local = ModelCfg(id="a", provider="ollama", model="x")
    throughput = ProviderLimits.for_run([local], RunOptions(concurrency=8))
    assert throughput.limit_for(local) == 2


def test_latency_mode_pins_every_endpoint_to_one() -> None:
    """A p95 taken from four concurrent requests to one GPU describes our
    scheduler, not the model."""
    from slmeval.config.models import ModelCfg

    models = [
        ModelCfg(id="a", provider="ollama", model="x"),
        ModelCfg(id="b", provider="fake", model="y"),
    ]
    limits = ProviderLimits.for_run(
        models, RunOptions(concurrency=16, serial_latency=True)
    )
    assert set(limits.limits.values()) == {1}


def test_an_endpoint_limit_never_exceeds_the_global_ceiling() -> None:
    from slmeval.config.models import ModelCfg

    cloud = ModelCfg(id="a", provider="fake", model="y")
    limits = ProviderLimits.for_run([cloud], RunOptions(concurrency=2))
    assert limits.limit_for(cloud) == 2


def test_the_scheduling_mode_decides_whether_latency_is_reportable() -> None:
    assert RunOptions().scheduling == THROUGHPUT
    assert not RunOptions().latency_reportable
    assert RunOptions(serial_latency=True).scheduling == LATENCY
    assert RunOptions(serial_latency=True).latency_reportable


@pytest.mark.asyncio
async def test_a_slot_is_bounded_by_its_limit() -> None:
    limits = ProviderLimits({"k": 2}, {"k": Endpoint("fake", "", False)})
    live = 0
    peak = 0

    async def hold() -> None:
        nonlocal live, peak
        async with limits.slot("k"):
            live += 1
            peak = max(peak, live)
            await asyncio.sleep(0.01)
            live -= 1

    await asyncio.gather(*[hold() for _ in range(6)])
    assert peak == 2


# --------------------------------------------------------------------------- #
# The spend budget
# --------------------------------------------------------------------------- #


def test_the_budget_stops_once_the_ceiling_is_crossed() -> None:
    budget = SpendBudget(1.0)
    budget.add(Priced(usd=0.6))
    assert not budget.exceeded
    budget.add(Priced(usd=0.6))
    assert budget.exceeded
    assert "1.20" in budget.reason()
    # Stated as "checked after each execution" rather than claimed as a hard
    # cap, because the cost of a call is not knowable until it returns.
    assert "after each execution" in budget.reason()


def test_no_ceiling_never_stops() -> None:
    budget = SpendBudget(None)
    budget.add(Priced(usd=10_000.0))
    assert not budget.exceeded


def test_unpriced_models_are_remembered_not_counted_as_zero() -> None:
    budget = SpendBudget(1.0)
    budget.add(Priced(usd=None, unpriced=("my-finetune",)))
    assert budget.spent == 0.0
    assert budget.unpriced == ("my-finetune",)


# --------------------------------------------------------------------------- #
# Running
# --------------------------------------------------------------------------- #


def test_a_small_matrix_runs_and_is_fully_persisted(project: Fixture) -> None:
    summary = project.go()
    assert summary.status is RunStatus.COMPLETED
    assert summary.executed == 4
    assert summary.complete

    stored = project.store.executions(summary.run_id)
    assert len(stored) == 4
    assert {e.status for e in stored} == {ExecStatus.OK}
    # The baseline answers "ok" and the weak candidate answers "bad", so the
    # dataset's assertion splits them.
    by_candidate = {
        c: {e.outcome_class for e in stored if e.candidate_id == c}
        for c in ("baseline", "weak")
    }
    assert by_candidate["baseline"] == {Outcome.PASSED}
    assert by_candidate["weak"] == {Outcome.FAILED_ASSERTION}


def test_the_run_header_records_its_provenance(project: Fixture) -> None:
    summary = project.go()
    run = project.store.require_run(summary.run_id)
    assert run.config_hash == project.config.config_hash()
    assert run.dataset_hash == project.dataset.content_hash
    assert run.scheduling == THROUGHPUT
    assert run.status is RunStatus.COMPLETED
    assert run.finished_at


def test_metrics_and_rollups_are_written(project: Fixture) -> None:
    summary = project.go()
    metrics = project.store.metrics(summary.run_id)
    assert {m.metric for m in metrics} >= {"schema_valid_rate", "assertions_passed"}

    rollups = project.store.rollups(summary.run_id)
    assert len(rollups) == 4  # 2 candidates x 2 cases
    baseline = [r for r in rollups if r.candidate_id == "baseline"]
    assert all(r.n == 1 and r.c == 1 for r in baseline)
    assert all(r.wilson_lo is not None for r in baseline)


def test_a_ninety_execution_matrix_completes_with_mixed_outcomes(
    tmp_path: Path,
) -> None:
    """Day 9's acceptance criterion, exactly: 3 candidates x 10 cases x 3 repeats."""
    fixture = build(
        tmp_path,
        cases=10,
        repeats=3,
        candidates=(("weak", "bad"), ("worse", "terrible")),
        concurrency=8,
    )
    try:
        summary = fixture.go()

        assert summary.planned == 90
        assert summary.executed == 90
        assert summary.status is RunStatus.COMPLETED
        assert len(fixture.store.executions(summary.run_id)) == 90

        outcomes = summary.outcomes
        assert outcomes[Outcome.PASSED] == 30
        assert outcomes[Outcome.FAILED_ASSERTION] == 60
        assert summary.harness_errors == 0

        # Every cell is there exactly once, which is what makes n believable.
        cells = {
            (e.candidate_id, e.case_id, e.repeat_idx)
            for e in fixture.store.executions(summary.run_id)
        }
        assert len(cells) == 90
        assert len(fixture.store.rollups(summary.run_id)) == 30
    finally:
        fixture.store.close()


def test_results_are_persisted_as_they_finish_not_at_the_end(
    project: Fixture,
) -> None:
    """A KeyboardInterrupt at 95% must not take the first 85 rows with it."""
    seen: list[int] = []
    runner = project.runner()

    def counting_evaluator(**kwargs: Any) -> Verdict:
        # Count what is already in SQLite at the moment this unit is scored.
        seen.append(len(project.store.executions(runner.last_run_id or "")))
        return evaluator_for(project.config)(**kwargs)

    runner.evaluator = counting_evaluator
    asyncio.run(runner.run())
    assert seen == sorted(seen)
    assert seen[-1] >= 1, "rows were already committed before the last unit ran"


# --------------------------------------------------------------------------- #
# Crash isolation
# --------------------------------------------------------------------------- #


def test_a_failure_inside_the_harness_is_isolated_and_attributed_to_us(
    project: Fixture,
) -> None:
    """Day 10's acceptance criterion: a case engineered to raise inside the
    harness is reported as harness_error without affecting the other rows."""
    runner = project.runner()
    real = evaluator_for(project.config)

    def exploding(**kwargs: Any) -> Verdict:
        if kwargs["ctx"].case_id == "c1":
            raise RuntimeError("engineered harness failure")
        return real(**kwargs)

    runner.evaluator = exploding
    summary = asyncio.run(runner.run())
    project.store.save_rollups(
        case_rollups(summary.run_id, project.store.executions(summary.run_id))
    )

    assert summary.executed == 4
    assert summary.harness_errors == 2
    broken = [r for r in summary.results if r.unit.case_id == "c1"]
    assert {r.outcome for r in broken} == {Outcome.HARNESS_ERROR}
    assert all(not r.judgeable for r in broken)

    # The other case is untouched, which is the whole point.
    fine = [r for r in summary.results if r.unit.case_id == "c0"]
    assert {r.outcome for r in fine} == {Outcome.PASSED, Outcome.FAILED_ASSERTION}

    # And a harness error never enters a pass rate.
    rollups = {
        (r.candidate_id, r.case_id): r for r in project.store.rollups(summary.run_id)
    }
    assert rollups[("baseline", "c1")].n == 0
    assert rollups[("baseline", "c0")].n == 1


def test_an_agent_that_raises_is_recorded_not_fatal(tmp_path: Path) -> None:
    """One malformed case must never abort the matrix."""
    fixture = build(tmp_path, cases=2, repeats=1)
    (tmp_path / f"{MODULE}.py").write_text(
        AGENT.replace(
            "    routing = completion.parsed_as(Routing)",
            "    routing = completion.parsed_as(Routing)\n"
            '    if case_input.get("q") == "q1":\n'
            '        raise ZeroDivisionError("the agent under test has a bug")',
        ),
        encoding="utf-8",
    )
    sys.modules.pop(MODULE, None)
    fixture.target = load_target(f"{MODULE}:run_case", root=tmp_path)
    try:
        summary = fixture.go()
        assert summary.executed == 2
        outcomes = {r.unit.case_id: r.outcome for r in summary.results}
        assert outcomes["c1"] is Outcome.CRASHED
        assert outcomes["c0"] is Outcome.PASSED
        # The agent's bug is the user's to fix, and it is not a harness error.
        assert summary.harness_errors == 0
        crashed = next(
            e
            for e in fixture.store.executions(summary.run_id)
            if e.case_id == "c1"
        )
        assert crashed.error_class == "ZeroDivisionError"
        assert crashed.status is ExecStatus.ERROR
    finally:
        fixture.store.close()


def test_the_step_budget_aborts_a_runaway_rather_than_hanging(tmp_path: Path) -> None:
    """A loop is a finding, not a hang."""
    fixture = build(tmp_path, cases=1, repeats=1)
    (tmp_path / f"{MODULE}.py").write_text(
        AGENT.replace(
            "    routing = completion.parsed_as(Routing)",
            "    for _ in range(50):\n"
            '        with node("spin"):\n'
            "            await chat_model().complete(\n"
            '                [{"role": "user", "content": "again"}], schema=Routing\n'
            "            )\n"
            "    routing = completion.parsed_as(Routing)",
        ),
        encoding="utf-8",
    )
    sys.modules.pop(MODULE, None)
    fixture.target = load_target(f"{MODULE}:run_case", root=tmp_path)
    try:
        summary = fixture.go()
        assert [r.outcome for r in summary.results] == [Outcome.STEP_BUDGET]
        stored = fixture.store.executions(summary.run_id)[0]
        assert stored.steps == 9  # max_steps 8, plus the call that tripped it
        assert "max_steps" in (stored.error_message or "")
    finally:
        fixture.store.close()


# --------------------------------------------------------------------------- #
# Resume
# --------------------------------------------------------------------------- #


def test_an_interrupt_leaves_the_finished_half_and_resume_completes_it(
    tmp_path: Path,
) -> None:
    """Day 10's acceptance criterion: Ctrl-C at 50%, then --resume finishes."""
    fixture = build(tmp_path, cases=4, repeats=1, candidates=(("weak", "bad"),), concurrency=1)
    try:
        runner = fixture.runner()
        real = evaluator_for(fixture.config)
        done = 0

        def interrupt_halfway(**kwargs: Any) -> Verdict:
            nonlocal done
            verdict = real(**kwargs)
            done += 1
            if done == 4:
                raise KeyboardInterrupt
            return verdict

        runner.evaluator = interrupt_halfway
        with pytest.raises(KeyboardInterrupt):
            asyncio.run(runner.run())

        run_id = runner.last_run_id
        assert run_id is not None
        # Three finished and were committed; the fourth was interrupted and has
        # no row at all, which is what makes it work rather than a result.
        assert len(fixture.store.executions(run_id)) == 3
        assert fixture.store.require_run(run_id).status is RunStatus.RUNNING

        summary = fixture.go(resume=run_id)
        assert summary.run_id == run_id
        assert summary.resumed == 3
        assert summary.executed == 5
        assert summary.planned == 8
        assert summary.complete

        # Eight distinct cells, no duplicates: the derived exec_id is what
        # makes a resumed re-run replace a row instead of doubling n.
        stored = fixture.store.executions(run_id)
        assert len(stored) == 8
        assert len({e.exec_id for e in stored}) == 8
        assert all(r.n == 1 for r in fixture.store.rollups(run_id))
    finally:
        fixture.store.close()


def test_resume_refuses_when_the_dataset_changed_underneath_it(
    project: Fixture, tmp_path: Path
) -> None:
    """Joining half a matrix measured one way onto half measured another would
    manufacture a difference. The refusal is the feature."""
    summary = project.go()
    (tmp_path / "cases.jsonl").write_text(
        '{"case_id": "c0", "input": {"q": "changed"}, '
        '"assert": {"state_contains": {"intent": "different"}}}\n',
        encoding="utf-8",
    )
    project.dataset = load_dataset(tmp_path / "cases.jsonl")

    with pytest.raises(ConfigError, match="cannot resume") as err:
        project.runner().plan(resume=summary.run_id)
    assert "dataset changed" in str(err.value)
    assert "Start a fresh run" in (err.value.hint or "")


def test_resuming_a_finished_run_is_a_no_op(project: Fixture) -> None:
    first = project.go()
    again = project.go(resume=first.run_id)
    assert again.executed == 0
    assert again.resumed == 4
    assert again.complete
    assert len(project.store.executions(first.run_id)) == 4


def test_an_unknown_run_id_is_a_usage_error(project: Fixture) -> None:
    with pytest.raises(Exception, match=r"(?i)run"):
        project.runner().plan(resume="20200101-000000-aaaaaa")


# --------------------------------------------------------------------------- #
# Budgets and fail-fast
# --------------------------------------------------------------------------- #


def test_fail_fast_stops_after_a_streak_not_on_the_first_failure(
    tmp_path: Path,
) -> None:
    """Benchmarks are expected to contain failures - that is the output - so
    aborting on the first would make the flag useless."""
    fixture = build(
        tmp_path, cases=6, repeats=1, candidates=(("weak", "bad"),), concurrency=1
    )
    try:
        summary = fixture.go(fail_fast=True)
        assert summary.status is RunStatus.CANCELLED
        assert summary.aborted_reason is not None
        assert "failed 3 executions in a row" in summary.aborted_reason
        # The baseline's six passes came first and are all there.
        assert summary.passed == 6
        assert not summary.complete
        assert summary.executed < summary.planned
    finally:
        fixture.store.close()


def test_a_spend_ceiling_aborts_the_run(project: Fixture) -> None:
    runner = project.runner(max_spend_usd=0.0001)
    # Price the scripted provider so there is something to spend.
    runner.prices = PriceTable(prices={"good": (1000.0, 1000.0), "weak": (1000.0, 1000.0)})
    summary = asyncio.run(runner.run())
    assert summary.status is RunStatus.CANCELLED
    assert "spend budget reached" in (summary.aborted_reason or "")
    assert summary.spend_usd > 0


def test_an_execution_that_outlives_its_timeout_is_recorded_as_a_timeout(
    tmp_path: Path,
) -> None:
    fixture = build(tmp_path, cases=1, repeats=1)
    (tmp_path / f"{MODULE}.py").write_text(
        AGENT.replace(
            "    routing = completion.parsed_as(Routing)",
            "    import asyncio\n"
            "    await asyncio.sleep(5)\n"
            "    routing = completion.parsed_as(Routing)",
        ),
        encoding="utf-8",
    )
    sys.modules.pop(MODULE, None)
    fixture.target = load_target(f"{MODULE}:run_case", root=tmp_path)
    try:
        summary = fixture.go(timeout_s=0.2)
        assert [r.outcome for r in summary.results] == [Outcome.TIMEOUT]
        assert fixture.store.executions(summary.run_id)[0].status is ExecStatus.TIMEOUT
    finally:
        fixture.store.close()


# --------------------------------------------------------------------------- #
# Observation
# --------------------------------------------------------------------------- #


@dataclass
class Recorder:
    plans: list[Any] = field(default_factory=list)
    started: list[WorkUnit] = field(default_factory=list)
    done: list[Any] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    finished: list[Any] = field(default_factory=list)

    def on_plan(self, plan: Any) -> None:
        self.plans.append(plan)

    def on_unit_start(self, unit: WorkUnit) -> None:
        self.started.append(unit)

    def on_unit_done(self, result: Any) -> None:
        self.done.append(result)

    def on_note(self, message: str) -> None:
        self.notes.append(message)

    def on_finish(self, summary: Any) -> None:
        self.finished.append(summary)


def test_progress_goes_to_an_observer_and_the_runner_never_prints(
    project: Fixture, capsys: pytest.CaptureFixture[str]
) -> None:
    recorder = Recorder()
    runner = project.runner()
    runner.observer = recorder
    asyncio.run(runner.run())

    assert len(recorder.plans) == 1
    assert len(recorder.started) == 4
    assert len(recorder.done) == 4
    assert len(recorder.finished) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_an_evaluator_crash_is_our_bug_and_is_warned_about(project: Fixture) -> None:
    runner = project.runner()

    def broken(**kwargs: Any) -> Verdict:
        raise RuntimeError("evaluator is broken")

    runner.evaluator = broken
    summary = asyncio.run(runner.run())
    assert {r.outcome for r in summary.results} == {Outcome.HARNESS_ERROR}
    stored = project.store.executions(summary.run_id)[0]
    assert "evaluator raised RuntimeError" in stored.warnings_json


def test_evaluation_can_be_switched_off(project: Fixture) -> None:
    """--no-eval leaves the runner's own classification in place, and does not
    promote a clean execution to a pass: "the agent returned" is not "the agent
    was right"."""
    summary = project.go(evaluate=False)
    assert {r.outcome for r in summary.results} == {Outcome.UNCLASSIFIED}
    assert not project.store.metrics(summary.run_id)


def test_the_classification_passed_to_the_evaluator_describes_the_execution(
    project: Fixture,
) -> None:
    seen: list[Classification] = []
    runner = project.runner()
    real = evaluator_for(project.config)

    def spy(**kwargs: Any) -> Verdict:
        seen.append(kwargs["classification"])
        return real(**kwargs)

    runner.evaluator = spy
    asyncio.run(runner.run())
    assert len(seen) == 4
    assert all(c.ok and c.needs_evaluation for c in seen)
