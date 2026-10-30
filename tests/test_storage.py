"""The trace store.

The headline test is :func:`test_a_run_survives_the_process_that_produced_it` -
step 12's acceptance criterion. Everything else guards a way the store could be
plausible and wrong:

- a resumed run that double-counts a unit, halving a candidate's pass rate;
- a retry that leaves two trajectories interleaved under one id;
- a half-written execution that reads as an agent which stopped early;
- a prune that collects a payload a live run still needs.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from slmeval import __version__
from slmeval.core.context import RunContext
from slmeval.core.trace import JsonStage, LLMCallTrace, NodeVisit, SchemaStatus, ToolCallTrace
from slmeval.errors import EnvironmentError_
from slmeval.sandbox.faults import Delivery, Fault, FaultKind, FaultVector
from slmeval.storage import codec, ingest
from slmeval.storage.db import RunNotFound, RunStore, make_exec_id
from slmeval.storage.migrations import SCHEMA_VERSION, current_version, migrate
from slmeval.storage.models import (
    EMPTY_PAYLOAD,
    CandidateRecord,
    CaseRollupRow,
    EvaluationRow,
    ExecStatus,
    ExecutionRecord,
    Outcome,
    Payload,
    RunStatus,
)
from slmeval.storage.provenance import Provenance, divergences, host_info, new_run_id

NOW = datetime(2026, 2, 3, 11, 42, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[RunStore]:
    with RunStore.for_project(tmp_path) as store:
        yield store


def a_run(
    store: RunStore,
    *,
    run_id: str | None = None,
    started_at: datetime | None = None,
    status: RunStatus = RunStatus.RUNNING,
    dataset_hash: str = "d00d",
    name: str = "refund-agent",
) -> str:
    from slmeval.storage.models import RunRecord

    rid = run_id or new_run_id(now=started_at or NOW)
    store.save_run(
        RunRecord(
            run_id=rid,
            name=name,
            started_at=(started_at or NOW).isoformat(),
            config_json='{"benchmark":{"name":"refund-agent"}}',
            config_hash="cfg123",
            dataset_hash=dataset_hash,
            slmeval_version=__version__,
            mode="replay",
            seed=42,
            status=status,
            runs_per_case=3,
        )
    )
    return rid


def an_execution(
    run_id: str,
    *,
    candidate_id: str = "gemma3-4b",
    case_id: str = "refund_207",
    repeat_idx: int = 0,
    status: ExecStatus = ExecStatus.OK,
    outcome: Outcome = Outcome.PASSED,
    output: Payload = EMPTY_PAYLOAD,
) -> ExecutionRecord:
    return ExecutionRecord(
        exec_id=make_exec_id(run_id, candidate_id, case_id, repeat_idx),
        run_id=run_id,
        candidate_id=candidate_id,
        case_id=case_id,
        repeat_idx=repeat_idx,
        status=status,
        outcome_class=outcome,
        duration_ms=312.5,
        steps=4,
        total_prompt_tokens=900,
        total_completion_tokens=120,
        output=output,
    )


# --------------------------------------------------------------------------- #
# Schema and migrations
# --------------------------------------------------------------------------- #


def test_fresh_file_is_at_the_current_version(store: RunStore) -> None:
    assert store.schema_version == SCHEMA_VERSION
    history = store.migration_history()
    assert [v for v, *_ in history] == [1]
    assert history[0][3] == __version__


def test_migrating_twice_is_a_no_op(tmp_path: Path) -> None:
    """Migration runs on every open, so it has to be cheap and idempotent -
    there is deliberately no separate `slmeval migrate` to forget."""
    path = tmp_path / "runs.sqlite"
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        assert migrate(conn) == [1]
        assert migrate(conn) == []
        assert current_version(conn) == SCHEMA_VERSION
    finally:
        conn.close()


def test_a_newer_schema_is_refused_not_guessed(tmp_path: Path) -> None:
    """Writing a half-understood schema would corrupt the one dataset a user
    cannot regenerate."""
    path = tmp_path / "runs.sqlite"
    conn = sqlite3.connect(path, isolation_level=None)
    try:
        migrate(conn)
        conn.execute(
            "INSERT INTO schema_version VALUES (99, 'from the future', ?, '9.9.9')",
            (NOW.isoformat(),),
        )
    finally:
        conn.close()

    with pytest.raises(EnvironmentError_) as err:
        RunStore(path)._conn()
    assert "newer slmeval" in str(err.value)
    assert "upgrade slmeval" in (err.value.hint or "")


def test_foreign_keys_are_on(store: RunStore) -> None:
    """`cache prune` is a one-line DELETE only because the cascades fire, and
    SQLite leaves them off by default, per connection."""
    assert store._conn().execute("PRAGMA foreign_keys").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# Runs, candidates, provenance
# --------------------------------------------------------------------------- #


def test_run_round_trip(store: RunStore) -> None:
    run_id = a_run(store)
    run = store.require_run(run_id)
    assert run.name == "refund-agent"
    assert run.status is RunStatus.RUNNING
    assert run.seed == 42
    assert run.finished_at is None


def test_finish_run_does_not_rewrite_provenance(store: RunStore) -> None:
    """The provenance written at the start is what the run actually used.
    Rewriting it at the end from whatever the config says now would re-describe
    history."""
    run_id = a_run(store)
    store.finish_run(run_id, status=RunStatus.COMPLETED)

    run = store.require_run(run_id)
    assert run.status is RunStatus.COMPLETED
    assert run.finished_at
    assert run.config_hash == "cfg123"
    assert run.dataset_hash == "d00d"


def test_run_ids_sort_chronologically() -> None:
    early = new_run_id(now=datetime(2026, 1, 1, 9, 0))
    late = new_run_id(now=datetime(2026, 1, 1, 9, 1))
    assert early < late
    assert len(new_run_id()) == len("20260101-090000-abcdef")


def test_run_ids_do_not_collide_within_a_second() -> None:
    """A CI matrix starts several runs in the same second."""
    ids = {new_run_id(now=NOW) for _ in range(200)}
    assert len(ids) == 200


def test_candidates_keep_declaration_order(store: RunStore) -> None:
    run_id = a_run(store)
    store.save_candidates(
        [
            CandidateRecord(run_id, "gemma3-4b", "{}", ordinal=1),
            CandidateRecord(run_id, "baseline", "{}", role="baseline", ordinal=0),
            CandidateRecord(run_id, "qwen2.5-7b", "{}", ordinal=2),
        ]
    )
    detail = store.reconstruct_run(run_id)
    assert [c.candidate_id for c in detail.candidates] == [
        "baseline",
        "gemma3-4b",
        "qwen2.5-7b",
    ]
    assert detail.baseline is not None
    assert detail.baseline.candidate_id == "baseline"


def test_host_info_carries_no_identity() -> None:
    """This file gets attached to bug reports."""
    info = host_info()
    assert set(info) == {"os", "release", "machine", "python", "implementation", "cpus"}
    assert not any("MOHIT" in str(v) for v in info.values())


def test_divergent_dataset_blocks_a_comparison(store: RunStore) -> None:
    """The refusal is the feature: a 12-point 'regression' caused by a dataset
    edit teaches people to distrust the numbers, once, permanently."""
    first = store.require_run(a_run(store, dataset_hash="aaaa"))
    second = store.require_run(a_run(store, dataset_hash="bbbb"))

    found = divergences(first, second)

    assert [d.field for d in found] == ["dataset_hash"]
    assert found[0].blocking is True


def test_a_version_difference_is_a_caveat_not_a_refusal(store: RunStore) -> None:
    """Different code computed the numbers - worth printing, rarely worth
    refusing over."""
    left = store.require_run(a_run(store))
    right = SimpleNamespace(
        dataset_hash=left.dataset_hash,
        config_hash=left.config_hash,
        mode=left.mode,
        seed=left.seed,
        slmeval_version="0.0.1",
    )

    found = divergences(left, right)

    assert [d.field for d in found] == ["slmeval_version"]
    assert found[0].blocking is False


def test_provenance_collect_stamps_the_build(tmp_path: Path) -> None:
    prov = Provenance.collect(config_hash="cfg", dataset_hash="ds", root=tmp_path)
    assert prov.slmeval_version == __version__
    assert codec.loads(prov.host)["os"]
    assert prov.started_at


# --------------------------------------------------------------------------- #
# Executions and payloads
# --------------------------------------------------------------------------- #


def test_execution_round_trip(store: RunStore) -> None:
    run_id = a_run(store)
    record = an_execution(run_id)
    store.save_execution(record)

    read = store.get_execution(record.exec_id)
    assert read == record


def test_small_payloads_stay_in_the_row(store: RunStore) -> None:
    payload = store.payload_of({"status": "refunded"})
    assert payload.inline is not None
    assert payload.digest is None
    assert store.blobs.stats().count == 0


def test_large_payloads_go_to_the_blob_store(store: RunStore) -> None:
    payload = store.payload_of({"rows": ["x" * 40 for _ in range(100)]})
    assert payload.inline is None
    assert payload.digest is not None
    assert store.blobs.exists(payload.digest)
    assert store.load_json(payload)["rows"][0] == "x" * 40


def test_a_pruned_payload_degrades_instead_of_raising(store: RunStore) -> None:
    """The trace row is intact, so the timeline around it still has to print."""
    payload = store.payload_of({"rows": ["y" * 40 for _ in range(100)]})
    assert payload.digest
    store.blobs.path_for(payload.digest).unlink()

    assert store.load(payload) is None
    assert store.load_json(payload) is None


def test_an_execution_and_its_traces_are_one_transaction(
    store: RunStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A committed execution row with only some of its tool calls does not read
    as a crash - it reads as an agent that stopped early, which is a finding the
    harness would have invented."""
    run_id = a_run(store)
    record = an_execution(run_id)
    nodes = [ingest.node_row(LLMCallTrace(seq=0, model_id="m", provider="p", model="m"),
                             exec_id=record.exec_id, store=store)]

    real_insert = RunStore._insert

    def exploding_insert(
        conn: sqlite3.Connection,
        table: str,
        columns: Sequence[str],
        rows: Sequence[tuple[object, ...]],
    ) -> None:
        if table == "node_traces":
            raise sqlite3.OperationalError("disk I/O error")
        real_insert(conn, table, columns, rows)

    monkeypatch.setattr(RunStore, "_insert", staticmethod(exploding_insert))
    with pytest.raises(sqlite3.OperationalError):
        store.save_execution(record, node_traces=nodes)

    monkeypatch.undo()
    assert store.get_execution(record.exec_id) is None
    assert store.executions(run_id) == []


def test_unknown_enum_values_do_not_break_a_read(store: RunStore) -> None:
    """A file written by a newer build can hold an outcome this build has never
    heard of. One unfamiliar label beats a crash."""
    run_id = a_run(store)
    record = an_execution(run_id)
    store.save_execution(record)
    store._conn().execute(
        "UPDATE executions SET outcome_class = 'invented_next_year' WHERE exec_id = ?",
        (record.exec_id,),
    )

    read = store.get_execution(record.exec_id)
    assert read is not None
    assert read.outcome_class is Outcome.UNCLASSIFIED


# --------------------------------------------------------------------------- #
# Resume and retries
# --------------------------------------------------------------------------- #


def test_exec_ids_are_deterministic() -> None:
    """What makes --resume safe: a random id would insert a second row for a
    unit already finished, doubling that candidate's n."""
    a = make_exec_id("20260203-114200-abc123", "gemma3-4b", "refund_207", 2)
    b = make_exec_id("20260203-114200-abc123", "gemma3-4b", "refund_207", 2)
    assert a == b
    assert a.endswith("gemma3-4b.refund_207.r2")


def test_rewriting_a_unit_replaces_it(store: RunStore) -> None:
    run_id = a_run(store)
    store.save_execution(an_execution(run_id, status=ExecStatus.ERROR, outcome=Outcome.CRASHED))
    store.save_execution(an_execution(run_id, status=ExecStatus.OK, outcome=Outcome.PASSED))

    rows = store.executions(run_id)
    assert len(rows) == 1
    assert rows[0].outcome_class is Outcome.PASSED


def test_a_retry_does_not_interleave_two_trajectories(store: RunStore) -> None:
    """INSERT OR REPLACE deletes the previous row, and with foreign_keys=ON
    that cascades away the previous attempt's traces before the new ones land."""
    run_id = a_run(store)
    record = an_execution(run_id)
    first = [
        ingest.tool_row(ToolCallTrace(seq=i, name=f"old_{i}"), exec_id=record.exec_id, store=store)
        for i in range(5)
    ]
    store.save_execution(record, tool_traces=first)

    second = [
        ingest.tool_row(ToolCallTrace(seq=0, name="new_0"), exec_id=record.exec_id, store=store)
    ]
    store.save_execution(record, tool_traces=second)

    tools = store.tool_traces(record.exec_id)
    assert [t.tool_name for t in tools] == ["new_0"]


def test_completed_units_drives_resume(store: RunStore) -> None:
    run_id = a_run(store)
    store.save_execution(an_execution(run_id, repeat_idx=0, status=ExecStatus.OK))
    store.save_execution(an_execution(run_id, repeat_idx=1, status=ExecStatus.ERROR))
    store.save_execution(an_execution(run_id, repeat_idx=2, status=ExecStatus.SKIPPED))

    done = store.completed_units(run_id)

    assert ("gemma3-4b", "refund_207", 0) in done
    # An error is a result; the model failed and that is data.
    assert ("gemma3-4b", "refund_207", 1) in done
    # Skipped was planned and never reached, so resuming has to pick it up.
    assert ("gemma3-4b", "refund_207", 2) not in done


def test_set_outcome_classifies_after_the_fact(store: RunStore) -> None:
    run_id = a_run(store)
    record = an_execution(run_id, outcome=Outcome.UNCLASSIFIED)
    store.save_execution(record)

    store.set_outcome(record.exec_id, Outcome.FORBIDDEN_TOOL, cost_usd=0.0042)

    read = store.get_execution(record.exec_id)
    assert read is not None
    assert read.outcome_class is Outcome.FORBIDDEN_TOOL
    assert read.cost_usd == pytest.approx(0.0042)


# --------------------------------------------------------------------------- #
# Id resolution
# --------------------------------------------------------------------------- #


def test_run_id_prefix_resolves(store: RunStore) -> None:
    run_id = a_run(store)
    assert store.resolve_run_id(run_id[:11]) == run_id
    assert store.resolve_run_id(run_id) == run_id


def test_ambiguous_run_id_is_refused(store: RunStore) -> None:
    """Picking the newest match would eventually print a report for a run the
    user was not asking about."""
    a_run(store, run_id="20260203-114200-aaaaaa")
    a_run(store, run_id="20260203-114200-bbbbbb")

    with pytest.raises(RunNotFound) as err:
        store.resolve_run_id("20260203")
    assert "matches 2 runs" in str(err.value)


def test_exec_id_resolves_by_suffix(store: RunStore) -> None:
    """The readable part of an exec id is at the end - it is what a user reads
    off the failure table."""
    run_id = a_run(store)
    record = an_execution(run_id)
    store.save_execution(record)

    assert store.resolve_exec_id("gemma3-4b.refund_207.r0") == record.exec_id


def test_unknown_ids_name_what_is_available(store: RunStore) -> None:
    with pytest.raises(RunNotFound):
        store.resolve_run_id("nope")
    with pytest.raises(RunNotFound):
        store.resolve_exec_id("nope")
    with pytest.raises(RunNotFound):
        store.reconstruct("nope")


# --------------------------------------------------------------------------- #
# Evaluations and rollups
# --------------------------------------------------------------------------- #


def test_evaluations_and_rollups_round_trip(store: RunStore) -> None:
    run_id = a_run(store)
    record = an_execution(run_id)
    store.save_execution(record)
    store.save_evaluations(
        [
            EvaluationRow(record.exec_id, tier=1, metric="schema_adherence", value_num=1.0),
            EvaluationRow(record.exec_id, tier=2, metric="tool_f1", value_num=0.91),
        ]
    )
    store.save_rollups(
        [CaseRollupRow(run_id, "gemma3-4b", "refund_207", n=3, c=2, pass_hat_k=0.666)]
    )

    metrics = {e.metric: e.value_num for e in store.evaluations(record.exec_id)}
    assert metrics == {"schema_adherence": 1.0, "tool_f1": 0.91}
    rollup = store.rollups(run_id)[0]
    assert (rollup.n, rollup.c) == (3, 2)


def test_re_scoring_replaces_a_metric(store: RunStore) -> None:
    """The evaluator pass runs again after a judge recalibration; two rows for
    one metric would double-count it."""
    run_id = a_run(store)
    record = an_execution(run_id)
    store.save_execution(record)
    store.save_evaluations([EvaluationRow(record.exec_id, 1, "tool_f1", value_num=0.5)])
    store.save_evaluations([EvaluationRow(record.exec_id, 1, "tool_f1", value_num=0.9)])

    rows = store.evaluations(record.exec_id)
    assert len(rows) == 1
    assert rows[0].value_num == 0.9


# --------------------------------------------------------------------------- #
# Fault vectors
# --------------------------------------------------------------------------- #


def test_fault_vector_round_trip() -> None:
    vector = FaultVector()
    vector.add(
        Fault(
            kind=FaultKind.EMPTY_RESULT,
            delivery=Delivery.RETURN,
            tool_name="fetch_order",
            call_index=1,
        ),
        node="order_lookup",
    )
    vector.add(
        Fault(
            kind=FaultKind.RATE_LIMIT_429,
            delivery=Delivery.RAISE,
            tool_name="issue_refund",
            call_index=0,
        ),
        applied=False,
    )

    back = ingest.decode_fault_vector(ingest.encode_fault_vector(vector))

    assert back.digest == vector.digest
    assert back.labels() == vector.labels()


def test_an_unnameable_fault_is_dropped_not_guessed() -> None:
    """A vector silently missing an entry would compare equal to a run that
    never scheduled it - the failure this field exists to catch. So an entry we
    cannot name must not become a plausible-looking different one."""
    text = codec.dumps(
        [
            {"kind": "invented_fault", "delivery": "return", "tool": "t", "call_index": 0},
            {"kind": "empty_result", "delivery": "return", "tool": "t", "call_index": 1},
        ]
    )
    back = ingest.decode_fault_vector(text)
    assert [f.kind for f in back] == [FaultKind.EMPTY_RESULT]


def test_empty_vector_is_not_null(store: RunStore) -> None:
    assert ingest.encode_fault_vector(None) == "[]"
    assert ingest.encode_fault_vector(FaultVector()) == "[]"


# --------------------------------------------------------------------------- #
# The acceptance test
# --------------------------------------------------------------------------- #


def a_five_step_context(run_id: str) -> RunContext:
    """One execution's worth of telemetry, as the runner would leave it."""
    ctx = RunContext(run_id=run_id, candidate_id="gemma3-4b", case_id="refund_207", attempt=1)
    for seq, name in enumerate(["intent_router", "order_lookup", "policy_check", "respond"]):
        ctx.trajectory.append(NodeVisit(path=(name,), seq=seq))
    ctx.llm_calls.extend(
        [
            LLMCallTrace(
                seq=0,
                model_id="gemma3-4b",
                provider="ollama",
                model="gemma3:4b-q4",
                node="intent_router",
                node_path=("intent_router",),
                schema_name="Routing",
                schema_status=SchemaStatus.VALID,
                structured_requested="auto",
                structured_mode_used="json_mode",
                rungs_tried=("native", "json_mode"),
                repair_attempts=1,
                json_stage=JsonStage.FENCED,
                prompt_tokens=412,
                completion_tokens=37,
                latency_ms=288.0,
                ttft_ms=91.0,
            ),
            LLMCallTrace(
                seq=1,
                model_id="gemma3-4b",
                provider="ollama",
                model="gemma3:4b-q4",
                node="policy_check",
                node_path=("policy_check",),
                schema_status=SchemaStatus.ERROR,
                tokens_estimated=True,
                retries=2,
                retry_ms=1400.0,
                error="connection reset",
                error_type="ProviderError",
                raw={"choices": [], "note": "kept for the failure report"},
            ),
        ]
    )
    ctx.tool_calls.extend(
        [
            ToolCallTrace(
                seq=0,
                name="fetch_order",
                node="order_lookup",
                node_path=("order_lookup",),
                arguments={"order_id": "A-1001"},
                cassette_key="abc123",
                outcome="replayed",
                match_tier="normalized",
                tool_version="2",
                latency_ms=0.4,
            ),
            ToolCallTrace(
                seq=1,
                name="issue_refund",
                node="respond",
                node_path=("respond",),
                arguments={"order_id": "A-1001", "amount": 42.5},
                outcome="miss",
                miss_verdict="hallucinated_parameter",
                error="CASSETTE_MISS",
            ),
        ]
    )
    ctx.steps = 4
    ctx.warnings.append("needed prompt repair on 1 call")
    ctx.fault_vector.add(
        Fault(
            kind=FaultKind.EMPTY_RESULT,
            delivery=Delivery.RETURN,
            tool_name="fetch_order",
            call_index=0,
        ),
        node="order_lookup",
    )
    return ctx


def test_a_run_survives_the_process_that_produced_it(tmp_path: Path) -> None:
    """Step 12's acceptance criterion.

    Write a run, close the store, drop every in-memory object, reopen from
    disk, and rebuild the timeline. If this passes, `slmeval report` and the CI
    gate can work from storage alone - which is the whole premise of week 2.
    """
    with RunStore.for_project(tmp_path) as store:
        run_id = a_run(store)
        store.save_candidates(
            [
                CandidateRecord(run_id, "baseline", '{"model":"gpt-4o"}', role="baseline"),
                CandidateRecord(run_id, "gemma3-4b", '{"model":"gemma3:4b-q4"}', ordinal=1),
            ]
        )
        ctx = a_five_step_context(run_id)
        record = ingest.save_context(
            store,
            ctx,
            status=ExecStatus.OK,
            outcome=Outcome.CASSETTE_MISS,
            duration_ms=1812.0,
            output={"answer": "refund issued", "detail": "x" * 4000},
            state={"refunded": True},
        )
        store.finish_run(run_id)
        exec_id = record.exec_id
        expected_llm = list(ctx.llm_calls)
        expected_tools = list(ctx.tool_calls)
        expected_digest = ctx.fault_vector.digest

    del ctx, record, store

    with RunStore.for_project(tmp_path) as reopened:
        detail = reopened.reconstruct(exec_id)
        llm, tools = ingest.traces_from_detail(detail)

        # The execution itself.
        assert detail.execution.run_id == run_id
        assert detail.execution.outcome_class is Outcome.CASSETTE_MISS
        assert detail.execution.steps == 4
        assert detail.execution.duration_ms == 1812.0
        assert detail.execution.repeat_idx == 1
        assert detail.execution.total_prompt_tokens == 412
        # One estimated call taints the total; the report must mark it `~`.
        assert detail.execution.tokens_estimated is True

        # The trajectory, including nodes that made no model call.
        assert detail.trajectory == (
            ("intent_router",),
            ("order_lookup",),
            ("policy_check",),
            ("respond",),
        )
        assert codec.decode_strings(detail.execution.warnings_json) == (
            "needed prompt repair on 1 call",
        )

        # The payload that was too big for a row came back through the blobs.
        assert detail.output["answer"] == "refund issued"
        assert detail.state == {"refunded": True}

        # The node timeline, field for field.
        assert len(llm) == 2
        first = llm[0]
        assert first.node == "intent_router"
        assert first.node_path == ("intent_router",)
        assert first.schema_status is SchemaStatus.VALID
        assert first.structured_mode_used == "json_mode"
        assert first.rungs_tried == ("native", "json_mode")
        assert first.json_stage is JsonStage.FENCED
        assert first.schema_lenient is True
        assert first.repair_attempts == 1
        assert first.total_tokens == 449
        assert llm[1].schema_status is SchemaStatus.ERROR
        assert llm[1].error_type == "ProviderError"
        assert llm[1].retries == 2
        # The raw body of the failing call is what the failure report prints.
        assert reopened.load_json(detail.node_traces[1].raw)["note"].startswith("kept for")
        assert detail.node_traces[0].raw.present is False

        # The tool timeline, with arguments and the miss verdict.
        assert [t.name for t in tools] == ["fetch_order", "issue_refund"]
        assert tools[0].match_tier == "normalized"
        assert tools[0].arguments == {"order_id": "A-1001"}
        assert tools[1].outcome == "miss"
        assert tools[1].miss_verdict == "hallucinated_parameter"
        assert detail.tool_arguments[1] == {"order_id": "A-1001", "amount": 42.5}

        # The faults it actually met, not the ones that were scheduled.
        assert ingest.decode_fault_vector(
            detail.execution.fault_vector_json
        ).digest == expected_digest

        # And the shapes match what went in.
        assert [(c.seq, c.model, c.node) for c in llm] == [
            (c.seq, c.model, c.node) for c in expected_llm
        ]
        assert [(t.seq, t.name, t.outcome) for t in tools] == [
            (t.seq, t.name, t.outcome) for t in expected_tools
        ]

        # The run header and its candidates, too.
        run_detail = reopened.reconstruct_run(run_id)
        assert run_detail.run.status is RunStatus.COMPLETED
        assert run_detail.baseline is not None
        assert len(run_detail.executions) == 1


# --------------------------------------------------------------------------- #
# Pruning
# --------------------------------------------------------------------------- #


def test_deleting_a_run_cascades(store: RunStore) -> None:
    run_id = a_run(store)
    ctx = a_five_step_context(run_id)
    ingest.save_context(store, ctx, status=ExecStatus.OK)
    store.save_rollups([CaseRollupRow(run_id, "gemma3-4b", "refund_207", n=1, c=1)])
    assert store.stats().tool_traces == 2

    store.delete_runs([run_id])

    stats = store.stats()
    assert (stats.runs, stats.executions, stats.node_traces, stats.tool_traces) == (0, 0, 0, 0)
    assert store.rollups(run_id) == []


def test_prune_keeps_the_last_few_whatever_their_age(store: RunStore) -> None:
    """Someone who has not benchmarked in two months still wants their last
    run when they come back - it is what the next one gets compared against."""
    old = [
        a_run(store, run_id=f"20250101-00000{i}-aaaaa{i}", started_at=datetime(2025, 1, 1, 0, 0, i))
        for i in range(5)
    ]
    for rid in old:
        store.finish_run(rid)

    stale, _ = store.prune(older_than=timedelta(days=1), keep_last=2, now=NOW)

    assert [r.run_id for r in stale] == old[:3]
    assert {r.run_id for r in store.list_runs()} == set(old[3:])


def test_prune_frees_payloads_nothing_references(store: RunStore) -> None:
    run_id = a_run(store, run_id="20250101-000000-aaaaaa", started_at=datetime(2025, 1, 1))
    store.finish_run(run_id)
    big = store.payload_of({"rows": ["z" * 40 for _ in range(100)]})
    assert big.digest
    store.save_execution(an_execution(run_id, output=big))

    _, swept = store.prune(older_than=timedelta(days=1), grace=timedelta(0), now=NOW)

    assert swept.deleted == 1
    assert not store.blobs.exists(big.digest)


def test_prune_keeps_payloads_shared_with_a_surviving_run(store: RunStore) -> None:
    """A payload shared by three runs is referenced until the last of them is
    gone. That sharing is the reason the blob store exists."""
    shared = store.payload_of({"rows": ["s" * 40 for _ in range(100)]})
    assert shared.digest
    old = a_run(store, run_id="20250101-000000-aaaaaa", started_at=datetime(2025, 1, 1))
    new = a_run(store, run_id="20260203-114200-bbbbbb", started_at=NOW)
    for rid in (old, new):
        store.finish_run(rid)
        store.save_execution(an_execution(rid, output=shared))

    stale, swept = store.prune(older_than=timedelta(days=1), grace=timedelta(0), now=NOW)

    assert [r.run_id for r in stale] == [old]
    assert swept.deleted == 0
    assert store.blobs.exists(shared.digest)


def test_prune_refuses_while_a_run_is_in_flight(store: RunStore) -> None:
    """Sweeping during a run races a process that is still writing rows, and
    the grace period narrows that window rather than closing it."""
    a_run(store, status=RunStatus.RUNNING)

    with pytest.raises(EnvironmentError_) as err:
        store.prune(older_than=timedelta(days=1))
    assert "still in progress" in str(err.value)
    assert "--force" in (err.value.hint or "")


def test_force_prunes_past_a_crash_leftover(store: RunStore) -> None:
    a_run(store, run_id="20250101-000000-aaaaaa", started_at=datetime(2025, 1, 1),
          status=RunStatus.RUNNING)

    stale, _ = store.prune(older_than=timedelta(days=1), force=True, now=NOW)

    assert len(stale) == 1


def test_dry_run_changes_nothing(store: RunStore) -> None:
    run_id = a_run(store, run_id="20250101-000000-aaaaaa", started_at=datetime(2025, 1, 1))
    store.finish_run(run_id)

    stale, _ = store.prune(older_than=timedelta(days=1), dry_run=True, now=NOW)

    assert [r.run_id for r in stale] == [run_id]
    assert store.get_run(run_id) is not None


# --------------------------------------------------------------------------- #
# Lifecycle, threads, Windows
# --------------------------------------------------------------------------- #


def test_concurrent_writers(tmp_path: Path) -> None:
    """The runner writes executions from several worker threads."""
    with RunStore.for_project(tmp_path) as store:
        run_id = a_run(store)
        errors: list[BaseException] = []

        def write(index: int) -> None:
            try:
                store.save_execution(an_execution(run_id, case_id=f"case_{index}"))
            except BaseException as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert len(store.executions(run_id)) == 8


def test_close_releases_the_file(tmp_path: Path) -> None:
    """On Windows an unclosed handle keeps the database locked, so a user
    cannot delete their own cache and a test cannot clean up after itself."""
    store = RunStore.for_project(tmp_path)
    a_run(store)
    store.close()

    (tmp_path / ".slmeval" / "runs.sqlite").unlink()
    with pytest.raises(RuntimeError):
        store.get_run("anything")


def test_stats_on_an_empty_project(tmp_path: Path) -> None:
    with RunStore.for_project(tmp_path) as store:
        stats = store.stats()
    assert stats.runs == 0
    assert stats.db_bytes > 0
    assert stats.oldest_run is None


def test_db_bytes_counts_the_wal(store: RunStore) -> None:
    """Reporting only the main file would show a user hunting for disk space a
    number that does not match their file manager."""
    a_run(store)
    main = store.path.stat().st_size
    assert store.stats().db_bytes >= main


def test_codec_tolerates_a_hand_edited_column() -> None:
    """A malformed JSON column degrades to empty; the rest of the timeline
    still prints."""
    assert codec.loads("{not json", default={}) == {}
    assert codec.decode_paths("[[") == ()
    assert codec.decode_strings(None) == ()


def test_codec_writes_stable_json() -> None:
    """Identical payloads have to hash identically or dedupe stops working."""
    assert codec.dumps({"b": 1, "a": 2}) == codec.dumps({"a": 2, "b": 1})
    assert " " not in codec.dumps({"a": [1, 2]})


def test_payload_of_a_value_that_is_not_json(store: RunStore) -> None:
    """A state object holding something exotic must not take down the write -
    `default=str` in the codec is what keeps a benchmark from dying at the
    persistence step after twenty minutes of work."""
    payload = store.payload_of({"when": NOW, "path": Path("x")})
    assert isinstance(store.load_json(payload)["when"], str)


def test_reconstruct_run_leaves_out_the_timelines(store: RunStore) -> None:
    """A 1,500-execution matrix's full timeline is tens of megabytes and the
    scorecard needs none of it."""
    run_id = a_run(store)
    ingest.save_context(store, a_five_step_context(run_id), status=ExecStatus.OK)

    detail = store.reconstruct_run(run_id)

    assert len(detail.executions) == 1
    assert not hasattr(detail, "node_traces")
