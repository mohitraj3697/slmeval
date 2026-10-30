"""Between the run context and the rows: trace objects in, trace objects out.

The runner holds :mod:`slmeval.core.trace` dataclasses; the store holds rows.
This module is the only place that knows both, which keeps two things true that
would otherwise drift. The runner never learns about SQLite - it hands over a
finished :class:`~slmeval.core.context.RunContext` and is done. And the report
layer reads rows, never live objects, so the acceptance test for step 12 is a
real test rather than a formality: write a run, drop every object, rehydrate,
and the timeline has to come back identical.

The rehydration direction is not symmetry for its own sake. It is the thing
that proves a run survived the process that produced it, and the evaluators in
week 2 consume rehydrated traces, so a field that writes but does not read back
is a metric that silently reads zero.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from slmeval.core.context import RunContext
from slmeval.core.trace import JsonStage, LLMCallTrace, SchemaStatus, ToolCallTrace
from slmeval.sandbox.faults import Delivery, FaultKind, FaultVector, InjectedFault
from slmeval.storage import codec
from slmeval.storage.db import RunStore, make_exec_id
from slmeval.storage.models import (
    EMPTY_PAYLOAD,
    EvaluationRow,
    ExecStatus,
    ExecutionRecord,
    NodeTraceRow,
    Outcome,
    ToolTraceRow,
)

# --------------------------------------------------------------- fault vector


def encode_fault_vector(vector: FaultVector | None) -> str:
    """Serialize the faults an execution actually met.

    The vector, not the schedule: a candidate that stopped after two calls
    never met the fault scheduled for the fourth, and the comparison table has
    to compare what happened.
    """
    if vector is None or not len(vector):
        return "[]"
    return codec.dumps(
        [
            {
                "seq": f.seq,
                "tool": f.tool_name,
                "call_index": f.call_index,
                "kind": f.kind.value,
                "delivery": f.delivery.value,
                "node": f.node,
                "applied": f.applied,
            }
            for f in vector
        ]
    )


def decode_fault_vector(text: str | None) -> FaultVector:
    """Rebuild a fault vector from its column.

    Unknown kinds and deliveries are dropped rather than guessed. A fault we
    cannot name cannot be compared for parity, and a vector silently missing an
    entry would compare *equal* to a run that never scheduled it - the one
    failure mode this field exists to catch.
    """
    raw = codec.loads(text, [])
    if not isinstance(raw, list):
        return FaultVector()
    faults: list[InjectedFault] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        try:
            kind = FaultKind(str(item.get("kind")))
            delivery = Delivery(str(item.get("delivery")))
        except ValueError:
            continue
        faults.append(
            InjectedFault(
                seq=int(item.get("seq", len(faults))),
                tool_name=str(item.get("tool", "")),
                call_index=int(item.get("call_index", 0)),
                kind=kind,
                delivery=delivery,
                node=item.get("node"),
                applied=bool(item.get("applied", True)),
            )
        )
    return FaultVector(faults=faults)


# ------------------------------------------------------------------ to rows


def node_row(trace: LLMCallTrace, *, exec_id: str, store: RunStore) -> NodeTraceRow:
    return NodeTraceRow(
        exec_id=exec_id,
        seq=trace.seq,
        node_name=trace.node or "",
        node_path=codec.encode_paths([trace.node_path] if trace.node_path else []),
        model_id=trace.model_id,
        provider=trace.provider,
        model=trace.model,
        latency_ms=trace.latency_ms,
        ttft_ms=trace.ttft_ms,
        retries=trace.retries,
        retry_ms=trace.retry_ms,
        prompt_tokens=trace.prompt_tokens,
        completion_tokens=trace.completion_tokens,
        tokens_estimated=trace.tokens_estimated,
        schema_name=trace.schema_name,
        schema_state=trace.schema_status.value,
        structured_requested=trace.structured_requested,
        structured_mode=trace.structured_mode_used,
        rungs_tried=codec.encode_strings(trace.rungs_tried),
        repair_attempts=trace.repair_attempts,
        json_stage=trace.json_stage.value if trace.json_stage else None,
        error_class=trace.error_type,
        error_message=trace.error,
        raw=store.payload_of(trace.raw) if trace.raw is not None else EMPTY_PAYLOAD,
    )


def tool_row(trace: ToolCallTrace, *, exec_id: str, store: RunStore) -> ToolTraceRow:
    return ToolTraceRow(
        exec_id=exec_id,
        seq=trace.seq,
        tool_name=trace.name,
        node_name=trace.node or "",
        node_path=codec.encode_paths([trace.node_path] if trace.node_path else []),
        args=store.payload_of(trace.arguments),
        cassette_key=trace.cassette_key or "",
        outcome=trace.outcome,
        match_tier=trace.match_tier,
        # A replay served from the LRU never touched SQLite. Worth recording,
        # because "replay is free" is a claim we make and this is the field
        # that can substantiate it.
        cache_hit=trace.outcome == "replayed",
        fault=trace.fault,
        fault_injected=trace.fault is not None,
        miss_verdict=trace.miss_verdict,
        tool_version=trace.tool_version or "",
        latency_ms=trace.latency_ms,
        error_message=trace.error,
    )


def execution_from_context(
    ctx: RunContext,
    *,
    store: RunStore,
    status: ExecStatus,
    outcome: Outcome = Outcome.UNCLASSIFIED,
    duration_ms: float = 0.0,
    started_at: str | None = None,
    attempt: int | None = None,
    cost_usd: float | None = None,
    error: BaseException | None = None,
    output: Any = None,
    state: Any = None,
) -> tuple[ExecutionRecord, list[NodeTraceRow], list[ToolTraceRow]]:
    """Turn a finished run context into the rows that describe it.

    Pure apart from the payload writes, which have to happen here because the
    inline/blob decision needs the store. The runner calls this and then
    :meth:`RunStore.save_execution <slmeval.storage.db.RunStore.save_execution>`;
    splitting the two keeps the transaction short and the mapping testable
    without a database.
    """
    exec_id = make_exec_id(ctx.run_id, ctx.candidate_id, ctx.case_id, ctx.attempt)
    prompt = sum(c.prompt_tokens or 0 for c in ctx.llm_calls)
    completion = sum(c.completion_tokens or 0 for c in ctx.llm_calls)
    seen_usage = any(
        c.prompt_tokens is not None or c.completion_tokens is not None for c in ctx.llm_calls
    )

    record = ExecutionRecord(
        exec_id=exec_id,
        run_id=ctx.run_id,
        candidate_id=ctx.candidate_id,
        case_id=ctx.case_id,
        repeat_idx=ctx.attempt,
        status=status,
        outcome_class=outcome,
        attempt=ctx.attempt if attempt is None else attempt,
        started_at=started_at or datetime.now(UTC).isoformat(),
        duration_ms=duration_ms,
        steps=ctx.steps,
        total_prompt_tokens=prompt if seen_usage else None,
        total_completion_tokens=completion if seen_usage else None,
        # One estimated call taints the total: an aggregate that mixes reported
        # and estimated usage has to be marked `~` in the report, and the only
        # way the report can know is this flag.
        tokens_estimated=any(c.tokens_estimated for c in ctx.llm_calls),
        cost_usd=cost_usd,
        fault_vector_json=encode_fault_vector(ctx.fault_vector),
        fault_digest=ctx.fault_vector.digest if ctx.fault_vector else "",
        trajectory_json=codec.encode_paths([visit.path for visit in ctx.trajectory]),
        warnings_json=codec.encode_strings(ctx.warnings),
        output=store.payload_of(output) if output is not None else EMPTY_PAYLOAD,
        state=store.payload_of(state) if state is not None else EMPTY_PAYLOAD,
        error_class=type(error).__name__ if error is not None else None,
        error_message=str(error) if error is not None else None,
    )
    nodes = [node_row(t, exec_id=exec_id, store=store) for t in ctx.llm_calls]
    tools = [tool_row(t, exec_id=exec_id, store=store) for t in ctx.tool_calls]
    return record, nodes, tools


def save_context(
    store: RunStore,
    ctx: RunContext,
    *,
    status: ExecStatus,
    evaluations: Sequence[EvaluationRow] = (),
    **kwargs: Any,
) -> ExecutionRecord:
    """Persist one finished execution. What the runner calls per work unit."""
    record, nodes, tools = execution_from_context(ctx, store=store, status=status, **kwargs)
    store.save_execution(
        record, node_traces=nodes, tool_traces=tools, evaluations=evaluations
    )
    return record


# ---------------------------------------------------------------- from rows


def as_llm_trace(row: NodeTraceRow, *, raw: Any = None) -> LLMCallTrace:
    """Rebuild a model-call trace from its row.

    ``raw`` is passed in rather than loaded here: it may be a blob, and the
    evaluators that consume these traces never look at it, so making every
    rehydration pay for a file read would be a tax on the common path.
    """
    paths = codec.decode_paths(row.node_path)
    return LLMCallTrace(
        seq=row.seq,
        model_id=row.model_id,
        provider=row.provider,
        model=row.model,
        node=row.node_name or None,
        node_path=paths[0] if paths else (),
        schema_name=row.schema_name,
        schema_status=_schema_status(row.schema_state),
        structured_requested=row.structured_requested,
        structured_mode_used=row.structured_mode,
        rungs_tried=codec.decode_strings(row.rungs_tried),
        repair_attempts=row.repair_attempts,
        json_stage=_json_stage(row.json_stage),
        prompt_tokens=row.prompt_tokens,
        completion_tokens=row.completion_tokens,
        tokens_estimated=row.tokens_estimated,
        latency_ms=row.latency_ms,
        ttft_ms=row.ttft_ms,
        retries=row.retries,
        retry_ms=row.retry_ms,
        error=row.error_message,
        error_type=row.error_class,
        raw=raw,
    )


def as_tool_trace(row: ToolTraceRow, *, arguments: Any = None) -> ToolCallTrace:
    """Rebuild a tool-call trace from its row."""
    paths = codec.decode_paths(row.node_path)
    return ToolCallTrace(
        seq=row.seq,
        name=row.tool_name,
        node=row.node_name or None,
        node_path=paths[0] if paths else (),
        arguments=arguments if isinstance(arguments, dict) else {},
        cassette_key=row.cassette_key or None,
        outcome=row.outcome,
        match_tier=row.match_tier,
        tool_version=row.tool_version or None,
        fault=row.fault,
        miss_verdict=row.miss_verdict,
        latency_ms=row.latency_ms,
        error=row.error_message,
    )


def traces_from_detail(detail: Any) -> tuple[list[LLMCallTrace], list[ToolCallTrace]]:
    """Rehydrate a reconstructed execution into trace objects.

    Takes an :class:`~slmeval.storage.models.ExecutionDetail`. This is the
    function step 12's acceptance test exercises, and the one the evaluators
    use so they can score a stored run without re-executing it - which is what
    makes ``slmeval report`` and the CI gate possible at all.
    """
    nodes = [as_llm_trace(row) for row in detail.node_traces]
    tools = [
        as_tool_trace(row, arguments=detail.tool_arguments.get(row.seq))
        for row in detail.tool_traces
    ]
    return nodes, tools


def _schema_status(value: str) -> SchemaStatus:
    try:
        return SchemaStatus(value)
    except ValueError:
        return SchemaStatus.NOT_REQUESTED


def _json_stage(value: str | None) -> JsonStage | None:
    if not value:
        return None
    try:
        return JsonStage(value)
    except ValueError:
        return None


def evaluation_rows(
    exec_id: str,
    metrics: Iterable[tuple[int, str, float | None, str | None]],
) -> list[EvaluationRow]:
    """Small convenience for the evaluators: ``(tier, metric, num, text)``."""
    return [
        EvaluationRow(exec_id=exec_id, tier=tier, metric=metric, value_num=num, value_text=text)
        for tier, metric, num, text in metrics
    ]


__all__ = [
    "as_llm_trace",
    "as_tool_trace",
    "decode_fault_vector",
    "encode_fault_vector",
    "evaluation_rows",
    "execution_from_context",
    "node_row",
    "save_context",
    "tool_row",
    "traces_from_detail",
]
