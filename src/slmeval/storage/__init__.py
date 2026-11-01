"""Trace storage: one SQLite file plus a content-addressed blob directory.

Traces are the asset. Everything the tool prints - the scorecard, the
drill-down diff, the CI gate - is a query over these rows, so the contract is
that a completed run is fully reconstructible from storage with every in-memory
object thrown away.

    from slmeval.storage import RunStore, new_run_id

    with RunStore.for_project(root) as store:
        store.save_run(record)
        ...
        detail = store.reconstruct(exec_id)
"""

from __future__ import annotations

from slmeval.storage.blobs import BlobStore, MissingBlob
from slmeval.storage.db import DB_NAME, RunNotFound, RunStore, make_exec_id
from slmeval.storage.migrations import SCHEMA_VERSION
from slmeval.storage.models import (
    NOT_THE_MODELS_FAULT,
    BlameRow,
    CandidateRecord,
    CaseRollupRow,
    EvaluationRow,
    ExecStatus,
    ExecutionDetail,
    ExecutionRecord,
    MetricSample,
    NodeSummaryRow,
    NodeTraceRow,
    Outcome,
    Payload,
    RunDetail,
    RunRecord,
    RunStatus,
    StoreStats,
    ToolTraceRow,
)
from slmeval.storage.provenance import Divergence, Provenance, divergences, new_run_id

__all__ = [
    "DB_NAME",
    "NOT_THE_MODELS_FAULT",
    "SCHEMA_VERSION",
    "BlameRow",
    "BlobStore",
    "CandidateRecord",
    "CaseRollupRow",
    "Divergence",
    "EvaluationRow",
    "ExecStatus",
    "ExecutionDetail",
    "ExecutionRecord",
    "MetricSample",
    "MissingBlob",
    "NodeSummaryRow",
    "NodeTraceRow",
    "Outcome",
    "Payload",
    "Provenance",
    "RunDetail",
    "RunNotFound",
    "RunRecord",
    "RunStatus",
    "RunStore",
    "StoreStats",
    "ToolTraceRow",
    "divergences",
    "make_exec_id",
    "new_run_id",
]
