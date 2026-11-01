"""Row types for the trace store.

Traces are the asset. Reports, diffs, regressions and the eventual hosted
product are all queries over these rows, so the shape here outlives any one
command and is worth more care than the code that writes it.

These are frozen dataclasses rather than Pydantic models for the same reason
``core.trace`` is: a 90-execution matrix over a 10-step agent writes tens of
thousands of them, and validating every field assignment on the way to SQLite
would be pure overhead. Validation belongs at the boundaries - config coming
in, report payloads going out.

Every enum stored in a column is a :class:`~enum.StrEnum`, so the database
holds a readable word rather than an integer whose meaning lives only in a
Python file. ``sqlite3 runs.sqlite "SELECT outcome_class, COUNT(*) ..."`` has
to be useful to a user debugging their own benchmark; that is most of why one
SQLite file was the right choice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Final

INLINE_LIMIT: Final = 1024
"""Payloads at or below this many bytes stay in the row.

Content addressing pays for itself on tool outputs - the same recorded payload
is replayed by every candidate, so 1,500 executions share one file - but a
20-byte payload in its own file costs a filesystem cluster plus an inode, and a
benchmark makes tens of thousands of them. Small payloads inline, large ones
shared; readers cannot tell the difference.
"""


class RunStatus(StrEnum):
    """Lifecycle of a whole run."""

    RUNNING = "running"
    """Written before the first execution. A run left in this state is a crash
    or a Ctrl-C, which is exactly what ``--resume`` looks for."""

    COMPLETED = "completed"

    FAILED = "failed"
    """The harness gave up - not a quality verdict. A run whose candidates all
    scored zero is ``completed``."""

    CANCELLED = "cancelled"


class ExecStatus(StrEnum):
    """What mechanically happened to one execution.

    Deliberately separate from :class:`Outcome`: this says whether the agent
    returned, and that says whether what it returned was any good. Collapsing
    the two is how a harness ends up reporting its own timeout as a model
    failure.
    """

    OK = "ok"
    """The target returned a result. It may still have failed every assertion."""

    ERROR = "error"
    """The target raised. ``error_class`` says what."""

    TIMEOUT = "timeout"

    SKIPPED = "skipped"
    """Planned but never run - a unit left over when a run was interrupted.
    Recorded rather than absent, so the shape of the matrix survives a crash."""


class Outcome(StrEnum):
    """The classified verdict for one execution.

    Filled by the evaluators in step 11, not by the runner. Day 8 only owns the
    column and the vocabulary - but the vocabulary has to be settled now,
    because the scorecard warnings are counts over these values and a rename
    later is a migration.
    """

    UNCLASSIFIED = "unclassified"
    """Stored but not yet judged. Never aggregated into a pass rate; a metric
    that silently counts unjudged executions as failures is worse than a gap."""

    PASSED = "passed"

    FAILED_ASSERTION = "failed_assertion"
    """Ran cleanly, produced the wrong answer or state. The ordinary failure."""

    SCHEMA_FAILURE = "schema_failure"
    """Could not produce output matching the requested schema. The headline SLM
    failure mode, and the reason it is not folded into ``failed_assertion``."""

    FORBIDDEN_TOOL = "forbidden_tool"
    """Called a tool the case forbids. A hard fail that must never be averaged
    away - the scorecard prints it above the table."""

    CASSETTE_MISS = "cassette_miss"
    """Asked for a tool call we never recorded, and the miss was diagnosed as
    the model's doing: a hallucinated tool or parameter."""

    UNCOVERED = "uncovered"
    """Also a miss, but diagnosed as a valid path we simply never recorded.
    Excluded from pass rates rather than counted against the candidate: marking
    a model down for exploring a route the baseline never took would be our own
    gap, reported as its failure."""

    TOOL_ERROR = "tool_error"
    """A tool raised - usually an injected fault the agent never recovered
    from, which is the numerator of the recovery rate."""

    PROVIDER_ERROR = "provider_error"
    """The model server never produced a response: down, unreachable, key
    rejected, model not pulled. Separate from every quality outcome, because
    "Ollama was not running" must never be reported as a regression."""

    STEP_BUDGET = "step_budget"
    """Hit ``max_steps``. Usually a loop."""

    TIMEOUT = "timeout"

    CRASHED = "crashed"
    """The target raised something unrelated to the model - a bug in the agent
    under test. The agent's own code is the user's to fix, so this counts
    against the run but is reported as the agent failing, not the model."""

    HARNESS_ERROR = "harness_error"
    """*We* broke. Counted separately and never attributed to the model or the
    agent, because conflating our own bugs with model failures is exactly how a
    benchmark loses the right to be believed. A scorecard prints these above
    the table as a caveat on the whole run and leaves them out of every rate."""


TERMINAL_STATUSES: Final = frozenset({ExecStatus.OK, ExecStatus.ERROR, ExecStatus.TIMEOUT})
"""Statuses meaning "this unit does not need running again".

``--resume`` re-runs anything outside this set, which is why ``SKIPPED`` is not
in it: a unit the previous run never reached is work, not a result.
"""


NOT_THE_MODELS_FAULT: Final = frozenset({Outcome.HARNESS_ERROR, Outcome.UNCOVERED})
"""Outcomes excluded from a candidate's pass rate entirely.

Not scored as failures and not scored as passes - dropped from the denominator.
``harness_error`` is our bug and ``uncovered`` is our missing recording; either
one counted against a candidate would be a regression the tool invented. Both
are surfaced as warnings above the table instead, where they belong.
"""


@dataclass(frozen=True, slots=True)
class Payload:
    """A JSON payload living either in the row or in the blob store.

    Exactly one side is set, or neither for "nothing recorded". Readers go
    through ``RunStore.load`` and never need to know which, so the inline/blob
    threshold can change between versions without touching a query.
    """

    inline: str | None = None
    digest: str | None = None

    @property
    def present(self) -> bool:
        return self.inline is not None or self.digest is not None


EMPTY_PAYLOAD: Final = Payload()


@dataclass(frozen=True, slots=True)
class RunRecord:
    """One invocation of ``slmeval run``, with its provenance.

    The provenance fields are not bookkeeping. Without ``config_hash`` and
    ``dataset_hash``, comparing two runs is guesswork dressed up as a table;
    with them, ``slmeval compare`` can *refuse* to compare runs built on
    different datasets unless forced. That refusal is a feature.
    """

    run_id: str
    name: str
    started_at: str
    config_json: str
    config_hash: str
    dataset_hash: str
    slmeval_version: str

    mode: str
    """``replay``, ``record`` or ``live`` - the tool mode the matrix ran under.
    A live run and a replay run of the same config do not have comparable
    latency numbers, and this is how a report knows not to put them in one
    column."""

    seed: int

    scheduling: str = "throughput"
    """``throughput`` or ``latency``. Throughput runs high concurrency and its
    latency numbers are not reportable: requests queue inside the provider, and
    a local model measured that way reads several times slower than it is.
    Stored so the report refuses to print a p95 column rather than printing a
    misleading one."""

    status: RunStatus = RunStatus.RUNNING
    finished_at: str | None = None
    dataset_path: str = ""
    git_sha: str = ""
    host_info: str = "{}"
    fault_rate: float = 0.0
    runs_per_case: int = 1
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateRecord:
    """One model configuration under test in a run.

    ``model_cfg_json`` is the whole resolved config, not a pointer at the file:
    ``eval.toml`` will have been edited by the time anyone reads this row, and a
    report that re-read the file would quietly describe a run that never
    happened.
    """

    run_id: str
    candidate_id: str
    model_cfg_json: str

    role: str = "candidate"
    """``baseline`` or ``candidate``. The scorecard computes its delta column
    against the baseline row, so which one that is has to survive the run."""

    ordinal: int = 0
    """Declaration order, so table row order is stable across reports rather
    than whatever the query planner happens to return today."""

    provider: str = ""
    model: str = ""


@dataclass(frozen=True, slots=True)
class ExecutionRecord:
    """One (candidate, case, repeat) cell of the matrix."""

    exec_id: str
    run_id: str
    candidate_id: str
    case_id: str
    repeat_idx: int
    status: ExecStatus
    outcome_class: Outcome = Outcome.UNCLASSIFIED

    attempt: int = 0
    """Incremented when crash isolation re-runs a unit. The retry replaces the
    row, but the count stays: "passed on the second attempt" is a finding about
    the candidate, not an implementation detail of the runner."""

    started_at: str = ""
    duration_ms: float = 0.0
    steps: int = 0
    total_prompt_tokens: int | None = None
    total_completion_tokens: int | None = None

    tokens_estimated: bool = False
    """True when any call here had its usage estimated rather than reported.
    Any aggregate containing one must be marked ``~``, so the flag rides on the
    execution and not only on the individual call."""

    cost_usd: float | None = None
    fault_vector_json: str = "[]"

    fault_digest: str = ""
    """Digest of the faults this execution actually met. Two candidates with
    different digests were not tested under the same conditions, and
    ``--require-fault-parity`` drops the pair rather than print both numbers as
    if they were."""

    trajectory_json: str = "[]"
    """Ordered node visits. Stored even though ``node_traces`` implies most of
    it, because a node can be entered without making a model call and the
    trajectory-distance metric counts the visit, not the call."""

    warnings_json: str = "[]"
    output: Payload = EMPTY_PAYLOAD
    state: Payload = EMPTY_PAYLOAD
    error_class: str | None = None
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class NodeTraceRow:
    """One model call, as stored. Mirrors
    :class:`~slmeval.core.trace.LLMCallTrace`."""

    exec_id: str
    seq: int
    node_name: str = ""
    node_path: str = "[]"

    model_id: str = ""
    """The candidate id, duplicated from the execution so per-node attribution
    is one indexed query instead of a join across the whole matrix."""

    provider: str = ""
    model: str = ""
    latency_ms: float = 0.0
    ttft_ms: float | None = None
    retries: int = 0
    retry_ms: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_estimated: bool = False
    schema_name: str | None = None
    schema_state: str = "not_requested"
    structured_requested: str = "auto"

    structured_mode: str | None = None
    """Which ladder rung actually worked. The most valuable field in this
    table - "needed json_mode on 100% of calls" is a line in the scorecard."""

    rungs_tried: str = "[]"
    repair_attempts: int = 0
    json_stage: str | None = None
    error_class: str | None = None
    error_message: str | None = None

    raw: Payload = EMPTY_PAYLOAD
    """The provider's raw response, kept for failing calls only. The failure
    report needs it; keeping it for every call would balloon the store."""


@dataclass(frozen=True, slots=True)
class ToolTraceRow:
    """One tool invocation, as stored. Mirrors
    :class:`~slmeval.core.trace.ToolCallTrace`."""

    exec_id: str
    seq: int
    tool_name: str
    node_name: str = ""
    node_path: str = "[]"
    args: Payload = EMPTY_PAYLOAD
    cassette_key: str = ""
    outcome: str = "live"
    match_tier: str | None = None
    cache_hit: bool = False
    fault: str | None = None
    fault_injected: bool = False
    miss_verdict: str | None = None
    tool_version: str = ""
    latency_ms: float = 0.0
    output: Payload = EMPTY_PAYLOAD
    error_message: str | None = None


@dataclass(frozen=True, slots=True)
class EvaluationRow:
    """One metric for one execution.

    Long and narrow rather than a column per metric, because week 3 keeps
    adding evaluators and a schema where every new metric is a migration is a
    schema that stops gaining metrics.
    """

    exec_id: str
    tier: int
    metric: str
    value_num: float | None = None
    value_text: str | None = None
    detail_json: str | None = None


@dataclass(frozen=True, slots=True)
class CaseRollupRow:
    """Consistency math for one (candidate, case) across its repeats.

    Derived from ``executions``, and stored anyway: ``pass^k`` and the Wilson
    interval are the headline numbers, a report must not recompute them and
    risk disagreeing with the run that printed them, and a later change to the
    estimator must not silently restate last month's results.
    """

    run_id: str
    candidate_id: str
    case_id: str

    n: int
    """Repeats that produced a judgeable result. Always printed - a pass rate
    without its n is a number pretending to be evidence."""

    c: int
    """How many of them passed."""

    pass_k: float | None = None
    pass_hat_k: float | None = None
    wilson_lo: float | None = None
    wilson_hi: float | None = None


@dataclass(frozen=True, slots=True)
class MetricSample:
    """One stored metric, with the cell it belongs to.

    The join the report needs and the drill-down does not: a scorecard
    aggregates thousands of metrics across a run, and doing it by reconstructing
    each execution in turn would make rendering a table slower than producing
    one.
    """

    candidate_id: str
    case_id: str
    repeat_idx: int
    exec_id: str
    tier: int
    metric: str
    value_num: float | None = None
    value_text: str | None = None
    outcome: Outcome = Outcome.UNCLASSIFIED


@dataclass(frozen=True, slots=True)
class NodeSummaryRow:
    """Model calls at one node, for one candidate, aggregated.

    Per-node attribution is the table that turns "this candidate is 13 points
    worse" into "this candidate breaks in ``policy_check``", which is the
    difference between a benchmark and a recommendation.
    """

    candidate_id: str
    node_name: str
    provider: str = ""
    model: str = ""
    calls: int = 0
    errors: int = 0
    schema_valid: int = 0
    schema_invalid: int = 0
    repair_attempts: int = 0
    latency_ms: float = 0.0
    """Total across calls. The mean is the report's to compute, so a caller
    that wants a different statistic is not stuck with ours."""
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def mean_latency_ms(self) -> float | None:
        return self.latency_ms / self.calls if self.calls else None

    @property
    def schema_valid_rate(self) -> float | None:
        judged = self.schema_valid + self.schema_invalid
        return self.schema_valid / judged if judged else None


@dataclass(frozen=True, slots=True)
class BlameRow:
    """Where a candidate's failing executions went wrong, counted by node."""

    candidate_id: str
    node_name: str
    kind: str
    """``model`` or ``tool`` - which half of the step failed."""
    error_class: str = ""
    count: int = 0


@dataclass(frozen=True, slots=True)
class ExecutionDetail:
    """One execution and everything recorded under it, payloads resolved.

    What ``slmeval show exec <id>`` renders, and the proof of step 12's
    acceptance test: built from rows alone, so dropping every in-memory object
    changes nothing about what gets printed.
    """

    execution: ExecutionRecord
    node_traces: tuple[NodeTraceRow, ...] = ()
    tool_traces: tuple[ToolTraceRow, ...] = ()
    evaluations: tuple[EvaluationRow, ...] = ()
    output: Any = None
    state: Any = None

    tool_outputs: dict[int, Any] = field(default_factory=dict)
    """Resolved tool return values, keyed by ``seq``. Loaded eagerly, because a
    timeline that re-opened the blob store per row would make the drill-down
    the slowest command in the tool."""

    tool_arguments: dict[int, Any] = field(default_factory=dict)

    @property
    def trajectory(self) -> tuple[tuple[str, ...], ...]:
        from slmeval.storage.codec import decode_paths

        return decode_paths(self.execution.trajectory_json)


@dataclass(frozen=True, slots=True)
class RunDetail:
    """A whole run, reconstructed. ``slmeval report`` reads exactly this."""

    run: RunRecord
    candidates: tuple[CandidateRecord, ...] = ()
    executions: tuple[ExecutionRecord, ...] = ()
    rollups: tuple[CaseRollupRow, ...] = ()

    @property
    def baseline(self) -> CandidateRecord | None:
        return next((c for c in self.candidates if c.role == "baseline"), None)


@dataclass(frozen=True, slots=True)
class StoreStats:
    """What ``slmeval cache stats`` prints."""

    runs: int
    running: int
    executions: int
    node_traces: int
    tool_traces: int
    evaluations: int
    db_bytes: int
    blobs: int
    blob_bytes: int
    oldest_run: str | None = None
    newest_run: str | None = None


__all__ = [
    "EMPTY_PAYLOAD",
    "INLINE_LIMIT",
    "NOT_THE_MODELS_FAULT",
    "TERMINAL_STATUSES",
    "BlameRow",
    "CandidateRecord",
    "CaseRollupRow",
    "EvaluationRow",
    "ExecStatus",
    "ExecutionDetail",
    "ExecutionRecord",
    "MetricSample",
    "NodeSummaryRow",
    "NodeTraceRow",
    "Outcome",
    "Payload",
    "RunDetail",
    "RunRecord",
    "RunStatus",
    "StoreStats",
    "ToolTraceRow",
]
