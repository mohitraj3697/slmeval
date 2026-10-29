"""Telemetry records.

These are plain dataclasses, not Pydantic models, and deliberately so: the run
context appends thousands of them per benchmark and validating every field
assignment would be pure overhead. Validation belongs at boundaries - config
coming in, report payloads going out - not in the hot path.

``slots=True`` throughout, because a 90-execution matrix over a 10-step agent is
tens of thousands of these objects in memory at once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class SchemaStatus(StrEnum):
    """Did the model produce output matching the requested schema?

    Tri-state plus an error state, because the blueprint's boolean conflated
    "the model emitted bad JSON" with "the HTTP call blew up". Mixing those
    dilutes the schema-adherence rate with unrelated crashes, and that rate is
    the headline SLM metric - it has to mean one thing.
    """

    NOT_REQUESTED = "not_requested"
    """No schema was asked for. Excluded from adherence rates entirely."""

    VALID = "valid"
    INVALID = "invalid"
    """A response came back and failed validation. Counts against the model."""

    ERROR = "error"
    """The call never produced a candidate response (timeout, 500, refusal).
    Counts as a failure, but never against schema adherence."""


class JsonStage(StrEnum):
    """Which rung of the JSON extraction cascade produced the object.

    Anything past ``DIRECT`` is a leniency event: the model did not emit clean
    JSON and our parser covered for it. A model that only "passes" because the
    parser is forgiving is a finding, not a pass, so the stage is recorded and
    reported rather than quietly discarded.
    """

    DIRECT = "direct"
    FENCED = "fenced"
    BALANCED = "balanced"
    REPAIRED = "repaired"


LENIENT_STAGES = frozenset({JsonStage.FENCED, JsonStage.BALANCED, JsonStage.REPAIRED})


@dataclass(slots=True)
class LLMCallTrace:
    """One model call, as the scorecard needs to see it."""

    seq: int
    model_id: str
    """The candidate id from eval.toml, so per-candidate aggregation is trivial."""
    provider: str
    model: str
    node: str | None = None
    node_path: tuple[str, ...] = ()
    """Full nesting path. Attribution for a call inside `router > classify` has
    to survive nesting or per-node blame is wrong."""

    schema_name: str | None = None
    schema_status: SchemaStatus = SchemaStatus.NOT_REQUESTED
    structured_requested: str = "auto"
    structured_mode_used: str | None = None
    """Which ladder rung actually worked. The single most valuable field here."""
    rungs_tried: tuple[str, ...] = ()
    repair_attempts: int = 0
    json_stage: JsonStage | None = None

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    tokens_estimated: bool = False
    """True when the provider returned no usage block and we estimated. Any
    aggregate containing one of these must be marked `~` in reports."""

    latency_ms: float = 0.0
    """Final attempt only. Retry time is in `retry_ms` so a slow-but-eventually
    -fine candidate cannot hide behind a healthy-looking p95."""
    ttft_ms: float | None = None
    retries: int = 0
    retry_ms: float = 0.0

    error: str | None = None
    error_type: str | None = None
    raw: dict[str, Any] | None = None
    """Kept only for failing calls - the failure report needs it, and keeping it
    for every call would balloon the trace store."""

    @property
    def schema_lenient(self) -> bool:
        return self.json_stage is not None and self.json_stage in LENIENT_STAGES

    @property
    def total_tokens(self) -> int | None:
        if self.prompt_tokens is None and self.completion_tokens is None:
            return None
        return (self.prompt_tokens or 0) + (self.completion_tokens or 0)


@dataclass(slots=True)
class ToolCallTrace:
    """One tool invocation. Populated by the sandbox in Step 8."""

    seq: int
    name: str
    node: str | None = None
    node_path: tuple[str, ...] = ()
    arguments: dict[str, Any] = field(default_factory=dict)
    cassette_key: str | None = None
    outcome: str = "live"
    """live | replayed | recorded | miss | fault"""
    match_tier: str | None = None
    """Which rung of the cassette cascade answered: exact | normalized | subset.
    Anything but `exact` is a Tier-1 finding - the model reached the recorded
    call, but not with the recorded arguments - so it is reported rather than
    kept as debug detail."""
    tool_version: str | None = None
    """The tool's version at call time. Compared against the cassette's to flag
    a replay of a payload recorded for code that has since changed."""
    fault: str | None = None
    """`kind:delivery` of the injected fault, e.g. `empty_result:return`. Set
    on every faulted call, whether the fault was raised or returned, so "was
    this call faulted" is one field and not an inference from the outcome."""
    miss_verdict: str | None = None
    """For `outcome == "miss"`: how the miss was classified -
    `hallucinated_parameter`, `hallucinated_tool`, `wrong_tool_for_step` or
    `unrecorded_valid_path`. On the trace rather than only in the diagnosis
    block because the gate has to exclude uncovered misses from a candidate's
    pass rate, and it reads traces."""
    latency_ms: float = 0.0
    error: str | None = None


@dataclass(slots=True)
class NodeVisit:
    """An entry into a node, in order. This is the trajectory."""

    path: tuple[str, ...]
    seq: int

    @property
    def name(self) -> str:
        return self.path[-1] if self.path else ""
