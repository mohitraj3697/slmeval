"""Declarative benchmark configuration.

This schema *is* the product's conceptual model — users read `eval.toml` to work
out what the tool does before they read the README. Two rules follow from that:

1. ``extra="forbid"`` everywhere. A typo must be an error, never a silently
   ignored key that makes someone think they configured something.
2. Secrets are referenced by environment-variable *name* (``api_key_env``),
   never by value. `eval.toml` gets committed to git and pasted into issues.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
"""Candidate ids become table headers, SQL keys and report filenames."""

ENTRYPOINT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*:[A-Za-z_][A-Za-z0-9_]*$")

Provider = Literal["openai", "anthropic", "ollama", "openai_compatible", "fake"]
ToolMode = Literal["record", "replay", "auto", "live"]
StructuredMode = Literal["auto", "tools", "json_schema", "json_mode", "prompt"]

#: Spelled out rather than generated from :class:`slmeval.sandbox.faults.FaultKind`
#: so a typo in `eval.toml` is a config error naming the valid values, and so the
#: config layer does not import the engine at module scope.
FaultKindName = Literal[
    "rate_limit_429",
    "timeout_504",
    "malformed_json",
    "empty_result",
    "partial_schema",
    "semantic_conflict",
    "latency_spike",
]
DeliveryName = Literal["raise", "return"]

DEFAULT_API_KEY_ENV: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "ollama": None,
    "openai_compatible": None,
    "fake": None,
}

DEFAULT_BASE_URL: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
}

#: vLLM has no native Windows build; it is reached through `openai_compatible`.
#: Keeping that out of the provider enum avoids promising something the
#: platform cannot deliver.
PROVIDER_NOTES: dict[str, str] = {
    "openai_compatible": (
        "covers llama.cpp server, LM Studio, vLLM, Together, Groq, OpenRouter "
        "- anything serving /v1/chat/completions"
    ),
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


class FaultsCfg(StrictModel):
    """Which faults to inject, and how they are delivered.

    Optional: the defaults in :mod:`slmeval.sandbox.faults` enable the whole
    taxonomy at its default delivery, so `fault_injection_rate` alone is enough
    to get started. This section exists because *delivery* changes what is
    being measured - a 429 raised tests the retry wrapper, the same 429
    returned as a payload tests whether the model reads its tool output at all
    - and that is a choice only the user can make for their agent.
    """

    kinds: list[FaultKindName] = Field(default_factory=list)
    """Faults to enable. Empty means all of them."""

    delivery: dict[FaultKindName, DeliveryName] = Field(default_factory=dict)
    """Per-fault override of raise vs return. Validated against what each
    fault can meaningfully be: a returned timeout and a raised `empty_result`
    are both refused, here, at config-load time."""

    weights: dict[FaultKindName, float] = Field(default_factory=dict)
    """Relative frequency per fault. `empty_result` and `semantic_conflict`
    are weighted double by default because they are where small models fail
    most interestingly and an even split puts almost none of either into a
    ten-case dataset."""

    # `Annotated[..., Field(...)] = default` rather than the `Field(default, ...)`
    # form used elsewhere in this file, and not a style preference: without the
    # pydantic mypy plugin, `Field(1.0, ge=...)` does not read as a default, so
    # this model would look un-constructible and `BenchmarkCfg.faults` below
    # could not default to one.
    delay_scale: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0
    """Multiplier on the real delays that `timeout_504` and `latency_spike`
    wait out. 1.0 is realistic; 0.0 makes a smoke run instant at the cost of
    never tripping the agent's own timeout."""

    @model_validator(mode="after")
    def _check_deliverable(self) -> FaultsCfg:
        """Refuse a delivery the fault has no meaningful form for.

        Done here rather than at the first injected fault, which could be an
        hour into a matrix run - everything needed to catch it is in this
        object.
        """
        from slmeval.sandbox.faults import fault_schedule

        try:
            fault_schedule(0, "_validate", **self.to_kwargs(rate=1.0))
        except ValueError as exc:
            raise ValueError(str(exc)) from exc
        return self

    def to_kwargs(self, *, rate: float) -> dict[str, Any]:
        """Keyword arguments for
        :func:`slmeval.sandbox.faults.fault_schedule`.

        The enum conversion lives here so the config layer can keep plain
        strings - a TOML file should never have to spell a Python enum - while
        the engine keeps its typed API.
        """
        from slmeval.sandbox.faults import Delivery, FaultKind

        return {
            "rate": rate,
            "kinds": [FaultKind(k) for k in self.kinds] or None,
            "delivery": {FaultKind(k): Delivery(v) for k, v in self.delivery.items()},
            "weights": {FaultKind(k): w for k, w in self.weights.items()},
            "delay_scale": self.delay_scale,
        }


class BenchmarkCfg(StrictModel):
    """Run-wide settings."""

    name: str = Field(min_length=1, max_length=120)
    dataset: Path
    runs_per_case: int = Field(3, ge=1, le=50)
    tool_mode: ToolMode = "replay"
    fault_injection_rate: float = Field(0.0, ge=0.0, le=1.0)
    seed: int = 42
    concurrency: int = Field(4, ge=1, le=256)
    timeout_s: float = Field(120.0, gt=0.0, le=3600.0)
    max_steps: int = Field(25, ge=1, le=1000)
    faults: FaultsCfg = Field(default_factory=FaultsCfg)


class TargetCfg(StrictModel):
    """Where the user's agent lives.

    The harness does not know or care which framework it uses; it calls one
    coroutine and supplies the model router and tool sandbox by ambient context.
    """

    entrypoint: str
    adapter: Literal["generic", "langgraph"] = "generic"

    sys_path: list[Path] = Field(default_factory=list)
    """Extra directories to import from, relative to `eval.toml`.

    Empty is right for the common layout, where the agent module sits next to
    the config. It exists for the two that do not: a `src/` layout, and a
    config that lives *inside* the package it benchmarks - as the bundled
    example does, needing `sys_path = [".."]` so `support_agent` is importable
    as a package rather than as the directory it happens to be run from.

    Spelled out in the config rather than guessed from the entrypoint, because
    the failure mode of guessing is importing a different module with the same
    name and benchmarking the wrong code.
    """

    @field_validator("entrypoint")
    @classmethod
    def _check_entrypoint(cls, v: str) -> str:
        if not ENTRYPOINT_RE.match(v):
            raise ValueError(
                f"expected 'module.path:callable_name', got {v!r} "
                "(example: 'myagent.graph:run_case')"
            )
        return v

    @property
    def module(self) -> str:
        return self.entrypoint.split(":", 1)[0]

    @property
    def attr(self) -> str:
        return self.entrypoint.split(":", 1)[1]


class ModelCfg(StrictModel):
    """One model configuration: the baseline, or a candidate to compare to it."""

    id: str
    provider: Provider
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(1024, ge=1, le=200_000)
    structured_output: StructuredMode = "auto"
    target_nodes: list[str] = Field(default_factory=list)
    """Nodes this model replaces. Empty = replaces the model everywhere."""
    extra_body: dict[str, Any] = Field(default_factory=dict)
    """Provider passthrough, e.g. {num_ctx = 8192, seed = 42} for Ollama."""

    @field_validator("id")
    @classmethod
    def _check_id(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError(
                f"{v!r} is not a valid id: use lowercase letters, digits, '.', '_' or '-', "
                "start with a letter or digit, max 64 chars"
            )
        return v

    @field_validator("base_url")
    @classmethod
    def _check_base_url(cls, v: str | None) -> str | None:
        if v is None:
            return None
        v = v.strip().rstrip("/")
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"must start with http:// or https://, got {v!r}")
        if v.endswith(("/chat/completions", "/completions")):
            raise ValueError(
                f"{v!r} points at an endpoint, not a base url - "
                "drop the '/chat/completions' suffix"
            )
        return v

    @field_validator("target_nodes")
    @classmethod
    def _check_nodes(cls, v: list[str]) -> list[str]:
        if len(set(v)) != len(v):
            dupes = sorted({n for n in v if v.count(n) > 1})
            raise ValueError(f"duplicate node names: {', '.join(dupes)}")
        return v

    @model_validator(mode="after")
    def _apply_provider_defaults(self) -> ModelCfg:
        if self.base_url is None and self.provider in DEFAULT_BASE_URL:
            self.base_url = DEFAULT_BASE_URL[self.provider]
        if self.provider == "openai_compatible" and not self.base_url:
            raise ValueError(
                "provider 'openai_compatible' requires base_url "
                "(e.g. 'http://localhost:8000/v1' for a local vLLM or llama.cpp server)"
            )
        if self.api_key_env is None:
            self.api_key_env = DEFAULT_API_KEY_ENV.get(self.provider)
        return self

    @property
    def replaces_everything(self) -> bool:
        return not self.target_nodes

    def describe(self) -> str:
        where = "all nodes" if self.replaces_everything else ", ".join(self.target_nodes)
        return f"{self.provider}:{self.model} -> {where}"


class BaselineCfg(ModelCfg):
    """The reference configuration. Its id defaults so `[baseline]` stays terse."""

    id: str = "baseline"


class JudgeCfg(ModelCfg):
    """The Tier 3 judge: which model grades, and what it must have proved.

    Optional, and off until `[judge]` names a model. Tiers 1 and 2 are free and
    carry most of the signal; a judge is for the remainder, and a judge nobody
    has calibrated is worse than no judge - it produces numbers with the same
    confidence as the measured ones.
    """

    id: str = "judge"
    provider: Provider = "openai"
    model: str = ""
    """Empty means no judge. Tier 3 simply does not run."""

    min_agreement: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7
    """Floor on the judge's measured agreement with human labels, read on the
    Wilson lower bound like everything else here.

    `slmeval judge calibrate` writes the measurement to `slmeval.judge.json`,
    which is committed for the same reason the baseline is: "we accepted a
    judge that agrees with us 71% of the time" should be a diff somebody
    approved, not a number in one laptop's cache."""

    allow_uncalibrated: bool = False
    """Run the judge with no calibration record at all. Off by default, and
    every figure derived from it is labelled UNCALIBRATED when it is on."""

    @property
    def enabled(self) -> bool:
        return bool(self.model)


class GateCfg(StrictModel):
    """Default thresholds for `slmeval gate`, so CI can call it bare.

    The whole point of committing these is that the thresholds a team agreed on
    live beside the benchmark and get reviewed with it, rather than in a CI
    YAML nobody reads. Every flag on `slmeval gate` overrides whatever is here.

    Quality limits are `None` by default - off. A threshold nobody chose is a
    threshold nobody will defend the first time it fires spuriously, and by
    then the whole gate gets disabled. The two that do have defaults are not
    quality judgements: they describe whether the run can be judged at all.
    """

    min_pass_k: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    """Floor on `pass^k`, read on the Wilson lower bound by default."""

    use_wilson_lower: bool = True
    """Gate on the lower bound, not the point estimate.

    With three repeats, 3/3 is a point estimate of 100% and a lower bound near
    44%. Reading the point estimate waves through a model that fails one
    request in four; see `slmeval.eval.stats`. Turning this off is a deliberate
    act and the gate reports which estimator it used.
    """

    max_cost_ratio: Annotated[float | None, Field(gt=0.0)] = None
    """Candidate cost per case, as a fraction of the baseline's. 0.4 = 60% off."""

    max_p95_ratio: Annotated[float | None, Field(gt=0.0)] = None
    """Only enforceable on a `--serial-latency` run; skipped, loudly, otherwise."""

    no_forbidden_tools: bool = True
    """On by default: a call to the tool that moves money is never acceptable
    at any pass rate, so noticing it is not something to opt in to."""

    max_regression: Annotated[float | None, Field(ge=0.0, le=1.0)] = None
    """Allowed drop in `pass^k` against the committed `slmeval.baseline.json`."""

    max_uncovered: Annotated[float, Field(ge=0.0, le=1.0)] = 0.2
    """Share of a candidate's executions that may take unrecorded paths before
    the run is judged uncoverable (exit 4) rather than failed (exit 1)."""

    max_provider_error_rate: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0
    """Share of executions that may fail to reach the model server before the
    job is called broken (exit 3). Zero by default - "Ollama was not running"
    must never read as a regression."""

    require_fault_parity: bool = False
    """Refuse to compare candidates that met different injected faults."""


class EvalConfig(StrictModel):
    """The whole `eval.toml`."""

    benchmark: BenchmarkCfg
    target: TargetCfg
    baseline: BaselineCfg
    candidates: list[ModelCfg] = Field(default_factory=list)
    gate: GateCfg = Field(default_factory=GateCfg)
    judge: JudgeCfg = JudgeCfg(temperature=0.0, max_tokens=512)
    """Off until `[judge]` names a model. 512 tokens is a verdict and two
    sentences of reasoning; a judge given room to write an essay writes one,
    and pays for it per execution."""

    @model_validator(mode="after")
    def _check_ids(self) -> EvalConfig:
        seen: dict[str, int] = {}
        for i, c in enumerate(self.candidates):
            if c.id in seen:
                raise ValueError(
                    f"duplicate candidate id {c.id!r} "
                    f"(candidates[{seen[c.id]}] and candidates[{i}]); "
                    "ids become report columns and must be unique"
                )
            seen[c.id] = i
        if self.baseline.id in seen:
            raise ValueError(
                f"candidate id {self.baseline.id!r} collides with the baseline id; "
                "rename the candidate or set a different [baseline] id"
            )
        return self

    @property
    def all_models(self) -> list[ModelCfg]:
        return [self.baseline, *self.candidates]

    def candidate(self, candidate_id: str) -> ModelCfg | None:
        return next((m for m in self.all_models if m.id == candidate_id), None)

    def config_hash(self) -> str:
        """Stable hash of the benchmark definition.

        Recorded with every run so a report can never silently compare results
        produced under different settings. Paths are excluded because the same
        benchmark run from two checkouts is still the same benchmark.
        """
        # `gate` is excluded with the dataset path, and for a sharper reason:
        # a gate threshold changes the *verdict* on a run, never what the run
        # measured. Two runs that differ only in their thresholds are perfectly
        # comparable, and hashing them apart would make `compare` refuse a
        # comparison that is entirely valid.
        exclude: dict[str, Any] = {"benchmark": {"dataset"}, "gate": True}
        if not self.judge.enabled:
            # A judge nobody configured measures nothing, so it must not change
            # the hash - otherwise every committed baseline in existence would
            # stop comparing the day this field was added, for a feature its
            # owner never turned on. An *enabled* judge is hashed, because a
            # pass rate that includes its verdicts is a different measurement.
            exclude["judge"] = True
        payload = self.model_dump(mode="json", exclude=exclude)
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
