"""``slmeval.baseline.json`` - the committed reference a gate compares against.

**A baseline has to be a file in the repository, not the last run in the local
database.** That is the whole design of this module and it is worth stating
plainly: gating against "whatever ran here most recently" produces different
verdicts on different machines, passes on the laptop of whoever last ran the
good model, and cannot be reviewed. A committed JSON file can be diffed in a
pull request, which makes "we accepted a two-point drop" a decision somebody
signed off rather than a number that drifted.

The file carries the metrics *and* the provenance: the config hash, the dataset
hash and the harness version that produced them. The gate checks those before
it compares anything, because a dataset edit would otherwise be published as a
regression - the single most credibility-destroying thing a measurement tool
can do.

Only what a gate reads is stored. Not the traces, not the per-execution rows:
this file lives in the repository and gets diffed by humans, so it has to stay
small enough to read.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from slmeval import __version__
from slmeval.errors import ConfigError
from slmeval.report.model import CandidateReport, Scorecard

SCHEMA_VERSION: Final = "slmeval.baseline/1"
"""Versioned like the report payload, and for the same reason: a gate in
somebody's CI reads this file, so the shape changes additively or not at all."""

BASELINE_FILENAME: Final = "slmeval.baseline.json"
"""Committed at the project root, beside ``eval.toml``."""


@dataclass(frozen=True, slots=True)
class CaseBaseline:
    """One case's consistency, as the reference recorded it."""

    n: int
    c: int

    @property
    def all_passed(self) -> bool:
        return self.n > 0 and self.c == self.n


@dataclass(frozen=True, slots=True)
class CandidateBaseline:
    """One row of the reference scorecard.

    Every figure is nullable and stays null when it was not measured. A
    baseline that recorded an unpriced model's cost as ``0`` would make every
    later candidate look infinitely more expensive, and one that recorded an
    unmeasured p95 as a number would have the gate enforcing a latency budget
    against noise.
    """

    candidate_id: str
    role: str = "candidate"
    provider: str = ""
    model: str = ""

    pass_hat_k: float | None = None
    wilson_lo: float | None = None
    wilson_hi: float | None = None
    cases: int = 0
    cases_all_pass: int = 0
    executions: int = 0

    schema_valid_rate: float | None = None
    tool_f1: float | None = None
    cost_per_case: float | None = None
    p95_ms: float | None = None
    unsafe_actions: int = 0

    per_case: dict[str, CaseBaseline] = field(default_factory=dict)
    """Kept so a gate can name *which* case regressed, not only that the rate
    fell. "refund_207 went from 3/3 to 1/3" is actionable; "pass^3 fell four
    points" sends somebody re-running the whole matrix to find out why."""

    def payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "role": self.role,
            "provider": self.provider,
            "model": self.model,
            "pass_hat_k": self.pass_hat_k,
            "wilson_lo": self.wilson_lo,
            "wilson_hi": self.wilson_hi,
            "cases": self.cases,
            "cases_all_pass": self.cases_all_pass,
            "executions": self.executions,
            "schema_valid_rate": self.schema_valid_rate,
            "tool_f1": self.tool_f1,
            "cost_per_case": self.cost_per_case,
            "p95_ms": self.p95_ms,
            "unsafe_actions": self.unsafe_actions,
            "per_case": {
                case_id: {"n": case.n, "c": case.c}
                for case_id, case in sorted(self.per_case.items())
            },
        }

    @classmethod
    def from_row(cls, row: CandidateReport, *, latency: bool) -> CandidateBaseline:
        return cls(
            candidate_id=row.candidate_id,
            role=row.role,
            provider=row.provider,
            model=row.model,
            pass_hat_k=row.pass_hat_k,
            wilson_lo=row.stats.lower,
            wilson_hi=row.stats.upper,
            cases=row.stats.cases,
            cases_all_pass=row.stats.cases_all_pass,
            executions=row.stats.executions,
            schema_valid_rate=row.schema_valid_rate,
            tool_f1=row.tool_f1,
            cost_per_case=row.cost_per_case,
            # Null unless the run was scheduled for latency. See
            # `Scorecard.latency_reportable`: a p95 from concurrent requests
            # describes our scheduler, and freezing one into a committed
            # baseline would make every later run gate against it.
            p95_ms=row.p95_ms if latency else None,
            unsafe_actions=row.unsafe_actions,
            per_case={
                case_id: CaseBaseline(n=stats.n, c=stats.c)
                for case_id, stats in sorted(row.rollups.items())
            },
        )

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> CandidateBaseline:
        cases = payload.get("per_case") or {}
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            role=str(payload.get("role", "candidate")),
            provider=str(payload.get("provider", "")),
            model=str(payload.get("model", "")),
            pass_hat_k=_number(payload.get("pass_hat_k")),
            wilson_lo=_number(payload.get("wilson_lo")),
            wilson_hi=_number(payload.get("wilson_hi")),
            cases=int(payload.get("cases") or 0),
            cases_all_pass=int(payload.get("cases_all_pass") or 0),
            executions=int(payload.get("executions") or 0),
            schema_valid_rate=_number(payload.get("schema_valid_rate")),
            tool_f1=_number(payload.get("tool_f1")),
            cost_per_case=_number(payload.get("cost_per_case")),
            p95_ms=_number(payload.get("p95_ms")),
            unsafe_actions=int(payload.get("unsafe_actions") or 0),
            per_case={
                str(case_id): CaseBaseline(
                    n=int(row.get("n") or 0), c=int(row.get("c") or 0)
                )
                for case_id, row in cases.items()
            },
        )


@dataclass(frozen=True, slots=True)
class Baseline:
    """A whole committed reference run.

    The provenance fields at the top are not metadata. ``divergences`` reads
    them, and a gate that compared across a dataset change would invent a
    regression out of an edit somebody made on purpose.
    """

    run_id: str
    name: str = ""
    saved_at: str = ""
    slmeval_version: str = __version__

    config_hash: str = ""
    dataset_hash: str = ""
    dataset_path: str = ""
    git_sha: str = ""

    mode: str = "replay"
    seed: int = 0
    runs_per_case: int = 1
    fault_rate: float = 0.0
    scheduling: str = "throughput"
    latency_reportable: bool = False

    candidates: dict[str, CandidateBaseline] = field(default_factory=dict)
    path: Path | None = None
    """Where it was loaded from, for error messages. Not serialised - a path
    baked into a committed file would be wrong on every other machine."""

    @property
    def baseline_row(self) -> CandidateBaseline | None:
        return next(
            (c for c in self.candidates.values() if c.role == "baseline"), None
        )

    def candidate(self, candidate_id: str) -> CandidateBaseline | None:
        return self.candidates.get(candidate_id)

    # ------------------------------------------------------------- build / io

    @classmethod
    def from_scorecard(
        cls, card: Scorecard, *, now: datetime | None = None
    ) -> Baseline:
        run = card.run
        return cls(
            run_id=run.run_id,
            name=run.name,
            saved_at=(now or datetime.now(UTC)).isoformat(),
            slmeval_version=run.slmeval_version,
            config_hash=run.config_hash,
            dataset_hash=run.dataset_hash,
            dataset_path=run.dataset_path,
            git_sha=run.git_sha,
            mode=run.mode,
            seed=run.seed,
            runs_per_case=run.runs_per_case,
            fault_rate=run.fault_rate,
            scheduling=run.scheduling,
            latency_reportable=card.latency_reportable,
            candidates={
                row.candidate_id: CandidateBaseline.from_row(
                    row, latency=card.latency_reportable
                )
                for row in card.rows
            },
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "run_id": self.run_id,
            "name": self.name,
            "saved_at": self.saved_at,
            "slmeval_version": self.slmeval_version,
            "config_hash": self.config_hash,
            "dataset_hash": self.dataset_hash,
            "dataset_path": self.dataset_path,
            "git_sha": self.git_sha,
            "mode": self.mode,
            "seed": self.seed,
            "runs_per_case": self.runs_per_case,
            "fault_rate": self.fault_rate,
            "scheduling": self.scheduling,
            "latency_reportable": self.latency_reportable,
            "candidates": {
                candidate_id: row.payload()
                for candidate_id, row in sorted(self.candidates.items())
            },
        }

    def dumps(self) -> str:
        """Pretty-printed with a trailing newline: this file is committed and
        diffed, and a one-line JSON blob makes every change look total."""
        return json.dumps(self.payload(), indent=2) + "\n"

    def save(self, path: Path) -> Path:
        path.write_text(self.dumps(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: Path) -> Baseline:
        """Read a committed baseline, refusing anything we cannot trust.

        Every failure here is a :class:`~slmeval.errors.ConfigError` with the
        file named, because the whole point of the file is to be editable by
        hand in a pull request - and a hand-edited file is a file that will
        occasionally be wrong.
        """
        if not path.is_file():
            raise ConfigError(
                f"no baseline file at {path}",
                hint="`slmeval baseline save` writes one from the latest run, "
                "and it is meant to be committed",
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as err:
            raise ConfigError(
                f"cannot read {path.name}: {err}",
                source=path,
                hint="re-create it with `slmeval baseline save`",
            ) from None
        if not isinstance(payload, Mapping):
            raise ConfigError(
                f"{path.name} is not a baseline object", source=path
            )
        schema = str(payload.get("schema", ""))
        if schema != SCHEMA_VERSION:
            raise ConfigError(
                f"{path.name} declares schema {schema or '(none)'}, "
                f"and this slmeval reads {SCHEMA_VERSION}",
                source=path,
                hint="re-save it with `slmeval baseline save` on this version",
            )
        rows = payload.get("candidates") or {}
        return cls(
            run_id=str(payload.get("run_id", "")),
            name=str(payload.get("name", "")),
            saved_at=str(payload.get("saved_at", "")),
            slmeval_version=str(payload.get("slmeval_version", "")),
            config_hash=str(payload.get("config_hash", "")),
            dataset_hash=str(payload.get("dataset_hash", "")),
            dataset_path=str(payload.get("dataset_path", "")),
            git_sha=str(payload.get("git_sha", "")),
            mode=str(payload.get("mode", "replay")),
            seed=int(payload.get("seed") or 0),
            runs_per_case=int(payload.get("runs_per_case") or 1),
            fault_rate=float(payload.get("fault_rate") or 0.0),
            scheduling=str(payload.get("scheduling", "throughput")),
            latency_reportable=bool(payload.get("latency_reportable")),
            candidates={
                str(candidate_id): CandidateBaseline.parse(row)
                for candidate_id, row in rows.items()
            },
            path=path,
        )


def baseline_path(root: Path, explicit: Path | None = None) -> Path:
    """Where the baseline lives: beside ``eval.toml`` unless told otherwise."""
    if explicit is not None:
        return Path(explicit).expanduser()
    return root / BASELINE_FILENAME


def _number(value: Any) -> float | None:
    """A float, or ``None`` - never a zero standing in for a missing figure."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


__all__ = [
    "BASELINE_FILENAME",
    "SCHEMA_VERSION",
    "Baseline",
    "CandidateBaseline",
    "CaseBaseline",
    "baseline_path",
]
