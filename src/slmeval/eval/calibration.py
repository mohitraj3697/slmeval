"""Measuring what the judge is worth.

A judge produces a number. The only thing that makes that number evidence is a
measured agreement with human labels, which is why this module exists before
the judge is wired into a single run.

Three decisions worth defending.

**Raw agreement is reported with Cohen's kappa beside it, never alone.** On a
set where 90% of the pairs pass, a judge that says "pass" to everything scores
90% agreement and knows nothing. Kappa corrects for exactly that - it asks how
much better than chance the judge did, given how lopsided the labels are - and
it is the figure that catches a judge whose high agreement comes from the base
rate. Both are printed; a reader who sees 0.92 agreement next to κ = 0.05 can
tell what they are looking at.

**Agreement is reported with its interval.** Thirty labels is a small sample,
and a point estimate from a small sample is the mistake the rest of this tool
exists to avoid. The same Wilson bound used for ``pass^k`` is used here, for
the same reason.

**A disagreement is listed, not just counted.** The deliverable of a
calibration run is not a score, it is the handful of pairs where the judge and
the human parted company - that is what tells you whether the rubric is
ambiguous, the judge is lenient, or the human was wrong.

The labels are the user's own work. slmeval ships an **illustrative** set in
``examples/judge/`` so the harness is runnable and testable offline, and that
file is marked as an example in every record; a calibration computed from it
reports itself as not a calibration. Shipping fabricated labels as if they were
human ones would be the single most dishonest thing this tool could do - the
whole argument for the judge is that somebody checked it.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field, ValidationError, field_validator

from slmeval.config.models import StrictModel
from slmeval.errors import ConfigError
from slmeval.eval.stats import Interval, wilson_interval
from slmeval.eval.tier3 import Judgement, Verdict

MIN_LABELS = 20
"""Below this, a calibration figure is noise wearing a percentage sign.

Not a hard refusal - a user exploring with five labels should see something -
but the report says the number is not usable, and the gate treats an
under-sampled calibration as absent.
"""

EXAMPLE_SOURCE = "example"
"""The marker on the labels slmeval ships. See the module docstring."""


class Label(StrictModel):
    """One human-labelled pair.

    JSONL, one object per line, the same shape as a golden case - because the
    person labelling these is the person who wrote the dataset, and asking them
    to learn a second format is asking them not to do it.
    """

    case_id: str
    rubric: str = Field(min_length=1)
    reference: str = Field(min_length=1)
    candidate: str = Field(min_length=1)
    verdict: Verdict
    """What a human decided. The ground truth this measures against."""
    question: str = ""
    note: str = ""
    source: str = "human"
    """``human`` for a real label. Anything else - and in particular
    ``example`` - makes the resulting figure report itself as not a
    calibration."""

    @field_validator("verdict", mode="before")
    @classmethod
    def _normalize(cls, value: Any) -> Any:
        if isinstance(value, bool):
            return "pass" if value else "fail"
        if isinstance(value, str):
            return value.strip().lower()
        return value


@dataclass(frozen=True, slots=True)
class Disagreement:
    """One pair the judge and the human read differently."""

    case_id: str
    human: str
    judge: str
    score: float
    reasoning: str

    def describe(self) -> str:
        return f"{self.case_id}: human {self.human}, judge {self.judge} ({self.score:.2f})"


@dataclass(frozen=True, slots=True)
class Agreement:
    """What a calibration run concluded."""

    n: int
    agreed: int
    judge_model: str
    interval: Interval | None = None
    kappa: float | None = None
    """Cohen's kappa. ``None`` when it is undefined - which happens when one
    side gave the same verdict to everything, and is itself the finding."""
    confusion: dict[tuple[str, str], int] = None  # type: ignore[assignment]
    disagreements: tuple[Disagreement, ...] = ()
    unclear: int = 0
    """Judgements of ``unclear``. Counted as disagreement when the human was
    decisive, and reported separately: a judge that abstains on a third of the
    set is a different problem from one that is wrong on a third."""
    sources: tuple[str, ...] = ()

    @property
    def rate(self) -> float | None:
        return self.agreed / self.n if self.n else None

    @property
    def lower(self) -> float | None:
        """The number to trust, as everywhere else in this tool."""
        return self.interval.lo if self.interval else None

    @property
    def usable(self) -> bool:
        """Whether this is a calibration at all.

        Two ways for it not to be: too few labels to say anything, or labels
        that were never human. Both have to be loud, because the alternative is
        a confident number with nothing behind it.
        """
        return self.n >= MIN_LABELS and not self.illustrative

    @property
    def illustrative(self) -> bool:
        return bool(self.sources) and set(self.sources) != {"human"}

    def warnings(self) -> tuple[str, ...]:
        out: list[str] = []
        if self.illustrative:
            out.append(
                "these labels are not human labels - "
                f"{', '.join(sorted(set(self.sources) - {'human'}))} - so this "
                "is a smoke test of the harness, not a calibration of the judge"
            )
        if self.n < MIN_LABELS:
            out.append(
                f"{self.n} label(s) cannot calibrate anything; {MIN_LABELS} is "
                "the floor, and the interval below shows why"
            )
        if self.kappa is None:
            out.append(
                "kappa is undefined because one side gave the same verdict to "
                "every pair - which is itself the finding"
            )
        elif self.kappa < 0.4:
            out.append(
                f"kappa {self.kappa:.2f}: barely better than chance given how "
                "lopsided these labels are, whatever the raw agreement says"
            )
        if self.unclear:
            out.append(
                f"the judge answered 'unclear' on {self.unclear} of {self.n} "
                "pair(s) - abstention, not error, and a different problem"
            )
        return tuple(out)

    def payload(self) -> dict[str, Any]:
        """The committed record, as ``slmeval.judge.json``."""
        return {
            "schema": "slmeval.calibration/1",
            "judge_model": self.judge_model,
            "n": self.n,
            "agreed": self.agreed,
            "agreement": self.rate,
            "agreement_lower": self.lower,
            "kappa": self.kappa,
            "unclear": self.unclear,
            "usable": self.usable,
            "illustrative": self.illustrative,
            "confusion": {f"{h}->{j}": c for (h, j), c in sorted(self.confusion.items())},
            "disagreements": [
                {
                    "case_id": d.case_id,
                    "human": d.human,
                    "judge": d.judge,
                    "score": d.score,
                    "reasoning": d.reasoning,
                }
                for d in self.disagreements
            ],
            "warnings": list(self.warnings()),
        }


def load_labels(path: Path | str) -> list[Label]:
    """Read a JSONL label file, refusing a bad line by number.

    By number, because a 30-line file with one typo is otherwise a validation
    error with no location, and the first thing the user does is re-read all
    thirty lines.
    """
    location = Path(path)
    if not location.exists():
        raise ConfigError(
            f"no label file at {location}",
            hint=(
                "Write one line of JSON per pair: case_id, rubric, reference, "
                "candidate, verdict. See examples/judge/labels.example.jsonl."
            ),
        )
    labels: list[Label] = []
    for number, line in enumerate(location.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("//"):
            continue
        try:
            labels.append(Label.model_validate(json.loads(line)))
        except json.JSONDecodeError as exc:
            raise ConfigError(
                f"{location.name}:{number} is not valid JSON: {exc.msg}",
                hint="one JSON object per line, no trailing commas",
            ) from exc
        except ValidationError as exc:
            first = exc.errors()[0]
            field = ".".join(str(p) for p in first["loc"]) or "(root)"
            raise ConfigError(
                f"{location.name}:{number}: {field} {first['msg']}",
                hint="verdict must be pass, fail or unclear",
            ) from exc
    if not labels:
        raise ConfigError(
            f"{location.name} holds no labels",
            hint="one JSON object per line; blank lines and // comments are skipped",
        )
    return labels


def kappa_of(confusion: dict[tuple[str, str], int], *, n: int) -> float | None:
    """Cohen's kappa over the confusion table.

    ``None`` when the expected agreement is 1.0, which is what happens when one
    side gave the same verdict to every pair. Returning 0.0 there would claim
    "no better than chance" about a situation where chance is undefined.
    """
    if not n:
        return None
    observed = sum(count for (human, judge), count in confusion.items() if human == judge) / n
    human_totals: dict[str, int] = {}
    judge_totals: dict[str, int] = {}
    for (human, judge), count in confusion.items():
        human_totals[human] = human_totals.get(human, 0) + count
        judge_totals[judge] = judge_totals.get(judge, 0) + count
    expected = sum(
        (human_totals.get(label, 0) / n) * (judge_totals.get(label, 0) / n)
        for label in set(human_totals) | set(judge_totals)
    )
    if expected >= 1.0:
        return None
    return (observed - expected) / (1.0 - expected)


def agreement(
    pairs: Iterable[tuple[Label, Judgement]], *, judge_model: str
) -> Agreement:
    """Compare a judge's verdicts against the human labels."""
    confusion: dict[tuple[str, str], int] = {}
    disagreements: list[Disagreement] = []
    sources: list[str] = []
    agreed = unclear = n = 0

    for label, judgement in pairs:
        n += 1
        sources.append(label.source)
        key = (label.verdict, judgement.verdict)
        confusion[key] = confusion.get(key, 0) + 1
        if judgement.verdict == "unclear" and label.verdict != "unclear":
            unclear += 1
        if label.verdict == judgement.verdict:
            agreed += 1
            continue
        disagreements.append(
            Disagreement(
                case_id=label.case_id,
                human=label.verdict,
                judge=judgement.verdict,
                score=judgement.score,
                reasoning=judgement.reasoning,
            )
        )

    return Agreement(
        n=n,
        agreed=agreed,
        judge_model=judge_model,
        interval=wilson_interval(n, agreed) if n else None,
        kappa=kappa_of(confusion, n=n),
        confusion=confusion,
        disagreements=tuple(disagreements),
        unclear=unclear,
        sources=tuple(sources),
    )


def stored_agreement(path: Path | str) -> float | None:
    """The committed calibration's lower bound, or ``None``.

    The lower bound and not the point estimate, and only when the record calls
    itself usable: a judge is allowed to run on measured agreement, and an
    illustrative or under-sampled record is not a measurement.
    """
    location = Path(path)
    if not location.exists():
        return None
    try:
        payload = json.loads(location.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict) or not payload.get("usable"):
        return None
    value = payload.get("agreement_lower")
    return float(value) if isinstance(value, int | float) else None


def calibration_path(root: Path, explicit: Path | None = None) -> Path:
    """``slmeval.judge.json`` beside eval.toml, unless told otherwise.

    Committed, like ``slmeval.baseline.json`` and for the same reason: "we
    accepted a judge that agrees with us 71% of the time" should be a diff
    somebody approved, not a number in one laptop's cache.
    """
    return explicit if explicit is not None else root / "slmeval.judge.json"


def verdict_counts(labels: Sequence[Label]) -> dict[str, int]:
    out: dict[str, int] = {}
    for label in labels:
        out[label.verdict] = out.get(label.verdict, 0) + 1
    return out


__all__ = [
    "EXAMPLE_SOURCE",
    "MIN_LABELS",
    "Agreement",
    "Disagreement",
    "Label",
    "agreement",
    "calibration_path",
    "kappa_of",
    "load_labels",
    "stored_agreement",
    "verdict_counts",
]
