"""The vocabulary the runner and the evaluators share.

Defined in ``core`` rather than in ``eval`` so the dependency points the right
way: the runner knows it must ask *something* for a verdict and persist what
comes back, and it knows nothing about trajectory distance or Wilson bounds.
That is what lets the runner be tested with a stub evaluator, and the
evaluators be tested with no runner at all.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from slmeval.core.context import RunContext
from slmeval.core.outcome import Classification
from slmeval.dataset.models import Case
from slmeval.storage import codec
from slmeval.storage.models import EvaluationRow, Outcome


@dataclass(frozen=True, slots=True)
class Metric:
    """One measurement about one execution.

    Long and narrow - a row per metric rather than a column per metric -
    because week 3 keeps adding evaluators, and a schema where every new metric
    is a migration is a schema that stops gaining metrics.
    """

    tier: int
    name: str
    value: float | None = None
    text: str | None = None
    detail: Any = None
    """Structured context, JSON-encoded on the way to the row. The *why* behind
    the number: the edit script behind a trajectory distance, the arguments
    behind a forbidden call. A metric nobody can explain is a metric nobody
    acts on."""

    def row(self, exec_id: str) -> EvaluationRow:
        return EvaluationRow(
            exec_id=exec_id,
            tier=self.tier,
            metric=self.name,
            value_num=self.value,
            value_text=self.text,
            detail_json=codec.dumps(self.detail) if self.detail is not None else None,
        )


@dataclass(frozen=True, slots=True)
class Verdict:
    """What the evaluators concluded about one execution."""

    outcome: Outcome
    metrics: tuple[Metric, ...] = ()
    warnings: tuple[str, ...] = ()
    """Things the user has to see even though the execution may have passed -
    a degraded cassette match, a schema that only validated after repair. A
    passing run that leaned on our leniency must not read as clean."""

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASSED

    def rows(self, exec_id: str) -> list[EvaluationRow]:
        return [m.row(exec_id) for m in self.metrics]

    def value(self, name: str) -> float | None:
        return next((m.value for m in self.metrics if m.name == name), None)


@dataclass(frozen=True, slots=True)
class NullVerdict:
    """Placeholder for a run with evaluation switched off."""

    outcome: Outcome = Outcome.UNCLASSIFIED
    metrics: tuple[Metric, ...] = field(default=())
    warnings: tuple[str, ...] = field(default=())


class Evaluator(Protocol):
    """Scores one finished execution.

    Called with the execution's own context, so it sees the real trace objects
    rather than rows - but it is never *given* the store, because an evaluator
    that could write would be able to disagree with the row the runner saved.
    """

    def __call__(
        self,
        *,
        ctx: RunContext,
        case: Case,
        state: dict[str, Any] | None,
        classification: Classification,
        reference: Sequence[Any] = (),
    ) -> Verdict: ...


def carry_classification(
    *,
    ctx: RunContext,
    case: Case,
    state: dict[str, Any] | None,
    classification: Classification,
    reference: Sequence[Any] = (),
) -> Verdict:
    """The no-op evaluator: keep the runner's own classification, add nothing.

    Used by tests and by ``--no-eval``. It deliberately does not mark a clean
    execution as passed: "the agent returned" is not "the agent was right", and
    a runner that could conclude the second on its own would make the
    evaluators optional.
    """
    return Verdict(outcome=classification.outcome)


__all__ = [
    "Evaluator",
    "Metric",
    "NullVerdict",
    "Verdict",
    "carry_classification",
]
