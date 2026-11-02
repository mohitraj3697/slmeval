"""Tier 3: the LLM judge, and the rules that keep it from being a vibe.

A judge is the only evaluator here that costs money and the only one that can
be wrong in a way that sounds confident. So the whole of this module is
guardrails, and each one exists because the obvious implementation is
misleading:

**It only judges what nothing cheaper could.** Tier 1 and Tier 2 run first, and
a judge call is made only when they both passed *and* the case carries a
``judge_rubric``. Judging an execution that already failed deterministically is
paying to grade a known failure; judging a case with no rubric is asking a model
to invent the standard it is marking against.

**Judgements are cached by content, not by run.** The key is
``hash(rubric, reference, candidate_output, judge_model)``. Re-rendering a
report, re-running a gate, or judging the same unchanged output next week costs
nothing. A tool that billed you to look at yesterday's table twice is a tool
people stop looking at.

**The judge never learns which model wrote which answer.** Outputs go in as a
randomised A/B pair with neutral labels, the order is derived from the case's
seed so it is reproducible, and the recorded judgement keeps the order it was
shown. Position bias in pairwise LLM judging is large and well documented; an
unblinded judge measures our prompt's framing as much as the model.

**It goes through the same structured-output ladder as anything else.** The
verdict is a schema, not a string we regex. A judge that answered in prose
would be the one place in this tool where output parsing was done by hope.

**And it refuses to produce a number nobody has calibrated.** A judge whose
agreement with human labels has not been measured is a random number generator
with a confident tone, so :class:`Judge` requires a calibration record unless
it is explicitly told to run uncalibrated - and anything derived from it then
carries the agreement figure. See :mod:`slmeval.eval.calibration`.

What is deliberately *not* here yet: the judge is not wired into the runner's
hot path. The skeleton, the cache, the blinding and the calibration harness all
exist and are tested; turning it on inside `slmeval run` waits until a
calibration figure from real labels says what the judge is worth.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

from slmeval import __version__
from slmeval.core.context import no_context
from slmeval.core.outcome import Classification
from slmeval.dataset.models import Case
from slmeval.eval.tier1 import Tier1
from slmeval.eval.tier2 import Tier2
from slmeval.providers.base import ChatClient

TIER = 3

Verdict = Literal["pass", "fail", "unclear"]
"""Three, not two.

``unclear`` is the one that earns its place: a judge forced to choose between
pass and fail on an answer it cannot assess will pick one, and that coin flip
enters the scorecard as a measurement. An explicit "I cannot tell" is reported
as unjudged and kept out of the rate, the same way an uncovered cassette path
is.
"""


class Judgement(BaseModel):
    """What the judge has to return. A schema, never prose."""

    verdict: Verdict = Field(
        description="pass if the candidate answer meets the rubric, fail if it "
        "does not, unclear if the rubric cannot be applied to it"
    )
    score: float = Field(
        ge=0.0, le=1.0, description="How well the rubric is met, 0 to 1"
    )
    reasoning: str = Field(
        max_length=600, description="One or two sentences. The rubric clause that decided it."
    )
    cited_span: str | None = Field(
        default=None,
        description="The exact substring of the candidate answer the verdict rests on",
    )

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"


SYSTEM_PROMPT = """\
You grade one answer against a rubric. You are a measuring instrument, not a \
collaborator.

Rules:
- Apply only the rubric. Not your own preferences, not what you would have written.
- Length, formatting and confidence are not evidence. A short correct answer \
beats a long one.
- You are not told which system produced which answer. Do not guess, and do \
not let the order they appear in affect the verdict.
- Quote the exact span of the answer your verdict rests on. If you cannot \
quote one, the verdict is `unclear`.
- `unclear` is a real answer. Use it when the rubric does not apply to what you \
were given, rather than choosing between pass and fail.\
"""


@dataclass(frozen=True, slots=True)
class BlindPair:
    """A reference and a candidate answer, in randomised order."""

    first: str
    second: str
    candidate_is: Literal["A", "B"]
    """Which label the candidate was given. Recorded with the judgement,
    because a cached verdict that could not say which side the candidate was on
    would be unauditable."""

    @property
    def order_shown(self) -> str:
        return f"candidate={self.candidate_is}"


def blind(reference: str, candidate: str, *, seed: int, case_id: str = "") -> BlindPair:
    """Put the two answers in a reproducible random order.

    Derived from the seed and the case id, so two candidates on the same case
    are shown in the same order - otherwise the blinding itself would be a
    difference between them - while two cases are not all shown the same way
    round, which is what a fixed order would do.
    """
    rng = random.Random(f"{seed}:{case_id}")
    if rng.random() < 0.5:
        return BlindPair(first=candidate, second=reference, candidate_is="A")
    return BlindPair(first=reference, second=candidate, candidate_is="B")


def normalize_answer(text: str) -> str:
    """Flatten the formatting cues a judge should not be reading.

    Collapsing whitespace and stripping Markdown emphasis will not make a long
    answer short, so this is a mitigation and not a fix - the prompt does the
    rest by saying plainly that length is not evidence. Worth doing anyway:
    bullet lists and bold text measurably shift pairwise judgements, and that
    is a property of the formatter, not of the model.
    """
    text = re.sub(r"[*_`#>]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def build_messages(rubric: str, pair: BlindPair, *, question: str = "") -> list[dict[str, str]]:
    """The judge prompt. Neutral labels, rubric first, answers last."""
    parts = [f"RUBRIC\n{rubric.strip()}"]
    if question:
        parts.append(f"REQUEST\n{normalize_answer(question)}")
    parts.append(f"ANSWER A\n{normalize_answer(pair.first)}")
    parts.append(f"ANSWER B\n{normalize_answer(pair.second)}")
    parts.append(
        "Grade ANSWER "
        f"{pair.candidate_is} against the rubric. The other answer is a "
        "reference for comparison only; do not grade it."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def cache_key(
    *, rubric: str, reference: str, candidate: str, judge_model: str
) -> str:
    """``hash(rubric, reference, candidate_output, judge_model)``.

    Length-prefixed rather than joined by a separator: "a" + "bc" and "ab" +
    "c" must not collide, and a rubric containing whatever separator we picked
    would otherwise be able to forge another pair's key.
    """
    digest = hashlib.sha256()
    for part in (rubric, reference, candidate, judge_model, f"tier3/{TIER}"):
        raw = part.encode("utf-8")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()[:32]


def rubric_hash(rubric: str) -> str:
    return hashlib.sha256(rubric.encode("utf-8")).hexdigest()[:16]


def should_judge(
    case: Case,
    *,
    classification: Classification,
    tier_1: Tier1,
    tier_2: Tier2,
) -> bool:
    """Whether this execution is worth paying a judge for.

    Three conditions, and each one saves money on a question already answered:
    the case has to carry a rubric, the execution has to have been judgeable at
    all and to have returned, and it has to have passed every deterministic
    assertion. ~80% of the signal is free; the judge is for the remainder.

    A case whose *only* assertion is the rubric is judged, and that is not an
    oversight: there the rubric is the assertion, which is exactly what Tier 3
    is for. (``Assertions.is_empty`` agrees - a rubric is not nothing.)
    """
    if not case.assertions.judge_rubric:
        return False
    if not classification.judgeable or not classification.ok:
        return False
    return not (tier_1.assertions.forbidden_called or not tier_1.assertions.ok)


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


@dataclass
class JudgementCache:
    """Content-addressed judgements, in the trace store.

    Takes a connection rather than opening one: the store already owns a WAL
    connection, and one more per judgement is how the blueprint exhausted
    Windows handles.
    """

    conn: sqlite3.Connection

    def get(self, key: str) -> Judgement | None:
        row = self.conn.execute(
            "SELECT verdict, score, reasoning, cited_span FROM judgements WHERE cache_key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return None
        return Judgement(
            verdict=row[0],
            score=row[1] if row[1] is not None else 0.0,
            reasoning=row[2] or "",
            cited_span=row[3],
        )

    def put(
        self,
        key: str,
        judgement: Judgement,
        *,
        judge_model: str,
        rubric: str,
        order_shown: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO judgements (cache_key, judge_model, rubric_hash, "
            "verdict, score, reasoning, cited_span, order_shown, prompt_tokens, "
            "completion_tokens, created_at, slmeval_version) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                key,
                judge_model,
                rubric_hash(rubric),
                judgement.verdict,
                judgement.score,
                judgement.reasoning,
                judgement.cited_span,
                order_shown,
                prompt_tokens,
                completion_tokens,
                datetime.now(UTC).isoformat(),
                __version__,
            ),
        )
        self.conn.commit()

    def __len__(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM judgements").fetchone()
        return int(row[0]) if row else 0


# --------------------------------------------------------------------------- #
# The judge
# --------------------------------------------------------------------------- #


@dataclass
class Judge:
    """One judge model, its cache, and its calibration record."""

    client: ChatClient
    seed: int = 42
    cache: JudgementCache | None = None
    agreement: float | None = None
    """Measured agreement with human labels, from ``slmeval judge calibrate``.
    ``None`` means nobody has measured it - see :attr:`uncalibrated_ok`."""
    uncalibrated_ok: bool = False
    calls: int = 0
    hits: int = 0
    usage: dict[str, int] = field(default_factory=lambda: {"prompt": 0, "completion": 0})

    @property
    def model(self) -> str:
        spec = self.client.spec
        return f"{spec.provider}:{spec.model}"

    def __post_init__(self) -> None:
        if self.agreement is None and not self.uncalibrated_ok:
            raise ValueError(
                "this judge has no measured agreement with human labels.\n"
                "  Run `slmeval judge calibrate <labels.jsonl>` first, or pass "
                "uncalibrated_ok=True and accept that the number means nothing.\n"
                "  A judge whose agreement you have not measured is a random "
                "number generator with a confident tone."
            )

    async def judge(
        self,
        *,
        rubric: str,
        reference: str,
        candidate: str,
        case_id: str = "",
        question: str = "",
    ) -> Judgement:
        """Grade one candidate answer. Cached, blinded, schema-checked."""
        key = cache_key(
            rubric=rubric,
            reference=reference,
            candidate=candidate,
            judge_model=self.model,
        )
        if self.cache is not None:
            cached = self.cache.get(key)
            if cached is not None:
                self.hits += 1
                return cached

        pair = blind(reference, candidate, seed=self.seed, case_id=case_id)
        messages = build_messages(rubric, pair, question=question)

        # Outside the agent's context, so the judge's own tokens never land in
        # the execution's trace and inflate the candidate's cost.
        with no_context():
            completion = await self.client.complete(messages, schema=Judgement)
        self.calls += 1
        self.usage["prompt"] += completion.prompt_tokens or 0
        self.usage["completion"] += completion.completion_tokens or 0

        judgement = completion.parsed_as(Judgement)
        if self.cache is not None:
            self.cache.put(
                key,
                judgement,
                judge_model=self.model,
                rubric=rubric,
                order_shown=pair.order_shown,
                prompt_tokens=completion.prompt_tokens,
                completion_tokens=completion.completion_tokens,
            )
        return judgement

    def note(self) -> str:
        """The line that must appear beside any judge-derived figure."""
        if self.agreement is None:
            return (
                f"judge {self.model} is UNCALIBRATED - its agreement with human "
                "labels has never been measured, so treat any score from it as "
                "unvalidated"
            )
        return (
            f"judge {self.model} agrees with human labels {self.agreement:.0%} "
            "of the time (`slmeval judge calibrate`)"
        )


def as_dict(judgement: Judgement) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(judgement.model_dump_json())
    return payload


def verdicts(judgements: Sequence[Judgement]) -> dict[str, int]:
    out: dict[str, int] = {}
    for judgement in judgements:
        out[judgement.verdict] = out.get(judgement.verdict, 0) + 1
    return out


__all__ = [
    "SYSTEM_PROMPT",
    "TIER",
    "BlindPair",
    "Judge",
    "Judgement",
    "JudgementCache",
    "Verdict",
    "as_dict",
    "blind",
    "build_messages",
    "cache_key",
    "normalize_answer",
    "rubric_hash",
    "should_judge",
    "verdicts",
]
