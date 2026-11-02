"""The Tier 3 judge, and every guardrail that makes its number mean something.

The judge is the only evaluator here that costs money and the only one that can
be wrong in a way that sounds right, so most of this file is about what it
*refuses* to do: judge a run that already failed, judge without a rubric, pay
twice for one question, see which model wrote which answer, or produce a figure
nobody has calibrated.

Nothing here touches a network. The judge is driven by the scripted `fake`
provider, which is also how a user can try the whole flow with no key - see
``examples/judge``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from slmeval.config.models import JudgeCfg
from slmeval.core.context import RunContext, get_context, use_context
from slmeval.core.outcome import OK, Classification
from slmeval.dataset.models import Assertions, Case
from slmeval.eval.tier1 import tier1
from slmeval.eval.tier2 import tier2
from slmeval.eval.tier3 import (
    Judge,
    Judgement,
    JudgementCache,
    blind,
    build_messages,
    cache_key,
    normalize_answer,
    should_judge,
    verdicts,
)
from slmeval.providers.fake import clear_scripts
from slmeval.providers.registry import build_client
from slmeval.storage.db import RunStore
from slmeval.storage.models import ExecStatus, Outcome

RUBRIC = "The answer must name the order id and the carrier."
REFERENCE = "Your order ORD-9921 is with DHL."
CANDIDATE = "ORD-9921 is on its way with DHL."

PASS_JSON = (
    '{"verdict": "pass", "score": 0.9, "reasoning": "names both", '
    '"cited_span": "ORD-9921"}'
)
FAIL_JSON = (
    '{"verdict": "fail", "score": 0.1, "reasoning": "names neither", '
    '"cited_span": null}'
)


@pytest.fixture(autouse=True)
def scripts() -> None:
    clear_scripts()


def judge_cfg(model: str = "scripted-judge", *script: str) -> JudgeCfg:
    return JudgeCfg(
        id="judge",
        provider="fake",
        model=model,
        temperature=0.0,
        max_tokens=512,
        extra_body={"script": [{"text": text} for text in script]} if script else {},
    )


def make_judge(*script: str, cache: JudgementCache | None = None, **kwargs: object) -> Judge:
    cfg = judge_cfg("scripted-judge", *(script or (PASS_JSON,)))
    return Judge(
        client=build_client(cfg),
        cache=cache,
        uncalibrated_ok=True,
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------- #
# When the judge runs at all
# --------------------------------------------------------------------------- #


def scored(
    *, rubric: str | None, state: dict[str, object] | None = None, forbidden: bool = False
) -> tuple[Case, object, object]:
    assertions = Assertions(
        state_contains={"intent": "order_status"},
        forbidden_tools=["issue_refund"] if forbidden else [],
        judge_rubric=rubric,
    )
    case = Case(case_id="c1", input={"q": "x"}, assertions=assertions)
    first = tier1(
        assertions=assertions,
        llm_calls=[],
        tool_calls=[],
        state=state if state is not None else {"intent": "order_status"},
    )
    second = tier2(
        tool_calls=[], trajectory=[], steps=1, accepted_trajectories=[], reference=None
    )
    return case, first, second


def test_no_rubric_means_no_judge_call() -> None:
    """A case with no rubric would be asking a model to invent the standard it
    is marking against."""
    case, first, second = scored(rubric=None)
    assert not should_judge(case, classification=OK, tier_1=first, tier_2=second)


def test_a_case_with_a_rubric_that_passed_the_cheap_tiers_is_judged() -> None:
    case, first, second = scored(rubric=RUBRIC)
    assert should_judge(case, classification=OK, tier_1=first, tier_2=second)


def test_an_execution_that_already_failed_is_never_judged() -> None:
    """Paying to grade a known failure is pure cost."""
    case, first, second = scored(rubric=RUBRIC, state={"intent": "refund"})
    assert not first.assertions.ok
    assert not should_judge(case, classification=OK, tier_1=first, tier_2=second)


@pytest.mark.parametrize(
    "outcome", [Outcome.HARNESS_ERROR, Outcome.UNCOVERED, Outcome.CRASHED]
)
def test_an_unjudgeable_or_failed_execution_is_never_judged(outcome: Outcome) -> None:
    """A harness error or an uncovered path is our gap, not a quality signal,
    and it must not become a judge bill either. A crash is a failure the cheap
    tiers already named."""
    case, first, second = scored(rubric=RUBRIC)
    broken = Classification(status=ExecStatus.ERROR, outcome=outcome)
    assert not should_judge(case, classification=broken, tier_1=first, tier_2=second)


def test_a_case_whose_only_assertion_is_the_rubric_is_judged() -> None:
    """There the rubric *is* the assertion, which is what Tier 3 is for."""
    case = Case(case_id="c1", input={"q": "x"}, assertions=Assertions(judge_rubric=RUBRIC))
    assert not case.assertions.is_empty, "a rubric is not nothing"
    first = tier1(assertions=case.assertions, llm_calls=[], tool_calls=[], state={})
    second = tier2(
        tool_calls=[], trajectory=[], steps=1, accepted_trajectories=[], reference=None
    )
    assert should_judge(case, classification=OK, tier_1=first, tier_2=second)


def test_a_case_with_no_assertions_at_all_is_never_judged() -> None:
    """No rubric, nothing to grade against, nothing to pay for."""
    case = Case(case_id="c1", input={"q": "x"}, assertions=Assertions())
    first = tier1(assertions=case.assertions, llm_calls=[], tool_calls=[], state={})
    second = tier2(
        tool_calls=[], trajectory=[], steps=1, accepted_trajectories=[], reference=None
    )
    assert case.assertions.is_empty
    assert not should_judge(case, classification=OK, tier_1=first, tier_2=second)


# --------------------------------------------------------------------------- #
# Blinding
# --------------------------------------------------------------------------- #


def test_the_order_is_random_but_reproducible() -> None:
    """Reproducible because two candidates on one case must be shown the same
    way round - otherwise the blinding is itself a difference between them."""
    first = blind(REFERENCE, CANDIDATE, seed=42, case_id="c1")
    again = blind(REFERENCE, CANDIDATE, seed=42, case_id="c1")
    assert (first.first, first.candidate_is) == (again.first, again.candidate_is)


def test_two_cases_are_not_all_shown_the_same_way_round() -> None:
    """A fixed order would make position bias a constant offset on every case."""
    sides = {blind(REFERENCE, CANDIDATE, seed=42, case_id=f"c{i}").candidate_is for i in range(40)}
    assert sides == {"A", "B"}


def test_the_pair_always_holds_both_answers() -> None:
    for case_id in ("a", "b", "c", "d", "e"):
        pair = blind(REFERENCE, CANDIDATE, seed=7, case_id=case_id)
        assert {pair.first, pair.second} == {REFERENCE, CANDIDATE}


def test_the_prompt_never_says_which_model_wrote_which_answer() -> None:
    pair = blind(REFERENCE, CANDIDATE, seed=42, case_id="c1")
    text = "\n".join(m["content"] for m in build_messages(RUBRIC, pair, question="where is it"))
    assert "ANSWER A" in text and "ANSWER B" in text
    for leak in ("candidate", "baseline", "reference answer", "gpt", "gemma"):
        assert leak not in text.lower(), leak
    # And it says plainly that length is not evidence, which is the other half
    # of the bias mitigation the normalizer cannot do.
    assert "length" in text.lower()


def test_formatting_cues_are_flattened() -> None:
    assert normalize_answer("**ORD-9921**  is\n\nwith `DHL`") == "ORD-9921 is with DHL"


# --------------------------------------------------------------------------- #
# The cache
# --------------------------------------------------------------------------- #


def test_the_cache_key_covers_every_input_and_nothing_else() -> None:
    base = {
        "rubric": RUBRIC,
        "reference": REFERENCE,
        "candidate": CANDIDATE,
        "judge_model": "openai:gpt-4o-mini",
    }
    key = cache_key(**base)  # type: ignore[arg-type]
    assert key == cache_key(**base)  # type: ignore[arg-type]
    for field in base:
        changed = {**base, field: base[field] + " !"}
        assert cache_key(**changed) != key, field  # type: ignore[arg-type]


def test_the_cache_key_cannot_be_forged_by_a_rubric_containing_a_separator() -> None:
    """Length-prefixed, so "a" + "bc" and "ab" + "c" do not collide."""
    one = cache_key(rubric="a", reference="bc", candidate="x", judge_model="m")
    two = cache_key(rubric="ab", reference="c", candidate="x", judge_model="m")
    assert one != two


@pytest.mark.asyncio
async def test_a_second_judgement_of_the_same_output_costs_nothing(tmp_path: Path) -> None:
    """Re-rendering a report must not be a billable operation."""
    with RunStore.for_project(tmp_path) as store:
        cache = JudgementCache(store.connection())
        judge = make_judge(PASS_JSON, cache=cache)

        first = await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)
        assert judge.calls == 1 and judge.hits == 0

        second = await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)
        assert judge.calls == 1, "the judge was called twice for one question"
        assert judge.hits == 1
        assert (second.verdict, second.score) == (first.verdict, first.score)
        assert len(cache) == 1


@pytest.mark.asyncio
async def test_the_cache_survives_the_process_that_wrote_it(tmp_path: Path) -> None:
    """Content-addressed and not scoped to a run: judging the same unchanged
    output next week has to cost nothing."""
    with RunStore.for_project(tmp_path) as store:
        judge = make_judge(PASS_JSON, cache=JudgementCache(store.connection()))
        await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)

    with RunStore.for_project(tmp_path) as reopened:
        judge = make_judge(FAIL_JSON, cache=JudgementCache(reopened.connection()))
        again = await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)
        # The cached verdict, not the newly scripted one: proof it came off disk.
        assert again.verdict == "pass"
        assert judge.calls == 0


def test_the_cache_records_which_side_the_candidate_was_on(tmp_path: Path) -> None:
    """A cached verdict that could not say which side the candidate was shown
    on would be unauditable."""
    with RunStore.for_project(tmp_path) as store:
        conn = store.connection()
        JudgementCache(conn).put(
            "k1",
            Judgement(verdict="pass", score=1.0, reasoning="fine"),
            judge_model="fake:j",
            rubric=RUBRIC,
            order_shown="candidate=B",
        )
        row = conn.execute("SELECT order_shown FROM judgements WHERE cache_key = 'k1'").fetchone()
        assert row[0] == "candidate=B"


def test_the_judgement_table_is_a_migration(tmp_path: Path) -> None:
    with RunStore.for_project(tmp_path) as store:
        names = {
            r[0]
            for r in store.connection().execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert "judgements" in names


# --------------------------------------------------------------------------- #
# Calibration is not optional
# --------------------------------------------------------------------------- #


def test_a_judge_with_no_measured_agreement_refuses_to_exist() -> None:
    """A judge whose agreement nobody has measured is a random number
    generator with a confident tone."""
    cfg = judge_cfg("scripted-judge", PASS_JSON)
    with pytest.raises(ValueError, match="no measured agreement"):
        Judge(client=build_client(cfg))


def test_an_uncalibrated_judge_says_so_wherever_it_is_quoted() -> None:
    judge = make_judge(PASS_JSON)
    assert "UNCALIBRATED" in judge.note()


def test_a_calibrated_judge_carries_its_agreement_figure() -> None:
    judge = make_judge(PASS_JSON, agreement=0.84)
    note = judge.note()
    assert "84%" in note
    assert "UNCALIBRATED" not in note


# --------------------------------------------------------------------------- #
# Everything else
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_the_verdict_is_a_schema_not_a_regex() -> None:
    """Prose from the judge is a provider error, not something to pattern-match
    out - the judge goes through the same ladder as any other call."""
    judge = make_judge("I think it is probably fine, honestly")
    from slmeval.providers.base import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)


@pytest.mark.asyncio
async def test_a_fenced_verdict_is_still_accepted() -> None:
    """A judge that wraps its JSON in a fence is annoying, not wrong."""
    judge = make_judge(f"Here you go:\n```json\n{PASS_JSON}\n```")
    judgement = await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)
    assert judgement.passed


@pytest.mark.asyncio
async def test_the_judges_own_tokens_never_land_in_the_agents_trace() -> None:
    """Otherwise a judged candidate's cost would include the cost of grading
    it, and the Pareto chart the whole tool exists to print would be wrong."""
    ctx = RunContext(run_id="r", candidate_id="c", case_id="c1")
    judge = make_judge(PASS_JSON)
    with use_context(ctx):
        assert get_context() is ctx
        await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate=CANDIDATE)

    assert ctx.llm_calls == []
    assert ctx.steps == 0
    # The judge keeps its own tally, so the cost is reportable - just not as
    # the candidate's.
    assert judge.usage["prompt"] > 0


@pytest.mark.asyncio
async def test_unclear_is_a_real_answer() -> None:
    """A judge forced to choose on an answer it cannot assess picks one, and
    that coin flip would enter the scorecard as a measurement."""
    judge = make_judge(
        '{"verdict": "unclear", "score": 0.5, "reasoning": "the rubric does not '
        'apply", "cited_span": null}'
    )
    judgement = await judge.judge(rubric=RUBRIC, reference=REFERENCE, candidate="¿Dónde está?")
    assert judgement.verdict == "unclear"
    assert not judgement.passed


def test_a_score_outside_the_range_is_refused() -> None:
    with pytest.raises(ValueError):
        Judgement(verdict="pass", score=1.4, reasoning="over-enthusiastic")


def test_verdicts_counts_what_it_was_given() -> None:
    assert verdicts(
        [
            Judgement(verdict="pass", score=1.0, reasoning="a"),
            Judgement(verdict="pass", score=1.0, reasoning="b"),
            Judgement(verdict="fail", score=0.0, reasoning="c"),
        ]
    ) == {"pass": 2, "fail": 1}


def test_the_cache_takes_a_connection_rather_than_opening_one(tmp_path: Path) -> None:
    """One connection per judgement is the blueprint's handle-exhaustion bug,
    and on Windows it surfaces as a lock error rather than as its cause."""
    with RunStore.for_project(tmp_path) as store:
        conn = store.connection()
        cache = JudgementCache(conn)
        assert isinstance(cache.conn, sqlite3.Connection)
        assert cache.conn is store.connection()
