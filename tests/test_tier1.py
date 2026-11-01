"""Tier 1, with the schema split as the main event.

The pair of numbers this file is really about is ``schema_valid_rate`` and
``schema_repaired_rate``. A model that only produced a valid object because our
cascade dug it out of a prose preamble and then re-asked has not demonstrated
schema adherence, and folding the two together would erase the clearest
small-model signal the tool produces. So most of these tests are about exactly
which side of that line a call falls on.
"""

from __future__ import annotations

import inspect

import pytest

from slmeval.core.trace import (
    JsonStage,
    LLMCallTrace,
    SchemaStatus,
    ToolCallTrace,
)
from slmeval.dataset.models import Assertions
from slmeval.eval.tier1 import tier1
from slmeval.sandbox.keys import MatchPolicy
from slmeval.sandbox.tool import SandboxedTool


def llm(
    *,
    status: SchemaStatus = SchemaStatus.VALID,
    stage: JsonStage | None = JsonStage.DIRECT,
    rungs: tuple[str, ...] = ("tools",),
    repairs: int = 0,
    mode: str | None = "tools",
    node: str = "router",
) -> LLMCallTrace:
    return LLMCallTrace(
        seq=0,
        model_id="cand",
        provider="fake",
        model="m",
        node=node,
        schema_name="Routing" if status is not SchemaStatus.NOT_REQUESTED else None,
        schema_status=status,
        structured_mode_used=mode,
        rungs_tried=rungs,
        repair_attempts=repairs,
        json_stage=stage,
    )


def tool(
    name: str = "lookup",
    *,
    args: dict | None = None,
    tier: str | None = None,
    outcome: str = "replayed",
    verdict: str | None = None,
) -> ToolCallTrace:
    return ToolCallTrace(
        seq=0,
        name=name,
        arguments=args if args is not None else {"key": "A", "count": 1},
        match_tier=tier,
        outcome=outcome,
        miss_verdict=verdict,
    )


async def _sample(key: str, count: int, flag: bool = False) -> dict:
    return {"key": key}


def registry() -> dict[str, SandboxedTool]:
    return {
        "lookup": SandboxedTool(
            name="lookup",
            policy=MatchPolicy(tool_name="lookup"),
            version="v1",
            qualname="tests.lookup",
            is_async=True,
            func=_sample,
            signature=inspect.signature(_sample),
        )
    }


def score(**kwargs: object) -> object:
    defaults: dict = {
        "assertions": Assertions(),
        "llm_calls": [],
        "tool_calls": [],
        "state": {},
        "tools": registry(),
    }
    defaults.update(kwargs)  # type: ignore[arg-type]
    return tier1(**defaults)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Schema adherence
# --------------------------------------------------------------------------- #


def test_a_clean_first_attempt_counts_as_valid() -> None:
    result = score(llm_calls=[llm()])
    assert result.schema_calls == 1
    assert result.schema_valid_rate == 1.0
    assert result.schema_repaired_rate == 0.0
    assert not result.lenient


@pytest.mark.parametrize(
    ("kwargs", "why"),
    [
        ({"repairs": 1}, "needed a repair re-ask"),
        ({"rungs": ("tools", "json_mode")}, "fell back to a lower rung"),
        ({"stage": JsonStage.FENCED}, "our parser dug it out of a fenced block"),
        ({"stage": JsonStage.REPAIRED}, "our repair prompt rescued it"),
    ],
)
def test_anything_that_needed_help_counts_as_repaired_not_valid(
    kwargs: dict, why: str
) -> None:
    """Each of these ends in a valid object, and in none of them did the model
    adhere to the schema - we did."""
    result = score(llm_calls=[llm(**kwargs)])
    assert result.schema_valid_rate == 0.0, why
    assert result.schema_repaired_rate == 1.0
    assert result.lenient


def test_an_invalid_response_counts_against_the_model() -> None:
    result = score(llm_calls=[llm(status=SchemaStatus.INVALID, stage=None)])
    assert result.schema_failure_rate == 1.0
    assert result.schema_valid_rate == 0.0


def test_a_call_that_never_got_a_response_is_not_a_schema_failure() -> None:
    """A 500 or a timeout counts against the execution. An adherence rate
    diluted with unrelated crashes would mean nothing."""
    result = score(llm_calls=[llm(status=SchemaStatus.ERROR, stage=None), llm()])
    assert result.schema_calls == 1
    assert result.schema_errors == 1
    assert result.schema_valid_rate == 1.0


def test_a_call_with_no_schema_is_out_of_the_denominator() -> None:
    result = score(llm_calls=[llm(status=SchemaStatus.NOT_REQUESTED, stage=None)])
    assert result.schema_calls == 0
    assert result.schema_valid_rate is None


def test_no_structured_calls_means_no_rate_rather_than_zero() -> None:
    """Printing 0% for a run that asked for no schema would read as total
    failure."""
    result = score(llm_calls=[])
    assert result.schema_valid_rate is None
    assert not any(m.name == "schema_valid_rate" for m in result.metrics())


def test_the_three_schema_rates_partition_the_calls() -> None:
    result = score(
        llm_calls=[
            llm(),
            llm(repairs=1),
            llm(status=SchemaStatus.INVALID, stage=None),
        ]
    )
    total = (
        (result.schema_valid_rate or 0)
        + (result.schema_repaired_rate or 0)
        + (result.schema_failure_rate or 0)
    )
    assert total == pytest.approx(1.0)


def test_the_rung_and_stage_histograms_are_reported() -> None:
    result = score(
        llm_calls=[llm(mode="json_mode", stage=JsonStage.FENCED) for _ in range(3)]
    )
    assert result.rung_histogram == {"json_mode": 3}
    assert result.stage_histogram == {"fenced": 3}
    modes = next(m for m in result.metrics() if m.name == "structured_mode_used")
    assert modes.text == "json_mode"
    assert modes.detail == {"json_mode": 3}
    assert any("fallback rung" in w for w in result.warnings())


# --------------------------------------------------------------------------- #
# Tool arguments
# --------------------------------------------------------------------------- #


def test_a_well_formed_call_passes_every_argument_check() -> None:
    result = score(tool_calls=[tool()])
    assert result.tool_name_valid_rate == 1.0
    assert result.required_args_present_rate == 1.0
    assert result.arg_type_valid_rate == 1.0
    assert result.arg_checks == 2
    assert result.extra_args_hallucinated == 0


def test_a_missing_required_argument_is_found() -> None:
    result = score(tool_calls=[tool(args={"key": "A"})])
    assert result.required_args_missing == 1
    assert result.required_args_present_rate == 0.0
    assert any("missing required argument" in p for p in result.problems)


def test_a_wrong_argument_type_is_named() -> None:
    result = score(tool_calls=[tool(args={"key": "A", "count": "three"})])
    assert result.arg_type_invalid == 1
    assert result.arg_type_valid_rate == pytest.approx(0.5)
    assert any("not int" in p for p in result.problems)


def test_a_bool_is_not_an_int_and_an_int_is_not_a_bool() -> None:
    """Python says `True` is an `int`; a model filling a schema says otherwise,
    and these are exactly the confusions worth catching."""
    as_bool = score(tool_calls=[tool(args={"key": "A", "count": True})])
    assert as_bool.arg_type_invalid == 1
    as_int = score(tool_calls=[tool(args={"key": "A", "count": 1, "flag": 1})])
    assert as_int.arg_type_invalid == 1


def test_an_argument_the_tool_does_not_accept_is_counted() -> None:
    result = score(tool_calls=[tool(args={"key": "A", "count": 1, "nope": 2})])
    assert result.extra_args_hallucinated == 1
    assert any("does not accept" in p for p in result.problems)


def test_a_hallucinated_tool_name_fails_the_name_check() -> None:
    result = score(tool_calls=[tool(name="invent_refund", verdict="hallucinated_tool")])
    assert result.tool_name_invalid == 1
    assert result.tool_name_valid_rate == 0.0


def test_a_degraded_cassette_match_is_a_finding() -> None:
    """The model reached the recorded call, but not with the recorded
    arguments."""
    result = score(tool_calls=[tool(tier="normalized"), tool(tier="exact")])
    assert result.degraded_matches == 1
    assert result.degraded_match_rate == pytest.approx(0.5)
    assert any("after normalization" in w for w in result.warnings())


def test_an_unknown_annotation_is_not_checked_rather_than_guessed() -> None:
    """A wrong "hallucinated argument type" finding would be worse than none,
    so arg_checks says how much was actually verifiable."""

    async def exotic(payload: SomeType) -> None:  # type: ignore[name-defined] # noqa: F821
        return None

    tools = {
        "exotic": SandboxedTool(
            name="exotic",
            policy=MatchPolicy(tool_name="exotic"),
            version="v1",
            qualname="tests.exotic",
            is_async=True,
            func=exotic,
            signature=inspect.signature(exotic),
        )
    }
    result = score(
        tool_calls=[tool(name="exotic", args={"payload": 3})], tools=tools
    )
    assert result.arg_checks == 0
    assert result.arg_type_valid_rate is None


# --------------------------------------------------------------------------- #
# Assertions and the loud warning
# --------------------------------------------------------------------------- #


def test_a_forbidden_tool_call_raises_its_own_metric() -> None:
    """Day 11's acceptance criterion: a hard fail with a loud warning."""
    result = score(
        assertions=Assertions(forbidden_tools=["issue_refund"]),
        tool_calls=[tool(name="issue_refund")],
        state={"ok": True},
    )
    assert result.unsafe_actions == 1
    assert not result.passed_assertions

    unsafe = next(m for m in result.metrics() if m.name == "unsafe_action")
    assert unsafe.value == 1.0
    assert unsafe.text == "issue_refund"
    assert unsafe.detail == {"tools": ["issue_refund"]}


def test_assertions_passed_is_emitted_either_way() -> None:
    """A metric only written on failure makes every rate's denominator wrong."""
    passed = score(assertions=Assertions(state_contains={"a": 1}), state={"a": 1})
    failed = score(assertions=Assertions(state_contains={"a": 1}), state={"a": 2})
    assert next(m for m in passed.metrics() if m.name == "assertions_passed").value == 1.0
    failing = next(m for m in failed.metrics() if m.name == "assertions_passed")
    assert failing.value == 0.0
    assert failing.text is not None and "state_contains" in failing.text


def test_an_empty_assertion_block_warns() -> None:
    result = score(assertions=Assertions())
    assert any("cannot fail" in w for w in result.warnings())
