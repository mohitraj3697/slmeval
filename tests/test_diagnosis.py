"""The cassette-miss diagnosis.

Most of these tests are about one thing: **the harness must not invent
findings.** A false "hallucinated_parameter" is indistinguishable in a report
from a real one, and the first time a user checks one by hand and finds the
model was right, every other number in the table becomes worthless. So the
classification tests below spend more effort on the cases that must *not* be
called hallucinations than on the ones that must.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from slmeval.sandbox.diagnosis import (
    MANGLE_DISTANCE,
    Verdict,
    argument_distance,
    diagnose,
    diff_arguments,
    glyphs,
    nearest_recorded,
)
from slmeval.sandbox.envelope import encode_json
from slmeval.sandbox.keys import MatchTier, declare
from slmeval.sandbox.store import Cassette, CassetteStore


@pytest.fixture
def store(tmp_path: Path) -> Iterator[CassetteStore]:
    with CassetteStore(tmp_path / "cassettes.sqlite") as store:
        yield store


def record(
    store: CassetteStore,
    *,
    case_id: str = "refund_104",
    tool_name: str = "fetch_order_from_erp",
    arguments: dict[str, Any] | None = None,
    response: Any = None,
) -> None:
    """Record one call through the real key policy.

    Through `declare` rather than by writing rows directly, so these tests
    exercise the same keys and tiers a real recording produces - a diagnosis
    that only worked against hand-built rows would be worthless.
    """
    args = {"order_id": "ORD-9921"} if arguments is None else arguments
    policy = declare(tool_name=tool_name)
    store.record(
        policy.compute(case_id, args),
        arguments=args,
        response={"status": "shipped"} if response is None else response,
    )


# --------------------------------------------------------------------------- #
# Argument distance and the diff
# --------------------------------------------------------------------------- #


def test_identical_arguments_are_distance_zero() -> None:
    args = {"order_id": "ORD-9921", "limit": 10}
    assert argument_distance(args, dict(args)) == 0.0


def test_an_unequal_string_is_never_distance_zero() -> None:
    # Without a floor, a one-character mangle of a long id scores as a perfect
    # match and the nearest search has nothing to rank by.
    assert argument_distance({"q": "A" * 200}, {"q": "A" * 199 + "B"}) > 0.0


def test_a_missing_argument_costs_a_full_penalty() -> None:
    assert argument_distance({}, {"order_id": "ORD-9921"}) == 1.0


def test_distance_is_averaged_so_tools_are_comparable() -> None:
    # Summed, a tool with eight arguments and one mangle would rank worse than
    # a one-argument tool that is entirely different, and the single nearest
    # neighbour shown in the block would be the wrong one.
    wide_called = {f"a{i}": i for i in range(8)} | {"bad": "x"}
    wide_recorded = {f"a{i}": i for i in range(8)} | {"bad": "y"}
    narrow = argument_distance({"only": "aaaa"}, {"only": "zzzz"})
    assert argument_distance(wide_called, wide_recorded) < narrow


def test_true_is_not_one() -> None:
    # `True == 1` in Python. A model passing a boolean where a number was
    # recorded made a type mistake, not a tiny numeric one.
    assert argument_distance({"n": True}, {"n": 1}) == 1.0


def test_a_shorter_list_is_not_a_perfect_match() -> None:
    assert argument_distance({"ids": [1, 2]}, {"ids": [1, 2, 3]}) > 0.0


@pytest.mark.parametrize(
    ("called", "recorded", "note"),
    [
        ("9921", "ORD-9921", "dropped prefix"),
        ("ORD-9921", "9921", "added prefix"),
        ("ORD", "ORD-9921", "dropped suffix"),
        ("ORD-9921-A", "ORD-9921", "added suffix"),
        ("ord-9921", "ORD-9921", "case differs"),
        (" ORD-9921 ", "ORD-9921", "whitespace differs"),
        ("ORD-7733", "ORD-9921", "different value"),
    ],
)
def test_string_differences_are_named(called: str, recorded: str, note: str) -> None:
    # The note is what makes the block worth reading: the values tell a user
    # something is wrong, the note tells them what.
    (diff,) = diff_arguments({"order_id": called}, {"order_id": recorded})
    assert diff.note == note


def test_digits_only_is_recognised() -> None:
    (diff,) = diff_arguments({"order_id": "1042"}, {"order_id": "INV/10/42"})
    assert diff.note == "digits only"


def test_a_type_change_is_reported_but_is_not_a_mangle() -> None:
    # A type change is as often our signature drifting as the model erring, so
    # it is shown and not treated as proof of a hallucination.
    (diff,) = diff_arguments({"qty": "5"}, {"qty": 5})
    assert diff.note.startswith("type changed")
    assert not diff.mangle


def test_a_different_value_is_not_a_mangle() -> None:
    # The single most important line in this file. A model looking up a
    # legitimately different record produces exactly this, and calling it a
    # hallucination is how a harness manufactures failures.
    (diff,) = diff_arguments({"order_id": "ORD-7733"}, {"order_id": "ORD-9921"})
    assert not diff.mangle


def test_only_differing_arguments_appear() -> None:
    diffs = diff_arguments(
        {"order_id": "9921", "limit": 10, "verbose": False},
        {"order_id": "ORD-9921", "limit": 10, "verbose": False},
    )
    assert [d.name for d in diffs] == ["order_id"]


def test_an_omitted_argument_is_a_mangle() -> None:
    (diff,) = diff_arguments({}, {"order_id": "ORD-9921"})
    assert diff.note == "argument not passed"
    assert diff.mangle
    assert diff.called_missing


# --------------------------------------------------------------------------- #
# Nearest neighbour
# --------------------------------------------------------------------------- #


def test_nearest_picks_the_closest_recording(store: CassetteStore) -> None:
    record(store, arguments={"order_id": "ORD-1111"})
    record(store, arguments={"order_id": "ORD-9921"})
    record(store, arguments={"order_id": "CUST-4242"})

    nearest = nearest_recorded(
        {"order_id": "9921"}, store.neighbours("refund_104", "fetch_order_from_erp")
    )
    assert nearest is not None
    assert nearest.arguments == {"order_id": "ORD-9921"}


def test_nearest_searches_exact_tier_rows_only(store: CassetteStore) -> None:
    # The normalized and subset rows of the same call hold reduced argument
    # forms. Offering a user a normalized rendering of their own recording as
    # "nearest recorded" would have them hunting for a call they never made.
    policy = declare(tool_name="fetch_order_from_erp", normalize={"order_id": "upper_strip"})
    store.record(
        policy.compute("refund_104", {"order_id": " ord-9921 "}),
        arguments={"order_id": " ord-9921 "},
        response={"status": "shipped"},
    )
    rows = store.neighbours("refund_104", "fetch_order_from_erp")
    assert [row.match_tier for row in rows] == [MatchTier.EXACT]


def test_an_undecodable_cassette_is_skipped_not_fatal(store: CassetteStore) -> None:
    # A cassette recorded against a class that has since been renamed must not
    # turn the user's finding into a second, unrelated failure.
    record(store, arguments={"order_id": "ORD-9921"})
    store.put(
        Cassette(
            key="broken",
            case_id="refund_104",
            tool_name="fetch_order_from_erp",
            match_tier=MatchTier.EXACT,
            canonical_args="{}",
            args_json=encode_json({"__t": "pydantic", "cls": "gone.Nope", "v": {}}),
            response_envelope=encode_json({}),
            recorded_at=datetime.now(UTC).isoformat(),
        )
    )
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.nearest_arguments == {"order_id": "ORD-9921"}


# --------------------------------------------------------------------------- #
# Classification
# --------------------------------------------------------------------------- #


def test_a_mangled_argument_is_a_parameter_hallucination(store: CassetteStore) -> None:
    record(store, arguments={"order_id": "ORD-9921"})
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.HALLUCINATED_PARAMETER
    assert not finding.uncovered
    assert finding.recorded_calls == 1


def test_a_different_identifier_is_uncovered_not_hallucinated(store: CassetteStore) -> None:
    # Identifiers in one system share their prefix, so a genuinely different
    # order scores around 50% similar to the recorded one. A threshold loose
    # enough to call that a hallucination accuses the model every time it
    # legitimately looks up a second record.
    record(store, arguments={"order_id": "ORD-9921"})
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-7733"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.UNRECORDED_VALID_PATH
    assert finding.uncovered
    assert finding.distance is not None
    assert finding.distance > MANGLE_DISTANCE


def test_a_near_typo_still_reads_as_a_hallucination(store: CassetteStore) -> None:
    # The backstop for a corruption the named mangles do not cover.
    record(store, arguments={"order_id": "ORD-9921"})
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-9912"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.HALLUCINATED_PARAMETER


def test_an_unknown_tool_name_is_an_invented_tool(store: CassetteStore) -> None:
    record(store)
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erps",
        arguments={"order_id": "ORD-9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.HALLUCINATED_TOOL
    assert finding.suggestions == ("fetch_order_from_erp",)


def test_a_real_tool_at_the_wrong_step(store: CassetteStore) -> None:
    # `issue_refund` exists and other tools were recorded for this case, so
    # the model reached for the wrong instrument rather than inventing one.
    record(store, tool_name="fetch_order_from_erp")
    finding = diagnose(
        case_id="refund_104",
        tool_name="issue_refund",
        arguments={"order_id": "ORD-9921"},
        store=store,
        known_tools=["fetch_order_from_erp", "issue_refund"],
    )
    assert finding.verdict is Verdict.WRONG_TOOL_FOR_STEP
    assert finding.also_recorded == ("fetch_order_from_erp",)


def test_an_empty_store_is_never_a_hallucination(store: CassetteStore) -> None:
    # Nothing recorded means nothing to compare against. A verdict here would
    # be the harness reading a finding out of its own empty store.
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.UNRECORDED_VALID_PATH


def test_no_store_still_produces_a_finding() -> None:
    # The store can be missing (a harness bug upstream) and the miss is still
    # real. Degrade the diagnosis, never the finding.
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-9921"},
        store=None,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.verdict is Verdict.UNRECORDED_VALID_PATH
    assert finding.nearest is None


def test_a_broken_store_does_not_replace_the_finding(store: CassetteStore) -> None:
    store.close()  # every query from here raises
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert finding.recorded_calls == 0
    assert "CASSETTE_MISS" in finding.render()


def test_only_the_uncovered_verdict_is_uncovered() -> None:
    uncovered = [v for v in Verdict if v.uncovered]
    assert uncovered == [Verdict.UNRECORDED_VALID_PATH]


# --------------------------------------------------------------------------- #
# The block itself
# --------------------------------------------------------------------------- #


def test_the_block_reads_like_the_spec(store: CassetteStore) -> None:
    # This is the output people screenshot, so its shape is a test and not an
    # implementation detail.
    record(store, arguments={"order_id": "ORD-9921"})
    block = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
        candidate_id="gemma3:4b",
        node="order_lookup",
    ).render(ascii_mode=False)

    lines = block.splitlines()
    assert lines[0] == "✗ refund_104 · gemma3:4b · node=order_lookup · CASSETTE_MISS"
    assert lines[1] == "  tool: fetch_order_from_erp"
    assert lines[2] == '  called with: {"order_id": "9921"}'
    assert lines[3] == '  nearest recorded: {"order_id": "ORD-9921"}'
    assert lines[4] == '  diff: order_id: "ORD-9921" → "9921"   (dropped prefix)'
    assert lines[5] == "  verdict: hallucinated_parameter"


def test_an_uncovered_block_offers_fill_gaps(store: CassetteStore) -> None:
    record(store, arguments={"order_id": "ORD-9921"})
    block = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "ORD-7733"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    ).render()
    assert "--fill-gaps" in block
    assert "coverage gap" in block


def test_the_block_degrades_on_a_legacy_codepage(store: CassetteStore) -> None:
    record(store)
    block = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "9921"},
        store=store,
        known_tools=["fetch_order_from_erp"],
    ).render(ascii_mode=True)
    assert block.startswith("XX ")
    assert "→" not in block
    assert "->" in block


def test_glyphs_are_explicit_when_asked() -> None:
    assert glyphs(ascii_mode=True) == ("XX", "->")
    assert glyphs(ascii_mode=False) == ("✗", "→")


def test_rendering_survives_an_unserializable_argument(store: CassetteStore) -> None:
    # A renderer on the failure path is not allowed to raise, whatever the
    # model managed to put in the arguments.
    record(store)
    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": object()},
        store=store,
        known_tools=["fetch_order_from_erp"],
    )
    assert "CASSETTE_MISS" in finding.render()


def test_str_is_the_block() -> None:
    finding = diagnose(
        case_id="refund_104",
        tool_name="whatever",
        arguments={},
        known_tools=["whatever"],
    )
    assert str(finding) == finding.render()
