"""The cassette key and its matching policy.

The product rests on one distinction, and these tests are that distinction:

- a **genuine parameter hallucination** must miss every tier, so it reaches the
  diagnosis block;
- a **formatting difference** must hit, because a miss there is our bug reported
  as the model's;
- a **subset hit** must be distinguishable from an exact hit, because it is the
  finding "the model reached the right call with the wrong extra arguments" and
  it must never be reported as a clean pass.

Everything else here guards the ways that distinction can be quietly broken.
"""

from __future__ import annotations

import unicodedata
from datetime import UTC, datetime
from typing import Any

import pytest

from slmeval.sandbox.keys import (
    MatchPolicy,
    MatchTier,
    PolicyError,
    canonicalize,
    declare,
)

CASE = "refund_104"


def fetch_order_from_erp(order_id: str, request_id: str, include_lines: bool = False) -> dict:
    """The blueprint's example tool. Only its signature is under test."""
    return {}


def erp_policy(**overrides: Any) -> MatchPolicy:
    kwargs: dict[str, Any] = {
        "tool_name": "fetch_order_from_erp",
        "match_on": ["order_id"],
        "normalize": {"order_id": "upper_strip"},
        "ignore_args": ["request_id"],
    }
    kwargs.update(overrides)
    return declare(fetch_order_from_erp, **kwargs)


# --------------------------------------------------------------------------- #
# The cascade
# --------------------------------------------------------------------------- #


def test_the_cascade_is_exact_then_normalized_then_subset() -> None:
    keys = erp_policy().compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    assert [entry.tier for entry in keys] == [
        MatchTier.EXACT,
        MatchTier.NORMALIZED,
        MatchTier.SUBSET,
    ]


def test_without_match_on_there_is_no_subset_tier() -> None:
    keys = erp_policy(match_on=None).compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    assert [entry.tier for entry in keys] == [MatchTier.EXACT, MatchTier.NORMALIZED]
    assert keys.subset is None


def test_the_same_call_produces_the_same_keys() -> None:
    policy = erp_policy()
    first = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    second = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r2"})
    assert first.keys == second.keys


def test_argument_order_cannot_change_the_key() -> None:
    policy = erp_policy()
    a = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r"})
    b = policy.compute(CASE, {"request_id": "r", "order_id": "ORD-9921"})
    assert a.keys == b.keys


def test_a_formatting_difference_hits_the_normalized_tier() -> None:
    # " ord-9921 " and "ORD-9921" are the same order. The exact tier must miss
    # (they really are different bytes) and the normalized tier must hit, or the
    # harness reports a hallucination it invented itself.
    policy = erp_policy()
    recorded = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    called = policy.compute(CASE, {"order_id": " ord-9921 ", "request_id": "r2"})

    assert called.exact != recorded.exact
    assert called.normalized == recorded.normalized


def test_a_hallucinated_parameter_misses_every_tier() -> None:
    # The headline case: "9921" instead of "ORD-9921" is the model dropping a
    # prefix. No tier may rescue it, because this miss is the finding.
    policy = erp_policy()
    recorded = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    called = policy.compute(CASE, {"order_id": "9921", "request_id": "r2"})
    assert not set(called.keys) & set(recorded.keys)


def test_an_ignored_argument_does_not_break_the_exact_tier() -> None:
    # request_id is different on every real call. If it reached the key, the
    # exact tier could never hit and every call would be reported as degraded.
    policy = erp_policy()
    recorded = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    called = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "totally-different"})
    assert called.exact == recorded.exact


def test_a_wrong_extra_argument_is_a_subset_hit_not_an_exact_hit() -> None:
    # include_lines=True reaches the right order by the right id, but it is not
    # the recorded call. Exact and normalized must miss; subset must hit, so the
    # report can say "right call, wrong extra argument" instead of "pass".
    policy = erp_policy()
    recorded = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    called = policy.compute(
        CASE, {"order_id": "ORD-9921", "request_id": "r2", "include_lines": True}
    )
    assert called.exact != recorded.exact
    assert called.normalized != recorded.normalized
    assert called.subset == recorded.subset


def test_a_subset_key_cannot_be_mistaken_for_an_exact_key() -> None:
    # Without the tier in the hash, the subset key of a full call would equal
    # the exact key of a call that passed only the match_on arguments. A model
    # that dropped a required argument would then hit the other cassette and be
    # reported as an exact match - the worst kind of silent pass.
    policy = declare(fetch_order_from_erp, tool_name="fetch_order_from_erp", match_on=["order_id"])
    full = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r1"})
    dropped = policy.compute(CASE, {"order_id": "ORD-9921"})
    assert full.subset != dropped.exact


def test_keys_are_scoped_to_the_case() -> None:
    policy = erp_policy()
    args = {"order_id": "ORD-9921", "request_id": "r"}
    assert not set(policy.compute("refund_104", args).keys) & set(
        policy.compute("refund_207", args).keys
    )


def test_keys_are_scoped_to_the_tool() -> None:
    args = {"order_id": "ORD-9921", "request_id": "r"}
    a = erp_policy().compute(CASE, args)
    b = erp_policy(tool_name="cancel_order_in_erp").compute(CASE, args)
    assert not set(a.keys) & set(b.keys)


def test_a_call_with_no_case_id_is_refused() -> None:
    # No case id means the sandbox had no run context. Filing the response
    # anyway would let one case's cassettes answer another case's calls.
    with pytest.raises(PolicyError) as caught:
        erp_policy().compute("", {"order_id": "ORD-1", "request_id": "r"})
    assert "case id" in str(caught.value)


# --------------------------------------------------------------------------- #
# Defaults
# --------------------------------------------------------------------------- #


def test_a_default_spelled_out_is_the_same_call_as_one_left_out() -> None:
    # Recording f(order_id=..) and replaying f(order_id=.., include_lines=False)
    # is the same call. A verbose model that states a default has not done
    # anything wrong, and without this half its calls would read as misses.
    policy = erp_policy()
    terse = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r"})
    verbose = policy.compute(
        CASE, {"order_id": "ORD-9921", "request_id": "r", "include_lines": False}
    )
    assert terse.exact == verbose.exact


def test_omitting_a_required_argument_still_misses() -> None:
    # Required parameters have no default to fill, so this stays a miss - which
    # is correct: it is the finding.
    policy = erp_policy()
    full = policy.compute(CASE, {"order_id": "ORD-9921", "request_id": "r"})
    missing = policy.compute(CASE, {"request_id": "r"})
    assert not set(full.keys) & set(missing.keys)


# --------------------------------------------------------------------------- #
# Canonicalization
# --------------------------------------------------------------------------- #


def test_unicode_composition_does_not_change_the_key() -> None:
    # macOS filesystem APIs hand out NFD where Linux and Windows produce NFC.
    # Without normalization, a cassette recorded on a laptop misses in CI - a
    # false miss caused entirely by the recording machine.
    policy = declare(tool_name="lookup_customer", fn=None)
    composed = unicodedata.normalize("NFC", "café")
    decomposed = unicodedata.normalize("NFD", "café")
    assert composed != decomposed
    assert policy.compute(CASE, {"name": composed}).exact == (
        policy.compute(CASE, {"name": decomposed}).exact
    )


def test_a_datetime_argument_is_not_the_string_that_looks_like_it() -> None:
    # Arguments go through the typed envelope, not str(). A tool called with a
    # datetime and a tool called with its ISO rendering are different calls.
    policy = declare(tool_name="orders_since", fn=None)
    moment = datetime(2026, 10, 28, 10, 0, tzinfo=UTC)
    assert policy.compute(CASE, {"since": moment}).exact != (
        policy.compute(CASE, {"since": moment.isoformat()}).exact
    )


def test_canonical_form_is_sorted_and_compact() -> None:
    assert canonicalize({"b": 1, "a": 2}, tool_name="t") == '{"a":2,"b":1}'


def test_the_canonical_arguments_are_kept_for_the_diagnosis() -> None:
    # The miss diagnosis shows the user what was compared, so the string the
    # hash was taken over is carried alongside the key rather than recomputed.
    keys = erp_policy().compute(CASE, {"order_id": " ord-9921 ", "request_id": "r"})
    assert "ord-9921" in keys.entries[0].canonical_args
    assert "ORD-9921" in keys.entries[1].canonical_args
    assert "request_id" not in keys.entries[0].canonical_args


def test_keys_are_hex_and_fixed_width() -> None:
    for entry in erp_policy().compute(CASE, {"order_id": "ORD-1", "request_id": "r"}):
        assert len(entry.key) == 32
        assert all(character in "0123456789abcdef" for character in entry.key)


# --------------------------------------------------------------------------- #
# Declaration-time validation
# --------------------------------------------------------------------------- #


def test_an_unknown_argument_name_is_caught_at_declaration() -> None:
    # Import time, not two hours into a matrix run - everything needed to catch
    # it is available the moment the decorator executes.
    with pytest.raises(PolicyError) as caught:
        erp_policy(match_on=["order_number"])
    assert "order_number" in str(caught.value)


def test_a_near_miss_argument_name_suggests_the_right_one() -> None:
    with pytest.raises(PolicyError) as caught:
        erp_policy(match_on=["order_i"])
    assert "order_id" in (caught.value.hint or "")


@pytest.mark.parametrize(
    "overrides",
    [
        {"ignore_args": ["nope"]},
        {"normalize": {"nope": "strip"}},
    ],
)
def test_every_declaration_list_is_checked_against_the_signature(
    overrides: dict[str, Any],
) -> None:
    with pytest.raises(PolicyError):
        erp_policy(**overrides)


def test_an_empty_match_on_is_refused() -> None:
    # match_on=[] would make every call to the tool match one cassette.
    with pytest.raises(PolicyError) as caught:
        erp_policy(match_on=[])
    assert "every call" in str(caught.value)


def test_a_duplicated_match_on_argument_is_refused() -> None:
    with pytest.raises(PolicyError):
        erp_policy(match_on=["order_id", "order_id"])


def test_matching_on_an_ignored_argument_is_a_contradiction() -> None:
    with pytest.raises(PolicyError) as caught:
        erp_policy(match_on=["request_id"], ignore_args=["request_id"], normalize={})
    assert "request_id" in str(caught.value)


def test_normalizing_an_ignored_argument_is_a_contradiction() -> None:
    with pytest.raises(PolicyError) as caught:
        erp_policy(normalize={"request_id": "strip"})
    assert "request_id" in str(caught.value)


def test_an_unknown_normalizer_names_the_tool() -> None:
    with pytest.raises(PolicyError) as caught:
        erp_policy(normalize={"order_id": "uppercase"})
    message = str(caught.value)
    assert "fetch_order_from_erp" in message
    assert "upper_strip" in message


def test_a_kwargs_tool_accepts_names_the_signature_cannot_show() -> None:
    # A tool that takes **kwargs receives argument names dynamically, so the
    # signature proves nothing and refusing the declaration would be wrong.
    def flexible(**kwargs: Any) -> dict:
        return {}

    policy = declare(flexible, tool_name="flexible", match_on=["whatever"])
    assert policy.accepts_kwargs is True
    assert policy.match_on == ("whatever",)


def test_a_policy_can_be_declared_without_a_callable() -> None:
    # The adapter sees some tools only by name; they get a policy with no
    # signature validation rather than no policy.
    policy = declare(tool_name="opaque", fn=None, match_on=["anything"])
    assert policy.parameters == ()
    assert policy.compute(CASE, {"anything": 1}).subset is not None


def test_self_is_not_treated_as_an_argument() -> None:
    class Client:
        def fetch(self, order_id: str) -> dict:
            return {}

    policy = declare(Client.fetch, tool_name="fetch", match_on=["order_id"])
    assert policy.parameters == ("order_id",)
