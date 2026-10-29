"""The named normalizers.

Each test is one of two claims:

- *this pair of calls means the same thing*, so the normalizer must collapse
  them (otherwise the harness invents a cassette miss and blames the model), or
- *this pair means different things*, so the normalizer must keep them apart
  (otherwise a real hallucination is laundered into a pass).

The second kind matters more, and it is where the SQL fallback earns its
complexity: folding the case of a string literal would make ``id = 'X'`` and
``id = 'x'`` the same call.
"""

from __future__ import annotations

from typing import Any

import pytest

from slmeval.sandbox import normalizers
from slmeval.sandbox.normalizers import UnknownNormalizer, resolve, resolve_all, sql_fallback


@pytest.mark.parametrize(
    ("spec", "value", "expected"),
    [
        ("strip", "  x  ", "x"),
        ("lower", "ABC", "abc"),
        ("upper", "abc", "ABC"),
        ("lower_strip", "  ABC ", "abc"),
        ("upper_strip", " ord-9921 ", "ORD-9921"),
        ("collapse_ws", "a   b\n\tc ", "a b c"),
        ("round:2", 0.30000000000000004, 0.3),
        ("round:2", 1.005, 1.0),
        ("round:0", 2.6, 3),
        ("sort_list", ["b", "a", "c"], ["a", "b", "c"]),
        ("sort_list", [2, 1], [1, 2]),
        ("json_canonical", '{"b": 1, "a": 2}', '{"a":2,"b":1}'),
        ("json_canonical", {"b": 1, "a": 2}, '{"a":2,"b":1}'),
    ],
)
def test_normalizers_collapse_equivalent_values(spec: str, value: Any, expected: Any) -> None:
    assert resolve(spec)(value) == expected


def test_json_canonical_converges_string_and_object_forms() -> None:
    # One side recording a dict while the other emits the equivalent JSON string
    # is not a difference the model should be charged for.
    canonical = resolve("json_canonical")
    assert canonical('{"a": 1, "b": [2, 3]}') == canonical({"b": [2, 3], "a": 1})


@pytest.mark.parametrize(
    ("spec", "value"),
    [
        ("round:2", "two"),
        ("round:2", None),
        ("round:2", True),
        ("upper_strip", 9921),
        ("upper_strip", None),
        ("sort_list", "abc"),
        ("sort_list", {"a": 1}),
        ("json_canonical", 7),
        ("json_canonical", "not json at all"),
        ("sql", 42),
    ],
)
def test_a_normalizer_handed_the_wrong_shape_passes_it_through(spec: str, value: Any) -> None:
    # Total and type-tolerant on purpose. A model passing "two" where the tool
    # declares `quantity: int` is the hallucination we want the key to expose as
    # a miss; raising here would crash the benchmark on the model's mistake and
    # lose the finding.
    assert resolve(spec)(value) == value


def test_round_does_not_round_booleans() -> None:
    # bool is an int subclass, so a naive numeric check turns True into 1.
    rounded = resolve("round:2")(True)
    assert rounded is True


# --------------------------------------------------------------------------- #
# Declaration-time errors
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec",
    ["uppper_strip", "", "round", "round:x", "round:99", "round:-1", "strip:2", "sql:postgres"],
)
def test_a_bad_spec_is_rejected_when_the_tool_is_declared(spec: str) -> None:
    with pytest.raises(UnknownNormalizer):
        resolve(spec)


def test_an_unknown_normalizer_lists_the_real_ones() -> None:
    with pytest.raises(UnknownNormalizer) as caught:
        resolve("uppercase")
    message = str(caught.value)
    assert "upper_strip" in message
    assert "round:N" in message


def test_resolve_all_names_the_offending_argument() -> None:
    with pytest.raises(UnknownNormalizer) as caught:
        resolve_all({"order_id": "upper_strip", "query": "sqll"})
    assert "query" in str(caught.value)


def test_a_non_string_spec_is_rejected() -> None:
    with pytest.raises(UnknownNormalizer):
        resolve(object())  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# SQL
# --------------------------------------------------------------------------- #


def test_sql_backend_is_reported() -> None:
    # Which parser normalized a query changes what "these differ" means in a
    # miss diagnosis, so it is reportable rather than internal.
    assert normalizers.sql_backend() in {"sqlglot", "fallback"}


@pytest.mark.parametrize(
    ("left", "right"),
    [
        (
            "SELECT * FROM orders WHERE id='X'",
            "select  *  from orders where id = 'X'",
        ),
        (
            "SELECT * FROM orders WHERE id = 'X';",
            "SELECT * FROM orders WHERE id = 'X'",
        ),
        (
            "select a, b from t",
            "select a,b from t",
        ),
        (
            "select count(*) from t",
            "select count( * ) from t",
        ),
        (
            "select * from t where a in (1,2)",
            "select * from t where a in ( 1 , 2 )",
        ),
        (
            "select * from t\nwhere a = 1",
            "select * from t where a = 1",
        ),
    ],
)
def test_semantically_identical_queries_normalize_alike(left: str, right: str) -> None:
    assert normalizers.sql(left) == normalizers.sql(right)


def test_sql_preserves_the_case_of_string_literals() -> None:
    # `id = 'X'` and `id = 'x'` select different rows in most databases.
    # Collapsing them would turn a real parameter error into a silent pass -
    # the single worst thing a normalizer can do.
    assert normalizers.sql("select * from t where id = 'X'") != normalizers.sql(
        "select * from t where id = 'x'"
    )
    assert "'X'" in sql_fallback("SELECT * FROM t WHERE id = 'X'")


def test_sql_fallback_folds_keywords_and_bare_identifiers() -> None:
    assert sql_fallback("SELECT Id FROM Orders") == "select id from orders"


def test_sql_fallback_preserves_quoted_identifiers() -> None:
    # A double-quoted identifier is case-sensitive in standard SQL.
    assert '"Orders"' in sql_fallback('SELECT * FROM "Orders"')


def test_sql_fallback_drops_comments() -> None:
    assert sql_fallback("select 1 -- why not\n") == sql_fallback("select 1")
    assert sql_fallback("select /* inline */ 1") == sql_fallback("select 1")


def test_sql_fallback_keeps_multi_character_operators_intact() -> None:
    assert sql_fallback("select * from t where a <> 1") == "select * from t where a <> 1"
    assert sql_fallback("select * from t where a>=1") == "select * from t where a >= 1"


def test_sql_fallback_is_indifferent_to_nested_paren_spacing() -> None:
    assert sql_fallback("select ((a)) from t") == sql_fallback("select ( ( a ) ) from t")


def test_sql_fallback_leaves_unparseable_text_deterministic() -> None:
    # An argument the parser rejects still has to hash, and may itself be the
    # finding. The only requirement is that it is stable.
    garbage = "SELEKT ** FRUM"
    assert sql_fallback(garbage) == sql_fallback(garbage)
