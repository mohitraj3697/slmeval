"""Golden dataset loading and identity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from slmeval.dataset import Assertions, Case, load_dataset
from slmeval.errors import DatasetError

GOOD = (
    '{"case_id": "a", "input": {"q": "x"}, "tags": ["t1"], '
    '"assert": {"required_tools": ["fetch"], "forbidden_tools": ["refund"]}}'
)


def write(tmp_path: Path, *lines: str) -> Path:
    path = tmp_path / "cases.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_loads_a_good_file(tmp_path: Path) -> None:
    dataset = load_dataset(write(tmp_path, GOOD))
    assert len(dataset) == 1
    assert dataset.tags == ["t1"]
    assert dataset.case("a") is not None
    assert dataset.case("missing") is None


def test_blank_lines_and_comments_are_skipped(tmp_path: Path) -> None:
    dataset = load_dataset(write(tmp_path, "// a comment", "", GOOD, "   "))
    assert len(dataset) == 1


def test_duplicate_case_id_names_both_lines(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc:
        load_dataset(write(tmp_path, GOOD, GOOD))
    message = exc.value.problems[0].message
    assert "duplicate case_id 'a'" in message
    assert "line 1" in message


def test_all_problems_are_collected_not_just_the_first(tmp_path: Path) -> None:
    """One pass of fixes, not eleven runs of the command."""
    with pytest.raises(DatasetError) as exc:
        load_dataset(write(tmp_path, "not json", '{"case_id": "!!", "input": {}}', "[1,2]"))
    assert len(exc.value.problems) >= 3


def test_bad_regex_is_caught_before_the_benchmark_runs(tmp_path: Path) -> None:
    line = json.dumps(
        {"case_id": "a", "input": {"q": 1}, "assert": {"final_answer_regex": "(unclosed"}}
    )
    with pytest.raises(DatasetError) as exc:
        load_dataset(write(tmp_path, line))
    assert "not a valid regular expression" in exc.value.problems[0].message


def test_empty_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(DatasetError, match="no cases"):
        load_dataset(write(tmp_path, ""))


def test_empty_input_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DatasetError) as exc:
        load_dataset(write(tmp_path, '{"case_id": "a", "input": {}}'))
    assert "must not be empty" in exc.value.problems[0].message


def test_unknown_assertion_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled assertion silently never runs, which is worse than a crash."""
    line = json.dumps({"case_id": "a", "input": {"q": 1}, "assert": {"requird_tools": ["t"]}})
    with pytest.raises(DatasetError):
        load_dataset(write(tmp_path, line))


def test_contradictory_tool_lists_rejected() -> None:
    with pytest.raises(ValueError, match="both required and forbidden"):
        Assertions(required_tools=["t"], forbidden_tools=["t"])


def test_empty_assertions_are_detectable() -> None:
    assert Assertions().is_empty
    assert not Assertions(max_tool_calls=0).is_empty
    assert not Assertions(required_tools=["t"]).is_empty


def test_assert_is_exposed_under_its_json_name() -> None:
    case = Case.model_validate({"case_id": "a", "input": {"q": 1}, "assert": {"max_tool_calls": 2}})
    assert case.assertions.max_tool_calls == 2
    assert "assert" in case.model_dump(by_alias=True)


# -- dataset identity ------------------------------------------------------


def test_hash_ignores_formatting(tmp_path: Path) -> None:
    """Reformatting the file must not look like a different benchmark."""
    reordered = json.dumps(
        {
            "input": {"q": "x"},
            "assert": {"forbidden_tools": ["refund"], "required_tools": ["fetch"]},
            "tags": ["t1"],
            "case_id": "a",
        },
        indent=None,
    )
    a = load_dataset(write(tmp_path, GOOD))
    b = load_dataset(write(tmp_path, reordered))
    assert a.content_hash == b.content_hash


def test_hash_changes_when_an_assertion_changes(tmp_path: Path) -> None:
    a = load_dataset(write(tmp_path, GOOD))
    b = load_dataset(write(tmp_path, GOOD.replace('"fetch"', '"fetch_v2"')))
    assert a.content_hash != b.content_hash


def test_hash_changes_when_a_case_is_added(tmp_path: Path) -> None:
    a = load_dataset(write(tmp_path, GOOD))
    b = load_dataset(write(tmp_path, GOOD, GOOD.replace('"a"', '"b"')))
    assert a.content_hash != b.content_hash
