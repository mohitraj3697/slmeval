"""Mapping a validation error back to a line in the source TOML.

The case that matters is the second one: `model` inside the *second*
[[candidates]] must never be reported as the first one's line, or every
multi-candidate error points at the wrong place.
"""

from __future__ import annotations

from slmeval.config.locate import locate

SAMPLE = """\
[benchmark]
name = "x"
runs_per_case = 3

[baseline]
provider = "openai"
model = "gpt-4o"

[[candidates]]
id = "a"
model = "first"

[[candidates]]
id = "b"
model = "second"
"""


def test_key_inside_a_table() -> None:
    assert locate(SAMPLE, ("benchmark", "runs_per_case")) == 3


def test_key_in_the_first_array_entry() -> None:
    assert locate(SAMPLE, ("candidates", 0, "model")) == 11


def test_key_in_the_second_array_entry_is_not_the_first() -> None:
    assert locate(SAMPLE, ("candidates", 1, "model")) == 15


def test_same_key_name_in_a_different_table() -> None:
    assert locate(SAMPLE, ("baseline", "model")) == 7


def test_table_header_when_there_is_no_key() -> None:
    assert locate(SAMPLE, ("baseline",)) == 5
    assert locate(SAMPLE, ("candidates", 1)) == 13


def test_missing_key_falls_back_to_the_header() -> None:
    assert locate(SAMPLE, ("baseline", "temperature")) == 5


def test_unknown_section_returns_none_rather_than_a_wrong_guess() -> None:
    assert locate(SAMPLE, ("nope",)) is None


def test_empty_loc_and_empty_text() -> None:
    assert locate(SAMPLE, ()) is None
    assert locate("", ("benchmark", "name")) is None


def test_quoted_keys_are_found() -> None:
    text = '[benchmark]\n"name" = "x"\n'
    assert locate(text, ("benchmark", "name")) == 2


def test_header_with_a_trailing_comment() -> None:
    text = "[[candidates]]  # first one\nid = \"a\"\n"
    assert locate(text, ("candidates", 0, "id")) == 2
