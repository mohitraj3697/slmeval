"""The extraction cascade, stage by stage.

Every case here is a shape a real small model produced. The assertion is never
just "it parsed" - it is "it parsed AND the report will say our parser had to
help", because a model that passes only because the parser is forgiving is a
finding, not a pass.
"""

from __future__ import annotations

import pytest

from slmeval.core.trace import JsonStage
from slmeval.providers.jsonx import JsonExtractionError, extract_json


@pytest.mark.parametrize(
    ("text", "stage", "value"),
    [
        ('{"a": 1}', JsonStage.DIRECT, {"a": 1}),
        ('  {"a": 1}\n', JsonStage.DIRECT, {"a": 1}),
        ("[1, 2]", JsonStage.DIRECT, [1, 2]),
        ('{"a": "}{"}', JsonStage.DIRECT, {"a": "}{"}),
        ('```json\n{"a": 1}\n```', JsonStage.FENCED, {"a": 1}),
        ('```\n{"a": 1}\n```', JsonStage.FENCED, {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', JsonStage.FENCED, {"a": 1}),
        ('Sure! {"a": 1} Let me know.', JsonStage.BALANCED, {"a": 1}),
        ('{"a": 1} {"a": 2}', JsonStage.BALANCED, {"a": 1}),
        ('{"a": 1,}', JsonStage.REPAIRED, {"a": 1}),
        ('{"a": [1, 2,],}', JsonStage.REPAIRED, {"a": [1, 2]}),
        ("{“a”: 1}", JsonStage.REPAIRED, {"a": 1}),
        ("{'a': True, 'b': None}", JsonStage.REPAIRED, {"a": True, "b": None}),
    ],
)
def test_cascade_reports_the_stage_that_won(text: str, stage: JsonStage, value: object) -> None:
    extraction = extract_json(text)
    assert extraction.value == value
    assert extraction.stage is stage


def test_only_direct_is_not_a_leniency_event() -> None:
    assert extract_json('{"a": 1}').lenient is False
    assert extract_json('```json\n{"a": 1}\n```').lenient is True
    assert extract_json('{"a": 1,}').lenient is True


def test_fence_body_wins_over_a_preamble_that_also_looks_like_json() -> None:
    """A worked example in the prose must not beat the real answer."""
    text = 'For example {"a": 0} would work. Here is yours:\n```json\n{"a": 7}\n```'
    extraction = extract_json(text)
    assert extraction.value == {"a": 7}
    assert extraction.stage is JsonStage.FENCED


def test_balanced_scan_is_string_aware() -> None:
    text = 'Output: {"note": "use } and { carefully", "n": 2} done'
    assert extract_json(text).value == {"note": "use } and { carefully", "n": 2}


def test_balanced_scan_handles_escaped_quotes() -> None:
    text = 'Answer: {"q": "say \\"hi\\"", "n": 1}!'
    assert extract_json(text).value == {"q": 'say "hi"', "n": 1}


def test_nested_objects_survive() -> None:
    text = 'ok: {"a": {"b": [{"c": 1}]}} end'
    assert extract_json(text).value == {"a": {"b": [{"c": 1}]}}


@pytest.mark.parametrize("text", ["", "   ", None, "I cannot help with that.", "{", "{'a': foo}"])
def test_hopeless_output_raises(text: str | None) -> None:
    with pytest.raises(JsonExtractionError):
        extract_json(text)


def test_the_error_shows_what_the_model_actually_emitted() -> None:
    """"invalid JSON" with no sample is an unactionable bug report."""
    with pytest.raises(JsonExtractionError) as excinfo:
        extract_json("I'm sorry, I can't do that.")
    assert "I'm sorry" in str(excinfo.value)


def test_the_error_preview_is_truncated() -> None:
    with pytest.raises(JsonExtractionError) as excinfo:
        extract_json("no json here " * 500)
    message = str(excinfo.value)
    assert "chars)" in message
    assert len(message) < 1000


def test_literal_eval_cannot_be_used_to_execute_anything() -> None:
    """The repair stage evaluates literals only. Nothing here is callable."""
    for hostile in ["__import__('os').system('echo hi')", "{'a': open('x')}", "lambda: 1"]:
        with pytest.raises(JsonExtractionError):
            extract_json(hostile)


def test_repair_refuses_values_json_cannot_round_trip() -> None:
    """A tuple or a set would not survive the trip into a cassette."""
    with pytest.raises(JsonExtractionError):
        extract_json("(1, 2)")
    with pytest.raises(JsonExtractionError):
        extract_json("{1, 2}")
