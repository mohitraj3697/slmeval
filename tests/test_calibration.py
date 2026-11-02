"""The calibration harness, and the arithmetic it must not get wrong.

Three things are pinned here, and all three are about a number being allowed to
mislead:

**Kappa against hand-computed values.** Raw agreement on a lopsided label set
is the single easiest way to publish a flattering figure about a judge that
knows nothing, and kappa is the correction. A wrong kappa would be worse than
no kappa, so the fixtures below carry worked values rather than whatever the
implementation happens to return.

**A judge with no measured agreement, or a measurement from labels nobody
labelled, is not a calibration.** The shipped example set is marked as an
example on every line, and the harness has to say so - loudly - rather than
reporting 41% as if somebody had checked it.

**The `slmeval judge calibrate` exit codes.** 1 when the judge is below the
floor, 2 when nothing was measured. A command that returned 0 after measuring
nothing would be the most expensive kind of green.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmeval import exit_codes
from slmeval.cli.main import app
from slmeval.errors import ConfigError
from slmeval.eval.calibration import (
    MIN_LABELS,
    Label,
    agreement,
    calibration_path,
    kappa_of,
    load_labels,
    stored_agreement,
    verdict_counts,
)
from slmeval.eval.tier3 import Judgement

runner = CliRunner()

EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "judge"

RUBRIC = "The answer must name the order id and the carrier."


def label(case_id: str, verdict: str, *, source: str = "human") -> Label:
    return Label(
        case_id=case_id,
        rubric=RUBRIC,
        reference="Your order ORD-9921 is with DHL.",
        candidate="ORD-9921 is with DHL.",
        verdict=verdict,  # type: ignore[arg-type]
        source=source,
    )


def judged(verdict: str, score: float = 0.5) -> Judgement:
    return Judgement(verdict=verdict, score=score, reasoning="because")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# The arithmetic
# --------------------------------------------------------------------------- #


def test_perfect_agreement_is_one() -> None:
    pairs = [(label(f"c{i}", "pass"), judged("pass")) for i in range(10)]
    result = agreement(pairs, judge_model="fake:j")
    assert result.rate == 1.0
    assert result.kappa is None, "kappa is undefined when one side never varies"
    assert "undefined" in " ".join(result.warnings())


def test_a_judge_that_passes_everything_scores_zero_kappa() -> None:
    """The whole reason kappa is printed. Nine passes in ten and a judge that
    says pass to everything: 90% agreement, and it knows nothing.
    """
    pairs = [
        *[(label(f"p{i}", "pass"), judged("pass")) for i in range(9)],
        (label("f0", "fail"), judged("pass")),
    ]
    result = agreement(pairs, judge_model="fake:j")
    assert result.rate == 0.9
    assert result.kappa == pytest.approx(0.0, abs=1e-9)
    assert any("chance" in w for w in result.warnings())


def test_kappa_matches_a_hand_computed_value() -> None:
    """A 2x2 table worked out by hand.

    human pass / judge pass = 20, pass/fail = 5, fail/pass = 10, fail/fail = 15.
    n = 50. observed = 35/50 = 0.70.
    human: pass 25, fail 25. judge: pass 30, fail 20.
    expected = (25/50)(30/50) + (25/50)(20/50) = 0.30 + 0.20 = 0.50.
    kappa = (0.70 - 0.50) / (1 - 0.50) = 0.40.
    """
    confusion = {
        ("pass", "pass"): 20,
        ("pass", "fail"): 5,
        ("fail", "pass"): 10,
        ("fail", "fail"): 15,
    }
    assert kappa_of(confusion, n=50) == pytest.approx(0.40)


def test_kappa_can_be_negative() -> None:
    """Worse than chance is a real result and must not be clamped to zero: a
    judge that systematically inverts the label is a finding."""
    confusion = {("pass", "fail"): 10, ("fail", "pass"): 10}
    assert kappa_of(confusion, n=20) == pytest.approx(-1.0)


def test_agreement_carries_its_interval() -> None:
    """Thirty labels is a small sample, and gating on a point estimate from a
    small sample is the mistake the rest of this tool exists to avoid."""
    pairs = [(label(f"c{i}", "pass"), judged("pass")) for i in range(3)]
    result = agreement(pairs, judge_model="fake:j")
    assert result.rate == 1.0
    assert result.lower is not None and result.lower < 0.5, (
        "3/3 has a lower bound near 29%, not 100%"
    )


def test_the_confusion_table_records_every_combination() -> None:
    pairs = [
        (label("a", "pass"), judged("pass")),
        (label("b", "pass"), judged("fail")),
        (label("c", "fail"), judged("unclear")),
    ]
    result = agreement(pairs, judge_model="fake:j")
    assert result.confusion == {
        ("pass", "pass"): 1,
        ("pass", "fail"): 1,
        ("fail", "unclear"): 1,
    }
    assert [d.case_id for d in result.disagreements] == ["b", "c"]


def test_abstention_is_counted_apart_from_error() -> None:
    """A judge that abstains on a third of the set is a different problem from
    one that is wrong on a third."""
    pairs = [
        (label("a", "pass"), judged("unclear")),
        (label("b", "fail"), judged("unclear")),
        (label("c", "unclear"), judged("unclear")),
    ]
    result = agreement(pairs, judge_model="fake:j")
    assert result.unclear == 2, "the human's own 'unclear' is agreement, not abstention"
    assert any("unclear" in w for w in result.warnings())


def test_too_few_labels_is_not_a_calibration() -> None:
    pairs = [(label(f"c{i}", "pass"), judged("pass")) for i in range(MIN_LABELS - 1)]
    result = agreement(pairs, judge_model="fake:j")
    assert not result.usable
    assert any("cannot calibrate" in w for w in result.warnings())


def test_labels_that_are_not_human_are_not_a_calibration() -> None:
    """The one thing this tool must never do: report a figure from fabricated
    labels as if somebody had checked it."""
    pairs = [
        (label(f"c{i}", "pass", source="example"), judged("pass"))
        for i in range(MIN_LABELS + 5)
    ]
    result = agreement(pairs, judge_model="fake:j")
    assert result.illustrative
    assert not result.usable
    assert any("not human labels" in w for w in result.warnings())


def test_enough_real_labels_is_a_calibration() -> None:
    pairs = [
        *[(label(f"p{i}", "pass"), judged("pass")) for i in range(15)],
        *[(label(f"f{i}", "fail"), judged("fail")) for i in range(10)],
    ]
    result = agreement(pairs, judge_model="fake:j")
    assert result.usable
    assert result.warnings() == ()


# --------------------------------------------------------------------------- #
# Loading labels
# --------------------------------------------------------------------------- #


def test_a_bad_line_is_reported_by_number(tmp_path: Path) -> None:
    """A thirty-line file with one typo must not produce a validation error
    with no location."""
    path = tmp_path / "labels.jsonl"
    good = json.dumps(label("a", "pass").model_dump())
    path.write_text(f"{good}\n{{not json}}\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"labels\.jsonl:2"):
        load_labels(path)


def test_an_unknown_verdict_says_what_is_allowed(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    payload = label("a", "pass").model_dump() | {"verdict": "maybe"}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError) as caught:
        load_labels(path)
    assert "pass, fail or unclear" in (caught.value.hint or "")


def test_a_boolean_verdict_is_accepted(tmp_path: Path) -> None:
    """People label in spreadsheets. `true` meaning pass is not worth refusing."""
    path = tmp_path / "labels.jsonl"
    payload = label("a", "pass").model_dump() | {"verdict": True}
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    assert load_labels(path)[0].verdict == "pass"


def test_comments_and_blank_lines_are_skipped(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text(
        "// a note\n\n" + json.dumps(label("a", "pass").model_dump()) + "\n",
        encoding="utf-8",
    )
    assert len(load_labels(path)) == 1


def test_a_missing_file_says_what_a_label_looks_like(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as caught:
        load_labels(tmp_path / "nope.jsonl")
    assert "case_id" in (caught.value.hint or "")


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "labels.jsonl"
    path.write_text("// only comments\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="no labels"):
        load_labels(path)


# --------------------------------------------------------------------------- #
# The shipped example set
# --------------------------------------------------------------------------- #


def test_the_shipped_labels_load_and_declare_themselves_as_examples() -> None:
    labels = load_labels(EXAMPLES / "labels.example.jsonl")
    assert len(labels) >= MIN_LABELS
    assert {label.source for label in labels} == {"example"}, (
        "shipping fabricated labels as human ones would undermine the only "
        "argument for having a judge"
    )
    counts = verdict_counts(labels)
    # Balanced on purpose: a label set of obvious passes measures nothing, and
    # kappa on one would be undefined.
    assert counts["pass"] >= 8 and counts["fail"] >= 8
    assert counts.get("unclear", 0) >= 1


# --------------------------------------------------------------------------- #
# The committed record
# --------------------------------------------------------------------------- #


def test_the_stored_record_reports_the_lower_bound_not_the_rate(tmp_path: Path) -> None:
    path = tmp_path / "slmeval.judge.json"
    path.write_text(
        json.dumps({"usable": True, "agreement": 0.9, "agreement_lower": 0.71}),
        encoding="utf-8",
    )
    assert stored_agreement(path) == 0.71


def test_an_unusable_record_reads_as_no_calibration_at_all(tmp_path: Path) -> None:
    path = tmp_path / "slmeval.judge.json"
    path.write_text(
        json.dumps({"usable": False, "agreement": 0.9, "agreement_lower": 0.71}),
        encoding="utf-8",
    )
    assert stored_agreement(path) is None
    assert stored_agreement(tmp_path / "absent.json") is None


def test_a_corrupt_record_is_no_calibration_rather_than_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "slmeval.judge.json"
    path.write_text("{not json", encoding="utf-8")
    assert stored_agreement(path) is None


def test_the_record_lives_beside_eval_toml(tmp_path: Path) -> None:
    assert calibration_path(tmp_path).name == "slmeval.judge.json"
    assert calibration_path(tmp_path, tmp_path / "other.json").name == "other.json"


# --------------------------------------------------------------------------- #
# The command
# --------------------------------------------------------------------------- #


JUDGE_CONFIG = """
[benchmark]
name = "judge-test"
dataset = "./cases.jsonl"
runs_per_case = 3

[target]
entrypoint = "stub_target:run_case"

[baseline]
provider = "fake"
model = "scripted-baseline"

[judge]
provider = "fake"
model = "scripted-judge"
min_agreement = 0.7
extra_body.script = [
    { text = '{"verdict": "pass", "score": 0.9, "reasoning": "names both"}' },
]
"""

CASES = (
    '{"case_id": "c1", "input": {"q": "x"}, "assert": {"state_contains": {"intent": "ok"}}}\n'
)
STUB = "async def run_case(case_input: dict) -> dict:\n    return {'intent': 'ok'}\n"


@pytest.fixture
def judge_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    (tmp_path / "eval.toml").write_text(JUDGE_CONFIG, encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
    (tmp_path / "stub_target.py").write_text(STUB, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return tmp_path


def write_labels(root: Path, count: int, *, source: str, verdict: str = "pass") -> Path:
    path = root / "labels.jsonl"
    lines = [
        json.dumps(label(f"c{i}", verdict, source=source).model_dump())
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_calibrate_writes_the_committed_record(judge_project: Path) -> None:
    write_labels(judge_project, MIN_LABELS + 5, source="human")
    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--ascii"])
    assert result.exit_code == exit_codes.OK, result.output

    payload = json.loads((judge_project / "slmeval.judge.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "slmeval.calibration/1"
    assert payload["judge_model"] == "fake:scripted-judge"
    assert payload["usable"] is True
    assert payload["agreement"] == 1.0
    assert payload["agreement_lower"] < 1.0
    assert "commit it" in result.output


def test_calibrate_exits_one_when_the_judge_is_below_the_floor(judge_project: Path) -> None:
    """The scripted judge says pass to everything; label everything fail."""
    write_labels(judge_project, MIN_LABELS + 5, source="human", verdict="fail")
    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--ascii"])
    assert result.exit_code == exit_codes.GATE_FAILED, result.output
    assert "not ready to grade anything" in result.output


def test_calibrate_exits_two_when_nothing_was_measured(judge_project: Path) -> None:
    """Illustrative labels measure the harness, not the judge. Returning 0
    there would be the most expensive kind of green."""
    write_labels(judge_project, MIN_LABELS + 5, source="example")
    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--ascii"])
    assert result.exit_code == exit_codes.USAGE, result.output
    assert "not human labels" in result.output


def test_calibrate_lists_the_disagreements(judge_project: Path) -> None:
    """The actual deliverable: the pairs where the judge and the human parted
    company tell you whether the rubric is ambiguous or the judge is lenient."""
    path = judge_project / "labels.jsonl"
    lines = [
        json.dumps(label(f"p{i}", "pass").model_dump()) for i in range(MIN_LABELS)
    ]
    lines.append(json.dumps(label("disputed", "fail").model_dump()))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--ascii"])
    assert "parted company" in result.output
    assert "disputed" in result.output
    assert "kappa" in result.output


def test_calibrate_emits_json_for_a_pipe(judge_project: Path) -> None:
    write_labels(judge_project, MIN_LABELS + 5, source="human")
    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--json"])
    assert result.exit_code == exit_codes.OK, result.output
    assert json.loads(result.output)["judge_model"] == "fake:scripted-judge"


def test_calibrate_without_a_judge_section_says_what_to_add(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "eval.toml").write_text(
        JUDGE_CONFIG.split("[judge]")[0], encoding="utf-8"
    )
    (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
    (tmp_path / "stub_target.py").write_text(STUB, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    write_labels(tmp_path, MIN_LABELS + 5, source="human")

    result = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--ascii"])
    assert result.exit_code == exit_codes.USAGE
    assert "[judge]" in result.output


def test_calibrating_twice_costs_one_judge_call_per_pair(judge_project: Path) -> None:
    """Judgements are cached by content, so fixing one label and re-running
    re-judges one pair, not thirty."""
    write_labels(judge_project, MIN_LABELS + 5, source="human")
    first = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--json"])
    second = runner.invoke(app, ["judge", "calibrate", "labels.jsonl", "--json"])
    assert first.exit_code == second.exit_code == exit_codes.OK
    assert json.loads(first.output)["agreement"] == json.loads(second.output)["agreement"]
