"""``slmeval gate`` and ``slmeval baseline`` end to end.

Step 14's acceptance criteria, as tests: a weak candidate fails the gate, and a
run whose model server was unreachable exits 3 and is reported as
infrastructure rather than as a regression. Both go through the real CLI
against a real stored run, so the exit code a CI pipeline would see is the one
being asserted.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmeval import exit_codes
from slmeval.cli.main import app
from slmeval.report.baseline import BASELINE_FILENAME
from tests.offline import (
    behaves,
    build_and_run,
    candidate,
    clone_project,
    forget_the_agent_module,  # noqa: F401 - autouse fixture
    offline_project,
    replies,
    routes,
    write_project,
)
from tests.test_cli import text_of

runner = CliRunner()

#: A candidate that answers both cases correctly, so the gate it faces is
#: about its pass rate and not about cassette coverage: a single script would
#: have it look up ORD-9921 for the case about ORD-7310, which is a path the
#: baseline never recorded and would exit 4 before any threshold was read.
STRONG = candidate(
    "strong",
    routes("order_status", "ORD-9921"),
    replies(),
    extra="""extra_body.script_by_case.status_other = [
    { text = '{"intent": "order_status", "order_id": "ORD-7310"}' },
    { text = '{"final_answer": "Your order ORD-7310 is with DHL."}' },
]""",
)
WEAK = candidate("weak", behaves("not_json"))
UNREACHABLE = candidate("down", behaves("unreachable"))


_TEMPLATE: dict[str, Path] = {}


def project_with(
    candidates: str,
    tmp_path: Path,
    factory: pytest.TempPathFactory,
    *,
    repeats: int = 3,
    key: str = "",
) -> Path:
    """A finished run on disk, cloned from a template built once per candidate.

    The matrix runs through the harness and everything asserted afterwards goes
    through the CLI, so what these tests pin is the command's exit code - the
    whole contract of this module. Cloning rather than re-running keeps the
    suite inside its thirty-second budget; most of these tests write a baseline
    file or break the agent, so they cannot share one project.
    """
    name = key or candidates
    if name not in _TEMPLATE:
        _TEMPLATE[name] = build_and_run(
            factory.mktemp("gate-template"), candidates=candidates, repeats=repeats
        )
    return clone_project(_TEMPLATE[name], tmp_path / "project")


@pytest.fixture
def strong(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = project_with(STRONG, tmp_path, tmp_path_factory, key="strong")
    monkeypatch.chdir(root)
    return root


@pytest.fixture
def weak(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    root = project_with(WEAK, tmp_path, tmp_path_factory, key="weak")
    monkeypatch.chdir(root)
    return root


# --------------------------------------------------------------------------- #
# The acceptance criteria
# --------------------------------------------------------------------------- #


def test_a_weak_candidate_fails_the_gate(weak: Path) -> None:
    result = runner.invoke(app, ["gate", "--min-pass-k", "0.9", "--ascii"])
    assert result.exit_code == exit_codes.GATE_FAILED
    out = text_of(result)
    assert "GATE FAILED" in out
    assert "weak" in out
    assert "pass_k" in out


def test_an_unreachable_provider_exits_three_and_says_it_is_not_a_regression(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The distinction the whole exit-code contract exists for.

    If this ever merges with exit 1, a team learns that a red gate might just
    mean the GPU box was asleep - and then they stop reading it.
    """
    monkeypatch.chdir(
        project_with(UNREACHABLE, tmp_path, tmp_path_factory, repeats=1, key="down")
    )
    result = runner.invoke(app, ["gate", "--min-pass-k", "0.9", "--ascii"])

    assert result.exit_code == exit_codes.ENVIRONMENT
    out = text_of(result)
    assert "INFRASTRUCTURE PROBLEM" in out
    assert "not a regression" in out


def test_a_good_candidate_passes_and_exits_zero(strong: Path) -> None:
    result = runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--ascii"])
    assert result.exit_code == exit_codes.OK
    assert "GATE PASSED" in text_of(result)


def test_a_passing_gate_is_quiet_until_asked(strong: Path) -> None:
    """One line in a CI log when everything is fine; the whole list on -v,
    which is what you want once, while setting the thresholds up."""
    quiet = text_of(runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--ascii"]))
    assert "no forbidden tool was called" not in quiet

    loud = text_of(runner.invoke(app, ["gate", "--min-pass-k", "0.3", "-v", "--ascii"]))
    assert "no forbidden tool was called" in loud
    assert "forbidden_tools" in loud


def test_the_exit_codes_are_printed_where_somebody_will_read_them(strong: Path) -> None:
    """They are a contract people script against, so the command documents
    itself in the one place a CI log is actually read."""
    out = text_of(runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--ascii"]))
    assert "3 environment" in out
    assert "4 coverage" in out


# --------------------------------------------------------------------------- #
# Usage
# --------------------------------------------------------------------------- #


def test_a_gate_with_no_candidate_is_a_usage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A green gate that checked nothing is the one outcome worse than red."""
    with offline_project(tmp_path) as runs:
        runs.record()
    write_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["gate", "--min-pass-k", "0.9"])
    assert result.exit_code == exit_codes.USAGE
    assert "no candidate to gate" in text_of(result)


def test_an_unknown_format_is_rejected(strong: Path) -> None:
    result = runner.invoke(app, ["gate", "--format", "xlsx"])
    assert result.exit_code != exit_codes.OK


def test_the_configured_thresholds_are_used_with_no_flags(weak: Path) -> None:
    """CI should be able to call `slmeval gate` bare, with the thresholds the
    team agreed on living beside the benchmark where they get reviewed."""
    config = weak / "eval.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + '\n[gate]\nmin_pass_k = 0.9\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["gate", "--ascii"])
    assert result.exit_code == exit_codes.GATE_FAILED
    assert "pass_k" in text_of(result)


def test_a_flag_overrides_the_configured_threshold(weak: Path) -> None:
    config = weak / "eval.toml"
    config.write_text(
        config.read_text(encoding="utf-8") + '\n[gate]\nmin_pass_k = 0.9\n',
        encoding="utf-8",
    )
    assert runner.invoke(app, ["gate", "--min-pass-k", "0.0"]).exit_code == exit_codes.OK


# --------------------------------------------------------------------------- #
# Formats
# --------------------------------------------------------------------------- #


def test_json_is_parseable_and_carries_the_exit_code(weak: Path) -> None:
    result = runner.invoke(app, ["gate", "--min-pass-k", "0.9", "--format", "json"])
    payload = json.loads(result.output)
    assert payload["schema"] == "slmeval.gate/1"
    assert payload["passed"] is False
    assert payload["exit_code"] == exit_codes.GATE_FAILED
    assert payload["thresholds"]["use_wilson_lower"] is True
    assert any(c["name"] == "pass_k" and not c["passed"] for c in payload["checks"])


def test_markdown_leads_with_the_verdict(weak: Path) -> None:
    out = runner.invoke(
        app, ["gate", "--min-pass-k", "0.9", "--format", "md"]
    ).output
    assert out.lstrip().startswith("### slmeval gate")
    assert "**FAILED**" in out
    assert "| `weak` |" in out


def test_markdown_says_outright_when_it_is_not_a_regression(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The PR comment is where a reviewer meets the gate, so the distinction
    has to survive into Markdown and not only into the terminal."""
    monkeypatch.chdir(
        project_with(UNREACHABLE, tmp_path, tmp_path_factory, repeats=1, key="down")
    )
    out = runner.invoke(
        app, ["gate", "--min-pass-k", "0.9", "--format", "md"]
    ).output
    assert "not a model regression" in out


def test_junit_marks_a_skipped_check_as_skipped(weak: Path) -> None:
    """A dashboard should show an unmeasurable threshold as neither green nor
    red, which is exactly what JUnit's `skipped` means."""
    out = runner.invoke(
        app,
        ["gate", "--min-pass-k", "0.9", "--max-p95-ratio", "2.0", "--format", "junit"],
    ).output
    assert "<testsuites" in out
    assert "<skipped" in out
    assert 'name="p95_ratio"' in out


# --------------------------------------------------------------------------- #
# baseline save / show
# --------------------------------------------------------------------------- #


def test_baseline_save_writes_a_committable_file(strong: Path) -> None:
    result = runner.invoke(app, ["baseline", "save"])
    assert result.exit_code == exit_codes.OK
    path = strong / BASELINE_FILENAME
    assert path.is_file()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "slmeval.baseline/1"
    assert payload["dataset_hash"]
    assert "strong" in payload["candidates"]
    assert "commit it" in text_of(result)


def test_baseline_save_will_not_quietly_overwrite(strong: Path) -> None:
    """Moving a baseline is a decision somebody should review, so it takes a
    flag and leaves a diff."""
    runner.invoke(app, ["baseline", "save"])
    second = runner.invoke(app, ["baseline", "save"])
    assert second.exit_code == exit_codes.USAGE
    assert "--force" in text_of(second)
    assert runner.invoke(app, ["baseline", "save", "--force"]).exit_code == exit_codes.OK


def test_baseline_show_prints_the_stored_numbers(strong: Path) -> None:
    runner.invoke(app, ["baseline", "save"])
    out = text_of(runner.invoke(app, ["baseline", "show"]))
    assert "strong" in out
    assert "wilson lo" in out


def test_baseline_show_without_a_file_says_so(strong: Path) -> None:
    result = runner.invoke(app, ["baseline", "show"])
    assert result.exit_code == exit_codes.USAGE
    assert "no baseline file" in text_of(result)


def test_gate_uses_the_committed_baseline_for_the_regression_check(strong: Path) -> None:
    runner.invoke(app, ["baseline", "save"])
    result = runner.invoke(
        app,
        ["gate", "--min-pass-k", "0.3", "--max-regression", "0.05", "-v", "--ascii"],
    )
    assert result.exit_code == exit_codes.OK
    assert "held or improved" in text_of(result)


def test_gate_refuses_a_baseline_built_on_a_different_dataset(strong: Path) -> None:
    """The refusal the provenance hashes exist for.

    Comparing across a dataset edit publishes a regression nobody caused, and a
    tool that does that once is not believed again.
    """
    runner.invoke(app, ["baseline", "save"])
    path = strong / BASELINE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_hash"] = "0000000000000000"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--max-regression", "0.05"])
    assert result.exit_code == exit_codes.USAGE
    out = text_of(result)
    assert "the dataset changed" in out
    assert "--force" in out


def test_a_stale_baseline_does_not_block_a_gate_that_ignores_it(strong: Path) -> None:
    """A committed baseline is only read when --max-regression asks for it.

    Refusing a pass-rate gate because a file it was never going to open
    predates a dataset edit would be the tool blocking valid work over
    bookkeeping.
    """
    runner.invoke(app, ["baseline", "save"])
    path = strong / BASELINE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_hash"] = "0000000000000000"
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert (
        runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--ascii"]).exit_code
        == exit_codes.OK
    )


def test_force_compares_anyway(strong: Path) -> None:
    runner.invoke(app, ["baseline", "save"])
    path = strong / BASELINE_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["dataset_hash"] = "0000000000000000"
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = runner.invoke(
        app,
        ["gate", "--min-pass-k", "0.3", "--max-regression", "0.05", "--force", "--ascii"],
    )
    assert result.exit_code == exit_codes.OK


def test_a_named_baseline_that_is_missing_is_an_error(strong: Path) -> None:
    """A missing default file is fine - most gates do not check a regression.
    A missing file the user *named* is not: they meant to compare."""
    result = runner.invoke(
        app,
        [
            "gate",
            "--min-pass-k",
            "0.3",
            "--max-regression",
            "0.05",
            "--baseline",
            "nope.json",
        ],
    )
    assert result.exit_code == exit_codes.USAGE
    assert "no baseline file" in text_of(result)


def test_naming_a_baseline_with_no_allowance_is_a_usage_error(strong: Path) -> None:
    """Otherwise the gate would run while silently ignoring half the flags."""
    runner.invoke(app, ["baseline", "save"])
    result = runner.invoke(
        app, ["gate", "--min-pass-k", "0.3", "--baseline", BASELINE_FILENAME]
    )
    assert result.exit_code == exit_codes.USAGE
    assert "--max-regression" in text_of(result)


def test_gate_does_not_need_the_agent_to_import(strong: Path) -> None:
    (strong / "offline_agent.py").write_text("import nonexistent\n", encoding="utf-8")
    assert (
        runner.invoke(app, ["gate", "--min-pass-k", "0.3", "--ascii"]).exit_code
        == exit_codes.OK
    )
