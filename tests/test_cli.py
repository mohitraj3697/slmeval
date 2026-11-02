"""End-to-end CLI behaviour, including the exit-code contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner, Result

from slmeval import __version__, exit_codes
from slmeval.cli.main import app
from tests.conftest import MINIMAL_CONFIG, write_config

runner = CliRunner()


def text_of(result: Result) -> str:
    """Both streams, whichever way the installed Click separates them."""
    out = result.output or ""
    try:
        if result.stderr:
            out += result.stderr
    except (ValueError, AttributeError):  # pragma: no cover - version dependent
        pass
    return out


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == exit_codes.OK
    assert __version__ in result.output


def test_help_lists_working_and_planned_commands() -> None:
    out = text_of(runner.invoke(app, ["--help"]))
    for command in ("init", "doctor", "validate", "run", "gate"):
        assert command in out


def test_no_args_shows_help_rather_than_an_error() -> None:
    assert "Usage" in text_of(runner.invoke(app, []))


# -- init ------------------------------------------------------------------


def test_init_then_validate(tmp_path: Path, monkeypatch) -> None:
    """The cold start: no API key, no local model, still valid."""
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["init"]).exit_code == exit_codes.OK
    for name in ("eval.toml", "datasets/golden_cases.jsonl", "slmeval_target.py", ".gitignore"):
        assert (tmp_path / name).exists(), name
    assert runner.invoke(app, ["validate"]).exit_code == exit_codes.OK


def test_init_is_not_destructive(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    (tmp_path / "eval.toml").write_text("# mine\n", encoding="utf-8")

    out = text_of(runner.invoke(app, ["init"]))
    assert "skipped" in out
    assert (tmp_path / "eval.toml").read_text(encoding="utf-8") == "# mine\n"

    runner.invoke(app, ["init", "--force"])
    assert "[benchmark]" in (tmp_path / "eval.toml").read_text(encoding="utf-8")


def test_init_appends_to_an_existing_gitignore(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text("*.pyc\n", encoding="utf-8")
    runner.invoke(app, ["init"])
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert "*.pyc" in content
    assert ".slmeval/" in content


def test_init_does_not_duplicate_the_ignore_entry(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".gitignore").write_text(".slmeval/\n", encoding="utf-8")
    runner.invoke(app, ["init"])
    content = (tmp_path / ".gitignore").read_text(encoding="utf-8")
    assert content.count(".slmeval/") == 1


def test_init_names_the_benchmark_after_the_directory(tmp_path: Path, monkeypatch) -> None:
    project = tmp_path / "Refund Agent"
    project.mkdir()
    monkeypatch.chdir(project)
    runner.invoke(app, ["init"])
    assert 'name = "refund-agent-slm-downsizing"' in (project / "eval.toml").read_text(
        encoding="utf-8"
    )


# -- validate --------------------------------------------------------------


def test_validate_reports_the_matrix_size(project: Path, monkeypatch) -> None:
    monkeypatch.chdir(project)
    out = text_of(runner.invoke(app, ["validate"]))
    assert "executions" in out


def test_validate_json_carries_the_hashes_ci_keys_on(project: Path, monkeypatch) -> None:
    """A CI workflow keys its cassette cache on the dataset's content hash.

    Keying on anything else - a branch name, a commit - restores a store
    recorded against a different dataset, and every uncovered path then reads
    as a model regression. So the hash has to be obtainable without running the
    benchmark, and parseable without Rich in the way.
    """
    monkeypatch.chdir(project)
    result = runner.invoke(app, ["validate", "--json"])
    assert result.exit_code == exit_codes.OK

    payload = json.loads(result.output)
    assert payload["schema"] == "slmeval.validate/1"
    assert payload["dataset_hash"] and payload["config_hash"]
    assert payload["cases"] >= 1
    assert payload["executions"] == (
        (1 + len(payload["candidates"])) * payload["cases"] * payload["runs_per_case"]
    )
    assert payload["adapter"] in {"generic", "langgraph"}

    # The same dataset hashes the same twice, or the cache key is a lottery.
    again = json.loads(runner.invoke(app, ["validate", "--json"]).output)
    assert again["dataset_hash"] == payload["dataset_hash"]

    # And a changed dataset is a different key.
    dataset = Path(payload["dataset"])
    dataset.write_text(
        dataset.read_text(encoding="utf-8")
        + '{"case_id": "extra", "input": {"q": "x"}, "assert": {"state": {"a": 1}}}\n',
        encoding="utf-8",
    )
    edited = json.loads(runner.invoke(app, ["validate", "--json"]).output)
    assert edited["dataset_hash"] != payload["dataset_hash"]


def test_validate_warns_about_a_misspelled_target_node(tmp_path: Path, monkeypatch) -> None:
    """The silent killer: a typo here means the candidate changes nothing and
    scores identically to the baseline."""
    body = MINIMAL_CONFIG + """
[[candidates]]
id = "c"
provider = "fake"
model = "m"
target_nodes = ["intent_routr"]
"""
    path = write_config(tmp_path, body)
    (tmp_path / "datasets" / "cases.jsonl").write_text(
        '{"case_id": "c1", "input": {"q": "x"}, '
        '"assert": {"reference_trajectory": [["intent_router"]]}}\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate", "--config", str(path)])
    out = text_of(result)
    assert "intent_routr" in out
    assert "scores identically" in out or "changes nothing" in out
    assert result.exit_code == exit_codes.OK  # a warning, not a failure


def test_strict_turns_warnings_into_a_failure(tmp_path: Path, monkeypatch) -> None:
    body = MINIMAL_CONFIG.replace('name = "t"', 'name = "t"\nruns_per_case = 1')
    path = write_config(tmp_path, body)
    (tmp_path / "slmeval_target.py").write_text(
        "async def run_case(case_input):\n    return {}\n", encoding="utf-8"
    )
    body = (tmp_path / "eval.toml").read_text(encoding="utf-8")
    (tmp_path / "eval.toml").write_text(
        body.replace('entrypoint = "mod:run_case"', 'entrypoint = "slmeval_target:run_case"'),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    assert runner.invoke(app, ["validate", "--config", str(path)]).exit_code == exit_codes.OK
    assert (
        runner.invoke(app, ["validate", "--config", str(path), "--strict"]).exit_code
        == exit_codes.USAGE
    )


def test_strict_reports_an_unimportable_target(project: Path, monkeypatch) -> None:
    monkeypatch.chdir(project)
    (project / "slmeval_target.py").unlink()
    result = runner.invoke(app, ["validate", "--strict"])
    assert result.exit_code == exit_codes.USAGE
    assert "could not import" in text_of(result)


def test_broken_config_exits_usage_and_names_the_field(tmp_path: Path, monkeypatch) -> None:
    path = write_config(tmp_path, MINIMAL_CONFIG.replace('provider = "fake"', 'provider = "nope"'))
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate", "--config", str(path)])
    assert result.exit_code == exit_codes.USAGE
    assert "baseline.provider" in text_of(result)


def test_missing_config_exits_usage(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["validate"])
    assert result.exit_code == exit_codes.USAGE


# -- doctor ----------------------------------------------------------------


def test_doctor_offline_passes(project: Path, monkeypatch) -> None:
    monkeypatch.chdir(project)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(app, ["doctor", "--skip-network"])
    assert result.exit_code == exit_codes.OK


def test_doctor_without_a_config_still_reports_the_machine(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--skip-network"])
    assert result.exit_code == exit_codes.OK
    assert "slmeval init" in text_of(result)


def test_doctor_reports_a_broken_config_as_a_failed_check(project: Path, monkeypatch) -> None:
    """Doctor's job is to report problems, not to crash on them."""
    monkeypatch.chdir(project)
    (project / "eval.toml").write_text("[benchmark]\nname = \n", encoding="utf-8")
    result = runner.invoke(app, ["doctor", "--skip-network"])
    assert result.exit_code == exit_codes.ENVIRONMENT
    assert "slmeval validate" in text_of(result)


def test_doctor_missing_key_is_environment_not_regression(tmp_path: Path, monkeypatch) -> None:
    """Exit 3, never 1 — CI must tell a missing key from a model regression."""
    body = MINIMAL_CONFIG.replace(
        '[baseline]\nprovider = "fake"\nmodel = "m"',
        '[baseline]\nprovider = "openai"\nmodel = "gpt-4o"',
    )
    path = write_config(tmp_path, body)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["doctor", "--skip-network", "--config", str(path)])
    assert result.exit_code == exit_codes.ENVIRONMENT
    assert "OPENAI_API_KEY" in text_of(result)


# -- dataset ---------------------------------------------------------------


def test_dataset_validate(project: Path) -> None:
    result = runner.invoke(
        app, ["dataset", "validate", str(project / "datasets" / "golden_cases.jsonl")]
    )
    assert result.exit_code == exit_codes.OK
    assert "valid" in text_of(result)


def test_dataset_validate_reports_problems(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    result = runner.invoke(app, ["dataset", "validate", str(path)])
    assert result.exit_code == exit_codes.USAGE
    assert "invalid JSON" in text_of(result)


# -- planned commands ------------------------------------------------------


@pytest.mark.parametrize("command", ["compare", "cassette"])
def test_planned_commands_explain_themselves(command: str) -> None:
    result = runner.invoke(app, [command])
    assert result.exit_code == exit_codes.USAGE
    assert "not implemented yet" in text_of(result)


def test_planned_commands_tolerate_arguments() -> None:
    """A scripted invocation should get the message, not a usage error."""
    result = runner.invoke(app, ["compare", "--baseline", "a", "--candidate", "b"])
    assert "not implemented yet" in text_of(result)


@pytest.mark.parametrize("command", ["record", "run", "report", "gate"])
def test_week_two_commands_are_real_now(command: str) -> None:
    """They need a project, so they fail on a missing config - not on being stubs."""
    result = runner.invoke(app, [command, "--config", "nope.toml"])
    assert result.exit_code == exit_codes.USAGE
    assert "config file not found" in text_of(result)
    assert "not implemented yet" not in text_of(result)


@pytest.mark.parametrize(
    "command",
    [
        ["show", "run"],
        ["show", "exec", "x"],
        ["show", "case", "x"],
        ["show", "uncovered"],
        ["baseline", "save"],
        ["baseline", "show"],
    ],
)
def test_week_three_subcommands_are_real_now(command: list[str]) -> None:
    result = runner.invoke(app, [*command, "--config", "nope.toml"])
    assert result.exit_code == exit_codes.USAGE
    assert "config file not found" in text_of(result)
    assert "not implemented yet" not in text_of(result)
