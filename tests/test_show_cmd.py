"""``slmeval show`` end to end: the four views, and what they refuse.

The behaviours worth pinning are the refusals. A `show` that guessed - picked
the newest run matching a prefix, or diffed against a baseline repeat that had
itself failed - would hand a user a comparison they would act on and that was
never true.

These invoke the real CLI against a real stored run, so the wiring is under
test and not just the renderers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmeval import exit_codes
from slmeval.cli.main import app
from tests.offline import (
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

SLOPPY = candidate("sloppy-router", routes("order_status", "9921"), replies())

_TEMPLATE: dict[str, Path] = {}


@pytest.fixture
def stored(
    tmp_path: Path,
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """A project with one recorded baseline and one sloppy candidate, run.

    Built through the offline harness and then handed to the CLI as a
    directory, because every assertion below is about the command rather than
    the runner - and the CLI has to find the config, the store and the
    cassettes by itself.

    Built once and cloned per test. Several of these tests mutate the project -
    they break the agent, or add a second run - so they cannot share one, and
    running the matrix twenty times over would put a quarter of the whole
    suite's thirty-second budget into this one file.
    """
    if "show" not in _TEMPLATE:
        _TEMPLATE["show"] = build_and_run(
            tmp_path_factory.mktemp("show-template"), candidates=SLOPPY, repeats=2
        )
    root = clone_project(_TEMPLATE["show"], tmp_path / "project")
    monkeypatch.chdir(root)
    return root


def run_id_of(stored: Path) -> str:
    payload = json.loads(text_of(runner.invoke(app, ["show", "run", "--json"])))
    return str(payload["run_id"])


# --------------------------------------------------------------------------- #
# show run
# --------------------------------------------------------------------------- #


def test_show_run_lists_the_failures_first(stored: Path) -> None:
    result = runner.invoke(app, ["show", "run", "--ascii"])
    assert result.exit_code == exit_codes.OK
    out = text_of(result)
    assert "cassette_miss" in out
    # The cells a user came to find are above the passes.
    assert out.index("cassette_miss") < out.index("passed")


def test_show_run_can_narrow_to_the_failures(stored: Path) -> None:
    out = text_of(runner.invoke(app, ["show", "run", "--failed", "--ascii"]))
    assert "cassette_miss" in out
    assert "passed" not in out


def test_show_run_as_json_is_parseable(stored: Path) -> None:
    payload = json.loads(text_of(runner.invoke(app, ["show", "run", "--json"])))
    assert payload["mode"] == "replay"
    assert {c["candidate_id"] for c in payload["candidates"]} == {
        "baseline",
        "sloppy-router",
    }
    assert payload["executions"]


def test_show_run_refuses_an_ambiguous_prefix(stored: Path) -> None:
    """Two runs starting with the same digits must not resolve to one of them.

    Picking the newest would eventually print a report for a run the user was
    not asking about, which is worse than making them type four more
    characters.
    """
    runner.invoke(app, ["run", "--tool-mode", "replay"])
    result = runner.invoke(app, ["show", "run", "2"])
    assert result.exit_code == exit_codes.USAGE
    assert "matches" in text_of(result)


def test_show_run_without_any_run_says_so(tmp_path: Path, monkeypatch) -> None:
    write_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["show", "run"])
    assert result.exit_code == exit_codes.USAGE
    assert "no runs stored yet" in text_of(result)


# --------------------------------------------------------------------------- #
# show exec
# --------------------------------------------------------------------------- #


def test_show_exec_takes_the_readable_tail(stored: Path) -> None:
    """``candidate.case.rN`` is what a user reads off the run table, and it has
    to be enough - re-typing the run id in front of it is friction for no gain."""
    result = runner.invoke(
        app, ["show", "exec", "sloppy-router.status_shipped.r0", "--ascii"]
    )
    assert result.exit_code == exit_codes.OK
    out = text_of(result)
    assert "trajectory" in out
    assert "intent_router" in out
    # The diagnosis block, rebuilt from the cassette store rather than from the
    # one line the exception left in the database.
    assert "nearest recorded" in out
    assert "dropped prefix" in out


def test_show_exec_as_json_carries_the_timeline(stored: Path) -> None:
    payload = json.loads(
        text_of(
            runner.invoke(app, ["show", "exec", "baseline.status_shipped.r0", "--json"])
        )
    )
    assert payload["outcome"] == "passed"
    assert payload["judgeable"] is True
    assert [c["node"] for c in payload["model_calls"]] == [
        "intent_router",
        "compose_reply",
    ]
    assert payload["tool_calls"][0]["tool"] == "fetch_order"


def test_show_exec_rejects_an_unknown_id(stored: Path) -> None:
    result = runner.invoke(app, ["show", "exec", "nope.nope.r9"])
    assert result.exit_code == exit_codes.USAGE
    assert "no execution id matches" in text_of(result)


# --------------------------------------------------------------------------- #
# show case
# --------------------------------------------------------------------------- #


def test_show_case_diffs_against_the_baseline(stored: Path) -> None:
    result = runner.invoke(
        app,
        ["show", "case", "status_shipped", "--candidate", "sloppy-router", "--ascii"],
    )
    assert result.exit_code == exit_codes.OK
    out = text_of(result)
    assert "trajectory" in out
    assert "ORD-9921" in out  # the baseline's call, on the left
    assert "hallucinated_parameter" in out
    assert "diverged at step" in out


def test_show_case_defaults_to_every_candidate(stored: Path) -> None:
    out = text_of(runner.invoke(app, ["show", "case", "status_shipped", "--ascii"]))
    assert "sloppy-router" in out
    assert "baseline -> sloppy-router" in out


def test_show_case_json_marks_the_diverging_step(stored: Path) -> None:
    payload = json.loads(
        text_of(runner.invoke(app, ["show", "case", "status_shipped", "--json"]))
    )
    diff = payload["diffs"][0]
    assert diff["comparable"] is True
    assert diff["identical"] is False
    assert diff["fault_parity"] is True
    assert any(row["op"] != "equal" for row in diff["tool_calls"])


def test_show_case_rejects_an_unknown_candidate(stored: Path) -> None:
    result = runner.invoke(
        app, ["show", "case", "status_shipped", "--candidate", "ghost"]
    )
    assert result.exit_code == exit_codes.USAGE
    assert "no executions for candidate" in text_of(result)


def test_show_case_rejects_an_unknown_case(stored: Path) -> None:
    result = runner.invoke(app, ["show", "case", "not_a_case"])
    assert result.exit_code == exit_codes.USAGE
    assert "has no execution for case" in text_of(result)


def test_show_case_rejects_a_repeat_that_was_never_run(stored: Path) -> None:
    result = runner.invoke(
        app, ["show", "case", "status_shipped", "--repeat", "7"]
    )
    assert result.exit_code == exit_codes.USAGE
    assert "no repeat 7" in text_of(result)


# --------------------------------------------------------------------------- #
# show uncovered
# --------------------------------------------------------------------------- #


def test_show_uncovered_lists_only_the_coverage_gaps(stored: Path) -> None:
    """The default list is what `record --fill-gaps` should go and record.

    A hallucinated order id in it would be re-recorded as ground truth, and
    every later run would grade against the model's own mistake.
    """
    out = text_of(runner.invoke(app, ["show", "uncovered", "--ascii"]))
    assert "unrecorded_valid_path" in out
    assert "hallucinated_parameter" not in out
    assert "record --fill-gaps" in out


def test_show_uncovered_all_includes_the_hallucinations(stored: Path) -> None:
    out = text_of(runner.invoke(app, ["show", "uncovered", "--all", "--ascii"]))
    assert "hallucinated_parameter" in out
    assert "unrecorded_valid_path" in out


def test_show_uncovered_json_separates_ours_from_theirs(stored: Path) -> None:
    payload = json.loads(
        text_of(runner.invoke(app, ["show", "uncovered", "--all", "--json"]))
    )
    assert payload["coverage_gaps"] >= 1
    assert payload["hallucinated_calls"] >= 1


def test_show_uncovered_on_a_fully_covered_run_is_not_a_warning(
    tmp_path: Path, monkeypatch
) -> None:
    """A clean run has to read as clean. A command that printed an empty table
    with a scary heading would train people to ignore the real one."""
    with offline_project(tmp_path) as runs:
        runs.record()
        runs.run(tool_mode="replay")
    write_project(tmp_path)
    monkeypatch.chdir(tmp_path)

    out = text_of(runner.invoke(app, ["show", "uncovered", "--ascii"]))
    assert "covered by a recording" in out


# --------------------------------------------------------------------------- #
# Independence from the agent
# --------------------------------------------------------------------------- #


def test_show_works_after_the_agent_stops_importing(stored: Path) -> None:
    """Results from last Tuesday survive a refactor of the agent.

    ``report`` makes this promise and so does ``show``: refusing to open stored
    traces because the code that produced them has moved on would be the tool
    losing data it already holds.
    """
    (stored / "offline_agent.py").write_text("import nonexistent_module\n", encoding="utf-8")
    assert runner.invoke(app, ["show", "run", "--ascii"]).exit_code == exit_codes.OK
    assert (
        runner.invoke(
            app, ["show", "case", "status_shipped", "--ascii"]
        ).exit_code
        == exit_codes.OK
    )
