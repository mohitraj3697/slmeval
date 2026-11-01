"""``slmeval record``, ``run`` and ``report`` from the outside.

The whole of Week 2's exit criterion, exercised the way a user meets it: three
commands in a project directory, offline, with no API key and no GPU. If this
file passes, the demo works.

Exit codes are a documented contract that people script against, so they are
asserted rather than ignored - and in particular an unreachable provider has to
exit 3, not 1, because a CI job that reports "the model server was down" as a
regression teaches people to ignore the gate.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmeval import exit_codes
from slmeval.cli.main import app
from slmeval.storage.db import RunStore

runner = CliRunner()

AGENT = '''
from __future__ import annotations

from pydantic import BaseModel

from slmeval import chat_model, node, sandbox_tool


class Routing(BaseModel):
    intent: str


@sandbox_tool(match_on=["intent"])
async def lookup(intent: str) -> dict:
    return {"found": intent.upper()}


async def run_case(case_input: dict) -> dict:
    with node("router"):
        completion = await chat_model().complete(
            [{"role": "user", "content": str(case_input.get("q", ""))}],
            schema=Routing,
        )
    routing = completion.parsed_as(Routing)
    with node("lookup"):
        found = await lookup(routing.intent)
    return {"intent": routing.intent, "final_answer": f"found {found['found']}"}
'''

CONFIG = """
[benchmark]
name = "cli-demo"
dataset = "./cases.jsonl"
runs_per_case = 2
tool_mode = "replay"
seed = 42
concurrency = 2
timeout_s = 10.0
max_steps = 8

[target]
entrypoint = "cli_agent:run_case"

[baseline]
provider = "fake"
model = "good"
extra_body.script = [{ text = '{"intent": "ok"}' }]

[[candidates]]
id = "weak"
provider = "fake"
model = "weak"
extra_body.script = [{ text = '{"intent": "ok"}' }]
"""

CASES = (
    '{"case_id": "c0", "input": {"q": "a"}, '
    '"assert": {"state_contains": {"intent": "ok"}, "required_tools": ["lookup"]}}\n'
    '{"case_id": "c1", "input": {"q": "b"}, '
    '"assert": {"state_contains": {"intent": "ok"}}}\n'
)


@pytest.fixture
def demo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    (tmp_path / "cli_agent.py").write_text(AGENT, encoding="utf-8")
    (tmp_path / "eval.toml").write_text(CONFIG, encoding="utf-8")
    (tmp_path / "cases.jsonl").write_text(CASES, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    try:
        yield tmp_path
    finally:
        sys.modules.pop("cli_agent", None)


def text_of(result: object) -> str:
    return getattr(result, "output", "") or ""


def invoke(*args: str) -> object:
    return runner.invoke(app, [*args, "--ascii"])


# --------------------------------------------------------------------------- #
# record
# --------------------------------------------------------------------------- #


def test_record_captures_the_baselines_tool_calls(demo: Path) -> None:
    result = invoke("record")
    assert result.exit_code == exit_codes.OK, text_of(result)
    assert "recorded" in text_of(result)
    assert "no network, no API keys, no GPU" in text_of(result)
    assert (demo / ".slmeval" / "cassettes.sqlite").is_file()


def test_record_runs_only_the_baseline(demo: Path) -> None:
    """Letting a candidate record the cassettes would bake its mistakes into
    the reference every other candidate is measured against."""
    invoke("record")
    with RunStore.for_project(demo) as store:
        latest = store.latest_run()
        assert latest is not None
        executions = store.executions(latest.run_id)
        assert {e.candidate_id for e in executions} == {"baseline"}
        # One repeat, not runs_per_case: cassettes are scoped per case, so
        # recording three times would overwrite the same rows twice while
        # tripling the calls to someone's real API.
        assert {e.repeat_idx for e in executions} == {0}


def test_a_case_that_cannot_be_recorded_exits_with_a_coverage_error(
    demo: Path,
) -> None:
    """No coverage means every candidate will miss on that case and read as a
    model failure. Better to say so than let it happen quietly."""
    (demo / "cases.jsonl").write_text(
        CASES
        + '{"case_id": "boom", "input": {"q": "boom"}, '
        '"assert": {"state_contains": {"intent": "never"}}}\n',
        encoding="utf-8",
    )
    result = invoke("record")
    assert result.exit_code == exit_codes.COVERAGE
    assert "were not recorded" in text_of(result)
    assert "boom" in text_of(result)


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def test_the_whole_demo_works_offline(demo: Path) -> None:
    """Week 2's exit criterion: record once, then run produces the table."""
    assert invoke("record").exit_code == exit_codes.OK

    result = invoke("run")
    assert result.exit_code == exit_codes.OK, text_of(result)
    out = text_of(result)
    assert "cli-demo" in out
    assert "Candidate" in out
    assert "pass^2" in out
    assert "95% CI" in out
    assert "gate on the lower bound" in out
    assert "per-node attribution" in out


def test_run_is_persisted_and_reportable(demo: Path) -> None:
    invoke("record")
    invoke("run")
    with RunStore.for_project(demo) as store:
        latest = store.latest_run()
        assert latest is not None
        # 2 candidates x 2 cases x 2 repeats.
        assert len(store.executions(latest.run_id)) == 8
        assert len(store.rollups(latest.run_id)) == 4
        assert store.metrics(latest.run_id)


def test_dry_run_plans_without_executing(demo: Path) -> None:
    result = invoke("run", "--dry-run")
    assert result.exit_code == exit_codes.OK
    payload = json.loads(text_of(result).strip())
    assert payload["to_run"] == 8
    assert payload["candidates"] == ["baseline", "weak"]
    with RunStore.for_project(demo) as store:
        assert store.latest_run() is None


def test_run_can_narrow_to_one_case_and_one_candidate(demo: Path) -> None:
    invoke("record")
    result = invoke("run", "--case", "c0", "--candidate", "weak", "--runs", "1")
    assert result.exit_code == exit_codes.OK, text_of(result)
    with RunStore.for_project(demo) as store:
        latest = store.latest_run()
        assert latest is not None
        executions = store.executions(latest.run_id)
        # The baseline comes along whether or not it was asked for: every delta
        # is measured against it.
        assert {e.candidate_id for e in executions} == {"baseline", "weak"}
        assert {e.case_id for e in executions} == {"c0"}


def test_run_writes_the_json_report_when_asked(demo: Path) -> None:
    invoke("record")
    result = invoke("run", "--json", "report.json")
    assert result.exit_code == exit_codes.OK
    payload = json.loads((demo / "report.json").read_text(encoding="utf-8"))
    assert payload["schema"].startswith("slmeval.report/")
    assert payload["run"]["name"] == "cli-demo"


def test_a_bad_tool_mode_is_rejected_before_anything_runs(demo: Path) -> None:
    result = invoke("run", "--tool-mode", "pretend")
    assert result.exit_code != exit_codes.OK
    assert "tool mode must be one of" in text_of(result)


def test_an_unreachable_provider_exits_three_not_one(demo: Path) -> None:
    """The distinction that stops "Ollama is down" from reading as "the model
    regressed"."""
    (demo / "eval.toml").write_text(
        CONFIG.replace(
            """[baseline]
provider = "fake"
model = "good"
extra_body.script = [{ text = '{"intent": "ok"}' }]""",
            """[baseline]
provider = "openai_compatible"
model = "nothing-here"
base_url = "http://127.0.0.1:9"
""",
        ).replace(
            """[[candidates]]
id = "weak"
provider = "fake"
model = "weak"
extra_body.script = [{ text = '{"intent": "ok"}' }]""",
            "",
        ),
        encoding="utf-8",
    )
    result = invoke("run", "--runs", "1")
    assert result.exit_code == exit_codes.ENVIRONMENT, text_of(result)


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #


def test_report_re_renders_the_stored_run(demo: Path) -> None:
    invoke("record")
    first = invoke("run")
    again = invoke("report")
    assert again.exit_code == exit_codes.OK, text_of(again)
    # The table a finished run printed and the one `report` prints have to be
    # the same table, or the trace store is not the asset.
    assert _table_of(text_of(first)) == _table_of(text_of(again))


def _table_of(output: str) -> str:
    lines = output.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip().startswith("Candidate"))
    end = next(
        (
            i
            for i, line in enumerate(lines[start:], start)
            if "pass^" in line and "fraction" in line
        ),
        len(lines),
    )
    return "\n".join(line.rstrip() for line in lines[start:end]).strip()


def test_report_works_without_the_agent_importing(demo: Path) -> None:
    """A checkout whose agent has since been refactored still has valid results,
    and refusing to print them would be the tool losing data it already has."""
    invoke("record")
    invoke("run")
    (demo / "cli_agent.py").write_text("raise ImportError('refactored away')\n", encoding="utf-8")
    sys.modules.pop("cli_agent", None)

    result = invoke("report")
    assert result.exit_code == exit_codes.OK, text_of(result)
    assert "Candidate" in text_of(result)


def test_report_can_export_json_md_and_csv(demo: Path) -> None:
    invoke("record")
    invoke("run")

    payload = json.loads(text_of(runner.invoke(app, ["report", "--format", "json"])))
    assert payload["run"]["name"] == "cli-demo"

    markdown = text_of(runner.invoke(app, ["report", "--format", "md"]))
    assert markdown.lstrip().startswith("### cli-demo")

    csv_text = text_of(runner.invoke(app, ["report", "--format", "csv"]))
    assert csv_text.splitlines()[0].startswith("run_id,candidate_id")


def test_report_can_write_to_a_file(demo: Path) -> None:
    invoke("record")
    invoke("run")
    result = runner.invoke(
        app, ["report", "--format", "md", "--out", "report.md", "--ascii"]
    )
    assert result.exit_code == exit_codes.OK
    assert (demo / "report.md").read_text(encoding="utf-8").startswith("### cli-demo")


def test_report_with_no_runs_says_so(demo: Path) -> None:
    result = invoke("report")
    assert result.exit_code == exit_codes.USAGE
    assert "no runs stored yet" in text_of(result)


def test_report_rejects_an_unknown_format(demo: Path) -> None:
    result = invoke("report", "--format", "yaml")
    assert result.exit_code != exit_codes.OK
    assert "format must be one of" in text_of(result)


def test_report_shows_the_per_case_table_on_request(demo: Path) -> None:
    invoke("record")
    invoke("run")
    result = invoke("report", "--cases")
    assert result.exit_code == exit_codes.OK
    assert "c0" in text_of(result)
    assert "2/2" in text_of(result)
