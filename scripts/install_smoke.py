"""Install the built wheel into a fresh virtualenv and use it like a user would.

Step 15's last acceptance criterion, and the one the test suite cannot give us:
`pytest` runs against the working tree, so it proves nothing about what `pip
install slmeval` actually ships. A missing package in `[tool.hatch.build]`, a
data file that never made it into the wheel, a console script that is not
executable on Windows - all of those are green in pytest and broken for every
user.

So this runs outside the suite, against the artifact:

1. a throwaway venv, with nothing but the wheel in it;
2. `slmeval init` + `validate` + `doctor` in an empty directory - the first
   three minutes of a new user, which must work with no API key and no GPU;
3. a scripted benchmark of our own, run end to end with `record`, `run` and
   `report`, which must exit 0 and print a scorecard;
4. the drill-down (`show run`, `show case`) and the gate, including that a
   floor nothing can clear exits exactly 1 - the number CI branches on.

Nothing here touches the network or a model server. `doctor` is allowed to exit
3 (environment) because a machine with no Ollama is the normal case in CI; every
other command must exit 0.

    python scripts/install_smoke.py                 # build the wheel, then test it
    python scripts/install_smoke.py --wheel dist/slmeval_cli-0.1.0-py3-none-any.whl
    python scripts/install_smoke.py --keep          # leave the temp dir for poking at
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

COLD_START_BUDGET_S = 300.0
"""The README's promise: `pip install` to a scorecard in under five minutes,
with no API key. Measured, because it is the whole funnel - and because a
dependency resolution that quietly grew to four minutes would not show up in
any other check here."""

AGENT = '''\
from pydantic import BaseModel

from slmeval import chat_model, node, sandbox_tool


class Routing(BaseModel):
    intent: str
    order_id: str


@sandbox_tool(match_on=["order_id"])
async def fetch_order(order_id: str) -> dict:
    return {"order_id": order_id, "status": "shipped"}


async def run_case(case_input: dict) -> dict:
    with node("intent_router"):
        routing = (
            await chat_model().complete(
                [{"role": "user", "content": str(case_input.get("user_query", ""))}],
                schema=Routing,
            )
        ).parsed_as(Routing)
    with node("order_lookup"):
        order = await fetch_order(routing.order_id)
    return {
        "intent": routing.intent,
        "order_id": routing.order_id,
        "final_answer": f"Order {order['order_id']} is {order['status']}.",
    }
'''

CONFIG = """\
[benchmark]
name = "install-smoke"
dataset = "./cases.jsonl"
runs_per_case = 2
tool_mode = "replay"
seed = 42
timeout_s = 30.0
max_steps = 8

[target]
entrypoint = "smoke_agent:run_case"

# The scripted provider, so this needs no key, no GPU and no network.
[baseline]
provider = "fake"
model = "scripted"
extra_body.script = [{ text = '{"intent": "order_status", "order_id": "ORD-9921"}' }]

# A candidate that answers correctly but cannot emit bare JSON. It passes, and
# the scorecard says it only passed with help - which is the whole product in
# one row, and worth asserting the installed wheel can still produce.
[[candidates]]
id = "fenced-json"
provider = "fake"
model = "fenced-json"
extra_body.unsupported = ["tools", "json_schema"]
extra_body.script = [
    { kind = "fenced_json", text = '{"intent": "order_status", "order_id": "ORD-9921"}' },
]
"""

CASES = (
    '{"case_id": "status_shipped", "input": {"user_query": "Where is ORD-9921?"}, '
    '"assert": {"state_contains": {"intent": "order_status"}, '
    '"required_tools": ["fetch_order"], "max_tool_calls": 2}}\n'
)


class Failed(RuntimeError):
    pass


def run(
    command: list[str],
    *,
    cwd: Path,
    allow: tuple[int, ...] = (0,),
    env: dict[str, str] | None = None,
    stdout_only: bool = False,
) -> str:
    """Run a command and return what it printed.

    ``stdout_only`` for the commands whose output is *data* - ``--json`` and
    ``--format junit`` are promises about a pipe, and a single warning on
    stderr folded into the same string would make this script's own parsing
    lie about whether the promise held.
    """
    printable = " ".join(Path(c).name if os.sep in c else c for c in command)
    print(f"  $ {printable}", flush=True)
    done = subprocess.run(
        command,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        env={**os.environ, **(env or {})},
    )
    both = (done.stdout or "") + (done.stderr or "")
    if done.returncode not in allow:
        raise Failed(
            f"{printable} exited {done.returncode}, expected one of {allow}\n{both}"
        )
    return (done.stdout or "") if stdout_only else both


def venv_bin(root: Path) -> Path:
    """The venv's script directory, under its own name on each platform."""
    return root / ("Scripts" if os.name == "nt" else "bin")


def build_wheel() -> Path:
    """Build a wheel into dist/ and return the newest one.

    `uv build` when uv is there, `python -m build` otherwise: a uv-managed
    virtualenv has no pip in it, so insisting on one would make this script
    unrunnable in the environment the project is actually developed in.
    """
    print("building the wheel")
    uv = shutil.which("uv")
    if uv:
        run([uv, "build", "--wheel"], cwd=REPO)
    else:
        run([sys.executable, "-m", "pip", "install", "--quiet", "build"], cwd=REPO)
        run([sys.executable, "-m", "build", "--wheel"], cwd=REPO)

    wheels = sorted((REPO / "dist").glob("slmeval_cli-*.whl"), key=lambda p: p.stat().st_mtime)
    if not wheels:
        raise Failed("no wheel in dist/ after building one")
    return wheels[-1]


def fresh_venv(root: Path, wheel: Path) -> Path:
    """A venv holding the wheel and nothing else. Returns the slmeval command."""
    print(f"installing {wheel.name} into a fresh venv")
    uv = shutil.which("uv")
    if uv:
        run([uv, "venv", "--quiet", str(root)], cwd=REPO)
    else:
        run([sys.executable, "-m", "venv", str(root)], cwd=REPO)

    python = venv_bin(root) / ("python.exe" if os.name == "nt" else "python")
    if uv:
        run([uv, "pip", "install", "--quiet", str(wheel)], cwd=REPO, env={"VIRTUAL_ENV": str(root)})
    else:
        run([str(python), "-m", "pip", "install", "--quiet", str(wheel)], cwd=REPO)

    command = venv_bin(root) / ("slmeval.exe" if os.name == "nt" else "slmeval")
    if not command.exists():
        raise Failed(f"the wheel installed no console script at {command}")
    return command


def check_first_run(slmeval: Path, root: Path) -> None:
    """`init`, `validate`, `doctor` in an empty directory."""
    print("first run: init, validate, doctor")
    run([str(slmeval), "init", "--name", "smoke"], cwd=root)
    for name in ("eval.toml", "slmeval_target.py"):
        if not (root / name).exists():
            raise Failed(f"init did not write {name}")
    run([str(slmeval), "validate"], cwd=root)

    # The machine-readable form, which is what a CI workflow keys its cassette
    # cache on. Rich must stay out of it or no parser can read it.
    payload = json.loads(
        run([str(slmeval), "validate", "--json"], cwd=root, stdout_only=True)
    )
    for key in ("dataset_hash", "config_hash", "cases"):
        if not payload.get(key):
            raise Failed(f"validate --json carried no {key}: {payload}")
    # 3 is "no model server here", which is the normal state of a CI runner and
    # not a broken install.
    doctor = run([str(slmeval), "doctor"], cwd=root, allow=(0, 3))
    if "slmeval" not in doctor:
        raise Failed(f"doctor printed nothing recognisable:\n{doctor}")


def check_scorecard(slmeval: Path, root: Path) -> None:
    """A scripted benchmark, end to end, exactly as the README promises."""
    print("benchmark: record, run, report")
    (root / "smoke_agent.py").write_text(AGENT, encoding="utf-8")
    (root / "eval.toml").write_text(CONFIG, encoding="utf-8")
    (root / "cases.jsonl").write_text(CASES, encoding="utf-8")

    run([str(slmeval), "record", "--ascii"], cwd=root)
    output = run([str(slmeval), "run", "--ascii"], cwd=root)
    for expected in ("Candidate", "pass^2", "fenced-json"):
        if expected not in output:
            raise Failed(f"the scorecard is missing {expected!r}:\n{output}")
    if "0 pass" in output:
        raise Failed(f"nothing passed, so the wheel is not replaying tool calls:\n{output}")

    again = run([str(slmeval), "report", "--ascii"], cwd=root)
    if "Candidate" not in again:
        raise Failed(f"report could not re-render the stored run:\n{again}")

    # The exports CI consumes: JUnit for an existing dashboard, Markdown for a
    # pull-request comment. Both are shipped promises, so both are checked
    # against the artifact rather than only against the working tree.
    run([str(slmeval), "report", "--format", "junit", "--out", "junit.xml"], cwd=root)
    junit = (root / "junit.xml").read_text(encoding="utf-8")
    if "<testsuite" not in junit or "status_shipped" not in junit:
        raise Failed(f"the JUnit export has no per-case rows:\n{junit}")

    markdown = run([str(slmeval), "report", "--format", "md"], cwd=root, stdout_only=True)
    if "| Candidate" not in markdown:
        raise Failed(f"the Markdown export has no table:\n{markdown}")

    run([str(slmeval), "--version"], cwd=root)


def check_drilldown(slmeval: Path, root: Path) -> None:
    """`show` has to work off the stored rows alone, in the installed wheel."""
    print("drill-down: show run, show case")
    listing = run([str(slmeval), "show", "run", "--ascii"], cwd=root)
    if "executions" not in listing:
        raise Failed(f"show run printed no execution table:\n{listing}")

    diff = run([str(slmeval), "show", "case", "status_shipped", "--ascii"], cwd=root)
    if "trajectory" not in diff:
        raise Failed(f"show case printed no trajectory diff:\n{diff}")


def check_gate(slmeval: Path, root: Path) -> None:
    """The exit-code contract, against the installed artifact.

    A floor of 0 has to pass and a floor of 1 has to exit exactly 1 - not 2 and
    not a traceback. CI branches on these numbers, so a wheel that got them
    wrong would be worse than a wheel that did not install.
    """
    print("gate: exit codes")
    passing = run([str(slmeval), "gate", "--min-pass-k", "0", "--ascii"], cwd=root)
    if "GATE PASSED" not in passing:
        raise Failed(f"a floor of zero did not pass:\n{passing}")

    failing = run(
        [str(slmeval), "gate", "--min-pass-k", "1", "--ascii"], cwd=root, allow=(1,)
    )
    if "GATE FAILED" not in failing:
        raise Failed(f"a floor of one did not fail the gate:\n{failing}")

    run([str(slmeval), "baseline", "save"], cwd=root)
    if not (root / "slmeval.baseline.json").exists():
        raise Failed("baseline save wrote no slmeval.baseline.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="an already built wheel to test")
    parser.add_argument("--keep", action="store_true", help="keep the temporary directory")
    args = parser.parse_args()

    wheel = args.wheel.resolve() if args.wheel else build_wheel()
    workspace = Path(tempfile.mkdtemp(prefix="slmeval-smoke-"))
    print(f"workspace: {workspace}")

    # The README promises a scorecard within five minutes of `pip install`.
    # That promise is the funnel, so it is measured here rather than assumed -
    # from the empty venv to the printed table, on whichever runner this is.
    started = time.monotonic()
    try:
        slmeval = fresh_venv(workspace / "venv", wheel)
        first = workspace / "first-run"
        first.mkdir()
        check_first_run(slmeval, first)

        benchmark = workspace / "benchmark"
        benchmark.mkdir()
        check_scorecard(slmeval, benchmark)
        check_drilldown(slmeval, benchmark)
        check_gate(slmeval, benchmark)

        python = venv_bin(workspace / "venv") / ("python.exe" if os.name == "nt" else "python")
        run([str(python), "-m", "slmeval", "--version"], cwd=benchmark)
    except Failed as exc:
        print(f"\nFAILED: {exc}", file=sys.stderr)
        return 1
    finally:
        if args.keep:
            print(f"kept {workspace}")
        else:
            shutil.rmtree(workspace, ignore_errors=True)

    elapsed = time.monotonic() - started
    print(f"\nok - {wheel.name} installs and produces a scorecard")
    print(f"cold start: {elapsed:.0f}s from an empty virtualenv to a gated scorecard")
    if elapsed > COLD_START_BUDGET_S:
        print(
            f"FAILED: that is over the {COLD_START_BUDGET_S:.0f}s the README promises. "
            "The first five minutes decide whether this gets a user.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
