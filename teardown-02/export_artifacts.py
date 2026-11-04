"""Write the files in `artifacts/` from the runs stored in `.slmeval/`.

    cd teardown-02
    slmeval record
    slmeval run
    slmeval run --serial-latency --case <id> ...   # the 5 cases are in README.md
    python export_artifacts.py <full-run-id> <latency-run-id>

This reads stored runs and calls no model. It exists so that nothing in
`artifacts/` is typed by hand, and so that the replies behind the write-up's
claims about wrong answers can be read directly.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
ROOT = HERE.parent
ROUTER = "llama3.2-3b-router"
OTHER = "gemma3-4b-router"


def capture(name: str, *argv: str, allowed: tuple[int, ...] = (0,)) -> None:
    env = {**os.environ, "COLUMNS": "110", "NO_COLOR": "1", "PYTHONUTF8": "1", "TERM": "dumb"}
    done = subprocess.run(
        [sys.executable, "-m", "slmeval", *argv],
        cwd=HERE,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if done.returncode not in allowed:
        raise SystemExit(f"{name}: exit {done.returncode}\n{done.stdout}\n{done.stderr}")
    (ARTIFACTS / name).write_text(portable(done.stdout), encoding="utf-8", newline="\n")
    print(f"{name}: exit {done.returncode}")


def portable(text: str) -> str:
    """Take this machine's absolute paths out of a published file."""
    root = str(ROOT)
    for form in (root.replace("\\", "\\\\"), root.replace("\\", "/"), root):
        text = re.sub(
            re.escape(form) + r'(?:\\\\|[\\/])(?P<tail>[^\s",]*)',
            lambda m: m.group("tail").replace("\\\\", "/").replace("\\", "/"),
            text,
        )
    return text


def replies(run_id: str) -> None:
    """Every distinct reply per case, and which configuration wrote it."""
    db = sqlite3.connect(HERE / ".slmeval" / "runs.sqlite")
    rows = db.execute(
        "select case_id, candidate_id, outcome_class, state_json from executions "
        "where run_id = ? order by case_id, candidate_id, repeat_idx",
        (run_id,),
    ).fetchall()
    db.close()
    grouped: dict[str, dict[tuple[str, str, str], int]] = {}
    for case_id, candidate, outcome, state in rows:
        answer = (json.loads(state) if state else {}).get("final_answer") or "(no reply)"
        by_reply = grouped.setdefault(case_id, {})
        key = (candidate, outcome, answer)
        by_reply[key] = by_reply.get(key, 0) + 1
    lines = [f"Every distinct reply in run {run_id}, with how many of the 3 repeats wrote it.", ""]
    for case_id in sorted(grouped):
        lines.append(case_id)
        for (candidate, outcome, answer), count in grouped[case_id].items():
            lines.append(f"  {candidate} {count}x {outcome}")
            lines.append(f"    {answer}")
        lines.append("")
    (ARTIFACTS / "replies.txt").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("replies.txt")


def main() -> None:
    full, latency = sys.argv[1:3]
    ARTIFACTS.mkdir(exist_ok=True)
    capture("scorecard.txt", "report", "--run-id", full, "--format", "table")
    capture("scorecard.md", "report", "--run-id", full, "--format", "md", "--cases")
    capture("report.json", "report", "--run-id", full, "--format", "json")
    capture("scorecard-latency.txt", "report", "--run-id", latency, "--format", "table")
    capture("report-latency.json", "report", "--run-id", latency, "--format", "json")
    capture("failures.txt", "show", "run", full, "--failed", "--limit", "0")
    capture("trace-injection-refund.txt", "show", "exec", f"{full}.{ROUTER}.inject_ignore.r0")
    capture("trace-dropped-prefix.txt", "show", "exec", f"{full}.{OTHER}.status_delivered.r0")
    capture("trace-invented-order.txt", "show", "exec", f"{full}.{OTHER}.no_order_thanks.r0")
    capture("gate.txt", "gate", "--run-id", full, "--min-pass-k", "0.5", allowed=(0, 1))
    replies(full)


if __name__ == "__main__":
    main()
