"""Regenerate every artifact published in `teardown/`.

A teardown is only evidence if the reader can produce it. So nothing under
`teardown/artifacts/` is written by hand: this script runs the same commands
the README tells a reader to run, captures their output verbatim, and writes
it where the write-up links to it. If a number in the write-up disagrees with
an artifact, the artifact is right and the write-up is stale.

    python scripts/teardown.py              # record if needed, then regenerate
    python scripts/teardown.py --check      # fail if any artifact would change

`--check` is the useful one: it is what keeps a published teardown honest as
the tool underneath it changes. A commit that alters the scorecard and leaves
the write-up claiming the old numbers fails it.

Two runs over one matrix, and they are separate on purpose:

* the clean run is the headline table - the cassette miss, the forbidden
  `issue_refund`, and the model that passes every case with repaired JSON;
* the faulted run is the resilience experiment, where every injected ERP
  failure takes the agent down with a `KeyError` and the recovery rate is 0%.

Folding them together would hide the safety finding behind an injected
outage: at a 30% fault rate the refund case crashes before the agent ever
reaches the tool that moves money.

Everything here is offline. No API key, no model server, no spend.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO / "teardown"
ARTIFACTS = PROJECT / "artifacts"

FAULT_RATE = "0.3"
"""High enough that most cases meet a fault at n=4, low enough that the table
still has passing rows to compare against. Stated in the write-up, because a
recovery rate is meaningless without it."""

WIDTH = "100"
"""Fixed terminal width for every captured command.

Rich wraps to the terminal, so without this the artifacts would differ between
machines - and a teardown whose files change depending on who regenerated them
is not reproducible in the only sense that matters here.
"""


@dataclass(frozen=True)
class Artifact:
    """One published file and the command that produces it."""

    name: str
    argv: tuple[str, ...]
    why: str

    #: Exit codes that are a correct result rather than a failure. `gate`
    #: exiting 1 on a candidate that called `issue_refund` is the whole point
    #: of running it, so 1 has to be allowed - but only where it is named.
    allowed_exits: tuple[int, ...] = (0,)


def clean_run() -> tuple[Artifact, ...]:
    """The headline matrix: 4 models x 4 cases x 3 repeats, no faults."""
    return (
        Artifact(
            "scorecard.md",
            ("report", "--format", "md", "--cases"),
            "the table, as the write-up embeds it",
        ),
        Artifact(
            "scorecard.txt",
            ("report", "--format", "table"),
            "the same table as the terminal prints it",
        ),
        Artifact(
            "report.json",
            ("report", "--format", "json"),
            "every number in the table, versioned, for anyone checking our arithmetic",
        ),
        Artifact(
            "failures.txt",
            ("show", "run", "--failed", "--limit", "0"),
            "every execution that did not pass, worst first",
        ),
        Artifact(
            "diagnosis-cassette-miss.txt",
            ("show", "case", "status_shipped", "--candidate", "sloppy-router"),
            "finding 1: the per-argument diff on a hallucinated order id",
        ),
        Artifact(
            "trace-forbidden-refund.txt",
            ("show", "case", "refund_needs_confirmation", "--candidate", "refund-happy"),
            "finding 2: the trajectory that reaches issue_refund",
        ),
        Artifact(
            "trace-repaired-json.txt",
            ("show", "exec", "fenced-json.status_shipped.r0"),
            "finding 3: a pass that needed a lower rung to get there",
        ),
        Artifact(
            "uncovered.txt",
            ("show", "uncovered", "--all"),
            "the coverage gaps, told apart from the hallucinations",
        ),
        Artifact(
            "gate.txt",
            ("gate", "--min-pass-k", "0.5"),
            "what CI does with this run: exit 1, on the forbidden tool call",
            allowed_exits=(0, 1),
        ),
    )


def faulted_run() -> tuple[Artifact, ...]:
    """The same matrix under injected ERP failures."""
    return (
        Artifact(
            "scorecard-faults.md",
            ("report", "--format", "md"),
            "the table under faults, where the recovery rate appears",
        ),
        Artifact(
            "failures-faults.txt",
            ("show", "run", "--failed", "--limit", "0"),
            "finding 4: every fault took the agent down with a KeyError",
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write. Exit 1 if any artifact would change.",
    )
    parser.add_argument(
        "--rerecord",
        action="store_true",
        help="Re-record the cassettes even if they are already there.",
    )
    args = parser.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    cassettes = PROJECT / ".slmeval" / "cassettes.sqlite"
    if args.rerecord or not cassettes.exists():
        _say("recording the agent's tool calls once, against the in-memory ERP")
        _run(("record",))

    _say("the clean matrix")
    _run(("run", "--quiet"))
    for artifact in clean_run():
        written[artifact.name] = _capture(artifact)

    _say(f"the same matrix at a {FAULT_RATE} fault rate")
    _run(("run", "--quiet", "--fault-rate", FAULT_RATE))
    for artifact in faulted_run():
        written[artifact.name] = _capture(artifact)

    if args.check:
        return _check(written)
    for name, text in written.items():
        (ARTIFACTS / name).write_text(text, encoding="utf-8", newline="\n")
    _say(f"{len(written)} artifact(s) written to {ARTIFACTS}")
    return 0


VOLATILE: tuple[tuple[str, str], ...] = (
    (r"\d{8}-\d{6}-[0-9a-f]{6}", "<run-id>"),
    (r"\d{4}-\d{2}-\d{2}T[\d:.]+\+00:00", "<timestamp>"),
    (r'("git_sha": ")[^"]*', r"\1<git-sha>"),
    (r"\b[0-9a-f]{7,40}-dirty\b", "<git-sha>"),
    (r'("[a-z0-9_]*_ms": )(?:[\d.]+|null)', r"\1<ms>"),
    (r"(\s)\d+(?:\.\d+)?(\s*ms\b)", r"\1<ms>\2"),
    (r'("(?:cpus|machine|os|release|python|implementation)": )("?[^",}]*"?)', r"\1<host>"),
)
"""What differs between two correct runs of the same matrix, and is therefore
not evidence of anything.

Used only to compare - never to rewrite what gets published. A reader wants
the real run id, the real wall clock and the real host in the artifact they
are reading; `--check` wants to know whether the *findings* moved. Scrubbing
the published file would have cost both.
"""


def _stable(text: str) -> str:
    """One artifact, with the volatile fields masked, for comparison only."""
    for pattern, replacement in VOLATILE:
        text = re.sub(pattern, replacement, text)
    return text


def _check(written: dict[str, str]) -> int:
    """Diff what we just produced against what is committed."""
    stale: list[str] = []
    for name, fresh in written.items():
        path = ARTIFACTS / name
        if not path.exists():
            stale.append(f"{name}: not committed")
            continue
        current = _stable(path.read_text(encoding="utf-8"))
        fresh = _stable(fresh)
        if current == fresh:
            continue
        stale.append(name)
        diff = difflib.unified_diff(
            current.splitlines(),
            fresh.splitlines(),
            fromfile=f"committed/{name}",
            tofile=f"regenerated/{name}",
            lineterm="",
            n=1,
        )
        print("\n".join(list(diff)[:40]), file=sys.stderr)

    if not stale:
        _say(f"{len(written)} artifact(s) match what is published")
        return 0
    print(
        f"\n{len(stale)} artifact(s) are stale: {', '.join(stale)}\n"
        "The published teardown no longer matches what the tool produces.\n"
        "Regenerate with `python scripts/teardown.py`, then read the diff before\n"
        "committing it - a number that moved is either a fix or a regression, and\n"
        "the write-up has to say which.",
        file=sys.stderr,
    )
    return 1


def _capture(artifact: Artifact) -> str:
    """Run one command and return its output, with a header saying what it was."""
    completed = _run(artifact.argv, capture=True, allowed=artifact.allowed_exits)
    command = "slmeval " + " ".join(artifact.argv)
    header = (
        f"$ {command}\n"
        f"# {artifact.why}\n"
        f"# regenerated by scripts/teardown.py - do not edit\n"
    )
    if completed.returncode:
        header += f"# exit {completed.returncode}\n"
    body = _portable(completed.stdout.replace("\r\n", "\n")).rstrip() + "\n"
    return f"{header}\n{body}"


def _portable(text: str) -> str:
    """Take this checkout's absolute paths out of a published artifact.

    `report.json` records the dataset it read by absolute path, which is right
    for a run on your own machine and wrong in a file committed to a public
    repository: it pins the artifact to one laptop and publishes a directory
    layout nobody asked about. Replaced with the repo-relative path, which is
    the part a reader can actually use.
    """
    root = str(REPO)
    # Three spellings of the same prefix: native, posix, and JSON-escaped -
    # `report.json` carries the third. Longest first, so the escaped form is
    # consumed before a shorter variant can match half of it.
    spellings = sorted({root, root.replace("\\", "/"), root.replace("\\", "\\\\")}, key=len)
    for form in reversed(spellings):
        # Only the matched tail is rewritten. A blanket backslash substitution
        # would also rewrite the regexes the dataset's assertions print.
        text = re.sub(
            re.escape(form) + r'(?:\\\\|[\\/])(?P<tail>[^\s",]*)',
            lambda m: m.group("tail").replace("\\\\", "/").replace("\\", "/"),
            text,
        )
    return text


def _run(
    argv: tuple[str, ...],
    *,
    capture: bool = False,
    allowed: tuple[int, ...] = (0,),
) -> subprocess.CompletedProcess[str]:
    """Invoke the CLI in `teardown/` the way a reader would.

    Through `-m slmeval`, not the console script, so this works from a checkout
    with no install step - and so it is unambiguous which slmeval ran.
    """
    env = dict(os.environ)
    # UTF-8 and a fixed width, or the artifacts would carry this machine's
    # console codepage and terminal size.
    env["PYTHONIOENCODING"] = "utf-8"
    env["COLUMNS"] = WIDTH
    env.pop("NO_COLOR", None)

    completed = subprocess.run(
        [sys.executable, "-m", "slmeval", *argv],
        cwd=PROJECT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    if completed.returncode not in allowed:
        sys.stderr.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        raise SystemExit(
            f"`slmeval {' '.join(argv)}` exited {completed.returncode}, "
            f"expected one of {allowed}"
        )
    if not capture:
        print(_tail(completed.stdout), file=sys.stderr)
    return completed


def _tail(text: str, lines: int = 6) -> str:
    kept = [line for line in text.replace("\r\n", "\n").split("\n") if line.strip()]
    return "\n".join(f"    {line}" for line in kept[-lines:])


def _say(message: str) -> None:
    print(f"\n==> {message}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
