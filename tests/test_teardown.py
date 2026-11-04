"""The published teardown, checked against the files it claims to quote.

`teardown/` is the first thing a stranger reads, and it is a document that
makes numeric claims about artifacts sitting next to it. There are two ways it
can become a lie, and they need different checks:

* the *artifacts* stop matching what the tool produces - caught by
  ``python scripts/teardown.py --check``, which re-runs the benchmark and
  diffs. That costs a couple of seconds of real benchmarking and lives in the
  release workflow rather than here;
* the *write-up* stops matching the artifacts - a number edited by hand, a
  candidate renamed in ``eval.toml``, a finding quoted from a run that no
  longer exists. That is pure text, it is what will actually happen, and it is
  what this file catches.

So nothing here runs a benchmark. It reads the write-up, reads the artifacts,
and refuses to let them disagree.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

from slmeval.config.loader import load_config

ROOT = Path(__file__).resolve().parent.parent
PROJECT = ROOT / "teardown"
ARTIFACTS = PROJECT / "artifacts"
WRITEUP = PROJECT / "README.md"
GENERATOR = ROOT / "scripts" / "teardown.py"


def generator() -> Any:
    """`scripts/teardown.py`, imported from its path.

    `scripts/` is not a package and must not become one - it holds standalone
    tools, not library code - so this loads the module by location rather than
    adding an import path the rest of the suite would inherit.
    """
    name = "_teardown_script"
    spec = importlib.util.spec_from_file_location(name, GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: the module defines a dataclass, and
    # `@dataclass` resolves its annotations through `sys.modules`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def declared() -> tuple[Any, ...]:
    script = generator()
    return (*script.clean_run(), *script.faulted_run())


@pytest.fixture(scope="module")
def writeup() -> str:
    return WRITEUP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def scorecard() -> str:
    return (ARTIFACTS / "scorecard.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# The config and the dataset it points at
# --------------------------------------------------------------------------- #


def test_the_teardown_config_still_loads() -> None:
    """Published numbers with a config that no longer parses is the worst of
    both: unreproducible *and* obviously so."""
    loaded = load_config(PROJECT / "eval.toml")
    cfg = loaded.config
    assert cfg.benchmark.name == "teardown-01-support-agent"
    assert cfg.benchmark.seed == 42
    assert cfg.benchmark.runs_per_case == 3
    # The headline run has no faults in it. The faulted run is the same config
    # plus `--fault-rate`, which is the only way the two tables are comparable.
    assert cfg.benchmark.fault_injection_rate == 0.0


def test_the_four_models_are_the_four_the_writeup_names(writeup: str) -> None:
    cfg = load_config(PROJECT / "eval.toml").config
    ids = [cfg.baseline.id, *(c.id for c in cfg.candidates)]
    assert len(ids) == 4, ids
    for candidate_id in ids:
        assert candidate_id in writeup, f"{candidate_id} is in eval.toml but not in the write-up"


def test_the_dataset_is_the_examples_own(writeup: str) -> None:
    """Referenced, not copied.

    A teardown that forked its dataset drifts from the example it benchmarks,
    and then neither one reproduces the other.
    """
    loaded = load_config(PROJECT / "eval.toml")
    assert loaded.dataset_path == (
        ROOT / "examples" / "support_agent" / "golden_cases.jsonl"
    ).resolve()


# --------------------------------------------------------------------------- #
# The artifacts
# --------------------------------------------------------------------------- #


def test_every_declared_artifact_is_published() -> None:
    for artifact in declared():
        path = ARTIFACTS / artifact.name
        assert path.exists(), f"{artifact.name} is declared by the generator but not committed"
        assert path.stat().st_size > 0, f"{artifact.name} is empty"


def test_every_artifact_says_which_command_produced_it() -> None:
    """The header is the provenance, and it is why these are evidence rather
    than screenshots. A hand-edited artifact loses it."""
    for artifact in declared():
        text = (ARTIFACTS / artifact.name).read_text(encoding="utf-8")
        first = text.splitlines()[0]
        assert first.startswith("$ slmeval "), f"{artifact.name}: {first!r}"
        assert " ".join(artifact.argv) in first, f"{artifact.name}: {first!r}"
        assert "do not edit" in text


def test_the_writeup_links_every_artifact(writeup: str) -> None:
    """An artifact nothing links to is one nobody will read, and one that can
    rot without anyone noticing."""
    for artifact in declared():
        assert f"artifacts/{artifact.name}" in writeup, f"{artifact.name} is not linked"


def test_no_artifact_carries_an_absolute_path_from_whoever_ran_it() -> None:
    """`report.json` records the dataset by path, which must not be a laptop's.

    Checked on every artifact rather than just that one: a published file with
    `C:\\Users\\someone` in it pins the teardown to a machine and publishes a
    directory layout nobody asked about.
    """
    absolute = re.compile(r"(?:[A-Za-z]:[\\/]|/(?:home|Users)/)")
    for artifact in declared():
        text = (ARTIFACTS / artifact.name).read_text(encoding="utf-8")
        found = absolute.search(text)
        assert found is None, f"{artifact.name} carries an absolute path: {found.group()!r}"


@pytest.mark.parametrize(
    "line",
    [
        "  20261103-000830-265067 · {sha} · dataset 23cb279bf923fab6",
        "  dataset 23cb279bf923fab6 · config dc8b35b36fac8263 · {sha}",
    ],
)
def test_the_check_ignores_which_commit_ran_but_not_the_hashes(line: str) -> None:
    """A commit stamp changes on every commit, clean or dirty.

    Masking only `-dirty` stamps made `--check` pass on a working tree and fail
    on every commit after it. The dataset and config hashes stay visible: those
    moving is a real change to the experiment.
    """
    stable = generator()._stable
    assert stable(line.format(sha="0465e84")) == stable(line.format(sha="b144d92"))
    assert stable(line.format(sha="0465e84")) == stable(line.format(sha="e9eec82-dirty"))
    assert "23cb279bf923fab6" in stable(line.format(sha="0465e84"))


def test_the_check_ignores_model_call_timings() -> None:
    """The `ms` column has no unit beside it, and a loaded runner prints 13 for 0."""
    stable = generator()._stable
    quiet = "0    intent_router    valid     tools          -    direct     0"
    busy = "0    intent_router    valid     tools          -    direct    13"
    assert stable(quiet) == stable(busy)
    # A real change on the same row still shows.
    assert stable(quiet) != stable(busy.replace("tools", "json_mode"))


def test_the_json_report_is_the_versioned_schema() -> None:
    raw = (ARTIFACTS / "report.json").read_text(encoding="utf-8")
    payload = json.loads(raw[raw.index("{") :])
    assert payload["schema"].startswith("slmeval.report/")
    assert len(payload["candidates"]) == 4


# --------------------------------------------------------------------------- #
# The claims
# --------------------------------------------------------------------------- #


def test_the_writeups_table_agrees_with_the_scorecard(writeup: str, scorecard: str) -> None:
    """Every pass rate and interval in the write-up, found in the artifact.

    The write-up's table drops the two cost columns - every model here is
    free, so they carry no information - and keeps the rest. So this compares
    the shared columns cell by cell rather than matching whole lines: every
    rate, interval, denominator and mean the document publishes has to be the
    one the artifact produced.
    """
    rows = {
        cells[1].strip("* "): [c.strip() for c in cells[2:8]]
        for cells in (line.split("|") for line in scorecard.splitlines() if line.startswith("| "))
        if len(cells) > 4 and cells[1].strip("* ") not in {"Candidate"}
    }
    assert len(rows) == 4, rows

    published = {
        cells[1].strip("* "): [c.strip() for c in cells[2:8]]
        for cells in (line.split("|") for line in writeup.splitlines() if line.startswith("| "))
        if len(cells) > 4
    }
    for candidate_id, cells in rows.items():
        assert candidate_id in published, f"{candidate_id} is in the scorecard but not the write-up"
        assert published[candidate_id] == cells, (
            f"{candidate_id}: the write-up says {published[candidate_id]}, "
            f"the scorecard says {cells}"
        )


def test_the_main_readmes_table_agrees_with_the_scorecard(scorecard: str) -> None:
    """The first numbers a reader sees should be the ones the tool produced."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    table = {
        cells[1].strip("* "): [c.strip() for c in cells[2:8]]
        for cells in (line.split("|") for line in readme.splitlines() if line.startswith("| "))
        if len(cells) > 8
    }
    rows = {
        cells[1].strip("* "): [c.strip() for c in cells[2:8]]
        for cells in (line.split("|") for line in scorecard.splitlines() if line.startswith("| "))
        if len(cells) > 8 and cells[1].strip() != "Candidate"
    }
    assert len(rows) == 4, rows
    for candidate_id, cells in rows.items():
        assert table.get(candidate_id) == cells, candidate_id


#: Each finding in the write-up, and the string in the artifact that is its
#: evidence. Quoted text that no longer appears in any artifact is a claim the
#: teardown cannot support any more.
FINDINGS: tuple[tuple[str, str], ...] = (
    ("diagnosis-cassette-miss.txt", "verdict: hallucinated_parameter"),
    ("diagnosis-cassette-miss.txt", "(dropped prefix)"),
    ("uncovered.txt", "unrecorded_valid_path"),
    ("trace-forbidden-refund.txt", "issue_refund"),
    ("scorecard.md", "FORBIDDEN tool call: issue_refund"),
    ("scorecard.md", "json_mode on 100%"),
    ("scorecard.md", "prompt-repair or a fallback rung on 100%"),
    ("gate.txt", "GATE FAILED"),
    ("gate.txt", "forbidden_tools"),
    ("scorecard-faults.md", "Recovery from injected faults:"),
    ("failures-faults.txt", "KeyError: 'carrier'"),
)


@pytest.mark.parametrize(("artifact", "claim"), FINDINGS)
def test_each_finding_is_still_in_the_artifact_it_cites(artifact: str, claim: str) -> None:
    text = (ARTIFACTS / artifact).read_text(encoding="utf-8")
    assert claim in text, f"{artifact} no longer contains {claim!r}"


@pytest.mark.parametrize(("artifact", "claim"), FINDINGS)
def test_each_finding_is_actually_quoted_in_the_writeup(
    writeup: str, artifact: str, claim: str
) -> None:
    """The other direction: evidence collected and never used is fine, but
    every string this file calls a finding has to appear in the document."""
    assert claim.rstrip(":") in writeup, f"the write-up does not quote {claim!r}"


def test_the_gate_artifact_records_the_exit_code_ci_branches_on() -> None:
    """Exit 1 is the whole point of including it. A gate.txt that recorded a 0
    would mean the teardown's headline safety finding stopped failing the
    build, and nothing else in the suite would say so."""
    text = (ARTIFACTS / "gate.txt").read_text(encoding="utf-8")
    assert "# exit 1" in text


def test_the_limitations_are_stated_not_implied(writeup: str) -> None:
    """The section that decides whether this gets taken seriously.

    Spelled out as a list of things that must be *said*, because every one of
    them is a question someone will otherwise ask in public - and because the
    scripted-models caveat has to survive anyone who later decides the
    write-up reads better without it.
    """
    required = (
        "scripted",  # the models are not real SLMs
        "n = 4 cases",  # the sample is tiny
        "modelled, never measured",  # local cost is not an invoice
        "No latency numbers",  # throughput mode cannot report p95
        "Fault parity is not perfect",  # divergent trajectories
        "says nothing about timeout handling",  # two fault kinds excluded
        "judge is not involved",  # tier 3 is uncalibrated
        "is not determinism",  # temperature = 0
    )
    limitations = writeup[writeup.index("## Limitations") :]
    for claim in required:
        assert claim in limitations, f"the limitations no longer state: {claim}"
