"""Run it twice, get the same answer.

A benchmark whose numbers move when nothing moved is not a measurement, and the
ways that happens are boring and platform-specific: a hash that iterates a
dict, a path separator in a key, a CRLF checkout, a scheduler that interleaves
two executions differently. None of them show up in a unit test of the formula.

So these run whole benchmarks and compare the results of one against the other,
and they run on Windows and Linux in CI - the two places those divergences
differ. Where a value has to be identical *across* machines rather than merely
stable on one, it is pinned to a literal: a digest written down here is the only
thing that catches a Windows-only key change before a user does.

Everything is scripted and offline, so "the same answer" is a real claim rather
than a tolerance.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from slmeval.dataset.loader import dataset_hash, load_dataset
from slmeval.report.export import to_json
from slmeval.report.model import build_scorecard
from slmeval.sandbox.faults import fault_schedule
from slmeval.sandbox.keys import declare
from tests.offline import (
    ONE_CASE,
    behaves,
    candidate,
    forget_the_agent_module,  # noqa: F401 - autouse fixture
    offline_project,
    replies,
    routes,
    write_project,
)

FAULTS = (
    "[benchmark.faults]\n"
    'kinds = ["empty_result", "rate_limit_429"]\n'
    "delay_scale = 0.0\n"
)

MIXED = "".join(
    [
        candidate("sloppy-id", routes("order_status", "9921"), replies()),
        candidate("prose-only", behaves("not_json")),
        candidate(
            "fenced",
            behaves("fenced_json", '{"intent": "order_status", "order_id": "ORD-9921"}'),
            behaves("fenced_json", '{"final_answer": "with DHL"}'),
            extra='extra_body.unsupported = ["tools", "json_schema"]',
        ),
    ]
)

# What differs between two runs of the same benchmark by definition. Everything
# else has to match, which is the point: scrub timings and ids, and any
# remaining difference is a real one.
VOLATILE = re.compile(r"(_ms$|_at$|^run_id$|^exec_id$|^git_sha$|^dataset_path$|^duration)")


def stable(value: Any) -> Any:
    """The report with its clocks and ids removed."""
    if isinstance(value, dict):
        return {k: stable(v) for k, v in value.items() if not VOLATILE.search(k)}
    if isinstance(value, list):
        return [stable(v) for v in value]
    return value


def report_of(runs: Any, run_id: str) -> Any:
    return stable(json.loads(to_json(build_scorecard(runs.store, run_id))))


# --------------------------------------------------------------------------- #
# The same benchmark, twice
# --------------------------------------------------------------------------- #


def test_two_runs_of_one_benchmark_agree_to_the_digit(tmp_path: Path) -> None:
    """The headline claim. Four candidates, two repeats, a cassette miss and a
    schema failure among them - and the two reports are the same document."""
    with offline_project(tmp_path, candidates=MIXED, cases=ONE_CASE, repeats=2) as runs:
        runs.record()
        first = runs.run(tool_mode="replay")
        second = runs.run(tool_mode="replay")

        assert first.run_id != second.run_id
        assert report_of(runs, first.run_id) == report_of(runs, second.run_id)


def test_the_scheduler_cannot_change_the_result(tmp_path: Path) -> None:
    """Concurrency is a throughput choice. If it also moved the numbers, every
    published figure would carry an invisible "at concurrency 4" caveat."""
    with offline_project(tmp_path, candidates=MIXED, cases=ONE_CASE, repeats=2) as runs:
        runs.record()
        serial = runs.run(tool_mode="replay", concurrency=1)
        parallel = runs.run(tool_mode="replay", concurrency=8)

        assert report_of(runs, serial.run_id) == report_of(runs, parallel.run_id)


def test_a_run_in_another_directory_is_the_same_benchmark(tmp_path: Path) -> None:
    """Two checkouts, two absolute paths, one benchmark. A path that reached a
    hash would make every CI run incomparable with every local one."""
    hashes = []
    for name in ("alpha", "beta"):
        root = tmp_path / name
        root.mkdir()
        with offline_project(root, candidates=MIXED, cases=ONE_CASE) as runs:
            hashes.append((runs.config.config_hash(), runs.project.dataset.content_hash))

    assert hashes[0] == hashes[1]


# --------------------------------------------------------------------------- #
# Fault vectors
# --------------------------------------------------------------------------- #


def test_one_seed_means_one_experiment(tmp_path: Path) -> None:
    """Every candidate must meet the same failures at the same steps, in both
    runs. Fault parity across candidates is what makes two rows of the table
    comparable at all, so the digest is stored and compared rather than
    assumed."""
    with offline_project(
        tmp_path,
        candidates=MIXED,
        cases=ONE_CASE,
        repeats=2,
        # Every call faulted: at one tool call per execution, a rate below 1.0
        # leaves this test asserting that two empty vectors match.
        fault_rate=1.0,
        extra_benchmark=FAULTS,
    ) as runs:
        runs.record()
        first = runs.run(tool_mode="replay")
        second = runs.run(tool_mode="replay")

        def digests(run_id: str) -> dict[tuple[str, str, int], str]:
            return {
                (r.candidate_id, r.case_id, r.repeat_idx): r.fault_digest
                for r in runs.store.executions(run_id)
            }

        assert digests(first.run_id) == digests(second.run_id)
        assert any(digests(first.run_id).values()), "no faults were injected at all"


def test_the_schedule_is_a_pure_function_of_the_seed() -> None:
    """Rebuilt from scratch in a fresh process it has to come out the same, so
    `--resume` and `report` describe the run that happened rather than a new
    draw."""
    tools = ["fetch_order", "issue_refund"]
    one = fault_schedule(7, "status_shipped", rate=0.4)
    again = fault_schedule(7, "status_shipped", rate=0.4)
    other_seed = fault_schedule(8, "status_shipped", rate=0.4)
    other_case = fault_schedule(7, "refund_207", rate=0.4)

    assert one.digest(tools) == again.digest(tools)
    assert one.digest(tools) != other_seed.digest(tools)
    # Per case, or one long case would shift every later case's faults.
    assert one.digest(tools) != other_case.digest(tools)


def test_the_fault_draw_is_pinned_across_platforms() -> None:
    """A literal, because this is the value that must not differ between a
    Windows laptop and a Linux runner. If this changes, every stored fault
    digest in every project becomes a lie - so it changes only deliberately."""
    schedule = fault_schedule(42, "status_shipped", rate=1.0)
    faults = [schedule.fault_for("fetch_order", i) for i in range(3)]

    assert [f.kind.value for f in faults if f] == [
        "timeout_504",
        "empty_result",
        "empty_result",
    ]


# --------------------------------------------------------------------------- #
# Cassette keys
# --------------------------------------------------------------------------- #


def fetch_order(order_id: str, request_id: str = "r") -> dict:
    return {"order_id": order_id}


def test_a_cassette_key_is_the_same_hash_everywhere() -> None:
    """Pinned to a literal for the same reason as the fault draw: a key that
    differs by platform turns a shared cassette store into a wall of misses
    that reads exactly like the model falling apart."""
    policy = declare(
        fetch_order,
        tool_name="fetch_order",
        match_on=["order_id"],
        normalize={"order_id": "lower_strip"},
    )
    keys = policy.compute("status_shipped", {"order_id": "ORD-9921", "request_id": "r1"})

    assert keys.exact == "09ef0bcf75dc712b6563e8c38f1aae60"
    assert keys.normalized == "317083a713ba3b7ef00d8bf52496e4c1"
    assert keys.subset == "365ad429b308e151b468f11320b11ea1"


def test_a_key_does_not_depend_on_how_the_arguments_were_spelled() -> None:
    """Argument order, and a default spelled out rather than left out, are the
    two ways the same call reaches us looking different."""
    policy = declare(fetch_order, tool_name="fetch_order", match_on=["order_id"])
    a = policy.compute("c", {"order_id": "ORD-1", "request_id": "r"})
    b = policy.compute("c", {"request_id": "r", "order_id": "ORD-1"})
    c = policy.compute("c", {"order_id": "ORD-1"})

    assert a.keys == b.keys == c.keys


# --------------------------------------------------------------------------- #
# Reading the project
# --------------------------------------------------------------------------- #


def test_a_crlf_checkout_hashes_like_an_lf_one(tmp_path: Path) -> None:
    """Git hands a Windows checkout CRLF line endings. If that changed the
    dataset hash, every provenance check would fail across platforms and the
    report would refuse to compare runs that are in fact the same."""
    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    lf.write_bytes(ONE_CASE.encode("utf-8"))
    crlf.write_bytes(ONE_CASE.replace("\n", "\r\n").encode("utf-8"))

    assert dataset_hash(load_dataset(lf).cases) == dataset_hash(load_dataset(crlf).cases)


def test_a_bom_does_not_change_the_benchmark(tmp_path: Path) -> None:
    """Notepad writes one. It must not become a different config."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plain = write_project(tmp_path / "a", candidates=MIXED)
    with_bom = write_project(tmp_path / "b", candidates=MIXED)
    with_bom.write_bytes(b"\xef\xbb\xbf" + with_bom.read_bytes())

    from slmeval.config.loader import load_config

    assert load_config(plain).config.config_hash() == load_config(with_bom).config.config_hash()
