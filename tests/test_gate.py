"""The gate: thresholds, severities, and the exit-code contract.

The exit code is the product, so most of these tests assert a number rather
than a word. The two that matter more than all the others:

- a weak candidate exits 1, and
- an unreachable provider exits 3 while saying it is not a regression.

If those two ever merge, a team learns to ignore the gate, and a gate people
ignore is worth less than no gate at all.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from slmeval import exit_codes
from slmeval.eval.stats import Aggregate, Consistency, aggregate, consistency
from slmeval.gate import (
    COST_RATIO,
    COVERAGE,
    FORBIDDEN_TOOLS,
    P95_RATIO,
    PASS_K,
    PROVIDER,
    REGRESSION,
    Severity,
    Thresholds,
    evaluate,
)
from slmeval.report.baseline import Baseline, CandidateBaseline, CaseBaseline
from slmeval.report.model import CandidateReport, Incident, Scorecard
from slmeval.storage.models import Outcome, RunRecord, RunStatus


def run_record(**kwargs: object) -> RunRecord:
    base = {
        "run_id": "20260101-000000-abcdef",
        "name": "t",
        "started_at": "2026-01-01T00:00:00+00:00",
        "config_json": "{}",
        "config_hash": "cfg",
        "dataset_hash": "ds",
        "slmeval_version": "0.1.0",
        "mode": "replay",
        "seed": 42,
        "status": RunStatus.COMPLETED,
        "runs_per_case": 3,
    }
    base.update(kwargs)
    return RunRecord(**base)  # type: ignore[arg-type]


def row(
    candidate_id: str,
    *,
    role: str = "candidate",
    cases: int = 10,
    all_pass: int = 10,
    cost: float | None = 0.001,
    p95: float | None = None,
    unsafe: int = 0,
    uncovered: int = 0,
    provider_errors: int = 0,
    executions: int | None = None,
) -> CandidateReport:
    """One scorecard row, built straight rather than through a whole run.

    The gate is a pure function of a scorecard, so its tests do not need a
    matrix - and building the rows here is what lets a single test pin "3/3
    passes fails a 90% floor on the lower bound", which no scripted run could
    express as clearly.
    """
    total = executions if executions is not None else cases * 3
    outcomes: dict[Outcome, int] = {}
    if provider_errors:
        outcomes[Outcome.PROVIDER_ERROR] = provider_errors
    if uncovered:
        outcomes[Outcome.UNCOVERED] = uncovered
    judged = max(0, total - provider_errors - uncovered)
    if judged:
        outcomes[Outcome.PASSED] = judged

    rollups = {
        f"case_{i}": consistency(3, 3 if i < all_pass else 1)
        for i in range(cases)
    }
    return CandidateReport(
        candidate_id=candidate_id,
        role=role,
        model=candidate_id,
        stats=aggregate(list(rollups.values())),
        rollups=rollups,
        outcomes=outcomes,
        cost_total=None if cost is None else cost * cases,
        cost_cases=cases,
        durations_ms=() if p95 is None else tuple([p95] * 20),
        unsafe_actions=unsafe,
        incidents=(
            (Incident(candidate_id=candidate_id, case_id="case_0", tools=("issue_refund",)),)
            if unsafe
            else ()
        ),
        uncovered_cases=("case_9",) if uncovered else (),
        uncovered_executions=uncovered,
    )


def card(*rows: CandidateReport, scheduling: str = "throughput", **run: object) -> Scorecard:
    return Scorecard(
        run=run_record(scheduling=scheduling, **run),
        rows=rows,
        cases=tuple(f"case_{i}" for i in range(10)),
        repeats=3,
    )


# --------------------------------------------------------------------------- #
# The exit-code contract
# --------------------------------------------------------------------------- #


def test_a_good_candidate_passes() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")),
        Thresholds(min_pass_k=0.5),
    )
    assert report.passed
    assert report.exit_code == exit_codes.OK


def test_a_weak_candidate_fails_the_gate() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small", all_pass=4)),
        Thresholds(min_pass_k=0.9),
    )
    assert report.exit_code == exit_codes.GATE_FAILED
    assert report.severity is Severity.GATE
    assert [c.name for c in report.failures] == [PASS_K]


def test_an_unreachable_provider_is_environment_not_regression() -> None:
    """The single most important line in the module.

    A CI job that reported "Ollama was not running" as "the model got worse"
    would teach a team to ignore the gate, and after that it is worth nothing.
    """
    report = evaluate(
        card(
            row("baseline", role="baseline"),
            row("small", all_pass=0, provider_errors=30, executions=30),
        ),
        Thresholds(min_pass_k=0.9),
    )
    assert report.exit_code == exit_codes.ENVIRONMENT
    assert report.severity is Severity.ENVIRONMENT
    assert any(c.name == PROVIDER for c in report.failures)


def test_thin_coverage_is_its_own_exit_code() -> None:
    report = evaluate(
        card(
            row("baseline", role="baseline"),
            row("small", all_pass=5, uncovered=20, executions=30),
        ),
        Thresholds(min_pass_k=0.1),
    )
    assert report.exit_code == exit_codes.COVERAGE
    assert any(c.name == COVERAGE for c in report.failures)


def test_a_forbidden_call_outranks_a_coverage_shortfall() -> None:
    """An observed safety violation must never be masked by a missing recording.

    Coverage outranks the rate-based checks because a rate over a third of the
    executions is not a measurement. A forbidden tool call is not a rate: it is
    an event somebody observed, and it stays exit 1.
    """
    report = evaluate(
        card(
            row("baseline", role="baseline"),
            row("small", unsafe=1, uncovered=20, executions=30),
        ),
        Thresholds(min_pass_k=0.1),
    )
    assert report.exit_code == exit_codes.GATE_FAILED
    names = {c.name for c in report.failures}
    assert FORBIDDEN_TOOLS in names
    # The coverage shortfall is still reported, just not in the exit code.
    assert COVERAGE in names


def test_environment_outranks_coverage() -> None:
    report = evaluate(
        card(
            row("baseline", role="baseline"),
            row("small", all_pass=0, provider_errors=20, uncovered=10, executions=30),
        ),
        Thresholds(min_pass_k=0.9),
    )
    assert report.exit_code == exit_codes.ENVIRONMENT


def test_nothing_to_gate_is_a_usage_error_not_a_pass() -> None:
    """The dangerous failure mode of a gate is green-while-measuring-nothing."""
    report = evaluate(card(row("baseline", role="baseline")), Thresholds(min_pass_k=0.9))
    assert report.exit_code == exit_codes.USAGE
    assert "no candidate to gate" in report.usage_error


def test_no_threshold_at_all_is_a_usage_error() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")),
        Thresholds(no_forbidden_tools=False),
    )
    assert report.exit_code == exit_codes.USAGE
    assert "nothing to gate on" in report.usage_error


# --------------------------------------------------------------------------- #
# The estimator
# --------------------------------------------------------------------------- #


def test_three_out_of_three_does_not_clear_a_ninety_percent_floor() -> None:
    """The reason the gate reads the lower bound, in one assertion.

    Ten cases all passing three repeats is a point estimate of 100% and a
    Wilson lower bound near 72%. A gate at 90% reading the point estimate would
    approve it; reading the bound, it does not, because ten cases cannot
    demonstrate 90% at 95% confidence.
    """
    small = row("small", cases=10, all_pass=10)
    assert small.pass_hat_k == 1.0
    assert small.stats.lower is not None and small.stats.lower < 0.9

    strict = evaluate(card(row("baseline", role="baseline"), small), Thresholds(min_pass_k=0.9))
    assert strict.exit_code == exit_codes.GATE_FAILED

    loose = evaluate(
        card(row("baseline", role="baseline"), small),
        Thresholds(min_pass_k=0.9, use_wilson_lower=False),
    )
    assert loose.exit_code == exit_codes.OK


def test_the_detail_names_which_estimator_was_used() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")),
        Thresholds(min_pass_k=0.9, use_wilson_lower=False),
    )
    check = next(c for c in report.checks if c.name == PASS_K)
    assert "point estimate" in check.detail


def test_a_candidate_with_nothing_judgeable_cannot_be_gated() -> None:
    nothing = replace(
        row("small"), stats=Aggregate(), rollups={}, uncovered_executions=30,
        outcomes={Outcome.UNCOVERED: 30},
    )
    report = evaluate(
        card(row("baseline", role="baseline"), nothing), Thresholds(min_pass_k=0.9)
    )
    assert report.exit_code == exit_codes.COVERAGE
    assert any("no judgeable execution" in c.detail for c in report.failures)


# --------------------------------------------------------------------------- #
# Ratios
# --------------------------------------------------------------------------- #


def test_the_cost_ratio_is_measured_against_the_baseline() -> None:
    report = evaluate(
        card(
            row("baseline", role="baseline", cost=0.01),
            row("cheap", cost=0.002),
            row("dear", cost=0.008),
        ),
        Thresholds(min_pass_k=0.1, max_cost_ratio=0.5),
    )
    checks = {c.candidate_id: c for c in report.checks if c.name == COST_RATIO}
    assert checks["cheap"].passed
    assert not checks["dear"].passed
    assert report.exit_code == exit_codes.GATE_FAILED


def test_an_unpriced_candidate_is_skipped_not_passed() -> None:
    """A check that could not run must not contribute a quiet green."""
    report = evaluate(
        card(row("baseline", role="baseline", cost=0.01), row("local", cost=None)),
        Thresholds(min_pass_k=0.1, max_cost_ratio=0.5),
    )
    check = next(c for c in report.checks if c.name == COST_RATIO)
    assert check.skipped
    assert not check.failed
    # Skipped is neither a pass nor a failure, and it does not move the code -
    # failing a build over an unpriced model would read as a regression.
    assert report.exit_code == exit_codes.OK
    assert check in report.skipped


def test_a_latency_budget_is_skipped_on_a_throughput_run() -> None:
    """A p95 from concurrent requests describes the scheduler, so there is
    nothing here to enforce - and the gate says so rather than passing."""
    report = evaluate(
        card(
            row("baseline", role="baseline", p95=1000),
            row("small", p95=4000),
            scheduling="throughput",
        ),
        Thresholds(min_pass_k=0.1, max_p95_ratio=2.0),
    )
    check = next(c for c in report.checks if c.name == P95_RATIO)
    assert check.skipped
    assert "--serial-latency" in check.detail
    assert report.exit_code == exit_codes.OK


def test_a_latency_budget_is_enforced_on_a_latency_run() -> None:
    report = evaluate(
        card(
            row("baseline", role="baseline", p95=1000),
            row("slow", p95=4000),
            scheduling="latency",
        ),
        Thresholds(min_pass_k=0.1, max_p95_ratio=2.0),
    )
    check = next(c for c in report.checks if c.name == P95_RATIO)
    assert not check.passed
    assert check.observed == pytest.approx(4.0)


# --------------------------------------------------------------------------- #
# Forbidden tools
# --------------------------------------------------------------------------- #


def test_a_forbidden_call_is_never_averaged_away() -> None:
    """One forbidden call in thirty executions is not "97% safe"."""
    report = evaluate(
        card(row("baseline", role="baseline"), row("small", unsafe=1)),
        Thresholds(min_pass_k=0.1),
    )
    check = next(c for c in report.checks if c.name == FORBIDDEN_TOOLS)
    assert not check.passed
    assert check.limit == 0.0
    assert "issue_refund" in check.detail
    assert report.exit_code == exit_codes.GATE_FAILED


def test_the_baseline_is_checked_for_forbidden_calls_too() -> None:
    """A refund issued without authorisation is not acceptable because the
    frontier model issued it."""
    report = evaluate(
        card(row("baseline", role="baseline", unsafe=1), row("small")),
        Thresholds(min_pass_k=0.1),
    )
    assert any(
        c.candidate_id == "baseline" and c.name == FORBIDDEN_TOOLS for c in report.failures
    )


def test_the_baseline_is_not_gated_on_quality() -> None:
    """It is the reference every delta is measured against. Gating it against
    itself would always pass and tell nobody anything."""
    report = evaluate(
        card(row("baseline", role="baseline", all_pass=0), row("small")),
        Thresholds(min_pass_k=0.9),
    )
    assert not any(c.candidate_id == "baseline" and c.name == PASS_K for c in report.checks)


# --------------------------------------------------------------------------- #
# Against a committed baseline
# --------------------------------------------------------------------------- #


def stored_baseline(**candidates: CandidateBaseline) -> Baseline:
    return Baseline(
        run_id="20251201-000000-aaaaaa",
        config_hash="cfg",
        dataset_hash="ds",
        slmeval_version="0.1.0",
        mode="replay",
        seed=42,
        candidates=dict(candidates),
    )


def frozen(candidate_id: str, *, cases: int = 10, all_pass: int = 10) -> CandidateBaseline:
    rollups = [consistency(3, 3 if i < all_pass else 1) for i in range(cases)]
    stats = aggregate(rollups)
    return CandidateBaseline(
        candidate_id=candidate_id,
        pass_hat_k=stats.pass_hat_k,
        wilson_lo=stats.lower,
        cases=cases,
        cases_all_pass=all_pass,
        per_case={
            f"case_{i}": CaseBaseline(n=3, c=3 if i < all_pass else 1)
            for i in range(cases)
        },
    )


def test_a_regression_against_the_committed_file_fails() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small", all_pass=5)),
        Thresholds(min_pass_k=0.1, max_regression=0.05),
        baseline=stored_baseline(small=frozen("small", all_pass=10)),
    )
    check = next(c for c in report.checks if c.name == REGRESSION)
    assert not check.passed
    assert report.exit_code == exit_codes.GATE_FAILED


def test_the_regression_check_names_the_cases_that_got_worse() -> None:
    """The point of storing per-case counts in the file.

    "pass^3 fell four points" sends somebody re-running the matrix; "case_9
    went 3/3 to 1/3" sends them to one case.
    """
    report = evaluate(
        card(row("baseline", role="baseline"), row("small", all_pass=9)),
        Thresholds(min_pass_k=0.1, max_regression=0.0),
        baseline=stored_baseline(small=frozen("small", all_pass=10)),
    )
    check = next(c for c in report.checks if c.name == REGRESSION)
    assert "case_9 (3/3 to 1/3)" in check.detail


def test_an_improvement_is_not_a_regression() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small", all_pass=10)),
        Thresholds(min_pass_k=0.1, max_regression=0.0),
        baseline=stored_baseline(small=frozen("small", all_pass=6)),
    )
    check = next(c for c in report.checks if c.name == REGRESSION)
    assert check.passed
    assert "held or improved" in check.detail


def test_a_new_candidate_has_nothing_to_regress_against() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("brand-new")),
        Thresholds(min_pass_k=0.1, max_regression=0.05),
        baseline=stored_baseline(small=frozen("small")),
    )
    check = next(c for c in report.checks if c.name == REGRESSION)
    assert check.skipped
    assert "not in the committed baseline" in check.detail


def test_asking_for_a_regression_check_without_a_baseline_is_skipped_loudly() -> None:
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")),
        Thresholds(min_pass_k=0.1, max_regression=0.05),
        baseline=None,
    )
    check = next(c for c in report.checks if c.name == REGRESSION)
    assert check.skipped
    assert "baseline save" in check.detail


def test_a_version_difference_is_a_caveat_not_a_refusal() -> None:
    stored = replace(stored_baseline(small=frozen("small")), slmeval_version="0.0.1")
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")),
        Thresholds(min_pass_k=0.1, max_regression=0.05),
        baseline=stored,
    )
    assert any("different slmeval versions" in note for note in report.warnings)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


def test_flags_override_the_config_and_unset_flags_do_not() -> None:
    from slmeval.config.models import GateCfg

    cfg = GateCfg(min_pass_k=0.8, max_cost_ratio=0.4)
    thresholds = Thresholds.from_config(cfg, min_pass_k=0.95, max_cost_ratio=None)
    assert thresholds.min_pass_k == 0.95
    assert thresholds.max_cost_ratio == 0.4
    assert thresholds.use_wilson_lower is True


def test_a_gate_with_only_safety_on_says_so() -> None:
    """Green here means "called nothing forbidden", not "good enough", and the
    difference has to be said out loud."""
    report = evaluate(
        card(row("baseline", role="baseline"), row("small")), Thresholds()
    )
    assert report.passed
    assert any("no pass^k floor" in note for note in report.warnings)


# --------------------------------------------------------------------------- #
# The committed file
# --------------------------------------------------------------------------- #


def test_a_baseline_round_trips(tmp_path: Path) -> None:
    before = stored_baseline(small=frozen("small", all_pass=7))
    path = before.save(tmp_path / "slmeval.baseline.json")
    after = Baseline.load(path)

    assert after.run_id == before.run_id
    assert after.dataset_hash == "ds"
    stored = after.candidate("small")
    assert stored is not None
    assert stored.pass_hat_k == pytest.approx(0.7)
    assert stored.per_case["case_8"] == CaseBaseline(n=3, c=1)


def test_a_baseline_from_a_future_schema_is_refused(tmp_path: Path) -> None:
    """Rather than parsed on a best-effort basis.

    A file this version half-understands would make the gate compare against
    numbers it got wrong, which is worse than asking the user to re-save.
    """
    path = tmp_path / "slmeval.baseline.json"
    path.write_text('{"schema": "slmeval.baseline/99"}', encoding="utf-8")
    with pytest.raises(Exception, match="schema"):
        Baseline.load(path)


def test_an_unmeasured_figure_stays_null_in_the_file(tmp_path: Path) -> None:
    """A baseline that wrote an unpriced model's cost as 0 would make every
    later candidate look infinitely more expensive."""
    source = card(row("baseline", role="baseline", cost=None), row("small", p95=900))
    saved = Baseline.from_scorecard(source)
    path = saved.save(tmp_path / "b.json")
    loaded = Baseline.load(path)

    reference = loaded.candidate("baseline")
    assert reference is not None and reference.cost_per_case is None
    # Throughput run: the p95 is the scheduler's, so it is not frozen into a
    # file that later runs would be gated against.
    small = loaded.candidate("small")
    assert small is not None and small.p95_ms is None


def test_a_latency_run_does_freeze_its_p95(tmp_path: Path) -> None:
    source = card(row("small", p95=900), scheduling="latency")
    loaded = Baseline.load(Baseline.from_scorecard(source).save(tmp_path / "b.json"))
    small = loaded.candidate("small")
    assert small is not None and small.p95_ms == 900


def test_consistency_fixture_is_what_the_gate_reads() -> None:
    """Guards the arithmetic the two assertions above depend on."""
    stats = aggregate([Consistency(n=3, c=3, k=3)] * 10)
    assert stats.pass_hat_k == 1.0
    assert stats.lower is not None
    assert stats.lower == pytest.approx(0.722, abs=0.005)
