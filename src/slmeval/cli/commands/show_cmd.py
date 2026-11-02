"""``slmeval show`` - the drill-down, from the scorecard to the exact step.

The scorecard is the headline; this is the audit trail under it. Four views,
each one a question somebody asks while staring at a red cell:

``show run <id>``
    Which executions exist and how did each end? Worst first.
``show exec <id>``
    What did this one actually do - trajectory, model calls, tool calls,
    metrics, final state?
``show case <id> --candidate <id>``
    Where did it part company with the baseline? Side by side, divergence
    marked, cassette-miss diagnosis inline.
``show uncovered``
    Which tool calls had no recording, so where does ``record --fill-gaps``
    need to go?

Like ``report``, none of this loads the user's agent and none of it re-executes
anything. A checkout whose agent has since been refactored still has valid
results from last Tuesday, and a view that could not open them would be the
tool losing data it already holds.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import typer

from slmeval.cli.loading import LATEST, Project, open_project, resolve_run
from slmeval.cli.render import console
from slmeval.cli.wrap import handle_errors
from slmeval.errors import ConfigError
from slmeval.report.drilldown import (
    Diff,
    build_diff,
    diff_payload,
    execution_payload,
    gaps,
    gaps_payload,
    known_tool_names,
    rebuild_diagnosis,
    render_diff,
    render_execution,
    render_gaps,
    render_run,
    tool_steps,
)
from slmeval.sandbox.diagnosis import MissDiagnosis
from slmeval.storage.db import RunStore
from slmeval.storage.models import ExecutionDetail, ExecutionRecord, Outcome

MISS_OUTCOMES = frozenset({Outcome.CASSETTE_MISS, Outcome.UNCOVERED})

show_app = typer.Typer(
    no_args_is_help=True,
    help="Drill into a run, an execution, or one case against the baseline.",
)

_ConfigOpt = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="Path to eval.toml. Defaults to the nearest one."),
]
_AsciiOpt = Annotated[
    bool, typer.Option("--ascii", help="ASCII only, for legacy Windows consoles.")
]
_JsonOpt = Annotated[
    bool, typer.Option("--json", help="Emit the view as JSON instead of a table.")
]
_RunOpt = Annotated[
    str,
    typer.Option("--run-id", help="A run id or prefix, or 'latest'."),
]


# --------------------------------------------------------------------------- #
# show run
# --------------------------------------------------------------------------- #


@show_app.command("run")
@handle_errors
def show_run(
    run_id: Annotated[
        str, typer.Argument(help="A run id or prefix, or 'latest'.")
    ] = LATEST,
    config: _ConfigOpt = None,
    candidate: Annotated[
        str | None, typer.Option("--candidate", help="Only this candidate's cells.")
    ] = None,
    case: Annotated[str | None, typer.Option("--case", help="Only this case.")] = None,
    failed: Annotated[
        bool, typer.Option("--failed", help="Only the executions that did not pass.")
    ] = False,
    limit: Annotated[
        int, typer.Option("--limit", min=0, help="Rows to show. 0 for all.")
    ] = 40,
    as_json: _JsonOpt = False,
    ascii_mode: _AsciiOpt = False,
) -> None:
    """List a run's executions, worst first."""
    with open_project(config, load_agent=False) as project:
        resolved = resolve_run(project.store, run_id)
        run = project.store.require_run(resolved)
        candidates = project.store.candidates(resolved)
        executions = _filtered(
            project.store.executions(resolved, candidate_id=candidate, case_id=case),
            failed=failed,
        )
        if as_json:
            _print_json(
                {
                    "run_id": run.run_id,
                    "name": run.name,
                    "status": run.status.value,
                    "mode": run.mode,
                    "seed": run.seed,
                    "runs_per_case": run.runs_per_case,
                    "config_hash": run.config_hash,
                    "dataset_hash": run.dataset_hash,
                    "candidates": [
                        {
                            "candidate_id": c.candidate_id,
                            "role": c.role,
                            "provider": c.provider,
                            "model": c.model,
                        }
                        for c in candidates
                    ],
                    "executions": [
                        {
                            "exec_id": e.exec_id,
                            "candidate_id": e.candidate_id,
                            "case_id": e.case_id,
                            "repeat_idx": e.repeat_idx,
                            "status": e.status.value,
                            "outcome": e.outcome_class.value,
                            "steps": e.steps,
                            "duration_ms": e.duration_ms,
                            "error_class": e.error_class,
                            "error_message": e.error_message,
                        }
                        for e in executions
                    ],
                }
            )
            return
        out = console()
        out.print(
            render_run(
                run,
                candidates,
                executions,
                ascii_mode=ascii_mode or None,
                limit=limit,
            )
        )
        out.print()


# --------------------------------------------------------------------------- #
# show exec
# --------------------------------------------------------------------------- #


@show_app.command("exec")
@handle_errors
def show_exec(
    exec_id: Annotated[
        str,
        typer.Argument(
            help="An execution id, or its readable tail: gemma3-4b.refund_207.r0"
        ),
    ],
    config: _ConfigOpt = None,
    run_id: _RunOpt = LATEST,
    as_json: _JsonOpt = False,
    ascii_mode: _AsciiOpt = False,
) -> None:
    """Show one execution's timeline, metrics and final state.

    The id is resolved inside one run - the latest unless ``--run-id`` says
    otherwise - because the readable tail is identical in every run of a
    project: ``record`` alone leaves a second execution under every one of
    them. A full execution id skips the question.
    """
    with open_project(config, load_agent=False) as project:
        resolved = _resolve_exec(project.store, exec_id, run_id)
        detail = project.store.reconstruct(resolved)
        if as_json:
            _print_json(execution_payload(detail))
            return
        out = console()
        out.print(
            render_execution(
                detail,
                ascii_mode=ascii_mode or None,
                diagnoses=_diagnoses(detail, project),
            )
        )


# --------------------------------------------------------------------------- #
# show case
# --------------------------------------------------------------------------- #


@show_app.command("case")
@handle_errors
def show_case(
    case_id: Annotated[str, typer.Argument(help="A case id from the dataset.")],
    config: _ConfigOpt = None,
    run_id: _RunOpt = LATEST,
    candidate: Annotated[
        list[str] | None,
        typer.Option(
            "--candidate",
            help="Diff only these candidates. Defaults to every non-baseline one.",
        ),
    ] = None,
    repeat: Annotated[
        int | None,
        typer.Option(
            "--repeat",
            min=0,
            help="Which repeat to show. Defaults to the worst one, which is the "
            "reason you are here.",
        ),
    ] = None,
    as_json: _JsonOpt = False,
    ascii_mode: _AsciiOpt = False,
) -> None:
    """Diff one case against the baseline, side by side."""
    with open_project(config, load_agent=False) as project:
        resolved = resolve_run(project.store, run_id)
        store = project.store
        executions = store.executions(resolved, case_id=case_id)
        if not executions:
            raise ConfigError(
                f"run {resolved} has no execution for case {case_id!r}",
                hint="`slmeval show run` lists the cases a run covered",
            )

        baseline_id = _baseline_id(store, resolved)
        wanted = _wanted_candidates(executions, baseline_id, candidate)
        diffs = [
            _diff_for(
                project,
                case_id=case_id,
                executions=executions,
                baseline_id=baseline_id,
                candidate_id=candidate_id,
                repeat=repeat,
            )
            for candidate_id in wanted
        ]

        if as_json:
            _print_json(
                {
                    "run_id": resolved,
                    "case_id": case_id,
                    "diffs": [diff_payload(diff) for diff in diffs],
                }
            )
            return

        out = console()
        for diff in diffs:
            out.print(render_diff(diff, ascii_mode=ascii_mode or None))
        out.print()


# --------------------------------------------------------------------------- #
# show uncovered
# --------------------------------------------------------------------------- #


@show_app.command("uncovered")
@handle_errors
def show_uncovered(
    run_id: Annotated[
        str, typer.Argument(help="A run id or prefix, or 'latest'.")
    ] = LATEST,
    config: _ConfigOpt = None,
    candidate: Annotated[
        str | None, typer.Option("--candidate", help="Only this candidate's misses.")
    ] = None,
    all_misses: Annotated[
        bool,
        typer.Option(
            "--all",
            help="Include hallucinated calls too, not only the coverage gaps.",
        ),
    ] = False,
    as_json: _JsonOpt = False,
    ascii_mode: _AsciiOpt = False,
) -> None:
    """List the tool calls no cassette covered - the input to `record --fill-gaps`.

    Only the uncovered verdict by default. The other three verdicts are model
    failures and belong in the failure report, not in a list of recordings to
    go and make; mixing them would have a user re-recording a hallucinated
    order id as if it were ground truth.
    """
    with open_project(config, load_agent=False) as project:
        resolved = resolve_run(project.store, run_id)
        executions = [
            e
            for e in project.store.executions(resolved, candidate_id=candidate)
            if e.outcome_class in MISS_OUTCOMES
        ]
        details = [
            (e.candidate_id, project.store.reconstruct(e.exec_id)) for e in executions
        ]
        found = gaps(details, uncovered_only=not all_misses)
        if as_json:
            _print_json({"run_id": resolved, **gaps_payload(found)})
            return
        out = console()
        out.print(render_gaps(found, ascii_mode=ascii_mode or None, run_id=resolved))


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _resolve_exec(store: RunStore, exec_id: str, run_id: str) -> str:
    """One execution id, resolved inside a run before the whole project.

    Inside a run first, because ``candidate.case.rN`` is not unique across a
    project and the id a user just read off ``show run`` belongs to the run
    they were looking at. Falling back to a project-wide search afterwards is
    still worth doing - an id pasted from an older run's table should not need
    its run spelled out - and that search refuses ambiguity rather than
    guessing.
    """
    if store.get_execution(exec_id) is not None:
        return exec_id
    scoped = f"{resolve_run(store, run_id)}."
    within = f"{scoped}{exec_id.removeprefix(scoped)}"
    if store.get_execution(within) is not None:
        return within
    return store.resolve_exec_id(exec_id)


def _filtered(
    executions: Sequence[ExecutionRecord], *, failed: bool
) -> list[ExecutionRecord]:
    if not failed:
        return list(executions)
    return [e for e in executions if e.outcome_class is not Outcome.PASSED]


def _baseline_id(store: RunStore, run_id: str) -> str:
    rows = store.candidates(run_id)
    found = next((c for c in rows if c.role == "baseline"), None)
    return found.candidate_id if found else ""


def _wanted_candidates(
    executions: Sequence[ExecutionRecord], baseline_id: str, asked: list[str] | None
) -> list[str]:
    present = list(dict.fromkeys(e.candidate_id for e in executions))
    if asked:
        unknown = [c for c in asked if c not in present]
        if unknown:
            raise ConfigError(
                f"no executions for candidate(s) {', '.join(unknown)} in this run",
                hint=f"this run has: {', '.join(present)}",
            )
        return list(asked)
    others = [c for c in present if c != baseline_id]
    # With no candidates at all - a baseline-only run, which `record` produces -
    # diffing the baseline against itself is the only honest thing left, and it
    # still shows the timeline.
    return others or present


def _diff_for(
    project: Project,
    *,
    case_id: str,
    executions: Sequence[ExecutionRecord],
    baseline_id: str,
    candidate_id: str,
    repeat: int | None,
) -> Diff:
    mine = [e for e in executions if e.candidate_id == candidate_id]
    chosen = _pick_repeat(mine, repeat)
    if chosen is None:
        raise ConfigError(
            f"candidate {candidate_id!r} has no repeat {repeat} for case {case_id!r}",
            hint=f"stored repeats: {', '.join(str(e.repeat_idx) for e in mine)}",
        )
    reference = _pick_baseline(
        [e for e in executions if e.candidate_id == baseline_id], chosen.repeat_idx
    )
    return build_diff(
        case_id=case_id,
        candidate=project.store.reconstruct(chosen.exec_id),
        baseline=(
            project.store.reconstruct(reference.exec_id)
            if reference is not None and reference.exec_id != chosen.exec_id
            else None
        ),
        candidate_id=candidate_id,
        baseline_id=baseline_id or "baseline",
        cassettes=project.cassettes,
    )


def _pick_repeat(
    mine: Sequence[ExecutionRecord], repeat: int | None
) -> ExecutionRecord | None:
    """The repeat to show: the one the user asked for, or the worst one.

    The worst one by default, because a user opening the drill-down for a case
    that failed two runs in three does not want to be shown the pass. "Worst"
    means failed before passed, and an earlier repeat before a later one so the
    choice is reproducible.
    """
    if repeat is not None:
        return next((e for e in mine if e.repeat_idx == repeat), None)
    if not mine:
        return None
    return sorted(
        mine,
        key=lambda e: (e.outcome_class is Outcome.PASSED, e.repeat_idx),
    )[0]


def _pick_baseline(
    rows: Sequence[ExecutionRecord], repeat_idx: int
) -> ExecutionRecord | None:
    """The baseline execution to diff against: the matching repeat if it passed.

    A failing baseline repeat is a poor reference - the left column would be
    just as wrong as the right one, and the user would be comparing two
    failures and seeing no divergence. So a passing repeat wins over the
    matching index, and the matching index wins over anything else.
    """
    if not rows:
        return None
    passing = [e for e in rows if e.outcome_class is Outcome.PASSED]
    pool = passing or list(rows)
    exact = next((e for e in pool if e.repeat_idx == repeat_idx), None)
    return exact or pool[0]


def _diagnoses(detail: ExecutionDetail, project: Project) -> list[MissDiagnosis]:
    """Rebuild every cassette-miss block in one execution."""
    known = known_tool_names([detail], project.cassettes)
    found: list[MissDiagnosis] = []
    for step in tool_steps(detail):
        finding = rebuild_diagnosis(
            step,
            case_id=detail.execution.case_id,
            candidate_id=detail.execution.candidate_id,
            cassettes=project.cassettes,
            known_tools=known,
        )
        if finding is not None:
            found.append(finding)
    return found


def _print_json(payload: dict[str, Any]) -> None:
    """``print``, not rich-print: this is data on its way into a pipe, and Rich
    would wrap and style it into something no parser can read."""
    print(json.dumps(payload, indent=2, default=str))


def register(app: typer.Typer) -> None:
    app.add_typer(show_app, name="show")


__all__ = ["register", "show_app"]
