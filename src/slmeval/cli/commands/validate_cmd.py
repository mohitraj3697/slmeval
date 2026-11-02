"""`slmeval validate` — check the config and dataset before spending GPU time.

Also home to the cross-checks that neither schema can do alone. The most
valuable one is the `target_nodes` spell-check: "I benchmarked for an hour and
got identical numbers" is always a candidate pointing at a node name that does
not exist.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.table import Table
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.render import console, symbols
from slmeval.cli.wrap import handle_errors
from slmeval.config import (
    LoadedConfig,
    find_config,
    load_config,
    require_dataset,
    resolve_entrypoint,
)
from slmeval.config.locate import locate
from slmeval.dataset import Dataset, load_dataset
from slmeval.errors import Problem

MIN_RUNS_FOR_CONSISTENCY = 3


def cross_check(loaded: LoadedConfig, dataset: Dataset) -> list[Problem]:
    """Problems that need both the config and the dataset to see."""
    cfg = loaded.config
    text = loaded.text
    warnings: list[Problem] = []

    if not cfg.candidates:
        warnings.append(
            Problem(
                path="candidates",
                message="no candidates defined - there is nothing to compare the baseline against",
            )
        )

    # Node names, spell-checked against the trajectories declared in the dataset.
    known_nodes = {
        node for case in dataset.cases for path in case.assertions.reference_trajectory
        for node in path
    }
    if known_nodes:
        for i, candidate in enumerate(cfg.candidates):
            unknown = [n for n in candidate.target_nodes if n not in known_nodes]
            if unknown:
                warnings.append(
                    Problem(
                        path=f"candidates[{i}].target_nodes",
                        message=f"{', '.join(repr(n) for n in unknown)} "
                        f"appear in no reference_trajectory. If misspelled, this candidate "
                        f"changes nothing and will score identically to the baseline. "
                        f"Known nodes: {', '.join(sorted(known_nodes))}",
                        line=locate(text, ("candidates", i, "target_nodes")),
                    )
                )
    elif any(c.target_nodes for c in cfg.candidates):
        warnings.append(
            Problem(
                path="assert.reference_trajectory",
                message="no case declares a reference_trajectory, so target_nodes cannot be "
                "checked for typos",
            )
        )

    if cfg.benchmark.runs_per_case < MIN_RUNS_FOR_CONSISTENCY:
        warnings.append(
            Problem(
                path="benchmark.runs_per_case",
                message=f"{cfg.benchmark.runs_per_case} run per case cannot measure "
                "consistency. Agents are not deterministic even at temperature 0; "
                "3 is the floor for pass^k.",
                line=locate(text, ("benchmark", "runs_per_case")),
            )
        )

    if cfg.benchmark.tool_mode == "live":
        warnings.append(
            Problem(
                path="benchmark.tool_mode",
                message="'live' executes your real tools against real APIs on every run",
                line=locate(text, ("benchmark", "tool_mode")),
            )
        )

    if cfg.benchmark.fault_injection_rate > 0 and cfg.benchmark.tool_mode == "record":
        warnings.append(
            Problem(
                path="benchmark.fault_injection_rate",
                message="faults are not injected while recording (the recording would capture "
                "the fault instead of the real response), so this setting has no effect",
                line=locate(text, ("benchmark", "fault_injection_rate")),
            )
        )

    empty = [c.case_id for c in dataset.cases if c.assertions.is_empty]
    if empty:
        warnings.append(
            Problem(
                path="dataset",
                message=f"case(s) {', '.join(empty[:5])} declare no assertions and therefore "
                "can never fail",
            )
        )

    if len(dataset.cases) >= 3 and not any(c.assertions.forbidden_tools for c in dataset.cases):
        warnings.append(
            Problem(
                path="dataset",
                message="no case declares forbidden_tools. A small model taking a destructive "
                "action it was not asked to take is the failure worth catching first.",
            )
        )

    return warnings


def as_payload(
    loaded: LoadedConfig, dataset: Dataset, warnings: list[Problem]
) -> dict[str, Any]:
    """The same summary as a machine-readable dict.

    Exists for one job, and it is worth naming: a CI workflow has to key its
    cassette cache on something, and the only correct key is the *content* hash
    of the dataset the cassettes were recorded for. A workflow that keyed on a
    branch name, or on nothing, would restore a stale cache and then report
    every uncovered path as a model regression.

    The config hash rides along because it is what `gate` refuses to compare
    across, so a workflow can print both and a reviewer can see at a glance
    whether two runs were even the same experiment.
    """
    cfg = loaded.config
    return {
        "schema": "slmeval.validate/1",
        "ok": True,
        "warnings": [
            {"path": w.path, "message": w.message, "line": w.line} for w in warnings
        ],
        "benchmark": cfg.benchmark.name,
        "config": str(loaded.path),
        "config_hash": cfg.config_hash(),
        "dataset": str(dataset.path),
        "dataset_hash": dataset.content_hash,
        "cases": len(dataset),
        "tags": list(dataset.tags),
        "target": cfg.target.entrypoint,
        "adapter": cfg.target.adapter,
        "baseline": cfg.baseline.id,
        "candidates": [c.id for c in cfg.candidates],
        "runs_per_case": cfg.benchmark.runs_per_case,
        "tool_mode": cfg.benchmark.tool_mode,
        "executions": len(cfg.all_models) * len(dataset) * cfg.benchmark.runs_per_case,
    }


def _summary(loaded: LoadedConfig, dataset: Dataset) -> Table:
    cfg = loaded.config
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", no_wrap=True)
    table.add_column(overflow="fold")

    table.add_row("benchmark", f"{cfg.benchmark.name}  [config {cfg.config_hash()}]")
    table.add_row("config", str(loaded.path))
    table.add_row(
        "dataset",
        f"{dataset.path.name} - {len(dataset)} case(s), hash {dataset.content_hash}"
        + (f", tags: {', '.join(dataset.tags)}" if dataset.tags else ""),
    )
    table.add_row("target", f"{cfg.target.entrypoint}  (adapter: {cfg.target.adapter})")
    table.add_row(
        "matrix",
        f"{len(cfg.all_models)} model(s) x {len(dataset)} case(s) x "
        f"{cfg.benchmark.runs_per_case} run(s) = "
        f"{len(cfg.all_models) * len(dataset) * cfg.benchmark.runs_per_case} executions",
    )
    table.add_row(
        "tools",
        f"mode={cfg.benchmark.tool_mode}, "
        f"faults={cfg.benchmark.fault_injection_rate:.0%}",
    )
    table.add_row("baseline", cfg.baseline.describe())
    for candidate in cfg.candidates:
        table.add_row(f"candidate {candidate.id}", candidate.describe())
    return table


def register(app: typer.Typer) -> None:
    @app.command()
    @handle_errors
    def validate(
        config: Annotated[
            Path | None,
            typer.Option("--config", "-c", help="Path to eval.toml. Default: search upwards."),
        ] = None,
        strict: Annotated[
            bool,
            typer.Option(
                "--strict",
                help="Import the target entrypoint and treat warnings as errors (for CI).",
            ),
        ] = False,
        as_json: Annotated[
            bool,
            typer.Option(
                "--json",
                help="Emit the summary - including the config and dataset hashes - "
                "as JSON. For CI cache keys and provenance checks.",
            ),
        ] = False,
    ) -> None:
        """Validate eval.toml, the dataset, and the links between them."""
        out = console()
        sym = symbols()

        loaded = load_config(find_config(config))
        dataset = load_dataset(require_dataset(loaded))
        warnings = cross_check(loaded, dataset)

        if as_json:
            # print, not rich-print: this is data on its way into a pipe or a
            # `$GITHUB_OUTPUT`, and Rich would wrap and style it into something
            # no parser can read.
            print(json.dumps(as_payload(loaded, dataset, warnings), indent=2))
            if warnings and strict:
                raise typer.Exit(exit_codes.USAGE)
            return

        out.print()
        out.print(_summary(loaded, dataset))

        if strict:
            func = resolve_entrypoint(loaded)
            kind = "async" if inspect.iscoroutinefunction(func) else "sync"
            out.print()
            out.print(
                Text(f"  {sym.ok} ", style="green"),
                end="",
            )
            out.print(
                Text(
                    f"imported {loaded.config.target.entrypoint} ({kind})",
                    style="dim",
                )
            )
            if kind == "sync":
                warnings.append(
                    Problem(
                        path="target.entrypoint",
                        message="target is a plain function; it will be run in a worker thread, "
                        "where contextvars-based tool interception is more fragile. An "
                        "`async def` target is strongly preferred.",
                    )
                )

        out.print()
        for warning in warnings:
            out.print(Text(f"  {sym.warn} {warning.path}", style="yellow"), end="")
            out.print(Text(f"  {sym.arrow}  {warning.message}"))
            if warning.line:
                out.print(
                    Text(
                        f"      {loaded.path.name}:{warning.line}",
                        style="dim",
                    )
                )
        if warnings:
            out.print()

        count = len(warnings)
        if count and strict:
            out.print(
                Text(
                    f"  {sym.fail} {count} warning(s), and --strict was given",
                    style="bold red",
                )
            )
            out.print()
            raise typer.Exit(exit_codes.USAGE)

        label = f"valid - {count} warning(s)" if count else "valid"
        out.print(Text(f"  {sym.ok} {label}", style="bold green"))
        out.print()
