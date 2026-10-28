"""`slmeval doctor` — can this machine run the benchmark right now?"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.text import Text

from slmeval import exit_codes
from slmeval.cli.render import checks_summary, console, render_checks
from slmeval.cli.wrap import handle_errors
from slmeval.config import LoadedConfig, find_config, load_config
from slmeval.doctor import Status, run_checks, worst_status
from slmeval.errors import SlmevalError


def _try_load(config: Path | None) -> tuple[LoadedConfig | None, str | None, Path]:
    """Load the config if there is one, reporting failure as a check rather than
    an exception — doctor's whole job is to report problems, not to hit them."""
    try:
        path = find_config(config)
    except SlmevalError as err:
        return None, (err.message if config is not None else None), Path.cwd()

    try:
        return load_config(path), None, path.parent
    except SlmevalError as err:
        return None, err.message, path.parent


def register(app: typer.Typer) -> None:
    @app.command()
    @handle_errors
    def doctor(
        config: Annotated[
            Path | None,
            typer.Option("--config", "-c", help="Path to eval.toml. Default: search upwards."),
        ] = None,
        skip_network: Annotated[
            bool,
            typer.Option("--skip-network", help="Do not probe providers (offline / CI use)."),
        ] = False,
    ) -> None:
        """Check Python, storage, credentials and provider reachability."""
        out = console()
        loaded, config_error, root = _try_load(config)

        checks = run_checks(
            root=root,
            loaded=loaded,
            config_error=config_error,
            skip_network=skip_network,
        )

        out.print()
        out.print(render_checks(checks))
        out.print()
        out.print(Text("  ", end=""), end="")
        out.print(checks_summary(checks))
        out.print()

        status = worst_status(checks)
        if status is Status.FAIL:
            # Exit 3, never 1. A broken environment is not a model regression,
            # and CI must be able to tell them apart.
            raise typer.Exit(exit_codes.ENVIRONMENT)
