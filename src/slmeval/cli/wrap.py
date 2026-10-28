"""Turn user-fixable errors into readable output and a documented exit code.

Applied to every command, so the behaviour is identical whether the CLI is
invoked by a shell, by CI, or by `typer.testing.CliRunner` in the test suite.
A harness bug keeps its traceback — that distinction is deliberate.
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from typing import Any, TypeVar

import typer

from slmeval.cli.render import render_error
from slmeval.errors import SlmevalError

F = TypeVar("F", bound=Callable[..., Any])


def handle_errors(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except SlmevalError as err:
            render_error(err)
            raise typer.Exit(err.exit_code) from None

    return wrapper  # type: ignore[return-value]
