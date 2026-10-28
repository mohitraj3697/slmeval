"""User-facing errors.

Anything raised from these types is a problem the *user* can fix, and the CLI
renders it as a readable block plus an exit code — never a traceback. Harness
bugs keep their traceback, because those are ours to fix and the distinction
matters: a tool that reports its own crashes as user errors trains people to
ignore its output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from slmeval import exit_codes


@dataclass(frozen=True)
class Problem:
    """One specific thing that is wrong, located if possible."""

    path: str
    message: str
    line: int | None = None
    value: str | None = None


class SlmevalError(Exception):
    """Base class for errors that are the user's to fix."""

    exit_code: int = exit_codes.USAGE

    def __init__(
        self,
        message: str,
        *,
        hint: str | None = None,
        source: Path | None = None,
        problems: list[Problem] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.hint = hint
        self.source = source
        self.problems: list[Problem] = problems or []


class ConfigError(SlmevalError):
    """eval.toml is missing, unparseable, or invalid."""


class DatasetError(SlmevalError):
    """The golden dataset is missing or malformed."""


class EnvironmentError_(SlmevalError):
    """Provider unreachable, model not pulled, cache unwritable.

    Exits 3 so CI can tell a broken environment from a real regression.
    """

    exit_code = exit_codes.ENVIRONMENT


@dataclass
class ValidationReport:
    """Collected problems for commands that check rather than fail fast."""

    problems: list[Problem] = field(default_factory=list)

    def add(self, path: str, message: str, line: int | None = None) -> None:
        self.problems.append(Problem(path=path, message=message, line=line))

    @property
    def ok(self) -> bool:
        return not self.problems
