"""What a run was, so a later comparison is evidence and not guesswork.

Two scorecards side by side are only meaningful if the things around the models
were held still: same dataset, same config, same harness. Nobody remembers
whether they edited a case last Tuesday. So every run records what it was built
from, and ``slmeval compare`` can refuse a comparison whose premises differ.

That refusal is the feature. A tool that cheerfully prints a 12-point
regression caused by a dataset edit teaches people to distrust its numbers,
once, permanently.
"""

from __future__ import annotations

import os
import platform
import secrets
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from slmeval import __version__
from slmeval.storage import codec

GIT_TIMEOUT_S: Final = 2.0
"""Provenance is never worth stalling a run for. A slow or wedged git - a
network-mounted repository, a stale index lock - degrades to "no SHA recorded",
which is a gap in a report, not a failed benchmark.
"""


def new_run_id(*, now: datetime | None = None) -> str:
    """``YYYYMMDD-HHMMSS-<6 hex>``.

    Lexically sortable, so ``ORDER BY run_id`` is chronological and a user
    skimming ``.slmeval/`` can read the dates. The random suffix because two
    runs started in the same second on one machine is not hypothetical - it is
    what a CI matrix does.
    """
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{secrets.token_hex(3)}"


def git_sha(root: Path | None = None) -> str:
    """Short commit SHA, with ``-dirty`` when the tree has changes.

    The dirty marker earns its place: "which commit scored 83%" is useless if
    the answer is a commit plus uncommitted edits nobody kept. Marked as dirty,
    a user at least knows not to trust the SHA.

    Every failure returns the empty string. Git may be absent, the directory
    may not be a repository, and neither is a reason to fail a benchmark.
    """
    cwd = str(root or Path.cwd())
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if sha.returncode != 0:
            return ""
        value = sha.stdout.strip()
        if not value:
            return ""
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=GIT_TIMEOUT_S,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            return f"{value}-dirty"
        return value
    except (OSError, subprocess.SubprocessError):
        return ""


def host_info() -> dict[str, Any]:
    """Enough about the machine to explain a latency difference.

    Latency and p95 are headline columns, and they are properties of the
    machine as much as the model. A run from a laptop on battery and one from a
    CI runner are not the same measurement; recorded here, that is visible
    instead of mysterious.

    Deliberately no hostname, no username, no local paths beyond the project
    the user chose: this file gets attached to bug reports and pasted into
    issues.
    """
    return {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "implementation": sys.implementation.name,
        "cpus": os.cpu_count() or 0,
    }


@dataclass(frozen=True, slots=True)
class Provenance:
    """The inputs to one run, hashed and stamped."""

    config_hash: str
    dataset_hash: str
    dataset_path: str = ""
    git_sha: str = ""
    slmeval_version: str = __version__
    host: str = "{}"
    started_at: str = ""

    @classmethod
    def collect(
        cls,
        *,
        config_hash: str,
        dataset_hash: str,
        dataset_path: Path | str = "",
        root: Path | None = None,
    ) -> Provenance:
        return cls(
            config_hash=config_hash,
            dataset_hash=dataset_hash,
            dataset_path=str(dataset_path),
            git_sha=git_sha(root),
            slmeval_version=__version__,
            host=codec.dumps(host_info()),
            started_at=datetime.now(UTC).isoformat(),
        )


# Fields that must match for two runs to be comparable, and what to say when
# they do not. Ordered by how badly each one invalidates a comparison, because
# the CLI shows the first line most prominently.
_COMPARABLE: Final = (
    ("dataset_hash", "the dataset changed between these runs"),
    ("config_hash", "the benchmark config changed between these runs"),
    ("mode", "one run replayed cassettes and the other did not"),
    ("seed", "the runs used different seeds, so they met different faults"),
    ("slmeval_version", "the runs were produced by different slmeval versions"),
)


@dataclass(frozen=True, slots=True)
class Divergence:
    """One reason two runs are not comparable."""

    field: str
    reason: str
    left: str
    right: str

    @property
    def blocking(self) -> bool:
        """Whether this difference should stop a comparison outright.

        A different ``slmeval_version`` is a caveat - the numbers were computed
        by different code, worth printing, rarely worth refusing over. A
        different dataset, config, mode or seed means the two tables answer
        different questions, and putting them in one view manufactures a
        regression out of thin air.
        """
        return self.field != "slmeval_version"


def divergences(left: object, right: object) -> list[Divergence]:
    """Why two :class:`~slmeval.storage.models.RunRecord` values differ.

    Takes anything with the provenance attributes rather than importing the
    record type, so the config layer can check a pending run against a stored
    one before spending twenty minutes on a comparison nobody can use.
    """
    out: list[Divergence] = []
    for field_name, reason in _COMPARABLE:
        a = getattr(left, field_name, None)
        b = getattr(right, field_name, None)
        if a != b:
            out.append(
                Divergence(field=field_name, reason=reason, left=str(a), right=str(b))
            )
    return out


__all__ = [
    "GIT_TIMEOUT_S",
    "Divergence",
    "Provenance",
    "divergences",
    "git_sha",
    "host_info",
    "new_run_id",
]
