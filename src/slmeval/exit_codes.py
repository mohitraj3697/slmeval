"""Process exit codes — a documented contract, not an implementation detail.

CI pipelines and shell scripts branch on these, so they are frozen from 0.1.0.
The important distinction is 1 vs 3: a regression must never be confused with
"Ollama was not running", or nobody will trust the gate.
"""

from __future__ import annotations

from typing import Final

OK: Final = 0
"""Everything passed."""

GATE_FAILED: Final = 1
"""The run completed and a quality gate was not met. A real regression."""

USAGE: Final = 2
"""Bad invocation: malformed config, unknown candidate, missing dataset."""

ENVIRONMENT: Final = 3
"""Infrastructure problem: provider unreachable, model not pulled, cache unwritable.
Not a model regression. CI should treat this as a broken job, not a failed test."""

COVERAGE: Final = 4
"""Cassette coverage insufficient to judge the run (too many uncovered paths)."""

__all__ = ["COVERAGE", "ENVIRONMENT", "GATE_FAILED", "OK", "USAGE"]
