"""One module per command."""

from __future__ import annotations

from slmeval.cli.commands import (
    cache_cmd,
    dataset_cmd,
    doctor_cmd,
    init_cmd,
    record_cmd,
    report_cmd,
    run_cmd,
    stubs,
    validate_cmd,
)

__all__ = [
    "cache_cmd",
    "dataset_cmd",
    "doctor_cmd",
    "init_cmd",
    "record_cmd",
    "report_cmd",
    "run_cmd",
    "stubs",
    "validate_cmd",
]
