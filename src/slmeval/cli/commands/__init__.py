"""One module per command."""

from __future__ import annotations

from slmeval.cli.commands import dataset_cmd, doctor_cmd, init_cmd, stubs, validate_cmd

__all__ = ["dataset_cmd", "doctor_cmd", "init_cmd", "stubs", "validate_cmd"]
