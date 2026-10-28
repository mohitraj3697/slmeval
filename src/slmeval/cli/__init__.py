"""Terminal interface."""

from __future__ import annotations

__all__ = ["app", "main"]


def __getattr__(name: str) -> object:
    # Lazy so `import slmeval.cli` stays cheap and import cycles cannot form
    # between main.py and the command modules.
    if name in __all__:
        from slmeval.cli import main as _main

        return getattr(_main, name)
    raise AttributeError(name)
