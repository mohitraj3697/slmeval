"""slmeval — benchmark agent workflows across small language models.

The public API is intentionally tiny. Users integrate with two touch points:

    from slmeval import chat_model, node, sandbox_tool

Both are no-ops outside a benchmark run, so the integration is safe to leave in
production code. They land in Week 1 (Steps 5-8); this release ships the
scaffold, environment doctor, and configuration layer.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
