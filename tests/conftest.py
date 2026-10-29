"""Shared fixtures.

The whole suite runs offline: no GPU, no API keys, no network. A test suite you
stop running because it is slow or flaky protects nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest

from slmeval.cli.commands.init_cmd import scaffold
from slmeval.sandbox.tool import REGISTRY


@pytest.fixture(autouse=True)
def isolated_tool_registry() -> Iterator[None]:
    """Restore the sandboxed-tool registry around every test.

    `@sandbox_tool` registers by name at import time, into one process-wide
    registry, and registering two different functions under one name is a hard
    error - cassettes are keyed by tool name, so a silent collision would have
    two tools answering each other's recordings. That check is right, which
    means any test module that declares or imports a tool called `fetch_order`
    would otherwise poison every module after it. Snapshotting here, rather
    than in each file, is what keeps that from being a trap for the next test.
    """
    saved = dict(REGISTRY)
    yield
    REGISTRY.clear()
    REGISTRY.update(saved)


@pytest.fixture(autouse=True)
def stable_terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin width and colour so substring assertions are not at the mercy of
    whatever terminal the suite happens to run in."""
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.setenv("LINES", "50")
    monkeypatch.setenv("NO_COLOR", "1")
    monkeypatch.delenv("TERM", raising=False)


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """A scaffolded project, exactly as `slmeval init` leaves it."""
    scaffold(tmp_path, name="test-benchmark")
    return tmp_path


@pytest.fixture
def mock_http() -> Iterator[Callable[[Callable[[httpx.Request], httpx.Response]], httpx.Client]]:
    """Build an httpx.Client backed by a scripted handler."""
    clients: list[httpx.Client] = []

    def factory(handler: Callable[[httpx.Request], httpx.Response]) -> httpx.Client:
        client = httpx.Client(transport=httpx.MockTransport(handler))
        clients.append(client)
        return client

    yield factory
    for client in clients:
        client.close()


def write_config(directory: Path, body: str) -> Path:
    """Write an eval.toml with a dataset that exists, so only `body` is under test."""
    datasets = directory / "datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    (datasets / "cases.jsonl").write_text(
        '{"case_id": "c1", "input": {"q": "hi"}, "assert": {"required_tools": ["t"]}}\n',
        encoding="utf-8",
    )
    path = directory / "eval.toml"
    path.write_text(body, encoding="utf-8")
    return path


MINIMAL_CONFIG = """\
[benchmark]
name = "t"
dataset = "./datasets/cases.jsonl"

[target]
entrypoint = "mod:run_case"

[baseline]
provider = "fake"
model = "m"
"""
