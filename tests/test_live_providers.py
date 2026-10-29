"""Step 6's acceptance test, against real providers.

Excluded from the default suite, because the offline suite has to stay free,
fast and runnable in CI with no GPU and no keys. These exist because the mock
transport can prove we *send* the right request but not that a real server
accepts it - and "the server accepted it" is precisely the thing that differs
between OpenAI, Ollama and llama.cpp.

Run them deliberately:

    pytest -m live                       # everything reachable
    pytest -m live -k ollama             # local only, costs nothing
    pytest -m live -k openai             # spends a fraction of a cent

Each test skips rather than fails when its provider is absent, so a partial
environment still gives a useful signal.
"""

from __future__ import annotations

import os

import httpx
import pytest
from pydantic import BaseModel, Field

from slmeval.config import ModelCfg
from slmeval.providers.registry import build_client

pytestmark = [pytest.mark.live, pytest.mark.asyncio]

OLLAMA_TAGS = "http://localhost:11434/api/tags"
OLLAMA_MODEL = os.environ.get("SLMEVAL_LIVE_OLLAMA_MODEL", "gemma3:4b")
OPENAI_MODEL = os.environ.get("SLMEVAL_LIVE_OPENAI_MODEL", "gpt-4o-mini")

PROMPT = [
    {"role": "system", "content": "You classify customer support messages."},
    {"role": "user", "content": "Where is my order ORD-9921? I ordered it last Tuesday."},
]


class Routing(BaseModel):
    """The intent of a customer support message."""

    intent: str = Field(description="one of: order_status, refund, other")
    order_id: str = Field(description="the order id mentioned, or an empty string")


def ollama_models() -> list[str]:
    try:
        response = httpx.get(OLLAMA_TAGS, timeout=3.0)
        response.raise_for_status()
    except (httpx.HTTPError, ValueError):
        return []
    return [m.get("name", "") for m in response.json().get("models", [])]


def report(label: str, result: object) -> None:
    """Print what the ladder did. This output IS the acceptance criterion."""
    assert hasattr(result, "structured_mode_used")
    print(
        f"\n  {label}: rung={result.structured_mode_used} "  # type: ignore[attr-defined]
        f"tried={result.rungs_tried} "  # type: ignore[attr-defined]
        f"json_stage={result.json_stage} "  # type: ignore[attr-defined]
        f"repairs={result.repair_attempts} "  # type: ignore[attr-defined]
        f"tokens={result.prompt_tokens}/{result.completion_tokens}"  # type: ignore[attr-defined]
        f"{'~' if result.tokens_estimated else ''} "  # type: ignore[attr-defined]
        f"latency={result.latency_ms:.0f}ms "  # type: ignore[attr-defined]
        f"retries={result.retries}"  # type: ignore[attr-defined]
    )


async def test_openai_returns_a_validated_schema() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY is not set")

    client = build_client(
        ModelCfg(id="live-openai", provider="openai", model=OPENAI_MODEL, max_tokens=200)
    )
    try:
        result = await client.complete(PROMPT, schema=Routing)
    finally:
        await client.aclose()

    report(f"openai/{OPENAI_MODEL}", result)
    assert isinstance(result.parsed, Routing)
    assert "9921" in result.parsed.order_id
    # A frontier model should not need our parser's help or a re-ask. If this
    # ever fails, the finding is real.
    assert result.structured_mode_used == "tools"
    assert result.repair_attempts == 0
    assert result.tokens_estimated is False


async def test_ollama_returns_a_validated_schema() -> None:
    available = ollama_models()
    if not available:
        pytest.skip("Ollama is not reachable at localhost:11434 (`ollama serve`)")
    if not any(m.startswith(OLLAMA_MODEL.split(":")[0]) for m in available):
        pytest.skip(
            f"{OLLAMA_MODEL} is not pulled (`ollama pull {OLLAMA_MODEL}`); have {available}"
        )

    client = build_client(
        ModelCfg(id="live-ollama", provider="ollama", model=OLLAMA_MODEL, max_tokens=200)
    )
    try:
        result = await client.complete(PROMPT, schema=Routing)
    finally:
        await client.aclose()

    report(f"ollama/{OLLAMA_MODEL}", result)
    assert isinstance(result.parsed, Routing)
    # Deliberately no assertion on which rung worked: that it may be json_mode,
    # and may need a repair, is the whole premise of the tool.
    assert result.structured_mode_used in ("tools", "json_schema", "json_mode", "prompt")


async def test_the_same_call_works_against_every_reachable_provider() -> None:
    """Step 6's done-when: one `complete(schema=X)` against cloud, local and fake."""
    from slmeval.providers.fake import FakeTurn, register_script

    register_script(
        "preamble-json",
        [FakeTurn(text='Sure! Here is the JSON:\n```json\n{"intent": "order_status", '
                       '"order_id": "ORD-9921"}\n```\nLet me know if you need more.')],
    )

    configs = [
        ModelCfg(id="fake", provider="fake", model="preamble-json"),
    ]
    if os.environ.get("OPENAI_API_KEY"):
        configs.append(ModelCfg(id="openai", provider="openai", model=OPENAI_MODEL, max_tokens=200))
    if ollama_models():
        configs.append(ModelCfg(id="ollama", provider="ollama", model=OLLAMA_MODEL, max_tokens=200))

    for cfg in configs:
        client = build_client(cfg)
        try:
            result = await client.complete(PROMPT, schema=Routing)
        finally:
            await client.aclose()
        report(cfg.id, result)
        assert isinstance(result.parsed, Routing)
        assert result.structured_mode_used is not None

    if len(configs) == 1:
        pytest.skip("only the fake provider was reachable; set a key or start Ollama for more")
