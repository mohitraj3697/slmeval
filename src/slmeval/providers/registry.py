"""Turning a `[baseline]` or `[[candidates]]` table into a live client.

Two things happen here that are easy to get wrong and expensive to get wrong
late:

- **Secrets are resolved from the environment, once.** `eval.toml` holds the
  variable *name*; the value is read here and lives only on the
  :class:`ClientSpec`, whose ``__repr__`` refuses to print it.
- **Operational defaults are per provider.** A 7B model on a laptop GPU gets
  ``max_concurrency = 1`` and a long read timeout; a cloud endpoint gets eight
  in flight and a short one. Using one number for both either throttles the
  cloud run to nothing or thrashes the local GPU - and a local latency figure
  measured under concurrency is worse than no figure at all.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from slmeval.config import ModelCfg
from slmeval.errors import ConfigError, EnvironmentError_
from slmeval.providers.base import ChatClient, ClientSpec, RetryPolicy

DEFAULT_BASE_URL: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
}

LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"})

REQUIRES_KEY = frozenset({"openai"})


@dataclass(frozen=True, slots=True)
class Operational:
    """Per-provider connection behaviour."""

    max_concurrency: int
    connect_timeout_s: float
    read_timeout_s: float
    retry: RetryPolicy


CLOUD = Operational(
    max_concurrency=8,
    connect_timeout_s=10.0,
    read_timeout_s=120.0,
    retry=RetryPolicy(attempts=3),
)

LOCAL = Operational(
    # One. A laptop serving a 7B model has no spare capacity, and a p95 taken
    # from four concurrent requests to one GPU describes our scheduler, not the
    # model. Step 10's throughput mode raises this deliberately and says so.
    max_concurrency=1,
    connect_timeout_s=5.0,
    # Generous: the first call to an unloaded Ollama model pays for weights
    # coming off disk, which on a cold 7B can be 60s before a single token.
    read_timeout_s=300.0,
    retry=RetryPolicy(attempts=2, base_delay_s=1.0),
)

FAKE = Operational(
    max_concurrency=16,
    connect_timeout_s=1.0,
    read_timeout_s=1.0,
    retry=RetryPolicy(attempts=2, base_delay_s=0.0, max_delay_s=0.0, jitter=0.0),
)


def is_local(provider: str, base_url: str | None) -> bool:
    """Whether this endpoint is a model running on the user's own machine."""
    if provider == "ollama":
        return True
    if provider != "openai_compatible" or not base_url:
        return False
    host = (urlparse(base_url).hostname or "").lower()
    return host in LOCAL_HOSTS or host.endswith(".local")


def operational_for(provider: str, base_url: str | None) -> Operational:
    if provider == "fake":
        return FAKE
    return LOCAL if is_local(provider, base_url) else CLOUD


def resolve_spec(cfg: ModelCfg, *, timeout_s: float | None = None) -> ClientSpec:
    """Config plus environment plus provider defaults, as one flat spec."""
    base_url = cfg.base_url or DEFAULT_BASE_URL.get(cfg.provider)
    ops = operational_for(cfg.provider, base_url)
    api_key = _resolve_key(cfg)

    return ClientSpec(
        id=cfg.id,
        provider=cfg.provider,
        model=cfg.model,
        base_url=base_url,
        api_key=api_key,
        api_key_env=cfg.api_key_env,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        structured_output=cfg.structured_output,
        extra_body=dict(cfg.extra_body),
        connect_timeout_s=ops.connect_timeout_s,
        # `benchmark.timeout_s` is the per-case budget, so it is a ceiling on a
        # single call but never a reason to wait longer than the provider's own
        # default would.
        read_timeout_s=min(ops.read_timeout_s, timeout_s) if timeout_s else ops.read_timeout_s,
        max_concurrency=ops.max_concurrency,
        retry=ops.retry,
    )


def build_client(
    cfg: ModelCfg,
    *,
    timeout_s: float | None = None,
    http: httpx.AsyncClient | None = None,
) -> ChatClient:
    """The client for one model configuration."""
    spec = resolve_spec(cfg, timeout_s=timeout_s)

    if spec.provider == "fake":
        from slmeval.providers.fake import FakeClient

        return FakeClient(spec)

    if spec.provider == "anthropic":
        raise ConfigError(
            f"candidate {cfg.id!r} uses provider 'anthropic', which is not wired up yet",
            hint=(
                "Anthropic needs its own client (it does not speak "
                "/v1/chat/completions) and lands in week 3. Until then, reach "
                "Claude through an OpenAI-compatible gateway with "
                "provider = 'openai_compatible' and a base_url."
            ),
        )

    if spec.provider in ("openai", "openai_compatible", "ollama"):
        from slmeval.providers.openai_compat import OpenAICompatClient

        return OpenAICompatClient(spec, http=http)

    raise ConfigError(f"unknown provider {spec.provider!r} for candidate {cfg.id!r}")


def _resolve_key(cfg: ModelCfg) -> str | None:
    """Read the key from the environment, or fail with the variable's name.

    Exits 3, not 1: an unset key is a broken environment, and a CI job that
    reports it as a model regression teaches people to ignore the gate.
    """
    if not cfg.api_key_env:
        return None
    value = os.environ.get(cfg.api_key_env)
    if value:
        return value
    if cfg.provider in REQUIRES_KEY:
        raise EnvironmentError_(
            f"${cfg.api_key_env} is not set, and candidate {cfg.id!r} "
            f"({cfg.provider}) needs it",
            hint=(
                "set it in this shell, or switch the candidate to a local "
                "provider. In replay mode a benchmark needs no API keys at "
                "all - only recording does."
            ),
        )
    return None


__all__ = [
    "CLOUD",
    "DEFAULT_BASE_URL",
    "FAKE",
    "LOCAL",
    "Operational",
    "build_client",
    "is_local",
    "operational_for",
    "resolve_spec",
]
