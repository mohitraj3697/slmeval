"""One client for every server that speaks ``/v1/chat/completions``.

OpenAI, Ollama, llama.cpp's server, LM Studio, vLLM, Together, Groq and
OpenRouter all speak this dialect, so one 250-line client covers the entire MVP
provider list. Dropping vLLM as a "native" provider and reaching it through
``openai_compatible`` + ``base_url`` loses nothing and gains six backends - and
it is the only option on Windows, where vLLM has no build.

What this client is careful about:

- **A rejected mechanism is data, not an error.** A 400 naming
  ``response_format`` means this server cannot do that rung. It is raised as
  :class:`UnsupportedFeature` so the ladder drops down, and remembered so the
  run does not pay for the same discovery 900 times.
- **Latency excludes retries.** ``latency_ms`` times the final attempt only.
- **Usage is read, not assumed.** Servers that omit ``usage`` get an estimate
  that is flagged as one all the way to the report.
- **Local concurrency defaults to 1.** Measuring a laptop GPU under concurrency
  and publishing the latency would be the fastest way to lose credibility.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Sequence
from typing import Any

import httpx
from pydantic import BaseModel

from slmeval.core.trace import JsonStage
from slmeval.providers.base import (
    ClientSpec,
    Msg,
    ProviderRequestError,
    ProviderServerError,
    ProviderUnreachable,
    RateLimited,
    Rung,
    ToolCall,
    ToolSpec,
    UnsupportedFeature,
)
from slmeval.providers.jsonx import JsonExtractionError, extract_json
from slmeval.providers.ladder import LadderClient, RungResult, with_schema_instruction
from slmeval.providers.retry import with_retries
from slmeval.providers.schema import response_format_json_schema
from slmeval.providers.tokens import usage_from_response

CHAT_PATH = "/chat/completions"

RUNG_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tools": ("tool", "function"),
    "json_schema": ("json_schema", "response_format", "schema"),
    "json_mode": ("json_object", "response_format"),
}
"""Words that, in a 4xx body, mean "this server cannot do that rung".

Keyword matching rather than status codes alone, because the servers disagree:
llama.cpp returns 500 for an unknown ``response_format``, Ollama returns 400,
and some proxies return 422.
"""

UNSUPPORTED_MARKERS = (
    "not supported",
    "unsupported",
    "unrecognized",
    "unknown field",
    "unexpected keyword",
    "invalid type",
    "does not support",
    "no such field",
)

OLLAMA_V1_IGNORED = frozenset(
    {
        "num_ctx",
        "num_gpu",
        "num_predict",
        "num_thread",
        "repeat_penalty",
        "repeat_last_n",
        "top_k",
        "min_p",
        "tfs_z",
        "mirostat",
        "mirostat_eta",
        "mirostat_tau",
        "keep_alive",
    }
)
"""Ollama options its OpenAI-compatible endpoint silently drops.

Silently is the problem. A user who sets ``num_ctx = 8192`` and gets a 2048
context has measured the wrong thing and has no way to know, so we say so once.
"""

MAX_BODY_CHARS = 600


class OpenAICompatClient(LadderClient):
    """Chat completions over HTTP, with the ladder on top."""

    def __init__(
        self,
        spec: ClientSpec,
        *,
        http: httpx.AsyncClient | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(spec)
        if not spec.base_url:
            raise ValueError(f"{spec.provider!r} needs a base_url")
        self._base_url = spec.base_url.rstrip("/")
        self._owns_http = http is None
        self._http = http or httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=spec.connect_timeout_s,
                read=spec.read_timeout_s,
                write=spec.connect_timeout_s,
                pool=spec.connect_timeout_s,
            ),
        )
        self._rng = rng or random.Random(0)
        self._warned: set[str] = set()
        # Per-client, because the limit belongs to the *server*: one client is
        # shared by every concurrent execution of this candidate, and a laptop
        # serving a 7B model wants max_concurrency = 1 no matter how wide the
        # matrix runs. Acquired around the request only, never held across a
        # backoff sleep.
        self._gate = asyncio.Semaphore(max(1, spec.max_concurrency))

    async def aclose(self) -> None:
        if self._owns_http and not self._http.is_closed:
            await self._http.aclose()

    # -- one rung ---------------------------------------------------------- #

    async def attempt(
        self,
        rung: Rung | None,
        messages: Sequence[Msg],
        *,
        schema: type[BaseModel] | None,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> RungResult:
        wire_messages = list(messages)
        if rung in ("json_mode", "prompt") and schema is not None:
            wire_messages = with_schema_instruction(wire_messages, schema)

        payload = self._payload(
            rung, wire_messages, schema=schema, tools=tools, tool_choice=tool_choice
        )

        attempted = await with_retries(
            lambda: self._post(payload, rung),
            self.spec.retry,
            rng=self._rng,
        )
        raw, latency_ms = attempted.value

        text, calls = self._read_choice(raw)
        return RungResult(
            text=text,
            tool_calls=calls,
            raw=raw,
            usage=usage_from_response(
                raw,
                messages=wire_messages,
                output_text=text,
                output_tool_calls=calls,
            ),
            latency_ms=latency_ms,
            # Not streamed, so there is no first token to time. Streamed TTFT
            # arrives with latency mode in Step 10; reporting 0 here would be a
            # fabricated measurement.
            ttft_ms=None,
            retries=attempted.retries,
            retry_ms=attempted.retry_ms,
        )

    def _payload(
        self,
        rung: Rung | None,
        messages: Sequence[Msg],
        *,
        schema: type[BaseModel] | None,
        tools: Sequence[ToolSpec] | None,
        tool_choice: str | None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.spec.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": self.spec.temperature,
            "max_tokens": self.spec.max_tokens,
            "stream": False,
        }

        if rung == "tools" and schema is not None:
            # The schema becomes a single forced tool. The ladder guarantees the
            # caller passed no tools of their own on this rung, so there is no
            # ambiguity about which call to read.
            schema_tool = ToolSpec.from_model(schema)
            body["tools"] = [schema_tool.to_wire()]
            body["tool_choice"] = {"type": "function", "function": {"name": schema_tool.name}}
        else:
            if tools:
                body["tools"] = [t.to_wire() for t in tools]
                if tool_choice:
                    body["tool_choice"] = tool_choice
            if rung == "json_schema" and schema is not None:
                body["response_format"] = response_format_json_schema(schema)
            elif rung == "json_mode" and schema is not None:
                body["response_format"] = {"type": "json_object"}

        self._warn_ignored_extras()
        body.update(self.spec.extra_body)
        return body

    def _warn_ignored_extras(self) -> None:
        """Say once when Ollama's /v1 endpoint will drop a configured option."""
        if self.spec.provider != "ollama":
            return
        dropped = sorted(set(self.spec.extra_body) & OLLAMA_V1_IGNORED)
        if not dropped or "extra_body" in self._warned:
            return
        self._warned.add("extra_body")
        import warnings

        warnings.warn(
            f"[{self.spec.id}] Ollama's OpenAI-compatible endpoint ignores "
            f"{', '.join(dropped)}. The run will NOT use those values. Set them "
            "in a Modelfile and point `model` at the derived model instead.",
            UserWarning,
            stacklevel=4,
        )

    # -- HTTP -------------------------------------------------------------- #

    async def _post(
        self, payload: dict[str, Any], rung: Rung | None
    ) -> tuple[dict[str, Any], float]:
        headers = {"Content-Type": "application/json"}
        if self.spec.api_key:
            headers["Authorization"] = f"Bearer {self.spec.api_key}"

        # The clock starts after the gate, never before it: time spent queueing
        # behind our own concurrency limit is our scheduling, not the model's
        # latency, and folding it in would make a p95 meaningless.
        try:
            async with self._gate:
                started = time.perf_counter()
                response = await self._http.post(
                    f"{self._base_url}{CHAT_PATH}", json=payload, headers=headers
                )
                latency_ms = (time.perf_counter() - started) * 1000.0
        except (httpx.ConnectError, httpx.ConnectTimeout) as exc:
            raise ProviderUnreachable(
                f"cannot reach {self._base_url} ({exc.__class__.__name__})"
            ) from exc
        except httpx.TimeoutException as exc:
            raise ProviderUnreachable(
                f"{self.spec.model} did not respond within "
                f"{self.spec.read_timeout_s:g}s - a cold local model can take "
                "longer on its first call than `timeout_s` allows"
            ) from exc
        except httpx.HTTPError as exc:
            raise ProviderUnreachable(
                f"transport error talking to {self._base_url}: {exc}"
            ) from exc

        if response.status_code >= 400:
            self._raise_for_status(response, rung)

        try:
            raw = response.json()
        except ValueError as exc:
            raise ProviderRequestError(
                f"{self._base_url} returned non-JSON ({response.status_code}): "
                f"{_clip(response.text)}"
            ) from exc
        if not isinstance(raw, dict):
            raise ProviderRequestError(f"expected a JSON object, got {type(raw).__name__}")
        return raw, latency_ms

    def _raise_for_status(self, response: httpx.Response, rung: Rung | None) -> None:
        status = response.status_code
        body = response.text or ""

        if status == 429:
            raise RateLimited("rate limited", status=status, body=_clip(body))

        if rung is not None and self._looks_unsupported(body, rung):
            raise UnsupportedFeature(rung, f"HTTP {status}: {_clip(body)}")

        if status >= 500:
            raise ProviderServerError(
                f"{self._base_url} returned {status}", status=status, body=_clip(body)
            )
        if status in (401, 403):
            key_hint = (
                f"check ${self.spec.api_key_env}"
                if self.spec.api_key_env
                else "check the server's auth settings"
            )
            raise ProviderRequestError(
                f"{self._base_url} rejected the credentials ({status}) - {key_hint}",
                status=status,
                body=_clip(body),
            )
        if status == 404:
            raise ProviderRequestError(
                f"{self._base_url}{CHAT_PATH} not found ({status}) - is base_url "
                f"missing or carrying an extra path segment? Model: {self.spec.model!r}",
                status=status,
                body=_clip(body),
            )
        raise ProviderRequestError(
            f"{self._base_url} returned {status}: {_clip(body)}", status=status, body=_clip(body)
        )

    @staticmethod
    def _looks_unsupported(body: str, rung: Rung) -> bool:
        low = body.lower()
        keywords = RUNG_KEYWORDS.get(rung, ())
        if not keywords or not any(k in low for k in keywords):
            return False
        return any(marker in low for marker in UNSUPPORTED_MARKERS)

    # -- response parsing -------------------------------------------------- #

    def _read_choice(self, raw: dict[str, Any]) -> tuple[str | None, list[ToolCall]]:
        choices = raw.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ProviderRequestError(f"response had no choices: {_clip(json.dumps(raw))}")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        if not isinstance(message, dict):
            raise ProviderRequestError("response choice had no message object")

        content = message.get("content")
        text = content if isinstance(content, str) else None

        calls: list[ToolCall] = []
        for i, item in enumerate(message.get("tool_calls") or []):
            if not isinstance(item, dict):
                continue
            fn = item.get("function") or {}
            raw_args = fn.get("arguments")
            args, stage = _parse_arguments(raw_args)
            calls.append(
                ToolCall(
                    id=str(item.get("id") or f"call_{i}"),
                    name=str(fn.get("name") or ""),
                    arguments=args,
                    raw_arguments=raw_args if isinstance(raw_args, str) else None,
                    arguments_stage=stage,
                )
            )
        return text, calls


def _parse_arguments(raw_args: Any) -> tuple[dict[str, Any], JsonStage | None]:
    """Tool-call arguments, through the same cascade as any other model JSON.

    Small models emit broken JSON inside ``arguments`` just as readily as in
    message content, and a lenient parse there has to be counted the same way.
    """
    if isinstance(raw_args, dict):
        return raw_args, JsonStage.DIRECT
    if not isinstance(raw_args, str) or not raw_args.strip():
        return {}, None
    try:
        extraction = extract_json(raw_args)
    except JsonExtractionError:
        return {}, None
    if not isinstance(extraction.value, dict):
        return {}, None
    return extraction.value, extraction.stage


def _clip(text: str) -> str:
    text = " ".join(text.split())
    if len(text) <= MAX_BODY_CHARS:
        return text
    return text[:MAX_BODY_CHARS] + "..."


__all__ = ["OLLAMA_V1_IGNORED", "OpenAICompatClient"]
