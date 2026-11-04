# Configuration

`eval.toml` holds everything. `slmeval init` writes one with each key explained, and
`slmeval validate` checks it along with the dataset.

```toml
[benchmark]
name = "support-agent"
dataset = "./datasets/golden_cases.jsonl"
runs_per_case = 3
tool_mode = "replay"
seed = 42

[target]
entrypoint = "my_agent:run_case"

[baseline]
provider = "openai"
model = "gpt-4o-mini"

[[candidates]]
id = "gemma3-router"
provider = "ollama"
model = "gemma3:4b"
target_nodes = ["intent_router"]
```

`target_nodes` limits a candidate to the named steps, and the baseline serves the rest.
Leave it out to use the candidate on every step.

## Providers

| Provider | Use |
|---|---|
| `openai` | The OpenAI API |
| `ollama` | A local Ollama server |
| `openai_compatible` | Anything serving `/v1/chat/completions`: llama.cpp, LM Studio, vLLM, Together, Groq, OpenRouter |
| `fake` | A scripted model for offline runs |

`anthropic` is accepted by the config but isn't connected yet. Until it is, reach Claude
through an OpenAI-compatible gateway.

API keys stay in environment variables. The config holds only the variable's name, in
`api_key_env`, so `eval.toml` is safe to commit.

## Tool modes

| Mode | Behaviour |
|---|---|
| `replay` | Serve responses from cassettes and never call the tool |
| `record` | Call the real tool and store the response |
| `auto` | Replay when there is a recording, otherwise record |
| `live` | Always call the real tool |

## Structured output

Every call with a schema tries these mechanisms in order. slmeval records which one
worked, so you can see what a model needed:

| Rung | Mechanism |
|---|---|
| `tools` | Native function calling, with the schema as a forced tool |
| `json_schema` | Strict `response_format` |
| `json_mode` | `{"type": "json_object"}` with the schema in the prompt |
| `prompt` | The schema in the prompt only |
| `repair` | One retry that includes the validation error |

Set `structured_output = "auto"` to walk the list, or name one rung to hold every model
to it.

JSON that only parses because the parser was lenient, such as a fenced block, a prose
preamble or a trailing comma, is counted and reported.

Token counts come from the provider's `usage` block. If it doesn't send one, they are
estimated and marked `~`. Retries are left out of latency and counted separately.

## Latency

Local servers queue requests, so latency measured under concurrency mostly reflects the
queue. A normal run therefore reports no latency at all. To measure it:

```bash
slmeval run --serial-latency
```

This sends one request at a time per endpoint after a warm-up. It is much slower.

## The fake provider

`provider = "fake"` is a scripted model. It never touches the network, costs nothing and
behaves the same on every machine. Use it for a free, reproducible baseline, for testing
your wiring, or for testing how your agent handles a specific failure.

Each entry in a script answers one model call, in order, and the last entry repeats:

```toml
[[candidates]]
id = "weak-4b"
provider = "fake"
model = "weak-4b"
extra_body.unsupported = ["tools", "json_schema"]
extra_body.script = [
    { kind = "fenced_json", text = '{"intent": "order_status"}' },
    { kind = "rate_limit" },
    { text = '{"final_answer": "shipped"}', usage = [120, 18], latency_ms = 240 },
]
```

`unsupported` lists the rungs this fake server rejects. `kind` names a behaviour, and
`text` is the payload it wraps:

| `kind` | Behaviour |
|---|---|
| `valid_json` | Clean, schema-valid JSON |
| `fenced_json` | A prose preamble and a fenced JSON block |
| `prose_json` | The object buried in a sentence |
| `trailing_comma` | JSON that needs repair |
| `invalid_schema` | Parses, but the fields are wrong |
| `not_json` / `empty` | Prose only, or nothing |
| `refusal` | Answers in words on the forced-tool rung |
| `hallucinated_tool` / `hallucinated_args` | Calls a tool that doesn't exist, or with arguments it doesn't take |
| `timeout` / `unreachable` | The endpoint never answers |
| `rate_limit` / `server_error` / `bad_request` | HTTP 429, 5xx or 400 |
| `loop` | The same answer until `max_steps` stops the agent |

One model usually has to answer a whole dataset, so scripts can be set per case:

```toml
[baseline.extra_body.script_by_case]
refund_207 = [{ text = '{"intent": "refund", "order_id": "ORD-7310"}' }]
```

Longer scripts can live in their own TOML or JSON file, with the same three keys:

```toml
extra_body.script_file = "./fakes/weak-4b.toml"
```

The file's path resolves against `eval.toml`, and its contents go into the config hash,
so editing it counts as a different benchmark.

## Fault injection

```bash
slmeval run --fault-rate 0.3
```

Or set `fault_injection_rate` in `[benchmark]`. Choose the faults with a table:

```toml
[benchmark.faults]
kinds = ["empty_result", "semantic_conflict", "rate_limit_429"]
delivery = { rate_limit_429 = "return" }
weights = { empty_result = 3.0 }
```

| Fault | What it tests |
|---|---|
| `rate_limit_429` | Backoff and retry |
| `timeout_504` | Timeout handling. Waits two seconds for real. |
| `malformed_json` | Handling of garbage tool output |
| `empty_result` | An empty response where data was expected. Does the model invent data? |
| `partial_schema` | A payload missing fields the agent expects |
| `semantic_conflict` | Data that contradicts earlier state |
| `latency_spike` | A correct result, five seconds late |

`delivery` sets whether a fault is raised or returned. A raised 429 tests your retry
code. The same 429 returned as a payload tests whether the model reads its tool output.

Faults are scheduled per case and seed, and indexed by each tool's own call count. One
extra call to one tool therefore can't shift later faults onto different steps. Two
models that take different paths can still meet different faults, because a model that
stops early never reaches a later one. See
[Reading results](reading-results.md#fault-injection-results).
