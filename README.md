# slmeval

**Prove a 4B model can do the boring 80% of your agent before you ship it.**

Your agent calls a frontier model at every step. Some of those steps are intent
routing, field extraction and formatting — work a 4B model does for 2% of the
price. The reason nobody downsizes is that there is no way to prove the small
model won't break step 7 of a 10-step tool-calling loop.

`slmeval` records your agent's tool calls once, replays them deterministically
against every candidate model, and prints what each swap actually costs you.

```text
 refund-agent-slm-downsizing · 10 cases × 3 runs · replay · faults 10% · seed 42

 Candidate            pass^3   95% CI        Schema  Tool F1  Steps  p95 ms   Cost/case   Δ Cost
 ───────────────────────────────────────────────────────────────────────────────────────────────
 baseline gpt-4o      96.7%    [83%, 99%]    100%     0.98     4.1    1 240   $0.0182       —
 gemma3:4b-q4         83.3%    [66%, 93%]     88%     0.91     5.3      310   $0.0039    −78%
 qwen2.5-7b           93.3%    [78%, 98%]     97%     0.96     4.3      520   $0.0061    −66%

 ⚠ gemma3:4b-q4 needed json_mode on 100% of calls, prompt-repair on 12%
 ✗ 1 forbidden tool call: issue_refund in refund_207 (gemma3:4b-q4)
```

Because tool calls are replayed from a local cache, **a benchmark run in CI
needs no GPU and no API keys.**

---

## Status

Alpha, and honest about it. What works today:

| Command | |
|---|---|
| `slmeval init` | scaffold `eval.toml`, a starter dataset and a target stub |
| `slmeval doctor` | check Python, storage, credentials and provider reachability |
| `slmeval validate` | validate the config, the dataset, and the links between them |
| `slmeval dataset validate` | check a golden dataset on its own |

And as a library, the halves of the harness that are built:

| API | |
|---|---|
| `slmeval.node(...)` | names the agent step a call belongs to; nests correctly |
| `slmeval.chat_model(...)` | routes each call to the model this candidate assigns to that node |
| the provider gateway | OpenAI, Ollama and anything speaking `/v1/chat/completions`, plus `fake` |
| the structured-output ladder | reports which mechanism a candidate actually needed |

`record`, `run`, `report`, `compare`, `show`, `gate`, `cassette` and `cache` are
scaffolded and listed in `--help`, but not implemented yet. They say which step
delivers them when you call them. The tool sandbox (`@sandbox_tool`) and the
cassette store are next.

Not built yet, and deliberately: the `anthropic` provider (Claude does not speak
`/v1/chat/completions`; reach it through an OpenAI-compatible gateway until
then), and streamed time-to-first-token.

## Install

```bash
pip install slmeval
```

Core install pulls four packages and no model framework. Framework support is an
extra: `pip install "slmeval[langgraph]"`.

## Three commands

```bash
slmeval init        # writes eval.toml, datasets/golden_cases.jsonl, slmeval_target.py
slmeval doctor      # can this machine run it?
slmeval validate    # is the config sane?
```

`init` produces a config that validates with zero setup — no API key, no local
model — so you can see the shape of the thing before committing to it.

## How it works

1. **You point it at your agent.** `[target].entrypoint` in `eval.toml` names a
   coroutine that takes a case's input and returns the final state. Any
   framework, or none.
2. **You mark the seams.** `slmeval.chat_model()` where you build an LLM client,
   `@sandbox_tool` on your tool functions. Both are no-ops outside a benchmark
   run, so they are safe to leave in production code.
3. **It records once, replays forever.** The baseline run executes your real
   tools and caches every response. Candidate runs serve those responses from
   SQLite in microseconds — no network, no rate limits, no drift.
4. **It grades in tiers.** Deterministic checks (valid JSON? real tool name?
   required args?) cost nothing and catch most small-model failures. Trajectory
   and state checks catch redundant loops and wrong paths. An LLM judge runs
   last, only on cases that already passed everything cheaper.
5. **It reports consistency, not a single run.** Each case runs *k* times, and
   results are reported as `pass^k` with a Wilson confidence interval — because
   three passing runs is not 100%, and a gate that pretends otherwise will wave
   through a model that fails one request in four.

## Configuration

`eval.toml` is the whole interface, and the generated file documents every key.
The part that matters:

```toml
[[candidates]]
id = "gemma3-4b-local"
provider = "ollama"
model = "gemma3:4b"
target_nodes = ["intent_router"]   # swap this node only
```

`target_nodes` is the point of the tool. Swapping the model everywhere almost
always fails. Naming the two nodes that don't need frontier reasoning is where
the large cost cut with no accuracy loss lives.

Providers: `openai`, `ollama`, `openai_compatible` (llama.cpp server, LM Studio,
vLLM, Together, Groq, OpenRouter), and `fake` for tests. `anthropic` is accepted
by the config and not yet wired up.

### Structured output is measured, not assumed

The reason a small model breaks an agent is almost never that it is dumb — it is
that its structured-output path on that provider silently degrades. So every
call with a schema walks a ladder and slmeval records the rung that *worked*:

| Rung | Mechanism |
|---|---|
| `tools` | native function calling, schema as a forced tool |
| `json_schema` | strict `response_format` |
| `json_mode` | `{"type": "json_object"}` plus the schema in-prompt |
| `prompt` | schema in-prompt only |
| `repair` | one re-ask carrying the validation error |

Set `structured_output = "auto"` to walk it, or pin a rung so a published
comparison is honest. Either way the scorecard can say *gpt-4o used native tools
throughout; gemma3:4b needed json_mode on 100% of calls and prompt-repair on
12%*.

JSON that only parses because our parser was forgiving — a fenced block, a prose
preamble, a trailing comma, a Python dict — is counted as a leniency event and
reported, not quietly passed. A model that passes because of our regexes is a
finding, not a pass.

Token counts come from the provider's `usage` block when it sends one. When it
does not, they are estimated and marked `~` everywhere they appear. Retries are
excluded from latency and counted separately, so a candidate needing three
retries per call cannot hide behind a healthy p95.

## Privacy

Local-first, and not in a marketing sense:

- Traces, tool cassettes and reports stay in `./.slmeval/`.
- The only network traffic is to the model endpoints you configured.
- No telemetry. Not opt-out — absent.
- `eval.toml` holds the *name* of an environment variable (`api_key_env`), never
  a key, so it is safe to commit and safe to paste into an issue.

## Development

```bash
uv sync --extra dev
uv run pytest             # offline: no GPU, no keys, no network, no spend
uv run pytest -m live     # opt-in: hits whichever providers are reachable
uv run slmeval --help
```

The default suite runs fully offline against a scripted fake provider, in about
two seconds. `-m live` is excluded from it on purpose: a suite that costs money
or needs a GPU is a suite people stop running.

## License

Apache-2.0
