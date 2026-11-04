# slmeval

**Find out which steps of your agent a small model can handle, before you ship it.**

Your agent calls a frontier model at every step. Some of those steps may be
simple enough for a much cheaper small model: intent routing, field extraction,
formatting. The hard part is knowing the small model won't break step 7 of a
10-step tool-calling loop.

`slmeval` records your agent's tool calls once, replays them deterministically
against every candidate model, and prints what each swap actually costs you.

```text
  teardown-01-support-agent · 4 cases x 3 runs · replay · no faults · seed 42 · throughput

  ✗ FORBIDDEN tool call: issue_refund in refund_needs_confirmation (refund-happy), 3x
  ! fenced-json needed prompt-repair or a fallback rung on 100% of structured calls

Candidate            pass^3         95% CI     runs    Schema    Tool F1    Steps      $/case    Δ Cost
baseline baseline    100.0%    [51%, 100%]    12/12      100%       1.00      4.8    ~$0.0000         —
sloppy-router         33.3%      [6%, 79%]      3/9      100%       0.40      1.8    ~$0.0000         —
fenced-json          100.0%    [51%, 100%]    12/12        0%       1.00      4.8    ~$0.0000         —
refund-happy          75.0%     [30%, 95%]     9/12      100%       0.96      4.8    ~$0.0000         —
```

That is real output, shortened, from [Teardown #1](teardown). **The four models
in it are scripted stand-ins, not real SLMs**, each written to fail in one known
way, so it shows what the tool reports and says nothing about any real model.
They cost nothing, which is why the cost columns are empty.

Because tool calls are replayed from a local cache, **a benchmark run in CI
needs no GPU and no API keys.**

---

## Five minutes, no API key

```bash
pip install slmeval
cd examples/support_agent    # or: slmeval init, in your own project
slmeval record               # execute the real tools once, cache them
slmeval run                  # replay them against every candidate
```

That is the whole cold start, and nothing in it asks for a credential. The
bundled [example](examples/support_agent) runs a four-node agent against a
baseline and three candidates — one that passes, one that hallucinates an order
id, and one that issues a refund nobody authorised — on CPU, offline, in under
a second. Then point `[target].entrypoint` at your own agent and swap the
candidates for models you actually have.

---

## Status

Alpha, and honest about it. What works today:

| Command | |
|---|---|
| `slmeval init` | scaffold `eval.toml`, a starter dataset and a target stub |
| `slmeval doctor` | check Python, storage, credentials and provider reachability |
| `slmeval validate` | validate the config, the dataset, and the links between them. `--json` gives the hashes a CI cache keys on |
| `slmeval record` | execute the real tools once and cache them for replay |
| `slmeval run` | run the candidate matrix and print the scorecard |
| `slmeval report` | re-render the scorecard from a stored run, as a table, JSON, Markdown, CSV or JUnit XML |
| `slmeval show run\|exec\|case\|uncovered` | drill from a red cell to the exact step that diverged |
| `slmeval gate` | fail the build when a candidate regresses, with exit codes CI can branch on |
| `slmeval baseline save\|show` | write the committed reference a gate compares against |
| `slmeval judge calibrate` | measure an LLM judge's agreement with your own labels, before trusting it |
| `slmeval dataset validate` | check a golden dataset on its own |
| `slmeval cache stats` | what the trace store holds, and what it costs on disk |
| `slmeval cache prune` | drop runs older than a cutoff and free the payloads nothing references |

Plus a [GitHub Action](examples/github-action) that runs the whole thing on a
stock runner and posts the verdict to the pull request.

The bundled [example](examples/support_agent) runs offline, with no API key and
no GPU - four nodes, five sandboxed tools, three candidates and three different
ways to fail:

```bash
cd examples/support_agent
slmeval record      # execute the real tools once, cache them
slmeval run         # replay them against every candidate
```

[Teardown #1](teardown) is that example put through the whole tool and
published with every artifact it produced: the scorecard, the gate's exit 1,
the cassette-miss diagnosis, and the fault-injection run that found a bug in
our own example agent. Its models are scripted stand-ins rather than real SLMs,
which it says in its first paragraph and again in its limitations.

And as a library, the halves of the harness that are built:

| API | |
|---|---|
| `slmeval.node(...)` | names the agent step a call belongs to; nests correctly |
| `slmeval.chat_model(...)` | routes each call to the model this candidate assigns to that node |
| the provider gateway | OpenAI, Ollama and anything speaking `/v1/chat/completions`, plus `fake` |
| the structured-output ladder | reports which mechanism a candidate actually needed |
| `slmeval.instrument_langgraph(...)` | wraps every node of a LangGraph builder, so you mark nothing by hand |
| `slmeval.sandbox_tool(...)` | records a tool call once, then replays it — `record` / `replay` / `auto` / `live` |
| the cassette key policy | per-tool `match_on` / `normalize` / `ignore_args`, matched exact → normalized → `match_on` subset, reporting which tier hit |
| the typed envelope | a tool's `datetime`, `Decimal`, tuple, set, enum, Pydantic model or dataclass replays as the same *type* it recorded as — and a value that cannot be recorded faithfully is refused loudly instead of coerced |
| the cassette store | SQLite + WAL with an in-process LRU; 200 replays of a 4-tool-call agent take 48 ms with sockets blocked |
| the `generic` adapter | loads `module:callable`, runs it async or in a thread, keeps the run context intact across both |
| the `langgraph` adapter | one call wraps every node: latency, state key sets and exception type per node, and an identical trajectory to marking them by hand |
| the Tier 3 judge | schema'd verdict, blinded randomised A/B, content-addressed cache, and a calibration harness it refuses to run without — not yet wired into `run` |
| the trace store | one SQLite file with forward-only migrations; a completed run is reconstructible from the rows alone, with provenance (config hash, dataset hash, git SHA, host) recorded so `compare` can refuse a comparison whose premises differ |
| the payload store | large tool outputs and raw model responses addressed by SHA-256, so one recorded payload replayed by every candidate is stored once |

A worked example lives in [`examples/support_agent/`](examples/support_agent):
four nodes, five sandboxed tools, no framework, and tools that return
`datetime`, `Decimal` and `StrEnum` values so the round trip is actually tested.

`compare` and `cassette` are not in this release, so they are left out of
`--help` and exit 2 with a message if you call them. Everything they need is
already stored: a run's rows carry the provenance `compare` has to check and the
cassette keys `cassette verify` has to walk.

Two numbers in that table are worth reading twice. `pass^3` is the fraction of
cases that passed **every** repeat, not an average — an agent you cannot retry
has to pass every time. And the interval beside it is the Wilson 95% bound on
that fraction, which is what to gate on: three passes out of three is a point
estimate of 100% whose lower bound is about 44%, so a gate set on the point
estimate will cheerfully approve a model that fails one run in four.

### What a cassette miss tells you

A replay miss is not an exception string. The store is searched for the nearest
recorded call, the arguments are diffed, and the result is classified:

```text
✗ status_shipped · sloppy-router · node=order_lookup · CASSETTE_MISS
  tool: fetch_order
  called with: {"order_id": "9921", "request_id": "req-9921"}
  nearest recorded: {"order_id": "ORD-9921", "request_id": "req-ORD-9921"}
  diff: order_id: "ORD-9921" → "9921"   (dropped prefix)
        request_id: "req-ORD-9921" → "req-9921"   (different value)
  verdict: hallucinated_parameter
```

Four verdicts, and the fourth is the one that matters most:
`hallucinated_parameter`, `hallucinated_tool`, `wrong_tool_for_step`, and
`unrecorded_valid_path` — the model did something plausible that the baseline
never recorded. That last one is a gap in *our* coverage, not a model failure.
It is reported separately, kept out of the candidate's pass rate, and fixed by
re-recording. A harness that counted it as a failure would inflate its own
numbers, and the first person to check one by hand would stop believing the
rest of the table.

### From a red cell to the exact step

The scorecard says a candidate lost four points. `slmeval show` says where.

```bash
slmeval show run                      # every execution, worst first
slmeval show exec sloppy-router.status_shipped.r0   # one execution's whole timeline
slmeval show case status_shipped --candidate sloppy-router   # side by side
slmeval show uncovered                # what `record --fill-gaps` should record
```

`show case` is the drill-down: the baseline's trajectory on the left, the
candidate's on the right, aligned rather than zipped — so a candidate that
visits one node twice reads as one insertion instead of a shifted tail — with
the cassette-miss diagnosis block inline at the step that missed.

```text
  status_shipped  baseline → sloppy-router
  passed → cassette_miss · 6 → 2 steps

trajectory
#    baseline              sloppy-router
1    intent_router         intent_router
2    order_lookup          order_lookup
3    policy_check     -    —
4    compose_reply    -    —

tool calls
#    baseline                                           sloppy-router
1    fetch_order({"order_id":"ORD-9921","reques    ~    fetch_order({"order_id":"9921","request_id"
     t_id":"req-ORD-9921"})                             :"req-9921"})
2    fetch_customer({"customer_id":"CUS-41"})      -    —
3    fetch_tracking({"order_id":"ORD-9921"})       -    —
```

Every `show` view takes `--json`, and none of them loads your agent: results
from last Tuesday stay readable after you refactor the code that produced them.

### Gating a pull request

```bash
slmeval run --tool-mode replay          # no GPU, no API keys
slmeval gate --min-pass-k 0.90 --max-cost-ratio 0.40 --max-regression 0.05
```

Replay mode is what makes this adoptable: CI needs no GPU and no keys, because
every tool response comes out of SQLite. Put the thresholds in a `[gate]`
section of `eval.toml` and CI can call `slmeval gate` bare.

**The exit codes are a contract**, frozen from 0.1.0, because people script
against them:

| | |
|---|---|
| `0` | every threshold met |
| `1` | a threshold was not met. A real quality finding. |
| `2` | bad invocation — no candidate, no threshold, unreadable baseline file |
| `3` | the model server was unreachable. **Infrastructure, not a regression.** |
| `4` | not enough recorded ground truth to judge the run |

Distinguishing 1 from 3 is the whole point. A job that reports "Ollama was not
running" as "the model got worse" teaches a team to ignore the gate, and after
that the gate is worth nothing. The same goes for 4: an uncovered path is a
hole in *our* recordings.

Three more decisions worth knowing before you trust a green:

- **It gates on the Wilson lower bound, not the point estimate.** Three passes
  out of three is a lower bound near 44%. `--use-point-estimate` exists and is
  a deliberate act.
- **A check that could not run does not pass.** A latency budget on a
  throughput run, a cost ratio against an unpriced baseline — these print as
  "not measured" and never contribute a quiet green. They do not fail the build
  either, because that would read as a regression.
- **A gate with nothing to check exits 2.** No candidate in the run, or no
  threshold configured, is a usage error. Green-while-measuring-nothing is the
  one outcome worse than red.

A forbidden tool call is a hard fail at any pass rate, and it is the one
finding that still sets exit 1 even when the run was otherwise unjudgeable —
an observed safety violation is not a rate over executions that may be missing.

The baseline for `--max-regression` is a **committed file**, not the last run in
the local database:

```bash
slmeval baseline save     # writes slmeval.baseline.json — commit it
```

Gating against whatever ran here most recently gives different verdicts on
different machines and cannot be reviewed. A committed file turns "we accepted
a two-point drop" into a diff somebody approved. It carries the config hash and
the dataset hash, and `gate` refuses to compare across a change in either
unless you pass `--force` — publishing a dataset edit as a regression is the
fastest way for a measurement tool to stop being believed.

`--format md` gives the pull-request comment, `--format junit` feeds an
existing CI dashboard (`slmeval report --format junit` gives it one test case
per candidate per case), and `--format json` is for everything else.

### The GitHub Action

```yaml
- uses: actions/checkout@v4
- uses: mohitraj3697/slmeval/examples/github-action@main
  with:
    working-directory: evals
    cassettes: evals/cassettes.sqlite
```

That is the whole integration, and it runs on a stock `ubuntu-latest` runner:
no GPU, no API keys, no self-hosted anything. It restores the cassette store
from `actions/cache` keyed on the dataset's content hash, replays the matrix,
gates it, writes the scorecard to the job summary, and edits **one** sticky
pull-request comment rather than appending a table per push.

Each outcome gets its own annotation, because a gate that reports "Ollama was
down" as "the model got worse" is a gate a team learns to ignore:

| Exit | `verdict` | Annotation |
|---|---|---|
| 1 | `failed` | **regression** — a threshold was not met |
| 3 | `environment` | **infrastructure, not a regression** |
| 4 | `coverage` | **coverage, not a regression** — re-record |

3 and 4 still fail the job: a measurement that did not happen must not read as
a pass. They are simply never called a regression.

See [`examples/github-action/`](examples/github-action) for the inputs, the
outputs and the two routes for getting cassettes onto a runner. This repository
runs the Action against its own bundled example on every pull request —
[`gate-demo.yml`](.github/workflows/gate-demo.yml), with a job that asserts a
healthy run exits 0 and a job that asserts an unmeetable floor exits 1.

### Fault injection, and its honest limit

Faults are scheduled deterministically per `(case_id, seed)` and indexed **per
tool, by that tool's own Nth call** — not by a global call counter. That
distinction is the whole correctness argument: with one counter, a candidate
that makes one extra call to any tool shifts every later fault onto a different
step, and two candidates stop facing the same experiment while the table still
prints them side by side.

The limit we cannot engineer away: under divergent trajectories perfect fault
parity is impossible. A candidate that stops after two calls never meets the
fault scheduled for the fourth. So every run records the ordered **fault
vector** it actually met, the comparison table flags any candidate whose vector
differs from the baseline's, and `--require-fault-parity` discards those pairs
rather than pretending they are comparable.

Not built yet, and deliberately: the `anthropic` provider (Claude does not speak
`/v1/chat/completions`; reach it through an OpenAI-compatible gateway until
then), and streamed time-to-first-token.

### With LangGraph

```python
import slmeval
from langgraph.graph import StateGraph

builder = slmeval.instrument_langgraph(StateGraph(State))
builder.add_node("intent_router", route)
builder.add_node("order_lookup", lookup)
graph = builder.compile()
```

One line, and every node names itself in the scorecard — no `with node(...)`
blocks. It records each node's latency, the state keys that went in and came
out, and the type of any exception it raised; the model routing and tool
interception are unchanged, because both come from the ambient context either
way. A graph instrumented this way and the same agent marked by hand produce
**identical trajectories**, which is asserted against the bundled example in
[`tests/test_langgraph_adapter.py`](tests/test_langgraph_adapter.py) — if they
ever diverged, every Tier 2 metric would be measuring the adapter.

Needs the extra: `pip install "slmeval[langgraph]"`. Nothing in the core
imports a model framework.

### The LLM judge, and why it is off

Some things a rubric can express and an assertion cannot ("does the reply state
the return window?"). Tier 3 is for those, and it is **off until you name a
judge model** — tiers 1 and 2 cost nothing, re-run for free, and carry most of
the signal.

The part worth reading before you turn it on:

```bash
slmeval judge calibrate labels.jsonl    # writes slmeval.judge.json — commit it
```

**A judge whose agreement with human labels you have not measured is a random
number generator with a confident tone**, so the judge refuses to run without a
calibration record and every figure derived from it carries the agreement
number. Calibration reports the raw agreement *with its Wilson interval* and
*Cohen's kappa* beside it — on a label set where nine pairs in ten pass, a
judge that says "pass" to everything scores 90% and knows nothing, and kappa is
what deflates that. It also lists every pair where the judge and the human
disagreed, which is the actual deliverable.

Four more rules it is held to: it never grades a run that already failed
deterministically; it never learns which model wrote which answer (randomised
A/B in a reproducible order, with the formatting cues flattened); it never gets
paid twice for the same question (judgements are cached by
`hash(rubric, reference, candidate, judge_model)`, so re-rendering a report
costs $0); and it answers in a schema, through the same structured-output
ladder as anything else.

The skeleton, the cache, the blinding and the calibration harness are built and
tested. Tier 3 is **not yet wired into `slmeval run`** — that waits on a
calibration figure from real labels, which is the only thing that says what a
judge is worth. See [`examples/judge/`](examples/judge).

## Install

```bash
pip install slmeval
```

Core install pulls four packages and no model framework. Framework support is an
extra: `pip install "slmeval[langgraph]"` — which is needed only for the
[LangGraph adapter](#with-langgraph), not for benchmarking a LangGraph agent
you have marked by hand.

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
vLLM, Together, Groq, OpenRouter), and [`fake`](#the-fake-provider), a scripted
model for offline runs. `anthropic` is accepted by the config and not yet wired
up.

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
comparison is honest. Either way the scorecard says which rung each candidate needed:
in [Teardown #1](teardown), *fenced-json needed json_mode on 57% of structured
calls, and prompt-repair or a fallback rung on 100%*.

JSON that only parses because our parser was forgiving — a fenced block, a prose
preamble, a trailing comma, a Python dict — is counted as a leniency event and
reported, not quietly passed. A model that passes because of our regexes is a
finding, not a pass.

Token counts come from the provider's `usage` block when it sends one. When it
does not, they are estimated and marked `~` everywhere they appear. Retries are
excluded from latency and counted separately, so a candidate needing three
retries per call cannot hide behind a healthy p95.

### The `fake` provider

`provider = "fake"` is a scripted model: it never touches the network, costs
nothing, and does exactly the same thing on every machine. It is a shipped
feature, not a test fixture, and it is good for three things:

- **A reference run that is free and reproducible.** Point `[baseline]` at it
  and every candidate is measured against a fixed script rather than against
  whatever GPT-4o did that afternoon.
- **A smoke test of your own wiring.** `slmeval run` exercises the config, the
  dataset, the sandbox, the evaluators and the report without spending a cent.
- **Benchmarking the failure, not the model.** What breaks agents on small
  models is a prose preamble around the JSON, a dropped `ORD-` prefix, a 429 on
  step four. Those are scriptable; a real 4B model will do something slightly
  different tomorrow.

Each turn is one response, consumed in order, and the last one repeats - so a
one-turn script is a model that always answers that way:

```toml
[[candidates]]
id = "weak-4b"
provider = "fake"
model = "weak-4b"
extra_body.unsupported = ["tools", "json_schema"]   # rungs this server rejects
extra_body.script = [
    { kind = "fenced_json", text = '{"intent": "order_status"}' },
    { kind = "rate_limit" },
    { text = '{"final_answer": "shipped"}', usage = [120, 18], latency_ms = 240 },
]
```

`kind` names a misbehaviour and `text` is the payload it wraps:

| `kind` | What the model does |
|---|---|
| `valid_json` | clean, parseable, schema-valid |
| `fenced_json` | a prose preamble and a ```` ```json ```` block |
| `prose_json` | the object buried in a sentence |
| `trailing_comma` | JSON our repair stage has to fix |
| `invalid_schema` | parses, wrong fields |
| `not_json` / `empty` | prose only, or nothing |
| `refusal` | answers in words on the forced-tool rung |
| `hallucinated_tool` / `hallucinated_args` | calls a tool that does not exist, or with arguments it does not take |
| `timeout` / `unreachable` | the endpoint never answers |
| `rate_limit` / `server_error` / `bad_request` | 429, 5xx, 400 |
| `loop` | the same answer forever, until `max_steps` stops the agent |

One model has to answer a whole dataset, so scripts can be per case - a shared
script would have the scripted baseline fail most of its own cases:

```toml
[baseline.extra_body.script_by_case]
refund_207 = [{ text = '{"intent": "refund", "order_id": "ORD-7310"}' }]
```

Longer scripts belong in their own file, which keeps `eval.toml` readable:

```toml
extra_body.script_file = "./fakes/weak-4b.toml"   # TOML or JSON
```

The file holds the same three keys (`script`, `script_by_case`, `unsupported`),
its path resolves against `eval.toml`, and its *contents* go into the config
hash - so the same benchmark hashes the same in two checkouts, and editing the
fixture is correctly a different benchmark.

## Privacy

Local-first, and not in a marketing sense:

- Traces, tool cassettes and reports stay in `./.slmeval/`.
- The only network traffic is to the model endpoints you configured.
- No telemetry. Not opt-out — absent.
- `eval.toml` holds the *name* of an environment variable (`api_key_env`), never
  a key, so it is safe to commit and safe to paste into an issue.
- In CI it is narrower still: replay mode reads tool responses out of SQLite, so
  a pull-request check needs no credential at all and makes no outbound request.

There is no `--sync`, and if one is ever added it will be opt-in per command,
it will say exactly what it uploads before it uploads anything, and it will
never carry prompts, tool payloads or model names from a private config. Stating
that now is cheaper than being asked later.

## Development

```bash
uv sync --extra dev
uv run pytest             # offline: no GPU, no keys, no network, no spend
uv run pytest -m live     # opt-in: hits whichever providers are reachable
uv run slmeval --help
```

The default suite is ~1,090 tests in about thirty seconds, fully offline
against the scripted [`fake` provider](#the-fake-provider): every evaluator and
every failure classification is asserted against a scripted model rather than a
real one. `-m live` is excluded on purpose - a suite that costs money or needs a GPU
is a suite people stop running, and a suite people stop running protects
nothing. The 30-second budget is enforced in CI for the same reason.

Three parts of it are worth knowing about before you change anything:

```bash
uv run pytest tests/test_determinism.py       # same seed twice -> the same report
uv run pytest tests/test_failure_matrix.py    # one candidate per way a run can fail
uv run pytest tests/test_snapshots.py         # every rendered table, pinned
uv run pytest tests/test_example_benchmark.py # the bundled example, as a user runs it
uv run pytest tests/test_github_action.py     # the docs and the Action, against the real CLI
uv run pytest tests/test_langgraph_adapter.py # both adapters, same trajectory
```

The last two are there because documentation is the part of this project most
likely to rot. `test_github_action.py` extracts every command line beginning
with `slmeval` from the READMEs, the workflows and the composite action, and
checks it against the live CLI — so renaming a flag goes red here before it
breaks somebody's pipeline.

The tables under `tests/snapshots/` are golden files. When you change what the
scorecard prints, regenerate them and review the diff as part of the change:

```bash
SLMEVAL_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_snapshots.py
```

And because `pytest` runs against the working tree rather than the artifact,
the wheel gets its own check - a fresh virtualenv, the built wheel and nothing
else, then `init`, `validate`, `doctor`, `record`, `run` and `report` as a new
user meets them:

```bash
python scripts/install_smoke.py
```

## License

Apache-2.0
