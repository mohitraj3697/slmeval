# Changelog

Notable changes, newest first. Versions follow [semantic
versioning](https://semver.org); while the major version is 0, a minor bump may
change the CLI surface and the stored schema.

Two things in here are contracts rather than features, and a change to either
is a breaking change even if no Python signature moves: the **exit codes**
`slmeval gate` returns, which pipelines branch on, and the `schema` field of
the JSON report, which anything charting a run reads.

## [0.1.0] — 2026-11-04

First release. `slmeval` records an agent's tool calls once and replays them
against every candidate model, so a model-swap benchmark runs offline, with no
GPU and no API key, and reports what the swap costs in reliability rather than
only in tokens.

### The harness

- **Two touch points in your agent**, both no-ops outside a benchmark:
  `slmeval.node()` names the step a model call belongs to, and
  `slmeval.chat_model()` routes it to whatever model this candidate assigns to
  that node. `slmeval.instrument_langgraph()` replaces both for a LangGraph
  graph, with the example agent running identically under either adapter.
- **`@sandbox_tool`** records a tool call once and replays it for every
  candidate after that. Keys are matched through a four-tier cascade, so a miss
  is classified rather than just raised: a hallucinated parameter, a
  normalisation gap of ours, an uncovered path, or a genuinely new call. The
  diagnosis names the tool, prints the nearest recorded call and diffs the
  arguments one at a time.
- **A typed envelope** around every recorded response. A `Decimal` replays as a
  `Decimal` and an enum as an enum; a value that cannot be serialised is
  refused loudly at record time rather than silently stringified into something
  the agent will mis-handle on replay.
- **Deterministic fault injection** over seven kinds, scheduled per
  `(case_id, seed)` and indexed per tool by that tool's own Nth call — so an
  extra call to one tool cannot shift every later fault onto a different step.
  Each run records the fault vector it actually met.
- **Providers**: OpenAI, Ollama, anything speaking `/v1/chat/completions`, and
  `fake` for tests. Structured output goes down a ladder (tool call → JSON
  schema → JSON mode → prompt repair) and the report says which rung each call
  needed, because a model that only produces valid JSON after repair is a
  different operational proposition from one that does it directly.

### The measurement

- Tier 1 deterministic assertions, including forbidden tools as a hard fail
  that outranks the execution's own outcome.
- Tier 2 trajectory distance, tool precision/recall/F1, loop and stall
  detection, redundancy, and recovery from injected faults.
- Tier 3 LLM judge, off by default and refusing to report a number until
  `slmeval judge calibrate` has measured its agreement with your own labels.
- `pass@k`, `pass^k` and Wilson intervals on every rate, with the gate reading
  the lower bound rather than the point estimate.
- Cost per case, with local inference reported as **modelled** from token
  counts and a `--gpu-hourly` figure you supply — never as measured.
- Latency only in `--serial-latency` mode. A throughput run reports no p95 at
  all, because under concurrency that number is the scheduler's.
- Every failure is attributed before it is counted. A bug in slmeval
  (`harness_error`) and a path we never recorded (`uncovered`) are kept out of
  every rate and printed above the table; a crash in the agent under test is
  the agent's, wherever in the standard library its exception type happens to
  live. Those three numbers in one column labelled "failed" would add up to
  something that looks like evidence and is not.

### The CLI

`init`, `doctor`, `validate`, `record`, `run`, `report`, `show
run|exec|case|uncovered`, `gate`, `baseline save|show`, `judge calibrate`,
`dataset validate`, `cache stats|prune`. Reports render as a table, JSON,
Markdown, CSV or JUnit XML, and the Markdown one — the format that ends up in a
pull request — carries the warnings above the table and the fault-recovery rate
below it.

Exit codes: `0` pass, `1` gate failed, `2` usage, `3` environment, `4`
coverage.

Plus a [GitHub Action](examples/github-action) that runs the whole thing on a
stock runner, caches the cassettes on the dataset's content hash, and posts the
verdict to the pull request.

### Published with it

- [`examples/support_agent`](examples/support_agent) — four nodes, five
  sandboxed tools, three candidates and three different ways to fail. Runs
  offline.
- [`examples/judge`](examples/judge) — what calibrating the Tier 3 judge
  actually requires.
- [Teardown #1](teardown) — that example put through the whole tool, published
  with the scorecard, the failure traces, the cassette-miss diagnosis, the
  gate's exit 1, a fault-injection run, and the config and dataset that produce
  them. `python scripts/teardown.py --check` re-runs the benchmark and fails if
  a published artifact has gone stale; CI runs it on both platforms, so the
  document cannot quietly stop being true.

### Not in this release

`slmeval compare` and `slmeval cassette ls|verify|prune` are not built. They are
hidden from `--help` and exit 2 with a message if called. The `anthropic` provider is not built
(reach Claude through an OpenAI-compatible gateway), and there is no
time-to-first-token measurement.
