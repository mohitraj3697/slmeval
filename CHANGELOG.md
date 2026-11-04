# Changelog

Versions follow [semantic versioning](https://semver.org). While the major version is 0, a
minor release can change the CLI and the stored schema.

The exit codes of `slmeval gate` and the `schema` field of the JSON report are public
contracts. Any change to them is listed here as breaking.

## [Unreleased]

### Fixed

- `slmeval record` no longer reports a case as "not recorded" when the agent ran to the end and
  only failed an assertion. It now exits 0 for those cases and says the baseline failed them. It
  also warns that a forbidden tool really ran during recording. A case that stops early still
  exits 4.
- The per-node recommendation no longer says to keep the baseline on a node the candidate didn't
  change. When a failure surfaces in such a node, it points at the nodes the candidate did change.

### Changed

- Shorter wording for the note that latency isn't measured in throughput mode.
- The example support agent accepts lowercase order ids and has more orders. Its reply step now
  receives the delivery date, the policy reason and whether a refund was issued.

### Added

- [Teardown #2](teardown-02/README.md): three local models through Ollama on one laptop.

## [0.1.0] - 2026-11-04

First release, published on PyPI as `slmeval-cli`. The command and the Python import are
both `slmeval`.

### Harness

- `slmeval.node()` and `slmeval.chat_model()` mark steps and route model calls per candidate.
  Both do nothing outside a benchmark.
- `slmeval.instrument_langgraph()` marks every node of a LangGraph builder. Requires the
  `langgraph` extra.
- `@sandbox_tool` records a tool call once and replays it for every candidate. A miss gets a
  verdict and a diff of the arguments against the nearest recording.
- Recorded responses keep their types, including `Decimal`, `datetime` and enums. A value that
  can't be stored faithfully raises an error at record time.
- Seven kinds of fault injection, scheduled per case and seed. Each run records the faults it met.
- Providers for OpenAI, Ollama, anything serving `/v1/chat/completions`, and a scripted `fake`.
  Reports show which structured-output mechanism each call needed.

### Measurement

- Dataset assertions, including forbidden tools as a hard fail.
- Path distance, tool overlap with the baseline, loop detection and recovery from faults.
- `pass@k`, `pass^k` and Wilson intervals. The gate reads the lower bound by default.
- Cost per case. Local inference is reported as modelled from token counts and `--gpu-hourly`.
- Latency only in `--serial-latency` mode.
- Bugs in slmeval, unrecorded paths and unreachable providers stay out of pass rates and are
  reported separately.
- An LLM judge with a calibration harness, off by default. It isn't connected to `slmeval run` yet.

### CLI

`init`, `doctor`, `validate`, `record`, `run`, `report`, `show run|exec|case|uncovered`, `gate`,
`baseline save|show`, `judge calibrate`, `dataset validate` and `cache stats|prune`.

Reports render as a table, JSON, Markdown, CSV or JUnit XML.

Exit codes: 0 pass, 1 gate failed, 2 usage, 3 environment, 4 coverage.

### Also included

- A [GitHub Action](examples/github-action) that runs the benchmark, writes the job summary and
  keeps one pull request comment updated.
- A [runnable example](examples/support_agent) and a [judge calibration example](examples/judge).
- [Teardown #1](teardown), a published run with scripted models. CI re-runs it on Linux and Windows.

### Not in this release

- `slmeval compare` and `slmeval cassette`. They are hidden from `--help` and exit 2 if called.
- A native Anthropic provider. Reach Claude through an OpenAI-compatible gateway.
- Time-to-first-token measurement.
