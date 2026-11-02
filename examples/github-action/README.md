# slmeval in GitHub Actions

Block the merge when a model swap makes the agent worse.

```yaml
- uses: actions/checkout@v4
- uses: slmeval/slmeval/examples/github-action@main
  with:
    working-directory: evals
    cassettes: evals/cassettes.sqlite
```

That is the whole integration. **No GPU and no API keys on the runner** — the
agent's tool calls are replayed out of a local SQLite file, and the models are
whichever ones your `eval.toml` names (the bundled `fake` provider needs no
network at all). A standard `ubuntu-latest` runner is enough, which is the only
reason a benchmark like this can live in a pull-request check.

[`slmeval.yml`](slmeval.yml) is a complete workflow to copy into
`.github/workflows/`. [`action.yml`](action.yml) is the composite action it
calls.

## What the job does

1. `slmeval validate --json` — does the config load, and what is the dataset's
   content hash?
2. restores the cassette store from `actions/cache`, keyed on that hash
3. `slmeval run --tool-mode replay`
4. `slmeval gate`
5. writes the scorecard and the verdict to `$GITHUB_STEP_SUMMARY`, and edits
   one sticky pull-request comment
6. exits with the gate's own exit code

## The exit codes are the point

A gate that cannot tell "the model got worse" from "Ollama was down" teaches a
team to ignore it, and after that it is worth nothing. So each outcome gets its
own annotation, its own wording in the comment, and its own exit code:

| Exit | `verdict` output | Annotation |
|---|---|---|
| 0 | `passed` | notice |
| 1 | `failed` | **regression** — a threshold was not met |
| 2 | `usage` | the gate was misconfigured: no candidate, or no threshold |
| 3 | `environment` | **infrastructure, not a regression** — a provider was unreachable |
| 4 | `coverage` | **coverage, not a regression** — too little recorded ground truth to judge |

3 and 4 still fail the job. A measurement that did not happen must not read as
a pass — but it is labelled, everywhere it appears, as something other than a
model getting worse.

## Where the cassettes come from

Replay has to have something to replay. Record once, locally:

```bash
slmeval record          # executes your real tools exactly once
```

Then pick one of two routes.

**Commit the file** (recommended). Copy `.slmeval/cassettes.sqlite` to a stable
path in the repository and pass it as `cassettes:`. A committed cassette store
means CI is reproducible, is immune to cache eviction, and that re-recording is
a reviewable diff rather than an invisible change in a runner's cache.

**Use the cache.** Leave `cassettes` unset and the action restores
`.slmeval/` from `actions/cache` keyed on `slmeval-cassettes-<os>-<dataset
hash>`. The key is the dataset's content hash and there are deliberately no
`restore-keys`: a prefix hit would hand the run a store recorded against a
different dataset, and a stale cassette reads as a model failure. When the
cache misses, the run still completes and the gate exits 4 — "not enough
recorded ground truth to judge this" — which is the honest answer and is not
the same as red.

`record: missing` makes the action record on a cache miss. That executes your
real tools on a GitHub runner, so set it only when those tools are hermetic, or
when the provider is `fake` and the tools are local functions — which is
exactly the case for the bundled
[`examples/support_agent`](../support_agent), and why this repository can
demonstrate the whole flow in
[`.github/workflows/gate-demo.yml`](../../.github/workflows/gate-demo.yml) with
no secrets of any kind.

## Thresholds

Put them in `eval.toml`, not in the workflow:

```toml
[gate]
min_pass_k = 0.90
use_wilson_lower = true     # the default, and the one that matters
max_cost_ratio = 0.40
max_regression = 0.05
```

Then CI calls `slmeval gate` bare, the same gate runs on a laptop, and a change
to a threshold is a diff somebody approved rather than a line in a YAML file
nobody reads. `max_regression` compares against `slmeval.baseline.json`, which
is also committed — see the main [README](../../README.md#gating-a-pull-request).

## Inputs

| Input | Default | |
|---|---|---|
| `working-directory` | `.` | where `eval.toml` lives |
| `config` | | an `eval.toml` somewhere else |
| `version` | `slmeval` | pip requirement; `.` installs the checkout |
| `python-version` | `3.11` | |
| `cassettes` | | a committed cassette file to use |
| `cache` | `true` | restore/save `.slmeval/` via `actions/cache` |
| `record` | `never` | `missing` records when no cassettes were found |
| `run-args` | | extra arguments for `slmeval run` |
| `gate-args` | | extra arguments for `slmeval gate` |
| `comment` | `true` | sticky pull-request comment |
| `token` | `github.token` | needs `pull-requests: write` |
| `artifact` | `true` | upload the run database, JSON report and JUnit XML |

## Outputs

`exit-code`, `verdict`, `run-id`, `dataset-hash`, `config-hash`,
`cassettes-restored`.

## Artifacts

`slmeval-report.json` (the versioned JSON contract), `slmeval-report.md`,
`slmeval-gate.md`, `slmeval-junit.xml` (one test case per candidate per case,
for an existing dashboard) and `.slmeval/runs.sqlite` — from which `slmeval
report --run-id` and `slmeval show` reproduce the whole run locally, with
nothing re-executed.
