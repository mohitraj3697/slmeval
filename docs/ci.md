# CI

## Gate a run

```bash
slmeval run --tool-mode replay
slmeval gate --min-pass-k 0.90 --max-cost-ratio 0.40 --max-regression 0.05
```

Replay mode reads tool responses from SQLite, so CI needs no tool credentials. The
runner still has to reach your candidate models, unless they use the `fake` provider.

Thresholds can go in `eval.toml`, so CI can call `slmeval gate` with no arguments:

```toml
[gate]
min_pass_k = 0.5
max_regression = 0.05
```

| Option | Checks |
|---|---|
| `--min-pass-k` | Floor on `pass^k`, read on the Wilson lower bound by default |
| `--use-point-estimate` | Read the floor on the point estimate instead |
| `--max-cost-ratio` | Candidate cost against the baseline |
| `--max-p95-ratio` | Candidate p95 against the baseline. Needs a `--serial-latency` run. |
| `--max-regression` | Allowed drop against a committed baseline file |
| `--max-uncovered` | Share of executions allowed on unrecorded paths |
| `--require-fault-parity` | Refuse candidates that met different faults from the baseline |

A forbidden tool call fails the gate at any pass rate. Turning that off takes an
explicit `--allow-forbidden-tools`.

## Exit codes

These won't change without a major version bump.

| Code | Meaning | Is it a regression? |
|---|---|---|
| 0 | Every threshold was met | |
| 1 | A threshold was missed, or a forbidden tool was called | Yes |
| 2 | Bad invocation, such as no candidate or no threshold | No |
| 3 | A model server was unreachable | No |
| 4 | Not enough recorded responses to judge the run | No |

Keep 1 separate from 3 and 4 in your pipeline. Otherwise "Ollama wasn't running" looks
like "the model got worse".

A few more behaviours to know about:

- A check that can't run, such as a latency budget on a throughput run, prints "not
  measured". It neither passes nor fails.
- A gate with nothing to check exits 2.
- A forbidden tool call still exits 1 when the rest of the run couldn't be judged.

## Baselines

`--max-regression` compares against a committed file, not the last local run:

```bash
slmeval baseline save
```

This writes `slmeval.baseline.json`. Commit it, so that accepting a drop shows up as a
diff in review. The file carries the config and dataset hashes. `gate` refuses to compare
across a change in either unless you pass `--force`.

## Reports

```bash
slmeval gate --format md --out gate.md
slmeval report --format junit --out results.xml
```

## GitHub Action

```yaml
- uses: actions/checkout@v4
- uses: mohitraj3697/slmeval/examples/github-action@main
  with:
    working-directory: evals
    cassettes: evals/cassettes.sqlite
```

It runs on a stock `ubuntu-latest` runner. It validates the config, loads the cassettes,
replays the matrix, runs the gate, writes the scorecard to the job summary and keeps one
comment updated on the pull request.

Exit codes 3 and 4 fail the job, because no measurement happened, but they are labelled
as infrastructure and coverage problems, not regressions.

Before you commit a cassette file, remember that it contains your tools' real responses.
Check it for customer data and secrets, and use the cache route if it can't be committed.

See the [Action's README](../examples/github-action/README.md) for inputs, outputs and both
ways of getting cassettes onto a runner.
[`gate-demo.yml`](../.github/workflows/gate-demo.yml) runs the Action against the demo on
every push and pull request.
