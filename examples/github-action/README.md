# slmeval in GitHub Actions

Fail a pull request when a model change makes the agent worse.

```yaml
- uses: actions/checkout@v4
- uses: mohitraj3697/slmeval/examples/github-action@main
  with:
    working-directory: evals
    cassettes: evals/cassettes.sqlite
```

Tool calls are replayed from a local SQLite file, so the runner needs no GPU and no tool credentials.
The models are whichever ones `eval.toml` names, and the bundled `fake` provider needs no network. A
standard `ubuntu-latest` runner is enough.

[`slmeval.yml`](slmeval.yml) is a complete workflow to copy into `.github/workflows/`.
[`action.yml`](action.yml) is the composite action it calls.

## What the job does

1. `slmeval validate --json` loads the config and reads the dataset's content hash.
2. Restores the cassette store from `actions/cache`, keyed on that hash.
3. `slmeval run --tool-mode replay`
4. `slmeval gate`
5. Writes the scorecard and verdict to `$GITHUB_STEP_SUMMARY`, and keeps one comment updated on the
   pull request.
6. Exits with the gate's exit code.

## Exit codes

| Exit | `verdict` output | Annotation |
|---|---|---|
| 0 | `passed` | Notice |
| 1 | `failed` | Regression: a threshold was not met |
| 2 | `usage` | The gate was misconfigured: no candidate, or no threshold |
| 3 | `environment` | Infrastructure, not a regression: a provider was unreachable |
| 4 | `coverage` | Coverage, not a regression: too little recorded data to judge |

Codes 3 and 4 still fail the job, because nothing was measured, but they aren't labelled as a model
getting worse.

## Getting cassettes onto the runner

Record once, locally:

```bash
slmeval record
```

Then choose one of these.

**Commit the file.** Copy `.slmeval/cassettes.sqlite` to a stable path in the repository and pass it as
`cassettes:`. CI stays reproducible and re-recording shows up as a diff. Cassettes contain your tools'
real responses, so check them for customer data and secrets first.

**Use the cache.** Leave `cassettes` unset and the action restores `.slmeval/` from `actions/cache`,
keyed on `slmeval-cassettes-<os>-<dataset hash>`. There are no `restore-keys`, because a partial match
could restore recordings made for a different dataset. On a cache miss, the run finishes and the gate
exits 4.

`record: missing` makes the action record on a cache miss. That runs your real tools on a GitHub runner,
so only use it when those tools are self-contained. The bundled
[`examples/support_agent`](../support_agent) qualifies, which is how
[`gate-demo.yml`](../../.github/workflows/gate-demo.yml) runs the whole flow with no secrets.

## Thresholds

Put them in `eval.toml`, so the same gate runs on a laptop and CI can call `slmeval gate` with no
arguments:

```toml
[gate]
min_pass_k = 0.90
use_wilson_lower = true
max_cost_ratio = 0.40
max_regression = 0.05
```

`max_regression` compares against the committed `slmeval.baseline.json`. See the
[CI guide](../../docs/ci.md#baselines).

## Inputs

| Input | Default | |
|---|---|---|
| `working-directory` | `.` | Where `eval.toml` lives |
| `config` | | An `eval.toml` somewhere else |
| `version` | `slmeval` | pip requirement. `.` installs the checkout. |
| `python-version` | `3.11` | |
| `cassettes` | | A committed cassette file to use |
| `cache` | `true` | Restore and save `.slmeval/` through `actions/cache` |
| `record` | `never` | `missing` records when no cassettes were found |
| `run-args` | | Extra arguments for `slmeval run` |
| `gate-args` | | Extra arguments for `slmeval gate` |
| `comment` | `true` | Keep one comment updated on the pull request |
| `token` | `github.token` | Needs `pull-requests: write` |
| `artifact` | `true` | Upload the run database, JSON report and JUnit XML |

## Outputs

`exit-code`, `verdict`, `run-id`, `dataset-hash`, `config-hash`, `cassettes-restored`.

## Artifacts

`slmeval-report.json`, `slmeval-report.md`, `slmeval-gate.md`, `slmeval-junit.xml` and
`.slmeval/runs.sqlite`. With the database, `slmeval report --run-id` and `slmeval show` can inspect the
whole run locally without running anything again.
