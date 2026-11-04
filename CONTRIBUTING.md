# Contributing

Thanks for helping. Bug reports, documentation fixes and results from real models are all
useful. For a large change, please open an issue first so we can agree on the approach.

## Setup

You need Python 3.11 or newer and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/mohitraj3697/slmeval
cd slmeval
uv sync --extra dev
uv run slmeval --help
```

## Before you open a pull request

```bash
uv run ruff check src tests scripts
uv run mypy src
uv run pytest
```

The default test suite is offline: no GPU, API keys, network or spend. It takes about 30
seconds, and CI fails it above 120. Tests that call real providers are marked `live` and
only run when you ask for them:

```bash
uv run pytest -m live
```

CI also runs the suite on Linux and Windows, the LangGraph tests, an install check of the
built wheel, and the steps below.

## If your change alters what the tool prints

Rendered tables are stored as snapshots in `tests/snapshots/`. Regenerate them and review
the diff as part of your change:

```bash
SLMEVAL_UPDATE_SNAPSHOTS=1 uv run pytest tests/test_snapshots.py
```

The published teardown and the README images come from real runs. If a number moves,
regenerate them and say why in the pull request:

```bash
python scripts/teardown.py
python scripts/readme_visuals.py
```

`python scripts/teardown.py --check` re-runs the benchmark and fails if a published file
has gone stale. `tests/test_visuals.py` does the same for the charts.

## If your change alters the CLI

`tests/test_github_action.py` checks every `slmeval` command in the docs, the workflows
and the Action against the real CLI, so a renamed flag fails there first. Update the docs
in the same pull request.

These two are public contracts. Changing either needs a changelog entry that says so:

- the exit codes of `slmeval gate`
- the `schema` field of the JSON report

## Tests worth knowing

| File | What it covers |
|---|---|
| `tests/test_determinism.py` | The same seed gives the same report |
| `tests/test_failure_matrix.py` | One candidate for each way a run can fail |
| `tests/test_snapshots.py` | Every rendered table |
| `tests/test_example_benchmark.py` | The demo, run as a user runs it |
| `tests/test_langgraph_adapter.py` | Both adapters take the same path |
| `tests/test_teardown.py` | The write-up against its artifacts |

Please test new failure modes through the `fake` provider, so they stay fast and free.

## Style

- Ruff and mypy settings are in `pyproject.toml`. Source is strictly typed.
- Keep the core install small. Framework support belongs in an extra.
- Keep commits focused, and explain why in the message when it isn't obvious.

## Reporting bugs

Please use the [issue templates](https://github.com/mohitraj3697/slmeval/issues/new/choose).
The output of `slmeval doctor` helps a lot. Don't paste real customer data from a cassette
or trace into a public issue.

For security problems, see [SECURITY.md](SECURITY.md).

By contributing, you agree that your work is licensed under the
[Apache 2.0 license](LICENSE). This project follows the
[Code of Conduct](CODE_OF_CONDUCT.md).
