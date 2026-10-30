"""`slmeval cache` — the broom.

Benchmark traces grow to gigabytes, so these guard the two things a cleanup
command must never get wrong: an age it misread, and a deletion the user did
not agree to.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from typer.testing import CliRunner

from slmeval import exit_codes
from slmeval.cli.commands.cache_cmd import human_bytes, parse_age
from slmeval.cli.main import app
from slmeval.storage import RunRecord, RunStatus, RunStore
from tests.test_cli import text_of

runner = CliRunner()


def seed_run(root: Path, *, run_id: str, started: datetime, status: RunStatus) -> None:
    with RunStore.for_project(root) as store:
        store.save_run(
            RunRecord(
                run_id=run_id,
                name="refund-agent",
                started_at=started.isoformat(),
                config_json="{}",
                config_hash="cfg",
                dataset_hash="ds",
                slmeval_version="0.1.0",
                mode="replay",
                seed=42,
                status=status,
            )
        )


@pytest.fixture
def seeded(tmp_path: Path) -> Path:
    (tmp_path / "eval.toml").write_text("# placeholder\n", encoding="utf-8")
    seed_run(
        tmp_path,
        run_id="20250101-000000-aaaaaa",
        started=datetime(2025, 1, 1, tzinfo=UTC),
        status=RunStatus.COMPLETED,
    )
    seed_run(
        tmp_path,
        run_id=datetime.now(UTC).strftime("%Y%m%d-%H%M%S-bbbbbb"),
        started=datetime.now(UTC),
        status=RunStatus.COMPLETED,
    )
    return tmp_path


# --------------------------------------------------------------------------- #
# Age parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("30d", timedelta(days=30)),
        ("12h", timedelta(hours=12)),
        ("2w", timedelta(weeks=2)),
        ("90m", timedelta(minutes=90)),
        (" 45 S ", timedelta(seconds=45)),
    ],
)
def test_parse_age(text: str, expected: timedelta) -> None:
    assert parse_age(text) == expected


@pytest.mark.parametrize("text", ["30", "", "soon", "d30", "-5d", "1y"])
def test_a_bare_or_unknown_age_is_refused(text: str) -> None:
    """A bare `30` could mean seconds, days or runs, and the command it feeds
    deletes things. One retyped character beats a wrong guess."""
    from slmeval.errors import ConfigError

    with pytest.raises(ConfigError):
        parse_age(text)


def test_human_bytes() -> None:
    assert human_bytes(0) == "0 B"
    assert human_bytes(2048) == "2.0 KB"
    assert human_bytes(5 * 1024**3) == "5.0 GB"


# --------------------------------------------------------------------------- #
# stats
# --------------------------------------------------------------------------- #


def test_stats_reports_what_is_stored(seeded: Path) -> None:
    out = text_of(runner.invoke(app, ["cache", "stats", "--path", str(seeded)]))
    assert "runs" in out
    assert "schema" in out
    assert "payloads" in out


def test_stats_shows_migration_history(seeded: Path) -> None:
    """The first thing to ask for in a bug report about a stored run."""
    out = text_of(runner.invoke(app, ["cache", "stats", "--path", str(seeded), "-v"]))
    assert "migrations" in out
    assert "v1 initial" in out


def test_stats_works_before_the_first_run(tmp_path: Path) -> None:
    (tmp_path / "eval.toml").write_text("# placeholder\n", encoding="utf-8")
    result = runner.invoke(app, ["cache", "stats", "--path", str(tmp_path)])
    assert result.exit_code == exit_codes.OK
    assert "no runs stored yet" in text_of(result)


def test_stats_does_not_need_a_loadable_config(tmp_path: Path) -> None:
    """The point of `cache stats` is to work when something else is broken."""
    (tmp_path / "eval.toml").write_text("this is not [valid toml\n", encoding="utf-8")
    assert runner.invoke(app, ["cache", "stats", "--path", str(tmp_path)]).exit_code == 0


def test_outside_a_project_it_says_so(tmp_path: Path) -> None:
    result = runner.invoke(app, ["cache", "stats", "--path", str(tmp_path / "nope")])
    # A directory with no .slmeval is simply an empty store, not an error.
    assert result.exit_code == exit_codes.OK


# --------------------------------------------------------------------------- #
# prune
# --------------------------------------------------------------------------- #


def test_prune_lists_what_it_would_delete(seeded: Path) -> None:
    result = runner.invoke(
        app,
        ["cache", "prune", "--path", str(seeded), "--older-than", "30d", "--keep-last", "0",
         "--dry-run"],
    )
    out = text_of(result)
    assert "20250101-000000-aaaaaa" in out
    assert "nothing deleted" in out
    with RunStore.for_project(seeded) as store:
        assert store.get_run("20250101-000000-aaaaaa") is not None


def test_prune_asks_before_deleting(seeded: Path) -> None:
    """Deleting benchmark history is not undoable, and what is being deleted is
    what the next gate compares against."""
    result = runner.invoke(
        app,
        ["cache", "prune", "--path", str(seeded), "--older-than", "30d", "--keep-last", "0"],
        input="n\n",
    )
    assert "cancelled" in text_of(result)
    with RunStore.for_project(seeded) as store:
        assert store.get_run("20250101-000000-aaaaaa") is not None


def test_prune_deletes_on_confirmation(seeded: Path) -> None:
    result = runner.invoke(
        app,
        ["cache", "prune", "--path", str(seeded), "--older-than", "30d", "--keep-last", "0", "-y"],
    )
    assert result.exit_code == exit_codes.OK
    assert "deleted 1 run" in text_of(result)
    with RunStore.for_project(seeded) as store:
        assert store.get_run("20250101-000000-aaaaaa") is None
        assert len(store.list_runs()) == 1


def test_prune_defaults_keep_recent_runs(seeded: Path) -> None:
    """--keep-last defaults to 3, so the lazy invocation cannot leave a user
    with no history at all."""
    result = runner.invoke(
        app, ["cache", "prune", "--path", str(seeded), "--older-than", "1h", "-y"]
    )
    assert "nothing older than 1h to prune" in text_of(result)


def test_prune_refuses_while_a_run_is_in_flight(tmp_path: Path) -> None:
    (tmp_path / "eval.toml").write_text("# placeholder\n", encoding="utf-8")
    seed_run(
        tmp_path,
        run_id="20250101-000000-cccccc",
        started=datetime(2025, 1, 1, tzinfo=UTC),
        status=RunStatus.RUNNING,
    )
    result = runner.invoke(
        app,
        ["cache", "prune", "--path", str(tmp_path), "--older-than", "1d", "--keep-last", "0", "-y"],
    )
    assert result.exit_code == exit_codes.ENVIRONMENT
    assert "still in progress" in text_of(result)


def test_force_prunes_past_a_crash_leftover(tmp_path: Path) -> None:
    (tmp_path / "eval.toml").write_text("# placeholder\n", encoding="utf-8")
    seed_run(
        tmp_path,
        run_id="20250101-000000-cccccc",
        started=datetime(2025, 1, 1, tzinfo=UTC),
        status=RunStatus.RUNNING,
    )
    result = runner.invoke(
        app,
        ["cache", "prune", "--path", str(tmp_path), "--older-than", "1d", "--keep-last", "0",
         "-y", "--force"],
    )
    assert result.exit_code == exit_codes.OK
    assert "deleted 1 run" in text_of(result)


def test_a_bad_age_exits_with_the_usage_code(seeded: Path) -> None:
    result = runner.invoke(
        app, ["cache", "prune", "--path", str(seeded), "--older-than", "forever"]
    )
    assert result.exit_code == exit_codes.USAGE


def test_cache_is_no_longer_a_stub() -> None:
    out = text_of(runner.invoke(app, ["cache", "--help"]))
    assert "not implemented yet" not in out
    assert "stats" in out and "prune" in out
