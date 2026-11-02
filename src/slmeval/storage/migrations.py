"""Forward-only schema migrations for ``runs.sqlite``.

This exists on day one, before there is anything to migrate, and that is the
whole point: the schema *will* change in week 2, and by then users will have
runs they care about. A tool whose upgrade path is "delete your benchmark
history" does not get a second chance.

Three decisions worth defending.

**A ``schema_version`` table, not ``PRAGMA user_version``.** The cassette store
uses the pragma, and for a cache that can be deleted and re-recorded that is
the right call. Run history cannot be re-recorded, so when a user reports that
a report renders oddly, the first question is which migrations this file has
seen and which build applied them. An integer cannot answer that; a table with
``applied_at`` and ``slmeval_version`` can.

**Each migration and its version row commit together.** SQLite has
transactional DDL, so the statements and the bookkeeping insert go in one
transaction. A crash mid-upgrade therefore leaves the file at the previous
version with the previous schema, never at a version it does not actually
have - which would be unrecoverable without hand-editing.

**Forward only.** No downgrades. A newer build opening an older file upgrades
it; an older build opening a newer file refuses loudly and tells the user to
upgrade. Writing a half-understood schema is how you corrupt the one dataset
the user cannot regenerate.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from slmeval import __version__
from slmeval.errors import EnvironmentError_

_VERSION_TABLE: Final = """
CREATE TABLE IF NOT EXISTS schema_version (
    version         INTEGER PRIMARY KEY,
    name            TEXT NOT NULL,
    applied_at      TEXT NOT NULL,
    slmeval_version TEXT NOT NULL
)
"""

_V1: Final = (
    # ------------------------------------------------------------------ runs
    """
    CREATE TABLE IF NOT EXISTS runs (
        run_id          TEXT PRIMARY KEY,
        name            TEXT NOT NULL,
        started_at      TEXT NOT NULL,
        finished_at     TEXT,
        config_json     TEXT NOT NULL,
        config_hash     TEXT NOT NULL,
        dataset_hash    TEXT NOT NULL,
        dataset_path    TEXT NOT NULL DEFAULT '',
        slmeval_version TEXT NOT NULL,
        git_sha         TEXT NOT NULL DEFAULT '',
        host_info       TEXT NOT NULL DEFAULT '{}',
        mode            TEXT NOT NULL,
        seed            INTEGER NOT NULL,
        fault_rate      REAL NOT NULL DEFAULT 0.0,
        runs_per_case   INTEGER NOT NULL DEFAULT 1,
        status          TEXT NOT NULL,
        error_message   TEXT
    )
    """,
    # `slmeval report` with no arguments means "the last run", and `cache
    # prune` walks runs oldest-first. Both are this index.
    "CREATE INDEX IF NOT EXISTS runs_by_start ON runs (started_at DESC)",
    # ------------------------------------------------------------ candidates
    """
    CREATE TABLE IF NOT EXISTS candidates (
        run_id         TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        candidate_id   TEXT NOT NULL,
        role           TEXT NOT NULL DEFAULT 'candidate',
        ordinal        INTEGER NOT NULL DEFAULT 0,
        provider       TEXT NOT NULL DEFAULT '',
        model          TEXT NOT NULL DEFAULT '',
        model_cfg_json TEXT NOT NULL,
        PRIMARY KEY (run_id, candidate_id)
    )
    """,
    # ------------------------------------------------------------ executions
    """
    CREATE TABLE IF NOT EXISTS executions (
        exec_id                 TEXT PRIMARY KEY,
        run_id                  TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        candidate_id            TEXT NOT NULL,
        case_id                 TEXT NOT NULL,
        repeat_idx              INTEGER NOT NULL,
        attempt                 INTEGER NOT NULL DEFAULT 0,
        status                  TEXT NOT NULL,
        outcome_class           TEXT NOT NULL DEFAULT 'unclassified',
        started_at              TEXT NOT NULL DEFAULT '',
        duration_ms             REAL NOT NULL DEFAULT 0.0,
        steps                   INTEGER NOT NULL DEFAULT 0,
        total_prompt_tokens     INTEGER,
        total_completion_tokens INTEGER,
        tokens_estimated        INTEGER NOT NULL DEFAULT 0,
        cost_usd                REAL,
        fault_vector_json       TEXT NOT NULL DEFAULT '[]',
        fault_digest            TEXT NOT NULL DEFAULT '',
        trajectory_json         TEXT NOT NULL DEFAULT '[]',
        warnings_json           TEXT NOT NULL DEFAULT '[]',
        output_json             TEXT,
        output_hash             TEXT,
        state_json              TEXT,
        state_hash              TEXT,
        error_class             TEXT,
        error_message           TEXT,
        -- The constraint that makes `--resume` safe. A resumed run recomputes
        -- the same exec_id for a unit it already finished, so a double-write
        -- replaces one row instead of silently doubling a candidate's n and
        -- halving its pass rate.
        UNIQUE (run_id, candidate_id, case_id, repeat_idx)
    )
    """,
    "CREATE INDEX IF NOT EXISTS executions_by_cell ON executions "
    "(run_id, candidate_id, case_id)",
    "CREATE INDEX IF NOT EXISTS executions_by_outcome ON executions (run_id, outcome_class)",
    # ----------------------------------------------------------- node traces
    """
    CREATE TABLE IF NOT EXISTS node_traces (
        exec_id             TEXT NOT NULL REFERENCES executions(exec_id) ON DELETE CASCADE,
        seq                 INTEGER NOT NULL,
        node_name           TEXT NOT NULL DEFAULT '',
        node_path           TEXT NOT NULL DEFAULT '[]',
        model_id            TEXT NOT NULL DEFAULT '',
        provider            TEXT NOT NULL DEFAULT '',
        model               TEXT NOT NULL DEFAULT '',
        latency_ms          REAL NOT NULL DEFAULT 0.0,
        ttft_ms             REAL,
        retries             INTEGER NOT NULL DEFAULT 0,
        retry_ms            REAL NOT NULL DEFAULT 0.0,
        prompt_tokens       INTEGER,
        completion_tokens   INTEGER,
        tokens_estimated    INTEGER NOT NULL DEFAULT 0,
        schema_name         TEXT,
        schema_state        TEXT NOT NULL DEFAULT 'not_requested',
        structured_requested TEXT NOT NULL DEFAULT 'auto',
        structured_mode     TEXT,
        rungs_tried         TEXT NOT NULL DEFAULT '[]',
        repair_attempts     INTEGER NOT NULL DEFAULT 0,
        json_stage          TEXT,
        error_class         TEXT,
        error_message       TEXT,
        raw_json            TEXT,
        raw_hash            TEXT,
        PRIMARY KEY (exec_id, seq)
    )
    """,
    # Per-node attribution - "which node did this candidate break in" - is the
    # consulting wedge in one table, and it reads this index.
    "CREATE INDEX IF NOT EXISTS node_traces_by_node ON node_traces (node_name, model_id)",
    # ----------------------------------------------------------- tool traces
    """
    CREATE TABLE IF NOT EXISTS tool_traces (
        exec_id        TEXT NOT NULL REFERENCES executions(exec_id) ON DELETE CASCADE,
        seq            INTEGER NOT NULL,
        tool_name      TEXT NOT NULL,
        node_name      TEXT NOT NULL DEFAULT '',
        node_path      TEXT NOT NULL DEFAULT '[]',
        args_json      TEXT,
        args_hash      TEXT,
        cassette_key   TEXT NOT NULL DEFAULT '',
        outcome        TEXT NOT NULL DEFAULT 'live',
        match_tier     TEXT,
        cache_hit      INTEGER NOT NULL DEFAULT 0,
        fault          TEXT,
        fault_injected INTEGER NOT NULL DEFAULT 0,
        miss_verdict   TEXT,
        tool_version   TEXT NOT NULL DEFAULT '',
        latency_ms     REAL NOT NULL DEFAULT 0.0,
        output_json    TEXT,
        output_hash    TEXT,
        error_message  TEXT,
        PRIMARY KEY (exec_id, seq)
    )
    """,
    # "Every forbidden-tool call in this run" and "every cassette miss" are
    # both scans without this, and both run on the failure path where the user
    # is already waiting.
    "CREATE INDEX IF NOT EXISTS tool_traces_by_tool ON tool_traces (tool_name, outcome)",
    # ---------------------------------------------------------- evaluations
    """
    CREATE TABLE IF NOT EXISTS evaluations (
        exec_id     TEXT NOT NULL REFERENCES executions(exec_id) ON DELETE CASCADE,
        tier        INTEGER NOT NULL,
        metric      TEXT NOT NULL,
        value_num   REAL,
        value_text  TEXT,
        detail_json TEXT,
        PRIMARY KEY (exec_id, tier, metric)
    )
    """,
    "CREATE INDEX IF NOT EXISTS evaluations_by_metric ON evaluations (metric)",
    # --------------------------------------------------------- case rollups
    """
    CREATE TABLE IF NOT EXISTS case_rollups (
        run_id       TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
        candidate_id TEXT NOT NULL,
        case_id      TEXT NOT NULL,
        n            INTEGER NOT NULL,
        c            INTEGER NOT NULL,
        pass_k       REAL,
        pass_hat_k   REAL,
        wilson_lo    REAL,
        wilson_hi    REAL,
        PRIMARY KEY (run_id, candidate_id, case_id)
    )
    """,
)


_V2: Final = (
    # Which scheduling mode produced the run. Not a cosmetic flag: local-model
    # latency measured under concurrency is meaningless - requests queue in the
    # server and the numbers come out several times too slow - so a report has
    # to know whether it may print a latency column at all, and that knowledge
    # must survive the process that measured it.
    "ALTER TABLE runs ADD COLUMN scheduling TEXT NOT NULL DEFAULT 'throughput'",
)


@dataclass(frozen=True, slots=True)
class Migration:
    """One forward step. Never edited after release - a changed migration is a
    different schema wearing the same version number, and the two users who
    upgraded on either side of the edit now disagree about what their file
    contains."""

    version: int
    name: str
    statements: tuple[str, ...]


_V3: Final = (
    # The Tier 3 judgement cache.
    #
    # Not scoped to a run, and that is the entire design: a judgement is a
    # function of (rubric, reference, candidate output, judge model) and
    # nothing else, so re-rendering a report, re-running a gate or judging the
    # same unchanged output next week has to cost nothing. A cache keyed on the
    # run would make `slmeval report` a billable operation, and a tool that
    # charges you to look at yesterday's table twice is a tool people stop
    # looking at.
    #
    # `order_shown` is kept because the judge is asked about a randomised A/B
    # pair - position bias is real and large - and a cached verdict that could
    # not say which side the candidate was on would be unauditable.
    """
    CREATE TABLE IF NOT EXISTS judgements (
        cache_key       TEXT PRIMARY KEY,
        judge_model     TEXT NOT NULL,
        rubric_hash     TEXT NOT NULL,
        verdict         TEXT NOT NULL,
        score           REAL,
        reasoning       TEXT,
        cited_span      TEXT,
        order_shown     TEXT NOT NULL,
        prompt_tokens   INTEGER,
        completion_tokens INTEGER,
        created_at      TEXT NOT NULL,
        slmeval_version TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS judgements_by_model ON judgements (judge_model)",
)


MIGRATIONS: Final[tuple[Migration, ...]] = (
    Migration(version=1, name="initial", statements=_V1),
    Migration(version=2, name="run_scheduling_mode", statements=_V2),
    Migration(version=3, name="judgement_cache", statements=_V3),
)

SCHEMA_VERSION: Final = MIGRATIONS[-1].version


def current_version(conn: sqlite3.Connection) -> int:
    """Highest applied migration, or 0 for a fresh file."""
    conn.execute(_VERSION_TABLE)
    row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
    return int(row[0]) if row and row[0] is not None else 0


def applied(conn: sqlite3.Connection) -> list[tuple[int, str, str, str]]:
    """The upgrade history, oldest first. Printed by ``slmeval cache stats
    --verbose`` and the first thing to ask for in a bug report."""
    conn.execute(_VERSION_TABLE)
    return [
        (int(r[0]), str(r[1]), str(r[2]), str(r[3]))
        for r in conn.execute(
            "SELECT version, name, applied_at, slmeval_version "
            "FROM schema_version ORDER BY version"
        ).fetchall()
    ]


def migrate(conn: sqlite3.Connection, *, path: Path | None = None) -> list[int]:
    """Bring ``conn`` up to :data:`SCHEMA_VERSION`. Returns versions applied.

    Idempotent and safe to call on every open: the common case is a single
    ``SELECT MAX(version)`` against a one-row table, which costs nothing, and
    doing it at open means there is no separate ``slmeval migrate`` command for
    a user to forget.
    """
    version = current_version(conn)
    if version > SCHEMA_VERSION:
        where = f"{path} " if path else ""
        raise EnvironmentError_(
            f"{where}was written by a newer slmeval "
            f"(store schema v{version}, this build reads v{SCHEMA_VERSION})",
            hint="upgrade slmeval - this build will not write to a schema it does not know",
        )

    done: list[int] = []
    for migration in MIGRATIONS:
        if migration.version <= version:
            continue
        # BEGIN IMMEDIATE takes the write lock up front. Two processes opening
        # the store at the same moment - a run and a `cache stats` in another
        # terminal - would otherwise both read version 0 and both try to apply
        # migration 1; the loser gets a clean "database is locked" wait rather
        # than a half-applied upgrade.
        conn.execute("BEGIN IMMEDIATE")
        try:
            # Re-read inside the transaction: the process that held the lock
            # may have applied this very migration while we waited.
            if current_version(conn) >= migration.version:
                conn.execute("COMMIT")
                continue
            for statement in migration.statements:
                conn.execute(statement)
            conn.execute(
                "INSERT INTO schema_version (version, name, applied_at, slmeval_version) "
                "VALUES (?, ?, ?, ?)",
                (
                    migration.version,
                    migration.name,
                    datetime.now(UTC).isoformat(),
                    __version__,
                ),
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        done.append(migration.version)
    return done


__all__ = ["MIGRATIONS", "SCHEMA_VERSION", "Migration", "applied", "current_version", "migrate"]
