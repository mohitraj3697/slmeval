"""The trace store: one SQLite file per project, plus a blob directory.

``.slmeval/runs.sqlite`` is the asset this whole tool produces. The scorecard,
the drill-down diff, the CI gate and any future hosted product are queries over
it, which sets the bar for this module: a completed run has to be fully
reconstructible from these rows, with every in-memory object thrown away.

The connection discipline is copied from :mod:`slmeval.sandbox.store` because
it was learned the hard way and the reasons have not changed: one connection
per thread, opened once under a lock, every pragma set at open, closed
explicitly. ``with sqlite3.connect(...)`` commits but does not close, and on
Windows a leaked handle keeps the file locked, so a user cannot delete their
own cache and a test cannot remove its temporary directory.

What is different here is the write pattern. The cassette store writes
independent rows; this one writes a tree - an execution, then its node traces,
tool traces and evaluations. All of it goes in one transaction. An execution
row with half its tool calls committed does not read as a crash, it reads as an
agent that stopped after three steps, and the entire product is built on
trusting that a stored trajectory is the trajectory that happened.
"""

from __future__ import annotations

import contextlib
import sqlite3
import threading
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, TypeVar

from slmeval.errors import EnvironmentError_, SlmevalError
from slmeval.storage import codec
from slmeval.storage.blobs import SWEEP_GRACE, BlobStore, MissingBlob, SweepResult
from slmeval.storage.migrations import SCHEMA_VERSION, migrate
from slmeval.storage.migrations import applied as applied_migrations
from slmeval.storage.models import (
    EMPTY_PAYLOAD,
    INLINE_LIMIT,
    TERMINAL_STATUSES,
    CandidateRecord,
    CaseRollupRow,
    EvaluationRow,
    ExecStatus,
    ExecutionDetail,
    ExecutionRecord,
    NodeTraceRow,
    Outcome,
    Payload,
    RunDetail,
    RunRecord,
    RunStatus,
    StoreStats,
    ToolTraceRow,
)

DB_NAME: Final = "runs.sqlite"

RUN_COLUMNS: Final = (
    "run_id",
    "name",
    "started_at",
    "finished_at",
    "config_json",
    "config_hash",
    "dataset_hash",
    "dataset_path",
    "slmeval_version",
    "git_sha",
    "host_info",
    "mode",
    "seed",
    "fault_rate",
    "runs_per_case",
    "status",
    "error_message",
)

CANDIDATE_COLUMNS: Final = (
    "run_id",
    "candidate_id",
    "role",
    "ordinal",
    "provider",
    "model",
    "model_cfg_json",
)

EXEC_COLUMNS: Final = (
    "exec_id",
    "run_id",
    "candidate_id",
    "case_id",
    "repeat_idx",
    "attempt",
    "status",
    "outcome_class",
    "started_at",
    "duration_ms",
    "steps",
    "total_prompt_tokens",
    "total_completion_tokens",
    "tokens_estimated",
    "cost_usd",
    "fault_vector_json",
    "fault_digest",
    "trajectory_json",
    "warnings_json",
    "output_json",
    "output_hash",
    "state_json",
    "state_hash",
    "error_class",
    "error_message",
)

NODE_COLUMNS: Final = (
    "exec_id",
    "seq",
    "node_name",
    "node_path",
    "model_id",
    "provider",
    "model",
    "latency_ms",
    "ttft_ms",
    "retries",
    "retry_ms",
    "prompt_tokens",
    "completion_tokens",
    "tokens_estimated",
    "schema_name",
    "schema_state",
    "structured_requested",
    "structured_mode",
    "rungs_tried",
    "repair_attempts",
    "json_stage",
    "error_class",
    "error_message",
    "raw_json",
    "raw_hash",
)

TOOL_COLUMNS: Final = (
    "exec_id",
    "seq",
    "tool_name",
    "node_name",
    "node_path",
    "args_json",
    "args_hash",
    "cassette_key",
    "outcome",
    "match_tier",
    "cache_hit",
    "fault",
    "fault_injected",
    "miss_verdict",
    "tool_version",
    "latency_ms",
    "output_json",
    "output_hash",
    "error_message",
)

EVAL_COLUMNS: Final = ("exec_id", "tier", "metric", "value_num", "value_text", "detail_json")

ROLLUP_COLUMNS: Final = (
    "run_id",
    "candidate_id",
    "case_id",
    "n",
    "c",
    "pass_k",
    "pass_hat_k",
    "wilson_lo",
    "wilson_hi",
)

_BLOB_REFERENCES: Final = (
    ("executions", "output_hash"),
    ("executions", "state_hash"),
    ("node_traces", "raw_hash"),
    ("tool_traces", "args_hash"),
    ("tool_traces", "output_hash"),
)
"""Every column that can hold a blob digest.

The sweep reads all of them, and a new blob-backed column added without being
listed here is a data-loss bug: its payloads would look unreferenced and the
next prune would delete them. Hence one table, not a query per call site.
"""

_E = TypeVar("_E", RunStatus, ExecStatus, Outcome)


class RunNotFound(SlmevalError):
    """No run with that id, or an ambiguous id fragment."""


def make_exec_id(run_id: str, candidate_id: str, case_id: str, repeat_idx: int) -> str:
    """The deterministic id for one matrix cell.

    Derived rather than random, and that is what makes ``--resume`` safe: a
    resumed run recomputes the same id for a unit it already finished, so a
    double-write replaces a row instead of inserting a second one. A random id
    would double that candidate's ``n`` and halve its pass rate - a regression
    the harness invented, which is the worst class of bug we can ship.

    It is also readable, which matters because users type it:
    ``20260130-114200-a1b2c3.gemma3-4b.refund_207.r2``.
    """
    return f"{run_id}.{candidate_id}.{case_id}.r{repeat_idx}"


def _enum(kind: type[_E], value: object, default: _E) -> _E:
    """Read an enum column without trusting it.

    A file written by a newer build can hold an ``outcome_class`` this build has
    never heard of. Falling back beats raising: the rest of the run is still
    readable, and a report that renders with one unfamiliar label is worth more
    than a crash.
    """
    try:
        return kind(str(value))
    except ValueError:
        return default


def _bool(value: object) -> bool:
    return bool(value)


class RunStore:
    """Read and write runs for one project."""

    def __init__(
        self,
        path: Path | str,
        *,
        blobs: BlobStore | None = None,
        inline_limit: int = INLINE_LIMIT,
    ) -> None:
        self.path = Path(path)
        self.blobs = blobs or BlobStore(self.path.parent / "blobs")
        self.inline_limit = max(0, inline_limit)
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def for_project(cls, root: Path | str, **kwargs: Any) -> RunStore:
        """The store under a project's ``.slmeval/`` directory."""
        cache = Path(root) / ".slmeval"
        cache.mkdir(parents=True, exist_ok=True)
        return cls(cache / DB_NAME, **kwargs)

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                self.path,
                timeout=5.0,
                # The store is closed at the end of a run, when the workers
                # that opened these connections may already be gone.
                check_same_thread=False,
                # Autocommit: every transaction below is explicit, which beats
                # a driver deciding to hold a write lock on our behalf.
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise EnvironmentError_(
                f"cannot open the trace store at {self.path}: {exc}",
                hint="run `slmeval doctor` to check the cache directory is writable",
            ) from exc

        conn.row_factory = sqlite3.Row
        # busy_timeout first, and the order is load-bearing: the statements
        # that most need it are the journal_mode switch and the migration DDL
        # that follow.
        conn.execute("PRAGMA busy_timeout=5000")
        # WAL so a `cache stats` or a report in another terminal never blocks
        # a run that is writing.
        conn.execute("PRAGMA journal_mode=WAL")
        # NORMAL, not FULL. Losing the last execution of a run to a power cut
        # costs one re-run; fsyncing every one of ~40 writes per execution
        # costs a measurable share of a benchmark's wall clock.
        conn.execute("PRAGMA synchronous=NORMAL")
        # Off by default in SQLite, per connection, and the cascades that make
        # `cache prune` a one-line DELETE depend on it. Must be set outside a
        # transaction, which is why it is here and not in migrate().
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            migrate(conn, path=self.path)
        except BaseException:
            conn.close()
            raise
        return conn

    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("this RunStore is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # Serialized, and not out of caution. Switching journal_mode takes
            # a brief exclusive lock for which SQLite does *not* run the busy
            # handler, so busy_timeout cannot save two workers that open the
            # store at the same instant. Opening happens once per thread.
            with self._conn_lock:
                conn = self._open()
                self._connections.append(conn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close every connection this store opened, from any thread.

        Not optional on Windows, where an open handle keeps the file locked and
        the user cannot delete their own ``.slmeval`` directory.
        """
        self._closed = True
        with self._conn_lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._local = threading.local()

    def __enter__(self) -> RunStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def schema_version(self) -> int:
        return SCHEMA_VERSION

    def migration_history(self) -> list[tuple[int, str, str, str]]:
        """``(version, name, applied_at, slmeval_version)``, oldest first.

        The first thing to ask for when a user reports that a report renders
        oddly: which migrations this file has seen, and which build applied
        them.
        """
        return applied_migrations(self._conn())

    # --------------------------------------------------------------- payloads

    def payload(self, text: str | None) -> Payload:
        """Decide where a serialized payload lives, and put it there."""
        if text is None:
            return EMPTY_PAYLOAD
        if len(text.encode("utf-8")) <= self.inline_limit:
            return Payload(inline=text)
        return Payload(digest=self.blobs.put_text(text))

    def payload_of(self, value: Any) -> Payload:
        """Serialize a value through the shared codec and store it."""
        return self.payload(codec.dumps(value))

    def load(self, payload: Payload) -> str | None:
        """Resolve a payload to its text, or ``None`` if nothing was stored.

        A payload whose blob has been pruned resolves to ``None`` rather than
        raising: the trace row is intact and the timeline around it still has
        to print. Callers that need to tell "nothing recorded" from "payload
        gone" check :attr:`Payload.present` first.
        """
        if payload.inline is not None:
            return payload.inline
        if payload.digest is None:
            return None
        try:
            return self.blobs.get_text(payload.digest)
        except MissingBlob:
            return None

    def load_json(self, payload: Payload) -> Any:
        return codec.loads(self.load(payload))

    # ------------------------------------------------------------ writing runs

    def save_run(self, run: RunRecord) -> None:
        """Insert or replace a run header. Called once at the start, and again
        by :meth:`finish_run`."""
        self._write(
            "runs",
            RUN_COLUMNS,
            [
                (
                    run.run_id,
                    run.name,
                    run.started_at,
                    run.finished_at,
                    run.config_json,
                    run.config_hash,
                    run.dataset_hash,
                    run.dataset_path,
                    run.slmeval_version,
                    run.git_sha,
                    run.host_info,
                    run.mode,
                    run.seed,
                    run.fault_rate,
                    run.runs_per_case,
                    run.status.value,
                    run.error_message,
                )
            ],
        )

    def finish_run(
        self,
        run_id: str,
        *,
        status: RunStatus = RunStatus.COMPLETED,
        error_message: str | None = None,
        finished_at: str | None = None,
    ) -> None:
        """Stamp a run as over.

        An UPDATE, not a replace of the header: the provenance written at the
        start is what the run actually used, and rewriting it at the end from
        whatever the config says now would quietly re-describe history.
        """
        conn = self._conn()
        conn.execute(
            "UPDATE runs SET status = ?, finished_at = ?, error_message = ? WHERE run_id = ?",
            (
                status.value,
                finished_at or datetime.now(UTC).isoformat(),
                error_message,
                run_id,
            ),
        )

    def save_candidates(self, candidates: Iterable[CandidateRecord]) -> None:
        self._write(
            "candidates",
            CANDIDATE_COLUMNS,
            [
                (
                    c.run_id,
                    c.candidate_id,
                    c.role,
                    c.ordinal,
                    c.provider,
                    c.model,
                    c.model_cfg_json,
                )
                for c in candidates
            ],
        )

    def save_execution(
        self,
        execution: ExecutionRecord,
        *,
        node_traces: Sequence[NodeTraceRow] = (),
        tool_traces: Sequence[ToolTraceRow] = (),
        evaluations: Sequence[EvaluationRow] = (),
    ) -> None:
        """Write one execution and its whole trace tree in a single transaction.

        Atomicity is the point. A committed execution row with only some of its
        tool calls does not look like a crash - it looks like an agent that
        stopped early, which is a finding the harness would have invented.

        The execution row goes first because the trace tables reference it. On
        a retry, ``INSERT OR REPLACE`` deletes the previous row, and with
        ``foreign_keys=ON`` that cascades away the previous attempt's traces
        before the new ones are inserted - so a re-run never leaves two
        trajectories interleaved under one id.
        """
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._insert(conn, "executions", EXEC_COLUMNS, [_exec_tuple(execution)])
            self._insert(
                conn, "node_traces", NODE_COLUMNS, [_node_tuple(r) for r in node_traces]
            )
            self._insert(
                conn, "tool_traces", TOOL_COLUMNS, [_tool_tuple(r) for r in tool_traces]
            )
            self._insert(
                conn, "evaluations", EVAL_COLUMNS, [_eval_tuple(r) for r in evaluations]
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    def save_evaluations(self, rows: Iterable[EvaluationRow]) -> None:
        """Write metrics for executions already stored - the evaluator pass."""
        self._write("evaluations", EVAL_COLUMNS, [_eval_tuple(r) for r in rows])

    def set_outcome(
        self,
        exec_id: str,
        outcome: Outcome,
        *,
        cost_usd: float | None = None,
    ) -> None:
        """Classify a stored execution.

        Evaluation happens after the fact, so the verdict is an update rather
        than part of the original write. ``cost_usd`` rides along because it is
        computed from the same token counts at the same moment.
        """
        conn = self._conn()
        if cost_usd is None:
            conn.execute(
                "UPDATE executions SET outcome_class = ? WHERE exec_id = ?",
                (outcome.value, exec_id),
            )
        else:
            conn.execute(
                "UPDATE executions SET outcome_class = ?, cost_usd = ? WHERE exec_id = ?",
                (outcome.value, cost_usd, exec_id),
            )

    def save_rollups(self, rows: Iterable[CaseRollupRow]) -> None:
        self._write(
            "case_rollups",
            ROLLUP_COLUMNS,
            [
                (
                    r.run_id,
                    r.candidate_id,
                    r.case_id,
                    r.n,
                    r.c,
                    r.pass_k,
                    r.pass_hat_k,
                    r.wilson_lo,
                    r.wilson_hi,
                )
                for r in rows
            ],
        )

    def _write(
        self,
        table: str,
        columns: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        if not rows:
            return
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            self._insert(conn, table, columns, rows)
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")

    @staticmethod
    def _insert(
        conn: sqlite3.Connection,
        table: str,
        columns: Sequence[str],
        rows: Sequence[tuple[Any, ...]],
    ) -> None:
        if not rows:
            return
        placeholders = ", ".join("?" * len(columns))
        conn.executemany(
            f"INSERT OR REPLACE INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
            rows,
        )

    # ------------------------------------------------------------ reading runs

    def get_run(self, run_id: str) -> RunRecord | None:
        row = self._conn().execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return _run_from_row(row) if row else None

    def require_run(self, run_id: str) -> RunRecord:
        run = self.get_run(run_id)
        if run is None:
            raise RunNotFound(
                f"no run {run_id!r} in {self.path}",
                hint="`slmeval cache stats` lists what this project has stored",
            )
        return run

    def list_runs(self, *, limit: int = 20, name: str | None = None) -> list[RunRecord]:
        """Most recent first - which is what every caller wants and what
        ``runs_by_start`` indexes."""
        sql = f"SELECT {', '.join(RUN_COLUMNS)} FROM runs"
        params: list[Any] = []
        if name:
            sql += " WHERE name = ?"
            params.append(name)
        sql += " ORDER BY started_at DESC LIMIT ?"
        params.append(limit)
        return [_run_from_row(r) for r in self._conn().execute(sql, params).fetchall()]

    def latest_run(self, *, name: str | None = None) -> RunRecord | None:
        runs = self.list_runs(limit=1, name=name)
        return runs[0] if runs else None

    def resolve_run_id(self, fragment: str) -> str:
        """Accept an unambiguous prefix of a run id.

        Run ids are long enough that nobody types them in full, and a
        shell-friendly prefix is the difference between a usable ``show``
        command and a copy-paste ritual. Ambiguity is refused rather than
        guessed - picking the newest match would eventually print a report for
        a run the user was not asking about.
        """
        if not fragment:
            raise RunNotFound("no run id given")
        rows = self._conn().execute(
            "SELECT run_id FROM runs WHERE run_id = ? OR run_id LIKE ? ORDER BY started_at DESC",
            (fragment, f"{fragment}%"),
        ).fetchall()
        exact = [str(r[0]) for r in rows if str(r[0]) == fragment]
        if exact:
            return exact[0]
        if not rows:
            raise RunNotFound(
                f"no run id starts with {fragment!r}",
                hint="`slmeval cache stats` lists stored runs",
            )
        if len(rows) > 1:
            ids = ", ".join(str(r[0]) for r in rows[:5])
            raise RunNotFound(
                f"{fragment!r} matches {len(rows)} runs: {ids}",
                hint="use more characters of the run id",
            )
        return str(rows[0][0])

    def candidates(self, run_id: str) -> list[CandidateRecord]:
        rows = self._conn().execute(
            f"SELECT {', '.join(CANDIDATE_COLUMNS)} FROM candidates "
            "WHERE run_id = ? ORDER BY ordinal, candidate_id",
            (run_id,),
        ).fetchall()
        return [_candidate_from_row(r) for r in rows]

    def executions(
        self,
        run_id: str,
        *,
        candidate_id: str | None = None,
        case_id: str | None = None,
    ) -> list[ExecutionRecord]:
        sql = f"SELECT {', '.join(EXEC_COLUMNS)} FROM executions WHERE run_id = ?"
        params: list[Any] = [run_id]
        if candidate_id is not None:
            sql += " AND candidate_id = ?"
            params.append(candidate_id)
        if case_id is not None:
            sql += " AND case_id = ?"
            params.append(case_id)
        sql += " ORDER BY candidate_id, case_id, repeat_idx"
        return [_exec_from_row(r) for r in self._conn().execute(sql, params).fetchall()]

    def get_execution(self, exec_id: str) -> ExecutionRecord | None:
        row = self._conn().execute(
            f"SELECT {', '.join(EXEC_COLUMNS)} FROM executions WHERE exec_id = ?", (exec_id,)
        ).fetchone()
        return _exec_from_row(row) if row else None

    def resolve_exec_id(self, fragment: str) -> str:
        """Accept a suffix or a full execution id.

        A suffix, because the readable part of an exec id is at the end -
        ``gemma3-4b.refund_207.r2`` is what a user reads off the failure table,
        and prefixing it with the run id again is friction for no gain.
        """
        if not fragment:
            raise RunNotFound("no execution id given")
        rows = self._conn().execute(
            "SELECT exec_id FROM executions WHERE exec_id = ? OR exec_id LIKE ? "
            "ORDER BY exec_id LIMIT 6",
            (fragment, f"%{fragment}"),
        ).fetchall()
        exact = [str(r[0]) for r in rows if str(r[0]) == fragment]
        if exact:
            return exact[0]
        if not rows:
            raise RunNotFound(f"no execution id matches {fragment!r}")
        if len(rows) > 1:
            ids = ", ".join(str(r[0]) for r in rows[:5])
            raise RunNotFound(
                f"{fragment!r} matches {len(rows)} executions: {ids}",
                hint="include the candidate and case, e.g. `gemma3-4b.refund_207.r0`",
            )
        return str(rows[0][0])

    def node_traces(self, exec_id: str) -> list[NodeTraceRow]:
        rows = self._conn().execute(
            f"SELECT {', '.join(NODE_COLUMNS)} FROM node_traces WHERE exec_id = ? ORDER BY seq",
            (exec_id,),
        ).fetchall()
        return [_node_from_row(r) for r in rows]

    def tool_traces(self, exec_id: str) -> list[ToolTraceRow]:
        rows = self._conn().execute(
            f"SELECT {', '.join(TOOL_COLUMNS)} FROM tool_traces WHERE exec_id = ? ORDER BY seq",
            (exec_id,),
        ).fetchall()
        return [_tool_from_row(r) for r in rows]

    def evaluations(self, exec_id: str) -> list[EvaluationRow]:
        rows = self._conn().execute(
            f"SELECT {', '.join(EVAL_COLUMNS)} FROM evaluations WHERE exec_id = ? "
            "ORDER BY tier, metric",
            (exec_id,),
        ).fetchall()
        return [_eval_from_row(r) for r in rows]

    def rollups(self, run_id: str) -> list[CaseRollupRow]:
        rows = self._conn().execute(
            f"SELECT {', '.join(ROLLUP_COLUMNS)} FROM case_rollups WHERE run_id = ? "
            "ORDER BY candidate_id, case_id",
            (run_id,),
        ).fetchall()
        return [_rollup_from_row(r) for r in rows]

    def completed_units(self, run_id: str) -> set[tuple[str, str, int]]:
        """``(candidate_id, case_id, repeat_idx)`` triples that need no re-run.

        What ``--resume`` subtracts from the planned matrix. Only terminal
        statuses count: a unit recorded as ``skipped`` was planned and never
        reached, so resuming has to pick it up rather than treat it as done.
        """
        wanted = ", ".join("?" * len(TERMINAL_STATUSES))
        rows = self._conn().execute(
            "SELECT candidate_id, case_id, repeat_idx FROM executions "
            f"WHERE run_id = ? AND status IN ({wanted})",
            (run_id, *sorted(s.value for s in TERMINAL_STATUSES)),
        ).fetchall()
        return {(str(r[0]), str(r[1]), int(r[2])) for r in rows}

    # ----------------------------------------------------------- reconstruction

    def reconstruct(self, exec_id: str) -> ExecutionDetail:
        """One execution, its timeline and its payloads - from rows alone.

        This method is step 12's acceptance test. ``slmeval show exec <id>``
        renders what it returns, and nothing in it comes from a live object: if
        this is complete, a run survives the process that produced it.
        """
        execution = self.get_execution(exec_id)
        if execution is None:
            raise RunNotFound(
                f"no execution {exec_id!r} in {self.path}",
                hint="`slmeval show run <run-id>` lists the executions of a run",
            )
        tools = tuple(self.tool_traces(exec_id))
        return ExecutionDetail(
            execution=execution,
            node_traces=tuple(self.node_traces(exec_id)),
            tool_traces=tools,
            evaluations=tuple(self.evaluations(exec_id)),
            output=self.load_json(execution.output),
            state=self.load_json(execution.state),
            tool_outputs={t.seq: self.load_json(t.output) for t in tools},
            tool_arguments={t.seq: self.load_json(t.args) for t in tools},
        )

    def reconstruct_run(self, run_id: str) -> RunDetail:
        """A whole run's header, candidates, cells and rollups.

        Without the per-execution traces: a 1,500-execution matrix's full
        timeline is tens of megabytes, and the scorecard needs none of it. The
        drill-down asks for one execution at a time through
        :meth:`reconstruct`.
        """
        run = self.require_run(run_id)
        return RunDetail(
            run=run,
            candidates=tuple(self.candidates(run_id)),
            executions=tuple(self.executions(run_id)),
            rollups=tuple(self.rollups(run_id)),
        )

    # ------------------------------------------------------------- maintenance

    def referenced_blobs(self) -> set[str]:
        """Every blob digest any row points at, across the whole store.

        Whole store, never one run: a payload shared by three runs stays live
        until the last of them is deleted, and that sharing is the reason the
        blob store exists. A partial set passed to the sweep deletes live data.
        """
        conn = self._conn()
        out: set[str] = set()
        for table, column in _BLOB_REFERENCES:
            rows = conn.execute(
                f"SELECT DISTINCT {column} FROM {table} WHERE {column} IS NOT NULL"
            ).fetchall()
            out.update(str(r[0]) for r in rows if r[0])
        return out

    def delete_runs(self, run_ids: Iterable[str]) -> int:
        """Delete runs and everything under them.

        One DELETE per run; the cascades take the candidates, executions,
        traces, evaluations and rollups. Blobs are not touched here - they are
        shared, so freeing them is a sweep, not a delete.
        """
        ids = list(run_ids)
        if not ids:
            return 0
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany("DELETE FROM runs WHERE run_id = ?", [(i,) for i in ids])
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        return len(ids)

    def runs_older_than(self, cutoff: datetime, *, keep_last: int = 0) -> list[RunRecord]:
        """Candidates for pruning, oldest first.

        ``keep_last`` is applied after the age filter and over the whole store,
        so ``--older-than 30d --keep-last 5`` cannot leave a user with no
        history at all. Someone who has not benchmarked in two months still
        wants their last run when they come back - that is the one the next run
        gets compared against.
        """
        stamp = cutoff.isoformat()
        rows = self._conn().execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM runs "
            "WHERE started_at < ? ORDER BY started_at ASC",
            (stamp,),
        ).fetchall()
        stale = [_run_from_row(r) for r in rows]
        if keep_last <= 0:
            return stale
        newest = {r.run_id for r in self.list_runs(limit=keep_last)}
        return [r for r in stale if r.run_id not in newest]

    def prune(
        self,
        *,
        older_than: timedelta,
        keep_last: int = 0,
        dry_run: bool = False,
        force: bool = False,
        grace: timedelta = SWEEP_GRACE,
        now: datetime | None = None,
    ) -> tuple[list[RunRecord], SweepResult]:
        """Delete old runs, then collect the payloads nothing references.

        Refuses while a run is in flight unless ``force``. Sweeping during a
        run means racing a process that is still writing rows, and the grace
        period only narrows that window rather than closing it. A stale
        ``running`` row left by a crash blocks this too, which is the right
        trade: a clear refusal with a way out beats silently deleting a payload
        a live run needs.
        """
        running = self.running_runs()
        if running and not dry_run and not force:
            ids = ", ".join(r.run_id for r in running[:3])
            raise EnvironmentError_(
                f"{len(running)} run(s) are still in progress: {ids}",
                hint="wait for them to finish, or `slmeval cache prune --force` "
                "if they were left behind by a crash",
            )
        cutoff = (now or datetime.now(UTC)) - older_than
        stale = self.runs_older_than(cutoff, keep_last=keep_last)
        if not dry_run and stale:
            self.delete_runs(r.run_id for r in stale)
        swept = self.blobs.sweep(self.referenced_blobs(), grace=grace, dry_run=dry_run)
        if not dry_run and stale:
            self.vacuum()
        return stale, swept

    def running_runs(self) -> list[RunRecord]:
        rows = self._conn().execute(
            f"SELECT {', '.join(RUN_COLUMNS)} FROM runs WHERE status = ? ORDER BY started_at",
            (RunStatus.RUNNING.value,),
        ).fetchall()
        return [_run_from_row(r) for r in rows]

    def vacuum(self) -> None:
        """Return freed pages to the filesystem.

        Deleting rows leaves the file the same size, and a user who just pruned
        a gigabyte of traces to reclaim disk will check. Must run outside a
        transaction, which autocommit gives us.
        """
        with contextlib.suppress(sqlite3.Error):
            self._conn().execute("VACUUM")

    def stats(self) -> StoreStats:
        conn = self._conn()
        runs, running = conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(status = ?), 0) FROM runs",
            (RunStatus.RUNNING.value,),
        ).fetchone()
        counts: dict[str, int] = {}
        for table in ("executions", "node_traces", "tool_traces", "evaluations"):
            counts[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        span = conn.execute("SELECT MIN(started_at), MAX(started_at) FROM runs").fetchone()
        blobs = self.blobs.stats()
        return StoreStats(
            runs=int(runs),
            running=int(running),
            executions=counts["executions"],
            node_traces=counts["node_traces"],
            tool_traces=counts["tool_traces"],
            evaluations=counts["evaluations"],
            db_bytes=self._db_bytes(),
            blobs=blobs.count,
            blob_bytes=blobs.size_bytes,
            oldest_run=span[0],
            newest_run=span[1],
        )

    def _db_bytes(self) -> int:
        """On-disk size, WAL included.

        The write-ahead log can be a large fraction of the total during and
        after a run, so reporting only the main file would tell a user looking
        for their disk space a number that does not match what their file
        manager shows.
        """
        total = 0
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(self.path) + suffix)
            with contextlib.suppress(OSError):
                if candidate.exists():
                    total += candidate.stat().st_size
        return total


# --------------------------------------------------------------- row mapping


def _exec_tuple(e: ExecutionRecord) -> tuple[Any, ...]:
    return (
        e.exec_id,
        e.run_id,
        e.candidate_id,
        e.case_id,
        e.repeat_idx,
        e.attempt,
        e.status.value,
        e.outcome_class.value,
        e.started_at,
        e.duration_ms,
        e.steps,
        e.total_prompt_tokens,
        e.total_completion_tokens,
        int(e.tokens_estimated),
        e.cost_usd,
        e.fault_vector_json,
        e.fault_digest,
        e.trajectory_json,
        e.warnings_json,
        e.output.inline,
        e.output.digest,
        e.state.inline,
        e.state.digest,
        e.error_class,
        e.error_message,
    )


def _node_tuple(r: NodeTraceRow) -> tuple[Any, ...]:
    return (
        r.exec_id,
        r.seq,
        r.node_name,
        r.node_path,
        r.model_id,
        r.provider,
        r.model,
        r.latency_ms,
        r.ttft_ms,
        r.retries,
        r.retry_ms,
        r.prompt_tokens,
        r.completion_tokens,
        int(r.tokens_estimated),
        r.schema_name,
        r.schema_state,
        r.structured_requested,
        r.structured_mode,
        r.rungs_tried,
        r.repair_attempts,
        r.json_stage,
        r.error_class,
        r.error_message,
        r.raw.inline,
        r.raw.digest,
    )


def _tool_tuple(r: ToolTraceRow) -> tuple[Any, ...]:
    return (
        r.exec_id,
        r.seq,
        r.tool_name,
        r.node_name,
        r.node_path,
        r.args.inline,
        r.args.digest,
        r.cassette_key,
        r.outcome,
        r.match_tier,
        int(r.cache_hit),
        r.fault,
        int(r.fault_injected),
        r.miss_verdict,
        r.tool_version,
        r.latency_ms,
        r.output.inline,
        r.output.digest,
        r.error_message,
    )


def _eval_tuple(r: EvaluationRow) -> tuple[Any, ...]:
    return (r.exec_id, r.tier, r.metric, r.value_num, r.value_text, r.detail_json)


def _run_from_row(row: Mapping[str, Any] | sqlite3.Row) -> RunRecord:
    return RunRecord(
        run_id=str(row["run_id"]),
        name=str(row["name"]),
        started_at=str(row["started_at"]),
        finished_at=row["finished_at"],
        config_json=str(row["config_json"]),
        config_hash=str(row["config_hash"]),
        dataset_hash=str(row["dataset_hash"]),
        dataset_path=str(row["dataset_path"]),
        slmeval_version=str(row["slmeval_version"]),
        git_sha=str(row["git_sha"]),
        host_info=str(row["host_info"]),
        mode=str(row["mode"]),
        seed=int(row["seed"]),
        fault_rate=float(row["fault_rate"]),
        runs_per_case=int(row["runs_per_case"]),
        status=_enum(RunStatus, row["status"], RunStatus.COMPLETED),
        error_message=row["error_message"],
    )


def _candidate_from_row(row: sqlite3.Row) -> CandidateRecord:
    return CandidateRecord(
        run_id=str(row["run_id"]),
        candidate_id=str(row["candidate_id"]),
        model_cfg_json=str(row["model_cfg_json"]),
        role=str(row["role"]),
        ordinal=int(row["ordinal"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
    )


def _exec_from_row(row: sqlite3.Row) -> ExecutionRecord:
    return ExecutionRecord(
        exec_id=str(row["exec_id"]),
        run_id=str(row["run_id"]),
        candidate_id=str(row["candidate_id"]),
        case_id=str(row["case_id"]),
        repeat_idx=int(row["repeat_idx"]),
        status=_enum(ExecStatus, row["status"], ExecStatus.ERROR),
        outcome_class=_enum(Outcome, row["outcome_class"], Outcome.UNCLASSIFIED),
        attempt=int(row["attempt"]),
        started_at=str(row["started_at"]),
        duration_ms=float(row["duration_ms"]),
        steps=int(row["steps"]),
        total_prompt_tokens=row["total_prompt_tokens"],
        total_completion_tokens=row["total_completion_tokens"],
        tokens_estimated=_bool(row["tokens_estimated"]),
        cost_usd=row["cost_usd"],
        fault_vector_json=str(row["fault_vector_json"]),
        fault_digest=str(row["fault_digest"]),
        trajectory_json=str(row["trajectory_json"]),
        warnings_json=str(row["warnings_json"]),
        output=Payload(inline=row["output_json"], digest=row["output_hash"]),
        state=Payload(inline=row["state_json"], digest=row["state_hash"]),
        error_class=row["error_class"],
        error_message=row["error_message"],
    )


def _node_from_row(row: sqlite3.Row) -> NodeTraceRow:
    return NodeTraceRow(
        exec_id=str(row["exec_id"]),
        seq=int(row["seq"]),
        node_name=str(row["node_name"]),
        node_path=str(row["node_path"]),
        model_id=str(row["model_id"]),
        provider=str(row["provider"]),
        model=str(row["model"]),
        latency_ms=float(row["latency_ms"]),
        ttft_ms=row["ttft_ms"],
        retries=int(row["retries"]),
        retry_ms=float(row["retry_ms"]),
        prompt_tokens=row["prompt_tokens"],
        completion_tokens=row["completion_tokens"],
        tokens_estimated=_bool(row["tokens_estimated"]),
        schema_name=row["schema_name"],
        schema_state=str(row["schema_state"]),
        structured_requested=str(row["structured_requested"]),
        structured_mode=row["structured_mode"],
        rungs_tried=str(row["rungs_tried"]),
        repair_attempts=int(row["repair_attempts"]),
        json_stage=row["json_stage"],
        error_class=row["error_class"],
        error_message=row["error_message"],
        raw=Payload(inline=row["raw_json"], digest=row["raw_hash"]),
    )


def _tool_from_row(row: sqlite3.Row) -> ToolTraceRow:
    return ToolTraceRow(
        exec_id=str(row["exec_id"]),
        seq=int(row["seq"]),
        tool_name=str(row["tool_name"]),
        node_name=str(row["node_name"]),
        node_path=str(row["node_path"]),
        args=Payload(inline=row["args_json"], digest=row["args_hash"]),
        cassette_key=str(row["cassette_key"]),
        outcome=str(row["outcome"]),
        match_tier=row["match_tier"],
        cache_hit=_bool(row["cache_hit"]),
        fault=row["fault"],
        fault_injected=_bool(row["fault_injected"]),
        miss_verdict=row["miss_verdict"],
        tool_version=str(row["tool_version"]),
        latency_ms=float(row["latency_ms"]),
        output=Payload(inline=row["output_json"], digest=row["output_hash"]),
        error_message=row["error_message"],
    )


def _eval_from_row(row: sqlite3.Row) -> EvaluationRow:
    return EvaluationRow(
        exec_id=str(row["exec_id"]),
        tier=int(row["tier"]),
        metric=str(row["metric"]),
        value_num=row["value_num"],
        value_text=row["value_text"],
        detail_json=row["detail_json"],
    )


def _rollup_from_row(row: sqlite3.Row) -> CaseRollupRow:
    return CaseRollupRow(
        run_id=str(row["run_id"]),
        candidate_id=str(row["candidate_id"]),
        case_id=str(row["case_id"]),
        n=int(row["n"]),
        c=int(row["c"]),
        pass_k=row["pass_k"],
        pass_hat_k=row["pass_hat_k"],
        wilson_lo=row["wilson_lo"],
        wilson_hi=row["wilson_hi"],
    )


__all__ = [
    "DB_NAME",
    "RunNotFound",
    "RunStore",
    "make_exec_id",
]
