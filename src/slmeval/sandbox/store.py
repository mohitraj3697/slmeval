"""The cassette store: SQLite on disk, an LRU in front of it.

Replay is the feature this table makes possible - 200 trajectories against four
models with no GPU, no API keys and no network - so its cost per tool call is
not an implementation detail. At 10 tool calls per execution and 90 executions,
a millisecond per lookup is a second of wall clock spent on bookkeeping, and the
headline claim is that replay is free.

Two things that look fine and are not:

**``with sqlite3.connect(...) as conn:`` commits but does not close.** It is a
transaction context manager, not a connection one. Opening per tool call leaks a
handle per call, and on Windows that becomes ``OSError: too many open files`` or
a locked database partway through a long run - after the expensive part, which
is the worst possible time. So: one connection per thread, opened once, closed
explicitly, with every pragma set at open.

**Opening a connection per ``get()`` is itself the dominant cost of a "0ms"
replay.** Connection setup dwarfs an indexed primary-key lookup. The LRU in
front means the second and later replays of a case never touch SQLite at all,
which is what turns 200 replays into something that finishes while you watch.

There is deliberately **no negative cache**. Caching misses would save a couple
of microseconds and introduce an invalidation bug whose symptom is a phantom
``CASSETTE_MISS`` - the harness reporting a hallucination it invented, which is
the one failure that would cost us a user's trust outright. Not a trade worth
making for 2µs.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import threading
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from slmeval.errors import EnvironmentError_
from slmeval.sandbox import envelope
from slmeval.sandbox.keys import CassetteKeys, MatchTier

SCHEMA_VERSION: Final = 1

DEFAULT_LRU_SIZE: Final = 4096
"""Roughly a full matrix's worth of distinct tool calls. A cassette payload is
small; holding a few thousand costs a few megabytes and removes SQLite from the
hot path entirely."""

_DDL: Final = (
    """
    CREATE TABLE IF NOT EXISTS cassettes (
        key               TEXT PRIMARY KEY,
        case_id           TEXT NOT NULL,
        tool_name         TEXT NOT NULL,
        match_tier        TEXT NOT NULL,
        canonical_args    TEXT NOT NULL,
        args_json         TEXT NOT NULL,
        response_envelope TEXT NOT NULL,
        recorded_at       TEXT NOT NULL,
        dataset_hash      TEXT NOT NULL DEFAULT '',
        tool_version      TEXT NOT NULL DEFAULT ''
    )
    """,
    # The miss diagnosis asks "what did we record for this tool, in this case?"
    # so it can show the nearest recorded arguments. Without this index that
    # question is a table scan, and it runs on every failure.
    "CREATE INDEX IF NOT EXISTS cassettes_by_call ON cassettes (case_id, tool_name)",
)

_COLUMNS: Final = (
    "key",
    "case_id",
    "tool_name",
    "match_tier",
    "canonical_args",
    "args_json",
    "response_envelope",
    "recorded_at",
    "dataset_hash",
    "tool_version",
)


@dataclass(frozen=True, slots=True)
class Cassette:
    """One recorded tool response, as stored.

    ``match_tier`` is the tier this row answers, and it is authoritative rather
    than inferred: the tier is part of the key's hash, so a row found by a
    subset-tier key can never be a row that was filed at the exact tier.
    """

    key: str
    case_id: str
    tool_name: str
    match_tier: MatchTier
    canonical_args: str
    args_json: str
    response_envelope: str
    recorded_at: str
    dataset_hash: str = ""
    tool_version: str = ""

    def arguments(self) -> Any:
        """The arguments as originally recorded - what a diff is built against."""
        return envelope.decode_json(self.args_json)

    def response(self) -> Any:
        """The recorded return value, rebuilt with its original types."""
        return envelope.decode_json(self.response_envelope)


@dataclass(frozen=True, slots=True)
class Hit:
    """A successful lookup, and the tier that produced it."""

    cassette: Cassette
    response: Any

    @property
    def tier(self) -> MatchTier:
        return self.cassette.match_tier

    @property
    def degraded(self) -> bool:
        """True when the call did not match exactly.

        A normalized hit means our normalizers covered a formatting difference;
        a subset hit means the model reached the right call with something else
        wrong. Both are reportable, so neither is allowed to look like a clean
        pass.
        """
        return self.cassette.match_tier is not MatchTier.EXACT


@dataclass(frozen=True, slots=True)
class StoreStats:
    cassettes: int
    cases: int
    tools: int
    size_bytes: int
    by_tier: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class _Cached:
    """An LRU entry: the row, plus its envelope parsed once.

    The *parsed* envelope is cached, never the decoded response. Handing the
    same object to 200 replays would let the first execution mutate what the
    other 199 see - a cross-contamination that looks exactly like
    nondeterminism in the model. Decoding per call rebuilds fresh objects and
    costs microseconds.
    """

    cassette: Cassette
    parsed: Any


class CassetteStore:
    """Recorded tool responses for one project.

    Thread-safe and usable from several event-loop threads: each thread gets its
    own SQLite connection, and the LRU is guarded by a lock.
    """

    def __init__(self, path: Path | str, *, lru_size: int = DEFAULT_LRU_SIZE) -> None:
        self.path = Path(path)
        self._lru_size = max(0, lru_size)
        self._cache: OrderedDict[str, _Cached] = OrderedDict()
        self._cache_lock = threading.Lock()
        self._local = threading.local()
        self._connections: list[sqlite3.Connection] = []
        self._conn_lock = threading.Lock()
        self._closed = False
        self.hits = 0
        self.misses = 0
        self.cache_hits = 0

    # ------------------------------------------------------------- lifecycle

    @classmethod
    def for_project(cls, root: Path | str, **kwargs: Any) -> CassetteStore:
        """The store under a project's ``.slmeval/`` cache directory."""
        cache = Path(root) / ".slmeval"
        cache.mkdir(parents=True, exist_ok=True)
        return cls(cache / "cassettes.sqlite", **kwargs)

    def _open(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(
                self.path,
                timeout=5.0,
                # The store is closed at the end of a run, when the workers that
                # opened these connections may already be gone. Without this,
                # closing another thread's connection raises ProgrammingError
                # and the handle leaks anyway.
                check_same_thread=False,
                # Autocommit. Every pragma and every multi-row write below is
                # explicit about its transaction, which beats a driver deciding
                # to hold a write lock on our behalf.
                isolation_level=None,
            )
        except sqlite3.Error as exc:
            raise EnvironmentError_(
                f"cannot open the cassette store at {self.path}: {exc}",
                hint="run `slmeval doctor` to check the cache directory is writable",
            ) from exc

        conn.row_factory = sqlite3.Row
        # busy_timeout FIRST, and the order is load-bearing. Switching
        # journal_mode needs a brief exclusive lock, and `CREATE TABLE` below
        # needs a write lock; with the default timeout of 0 a second worker
        # opening the store at the same moment gets `database is locked`
        # immediately instead of waiting. Setting the timeout after those
        # statements protects everything except the two that actually contend.
        conn.execute("PRAGMA busy_timeout=5000")
        # WAL so readers never block the recorder. NORMAL because losing a
        # cassette to a power cut costs one re-record, not correctness.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema(conn)
        return conn

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        version = int(conn.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            conn.close()
            raise EnvironmentError_(
                f"{self.path} was written by a newer slmeval "
                f"(store schema v{version}, this build reads v{SCHEMA_VERSION})",
                hint="upgrade slmeval, or delete the .slmeval cache to re-record",
            )
        for statement in _DDL:
            conn.execute(statement)
        if version != SCHEMA_VERSION:
            conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")

    def _conn(self) -> sqlite3.Connection:
        if self._closed:
            raise RuntimeError("this CassetteStore is closed")
        conn: sqlite3.Connection | None = getattr(self._local, "conn", None)
        if conn is None:
            # Opening is serialized, and that is not belt-and-braces. Switching
            # journal_mode takes a brief exclusive lock, and SQLite does *not*
            # run the busy handler for it - so `busy_timeout` cannot save two
            # workers that open the store at the same instant, and one of them
            # gets `database is locked` straight away. Every later statement is
            # covered by the timeout; this one has to be covered by us.
            # Opening happens once per thread, so the lock costs nothing.
            with self._conn_lock:
                conn = self._open()
                self._connections.append(conn)
            self._local.conn = conn
        return conn

    def close(self) -> None:
        """Close every connection this store opened, from any thread.

        Explicit, and not optional: on Windows an unclosed handle keeps the
        database file locked, so a test's temporary directory cannot be removed
        and a user cannot delete their own cache.
        """
        self._closed = True
        with self._conn_lock:
            connections, self._connections = self._connections, []
        for conn in connections:
            # A connection that is already gone is not a reason to leave the
            # rest of them open.
            with contextlib.suppress(sqlite3.Error):
                conn.close()
        self._local = threading.local()
        with self._cache_lock:
            self._cache.clear()

    def __enter__(self) -> CassetteStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------- reading

    def _cache_get(self, key: str) -> _Cached | None:
        if not self._lru_size:
            return None
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
            return entry

    def _cache_put(self, key: str, entry: _Cached) -> None:
        if not self._lru_size:
            return
        with self._cache_lock:
            self._cache[key] = entry
            self._cache.move_to_end(key)
            while len(self._cache) > self._lru_size:
                self._cache.popitem(last=False)

    def _fetch(self, key: str) -> _Cached | None:
        cached = self._cache_get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached
        row = self._conn().execute(
            f"SELECT {', '.join(_COLUMNS)} FROM cassettes WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        cassette = _from_row(row)
        entry = _Cached(cassette=cassette, parsed=json.loads(cassette.response_envelope))
        self._cache_put(key, entry)
        return entry

    def get(self, key: str) -> Cassette | None:
        entry = self._fetch(key)
        return entry.cassette if entry else None

    def lookup(self, keys: CassetteKeys) -> Hit | None:
        """Walk the cascade: exact, then normalized, then subset.

        The first hit wins, and its tier is the finding. Trying them in order -
        rather than matching fuzzily and reporting a confidence - is what makes
        "the model got an extra argument wrong" a distinct, reportable outcome
        instead of a shrug.
        """
        for entry in keys:
            found = self._fetch(entry.key)
            if found is not None:
                self.hits += 1
                return Hit(cassette=found.cassette, response=envelope.decode(found.parsed))
        self.misses += 1
        return None

    def neighbours(self, case_id: str, tool_name: str, *, limit: int = 50) -> list[Cassette]:
        """Every exact-tier cassette for one (case, tool).

        What the miss diagnosis searches for the nearest recorded arguments.
        Exact-tier only: the normalized and subset rows of the same call hold
        reduced argument forms, and showing a user a normalized rendering of
        their own recording as "nearest recorded" would be confusing.
        """
        rows = self._conn().execute(
            f"SELECT {', '.join(_COLUMNS)} FROM cassettes "
            "WHERE case_id = ? AND tool_name = ? AND match_tier = ? "
            "ORDER BY recorded_at LIMIT ?",
            (case_id, tool_name, MatchTier.EXACT.value, limit),
        ).fetchall()
        return [_from_row(row) for row in rows]

    def tool_names(self, case_id: str | None = None) -> list[str]:
        """Tools with at least one recording. Day 7 uses this to tell a
        hallucinated tool name from an unrecorded path."""
        if case_id is None:
            rows = self._conn().execute(
                "SELECT DISTINCT tool_name FROM cassettes ORDER BY tool_name"
            ).fetchall()
        else:
            rows = self._conn().execute(
                "SELECT DISTINCT tool_name FROM cassettes WHERE case_id = ? ORDER BY tool_name",
                (case_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    # ---------------------------------------------------------------- writing

    def record(
        self,
        keys: CassetteKeys,
        *,
        arguments: Mapping[str, Any],
        response: Any,
        tool_version: str = "",
        dataset_hash: str = "",
    ) -> Hit:
        """Store one real tool call under every key it can be found by.

        The response is encoded once and shared by all three rows, and the
        arguments are stored as the caller passed them so a later diff shows
        real values rather than a canonical rendering.

        Two calls that differ only outside ``match_on`` collapse to the same
        subset key, so the last recording wins *at that tier*. That is correct
        rather than unfortunate: the subset tier answers "did the model reach
        the right call", and its payload is an acknowledged stand-in.
        """
        what = f"tool {keys.tool_name!r} return value"
        response_envelope = envelope.encode_json(response, what=what)
        args_json = envelope.encode_json(
            dict(arguments), what=f"arguments to tool {keys.tool_name!r}"
        )
        recorded_at = datetime.now(UTC).isoformat()

        cassettes = [
            Cassette(
                key=entry.key,
                case_id=keys.case_id,
                tool_name=keys.tool_name,
                match_tier=entry.tier,
                canonical_args=entry.canonical_args,
                args_json=args_json,
                response_envelope=response_envelope,
                recorded_at=recorded_at,
                dataset_hash=dataset_hash,
                tool_version=tool_version,
            )
            for entry in keys
        ]
        self.put_many(cassettes)
        return Hit(cassette=cassettes[0], response=response)

    def put(self, cassette: Cassette) -> None:
        self.put_many([cassette])

    def put_many(self, cassettes: Iterable[Cassette]) -> None:
        """Write rows in one transaction, and drop their cache entries.

        Invalidating rather than updating the cache: a re-record whose payload
        differs must not be shadowed by what an earlier execution read.
        """
        rows = [
            (
                c.key,
                c.case_id,
                c.tool_name,
                c.match_tier.value,
                c.canonical_args,
                c.args_json,
                c.response_envelope,
                c.recorded_at,
                c.dataset_hash,
                c.tool_version,
            )
            for c in cassettes
        ]
        if not rows:
            return
        placeholders = ", ".join("?" * len(_COLUMNS))
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.executemany(
                f"INSERT OR REPLACE INTO cassettes ({', '.join(_COLUMNS)}) "
                f"VALUES ({placeholders})",
                rows,
            )
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        with self._cache_lock:
            for row in rows:
                self._cache.pop(str(row[0]), None)

    def delete_case(self, case_id: str) -> int:
        """Drop every cassette for one case - what ``record --force`` needs."""
        conn = self._conn()
        conn.execute("BEGIN IMMEDIATE")
        try:
            cursor = conn.execute("DELETE FROM cassettes WHERE case_id = ?", (case_id,))
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
        with self._cache_lock:
            self._cache.clear()
        return int(cursor.rowcount)

    # ------------------------------------------------------------------ stats

    def stats(self) -> StoreStats:
        conn = self._conn()
        total, cases, tools = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT case_id), COUNT(DISTINCT tool_name) FROM cassettes"
        ).fetchone()
        by_tier = {
            str(row[0]): int(row[1])
            for row in conn.execute(
                "SELECT match_tier, COUNT(*) FROM cassettes GROUP BY match_tier"
            ).fetchall()
        }
        size = self.path.stat().st_size if self.path.exists() else 0
        return StoreStats(
            cassettes=int(total),
            cases=int(cases),
            tools=int(tools),
            size_bytes=size,
            by_tier=by_tier,
        )

    def __len__(self) -> int:
        return int(self._conn().execute("SELECT COUNT(*) FROM cassettes").fetchone()[0])


def _from_row(row: sqlite3.Row) -> Cassette:
    return Cassette(
        key=row["key"],
        case_id=row["case_id"],
        tool_name=row["tool_name"],
        match_tier=MatchTier(row["match_tier"]),
        canonical_args=row["canonical_args"],
        args_json=row["args_json"],
        response_envelope=row["response_envelope"],
        recorded_at=row["recorded_at"],
        dataset_hash=row["dataset_hash"],
        tool_version=row["tool_version"],
    )


__all__ = [
    "DEFAULT_LRU_SIZE",
    "SCHEMA_VERSION",
    "Cassette",
    "CassetteStore",
    "Hit",
    "StoreStats",
]
