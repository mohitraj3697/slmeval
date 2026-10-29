"""The cassette store.

Most of these guard a specific way the store could be fast-but-wrong or
correct-but-leaky:

- a handle leak, which on Windows surfaces as a locked database partway through
  a long run rather than as an error you can read;
- a cached object shared between replays, which looks like model
  nondeterminism;
- a stale cache entry after a re-record, which looks like a cassette miss - the
  harness reporting a hallucination it invented.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from slmeval.errors import EnvironmentError_
from slmeval.sandbox.keys import MatchTier, declare
from slmeval.sandbox.store import CassetteStore

CASE = "status_shipped"


def fetch_order(order_id: str, request_id: str = "") -> dict:
    return {}


@pytest.fixture
def store(tmp_path: Path):
    with CassetteStore(tmp_path / "cassettes.sqlite") as store:
        yield store


@pytest.fixture
def policy():
    return declare(
        fetch_order,
        tool_name="fetch_order",
        match_on=["order_id"],
        normalize={"order_id": "upper_strip"},
        ignore_args=["request_id"],
    )


def record_one(store: CassetteStore, policy, args: dict, response: object):
    return store.record(policy.compute(CASE, args), arguments=args, response=response)


# --------------------------------------------------------------------------- #
# Round trip
# --------------------------------------------------------------------------- #


def test_a_recorded_call_replays(store: CassetteStore, policy) -> None:
    args = {"order_id": "ORD-9921", "request_id": "r1"}
    record_one(store, policy, args, {"state": "shipped"})

    hit = store.lookup(policy.compute(CASE, args))
    assert hit is not None
    assert hit.response == {"state": "shipped"}
    assert hit.tier is MatchTier.EXACT
    assert hit.degraded is False


def test_recording_writes_one_row_per_tier(store: CassetteStore, policy) -> None:
    # Three rows is the deliberate cost of being able to say *which* tier
    # matched - see the note in keys.py.
    record_one(store, policy, {"order_id": "ORD-9921", "request_id": "r"}, {})
    assert len(store) == 3
    assert store.stats().by_tier == {"exact": 1, "normalized": 1, "subset": 1}


def test_rich_types_survive_the_store(store: CassetteStore, policy) -> None:
    response = {
        "total": Decimal("64.50"),
        "placed_at": datetime(2026, 10, 24, 12, 0, tzinfo=UTC),
        "checkpoints": ("Leipzig", "Brussels"),
    }
    args = {"order_id": "ORD-9921"}
    record_one(store, policy, args, response)

    hit = store.lookup(policy.compute(CASE, args))
    assert hit is not None
    assert hit.response == response
    assert isinstance(hit.response["total"], Decimal)
    assert isinstance(hit.response["checkpoints"], tuple)


def test_a_formatting_difference_replays_as_a_degraded_hit(
    store: CassetteStore, policy
) -> None:
    record_one(store, policy, {"order_id": "ORD-9921", "request_id": "r1"}, {"ok": True})

    hit = store.lookup(policy.compute(CASE, {"order_id": " ord-9921 ", "request_id": "r2"}))
    assert hit is not None
    assert hit.tier is MatchTier.NORMALIZED
    # Degraded, so the scorecard cannot show this as a clean pass: our
    # normalizers did work the model should have done.
    assert hit.degraded is True


def test_a_hallucinated_argument_misses(store: CassetteStore, policy) -> None:
    record_one(store, policy, {"order_id": "ORD-9921"}, {"ok": True})
    assert store.lookup(policy.compute(CASE, {"order_id": "9921"})) is None


def test_cassettes_do_not_leak_between_cases(store: CassetteStore, policy) -> None:
    args = {"order_id": "ORD-9921"}
    store.record(policy.compute("case_a", args), arguments=args, response={"from": "a"})
    assert store.lookup(policy.compute("case_b", args)) is None


# --------------------------------------------------------------------------- #
# Isolation between replays
# --------------------------------------------------------------------------- #


def test_each_replay_gets_its_own_objects(store: CassetteStore, policy) -> None:
    # 200 replays of one case must not share a mutable payload. If the first
    # execution's agent appends to a list the cassette handed it, the other 199
    # see the mutation and the run looks nondeterministic - in the model.
    args = {"order_id": "ORD-9921"}
    record_one(store, policy, args, {"lines": [{"sku": "A-1"}]})

    first = store.lookup(policy.compute(CASE, args))
    assert first is not None
    first.response["lines"].append({"sku": "INJECTED"})

    second = store.lookup(policy.compute(CASE, args))
    assert second is not None
    assert second.response == {"lines": [{"sku": "A-1"}]}


def test_a_rerecord_is_not_shadowed_by_the_cache(store: CassetteStore, policy) -> None:
    # The nastiest cache bug available here: a re-record whose new payload is
    # hidden behind what an earlier execution read would make `slmeval record`
    # look like it silently did nothing.
    args = {"order_id": "ORD-9921"}
    record_one(store, policy, args, {"state": "shipped"})
    assert store.lookup(policy.compute(CASE, args)).response == {"state": "shipped"}

    record_one(store, policy, args, {"state": "delivered"})
    assert store.lookup(policy.compute(CASE, args)).response == {"state": "delivered"}


def test_the_cache_removes_sqlite_from_the_hot_path(store: CassetteStore, policy) -> None:
    args = {"order_id": "ORD-9921"}
    record_one(store, policy, args, {"ok": True})
    keys = policy.compute(CASE, args)

    store.lookup(keys)
    before = store.cache_hits
    for _ in range(50):
        store.lookup(keys)
    assert store.cache_hits - before == 50


def test_an_lru_size_of_zero_still_works(tmp_path: Path, policy) -> None:
    # Config knob, so it has to be safe at its extreme: no cache at all means
    # every lookup goes to SQLite, which is slower and still correct.
    with CassetteStore(tmp_path / "c.sqlite", lru_size=0) as store:
        args = {"order_id": "ORD-1"}
        record_one(store, policy, args, {"ok": True})
        assert store.lookup(policy.compute(CASE, args)).response == {"ok": True}
        assert store.cache_hits == 0


# --------------------------------------------------------------------------- #
# Connections
# --------------------------------------------------------------------------- #


def test_the_store_is_usable_from_several_threads(store: CassetteStore, policy) -> None:
    # The runner executes the matrix concurrently, and a sync agent runs in a
    # worker thread. One connection per thread, and WAL, is what makes that work
    # without `database is locked`.
    errors: list[BaseException] = []

    def work(index: int) -> None:
        try:
            args = {"order_id": f"ORD-{index}"}
            record_one(store, policy, args, {"index": index})
            hit = store.lookup(policy.compute(CASE, args))
            assert hit is not None and hit.response == {"index": index}
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
    assert len(store) == 8 * 3


def test_close_releases_the_file_even_across_threads(tmp_path: Path, policy) -> None:
    # The actual Windows failure this guards: an unclosed handle keeps the
    # database locked, so the cache cannot be deleted and a temp directory
    # cannot be cleaned up.
    store = CassetteStore(tmp_path / "c.sqlite")
    args = {"order_id": "ORD-1"}
    record_one(store, policy, args, {"ok": True})

    done = threading.Event()

    def touch() -> None:
        store.lookup(policy.compute(CASE, args))
        done.set()

    thread = threading.Thread(target=touch)
    thread.start()
    thread.join()
    assert done.is_set()

    store.close()
    store.path.unlink()
    for suffix in ("-wal", "-shm"):
        Path(str(store.path) + suffix).unlink(missing_ok=True)


def test_using_a_closed_store_says_so(store: CassetteStore) -> None:
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.get("whatever")


def test_wal_is_enabled(store: CassetteStore, policy) -> None:
    record_one(store, policy, {"order_id": "ORD-1"}, {})
    mode = store._conn().execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"


# --------------------------------------------------------------------------- #
# Schema and housekeeping
# --------------------------------------------------------------------------- #


def test_a_store_from_a_newer_slmeval_is_refused(tmp_path: Path) -> None:
    # Refused with an explanation, rather than crashing on a column that does
    # not exist yet.
    path = tmp_path / "future.sqlite"
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA user_version=99")
    conn.close()

    with pytest.raises(EnvironmentError_) as caught:
        CassetteStore(path).get("x")
    assert "newer slmeval" in str(caught.value)


def test_an_existing_store_reopens(tmp_path: Path, policy) -> None:
    path = tmp_path / "c.sqlite"
    args = {"order_id": "ORD-1"}
    with CassetteStore(path) as first:
        record_one(first, policy, args, {"ok": True})
    with CassetteStore(path) as second:
        assert second.lookup(policy.compute(CASE, args)).response == {"ok": True}


def test_neighbours_returns_only_exact_rows(store: CassetteStore, policy) -> None:
    # What the miss diagnosis searches. Showing a user a *normalized* rendering
    # of their own recording as "nearest recorded" would be baffling.
    record_one(store, policy, {"order_id": "ORD-9921"}, {})
    record_one(store, policy, {"order_id": "ORD-7310"}, {})

    neighbours = store.neighbours(CASE, "fetch_order")
    assert len(neighbours) == 2
    assert {n.match_tier for n in neighbours} == {MatchTier.EXACT}
    assert {n.arguments()["order_id"] for n in neighbours} == {"ORD-9921", "ORD-7310"}


def test_tool_names_lists_what_was_recorded(store: CassetteStore, policy) -> None:
    record_one(store, policy, {"order_id": "ORD-1"}, {})
    assert store.tool_names() == ["fetch_order"]
    assert store.tool_names(case_id="other_case") == []


def test_delete_case_clears_that_case_only(store: CassetteStore, policy) -> None:
    args = {"order_id": "ORD-1"}
    store.record(policy.compute("case_a", args), arguments=args, response={})
    store.record(policy.compute("case_b", args), arguments=args, response={})

    assert store.delete_case("case_a") == 3
    assert store.lookup(policy.compute("case_a", args)) is None
    assert store.lookup(policy.compute("case_b", args)) is not None


def test_stats_report_the_shape_of_the_cache(store: CassetteStore, policy) -> None:
    record_one(store, policy, {"order_id": "ORD-1"}, {})
    store.record(policy.compute("case_b", {"order_id": "ORD-2"}), arguments={}, response={})

    stats = store.stats()
    assert stats.cassettes == 6
    assert stats.cases == 2
    assert stats.tools == 1
    assert stats.size_bytes > 0


def test_for_project_puts_the_store_under_the_cache_dir(tmp_path: Path) -> None:
    with CassetteStore.for_project(tmp_path) as store:
        assert store.path == tmp_path / ".slmeval" / "cassettes.sqlite"
        assert store.path.parent.is_dir()
