"""The content-addressed payload store.

The tests that matter here are not "can it store bytes" - they are the ways a
blob store can quietly lose or corrupt data:

- a digest from a row used as a path, which must never escape the store;
- a partial write left under the hash of the complete content, which every
  future read would then trust;
- a sweep that collects a payload a run is about to reference.
"""

from __future__ import annotations

import os
import time
from datetime import timedelta
from pathlib import Path

import pytest

from slmeval.storage.blobs import BlobStore, MissingBlob

PAYLOAD = b'{"order_id": "A-1001", "status": "shipped"}'


@pytest.fixture
def blobs(tmp_path: Path) -> BlobStore:
    return BlobStore(tmp_path / "blobs")


# --------------------------------------------------------------------------- #
# Round trip and dedupe
# --------------------------------------------------------------------------- #


def test_round_trip(blobs: BlobStore) -> None:
    digest = blobs.put(PAYLOAD)
    assert len(digest) == 64
    assert blobs.get(digest) == PAYLOAD
    assert blobs.exists(digest)


def test_identical_payloads_stored_once(blobs: BlobStore) -> None:
    """The entire reason this module exists: one recorded tool output replayed
    by every candidate must cost one file, not ninety."""
    digests = {blobs.put(PAYLOAD) for _ in range(90)}
    assert len(digests) == 1
    assert blobs.stats().count == 1
    assert blobs.deduped == 89


def test_json_round_trip_is_stable(blobs: BlobStore) -> None:
    """Key order must not change the digest, or dedupe stops working for
    payloads that are the same object built two ways."""
    a = blobs.put_json({"b": 2, "a": 1})
    b = blobs.put_json({"a": 1, "b": 2})
    assert a == b
    assert blobs.get_json(a) == {"a": 1, "b": 2}


def test_sharded_layout(blobs: BlobStore) -> None:
    """Tens of thousands of files in one directory is slow to enumerate and
    hostile to anyone who opens it."""
    digest = blobs.put(PAYLOAD)
    assert blobs.path_for(digest).parent.name == digest[:2]
    assert blobs.path_for(digest).name == digest[2:]


# --------------------------------------------------------------------------- #
# Hostile input
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "digest",
    [
        "../../../etc/passwd",
        "..\\..\\windows\\system32",
        "not-a-hash",
        "ABCD" * 16,  # uppercase is not our encoding
        "",
        "a" * 63,
    ],
)
def test_bad_digest_is_refused_before_touching_disk(blobs: BlobStore, digest: str) -> None:
    """A hash read back from a hand-edited database is still input."""
    with pytest.raises(MissingBlob):
        blobs.path_for(digest)


def test_missing_payload_names_itself(blobs: BlobStore) -> None:
    digest = "0" * 64
    with pytest.raises(MissingBlob) as err:
        blobs.get(digest)
    assert "000000000000" in str(err.value)


def test_stray_files_are_ignored(blobs: BlobStore) -> None:
    """A leftover temp file from an interrupted write, or anything else a user
    drops in, must not be reported as a payload or swept as garbage."""
    digest = blobs.put(PAYLOAD)
    shard = blobs.path_for(digest).parent
    (shard / ".tmp-interrupted").write_bytes(b"half")
    (blobs.root / "readme.txt").write_text("hello")
    assert blobs.stats().count == 1
    assert [d for d, _ in blobs.iter_digests()] == [digest]


# --------------------------------------------------------------------------- #
# Sweeping
# --------------------------------------------------------------------------- #


def age(path: Path, *, days: int) -> None:
    old = path.stat().st_mtime - days * 86_400
    os.utime(path, (old, old))


def test_sweep_keeps_referenced(blobs: BlobStore) -> None:
    keep = blobs.put(b"referenced")
    drop = blobs.put(b"orphan")
    age(blobs.path_for(keep), days=2)
    age(blobs.path_for(drop), days=2)

    result = blobs.sweep({keep})

    assert result.deleted == 1
    assert result.kept_referenced == 1
    assert blobs.exists(keep)
    assert not blobs.exists(drop)


def test_sweep_spares_recent_payloads(blobs: BlobStore) -> None:
    """The window between writing a payload and committing the row that points
    at it. Without this, pruning mid-run deletes live data."""
    fresh = blobs.put(b"written seconds ago, row not committed yet")

    result = blobs.sweep(set())

    assert result.deleted == 0
    assert result.kept_recent == 1
    assert blobs.exists(fresh)


def test_sweep_grace_is_configurable(blobs: BlobStore) -> None:
    fresh = blobs.put(b"orphan")
    result = blobs.sweep(set(), grace=timedelta(0))
    assert result.deleted == 1
    assert not blobs.exists(fresh)


def test_no_grace_does_not_depend_on_file_timestamps(blobs: BlobStore) -> None:
    """A fresh file's timestamp can land just after the time the sweep reads.

    That failed `test_prune_frees_payloads_nothing_references` intermittently on
    Windows CI. The timestamp is pushed forward here so the case is not left to
    chance. A real grace period still spares the file.
    """
    fresh = blobs.put(b"orphan")
    ahead = time.time() + 3600
    os.utime(blobs.path_for(fresh), (ahead, ahead))

    assert blobs.sweep(set(), grace=timedelta(seconds=1)).kept_recent == 1

    result = blobs.sweep(set(), grace=timedelta(0))
    assert result.deleted == 1
    assert result.kept_recent == 0
    assert not blobs.exists(fresh)


def test_dry_run_deletes_nothing(blobs: BlobStore) -> None:
    orphan = blobs.put(b"orphan")
    age(blobs.path_for(orphan), days=2)

    result = blobs.sweep(set(), dry_run=True)

    assert result.deleted == 1
    assert result.freed_bytes > 0
    assert blobs.exists(orphan)


def test_re_reference_refreshes_the_clock(blobs: BlobStore) -> None:
    """A payload first written by an old run and re-referenced by a new one
    must look new, or pruning the old run collects it while the new run's row
    is still uncommitted."""
    digest = blobs.put(PAYLOAD)
    age(blobs.path_for(digest), days=40)

    blobs.put(PAYLOAD)  # a new run stores the same payload

    assert blobs.sweep(set()).deleted == 0
    assert blobs.exists(digest)


def test_sweep_removes_empty_shards(blobs: BlobStore) -> None:
    digest = blobs.put(PAYLOAD)
    shard = blobs.path_for(digest).parent
    age(blobs.path_for(digest), days=2)

    blobs.sweep(set())

    assert not shard.exists()


def test_stats_on_a_store_that_was_never_written(tmp_path: Path) -> None:
    """`cache stats` runs before the first benchmark does."""
    stats = BlobStore(tmp_path / "never").stats()
    assert stats == type(stats)(count=0, size_bytes=0)
