"""Content-addressed storage for large payloads.

Full tool outputs and raw model responses do not belong in a database row. A
single recorded payload is replayed by every candidate on every repeat, so a
10-case x 3-candidate x 3-repeat matrix would store the same JSON ninety times;
addressed by its SHA-256 it is stored once, and the rows hold a 64-character
hash. On a real benchmark that is the difference between a 40 MB file and a
400 MB one, and the saving grows with exactly the thing we want users to do
more of - run the matrix again.

Four things here are deliberate rather than incidental.

**Writes are atomic.** Content is written to a temporary file and moved into
place with ``os.replace``. A Ctrl-C halfway through a write would otherwise
leave a truncated file *under the hash of its full contents* - a corrupt
payload that every future lookup trusts, because the name is the checksum and
we would never check it again.

**A digest is validated before it touches the filesystem.** A hash read back
from a row is still input: a hand-edited database with ``../../../etc/passwd``
in ``output_hash`` must get a refusal, not a file read. Cheap, and the kind of
thing that is embarrassing to add later.

**Files are sharded by the first two hex characters.** A benchmark produces
tens of thousands of blobs, and a single directory with 60,000 entries is slow
to enumerate on NTFS and hostile to anyone who opens it in a file manager.

**Garbage collection is mark-and-sweep with a grace period.** The database is
the only authority on what is referenced; the sweep deletes what it cannot find
a reference to. The grace period covers the gap between writing a blob and
committing the row that points at it - without it, pruning while a run is in
flight can delete a payload the run is about to reference.
"""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import tempfile
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

from slmeval.errors import EnvironmentError_, SlmevalError
from slmeval.storage import codec

DIGEST_RE: Final = re.compile(r"^[0-9a-f]{64}$")

SWEEP_GRACE: Final = timedelta(hours=1)
"""How recently a blob must have been touched to be spared by a sweep.

Generous on purpose. The cost of being wrong in one direction is a few
kilobytes of dead payload that the next prune collects; in the other it is a
report that cannot render a trace it has a row for. An hour is long enough to
cover any plausible gap between a blob write and its commit, including a run
paused by a sleeping laptop.
"""


class MissingBlob(SlmevalError):
    """A row points at a payload the store does not have.

    Its own type because it has a specific cause worth naming in the message:
    someone deleted ``.slmeval/blobs`` while keeping ``runs.sqlite``, or pruned
    with a clock skewed far enough to defeat the grace period. Either way the
    run's shape is intact and only this payload is gone, so the caller can
    degrade to "payload unavailable" instead of failing the whole report.
    """


@dataclass(frozen=True, slots=True)
class BlobStats:
    count: int
    size_bytes: int


@dataclass(frozen=True, slots=True)
class SweepResult:
    """What a garbage collection pass did, and what it chose not to touch."""

    deleted: int
    freed_bytes: int
    kept_referenced: int
    kept_recent: int
    """Unreferenced blobs spared by the grace period. A non-zero count here
    while a run is in flight is the mechanism working, not a leak."""


class BlobStore:
    """Payloads addressed by the SHA-256 of their bytes.

    Thread-safe without a lock, which is the quiet benefit of content
    addressing: two threads writing the same payload write identical bytes to
    different temporary files and race only to rename, and either winner leaves
    the correct content in place.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.writes = 0
        self.deduped = 0
        """Writes that found the payload already present. The ratio of these to
        ``writes`` is what ``cache stats`` reports as the dedupe rate, and it is
        the number that justifies this module existing."""

    @classmethod
    def for_project(cls, root: Path | str) -> BlobStore:
        return cls(Path(root) / ".slmeval" / "blobs")

    # ------------------------------------------------------------------ paths

    def path_for(self, digest: str) -> Path:
        """Where a digest lives. Validates first, always."""
        if not DIGEST_RE.match(digest):
            raise MissingBlob(
                f"not a payload hash: {digest[:80]!r}",
                hint="the trace store may be corrupt; `slmeval cache stats` reports its state",
            )
        return self.root / digest[:2] / digest[2:]

    def exists(self, digest: str) -> bool:
        return self.path_for(digest).is_file()

    # ---------------------------------------------------------------- writing

    def put(self, data: bytes) -> str:
        """Store bytes, return their digest. A no-op if already present."""
        digest = hashlib.sha256(data).hexdigest()
        target = self.path_for(digest)
        self.writes += 1
        if target.is_file():
            self.deduped += 1
            # Touch it. The sweep spares recently-written blobs, and a payload
            # re-referenced by a new run has to look new - otherwise pruning
            # the old run that first wrote it can collect a blob the new run's
            # uncommitted row already points at.
            with contextlib.suppress(OSError):
                os.utime(target)
            return digest

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            # Same directory as the target, so the move is a rename within one
            # filesystem and therefore atomic. A temp file in the system temp
            # directory would make this a copy across devices, which is neither
            # atomic nor fast.
            fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=".tmp-")
            tmp = Path(tmp_name)
            try:
                with os.fdopen(fd, "wb") as handle:
                    handle.write(data)
                # Windows refuses to replace an open file, and will refuse the
                # rename if another thread won the race to the same digest.
                # Identical content, so that loser is a success.
                os.replace(tmp, target)
            except OSError:
                tmp.unlink(missing_ok=True)
                if target.is_file():
                    self.deduped += 1
                    return digest
                raise
        except OSError as exc:
            raise EnvironmentError_(
                f"cannot write to the payload store at {self.root}: {exc}",
                hint="run `slmeval doctor` to check the cache directory is writable",
            ) from exc
        return digest

    def put_text(self, text: str) -> str:
        return self.put(text.encode("utf-8"))

    def put_json(self, value: Any) -> str:
        """Store a JSON-able value through the shared codec, so the same value
        always hashes to the same digest."""
        return self.put_text(codec.dumps(value))

    # ---------------------------------------------------------------- reading

    def get(self, digest: str) -> bytes:
        path = self.path_for(digest)
        try:
            return path.read_bytes()
        except FileNotFoundError as exc:
            raise MissingBlob(
                f"payload {digest[:12]} is referenced by a trace but not in the store",
                hint="it was pruned or deleted; re-run the benchmark to regenerate it",
            ) from exc
        except OSError as exc:
            raise EnvironmentError_(f"cannot read payload {digest[:12]}: {exc}") from exc

    def get_text(self, digest: str) -> str:
        return self.get(digest).decode("utf-8")

    def get_json(self, digest: str) -> Any:
        return codec.loads(self.get_text(digest))

    def size_of(self, digest: str) -> int:
        try:
            return self.path_for(digest).stat().st_size
        except OSError:
            return 0

    # ------------------------------------------------------- enumerate & stats

    def iter_digests(self) -> Iterator[tuple[str, Path]]:
        """Every stored payload. Skips temporary files from an interrupted
        write and anything whose name is not a digest, so a stray file in the
        blob directory is ignored rather than swept or reported as a payload."""
        if not self.root.is_dir():
            return
        for shard in sorted(self.root.iterdir()):
            if not shard.is_dir() or len(shard.name) != 2:
                continue
            for entry in shard.iterdir():
                if not entry.is_file():
                    continue
                digest = shard.name + entry.name
                if DIGEST_RE.match(digest):
                    yield digest, entry

    def stats(self) -> BlobStats:
        count = 0
        size = 0
        for _digest, path in self.iter_digests():
            count += 1
            with contextlib.suppress(OSError):
                size += path.stat().st_size
        return BlobStats(count=count, size_bytes=size)

    # ------------------------------------------------------------------ sweep

    def sweep(
        self,
        referenced: Iterable[str],
        *,
        grace: timedelta = SWEEP_GRACE,
        dry_run: bool = False,
    ) -> SweepResult:
        """Delete payloads nothing references.

        ``referenced`` must be every hash in the database, not the hashes of
        one run - a payload shared by three runs is referenced until the last
        of them is gone, and that is the entire point of content addressing.
        Passing a partial set deletes live data, so the only caller is
        :meth:`RunStore.prune <slmeval.storage.db.RunStore.prune>`, which reads
        the whole store.
        """
        keep = set(referenced)
        cutoff = datetime.now(UTC) - grace
        deleted = 0
        freed = 0
        kept_recent = 0
        kept_referenced = 0

        for digest, path in self.iter_digests():
            if digest in keep:
                kept_referenced += 1
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            if datetime.fromtimestamp(stat.st_mtime, UTC) > cutoff:
                kept_recent += 1
                continue
            if not dry_run:
                try:
                    path.unlink()
                except OSError:
                    # A payload another process is reading, or already gone.
                    # Neither is a reason to abandon the rest of the sweep.
                    continue
            deleted += 1
            freed += stat.st_size

        if not dry_run:
            self._drop_empty_shards()
        return SweepResult(
            deleted=deleted,
            freed_bytes=freed,
            kept_referenced=kept_referenced,
            kept_recent=kept_recent,
        )

    def _drop_empty_shards(self) -> None:
        if not self.root.is_dir():
            return
        for shard in self.root.iterdir():
            if shard.is_dir() and len(shard.name) == 2:
                with contextlib.suppress(OSError):
                    shard.rmdir()


__all__ = [
    "DIGEST_RE",
    "SWEEP_GRACE",
    "BlobStats",
    "BlobStore",
    "MissingBlob",
    "SweepResult",
]
