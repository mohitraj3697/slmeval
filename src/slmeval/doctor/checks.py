"""Environment checks.

`doctor` exists so that "it doesn't work" becomes one pasted block instead of a
twenty-message thread. In practice almost every failure report for a tool like
this is one of five things: the local server isn't running, the model isn't
pulled, the API key isn't exported, the cache is on a cloud-synced folder, or
the disk is full. All five are checked here.

Nothing in this module prints a secret. API keys are reported as present or
absent — never their value, never their length, never a hash.
"""

from __future__ import annotations

import os
import platform
import shutil
import sqlite3
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import httpx

from slmeval import __version__
from slmeval.config import LoadedConfig, ModelCfg

PROBE_TIMEOUT_S = 3.0
MIN_FREE_BYTES = 2 * 1024**3

SYNC_FOLDER_MARKERS = ("onedrive", "dropbox", "google drive", "googledrive", "icloud")


class Status(StrEnum):
    OK = "ok"
    WARN = "warn"
    FAIL = "fail"
    SKIP = "skip"


@dataclass
class Check:
    name: str
    status: Status
    detail: str
    hint: str | None = None


def check_python() -> Check:
    v = sys.version_info
    text = f"{v.major}.{v.minor}.{v.micro}"
    if v < (3, 11):
        return Check(
            "python",
            Status.FAIL,
            f"{text} - slmeval needs 3.11+",
            hint="3.11 is the floor: tomllib and asyncio.TaskGroup are used throughout",
        )
    return Check("python", Status.OK, f"{text} at {sys.executable}")


def _os_name() -> str:
    """platform.release() still says "10" on Windows 11; correct it by build."""
    release = platform.release()
    if sys.platform == "win32" and release == "10":
        try:
            build = int(platform.version().split(".")[2])
        except (IndexError, ValueError):
            return release
        if build >= 22000:
            return f"11 (build {build})"
        return f"10 (build {build})"
    return release


def check_platform() -> Check:
    detail = f"{platform.system()} {_os_name()} ({platform.machine()})"
    if sys.platform == "win32":
        detail += " - use provider 'openai_compatible' for vLLM (no native Windows build)"
    return Check("platform", Status.OK, detail)


def check_version() -> Check:
    return Check("slmeval", Status.OK, __version__)


def check_cache_dir(root: Path) -> Check:
    """Can we create the cache, and does SQLite get write-ahead logging?

    WAL is what lets many concurrent workers read and write cassettes without
    `database is locked`. It silently degrades on network and synced
    filesystems, so it is worth confirming rather than assuming.
    """
    cache = root / ".slmeval"
    try:
        cache.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        return Check(
            "cache dir",
            Status.FAIL,
            f"cannot create {cache}: {exc}",
            hint="check directory permissions, or run from a writable directory",
        )

    probe = cache / "_doctor_probe.sqlite"
    try:
        conn = sqlite3.connect(probe, timeout=5.0)
        try:
            mode = conn.execute("PRAGMA journal_mode=WAL;").fetchone()[0]
            conn.execute("CREATE TABLE IF NOT EXISTS probe (x INTEGER)")
            conn.execute("INSERT INTO probe VALUES (1)")
            conn.commit()
        finally:
            conn.close()
    except (OSError, sqlite3.Error) as exc:
        return Check(
            "cache dir",
            Status.FAIL,
            f"SQLite cannot write in {cache}: {exc}",
            hint="traces and tool cassettes live here; the benchmark cannot run without it",
        )
    finally:
        for suffix in ("", "-wal", "-shm"):
            Path(str(probe) + suffix).unlink(missing_ok=True)

    if str(mode).lower() != "wal":
        return Check(
            "cache dir",
            Status.WARN,
            f"{cache} - SQLite fell back to journal_mode={mode}",
            hint="concurrent runs may hit lock contention; "
            "a local (non-network, non-synced) disk fixes this",
        )
    return Check("cache dir", Status.OK, f"{cache} (WAL enabled)")


def check_sync_folder(root: Path) -> Check:
    """Cloud-sync folders corrupt SQLite WAL files under concurrent writes."""
    parts = [p.lower() for p in root.resolve().parts]
    hit = next(
        (p for p in parts if any(marker in p for marker in SYNC_FOLDER_MARKERS)),
        None,
    )
    if hit:
        return Check(
            "storage location",
            Status.WARN,
            f"path contains {hit!r} - looks like a cloud-synced folder",
            hint="SQLite WAL and file sync do not mix; move the project to a local path, "
            "or exclude .slmeval/ from syncing",
        )
    return Check("storage location", Status.OK, "not a known cloud-sync folder")


def check_disk(root: Path) -> Check:
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return Check("disk space", Status.WARN, f"could not determine: {exc}")
    free_gb = usage.free / 1024**3
    if usage.free < MIN_FREE_BYTES:
        return Check(
            "disk space",
            Status.WARN,
            f"{free_gb:.1f} GB free",
            hint="traces and cassettes grow quickly; `slmeval cache prune` reclaims space",
        )
    return Check("disk space", Status.OK, f"{free_gb:.1f} GB free")


def check_config(loaded: LoadedConfig | None, error: str | None) -> Check:
    if error is not None:
        return Check("config", Status.FAIL, error, hint="run `slmeval validate` for details")
    if loaded is None:
        return Check(
            "config",
            Status.SKIP,
            "no eval.toml found - provider checks skipped",
            hint="run `slmeval init` to create one",
        )
    cfg = loaded.config
    return Check(
        "config",
        Status.OK,
        f"{loaded.path} - benchmark {cfg.benchmark.name!r}, "
        f"{len(cfg.candidates)} candidate(s), hash {cfg.config_hash()}",
    )


# --------------------------------------------------------------------------
# Provider reachability
# --------------------------------------------------------------------------


def _ollama_root(base_url: str) -> str:
    """`http://host:11434/v1` -> `http://host:11434`, where /api/tags lives."""
    return base_url.rstrip("/").removesuffix("/v1")


def _tag_matches(wanted: str, available: str) -> bool:
    """Ollama reports implicit tags as `name:latest`, so compare generously."""
    if wanted == available:
        return True
    if available == f"{wanted}:latest":
        return True
    return ":" not in wanted and available.split(":", 1)[0] == wanted


def check_ollama(
    base_url: str,
    wanted_models: Iterable[str],
    *,
    client: httpx.Client | None = None,
) -> list[Check]:
    """Probe an Ollama server and confirm the configured models are pulled."""
    root = _ollama_root(base_url)
    owns_client = client is None
    client = client or httpx.Client(timeout=PROBE_TIMEOUT_S)
    try:
        try:
            response = client.get(f"{root}/api/tags")
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            return [
                Check(
                    f"ollama @ {root}",
                    Status.FAIL,
                    f"unreachable: {type(exc).__name__}",
                    hint="start it with `ollama serve`, or correct base_url in eval.toml",
                )
            ]
    finally:
        if owns_client:
            client.close()

    available = [m.get("name", "") for m in payload.get("models", [])]
    checks = [
        Check(
            f"ollama @ {root}",
            Status.OK,
            f"reachable, {len(available)} model(s) pulled"
            + (f": {', '.join(sorted(available)[:4])}" if available else ""),
        )
    ]

    for wanted in sorted(set(wanted_models)):
        match = next((a for a in available if _tag_matches(wanted, a)), None)
        if match is None:
            checks.append(
                Check(
                    f"model {wanted}",
                    Status.FAIL,
                    "not pulled",
                    hint=f"ollama pull {wanted}",
                )
            )
        elif match != wanted:
            checks.append(Check(f"model {wanted}", Status.OK, f"resolves to {match}"))
        else:
            checks.append(Check(f"model {wanted}", Status.OK, "pulled"))
    return checks


def check_openai_compatible(
    base_url: str,
    api_key_env: str | None,
    *,
    client: httpx.Client | None = None,
) -> Check:
    headers = {}
    if api_key_env and (key := os.environ.get(api_key_env)):
        headers["Authorization"] = f"Bearer {key}"

    owns_client = client is None
    client = client or httpx.Client(timeout=PROBE_TIMEOUT_S)
    try:
        response = client.get(f"{base_url.rstrip('/')}/models", headers=headers)
        if response.status_code in (401, 403):
            return Check(
                f"server @ {base_url}",
                Status.FAIL,
                f"reachable but rejected the credentials (HTTP {response.status_code})",
                hint=f"check ${api_key_env}" if api_key_env else "set api_key_env in eval.toml",
            )
        response.raise_for_status()
        return Check(f"server @ {base_url}", Status.OK, "reachable")
    except (httpx.HTTPError, ValueError) as exc:
        return Check(
            f"server @ {base_url}",
            Status.FAIL,
            f"unreachable: {type(exc).__name__}",
            hint="start the server, or correct base_url in eval.toml",
        )
    finally:
        if owns_client:
            client.close()


def check_api_keys(models: Iterable[ModelCfg]) -> list[Check]:
    """Presence only. The value never leaves os.environ."""
    needed: dict[str, list[str]] = {}
    for m in models:
        if m.provider in ("openai", "anthropic") and m.api_key_env:
            needed.setdefault(m.api_key_env, []).append(m.id)

    checks = []
    for env_name, users in sorted(needed.items()):
        who = ", ".join(sorted(users))
        if os.environ.get(env_name):
            checks.append(Check(f"${env_name}", Status.OK, f"set - used by {who}"))
        else:
            checks.append(
                Check(
                    f"${env_name}",
                    Status.FAIL,
                    f"not set - required by {who}",
                    hint=f"export {env_name}=... (or add it to a .env file)",
                )
            )
    return checks


def run_checks(
    *,
    root: Path,
    loaded: LoadedConfig | None,
    config_error: str | None = None,
    skip_network: bool = False,
    client: httpx.Client | None = None,
) -> list[Check]:
    """Run every applicable check, in the order a user wants to read them."""
    checks: list[Check] = [
        check_version(),
        check_python(),
        check_platform(),
        check_cache_dir(root),
        check_sync_folder(root),
        check_disk(root),
        check_config(loaded, config_error),
    ]

    if loaded is None:
        return checks

    models = loaded.config.all_models
    checks.extend(check_api_keys(models))

    if skip_network:
        checks.append(Check("providers", Status.SKIP, "--skip-network given"))
        return checks

    ollama_models: dict[str, list[str]] = {}
    seen_servers: set[str] = set()
    for m in models:
        if m.provider == "ollama" and m.base_url:
            ollama_models.setdefault(m.base_url, []).append(m.model)
        elif m.provider == "openai_compatible" and m.base_url not in seen_servers and m.base_url:
            seen_servers.add(m.base_url)
            checks.append(check_openai_compatible(m.base_url, m.api_key_env, client=client))

    for base_url, wanted in sorted(ollama_models.items()):
        checks.extend(check_ollama(base_url, wanted, client=client))

    if not ollama_models and not seen_servers:
        checks.append(
            Check("providers", Status.SKIP, "no local providers configured")
        )
    return checks


def worst_status(checks: Iterable[Check]) -> Status:
    statuses = [c.status for c in checks]
    if Status.FAIL in statuses:
        return Status.FAIL
    if Status.WARN in statuses:
        return Status.WARN
    return Status.OK
