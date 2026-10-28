"""Environment checks, with the network scripted."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from slmeval.config.models import ModelCfg
from slmeval.doctor import Status
from slmeval.doctor.checks import (
    _tag_matches,
    check_api_keys,
    check_cache_dir,
    check_disk,
    check_ollama,
    check_openai_compatible,
    check_python,
    check_sync_folder,
)


def tags_response(*names: str) -> httpx.Response:
    return httpx.Response(200, json={"models": [{"name": n, "size": 1} for n in names]})


# -- ollama ----------------------------------------------------------------


def test_reachable_server_with_the_model_pulled(mock_http) -> None:
    client = mock_http(lambda request: tags_response("gemma3:4b", "qwen2.5:7b"))
    checks = check_ollama("http://localhost:11434/v1", ["gemma3:4b"], client=client)
    assert [c.status for c in checks] == [Status.OK, Status.OK]


def test_unreachable_server_fails_with_the_fix(mock_http) -> None:
    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    checks = check_ollama("http://localhost:11434/v1", ["gemma3:4b"], client=mock_http(refuse))
    assert checks[0].status is Status.FAIL
    assert "ollama serve" in (checks[0].hint or "")


def test_missing_model_prints_the_pull_command(mock_http) -> None:
    client = mock_http(lambda request: tags_response("llama3.2:1b"))
    checks = check_ollama("http://localhost:11434/v1", ["gemma3:4b"], client=client)
    missing = [c for c in checks if c.status is Status.FAIL]
    assert len(missing) == 1
    assert missing[0].hint == "ollama pull gemma3:4b"


def test_tags_endpoint_is_derived_from_the_v1_base_url(mock_http) -> None:
    """base_url is the OpenAI-compatible path; /api/tags lives at the root."""
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return tags_response()

    check_ollama("http://localhost:11434/v1", [], client=mock_http(handler))
    assert seen == ["http://localhost:11434/api/tags"]


def test_malformed_json_is_reported_not_raised(mock_http) -> None:
    client = mock_http(lambda request: httpx.Response(200, text="<html>nope</html>"))
    checks = check_ollama("http://localhost:11434/v1", [], client=client)
    assert checks[0].status is Status.FAIL


@pytest.mark.parametrize(
    ("wanted", "available", "expected"),
    [
        ("gemma3:4b", "gemma3:4b", True),
        ("gemma3", "gemma3:latest", True),
        ("gemma3:4b", "gemma3:4b:latest", True),
        ("gemma3:4b", "gemma3:7b", False),
        ("gemma3:4b", "gemma3", False),
        ("qwen2.5", "qwen2.5:7b", True),
    ],
)
def test_tag_matching(wanted: str, available: str, expected: bool) -> None:
    assert _tag_matches(wanted, available) is expected


# -- openai-compatible servers ---------------------------------------------


def test_rejected_credentials_are_distinguished_from_being_down(mock_http) -> None:
    client = mock_http(lambda request: httpx.Response(401, json={"error": "nope"}))
    check = check_openai_compatible("http://localhost:8000/v1", "MY_KEY", client=client)
    assert check.status is Status.FAIL
    assert "rejected the credentials" in check.detail


def test_reachable_server_is_ok(mock_http) -> None:
    client = mock_http(lambda request: httpx.Response(200, json={"data": []}))
    assert check_openai_compatible("http://x:1/v1", None, client=client).status is Status.OK


def test_key_is_sent_as_a_bearer_token_when_set(mock_http, monkeypatch) -> None:
    monkeypatch.setenv("MY_KEY", "sk-secret")
    headers: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        headers.append(request.headers.get("authorization", ""))
        return httpx.Response(200, json={"data": []})

    check_openai_compatible("http://x:1/v1", "MY_KEY", client=mock_http(handler))
    assert headers == ["Bearer sk-secret"]


# -- credentials -----------------------------------------------------------


def test_missing_key_fails_and_names_who_needs_it(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    models = [ModelCfg(id="c1", provider="openai", model="gpt-4o")]
    checks = check_api_keys(models)
    assert checks[0].status is Status.FAIL
    assert "c1" in checks[0].detail


def test_key_value_never_appears_in_output(monkeypatch) -> None:
    """Doctor output gets pasted into issues and screenshots."""
    secret = "sk-do-not-leak-me"
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    checks = check_api_keys([ModelCfg(id="c1", provider="openai", model="gpt-4o")])
    rendered = json.dumps([c.__dict__ for c in checks], default=str)
    assert secret not in rendered
    assert str(len(secret)) not in rendered
    assert checks[0].status is Status.OK


def test_local_providers_need_no_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    models = [
        ModelCfg(id="a", provider="ollama", model="m"),
        ModelCfg(id="b", provider="fake", model="m"),
    ]
    assert check_api_keys(models) == []


# -- storage ---------------------------------------------------------------


def test_cache_dir_gets_wal(tmp_path: Path) -> None:
    check = check_cache_dir(tmp_path)
    assert check.status is Status.OK
    assert "WAL" in check.detail
    assert (tmp_path / ".slmeval").is_dir()
    # The probe database must not be left behind.
    assert not list((tmp_path / ".slmeval").glob("_doctor_probe*"))


@pytest.mark.parametrize("folder", ["OneDrive", "Dropbox", "Google Drive"])
def test_cloud_sync_folders_are_flagged(tmp_path: Path, folder: str) -> None:
    """SQLite WAL and file sync corrupt each other; this is the #1 support ticket."""
    path = tmp_path / folder / "project"
    path.mkdir(parents=True)
    check = check_sync_folder(path)
    assert check.status is Status.WARN
    assert "sync" in (check.hint or "")


def test_ordinary_folder_is_not_flagged(tmp_path: Path) -> None:
    assert check_sync_folder(tmp_path).status is Status.OK


def test_python_and_disk_checks_pass_here(tmp_path: Path) -> None:
    assert check_python().status is Status.OK
    assert check_disk(tmp_path).status in (Status.OK, Status.WARN)
