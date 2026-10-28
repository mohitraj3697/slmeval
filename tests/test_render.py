"""Output rendering.

Snapshot-style assertions on a recording console, so a column change or a lost
fix-hint fails a test instead of quietly shipping.
"""

from __future__ import annotations

import io
from pathlib import Path

from rich.console import Console

from slmeval.cli import render
from slmeval.doctor import Check, Status
from slmeval.errors import ConfigError, Problem


def capture(renderable: object, *, width: int = 120) -> str:
    console = Console(file=io.StringIO(), width=width, no_color=True, record=True)
    console.print(renderable)
    return console.export_text()


def test_fix_hints_are_shown_for_failures_only() -> None:
    checks = [
        Check("ok thing", Status.OK, "fine", hint="you will never see this"),
        Check("broken", Status.FAIL, "it is down", hint="turn it on"),
    ]
    out = capture(render.render_checks(checks, ascii_mode=True))
    assert "turn it on" in out
    assert "you will never see this" not in out


def test_summary_counts_every_status() -> None:
    checks = [
        Check("a", Status.OK, ""),
        Check("b", Status.WARN, ""),
        Check("c", Status.FAIL, ""),
        Check("d", Status.SKIP, ""),
    ]
    out = capture(render.checks_summary(checks))
    assert "1 failed" in out
    assert "1 warning" in out
    assert "1 skipped" in out
    assert "1 ok" in out


def test_ascii_mode_emits_no_unicode() -> None:
    """Legacy Windows consoles cannot encode the box glyphs."""
    out = capture(render.render_checks([Check("x", Status.OK, "y")], ascii_mode=True))
    assert out.isascii()


def test_symbols_fall_back_when_the_stream_cannot_encode() -> None:
    class Cp1252Stream:
        encoding = "cp1252"

    class Utf8Stream:
        encoding = "utf-8"

    assert not render.supports_unicode(Cp1252Stream())
    assert render.supports_unicode(Utf8Stream())
    assert render.symbols(None) is not None


def test_make_encoding_safe_is_a_noop_on_odd_streams() -> None:
    """Must never raise, whatever it is handed."""
    render.make_encoding_safe(None)
    render.make_encoding_safe(object())
    render.make_encoding_safe(io.StringIO())


def test_error_shows_the_offending_source_line(tmp_path: Path, capsys) -> None:
    path = tmp_path / "eval.toml"
    path.write_text("[benchmark]\nname = 1\nruns_per_case = 500\n", encoding="utf-8")
    err = ConfigError(
        "eval.toml is invalid (1 problem)",
        source=path,
        problems=[Problem(path="benchmark.runs_per_case", message="too big", line=3)],
        hint="lower it",
    )
    render.render_error(err, ascii_mode=True)
    out = capsys.readouterr().err
    assert "benchmark.runs_per_case" in out
    assert "runs_per_case = 500" in out
    assert "eval.toml:3" in out
    assert "lower it" in out


def test_error_without_a_line_falls_back_to_the_value(capsys) -> None:
    err = ConfigError(
        "nope",
        problems=[Problem(path="a.b", message="bad", value="'zzz'")],
    )
    render.render_error(err, ascii_mode=True)
    out = capsys.readouterr().err
    assert "got 'zzz'" in out


def test_error_survives_a_missing_source_file(tmp_path: Path, capsys) -> None:
    err = ConfigError(
        "nope",
        source=tmp_path / "gone.toml",
        problems=[Problem(path="a", message="bad", line=9)],
    )
    render.render_error(err, ascii_mode=True)
    assert "bad" in capsys.readouterr().err
