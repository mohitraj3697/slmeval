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


# --------------------------------------------------------------------------- #
# The cassette-miss block
# --------------------------------------------------------------------------- #


def test_render_miss_keeps_the_plain_block_verbatim() -> None:
    # The layout comes from MissDiagnosis.render and this adds style only, so
    # the block a user screenshots from the terminal and the one they paste out
    # of a CI log are the same block.
    from slmeval.cli.render import render_miss
    from slmeval.sandbox.diagnosis import diagnose

    finding = diagnose(
        case_id="refund_104",
        tool_name="fetch_order_from_erp",
        arguments={"order_id": "9921"},
        known_tools=["fetch_order_from_erp"],
        candidate_id="gemma3:4b",
    )
    assert render_miss(finding, ascii_mode=True).plain == finding.render(ascii_mode=True)


def test_an_uncovered_miss_is_not_coloured_like_a_failure() -> None:
    # An uncovered path is a gap in our coverage, not a model failure, and
    # painting it red would undo the distinction the diagnosis exists to draw.
    from slmeval.cli.render import VERDICT_STYLE
    from slmeval.sandbox.diagnosis import Verdict

    assert "red" not in VERDICT_STYLE[Verdict.UNRECORDED_VALID_PATH]
    assert VERDICT_STYLE[Verdict.HALLUCINATED_PARAMETER] == "red"


def test_every_verdict_has_a_style() -> None:
    from slmeval.cli.render import VERDICT_STYLE
    from slmeval.sandbox.diagnosis import Verdict

    assert set(VERDICT_STYLE) == set(Verdict)


def test_a_piped_console_has_the_same_width_on_every_platform(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Rich drops a column on Windows when it finds no console to talk to.

    Published artifacts and snapshots are captured through a pipe, so that one
    column re-wrapped them and made Windows and Linux disagree.
    """
    monkeypatch.setenv("COLUMNS", "100")
    with (tmp_path / "out.txt").open("w", encoding="utf-8") as piped:
        monkeypatch.setattr("sys.stdout", piped)
        assert render.console().width == 100
