"""Draw the images in the README from the teardown's published data.

    python scripts/readme_visuals.py            # write everything to docs/assets
    python scripts/readme_visuals.py --check    # fail if a chart is out of date

Charts are built from `teardown/artifacts/report.json` and `report-faults.json`,
so a chart cannot show a number the tool did not produce. `tests/test_visuals.py`
runs the same comparison as `--check`.

`scorecard.svg` is a capture of `slmeval report`, shortened. Rich draws it, so its
bytes depend on the Rich version and `--check` skips it. The tests check its
numbers against the published scorecard instead.

Standard library only for the charts: no plotting dependency for a few SVG files.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

REPO = Path(__file__).resolve().parent.parent
TEARDOWN = REPO / "teardown"
ARTIFACTS = TEARDOWN / "artifacts"
ASSETS = REPO / "docs" / "assets"
DATASET = REPO / "examples" / "support_agent" / "golden_cases.jsonl"

GATE_FLOOR = 0.5
"""The floor `gate.txt` was produced with. 4/4 cases has a Wilson lower bound of
51%, so this is as high as a floor can go on four cases."""

SANS = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

STYLE = f"""
text {{ font-family: {SANS}; font-size: 13px; fill: #0b0b0b; }}
.mono {{ font-family: {MONO}; font-size: 12.5px; }}
.title {{ font-size: 15px; font-weight: 600; }}
.sub {{ fill: #52514e; }}
.muted {{ fill: #6f6d68; font-size: 12px; }}
.card {{ fill: #fcfcfb; stroke: #e1e0d9; }}
.grid {{ stroke: #e1e0d9; stroke-width: 1; }}
.axis {{ stroke: #c3c2b7; stroke-width: 1; }}
.floor {{ stroke: #52514e; stroke-width: 1.5; }}
.ring {{ stroke: #fcfcfb; stroke-width: 2; }}
.s1 {{ fill: #2a78d6; }}
.s2 {{ fill: #eb6834; }}
.line1 {{ stroke: #2a78d6; stroke-width: 2; stroke-linecap: round; }}
.link {{ stroke: #898781; stroke-width: 2; stroke-linecap: round; }}
.good {{ fill: #0a860a; }}
.bad {{ fill: #d03b3b; }}
.on-bad {{ fill: #ffffff; font-weight: 600; }}
.tile {{ fill: #f4f3ef; }}
.box {{ fill: #f4f3ef; stroke: #c3c2b7; }}
.band {{ fill: #f4f3ef; }}
.arrow {{ stroke: #52514e; stroke-width: 1.5; fill: none; }}
.head {{ fill: #52514e; }}
@media (prefers-color-scheme: dark) {{
  text {{ fill: #f0efec; }}
  .sub {{ fill: #c3c2b7; }}
  .muted {{ fill: #a3a199; }}
  .card {{ fill: #1a1a19; stroke: #383835; }}
  .grid {{ stroke: #2c2c2a; }}
  .axis {{ stroke: #383835; }}
  .floor {{ stroke: #c3c2b7; }}
  .ring {{ stroke: #1a1a19; }}
  .s1 {{ fill: #3987e5; }}
  .s2 {{ fill: #d95926; }}
  .line1 {{ stroke: #3987e5; }}
  .good {{ fill: #0ca30c; }}
  .on-bad {{ fill: #ffffff; }}
  .tile {{ fill: #262624; }}
  .box {{ fill: #262624; stroke: #4a4a46; }}
  .band {{ fill: #222220; }}
  .arrow {{ stroke: #c3c2b7; }}
  .head {{ fill: #c3c2b7; }}
}}
"""


@dataclass(frozen=True)
class Row:
    """The numbers one chart row needs."""

    name: str
    baseline: bool
    pass_rate: float
    low: float
    high: float
    unsafe: int
    faults_met: int
    recovery: float | None


def load_report(name: str) -> dict:
    raw = (ARTIFACTS / name).read_text(encoding="utf-8")
    return json.loads(raw[raw.index("{") :])


def rows(report: dict) -> list[Row]:
    return [
        Row(
            name=c["candidate_id"],
            baseline=c["role"] == "baseline",
            pass_rate=c["pass_hat_k"],
            low=c["wilson_lo"],
            high=c["wilson_hi"],
            unsafe=c["unsafe_actions"],
            faults_met=c["faults_met"],
            recovery=c["recovery_rate"],
        )
        for c in report["candidates"]
    ]


def pct(value: float) -> str:
    return f"{value * 100:.0f}%"


def text(x: float, y: float, content: str, cls: str = "", anchor: str = "start") -> str:
    attrs = f' class="{cls}"' if cls else ""
    if anchor != "start":
        attrs += f' text-anchor="{anchor}"'
    return f'<text x="{x:g}" y="{y:g}"{attrs}>{escape(content)}</text>'


def line(x1: float, y1: float, x2: float, y2: float, cls: str) -> str:
    return f'<line x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}" class="{cls}"/>'


def dot(x: float, y: float, cls: str) -> str:
    return f'<circle cx="{x:g}" cy="{y:g}" r="5.5" class="{cls} ring"/>'


def svg(width: int, height: int, title: str, description: str, body: list[str]) -> str:
    content = "\n".join(body)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">\n'
        f"<title>{escape(title)}</title>\n<desc>{escape(description)}</desc>\n"
        f"<style>{STYLE}</style>\n"
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="6" class="card"/>\n'
        f"{content}\n</svg>\n"
    )


def axis(left: float, right: float, top: float, bottom: float) -> list[str]:
    """The shared 0-100% scale: hairline grid, labels underneath."""
    parts = []
    for tick in (0, 25, 50, 75, 100):
        x = left + (right - left) * tick / 100
        parts.append(line(x, top, x, bottom, "axis" if tick == 0 else "grid"))
        parts.append(text(x, bottom + 17, f"{tick}%", "muted", "middle"))
    return parts


def pass_rate_chart(report: dict) -> str:
    """Point estimate and interval per model, against the gate's floor."""
    data = rows(report)
    width, label_x, left, right = 860, 24, 170, 590
    top, step = 96, 40
    bottom = top + step * len(data) - 8

    def scale(value: float) -> float:
        return left + (right - left) * value

    body = [
        text(24, 32, "Pass rate with its 95% interval", "title"),
        text(
            24,
            52,
            "pass^3: share of cases that passed all 3 repeats. Four cases, scripted models.",
            "sub",
        ),
        text(scale(GATE_FLOOR), top - 12, f"gate floor {pct(GATE_FLOOR)}", "muted", "middle"),
        text(610, top - 12, "pass^3", "muted"),
        text(662, top - 12, "95% CI", "muted"),
        text(742, top - 12, "lower bound vs floor", "muted"),
    ]
    body += axis(left, right, top, bottom)
    body.append(line(scale(GATE_FLOOR), top, scale(GATE_FLOOR), bottom, "floor"))

    for index, row in enumerate(data):
        y = top + step * index + 16
        body.append(text(label_x, y + 4, row.name, "mono"))
        body.append(line(scale(row.low), y, scale(row.high), y, "line1"))
        body.append(dot(scale(row.pass_rate), y, "s1"))
        body.append(text(610, y + 4, pct(row.pass_rate)))
        body.append(text(662, y + 4, f"{pct(row.low)} to {pct(row.high)}"))
        if row.baseline:
            body.append(text(742, y + 4, "reference", "sub"))
        elif row.low >= GATE_FLOOR:
            body.append(text(742, y + 4, "✓", "good"))
            body.append(text(756, y + 4, "clears"))
        else:
            body.append(text(742, y + 4, "✗", "bad"))
            body.append(text(756, y + 4, "below"))

    footer = bottom + 44
    unsafe = [r.name for r in data if r.unsafe]
    note = (
        f"{', '.join(unsafe)} also called a forbidden tool, which fails the gate at any pass rate."
    )
    body.append(text(24, footer, note, "sub"))
    return svg(
        width,
        footer + 18,
        "Pass rate with its 95% interval, by model",
        "; ".join(
            f"{r.name}: {pct(r.pass_rate)}, interval {pct(r.low)} to {pct(r.high)}" for r in data
        ),
        body,
    )


def agent_steps() -> list[str]:
    """The agent's steps in order, from the dataset's declared path."""
    for raw in DATASET.read_text(encoding="utf-8").splitlines():
        paths = json.loads(raw).get("assert", {}).get("reference_trajectory")
        if paths:
            return list(paths[0])
    raise SystemExit(f"no reference_trajectory in {DATASET}")


def node_chart(report: dict) -> str:
    """Which agent step each model got through, and where it failed."""
    steps = agent_steps()
    data = rows(report)
    calls = {(n["candidate_id"], n["node"]): n["calls"] - n["errors"] for n in report["nodes"]}
    fails = {(b["candidate_id"], b["node"]): b["count"] for b in report["blame"]}

    left, tile_width, tile_height, gap = 170, 150, 30, 10
    top = 100
    width = left + (tile_width + gap) * len(steps) + 14

    body = [
        text(24, 32, "Where each model failed", "title"),
        text(24, 52, "Executions per agent step, in the order the agent runs them.", "sub"),
    ]
    for column, step in enumerate(steps):
        x = left + (tile_width + gap) * column
        body.append(text(x + tile_width / 2, top - 12, step, "mono", "middle"))
    for index, row in enumerate(data):
        y = top + (tile_height + gap) * index
        body.append(text(24, y + 20, row.name, "mono"))
        for column, step in enumerate(steps):
            x = left + (tile_width + gap) * column
            mid = x + tile_width / 2
            failed = fails.get((row.name, step), 0)
            passed = calls.get((row.name, step), 0)
            cls = "bad" if failed else "tile"
            body.append(
                f'<rect x="{x:g}" y="{y:g}" width="{tile_width}" height="{tile_height}" '
                f'rx="4" class="{cls}"/>'
            )
            if failed:
                body.append(text(mid, y + 20, f"✗ {failed} failed", "on-bad", "middle"))
            elif passed:
                body.append(text(mid - 26, y + 20, "✓", "good"))
                body.append(text(mid - 12, y + 20, f"{passed} ok"))
            else:
                body.append(text(mid, y + 20, "no model call", "muted", "middle"))

    footer = top + (tile_height + gap) * len(data) + 18
    body.append(
        text(
            24,
            footer,
            "order_lookup and policy_check only call tools, so a pass there shows no model call.",
            "muted",
        )
    )
    body.append(
        text(24, footer + 18, "A failure stops the run, so later steps stay empty.", "muted")
    )
    description = "; ".join(
        f"{candidate} failed {count} times at {node}" for (candidate, node), count in fails.items()
    )
    return svg(width, footer + 34, "Where each model failed", description, body)


def faults_chart(clean: dict, faulted: dict) -> str:
    """The same matrix with and without injected tool failures."""
    before = rows(clean)
    after = {row.name: row for row in rows(faulted)}
    width, left, right = 860, 170, 620
    top, step = 106, 36
    bottom = top + step * len(before) - 6

    def scale(value: float) -> float:
        return left + (right - left) * value

    body = [
        text(24, 32, "Pass rate with and without injected tool failures", "title"),
        text(
            24,
            52,
            "Same models, cases and seed. 30% of tool calls faulted in the second run.",
            "sub",
        ),
        '<circle cx="31" cy="76" r="5.5" class="s1"/>',
        text(43, 80, "no faults"),
        '<circle cx="124" cy="76" r="5.5" class="s2"/>',
        text(136, 80, "30% fault rate"),
        text(650, top - 12, "pass^3", "muted"),
        text(734, top - 12, "faults met", "muted"),
    ]
    body += axis(left, right, top, bottom)

    for index, row in enumerate(before):
        other = after[row.name]
        y = top + step * index + 15
        body.append(text(24, y + 4, row.name, "mono"))
        body.append(line(scale(row.pass_rate), y, scale(other.pass_rate), y, "link"))
        body.append(dot(scale(row.pass_rate), y, "s1"))
        body.append(dot(scale(other.pass_rate), y, "s2"))
        body.append(text(650, y + 4, f"{pct(row.pass_rate)} to {pct(other.pass_rate)}"))
        body.append(text(734, y + 4, str(other.faults_met)))

    met = sum(r.faults_met for r in after.values())
    recovered = sum(round((r.recovery or 0) * r.faults_met) for r in after.values())
    footer = bottom + 44
    body.append(
        text(
            24,
            footer,
            f"Recovered from {recovered} of {met} faults. The cause is the example agent: "
            "it crashes on an empty tool response.",
            "sub",
        )
    )
    return svg(
        width,
        footer + 18,
        "Pass rate with and without injected tool failures",
        "; ".join(
            f"{r.name}: {pct(r.pass_rate)} without faults, "
            f"{pct(after[r.name].pass_rate)} with faults"
            for r in before
        ),
        body,
    )


def box(x: float, y: float, w: float, h: float, lines: list[tuple[str, str]]) -> list[str]:
    """A rounded box with centred lines of text."""
    parts = [f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="5" class="box"/>']
    start = y + h / 2 - (len(lines) - 1) * 9 + 4
    for index, (content, cls) in enumerate(lines):
        parts.append(text(x + w / 2, start + index * 18, content, cls, "middle"))
    return parts


def arrow(points: list[tuple[float, float]]) -> list[str]:
    """A polyline with a small filled head at its last point."""
    path = " ".join(f"{x:g},{y:g}" for x, y in points)
    (x1, y1), (x2, y2) = points[-2], points[-1]
    if x2 > x1:
        head = f"{x2},{y2} {x2 - 7},{y2 - 4} {x2 - 7},{y2 + 4}"
    elif y2 > y1:
        head = f"{x2},{y2} {x2 - 4},{y2 - 7} {x2 + 4},{y2 - 7}"
    else:
        head = f"{x2},{y2} {x2 - 4},{y2 + 7} {x2 + 4},{y2 + 7}"
    return [
        f'<polyline points="{path}" class="arrow"/>',
        f'<polygon points="{head}" class="head"/>',
    ]


def how_it_works() -> str:
    """Record once, then replay for every model."""
    body = [
        text(24, 32, "How a benchmark runs", "title"),
        '<rect x="14" y="48" width="832" height="104" rx="5" class="band"/>',
        text(26, 68, "Once", "sub"),
        '<rect x="14" y="162" width="832" height="150" rx="5" class="band"/>',
        text(26, 182, "Every run: offline, no API keys for tools", "sub"),
    ]
    body += box(26, 78, 190, 62, [("Your agent", ""), ("with its real tools", "muted")])
    body += box(290, 78, 190, 62, [("slmeval record", "mono"), ("runs the tools once", "muted")])
    body += box(554, 78, 190, 62, [("cassettes.sqlite", "mono"), ("every tool response", "muted")])
    body += arrow([(216, 109), (290, 109)])
    body += arrow([(480, 109), (554, 109)])

    body += box(
        26,
        200,
        150,
        92,
        [
            ("Candidates", ""),
            ("baseline", "mono"),
            ("small model A", "mono"),
            ("small model B", "mono"),
        ],
    )
    body += box(
        226,
        200,
        150,
        92,
        [("slmeval run", "mono"), ("models x cases", "muted"), ("x repeats", "muted")],
    )
    body += box(
        426,
        200,
        150,
        92,
        [
            ("Checks", ""),
            ("assertions", "muted"),
            ("tool calls, path", "muted"),
            ("pass^k, interval", "muted"),
        ],
    )
    body += box(
        626,
        200,
        94,
        92,
        [("Report", ""), ("table", "muted"), ("JSON", "muted"), ("JUnit", "muted")],
    )
    body += box(748, 200, 86, 92, [("slmeval", "mono"), ("gate", "mono"), ("exit 0 / 1", "muted")])
    body += arrow([(176, 246), (226, 246)])
    body += arrow([(376, 246), (426, 246)])
    body += arrow([(576, 246), (626, 246)])
    body += arrow([(720, 246), (748, 246)])
    # Down from the cassette store and left into `slmeval run`.
    body += arrow([(649, 140), (649, 172), (301, 172), (301, 200)])
    return svg(
        860,
        326,
        "How a benchmark runs",
        "slmeval record runs your agent's real tools once and stores every response. "
        "slmeval run then replays those responses for each candidate model, checks the "
        "results and reports them, and slmeval gate turns the report into an exit code.",
        body,
    )


def charts() -> dict[str, str]:
    """Every image that is drawn purely from committed data."""
    clean = load_report("report.json")
    faulted = load_report("report-faults.json")
    return {
        "pass-rates.svg": pass_rate_chart(clean),
        "where-models-failed.svg": node_chart(clean),
        "fault-injection.svg": faults_chart(clean, faulted),
        "how-it-works.svg": how_it_works(),
    }


def capture_scorecard() -> str:
    """A run of the teardown, drawn as a terminal. Real output, shortened."""
    from rich.console import Console
    from rich.text import Text

    width = 104
    env = dict(os.environ)
    env.update(
        PYTHONIOENCODING="utf-8",
        COLUMNS=str(width),
        FORCE_COLOR="1",
        TERM="xterm-256color",
        COLORTERM="truecolor",
    )
    env.pop("NO_COLOR", None)

    def cli(*argv: str) -> str:
        return subprocess.run(
            [sys.executable, "-m", "slmeval", *argv],
            cwd=TEARDOWN,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        ).stdout

    if not (TEARDOWN / ".slmeval" / "cassettes.sqlite").exists():
        cli("record", "--quiet")
    cli("run", "--quiet")
    lines = cli("report").replace("\r\n", "\n").split("\n")

    # What is left out: the line with the run id and commit, which change every
    # run, and the notes that need more explaining than a picture can do.
    skipping = False
    kept: list[Text] = []
    dropped = (
        "explored paths",
        "some token counts",
        "latency not measured",
        "pass^3 =",
        "gate on the lower bound",
        "prices as of",
    )
    for raw in lines:
        plain = Text.from_ansi(raw).plain
        stripped = plain.strip()
        new_note = stripped[:2] in {"✗ ", "! ", "· "}
        if new_note or not stripped:
            skipping = False
        if any(mark in plain for mark in dropped) or "· dataset " in plain:
            skipping = True
        if not skipping:
            kept.append(Text.from_ansi(raw))
    while kept and not kept[0].plain.strip():
        kept.pop(0)
    while kept and not kept[-1].plain.strip():
        kept.pop()

    console = Console(
        record=True,
        width=width,
        force_terminal=True,
        legacy_windows=False,
        file=io.StringIO(),
    )
    for item in kept:
        console.print(item, no_wrap=True, overflow="crop")
    drawn = console.export_svg(title="slmeval report", unique_id="scorecard")
    # Rich leaves out the element screen readers announce.
    head = drawn.index(">") + 1
    return f"{drawn[:head]}<title>Output of slmeval report, shortened</title>{drawn[head:]}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Compare only. Exit 1 if stale.")
    args = parser.parse_args()

    drawn = charts()
    if args.check:
        stale = [
            name
            for name, content in drawn.items()
            if not (ASSETS / name).exists()
            or (ASSETS / name).read_text(encoding="utf-8") != content
        ]
        if stale:
            print(
                f"out of date: {', '.join(stale)}. Run scripts/readme_visuals.py.", file=sys.stderr
            )
            return 1
        print(f"{len(drawn)} image(s) match the published data")
        return 0

    ASSETS.mkdir(parents=True, exist_ok=True)
    drawn["scorecard.svg"] = capture_scorecard()
    for name, content in drawn.items():
        (ASSETS / name).write_text(content, encoding="utf-8", newline="\n")
    print(f"{len(drawn)} image(s) written to {ASSETS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
