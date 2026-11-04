"""The README's images and the links around them.

The charts are drawn from the teardown's published data, so they should match what
the script draws now. The README and docs link to files by path, so those paths
should exist. The issue templates are YAML that nothing else in the suite parses.
"""

from __future__ import annotations

import html
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

import pytest

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "docs" / "assets"
SCRIPT = ROOT / "scripts" / "readme_visuals.py"
DOCS = [ROOT / "README.md", ROOT / "CONTRIBUTING.md", *sorted((ROOT / "docs").glob("*.md"))]
REPO_URL = re.compile(
    r"https://github\.com/mohitraj3697/slmeval/(?:raw|blob|tree)/main/([^)\s\"]+)"
)
RELATIVE_LINK = re.compile(r"\]\((?!https?://|#|mailto:)([^)\s#]+)(?:#[^)\s]*)?\)")


def visuals() -> Any:
    name = "_readme_visuals"
    spec = importlib.util.spec_from_file_location(name, SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name", ["pass-rates.svg", "where-models-failed.svg", "fault-injection.svg", "how-it-works.svg"]
)
def test_each_chart_matches_the_published_data(name: str) -> None:
    drawn = visuals().charts()[name]
    assert (ASSETS / name).read_text(encoding="utf-8") == drawn, (
        f"{name} is out of date. Run scripts/readme_visuals.py."
    )


@pytest.mark.parametrize("path", sorted(ASSETS.glob("*.svg")), ids=lambda p: p.name)
def test_each_image_is_well_formed_and_named(path: Path) -> None:
    root = ElementTree.parse(path).getroot()
    assert root.tag.endswith("svg")
    titles = [child for child in root if child.tag.endswith("title")]
    assert titles, f"{path.name} has no <title> for screen readers"


def test_the_scorecard_capture_shows_the_published_numbers() -> None:
    """It comes from a live run, so it gets compared number by number."""
    raw = (ASSETS / "scorecard.svg").read_text(encoding="utf-8")
    shown = " ".join(html.unescape(raw).replace("\xa0", " ").split())
    scorecard = (ROOT / "teardown" / "artifacts" / "scorecard.md").read_text(encoding="utf-8")
    rows = [
        [cell.strip().strip("*") for cell in line.split("|")[1:5]]
        for line in scorecard.splitlines()
        if line.startswith("| ") and not line.startswith("| Candidate")
    ]
    assert len(rows) == 4
    for candidate, pass_rate, interval, runs in rows:
        for value in (candidate, pass_rate, interval, runs):
            assert value in shown, f"{value} ({candidate}) is not in scorecard.svg"
    assert "FORBIDDEN tool call" in shown


def test_the_readme_labels_the_models_as_scripted() -> None:
    readme = " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())
    first_image = readme.index("docs/assets/scorecard.svg")
    assert "scripted stand-ins, not real SLMs" in readme[first_image : first_image + 500]


@pytest.mark.parametrize("path", DOCS, ids=lambda p: p.name)
def test_links_into_this_repository_resolve(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for target in REPO_URL.findall(text):
        assert (ROOT / target.split("#")[0]).exists(), f"{path.name} links to missing {target}"
    for target in RELATIVE_LINK.findall(text):
        assert (path.parent / target).exists(), f"{path.name} links to missing {target}"


def test_the_readme_images_use_absolute_links() -> None:
    """PyPI shows this README too, and it can't resolve a relative image path."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    images = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)
    assert images
    assert all(image.startswith("https://") for image in images), images


@pytest.mark.parametrize("path", sorted((ROOT / ".github").rglob("*.yml")), ids=lambda p: p.name)
def test_github_config_parses(path: Path) -> None:
    yaml = pytest.importorskip("yaml")
    assert isinstance(yaml.safe_load(path.read_text(encoding="utf-8")), dict)
