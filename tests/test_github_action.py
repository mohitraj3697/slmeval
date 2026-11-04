"""The shipped workflows, checked against the CLI they call.

A GitHub Action is documentation that executes, which makes it the thing most
likely to rot: renaming a flag breaks every user's pipeline and nothing in the
Python suite notices. There is no runner here to execute these files, but
everything that matters about them is checkable without one -

- every ``slmeval`` command line in the action, in the demo workflow and in the
  action's README names a real command and real options;
- the exit-code mapping the action branches on is the one
  :mod:`slmeval.exit_codes` defines, with nothing invented and nothing missing;
- the cassette cache is keyed on the dataset's content hash, with no
  ``restore-keys`` prefix fallback that could restore a stale store.

The last two are the contract. Exit 1 and exit 3 mean different things to a
reviewer, and a cache that silently serves cassettes recorded for a different
dataset turns our own staleness into somebody's model regression.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import typer.main

from slmeval import exit_codes
from slmeval.cli.main import app

ROOT = Path(__file__).resolve().parent.parent
ACTION = ROOT / "examples" / "github-action" / "action.yml"
WORKFLOW = ROOT / "examples" / "github-action" / "slmeval.yml"
ACTION_README = ROOT / "examples" / "github-action" / "README.md"
DEMO = ROOT / ".github" / "workflows" / "gate-demo.yml"

#: `slmeval <command> [<subcommand>] [--flags…]`, to the end of that line.
#:
#: Deliberately a regex and not a YAML parse: the command lines live inside
#: `run:` blocks as shell, so a parser would hand back the same strings and
#: cost a dependency the core install does not have.
#:
#: The boundaries carry weight. ``slmeval-cassettes-…`` is a cache key and
#: ``mohitraj3697/slmeval/examples/…`` is an action reference - neither is a command
#: line - and matching ``[ \t]`` rather than ``\s`` keeps a YAML ``id: slmeval``
#: from swallowing whatever is on the next line.
INVOCATION = re.compile(r"(?<![\w/.-])slmeval(?![\w/.-])[ \t]+([^\n`]+)")

#: Tokens that are a shell's business rather than the CLI's.
SHELL = re.compile(r"^(\$|>|<|\||&|\d>|;)")


def cli_commands() -> dict[str, Any]:
    """The registered commands, read off the built CLI.

    Duck-typed rather than annotated against Click: recent Typer vendors its
    own copy as ``typer._click``, so importing ``click`` here would make this
    test's correctness depend on which Click the environment happens to have.
    """
    group = typer.main.get_command(app)
    commands = getattr(group, "commands", None)
    assert commands, "the CLI is not a command group any more"
    return dict(commands)


def option_names(command: Any) -> set[str]:
    # `-h` / `--help` come from the app's context settings rather than from a
    # parameter, so they are not in `params` and have to be added by hand.
    names: set[str] = {"-h", "--help"}
    for param in command.params:
        names.update(opt for opt in param.opts if opt.startswith("-"))
        names.update(opt for opt in getattr(param, "secondary_opts", ()) if opt.startswith("-"))
    return names


def code_only(text: str) -> str:
    """Markdown with the prose removed: fenced blocks and inline code only.

    Prose says things like "slmeval records the rung that worked", which is a
    sentence and not a command line. Reading only the code is what keeps this
    test asserting about the CLI rather than about English.
    """
    out: list[str] = []
    fenced = False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            out.append(line)
        else:
            out.extend(re.findall(r"`([^`]+)`", line))
    return "\n".join(out)


def shell_only(text: str) -> str:
    """YAML with its comments removed.

    These files explain themselves at length, and a comment saying "installs
    slmeval into the runner's Python" is prose. Only whole-line comments are
    dropped: a `#` inside a `run:` block could be part of a command.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


def invocations(text: str) -> list[list[str]]:
    """Every ``slmeval …`` command line in a file, as argument lists.

    Shell variables are dropped rather than guessed at: ``$CONFIG_ARG`` is a
    real part of the action and an unknowable part of a command line, and
    expanding it here would be this test asserting against its own fiction.
    """
    out: list[list[str]] = []
    for line in text.splitlines():
        for match in INVOCATION.finditer(line):
            if not _is_a_command(line[: match.start()]):
                continue
            raw = match.group(1).strip().rstrip("\\").strip()
            tokens: list[str] = []
            for token in raw.split():
                if token.startswith("$"):
                    continue
                if SHELL.match(token):
                    break
                cleaned = _clean(token)
                if cleaned:
                    tokens.append(cleaned)
            if tokens:
                out.append(tokens)
    return out


def _is_a_command(prefix: str) -> bool:
    """Whether what precedes ``slmeval`` on its line makes it a command.

    A YAML file says ``author: slmeval contributors`` and ``id: slmeval`` as
    well as ``run: slmeval validate --json``, and only the last of those is a
    command line. So a key is allowed in front of ``slmeval`` when it is
    ``run``, and not otherwise.
    """
    head = prefix.strip()
    if not head or head in {"-", ">", "|"}:
        return True
    if head.endswith(":"):
        return head.rsplit(" ", 1)[-1] == "run:"
    return head.endswith(("&&", "||", ";", "(", "$", "uv run", "python -m"))


def _clean(token: str) -> str:
    """One token as the CLI would see it.

    The docs write alternatives as ``run|exec|case``, escape the pipe for a
    Markdown table and wrap the lot in backticks. The first alternative is the
    one to check - the rest are the same command's siblings - and the
    punctuation around it is prose.
    """
    token = token.strip("`'\"(),.;").replace("\\", "")
    return token.split("|", 1)[0]


README = ROOT / "README.md"
EXAMPLE_README = ROOT / "examples" / "support_agent" / "README.md"
JUDGE_README = ROOT / "examples" / "judge" / "README.md"
GUIDES = sorted((ROOT / "docs").glob("*.md"))
CONTRIBUTING = ROOT / "CONTRIBUTING.md"
RELEASING = ROOT / "RELEASING.md"
TEARDOWN_README = ROOT / "teardown" / "README.md"

FILES = [
    ACTION,
    WORKFLOW,
    ACTION_README,
    DEMO,
    README,
    EXAMPLE_README,
    JUDGE_README,
    CONTRIBUTING,
    RELEASING,
    TEARDOWN_README,
    *GUIDES,
]

#: The files whose whole job is to show commands. One of these going empty
#: means the extraction broke, not that the docs got shorter.
DOCUMENTS_COMMANDS = [ACTION, ACTION_README, README, EXAMPLE_README, JUDGE_README, TEARDOWN_README]


def command_lines(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    source = code_only(text) if path.suffix == ".md" else shell_only(text)
    return invocations(source)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_the_files_are_shipped(path: Path) -> None:
    assert path.exists(), f"{path} is referenced by the docs and the tests"


@pytest.mark.parametrize("path", DOCUMENTS_COMMANDS, ids=lambda p: p.name)
def test_the_extraction_still_finds_command_lines(path: Path) -> None:
    """Guards the guard: a drift test that silently matches nothing is worse
    than no drift test, because it reads as a passing check."""
    assert command_lines(path), f"no slmeval invocations extracted from {path.name}"


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_every_command_line_is_one_the_cli_accepts(path: Path) -> None:
    """The drift test. Rename a flag and this goes red before a user's CI does."""
    commands = cli_commands()
    lines = command_lines(path)

    for tokens in lines:
        head = tokens[0]
        if head.startswith("-"):
            # `slmeval --version`, and the bare `slmeval` of a pip check.
            root = typer.main.get_command(app)
            assert head in option_names(root), f"{path.name}: unknown root option {head}"
            continue
        assert head in commands, (
            f"{path.name}: `slmeval {head}` is not a command "
            f"(have: {', '.join(sorted(commands))})"
        )
        command = commands[head]
        rest = tokens[1:]
        children = getattr(command, "commands", None)
        if children and rest and not rest[0].startswith("-"):
            sub = dict(children)
            assert rest[0] in sub, (
                f"{path.name}: `slmeval {head} {rest[0]}` is not a subcommand "
                f"(have: {', '.join(sorted(sub))})"
            )
            command, rest = sub[rest[0]], rest[1:]
        allowed = option_names(command)
        for token in rest:
            if not token.startswith("-"):
                continue
            name = token.split("=", 1)[0]
            assert name in allowed, (
                f"{path.name}: `slmeval {head}` has no option {name} "
                f"(have: {', '.join(sorted(allowed))})"
            )


def test_the_action_branches_on_the_documented_exit_codes() -> None:
    """Every code in the contract, and no code outside it.

    The action translates the number into a word the reviewer reads, so a
    missing arm would quietly report a regression as `unknown` and a stray one
    would be a contract this tool does not have.
    """
    text = ACTION.read_text(encoding="utf-8")
    expected = {
        exit_codes.OK: "passed",
        exit_codes.GATE_FAILED: "failed",
        exit_codes.USAGE: "usage",
        exit_codes.ENVIRONMENT: "environment",
        exit_codes.COVERAGE: "coverage",
    }
    mapped = dict(re.findall(r"^\s{10}(\d)\) verdict=([a-z]+) ;;$", text, re.MULTILINE))
    assert {int(k): v for k, v in mapped.items()} == expected

    for verdict in expected.values():
        assert f"{verdict})" in text, f"the annotation step has no arm for {verdict}"

    # 3 and 4 fail the job, and say so in words that are not "regression".
    assert "infrastructure, not a regression" in text
    assert "coverage, not a regression" in text


def test_the_cassette_cache_is_keyed_on_the_dataset_hash() -> None:
    """No prefix fallback. A stale cassette reads as a model failure."""
    text = ACTION.read_text(encoding="utf-8")
    assert "key: slmeval-cassettes-${{ runner.os }}-${{ steps.validate.outputs.dataset-hash }}" in (
        text
    )
    assert not re.search(r"^\s*restore-keys:", text, re.MULTILINE)


def test_recording_in_ci_is_opt_in() -> None:
    """`record` executes the user's real tools. It must never be the default."""
    text = ACTION.read_text(encoding="utf-8")
    assert re.search(r"  record:\n(?:.*\n)*?    default: never\n", text)
    assert "inputs.record == 'missing'" in text


def test_the_demo_workflow_asserts_both_verdicts() -> None:
    """A demo that can only go green proves the half nobody needs."""
    text = DEMO.read_text(encoding="utf-8")
    assert "verdict }}' = passed" in text
    assert "verdict }}' = failed" in text
    assert "continue-on-error: true" in text
    # No secrets, no GPU, no model server. If that ever stops being true the
    # action stops being adoptable, and this is the line that notices.
    assert "secrets." not in text


def test_the_action_needs_no_secrets() -> None:
    text = ACTION.read_text(encoding="utf-8")
    assert "secrets." not in text
    assert "--tool-mode replay" in text


# --------------------------------------------------------------------------- #
# Structure
# --------------------------------------------------------------------------- #

YAML_FILES = [*sorted((ROOT / ".github" / "workflows").glob("*.yml")), ACTION, WORKFLOW]


@pytest.mark.parametrize("path", YAML_FILES, ids=lambda p: p.name)
def test_every_shipped_yaml_file_parses(path: Path) -> None:
    """A missing colon in `action.yml` breaks every user's pipeline.

    GitHub is the only thing that really validates these, and it does so after
    a push. Parsing them here catches the class of mistake that costs a release
    - and it is the reason `pyyaml` is in the dev extra.
    """
    yaml = pytest.importorskip("yaml")
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict), f"{path.name} is not a mapping"
    if path == ACTION:
        assert document["runs"]["using"] == "composite"
        assert document["inputs"].keys() >= {"working-directory", "cache", "record", "comment"}
        assert document["outputs"].keys() >= {"exit-code", "verdict", "run-id", "dataset-hash"}
        # Every composite step either runs a shell or uses an action.
        for step in document["runs"]["steps"]:
            assert ("run" in step and step.get("shell") == "bash") or "uses" in step, step
    else:
        assert document.get("jobs"), f"{path.name} declares no jobs"


def test_the_pull_request_comment_asks_for_the_permission_it_needs() -> None:
    """A comment step with no `pull-requests: write` fails at the API call,
    after the run - the worst place to find out."""
    yaml = pytest.importorskip("yaml")
    for path in (WORKFLOW, DEMO):
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert document["permissions"]["pull-requests"] == "write", path.name


def test_the_install_name_is_the_published_one() -> None:
    """PyPI refused `slmeval` as too close to `slm-eval`, so the package is
    `slmeval-cli`. A bare `pip install slmeval` fails, or in time could install
    somebody else's package."""
    import tomllib

    yaml = pytest.importorskip("yaml")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    action = yaml.safe_load(ACTION.read_text(encoding="utf-8"))
    assert action["inputs"]["version"]["default"] == project["name"]

    bare = re.compile(r"""pip install ["']?slmeval(?![\w-])""")
    for path in [README, *GUIDES, RELEASING, CONTRIBUTING, EXAMPLE_README, JUDGE_README]:
        found = bare.search(path.read_text(encoding="utf-8"))
        assert found is None, f"{path.name} says {found.group()!r}"
