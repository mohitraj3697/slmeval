"""Load, locate and validate `eval.toml`.

A typo in the config must fail in under a second with a line-level message —
never forty minutes into a benchmark run.
"""

from __future__ import annotations

import importlib
import re
import sys
import sysconfig
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from slmeval.config.locate import locate
from slmeval.config.models import EvalConfig, ModelCfg
from slmeval.errors import ConfigError, Problem

DEFAULT_CONFIG_NAME = "eval.toml"
CACHE_DIR_NAME = ".slmeval"

_TOML_POS_RE = re.compile(r"at line (\d+), column (\d+)")


@dataclass
class LoadedConfig:
    """A validated config plus everything needed to report on it."""

    config: EvalConfig
    path: Path
    text: str

    @property
    def root(self) -> Path:
        """Directory the config lives in. All relative paths resolve against it."""
        return self.path.parent

    @property
    def cache_dir(self) -> Path:
        return self.root / CACHE_DIR_NAME

    @property
    def dataset_path(self) -> Path:
        ds = self.config.benchmark.dataset
        return ds if ds.is_absolute() else (self.root / ds).resolve()


def find_config(explicit: Path | None = None, *, start: Path | None = None) -> Path:
    """Locate `eval.toml`, walking up from ``start`` like git does."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if not path.exists():
            raise ConfigError(
                f"config file not found: {path}",
                hint="run `slmeval init` to create one",
            )
        if path.is_dir():
            candidate = path / DEFAULT_CONFIG_NAME
            if not candidate.exists():
                raise ConfigError(f"no {DEFAULT_CONFIG_NAME} in directory {path}")
            return candidate.resolve()
        return path.resolve()

    here = (start or Path.cwd()).resolve()
    for directory in (here, *here.parents):
        candidate = directory / DEFAULT_CONFIG_NAME
        if candidate.is_file():
            return candidate
    raise ConfigError(
        f"no {DEFAULT_CONFIG_NAME} found in {here} or any parent directory",
        hint="run `slmeval init` to create one, or pass --config PATH",
    )


def load_config(path: Path) -> LoadedConfig:
    """Parse and validate a config file. Raises :class:`ConfigError` on any problem."""
    path = Path(path)
    try:
        # utf-8-sig: a file saved by Notepad carries a BOM, which tomllib
        # otherwise reports as a syntax error on line 1.
        text = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError(f"could not read {path}: {exc}") from exc

    try:
        data: dict[str, Any] = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        line = None
        if m := _TOML_POS_RE.search(str(exc)):
            line = int(m.group(1))
        raise ConfigError(
            "eval.toml is not valid TOML",
            source=path,
            problems=[Problem(path="<syntax>", message=str(exc), line=line)],
        ) from exc

    try:
        config = EvalConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(
            f"eval.toml is invalid ({exc.error_count()} "
            f"problem{'s' if exc.error_count() != 1 else ''})",
            source=path,
            problems=_problems_from(exc, text),
        ) from exc

    loaded = LoadedConfig(config=config, path=path, text=text)
    _settle_fake_scripts(loaded)
    return loaded


def _settle_fake_scripts(loaded: LoadedConfig) -> None:
    """Validate every `fake` script, and inline the fixture files it points at.

    Two things happen here, both so that a scripted model behaves like a
    configured one rather than a special case.

    A ``script_file`` is read now and its table merged into the candidate's
    ``extra_body``, then the key is dropped. That resolves the path against the
    directory holding ``eval.toml`` - ``slmeval run -c ../other/eval.toml`` has
    to mean what running from that directory means - and it puts the fixture's
    *content* into :meth:`EvalConfig.config_hash` instead of an absolute path,
    so the same benchmark hashes the same on two machines and an edited fixture
    is correctly a different benchmark.

    And every script is parsed here, while we still have the file and its line
    numbers. Left to the provider, a typo would surface as a ``ValueError``
    raised inside slmeval midway through a matrix, which the outcome classifier
    reads - correctly, by its own rules - as a harness bug.
    """
    from slmeval.providers.fake import load_script_file, validate_extra_body

    for loc, cfg in _model_locations(loaded.config):
        if cfg.provider != "fake":
            continue
        extra = cfg.extra_body
        key = "script_file" if "script_file" in extra else "script"
        try:
            if "script_file" in extra:
                if "script" in extra:
                    raise ValueError(
                        "set extra_body.script or extra_body.script_file, not both - "
                        "two scripts for one model has no defined order"
                    )
                fixture = Path(str(extra.pop("script_file"))).expanduser()
                if not fixture.is_absolute():
                    fixture = (loaded.root / fixture).resolve()
                table = load_script_file(fixture)
                # Inline entries win over the fixture's, per case, so a file
                # holding the whole dataset plus one override reads the way it
                # looks.
                by_case = dict(table.get("script_by_case") or {})
                by_case.update(extra.get("script_by_case") or {})
                if by_case:
                    extra["script_by_case"] = by_case
                for name in ("script", "unsupported"):
                    if name in table and name not in extra:
                        extra[name] = table[name]
            validate_extra_body(extra)
        except ValueError as exc:
            raise ConfigError(
                f"{cfg.id}: invalid script for provider 'fake'",
                source=loaded.path,
                problems=[
                    Problem(
                        path=format_loc((*loc, "extra_body", key)),
                        message=str(exc),
                        # Either spelling: a dotted `extra_body.script = ...`
                        # line, or a `[baseline.extra_body]` table.
                        line=(
                            locate(loaded.text, (*loc, "extra_body", key))
                            or locate(loaded.text, (*loc, f"extra_body.{key}"))
                        ),
                    )
                ],
                hint=(
                    "a fake script is a list of turns: "
                    "extra_body.script = [{ kind = \"fenced_json\", text = '{}' }]"
                ),
            ) from exc


def _model_locations(config: EvalConfig) -> list[tuple[tuple[str | int, ...], ModelCfg]]:
    """Every model with the config path it was written at, for error messages."""
    out: list[tuple[tuple[str | int, ...], ModelCfg]] = [(("baseline",), config.baseline)]
    out += [(("candidates", i), cfg) for i, cfg in enumerate(config.candidates)]
    return out


def _problems_from(exc: ValidationError, text: str) -> list[Problem]:
    problems: list[Problem] = []
    for err in exc.errors():
        loc = err["loc"]
        message = err["msg"]
        # Pydantic prefixes custom ValueError messages; the prefix adds nothing
        # for a user reading their own config file.
        message = message.removeprefix("Value error, ")
        if err["type"] == "extra_forbidden":
            message = "unknown key (check the spelling, or remove it)"
        problems.append(
            Problem(
                path=format_loc(loc),
                message=message,
                line=locate(text, tuple(loc)),
                value=_short(err.get("input")),
            )
        )
    return problems


def format_loc(loc: tuple[int | str, ...]) -> str:
    """``("candidates", 0, "provider")`` -> ``candidates[0].provider``."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out or "config"


def _short(value: Any, limit: int = 60) -> str | None:
    if value is None or isinstance(value, dict):
        return None
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "..."


def require_dataset(loaded: LoadedConfig) -> Path:
    """Resolve the dataset path, erroring clearly if it is not there."""
    path = loaded.dataset_path
    if not path.exists():
        raise ConfigError(
            f"dataset not found: {path}",
            source=loaded.path,
            problems=[
                Problem(
                    path="benchmark.dataset",
                    message=f"resolved to {path} (relative to {loaded.root})",
                    line=locate(loaded.text, ("benchmark", "dataset")),
                )
            ],
            hint="create the file, or point benchmark.dataset at an existing .jsonl",
        )
    return path


def _within(path: Path | str, root: Path) -> bool:
    try:
        return Path(path).resolve().is_relative_to(root.resolve())
    except (OSError, ValueError):  # pragma: no cover - unresolvable path
        return False


def _is_installed(origin: str) -> bool:
    """Did this module come from the standard library or site-packages?"""
    locations = sysconfig.get_paths()
    for key in ("purelib", "platlib", "stdlib", "platstdlib"):
        location = locations.get(key)
        if location and _within(origin, Path(location)):
            return True
    return False


_injected_roots: set[str] = set()


def _prepare_sys_path(root: str) -> None:
    """Make the active project's root importable, and only that one.

    The agent module lives next to eval.toml, which is not necessarily the
    working directory, so the root has to go on `sys.path`. But `sys.path` is
    global and append-only: left alone, a second config loaded in the same
    process could still import the first project's agent and silently benchmark
    the wrong code. So each resolve withdraws the roots a previous resolve
    injected before adding its own.
    """
    for previous in _injected_roots:
        while previous in sys.path:
            sys.path.remove(previous)
    _injected_roots.clear()

    while root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    _injected_roots.add(root)


def _import_target_module(module_name: str, root: Path) -> Any:
    """Import a module, preferring the copy that belongs to this project.

    `sys.modules` is global and sticky: once any module named `slmeval_target`
    has been imported, a later import of that name returns the first one no
    matter which directory we are now pointed at. That silently validates a
    target that no longer exists, and in a monorepo it would benchmark the wrong
    agent. So when the module we get back is neither inside this project nor an
    installed package, drop it and resolve once more from scratch.
    """
    importlib.invalidate_caches()
    module = importlib.import_module(module_name)
    origin = getattr(module, "__file__", None)

    if origin is None or _within(origin, root) or _is_installed(origin):
        return module

    stale = {
        name: mod
        for name, mod in list(sys.modules.items())
        if name == module_name or name.startswith(f"{module_name}.")
    }
    for name in stale:
        del sys.modules[name]

    importlib.invalidate_caches()
    try:
        return importlib.import_module(module_name)
    except ImportError:
        # Nothing on the path can provide it any more. Put back what we removed
        # so we do not corrupt the interpreter on the way out.
        sys.modules.update(stale)
        raise


def resolve_entrypoint(loaded: LoadedConfig) -> Any:
    """Import the user's agent callable.

    Only used by `validate --strict` and `run`: importing the target can be slow
    and can execute arbitrary module-level code, so plain validation never does it.
    """
    target = loaded.config.target
    _prepare_sys_path(str(loaded.root))

    try:
        module = _import_target_module(target.module, loaded.root)
    except ImportError as exc:
        raise ConfigError(
            f"could not import {target.module!r} from {loaded.root}",
            source=loaded.path,
            problems=[Problem(path="target.entrypoint", message=str(exc))],
            hint="check the module path, and that its own imports are installed",
        ) from exc

    try:
        func = getattr(module, target.attr)
    except AttributeError as exc:
        available = ", ".join(n for n in dir(module) if not n.startswith("_")) or "nothing"
        raise ConfigError(
            f"{target.module!r} has no attribute {target.attr!r}",
            source=loaded.path,
            problems=[Problem(path="target.entrypoint", message=f"module defines: {available}")],
        ) from exc

    if not callable(func):
        raise ConfigError(
            f"{target.entrypoint} is not callable (it is a {type(func).__name__})",
            source=loaded.path,
        )
    return func
