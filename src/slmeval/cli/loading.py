"""Getting from a command-line invocation to a loaded project.

Four commands need the same five things - config, dataset, target, trace store,
cassette store - and they have to fail in the same order every time: config
before dataset, dataset before importing the user's agent. A typo in
``eval.toml`` must not be reported as an import error, and importing the user's
module before validating their dataset makes a trivial mistake arrive as a
stack trace from somebody else's code.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from slmeval.config.loader import LoadedConfig, find_config, load_config
from slmeval.config.models import EvalConfig
from slmeval.core.adapter import Target, load_target
from slmeval.dataset.loader import Dataset, load_dataset
from slmeval.errors import ConfigError, DatasetError
from slmeval.eval.cost import PriceTable
from slmeval.sandbox.store import CassetteStore
from slmeval.storage.db import RunStore


@dataclass(slots=True)
class Project:
    """Everything a run needs, opened and ready."""

    loaded: LoadedConfig
    dataset: Dataset
    target: Target
    store: RunStore
    cassettes: CassetteStore
    prices: PriceTable

    @property
    def config(self) -> EvalConfig:
        return self.loaded.config

    @property
    def root(self) -> Path:
        return self.loaded.root

    def close(self) -> None:
        # Suppressed individually: a failure closing one store must not leave
        # the other open, and neither is worth masking the error that brought
        # us here.
        with contextlib.suppress(Exception):
            self.cassettes.close()
        with contextlib.suppress(Exception):
            self.store.close()


@contextlib.contextmanager
def open_project(
    config_path: Path | None = None,
    *,
    gpu_hourly_usd: float | None = None,
    load_agent: bool = True,
) -> Iterator[Project]:
    """Open a project, closing both stores on the way out.

    ``load_agent=False`` skips importing the user's module, for commands that
    only read stored results. ``slmeval report`` must work on a checkout whose
    agent no longer imports - the results are still valid, and refusing to
    print them would be the tool losing data it already has.
    """
    path = find_config(config_path)
    loaded = load_config(path)
    dataset = _dataset(loaded)

    target = (
        load_target(
            loaded.config.target.entrypoint,
            adapter=loaded.config.target.adapter,
            root=loaded.root,
            # Resolved against eval.toml, not the working directory: the same
            # command has to mean the same thing run from the project root and
            # from two levels down.
            extra_paths=[
                p if p.is_absolute() else (loaded.root / p).resolve()
                for p in loaded.config.target.sys_path
            ],
        )
        if load_agent
        else Target(entrypoint=loaded.config.target.entrypoint, func=_unloaded, is_async=False)
    )

    store = RunStore.for_project(loaded.root)
    cassettes = CassetteStore.for_project(loaded.root)
    project = Project(
        loaded=loaded,
        dataset=dataset,
        target=target,
        store=store,
        cassettes=cassettes,
        prices=PriceTable.load(loaded.root, gpu_hourly_usd=gpu_hourly_usd),
    )
    try:
        yield project
    finally:
        project.close()


def _dataset(loaded: LoadedConfig) -> Dataset:
    path = loaded.dataset_path
    if not path.is_file():
        raise DatasetError(
            f"dataset not found: {path}",
            hint=(
                f"[benchmark].dataset in {loaded.path.name} points at "
                f"{loaded.config.benchmark.dataset}, resolved against {loaded.root}"
            ),
        )
    return load_dataset(path)


def _unloaded(_: object) -> object:  # pragma: no cover - never called
    raise ConfigError("the agent was not loaded for this command")


__all__ = ["Project", "open_project"]
