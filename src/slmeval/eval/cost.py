"""What a run cost, and what a local model would have cost.

Two numbers that are not the same kind of number, kept visibly apart.

**Cloud cost is measured.** Tokens reported by the provider, multiplied by a
price. The prices live in a table with an ``as_of`` date and can be overridden
from ``prices.toml``, because prices change and a hardcoded table that has
quietly gone stale is not an approximation - it is a wrong answer printed with
confidence.

**Local cost is modelled.** A model on your own GPU bills nothing, so the
direct cost is zero - true, and useless on its own: it makes every local
candidate look infinitely cheap and the Pareto view meaningless. So there is
also ``shadow_cost = wall_seconds * gpu_hourly_rate / 3600``, with a rate the
user sets, and every report that prints it says *modelled*. A modelled number
labelled as one is useful. The same number in a column called "cost" is a lie
with a decimal point.

The third case matters most for trust: **a model we have no price for costs
``None``, not zero.** An unpriced model silently totalling zero would make a
candidate look free, and "free" is the most persuasive cell in the table.
Unknown stays unknown, the report marks it, and the user is told which model to
price.
"""

from __future__ import annotations

import tomllib
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from slmeval.config.models import ModelCfg
from slmeval.errors import ConfigError, Problem

PRICES_FILE: Final = "prices.toml"

PRICES_AS_OF: Final = "2026-05-01"
"""When the built-in table was last checked.

Printed in the report footer. A user comparing a cost column against their
actual invoice needs to know which month's prices produced it, and a table with
no date cannot be audited - or corrected.
"""

#: USD per **million** tokens, ``(prompt, completion)``.
#:
#: Deliberately short. Every entry is a claim about someone's billing page, and
#: a long table is a long list of things to go stale. Anything missing is
#: priced from ``prices.toml``, and anything still missing stays unpriced
#: rather than guessed.
BUILTIN_PRICES: Final[dict[str, tuple[float, float]]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "claude-opus-4": (15.00, 75.00),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (0.80, 4.00),
}

LOCAL_PROVIDERS: Final = frozenset({"ollama", "openai_compatible"})
"""Providers that bill nothing per token.

``openai_compatible`` is the awkward one - it covers a llama.cpp server on
localhost *and* a hosted gateway - so an explicit price entry for the model
always wins over the local assumption.
"""

FREE_PROVIDERS: Final = frozenset({"fake"})
"""The scripted provider costs nothing and is not modelled either: a shadow
cost for a provider that never ran a GPU would be fabricated."""


@dataclass(frozen=True, slots=True)
class Priced:
    """One execution's cost, with its provenance attached.

    ``usd`` and ``shadow_usd`` are separate fields rather than one sum, because
    adding a measured cost to a modelled one produces a figure that cannot be
    checked against anything.
    """

    usd: float | None = None
    """Measured spend. ``None`` means "no price for this model" - never zero."""

    shadow_usd: float | None = None
    """Modelled GPU cost for a local model. ``None`` when no rate was set."""

    estimated: bool = False
    """True when any token count behind ``usd`` was estimated rather than
    reported by the provider. Aggregates containing one are marked ``~``."""

    local: bool = False

    unpriced: tuple[str, ...] = ()
    """Models that had no price. The report names them, so "why is this column
    empty" has an answer the user can act on."""

    @property
    def known(self) -> bool:
        return self.usd is not None


@dataclass(frozen=True, slots=True)
class PriceTable:
    """Token prices, plus the GPU rate used to model local runs."""

    prices: dict[str, tuple[float, float]] = field(
        default_factory=lambda: dict(BUILTIN_PRICES)
    )
    gpu_hourly_usd: float | None = None
    as_of: str = PRICES_AS_OF
    source: str = "built-in"

    @classmethod
    def load(
        cls,
        root: Path | str | None = None,
        *,
        gpu_hourly_usd: float | None = None,
    ) -> PriceTable:
        """The built-in table, overlaid with ``prices.toml`` if the project has one.

        Overlaid rather than replaced: a user correcting one model's price
        should not have to retype the whole table, and a file that replaced it
        would silently un-price everything they left out.
        """
        table = cls(gpu_hourly_usd=gpu_hourly_usd)
        if root is None:
            return table
        path = Path(root) / PRICES_FILE
        if not path.is_file():
            return table
        return table.overlay(path)

    def overlay(self, path: Path) -> PriceTable:
        """Merge a ``prices.toml``::

            as_of = "2026-06-01"
            gpu_hourly_usd = 0.80

            [prices]
            "gpt-4o" = { input = 2.50, output = 10.00 }
            "my-finetune" = { input = 1.00, output = 3.00 }
        """
        try:
            data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8-sig"))
        except OSError as exc:
            raise ConfigError(f"could not read {path}: {exc}") from exc
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(
                f"{path.name} is not valid TOML",
                source=path,
                problems=[Problem(path="<syntax>", message=str(exc))],
            ) from exc

        merged = dict(self.prices)
        raw = data.get("prices", {})
        if not isinstance(raw, dict):
            raise ConfigError(f"{path.name}: [prices] must be a table of model names")
        for model, entry in raw.items():
            merged[str(model)] = _read_price(path, str(model), entry)

        rate = data.get("gpu_hourly_usd", self.gpu_hourly_usd)
        if rate is not None and (not isinstance(rate, int | float) or rate < 0):
            raise ConfigError(
                f"{path.name}: gpu_hourly_usd must be a number >= 0, got {rate!r}"
            )

        return PriceTable(
            prices=merged,
            gpu_hourly_usd=float(rate) if rate is not None else None,
            as_of=str(data.get("as_of", self.as_of)),
            source=str(path),
        )

    def price_for(self, model: str) -> tuple[float, float] | None:
        """Prices for a model name, trying the exact name then its family.

        Providers version models in the name - ``gpt-4o-2026-01-15`` - and a
        table keyed only on exact names would un-price a run the day a user
        pins a snapshot. Longest matching prefix wins, so a specific entry
        always beats the family it belongs to.
        """
        exact = self.prices.get(model)
        if exact is not None:
            return exact
        best: tuple[int, tuple[float, float]] | None = None
        for name, price in self.prices.items():
            if model.startswith(name) and (best is None or len(name) > best[0]):
                best = (len(name), price)
        return best[1] if best else None

    def cost_of(
        self,
        model_cfg: ModelCfg,
        *,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        estimated: bool = False,
        wall_seconds: float = 0.0,
    ) -> Priced:
        """Price one model's share of an execution.

        An explicit price always wins, whatever the provider. That ordering is
        what makes two things possible: reaching a hosted gateway through
        ``openai_compatible``, which otherwise looks local and free; and
        pricing a scripted ``fake`` run, so "what would this swap have cost on
        gpt-4o" is answerable offline instead of needing the real spend.
        """
        price = self.price_for(model_cfg.model)
        if price is None and model_cfg.provider in FREE_PROVIDERS:
            return Priced(usd=0.0)
        if price is None and model_cfg.provider in LOCAL_PROVIDERS:
            return Priced(
                usd=0.0,
                shadow_usd=self.shadow_cost(wall_seconds),
                estimated=estimated,
                local=True,
            )
        if price is None:
            return Priced(usd=None, estimated=estimated, unpriced=(model_cfg.model,))

        per_in, per_out = price
        usd = (prompt_tokens or 0) / 1_000_000 * per_in
        usd += (completion_tokens or 0) / 1_000_000 * per_out
        return Priced(usd=usd, estimated=estimated)

    def shadow_cost(self, wall_seconds: float) -> float | None:
        """Modelled GPU cost. ``None`` until the user sets a rate.

        No default rate, on purpose. A made-up $/hour would propagate into a
        published comparison as though it were measured, and there is no
        defensible default across a laptop, a 4090 and an H100.
        """
        if self.gpu_hourly_usd is None:
            return None
        return max(0.0, wall_seconds) * self.gpu_hourly_usd / 3600.0


def _read_price(path: Path, model: str, entry: Any) -> tuple[float, float]:
    if isinstance(entry, dict):
        try:
            return float(entry["input"]), float(entry["output"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"{path.name}: price for {model!r} needs numeric 'input' and "
                "'output' (USD per million tokens)"
            ) from exc
    if isinstance(entry, list | tuple) and len(entry) == 2:
        return float(entry[0]), float(entry[1])
    raise ConfigError(
        f"{path.name}: price for {model!r} must be "
        "{ input = 2.50, output = 10.00 } or [2.50, 10.00]"
    )


def total(costs: Sequence[Priced]) -> Priced:
    """Sum a run's costs, keeping "unknown" from collapsing into zero.

    If any execution was unpriced the total is ``None``: a total that quietly
    omitted the three most expensive candidates would read as a cheap run, and
    that is the one direction a cost figure must never be wrong in.
    """
    if not costs:
        return Priced(usd=0.0)
    unpriced = tuple(dict.fromkeys(m for c in costs for m in c.unpriced))
    known = [c.usd for c in costs if c.usd is not None]
    shadows = [c.shadow_usd for c in costs if c.shadow_usd is not None]
    return Priced(
        usd=sum(known) if not unpriced else None,
        shadow_usd=sum(shadows) if shadows else None,
        estimated=any(c.estimated for c in costs),
        local=all(c.local for c in costs),
        unpriced=unpriced,
    )


__all__ = [
    "BUILTIN_PRICES",
    "FREE_PROVIDERS",
    "LOCAL_PROVIDERS",
    "PRICES_AS_OF",
    "PRICES_FILE",
    "PriceTable",
    "Priced",
    "total",
]
