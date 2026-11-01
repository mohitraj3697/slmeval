"""Cost: measured cloud spend, modelled local cost, and "we do not know".

The test this file exists for is
``test_an_unpriced_model_costs_none_not_zero``. An unpriced model silently
totalling zero would make a candidate look free, and "free" is the most
persuasive cell in the table - so unknown has to stay unknown all the way
through the aggregation.
"""

from __future__ import annotations

import pytest

from slmeval.config.models import ModelCfg
from slmeval.errors import ConfigError
from slmeval.eval.cost import (
    PRICES_AS_OF,
    Priced,
    PriceTable,
    total,
)


def cfg(provider: str, model: str, **kwargs: object) -> ModelCfg:
    return ModelCfg(id="c", provider=provider, model=model, **kwargs)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Lookup
# --------------------------------------------------------------------------- #


def test_an_exact_model_name_is_priced() -> None:
    table = PriceTable()
    assert table.price_for("gpt-4o") == (2.50, 10.00)


def test_a_pinned_snapshot_falls_back_to_its_family() -> None:
    """Providers version models in the name, and a table keyed only on exact
    names would un-price a run the day a user pins a snapshot."""
    table = PriceTable()
    assert table.price_for("gpt-4o-2026-01-15") == (2.50, 10.00)


def test_the_longest_matching_prefix_wins() -> None:
    """`gpt-4o-mini` must not be priced as `gpt-4o`, which is 16x more."""
    table = PriceTable()
    assert table.price_for("gpt-4o-mini") == (0.15, 0.60)
    assert table.price_for("gpt-4o-mini-2026-01-15") == (0.15, 0.60)


def test_an_unknown_model_has_no_price() -> None:
    assert PriceTable().price_for("gemma3:4b") is None


# --------------------------------------------------------------------------- #
# Costing one execution
# --------------------------------------------------------------------------- #


def test_cloud_cost_is_tokens_times_price() -> None:
    priced = PriceTable().cost_of(
        cfg("openai", "gpt-4o"), prompt_tokens=1000, completion_tokens=500
    )
    # 1000/1e6 * 2.50 + 500/1e6 * 10.00
    assert priced.usd == pytest.approx(0.0025 + 0.005)
    assert priced.known
    assert not priced.local


def test_an_unpriced_model_costs_none_not_zero() -> None:
    priced = PriceTable().cost_of(
        cfg("openai", "some-new-model"), prompt_tokens=1000, completion_tokens=1000
    )
    assert priced.usd is None
    assert not priced.known
    assert priced.unpriced == ("some-new-model",)


def test_a_local_model_bills_nothing_and_is_modelled_separately() -> None:
    table = PriceTable(gpu_hourly_usd=3600.0)  # $1/second, to make the maths plain
    priced = table.cost_of(
        cfg("ollama", "gemma3:4b"),
        prompt_tokens=100,
        completion_tokens=100,
        wall_seconds=2.0,
    )
    assert priced.usd == 0.0
    assert priced.local
    assert priced.shadow_usd == pytest.approx(2.0)


def test_shadow_cost_needs_a_rate_the_user_set() -> None:
    """No default $/hour. A made-up rate would propagate into a published
    comparison as though it were measured, and there is no defensible default
    across a laptop, a 4090 and an H100."""
    priced = PriceTable().cost_of(
        cfg("ollama", "gemma3:4b"),
        prompt_tokens=1,
        completion_tokens=1,
        wall_seconds=10.0,
    )
    assert priced.usd == 0.0
    assert priced.shadow_usd is None


def test_the_scripted_provider_is_free_and_not_modelled() -> None:
    """A shadow cost for a provider that never ran a GPU would be fabricated."""
    priced = PriceTable(gpu_hourly_usd=1.0).cost_of(
        cfg("fake", "scripted"), prompt_tokens=100, completion_tokens=100, wall_seconds=5
    )
    assert priced.usd == 0.0
    assert priced.shadow_usd is None


def test_an_explicit_price_beats_the_free_and_local_assumptions() -> None:
    """A hosted gateway reached through `openai_compatible` looks local, and a
    scripted run priced on purpose answers "what would this have cost"."""
    table = PriceTable(prices={"gemma3:4b": (1.0, 2.0), "scripted": (1.0, 1.0)})
    hosted = table.cost_of(
        cfg("openai_compatible", "gemma3:4b", base_url="https://gateway.example.com/v1"),
        prompt_tokens=1_000_000,
        completion_tokens=0,
    )
    assert hosted.usd == pytest.approx(1.0)
    assert not hosted.local

    scripted = table.cost_of(
        cfg("fake", "scripted"), prompt_tokens=1_000_000, completion_tokens=0
    )
    assert scripted.usd == pytest.approx(1.0)


def test_estimated_tokens_are_flagged_through_to_the_cost() -> None:
    priced = PriceTable().cost_of(
        cfg("openai", "gpt-4o"),
        prompt_tokens=10,
        completion_tokens=10,
        estimated=True,
    )
    assert priced.estimated


# --------------------------------------------------------------------------- #
# Totals
# --------------------------------------------------------------------------- #


def test_a_total_containing_an_unpriced_model_is_unknown() -> None:
    """A total that quietly omitted the expensive executions would read as a
    cheap run, which is the one direction a cost figure must never be wrong
    in."""
    rolled = total([Priced(usd=1.0), Priced(usd=None, unpriced=("x",))])
    assert rolled.usd is None
    assert rolled.unpriced == ("x",)


def test_a_total_sums_what_it_knows() -> None:
    rolled = total([Priced(usd=1.0), Priced(usd=0.5)])
    assert rolled.usd == pytest.approx(1.5)


def test_a_total_keeps_shadow_cost_apart_from_measured_cost() -> None:
    """Adding a measured cost to a modelled one produces a figure that cannot
    be checked against anything."""
    rolled = total([Priced(usd=0.0, shadow_usd=0.2), Priced(usd=0.0, shadow_usd=0.3)])
    assert rolled.usd == 0.0
    assert rolled.shadow_usd == pytest.approx(0.5)


def test_a_total_inherits_the_estimate_flag() -> None:
    rolled = total([Priced(usd=1.0), Priced(usd=1.0, estimated=True)])
    assert rolled.estimated


def test_an_empty_total_is_zero() -> None:
    assert total([]).usd == 0.0


# --------------------------------------------------------------------------- #
# prices.toml
# --------------------------------------------------------------------------- #


def test_a_prices_file_overlays_rather_than_replaces(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A user correcting one price should not have to retype the table, and a
    file that replaced it would silently un-price everything left out."""
    (tmp_path / "prices.toml").write_text(
        """
as_of = "2026-06-01"
gpu_hourly_usd = 0.80

[prices]
"gpt-4o" = { input = 3.00, output = 12.00 }
"my-finetune" = [1.0, 3.0]
""",
        encoding="utf-8",
    )
    table = PriceTable.load(tmp_path)
    assert table.price_for("gpt-4o") == (3.00, 12.00)
    assert table.price_for("my-finetune") == (1.0, 3.0)
    # Untouched entries survive.
    assert table.price_for("gpt-4o-mini") == (0.15, 0.60)
    assert table.as_of == "2026-06-01"
    assert table.gpu_hourly_usd == 0.80
    assert "prices.toml" in table.source


def test_no_prices_file_means_the_built_in_table(tmp_path) -> None:  # type: ignore[no-untyped-def]
    table = PriceTable.load(tmp_path, gpu_hourly_usd=1.5)
    assert table.as_of == PRICES_AS_OF
    assert table.source == "built-in"
    assert table.gpu_hourly_usd == 1.5


def test_a_malformed_price_is_a_readable_config_error(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "prices.toml").write_text(
        '[prices]\n"x" = { input = 1.0 }\n', encoding="utf-8"
    )
    with pytest.raises(ConfigError, match="numeric 'input' and 'output'"):
        PriceTable.load(tmp_path)


def test_a_negative_gpu_rate_is_refused(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "prices.toml").write_text("gpu_hourly_usd = -1\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="must be a number >= 0"):
        PriceTable.load(tmp_path)


def test_unparseable_toml_names_the_file(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "prices.toml").write_text("not = = toml\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid TOML"):
        PriceTable.load(tmp_path)


def test_the_as_of_date_is_carried_so_a_cost_column_can_be_audited() -> None:
    """A table with no date cannot be checked against an invoice - or
    corrected."""
    assert PriceTable().as_of == PRICES_AS_OF
    assert PRICES_AS_OF.count("-") == 2
