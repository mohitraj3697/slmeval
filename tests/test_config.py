"""Config schema and loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from slmeval.config import find_config, load_config, require_dataset
from slmeval.config.models import EvalConfig, ModelCfg
from slmeval.errors import ConfigError
from tests.conftest import MINIMAL_CONFIG, write_config


def test_scaffolded_config_is_valid(project: Path) -> None:
    """`init` must produce something `validate` accepts with zero setup.
    If this breaks, the five-minute cold start breaks."""
    loaded = load_config(project / "eval.toml")
    assert loaded.config.benchmark.name == "test-benchmark"
    assert require_dataset(loaded).exists()


def test_duplicate_candidate_id_rejected(tmp_path: Path) -> None:
    body = MINIMAL_CONFIG + """
[[candidates]]
id = "dup"
provider = "fake"
model = "a"

[[candidates]]
id = "dup"
provider = "fake"
model = "b"
"""
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    assert "duplicate candidate id 'dup'" in str(exc.value.problems[0].message)


def test_candidate_cannot_shadow_baseline_id(tmp_path: Path) -> None:
    body = MINIMAL_CONFIG + """
[[candidates]]
id = "baseline"
provider = "fake"
model = "a"
"""
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    assert "collides with the baseline" in exc.value.problems[0].message


def test_unknown_provider_names_the_field(tmp_path: Path) -> None:
    body = MINIMAL_CONFIG.replace('provider = "fake"', 'provider = "ollamaa"')
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    problem = exc.value.problems[0]
    assert problem.path == "baseline.provider"
    assert problem.line is not None


def test_out_of_range_fault_rate_rejected(tmp_path: Path) -> None:
    body = MINIMAL_CONFIG.replace('name = "t"', 'name = "t"\nfault_injection_rate = 1.5')
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    assert exc.value.problems[0].path == "benchmark.fault_injection_rate"


def test_unknown_key_is_an_error_not_a_shrug(tmp_path: Path) -> None:
    """A silently ignored key makes someone think they configured something."""
    body = MINIMAL_CONFIG + (
        '\n[[candidates]]\nid = "c"\nprovider = "fake"\nmodel = "m"\nfault_rate = 0.3\n'
    )
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    problem = exc.value.problems[0]
    assert problem.path == "candidates[0].fault_rate"
    assert "unknown key" in problem.message


def test_toml_syntax_error_reports_a_line(tmp_path: Path) -> None:
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, "[benchmark]\nname = \n"))
    assert exc.value.problems[0].line == 2


def test_bom_prefixed_file_loads(tmp_path: Path) -> None:
    """A config saved by Notepad carries a BOM; tomllib alone chokes on it."""
    path = write_config(tmp_path, MINIMAL_CONFIG)
    path.write_bytes(b"\xef\xbb\xbf" + MINIMAL_CONFIG.encode("utf-8"))
    assert load_config(path).config.benchmark.name == "t"


def test_missing_dataset_is_located(tmp_path: Path) -> None:
    body = MINIMAL_CONFIG.replace("./datasets/cases.jsonl", "./datasets/nope.jsonl")
    loaded = load_config(write_config(tmp_path, body))
    with pytest.raises(ConfigError) as exc:
        require_dataset(loaded)
    assert exc.value.problems[0].path == "benchmark.dataset"


def test_dataset_path_is_relative_to_the_config_not_cwd(tmp_path: Path, monkeypatch) -> None:
    loaded = load_config(write_config(tmp_path, MINIMAL_CONFIG))
    monkeypatch.chdir(tmp_path.parent)
    assert require_dataset(loaded).exists()


def test_find_config_walks_up_parents(tmp_path: Path, monkeypatch) -> None:
    write_config(tmp_path, MINIMAL_CONFIG)
    nested = tmp_path / "src" / "deep"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    assert find_config() == tmp_path / "eval.toml"


def test_find_config_without_one_suggests_init(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ConfigError) as exc:
        find_config()
    assert "slmeval init" in (exc.value.hint or "")


# -- provider defaults -----------------------------------------------------


def test_ollama_gets_a_default_base_url() -> None:
    model = ModelCfg(id="c", provider="ollama", model="gemma3:4b")
    assert model.base_url == "http://localhost:11434/v1"


def test_openai_gets_a_default_key_env_but_never_a_key() -> None:
    model = ModelCfg(id="c", provider="openai", model="gpt-4o")
    assert model.api_key_env == "OPENAI_API_KEY"
    assert "api_key" not in model.model_dump()


def test_openai_compatible_requires_a_base_url() -> None:
    with pytest.raises(ValueError, match="requires base_url"):
        ModelCfg(id="c", provider="openai_compatible", model="m")


def test_endpoint_url_is_rejected_with_an_explanation() -> None:
    with pytest.raises(ValueError, match="points at an endpoint"):
        ModelCfg(
            id="c",
            provider="openai_compatible",
            model="m",
            base_url="http://localhost:8000/v1/chat/completions",
        )


def test_trailing_slash_is_normalised() -> None:
    model = ModelCfg(id="c", provider="openai_compatible", model="m", base_url="http://x:1/v1/")
    assert model.base_url == "http://x:1/v1"


@pytest.mark.parametrize("bad", ["Upper", "1 space", "-leading", "a" * 65, ""])
def test_invalid_candidate_ids_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        ModelCfg(id=bad, provider="fake", model="m")


@pytest.mark.parametrize("bad", ["nocolon", "mod:", ":fn", "mod:fn:extra", "1mod:fn"])
def test_invalid_entrypoints_rejected(tmp_path: Path, bad: str) -> None:
    body = MINIMAL_CONFIG.replace('entrypoint = "mod:run_case"', f'entrypoint = "{bad}"')
    with pytest.raises(ConfigError) as exc:
        load_config(write_config(tmp_path, body))
    assert exc.value.problems[0].path == "target.entrypoint"


def test_duplicate_target_nodes_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate node names"):
        ModelCfg(id="c", provider="fake", model="m", target_nodes=["a", "a"])


# -- provenance ------------------------------------------------------------


def _config(**benchmark_overrides: object) -> EvalConfig:
    data: dict = {
        "benchmark": {"name": "t", "dataset": "d.jsonl", **benchmark_overrides},
        "target": {"entrypoint": "m:f"},
        "baseline": {"provider": "fake", "model": "m"},
    }
    return EvalConfig.model_validate(data)


def test_config_hash_ignores_formatting_and_dataset_path() -> None:
    """The same benchmark checked out at a different path is the same benchmark."""
    assert _config().config_hash() == _config(dataset="elsewhere/d.jsonl").config_hash()


def test_config_hash_changes_with_a_setting_that_affects_results() -> None:
    assert _config().config_hash() != _config(seed=7).config_hash()
    assert _config().config_hash() != _config(runs_per_case=5).config_hash()


def test_a_judge_nobody_configured_does_not_change_the_hash() -> None:
    """Adding an off-by-default section must not invalidate every committed
    baseline in existence: `gate --max-regression` refuses to compare across a
    config-hash change, so a gratuitous change here breaks CI for a feature
    its owner never turned on."""
    data: dict = {
        "benchmark": {"name": "t", "dataset": "d.jsonl"},
        "target": {"entrypoint": "m:f"},
        "baseline": {"provider": "fake", "model": "m"},
    }
    without = EvalConfig.model_validate(data).config_hash()
    explicit_off = EvalConfig.model_validate(
        {**data, "judge": {"provider": "openai", "model": ""}}
    ).config_hash()
    assert without == explicit_off


def test_an_enabled_judge_does_change_the_hash() -> None:
    """A pass rate that includes a judge's verdicts is a different
    measurement, and two runs judged by different models are not comparable."""
    data: dict = {
        "benchmark": {"name": "t", "dataset": "d.jsonl"},
        "target": {"entrypoint": "m:f"},
        "baseline": {"provider": "fake", "model": "m"},
    }
    off = EvalConfig.model_validate(data).config_hash()
    on = EvalConfig.model_validate(
        {**data, "judge": {"provider": "openai", "model": "gpt-4o-mini"}}
    ).config_hash()
    other = EvalConfig.model_validate(
        {**data, "judge": {"provider": "openai", "model": "gpt-4o"}}
    ).config_hash()
    assert off != on != other
    assert off != other
