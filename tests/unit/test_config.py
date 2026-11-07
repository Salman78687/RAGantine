"""Config loading, validation and YAML round-trip (SPEC §B2)."""

from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from ragantine.config import (
    AppConfig,
    ModelsConfig,
    Settings,
    dump_config,
    load_config,
    load_models_config,
)
from ragantine.schemas import SourceTier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "configs" / "default.yaml"
MODELS = ROOT / "configs" / "models.yaml"


def _raw(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = yaml.safe_load(path.read_text())
    return data


def test_default_config_loads_with_spec_values() -> None:
    cfg = load_config(DEFAULT)
    assert cfg.seed == 1337
    assert cfg.corpus.n_articles == 5000
    assert cfg.corpus.tier_mix[SourceTier.CURATED] == pytest.approx(0.70)
    assert cfg.retrieval.fused_k == 20 and cfg.retrieval.final_k == 5
    assert cfg.defenses.enabled == []
    assert cfg.defenses.d4.mode == "delimit_and_datamark"
    assert cfg.bench.seeds == [1337, 2024, 7]


def test_config_round_trip(tmp_path: Path) -> None:
    cfg = load_config(DEFAULT)
    out = tmp_path / "run" / "config.yaml"
    dump_config(cfg, out)
    assert load_config(out) == cfg


def test_models_config_round_trip(tmp_path: Path) -> None:
    models = load_models_config(MODELS)
    assert models.defaults.generator in models.models
    out = tmp_path / "models.yaml"
    dump_config(models, out)
    assert load_models_config(out) == models


def test_env_var_selects_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = _raw(DEFAULT)
    raw["seed"] = 7
    alt = tmp_path / "alt.yaml"
    alt.write_text(yaml.safe_dump(raw))
    monkeypatch.setenv("RAGANTINE_CONFIG", str(alt))
    assert load_config().seed == 7


def test_unknown_key_rejected() -> None:
    raw = _raw(DEFAULT)
    raw["retrieval"]["topk"] = 3
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_tier_mix_must_sum_to_one() -> None:
    raw = _raw(DEFAULT)
    raw["corpus"]["tier_mix"]["curated"] = 0.9
    with pytest.raises(ValidationError, match=r"sum to 1\.0"):
        AppConfig.model_validate(raw)


def test_unknown_defense_id_rejected() -> None:
    raw = _raw(DEFAULT)
    raw["defenses"]["enabled"] = ["d1", "d9"]
    with pytest.raises(ValidationError):
        AppConfig.model_validate(raw)


def test_final_k_cannot_exceed_fused_k() -> None:
    raw = _raw(DEFAULT)
    raw["retrieval"]["final_k"] = 50
    with pytest.raises(ValidationError, match="final_k"):
        AppConfig.model_validate(raw)


def test_models_defaults_must_reference_known_alias() -> None:
    raw = _raw(MODELS)
    raw["defaults"]["judge"] = "no-such-model"
    with pytest.raises(ValidationError, match="judge"):
        ModelsConfig.model_validate(raw)


def test_model_with_unknown_provider_rejected() -> None:
    raw = _raw(MODELS)
    raw["models"]["open-large"]["provider"] = "nowhere"
    with pytest.raises(ValidationError, match="unknown provider"):
        ModelsConfig.model_validate(raw)


def test_settings_keep_keys_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-not-a-real-key")
    settings = Settings(_env_file=None)
    assert settings.groq_api_key is not None
    assert "test-not-a-real-key" not in repr(settings)
    assert settings.groq_api_key.get_secret_value() == "test-not-a-real-key"


def test_judge_differs_from_generator_when_possible() -> None:
    models = load_models_config(MODELS)
    judge = models.defaults.judge
    assert models.judge_for("open-large") == judge
    assert models.judge_for(judge) == models.defaults.judge_fallback


def test_placeholder_keys_count_as_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "<your-groq-api-key>")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.api_key("GROQ_API_KEY") is None
    assert settings.api_key("OPENAI_API_KEY") is None
    monkeypatch.setenv("OPENAI_API_KEY", "test-not-a-real-key")
    assert settings.api_key("OPENAI_API_KEY") == "test-not-a-real-key"


def test_model_cost() -> None:
    entry = load_models_config(MODELS).models["open-large"]
    assert entry.cost_usd(1_000_000, 0) == pytest.approx(entry.price_in_per_1m)


def test_overrides_apply_and_validate() -> None:
    from ragantine.config import apply_overrides

    cfg = load_config(DEFAULT)
    out = apply_overrides(
        cfg, ["defenses.d2.variant=zeroshot", "retrieval.final_k=3", "defenses.enabled=[d1, d3]"]
    )
    assert (
        out.defenses.d2.variant == "zeroshot"
        and out.retrieval.final_k == 3
        and out.defenses.enabled == ["d1", "d3"]
    )
    assert cfg.defenses.d2.variant == "finetuned"  # original untouched
    with pytest.raises(KeyError):
        apply_overrides(cfg, ["retrieval.nope=1"])
    with pytest.raises(ValidationError):
        apply_overrides(cfg, ["retrieval.final_k=999"])
    with pytest.raises(ValueError):
        apply_overrides(cfg, ["no-equals-sign"])


def test_v01_free_config_only_changes_scale() -> None:
    full = load_config(DEFAULT)
    free = load_config(ROOT / "configs" / "v01-free.yaml")
    assert (
        free.bench.targets_per_attack == 10
        and free.bench.seeds == [1337]
        and free.corpus.n_articles == 300
    )
    same = ["chunking", "embedding", "index", "retrieval", "paths"]
    assert all(getattr(free, k) == getattr(full, k) for k in same)
    # only D2's threshold may differ: `bench d2-threshold --write` tunes it on the dev split
    tuned = free.defenses.model_copy(
        update={"d2": free.defenses.d2.model_copy(update={"threshold": full.defenses.d2.threshold})}
    )
    assert tuned == full.defenses
