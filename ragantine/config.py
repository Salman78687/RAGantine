"""Typed configuration loaded from configs/default.yaml and configs/models.yaml (SPEC §B2).

Secrets and endpoints come from the environment / ``.env`` via pydantic-settings;
everything else comes from YAML. ``dump_config`` writes the resolved run config so
every run can snapshot it into ``results/<run_id>/config.yaml``.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ragantine.schemas import DefenseId, SourceTier

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_MODELS_PATH = Path("configs/models.yaml")


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _check_mix(mix: dict[Any, float], name: str) -> None:
    if any(v < 0 for v in mix.values()):
        raise ValueError(f"{name} weights must be non-negative")
    if not math.isclose(sum(mix.values()), 1.0, abs_tol=1e-6):
        raise ValueError(f"{name} must sum to 1.0, got {sum(mix.values())}")


# ---------------------------------------------------------------- environment


class Settings(BaseSettings):
    """Environment-sourced settings. Keys are SecretStr so they never print."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    groq_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    opensearch_url: str = "http://localhost:9200"
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None
    langfuse_host: str = "https://cloud.langfuse.com"
    hf_token: SecretStr | None = None
    ragantine_config: Path = DEFAULT_CONFIG_PATH
    ragantine_models_config: Path = DEFAULT_MODELS_PATH

    def api_key(self, env_name: str) -> str | None:
        """Return the key stored under ``env_name``, or None if unset or a ``<placeholder>``."""
        raw = os.environ.get(env_name)
        if raw is None:
            field = env_name.lower()
            value = getattr(self, field, None)
            raw = value.get_secret_value() if isinstance(value, SecretStr) else None
        if not raw or raw.strip().startswith("<"):
            return None
        return raw.strip()


# ---------------------------------------------------------------- default.yaml


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    models_dir: Path = Path("models")

    @property
    def corpus_dir(self) -> Path:
        return self.data_dir / "corpus"

    @property
    def bench_dir(self) -> Path:
        return self.data_dir / "bench"

    @property
    def llm_cache_dir(self) -> Path:
        return self.data_dir / "cache" / "llm"


class CorpusConfig(_Strict):
    hf_dataset: str
    hf_config: str
    n_articles: int = Field(gt=0)
    min_chars: int = Field(ge=0)
    max_chars: int = Field(gt=0)
    scan_limit: int = Field(default=100_000, gt=0)
    tier_mix: dict[SourceTier, float]

    @model_validator(mode="after")
    def _validate(self) -> CorpusConfig:
        if self.min_chars > self.max_chars:
            raise ValueError("corpus.min_chars must be <= corpus.max_chars")
        _check_mix(self.tier_mix, "corpus.tier_mix")
        return self


class ChunkingConfig(_Strict):
    target_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    min_tokens: int = Field(default=30, ge=0)
    tokenizer: str

    @model_validator(mode="after")
    def _validate(self) -> ChunkingConfig:
        if self.overlap_tokens >= self.target_tokens:
            raise ValueError("chunking.overlap_tokens must be < target_tokens")
        return self


class EmbeddingConfig(_Strict):
    model: str
    dimension: int = Field(default=768, gt=0)
    batch_size: int = Field(gt=0)
    query_prefix: str


class HnswConfig(_Strict):
    m: int = Field(gt=0)
    ef_construction: int = Field(gt=0)
    ef_search: int = Field(gt=0)


class IndexConfig(_Strict):
    name: str
    quarantine_name: str
    hnsw: HnswConfig


class RetrievalConfig(_Strict):
    bm25_k: int = Field(gt=0)
    knn_k: int = Field(gt=0)
    rrf_k: int = Field(gt=0)
    fused_k: int = Field(gt=0)
    rerank_model: str
    final_k: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate(self) -> RetrievalConfig:
        if self.final_k > self.fused_k:
            raise ValueError("retrieval.final_k must be <= fused_k")
        return self


class D1Config(_Strict):
    flag_min_hidden_chars: int = Field(ge=0)


class D2Config(_Strict):
    model_path: str
    threshold: float = Field(ge=0.0, le=1.0)
    apply_at: list[Literal["ingestion", "retrieval"]]
    variant: Literal["finetuned", "zeroshot"] = "finetuned"
    zeroshot_model: str = "protectai/deberta-v3-base-prompt-injection-v2"


class D3Config(_Strict):
    tier_weights: dict[SourceTier, float]
    max_low_tier_in_final: int = Field(ge=0)


class D4Config(_Strict):
    mode: Literal["delimit_and_datamark", "delimit_only"]


class D5Config(_Strict):
    min_supporting_docs: int = Field(gt=0)
    nli_model: str
    support_threshold: float = Field(ge=0.0, le=1.0)


class D6Config(_Strict):
    probe_queries: int = Field(gt=0)
    hub_z_threshold: float


class DefensesConfig(_Strict):
    enabled: list[DefenseId] = Field(default_factory=list)
    d1: D1Config
    d2: D2Config
    d3: D3Config
    d4: D4Config
    d5: D5Config
    d6: D6Config


class BenchConfig(_Strict):
    attacks: list[Literal["A1", "A2", "A3", "A4", "A5", "A6"]]
    targets_per_attack: int = Field(gt=0)
    poison_docs_per_target: int = Field(gt=0)
    poison_tier_mix: dict[SourceTier, float]
    n_clean_questions: int = Field(ge=0)
    n_paraphrase: int = Field(ge=0)
    dev_fraction: float = Field(gt=0.0, lt=1.0)
    seeds: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate(self) -> BenchConfig:
        _check_mix(self.poison_tier_mix, "bench.poison_tier_mix")
        return self


class AppConfig(_Strict):
    """Resolved run configuration (configs/default.yaml)."""

    seed: int
    paths: PathsConfig = Field(default_factory=PathsConfig)
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: RetrievalConfig
    defenses: DefensesConfig
    bench: BenchConfig


# ---------------------------------------------------------------- models.yaml


class ProviderConfig(_Strict):
    base_url: str
    api_key_env: str


class ModelEntry(_Strict):
    provider: str
    model: str
    # USD per 1M tokens, used for cost reporting (SPEC §B5.6); optional until set.
    price_in_per_1m: float | None = Field(default=None, ge=0)
    price_out_per_1m: float | None = Field(default=None, ge=0)
    # Provider-specific request params (e.g. reasoning_effort for gpt-oss on Groq).
    extra_params: dict[str, Any] = Field(default_factory=dict)
    # Some reasoning models reject `temperature`; the client also learns this at runtime.
    supports_temperature: bool = True
    # Client-side pacing for providers with a per-minute token budget (e.g. Groq free tier).
    tokens_per_minute: int | None = Field(default=None, gt=0)
    # ...and for providers that limit requests per minute (e.g. Gemini API free tier).
    requests_per_minute: int | None = Field(default=None, gt=0)

    def cost_usd(self, prompt_tokens: int, completion_tokens: int) -> float | None:
        if self.price_in_per_1m is None or self.price_out_per_1m is None:
            return None
        return (
            prompt_tokens * self.price_in_per_1m + completion_tokens * self.price_out_per_1m
        ) / 1_000_000


class ModelDefaults(_Strict):
    generator: str
    judge: str
    judge_fallback: str | None = None  # used when the generator under test is the judge model
    poison_writer: str
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0)
    max_concurrency: int = Field(gt=0)
    # 429s honour the provider's Retry-After; free tiers need more patience than 5xx retries
    rate_limit_attempts: int = Field(default=5, gt=0)
    # also applied to timeouts / connection errors, which a loaded laptop hits under contention
    connect_timeout_s: float = Field(default=30.0, gt=0)
    request_timeout_s: float = Field(default=120.0, gt=0)


class ModelsConfig(_Strict):
    providers: dict[str, ProviderConfig]
    models: dict[str, ModelEntry]
    defaults: ModelDefaults

    @model_validator(mode="after")
    def _validate(self) -> ModelsConfig:
        for alias, entry in self.models.items():
            if entry.provider not in self.providers:
                raise ValueError(f"model {alias!r} uses unknown provider {entry.provider!r}")
        for role in ("generator", "judge", "judge_fallback", "poison_writer"):
            alias = getattr(self.defaults, role)
            if alias is not None and alias not in self.models:
                raise ValueError(f"defaults.{role} refers to unknown model alias {alias!r}")
        return self

    def judge_for(self, generator_alias: str) -> str:
        """Judge alias for scoring ``generator_alias``: a different model when possible (SPEC §B1)."""
        judge = self.defaults.judge
        same = (
            self.models[judge].model == self.models.get(generator_alias, self.models[judge]).model
        )
        if same and generator_alias in self.models and self.defaults.judge_fallback:
            return self.defaults.judge_fallback
        return judge


# ---------------------------------------------------------------- loading


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return data


def _resolve(path: Path | str | None, env_var: str, default: Path) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(env_var, default))


def load_config(path: Path | str | None = None) -> AppConfig:
    """Load the run config from ``path``, else $RAGANTINE_CONFIG, else configs/default.yaml."""
    return AppConfig.model_validate(
        _read_yaml(_resolve(path, "RAGANTINE_CONFIG", DEFAULT_CONFIG_PATH))
    )


def load_models_config(path: Path | str | None = None) -> ModelsConfig:
    """Load model aliases from ``path``, else $RAGANTINE_MODELS_CONFIG, else configs/models.yaml."""
    return ModelsConfig.model_validate(
        _read_yaml(_resolve(path, "RAGANTINE_MODELS_CONFIG", DEFAULT_MODELS_PATH))
    )


def dump_config(cfg: BaseModel, path: Path | str) -> None:
    """Write a config model to YAML (used to snapshot resolved configs per run)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.model_dump(mode="json"), f, sort_keys=False, allow_unicode=True)


def apply_overrides(cfg: AppConfig, overrides: list[str] | None) -> AppConfig:
    """Return a validated copy with dotted overrides, e.g. ``defenses.d2.variant=zeroshot``.

    Values are parsed as YAML scalars/lists (``0.9``, ``true``, ``[ingestion]``).
    """
    if not overrides:
        return cfg
    data = cfg.model_dump(mode="json")
    for item in overrides:
        key, sep, raw = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"override must look like a.b.c=value, got {item!r}")
        node: Any = data
        parts = key.strip().split(".")
        for part in parts[:-1]:
            if not isinstance(node, dict) or part not in node:
                raise KeyError(f"unknown config key {key!r}")
            node = node[part]
        if not isinstance(node, dict) or parts[-1] not in node:
            raise KeyError(f"unknown config key {key!r}")
        node[parts[-1]] = yaml.safe_load(raw)
    return AppConfig.model_validate(data)
