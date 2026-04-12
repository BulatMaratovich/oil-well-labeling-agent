"""
Central configuration for the oil-well labeling framework.

Loads config/settings.yaml, then applies overrides from environment variables.
Single instance `settings` is imported everywhere; `app/config.py` stays as a
thin compatibility shim for the existing FastAPI UI.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_CONFIG_FILE = Path(__file__).parent / "settings.yaml"


def _load_yaml() -> dict[str, Any]:
    if not _CONFIG_FILE.exists():
        return {}
    with _CONFIG_FILE.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _get(data: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Nested dict lookup with a default."""
    value = data
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key, default)
    return value


@dataclass
class LLMConfig:
    provider: str = "mistral"
    model: str = "mistral-nemo"
    api_base: str = "https://api.mistral.ai/v1"
    timeout_sec: float = 30.0
    max_tokens_discovery: int = 900
    max_tokens_fact_extraction: int = 600
    max_tokens_explanation: int = 400
    max_tokens_rule_draft: int = 800
    temperature: float = 0.1


@dataclass
class SignalConfig:
    value_min: float = -1000.0
    value_max: float = 100_000.0
    data_quality_min_valid: float = 0.5
    dropout_threshold: float = 0.01
    min_dropout_duration_s: int = 300
    max_interpolation_gap_s: int = 600


@dataclass
class ProfilingConfig:
    pelt_model: str = "rbf"
    pelt_penalty_multiplier: float = 3.0
    pelt_min_segment_size: int = 8
    regime_cluster_k_max: int = 6
    regime_cluster_k_min: int = 2
    max_profile_age_days: int = 90


@dataclass
class ThresholdsConfig:
    novel_regime_min_novelty: float = 0.0
    atypical_amplitude_z: float = 2.5
    unusual_duration_p_low: float = 0.1
    unusual_duration_p_high: float = 0.9
    default_statistical_threshold_pct: float = 30.0


@dataclass
class RetrievalConfig:
    chroma_collection: str = "oil_well_labeling"
    chroma_persist_dir: str = "data/chroma"
    semantic_top_k: int = 5
    semantic_min_similarity: float = 0.5
    maintenance_window_days_before: int = 3
    maintenance_window_days_after: int = 1


@dataclass
class TimeoutsConfig:
    input_normalization: int = 10
    signal_sanitization: int = 15
    global_series_profiling: int = 30
    historical_profile_build: int = 10
    candidate_event_detection: int = 10
    local_segment_analysis: int = 10
    context_fact_extraction: int = 40
    rule_engine: int = 5
    asset_total: int = 180
    candidate_total: int = 90


@dataclass
class RetriesConfig:
    context_fact_extraction: int = 2
    global_series_profiling: int = 1


@dataclass
class PathsConfig:
    tasks_dir: str = "data/tasks"
    runs_dir: str = "runs"
    knowledge_base_dir: str = "data/knowledge_base"
    baselines_dir: str = "data/baselines"
    outputs_dir: str = "data/outputs"
    review_labels_dir: str = "data/review_labels"


@dataclass
class ExampleStoreConfig:
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_similar: int = 5
    min_similarity: float = 0.6


@dataclass
class Settings:
    llm: LLMConfig = field(default_factory=LLMConfig)
    signal: SignalConfig = field(default_factory=SignalConfig)
    profiling: ProfilingConfig = field(default_factory=ProfilingConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    timeouts: TimeoutsConfig = field(default_factory=TimeoutsConfig)
    retries: RetriesConfig = field(default_factory=RetriesConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    example_store: ExampleStoreConfig = field(default_factory=ExampleStoreConfig)


def _build_settings() -> Settings:
    raw = _load_yaml()

    def _env(key: str, fallback: Any) -> Any:
        return os.getenv(key, fallback)

    llm_raw = raw.get("llm", {})
    sig_raw = raw.get("signal", {})
    prof_raw = raw.get("profiling", {})
    thr_raw = raw.get("thresholds", {})
    ret_raw = raw.get("retrieval", {})
    to_raw = raw.get("timeouts", {})
    re_raw = raw.get("retries", {})
    path_raw = raw.get("paths", {})
    es_raw = raw.get("example_store", {})

    return Settings(
        llm=LLMConfig(
            provider=_env("LLM_PROVIDER", llm_raw.get("provider", "mistral")),
            model=_env("LLM_MODEL", llm_raw.get("model", "mistral-nemo")),
            api_base=_env("LLM_API_BASE", llm_raw.get("api_base", "https://api.mistral.ai/v1")),
            timeout_sec=float(_env("LLM_TIMEOUT_SEC", llm_raw.get("timeout_sec", 30))),
            max_tokens_discovery=int(llm_raw.get("max_tokens_discovery", 900)),
            max_tokens_fact_extraction=int(llm_raw.get("max_tokens_fact_extraction", 600)),
            max_tokens_explanation=int(llm_raw.get("max_tokens_explanation", 400)),
            max_tokens_rule_draft=int(llm_raw.get("max_tokens_rule_draft", 800)),
            temperature=float(llm_raw.get("temperature", 0.1)),
        ),
        signal=SignalConfig(
            value_min=float(sig_raw.get("value_min", -1000.0)),
            value_max=float(sig_raw.get("value_max", 100_000.0)),
            data_quality_min_valid=float(sig_raw.get("data_quality_min_valid", 0.5)),
            dropout_threshold=float(sig_raw.get("dropout_threshold", 0.01)),
            min_dropout_duration_s=int(sig_raw.get("min_dropout_duration_s", 300)),
            max_interpolation_gap_s=int(sig_raw.get("max_interpolation_gap_s", 600)),
        ),
        profiling=ProfilingConfig(
            pelt_model=prof_raw.get("pelt_model", "rbf"),
            pelt_penalty_multiplier=float(prof_raw.get("pelt_penalty_multiplier", 3.0)),
            pelt_min_segment_size=int(prof_raw.get("pelt_min_segment_size", 8)),
            regime_cluster_k_max=int(prof_raw.get("regime_cluster_k_max", 6)),
            regime_cluster_k_min=int(prof_raw.get("regime_cluster_k_min", 2)),
            max_profile_age_days=int(prof_raw.get("max_profile_age_days", 90)),
        ),
        thresholds=ThresholdsConfig(
            novel_regime_min_novelty=float(thr_raw.get("novel_regime_min_novelty", 0.0)),
            atypical_amplitude_z=float(thr_raw.get("atypical_amplitude_z", 2.5)),
            unusual_duration_p_low=float(thr_raw.get("unusual_duration_p_low", 0.1)),
            unusual_duration_p_high=float(thr_raw.get("unusual_duration_p_high", 0.9)),
            default_statistical_threshold_pct=float(thr_raw.get("default_statistical_threshold_pct", 30.0)),
        ),
        retrieval=RetrievalConfig(
            chroma_collection=ret_raw.get("chroma_collection", "oil_well_labeling"),
            chroma_persist_dir=ret_raw.get("chroma_persist_dir", "data/chroma"),
            semantic_top_k=int(ret_raw.get("semantic_top_k", 5)),
            semantic_min_similarity=float(ret_raw.get("semantic_min_similarity", 0.5)),
            maintenance_window_days_before=int(ret_raw.get("maintenance_window_days_before", 3)),
            maintenance_window_days_after=int(ret_raw.get("maintenance_window_days_after", 1)),
        ),
        timeouts=TimeoutsConfig(**{k: int(v) for k, v in to_raw.items() if hasattr(TimeoutsConfig, k)}),
        retries=RetriesConfig(**{k: int(v) for k, v in re_raw.items() if hasattr(RetriesConfig, k)}),
        paths=PathsConfig(**{k: str(v) for k, v in path_raw.items() if hasattr(PathsConfig, k)}),
        example_store=ExampleStoreConfig(
            embedding_model=es_raw.get("embedding_model", "all-MiniLM-L6-v2"),
            top_k_similar=int(es_raw.get("top_k_similar", 5)),
            min_similarity=float(es_raw.get("min_similarity", 0.6)),
        ),
    )


settings = _build_settings()
