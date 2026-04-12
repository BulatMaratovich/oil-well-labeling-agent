"""
Canonical dataclasses for the oil-well labeling pipeline.

These are plain data containers — no business logic here.
All pipeline stages communicate via these types.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

@dataclass
class DateRange:
    start: datetime
    end: datetime


# ---------------------------------------------------------------------------
# Stage 1 — Input Normalizer output
# ---------------------------------------------------------------------------

@dataclass
class CanonicalTimeSeries:
    """A single asset's time series after schema normalisation."""
    asset_id: str
    timestamp_col: str
    signal_col: str
    unit: Optional[str]
    # Underlying DataFrame with at least [timestamp_col, signal_col] columns.
    # Not serialised to JSON — only used in-memory during a run.
    values: pd.DataFrame = field(repr=False, compare=False)


# ---------------------------------------------------------------------------
# Stage 2 — Signal Sanitizer output
# ---------------------------------------------------------------------------

@dataclass
class QualityFlags:
    missing_pct: float                        # fraction 0–1
    dropout_spans: list[DateRange] = field(default_factory=list)
    noise_level: Optional[float] = None       # std / median, dimensionless
    clamp_events: int = 0                     # count of clamped values
    interpolated_gaps: int = 0                # count of interpolated gaps


# ---------------------------------------------------------------------------
# Stage 3 — Global Series Profiler output
# ---------------------------------------------------------------------------

@dataclass
class Regime:
    start: datetime
    end: datetime
    regime_id: str
    regime_type: str          # cluster label, e.g. "type_0", "type_1"
    duration_h: float
    mean_power: Optional[float] = None
    std_power: Optional[float] = None


@dataclass
class RegimeSequence:
    asset_id: str
    regimes: list[Regime] = field(default_factory=list)
    no_regime_structure: bool = False   # True when PELT found only 1 segment


# ---------------------------------------------------------------------------
# Stage 5 — Candidate Event Detector output
# ---------------------------------------------------------------------------

@dataclass
class CandidateEvent:
    candidate_id: str
    asset_id: str
    segment: DateRange
    deviation_type: str        # "novel_regime" | "atypical_amplitude" |
                               # "unusual_duration" | "abrupt_transition" |
                               # "full_series_review"
    deviation_score: float
    context_query: str         # free-text hint for context retrieval
    preceding_regime_type: Optional[str] = None
    following_regime_type: Optional[str] = None
    flags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Stage 6 — Local Segment Analyzer output
# ---------------------------------------------------------------------------

@dataclass
class LocalFeatures:
    candidate_id: str
    power_mean: Optional[float] = None
    power_std: Optional[float] = None
    power_p10: Optional[float] = None
    power_p90: Optional[float] = None
    min_power: Optional[float] = None
    max_power: Optional[float] = None
    zero_fraction: Optional[float] = None      # fraction of near-zero values
    transition_sharpness: Optional[float] = None  # delta at segment boundary
    segment_duration_h: Optional[float] = None
    preceding_regime_type: Optional[str] = None
    following_regime_type: Optional[str] = None


# ---------------------------------------------------------------------------
# Stage 7 — Context Fact Extractor output
# ---------------------------------------------------------------------------

@dataclass
class StructuredFacts:
    doc_id: str
    event_type: Optional[str] = None      # "planned_stop" | "belt_replacement" | …
    event_date: Optional[datetime] = None
    asset_id: Optional[str] = None
    duration_h: Optional[float] = None
    action_summary: Optional[str] = None
    parts_replaced: list[str] = field(default_factory=list)
    extraction_confidence: str = "ok"     # "ok" | "low" | "failed"


@dataclass
class MaintenanceDocument:
    doc_id: str
    asset_id: str
    event_date: datetime
    raw_text: str
    source: str = "maintenance_log"


@dataclass
class EquipmentDocument:
    asset_id: str
    equipment_family: str
    installation_date: Optional[datetime] = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleDocument:
    rule_id: str
    description: str
    label: str
    embedding_text: str


@dataclass
class ContextBundle:
    candidate_id: str
    maintenance_docs: list[MaintenanceDocument] = field(default_factory=list)
    maintenance_facts: list[StructuredFacts] = field(default_factory=list)
    equipment_doc: Optional[EquipmentDocument] = None
    rule_docs: list[RuleDocument] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)   # "fact_extraction_failed", "low_context", …


# ---------------------------------------------------------------------------
# Stage 8 — Rule Engine output
# ---------------------------------------------------------------------------

@dataclass
class RuleTrace:
    rules_evaluated: list[str] = field(default_factory=list)
    rules_fired: list[str] = field(default_factory=list)
    rules_blocked: list[str] = field(default_factory=list)
    winning_rule: Optional[str] = None
    conflict: bool = False
    abstain_reason: Optional[str] = None   # "no_rule_matched" | "rule_conflict" | …


@dataclass
class RuleResult:
    label: str                              # from label_taxonomy or "unknown"
    rule_trace: RuleTrace = field(default_factory=RuleTrace)
    abstain_reason: Optional[str] = None
    conflict_flag: bool = False


# ---------------------------------------------------------------------------
# Stage 9–10 — Human Review / Label Record
# ---------------------------------------------------------------------------

@dataclass
class LabelRecord:
    record_id: str
    task_id: str
    asset_id: str
    segment: DateRange
    deviation_type: str
    local_features: Optional[LocalFeatures]
    rule_result: RuleResult
    final_label: str
    was_override: bool = False             # True if engineer changed the label
    correction_reason: Optional[str] = None
    confirmed_at: Optional[datetime] = None
    run_id: Optional[str] = None
    status: str = "pending"                # "pending" | "accepted" | "rejected" | "ambiguous"


# ---------------------------------------------------------------------------
# WellProfile — Historical Profile Builder
# ---------------------------------------------------------------------------

@dataclass
class RegimeBaseline:
    regime_type: str
    mean_power: float
    std_power: float
    p10_power: float
    p90_power: float
    typical_duration_h: float
    observation_count: int


@dataclass
class WellProfile:
    well_id: str
    baseline_regimes: list[RegimeBaseline] = field(default_factory=list)
    known_stops: list[DateRange] = field(default_factory=list)
    known_replacements: list[DateRange] = field(default_factory=list)
    profile_source: str = "well_history"   # "well_history" | "population_fallback"
    first_seen: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    flags: list[str] = field(default_factory=list)   # "no_well_history", "profile_stale"
