from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional
from uuid import uuid4


def new_id() -> str:
    return uuid4().hex


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    nullable_pct: float
    sample_values: list[str]
    numeric: bool
    datetime_like: bool
    unique_ratio: float


@dataclass
class DataProfile:
    rows: int
    columns: list[ColumnProfile]
    timestamp_candidates: list[str]
    well_candidates: list[str]
    numeric_candidates: list[str]
    inferred_well_column: Optional[str]
    inferred_time_column: Optional[str]
    detected_multiple_wells: bool
    inferred_window_size: Optional[int]
    unique_well_count: int
    time_min: Optional[str]
    time_max: Optional[str]
    sheet_names: list[str]
    source_sheet_column: Optional[str]


@dataclass
class SignalSpec:
    name: str
    unit: Optional[str] = None
    role: str = "candidate_signal"
    selected_for_review: bool = False


@dataclass
class ReviewPolicy:
    policy_name: str = "human_review_required"
    auto_label_allowed: bool = False
    allow_modify: bool = True
    allow_reject: bool = True
    allow_point: bool = True
    allow_interval: bool = True


@dataclass
class TaskSpec:
    task_id: str = field(default_factory=new_id)
    version: str = "0.1.0"
    title: Optional[str] = None
    equipment_family: str = "generic_well_timeseries"
    primary_deviation: str = "user_defined_deviation"
    signal_schema: list[SignalSpec] = field(default_factory=list)
    segmentation_strategy: str = "regime_segment_with_manual_point_or_interval_refinement"
    feature_profile: list[str] = field(
        default_factory=lambda: [
            "power_mean",
            "power_std",
            "power_p10",
            "power_p90",
            "transition_sharpness",
            "segment_duration",
        ]
    )
    label_taxonomy: list[str] = field(
        default_factory=lambda: [
            "belt_break",
            "planned_stop",
            "planned_maintenance",
            "sensor_issue",
            "stable_unusual_regime",
            "unknown",
        ]
    )
    unknown_label: str = "unknown"
    context_sources: list[str] = field(
        default_factory=lambda: [
            "maintenance_reports",
            "equipment_metadata",
            "engineer_review_notes",
        ]
    )
    baseline_strategy: str = "per_well_history"
    quality_rules: list[str] = field(
        default_factory=lambda: [
            "require_time_axis",
            "drop_non_numeric_signal_values",
            "flag_dropout_segments",
        ]
    )
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)
    normal_operation_definition: Optional[str] = None
    confounders: list[str] = field(
        default_factory=lambda: [
            "planned_stop",
            "planned_maintenance",
            "sensor_issue",
            "load_change",
        ]
    )
    minimum_segment_duration: Optional[int] = None
    expected_deviation_frequency: Optional[str] = None
    statistical_threshold_pct: Optional[float] = None
    well_column: Optional[str] = None
    time_column: Optional[str] = None


@dataclass
class RecommendationPoint:
    mode: str = "interval"
    x: Optional[str] = None
    y: Optional[float] = None
    x_end: Optional[str] = None
    trace_name: Optional[str] = None
    locked: bool = False


@dataclass
class SavedAnnotation:
    annotation_id: str = field(default_factory=new_id)
    filename: Optional[str] = None
    well_column: Optional[str] = None
    well_value: Optional[str] = None
    recommendation_mode: str = "point"
    x: Optional[str] = None
    x_end: Optional[str] = None
    y: Optional[float] = None
    trace_name: Optional[str] = None
    series: list[str] = field(default_factory=list)
    window_size: Optional[int] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    label: Optional[str] = None
    correction_reason: Optional[str] = None
    created_at: Optional[str] = None
    candidate_id: Optional[str] = None
    proposed_label: Optional[str] = None
    proposed_rule: Optional[str] = None
    review_action: Optional[str] = None
    review_status: str = "accepted"
    routing: Optional[str] = None
    candidate_source: Optional[str] = None
    deviation_type: Optional[str] = None
    deviation_score: Optional[float] = None
    explanation: Optional[str] = None
    rule_trace: dict[str, Any] = field(default_factory=dict)
    local_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionState:
    session_id: str = field(default_factory=new_id)
    filename: Optional[str] = None
    dataframe_json: Optional[str] = None
    profile: Optional[DataProfile] = None
    selected_well_column: Optional[str] = None
    selected_time_column: Optional[str] = None
    selected_series: list[str] = field(default_factory=list)
    selected_well_value: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    anomaly_goal: Optional[str] = None
    chart_preferences: Optional[str] = None
    window_size: Optional[int] = None
    statistical_threshold_pct: Optional[float] = None
    recommendation_mode: str = "interval"
    recommendation: RecommendationPoint = field(default_factory=RecommendationPoint)
    saved_annotations: list[SavedAnnotation] = field(default_factory=list)
    annotations_path: Optional[str] = None
    task_spec: Optional[TaskSpec] = None
    task_spec_path: Optional[str] = None
    messages: list[dict[str, str]] = field(default_factory=list)
    review_candidates: list[dict[str, Any]] = field(default_factory=list)
    review_cache_key: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "filename": self.filename,
            "profile": self.profile,
            "selected_well_column": self.selected_well_column,
            "selected_time_column": self.selected_time_column,
            "selected_series": self.selected_series,
            "selected_well_value": self.selected_well_value,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "anomaly_goal": self.anomaly_goal,
            "chart_preferences": self.chart_preferences,
            "window_size": self.window_size,
            "statistical_threshold_pct": self.statistical_threshold_pct,
            "recommendation_mode": self.recommendation_mode,
            "recommendation": self.recommendation,
            "saved_annotations": self.saved_annotations,
            "annotations_path": self.annotations_path,
            "task_spec": self.task_spec,
            "task_spec_path": self.task_spec_path,
            "messages": self.messages,
            "review_candidates": self.review_candidates,
            "review_cache_key": self.review_cache_key,
        }
