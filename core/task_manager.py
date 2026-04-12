"""
Core TaskSpec management.

Handles create / load / save / update of TaskSpec.
app/task_manager.py delegates here; all new pipeline code imports from here.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4


TASKS_DIR = Path("data/tasks")
TASKS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TAXONOMY_DEFAULT = [
    "belt_break",
    "planned_stop",
    "planned_maintenance",
    "sensor_issue",
    "stable_unusual_regime",
    "unknown",
]


def new_id() -> str:
    return uuid4().hex


# ---------------------------------------------------------------------------
# Extended TaskSpec (superset of app/models.py TaskSpec)
# ---------------------------------------------------------------------------

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

    # Equipment / domain
    equipment_family: str = "generic_well_timeseries"
    primary_deviation: str = "user_defined_deviation"

    # Signals
    signal_schema: list[SignalSpec] = field(default_factory=list)
    well_column: Optional[str] = None
    time_column: Optional[str] = None

    # Segmentation
    segmentation_strategy: str = "regime_segment_with_manual_point_or_interval_refinement"
    minimum_segment_duration: Optional[int] = None

    # Features extracted for each candidate
    feature_profile: list[str] = field(default_factory=lambda: [
        "power_mean", "power_std", "power_p10", "power_p90",
        "transition_sharpness", "segment_duration",
    ])

    # Labeling
    label_taxonomy: list[str] = field(default_factory=lambda: list(LABEL_TAXONOMY_DEFAULT))
    unknown_label: str = "unknown"
    confounders: list[str] = field(default_factory=lambda: [
        "planned_stop", "planned_maintenance", "sensor_issue", "load_change",
    ])

    # Context
    context_sources: list[str] = field(default_factory=lambda: [
        "maintenance_reports", "equipment_metadata", "engineer_review_notes",
    ])

    # Baselines
    baseline_strategy: str = "per_well_history"

    # Signal quality
    quality_rules: list[str] = field(default_factory=lambda: [
        "require_time_axis",
        "drop_non_numeric_signal_values",
        "flag_dropout_segments",
    ])
    signal_min: Optional[float] = None
    signal_max: Optional[float] = None
    dropout_threshold: Optional[float] = None
    min_dropout_duration_s: Optional[int] = None
    max_interpolation_gap_s: Optional[int] = None

    # Review policy
    review_policy: ReviewPolicy = field(default_factory=ReviewPolicy)

    # Discovery fields
    normal_operation_definition: Optional[str] = None
    expected_deviation_frequency: Optional[str] = None
    statistical_threshold_pct: Optional[float] = None

    # Known event types for fact extraction
    known_event_types: list[str] = field(default_factory=lambda: [
        "planned_stop", "belt_replacement", "rod_replacement",
        "equipment_service", "sensor_check",
    ])


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------

def build_task_id(filename: str | None) -> str:
    stem = Path(filename or "task").stem.lower()
    slug = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "task"
    return f"{slug}_{new_id()[:8]}"


def default_task_spec_path(task_id: str) -> Path:
    return TASKS_DIR / task_id / "task_spec.json"


def persist_task_spec(spec: TaskSpec, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(asdict(spec), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_task_spec(task_id: str) -> Optional[TaskSpec]:
    path = default_task_spec_path(task_id)
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return _task_spec_from_dict(raw)


def _task_spec_from_dict(raw: dict[str, Any]) -> TaskSpec:
    spec = TaskSpec()
    for key, value in raw.items():
        if key == "signal_schema":
            setattr(spec, key, [
                SignalSpec(**{k: v for k, v in item.items() if k in SignalSpec.__dataclass_fields__})
                for item in (value or [])
            ])
        elif key == "review_policy" and isinstance(value, dict):
            spec.review_policy = ReviewPolicy(**{
                k: v for k, v in value.items()
                if k in ReviewPolicy.__dataclass_fields__
            })
        elif hasattr(spec, key):
            setattr(spec, key, value)
    return spec


def apply_task_spec_updates(spec: TaskSpec, updates: dict[str, Any]) -> TaskSpec:
    if not updates:
        return spec

    str_fields = (
        "equipment_family", "primary_deviation",
        "normal_operation_definition", "expected_deviation_frequency",
    )
    for f in str_fields:
        if isinstance(updates.get(f), str) and updates[f].strip():
            setattr(spec, f, updates[f].strip())

    for f in ("confounders", "context_sources", "known_event_types"):
        if isinstance(updates.get(f), list):
            cleaned = [str(item).strip() for item in updates[f] if str(item).strip()]
            if cleaned:
                setattr(spec, f, cleaned)

    for f in ("signal_min", "signal_max", "dropout_threshold"):
        val = updates.get(f)
        if val not in (None, ""):
            try:
                setattr(spec, f, float(val))
            except (TypeError, ValueError):
                pass

    for f in ("minimum_segment_duration", "min_dropout_duration_s", "max_interpolation_gap_s"):
        val = updates.get(f)
        if val not in (None, ""):
            try:
                parsed = int(val)
                if parsed > 0:
                    setattr(spec, f, parsed)
            except (TypeError, ValueError):
                pass

    val = updates.get("statistical_threshold_pct")
    if val not in (None, ""):
        try:
            parsed = float(val)
            if parsed > 0:
                spec.statistical_threshold_pct = parsed
        except (TypeError, ValueError):
            pass

    return spec
