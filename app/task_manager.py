"""
app/task_manager.py — compatibility shim.

All logic lives in core/task_manager.py.
This module re-exports the same public API that app/main.py expects.
"""
from __future__ import annotations

from typing import Any

from app.models import SessionState, SignalSpec as AppSignalSpec, TaskSpec as AppTaskSpec
from core.task_manager import (
    TaskSpec as CoreTaskSpec,
    SignalSpec as CoreSignalSpec,
    apply_task_spec_updates as core_apply_updates,
    build_task_id,
    default_task_spec_path,
    persist_task_spec as core_persist,
    load_task_spec,
)


__all__ = [
    "build_initial_task_spec",
    "sync_task_spec_from_state",
    "apply_task_spec_updates",
    "persist_task_spec",
    "default_task_spec_path",
    "load_task_spec",
]


def build_initial_task_spec(state: SessionState) -> AppTaskSpec:
    profile = state.profile
    signal_schema: list[AppSignalSpec] = []
    if profile:
        for name in profile.numeric_candidates:
            signal_schema.append(AppSignalSpec(
                name=name,
                unit=None,
                role="candidate_signal",
                selected_for_review=name in state.selected_series,
            ))

    spec = AppTaskSpec(
        task_id=build_task_id(state.filename),
        title=f"Task for {state.filename}" if state.filename else "Task",
        signal_schema=signal_schema,
        minimum_segment_duration=state.window_size,
        well_column=state.selected_well_column,
        time_column=state.selected_time_column,
    )
    sync_task_spec_from_state(spec, state)
    return spec


def sync_task_spec_from_state(spec: AppTaskSpec, state: SessionState) -> AppTaskSpec:
    if state.filename:
        spec.title = spec.title or f"Task for {state.filename}"
    if state.selected_time_column:
        spec.time_column = state.selected_time_column
    if state.selected_well_column:
        spec.well_column = state.selected_well_column
    if state.window_size:
        spec.minimum_segment_duration = state.window_size
    if state.statistical_threshold_pct is not None:
        spec.statistical_threshold_pct = state.statistical_threshold_pct
    if state.anomaly_goal:
        spec.primary_deviation = state.anomaly_goal

    spec.review_policy.allow_point = True
    spec.review_policy.allow_interval = True

    if spec.signal_schema:
        selected = set(state.selected_series)
        for signal in spec.signal_schema:
            signal.selected_for_review = signal.name in selected

    return spec


def apply_task_spec_updates(spec: AppTaskSpec, updates: dict[str, Any]) -> AppTaskSpec:
    """Apply LLM-inferred updates to an AppTaskSpec in-place."""
    if not updates:
        return spec

    if isinstance(updates.get("equipment_family"), str) and updates["equipment_family"].strip():
        spec.equipment_family = updates["equipment_family"].strip()
    if isinstance(updates.get("primary_deviation"), str) and updates["primary_deviation"].strip():
        spec.primary_deviation = updates["primary_deviation"].strip()
    if isinstance(updates.get("normal_operation_definition"), str) and updates["normal_operation_definition"].strip():
        spec.normal_operation_definition = updates["normal_operation_definition"].strip()
    if isinstance(updates.get("expected_deviation_frequency"), str) and updates["expected_deviation_frequency"].strip():
        spec.expected_deviation_frequency = updates["expected_deviation_frequency"].strip()

    val = updates.get("statistical_threshold_pct")
    if val not in (None, ""):
        try:
            parsed = float(val)
            if parsed > 0:
                spec.statistical_threshold_pct = parsed
        except (TypeError, ValueError):
            pass

    for field_name in ("confounders", "context_sources"):
        lst = updates.get(field_name)
        if isinstance(lst, list):
            cleaned = [str(item).strip() for item in lst if str(item).strip()]
            if cleaned:
                setattr(spec, field_name, cleaned)

    val = updates.get("minimum_segment_duration")
    if val not in (None, ""):
        try:
            parsed = int(val)
            if parsed > 0:
                spec.minimum_segment_duration = parsed
        except (TypeError, ValueError):
            pass

    return spec


def persist_task_spec(spec: AppTaskSpec, path: str) -> None:
    from dataclasses import asdict
    from pathlib import Path
    import json
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(asdict(spec), ensure_ascii=False, indent=2), encoding="utf-8")
