from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from app.models import (
    ColumnProfile,
    DataProfile,
    RecommendationPoint,
    ReviewPolicy,
    SavedAnnotation,
    SessionState,
    SignalSpec,
    TaskSpec,
)
from core.canonical_schema import MaintenanceDocument

SESSIONS_DIR = Path("data/sessions")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


class FileSessionStore:
    def __init__(self, root_dir: Path | str = SESSIONS_DIR) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def load(self, session_id: str) -> Optional[SessionState]:
        path = self._state_path(session_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return _session_from_dict(raw)

    def save(self, state: SessionState) -> None:
        session_dir = self._session_dir(state.session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        target = self._state_path(state.session_id)
        payload = json.dumps(asdict(state), ensure_ascii=False, indent=2, default=str)
        _atomic_write(target, payload)

    def save_uploaded_dataset(self, session_id: str, filename: str, payload: bytes) -> Path:
        session_dir = self._session_dir(session_id)
        session_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(filename or "dataset.csv").suffix or ".csv"
        target = session_dir / f"source{suffix}"
        target.write_bytes(payload)
        return target

    def _session_dir(self, session_id: str) -> Path:
        return self.root_dir / session_id

    def _state_path(self, session_id: str) -> Path:
        return self._session_dir(session_id) / "state.json"


def _atomic_write(path: Path, content: str) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.replace(path)


def _session_from_dict(raw: dict[str, Any]) -> SessionState:
    state = SessionState()
    state.session_id = str(raw.get("session_id") or state.session_id)
    state.filename = raw.get("filename")
    state.dataframe_json = raw.get("dataframe_json")
    state.dataframe_path = raw.get("dataframe_path")
    state.profile = _data_profile_from_dict(raw.get("profile"))
    state.selected_well_column = raw.get("selected_well_column")
    state.selected_time_column = raw.get("selected_time_column")
    state.selected_series = [str(item) for item in (raw.get("selected_series") or [])]
    state.selected_well_value = raw.get("selected_well_value")
    state.date_from = raw.get("date_from")
    state.date_to = raw.get("date_to")
    state.anomaly_goal = raw.get("anomaly_goal")
    state.chart_preferences = raw.get("chart_preferences")
    state.window_size = _coerce_int(raw.get("window_size"))
    state.statistical_threshold_pct = _coerce_float(raw.get("statistical_threshold_pct"))
    state.recommendation_mode = str(raw.get("recommendation_mode") or state.recommendation_mode)
    state.recommendation = _recommendation_from_dict(raw.get("recommendation"))
    state.saved_annotations = [
        _saved_annotation_from_dict(item)
        for item in (raw.get("saved_annotations") or [])
        if isinstance(item, dict)
    ]
    state.annotations_path = raw.get("annotations_path")
    state.task_spec = _task_spec_from_dict(raw.get("task_spec"))
    state.task_spec_path = raw.get("task_spec_path")
    state.maintenance_documents = [
        _maintenance_document_from_dict(item)
        for item in (raw.get("maintenance_documents") or [])
        if isinstance(item, dict)
    ]
    state.maintenance_upload_name = raw.get("maintenance_upload_name")
    state.maintenance_documents_path = raw.get("maintenance_documents_path")
    state.maintenance_context_summary = dict(raw.get("maintenance_context_summary") or {})
    state.messages = [
        {"role": str(item.get("role") or ""), "content": str(item.get("content") or "")}
        for item in (raw.get("messages") or [])
        if isinstance(item, dict)
    ]
    state.review_candidates = [
        dict(item)
        for item in (raw.get("review_candidates") or [])
        if isinstance(item, dict)
    ]
    state.review_cache_key = raw.get("review_cache_key")
    return state


def _data_profile_from_dict(raw: Any) -> Optional[DataProfile]:
    if not isinstance(raw, dict):
        return None
    columns = [
        ColumnProfile(**{
            key: value
            for key, value in item.items()
            if key in ColumnProfile.__dataclass_fields__
        })
        for item in (raw.get("columns") or [])
        if isinstance(item, dict)
    ]
    payload = {
        key: value
        for key, value in raw.items()
        if key in DataProfile.__dataclass_fields__ and key != "columns"
    }
    payload["columns"] = columns
    return DataProfile(**payload)


def _recommendation_from_dict(raw: Any) -> RecommendationPoint:
    if not isinstance(raw, dict):
        return RecommendationPoint()
    payload = {
        key: value
        for key, value in raw.items()
        if key in RecommendationPoint.__dataclass_fields__
    }
    return RecommendationPoint(**payload)


def _saved_annotation_from_dict(raw: dict[str, Any]) -> SavedAnnotation:
    payload = {
        key: value
        for key, value in raw.items()
        if key in SavedAnnotation.__dataclass_fields__
    }
    return SavedAnnotation(**payload)


def _task_spec_from_dict(raw: Any) -> Optional[TaskSpec]:
    if not isinstance(raw, dict):
        return None
    spec = TaskSpec()
    for key, value in raw.items():
        if key == "signal_schema" and isinstance(value, list):
            spec.signal_schema = [
                SignalSpec(**{
                    item_key: item_value
                    for item_key, item_value in item.items()
                    if item_key in SignalSpec.__dataclass_fields__
                })
                for item in value
                if isinstance(item, dict)
            ]
        elif key == "review_policy" and isinstance(value, dict):
            spec.review_policy = ReviewPolicy(**{
                item_key: item_value
                for item_key, item_value in value.items()
                if item_key in ReviewPolicy.__dataclass_fields__
            })
        elif key in TaskSpec.__dataclass_fields__:
            setattr(spec, key, value)
    return spec


def _maintenance_document_from_dict(raw: dict[str, Any]) -> MaintenanceDocument:
    event_date = _coerce_datetime(raw.get("event_date")) or datetime.utcnow()
    return MaintenanceDocument(
        doc_id=str(raw.get("doc_id") or ""),
        asset_id=str(raw.get("asset_id") or "unknown"),
        event_date=event_date,
        raw_text=str(raw.get("raw_text") or ""),
        source=str(raw.get("source") or "maintenance_log"),
    )


def _coerce_datetime(value: Any) -> Optional[datetime]:
    if not value:
        return None
    text = str(value)
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _coerce_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
