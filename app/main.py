from __future__ import annotations

import json
from dataclasses import asdict
from io import StringIO
from pathlib import Path
from typing import Optional
from datetime import datetime, timezone

import pandas as pd
from fastapi.encoders import jsonable_encoder
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.data_utils import (
    detect_candidate_intervals,
    filter_dataframe,
    get_scope_time_range,
    load_tabular_file,
    normalize_for_plot,
    profile_dataframe,
)
from app.llm_assistant import (
    apply_discovery_updates,
    build_initial_message,
    generate_reply,
    infer_message_updates,
    infer_series_from_message,
    infer_settings_from_message,
)
from app.config import settings
from app.models import RecommendationPoint, SavedAnnotation, SessionState, new_id
from app.task_manager import (
    apply_task_spec_updates,
    build_initial_task_spec,
    default_task_spec_path,
    persist_task_spec,
    sync_task_spec_from_state,
)
from app.stat_analysis import analyze_candidate_intervals


app = FastAPI(title="Oil Well Labeling UI")
templates = Jinja2Templates(directory="app/templates")
SESSIONS: dict[str, SessionState] = {}
LABELS_DIR = Path("data/review_labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)


def _get_session(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return state


def _default_well_value(state: SessionState, df: Optional[pd.DataFrame] = None) -> Optional[str]:
    profile = state.profile
    if profile and profile.sheet_names:
        return str(profile.sheet_names[0])
    if (
        df is not None
        and state.selected_well_column
        and state.selected_well_column in df.columns
    ):
        values = df[state.selected_well_column].dropna().astype(str)
        if not values.empty:
            return str(values.iloc[0])
    return None


def _parse_state_dataframe(state: SessionState) -> pd.DataFrame:
    if not state.dataframe_json:
        raise HTTPException(status_code=400, detail="Нет загруженных данных")
    return pd.read_json(StringIO(state.dataframe_json), orient="split")


def _persist_annotations(state: SessionState) -> None:
    if not state.annotations_path:
        return
    path = Path(state.annotations_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(item) for item in state.saved_annotations], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _filtered_annotations(state: SessionState, well_value: Optional[str] = None) -> list[SavedAnnotation]:
    if not well_value:
        return list(state.saved_annotations)
    return [item for item in state.saved_annotations if item.well_value == well_value]


def _find_annotation(state: SessionState, annotation_id: str) -> tuple[int, SavedAnnotation]:
    for index, annotation in enumerate(state.saved_annotations):
        if annotation.annotation_id == annotation_id:
            return index, annotation
    raise HTTPException(status_code=404, detail="Разметка не найдена")


# Labels that indicate the candidate was NOT a deviation — user overrode the system's implicit suggestion
_CONFOUNDER_LABELS = frozenset({"planned_stop", "planned_maintenance", "sensor_issue"})


def _write_to_task_memory(state: SessionState, annotation: SavedAnnotation) -> None:
    """Persist annotation as a LabelRecord in TaskMemory for learning."""
    if not state.task_spec or not annotation.label:
        return
    try:
        from learning.task_memory import TaskMemory
        from core.canonical_schema import DateRange, LabelRecord, RuleResult, RuleTrace

        def _parse_ts(s: Optional[str]) -> datetime:
            if not s:
                return datetime.now(tz=timezone.utc)
            try:
                dt = datetime.fromisoformat(s)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except ValueError:
                return datetime.now(tz=timezone.utc)

        record = LabelRecord(
            record_id=annotation.annotation_id,
            task_id=state.task_spec.task_id,
            asset_id=annotation.well_value or state.selected_well_value or "unknown",
            segment=DateRange(
                start=_parse_ts(annotation.x),
                end=_parse_ts(annotation.x_end) if annotation.x_end else _parse_ts(annotation.x),
            ),
            deviation_type=state.anomaly_goal or state.task_spec.primary_deviation,
            local_features=None,
            # The system's implicit suggestion for any candidate is "candidate_deviation"
            rule_result=RuleResult(label="candidate_deviation", rule_trace=RuleTrace()),
            final_label=annotation.label,
            was_override=annotation.label in _CONFOUNDER_LABELS,
            correction_reason=annotation.correction_reason,
            confirmed_at=datetime.now(tz=timezone.utc),
            status="accepted",
        )
        TaskMemory(state.task_spec.task_id).add(record)
    except Exception:
        pass  # memory write failure must not break annotation save


def _save_annotation(state: SessionState, label: Optional[str] = None, correction_reason: Optional[str] = None) -> SavedAnnotation:
    annotation = SavedAnnotation(
        filename=state.filename,
        well_column=state.selected_well_column,
        well_value=state.selected_well_value,
        recommendation_mode=state.recommendation.mode,
        x=state.recommendation.x,
        x_end=state.recommendation.x_end,
        y=state.recommendation.y,
        trace_name=state.recommendation.trace_name,
        series=list(state.selected_series),
        window_size=state.window_size,
        date_from=state.date_from,
        date_to=state.date_to,
        label=label or None,
        correction_reason=correction_reason or None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )
    state.saved_annotations.append(annotation)
    _persist_annotations(state)
    _write_to_task_memory(state, annotation)
    return annotation


def _filter_unannotated_candidates(
    candidates: list[dict],
    saved_annotations: list[SavedAnnotation],
    well_value: Optional[str],
) -> list[dict]:
    """Remove candidates whose time range is already covered by a saved annotation."""
    if not candidates or not saved_annotations:
        return candidates
    # Build list of (start, end) for annotations on this well that have a label
    annotated: list[tuple[str, str]] = [
        (a.x, a.x_end or a.x)
        for a in saved_annotations
        if a.label and a.x and (well_value is None or a.well_value == well_value)
    ]
    if not annotated:
        return candidates

    def _overlaps(c: dict) -> bool:
        c_start = c.get("start", "") or ""
        c_end = c.get("end", c_start) or c_start
        for a_start, a_end in annotated:
            # Overlap: neither is entirely before the other
            if c_start <= a_end and c_end >= a_start:
                return True
        return False

    return [c for c in candidates if not _overlaps(c)]


def _persist_task_spec(state: SessionState) -> None:
    if not state.task_spec or not state.task_spec_path:
        return
    sync_task_spec_from_state(state.task_spec, state)
    persist_task_spec(state.task_spec, state.task_spec_path)


def _suggest_recommendation(
    plot_payload: dict[str, object],
    window_size: Optional[int],
    mode: str,
    candidate_intervals: Optional[list[dict[str, object]]] = None,
) -> RecommendationPoint:
    traces = plot_payload["traces"]
    if not traces:
        return RecommendationPoint(mode=mode)
    first_trace = traces[0]
    x_values = first_trace["x"]
    y_values = first_trace["y"]
    if not x_values:
        return RecommendationPoint(mode=mode)
    if candidate_intervals:
        first_candidate = candidate_intervals[0]
        start_x = str(first_candidate.get("start"))
        end_x = str(first_candidate.get("end"))
        start_idx = x_values.index(start_x) if start_x in x_values else 0
        end_idx = x_values.index(end_x) if end_x in x_values else min(start_idx + 1, len(x_values) - 1)
        center_idx = min((start_idx + end_idx) // 2, len(y_values) - 1)
        y_value = y_values[center_idx] if center_idx < len(y_values) else None
        if mode == "interval":
            return RecommendationPoint(
                mode=mode,
                x=start_x,
                x_end=end_x,
                y=y_value,
                trace_name=str(first_candidate.get("series_name") or first_trace.get("name")),
                locked=False,
            )
        return RecommendationPoint(
            mode="point",
            x=str(x_values[center_idx]),
            y=y_value,
            trace_name=str(first_candidate.get("series_name") or first_trace.get("name")),
            locked=False,
        )
    center_idx = len(x_values) // 2
    if window_size and window_size > 0:
        half = max(window_size // 2, 1)
        center_idx = min(max(half, center_idx), max(len(x_values) - half - 1, 0))
    y_value = y_values[center_idx] if center_idx < len(y_values) else None
    if mode == "interval":
        half = max((window_size or 10) // 2, 1)
        start_idx = max(center_idx - half, 0)
        end_idx = min(center_idx + half, len(x_values) - 1)
        return RecommendationPoint(
            mode=mode,
            x=str(x_values[start_idx]),
            x_end=str(x_values[end_idx]),
            y=y_value,
            locked=False,
        )
    return RecommendationPoint(mode=mode, x=str(x_values[center_idx]), y=y_value, locked=False)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "mistral_model": settings.mistral_resolved_model,
            "mistral_configured": settings.mistral_configured,
        },
    )


@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    content = await file.read()
    try:
        df = load_tabular_file(file.filename or "dataset.csv", content)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    profile = profile_dataframe(df)
    session = SessionState(
        session_id=new_id(),
        filename=file.filename,
        dataframe_json=df.to_json(orient="split", date_format="iso"),
        profile=profile,
        selected_well_column=profile.inferred_well_column,
        selected_time_column=profile.inferred_time_column,
        selected_series=profile.numeric_candidates[:1],
        selected_well_value=profile.sheet_names[0] if profile.detected_multiple_wells and profile.sheet_names else None,
        window_size=profile.inferred_window_size,
        annotations_path=str(LABELS_DIR / f"{file.filename or 'session'}_{new_id()}.json"),
    )
    session.task_spec = build_initial_task_spec(session)
    session.task_spec_path = str(default_task_spec_path(session.task_spec.task_id))
    _persist_task_spec(session)
    initial_message = build_initial_message(session)
    session.messages.append({"role": "assistant", "content": initial_message})
    SESSIONS[session.session_id] = session
    return JSONResponse(
        jsonable_encoder({
            "session_id": session.session_id,
            "profile": asdict(profile),
            "messages": session.messages,
            "defaults": {
                "well_column": session.selected_well_column,
                "well_value": session.selected_well_value,
                "time_column": session.selected_time_column,
                "series": session.selected_series,
                "window_size": session.window_size,
                "date_from": session.date_from,
                "date_to": session.date_to,
                "recommendation_mode": session.recommendation_mode,
                "statistical_threshold_pct": session.statistical_threshold_pct,
                "annotations_path": session.annotations_path,
                "task_spec_path": session.task_spec_path,
            },
            "task_spec": asdict(session.task_spec) if session.task_spec else None,
            "llm": {
                "provider": "mistral",
                "model": settings.mistral_resolved_model,
                "configured": settings.mistral_configured,
            },
        })
    )


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, request: Request) -> JSONResponse:
    state = _get_session(session_id)
    payload = await request.json()
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение пустое")

    inferred = infer_settings_from_message(user_message)
    inferred_series = infer_series_from_message(
        user_message,
        state.profile.numeric_candidates if state.profile else [],
    )
    local_result = infer_message_updates(user_message, state)
    local_updates = local_result.get("updates") or {}

    state.selected_series = (
        local_updates.get("selected_series")
        or inferred_series
        or state.selected_series
        or payload.get("series")
    )
    state.selected_time_column = payload.get("time_column") or state.selected_time_column
    state.selected_well_column = payload.get("well_column") or state.selected_well_column
    state.selected_well_value = (
        local_updates.get("selected_well_value")
        or state.selected_well_value
        or payload.get("well_value")
    )
    state.date_from = (
        local_updates.get("date_from")
        or inferred.get("date_from")
        or state.date_from
        or payload.get("date_from")
    )
    state.date_to = (
        local_updates.get("date_to")
        or inferred.get("date_to")
        or state.date_to
        or payload.get("date_to")
    )
    state.anomaly_goal = (
        local_updates.get("anomaly_goal")
        or inferred.get("anomaly_goal")
        or state.anomaly_goal
        or payload.get("anomaly_goal")
    )
    state.chart_preferences = (
        local_updates.get("chart_preferences")
        or inferred.get("chart_preferences")
        or state.chart_preferences
        or payload.get("chart_preferences")
    )
    incoming_threshold = local_updates.get("statistical_threshold_pct")
    if incoming_threshold in (None, ""):
        incoming_threshold = inferred.get("statistical_threshold_pct")
    if incoming_threshold in (None, ""):
        incoming_threshold = state.statistical_threshold_pct
    if incoming_threshold in (None, ""):
        incoming_threshold = payload.get("statistical_threshold_pct")
    if incoming_threshold not in (None, ""):
        state.statistical_threshold_pct = float(incoming_threshold)
    state.recommendation_mode = (
        local_updates.get("recommendation_mode")
        or local_result.get("updates", {}).get("recommendation_mode")
        or inferred.get("recommendation_mode")
        or state.recommendation_mode
        or payload.get("recommendation_mode")
    )

    window_size = local_updates.get("window_size") or state.window_size or payload.get("window_size")
    if window_size in (None, ""):
        window_size = inferred.get("window_size")
    if window_size not in (None, ""):
        state.window_size = int(window_size)

    state.messages.append({"role": "user", "content": user_message})
    llm_result = generate_reply(state, user_message)
    apply_discovery_updates(state, llm_result.get("updates") or {})
    if state.task_spec:
        apply_task_spec_updates(state.task_spec, llm_result.get("task_spec_updates") or {})
        sync_task_spec_from_state(state.task_spec, state)
        _persist_task_spec(state)
    reply = llm_result["reply"]
    state.messages.append({"role": "assistant", "content": reply})
    return JSONResponse(
        jsonable_encoder(
            {
                "messages": state.messages,
                "reply": reply,
                "llm": {
                    "mode": llm_result.get("mode"),
                    "ready_for_first_pass": llm_result.get("ready_for_first_pass", False),
                    "error": llm_result.get("error"),
                },
                "task_spec": asdict(state.task_spec) if state.task_spec else None,
                "state": {
                    "selected_series": state.selected_series,
                    "selected_well_value": state.selected_well_value,
                    "date_from": state.date_from,
                    "date_to": state.date_to,
                    "anomaly_goal": state.anomaly_goal,
                    "chart_preferences": state.chart_preferences,
                    "recommendation_mode": state.recommendation_mode,
                    "window_size": state.window_size,
                    "statistical_threshold_pct": state.statistical_threshold_pct,
                    "task_spec_path": state.task_spec_path,
                },
            }
        )
    )


@app.get("/api/plot/{session_id}")
async def get_plot(
    session_id: str,
    time_column: Optional[str] = None,
    well_column: Optional[str] = None,
    well_value: Optional[str] = None,
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    series: Optional[str] = None,
    window_size: Optional[int] = None,
    statistical_threshold_pct: Optional[float] = None,
    recommendation_mode: Optional[str] = None,
) -> JSONResponse:
    state = _get_session(session_id)
    df = _parse_state_dataframe(state)

    resolved_series = [item for item in (series or "").split(",") if item] or state.selected_series
    resolved_time_column = time_column or state.selected_time_column
    resolved_well_column = well_column or state.selected_well_column
    resolved_well_value = well_value or state.selected_well_value
    if (
        not resolved_well_value
        and state.profile
        and state.profile.detected_multiple_wells
    ):
        resolved_well_value = _default_well_value(state, df)
    resolved_date_from = date_from or state.date_from
    resolved_date_to = date_to or state.date_to
    resolved_window_size = window_size if window_size is not None else state.window_size
    resolved_statistical_threshold_pct = (
        statistical_threshold_pct if statistical_threshold_pct is not None else state.statistical_threshold_pct
    )
    resolved_recommendation_mode = recommendation_mode or state.recommendation_mode
    state.selected_time_column = resolved_time_column
    state.selected_well_column = resolved_well_column
    state.selected_well_value = resolved_well_value
    state.date_from = resolved_date_from
    state.date_to = resolved_date_to
    state.selected_series = resolved_series
    state.window_size = resolved_window_size
    state.statistical_threshold_pct = resolved_statistical_threshold_pct
    state.recommendation_mode = resolved_recommendation_mode
    _persist_task_spec(state)

    plot_payload = normalize_for_plot(
        df=df,
        time_column=resolved_time_column,
        well_column=resolved_well_column,
        well_value=resolved_well_value,
        date_from=resolved_date_from,
        date_to=resolved_date_to,
        series_names=resolved_series,
    )
    scope_time_range = get_scope_time_range(
        df=df,
        time_column=resolved_time_column,
        well_column=resolved_well_column,
        well_value=resolved_well_value,
    )
    filtered_frame = filter_dataframe(
        df=df,
        time_column=resolved_time_column,
        well_column=resolved_well_column,
        well_value=resolved_well_value,
        date_from=resolved_date_from,
        date_to=resolved_date_to,
    )
    candidate_intervals = detect_candidate_intervals(
        plot_payload=plot_payload,
        anomaly_goal=state.anomaly_goal,
        window_size=resolved_window_size,
        statistical_threshold_pct=resolved_statistical_threshold_pct,
    )
    # Remove already-annotated intervals so the candidate list only shows unreviewed items
    candidate_intervals = _filter_unannotated_candidates(
        candidate_intervals, state.saved_annotations, resolved_well_value
    )
    candidate_interval_stats = analyze_candidate_intervals(
        frame=filtered_frame,
        time_column=resolved_time_column,
        series_name=resolved_series[0] if resolved_series else None,
        candidates=candidate_intervals,
        with_ruptures=True,
    )
    if (
        not state.recommendation.x
        or state.recommendation.mode != resolved_recommendation_mode
        or not state.recommendation.locked
    ):
        state.recommendation = _suggest_recommendation(
            plot_payload,
            resolved_window_size,
            resolved_recommendation_mode,
            candidate_intervals=candidate_intervals,
        )

    plot_warning = None
    if plot_payload.get("row_count", 0) == 0:
        if resolved_well_value and (scope_time_range.get("time_min") or scope_time_range.get("time_max")):
            plot_warning = (
                f"Для скважины `{resolved_well_value}` нет данных в выбранном диапазоне. "
                f"Доступный диапазон по этой скважине: "
                f"{scope_time_range.get('time_min') or '?'} .. {scope_time_range.get('time_max') or '?'}."
            )
        else:
            plot_warning = "Для выбранных фильтров данных не найдено."

    return JSONResponse(
        jsonable_encoder({
            "plot": plot_payload,
            "candidate_intervals": candidate_intervals,
            "candidate_interval_stats": candidate_interval_stats,
            "recommendation": asdict(state.recommendation),
            "selected_well_value": resolved_well_value,
            "plot_warning": plot_warning,
            "scope_time_range": scope_time_range,
            "window_size": resolved_window_size,
            "statistical_threshold_pct": resolved_statistical_threshold_pct,
            "recommendation_mode": resolved_recommendation_mode,
            "saved_annotations": [asdict(item) for item in _filtered_annotations(state, resolved_well_value)],
        })
    )


@app.post("/api/recommendation/{session_id}")
async def set_recommendation(session_id: str, request: Request) -> JSONResponse:
    state = _get_session(session_id)
    payload = await request.json()
    state.recommendation = RecommendationPoint(
        mode=payload.get("mode", state.recommendation_mode),
        x=payload.get("x"),
        y=payload.get("y"),
        x_end=payload.get("x_end"),
        trace_name=payload.get("trace_name"),
        locked=bool(payload.get("locked", True)),
    )
    state.recommendation_mode = state.recommendation.mode
    saved_annotation = None
    if state.recommendation.locked and state.recommendation.x:
        saved_annotation = _save_annotation(
            state,
            label=payload.get("label"),
            correction_reason=payload.get("correction_reason"),
        )
    return JSONResponse(
        jsonable_encoder(
            {
                "recommendation": asdict(state.recommendation),
                "saved_annotation": asdict(saved_annotation) if saved_annotation else None,
                "saved_annotations": [
                    asdict(item) for item in _filtered_annotations(state, state.selected_well_value)
                ],
            }
        )
    )


@app.patch("/api/annotations/{session_id}/{annotation_id}")
async def update_annotation(session_id: str, annotation_id: str, request: Request) -> JSONResponse:
    state = _get_session(session_id)
    _, annotation = _find_annotation(state, annotation_id)
    payload = await request.json()

    annotation.recommendation_mode = payload.get("mode", annotation.recommendation_mode)
    annotation.x = payload.get("x")
    annotation.x_end = payload.get("x_end")
    annotation.y = payload.get("y")
    annotation.trace_name = payload.get("trace_name")
    annotation.well_column = state.selected_well_column
    annotation.well_value = state.selected_well_value
    annotation.series = list(state.selected_series)
    annotation.window_size = state.window_size
    annotation.date_from = state.date_from
    annotation.date_to = state.date_to
    if "label" in payload:
        annotation.label = payload["label"] or None
    if "correction_reason" in payload:
        annotation.correction_reason = payload["correction_reason"] or None

    state.recommendation = RecommendationPoint(
        mode=annotation.recommendation_mode,
        x=annotation.x,
        y=annotation.y,
        x_end=annotation.x_end,
        trace_name=annotation.trace_name,
        locked=True,
    )
    state.recommendation_mode = annotation.recommendation_mode
    _persist_annotations(state)
    _write_to_task_memory(state, annotation)

    return JSONResponse(
        jsonable_encoder(
            {
                "recommendation": asdict(state.recommendation),
                "saved_annotation": asdict(annotation),
                "saved_annotations": [
                    asdict(item) for item in _filtered_annotations(state, state.selected_well_value)
                ],
            }
        )
    )


@app.delete("/api/annotations/{session_id}/{annotation_id}")
async def delete_annotation(session_id: str, annotation_id: str) -> JSONResponse:
    state = _get_session(session_id)
    _, annotation = _find_annotation(state, annotation_id)
    state.saved_annotations = [
        item for item in state.saved_annotations if item.annotation_id != annotation_id
    ]
    if state.recommendation.locked and state.recommendation.x == annotation.x and state.recommendation.x_end == annotation.x_end:
        state.recommendation = RecommendationPoint(mode=state.recommendation_mode, locked=False)
    _persist_annotations(state)
    return JSONResponse(
        jsonable_encoder(
            {
                "deleted_annotation_id": annotation_id,
                "saved_annotations": [
                    asdict(item) for item in _filtered_annotations(state, state.selected_well_value)
                ],
            }
        )
    )


@app.get("/api/annotations/{session_id}")
async def get_annotations(session_id: str, well_value: Optional[str] = None) -> JSONResponse:
    state = _get_session(session_id)
    annotations = _filtered_annotations(state, well_value)
    return JSONResponse(
        jsonable_encoder(
            {
                "annotations": [asdict(item) for item in annotations],
                "annotations_path": state.annotations_path,
            }
        )
    )


@app.get("/api/session/{session_id}")
async def get_session(session_id: str) -> JSONResponse:
    state = _get_session(session_id)
    return JSONResponse(json.loads(json.dumps(asdict(state), default=str)))


@app.get("/api/learn/{session_id}")
async def get_learning_summary(session_id: str) -> JSONResponse:
    """Return correction patterns and mined rule drafts from TaskMemory."""
    state = _get_session(session_id)
    if not state.task_spec:
        return JSONResponse({"n_total": 0, "n_confirmed": 0, "n_corrections": 0,
                             "patterns": [], "rule_drafts": []})
    try:
        from learning.task_memory import TaskMemory
        from learning.rule_miner import mine

        memory = TaskMemory(state.task_spec.task_id)
        patterns = memory.correction_patterns()
        drafts = mine(memory, llm_client=None, min_pattern_count=2)
        return JSONResponse({
            "n_total": len(memory.all()),
            "n_confirmed": len(memory.confirmed()),
            "n_corrections": len(memory.corrections()),
            "patterns": patterns,
            "rule_drafts": [
                {
                    "rule_id": d.rule_id,
                    "label": d.label,
                    "description": d.description,
                    "rationale": d.rationale,
                    "priority": d.priority,
                }
                for d in drafts
            ],
        })
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)


@app.get("/api/task-spec/{session_id}")
async def get_task_spec(session_id: str) -> JSONResponse:
    state = _get_session(session_id)
    return JSONResponse(
        jsonable_encoder(
            {
                "task_spec": asdict(state.task_spec) if state.task_spec else None,
                "task_spec_path": state.task_spec_path,
            }
        )
    )
