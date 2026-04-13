from __future__ import annotations

import hashlib
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

from agents.explanation_agent import explain as explain_review_candidate
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
)
from app.maintenance_utils import load_maintenance_documents, serialize_maintenance_document
from app.mistral_client import MistralChatClient
from app.session_store import FileSessionStore
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
from core.canonical_schema import (
    CandidateEvent,
    ContextBundle,
    DateRange,
    LocalFeatures,
    MaintenanceDocument,
    RuleResult,
    RuleTrace,
    StructuredFacts,
)
from core.pipeline_runner import PipelineRunner, build_context_bundle
from core.policy_engine import route as route_candidate
from core.task_manager import (
    ReviewPolicy as CoreReviewPolicy,
    SignalSpec as CoreSignalSpec,
    TaskSpec as CoreTaskSpec,
)


app = FastAPI(title="Oil Well Labeling UI")
templates = Jinja2Templates(directory="app/templates")
LABELS_DIR = Path("data/review_labels")
LABELS_DIR.mkdir(parents=True, exist_ok=True)
SESSION_STORE = FileSessionStore()


def _default_window_size(profile) -> Optional[int]:
    """Return a default window size in *data points* (not seconds).

    ``profile.inferred_window_size`` is the median sampling step in seconds.
    It must not be used directly as a point count — that causes the statistical
    detector to require far more data points than actually exist.
    """
    rows = getattr(profile, "rows", 0) or 0
    if rows < 12:
        return None
    # ~5 % of the series, clamped to [6, 50] points
    return max(6, min(50, rows // 20))


def _get_session(session_id: str) -> SessionState:
    state = SESSION_STORE.load(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    return state


def _save_session(state: SessionState) -> None:
    SESSION_STORE.save(state)


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
    if state.dataframe_path:
        path = Path(state.dataframe_path)
        if not path.exists():
            raise HTTPException(status_code=400, detail="Файл исходных данных сессии не найден")
        return load_tabular_file(state.filename or path.name, path.read_bytes())
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


def _parse_ts(value: Optional[str]) -> datetime:
    if not value:
        return datetime.now(tz=timezone.utc)
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return datetime.now(tz=timezone.utc)
    return parsed.to_pydatetime()


def _serialize_rule_trace(trace: RuleTrace) -> dict[str, object]:
    return asdict(trace)


def _deserialize_rule_trace(payload: Optional[dict[str, object]]) -> RuleTrace:
    payload = payload or {}
    return RuleTrace(
        rules_evaluated=list(payload.get("rules_evaluated") or []),
        rules_fired=list(payload.get("rules_fired") or []),
        rules_blocked=list(payload.get("rules_blocked") or []),
        winning_rule=payload.get("winning_rule"),
        conflict=bool(payload.get("conflict", False)),
        abstain_reason=payload.get("abstain_reason"),
    )


def _deserialize_local_features(payload: Optional[dict[str, object]], candidate_id: str) -> Optional[LocalFeatures]:
    if not payload:
        return None
    sanitized = {
        key: value
        for key, value in payload.items()
        if key in LocalFeatures.__dataclass_fields__
    }
    sanitized.setdefault("candidate_id", candidate_id)
    try:
        return LocalFeatures(**sanitized)
    except TypeError:
        return None


def _serialize_structured_fact(fact: StructuredFacts) -> dict[str, object]:
    return {
        "doc_id": fact.doc_id,
        "event_type": fact.event_type,
        "event_date": fact.event_date.isoformat() if fact.event_date else None,
        "asset_id": fact.asset_id,
        "duration_h": fact.duration_h,
        "action_summary": fact.action_summary,
        "parts_replaced": list(fact.parts_replaced or []),
        "extraction_confidence": fact.extraction_confidence,
    }


def _maintenance_documents_target(state: SessionState) -> Optional[Path]:
    if not state.task_spec:
        return None
    return Path(default_task_spec_path(state.task_spec.task_id)).parent / "maintenance_documents.json"


def _persist_maintenance_documents(state: SessionState) -> Optional[Path]:
    target = _maintenance_documents_target(state)
    if target is None:
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(
            [serialize_maintenance_document(doc) for doc in state.maintenance_documents],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return target


def _maintenance_signature(documents: list[MaintenanceDocument]) -> Optional[str]:
    if not documents:
        return None
    digest = hashlib.sha256()
    for doc in documents:
        digest.update((doc.doc_id or "").encode("utf-8"))
        digest.update((doc.asset_id or "").encode("utf-8"))
        digest.update(doc.event_date.isoformat().encode("utf-8"))
        digest.update((doc.raw_text or "").encode("utf-8"))
    return digest.hexdigest()


def _build_maintenance_context_summary(
    state: SessionState,
    *,
    pipeline_result=None,
    review_candidates: Optional[list[dict[str, object]]] = None,
) -> dict[str, object]:
    documents = state.maintenance_documents or []
    summary: dict[str, object] = {
        "filename": state.maintenance_upload_name,
        "document_count": len(documents),
        "documents_path": state.maintenance_documents_path,
        "llm_configured": settings.mistral_configured,
        "used_in_last_search": False,
        "fact_count": 0,
        "matched_candidate_count": 0,
        "low_confidence_fact_count": 0,
        "status": "not_uploaded" if not documents else "uploaded",
    }
    if not documents or pipeline_result is None:
        return summary

    summary["status"] = "applied"
    summary["used_in_last_search"] = True
    summary["fact_count"] = len(
        {
            (
                fact.doc_id,
                fact.event_type,
                fact.event_date.isoformat() if fact.event_date else "",
                fact.asset_id or "",
            )
            for fact in (pipeline_result.maintenance_facts or [])
        }
    )
    if review_candidates is not None:
        summary["matched_candidate_count"] = sum(
            1
            for item in review_candidates
            if item.get("maintenance_facts")
        )
    else:
        summary["matched_candidate_count"] = sum(
            1
            for bundle in (pipeline_result.context_bundles or [])
            if bundle.maintenance_facts
        )
    summary["low_confidence_fact_count"] = sum(
        1
        for fact in (pipeline_result.maintenance_facts or [])
        if fact.extraction_confidence in {"low", "failed"}
    )
    return summary


def _find_review_candidate(state: SessionState, candidate_id: Optional[str]) -> Optional[dict[str, object]]:
    if not candidate_id:
        return None
    return next(
        (item for item in state.review_candidates if item.get("candidate_id") == candidate_id),
        None,
    )


def _delete_from_task_memory(state: SessionState, annotation_id: str) -> None:
    if not state.task_spec:
        return
    try:
        from learning.task_memory import TaskMemory

        TaskMemory(state.task_spec.task_id).remove(annotation_id)
    except Exception:
        pass


def _build_core_task_spec(
    state: SessionState,
    *,
    selected_series: list[str],
    time_column: Optional[str],
    well_column: Optional[str],
    window_size: Optional[int],
    statistical_threshold_pct: Optional[float],
) -> CoreTaskSpec:
    source_spec = state.task_spec or build_initial_task_spec(state)
    units_by_signal = {
        item.name: item.unit
        for item in (source_spec.signal_schema or [])
    }
    signal_schema = [
        CoreSignalSpec(
            name=name,
            unit=units_by_signal.get(name),
            role="candidate_signal",
            selected_for_review=True,
        )
        for name in selected_series
    ]
    return CoreTaskSpec(
        task_id=source_spec.task_id,
        version=source_spec.version,
        title=source_spec.title,
        equipment_family=source_spec.equipment_family,
        primary_deviation=state.anomaly_goal or source_spec.primary_deviation,
        signal_schema=signal_schema,
        well_column=well_column,
        time_column=time_column,
        segmentation_strategy=source_spec.segmentation_strategy,
        minimum_segment_duration=window_size,
        feature_profile=list(source_spec.feature_profile),
        label_taxonomy=list(source_spec.label_taxonomy),
        unknown_label=source_spec.unknown_label,
        confounders=list(source_spec.confounders),
        context_sources=list(source_spec.context_sources),
        baseline_strategy=source_spec.baseline_strategy,
        quality_rules=list(source_spec.quality_rules),
        review_policy=CoreReviewPolicy(
            policy_name=source_spec.review_policy.policy_name,
            auto_label_allowed=False,
            allow_modify=source_spec.review_policy.allow_modify,
            allow_reject=source_spec.review_policy.allow_reject,
            allow_point=source_spec.review_policy.allow_point,
            allow_interval=source_spec.review_policy.allow_interval,
        ),
        normal_operation_definition=source_spec.normal_operation_definition,
        expected_deviation_frequency=source_spec.expected_deviation_frequency,
        statistical_threshold_pct=statistical_threshold_pct,
    )


def _fallback_candidate_id(
    well_value: Optional[str],
    series_name: Optional[str],
    start: str,
    end: str,
    reason: str,
) -> str:
    raw = "|".join([
        str(well_value or "unknown"),
        str(series_name or "unknown"),
        start,
        end,
        reason,
    ])
    return "fallback_" + hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]


def _unknown_rule_result(label: str = "unknown") -> RuleResult:
    return RuleResult(
        label=label,
        rule_trace=RuleTrace(abstain_reason="no_rule_matched"),
        abstain_reason="no_rule_matched",
    )


def _review_candidate_payload(
    candidate: CandidateEvent,
    *,
    rule_result: RuleResult,
    local_features: Optional[LocalFeatures],
    context: Optional[ContextBundle],
    source: str,
    routing: Optional[object] = None,
    explanation: Optional[str] = None,
) -> dict[str, object]:
    flags = list(candidate.flags)
    if getattr(routing, "disposition", None) == "mandatory_review":
        flags.append(str(getattr(routing, "reason", "mandatory_review")))
    if context:
        flags.extend(context.flags or [])
    return {
        "candidate_id": candidate.candidate_id,
        "start": candidate.segment.start.isoformat(),
        "end": candidate.segment.end.isoformat(),
        "series_name": candidate.series_name,
        "source": source,
        "deviation_type": candidate.deviation_type,
        "deviation_score": candidate.deviation_score,
        "reason": candidate.deviation_type,
        "proposed_label": rule_result.label,
        "routing": getattr(routing, "disposition", "mandatory_review"),
        "routing_reason": getattr(routing, "reason", rule_result.abstain_reason or "review"),
        "confidence": getattr(routing, "confidence", 0.0),
        "flags": list(dict.fromkeys(flags)),
        "winning_rule": rule_result.rule_trace.winning_rule,
        "rule_trace": _serialize_rule_trace(rule_result.rule_trace),
        "abstain_reason": rule_result.abstain_reason,
        "conflict_flag": rule_result.conflict_flag,
        "local_features": asdict(local_features) if local_features else {},
        "context_flags": list(context.flags or []) if context else [],
        "maintenance_facts": [
            _serialize_structured_fact(item)
            for item in (context.maintenance_facts or [])
        ] if context else [],
        "explanation": explanation or explain_review_candidate(rule_result),
    }


def _build_pipeline_review_candidates(
    pipeline_result,
    core_spec: CoreTaskSpec,
) -> list[dict[str, object]]:
    features_by_id = {
        item.candidate_id: item
        for item in pipeline_result.local_features
    }
    contexts_by_id = {
        item.candidate_id: item
        for item in pipeline_result.context_bundles
    }
    review_candidates: list[dict[str, object]] = []
    for index, candidate in enumerate(pipeline_result.candidates):
        rule_result = (
            pipeline_result.rule_results[index]
            if index < len(pipeline_result.rule_results)
            else _unknown_rule_result(core_spec.unknown_label)
        )
        context = contexts_by_id.get(candidate.candidate_id)
        local_features = features_by_id.get(candidate.candidate_id)
        routing = route_candidate(candidate, rule_result, core_spec)
        review_candidates.append(
            _review_candidate_payload(
                candidate,
                rule_result=rule_result,
                local_features=local_features,
                context=context,
                source="pipeline",
                routing=routing,
                explanation=explain_review_candidate(rule_result, context),
            )
        )
    return review_candidates


def _build_statistical_fallback_candidates(
    *,
    state: SessionState,
    core_spec: CoreTaskSpec,
    pipeline_result,
    plot_payload: dict[str, object],
    anomaly_goal: Optional[str],
    statistical_threshold_pct: Optional[float],
    window_size: Optional[int],
) -> list[dict[str, object]]:
    fallback = detect_candidate_intervals(
        plot_payload=plot_payload,
        anomaly_goal=anomaly_goal,
        window_size=window_size,
        statistical_threshold_pct=statistical_threshold_pct,
    )
    if not fallback:
        return []

    from rules.rule_engine import evaluate
    from rules.rule_schemas import RuleInput
    from rules.starter_ruleset import build_registry
    from signals.local_segment_analyzer import analyze

    registry = build_registry()
    series_by_name = {item.signal_col: item for item in pipeline_result.series}
    default_series = pipeline_result.series[0] if pipeline_result.series else None
    review_candidates: list[dict[str, object]] = []

    for item in fallback:
        start = str(item.get("start") or "")
        end = str(item.get("end") or start)
        reason = str(item.get("reason") or "statistical_shift")
        deviation_type = (
            "abrupt_transition"
            if reason == "change_in_local_amplitude"
            else "atypical_amplitude"
        )
        candidate = CandidateEvent(
            candidate_id=_fallback_candidate_id(
                state.selected_well_value,
                item.get("series_name"),
                start,
                end,
                reason,
            ),
            asset_id=state.selected_well_value or "unknown",
            segment=DateRange(start=_parse_ts(start), end=_parse_ts(end)),
            deviation_type=deviation_type,
            deviation_score=float(item.get("score") or 0.0),
            context_query=f"statistical_fallback:{reason}",
            series_name=item.get("series_name"),
            flags=["statistical_fallback"],
        )
        signal_series = series_by_name.get(candidate.series_name) or default_series
        local_features = analyze(candidate, signal_series) if signal_series else None
        context = build_context_bundle(
            candidate,
            maintenance_docs=state.maintenance_documents,
            maintenance_facts=getattr(pipeline_result, "maintenance_facts", []),
        )
        rule_result = evaluate(
            RuleInput(
                candidate=candidate,
                features=local_features,
                context=context,
                task_params={
                    "primary_deviation": core_spec.primary_deviation,
                    "equipment_family": core_spec.equipment_family,
                },
            ),
            registry,
            unknown_label=core_spec.unknown_label,
        )
        routing = route_candidate(candidate, rule_result, core_spec)
        payload = _review_candidate_payload(
            candidate,
            rule_result=rule_result,
            local_features=local_features,
            context=context,
            source="statistical_fallback",
            routing=routing,
            explanation=explain_review_candidate(rule_result, context),
        )
        payload["reason"] = reason
        review_candidates.append(payload)
    return review_candidates


def _review_cache_key(
    *,
    well_value: Optional[str],
    selected_series: list[str],
    time_column: Optional[str],
    well_column: Optional[str],
    window_size: Optional[int],
    anomaly_goal: Optional[str],
    statistical_threshold_pct: Optional[float],
    maintenance_signature: Optional[str],
) -> str:
    return json.dumps(
        {
            "well_value": well_value,
            "selected_series": selected_series,
            "time_column": time_column,
            "well_column": well_column,
            "window_size": window_size,
            "anomaly_goal": anomaly_goal,
            "statistical_threshold_pct": statistical_threshold_pct,
            "maintenance_signature": maintenance_signature,
        },
        ensure_ascii=False,
        sort_keys=True,
    )


def _build_review_candidates(
    state: SessionState,
    df: pd.DataFrame,
    *,
    selected_series: list[str],
    time_column: Optional[str],
    well_column: Optional[str],
    well_value: Optional[str],
    window_size: Optional[int],
    statistical_threshold_pct: Optional[float],
) -> list[dict[str, object]]:
    cache_key = _review_cache_key(
        well_value=well_value,
        selected_series=selected_series,
        time_column=time_column,
        well_column=well_column,
        window_size=window_size,
        anomaly_goal=state.anomaly_goal,
        statistical_threshold_pct=statistical_threshold_pct,
        maintenance_signature=_maintenance_signature(state.maintenance_documents),
    )
    if state.review_cache_key == cache_key and state.review_candidates:
        return list(state.review_candidates)

    if not time_column or not selected_series:
        state.review_cache_key = cache_key
        state.review_candidates = []
        state.maintenance_context_summary = _build_maintenance_context_summary(state)
        return []

    full_scope_frame = filter_dataframe(
        df=df,
        time_column=time_column,
        well_column=well_column,
        well_value=well_value,
        date_from=None,
        date_to=None,
    )
    if full_scope_frame.empty:
        state.review_cache_key = cache_key
        state.review_candidates = []
        state.maintenance_context_summary = _build_maintenance_context_summary(state)
        return []

    core_spec = _build_core_task_spec(
        state,
        selected_series=selected_series,
        time_column=time_column,
        well_column=well_column,
        window_size=window_size,
        statistical_threshold_pct=statistical_threshold_pct,
    )
    llm_client = MistralChatClient() if settings.mistral_configured else None
    runner = PipelineRunner(
        core_spec,
        llm_client=llm_client,
        llm_model=settings.mistral_resolved_model if settings.mistral_configured else None,
    )
    pipeline_result = runner.run(
        full_scope_frame.to_csv(index=False).encode("utf-8"),
        asset_id=well_value,
        filename="session.csv",
        maintenance_docs=state.maintenance_documents or None,
    )
    review_candidates = _build_pipeline_review_candidates(pipeline_result, core_spec)
    if not review_candidates:
        full_scope_plot = normalize_for_plot(
            df=full_scope_frame,
            time_column=time_column,
            well_column=None,
            well_value=None,
            date_from=None,
            date_to=None,
            series_names=selected_series,
        )
        review_candidates = _build_statistical_fallback_candidates(
            state=state,
            core_spec=core_spec,
            pipeline_result=pipeline_result,
            plot_payload=full_scope_plot,
            anomaly_goal=state.anomaly_goal,
            statistical_threshold_pct=statistical_threshold_pct,
            window_size=window_size,
        )
    state.maintenance_context_summary = _build_maintenance_context_summary(
        state,
        pipeline_result=pipeline_result,
        review_candidates=review_candidates,
    )

    state.review_cache_key = cache_key
    state.review_candidates = review_candidates
    return list(review_candidates)


def _resolve_review_decision(
    state: SessionState,
    payload: dict[str, object],
    candidate: Optional[dict[str, object]],
) -> tuple[str, Optional[str], str]:
    action = str(payload.get("review_action") or "").strip().lower()
    selected_label = payload.get("label")
    label = str(selected_label).strip() if isinstance(selected_label, str) and selected_label.strip() else None
    proposed_label = (
        str(candidate.get("proposed_label"))
        if candidate and candidate.get("proposed_label")
        else state.task_spec.unknown_label if state.task_spec else "unknown"
    )

    if action not in {"accept", "override", "reject", "ambiguous"}:
        action = "override" if label else "accept"

    if action == "override" and not label:
        raise HTTPException(status_code=400, detail="Для override выберите итоговую метку")

    if action == "accept":
        # If the user explicitly provided a label (manual annotation without a
        # candidate, or a correction during accept), honour it; otherwise fall
        # back to the system-proposed label.
        return action, label if label else proposed_label, "accepted"
    if action == "override":
        return action, label, "accepted"
    if action == "reject":
        return action, label or proposed_label, "rejected"
    fallback_label = state.task_spec.unknown_label if state.task_spec else "unknown"
    return action, label or fallback_label, "ambiguous"


def _write_to_task_memory(state: SessionState, annotation: SavedAnnotation) -> None:
    """Persist annotation as a LabelRecord in TaskMemory for learning."""
    if not state.task_spec or not annotation.x:
        return
    try:
        from learning.task_memory import TaskMemory
        from core.canonical_schema import LabelRecord

        rule_trace = _deserialize_rule_trace(annotation.rule_trace)
        proposed_label = annotation.proposed_label or state.task_spec.unknown_label
        final_label = annotation.label or proposed_label
        local_features = _deserialize_local_features(annotation.local_features, annotation.candidate_id or annotation.annotation_id)
        record = LabelRecord(
            record_id=annotation.annotation_id,
            task_id=state.task_spec.task_id,
            asset_id=annotation.well_value or state.selected_well_value or "unknown",
            segment=DateRange(
                start=_parse_ts(annotation.x),
                end=_parse_ts(annotation.x_end) if annotation.x_end else _parse_ts(annotation.x),
            ),
            deviation_type=annotation.deviation_type or state.anomaly_goal or state.task_spec.primary_deviation,
            local_features=local_features,
            rule_result=RuleResult(
                label=proposed_label,
                rule_trace=rule_trace,
                abstain_reason=rule_trace.abstain_reason,
                conflict_flag=rule_trace.conflict,
            ),
            final_label=final_label,
            was_override=annotation.review_action == "override",
            correction_reason=annotation.correction_reason,
            confirmed_at=datetime.now(tz=timezone.utc),
            status=annotation.review_status,
        )
        TaskMemory(state.task_spec.task_id).add(record)
    except Exception:
        pass  # memory write failure must not break annotation save


def _save_annotation(
    state: SessionState,
    *,
    label: Optional[str] = None,
    correction_reason: Optional[str] = None,
    review_action: Optional[str] = None,
    review_status: str = "accepted",
    candidate: Optional[dict[str, object]] = None,
) -> SavedAnnotation:
    proposed_label = str(candidate.get("proposed_label")) if candidate and candidate.get("proposed_label") else None
    final_label = label or proposed_label
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
        label=final_label or None,
        correction_reason=correction_reason or None,
        created_at=datetime.now(timezone.utc).isoformat(),
        candidate_id=str(candidate.get("candidate_id")) if candidate and candidate.get("candidate_id") else None,
        proposed_label=proposed_label,
        proposed_rule=str(candidate.get("winning_rule")) if candidate and candidate.get("winning_rule") else None,
        review_action=review_action,
        review_status=review_status,
        routing=str(candidate.get("routing")) if candidate and candidate.get("routing") else None,
        candidate_source=str(candidate.get("source")) if candidate and candidate.get("source") else None,
        deviation_type=str(candidate.get("deviation_type")) if candidate and candidate.get("deviation_type") else None,
        deviation_score=float(candidate.get("deviation_score")) if candidate and candidate.get("deviation_score") is not None else None,
        explanation=str(candidate.get("explanation")) if candidate and candidate.get("explanation") else None,
        rule_trace=dict(candidate.get("rule_trace") or {}) if candidate else {},
        local_features=dict(candidate.get("local_features") or {}) if candidate else {},
    )
    state.saved_annotations.append(annotation)
    _persist_annotations(state)
    _write_to_task_memory(state, annotation)
    return annotation


def _filter_unreviewed_candidates(
    candidates: list[dict],
    saved_annotations: list[SavedAnnotation],
    well_value: Optional[str],
) -> list[dict]:
    """Remove candidates that already have a completed review action."""
    if not candidates or not saved_annotations:
        return candidates
    reviewed_candidate_ids = {
        item.candidate_id
        for item in saved_annotations
        if item.candidate_id
        and item.review_status in {"accepted", "rejected", "ambiguous"}
        and (well_value is None or item.well_value == well_value)
    }
    annotated: list[tuple[str, str]] = [
        (a.x, a.x_end or a.x)
        for a in saved_annotations
        if a.x
        and a.review_status in {"accepted", "rejected", "ambiguous"}
        and (well_value is None or a.well_value == well_value)
    ]
    if not annotated and not reviewed_candidate_ids:
        return candidates

    def _overlaps(c: dict) -> bool:
        candidate_id = c.get("candidate_id")
        if candidate_id and candidate_id in reviewed_candidate_ids:
            return True
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


def _apply_chat_control_updates(state: SessionState, payload: dict[str, object]) -> None:
    threshold = payload.get("statistical_threshold_pct")
    if threshold not in (None, ""):
        try:
            parsed_threshold = float(threshold)
        except (TypeError, ValueError):
            parsed_threshold = None
        if parsed_threshold is not None and parsed_threshold > 0:
            state.statistical_threshold_pct = parsed_threshold

    mode = payload.get("recommendation_mode")
    if mode in {"point", "interval"}:
        state.recommendation_mode = str(mode)

    window_size = payload.get("window_size")
    if window_size not in (None, ""):
        try:
            parsed_window_size = int(window_size)
        except (TypeError, ValueError):
            parsed_window_size = None
        if parsed_window_size is not None and parsed_window_size > 0:
            state.window_size = parsed_window_size


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
    session_id = new_id()
    session = SessionState(
        session_id=session_id,
        filename=file.filename,
        profile=profile,
        selected_well_column=profile.inferred_well_column,
        selected_time_column=profile.inferred_time_column,
        selected_series=profile.numeric_candidates[:1],
        selected_well_value=profile.sheet_names[0] if profile.detected_multiple_wells and profile.sheet_names else None,
        window_size=_default_window_size(profile),
        annotations_path=str(LABELS_DIR / f"{file.filename or 'session'}_{new_id()}.json"),
    )
    dataset_path = SESSION_STORE.save_uploaded_dataset(session.session_id, file.filename or "dataset.csv", content)
    session.dataframe_path = str(dataset_path)
    session.task_spec = build_initial_task_spec(session)
    session.task_spec_path = str(default_task_spec_path(session.task_spec.task_id))
    _persist_task_spec(session)
    session.maintenance_context_summary = _build_maintenance_context_summary(session)
    initial_message = build_initial_message(session)
    session.messages.append({"role": "assistant", "content": initial_message})
    _save_session(session)
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
            "maintenance_context": session.maintenance_context_summary,
            "llm": {
                "provider": "mistral",
                "model": settings.mistral_resolved_model,
                "configured": settings.mistral_configured,
            },
        })
    )


@app.post("/api/maintenance/{session_id}")
async def upload_maintenance(session_id: str, file: UploadFile = File(...)) -> JSONResponse:
    state = _get_session(session_id)
    content = await file.read()
    fallback_asset_id = state.selected_well_value or _default_well_value(state)
    try:
        documents = load_maintenance_documents(
            file.filename or "maintenance.txt",
            content,
            fallback_asset_id=fallback_asset_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    state.maintenance_documents = documents
    state.maintenance_upload_name = file.filename or "maintenance.txt"
    persisted_path = _persist_maintenance_documents(state)
    state.maintenance_documents_path = str(persisted_path) if persisted_path else None
    state.review_candidates = []
    state.review_cache_key = None
    state.maintenance_context_summary = _build_maintenance_context_summary(state)
    _save_session(state)
    return JSONResponse(
        jsonable_encoder(
            {
                "maintenance_context": state.maintenance_context_summary,
            }
        )
    )


@app.post("/api/chat/{session_id}")
async def chat(session_id: str, request: Request) -> JSONResponse:
    state = _get_session(session_id)
    payload = await request.json()
    user_message = (payload.get("message") or "").strip()
    if not user_message:
        raise HTTPException(status_code=400, detail="Сообщение пустое")

    _apply_chat_control_updates(state, payload)

    state.messages.append({"role": "user", "content": user_message})
    llm_result = generate_reply(state, user_message)
    apply_discovery_updates(state, llm_result.get("updates") or {})
    if state.task_spec:
        apply_task_spec_updates(state.task_spec, llm_result.get("task_spec_updates") or {})
        sync_task_spec_from_state(state.task_spec, state)
        _persist_task_spec(state)
    reply = llm_result["reply"]
    state.messages.append({"role": "assistant", "content": reply})
    _save_session(state)
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
    use_scope_dates: bool = False,
    detect_candidates: bool = False,
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
    scope_time_range = get_scope_time_range(
        df=df,
        time_column=resolved_time_column,
        well_column=resolved_well_column,
        well_value=resolved_well_value,
    )
    if use_scope_dates:
        resolved_date_from = scope_time_range.get("time_min")
        resolved_date_to = scope_time_range.get("time_max")
    else:
        resolved_date_from = date_from or state.date_from or scope_time_range.get("time_min")
        resolved_date_to = date_to or state.date_to or scope_time_range.get("time_max")
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
    candidate_intervals: list[dict[str, object]] = []
    candidate_interval_stats: list[dict[str, object]] = []
    if detect_candidates:
        filtered_frame = filter_dataframe(
            df=df,
            time_column=resolved_time_column,
            well_column=resolved_well_column,
            well_value=resolved_well_value,
            date_from=resolved_date_from,
            date_to=resolved_date_to,
        )
        review_candidates = _build_review_candidates(
            state,
            df,
            selected_series=resolved_series,
            time_column=resolved_time_column,
            well_column=resolved_well_column,
            well_value=resolved_well_value,
            window_size=resolved_window_size,
            statistical_threshold_pct=resolved_statistical_threshold_pct,
        )
        candidate_intervals = _filter_unreviewed_candidates(
            review_candidates,
            state.saved_annotations,
            resolved_well_value,
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

    _save_session(state)
    return JSONResponse(
        jsonable_encoder({
            "plot": plot_payload,
            "candidate_intervals": candidate_intervals,
            "candidate_interval_stats": candidate_interval_stats,
            "recommendation": asdict(state.recommendation),
            "selected_well_value": resolved_well_value,
            "resolved_date_from": resolved_date_from,
            "resolved_date_to": resolved_date_to,
            "candidates_computed": detect_candidates,
            "plot_warning": plot_warning,
            "scope_time_range": scope_time_range,
            "window_size": resolved_window_size,
            "statistical_threshold_pct": resolved_statistical_threshold_pct,
            "recommendation_mode": resolved_recommendation_mode,
            "saved_annotations": [asdict(item) for item in _filtered_annotations(state, resolved_well_value)],
            "maintenance_context": state.maintenance_context_summary or _build_maintenance_context_summary(state),
        })
    )


@app.post("/api/recommendation/{session_id}")
async def set_recommendation(session_id: str, request: Request) -> JSONResponse:
    state = _get_session(session_id)
    payload = await request.json()
    candidate = _find_review_candidate(state, payload.get("candidate_id"))
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
        review_action, final_label, review_status = _resolve_review_decision(state, payload, candidate)
        saved_annotation = _save_annotation(
            state,
            label=final_label,
            correction_reason=payload.get("correction_reason"),
            review_action=review_action,
            review_status=review_status,
            candidate=candidate,
        )
    _save_session(state)
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
    candidate = _find_review_candidate(state, payload.get("candidate_id") or annotation.candidate_id)
    if candidate is None and annotation.candidate_id:
        candidate = {
            "candidate_id": annotation.candidate_id,
            "proposed_label": annotation.proposed_label,
            "winning_rule": annotation.proposed_rule,
            "routing": annotation.routing,
            "source": annotation.candidate_source,
            "deviation_type": annotation.deviation_type,
            "deviation_score": annotation.deviation_score,
            "rule_trace": annotation.rule_trace,
            "local_features": annotation.local_features,
            "explanation": annotation.explanation,
        }
    review_action, final_label, review_status = _resolve_review_decision(state, payload, candidate)

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
    annotation.label = final_label
    annotation.correction_reason = payload.get("correction_reason") or None
    annotation.candidate_id = (
        str(candidate.get("candidate_id"))
        if candidate and candidate.get("candidate_id")
        else annotation.candidate_id
    )
    annotation.proposed_label = (
        str(candidate.get("proposed_label"))
        if candidate and candidate.get("proposed_label")
        else annotation.proposed_label
    )
    annotation.proposed_rule = (
        str(candidate.get("winning_rule"))
        if candidate and candidate.get("winning_rule")
        else annotation.proposed_rule
    )
    annotation.review_action = review_action
    annotation.review_status = review_status
    annotation.routing = (
        str(candidate.get("routing"))
        if candidate and candidate.get("routing")
        else annotation.routing
    )
    annotation.candidate_source = (
        str(candidate.get("source"))
        if candidate and candidate.get("source")
        else annotation.candidate_source
    )
    annotation.deviation_type = (
        str(candidate.get("deviation_type"))
        if candidate and candidate.get("deviation_type")
        else annotation.deviation_type
    )
    annotation.deviation_score = (
        float(candidate.get("deviation_score"))
        if candidate and candidate.get("deviation_score") is not None
        else annotation.deviation_score
    )
    annotation.explanation = (
        str(candidate.get("explanation"))
        if candidate and candidate.get("explanation")
        else annotation.explanation
    )
    annotation.rule_trace = dict(candidate.get("rule_trace") or {}) if candidate else annotation.rule_trace
    annotation.local_features = dict(candidate.get("local_features") or {}) if candidate else annotation.local_features

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
    _save_session(state)

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
    _delete_from_task_memory(state, annotation_id)
    _save_session(state)
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
