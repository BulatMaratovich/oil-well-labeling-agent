from __future__ import annotations

import json
import re
from typing import Any
from datetime import datetime
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.config import settings
from app.models import SessionState


SYSTEM_PROMPT = (
    "Ты discovery-ассистент для промышленной разметки временных рядов. "
    "Работаешь в двух режимах:\n\n"
    "РЕЖИМ 1 — Настройка (до первого прохода):\n"
    "Помоги выбрать ряды, скважину, диапазон дат, тип отклонений, режим (точка/интервал) и размер окна. "
    "Если пользователь не знает с чего начать — скажи ему, что параметры можно задать прямо "
    "на панели инструментов слева: выбрать скважину в выпадающем списке, "
    "указать диапазон дат и нажать 'Обновить график'. "
    "Через чат удобно задавать аналитические параметры (тип отклонения, порог, режим).\n\n"
    "РЕЖИМ 2 — Разметка (после появления кандидатов):\n"
    "Справа отображается панель кандидатов — аномальных интервалов, найденных агентом. "
    "Для каждого кандидата: нажать 'Открыть' → изучить интервал на графике → "
    "выбрать метку (belt_break, planned_stop, sensor_issue и т.д.) → "
    "добавить комментарий при необходимости → нажать 'Интервал ✓'. "
    "Если предложенный диапазон неточен — выбрать другой интервал мышкой и уточнить причину. "
    "После каждой разметки агент автоматически пересматривает оставшихся кандидатов. "
    "Нельзя выдавать финальную разметку как факт. "
    "Задавай не больше двух вопросов за раз."
)

JSON_INSTRUCTION = (
    "Верни только JSON-объект без markdown и без пояснений. "
    "Схема ответа: "
    '{'
    '"assistant_reply": "короткий ответ пользователю", '
    '"updates": {'
    '"selected_series": ["ряд1", "ряд2"], '
    '"selected_well_value": "имя скважины или листа либо null", '
    '"date_from": "ISO-дата или null", '
    '"date_to": "ISO-дата или null", '
    '"anomaly_goal": "что искать или null", '
    '"chart_preferences": "как показывать графики или null", '
    '"recommendation_mode": "point|interval|null", '
    '"window_size": 120, '
    '"statistical_threshold_pct": 30'
    "}, "
    '"task_spec_updates": {'
    '"equipment_family": "rod_pump_unit or null", '
    '"primary_deviation": "belt_break or null", '
    '"normal_operation_definition": "what counts as normal or null", '
    '"confounders": ["planned_stop", "sensor_issue"], '
    '"context_sources": ["maintenance_reports", "equipment_metadata"], '
    '"minimum_segment_duration": 120, '
    '"expected_deviation_frequency": "rare|occasional|unknown|null"'
    ', "statistical_threshold_pct": 30'
    "}, "
    '"ready_for_first_pass": true'
    '}. '
    "Если не хочешь менять поле, верни null для скалярного поля или [] для selected_series."
)


def build_initial_message(state: SessionState) -> str:
    if not state.profile:
        return "Сначала загрузите файл."

    profile = state.profile
    parts: list[str] = []
    parts.append(f"Файл загружен: {state.filename}, строк: {profile.rows}.")

    if profile.inferred_time_column:
        parts.append(f"Похоже, временная ось: `{profile.inferred_time_column}`.")
    else:
        parts.append("Колонку времени автоматически определить не удалось.")

    if profile.inferred_well_column:
        if profile.detected_multiple_wells:
            if profile.source_sheet_column and profile.inferred_well_column == profile.source_sheet_column:
                parts.append(
                    f"Похоже, в Excel несколько листов, и каждый лист трактуется как отдельная скважина. Доступно листов: {profile.unique_well_count}."
                )
            else:
                parts.append(
                    f"Похоже, в данных несколько скважин: примерно {profile.unique_well_count}. Базовая колонка скважины: `{profile.inferred_well_column}`."
                )
        else:
            parts.append(f"Колонка идентификатора скважины: `{profile.inferred_well_column}`.")

    if profile.time_min and profile.time_max:
        parts.append(f"Диапазон времени: от `{profile.time_min}` до `{profile.time_max}`.")

    if profile.numeric_candidates:
        preview = ", ".join(f"`{name}`" for name in profile.numeric_candidates[:6])
        parts.append(f"Числовые ряды, которые можно вывести: {preview}.")

    parts.append(
        "Уточните, какие столбцы показать на графике, что считать отклонением, "
        "какой размер центрального окна нужен для рекомендации, и нужна ли рекомендация точкой или интервалом."
    )
    return " ".join(parts)


def _is_review_phase(state: SessionState) -> bool:
    """True if the user has already completed setup and is in the labeling phase."""
    return bool(
        state.selected_series
        and (state.anomaly_goal or (state.task_spec and state.task_spec.primary_deviation))
        and (state.selected_well_value or (state.profile and not state.profile.detected_multiple_wells))
    )


def _review_guidance_reply(state: SessionState) -> str:
    parts = [
        "Вы находитесь в режиме разметки. В правой панели перечислены кандидаты — "
        "интервалы, которые агент счёл аномальными."
    ]
    parts.append(
        "Для каждого кандидата: нажмите 'Открыть' → изучите интервал на графике → "
        "выберите метку из выпадающего списка → при желании добавьте комментарий в поле рядом → "
        "нажмите 'Интервал ✓' (или 'Точка ✓' если режим — точка)."
    )
    parts.append(
        "Метки: 'candidate_deviation' или 'belt_break' — это реальное отклонение; "
        "'planned_stop' / 'planned_maintenance' — плановые события; "
        "'sensor_issue' — проблема с датчиком; 'stable_unusual_regime' — нетипичный, но стабильный режим."
    )
    parts.append(
        "Если предложенный диапазон неточен — выделите нужный участок мышкой прямо на графике, "
        "затем выберите метку и подтвердите. После каждой разметки список кандидатов обновляется."
    )
    if state.selected_well_value:
        parts.append(
            f"Текущая скважина: {state.selected_well_value}. "
            "Чтобы перейти к другой — выберите её в выпадающем списке на панели инструментов."
        )
    return " ".join(parts)


def generate_reply(state: SessionState, user_message: str) -> dict[str, Any]:
    lower = user_message.lower()
    # Confused user in review phase → give labeling guidance without calling LLM
    if _is_review_phase(state) and any(m in lower for m in _CONFUSED_MARKERS):
        return {
            "reply": _review_guidance_reply(state),
            "updates": {},
            "task_spec_updates": {},
            "ready_for_first_pass": False,
            "mode": "guidance",
        }

    local_result = infer_message_updates(user_message, state)
    if settings.mistral_configured:
        try:
            llm_result = _generate_reply_with_mistral(state, user_message)
            llm_result["updates"] = _merge_updates(local_result["updates"], llm_result.get("updates") or {})
            llm_result["task_spec_updates"] = _merge_updates(
                local_result["task_spec_updates"], llm_result.get("task_spec_updates") or {}
            )
            llm_result["reply"] = build_guided_reply(
                state,
                llm_result["updates"],
                llm_result["task_spec_updates"],
                llm_result.get("ready_for_first_pass", False),
                llm_result.get("reply"),
            )
            llm_result["mode"] = "live"
            return llm_result
        except Exception as exc:
            fallback = _fallback_result(state, user_message, local_result)
            fallback["mode"] = "fallback"
            fallback["error"] = str(exc)
            return fallback

    fallback = _fallback_result(state, user_message, local_result)
    fallback["mode"] = "fallback"
    fallback["error"] = None
    return fallback


def apply_discovery_updates(state: SessionState, updates: dict[str, Any]) -> None:
    if not updates:
        return

    if state.profile:
        candidates = set(state.profile.numeric_candidates)
        selected_series = [
            item for item in (updates.get("selected_series") or []) if item in candidates
        ]
        if selected_series:
            state.selected_series = selected_series

        if state.profile.inferred_well_column:
            well_values = set(state.profile.sheet_names)
            if not well_values and state.selected_well_value:
                well_values.add(state.selected_well_value)
            selected_well_value = updates.get("selected_well_value")
            if selected_well_value and (not well_values or selected_well_value in well_values):
                state.selected_well_value = selected_well_value

    if updates.get("date_from"):
        df = str(updates["date_from"])
        # Bare date → start of day
        if re.match(r'^\d{4}-\d{2}-\d{2}$', df):
            df += "T00:00:00"
        state.date_from = df
    if updates.get("date_to"):
        dt = str(updates["date_to"])
        # Bare date → end of day so the entire day is included
        if re.match(r'^\d{4}-\d{2}-\d{2}$', dt):
            dt += "T23:59:59"
        state.date_to = dt
    if updates.get("anomaly_goal"):
        state.anomaly_goal = str(updates["anomaly_goal"])
    if updates.get("chart_preferences"):
        state.chart_preferences = str(updates["chart_preferences"])
    threshold = updates.get("statistical_threshold_pct")
    if threshold not in (None, ""):
        try:
            parsed_threshold = float(threshold)
        except (TypeError, ValueError):
            parsed_threshold = None
        if parsed_threshold is not None and parsed_threshold > 0:
            state.statistical_threshold_pct = parsed_threshold

    mode = updates.get("recommendation_mode")
    if mode in {"point", "interval"}:
        state.recommendation_mode = mode

    window_size = updates.get("window_size")
    if window_size not in (None, ""):
        try:
            parsed = int(window_size)
        except (TypeError, ValueError):
            parsed = None
        if parsed and parsed > 0:
            state.window_size = parsed


def infer_settings_from_message(user_message: str) -> dict[str, object]:
    lower = user_message.lower()
    inferred: dict[str, object] = {}

    anomaly_goal = infer_anomaly_goal_from_message(user_message)
    if anomaly_goal:
        inferred["anomaly_goal"] = anomaly_goal

    if any(marker in lower for marker in ("граф", "plot", "подграф", "маркер", "линия", "overlay", "subplot")):
        inferred["chart_preferences"] = user_message

    threshold_pct = infer_statistical_threshold_pct_from_message(user_message)
    if threshold_pct is not None:
        inferred["statistical_threshold_pct"] = threshold_pct

    match = re.search(r"окн\w*\D+(\d+)|window\D+(\d+)", lower)
    if match:
        inferred["window_size"] = int(next(group for group in match.groups() if group))

    if "интервал" in lower:
        inferred["recommendation_mode"] = "interval"
    if "точк" in lower:
        inferred["recommendation_mode"] = "point"

    date_from_match = re.search(r"(?:с|from)\s+(\d{4}-\d{2}-\d{2})", lower)
    date_to_match = re.search(r"(?:по|to|until)\s+(\d{4}-\d{2}-\d{2})", lower)
    if date_from_match:
        inferred["date_from"] = date_from_match.group(1)
    if date_to_match:
        inferred["date_to"] = date_to_match.group(1)

    return inferred


def infer_series_from_message(user_message: str, candidates: list[str]) -> list[str]:
    lower = user_message.lower()
    normalized_message = normalize_token(user_message)
    message_tokens = tokenize_text(user_message)
    matched: list[str] = []
    for candidate in candidates:
        candidate_lower = candidate.lower()
        normalized_candidate = normalize_token(candidate)
        if (
            candidate_lower in lower
            or (normalized_candidate and normalized_candidate in normalized_message)
            or candidate_tokens_match(candidate, message_tokens)
        ):
            matched.append(candidate)
    return matched


def infer_message_updates(user_message: str, state: SessionState) -> dict[str, Any]:
    updates: dict[str, Any] = {}
    task_spec_updates: dict[str, Any] = {}

    inferred = infer_settings_from_message(user_message)
    if inferred.get("date_from"):
        updates["date_from"] = inferred["date_from"]
    if inferred.get("date_to"):
        updates["date_to"] = inferred["date_to"]
    if inferred.get("anomaly_goal"):
        updates["anomaly_goal"] = inferred["anomaly_goal"]
    if inferred.get("chart_preferences"):
        updates["chart_preferences"] = inferred["chart_preferences"]
    if inferred.get("statistical_threshold_pct") is not None:
        updates["statistical_threshold_pct"] = inferred["statistical_threshold_pct"]
        task_spec_updates["statistical_threshold_pct"] = inferred["statistical_threshold_pct"]
    if inferred.get("recommendation_mode"):
        updates["recommendation_mode"] = inferred["recommendation_mode"]

    inferred_window_size = infer_window_size_from_message(user_message, state)
    if inferred_window_size:
        updates["window_size"] = inferred_window_size
        task_spec_updates["minimum_segment_duration"] = inferred_window_size

    if state.profile:
        selected_series = infer_series_from_message(user_message, state.profile.numeric_candidates)
        if selected_series:
            updates["selected_series"] = selected_series

        well_value = infer_well_from_message(user_message, state.profile.sheet_names)
        if well_value:
            updates["selected_well_value"] = well_value

        dates = infer_dates_from_message(user_message, state)
        if dates.get("date_from"):
            updates["date_from"] = dates["date_from"]
        if dates.get("date_to"):
            updates["date_to"] = dates["date_to"]

    task_spec_updates.update(infer_task_spec_updates_from_message(user_message))
    return {
        "updates": updates,
        "task_spec_updates": task_spec_updates,
    }


def _fallback_result(state: SessionState, user_message: str, local_result: dict[str, Any] | None = None) -> dict[str, Any]:
    local_result = local_result or {"updates": {}, "task_spec_updates": {}}
    return {
        "reply": build_guided_reply(
            state,
            local_result.get("updates") or {},
            local_result.get("task_spec_updates") or {},
            False,
            _fallback_reply(state, user_message),
        ),
        "updates": local_result.get("updates") or {},
        "task_spec_updates": local_result.get("task_spec_updates") or {},
        "ready_for_first_pass": False,
    }


_CONFUSED_MARKERS = (
    "что делать", "непонятно", "как работать", "с чего начать", "помогите",
    "дальше что", "не знаю", "подскажи", "что дальше", "как пользоваться",
    "помощь", "help", "инструкция", "не понимаю", "объясни",
)


def _fallback_reply(state: SessionState, user_message: str) -> str:
    lower = user_message.lower()

    # If user seems confused and setup is already done → give review guidance
    if _is_review_phase(state) and any(m in lower for m in _CONFUSED_MARKERS):
        return _review_guidance_reply(state)

    prompts: list[str] = []

    if not state.selected_series:
        prompts.append("Сначала выберите хотя бы один числовой ряд для графика.")
    if not state.selected_time_column:
        prompts.append("Нужно подтвердить колонку времени.")
    if state.profile and state.profile.detected_multiple_wells and not state.selected_well_value:
        prompts.append("В файле несколько скважин. Выберите, какую скважину открыть первой.")
    if state.profile and state.profile.rows > 100_000 and not (state.selected_well_value or state.date_from or state.date_to):
        prompts.append("Файл большой. Лучше сразу сузить данные по скважине или диапазону дат перед построением графика.")
    anomaly_markers = ("отклон", "аномал", "падени", "скач", "останов", "шум", "полк", "обрыв")
    chart_markers = ("граф", "plot", "подграф", "маркер", "линия", "overlay", "subplot")
    if not state.anomaly_goal and not any(marker in lower for marker in anomaly_markers):
        prompts.append("Опишите, какие отклонения вы хотите искать: резкое падение, длительная полка, шум, останов и т.д.")
    if (
        state.anomaly_goal
        and ("статист" in state.anomaly_goal.lower() or "%" in state.anomaly_goal or "процент" in state.anomaly_goal.lower())
        and state.statistical_threshold_pct is None
    ):
        prompts.append("Уточните порог статистического сдвига в процентах, например 30%.")
    if not state.chart_preferences and not any(marker in lower for marker in chart_markers):
        prompts.append("Уточните вид представления: один график, несколько подграфиков, наложение рядов, маркеры, диапазон окна.")
    if state.window_size is None and "окн" not in lower:
        prompts.append("Задайте размер центрального окна в точках или секундах.")
    if state.recommendation_mode not in ("point", "interval"):
        prompts.append("Уточните, рекомендация должна быть точкой или интервалом.")

    if prompts:
        tip = (
            " Подсказка: скважину, диапазон дат и порог можно задать прямо на панели инструментов — "
            "выпадающий список 'Лист / скважина', поля 'Дата от / до' и кнопка 'Обновить график'."
        )
        return " ".join(prompts) + tip

    return (
        "Параметров уже достаточно для первого прохода. Построю график и покажу кандидатов. "
        "Дальше: в правой панели откройте кандидата → выберите метку → нажмите 'Интервал ✓'. "
        "Если интервал неточен — выделите другой диапазон на графике и укажите причину."
    )


def build_guided_reply(
    state: SessionState,
    updates: dict[str, Any],
    task_spec_updates: dict[str, Any],
    llm_ready_for_first_pass: bool,
    llm_reply: str | None,
) -> str:
    projected = project_state(state, updates, task_spec_updates)
    summary = build_capture_summary(projected, updates, task_spec_updates)
    missing = get_missing_slots(projected)

    if not missing or llm_ready_for_first_pass:
        anomaly_goal = str(projected.get("anomaly_goal") or "").lower()
        if "статист" in anomaly_goal or "%" in anomaly_goal or "процент" in anomaly_goal:
            candidate_phrase = "предложу кандидаты, где статистические параметры сдвигаются относительно baseline"
        else:
            candidate_phrase = "предложу интервалы, где заметны изменения амплитуды"
        ready = (
            f"Покажу график по выбранной скважине и периоду и {candidate_phrase}. "
            "Кандидаты появятся в правой панели. "
            "Для каждого: нажмите 'Открыть' → выберите метку → нажмите 'Интервал ✓'. "
            "Если диапазон неточен — выделите другой на графике и укажите причину. "
            "После каждой разметки список кандидатов обновляется автоматически."
        )
        return f"{summary} {ready}".strip() if summary else ready

    next_question = build_next_question(projected, missing[0])
    if summary:
        return f"{summary} {next_question}".strip()
    if llm_reply and not asks_for_filled_slot(llm_reply, projected):
        return llm_reply
    return next_question


def project_state(state: SessionState, updates: dict[str, Any], task_spec_updates: dict[str, Any]) -> dict[str, Any]:
    task_spec = state.task_spec
    return {
        "selected_series": updates.get("selected_series") or state.selected_series,
        "selected_well_value": updates.get("selected_well_value") or state.selected_well_value,
        "date_from": updates.get("date_from") or state.date_from,
        "date_to": updates.get("date_to") or state.date_to,
        "anomaly_goal": updates.get("anomaly_goal") or state.anomaly_goal or task_spec_updates.get("primary_deviation"),
        "recommendation_mode": updates.get("recommendation_mode") or state.recommendation_mode,
        "window_size": updates.get("window_size") or state.window_size,
        "statistical_threshold_pct": (
            updates.get("statistical_threshold_pct")
            if updates.get("statistical_threshold_pct") is not None
            else state.statistical_threshold_pct
        ),
        "rows": state.profile.rows if state.profile else 0,
        "multiple_wells": bool(state.profile and state.profile.detected_multiple_wells),
        "normal_operation_definition": task_spec_updates.get("normal_operation_definition")
        or (task_spec.normal_operation_definition if task_spec else None),
    }


def build_capture_summary(projected: dict[str, Any], updates: dict[str, Any], task_spec_updates: dict[str, Any]) -> str:
    captured: list[str] = []
    if updates.get("selected_series"):
        captured.append("ряды: " + ", ".join(f"`{item}`" for item in projected["selected_series"]))
    if updates.get("selected_well_value"):
        captured.append(f"скважина: `{projected['selected_well_value']}`")
    if updates.get("date_from") or updates.get("date_to"):
        captured.append(
            "период: "
            + f"{projected.get('date_from') or '?'} .. {projected.get('date_to') or '?'}"
        )
    if updates.get("anomaly_goal"):
        captured.append(f"отклонение: {projected['anomaly_goal']}")
    if updates.get("recommendation_mode"):
        captured.append(
            "режим: " + ("интервалы" if projected["recommendation_mode"] == "interval" else "точки")
        )
    if updates.get("window_size"):
        captured.append(f"окно: {projected['window_size']} точек")
    if updates.get("statistical_threshold_pct") is not None:
        captured.append(f"порог: {projected['statistical_threshold_pct']}%")
    if task_spec_updates.get("normal_operation_definition"):
        captured.append("понял, что считать нормальной работой")

    if not captured:
        return ""
    return "Принял " + "; ".join(captured) + "."


def get_missing_slots(projected: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if not projected.get("selected_series"):
        missing.append("series")
    if projected.get("multiple_wells") and not projected.get("selected_well_value"):
        missing.append("well")
    if projected.get("rows", 0) > 100_000 and not (projected.get("date_from") and projected.get("date_to")):
        missing.append("date_range")
    if not projected.get("anomaly_goal"):
        missing.append("anomaly")
    if projected.get("recommendation_mode") not in {"point", "interval"}:
        missing.append("mode")
    if not projected.get("window_size"):
        missing.append("window")
    anomaly_goal = str(projected.get("anomaly_goal") or "").lower()
    if (
        ("статист" in anomaly_goal or "%" in anomaly_goal or "процент" in anomaly_goal)
        and projected.get("statistical_threshold_pct") in (None, "")
    ):
        missing.append("statistical_threshold")
    return missing


def build_next_question(projected: dict[str, Any], slot: str) -> str:
    if slot == "series":
        return "Уточните, какие ряды вывести: один или оба."
    if slot == "well":
        return "Какую скважину или лист открыть первой?"
    if slot == "date_range":
        return "Файл большой. Уточните диапазон дат для первого прохода."
    if slot == "anomaly":
        return "Что именно считать отклонением: изменение амплитуды, провал, полка, шум или другой паттерн?"
    if slot == "mode":
        return "Рекомендации нужны точками или интервалами?"
    if slot == "window":
        return "Какой размер окна взять для первого прохода?"
    if slot == "statistical_threshold":
        return "Какой порог статистического сдвига взять, например 30%?"
    return "Нужны ещё уточнения для первого прохода."


def asks_for_filled_slot(reply: str, projected: dict[str, Any]) -> bool:
    lower = reply.lower()
    return (
        ("скваж" in lower and projected.get("selected_well_value"))
        or ("дат" in lower and projected.get("date_from") and projected.get("date_to"))
        or ("интервал" in lower and projected.get("recommendation_mode") == "interval")
        or ("точк" in lower and projected.get("recommendation_mode") == "point")
        or ("отклон" in lower and projected.get("anomaly_goal"))
    )


def _merge_updates(primary: dict[str, Any], secondary: dict[str, Any]) -> dict[str, Any]:
    merged = dict(secondary)
    for key, value in primary.items():
        if value in (None, "", []):
            continue
        merged[key] = value
    return merged


def _generate_reply_with_mistral(state: SessionState, user_message: str) -> dict[str, Any]:
    context = {
        "profile": _serialize_profile(state),
        "current_state": {
            "selected_series": state.selected_series,
            "selected_well_column": state.selected_well_column,
            "selected_well_value": state.selected_well_value,
            "selected_time_column": state.selected_time_column,
            "date_from": state.date_from,
            "date_to": state.date_to,
            "anomaly_goal": state.anomaly_goal,
            "chart_preferences": state.chart_preferences,
            "window_size": state.window_size,
            "statistical_threshold_pct": state.statistical_threshold_pct,
            "recommendation_mode": state.recommendation_mode,
        },
        "task_spec": _serialize_task_spec(state),
        "recent_messages": state.messages[-8:],
        "user_message": user_message,
        "ui_constraints": {
            "detail_panels_limit": 2,
            "review_surface_available": True,
            "window_size_near_plot": True,
            "point_or_interval_supported": True,
        },
    }

    payload = {
        "model": settings.mistral_resolved_model,
        "temperature": 0.1,
        "max_tokens": 900,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": JSON_INSTRUCTION},
            {"role": "user", "content": json.dumps(context, ensure_ascii=False)},
        ],
    }

    response = _call_mistral_api(payload)
    content = _extract_content(response)
    parsed = _parse_json_object(content)
    updates = _sanitize_updates(parsed.get("updates") or {}, state)
    task_spec_updates = _sanitize_task_spec_updates(parsed.get("task_spec_updates") or {})
    reply = str(parsed.get("assistant_reply") or "").strip()
    if not reply:
        reply = _fallback_reply(state, user_message)

    return {
        "reply": reply,
        "updates": updates,
        "task_spec_updates": task_spec_updates,
        "ready_for_first_pass": bool(parsed.get("ready_for_first_pass")),
    }


def _serialize_profile(state: SessionState) -> dict[str, Any]:
    profile = state.profile
    if not profile:
        return {}
    return {
        "rows": profile.rows,
        "timestamp_candidates": profile.timestamp_candidates,
        "well_candidates": profile.well_candidates,
        "numeric_candidates": profile.numeric_candidates,
        "inferred_well_column": profile.inferred_well_column,
        "inferred_time_column": profile.inferred_time_column,
        "detected_multiple_wells": profile.detected_multiple_wells,
        "unique_well_count": profile.unique_well_count,
        "time_min": profile.time_min,
        "time_max": profile.time_max,
        "sheet_names": profile.sheet_names,
        "source_sheet_column": profile.source_sheet_column,
    }


def _serialize_task_spec(state: SessionState) -> dict[str, Any]:
    task_spec = state.task_spec
    if not task_spec:
        return {}
    return {
        "task_id": task_spec.task_id,
        "title": task_spec.title,
        "equipment_family": task_spec.equipment_family,
        "primary_deviation": task_spec.primary_deviation,
        "signal_schema": [
            {
                "name": signal.name,
                "unit": signal.unit,
                "role": signal.role,
                "selected_for_review": signal.selected_for_review,
            }
            for signal in task_spec.signal_schema
        ],
        "label_taxonomy": task_spec.label_taxonomy,
        "unknown_label": task_spec.unknown_label,
        "context_sources": task_spec.context_sources,
        "baseline_strategy": task_spec.baseline_strategy,
        "quality_rules": task_spec.quality_rules,
        "normal_operation_definition": task_spec.normal_operation_definition,
        "confounders": task_spec.confounders,
        "minimum_segment_duration": task_spec.minimum_segment_duration,
        "expected_deviation_frequency": task_spec.expected_deviation_frequency,
        "statistical_threshold_pct": task_spec.statistical_threshold_pct,
        "well_column": task_spec.well_column,
        "time_column": task_spec.time_column,
    }


def _call_mistral_api(payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        url=f"{settings.mistral_api_base.rstrip('/')}/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {settings.mistral_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.mistral_timeout_sec) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Mistral API HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Mistral API network error: {exc.reason}") from exc


def _extract_content(response: dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("Mistral API returned no choices")
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
        if parts:
            return "".join(parts)
    raise RuntimeError("Mistral API returned unsupported message content")


def _parse_json_object(content: str) -> dict[str, Any]:
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if not match:
            raise RuntimeError("Mistral API returned non-JSON content")
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise RuntimeError("Mistral API returned invalid JSON content") from exc


def _sanitize_updates(updates: dict[str, Any], state: SessionState) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}

    selected_series = updates.get("selected_series")
    if isinstance(selected_series, list) and state.profile:
        allowed = set(state.profile.numeric_candidates)
        sanitized["selected_series"] = [
            str(item) for item in selected_series if isinstance(item, str) and item in allowed
        ]

    selected_well_value = updates.get("selected_well_value")
    if isinstance(selected_well_value, str) and selected_well_value.strip():
        sanitized["selected_well_value"] = selected_well_value.strip()

    for field in ("date_from", "date_to", "anomaly_goal", "chart_preferences"):
        value = updates.get(field)
        if isinstance(value, str) and value.strip():
            sanitized[field] = value.strip()

    recommendation_mode = updates.get("recommendation_mode")
    if recommendation_mode in {"point", "interval"}:
        sanitized["recommendation_mode"] = recommendation_mode

    window_size = updates.get("window_size")
    if window_size not in (None, ""):
        try:
            parsed = int(window_size)
        except (TypeError, ValueError):
            parsed = None
        if parsed and parsed > 0:
            sanitized["window_size"] = parsed
    statistical_threshold_pct = updates.get("statistical_threshold_pct")
    if statistical_threshold_pct not in (None, ""):
        try:
            parsed_threshold = float(statistical_threshold_pct)
        except (TypeError, ValueError):
            parsed_threshold = None
        if parsed_threshold is not None and parsed_threshold > 0:
            sanitized["statistical_threshold_pct"] = parsed_threshold

    return sanitized


def _sanitize_task_spec_updates(updates: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}

    for field in ("equipment_family", "primary_deviation", "normal_operation_definition", "expected_deviation_frequency"):
        value = updates.get(field)
        if isinstance(value, str) and value.strip():
            sanitized[field] = value.strip()

    for field in ("confounders", "context_sources"):
        value = updates.get(field)
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                sanitized[field] = cleaned

    minimum_segment_duration = updates.get("minimum_segment_duration")
    if minimum_segment_duration not in (None, ""):
        try:
            parsed = int(minimum_segment_duration)
        except (TypeError, ValueError):
            parsed = None
        if parsed and parsed > 0:
            sanitized["minimum_segment_duration"] = parsed
    statistical_threshold_pct = updates.get("statistical_threshold_pct")
    if statistical_threshold_pct not in (None, ""):
        try:
            parsed_threshold = float(statistical_threshold_pct)
        except (TypeError, ValueError):
            parsed_threshold = None
        if parsed_threshold is not None and parsed_threshold > 0:
            sanitized["statistical_threshold_pct"] = parsed_threshold

    return sanitized


def infer_statistical_threshold_pct_from_message(user_message: str) -> float | None:
    lowered = user_message.lower().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*процент", lowered)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def infer_window_size_from_message(user_message: str, state: SessionState) -> int | None:
    lower = user_message.lower()

    simple_match = re.search(r"(\d+)\s*(час|часа|часов|ч)\b", lower)
    if simple_match:
        hours = int(simple_match.group(1))
        return duration_seconds_to_points(hours * 3600, state)

    minute_match = re.search(r"(\d+)\s*(мин|минута|минуты|минут)\b", lower)
    if minute_match:
        minutes = int(minute_match.group(1))
        return duration_seconds_to_points(minutes * 60, state)

    return None


def duration_seconds_to_points(duration_seconds: int, state: SessionState) -> int | None:
    if duration_seconds <= 0:
        return None
    sample_seconds = state.profile.inferred_window_size if state.profile else None
    if not sample_seconds or sample_seconds <= 0:
        return None
    return max(int(round(duration_seconds / sample_seconds)), 1)


def infer_well_from_message(user_message: str, well_values: list[str]) -> str | None:
    if not well_values:
        return None

    normalized_message = normalize_token(user_message)
    ranked: list[tuple[int, str]] = []
    for value in well_values:
        normalized_value = normalize_token(value)
        if not normalized_value:
            continue
        score = 0
        if normalized_value in normalized_message:
            score = len(normalized_value) + 100
        elif normalized_message in normalized_value and len(normalized_message) >= 4:
            score = len(normalized_message)
        if score:
            ranked.append((score, value))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0], reverse=True)
    if len(ranked) > 1 and ranked[0][0] == ranked[1][0] and ranked[0][1] != ranked[1][1]:
        return None
    return ranked[0][1]


def normalize_token(value: str) -> str:
    return re.sub(r"[^a-zа-я0-9]+", "", value.lower())


def tokenize_text(value: str) -> list[str]:
    return [token for token in re.findall(r"[a-zа-я0-9]+", value.lower()) if token]


def token_stem(value: str) -> str:
    return value[: max(4, min(6, len(value)))]


def candidate_tokens_match(candidate: str, message_tokens: list[str]) -> bool:
    candidate_tokens = [token for token in tokenize_text(candidate) if len(token) >= 4]
    if not candidate_tokens:
        return False

    matched_count = 0
    for candidate_token in candidate_tokens:
        stem = token_stem(candidate_token)
        if any(token.startswith(stem) or stem.startswith(token_stem(token)) for token in message_tokens):
            matched_count += 1

    return matched_count == len(candidate_tokens)


def infer_dates_from_message(user_message: str, state: SessionState) -> dict[str, str]:
    lower = user_message.lower()
    iso_matches = re.findall(r"\d{4}-\d{2}-\d{2}", lower)
    if len(iso_matches) >= 2:
        return {"date_from": iso_matches[0], "date_to": iso_matches[1]}
    if len(iso_matches) == 1:
        d = iso_matches[0]
        # "до конца дня" / "за весь день" → full-day range
        if _is_full_day_intent(lower):
            return {"date_from": d, "date_to": d + "T23:59:59"}
        return {"date_from": d}

    month_map = {
        "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
        "ма": 5, "июн": 6, "июл": 7, "август": 8,
        "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
    }
    matches = re.findall(
        r"(\d{1,2})\s+(январ\w*|феврал\w*|март\w*|апрел\w*|ма[йя]\w*|июн\w*|июл\w*|август\w*|сентябр\w*|октябр\w*|ноябр\w*|декабр\w*)(?:\s+(\d{4}))?",
        lower,
    )
    if not matches:
        return {}

    # Resolve default year using the actual month/day of the first match
    _first_month, _first_day = 1, 1
    for _dstr, _mstr, _ystr in matches[:1]:
        for _key, _val in month_map.items():
            if _mstr.startswith(_key):
                _first_month = _val
                try:
                    _first_day = int(_dstr)
                except ValueError:
                    pass
                break
    default_year = infer_default_year(state, month=_first_month, day=_first_day)

    parsed_dates: list[str] = []
    for day_raw, month_raw, year_raw in matches[:2]:
        month = None
        for key, value in month_map.items():
            if month_raw.startswith(key):
                month = value
                break
        if not month:
            continue
        year = int(year_raw) if year_raw else default_year
        try:
            parsed_dates.append(datetime(year, month, int(day_raw)).date().isoformat())
        except ValueError:
            continue

    if len(parsed_dates) >= 2:
        return {"date_from": parsed_dates[0], "date_to": parsed_dates[1]}
    if len(parsed_dates) == 1:
        d = parsed_dates[0]
        if _is_full_day_intent(lower):
            return {"date_from": d, "date_to": d + "T23:59:59"}
        return {"date_from": d}
    return {}


def _is_full_day_intent(lower: str) -> bool:
    """True when message implies a full calendar day rather than just a start."""
    markers = (
        "до конца дня", "до конца", "за весь день", "весь день",
        "с начала до конца", "от начала до конца",
        "с 00:00", "с нуля часов", "00:00 до 23", "до 23:59", "до 24:00",
    )
    return any(m in lower for m in markers)


def infer_default_year(state: SessionState, month: int = 1, day: int = 1) -> int:
    """Return the most appropriate year for a day+month with no explicit year.

    Priority:
    1. Year already in state.date_from / state.date_to
    2. Most recent year within the dataset range where (month, day) is valid
    3. Year from time_max
    4. Current year
    """
    # Priority 1: already-set context
    for candidate in (state.date_from, state.date_to):
        if candidate:
            try:
                return datetime.fromisoformat(candidate[:10]).year
            except ValueError:
                pass

    # Priority 2: most recent valid year within dataset bounds
    if state.profile:
        time_min_str = state.profile.time_min
        time_max_str = state.profile.time_max
        try:
            ds_start = datetime.fromisoformat(time_min_str).year if time_min_str else None
            ds_end = datetime.fromisoformat(time_max_str).year if time_max_str else None
            if ds_start and ds_end:
                for year in range(ds_end, ds_start - 1, -1):
                    try:
                        datetime(year, month, day)  # validates date exists
                        candidate_date = datetime(year, month, day)
                        ds_min = datetime.fromisoformat(time_min_str[:10])
                        ds_max = datetime.fromisoformat(time_max_str[:10])
                        if ds_min <= candidate_date <= ds_max:
                            return year
                    except ValueError:
                        continue
        except (ValueError, TypeError):
            pass

        # Priority 3: year from time_max
        if time_max_str:
            try:
                return datetime.fromisoformat(time_max_str).year
            except ValueError:
                pass

    return datetime.utcnow().year


def infer_task_spec_updates_from_message(user_message: str) -> dict[str, Any]:
    lower = user_message.lower()
    updates: dict[str, Any] = {}

    if any(marker in lower for marker in ("rod pump", "rod_pump", "станок-качал", "ск")):
        updates["equipment_family"] = "rod_pump_unit"
    if "belt break" in lower or "обрыв" in lower:
        updates["primary_deviation"] = "belt_break"

    normal_match = re.search(r"нормальн\w+\s+работ\w+\s+(?:это|—|-)\s+(.+)", user_message, re.IGNORECASE)
    if normal_match:
        updates["normal_operation_definition"] = normal_match.group(1).strip().rstrip(".")

    confounder_match = re.search(r"(?:конфаундеры|confounders)\s*[:\-]\s*(.+)", user_message, re.IGNORECASE)
    if confounder_match:
        parts = re.split(r",|;| и ", confounder_match.group(1))
        cleaned = [part.strip(" .") for part in parts if part.strip(" .")]
        if cleaned:
            updates["confounders"] = cleaned

    return updates


def infer_anomaly_goal_from_message(user_message: str) -> str | None:
    lower = user_message.lower()
    if "кандидат" in lower and "нет" in lower:
        return None
    if "статист" in lower and ("30%" in lower or "30 %" in lower or "30 процентов" in lower or "30 процент" in lower):
        return "изменение статистического параметра не менее чем на 30%"
    if "статист" in lower and "процент" in lower:
        return user_message.strip()
    if "изменение амплитуд" in lower:
        return "изменение амплитуды"
    if "амплитуд" in lower:
        return "изменение амплитуды"
    if "резк" in lower and "провал" in lower:
        return "резкие провалы"
    if any(marker in lower for marker in ("отклон", "аномал", "падени", "скач", "останов", "шум", "полк", "обрыв")) and "?" not in lower:
        return user_message.strip()
    return None
