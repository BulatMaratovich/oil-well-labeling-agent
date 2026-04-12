from __future__ import annotations

from typing import Any, Optional
import math

import pandas as pd


def safe_number(value: Any, digits: int = 6) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return round(parsed, digits)


def remove_three_sigma_outliers(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return clean, {"count_raw": 0, "count_clean": 0, "outliers_removed": 0}

    mean = clean.mean()
    std = clean.std(ddof=1)
    if pd.isna(std) or std == 0:
        return clean, {
            "count_raw": int(len(clean)),
            "count_clean": int(len(clean)),
            "outliers_removed": 0,
        }

    filtered = clean[(clean >= mean - 3 * std) & (clean <= mean + 3 * std)]
    if filtered.empty:
        filtered = clean
    return filtered, {
        "count_raw": int(len(clean)),
        "count_clean": int(len(filtered)),
        "outliers_removed": int(len(clean) - len(filtered)),
    }


def describe_series(series: pd.Series) -> dict[str, Any]:
    filtered, meta = remove_three_sigma_outliers(series)
    if filtered.empty:
        return {
            **meta,
            "mean": None,
            "median": None,
            "std": None,
            "variance": None,
            "min": None,
            "max": None,
            "range": None,
        }

    variance = filtered.var(ddof=1) if len(filtered) > 1 else 0.0
    std = filtered.std(ddof=1) if len(filtered) > 1 else 0.0
    return {
        **meta,
        "mean": safe_number(filtered.mean()),
        "median": safe_number(filtered.median()),
        "std": safe_number(std),
        "variance": safe_number(variance),
        "min": safe_number(filtered.min()),
        "max": safe_number(filtered.max()),
        "range": safe_number(filtered.max() - filtered.min()),
    }


def compare_stats(reference: dict[str, Any], focus: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key in ("mean", "median", "std", "variance", "range"):
        ref = reference.get(key)
        cur = focus.get(key)
        if ref is None or cur is None:
            result[f"delta_{key}"] = None
            result[f"relative_delta_{key}"] = None
            continue
        delta = cur - ref
        result[f"delta_{key}"] = safe_number(delta)
        result[f"relative_delta_{key}"] = safe_number(delta / ref) if ref not in (0, 0.0) else None
    return result


def run_ruptures(series: pd.Series) -> dict[str, Any]:
    try:
        import ruptures as rpt
    except Exception:
        return {"available": False, "change_points": [], "error": "ruptures_not_installed"}

    signal = pd.to_numeric(series, errors="coerce").dropna().to_numpy()
    if len(signal) < 8:
        return {"available": True, "change_points": [], "error": None}

    algo = rpt.Pelt(model="rbf").fit(signal)
    penalty = max(float(signal.std()) * 3, 1.0)
    breakpoints = algo.predict(pen=penalty)
    return {
        "available": True,
        "change_points": [int(item) for item in breakpoints[:-1]],
        "error": None,
    }


def build_assessment(
    reference_stats: dict[str, Any],
    focus_stats: dict[str, Any],
    comparison: dict[str, Any],
    ruptures: dict[str, Any],
) -> dict[str, Any]:
    rel_range = abs(comparison.get("relative_delta_range") or 0.0)
    rel_std = abs(comparison.get("relative_delta_std") or 0.0)
    rel_mean = abs(comparison.get("relative_delta_mean") or 0.0)
    rel_median = abs(comparison.get("relative_delta_median") or 0.0)

    spread_shift = rel_range >= 0.2 or rel_std >= 0.2
    central_shift = rel_mean >= 0.15 or rel_median >= 0.15
    materially_different = spread_shift or central_shift

    reasons: list[str] = []
    if spread_shift:
        if rel_range >= rel_std:
            reasons.append(f"размах изменился на {round(rel_range * 100, 1)}%")
        else:
            reasons.append(f"стандартное отклонение изменилось на {round(rel_std * 100, 1)}%")
    if central_shift:
        if rel_mean >= rel_median:
            reasons.append(f"среднее изменилось на {round(rel_mean * 100, 1)}%")
        else:
            reasons.append(f"медиана изменилась на {round(rel_median * 100, 1)}%")

    if not reasons:
        reasons.append("сильного статистического сдвига относительно суточного baseline не видно")

    preferred_method = "statistics"
    if ruptures.get("available") and ruptures.get("change_points"):
        preferred_method = "statistics_plus_ruptures"
        reasons.append(f"ruptures нашёл {len(ruptures['change_points'])} change-point candidates")
    elif not ruptures.get("available"):
        reasons.append("ruptures недоступен, используется только статистическая оценка")

    return {
        "material_change": materially_different,
        "spread_shift": spread_shift,
        "central_shift": central_shift,
        "preferred_method": preferred_method,
        "summary": "; ".join(reasons),
        "outliers_removed_reference": reference_stats.get("outliers_removed"),
        "outliers_removed_focus": focus_stats.get("outliers_removed"),
    }


def analyze_interval_against_day(
    frame: pd.DataFrame,
    time_column: str,
    value_column: str,
    window_start: str,
    window_end: Optional[str],
    with_ruptures: bool = True,
) -> dict[str, Any]:
    if frame.empty or time_column not in frame.columns or value_column not in frame.columns:
        return {
            "reference_stats": {},
            "focus_stats": {},
            "comparison": {},
            "ruptures": {"available": False, "change_points": [], "error": "invalid_input"},
            "assessment": {"summary": "Недостаточно данных для статистической оценки."},
        }

    working = frame[[time_column, value_column]].copy()
    working[time_column] = pd.to_datetime(working[time_column], errors="coerce", utc=True)
    working[value_column] = pd.to_numeric(working[value_column], errors="coerce")
    working = working.dropna(subset=[time_column, value_column]).sort_values(time_column)
    if working.empty:
        return {
            "reference_stats": {},
            "focus_stats": {},
            "comparison": {},
            "ruptures": {"available": False, "change_points": [], "error": "empty_series"},
            "assessment": {"summary": "После очистки не осталось валидных точек."},
        }

    start = pd.to_datetime(window_start, utc=True, errors="coerce")
    end = pd.to_datetime(window_end, utc=True, errors="coerce") if window_end else start
    if pd.isna(start):
        return {
            "reference_stats": {},
            "focus_stats": {},
            "comparison": {},
            "ruptures": {"available": False, "change_points": [], "error": "invalid_window"},
            "assessment": {"summary": "Не удалось распарсить окно candidate interval."},
        }
    if pd.isna(end) or end < start:
        end = start

    reference_day_start = start.normalize()
    reference_day_end = reference_day_start + pd.Timedelta(days=1)
    reference_frame = working[
        (working[time_column] >= reference_day_start) & (working[time_column] < reference_day_end)
    ]
    focus_frame = working[
        (working[time_column] >= start) & (working[time_column] <= end)
    ]

    reference_stats = describe_series(reference_frame[value_column]) if not reference_frame.empty else {}
    focus_stats = describe_series(focus_frame[value_column]) if not focus_frame.empty else {}
    comparison = compare_stats(reference_stats, focus_stats) if reference_stats and focus_stats else {}
    ruptures = run_ruptures(reference_frame[value_column]) if with_ruptures and not reference_frame.empty else {
        "available": False,
        "change_points": [],
        "error": "reference_window_empty" if reference_frame.empty else "disabled",
    }
    assessment = build_assessment(reference_stats, focus_stats, comparison, ruptures)

    return {
        "value_column": value_column,
        "reference_day": reference_day_start.date().isoformat(),
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "focus_duration_hours": safe_number((end - start).total_seconds() / 3600, digits=4),
        "reference_stats": reference_stats,
        "focus_stats": focus_stats,
        "comparison": comparison,
        "ruptures": ruptures,
        "assessment": assessment,
    }


def analyze_candidate_intervals(
    frame: pd.DataFrame,
    time_column: Optional[str],
    series_name: Optional[str],
    candidates: list[dict[str, Any]],
    with_ruptures: bool = True,
) -> list[dict[str, Any]]:
    if frame.empty or not time_column or not series_name or series_name not in frame.columns:
        return []

    analyzed: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates):
        start = candidate.get("start")
        end = candidate.get("end")
        if not start:
            continue
        stats = analyze_interval_against_day(
            frame=frame,
            time_column=time_column,
            value_column=series_name,
            window_start=str(start),
            window_end=str(end) if end else None,
            with_ruptures=with_ruptures,
        )
        analyzed.append(
            {
                "candidate_index": index,
                "start": start,
                "end": end,
                "series_name": series_name,
                "score": candidate.get("score"),
                "reason": candidate.get("reason"),
                "stats": stats,
            }
        )
    return analyzed
