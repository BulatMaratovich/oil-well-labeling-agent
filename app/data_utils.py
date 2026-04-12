from __future__ import annotations

from io import BytesIO
from typing import Optional
import math
import re

import pandas as pd

from app.models import ColumnProfile, DataProfile


WELL_HINTS = ("well", "скв", "m_geo", "object", "asset")
TIME_HINTS = ("time", "date", "timestamp", "valuedate", "дата", "время")


def _safe_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def load_tabular_file(filename: str, payload: bytes) -> pd.DataFrame:
    lower = filename.lower()
    if lower.endswith(".csv"):
        return pd.read_csv(BytesIO(payload))
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        workbook = pd.read_excel(BytesIO(payload), sheet_name=None)
        frames: list[pd.DataFrame] = []
        for sheet_name, frame in workbook.items():
            prepared = frame.copy()
            prepared["__sheet_name"] = sheet_name
            frames.append(prepared)
        if not frames:
            raise ValueError("Excel-файл не содержит читаемых листов.")
        return pd.concat(frames, ignore_index=True, sort=False)
    raise ValueError("Поддерживаются только CSV и Excel файлы.")


def _looks_like_datetime(series: pd.Series) -> bool:
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    return bool(parsed.notna().mean() >= 0.6)


def _datetime_dayfirst_hint(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(25)
    if sample.empty:
        return False
    pattern = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{4}(?:\s+\d{1,2}:\d{2}(?::\d{2})?)?$")
    matches = sum(1 for item in sample if pattern.match(item.strip()))
    return matches / len(sample) >= 0.6


def parse_datetime_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce", utc=True)
    dayfirst = _datetime_dayfirst_hint(series)
    return pd.to_datetime(series, errors="coerce", utc=True, dayfirst=dayfirst)


def _sample_values(series: pd.Series) -> list[str]:
    return [str(v) for v in series.dropna().head(3).tolist()]


def profile_dataframe(df: pd.DataFrame) -> DataProfile:
    rows = len(df)
    columns: list[ColumnProfile] = []
    timestamp_candidates: list[str] = []
    well_candidates: list[str] = []
    numeric_candidates: list[str] = []

    for column in df.columns:
        series = df[column]
        name = str(column)
        if name.startswith("__"):
            continue
        numeric = bool(pd.api.types.is_numeric_dtype(series))
        datetime_like = _looks_like_datetime(series)
        nullable_pct = round(float(series.isna().mean() * 100), 2) if rows else 0.0
        unique_ratio = round(float(series.nunique(dropna=True) / max(rows, 1)), 4)
        profile = ColumnProfile(
            name=name,
            dtype=str(series.dtype),
            nullable_pct=nullable_pct,
            sample_values=_sample_values(series),
            numeric=numeric,
            datetime_like=datetime_like,
            unique_ratio=unique_ratio,
        )
        columns.append(profile)

        lowered = name.lower()
        if datetime_like or any(token in lowered for token in TIME_HINTS):
            timestamp_candidates.append(name)
        is_identifier_like = any(token in lowered for token in WELL_HINTS)
        if is_identifier_like or (
            not numeric and 0.0001 < unique_ratio < 0.2
        ):
            well_candidates.append(name)
        if numeric and not is_identifier_like:
            numeric_candidates.append(name)

    sheet_names: list[str] = []
    source_sheet_column: Optional[str] = "__sheet_name" if "__sheet_name" in df.columns else None
    if source_sheet_column:
        sheet_names = [str(v) for v in df[source_sheet_column].dropna().astype(str).unique().tolist()]
        if len(sheet_names) > 1:
            well_candidates = [source_sheet_column] + [item for item in well_candidates if item != source_sheet_column]

    inferred_well_column = well_candidates[0] if well_candidates else None
    inferred_time_column = timestamp_candidates[0] if timestamp_candidates else None
    detected_multiple_wells = False
    unique_well_count = 0
    if inferred_well_column and rows:
        unique_well_count = int(df[inferred_well_column].nunique(dropna=True))
        detected_multiple_wells = bool(unique_well_count > 1)

    inferred_window_size: Optional[int] = None
    time_min: Optional[str] = None
    time_max: Optional[str] = None
    if inferred_time_column:
        ts = parse_datetime_series(df[inferred_time_column]).dropna().sort_values()
        if inferred_well_column and detected_multiple_wells and inferred_well_column in df.columns:
            per_well_steps: list[float] = []
            for _, group in df[[inferred_well_column, inferred_time_column]].dropna().groupby(inferred_well_column):
                group_ts = parse_datetime_series(group[inferred_time_column]).dropna().sort_values()
                if len(group_ts) < 3:
                    continue
                delta_seconds = group_ts.diff().dt.total_seconds().dropna()
                positive_steps = delta_seconds[delta_seconds > 0]
                if not positive_steps.empty:
                    per_well_steps.append(float(positive_steps.median()))
            if per_well_steps:
                inferred_window_size = int(pd.Series(per_well_steps, dtype="float64").median())
        if inferred_window_size is None and len(ts) > 2:
            delta_seconds = ts.diff().dt.total_seconds().dropna()
            positive_steps = delta_seconds[delta_seconds > 0]
            if not positive_steps.empty:
                inferred_window_size = int(positive_steps.median())
        if not ts.empty:
            time_min = ts.iloc[0].isoformat()
            time_max = ts.iloc[-1].isoformat()

    return DataProfile(
        rows=rows,
        columns=columns,
        timestamp_candidates=list(dict.fromkeys(timestamp_candidates)),
        well_candidates=list(dict.fromkeys(well_candidates)),
        numeric_candidates=numeric_candidates,
        inferred_well_column=inferred_well_column,
        inferred_time_column=inferred_time_column,
        detected_multiple_wells=detected_multiple_wells,
        inferred_window_size=inferred_window_size,
        unique_well_count=unique_well_count,
        time_min=time_min,
        time_max=time_max,
        sheet_names=sheet_names,
        source_sheet_column=source_sheet_column,
    )


def filter_dataframe(
    df: pd.DataFrame,
    time_column: Optional[str],
    well_column: Optional[str],
    well_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
) -> pd.DataFrame:
    frame = df.copy()
    if well_column and well_value not in (None, "") and well_column in frame.columns:
        frame = frame[frame[well_column].astype(str) == str(well_value)]

    if time_column and time_column in frame.columns:
        frame[time_column] = parse_datetime_series(frame[time_column])
        frame = frame.dropna(subset=[time_column]).sort_values(time_column)
        if date_from:
            parsed_from = pd.to_datetime(date_from, errors="coerce", utc=True)
            if not pd.isna(parsed_from):
                frame = frame[frame[time_column] >= parsed_from]
        if date_to:
            parsed_to = pd.to_datetime(date_to, errors="coerce", utc=True)
            if not pd.isna(parsed_to):
                frame = frame[frame[time_column] <= parsed_to]
    return frame


def normalize_for_plot(
    df: pd.DataFrame,
    time_column: Optional[str],
    well_column: Optional[str],
    well_value: Optional[str],
    date_from: Optional[str],
    date_to: Optional[str],
    series_names: list[str],
) -> dict[str, object]:
    frame = filter_dataframe(
        df=df,
        time_column=time_column,
        well_column=well_column,
        well_value=well_value,
        date_from=date_from,
        date_to=date_to,
    )
    if time_column and time_column in frame.columns:
        x = [
            value.isoformat() if hasattr(value, "isoformat") else str(value)
            for value in frame[time_column].tolist()
        ]
    else:
        x = [str(i) for i in range(len(frame))]

    traces = []
    for name in series_names:
        if name not in frame.columns:
            continue
        values = pd.to_numeric(frame[name], errors="coerce")
        traces.append(
            {
                "name": name,
                "x": x,
                "y": [None if pd.isna(v) else float(v) for v in values.tolist()],
            }
        )

    well_values = []
    if well_column and well_column in df.columns:
        well_values = [str(v) for v in df[well_column].dropna().astype(str).unique().tolist()[:500]]

    return {
        "traces": traces,
        "well_values": well_values,
        "row_count": len(frame),
        "date_from": x[0] if x else None,
        "date_to": x[-1] if x else None,
    }


def get_scope_time_range(
    df: pd.DataFrame,
    time_column: Optional[str],
    well_column: Optional[str],
    well_value: Optional[str],
) -> dict[str, Optional[str]]:
    if not time_column or time_column not in df.columns:
        return {"time_min": None, "time_max": None}
    frame = df.copy()
    if well_column and well_value not in (None, "") and well_column in frame.columns:
        frame = frame[frame[well_column].astype(str) == str(well_value)]
    if frame.empty:
        return {"time_min": None, "time_max": None}
    ts = parse_datetime_series(frame[time_column]).dropna().sort_values()
    if ts.empty:
        return {"time_min": None, "time_max": None}
    return {
        "time_min": ts.iloc[0].isoformat(),
        "time_max": ts.iloc[-1].isoformat(),
    }


def detect_candidate_intervals(
    plot_payload: dict[str, object],
    anomaly_goal: Optional[str],
    window_size: Optional[int],
    statistical_threshold_pct: Optional[float] = None,
    max_candidates: int = 6,
) -> list[dict[str, object]]:
    traces = plot_payload.get("traces") or []
    if not traces:
        return []

    primary_trace = traces[0]
    x_values = list(primary_trace.get("x") or [])
    y_values = list(primary_trace.get("y") or [])
    if len(x_values) < 12 or len(y_values) < 12:
        return []

    series = pd.Series(y_values, dtype="float64").dropna()
    if len(series) < 12:
        return []

    effective_window = max(int(window_size or 12), 6)
    anomaly_lower = (anomaly_goal or "").lower()
    if anomaly_lower and any(token in anomaly_lower for token in ("статист", "%", "процент")):
        candidates = detect_statistical_shift_intervals(
            x_values=x_values,
            y_values=y_values,
            window_size=effective_window,
            anomaly_goal=anomaly_goal,
            statistical_threshold_pct=statistical_threshold_pct,
            series_name=primary_trace.get("name"),
            max_candidates=max_candidates,
        )
        if candidates:
            return candidates
        if "амплитуд" not in anomaly_lower:
            return []
    if anomaly_lower and "амплитуд" not in anomaly_lower:
        return []

    amplitude_window = min(max(effective_window, 6), max(len(y_values) // 4, 6))
    baseline_window = min(max(amplitude_window * 4, amplitude_window + 4), max(len(y_values), amplitude_window * 4))

    full_series = pd.Series(y_values, dtype="float64")
    rolling_max = full_series.rolling(window=amplitude_window, min_periods=max(amplitude_window // 2, 3)).max()
    rolling_min = full_series.rolling(window=amplitude_window, min_periods=max(amplitude_window // 2, 3)).min()
    rolling_amplitude = rolling_max - rolling_min
    smoothed_amplitude = rolling_amplitude.rolling(
        window=max(amplitude_window // 2, 3),
        min_periods=max(amplitude_window // 3, 3),
    ).median()
    baseline = rolling_amplitude.rolling(
        window=min(baseline_window, len(full_series)),
        min_periods=max(amplitude_window, 6),
    ).median()
    deviation = (rolling_amplitude - baseline).abs()
    derivative = smoothed_amplitude.diff().abs()
    threshold = derivative.median(skipna=True) + derivative.std(skipna=True) * 1.0
    if pd.isna(threshold) or threshold <= 0:
        threshold = deviation.median(skipna=True) + deviation.std(skipna=True) * 1.0
    if pd.isna(threshold) or threshold <= 0:
        return []

    candidate_mask = derivative > threshold
    if not bool(candidate_mask.fillna(False).any()):
        fallback_threshold = deviation.median(skipna=True) + deviation.std(skipna=True) * 0.8
        candidate_mask = deviation > fallback_threshold
    active_positions = [idx for idx, value in candidate_mask.fillna(False).items() if bool(value)]
    if not active_positions:
        return []

    intervals: list[tuple[int, int]] = []
    start = active_positions[0]
    prev = active_positions[0]
    bridge = max(amplitude_window // 3, 2)
    for idx in active_positions[1:]:
        if idx - prev <= bridge:
            prev = idx
            continue
        intervals.append((start, prev))
        start = idx
        prev = idx
    intervals.append((start, prev))

    candidates: list[dict[str, object]] = []
    for start_idx, end_idx in intervals:
        safe_start = max(start_idx - amplitude_window // 2, 0)
        safe_end = min(end_idx + amplitude_window // 2, len(x_values) - 1)
        interval_deviation = deviation.iloc[safe_start:safe_end + 1]
        score = _safe_float(interval_deviation.max(skipna=True))
        if score is None or score <= 0:
            continue
        amplitude_before_slice = rolling_amplitude.iloc[max(safe_start - amplitude_window, 0):safe_start]
        amplitude_inside_slice = rolling_amplitude.iloc[safe_start:safe_end + 1]
        amplitude_before = _safe_float(amplitude_before_slice.median(skipna=True)) if not amplitude_before_slice.empty else 0.0
        amplitude_inside = _safe_float(amplitude_inside_slice.median(skipna=True)) if not amplitude_inside_slice.empty else 0.0
        candidates.append(
            {
                "start": str(x_values[safe_start]),
                "end": str(x_values[safe_end]),
                "score": round(score, 4),
                "series_name": primary_trace.get("name"),
                "reason": "change_in_local_amplitude",
                "amplitude_before": round(amplitude_before or 0.0, 4),
                "amplitude_inside": round(amplitude_inside or 0.0, 4),
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:max_candidates]


def parse_relative_threshold(anomaly_goal: Optional[str], default: float = 0.3) -> float:
    if not anomaly_goal:
        return default
    lowered = anomaly_goal.lower().replace(",", ".")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", lowered)
    if not match:
        match = re.search(r"(\d+(?:\.\d+)?)\s*процент", lowered)
    if not match:
        return default
    try:
        value = float(match.group(1)) / 100.0
    except ValueError:
        return default
    return max(value, 0.05)


def detect_statistical_shift_intervals(
    x_values: list[object],
    y_values: list[object],
    window_size: int,
    anomaly_goal: Optional[str],
    statistical_threshold_pct: Optional[float],
    series_name: object,
    max_candidates: int = 6,
) -> list[dict[str, object]]:
    threshold = (
        max(float(statistical_threshold_pct) / 100.0, 0.05)
        if statistical_threshold_pct not in (None, "")
        else parse_relative_threshold(anomaly_goal, default=0.3)
    )
    full_series = pd.Series(y_values, dtype="float64")
    valid_count = int(full_series.notna().sum())
    if valid_count < max(window_size * 3, 24):
        return []

    stat_window = min(max(window_size, 8), max(len(full_series) // 3, 8))
    baseline_window = min(max(stat_window * 4, stat_window + 8), len(full_series))
    min_periods = max(stat_window // 2, 4)

    rolling_mean = full_series.rolling(window=stat_window, min_periods=min_periods).mean()
    rolling_median = full_series.rolling(window=stat_window, min_periods=min_periods).median()
    rolling_std = full_series.rolling(window=stat_window, min_periods=min_periods).std()
    rolling_var = full_series.rolling(window=stat_window, min_periods=min_periods).var()
    rolling_range = (
        full_series.rolling(window=stat_window, min_periods=min_periods).max()
        - full_series.rolling(window=stat_window, min_periods=min_periods).min()
    )

    baseline_mean = rolling_mean.rolling(window=baseline_window, min_periods=max(stat_window, 6)).median()
    baseline_median = rolling_median.rolling(window=baseline_window, min_periods=max(stat_window, 6)).median()
    baseline_std = rolling_std.rolling(window=baseline_window, min_periods=max(stat_window, 6)).median()
    baseline_var = rolling_var.rolling(window=baseline_window, min_periods=max(stat_window, 6)).median()
    baseline_range = rolling_range.rolling(window=baseline_window, min_periods=max(stat_window, 6)).median()

    def relative_change(current: pd.Series, baseline: pd.Series) -> pd.Series:
        denominator = baseline.abs().where(lambda values: values > 1e-9)
        return ((current - baseline).abs() / denominator).replace([float("inf"), -float("inf")], pd.NA)

    rel_mean = relative_change(rolling_mean, baseline_mean)
    rel_median = relative_change(rolling_median, baseline_median)
    rel_std = relative_change(rolling_std, baseline_std)
    rel_var = relative_change(rolling_var, baseline_var)
    rel_range = relative_change(rolling_range, baseline_range)

    score_frame = pd.DataFrame(
        {
            "mean": rel_mean,
            "median": rel_median,
            "std": rel_std,
            "variance": rel_var,
            "range": rel_range,
        }
    )
    max_relative = score_frame.max(axis=1, skipna=True)
    if max_relative.dropna().empty:
        return []

    candidate_mask = max_relative >= threshold
    active_positions = [idx for idx, value in candidate_mask.fillna(False).items() if bool(value)]
    if not active_positions:
        return []

    intervals: list[tuple[int, int]] = []
    start = active_positions[0]
    prev = active_positions[0]
    bridge = max(stat_window // 2, 3)
    for idx in active_positions[1:]:
        if idx - prev <= bridge:
            prev = idx
            continue
        intervals.append((start, prev))
        start = idx
        prev = idx
    intervals.append((start, prev))

    candidates: list[dict[str, object]] = []
    for start_idx, end_idx in intervals:
        safe_start = max(start_idx - stat_window // 2, 0)
        safe_end = min(end_idx + stat_window // 2, len(x_values) - 1)
        score_slice = score_frame.iloc[safe_start:safe_end + 1]
        interval_max = score_slice.max(axis=0, skipna=True)
        score = _safe_float(score_slice.max(axis=1, skipna=True).max(skipna=True))
        if score is None or score < threshold:
            continue
        dominant_metric = interval_max.idxmax(skipna=True) if not interval_max.dropna().empty else "statistic"
        dominant_relative = _safe_float(interval_max.get(dominant_metric))
        candidates.append(
            {
                "start": str(x_values[safe_start]),
                "end": str(x_values[safe_end]),
                "score": round(score, 4),
                "series_name": series_name,
                "reason": "change_in_statistical_parameter",
                "dominant_metric": dominant_metric,
                "relative_change": round(dominant_relative or 0.0, 4),
                "threshold": round(threshold, 4),
            }
        )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    return candidates[:max_candidates]
