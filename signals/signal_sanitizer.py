"""
signals/signal_sanitizer.py — Stage 2: Signal Sanitizer.

Applies quality rules from the TaskSpec to a CanonicalTimeSeries:
  - Drop rows with non-numeric signal values (already coerced to NaN by normalizer)
  - Compute missing_pct and detect dropout spans
  - Optionally clamp signal_min / signal_max
  - Interpolate short gaps (max_interpolation_gap_s)
  - Compute noise level (std / |median|)

Returns the cleaned series plus a QualityFlags summary.
"""
from __future__ import annotations

from dataclasses import replace
from datetime import timedelta, timezone
from typing import Optional

import pandas as pd

from core.canonical_schema import CanonicalTimeSeries, DateRange, QualityFlags
from core.task_manager import TaskSpec


class SanitizationError(ValueError):
    """Raised when the series is unusable after sanitization."""


def sanitize(
    series: CanonicalTimeSeries,
    task_spec: TaskSpec,
) -> tuple[CanonicalTimeSeries, QualityFlags]:
    """Clean *series* according to rules in *task_spec*.

    Returns
    -------
    cleaned_series:
        A new :class:`~core.canonical_schema.CanonicalTimeSeries` with
        sanitized values.
    quality_flags:
        Summary of quality issues detected.
    """
    df = series.values.copy()
    sig = series.signal_col
    t = series.timestamp_col

    # ------------------------------------------------------------------
    # 1. Ensure numeric dtype (coerce leftovers from upstream)
    # ------------------------------------------------------------------
    df[sig] = pd.to_numeric(df[sig], errors="coerce")

    # ------------------------------------------------------------------
    # 2. Clamp to [signal_min, signal_max] if specified
    # ------------------------------------------------------------------
    clamp_events = 0
    if task_spec.signal_min is not None or task_spec.signal_max is not None:
        lo = task_spec.signal_min
        hi = task_spec.signal_max
        before = df[sig].notna().sum()
        if lo is not None:
            df[sig] = df[sig].where(df[sig].isna() | (df[sig] >= lo), other=lo)
        if hi is not None:
            df[sig] = df[sig].where(df[sig].isna() | (df[sig] <= hi), other=hi)
        after = df[sig].notna().sum()
        clamp_events = int(before - after)  # rows turned to NaN by clamping

    # ------------------------------------------------------------------
    # 3. Detect dropout spans BEFORE interpolation
    # ------------------------------------------------------------------
    total_rows = len(df)
    missing_before = int(df[sig].isna().sum())
    missing_pct = missing_before / max(total_rows, 1)

    dropout_spans = _detect_dropout_spans(
        df=df,
        time_col=t,
        signal_col=sig,
        dropout_threshold=task_spec.dropout_threshold,
        min_dropout_duration_s=task_spec.min_dropout_duration_s,
    )

    # ------------------------------------------------------------------
    # 4. Interpolate short gaps
    # ------------------------------------------------------------------
    interpolated_gaps = 0
    if task_spec.max_interpolation_gap_s and task_spec.max_interpolation_gap_s > 0:
        df, interpolated_gaps = _interpolate_short_gaps(
            df=df,
            time_col=t,
            signal_col=sig,
            max_gap_s=task_spec.max_interpolation_gap_s,
        )

    # ------------------------------------------------------------------
    # 5. Noise level  (std / |median|, clipped to [0, 10])
    # ------------------------------------------------------------------
    noise_level: Optional[float] = None
    valid = df[sig].dropna()
    if len(valid) >= 4:
        med = float(valid.median())
        std = float(valid.std(ddof=1))
        if abs(med) > 1e-9:
            noise_level = min(std / abs(med), 10.0)
        else:
            noise_level = None  # can't normalise against zero

    # ------------------------------------------------------------------
    # 6. Guard: if > 90 % missing after sanitization → unusable
    # ------------------------------------------------------------------
    final_missing = int(df[sig].isna().sum())
    if total_rows > 0 and final_missing / total_rows > 0.90:
        raise SanitizationError(
            f"Signal '{sig}' for asset '{series.asset_id}' is >90 % missing "
            f"after sanitization ({final_missing}/{total_rows} rows)."
        )

    flags = QualityFlags(
        missing_pct=missing_pct,
        dropout_spans=dropout_spans,
        noise_level=noise_level,
        clamp_events=clamp_events,
        interpolated_gaps=interpolated_gaps,
    )

    cleaned = replace(series, values=df)
    return cleaned, flags


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _detect_dropout_spans(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    dropout_threshold: Optional[float],
    min_dropout_duration_s: Optional[int],
) -> list[DateRange]:
    """Return contiguous NaN spans that exceed *min_dropout_duration_s*."""
    if time_col not in df.columns:
        return []

    # Build a boolean mask: True where signal is missing (below threshold or NaN)
    if dropout_threshold is not None:
        mask = df[signal_col].isna() | (df[signal_col] < dropout_threshold)
    else:
        mask = df[signal_col].isna()

    if not bool(mask.any()):
        return []

    times = df[time_col]
    spans: list[DateRange] = []
    in_span = False
    span_start = None

    for i, is_missing in enumerate(mask):
        if is_missing and not in_span:
            in_span = True
            span_start = times.iloc[i]
        elif not is_missing and in_span:
            in_span = False
            span_end = times.iloc[i - 1]
            spans.append(_make_date_range(span_start, span_end))
    if in_span and span_start is not None:
        spans.append(_make_date_range(span_start, times.iloc[-1]))

    if min_dropout_duration_s and min_dropout_duration_s > 0:
        min_td = timedelta(seconds=min_dropout_duration_s)
        spans = [s for s in spans if (s.end - s.start) >= min_td]

    return spans


def _interpolate_short_gaps(
    df: pd.DataFrame,
    time_col: str,
    signal_col: str,
    max_gap_s: int,
) -> tuple[pd.DataFrame, int]:
    """Linearly interpolate NaN runs shorter than *max_gap_s* seconds."""
    df = df.copy()
    mask = df[signal_col].isna()
    if not bool(mask.any()):
        return df, 0

    times = df[time_col]
    max_gap_td = timedelta(seconds=max_gap_s)

    # Find contiguous NaN runs
    interpolated_count = 0
    in_run = False
    run_start_i: int = 0

    def _try_interp(start_i: int, end_i: int) -> None:
        nonlocal interpolated_count
        t_start = times.iloc[start_i - 1] if start_i > 0 else None
        t_end = times.iloc[end_i + 1] if end_i < len(df) - 1 else None
        if t_start is None or t_end is None:
            return
        if (t_end - t_start) <= max_gap_td:
            df.iloc[start_i : end_i + 1, df.columns.get_loc(signal_col)] = None  # let pandas fill
            interpolated_count += end_i - start_i + 1

    for i, is_nan in enumerate(mask):
        if is_nan and not in_run:
            in_run = True
            run_start_i = i
        elif not is_nan and in_run:
            in_run = False
            _try_interp(run_start_i, i - 1)
    if in_run:
        _try_interp(run_start_i, len(df) - 1)

    df[signal_col] = df[signal_col].interpolate(method="time", limit_area="inside")
    return df, interpolated_count


def _make_date_range(start: object, end: object) -> DateRange:
    from datetime import datetime
    def _ensure_dt(v: object) -> datetime:
        if isinstance(v, datetime):
            return v
        if hasattr(v, "to_pydatetime"):
            return v.to_pydatetime()  # type: ignore[union-attr]
        return pd.Timestamp(v).to_pydatetime()
    return DateRange(start=_ensure_dt(start), end=_ensure_dt(end))
