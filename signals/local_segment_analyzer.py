"""
signals/local_segment_analyzer.py — Stage 6: Local Segment Analyzer.

For each CandidateEvent, extracts detailed local features from the raw
sanitized time series over the candidate's time window.

Features produced (matching TaskSpec.feature_profile):
  power_mean, power_std, power_p10, power_p90,
  min_power, max_power, zero_fraction,
  transition_sharpness, segment_duration_h,
  preceding_regime_type, following_regime_type
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from core.canonical_schema import (
    CandidateEvent,
    CanonicalTimeSeries,
    LocalFeatures,
)


def analyze(
    candidate: CandidateEvent,
    series: CanonicalTimeSeries,
    *,
    context_window_factor: float = 0.25,
) -> LocalFeatures:
    """Compute local features for *candidate* from the sanitized *series*.

    Parameters
    ----------
    candidate:
        The event whose segment we analyse.
    series:
        Sanitized time series for the same asset/signal.
    context_window_factor:
        Fraction of segment duration to include as left/right context
        when computing transition_sharpness.
    """
    df = series.values
    sig = series.signal_col
    t_col = series.timestamp_col

    times = pd.to_datetime(df[t_col], errors="coerce")
    values = pd.to_numeric(df[sig], errors="coerce")

    seg_start = pd.Timestamp(candidate.segment.start)
    seg_end = pd.Timestamp(candidate.segment.end)

    # ------------------------------------------------------------------
    # 1. Slice to the candidate window
    # ------------------------------------------------------------------
    mask = (times >= seg_start) & (times <= seg_end)
    seg_vals = values[mask].dropna()

    if len(seg_vals) == 0:
        return LocalFeatures(candidate_id=candidate.candidate_id)

    # ------------------------------------------------------------------
    # 2. Core statistics
    # ------------------------------------------------------------------
    arr = seg_vals.to_numpy(dtype=float)
    power_mean = float(np.mean(arr))
    power_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    power_p10 = float(np.percentile(arr, 10))
    power_p90 = float(np.percentile(arr, 90))
    min_power = float(np.min(arr))
    max_power = float(np.max(arr))
    zero_fraction = float(np.mean(np.abs(arr) < 1e-6))
    duration_h = max(
        (seg_end - seg_start).total_seconds() / 3600.0, 0.0
    )

    # ------------------------------------------------------------------
    # 3. Transition sharpness
    #    = |mean_inside – mean_outside| using a context window on each side
    # ------------------------------------------------------------------
    transition_sharpness = _compute_transition_sharpness(
        values=values,
        times=times,
        seg_start=seg_start,
        seg_end=seg_end,
        factor=context_window_factor,
    )

    return LocalFeatures(
        candidate_id=candidate.candidate_id,
        power_mean=power_mean,
        power_std=power_std,
        power_p10=power_p10,
        power_p90=power_p90,
        min_power=min_power,
        max_power=max_power,
        zero_fraction=zero_fraction,
        transition_sharpness=transition_sharpness,
        segment_duration_h=duration_h,
        preceding_regime_type=candidate.preceding_regime_type,
        following_regime_type=candidate.following_regime_type,
    )


def analyze_batch(
    candidates: list[CandidateEvent],
    series: CanonicalTimeSeries,
) -> list[LocalFeatures]:
    """Analyse multiple candidates over the same series."""
    return [analyze(c, series) for c in candidates]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_transition_sharpness(
    values: pd.Series,
    times: pd.Series,
    seg_start: pd.Timestamp,
    seg_end: pd.Timestamp,
    factor: float,
) -> Optional[float]:
    duration_s = (seg_end - seg_start).total_seconds()
    if duration_s <= 0:
        return None
    context_s = duration_s * factor

    ctx_start = seg_start - pd.Timedelta(seconds=context_s)
    ctx_end = seg_end + pd.Timedelta(seconds=context_s)

    before_mask = (times >= ctx_start) & (times < seg_start)
    after_mask = (times > seg_end) & (times <= ctx_end)
    inside_mask = (times >= seg_start) & (times <= seg_end)

    before = values[before_mask].dropna()
    after = values[after_mask].dropna()
    inside = values[inside_mask].dropna()

    if len(inside) == 0:
        return None

    mean_inside = float(inside.mean())

    if len(before) > 0 and len(after) > 0:
        mean_context = float(pd.concat([before, after]).mean())
    elif len(before) > 0:
        mean_context = float(before.mean())
    elif len(after) > 0:
        mean_context = float(after.mean())
    else:
        return None

    return round(abs(mean_inside - mean_context), 6)
