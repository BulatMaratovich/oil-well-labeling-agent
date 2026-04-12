"""
signals/global_series_profiler.py — Stage 3: Global Series Profiler.

Runs change-point detection (PELT via ruptures) on the sanitized signal,
then clusters the resulting segments into regime types (KMeans / fallback).

Returns a RegimeSequence — the global regime map for one asset signal.
"""
from __future__ import annotations

import hashlib
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd

from core.canonical_schema import (
    CanonicalTimeSeries,
    Regime,
    RegimeSequence,
)
from core.task_manager import TaskSpec


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def profile(
    series: CanonicalTimeSeries,
    task_spec: TaskSpec,
    *,
    n_clusters: int = 4,
    min_segment_rows: int = 6,
) -> RegimeSequence:
    """Detect regimes in *series* and cluster them.

    Parameters
    ----------
    series:
        Sanitized time series for a single asset / signal.
    task_spec:
        Used for ``minimum_segment_duration`` (in seconds).
    n_clusters:
        Maximum number of regime types.  Reduced automatically for short series.
    min_segment_rows:
        Minimum rows per PELT segment; shorter segments are merged into
        the preceding one.

    Returns
    -------
    RegimeSequence
        Contains the list of :class:`~core.canonical_schema.Regime` objects.
        ``no_regime_structure`` is ``True`` when only a single regime was found.
    """
    df = series.values
    sig = series.signal_col
    t_col = series.timestamp_col

    values = pd.to_numeric(df[sig], errors="coerce").values
    times = pd.to_datetime(df[t_col], errors="coerce")

    valid_mask = ~np.isnan(values)
    if valid_mask.sum() < 12:
        return RegimeSequence(
            asset_id=series.asset_id,
            regimes=[],
            no_regime_structure=True,
        )

    # ------------------------------------------------------------------
    # 1. Change-point detection with PELT
    # ------------------------------------------------------------------
    breakpoints = _detect_breakpoints(
        values=values,
        min_segment_rows=min_segment_rows,
        min_segment_s=task_spec.minimum_segment_duration,
        times=times,
    )

    # ------------------------------------------------------------------
    # 2. Build segment summaries
    # ------------------------------------------------------------------
    segments = _build_segments(
        values=values,
        times=times,
        breakpoints=breakpoints,
        min_segment_rows=min_segment_rows,
    )

    if not segments:
        return RegimeSequence(
            asset_id=series.asset_id,
            regimes=[],
            no_regime_structure=True,
        )

    # ------------------------------------------------------------------
    # 3. Cluster segments into regime types
    # ------------------------------------------------------------------
    regime_labels = _cluster_segments(segments, n_clusters=n_clusters)

    # ------------------------------------------------------------------
    # 4. Assemble Regime objects
    # ------------------------------------------------------------------
    regimes: list[Regime] = []
    for i, seg in enumerate(segments):
        mean_p = seg.get("mean")
        std_p = seg.get("std")
        regime_type = f"type_{regime_labels[i]}"
        regime_id = _stable_regime_id(series.asset_id, sig, seg["start_ts"], regime_type)
        regimes.append(
            Regime(
                start=seg["start_ts"].to_pydatetime(),
                end=seg["end_ts"].to_pydatetime(),
                regime_id=regime_id,
                regime_type=regime_type,
                duration_h=seg["duration_h"],
                mean_power=float(mean_p) if mean_p is not None else None,
                std_power=float(std_p) if std_p is not None else None,
            )
        )

    no_structure = len({r.regime_type for r in regimes}) == 1

    return RegimeSequence(
        asset_id=series.asset_id,
        regimes=regimes,
        no_regime_structure=no_structure,
        signal_name=sig,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_breakpoints(
    values: np.ndarray,
    min_segment_rows: int,
    min_segment_s: Optional[int],
    times: pd.Series,
) -> list[int]:
    """Return 0-based row indices of segment boundaries (excluding 0 and len)."""
    try:
        import ruptures as rpt  # type: ignore[import]

        clean = np.where(np.isnan(values), 0.0, values).reshape(-1, 1)
        n = len(clean)
        # Heuristic: allow up to n // min_segment_rows change-points, capped at 20
        max_bkps = min(max(n // max(min_segment_rows, 6), 1), 20)
        algo = rpt.Pelt(model="rbf", min_size=min_segment_rows, jump=1).fit(clean)
        result = algo.predict(pen=10.0)
        # ruptures returns [bkp1, bkp2, ..., n]  (1-based, last = len)
        return [b - 1 for b in result[:-1]]  # convert to 0-based end-of-segment

    except Exception:
        # Fallback: equal-width splits
        n = len(values)
        step = max(n // 5, min_segment_rows)
        return list(range(step, n, step))


def _build_segments(
    values: np.ndarray,
    times: pd.Series,
    breakpoints: list[int],
    min_segment_rows: int,
) -> list[dict]:
    """Split the series at *breakpoints* and compute per-segment statistics."""
    n = len(values)
    boundaries = [0] + [b + 1 for b in breakpoints] + [n]
    boundaries = sorted(set(boundaries))

    segments = []
    for i in range(len(boundaries) - 1):
        start_i = boundaries[i]
        end_i = boundaries[i + 1]
        if end_i - start_i < 2:
            continue

        seg_vals = values[start_i:end_i]
        seg_times = times.iloc[start_i:end_i]
        valid = seg_vals[~np.isnan(seg_vals)]

        start_ts = seg_times.iloc[0]
        end_ts = seg_times.iloc[-1]
        duration_h = max((end_ts - start_ts).total_seconds() / 3600.0, 0.0)

        segments.append({
            "start_i": start_i,
            "end_i": end_i,
            "start_ts": start_ts,
            "end_ts": end_ts,
            "duration_h": duration_h,
            "mean": float(valid.mean()) if len(valid) else None,
            "std": float(valid.std(ddof=1)) if len(valid) > 1 else 0.0,
            "p10": float(np.percentile(valid, 10)) if len(valid) else None,
            "p90": float(np.percentile(valid, 90)) if len(valid) else None,
        })

    return segments


def _cluster_segments(segments: list[dict], n_clusters: int) -> list[int]:
    """Assign an integer cluster label to each segment."""
    features = np.array([
        [
            s["mean"] if s["mean"] is not None else 0.0,
            s["std"] if s["std"] is not None else 0.0,
            s["duration_h"],
        ]
        for s in segments
    ], dtype=float)

    k = min(n_clusters, len(segments))
    if k <= 1:
        return [0] * len(segments)

    try:
        from sklearn.cluster import KMeans  # type: ignore[import]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import]

        scaler = StandardScaler()
        scaled = scaler.fit_transform(features)
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(scaled).tolist()
        return labels

    except Exception:
        # Ultra-simple fallback: bin by mean value
        means = features[:, 0]
        finite = means[np.isfinite(means)]
        if len(finite) == 0:
            return [0] * len(segments)
        edges = np.linspace(finite.min(), finite.max() + 1e-9, k + 1)
        return [int(np.searchsorted(edges[1:], m)) for m in means]


def _stable_regime_id(asset_id: str, signal: str, ts: object, regime_type: str) -> str:
    key = f"{asset_id}:{signal}:{ts}:{regime_type}"
    return "reg_" + hashlib.md5(key.encode()).hexdigest()[:10]
