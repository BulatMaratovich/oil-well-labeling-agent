"""
Automatic detection of anomalous windows in a device's time series.

Approach:
  1. Build a rolling baseline profile (what's "normal" for this device)
  2. Detect segments where the signal deviates significantly from baseline
  3. Merge nearby segments into single windows
  4. Add pre/post context around each window for transition analysis

Design choices:
  - Uses median + MAD (robust to outliers) instead of mean + std
  - Separate detection for level anomalies and amplitude anomalies
  - Minimum window duration threshold (avoid labeling single-point spikes)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class AnomalousWindow:
    device_id: str
    # Core window (the anomalous segment)
    start_idx: int
    end_idx: int
    start_time: datetime
    end_time: datetime
    # Extended context (pre/post window readings for transition analysis)
    context_start_idx: int
    context_end_idx: int
    # Anomaly characterization
    level_ratio: float        # median(window) / median(baseline)
    amplitude_ratio: float    # mad(window) / mad(baseline)
    anomaly_score: float      # combined deviation score 0-1
    anomaly_type: str         # "level_drop", "level_spike", "amplitude_change", "complex"
    # Current label (set during interactive session)
    label: Optional[str] = None
    confidence: Optional[float] = None
    routing: Optional[str] = None
    explanation: Optional[str] = None


def detect_windows(
    timestamps: list[datetime],
    values: list[float],
    device_id: str,
    # Detection parameters
    level_threshold: float = 0.4,      # flag if level deviates by > this fraction
    amplitude_threshold: float = 2.0,   # flag if amplitude changes by > this factor
    min_window_samples: int = 5,        # ignore anomalies shorter than this
    merge_gap_samples: int = 10,        # merge windows separated by fewer samples
    context_samples: int = 20,          # readings to include before/after window
) -> list[AnomalousWindow]:
    """
    Detect anomalous windows in a device's time series.

    Returns windows sorted by start time, each with pre/post context
    for transition analysis.
    """
    if len(values) < 30:
        return []  # not enough data to establish a baseline

    vals = np.array(values, dtype=float)
    n = len(vals)

    # ── Build rolling baseline ────────────────────────────────────────
    # Use a lookback window to estimate "normal" at each point.
    # Lookback of 60 samples ≈ 5 hours at 5-min intervals.
    lookback = min(60, n // 4)

    baseline_median = np.zeros(n)
    baseline_mad = np.zeros(n)

    for i in range(n):
        start = max(0, i - lookback)
        end = i  # only use past data, no future leakage
        if end - start < 10:
            # Not enough history yet: use full available data
            chunk = vals[:max(i + 1, 10)]
        else:
            chunk = vals[start:end]

        med = float(np.median(chunk))
        mad = float(np.median(np.abs(chunk - med))) * 1.4826  # normalize to std units
        baseline_median[i] = med
        baseline_mad[i] = max(mad, 1e-6)

    # ── Compute per-sample deviation scores ──────────────────────────
    level_deviation = np.abs(vals - baseline_median) / (baseline_median + 1e-6)

    # Amplitude: use rolling local std vs baseline mad
    local_std = np.zeros(n)
    window_size = 10
    for i in range(n):
        start = max(0, i - window_size // 2)
        end = min(n, i + window_size // 2)
        local_std[i] = float(np.std(vals[start:end]))

    amplitude_ratio_arr = local_std / baseline_mad

    # ── Flag anomalous samples ────────────────────────────────────────
    level_flag = level_deviation > level_threshold
    amplitude_flag = (amplitude_ratio_arr > amplitude_threshold) | (amplitude_ratio_arr < 1.0 / amplitude_threshold)

    anomaly_flag = level_flag | amplitude_flag

    # ── Extract contiguous anomalous segments ─────────────────────────
    segments: list[tuple[int, int]] = []
    in_segment = False
    seg_start = 0

    for i in range(n):
        if anomaly_flag[i] and not in_segment:
            seg_start = i
            in_segment = True
        elif not anomaly_flag[i] and in_segment:
            segments.append((seg_start, i - 1))
            in_segment = False

    if in_segment:
        segments.append((seg_start, n - 1))

    # ── Filter too-short segments ─────────────────────────────────────
    segments = [(s, e) for s, e in segments if (e - s + 1) >= min_window_samples]

    # ── Merge nearby segments ─────────────────────────────────────────
    if not segments:
        return []

    merged: list[tuple[int, int]] = [segments[0]]
    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    # ── Build AnomalousWindow objects ─────────────────────────────────
    windows: list[AnomalousWindow] = []

    for seg_start, seg_end in merged:
        win_vals = vals[seg_start:seg_end + 1]
        base_med = float(np.median(baseline_median[seg_start:seg_end + 1]))
        base_mad = float(np.median(baseline_mad[seg_start:seg_end + 1]))

        win_med = float(np.median(win_vals))
        win_mad = float(np.median(np.abs(win_vals - win_med))) * 1.4826

        lv_ratio = win_med / max(base_med, 1e-6)
        amp_ratio = win_mad / max(base_mad, 1e-6)

        # Classify anomaly type
        level_drop = lv_ratio < (1 - level_threshold)
        level_spike = lv_ratio > (1 + level_threshold)
        amp_change = amp_ratio > amplitude_threshold or amp_ratio < 1.0 / amplitude_threshold

        if level_drop and amp_change:
            atype = "complex"
        elif level_drop:
            atype = "level_drop"
        elif level_spike:
            atype = "level_spike"
        elif amp_change:
            atype = "amplitude_change"
        else:
            atype = "complex"

        # Anomaly score: combined deviation
        score = min(1.0, abs(1 - lv_ratio) * 0.6 + abs(1 - min(amp_ratio, 1 / (amp_ratio + 1e-6))) * 0.4)

        # Context indices
        ctx_start = max(0, seg_start - context_samples)
        ctx_end = min(n - 1, seg_end + context_samples)

        windows.append(AnomalousWindow(
            device_id=device_id,
            start_idx=seg_start,
            end_idx=seg_end,
            start_time=timestamps[seg_start],
            end_time=timestamps[seg_end],
            context_start_idx=ctx_start,
            context_end_idx=ctx_end,
            level_ratio=lv_ratio,
            amplitude_ratio=amp_ratio,
            anomaly_score=score,
            anomaly_type=atype,
        ))

    return windows


def parse_csv(
    file_path: str,
) -> tuple[list[str], dict[str, tuple[list[datetime], list[float]]]]:
    """
    Parse a CSV file into per-device time series.

    Returns:
        (column_names, {device_id: (timestamps, values)})

    Handles common formats:
      - Single device: timestamp, value
      - Multi device: timestamp, device_id, value
    Columns are identified by Claude in the interactive session.
    """
    import csv
    from pathlib import Path

    rows = []
    with open(file_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        columns = list(reader.fieldnames or [])
        for row in reader:
            rows.append(row)

    return columns, rows


def extract_series(
    rows: list[dict],
    timestamp_col: str,
    value_col: str,
    device_col: Optional[str] = None,
    default_device_id: str = "device_001",
) -> dict[str, tuple[list[datetime], list[float]]]:
    """
    Extract per-device time series from parsed CSV rows.

    Handles various timestamp formats automatically.
    """
    from dateutil import parser as dateparser

    series: dict[str, tuple[list[datetime], list[float]]] = {}

    for row in rows:
        device_id = row.get(device_col, default_device_id) if device_col else default_device_id
        ts_str = row.get(timestamp_col, "")
        val_str = row.get(value_col, "")

        try:
            ts = dateparser.parse(ts_str)
            val = float(val_str)
        except (ValueError, TypeError):
            continue  # skip unparseable rows

        if device_id not in series:
            series[device_id] = ([], [])
        series[device_id][0].append(ts)
        series[device_id][1].append(val)

    # Sort by timestamp
    for device_id in series:
        ts_list, val_list = series[device_id]
        pairs = sorted(zip(ts_list, val_list), key=lambda x: x[0])
        series[device_id] = ([p[0] for p in pairs], [p[1] for p in pairs])

    return series
