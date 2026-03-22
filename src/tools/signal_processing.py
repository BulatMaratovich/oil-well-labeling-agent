"""
Signal processing tools for time-series anomaly analysis.

Key design principles:
- All features are RELATIVE to the device's own profile, never absolute
- Handles irregular sampling (2-15 min intervals)
- Detects signal dropout via timestamp gaps (no values = no reading)
- Distinguishes: belt break (idle power) vs stop (near zero) vs dropout (no data)
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from src.models.schemas import (
    AnomalyReport,
    AnomalyType,
    DeviceProfile,
    DeviceStats,
    RelativeFeatures,
)

# A gap is considered a signal dropout if it's >3x the median interval
_DROPOUT_MULTIPLIER = 3.0

# Transition is "sharp" if level changes by >50% of operating range in one step
_SHARP_TRANSITION_THRESHOLD = 0.5


def compute_window_stats(
    timestamps: list[datetime],
    values: list[float],
) -> dict:
    """
    Compute raw statistics for a window of readings.

    Returns absolute values (kW) — caller must normalize via device profile.
    Also computes gap statistics: gaps in timestamps = signal dropout events.
    """
    if not values:
        raise ValueError("Cannot compute stats for empty window")

    vals = np.array(values, dtype=float)
    ts = np.array([t.timestamp() for t in timestamps], dtype=float)

    result: dict = {
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "median": float(np.median(vals)),
        "p10": float(np.percentile(vals, 10)),
        "p90": float(np.percentile(vals, 90)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "duration_s": float(ts[-1] - ts[0]) if len(ts) > 1 else 0.0,
        "samples": len(vals),
        "dropout_count": 0,
        "total_dropout_s": 0.0,
    }

    # Detect internal gaps (signal dropouts within the window)
    if len(ts) > 1:
        diffs = np.diff(ts)
        median_interval = float(np.median(diffs))
        threshold = median_interval * _DROPOUT_MULTIPLIER
        dropout_mask = diffs > threshold
        result["dropout_count"] = int(np.sum(dropout_mask))
        # Dropout duration = excess time beyond expected interval
        excess = diffs[dropout_mask] - median_interval
        result["total_dropout_s"] = float(np.sum(excess))
        result["median_interval_s"] = median_interval
    else:
        result["median_interval_s"] = 0.0

    return result


def compute_relative_features(
    window_stats: dict,
    profile: DeviceProfile,
    pre_timestamps: Optional[list[datetime]] = None,
    pre_values: Optional[list[float]] = None,
    gap_before_s: Optional[float] = None,
    gap_after_s: Optional[float] = None,
) -> RelativeFeatures:
    """
    Normalize window statistics relative to this device's operating profile.

    This is the critical normalization step: a 1.2 kW reading means nothing
    without knowing that this device normally runs at 0.8 kW (huge anomaly)
    vs 30 kW (negligible). All downstream reasoning uses these ratios.
    """
    op = profile.operating_stats

    # Guard against division by zero for very low power devices
    op_mean = op.mean if abs(op.mean) > 1e-6 else 1.0
    op_std = op.std if op.std > 1e-6 else 1.0

    level_ratio = window_stats["mean"] / op_mean
    amplitude_ratio = window_stats["std"] / op_std

    # Compute transition sharpness: how abruptly did the level change?
    transition_sharpness = 0.0
    if pre_values and len(pre_values) >= 2:
        pre_vals = np.array(pre_values[-5:], dtype=float)  # last 5 pre-window readings
        win_vals = np.array([window_stats["mean"]], dtype=float)
        pre_mean = float(np.mean(pre_vals))
        level_change = abs(window_stats["mean"] - pre_mean)
        # Normalize by operating range (p90 - p10)
        op_range = max(op.p90 - op.p10, op_std)
        transition_sharpness = min(1.0, level_change / op_range) if op_range > 1e-6 else 0.0

    return RelativeFeatures(
        level_ratio=level_ratio,
        amplitude_ratio=amplitude_ratio,
        transition_sharpness=transition_sharpness,
        gap_before_s=gap_before_s,
        gap_after_s=gap_after_s,
        duration_in_state_s=window_stats["duration_s"],
        samples_in_window=window_stats["samples"],
        window_mean=window_stats["mean"],
        window_std=window_stats["std"],
        window_min=window_stats["min"],
        window_max=window_stats["max"],
        window_p10=window_stats["p10"],
        window_p90=window_stats["p90"],
    )


def detect_anomaly_type(
    features: RelativeFeatures,
    profile: DeviceProfile,
) -> AnomalyReport:
    """
    Classify the type of anomaly based on relative features.

    This is deterministic classification — the LLM then reasons about
    which specific label within this anomaly type is most likely.
    """
    has_dropout = (features.gap_before_s is not None and features.gap_before_s > 120)
    dropout_s = features.gap_before_s if has_dropout else None

    # Primary classification by level deviation
    if features.level_ratio < 0.15:
        # Very low power — either stopped or belt break with very low idle
        if features.amplitude_ratio < 0.4:
            anomaly_type = AnomalyType.STABLE_BELOW_NORMAL
            description = (
                f"Power at {features.level_ratio:.0%} of normal with very stable signal "
                f"(oscillation {features.amplitude_ratio:.0%} of normal). "
                "Consistent with stopped motor or belt break at low idle."
            )
        else:
            anomaly_type = AnomalyType.LEVEL_DROP
            description = (
                f"Power dropped to {features.level_ratio:.0%} of normal. "
                f"Oscillation still {features.amplitude_ratio:.0%} of normal."
            )
    elif features.level_ratio < 0.6:
        # Significant drop — likely idle or low load
        if features.amplitude_ratio < 0.5:
            anomaly_type = AnomalyType.STABLE_BELOW_NORMAL
            description = (
                f"Power at {features.level_ratio:.0%} of normal, unusually stable "
                f"(oscillation only {features.amplitude_ratio:.0%} of normal). "
                "Motor may be running without mechanical load (belt break / idle)."
            )
        else:
            anomaly_type = AnomalyType.LEVEL_DROP
            description = (
                f"Power dropped to {features.level_ratio:.0%} of normal "
                f"with {features.amplitude_ratio:.0%} of normal oscillation."
            )
    elif features.level_ratio > 1.8:
        anomaly_type = AnomalyType.LEVEL_SPIKE
        description = (
            f"Power spiked to {features.level_ratio:.0%} of normal. "
            "Possible overload or runaway condition."
        )
    elif features.amplitude_ratio > 2.5:
        anomaly_type = AnomalyType.AMPLITUDE_INCREASE
        description = (
            f"Power oscillation increased to {features.amplitude_ratio:.0%} of normal "
            f"while mean level is {features.level_ratio:.0%} of normal. "
            "Possible mechanical instability or load fluctuation."
        )
    elif features.amplitude_ratio < 0.2:
        anomaly_type = AnomalyType.AMPLITUDE_DECREASE
        description = (
            f"Power oscillation dropped to {features.amplitude_ratio:.0%} of normal "
            f"while mean level is {features.level_ratio:.0%} of normal. "
            "Unusually stable signal — possible sensor issue or motor stopped."
        )
    elif has_dropout:
        anomaly_type = AnomalyType.SIGNAL_DROPOUT
        description = (
            f"Signal dropout of {features.gap_before_s:.0f}s detected before this window. "
            f"Current power at {features.level_ratio:.0%} of normal."
        )
    else:
        anomaly_type = AnomalyType.COMPLEX
        description = (
            f"Complex pattern: power at {features.level_ratio:.0%} of normal, "
            f"oscillation at {features.amplitude_ratio:.0%} of normal."
        )

    # Severity: how far from normal across both dimensions
    level_deviation = abs(1.0 - features.level_ratio)
    amplitude_deviation = abs(1.0 - features.amplitude_ratio)
    severity = min(1.0, level_deviation * 0.65 + amplitude_deviation * 0.35)

    return AnomalyReport(
        anomaly_type=anomaly_type,
        severity=severity,
        description=description,
        has_dropout_nearby=has_dropout,
        dropout_duration_s=dropout_s,
    )


def build_profile_from_samples(
    device_id: str,
    timestamps: list[datetime],
    values: list[float],
) -> DeviceStats:
    """
    Build operating statistics from a sequence of normal-operation readings.

    Used to initialize or update a device profile from historical data.
    The caller is responsible for ensuring these are normal-operation samples.
    """
    vals = np.array(values, dtype=float)
    # Amplitude: computed over rolling windows to capture oscillation
    window_size = min(10, len(vals) // 5)
    if window_size >= 2:
        windows = [vals[i:i+window_size] for i in range(0, len(vals) - window_size, window_size // 2)]
        window_stds = [float(np.std(w)) for w in windows if len(w) >= 2]
        amp_mean = float(np.mean(window_stds)) if window_stds else float(np.std(vals))
        amp_std = float(np.std(window_stds)) if window_stds else 0.0
    else:
        amp_mean = float(np.std(vals))
        amp_std = 0.0

    return DeviceStats(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        p10=float(np.percentile(vals, 10)),
        p90=float(np.percentile(vals, 90)),
        median=float(np.median(vals)),
    )


def detect_gaps_in_stream(
    timestamps: list[datetime],
    expected_interval_s: Optional[float] = None,
) -> list[tuple[datetime, datetime, float]]:
    """
    Detect signal dropout gaps in a timestamp sequence.

    Returns list of (gap_start, gap_end, duration_s) tuples.
    If expected_interval_s is None, uses median interval as baseline.
    """
    if len(timestamps) < 2:
        return []

    ts = np.array([t.timestamp() for t in timestamps])
    diffs = np.diff(ts)

    if expected_interval_s is None:
        expected_interval_s = float(np.median(diffs))

    threshold = expected_interval_s * _DROPOUT_MULTIPLIER
    gaps = []
    for i, diff in enumerate(diffs):
        if diff > threshold:
            gap_duration = diff - expected_interval_s
            gaps.append((timestamps[i], timestamps[i + 1], gap_duration))

    return gaps
