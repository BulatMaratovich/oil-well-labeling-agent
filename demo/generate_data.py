"""
Synthetic data generator for pipeline demonstration.

Generates realistic (but synthetic) time-series data for several device archetypes:
  - Small motor (0.5 kW): typical pump drive for low-volume wells
  - Medium motor (5 kW): standard pump drive
  - Large motor (35 kW): high-capacity pump drive

For each device, generates:
  - Historical normal-operation samples (for profile building)
  - Labeled anomaly examples (belt break, planned stop, amplitude change)

This lets us demonstrate the pipeline without real production data.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Callable

import numpy as np


def _make_timestamps(
    n: int,
    start: datetime,
    interval_minutes: float = 5.0,
    dropout_probability: float = 0.03,
) -> list[datetime]:
    """Generate irregular timestamps simulating signal dropout."""
    timestamps = [start]
    for _ in range(n - 1):
        # Occasionally skip a reading (signal dropout)
        skip = 2 if random.random() < dropout_probability else 1
        interval = interval_minutes * skip + random.gauss(0, 0.3)
        interval = max(interval, 0.5)
        timestamps.append(timestamps[-1] + timedelta(minutes=interval))
    return timestamps


def generate_normal_operation(
    n_samples: int,
    operating_mean: float,
    operating_std: float,
    amplitude: float,
    start: datetime,
) -> tuple[list[datetime], list[float]]:
    """Generate normal motor operation with realistic load oscillation."""
    timestamps = _make_timestamps(n_samples, start)

    # Load oscillation: sinusoidal component (pump cycle) + random noise
    cycle_period = random.randint(8, 20)  # samples per pump cycle
    t = np.arange(n_samples)
    cycle = amplitude * np.sin(2 * np.pi * t / cycle_period)

    noise = np.random.normal(0, operating_std * 0.3, n_samples)
    values = operating_mean + cycle + noise
    # Clamp to physically plausible range
    values = np.clip(values, operating_mean * 0.5, operating_mean * 1.8)

    return timestamps, list(values.astype(float))


def generate_belt_break(
    n_samples: int,
    operating_mean: float,
    idle_fraction: float,    # idle power as fraction of operating (0.1 - 0.45)
    operating_std: float,
    start: datetime,
) -> tuple[list[datetime], list[float]]:
    """
    Generate belt break event: sudden drop to idle power level.

    Motor continues running (non-zero power) but without mechanical load.
    Idle power is more stable (less oscillation than under load).
    """
    timestamps = _make_timestamps(n_samples, start)
    idle_power = operating_mean * idle_fraction
    # Much lower oscillation at idle: no pump load variance
    idle_std = operating_std * 0.15

    values = np.random.normal(idle_power, idle_std, n_samples)
    values = np.clip(values, 0, idle_power * 2)
    return timestamps, list(values.astype(float))


def generate_planned_stop(
    n_samples: int,
    stop_level: float,       # near-zero, typically 0.02-0.12 kW regardless of motor size
    start: datetime,
) -> tuple[list[datetime], list[float]]:
    """
    Generate planned motor stop: power drops to control electronics level.

    Very low absolute values — just lighting, control station, data transmission.
    This level is roughly constant regardless of motor size.
    """
    timestamps = _make_timestamps(n_samples, start)
    values = np.abs(np.random.normal(stop_level, stop_level * 0.4, n_samples))
    return timestamps, list(values.astype(float))


def generate_amplitude_anomaly(
    n_samples: int,
    operating_mean: float,
    operating_std: float,
    amplitude_multiplier: float,  # >1 = more oscillation, <1 = less
    start: datetime,
) -> tuple[list[datetime], list[float]]:
    """
    Generate amplitude anomaly: level stays roughly normal but oscillation changes.

    Could be bearing wear (increase) or blocked pump (decrease).
    """
    timestamps = _make_timestamps(n_samples, start)
    new_std = operating_std * amplitude_multiplier
    values = np.random.normal(operating_mean, new_std, n_samples)
    values = np.clip(values, 0, operating_mean * 3)
    return timestamps, list(values.astype(float))


# ── Device archetypes ────────────────────────────────────────────────────────

DEVICES = {
    "motor_small_001": {
        "operating_mean": 0.52,
        "operating_std": 0.08,
        "amplitude": 0.12,
        "idle_fraction": 0.22,
        "stop_level": 0.04,
        "description": "Small 0.5 kW motor (low-volume well pump)",
    },
    "motor_small_002": {
        "operating_mean": 0.61,
        "operating_std": 0.09,
        "amplitude": 0.14,
        "idle_fraction": 0.18,
        "stop_level": 0.05,
        "description": "Small 0.6 kW motor",
    },
    "motor_small_003": {
        "operating_mean": 0.48,
        "operating_std": 0.07,
        "amplitude": 0.10,
        "idle_fraction": 0.25,
        "stop_level": 0.03,
        "description": "Small 0.5 kW motor (worn bearings, slightly higher idle)",
    },
    "motor_medium_001": {
        "operating_mean": 4.85,
        "operating_std": 0.45,
        "amplitude": 0.65,
        "idle_fraction": 0.16,
        "stop_level": 0.06,
        "description": "Medium 5 kW motor (standard pump)",
    },
    "motor_medium_002": {
        "operating_mean": 6.20,
        "operating_std": 0.55,
        "amplitude": 0.80,
        "idle_fraction": 0.14,
        "stop_level": 0.07,
        "description": "Medium 6 kW motor",
    },
    "motor_medium_003": {
        "operating_mean": 5.10,
        "operating_std": 0.50,
        "amplitude": 0.70,
        "idle_fraction": 0.20,
        "stop_level": 0.08,
        "description": "Medium 5 kW motor (slightly higher idle due to age)",
    },
    "motor_large_001": {
        "operating_mean": 32.5,
        "operating_std": 2.1,
        "amplitude": 3.5,
        "idle_fraction": 0.28,
        "stop_level": 0.10,
        "description": "Large 32 kW motor (high-capacity pump)",
    },
    "motor_large_002": {
        "operating_mean": 38.0,
        "operating_std": 2.8,
        "amplitude": 4.2,
        "idle_fraction": 0.25,
        "stop_level": 0.11,
        "description": "Large 38 kW motor",
    },
}

# The device we will demonstrate the full pipeline on
DEMO_DEVICE = {
    "device_id": "motor_demo_001",
    "operating_mean": 5.85,
    "operating_std": 0.48,
    "amplitude": 0.72,
    "idle_fraction": 0.17,
    "stop_level": 0.07,
    "description": "Demo device: medium 6 kW motor (new, never seen a belt break)",
}


def build_dataset(seed: int = 42) -> dict:
    """
    Build a complete synthetic dataset for demonstration.

    Returns:
        {
          "devices": {device_id: {"normal": (ts, vals), "belt_break": (ts, vals), ...}},
          "demo": {"normal": (ts, vals), "belt_break_window": (ts, vals), "stop_window": (ts, vals)}
        }
    """
    np.random.seed(seed)
    random.seed(seed)

    base_time = datetime(2024, 1, 1, 8, 0, 0)
    dataset: dict = {"devices": {}, "demo": {}}

    # Generate data for known devices (will become the profile + example store)
    for device_id, cfg in DEVICES.items():
        t0 = base_time

        # Normal operation: 500 samples (~40 hours of data)
        ts_normal, vals_normal = generate_normal_operation(
            n_samples=500,
            operating_mean=cfg["operating_mean"],
            operating_std=cfg["operating_std"],
            amplitude=cfg["amplitude"],
            start=t0,
        )

        # Belt break: 80 samples
        t1 = ts_normal[-1] + timedelta(hours=2)
        ts_break, vals_break = generate_belt_break(
            n_samples=80,
            operating_mean=cfg["operating_mean"],
            idle_fraction=cfg["idle_fraction"],
            operating_std=cfg["operating_std"],
            start=t1,
        )

        # Planned stop: 40 samples
        t2 = ts_break[-1] + timedelta(hours=1)
        ts_stop, vals_stop = generate_planned_stop(
            n_samples=40,
            stop_level=cfg["stop_level"],
            start=t2,
        )

        dataset["devices"][device_id] = {
            "config": cfg,
            "normal": (ts_normal, vals_normal),
            "belt_break": (ts_break, vals_break),
            "planned_stop": (ts_stop, vals_stop),
        }

    # Demo device: unknown motor, profile from normal operation only
    t0 = base_time
    cfg = DEMO_DEVICE

    ts_normal, vals_normal = generate_normal_operation(
        n_samples=300,  # moderate history
        operating_mean=cfg["operating_mean"],
        operating_std=cfg["operating_std"],
        amplitude=cfg["amplitude"],
        start=t0,
    )

    # Pre-window: last 20 normal readings before the anomaly
    pre_ts = ts_normal[-20:]
    pre_vals = vals_normal[-20:]

    # Belt break window (what we're trying to label)
    t1 = pre_ts[-1] + timedelta(minutes=5)
    ts_break, vals_break = generate_belt_break(
        n_samples=60,
        operating_mean=cfg["operating_mean"],
        idle_fraction=cfg["idle_fraction"],
        operating_std=cfg["operating_std"],
        start=t1,
    )

    # Planned stop window (alternative scenario for comparison)
    t2 = ts_break[-1] + timedelta(hours=1)
    ts_stop, vals_stop = generate_planned_stop(
        n_samples=40,
        stop_level=cfg["stop_level"],
        start=t2,
    )

    dataset["demo"] = {
        "device_id": cfg["device_id"],
        "config": cfg,
        "normal": (ts_normal, vals_normal),
        "pre_window": (pre_ts, pre_vals),
        "belt_break_window": (ts_break, vals_break),
        "planned_stop_window": (ts_stop, vals_stop),
    }

    return dataset
