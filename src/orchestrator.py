"""
Pipeline orchestrator for the universal time-series labeling agent.

Coordinates the full pipeline:
  1. Load device profile (or cold-start)
  2. Compute signal features (deterministic)
  3. Detect anomaly type (deterministic)
  4. Find similar devices with labeled examples (k-NN)
  5. LLM reasoning → label + confidence (Claude)
  6. Route based on confidence thresholds

The orchestrator is scenario-agnostic: swap the ScenarioConfig to change
what anomaly type is being labeled without touching the pipeline code.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from src.agents.labeling_agent import label_window
from src.models.schemas import (
    DeviceProfile,
    DeviceStats,
    LabelProposal,
    ScenarioConfig,
    TimeSeriesWindow,
)
from src.tools.profile_store import ProfileStore
from src.tools.signal_processing import (
    compute_relative_features,
    compute_window_stats,
    detect_anomaly_type,
    detect_gaps_in_stream,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Full pipeline output including all intermediate results."""
    device_id: str
    scenario_id: str
    proposal: LabelProposal
    # Intermediate artifacts for auditability
    profile_summary: dict
    features_summary: dict
    anomaly_summary: dict
    similar_devices: list[str]
    processing_time_s: float


def load_scenario(scenario_path: str) -> ScenarioConfig:
    """Load a scenario definition from a YAML file."""
    with open(scenario_path) as f:
        data = yaml.safe_load(f)
    return ScenarioConfig(**data)


def _build_cold_start_profile(device_id: str, values: list[float]) -> DeviceProfile:
    """
    Build a minimal profile when no history exists for a device.

    For cold-start devices, we assume the provided values ARE normal operation
    (caller must ensure this). This gives us at least a scale estimate.
    """
    import numpy as np
    vals = np.array(values, dtype=float)

    operating_stats = DeviceStats(
        mean=float(np.mean(vals)),
        std=float(np.std(vals)),
        p10=float(np.percentile(vals, 10)),
        p90=float(np.percentile(vals, 90)),
        median=float(np.median(vals)),
    )
    # Amplitude estimated as local std (rough approximation for cold start)
    amplitude_stats = DeviceStats(
        mean=float(np.std(vals)),
        std=0.0, p10=0.0, p90=0.0,
        median=float(np.std(vals)),
    )

    logger.warning(
        "Cold-start profile for device %s (only %d samples). "
        "Profile will be refined with more data.",
        device_id, len(vals),
    )

    return DeviceProfile(
        device_id=device_id,
        operating_stats=operating_stats,
        amplitude_stats=amplitude_stats,
        sample_count=len(vals),
        last_updated=datetime.utcnow(),
    )


def _detect_boundary_gaps(
    window: TimeSeriesWindow,
) -> tuple[Optional[float], Optional[float]]:
    """
    Detect signal dropout gaps at the boundaries of the anomalous window.

    A gap before the window suggests communication failure rather than equipment event.
    A gap after suggests the equipment may have stopped entirely.
    """
    gap_before_s = None
    gap_after_s = None

    if window.pre_timestamps and window.timestamps:
        last_pre = window.pre_timestamps[-1].timestamp()
        first_win = window.timestamps[0].timestamp()
        gap = first_win - last_pre

        # Estimate expected interval from pre-window data
        if len(window.pre_timestamps) > 1:
            pre_ts = [t.timestamp() for t in window.pre_timestamps]
            import numpy as np
            median_interval = float(np.median(np.diff(pre_ts)))
            if gap > median_interval * 3:
                gap_before_s = gap - median_interval

    return gap_before_s, gap_after_s


class LabelingPipeline:
    """
    Universal time-series anomaly labeling pipeline.

    Usage:
        pipeline = LabelingPipeline(db_path="data/profiles.db")
        scenario = load_scenario("scenarios/belt_break.yaml")
        result = pipeline.run(window, scenario)
        print(result.proposal.label, result.proposal.confidence)
    """

    def __init__(self, db_path: str = "data/profiles.db", similar_devices_k: int = 5):
        self.store = ProfileStore(db_path=db_path)
        self.k = similar_devices_k

    def run(
        self,
        window: TimeSeriesWindow,
        scenario: ScenarioConfig,
        normal_baseline_values: Optional[list[float]] = None,
    ) -> PipelineResult:
        """
        Run the full labeling pipeline for one anomalous window.

        Args:
            window: The anomalous time-series window to label.
            scenario: What type of anomaly to look for.
            normal_baseline_values: Normal-operation samples for cold-start devices.
                If None and no profile exists, uses pre_values as baseline.
        """
        import time
        t0 = time.time()

        # ── Step 1: Device profile ──────────────────────────────────────
        profile = self.store.get_profile(window.device_id)

        if profile is None:
            baseline = normal_baseline_values or window.pre_values
            if baseline:
                profile = _build_cold_start_profile(window.device_id, baseline)
                self.store.save_profile(profile)
            else:
                raise ValueError(
                    f"No profile for device {window.device_id} and no baseline provided. "
                    "Cannot compute relative features."
                )

        logger.info(
            "Device %s profile: mean=%.2f kW, std=%.2f kW, %d samples",
            window.device_id, profile.operating_stats.mean,
            profile.operating_stats.std, profile.sample_count,
        )

        # ── Step 2: Signal feature extraction ──────────────────────────
        window_stats = compute_window_stats(window.timestamps, window.values)

        gap_before_s, gap_after_s = _detect_boundary_gaps(window)

        features = compute_relative_features(
            window_stats=window_stats,
            profile=profile,
            pre_timestamps=window.pre_timestamps or None,
            pre_values=window.pre_values or None,
            gap_before_s=gap_before_s,
            gap_after_s=gap_after_s,
        )

        logger.info(
            "Features: level_ratio=%.2f, amplitude_ratio=%.2f, transition=%.2f",
            features.level_ratio, features.amplitude_ratio, features.transition_sharpness,
        )

        # ── Step 3: Anomaly classification ─────────────────────────────
        anomaly_report = detect_anomaly_type(features, profile)

        logger.info(
            "Anomaly type: %s (severity=%.2f)",
            anomaly_report.anomaly_type.value, anomaly_report.severity,
        )

        # ── Step 4: Similar device retrieval ───────────────────────────
        similar_ids = self.store.get_similar_devices(
            profile=profile,
            k=self.k,
            scenario_id=window.scenario_id,
        )

        similar_examples = self.store.get_examples_for_devices(
            device_ids=similar_ids,
            scenario_id=window.scenario_id,
        )

        logger.info(
            "Found %d similar devices, %d labeled examples",
            len(similar_ids), len(similar_examples),
        )

        # ── Step 5: LLM labeling ────────────────────────────────────────
        proposal = label_window(
            scenario=scenario,
            profile=profile,
            features=features,
            anomaly_report=anomaly_report,
            similar_examples=similar_examples,
            similar_device_ids=similar_ids,
        )

        elapsed = time.time() - t0

        logger.info(
            "Label: '%s' (confidence=%.2f, routing=%s) in %.1fs",
            proposal.label, proposal.confidence, proposal.routing.value, elapsed,
        )

        return PipelineResult(
            device_id=window.device_id,
            scenario_id=window.scenario_id,
            proposal=proposal,
            profile_summary={
                "operating_mean_kw": profile.operating_stats.mean,
                "operating_std_kw": profile.operating_stats.std,
                "sample_count": profile.sample_count,
                "has_labeled_examples": profile.has_labeled_examples,
            },
            features_summary={
                "level_ratio": features.level_ratio,
                "amplitude_ratio": features.amplitude_ratio,
                "transition_sharpness": features.transition_sharpness,
                "window_mean_kw": features.window_mean,
                "duration_s": features.duration_in_state_s,
                "gap_before_s": features.gap_before_s,
            },
            anomaly_summary={
                "type": anomaly_report.anomaly_type.value,
                "severity": anomaly_report.severity,
                "description": anomaly_report.description,
            },
            similar_devices=similar_ids,
            processing_time_s=elapsed,
        )

    def seed_example(
        self,
        device_id: str,
        scenario_id: str,
        label: str,
        timestamps: list[datetime],
        values: list[float],
        description: str,
    ) -> None:
        """
        Add a validated labeled example to the knowledge base.

        Called after a human expert validates a label — this grows the
        few-shot pool for future similar device lookups.
        """
        profile = self.store.get_profile(device_id)
        if profile is None:
            raise ValueError(f"No profile for device {device_id}. Build profile first.")

        stats = compute_window_stats(timestamps, values)
        features = compute_relative_features(stats, profile)

        self.store.add_labeled_example(
            device_id=device_id,
            scenario_id=scenario_id,
            label=label,
            level_ratio=features.level_ratio,
            amplitude_ratio=features.amplitude_ratio,
            window_mean=features.window_mean,
            window_std=features.window_std,
            operating_mean=profile.operating_stats.mean,
            description=description,
        )
        logger.info("Seeded example: device=%s, label=%s", device_id, label)

    def close(self) -> None:
        self.store.close()
