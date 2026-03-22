"""
Core data models for the universal time-series labeling agent.

All features are relative to the device's own historical profile,
not absolute values — this is critical for handling 10,000+ devices
with vastly different power scales.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class RoutingDecision(str, Enum):
    AUTO_LABEL = "auto_label"          # confidence > threshold: label without review
    REVIEW = "review"                  # moderate confidence: show to human with explanation
    MANDATORY_REVIEW = "mandatory_review"  # low confidence: must be reviewed


class AnomalyType(str, Enum):
    LEVEL_DROP = "level_drop"                  # power dropped significantly vs normal
    LEVEL_SPIKE = "level_spike"                # power jumped significantly vs normal
    AMPLITUDE_INCREASE = "amplitude_increase"  # oscillation much higher than normal
    AMPLITUDE_DECREASE = "amplitude_decrease"  # oscillation much lower than normal
    STABLE_BELOW_NORMAL = "stable_below_normal"  # stable at sub-normal level (idle or stop)
    SIGNAL_DROPOUT = "signal_dropout"          # large gaps in timestamps
    COMPLEX = "complex"                        # multiple simultaneous changes


class DeviceStats(BaseModel):
    """Statistical summary of a device's signal in one operational mode."""
    mean: float
    std: float
    p10: float
    p90: float
    median: float


class DeviceProfile(BaseModel):
    """
    Per-device historical profile built from streaming data.

    This is the anchor for all relative feature computation.
    A device's "normal" is specific to that device — a 30 kW motor's
    idle looks completely different from a 0.5 kW motor's idle.
    """
    device_id: str
    operating_stats: DeviceStats    # stats when working under full load
    amplitude_stats: DeviceStats    # typical oscillation characteristics
    idle_stats: Optional[DeviceStats] = None   # stats when motor runs without belt load
    stop_stats: Optional[DeviceStats] = None   # stats when motor is fully stopped
    sample_count: int = 0
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    has_labeled_examples: bool = False


class TimeSeriesWindow(BaseModel):
    """
    A window of time-series data flagged for labeling.

    Includes the window itself plus pre-window context for transition analysis.
    No event labels — only the raw signal and timestamps.
    """
    device_id: str
    timestamps: list[datetime]
    values: list[float]
    scenario_id: str
    # Context readings before the anomalous window (for transition analysis)
    pre_timestamps: list[datetime] = Field(default_factory=list)
    pre_values: list[float] = Field(default_factory=list)


class RelativeFeatures(BaseModel):
    """
    Window features normalized relative to the device's own historical profile.

    Using absolute kW values fails at scale: a "1.2 kW anomaly" is huge
    for a 0.5 kW motor and negligible for a 30 kW motor.
    All ratios are device-relative.
    """
    # Primary discriminators — relative to device's operating profile
    level_ratio: float = Field(description="window_mean / operating_mean. <0.3 = major drop, >1.5 = spike")
    amplitude_ratio: float = Field(description="window_std / operating_std. <0.3 = suspiciously stable, >2.0 = high noise")
    transition_sharpness: float = Field(description="Normalized rate of change at window boundary. High = sudden event")

    # Signal quality context
    gap_before_s: Optional[float] = Field(None, description="Seconds of signal dropout immediately before this window")
    gap_after_s: Optional[float] = Field(None, description="Seconds of signal dropout immediately after this window")
    duration_in_state_s: float = Field(description="How long the device has been in this anomalous state")
    samples_in_window: int

    # Raw stats (in kW) for LLM context — never used as thresholds
    window_mean: float
    window_std: float
    window_min: float
    window_max: float
    window_p10: float
    window_p90: float


class AnomalyReport(BaseModel):
    """Output of deterministic anomaly detection — classifies the deviation type."""
    anomaly_type: AnomalyType
    severity: float = Field(ge=0.0, le=1.0, description="0=borderline, 1=extreme deviation")
    description: str
    has_dropout_nearby: bool
    dropout_duration_s: Optional[float] = None


class LabeledExample(BaseModel):
    """A labeled anomaly from a similar device, used as few-shot context."""
    device_id: str
    label: str
    level_ratio: float
    amplitude_ratio: float
    window_mean: float
    window_std: float
    operating_mean: float     # absolute scale of this device for context
    description: str          # brief human-readable summary


class CompetingLabel(BaseModel):
    label: str
    description: str


class ScenarioConfig(BaseModel):
    """
    Scenario definition — what anomaly to look for and how to recognize it.

    This is the 'universal' part: swapping this config changes what the agent
    looks for without any code changes.
    """
    scenario_id: str
    name: str
    description: str           # natural language: what does this anomaly look like?
    target_label: str          # the label to assign when detected
    competing_labels: list[CompetingLabel]  # other possible explanations
    physical_constraints: str  # physics-based reasoning hints
    confidence_thresholds: dict = Field(
        default={"auto_label": 0.85, "review": 0.60}
    )


class LabelProposalOutput(BaseModel):
    """
    Structured output from the LLM labeling agent.
    Used for JSON schema enforcement in the API call.
    """
    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="Proposed label: one of the scenario labels or 'uncertain'")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in the label (0=uncertain, 1=certain)")
    explanation: str = Field(description="Natural language explanation citing specific evidence from the device profile and features")
    evidence: list[str] = Field(description="Specific evidence items: each is a concrete observation supporting the label")
    physical_plausibility: str = Field(description="Assessment: is this pattern physically consistent with the proposed label?")


class LabelProposal(BaseModel):
    """Full label proposal including routing decision."""
    label: str
    confidence: float
    explanation: str
    evidence: list[str]
    physical_plausibility: str
    routing: RoutingDecision
    similar_devices_used: list[str] = Field(default_factory=list)
