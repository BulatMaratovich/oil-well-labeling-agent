"""
LLM-based labeling agent using Claude.

This is the reasoning core of the system. Given:
  - The device's historical operating profile (what's "normal" for this device)
  - Relative features (how this window deviates from normal)
  - Anomaly classification (what type of deviation)
  - Few-shot examples from similar devices (what their labeled anomalies looked like)
  - Scenario definition (what anomaly type we're looking for and how to recognize it)

Claude reasons about physical plausibility and outputs a label + confidence + explanation.

Key design choices:
  - Adaptive thinking enabled: this task involves weighing competing hypotheses
  - Structured JSON output: enforces label taxonomy, prevents hallucination
  - Conservative by default: "uncertain" routing rather than wrong auto-label
  - False positives cost more than false negatives (as specified by domain experts)
"""
from __future__ import annotations

import json

import anthropic

from src.models.schemas import (
    AnomalyReport,
    DeviceProfile,
    LabeledExample,
    LabelProposal,
    LabelProposalOutput,
    RelativeFeatures,
    RoutingDecision,
    ScenarioConfig,
)

_MODEL = "claude-opus-4-6"

_SYSTEM_PROMPT = """You are an expert industrial equipment diagnostics specialist analyzing \
power consumption time-series data from electric motors.

Your task: determine whether an anomalous power reading is a genuine equipment failure \
(specifically the target anomaly defined in the scenario) or has an alternative benign explanation.

Critical principles:
1. ALL feature values are RELATIVE to this device's own historical operating profile — \
not absolute kW values. A level_ratio of 0.3 means "30% of normal operating power for THIS device."
2. Physical plausibility matters: reason about what the signal pattern implies physically, \
not just whether numbers match a pattern.
3. Be CONSERVATIVE: a false positive (flagging normal as abnormal) is more costly than \
a false negative. When uncertain, route to human review rather than auto-labeling.
4. Different motors have very different scales: idle power for a 30 kW motor may be 8-10 kW, \
while a 0.5 kW motor's idle is 0.1-0.2 kW. Always reason in relative terms.

Output valid JSON matching the requested schema."""


def build_prompt(
    scenario: ScenarioConfig,
    profile: DeviceProfile,
    features: RelativeFeatures,
    anomaly_report: AnomalyReport,
    similar_examples: list[LabeledExample],
) -> str:
    """Construct the reasoning prompt for the labeling agent."""

    op = profile.operating_stats
    amp = profile.amplitude_stats

    # Profile context
    profile_section = f"""## Device Profile: {profile.device_id}
Normal operating power: {op.mean:.3f} kW (std: {op.std:.3f} kW, range: [{op.p10:.3f}, {op.p90:.3f}] kW)
Normal oscillation amplitude: {amp.mean:.3f} kW
Profile based on: {profile.sample_count} samples"""

    if profile.idle_stats:
        profile_section += f"\nKnown idle power level: {profile.idle_stats.mean:.3f} kW"
    if profile.stop_stats:
        profile_section += f"\nKnown stop power level: {profile.stop_stats.mean:.3f} kW"

    # Anomalous window features
    gap_info = ""
    if features.gap_before_s:
        gap_info = f"\n⚠ Signal dropout of {features.gap_before_s:.0f}s before this window (possible communication failure)"
    if features.gap_after_s:
        gap_info += f"\n⚠ Signal dropout of {features.gap_after_s:.0f}s after this window"

    features_section = f"""## Anomalous Window Features (all ratios relative to this device's profile)
- Level ratio: {features.level_ratio:.3f} (window mean is {features.level_ratio:.0%} of normal operating power)
- Amplitude ratio: {features.amplitude_ratio:.3f} (oscillation is {features.amplitude_ratio:.0%} of normal)
- Transition sharpness: {features.transition_sharpness:.3f} (0=gradual, 1=instantaneous drop)
- Duration in this state: {features.duration_in_state_s:.0f} seconds
- Absolute power: {features.window_mean:.3f} kW (std: {features.window_std:.3f} kW, range: [{features.window_min:.3f}, {features.window_max:.3f}] kW)
- Anomaly classification: {anomaly_report.anomaly_type.value} (severity: {anomaly_report.severity:.2f})
- {anomaly_report.description}{gap_info}"""

    # Few-shot examples from similar devices
    if similar_examples:
        examples_lines = []
        for ex in similar_examples:
            examples_lines.append(
                f"  - Device {ex.device_id} (operating at {ex.operating_mean:.2f} kW): "
                f"level={ex.level_ratio:.2f}x, amplitude={ex.amplitude_ratio:.2f}x → "
                f"label='{ex.label}' — {ex.description}"
            )
        examples_section = "## Similar Device Examples (few-shot context)\n" + "\n".join(examples_lines)
    else:
        examples_section = "## Similar Device Examples\nNo labeled examples available for similar devices. Reason from first principles."

    # Competing label descriptions
    competing = "\n".join([
        f"  - '{cl.label}': {cl.description}"
        for cl in scenario.competing_labels
    ])
    labels_section = f"""## Label Options
Target: '{scenario.target_label}'
Alternatives:
{competing}
  - 'uncertain': Cannot determine with confidence — route to human review"""

    # Schema for output
    schema = LabelProposalOutput.model_json_schema()

    return f"""# Anomaly Labeling Task

## Scenario: {scenario.name}
{scenario.description}

{profile_section}

{features_section}

{examples_section}

## Physical Constraints for This Scenario
{scenario.physical_constraints}

{labels_section}

## Instructions
1. Consider each label option and assess physical plausibility for THIS device
2. What would idle running look like for a device normally operating at {op.mean:.2f} kW?
3. Is the level_ratio consistent with the target label vs alternatives?
4. How does this compare to similar device examples above?
5. If multiple explanations fit equally well → choose 'uncertain'
6. Confidence guide: >0.85 = clear evidence; 0.6-0.85 = probable; <0.6 = uncertain

Output JSON matching this schema:
{json.dumps(schema, indent=2)}"""


def compute_routing(
    confidence: float,
    thresholds: dict,
) -> RoutingDecision:
    """Determine routing based on confidence and scenario thresholds."""
    if confidence >= thresholds.get("auto_label", 0.85):
        return RoutingDecision.AUTO_LABEL
    elif confidence >= thresholds.get("review", 0.60):
        return RoutingDecision.REVIEW
    else:
        return RoutingDecision.MANDATORY_REVIEW


def label_window(
    scenario: ScenarioConfig,
    profile: DeviceProfile,
    features: RelativeFeatures,
    anomaly_report: AnomalyReport,
    similar_examples: list[LabeledExample],
    similar_device_ids: list[str],
) -> LabelProposal:
    """
    Use Claude to reason about and label an anomalous window.

    This is the core LLM call. Claude receives:
    - The scenario definition (what we're looking for)
    - The device's operating profile (what's normal for this device)
    - Relative features (how the anomaly deviates from normal)
    - Few-shot examples from similar devices

    And outputs a label, confidence, explanation, and routing decision.
    """
    client = anthropic.Anthropic()

    prompt = build_prompt(
        scenario=scenario,
        profile=profile,
        features=features,
        anomaly_report=anomaly_report,
        similar_examples=similar_examples,
    )

    response = client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        thinking={"type": "adaptive"},
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
        output_config={
            "format": {
                "type": "json_schema",
                "schema": LabelProposalOutput.model_json_schema(),
            }
        },
    )

    # Extract text content (thinking blocks are separate)
    text_content = next(
        (block.text for block in response.content if block.type == "text"),
        None,
    )
    if text_content is None:
        raise ValueError("LLM returned no text content")

    raw = LabelProposalOutput.model_validate_json(text_content)

    routing = compute_routing(
        confidence=raw.confidence,
        thresholds=scenario.confidence_thresholds,
    )

    return LabelProposal(
        label=raw.label,
        confidence=raw.confidence,
        explanation=raw.explanation,
        evidence=raw.evidence,
        physical_plausibility=raw.physical_plausibility,
        routing=routing,
        similar_devices_used=similar_device_ids,
    )
