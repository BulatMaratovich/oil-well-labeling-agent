"""
core/policy_engine.py — Policy Engine.

Routes each (CandidateEvent, RuleResult) pair to one of three dispositions:

  "auto_label"       — high-confidence, rule fired cleanly → skip human review
  "review"           — normal human review required
  "mandatory_review" — conflict, abstain, or sensor issue → must be reviewed

Routing criteria are configured via ReviewPolicy from the TaskSpec.
The engine never writes to disk; it only produces RouteDecision objects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from core.canonical_schema import CandidateEvent, RuleResult
from core.task_manager import TaskSpec


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

@dataclass
class RouteDecision:
    candidate_id: str
    disposition: str          # "auto_label" | "review" | "mandatory_review"
    reason: str
    proposed_label: str
    confidence: float         # 0.0–1.0 heuristic


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

def route(
    candidate: CandidateEvent,
    rule_result: RuleResult,
    task_spec: TaskSpec,
) -> RouteDecision:
    """Determine how *candidate* should be handled after rule evaluation."""
    policy = task_spec.review_policy
    label = rule_result.label

    # 1. Conflict → mandatory review
    if rule_result.conflict_flag:
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            disposition="mandatory_review",
            reason="rule_conflict",
            proposed_label=label,
            confidence=0.0,
        )

    # 2. No rule matched → mandatory review
    if rule_result.abstain_reason == "no_rule_matched":
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            disposition="mandatory_review",
            reason="no_rule_matched",
            proposed_label=label,
            confidence=0.1,
        )

    # 3. Sensor issue (priority-1 rule) → mandatory review — data may be corrupt
    if label == "sensor_issue":
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            disposition="mandatory_review",
            reason="sensor_issue_detected",
            proposed_label=label,
            confidence=0.5,
        )

    # 4. Auto-label allowed by policy and confidence threshold met
    confidence = _estimate_confidence(candidate, rule_result)
    if (
        policy.auto_label_allowed
        and confidence >= 0.85
        and rule_result.rule_trace.winning_rule is not None
        and not rule_result.conflict_flag
    ):
        return RouteDecision(
            candidate_id=candidate.candidate_id,
            disposition="auto_label",
            reason=f"high_confidence_rule:{rule_result.rule_trace.winning_rule}",
            proposed_label=label,
            confidence=confidence,
        )

    # 5. Default → normal review
    return RouteDecision(
        candidate_id=candidate.candidate_id,
        disposition="review",
        reason=f"rule:{rule_result.rule_trace.winning_rule or 'none'}",
        proposed_label=label,
        confidence=confidence,
    )


def route_batch(
    candidates: list[CandidateEvent],
    rule_results: list[RuleResult],
    task_spec: TaskSpec,
) -> list[RouteDecision]:
    """Route a list of candidates; lengths must match."""
    return [
        route(c, r, task_spec)
        for c, r in zip(candidates, rule_results)
    ]


# ---------------------------------------------------------------------------
# Confidence heuristic
# ---------------------------------------------------------------------------

def _estimate_confidence(
    candidate: CandidateEvent,
    rule_result: RuleResult,
) -> float:
    """Simple heuristic: higher deviation score + clean rule trace → higher confidence."""
    base = 0.5

    # Winning rule without conflict → boost
    if rule_result.rule_trace.winning_rule and not rule_result.conflict_flag:
        base += 0.25

    # High deviation score → more certain something happened
    score = min(candidate.deviation_score, 5.0)
    base += score / 5.0 * 0.20

    # Abstain or conflict → penalise
    if rule_result.abstain_reason:
        base -= 0.30
    if rule_result.conflict_flag:
        base -= 0.40

    return round(max(0.0, min(1.0, base)), 3)
