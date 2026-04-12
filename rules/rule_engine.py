"""
rules/rule_engine.py — Stage 8: Rule Engine.

Evaluates a RuleRegistry against a RuleInput and produces a RuleResult
with a full RuleTrace.

Algorithm
---------
1. Iterate rules in priority order (1 → 4).
2. Collect all *firing* rules per priority tier.
3. The first tier that has at least one firing rule determines the outcome:
   - If exactly one rule fires in that tier → it wins, label assigned.
   - If multiple rules fire in that tier  → conflict flag set, label = "unknown".
4. If no rule fires across all tiers → abstain with "no_rule_matched".
"""
from __future__ import annotations

from core.canonical_schema import RuleResult, RuleTrace
from rules.rule_schemas import RuleInput, RuleRegistry


def evaluate(
    inp: RuleInput,
    registry: RuleRegistry,
    unknown_label: str = "unknown",
) -> RuleResult:
    """Run all enabled rules against *inp* and return a :class:`RuleResult`."""
    enabled = registry.all_enabled()

    trace = RuleTrace(
        rules_evaluated=[r.rule_id for r in enabled],
    )

    # Group firing rules by priority tier
    fired_by_priority: dict[int, list[str]] = {}
    for rule in enabled:
        if rule.matches(inp):
            fired_by_priority.setdefault(rule.priority, []).append(rule.rule_id)
            trace.rules_fired.append(rule.rule_id)
        else:
            trace.rules_blocked.append(rule.rule_id)

    if not fired_by_priority:
        trace.abstain_reason = "no_rule_matched"
        return RuleResult(
            label=unknown_label,
            rule_trace=trace,
            abstain_reason="no_rule_matched",
        )

    # Resolve by lowest (highest-precedence) priority tier
    winning_tier = min(fired_by_priority)
    winners = fired_by_priority[winning_tier]

    if len(winners) == 1:
        winning_rule_id = winners[0]
        winning_rule = registry.get(winning_rule_id)
        trace.winning_rule = winning_rule_id
        return RuleResult(
            label=winning_rule.label if winning_rule else unknown_label,
            rule_trace=trace,
        )

    # Conflict within tier
    trace.conflict = True
    trace.winning_rule = None
    trace.abstain_reason = f"rule_conflict_tier_{winning_tier}"
    return RuleResult(
        label=unknown_label,
        rule_trace=trace,
        abstain_reason=trace.abstain_reason,
        conflict_flag=True,
    )
