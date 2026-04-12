"""
rules/rule_schemas.py — Rule dataclasses.

A Rule is a pure declarative object: it describes *what* to check and *what*
label to assign when the condition fires.  All evaluation logic lives in
rule_engine.py.

Priority tiers (from documentation):
  1 — sensor / data-quality exclusions   (highest precedence)
  2 — confounder exclusions              (planned stops, maintenance, …)
  3 — stable unusual regime              (persistent shift, not a failure)
  4 — true deviations / belt-break class (lowest precedence, default tier)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from core.canonical_schema import CandidateEvent, ContextBundle, LocalFeatures


# ---------------------------------------------------------------------------
# Rule input bundle  (everything a rule condition can inspect)
# ---------------------------------------------------------------------------

@dataclass
class RuleInput:
    candidate: CandidateEvent
    features: Optional[LocalFeatures]
    context: Optional[ContextBundle]
    task_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Rule definition
# ---------------------------------------------------------------------------

# A condition is a callable: (RuleInput) -> bool
ConditionFn = Callable[["RuleInput"], bool]


@dataclass
class Rule:
    rule_id: str
    label: str                      # target label from label_taxonomy
    priority: int                   # 1–4 (lower = higher precedence)
    description: str
    condition: ConditionFn          # predicate — must NOT raise
    condition_params: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    # Human-readable rationale (used by explanation agent)
    rationale: str = ""

    def matches(self, inp: RuleInput) -> bool:
        """Evaluate the condition safely; return False on any exception."""
        if not self.enabled:
            return False
        try:
            return bool(self.condition(inp))
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Registry helper
# ---------------------------------------------------------------------------

class RuleRegistry:
    """Ordered container for Rule objects.

    Rules are stored sorted by (priority, rule_id) so iteration always
    visits higher-priority rules first.
    """

    def __init__(self) -> None:
        self._rules: list[Rule] = []

    def register(self, rule: Rule) -> None:
        self._rules.append(rule)
        self._rules.sort(key=lambda r: (r.priority, r.rule_id))

    def register_all(self, rules: list[Rule]) -> None:
        for r in rules:
            self.register(r)

    def get(self, rule_id: str) -> Optional[Rule]:
        return next((r for r in self._rules if r.rule_id == rule_id), None)

    def all_enabled(self) -> list[Rule]:
        return [r for r in self._rules if r.enabled]

    def __len__(self) -> int:
        return len(self._rules)

    def __iter__(self):
        return iter(self._rules)
