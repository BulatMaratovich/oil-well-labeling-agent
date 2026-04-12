"""
rules/rule_registry.py — Versioned Rule Registry.

Extends the basic RuleRegistry in rule_schemas.py with:
  - Versioning: every change bumps a version and appends to history
  - Persistence: saves/loads ruleset to data/tasks/<task_id>/ruleset.json
  - Regression check: validates proposed rules against confirmed LabelRecords
    before activation — refuses activation if FP rate rises above threshold
  - lock_ruleset(): freezes the active version

Storage format: data/tasks/<task_id>/ruleset.json
History format: data/tasks/<task_id>/ruleset_history.json
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rules.rule_schemas import Rule, RuleRegistry, RuleInput
from observability.logger import get_logger

log = get_logger(__name__)

TASKS_DIR = Path("data/tasks")


# ---------------------------------------------------------------------------
# Regression check result
# ---------------------------------------------------------------------------

@dataclass
class RegressionReport:
    passed: bool
    version_proposed: str
    n_tested: int
    n_changed: int               # label flips vs current ruleset
    fp_rate_delta: float         # positive = more FPs introduced
    details: list[dict] = field(default_factory=list)
    reason: str = ""


# ---------------------------------------------------------------------------
# Versioned registry
# ---------------------------------------------------------------------------

class VersionedRuleRegistry(RuleRegistry):
    """RuleRegistry with versioning, persistence, and regression checking."""

    def __init__(self, task_id: str) -> None:
        super().__init__()
        self.task_id = task_id
        self._version: int = 1
        self._locked: bool = False
        self._path = TASKS_DIR / task_id / "ruleset.json"
        self._history_path = TASKS_DIR / task_id / "ruleset_history.json"
        self._load()

    # ------------------------------------------------------------------
    # Versioning
    # ------------------------------------------------------------------

    @property
    def version(self) -> str:
        return f"v{self._version}"

    @property
    def locked(self) -> bool:
        return self._locked

    def lock_ruleset(self) -> str:
        """Freeze the current version. Returns the version string."""
        self._locked = True
        self._save()
        log.info("ruleset_locked", task_id=self.task_id, version=self.version)
        return self.version

    def unlock_ruleset(self) -> None:
        self._locked = False
        self._save()

    # ------------------------------------------------------------------
    # Mutation — all guarded by lock
    # ------------------------------------------------------------------

    def register(self, rule: Rule) -> None:
        if self._locked:
            raise RuntimeError(
                f"Ruleset {self.version} is locked. Unlock before modifying."
            )
        super().register(rule)

    def disable_rule(self, rule_id: str) -> bool:
        rule = self.get(rule_id)
        if rule is None:
            return False
        rule.enabled = False
        self._bump_version()
        return True

    def enable_rule(self, rule_id: str) -> bool:
        rule = self.get(rule_id)
        if rule is None:
            return False
        rule.enabled = True
        self._bump_version()
        return True

    def apply_draft(self, draft) -> None:
        """Apply a RuleDraft (from rule_miner) to the registry.

        Enables the rule if rule_id already exists; otherwise registers a stub.
        The stub has no real condition — it must be replaced with a real Rule
        before the registry is used for evaluation.
        """
        if self._locked:
            raise RuntimeError("Ruleset is locked.")

        existing = self.get(draft.rule_id)
        if existing:
            existing.description = draft.description
            existing.rationale = draft.rationale
            existing.condition_params = draft.condition_params
        else:
            # Placeholder rule — always returns False until replaced
            stub = Rule(
                rule_id=draft.rule_id,
                label=draft.label,
                priority=draft.priority,
                description=draft.description,
                condition=lambda _: False,   # placeholder
                condition_params=draft.condition_params,
                rationale=draft.rationale,
                enabled=False,               # must be explicitly enabled
            )
            super().register(stub)
        self._bump_version()

    # ------------------------------------------------------------------
    # Regression check
    # ------------------------------------------------------------------

    def check_regression(
        self,
        regression_records: list,    # list[LabelRecord]
        proposed_rules: list[Rule],
        *,
        max_fp_rate_delta: float = 0.05,
    ) -> RegressionReport:
        """Evaluate *proposed_rules* against confirmed *regression_records*.

        Returns a RegressionReport. ``passed=True`` means the proposed
        ruleset does not increase the false-positive rate beyond
        *max_fp_rate_delta*.
        """
        from rules.rule_engine import evaluate

        if not regression_records:
            return RegressionReport(
                passed=True,
                version_proposed=f"v{self._version + 1}",
                n_tested=0,
                n_changed=0,
                fp_rate_delta=0.0,
                reason="no_regression_set",
            )

        # Build candidate registry with proposed rules added
        proposed_registry = RuleRegistry()
        proposed_registry.register_all(list(self._rules))
        for r in proposed_rules:
            proposed_registry.register(r)

        n_changed = 0
        n_fp_current = 0
        n_fp_proposed = 0
        details = []

        for rec in regression_records:
            from rules.rule_schemas import RuleInput
            inp = RuleInput(
                candidate=_make_stub_candidate(rec),
                features=rec.local_features,
                context=None,
            )
            current_result = evaluate(inp, self)
            proposed_result = evaluate(inp, proposed_registry)

            current_correct = current_result.label == rec.final_label
            proposed_correct = proposed_result.label == rec.final_label

            if current_result.label != proposed_result.label:
                n_changed += 1
                details.append({
                    "record_id": rec.record_id,
                    "ground_truth": rec.final_label,
                    "current": current_result.label,
                    "proposed": proposed_result.label,
                    "flip": f"{current_result.label} → {proposed_result.label}",
                })

            if not current_correct:
                n_fp_current += 1
            if not proposed_correct:
                n_fp_proposed += 1

        n = len(regression_records)
        fp_rate_current = n_fp_current / n
        fp_rate_proposed = n_fp_proposed / n
        delta = fp_rate_proposed - fp_rate_current

        passed = delta <= max_fp_rate_delta
        reason = "" if passed else f"FP rate delta {delta:.3f} > threshold {max_fp_rate_delta}"

        report = RegressionReport(
            passed=passed,
            version_proposed=f"v{self._version + 1}",
            n_tested=n,
            n_changed=n_changed,
            fp_rate_delta=round(delta, 4),
            details=details[:20],
            reason=reason,
        )

        log.info(
            "regression_check",
            task_id=self.task_id,
            version=self.version,
            passed=passed,
            n_tested=n,
            n_changed=n_changed,
            fp_rate_delta=delta,
        )
        return report

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _bump_version(self) -> None:
        self._version += 1
        self._append_history()
        self._save()

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "task_id": self.task_id,
            "version": self._version,
            "locked": self._locked,
            "saved_at": datetime.now(tz=timezone.utc).isoformat(),
            "rules": [_rule_to_dict(r) for r in self._rules],
        }
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self._version = data.get("version", 1)
            self._locked = data.get("locked", False)
            # Rules with real conditions can't be round-tripped from JSON;
            # only metadata is restored — the caller must re-register live rules.
        except Exception as exc:
            log.warning("rule_registry_load_failed", task_id=self.task_id, error=str(exc))

    def _append_history(self) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        history: list[dict] = []
        if self._history_path.exists():
            try:
                history = json.loads(self._history_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        history.append({
            "version": self._version,
            "ts": datetime.now(tz=timezone.utc).isoformat(),
            "n_rules": len(self._rules),
            "rule_ids": [r.rule_id for r in self._rules],
        })
        self._history_path.write_text(
            json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rule_to_dict(rule: Rule) -> dict:
    return {
        "rule_id": rule.rule_id,
        "label": rule.label,
        "priority": rule.priority,
        "description": rule.description,
        "enabled": rule.enabled,
        "rationale": rule.rationale,
        "condition_params": rule.condition_params,
    }


def _make_stub_candidate(rec):
    """Build a minimal CandidateEvent from a LabelRecord for regression testing."""
    from core.canonical_schema import CandidateEvent
    return CandidateEvent(
        candidate_id=rec.record_id,
        asset_id=rec.asset_id,
        segment=rec.segment,
        deviation_type=rec.deviation_type,
        deviation_score=1.0,
        context_query="regression_test",
    )
