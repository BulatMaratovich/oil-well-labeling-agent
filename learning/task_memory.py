"""
learning/task_memory.py — TaskMemory.

Persists human-reviewed LabelRecords to disk, organised by task_id.
Provides queries needed by the rule miner and explanation agent:
  - confirmed examples  (was_override=False or engineer agreed)
  - rejected/corrected  (was_override=True)
  - ambiguous           (status="ambiguous")
  - regression set      (high-confidence confirmed examples used to test rule changes)

Storage: one JSON file per task at  data/tasks/<task_id>/memory.json
Each entry is a flat dict serialisation of LabelRecord.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

from core.canonical_schema import LabelRecord, RuleResult, RuleTrace

logger = logging.getLogger(__name__)

TASKS_DIR = Path("data/tasks")


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _iso(dt: Optional[datetime]) -> Optional[str]:
    return dt.isoformat() if dt else None


def _record_to_dict(rec: LabelRecord) -> dict:
    d = {
        "record_id": rec.record_id,
        "task_id": rec.task_id,
        "asset_id": rec.asset_id,
        "segment_start": _iso(rec.segment.start),
        "segment_end": _iso(rec.segment.end),
        "deviation_type": rec.deviation_type,
        "final_label": rec.final_label,
        "was_override": rec.was_override,
        "correction_reason": rec.correction_reason,
        "confirmed_at": _iso(rec.confirmed_at),
        "run_id": rec.run_id,
        "status": rec.status,
        # rule result summary
        "rule_label": rec.rule_result.label,
        "winning_rule": rec.rule_result.rule_trace.winning_rule,
        "rules_fired": rec.rule_result.rule_trace.rules_fired,
        "conflict": rec.rule_result.rule_trace.conflict,
        "abstain_reason": rec.rule_result.abstain_reason,
        # local features (optional)
        "features": asdict(rec.local_features) if rec.local_features else None,
    }
    return d


def _dict_to_record(d: dict) -> LabelRecord:
    from core.canonical_schema import DateRange, LocalFeatures

    def _dt(v):
        if not v:
            return datetime.now(tz=timezone.utc)
        return datetime.fromisoformat(v)

    segment = DateRange(start=_dt(d.get("segment_start")), end=_dt(d.get("segment_end")))

    feats = None
    if d.get("features"):
        fdict = d["features"]
        feats = LocalFeatures(**{k: v for k, v in fdict.items() if k in LocalFeatures.__dataclass_fields__})

    rule_result = RuleResult(
        label=d.get("rule_label", "unknown"),
        rule_trace=RuleTrace(
            winning_rule=d.get("winning_rule"),
            rules_fired=d.get("rules_fired") or [],
            conflict=d.get("conflict", False),
            abstain_reason=d.get("abstain_reason"),
        ),
        abstain_reason=d.get("abstain_reason"),
        conflict_flag=d.get("conflict", False),
    )

    return LabelRecord(
        record_id=d.get("record_id", uuid4().hex),
        task_id=d["task_id"],
        asset_id=d["asset_id"],
        segment=segment,
        deviation_type=d.get("deviation_type", "unknown"),
        local_features=feats,
        rule_result=rule_result,
        final_label=d["final_label"],
        was_override=d.get("was_override", False),
        correction_reason=d.get("correction_reason"),
        confirmed_at=_dt(d.get("confirmed_at")),
        run_id=d.get("run_id"),
        status=d.get("status", "accepted"),
    )


# ---------------------------------------------------------------------------
# TaskMemory
# ---------------------------------------------------------------------------

class TaskMemory:
    """Read/write store for reviewed LabelRecords for one task."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._path = TASKS_DIR / task_id / "memory.json"
        self._records: list[LabelRecord] = []
        self._load()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def add(self, record: LabelRecord) -> None:
        """Append a record and persist."""
        # Deduplicate by record_id
        self._records = [r for r in self._records if r.record_id != record.record_id]
        self._records.append(record)
        self._save()

    def add_batch(self, records: list[LabelRecord]) -> None:
        for rec in records:
            self._records = [r for r in self._records if r.record_id != rec.record_id]
            self._records.append(rec)
        self._save()

    def mark_status(self, record_id: str, status: str) -> bool:
        for rec in self._records:
            if rec.record_id == record_id:
                rec.status = status
                self._save()
                return True
        return False

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def all(self) -> list[LabelRecord]:
        return list(self._records)

    def confirmed(self) -> list[LabelRecord]:
        return [r for r in self._records if r.status == "accepted" and not r.was_override]

    def corrections(self) -> list[LabelRecord]:
        """Records where the engineer changed the rule suggestion."""
        return [r for r in self._records if r.was_override]

    def ambiguous(self) -> list[LabelRecord]:
        return [r for r in self._records if r.status == "ambiguous"]

    def regression_set(self, min_count: int = 3) -> list[LabelRecord]:
        """High-confidence confirmed records — used to test rule changes."""
        by_label: dict[str, list[LabelRecord]] = {}
        for rec in self.confirmed():
            by_label.setdefault(rec.final_label, []).append(rec)
        result = []
        for recs in by_label.values():
            if len(recs) >= min_count:
                result.extend(recs)
        return result

    def by_label(self, label: str) -> list[LabelRecord]:
        return [r for r in self._records if r.final_label == label]

    def correction_patterns(self) -> list[dict]:
        """Summarise correction patterns for the rule miner.

        Returns a list of dicts:
          {rule_suggested, engineer_label, count, correction_reasons}
        """
        from collections import Counter
        counts: Counter = Counter()
        reasons: dict[tuple, list] = {}
        for rec in self.corrections():
            key = (rec.rule_result.label, rec.final_label)
            counts[key] += 1
            reasons.setdefault(key, [])
            if rec.correction_reason:
                reasons[key].append(rec.correction_reason)
        return [
            {
                "rule_suggested": k[0],
                "engineer_label": k[1],
                "count": v,
                "correction_reasons": reasons.get(k, []),
            }
            for k, v in counts.most_common()
        ]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            self._records = [_dict_to_record(d) for d in raw]
        except Exception as exc:
            logger.warning("TaskMemory load failed (%s): %s", self._path, exc)
            self._records = []

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = [_record_to_dict(r) for r in self._records]
        self._path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
