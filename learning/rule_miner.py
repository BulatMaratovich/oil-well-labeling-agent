"""
learning/rule_miner.py — Rule Miner.

Analyses correction patterns from TaskMemory and proposes new or updated
rules using the LLM (rule_draft system prompt).

When the LLM is unavailable, returns a heuristic summary of patterns
without generating rule drafts.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from learning.task_memory import TaskMemory

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "rule_draft_system.txt"
_SYSTEM_PROMPT: Optional[str] = None

# Minimum number of same-type corrections before the miner fires
MIN_PATTERN_COUNT = 3


def _system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RuleDraft:
    action: str                          # "add_rule" | "modify_rule"
    rule_id: str
    priority: int
    label: str
    description: str
    condition_params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    raw_llm_json: Optional[str] = None   # original LLM output for audit


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def mine(
    memory: TaskMemory,
    *,
    existing_rules: Optional[list[dict]] = None,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 30.0,
    min_pattern_count: int = MIN_PATTERN_COUNT,
) -> list[RuleDraft]:
    """Analyse *memory* and return a list of rule draft proposals.

    Parameters
    ----------
    memory:
        TaskMemory for the current task.
    existing_rules:
        Summary of current rules (list of dicts with rule_id, label, priority).
        Passed to the LLM so it avoids duplicating them.
    llm_client:
        Mistral client.  When ``None`` returns heuristic patterns only.
    min_pattern_count:
        Minimum correction count for a pattern to trigger a draft.
    """
    patterns = memory.correction_patterns()
    significant = [p for p in patterns if p["count"] >= min_pattern_count]

    if not significant:
        return []

    if llm_client is None:
        return _heuristic_drafts(significant)

    payload = _build_payload(
        patterns=significant,
        corrections=memory.corrections(),
        existing_rules=existing_rules or [],
    )

    drafts: list[RuleDraft] = []
    for pattern in significant:
        try:
            draft = _call_llm_for_pattern(
                pattern=pattern,
                global_payload=payload,
                llm_client=llm_client,
                model=model,
                timeout=timeout,
            )
            if draft:
                drafts.append(draft)
        except Exception as exc:
            logger.warning("rule_miner LLM call failed for pattern %s: %s", pattern, exc)

    return drafts


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _build_payload(patterns, corrections, existing_rules) -> dict:
    """Build the context dict sent to the LLM."""
    sample_corrections = [
        {
            "rule_suggested": c.rule_result.label,
            "engineer_label": c.final_label,
            "reason": c.correction_reason,
            "deviation_type": c.deviation_type,
            "features": {
                "power_mean": c.local_features.power_mean if c.local_features else None,
                "zero_fraction": c.local_features.zero_fraction if c.local_features else None,
                "transition_sharpness": c.local_features.transition_sharpness if c.local_features else None,
            },
        }
        for c in corrections[:20]  # limit context size
    ]
    return {
        "patterns": patterns,
        "sample_corrections": sample_corrections,
        "existing_rules": existing_rules,
    }


def _call_llm_for_pattern(
    *,
    pattern: dict,
    global_payload: dict,
    llm_client,
    model: Optional[str],
    timeout: float,
) -> Optional[RuleDraft]:
    from app.config import settings

    resolved_model = model or settings.mistral_resolved_model
    user_text = json.dumps(
        {**global_payload, "target_pattern": pattern},
        ensure_ascii=False,
        indent=2,
    )

    response = llm_client.chat(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_text},
        ],
        timeout=timeout,
    )
    if hasattr(response, "choices"):
        raw = response.choices[0].message.content.strip()
    else:
        raw = response["choices"][0]["message"]["content"].strip()

    return _parse_draft(raw)


def _parse_draft(raw: str) -> Optional[RuleDraft]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())

    priority = data.get("priority", 4)
    try:
        priority = int(priority)
        priority = max(1, min(4, priority))
    except (TypeError, ValueError):
        priority = 4

    return RuleDraft(
        action=data.get("action", "add_rule"),
        rule_id=data.get("rule_id", "mined_rule"),
        priority=priority,
        label=data.get("label", "unknown"),
        description=data.get("description", ""),
        condition_params=data.get("condition_params") or {},
        rationale=data.get("rationale", ""),
        raw_llm_json=raw,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback
# ---------------------------------------------------------------------------

def _heuristic_drafts(patterns: list[dict]) -> list[RuleDraft]:
    """Return simple descriptive drafts without LLM."""
    drafts = []
    for p in patterns:
        rule_id = f"mined_{p['rule_suggested']}_to_{p['engineer_label']}".replace(" ", "_")
        drafts.append(RuleDraft(
            action="add_rule",
            rule_id=rule_id,
            priority=4,
            label=p["engineer_label"],
            description=(
                f"Инженер {p['count']} раз заменял «{p['rule_suggested']}» "
                f"на «{p['engineer_label']}»."
            ),
            rationale=(
                f"Паттерн коррекций: "
                + ("; ".join(p["correction_reasons"][:3]) or "причина не указана")
            ),
        ))
    return drafts
