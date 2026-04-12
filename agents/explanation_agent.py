"""
agents/explanation_agent.py — Explanation Agent.

Generates a human-readable explanation (in Russian) for why the Rule Engine
proposed a particular label, using the LLM with the explanation system prompt.

Falls back to a deterministic template explanation when the LLM is unavailable.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from core.canonical_schema import ContextBundle, RuleResult

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "explanation_system.txt"
_SYSTEM_PROMPT: Optional[str] = None


def _system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def explain(
    rule_result: RuleResult,
    context: Optional[ContextBundle] = None,
    *,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> str:
    """Return a 3–4 sentence Russian explanation of *rule_result*.

    Parameters
    ----------
    rule_result:
        Output of the Rule Engine for one candidate.
    context:
        Optional context bundle (maintenance facts, equipment info).
    llm_client:
        Pre-built Mistral client.  When ``None`` returns a template string.
    model:
        Model override.
    timeout:
        HTTP timeout.
    """
    if llm_client is None:
        return _template_explanation(rule_result)

    payload = _build_payload(rule_result, context)
    try:
        return _call_llm(payload, llm_client=llm_client, model=model, timeout=timeout)
    except Exception as exc:
        logger.warning("explanation_agent LLM call failed: %s", exc)
        return _template_explanation(rule_result)


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _build_payload(rule_result: RuleResult, context: Optional[ContextBundle]) -> str:
    import json
    trace = rule_result.rule_trace
    payload = {
        "proposed_label": rule_result.label,
        "rule_trace": {
            "winning_rule": trace.winning_rule,
            "rules_fired": trace.rules_fired,
            "rules_blocked": trace.rules_blocked,
            "conflict": trace.conflict,
            "abstain_reason": trace.abstain_reason,
        },
        "context_bundle": _summarise_context(context),
    }
    return json.dumps(payload, ensure_ascii=False)


def _summarise_context(context: Optional[ContextBundle]) -> dict:
    if context is None:
        return {}
    facts = [
        {
            "event_type": f.event_type,
            "event_date": f.event_date.isoformat() if f.event_date else None,
            "action_summary": f.action_summary,
            "confidence": f.extraction_confidence,
        }
        for f in (context.maintenance_facts or [])
    ]
    return {
        "maintenance_facts": facts,
        "flags": context.flags,
    }


def _call_llm(user_text: str, *, llm_client, model: Optional[str], timeout: float) -> str:
    from app.config import settings

    resolved_model = model or settings.mistral_resolved_model
    response = llm_client.chat(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": user_text},
        ],
        timeout=timeout,
    )
    if hasattr(response, "choices"):
        return response.choices[0].message.content.strip()
    return response["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Template fallback
# ---------------------------------------------------------------------------

def _template_explanation(rule_result: RuleResult) -> str:
    trace = rule_result.rule_trace
    label = rule_result.label

    if trace.abstain_reason == "no_rule_matched":
        return (
            f"Ни одно из правил не дало однозначного ответа для этого кандидата. "
            f"Метка установлена как «{label}» по умолчанию. "
            f"Рекомендуется ручная проверка."
        )

    if trace.conflict:
        fired = ", ".join(trace.rules_fired)
        return (
            f"Для этого кандидата сработало несколько правил одного приоритета: {fired}. "
            f"Из-за конфликта автоматическая метка не определена (выставлено «{label}»). "
            f"Необходима ручная проверка."
        )

    if trace.winning_rule:
        return (
            f"Метка «{label}» присвоена по правилу «{trace.winning_rule}». "
            f"Всего оценено правил: {len(trace.rules_evaluated)}, "
            f"сработало: {len(trace.rules_fired)}."
        )

    return f"Метка «{label}» установлена автоматически."
