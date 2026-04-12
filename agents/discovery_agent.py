"""
agents/discovery_agent.py — Discovery Agent.

Conducts a short dialogue with the engineer before labeling starts.
Extracts TaskSpec fields from the conversation using the LLM.

Two responsibilities:
  1. chat()  — send one user turn, get assistant reply (streaming-safe)
  2. extract_task_updates() — parse the full conversation history and
     return a dict of TaskSpec field updates

Uses config/prompts/discovery_system.txt as the system prompt.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "discovery_system.txt"
_SYSTEM_PROMPT: Optional[str] = None

# Fields that the discovery agent may update on TaskSpec
DISCOVERY_FIELDS = (
    "equipment_family",
    "primary_deviation",
    "normal_operation_definition",
    "expected_deviation_frequency",
    "statistical_threshold_pct",
    "confounders",
    "context_sources",
    "minimum_segment_duration",
)

# Extraction instruction appended when we ask the LLM to parse the conversation
_EXTRACT_SUFFIX = """

Дополнительно: проанализируй весь диалог выше и верни JSON-объект с полями TaskSpec,
которые ты смог определить из разговора. Включай только поля, которые явно упомянуты.
Схема:
{
  "equipment_family": "строка или null",
  "primary_deviation": "строка или null",
  "normal_operation_definition": "строка или null",
  "expected_deviation_frequency": "строка или null",
  "statistical_threshold_pct": число или null,
  "confounders": ["список"] или null,
  "minimum_segment_duration": число_секунд или null
}
Верни только JSON, без пояснений.
"""


def _system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Message history type
# ---------------------------------------------------------------------------

Message = dict[str, str]  # {"role": "user"|"assistant", "content": "..."}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def chat(
    history: list[Message],
    user_message: str,
    *,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> tuple[str, list[Message]]:
    """Send *user_message*, get assistant reply, return updated history.

    Parameters
    ----------
    history:
        Conversation history so far (mutated copy is returned).
    user_message:
        Latest user input.
    llm_client:
        Mistral client.  When ``None`` returns a static fallback reply.
    model / timeout:
        LLM parameters.

    Returns
    -------
    (assistant_reply, updated_history)
    """
    updated = list(history) + [{"role": "user", "content": user_message}]

    if llm_client is None:
        reply = _fallback_reply(updated)
        updated.append({"role": "assistant", "content": reply})
        return reply, updated

    try:
        reply = _call_llm(updated, llm_client=llm_client, model=model, timeout=timeout)
    except Exception as exc:
        logger.warning("discovery_agent chat LLM error: %s", exc)
        reply = _fallback_reply(updated)

    updated.append({"role": "assistant", "content": reply})
    return reply, updated


def extract_task_updates(
    history: list[Message],
    *,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Parse the conversation *history* and extract TaskSpec field updates.

    Returns an empty dict when nothing could be extracted or LLM unavailable.
    """
    if not history:
        return {}

    if llm_client is None:
        return {}

    extraction_history = list(history) + [
        {"role": "user", "content": _EXTRACT_SUFFIX}
    ]

    try:
        raw = _call_llm(extraction_history, llm_client=llm_client, model=model, timeout=timeout)
        return _parse_updates(raw)
    except Exception as exc:
        logger.warning("discovery_agent extract_task_updates error: %s", exc)
        return {}


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(
    history: list[Message],
    *,
    llm_client,
    model: Optional[str],
    timeout: float,
) -> str:
    from app.config import settings

    resolved_model = model or settings.mistral_resolved_model
    messages = [{"role": "system", "content": _system_prompt()}] + history

    response = llm_client.chat(
        model=resolved_model,
        messages=messages,
        timeout=timeout,
    )
    if hasattr(response, "choices"):
        return response.choices[0].message.content.strip()
    return response["choices"][0]["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _parse_updates(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    data = json.loads(text.strip())
    # Keep only known fields, drop nulls
    return {k: v for k, v in data.items() if k in DISCOVERY_FIELDS and v is not None}


# ---------------------------------------------------------------------------
# Fallback (no LLM)
# ---------------------------------------------------------------------------

def _fallback_reply(history: list[Message]) -> str:
    turn = len([m for m in history if m["role"] == "user"])
    if turn <= 1:
        return (
            "Привет! Загрузите файл с данными скважины. "
            "Какой тип отклонения вас интересует — обрыв ремня, остановы или другое?"
        )
    return "Хорошо, параметры приняты. Можно переходить к построению кандидатов."
