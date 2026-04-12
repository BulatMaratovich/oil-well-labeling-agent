"""
context/context_fact_extractor.py — Stage 7: Context Fact Extractor.

Sends raw maintenance document text to the LLM (Mistral) and parses the
structured StructuredFacts JSON response.

Uses the prompt at config/prompts/fact_extraction_system.txt.
Falls back gracefully when the LLM is unavailable or returns unparseable output.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.canonical_schema import MaintenanceDocument, StructuredFacts

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parent.parent / "config" / "prompts" / "fact_extraction_system.txt"
_SYSTEM_PROMPT: Optional[str] = None


def _system_prompt() -> str:
    global _SYSTEM_PROMPT
    if _SYSTEM_PROMPT is None:
        _SYSTEM_PROMPT = _PROMPT_PATH.read_text(encoding="utf-8")
    return _SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_facts(
    doc: MaintenanceDocument,
    *,
    llm_client=None,  # injected; None → rule-based fallback only
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> StructuredFacts:
    """Extract structured facts from *doc*.

    Parameters
    ----------
    doc:
        Raw maintenance document.
    llm_client:
        Optional pre-built Mistral client (``MistralClient``).  When ``None``
        the function returns a low-confidence stub.
    model:
        Model name override.  Falls back to app config default.
    timeout:
        HTTP timeout in seconds.
    """
    if llm_client is None:
        return _fallback_facts(doc)

    try:
        raw_json = _call_llm(doc.raw_text, llm_client=llm_client, model=model, timeout=timeout)
        return _parse_response(raw_json, doc)
    except Exception as exc:
        logger.warning("fact_extraction failed for doc %s: %s", doc.doc_id, exc)
        facts = _fallback_facts(doc)
        facts.extraction_confidence = "failed"
        return facts


def extract_facts_batch(
    docs: list[MaintenanceDocument],
    *,
    llm_client=None,
    model: Optional[str] = None,
    timeout: float = 30.0,
) -> list[StructuredFacts]:
    return [
        extract_facts(doc, llm_client=llm_client, model=model, timeout=timeout)
        for doc in docs
    ]


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def _call_llm(text: str, *, llm_client, model: Optional[str], timeout: float) -> str:
    from app.config import settings

    resolved_model = model or settings.mistral_resolved_model

    response = llm_client.chat(
        model=resolved_model,
        messages=[
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": text},
        ],
        timeout=timeout,
    )
    # Support both dict-style and object-style responses
    if hasattr(response, "choices"):
        return response.choices[0].message.content
    return response["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def _parse_response(raw: str, doc: MaintenanceDocument) -> StructuredFacts:
    text = raw.strip()
    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]

    data = json.loads(text)

    event_date: Optional[datetime] = None
    if data.get("event_date"):
        try:
            event_date = datetime.fromisoformat(str(data["event_date"]))
        except ValueError:
            pass

    duration_h: Optional[float] = None
    if data.get("duration_h") is not None:
        try:
            duration_h = float(data["duration_h"])
        except (TypeError, ValueError):
            pass

    parts = data.get("parts_replaced") or []
    if not isinstance(parts, list):
        parts = []

    confidence = data.get("extraction_confidence", "ok")
    if confidence not in ("ok", "low", "failed"):
        confidence = "low"

    return StructuredFacts(
        doc_id=doc.doc_id,
        event_type=data.get("event_type") or None,
        event_date=event_date,
        asset_id=data.get("asset_id") or doc.asset_id,
        duration_h=duration_h,
        action_summary=data.get("action_summary") or None,
        parts_replaced=[str(p) for p in parts if p],
        extraction_confidence=confidence,
    )


# ---------------------------------------------------------------------------
# Fallback (no LLM available)
# ---------------------------------------------------------------------------

def _fallback_facts(doc: MaintenanceDocument) -> StructuredFacts:
    """Return a minimal StructuredFacts when the LLM is unavailable."""
    return StructuredFacts(
        doc_id=doc.doc_id,
        event_type=None,
        event_date=doc.event_date,
        asset_id=doc.asset_id,
        extraction_confidence="low",
    )
