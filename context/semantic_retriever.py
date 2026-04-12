"""
context/semantic_retriever.py — File-based semantic retriever.

Implements retriever.md spec WITHOUT a database:
  - Documents are stored as JSON lines in data/tasks/<task_id>/docs.jsonl
  - Similarity is computed with TF-IDF cosine (scikit-learn)
  - Index is rebuilt on first query after new documents are added

Provides:
  SemanticRetriever.add(docs)      — index documents
  SemanticRetriever.query(text, k) — return top-k RuleDocument / StructuredFacts
  SemanticRetriever.save() / load()
"""
from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

from core.canonical_schema import RuleDocument, StructuredFacts
from observability.logger import get_logger

log = get_logger(__name__)

TASKS_DIR = Path("data/tasks")


# ---------------------------------------------------------------------------
# Generic document wrapper
# ---------------------------------------------------------------------------

@dataclass
class IndexedDoc:
    doc_id: str
    text: str           # searchable text
    doc_type: str       # "rule" | "fact" | "maintenance"
    payload: dict       # original document dict


# ---------------------------------------------------------------------------
# TF-IDF helpers (no sklearn required — pure Python fallback)
# ---------------------------------------------------------------------------

def _tokenise(text: str) -> list[str]:
    import re
    return re.findall(r"[а-яёa-z0-9]+", text.lower())


def _tf(tokens: list[str]) -> dict[str, float]:
    counts = Counter(tokens)
    n = max(len(tokens), 1)
    return {t: c / n for t, c in counts.items()}


class _TfidfIndex:
    def __init__(self) -> None:
        self._docs: list[IndexedDoc] = []
        self._tfs: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._dirty = True

    def add(self, docs: list[IndexedDoc]) -> None:
        self._docs.extend(docs)
        self._tfs.extend([_tf(_tokenise(d.text)) for d in docs])
        self._dirty = True

    def _build_idf(self) -> None:
        n = len(self._docs)
        if n == 0:
            self._idf = {}
            return
        df: Counter = Counter()
        for tf in self._tfs:
            for term in tf:
                df[term] += 1
        self._idf = {
            term: math.log((n + 1) / (count + 1)) + 1
            for term, count in df.items()
        }
        self._dirty = False

    def _vec(self, tf: dict[str, float]) -> dict[str, float]:
        return {t: v * self._idf.get(t, 1.0) for t, v in tf.items()}

    def _cosine(self, a: dict[str, float], b: dict[str, float]) -> float:
        shared = set(a) & set(b)
        if not shared:
            return 0.0
        dot = sum(a[t] * b[t] for t in shared)
        norm_a = math.sqrt(sum(v * v for v in a.values()))
        norm_b = math.sqrt(sum(v * v for v in b.values()))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def query(self, text: str, k: int = 5) -> list[tuple[IndexedDoc, float]]:
        if not self._docs:
            return []
        if self._dirty:
            self._build_idf()
        q_vec = self._vec(_tf(_tokenise(text)))
        scored = [
            (doc, self._cosine(q_vec, self._vec(tf)))
            for doc, tf in zip(self._docs, self._tfs)
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(doc, score) for doc, score in scored[:k] if score > 0]

    def __len__(self) -> int:
        return len(self._docs)


# ---------------------------------------------------------------------------
# SemanticRetriever
# ---------------------------------------------------------------------------

class SemanticRetriever:
    """File-backed semantic retriever for one task."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._path = TASKS_DIR / task_id / "docs.jsonl"
        self._index = _TfidfIndex()
        self._load()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def add_rule_docs(self, docs: list[RuleDocument]) -> None:
        indexed = [
            IndexedDoc(
                doc_id=d.rule_id,
                text=d.embedding_text or d.description,
                doc_type="rule",
                payload=asdict(d),
            )
            for d in docs
        ]
        self._index.add(indexed)
        self._save_append(indexed)
        log.info("retriever_indexed", task_id=self.task_id,
                 n=len(indexed), doc_type="rule")

    def add_fact_docs(self, facts: list[StructuredFacts]) -> None:
        indexed = [
            IndexedDoc(
                doc_id=f.doc_id,
                text=_fact_to_text(f),
                doc_type="fact",
                payload=_fact_to_dict(f),
            )
            for f in facts
        ]
        self._index.add(indexed)
        self._save_append(indexed)
        log.info("retriever_indexed", task_id=self.task_id,
                 n=len(indexed), doc_type="fact")

    def add_raw(self, docs: list[IndexedDoc]) -> None:
        self._index.add(docs)
        self._save_append(docs)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        text: str,
        k: int = 5,
        doc_type: Optional[str] = None,
    ) -> list[tuple[IndexedDoc, float]]:
        """Return top-k (doc, score) pairs for *text*.

        Parameters
        ----------
        text:   Query string (natural language or structured fact text).
        k:      Max results.
        doc_type: Filter to "rule" | "fact" | "maintenance" | None (all).
        """
        results = self._index.query(text, k=k * 3)  # over-fetch then filter
        if doc_type:
            results = [(d, s) for d, s in results if d.doc_type == doc_type]
        return results[:k]

    def query_rules(self, text: str, k: int = 3) -> list[RuleDocument]:
        results = self.query(text, k=k, doc_type="rule")
        out = []
        for doc, _ in results:
            p = doc.payload
            out.append(RuleDocument(
                rule_id=p.get("rule_id", doc.doc_id),
                description=p.get("description", ""),
                label=p.get("label", "unknown"),
                embedding_text=p.get("embedding_text", doc.text),
            ))
        return out

    def __len__(self) -> int:
        return len(self._index)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save_append(self, docs: list[IndexedDoc]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            for doc in docs:
                fh.write(json.dumps(asdict(doc), ensure_ascii=False) + "\n")

    def _load(self) -> None:
        if not self._path.exists():
            return
        docs: list[IndexedDoc] = []
        try:
            for line in self._path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                docs.append(IndexedDoc(**d))
            self._index.add(docs)
            log.info("retriever_loaded", task_id=self.task_id, n=len(docs))
        except Exception as exc:
            log.warning("retriever_load_failed", task_id=self.task_id, error=str(exc))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fact_to_text(f: StructuredFacts) -> str:
    parts = []
    if f.event_type:
        parts.append(f.event_type)
    if f.action_summary:
        parts.append(f.action_summary)
    if f.parts_replaced:
        parts.append(" ".join(f.parts_replaced))
    if f.asset_id:
        parts.append(f.asset_id)
    return " ".join(parts) or f.doc_id


def _fact_to_dict(f: StructuredFacts) -> dict:
    return {
        "doc_id": f.doc_id,
        "event_type": f.event_type,
        "event_date": f.event_date.isoformat() if f.event_date else None,
        "asset_id": f.asset_id,
        "action_summary": f.action_summary,
        "parts_replaced": f.parts_replaced,
        "extraction_confidence": f.extraction_confidence,
    }
