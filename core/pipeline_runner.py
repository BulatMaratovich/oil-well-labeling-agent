"""
core/pipeline_runner.py — Pipeline Runner.

Orchestrates the full 10-stage labeling pipeline for one asset:

  Stage 1  input_normalizer      → list[CanonicalTimeSeries]
  Stage 2  signal_sanitizer      → (CanonicalTimeSeries, QualityFlags)
  Stage 3  global_series_profiler→ RegimeSequence
  Stage 4  historical_profile_builder → WellProfile
  Stage 5  candidate_event_detector  → list[CandidateEvent]
  Stage 6  local_segment_analyzer    → list[LocalFeatures]
  Stage 7  context_fact_extractor    → list[StructuredFacts]  (optional)
  Stage 8  rule_engine               → list[RuleResult]
  Stage 9  (human review — handled by the web UI, not here)
  Stage 10 task_memory.add()         → persist LabelRecords

Returns a PipelineResult with all intermediate artefacts for the UI /
export layer to consume.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Optional
from pathlib import Path

from core.canonical_schema import (
    CandidateEvent,
    CanonicalTimeSeries,
    ContextBundle,
    LabelRecord,
    LocalFeatures,
    MaintenanceDocument,
    QualityFlags,
    RegimeSequence,
    RuleResult,
    StructuredFacts,
    WellProfile,
)
from core.task_manager import TaskSpec
from observability.logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    run_id: str
    task_id: str
    asset_id: str

    # Per-signal artefacts (one entry per selected signal)
    series: list[CanonicalTimeSeries] = field(default_factory=list)
    quality_flags: list[QualityFlags] = field(default_factory=list)
    regime_sequences: list[RegimeSequence] = field(default_factory=list)
    well_profile: Optional[WellProfile] = None

    # Candidate events (merged across all signals)
    candidates: list[CandidateEvent] = field(default_factory=list)
    local_features: list[LocalFeatures] = field(default_factory=list)

    # Context (optional)
    context_bundles: list[ContextBundle] = field(default_factory=list)
    maintenance_facts: list[StructuredFacts] = field(default_factory=list)

    # Rule engine outputs (same order as candidates)
    rule_results: list[RuleResult] = field(default_factory=list)

    # Errors / warnings accumulated during the run
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.errors)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class PipelineRunner:
    """Stateless orchestrator — create once, call run() repeatedly."""

    def __init__(
        self,
        task_spec: TaskSpec,
        *,
        rule_registry=None,    # rules.rule_schemas.RuleRegistry
        llm_client=None,       # Mistral client, optional
        llm_model: Optional[str] = None,
        existing_profile: Optional[WellProfile] = None,
    ) -> None:
        self.task_spec = task_spec
        self.rule_registry = rule_registry or _default_registry()
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.existing_profile = existing_profile

    def run(
        self,
        file_input,                   # Path | str | bytes
        asset_id: Optional[str] = None,
        filename: str = "data.csv",
        maintenance_docs: Optional[list] = None,
    ) -> PipelineResult:
        """Execute stages 1–8 for one asset.

        Parameters
        ----------
        file_input:
            Raw file bytes or path.
        asset_id:
            Well / asset identifier.  Required for multi-asset files.
        filename:
            Original filename (used for format detection when bytes).
        maintenance_docs:
            Optional list of MaintenanceDocument for context extraction.
        """
        run_id = uuid.uuid4().hex[:12]
        result = PipelineResult(
            run_id=run_id,
            task_id=self.task_spec.task_id,
            asset_id=asset_id or "unknown",
        )

        log.info("pipeline_start", run_id=run_id, task_id=self.task_spec.task_id,
                 asset_id=asset_id, filename=filename)

        # ── Stage 1: Input Normalizer ─────────────────────────────────
        try:
            from signals.input_normalizer import normalize
            all_series = normalize(
                file_input, self.task_spec,
                filename=filename, asset_id=asset_id,
            )
            result.series = all_series
            if all_series:
                result.asset_id = all_series[0].asset_id
            log.info("stage_complete", run_id=run_id, stage="input_normalizer",
                     n_series=len(all_series))
        except Exception as exc:
            result.errors.append(f"input_normalizer: {exc}")
            log.error("stage_failed", run_id=run_id, stage="input_normalizer", error=str(exc))
            return result

        # ── Stage 2: Signal Sanitizer ─────────────────────────────────
        clean_series: list[CanonicalTimeSeries] = []
        for s in result.series:
            try:
                from signals.signal_sanitizer import sanitize, SanitizationError
                cleaned, flags = sanitize(s, self.task_spec)
                clean_series.append(cleaned)
                result.quality_flags.append(flags)
            except Exception as exc:
                result.warnings.append(f"signal_sanitizer[{s.signal_col}]: {exc}")
                log.warning("stage_warning", run_id=run_id, stage="signal_sanitizer",
                            signal=s.signal_col, error=str(exc))
        result.series = clean_series
        if not clean_series:
            result.errors.append("All signals failed sanitization.")
            return result

        # ── Stage 3: Global Series Profiler ───────────────────────────
        for s in clean_series:
            try:
                from signals.global_series_profiler import profile
                seq = profile(s, self.task_spec)
                result.regime_sequences.append(seq)
            except Exception as exc:
                result.warnings.append(f"global_series_profiler[{s.signal_col}]: {exc}")

        log.info("stage_complete", run_id=run_id, stage="global_series_profiler",
                 n_sequences=len(result.regime_sequences))

        # ── Stage 4: Historical Profile Builder ──────────────────────
        try:
            from signals.historical_profile_builder import build_profile
            result.well_profile = build_profile(
                result.regime_sequences,
                existing_profile=self.existing_profile,
            )
        except Exception as exc:
            result.warnings.append(f"historical_profile_builder: {exc}")
            log.warning("stage_warning", run_id=run_id, stage="historical_profile_builder",
                        error=str(exc))

        # ── Stage 5: Candidate Event Detector ────────────────────────
        if result.well_profile:
            for seq in result.regime_sequences:
                try:
                    from signals.candidate_event_detector import detect
                    cands = detect(seq, result.well_profile, self.task_spec)
                    result.candidates.extend(cands)
                except Exception as exc:
                    result.warnings.append(f"candidate_event_detector: {exc}")

        log.info("stage_complete", run_id=run_id, stage="candidate_event_detector",
                 n_candidates=len(result.candidates))

        # ── Stage 6: Local Segment Analyzer ──────────────────────────
        # Map signal_col → series for fast lookup
        series_by_signal = {s.signal_col: s for s in clean_series}
        for cand in result.candidates:
            # Use first series as fallback if trace_name not matched
            sig_series = series_by_signal.get(
                cand.series_name,
                clean_series[0] if clean_series else None,
            )
            if sig_series is None:
                continue
            try:
                from signals.local_segment_analyzer import analyze
                feats = analyze(cand, sig_series)
                result.local_features.append(feats)
            except Exception as exc:
                result.warnings.append(f"local_segment_analyzer[{cand.candidate_id}]: {exc}")

        # ── Stage 7: Context Fact Extractor ───────────────────────────
        if maintenance_docs:
            try:
                from context.context_fact_extractor import extract_facts_batch
                facts = extract_facts_batch(
                    maintenance_docs,
                    llm_client=self.llm_client,
                    model=self.llm_model,
                )
                result.maintenance_facts = facts
                for cand in result.candidates:
                    bundle = build_context_bundle(
                        cand,
                        maintenance_docs=maintenance_docs,
                        maintenance_facts=facts,
                    )
                    if bundle is not None:
                        result.context_bundles.append(bundle)
            except Exception as exc:
                result.warnings.append(f"context_fact_extractor: {exc}")

        # ── Stage 8: Rule Engine ──────────────────────────────────────
        features_by_id = {f.candidate_id: f for f in result.local_features}
        context_by_id = {b.candidate_id: b for b in result.context_bundles}

        for cand in result.candidates:
            try:
                from rules.rule_engine import evaluate
                from rules.rule_schemas import RuleInput
                inp = RuleInput(
                    candidate=cand,
                    features=features_by_id.get(cand.candidate_id),
                    context=context_by_id.get(cand.candidate_id),
                    task_params={
                        "primary_deviation": self.task_spec.primary_deviation,
                        "equipment_family": self.task_spec.equipment_family,
                    },
                )
                rule_result = evaluate(
                    inp, self.rule_registry,
                    unknown_label=self.task_spec.unknown_label,
                )
                result.rule_results.append(rule_result)
            except Exception as exc:
                result.warnings.append(f"rule_engine[{cand.candidate_id}]: {exc}")

        log.info("pipeline_complete", run_id=run_id,
                 n_candidates=len(result.candidates),
                 n_rule_results=len(result.rule_results),
                 n_warnings=len(result.warnings))

        return result


# ---------------------------------------------------------------------------
# Default registry factory (lazy import to avoid circular deps)
# ---------------------------------------------------------------------------

def _default_registry():
    from rules.starter_ruleset import build_registry
    return build_registry()


def build_context_bundle(
    candidate: CandidateEvent,
    *,
    maintenance_docs: Optional[list[MaintenanceDocument]] = None,
    maintenance_facts: Optional[list[StructuredFacts]] = None,
) -> Optional[ContextBundle]:
    matched_docs = [
        doc for doc in (maintenance_docs or [])
        if _asset_matches(candidate.asset_id, doc.asset_id)
        and _date_matches(candidate, doc.event_date)
    ]
    matched_facts = [
        fact for fact in (maintenance_facts or [])
        if _asset_matches(candidate.asset_id, fact.asset_id)
        and _date_matches(candidate, fact.event_date)
    ]
    if not matched_docs and not matched_facts:
        return None

    flags: list[str] = []
    if matched_docs and not matched_facts:
        flags.append("maintenance_docs_without_matching_facts")
    if any(fact.extraction_confidence in {"low", "failed"} for fact in matched_facts):
        flags.append("low_confidence_context")

    return ContextBundle(
        candidate_id=candidate.candidate_id,
        maintenance_docs=matched_docs,
        maintenance_facts=matched_facts,
        flags=flags,
    )


def _asset_matches(candidate_asset_id: Optional[str], context_asset_id: Optional[str]) -> bool:
    candidate_asset = _normalize_asset_id(candidate_asset_id)
    context_asset = _normalize_asset_id(context_asset_id)
    if candidate_asset is None:
        return True
    if context_asset in {None, "unknown"}:
        return True
    return candidate_asset == context_asset


def _date_matches(candidate: CandidateEvent, event_date) -> bool:
    if event_date is None:
        return True
    event_day = event_date.date()
    return candidate.segment.start.date() <= event_day <= candidate.segment.end.date()


def _normalize_asset_id(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text.casefold()
