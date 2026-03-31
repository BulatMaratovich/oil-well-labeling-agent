# Spec: Pipeline Orchestrator and Stage Control

## Overview

The `PipelineRunner` orchestrates ten sequential analysis stages per run. It is not an LLM agent orchestrator — it dispatches deterministic and semi-deterministic processing stages, manages per-run state, enforces timeouts, and routes candidates through the Rule Engine to the human review queue.

The Rule Engine (not the LLM) is the decision authority for labeling.

---

## Pipeline Stages

```python
STAGES = [
    "input_normalization",
    "signal_sanitization",
    "global_series_profiling",
    "historical_profile_build",
    "candidate_event_detection",
    "local_segment_analysis",
    "context_fact_extraction",
    "rule_engine",
    "review_queue_assembly",
    "human_review",
]
```

### Timeout Configuration

```python
TIMEOUTS_S = {
    "input_normalization":      10,
    "signal_sanitization":      15,
    "global_series_profiling":  30,   # PELT on full series; O(n log n)
    "historical_profile_build": 10,
    "candidate_event_detection": 10,
    "local_segment_analysis":   10,   # per candidate
    "context_fact_extraction":  40,   # per candidate; includes LLM call
    "rule_engine":               5,   # per candidate; deterministic
    "asset_total":             180,   # hard wall-clock limit per asset
    "candidate_total":          90,   # hard wall-clock limit per candidate
}

RETRIES = {
    "context_fact_extraction": 2,    # LLM schema retry
    "global_series_profiling":  1,   # retry with relaxed parameters on failure
}
```

### On Stage Failure

```python
# Per-candidate failure
candidate.status = "skipped"
candidate.label = "unknown"
candidate.routing = "mandatory_review"
candidate.flags.append("processing_failed")
# Continue to next candidate — pipeline does NOT stop

# Per-asset failure (stages 1–4)
asset.status = "failed"
# Log error, skip asset, continue with next asset
```

---

## Stage Responsibilities

### Stage 1 — Input Normalization

- Parse CSV or DB extract
- Map columns to `CanonicalTimeSeries {asset_id, timestamp, signal_name, value, unit}`
- Validate: required columns present, timestamp parseable, signal column numeric
- On validation failure: abort run, return structured error report

### Stage 2 — Signal Sanitization

- Clamp values outside physical plausible range (configurable per TaskSpec)
- Detect and flag dropout spans (consecutive zeros or nulls above min duration)
- Interpolate short gaps (below max_gap_s threshold)
- Compute `QualityFlags {missing_pct, dropout_spans: list[DateRange], noise_level, clamp_events}`
- If `missing_pct > data_quality_min`: mark asset outcome `insufficient_data_quality`; skip stages 3–8
- `insufficient_data_quality` is an asset-level terminal outcome, not a candidate label in the export

### Stage 3 — Global Series Profiling

- Run PELT change-point detection on the full asset series
- Cluster stable segments into regime types (by power distribution similarity)
- Output: `RegimeSequence [{start, end, regime_id, regime_type, duration_h}]`
- `is_novel` is NOT computed here — WellProfile has not been loaded yet
- If no change-points detected: single-regime series; flag `no_regime_structure`

### Stage 4 — Historical Profile Build

- Load `WellProfile` for this asset from TaskMemory
- If no profile: use population fallback baseline; flag `no_well_history`
- Check profile age; flag `profile_stale` if older than `max_profile_age_days`
- Output: `WellProfile {baseline_regimes, known_stops, known_replacements}`

### Stage 5 — Candidate Event Detection

- Both `RegimeSequence` and `WellProfile` are now available; novelty and deviation are computed here
- Compare each regime against `WellProfile`:
  - Regime type not in `WellProfile.baseline_regimes` → novel_regime candidate
  - Known regime with atypical amplitude (z-score > `anomaly_z_threshold`) → atypical_amplitude candidate
  - Regime duration outside `[p10, p90]` of historical distribution → unusual_duration candidate
  - Change-point with high transition steepness → abrupt_transition candidate
- If profiling produced a single flat regime (`no_regime_structure`): emit one full-series candidate for review instead of returning an empty candidate list
- Assign `deviation_type` and `deviation_score` to each candidate
- Output: `list[CandidateEvent]`
- If no candidates: log `no_candidates_found`, mark asset `complete`; asset does not produce any label records in the export

### Stage 6 — Local Segment Analysis

- For each candidate only: extract detailed local features
- Features: power stats (mean, std, percentiles), cycle metrics, waveform shape, transition sharpness
- Output: `LocalFeatures` per candidate
- This stage runs on candidate segments only — not on the full series

### Stage 7 — Context Fact Extraction

- **Structured lookup** (deterministic):
  - Exact filter: maintenance logs for `asset_id` and `[candidate.start - 7d, candidate.end + 1d]`
  - Exact lookup: equipment metadata for `asset_id`
  - Output: raw matched documents
- **LLM fact extraction** (semi-deterministic, per matched report):
  - Input: raw report text wrapped in XML tags
  - Output: `StructuredFacts {event_type, date, asset_id, action, parts_replaced, duration_h}`
  - Schema-validated; retry once on parse failure
  - On failure: flag `fact_extraction_failed`; proceed with raw document text only
- Output: `ContextBundle {maintenance_facts, equipment_metadata, matched_docs}`

### Stage 8 — Rule Engine

- Load active ruleset from `TaskMemory.ruleset` (locked at run start)
- Apply rules in priority order per candidate:
  - Priority 0: series-level quality gate already handled before candidate generation
  - Priority 1: candidate-local sensor/data exclusions (`sensor_issue`)
  - Priority 2: confounder exclusions (`planned_stop`, `planned_maintenance`)
  - Priority 3: stable unusual regime
  - Priority 4: true deviation rules (`belt_break`, ...)
  - Priority 5: fallback `unknown`
- Detect rule conflicts (multiple priority-4 rules fire)
- Output: `RuleResult {label, rule_trace, abstain_reason?, conflict_flag}`
- Internal abstention is represented externally as `label = "unknown"` plus `abstain_reason`

### Stage 9 — Review Queue Assembly

All candidates are placed in the review queue. Routing is set based on RuleResult:

```python
ROUTING_VALUES = {
    "review":            "rule fired cleanly or soft flags present; engineer confirmation required",
    "mandatory_review":  "unknown/no-match, rule_conflict, processing_failed, or hard flags",
}
```

**All routing outcomes require engineer action.** There is no auto-label and no suggested-accept shortcut in PoC v1.

Hard flags:
```
rule_conflict, processing_failed, fact_extraction_failed, no_well_history,
schema_retry_exhausted, injection_suspected
```

Soft flags:
```
profile_stale, no_regime_structure, no_candidates_on_rerun,
fact_extraction_low_confidence, missing_equipment_metadata
```

### Stage 10 — Human Review

Display per candidate:
- Signal plot: `power_kW` over full candidate segment + surrounding context window (matplotlib PNG)
- Regime context: regime type, duration, position in full series
- Historical comparison: how this regime compares to WellProfile baseline
- Context facts: extracted StructuredFacts + raw document title/date
- Rule trace: which rules evaluated, which fired, which were blocked
- Similar confirmed cases (top-3 from ExampleStore by feature similarity)
- Proposed label + routing + flags

Engineer actions:
```
[A]ccept    → final_label = proposed_label; source = "agent_accepted"
[M]odify    → prompt for new label from TaskSpec.label_taxonomy; source = "engineer_override"
             → prompt for correction_reason (required for Modify)
[R]eject    → final_label = null; excluded from export; source = "rejected"
[B]ambig    → mark ambiguous; excluded from export and rule-mining input
[S]kip      → defer to end of queue (not allowed for mandatory_review)
```

---

## Discovery Phase (Pre-Run)

Before the first labeling run, a new task requires a discovery dialogue.

### DiscoveryAgent

LLM-based structured interview with the engineer:

```
Questions asked:
  1. What equipment / installation type?
  2. What signal(s) are available and what are their units?
  3. What is the primary deviation or failure to label?
  4. What counts as "normal operation" for this asset?
  5. What confounders exist? (planned stops, sensor issues, load changes, seasonal patterns)
  6. What external evidence is available? (maintenance reports, metadata, manuals)
  7. What is the minimum segment duration to consider as a candidate?
  8. What is the expected frequency of the target deviation?

Output: TaskSpec {
    task_id, equipment_family, signal_schema, segmentation_strategy,
    feature_profile, label_taxonomy, unknown_label, context_sources,
    baseline_strategy, quality_rules, review_policy
}
```

Domain adapter (`DomainAdapter`) bootstraps TaskSpec defaults for known equipment families. Unknown families start from a generic adapter and user-supplied schema.

---

## RunState Schema

```python
@dataclass
class RunState:
    run_id: str
    task_id: str
    task_version: str
    ruleset_version: str           # locked at run start
    input_file: str
    config_hash: str
    current_stage: str
    assets: list[AssetRecord]
    errors: list[ErrorRecord]
    started_at: datetime
    checkpointed_at: datetime
```

```python
@dataclass
class AssetRecord:
    asset_id: str
    status: str                    # pending | processing | completed | failed | skipped
    quality_flags: QualityFlags
    regime_sequence: list[Regime]
    candidates: list[CandidateRecord]
```

```python
@dataclass
class CandidateRecord:
    candidate_id: str
    segment: DateRange
    deviation_type: str
    deviation_score: float
    local_features: LocalFeatures
    context_bundle: ContextBundle  # stored to avoid re-extraction on resume
    rule_result: RuleResult
    routing: str
    flags: list[str]
    final_label: str | None
    correction_reason: str | None
    status: str                    # pending | completed | skipped
```

**Resume logic:** On restart, assets with `status = completed` and candidates with `status != pending` are skipped. Context bundles and rule results are reloaded from RunState — not recomputed.

---

## Stop Conditions

| Condition | Action |
|-----------|--------|
| All assets processed | Normal completion → human review queue |
| LLM API unavailable (retries exhausted) | Fact extraction stage skipped for remaining candidates; pipeline continues without facts |
| Input validation fails | Abort run, return structured error report |
| `KeyboardInterrupt` | Checkpoint written, clean exit |
