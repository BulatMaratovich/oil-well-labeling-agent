# Spec: Tools and API Contracts

## Overview

This document specifies the Python-level contracts for all modules in the pipeline. All signal processing and rule evaluation are deterministic. LLM calls are isolated to three modules: `context_fact_extractor.py`, `discovery_agent.py`, and `explanation_agent.py`.

---

## Signal Layer

### `input_normalizer.normalize()`

```python
def normalize(
    raw_file: str | Path,
    task_spec: TaskSpec,
) -> CanonicalTimeSeries:
```

- Maps raw CSV columns to `{asset_id, timestamp, signal_name, value, unit}`
- Validates required columns, timestamp format, numeric signal
- Raises `InputValidationError` on schema mismatch
- Timeout: 10 s
- No retries; validation errors abort the run

### `signal_sanitizer.sanitize()`

```python
def sanitize(
    series: CanonicalTimeSeries,
    task_spec: TaskSpec,
) -> tuple[CanonicalTimeSeries, QualityFlags]:
```

- Clamps values outside `[task_spec.signal_min, task_spec.signal_max]`
- Detects dropout spans: consecutive values below `task_spec.dropout_threshold` for `>= task_spec.min_dropout_duration_s`
- Interpolates gaps shorter than `task_spec.max_interpolation_gap_s`
- Returns clean signal + `QualityFlags {missing_pct, dropout_spans, noise_level, clamp_events}`
- Timeout: 15 s

### `global_series_profiler.profile()`

```python
def profile(
    series: CanonicalTimeSeries,
    task_spec: TaskSpec,
) -> RegimeSequence:
```

- Runs PELT change-point detection (ruptures library, `model="rbf"`)
- Clusters stable segments into regime types by power distribution similarity
- Returns `RegimeSequence`: list of `Regime {start, end, regime_id, regime_type}`
- Novelty is not computed here; it is derived later by `candidate_event_detector.detect()` after `WellProfile` is loaded
- Timeout: 30 s; on failure: retry once with relaxed PELT penalty
- Raises `ProfilingError` after retry failure

### `historical_profile_builder.get_profile()`

```python
def get_profile(
    asset_id: str,
    task_memory: TaskMemory,
) -> WellProfile:
```

- Loads `WellProfile` from `TaskMemory.well_profiles`
- If not found: returns population fallback profile, sets `profile_source = "population_fallback"`
- Sets `profile_stale = True` if `profile.last_updated < now - max_profile_age_days`
- Timeout: 10 s

### `candidate_event_detector.detect()`

```python
def detect(
    regime_sequence: RegimeSequence,
    well_profile: WellProfile,
    task_spec: TaskSpec,
) -> list[CandidateEvent]:
```

- Compares each regime against `WellProfile.baseline_regimes`:
  - Novel regime type → candidate with `deviation_type = "novel_regime"`
  - Z-score of regime mean power vs baseline > `task_spec.anomaly_z_threshold` → `"atypical_amplitude"`
  - Regime duration outside `[p10_duration, p90_duration]` → `"unusual_duration"`
  - Change-point steepness above threshold → `"abrupt_transition"`
- If the profiler produced a single flat regime and flagged `no_regime_structure`, emits one full-series candidate with that flag attached
- Returns empty list only if no candidates remain after historical comparison
- Timeout: 10 s

### `local_segment_analyzer.analyze()`

```python
def analyze(
    candidate: CandidateEvent,
    series: CanonicalTimeSeries,
    task_spec: TaskSpec,
) -> LocalFeatures:
```

- Extracts features only for the candidate segment (not the full series)
- Features: `{power_mean, power_std, power_p10, power_p90, min_power, max_power, zero_fraction, transition_sharpness, segment_duration_h, preceding_regime_type, following_regime_type}`
- Timeout: 10 s per candidate

---

## Context Layer

### `structured_lookup.retrieve_maintenance_logs()`

```python
def retrieve_maintenance_logs(
    asset_id: str,
    window_start: datetime,
    window_end: datetime,
    days_before: int = 7,
    days_after: int = 1,
) -> list[MaintenanceDocument]:
```

- In-memory dict filter; deterministic; O(n_logs)
- Returns raw documents with `{doc_id, date, asset_id, report_type, raw_text}`
- Returns `[]` on no match; no exception
- Timeout: < 1 ms

### `structured_lookup.retrieve_equipment_metadata()`

```python
def retrieve_equipment_metadata(asset_id: str) -> EquipmentDocument | None:
```

- In-memory dict lookup; O(1)
- Returns `None` if asset_id not found; sets `missing_equipment_metadata` flag
- Timeout: < 1 ms

### `semantic_retriever.query()`

```python
def query(
    deviation_type: str,
    n_results: int = 5,
    min_similarity: float = 0.40,
) -> list[RuleDocument]:
```

- ChromaDB query; local in-process
- Returns top-k documents above similarity threshold
- Returns `[]` on ChromaDB unavailable; logs warning; sets `low_context` flag
- Timeout: 3 s

### `context_fact_extractor.extract()`

```python
def extract(
    doc: MaintenanceDocument,
    task_spec: TaskSpec,
    llm_client: LLMClient,
) -> StructuredFacts:
```

- Wraps `doc.raw_text` in XML tags
- Calls LLM with fact extraction prompt
- Schema-validates response; retries once on parse failure
- Returns `StructuredFacts` with `extraction_confidence = "failed"` on exhausted retries
- Never raises; failure is represented in the returned object
- Timeout: 40 s (includes LLM call)

---

## Rule Engine

### `rule_engine.evaluate()`

```python
def evaluate(
    candidate: CandidateEvent,
    local_features: LocalFeatures,
    context_bundle: ContextBundle,
    ruleset: list[Rule],
    well_profile: WellProfile,
) -> RuleResult:
```

- Evaluates rules in priority order
- Returns on first match (except priority 4: all priority-4 rules evaluated for conflict detection)
- Conflict: multiple priority-4 rules fire → `label = "unknown"`, `conflict = True`
- No match at any priority: `label = "unknown"`, `abstain_reason = "no_rule_matched"`
- Internal abstention is surfaced externally as `label = "unknown"`; there is no separate `abstain` label
- Deterministic; no LLM; timeout: 5 s

### `rule_registry.load()`

```python
def load(task_id: str) -> list[Rule]:
```

- Loads active rules from `data/tasks/{task_id}/ruleset.json`
- Returns only rules with `active = True`
- Locks ruleset version at load time; version stored in RunState

### `rule_registry.check_regression()`

```python
def check_regression(
    new_rule: Rule,
    confirmed_examples: list[LabelRecord],
    existing_ruleset: list[Rule],
) -> RegressionResult:
```

- Re-evaluates rule engine with `new_rule` added
- Returns `RegressionResult {failed: bool, affected_example_ids: list[str]}`
- Called before any rule activation; blocks activation on failure

---

## LLM Client

### `llm_client.call()`

```python
def call(
    system_prompt: str,
    user_message: str,
    output_schema: type[BaseModel],
    max_tokens: int = 512,
    timeout_s: int = 30,
) -> BaseModel:
```

- Calls Anthropic Messages API with `response_format = {"type": "json_object"}`
- Validates response against `output_schema` (Pydantic model)
- Retry policy: up to 3 retries on HTTP 5xx with exponential backoff (2–8 s)
- 1 retry on timeout
- Raises `LLMUnavailableError` after exhausted retries
- Never receives raw signal arrays

### Token Budgets per Call Type

| Call | Input max | Output max | Total |
|------|-----------|------------|-------|
| Fact extraction (per report) | 2 000 | 300 | 2 300 |
| Discovery dialogue | 3 000 | 600 | 3 600 |
| Explanation generation | 1 300 | 400 | 1 700 |
| Rule draft suggestion | 1 500 | 300 | 1 800 |

---

## ExampleStore

### `example_store.add()`

```python
def add(record: LabelRecord) -> None:
```

- Appends to `examples.json`
- Computes and stores feature embedding in `example_embeddings.npy`

### `example_store.retrieve_similar()`

```python
def retrieve_similar(
    local_features: LocalFeatures,
    task_id: str,
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> list[LabelRecord]:
```

- Cosine similarity against all stored embeddings
- Returns top-k records above threshold
- Used for review UI context only

### `example_store.get_regression_set()`

```python
def get_regression_set(task_id: str) -> list[LabelRecord]:
```

- Returns all confirmed (non-ambiguous, non-rejected) examples
- Used by `rule_registry.check_regression()`

---

## API Contracts Summary

| Module | Deterministic | Network | Timeout | Failure mode |
|--------|--------------|---------|---------|--------------|
| `input_normalizer` | Yes | No | 10 s | raises `InputValidationError` |
| `signal_sanitizer` | Yes | No | 15 s | returns partial result + flags |
| `global_series_profiler` | Yes | No | 30 s | retry × 1; raises `ProfilingError` |
| `historical_profile_builder` | Yes | No | 10 s | returns fallback profile |
| `candidate_event_detector` | Yes | No | 10 s | returns empty list |
| `local_segment_analyzer` | Yes | No | 10 s | raises `AnalysisError`; candidate skipped |
| `structured_lookup` | Yes | No | <1 ms | returns `[]` or `None` |
| `semantic_retriever` | No (embeddings) | No | 3 s | returns `[]`; logs warning |
| `context_fact_extractor` | No (LLM) | Yes (LLM API) | 40 s | returns `StructuredFacts(confidence=failed)` |
| `rule_engine` | Yes | No | 5 s | returns `unknown` + `processing_failed` flag |
| `llm_client` | No | Yes | 30 s | raises `LLMUnavailableError` after retries |
| `example_store` | Yes | No | <100 ms | raises `StoreError` |

### Protection Rules

- No module executes code generated by LLM output or external input
- No module writes outside `data/` and `runs/` directories
- LLM client is the only module that makes external network calls
- Raw `power_kW` arrays are never passed to `llm_client.call()`
- All LLM outputs are schema-validated before use
