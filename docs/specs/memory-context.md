# Spec: Memory and Context Handling

## Overview

The system has two memory tiers:

1. **RunState** — per-run checkpoint; supports resume within a single labeling session
2. **TaskMemory** — persistent cross-run store; contains the ruleset, well profiles, confirmed examples, and known exceptions

The center of TaskMemory is the **ruleset**, not the example store. Confirmed examples are a regression set and a review aid — they do not drive labeling decisions directly.

There is no LLM conversation history. Each LLM call is stateless. Long-term improvement comes from growing the ruleset and updating well profiles, not from accumulating chat context.

---

## Tier 1 — RunState (Per-Run)

**Location:** `./runs/{run_id}/state.json`

**Lifecycle:**
- Created at pipeline start
- Updated after each asset and candidate completes
- Read on resume: assets and candidates with `status != pending` are skipped
- Retained for audit; not deleted automatically

**Schema:**

```json
{
  "run_id": "2024-01-15_143022",
  "task_id": "rod_pump_belt_break_v1",
  "task_version": "1.2",
  "ruleset_version": "1.4",
  "input_file": "wells_batch_42.csv",
  "config_hash": "a3f8...",
  "current_stage": "rule_engine",
  "assets": [
    {
      "asset_id": "Абдуллинская_2248",
      "status": "completed",
      "quality_flags": {
        "missing_pct": 0.02,
        "dropout_spans": [],
        "noise_level": "low",
        "clamp_events": 0
      },
      "regime_count": 4,
      "candidate_count": 2,
      "candidates": [
        {
          "candidate_id": "cand_001",
          "segment": ["2024-01-10T08:00:00Z", "2024-01-10T14:00:00Z"],
          "deviation_type": "novel_regime",
          "deviation_score": 0.82,
          "rule_result": {
            "label": "belt_break",
            "winning_rule": "rule_belt_break_low_amplitude_v1",
            "rule_trace": {...},
            "conflict": false
          },
          "routing": "review",
          "flags": [],
          "final_label": "belt_break",
          "correction_reason": null,
          "status": "completed"
        }
      ]
    }
  ],
  "errors": []
}
```

Context bundles and rule results are stored in RunState to avoid recomputation on resume.

---

## Tier 2 — TaskMemory (Cross-Run)

**Location:** `./data/tasks/{task_id}/memory.json`

**Lifecycle:**
- Created when TaskSpec is first saved after discovery dialogue
- Updated after every engineer action (accept/modify/reject/ambiguous) and after every rule change
- Never deleted automatically — retained as the task's accumulated knowledge

### Schema

```python
@dataclass
class TaskMemory:
    task_id: str
    task_spec: TaskSpec               # task definition: signals, taxonomy, context sources, quality rules
    ruleset: list[Rule]               # versioned active rules — primary decision logic
    well_profiles: dict[str, WellProfile]  # per-well historical baselines
    confirmed_examples: list[LabelRecord]  # accepted + corrected; used as regression set and review context
    rejected_examples: list[LabelRecord]   # rejected; hard negatives
    ambiguous_examples: list[LabelRecord]  # ambiguous; excluded from rule mining
    known_confounders: list[ConfounderRecord]  # explicit exceptions: "well X, period Y, reason Z"
    user_rules: list[UserRule]        # informal rule notes from engineers during review
    autonomy_status: AutonomyStatus
    updated_at: datetime
```

---

## Ruleset

The ruleset is the core of TaskMemory. It contains the explicit if/then rules that the Rule Engine evaluates.

### Rule Schema

```python
@dataclass
class Rule:
    rule_id: str
    version: str
    priority: int                  # 0=quality, 1=sensor, 2=confounder, 3=stable_unusual, 4=true_deviation
    label: str
    description: str               # human-readable; shown in rule trace and review UI
    condition_code: str            # reference to condition function in rule_registry.py
    condition_params: dict         # configurable thresholds for this rule
    source: str                    # "domain_expert_hardcoded" | "engineer_approved" | "rule_miner_draft"
    active: bool
    added_at: datetime
    last_modified_at: datetime
    last_modified_by: str
```

### Rule Versioning

Every rule change (add, modify, deactivate) appends a new entry to `ruleset_history.json`. The `ruleset_version` field in RunState records which version was active for a given run.

### Regression Check

Before any rule is activated:
```python
def check_regression(new_rule: Rule, confirmed_examples: list[LabelRecord]) -> RegressionResult:
    # Re-evaluate rule engine with new_rule included
    # Return list of examples whose label changes
    # If any change: block activation, return RegressionResult(failed=True, affected=[...])
```

---

## WellProfile

Per-well historical baseline stored in TaskMemory.

```python
@dataclass
class WellProfile:
    well_id: str
    baseline_regimes: list[RegimeBaseline]   # typical power stats per observed regime type
    known_stops: list[DateRange]             # verified planned stops from maintenance records
    known_replacements: list[DateRange]      # belt/rod replacements with dates
    first_seen: datetime
    last_updated: datetime
    profile_source: str             # "built_from_history" | "population_fallback"
```

```python
@dataclass
class RegimeBaseline:
    regime_type: str
    power_mean: float
    power_std: float
    power_p10: float
    power_p90: float
    typical_duration_h: float
    observation_count: int
```

---

## Confirmed Examples

Confirmed examples are **not** few-shot prompting material. They serve three purposes:

1. **Regression set** — verify that new rules do not break previously validated cases
2. **Review context** — show engineer the top-3 similar confirmed cases during review
3. **Rule mining input** — detect patterns in corrections to suggest new rules or confounder updates

```python
@dataclass
class LabelRecord:
    record_id: str
    task_id: str
    asset_id: str
    segment: DateRange
    deviation_type: str
    local_features: LocalFeatures       # stored for similarity search
    feature_embedding: np.ndarray       # dense vector for cosine similarity
    rule_result: RuleResult             # what the rule engine proposed
    final_label: str
    was_override: bool                  # True if engineer modified the proposed label
    correction_reason: str | None       # required when was_override = True
    confirmed_at: datetime
    run_id: str
```

### Similar Example Retrieval (For Review UI Only)

```python
def retrieve_similar(
    local_features: LocalFeatures,
    task_id: str,
    top_k: int = 3,
    min_similarity: float = 0.60,
) -> list[LabelRecord]:
    query_embedding = embed(format_features(local_features))
    similarities = cosine_similarity(query_embedding, example_embeddings)
    return top_k_above_threshold(similarities, min_similarity)
```

This retrieval is for review context only — it is not passed to the Rule Engine or to any LLM prompt.

---

## Known Confounders

Explicit exceptions stored in TaskMemory.

```python
@dataclass
class ConfounderRecord:
    confounder_id: str
    asset_id: str | None        # None = applies to all assets
    date_range: DateRange | None
    label: str                  # what to label this case as
    reason: str                 # why this is a confounder, not a failure
    added_by: str
    added_at: datetime
```

Example:
```json
{
  "confounder_id": "cf_001",
  "asset_id": "Абдуллинская_2248",
  "date_range": ["2024-01-08", "2024-01-12"],
  "label": "planned_stop",
  "reason": "Scheduled compressor maintenance, confirmed in VSP report #4412",
  "added_by": "engineer_session_7",
  "added_at": "2024-01-20T10:15:00Z"
}
```

The Rule Engine checks ConfounderRecords before applying deviation rules.

---

## LLM Context Budget per Call

Context is assembled once per agent call by `PromptBuilder`. No conversation history.

### DiscoveryAgent

```
slot                          | max tokens | source
------------------------------|------------|------------------------------------------
system_prompt                 |    ~500    | config/prompts/discovery_system.txt
elicitation questions         |    ~500    | hardcoded structured questions
sample CSV rows (optional)    |  ≤2 000    | first N rows from input file
------------------------------|------------|------------------------------------------
total                         |  ≤3 000    |
```

### ContextFactExtractor (per report)

```
slot                          | max tokens | source
------------------------------|------------|------------------------------------------
system_prompt                 |    ~400    | config/prompts/fact_extraction_system.txt
report text                   |  ≤1 500    | raw report text, XML-wrapped, truncated
output schema reminder        |    ~100    |
------------------------------|------------|------------------------------------------
total                         |  ≤2 000    |
```

### ExplanationAgent (per candidate, for review UI)

```
slot                          | max tokens | source
------------------------------|------------|------------------------------------------
system_prompt                 |    ~400    | config/prompts/explanation_system.txt
rule_trace (JSON)             |    ~400    | RuleResult serialized
label + taxonomy context      |    ~200    |
context facts summary         |    ~300    | StructuredFacts fields
------------------------------|------------|------------------------------------------
total                         |  ≤1 300    |
```

**No LLM call receives raw signal arrays or full LabelRecord lists.**

---

## Memory Policy Table

| Data type | Stored where | Scope | Notes |
|-----------|-------------|-------|-------|
| Raw signal (`power_kW` array) | In-memory only, per-asset | Discarded after sanitization | Never persisted |
| QualityFlags | RunState | Per-run | Retained for audit |
| RegimeSequence | RunState | Per-run | Retained for audit |
| LocalFeatures | RunState + ExampleStore (if confirmed) | Per-run; cross-run if confirmed | Feature embedding stored for similarity |
| ContextBundle | RunState | Per-run | Stored to avoid re-extraction on resume |
| RuleResult + RuleTrace | RunState + audit_log | Per-run | Full trace logged |
| FinalLabel | RunState + export | Per-run | Permanent output |
| Confirmed examples | ExampleStore (permanent) | Cross-run | Regression set + review context |
| Ruleset | TaskMemory (permanent, versioned) | Cross-run | Primary decision logic |
| WellProfiles | TaskMemory (permanent) | Cross-run | Updated after confirmed examples |
| ConfounderRecords | TaskMemory (permanent) | Cross-run | Explicit exceptions |
| LLM conversation history | Never | — | Each call is independent |
