# System Design: Industrial Time-Series Expert Labeling Framework (PoC)

## 1. Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Core paradigm** | Global-to-local rule-based labeling with human-in-the-loop | Anomalies are relational, not intrinsic properties of a single window — they are defined by deviation from historical norms, peer behavior, and explainable context |
| **Decision authority** | Rule Engine | Labeling logic lives in explicit, versioned, auditable if/then rules — not in LLM weights or model confidences |
| **LLM scope** | Narrowly scoped: discovery dialogue, free-text fact extraction, draft rule suggestions, review explanations | LLM is not the primary detector or classifier; it processes natural language and assists humans |
| **Unit of analysis** | Regime segment, not fixed window | A regime is a stable operating interval between structural change-points; anomalies are regimes that are new, atypical, or unexplained |
| **Anomaly definition** | Relational: new/atypical regime + no context explanation | A "strange window" is not an anomaly — an unexplained regime that deviates from the well's own history and is not covered by known confounders is a candidate |
| **Taxonomy** | Contrastive, including confounders | Labels include not only true failures but also look-alike non-failures: planned_stop, sensor_issue, stable_unusual_regime, unknown |
| **Learning mechanism** | Rule updates from engineer corrections; confirmed examples as regression set | System improves by growing the ruleset, not by accumulating few-shot prompting examples |
| **State persistence** | Per-run RunState + persistent TaskMemory (rules, profiles, baselines, exceptions) | Rules and profiles survive across runs; each run is resumable |
| **Reference domain** | Belt-break detection on rod pumping units | First domain pack; validates the framework without constraining its architecture |
| **PoC scope** | No peer comparison in critical path; no auto-label; hardcoded starter ruleset | Peer comparison and selective autonomy are Phase 2 |

---

## 2. Pipeline Stages

```text
Raw Signal
  │
  ▼
[1] Input Normalizer
  │  CSV / DB extract → CanonicalTimeSeries {asset_id, timestamp, signal, unit}
  ▼
[2] Signal Sanitizer
  │  drop/flag impossible values, clamp outliers, interpolate short gaps
  │  output: CleanSignal + QualityFlags {missing_pct, dropout_spans, noise_level, clamp_events}
  │  if missing_pct > threshold: mark asset_outcome = insufficient_data_quality, skip candidate generation
  ▼
[3] Global Series Profiler
  │  change-point detection on the full series (PELT / BOCPD)
  │  cluster stable segments into regime types
  │  output: RegimeSequence [{start, end, regime_id, regime_type, duration_h}]
  │  (novelty is NOT computed here — WellProfile not yet loaded)
  ▼
[4] Historical Profile Builder
  │  load or update per-well baseline stats per regime type
  │  compute: typical power distribution, cycle stats, known confounders for this well
  │  output: WellProfile {baseline_regimes, known_stops, known_replacements}
  ▼
[5] Candidate Event Detector
  │  compare RegimeSequence against WellProfile (both now available)
  │  flag: novel regime type, atypical amplitude, unusual duration, abrupt transition
  │  if profiler returned a single flat regime (no_regime_structure), emit one full-series candidate
  │  output: CandidateEvents [{segment, deviation_type, deviation_score, context_query}]
  │  if no candidates: log no_candidates_found, mark asset complete, no label record in export
  ▼
[6] Local Segment Analyzer
  │  extract detailed local features only for candidate segments
  │  output: LocalFeatures {power stats, cycle stats, waveform shape, transition sharpness}
  ▼
[7] Context Fact Extractor
  │  structured lookup: exact filter of maintenance logs + metadata by asset_id + date window
  │  LLM: parse free-text report content → StructuredFacts {event_type, date, action, parts}
  │  output: ContextBundle {maintenance_facts, equipment_metadata, known_rules_text}
  ▼
[8] Rule Engine
  │  apply RuleRegistry in priority order:
  │    1. Candidate-local sensor/data exclusions (sensor_issue)
  │    2. Confounder rules (planned_stop, planned_maintenance)
  │    3. Stable unusual regime rules (stable_unusual_regime)
  │    4. True deviation rules (belt_break, ...)
  │    5. Fallback unknown: case not covered or rule conflict
  │  output: RuleResult {label, rule_ids_fired, rule_trace, abstain_reason?}
  ▼
[9] Human Review
  │  display: signal plot + regime context + historical comparison + context facts + rule trace + similar confirmed cases
  │  engineer: Accept / Modify / Reject / Mark Ambiguous
  │  on Modify/Reject: capture correction_reason
  ▼
[10] Rule / Profile Update
     persist final label to confirmed examples
     if correction pattern detected → draft new rule or update confounder list
     update WellProfile baselines
     write to audit log
```

---

## 3. Module Structure

```text
core/
├── task_manager.py              — TaskSpec: create, load, version
├── canonical_schema.py          — dataclasses: CanonicalTimeSeries, Window, Regime, CandidateEvent,
│                                  RuleResult, ContextBundle, LabelRecord
├── pipeline_runner.py           — orchestrate stages 1–10 per run
├── state_manager.py             — RunState: checkpoint, resume
└── policy_engine.py             — route candidates to review / mandatory_review

signals/
├── input_normalizer.py          — raw CSV/DB → CanonicalTimeSeries
├── signal_sanitizer.py          — clamp, interpolate, flag quality issues
├── global_series_profiler.py    — PELT/BOCPD change-points, regime clustering
├── historical_profile_builder.py — per-well baselines, known regime types
├── candidate_event_detector.py  — compare regime vs history → CandidateEvents
└── local_segment_analyzer.py    — detailed local features for candidate segments

rules/
├── rule_engine.py               — apply RuleRegistry in priority order, produce RuleResult
├── rule_registry.py             — load, version, index rules; check for conflicts
├── rule_trace.py                — record which rules fired, in what order, with what inputs
└── rule_schemas.py              — Rule dataclass: id, version, priority, condition_fn, label, description

context/
├── structured_lookup.py         — exact filter/lookup: maintenance logs, metadata
├── context_fact_extractor.py    — LLM: free-text report → StructuredFacts
└── context_bundle.py            — assemble ContextBundle from all sources

learning/
├── example_store.py             — confirmed/rejected/ambiguous examples; regression checks
├── task_memory.py               — TaskMemory: TaskSpec, ruleset, profiles, exceptions, autonomy_status
└── rule_miner.py                — suggest new rules from correction patterns

agents/
├── discovery_agent.py           — LLM: interview user → TaskSpec
└── explanation_agent.py         — LLM: generate human-readable explanation for review UI

ui/
├── discovery_cli.py             — discovery dialogue terminal UI
├── review_ui.py                 — signal plot + regime context + rule trace + accept/modify/reject
└── export_ui.py                 — export labeled datasets + task snapshot

observability/
├── logger.py                    — structlog JSONL
├── metrics.py                   — candidate recall, rule coverage, FP rate on confounders, unknown-case rate
└── evaluations.py               — offline eval against ground truth; regression check on confirmed examples
```

---

## 4. Rule Engine Design

The Rule Engine is the central decision authority.

### Rule Schema

```python
@dataclass
class Rule:
    rule_id: str
    version: str
    priority: int                  # lower = applied first
    label: str                     # label to assign if rule fires
    condition: Callable[[CandidateEvent, ContextBundle, LocalFeatures], bool]
    description: str               # human-readable; shown in rule trace
    source: str                    # "domain_expert" | "engineer_correction" | "rule_miner_draft"
    active: bool
    added_at: datetime
    last_modified_at: datetime
```

### Application Order

```text
Priority 0 — Series-level quality gate
  if series_quality.missing_pct > threshold → skip asset before candidate generation
  this is an asset outcome, not a candidate label

Priority 1 — Sensor issue
  if dropout_span covers candidate segment → label = "sensor_issue"

Priority 2 — Known confounder exclusions
  if maintenance_facts contains planned_stop matching asset_id + date → label = "planned_stop"
  if maintenance_facts contains equipment_replacement matching asset_id + date → label = "planned_maintenance"

Priority 3 — Stable unusual regime
  if regime is novel but stable (long duration, low internal variance) AND no failure signature → label = "stable_unusual_regime"

Priority 4 — True deviation rules
  if <belt_break conditions> → label = "belt_break"
  ... (extensible per domain)

Priority 5 — Fallback unknown
  no rule fired → label = "unknown", routing = "mandatory_review"
  rule conflict (multiple priority-4 rules fire) → routing = "mandatory_review", flag = "rule_conflict"
```

### Rule Trace

Every RuleResult includes the full trace:

```python
@dataclass
class RuleTrace:
    rules_evaluated: list[str]     # rule_ids in order
    rules_fired: list[str]         # rule_ids that matched
    rules_blocked: list[str]       # rule_ids not reached due to higher-priority match
    winning_rule: str | None
    conflict: bool
    abstain_reason: str | None
```

### Versioning

Rules are stored in `data/tasks/{task_id}/ruleset.json`. Each change creates a new version entry. The RuleRegistry loads the active ruleset at pipeline start and locks it for the run.

---

## 5. Memory and State

### Two Memory Tiers

1. **RunState** — per-run checkpoint for resume
2. **TaskMemory** — persistent cross-run knowledge for a task

### TaskMemory Schema

```python
@dataclass
class TaskMemory:
    task_id: str
    task_spec: TaskSpec
    ruleset: list[Rule]                     # active rules (versioned)
    well_profiles: dict[str, WellProfile]   # per-well historical baselines
    confirmed_examples: list[LabelRecord]   # accepted/corrected examples → regression set
    rejected_examples: list[LabelRecord]    # rejected → hard negatives
    ambiguous_examples: list[LabelRecord]   # ambiguous → excluded from rule mining
    known_confounders: list[ConfounderRecord]  # exceptions: "well X, period Y, reason Z"
    user_rules: list[UserRule]              # explicit rule notes from engineers
    autonomy_status: AutonomyStatus
    updated_at: datetime
```

### WellProfile

```python
@dataclass
class WellProfile:
    well_id: str
    baseline_regimes: list[RegimeBaseline]   # typical power stats per regime type
    known_stops: list[DateRange]             # from verified maintenance records
    known_replacements: list[DateRange]      # belt/rod replacements
    first_seen: datetime
    last_updated: datetime
```

### No LLM conversation history

Each LLM call is stateless. Long-term learning is in rules and profiles, not in chat transcripts.

---

## 6. LLM Scope (Narrow and Explicit)

| LLM call | Input | Output | Where used |
|----------|-------|--------|------------|
| Discovery dialogue | User answers + sample data rows | TaskSpec fields | `discovery_agent.py` |
| Fact extraction | Free-text ADKU/VSP report | StructuredFacts {event_type, date, action, parts} | `context_fact_extractor.py` |
| Draft rule suggestion | Correction pattern + existing rules | Draft Rule text for engineer to approve | `rule_miner.py` |
| Review explanation | RuleTrace + ContextBundle + label | Human-readable explanation for review UI | `explanation_agent.py` |

**LLM does NOT:**
- detect anomalies
- assign final labels
- score candidate confidence as a probability
- replace rule engine logic
- see raw numerical signal arrays

---

## 7. Failure Modes and Fallbacks

| Failure | Detection | Fallback | Guardrail |
|---------|-----------|----------|-----------|
| Signal entirely missing or all zeros | QualityFlags.missing_pct = 1.0 | Skip asset; log warning | Reported in run summary |
| No change-points detected (flat series) | GlobalSeriesProfiler returns one regime | Treat full series as one candidate for quality review | Flag: `no_regime_structure` |
| WellProfile missing (new well) | Historical lookup returns empty | Use population median as fallback baseline; flag `no_well_history` | Lower deviation score confidence; route to review |
| All rules return no match | No rule fires | Label = `unknown`; routing = mandatory review; `abstain_reason = "no_rule_matched"` | Rule coverage metric tracks frequency |
| Rule conflict | Multiple priority-4 rules fire | Route to review; surface conflict in trace | Flag `rule_conflict` |
| Fact extraction LLM failure | Parse error or schema mismatch | Proceed with raw document text in review; flag `fact_extraction_failed` | Logs warning; no pipeline halt |
| LLM API unavailable | Timeout / 5xx | Fact extraction and explanations skip; rest of pipeline runs normally | Only LLM-dependent stages pause |
| New rule breaks confirmed example | Regression check fails in RuleRegistry | Block rule activation until engineer resolves | `regression_failed` flag; rule marked `inactive_pending_review` |

---

## 8. Label Taxonomy (PoC v1)

| Label | Meaning |
|-------|---------|
| `belt_break` | Belt failure signature confirmed by signal + rules + no confounder |
| `planned_stop` | Signal deviation explained by maintenance or planned shutdown |
| `planned_maintenance` | Equipment intervention (belt/rod replacement, service) |
| `sensor_issue` | Dropout, clamp saturation, or unreliable data quality |
| `stable_unusual_regime` | Novel but stable operating mode; not a failure |
| `unknown` | No rule fired or rule conflict; mandatory human review |

`insufficient_data_quality` is an asset-level terminal outcome during Stage 2, not a candidate label in the exported taxonomy.

Taxonomy is defined in `TaskSpec.label_taxonomy` and enforced at the Rule Engine output.

---

## 9. Technical Constraints (PoC)

| Constraint | Choice |
|-----------|--------|
| Team | 2–3 engineers |
| Timeline | Short; PoC validates framework on belt-break domain |
| Runtime | Python 3.11+, CPU only, single process |
| Deployment | Local or single instance; no container orchestration |
| LLM provider | Anthropic Claude (single provider) |
| Data | Verified ADKU/VSP reports as context; belt-break reference dataset |
| Peer comparison | Out of scope for PoC v1 (Phase 2) |
| Auto-label | Out of scope for PoC v1 |
| Starter ruleset | Hardcoded from domain expert knowledge; grows through review loop |
