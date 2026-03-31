# Spec: Observability and Evaluations

## Structured Logging

All events are written to `runs/{run_id}/audit_log.jsonl` as newline-delimited JSON via `structlog`.

### Event Types

#### Pipeline lifecycle

```json
{"event": "run_started",    "run_id": "...", "task_id": "...", "ruleset_version": "...", "input_file": "...", "asset_count": 42}
{"event": "run_completed",  "run_id": "...", "asset_count": 42, "candidate_count": 18, "review_queue_size": 14, "duration_s": 312}
```

#### Per-asset signal stages

```json
{"event": "signal_sanitized",   "asset_id": "...", "missing_pct": 0.02, "dropout_spans": 1, "clamp_events": 0}
{"event": "regimes_detected",   "asset_id": "...", "change_point_count": 3, "regime_type_count": 2}
{"event": "candidates_found",   "asset_id": "...", "candidate_count": 2, "deviation_types": ["novel_regime", "atypical_amplitude"]}
{"event": "no_candidates",      "asset_id": "...", "reason": "all_regimes_within_profile"}
{"event": "asset_skipped",      "asset_id": "...", "reason": "insufficient_data_quality", "missing_pct": 0.72}
```

#### Per-candidate context and rule stages

```json
{"event": "facts_extracted",    "candidate_id": "...", "doc_count": 2, "fact_count": 2, "confidence": ["high", "medium"]}
{"event": "fact_extraction_failed", "candidate_id": "...", "doc_id": "...", "error": "schema_mismatch"}
{"event": "rule_result",        "candidate_id": "...", "label": "belt_break", "winning_rule": "rule_belt_break_v2",
                                 "rules_evaluated": 7, "rules_fired": 1, "conflict": false, "abstain_reason": null,
                                 "flags": [], "routing": "review"}
```

#### Human review

```json
{"event": "human_action",  "candidate_id": "...", "proposed_label": "belt_break", "final_label": "belt_break",
                            "action": "accept", "correction_reason": null, "routing": "review",
                            "engineer_id": "session_7", "ts": "2024-01-15T15:42:00Z"}
{"event": "human_action",  "candidate_id": "...", "proposed_label": "belt_break", "final_label": "planned_stop",
                            "action": "modify", "correction_reason": "VSP report confirms scheduled belt replacement on same day",
                            "routing": "review", "engineer_id": "session_7", "ts": "..."}
```

#### Rule governance

```json
{"event": "rule_added",           "rule_id": "...", "version": "1.5", "source": "engineer_approved", "approved_by": "session_7"}
{"event": "rule_deactivated",     "rule_id": "...", "reason": "replaced_by_rule_belt_break_v3"}
{"event": "regression_passed",    "rule_id": "...", "examples_checked": 47}
{"event": "regression_failed",    "rule_id": "...", "affected_example_ids": ["ex_012", "ex_031"]}
{"event": "profile_updated",      "asset_id": "...", "regime_count": 5, "known_stops_added": 1}
{"event": "rule_draft_suggested", "pattern": "planned_stop firing on ±1d boundary", "draft_rule_id": "draft_001", "source_candidates": ["cand_007"]}
```

#### Errors

```json
{"event": "stage_failed",   "stage": "global_series_profiling", "asset_id": "...", "error": "ProfilingError", "pipeline_continues": true}
{"event": "llm_unavailable","stage": "context_fact_extraction",  "retries_exhausted": true, "fallback": "raw_docs_only"}
```

---

## Operational Metrics

Computed from audit_log at run end and reported in `runs/{run_id}/metrics.json`.

| Metric | Formula | Target |
|--------|---------|--------|
| Assets processed | `completed / total` | 100% |
| Asset failure rate | `failed / total` | < 5% |
| Candidate rate | `total_candidates / total_assets` | Task-specific |
| Unknown-case rate | `unknown / reviewed` | Track; alert if > 30% |
| Fact extraction success rate | `high+medium confidence / total extractions` | ≥ 80% |
| Rule coverage | `rule_fired / total_candidates` | ≥ 80% |
| Rule conflict rate | `conflicts / total_candidates` | < 10% |
| Review acceptance rate | `accepted / reviewed` | Track |
| Override rate | `modified / reviewed` | Track; alert if > 20% |
| Rejection rate | `rejected / reviewed` | Track; alert if > 15% |

---

## Evaluation Protocol

### Candidate Recall

**What it measures:** Does the pipeline surface real events as candidates before the Rule Engine runs?

```
candidate_recall = |{ground_truth_events found in CandidateEvents}| / |ground_truth_events|
```

- Ground truth: verified belt-break events from confirmed ADKU/VSP reports
- A ground truth event is "found" if at least one CandidateEvent overlaps its date range
- Target: ≥ 90%
- This metric is independent of Rule Engine quality — it measures the signal layer only

### Rule Coverage

**What it measures:** What fraction of candidates are resolved by a fired rule (not labeled `unknown`)?

```
rule_coverage = |{candidates with winning_rule != null}| / |total_candidates|
```

- Target: ≥ 80% after initial ruleset is tuned
- Low rule coverage → ruleset needs expansion
- Measured per run; trend over time shows ruleset maturity

### False Positive Rate on Confounders

**What it measures:** How often does the system label a known confounder (planned_stop, sensor_issue) as a true failure?

```
FP_confounder = |{confirmed_confounders labeled as belt_break or other failure}| / |confirmed_confounders|
```

- Ground truth for confounders: engineer-confirmed planned_stop / sensor_issue labels
- Target: < 10%
- This is the key differentiator from a naive anomaly detector

### Unknown-Case Quality

**What it measures:** Are `unknown` cases genuinely difficult, or is the system falling back to `unknown` on easy cases?

```
unknown_case_quality = |{unknown cases rated "genuinely ambiguous" by expert audit}| / |unknown cases audited|
```

- Method: spot-check 20% of `unknown` cases with domain expert
- Target: ≥ 90% rated genuinely ambiguous or novel
- Low quality → Rule Engine is too conservative; no-match / conflict handling needs tuning

### Fact Extraction Accuracy

**What it measures:** Do extracted StructuredFacts match the source document content?

```
fact_accuracy = |{extracted facts verified correct}| / |extracted facts spot-checked|
```

- Method: spot-check 50 random extractions against source documents
- Verify: event_type, event_date, asset_id match the document
- Target: ≥ 85%
- Low accuracy → prompt or schema needs revision

### Precision@Accepted

**What it measures:** Of all labels that engineers accepted (not modified), how many match ground truth?

```
precision_at_accepted = |{accepted labels matching ground truth}| / |total accepted labels|
```

- Ground truth: verified ADKU/VSP reports (not engineer acceptance itself)
- Target: ≥ 75%
- Note: this requires a separate ground truth set, not just the acceptance decision

### Regression Pass Rate

**What it measures:** Do new rules break previously validated cases?

```
regression_pass_rate = |{rule activations where regression_check_passed}| / |total rule activations|
```

- Target: 100% (regression check must pass before activation)
- Any failure blocks rule activation and is logged

---

## What Is Not a Primary Metric

| Excluded metric | Reason |
|-----------------|--------|
| LLM confidence / priority_score | There is no LLM confidence in the new design; Rule Engine produces deterministic labels |
| Label acceptance rate as accuracy proxy | Acceptance rate measures engineer workflow speed, not label correctness |
| Schema retry rate | LLM is not in the critical labeling path; retry rate is an operational health metric only |
| AUC / F1 on window classifier | There is no window classifier |

---

## Observability Gaps (PoC v1)

The following are known limitations that would need to be addressed in production:

- No real-time dashboard; metrics computed post-run only
- No automated alert on candidate recall drop (requires ground truth set to be preloaded)
- No drift detection on well profiles (manual audit required)
- No per-rule precision tracking across runs (ruleset analytics are post-hoc)
- No A/B evaluation of rule variants

---

## Demo Readiness Checklist

Before the PoC is considered demonstrable:

- [ ] Candidate recall ≥ 90% on reference belt-break dataset
- [ ] Rule coverage ≥ 80% on reference dataset
- [ ] FP rate on confounders < 10% on reference dataset
- [ ] Fact extraction accuracy ≥ 85% on 50-sample spot-check
- [ ] Regression pass rate = 100% (no rule has been activated with a failed regression)
- [ ] Review UI shows signal plot, regime context, rule trace, and context facts for each candidate
- [ ] At least one engineer correction has triggered a rule draft suggestion
- [ ] Labeled export contains all 5+ label classes with ≥ 5 examples each
