# Product Proposal: Industrial Time-Series Expert Labeling Framework

## 1. Problem Motivation

Industrial ML systems for equipment diagnostics require labeled historical time-series data. In practice, labels are scarce, expensive, inconsistent, and hard to trust — not primarily because labeling is slow, but because it is genuinely difficult:

- The same signal pattern can be a real failure, a planned stop, a sensor dropout, or a new stable operating mode.
- Correct interpretation requires combining signal behavior with maintenance records, equipment history, and field context.
- Different engineers label the same ambiguous segment differently.
- Existing labeled datasets are often biased toward obvious cases, missing the look-alike non-failures that matter most for model training.

### Why This Is Not a Classification Problem

The standard framing — "train a model to classify windows as normal or failure" — misses the core difficulty.

A window is not labeled in isolation. Correct labeling depends on:
- **What is normal for this well**, not for a generic pump
- **What happened around this time** (maintenance, replacements, seasonal load changes)
- **Whether the pattern is new** or has been seen before on this well
- **Whether the deviation is sustained** or a transient artifact

Naive window classifiers produce unacceptable false positive rates precisely because they cannot answer these questions.

### What Is Actually Needed

A labeling system that works like a domain expert:

1. Looks at the full series history before judging individual segments
2. Compares against what is normal for this specific well
3. Checks the maintenance and operational record before calling anything a failure
4. Applies explicit, auditable rules that engineers can inspect and correct
5. Distinguishes true deviations from look-alike confounders
6. Produces a contrastive labeled dataset: not just positives and negatives, but subtly-wrong cases too

---

## 2. Product Thesis

This project builds a **global-to-local rule-based expert labeling framework** for industrial time-series data.

The product does not replace domain experts. It gives them a structured, fast, auditable workflow:

- Formalizes the labeling task as a `TaskSpec` (signals, labels, confounders, context sources, rules)
- Analyzes the full series globally to find regimes and detect candidates
- Retrieves and extracts relevant context facts from maintenance records
- Applies explicit versioned rules to propose labels with a full rule trace
- Presents candidates to engineers with signal plots, regime context, historical comparisons, and fact summaries
- Grows the ruleset from engineer corrections, not from statistical model updates

**The core claim:** a small, well-designed ruleset — built and maintained by engineers — produces better, more trustworthy labeled data than a large implicit model.

---

## 3. Project Goal

Build a proof-of-concept framework that demonstrates the following on the belt-break reference domain:

1. Full-series analysis yields better candidates than window-by-window anomaly scoring
2. Explicit rules with a full trace are more auditable and correctable than model confidences
3. Distinguishing true failures from look-alike confounders is tractable with the right context
4. The labeled dataset produced includes contrastive negatives useful for downstream CatBoost training
5. The framework is reusable: a second deviation type or installation family needs a new TaskSpec and adapter, not a core redesign

---

## 4. Success Metrics

### Product Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Candidate recall | ≥ 90% of verified belt-break events appear as candidates | Evaluation against ground truth from verified ADKU/VSP |
| False positive rate on confounders | planned_stop / sensor_issue mislabeled as belt_break < 10% | Ground truth comparison |
| Rule coverage | ≥ 80% of reviewed candidates resolved by a fired rule (not `unknown`) | Audit log |
| Engineer review time | 50% reduction vs fully manual labeling | Timed comparison on reference dataset |
| Contrastive label balance | Dataset contains at least 4 of 6 label classes with ≥ 10 examples each | Export statistics |

### Quality Metrics

| Metric | Target | Method |
|--------|--------|--------|
| `precision@accepted` | ≥ 75% accepted labels match ground truth | Post-labeling ground truth comparison |
| Unknown-case quality | ≥ 90% of `unknown` cases are genuinely ambiguous or novel | Manual audit of unknown set |
| Fact extraction accuracy | ≥ 85% of extracted facts match source document | Spot check of 50 random extractions |
| Regression pass rate | 100% of confirmed examples still correctly labeled after rule updates | Automated regression check |

### Framework Metrics

| Metric | Target | Method |
|--------|--------|--------|
| Rule reuse | Belt-break ruleset requires no core code changes to add a second deviation type | Integration test |
| Profile persistence | WellProfile correctly loaded and updated across runs | Resume test |
| End-to-end latency P95 | < 120 seconds per candidate (including human review) | Timing in audit log |

---

## 5. Use Case Scenarios

### Scenario 1: Full Series Analysis Finds What Window-by-Window Misses

**Input.** A 6-month active power series for one rod pumping well.

**Expected behavior.** The Global Series Profiler detects a structural change-point on day 47. The Historical Profile Builder finds this regime type has not appeared in the well's history. The Candidate Event Detector flags it. The local analysis and context retrieval reveal no maintenance event. The Rule Engine fires the belt_break rule and routes to review.

**Value.** A window classifier looking at 2-hour windows around day 47 sees ambiguous data. The global profiler sees a regime that is genuinely new for this well.

---

### Scenario 2: Confounder Correctly Excluded

**Input.** A well showing an abrupt power drop followed by near-zero readings for 6 hours.

**Expected behavior.** The Candidate Event Detector flags the segment. The Context Fact Extractor parses a maintenance report and extracts: `{event_type: planned_stop, date: same day, asset_id: same well}`. The Rule Engine fires the planned_stop rule at Priority 2, blocking the belt_break rule. Label = `planned_stop`.

**Value.** The key differentiator: the system avoids a false positive because it has explicit context facts and explicit rules for confounders — not because an LLM guessed correctly.

---

### Scenario 3: Engineer Corrects a Rule, System Updates

**Input.** Engineer reviews a case labeled `planned_stop` but corrects it to `belt_break`. Correction note: "The maintenance report refers to the previous day, not this event."

**Expected behavior.** The correction is logged with the reason. The Rule Miner detects the pattern: planned_stop rule is firing on maintenance records from adjacent days. It drafts a rule update: "Planned_stop exclusion applies only if maintenance record date overlaps candidate segment, not if within ±1 day." Engineer approves. Updated rule is versioned. Regression check runs against all confirmed examples.

**Value.** The system learns from corrections by updating rules — producing an explicit, auditable, human-approved change to the labeling logic.

---

### Scenario 4: Stable Unusual Regime

**Input.** A well that has operated at 30% lower power for 3 months. No maintenance record explains it.

**Expected behavior.** The Global Series Profiler detects a long stable regime. The Candidate Event Detector flags the transition as a candidate. The Rule Engine checks: the new regime is stable (low internal variance, long duration), no failure signatures in local features, no context explanation. Label = `stable_unusual_regime`. Routed to review.

**Value.** The system produces a useful label that is neither `belt_break` nor `normal`. This class is critical for training CatBoost to distinguish developing degradation from abrupt failures.

---

### Scenario 5: Unknown on Genuinely Ambiguous Case

**Input.** A signal with moderate deviation, no maintenance context, ambiguous local features, and a pattern that partially matches two rules.

**Expected behavior.** The Rule Engine detects a rule conflict (belt_break rule and stable_unusual_regime rule both partially match). Label = `unknown`, routing = mandatory review. Review UI shows both matching rules and why each partially applies.

**Value.** The system does not force a label when evidence is insufficient. The ambiguous case is stored separately and can inform future rule refinement.

---

## 6. Constraints

### Technical Constraints

| Constraint | Implication |
|------------|-------------|
| CPU-only runtime | Signal processing, profiling, and rule evaluation run without GPU |
| Single LLM provider | Anthropic for discovery and fact extraction; no multi-provider fallback |
| Local-first deployment | No cloud services except LLM API |
| PoC-scale data | Belt-break reference domain; hundreds to low thousands of candidate events |
| Weak ground truth | Verified ADKU/VSP reports are the source of truth; engineer acceptance alone is not enough |

### Architecture Constraints

| Constraint | Implication |
|------------|-------------|
| LLM not in critical path for labeling | Pipeline must produce label proposals without LLM if fact extraction fails |
| Rules must be auditable | Every label must be traceable to a rule_id and a rule_trace |
| No auto-label in PoC v1 | All candidates go to engineer review |
| No peer comparison in PoC v1 | Phase 2 extension |
| Starter ruleset hardcoded from domain knowledge | Ruleset grows from engineer corrections over time |

### Operational Constraints

| Constraint | Implication |
|------------|-------------|
| Small team (2–3 engineers) | Modular architecture; clear stage boundaries |
| Short timeline | PoC validates framework, not all future extensions |
| No production infrastructure | Local files, local vector store, single-process Python |

---

## 7. Architecture Summary

The framework has ten pipeline stages organized into four phases:

**Phase 1 — Signal understanding (global)**
InputNormalizer → SignalSanitizer → GlobalSeriesProfiler → HistoricalProfileBuilder

**Phase 2 — Candidate identification (global-to-local)**
CandidateEventDetector → LocalSegmentAnalyzer

**Phase 3 — Labeling (context + rules)**
ContextFactExtractor → RuleEngine

**Phase 4 — Review and learning**
HumanReview → Rule/Profile Update

LLM is used in Phase 1 (discovery), Phase 3 (fact extraction from free text, explanation generation for review), and as a draft rule assistant after corrections. It is not used for anomaly detection or label assignment.

---

## 8. Reference Domain and Expansion Path

**PoC v1 reference domain:**
- Equipment: rod pumping units
- Signal: active power time series
- Task: belt-break deviation labeling
- Context: ADKU/VSP maintenance reports, equipment metadata
- Starter rules: hardcoded from domain expert knowledge
- Confounders: planned_stop, planned_maintenance, sensor_issue, stable_unusual_regime

**Expansion path:**
1. Validate framework on belt-break; grow ruleset through review loop
2. Add a second deviation type (e.g., idle motor) as new rules in the same TaskSpec
3. Add a second installation family via a new DomainAdapter with a new starter ruleset
4. Add peer comparison as a Phase 2 context signal

The core pipeline, rule engine, memory, and review loop do not change between expansions.

---

## 9. What This Product Is Not

- Not a one-click automatic failure detector
- Not a universal classifier that generalizes from zero examples
- Not a replacement for domain experts
- Not a real-time monitoring or alerting system
- Not a production MLOps stack

Its purpose: give industrial diagnostic teams a systematic, auditable, rule-based workflow for building high-quality, contrastive labeled datasets from historical time-series data.
