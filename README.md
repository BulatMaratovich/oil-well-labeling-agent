# Industrial Time-Series Expert Labeling Framework

## Problem

Industrial diagnostic teams have large archives of time-series data but too few high-quality labels. Manual labeling is slow, inconsistent, and produces datasets biased toward obvious cases — missing the look-alike non-failures that matter most for downstream model training.

The core difficulty is that **correct labeling is relational, not intrinsic**:

- The same power drop can be a belt break, a planned stop, a sensor dropout, or a new stable operating mode.
- Correct interpretation requires combining signal behavior with equipment history, maintenance records, and operational context.
- A "strange window" is not an anomaly. An unexplained regime that deviates from a well's own history — and is not covered by any known confounder — is a candidate for review.

Naive window classifiers fail because they answer the wrong question: "is this window unusual?" instead of "is this regime new, atypical, and unexplained for this specific well?"

## Product Idea

A **global-to-local rule-based expert labeling framework** for industrial time-series data.

The framework:

1. **Analyzes the full series** to find stable regimes and structural change-points — before looking at individual segments
2. **Compares regimes against per-well history** to identify what is genuinely new or atypical
3. **Extracts structured facts** from maintenance reports and equipment records
4. **Applies explicit, versioned rules** to propose labels with a full rule trace
5. **Routes candidates to engineers** with signal plots, regime context, historical comparison, and fact summaries
6. **Grows the ruleset** from engineer corrections — not from model weight updates

The labeling logic lives in auditable if/then rules, not in LLM weights or model confidences.

## What This Is Not

- Not a window-by-window anomaly detector
- Not an LLM classifier
- Not a replacement for domain experts
- Not a real-time alerting system

Its purpose: give diagnostic teams a systematic workflow for building **contrastive, high-quality labeled datasets** — including true failures, look-alike confounders, and genuinely ambiguous cases.

## Reference Domain

The first reference domain is **belt-break labeling for rod pumping units** using active power time series and ADKU/VSP maintenance reports. This validates the framework without constraining its architecture to one failure type or one installation family.

## Pipeline

```text
Raw Signal
  → Input Normalizer           normalize schema
  → Signal Sanitizer           clean, flag quality issues
  → Global Series Profiler     change-points, regime clustering
  → Historical Profile Builder per-well baseline, known confounders
  → Candidate Event Detector   new/atypical regimes vs well history
  → Local Segment Analyzer     detailed features for candidates only
  → Context Fact Extractor     parse maintenance reports → structured facts
  → Rule Engine                apply versioned if/then rules, produce rule trace
  → Human Review               plot + regime context + rule trace + accept/modify/reject
  → Rule / Profile Update      grow ruleset from corrections, update well profiles
```

## Label Taxonomy (PoC v1)

| Label | Meaning |
|-------|---------|
| `belt_break` | Belt failure confirmed by signal + rules + no confounder |
| `planned_stop` | Deviation explained by maintenance or planned shutdown |
| `planned_maintenance` | Equipment service event |
| `sensor_issue` | Dropout, saturation, or unreliable data |
| `stable_unusual_regime` | Novel but stable operating mode; not a failure |
| `unknown` | No rule fired or rule conflict; mandatory human review |

`insufficient_data_quality` is an asset-level skip outcome during signal sanitization, not an exported candidate label.

## Role of LLM

LLM is used in four narrow, well-defined places:

| Where | What |
|-------|------|
| Discovery dialogue | Elicit task definition from engineer → TaskSpec |
| Context Fact Extractor | Parse free-text ADKU/VSP reports → StructuredFacts |
| Rule Miner | Draft new rule text from correction patterns (engineer approves) |
| Explanation Agent | Generate human-readable explanation for review UI |

LLM does **not** detect anomalies, assign labels, or score candidate confidence.

## Repository Structure

```text
oil-well-labeling-agent/
├── README.md
├── docs/
│   ├── system-design.md          ← architecture source of truth
│   ├── product-proposal.md
│   ├── governance.md
│   ├── data-summary.md
│   ├── specs/
│   │   ├── agent-orchestrator.md
│   │   ├── memory-context.md
│   │   ├── retriever.md
│   │   ├── tools-apis.md
│   │   ├── observability-evals.md
│   │   └── serving-config.md
│   └── diagrams/
└── Some data/
    └── Belt/
```

## Status

Design-phase. Architecture source of truth: [docs/system-design.md](docs/system-design.md).

## License

Proof-of-concept for academic and research purposes.
