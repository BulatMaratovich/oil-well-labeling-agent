# Data Flow Diagram

How data is transformed, what is stored, and what is logged at each stage.

```mermaid
flowchart LR
    subgraph INPUT
        CSV["CSV / DB extract"]
        TASKSPEC["TaskSpec\n(loaded from TaskMemory)"]
    end

    subgraph STAGE1["Stage 1: Normalize + Sanitize"]
        CANON["CanonicalTimeSeries\n{asset_id, timestamp,\nsignal_name, value, unit}"]
        CLEAN["CleanSignal + QualityFlags\n{missing_pct, dropout_spans,\nnoise_level, clamp_events}"]
        CSV -->|InputNormalizer| CANON
        CANON -->|SignalSanitizer| CLEAN
    end

    subgraph STAGE2["Stage 2: Full-Series Profiling"]
        REGIMES["RegimeSequence\nlist[Regime]\n{start, end, regime_type,\nduration_h}"]
        PROFILE["WellProfile\nbaseline_regimes,\nknown_stops,\nknown_replacements"]
        CLEAN -->|GlobalSeriesProfiler| REGIMES
        TASKSPEC --> PROFILE
    end

    subgraph STAGE3["Stage 3: Candidate Detection"]
        CANDS["list[CandidateEvent]\n{segment, deviation_type,\ndeviation_score}"]
        REGIMES -->|compare vs history| CANDS
        PROFILE -->|historical baseline| CANDS
    end

    subgraph STAGE4["Stage 4: Candidate Analysis"]
        LOCAL["LocalFeatures\npower stats,\ntransition sharpness,\nsegment duration"]
        DOCS["matched_docs\nmaintenance logs,\nmetadata,\nengineering rules"]
        FACTS["StructuredFacts\n{event_type, event_date,\naction, parts_replaced,\nconfidence}"]
        BUNDLE["ContextBundle\n{maintenance_facts,\nequipment_metadata,\nrule_docs, flags}"]
        CANDS -->|LocalSegmentAnalyzer| LOCAL
        CANDS -->|StructuredLookup| DOCS
        DOCS -->|ContextFactExtractor| FACTS
        DOCS --> BUNDLE
        FACTS --> BUNDLE
    end

    subgraph STAGE5["Stage 5: Rule Evaluation"]
        RESULT["RuleResult\n{label, rule_trace,\nrouting, flags,\nabstain_reason}"]
        LOCAL -->|RuleEngine| RESULT
        BUNDLE -->|RuleEngine| RESULT
        PROFILE -->|RuleEngine| RESULT
    end

    subgraph STAGE6["Stage 6: Human Review"]
        SIMILAR["Similar confirmed cases\nfrom ExampleStore\n(review context only)"]
        FINAL["FinalLabelRecord\n{final_label, source,\ncorrection_reason}"]
        LOCAL -->|similarity search| SIMILAR
        RESULT --> FINAL
        SIMILAR --> FINAL
    end

    subgraph STORAGE["Persistent Storage"]
        STATE["runs/{run_id}/state.json\nRunState checkpoint"]
        TASK_MEM["data/tasks/{task_id}/memory.json\nTaskMemory: TaskSpec,\nruleset, well_profiles,\nconfirmed examples,\nconfounders"]
        RULESET["data/tasks/{task_id}/ruleset.json\nactive versioned rules"]
        AUDIT["audit_log.jsonl\nappend-only events"]
        EXPORT["labeled_dataset.json / parquet / csv"]
    end

    STAGE1 -->|write sanitized asset state| STATE
    STAGE2 -->|write regime sequence| STATE
    STAGE3 -->|write candidates| STATE
    STAGE4 -->|write context bundle| STATE
    STAGE5 -->|write rule result| STATE
    STAGE6 -->|write final labels| EXPORT
    STAGE6 -->|update examples + profiles| TASK_MEM
    STAGE5 -->|load active rules| RULESET

    STAGE1 -.->|log quality flags| AUDIT
    STAGE2 -.->|log regime boundaries| AUDIT
    STAGE3 -.->|log candidate counts| AUDIT
    STAGE4 -.->|log fact extraction| AUDIT
    STAGE5 -.->|log rule trace + routing| AUDIT
    STAGE6 -.->|log engineer action| AUDIT
```

## What Goes to the LLM (and What Doesn't)

| Data | Sent to LLM? | Notes |
|------|-------------|-------|
| Raw `power_kW` time-series array | **No** | Never leaves signal processing engine |
| Full-series regime description | No | Computed deterministically and stored locally |
| Local candidate features | No for labeling; may be summarized for explanations only | Rule Engine consumes them directly |
| Maintenance report raw text | Yes | Only for `ContextFactExtractor`; parsed into StructuredFacts |
| Engineer correction patterns | Yes | Only for `RuleMiner` draft suggestion |
| Rule trace + final label | Yes | Only for human-readable explanation generation |
| Confirmed examples | No for label decision | Used only in review UI and regression testing |

## What Is Stored

| Store | Format | Retention | Access |
|-------|--------|-----------|--------|
| `runs/{run_id}/state.json` | JSON | Per run; retained for audit | PipelineRunner only |
| `audit_log.jsonl` | JSON Lines | Permanent; one file per run | Dev team, post-run analysis |
| `labeled_dataset.json` / exported dataset files | JSON / Parquet / CSV | Permanent; final output | Export to downstream ML pipeline |
| `data/tasks/{task_id}/memory.json` | JSON | Persistent; updated after every run | TaskManager and review flow |
| `data/tasks/{task_id}/ruleset.json` | JSON | Persistent; versioned | RuleRegistry |
| ChromaDB index | Binary (SQLite + embeddings) | Persistent; rebuilt via `build_index.py` | Retriever only |

## What Is Logged in `audit_log.jsonl`

Every entry includes `run_id`, `task_id`, `asset_id` or `candidate_id`, `stage`, `timestamp_utc`.

Stage-specific fields:

- **signal_sanitized:** `missing_pct`, `dropout_spans`, `clamp_events`
- **regimes_detected:** `change_point_count`, `regime_count`
- **candidates_found:** `candidate_count`, `deviation_types`
- **facts_extracted:** `doc_count`, `fact_count`, `confidence`
- **rule_result:** `label`, `winning_rule`, `rules_evaluated`, `rules_fired`, `conflict`, `routing`
- **human_action:** `action`, `proposed_label`, `final_label`, `correction_reason`
- **rule_added / rule_deactivated / regression_passed / regression_failed**
