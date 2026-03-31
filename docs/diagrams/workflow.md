# Workflow / Graph Diagram

End-to-end execution flow for the rule-based global-to-local labeling pipeline.

## Phase A — Discovery and Task Setup

```mermaid
flowchart TD
    START([Engineer starts new labeling task]) --> CHECK_TASK{TaskSpec exists\nfor this task_id?}

    CHECK_TASK -->|No| DISCOVER[DiscoveryAgent\nelicitation dialogue with engineer]
    CHECK_TASK -->|Yes, complete| LOAD_SPEC[Load existing TaskSpec\nfrom TaskMemory]

    DISCOVER --> BUILD_SPEC[TaskSpec Builder\ncreates TaskSpec from TaskClarification]
    BUILD_SPEC --> ADAPTER{Known domain\npack exists?}
    ADAPTER -->|Yes| BOOTSTRAP[DomainAdapter.bootstrap()\nfills defaults for equipment family]
    ADAPTER -->|No| GENERIC[Use generic adapter\nand user-supplied schema]
    BOOTSTRAP --> SAVE_SPEC[Save TaskSpec to TaskMemory]
    GENERIC --> SAVE_SPEC
    LOAD_SPEC --> PHASE_B

    SAVE_SPEC --> PHASE_B([Phase B: Full-Series Analysis])
```

## Phase B — Full-Series Understanding

```mermaid
flowchart TD
    PHASE_B([Phase B: Full-Series Analysis]) --> NORM[InputNormalizer\nCSV → CanonicalTimeSeries]
    NORM --> VALID{Schema valid?}
    VALID -->|No| ABORT([Abort: return ValidationReport])
    VALID -->|Yes| SANITIZE[SignalSanitizer\nclamp impossible values,\ninterpolate short gaps,\nflag dropout spans]
    SANITIZE --> QCHECK{Data quality\nacceptable?}
    QCHECK -->|No| ASSET_SKIP[Asset skipped:\ninsufficient_data_quality]
    QCHECK -->|Yes| PROFILE[GlobalSeriesProfiler\nchange-points + regime segmentation]
    PROFILE --> HISTORY[HistoricalProfileBuilder\nload per-well profile\nor population fallback]
    HISTORY --> CANDIDATES[CandidateEventDetector\nnew / atypical / abrupt / unusual regimes]
    CANDIDATES --> ANY_CAND{Any candidates?}

    ASSET_SKIP --> EXPORT_SKIP([Audit log only\nno label records exported])
    ANY_CAND -->|No| NO_CAND([Asset complete:\nno candidate label records])
    ANY_CAND -->|Yes| PHASE_C([Phase C: Candidate Analysis])
```

## Phase C — Candidate Analysis

```mermaid
flowchart TD
    PHASE_C([Phase C: Candidate Analysis]) --> LOOP{For each candidate}

    LOOP --> LOCAL[LocalSegmentAnalyzer\ndetailed local features]
    LOCAL --> LOOKUP[Structured retrieval\nmaintenance logs + metadata + rule docs]
    LOOKUP --> FACTS[ContextFactExtractor\nLLM parses free-text reports\ninto StructuredFacts]
    FACTS --> RULES[RuleEngine\napply explicit rules\nin priority order]
    RULES --> ROUTE{Routing}

    ROUTE -->|review| REVIEW_Q[Review queue]
    ROUTE -->|mandatory_review| MREVIEW_Q[Mandatory review queue]

    LOOP -->|more candidates| LOOP
    LOOP -->|done| PHASE_D([Phase D: Human Review])
```

## Phase D — Human Review and Rule Growth

```mermaid
flowchart TD
    PHASE_D([Phase D: Human Review]) --> QUEUES[Merge review queues]
    QUEUES --> ENGINEER[Engineer reviews candidate\nwith plot + regime context\n+ fact summary + rule trace\n+ similar confirmed cases]

    ENGINEER --> ACTION{Engineer action}
    ACTION -->|Accept| CONFIRM[Confirm label]
    ACTION -->|Modify| MODIFY[Correct label\ncapture correction_reason]
    ACTION -->|Reject| REJECT[Reject from export]
    ACTION -->|Ambiguous| AMBIG[Mark ambiguous]
    ACTION -->|Skip| SKIP[Return to queue end\nnot allowed for mandatory_review]

    CONFIRM --> UPDATE[Update TaskMemory\nconfirmed examples,\nwell profiles,\naudit log]
    MODIFY --> UPDATE
    REJECT --> UPDATE
    AMBIG --> UPDATE

    UPDATE --> RULE_MINE{Repeated correction\npattern detected?}
    RULE_MINE -->|Yes| DRAFT[RuleMiner\nsuggest draft rule]
    RULE_MINE -->|No| EXPORT
    DRAFT --> APPROVE{Engineer approves\ndraft rule?}
    APPROVE -->|Yes| REGRESSION[Run regression check\nagainst confirmed examples]
    APPROVE -->|No| EXPORT[Export labeled dataset\n+ task snapshot\n+ audit_log.jsonl]
    REGRESSION -->|Pass| ACTIVATE[Activate new rule version]
    REGRESSION -->|Fail| BLOCK[Block rule activation\nkeep for manual revision]
    ACTIVATE --> EXPORT
    BLOCK --> EXPORT
```

---

## Key Design Decisions Visible in Workflow

| Decision | Where it appears |
|----------|-----------------|
| Discovery before labeling | Phase A is the mandatory entry point for new tasks |
| Full-series analysis before local analysis | Phase B profiles the entire asset before any candidate-level features are extracted |
| Candidate-first labeling | Phase C only analyzes candidate segments, not every window |
| Rule Engine is the decision authority | Rule application precedes every review path |
| Human review is mandatory in PoC v1 | All routed candidates go through Phase D |
| Confirmed examples are secondary memory | Used in review UI and regression checks, not as a decision mechanism |
| Rule growth replaces few-shot growth | Corrections lead to draft rules and regression-tested rule updates |

## Error Branch Summary

| Branch | Trigger | Outcome |
|--------|---------|---------|
| Abort | Input schema invalid | Entire run stops; ValidationReport printed |
| Asset skipped | Data quality too poor | Audit log only; no label records exported |
| No candidates | All regimes within historical profile | Asset complete; no candidate label records exported |
| Mandatory review | Rule conflict, `unknown` fallback, processing failure, hard flags | Engineer must resolve before session ends |
| Fact extraction failure | LLM parse/schema failure | Continue with raw docs visible in review |
| Regression failure | New rule breaks confirmed examples | Block rule activation |
