# C4 Context Diagram

Shows the system boundary, users, and external dependencies.

```mermaid
C4Context
    title Industrial Time-Series Expert Labeling Framework — System Context

    Person(engineer, "Field Engineer / Analyst", "Defines the task, reviews candidate labels, corrects rules, and exports validated datasets")

    System_Boundary(sys, "Industrial Time-Series Expert Labeling Framework") {
        System(labeling_framework, "Rule-Based Labeling Framework", "Global-to-local pipeline: signal sanitization, regime profiling, candidate detection, context fact extraction, rule evaluation, and human review")
    }

    System_Ext(anthropic, "Anthropic API", "LLM inference for discovery dialogue, fact extraction, draft rule suggestions, and review explanations")
    System_Ext(scada, "SCADA / Data Export", "CSV or DB export of active power time series; read-only input for PoC")
    System_Ext(kb, "Knowledge Base (local)", "Maintenance reports, equipment metadata, engineering rules, verified reports")

    ContainerDb(task_memory, "TaskMemory Store", "Local persistent store: TaskSpec, ruleset, well profiles, confirmed examples, confounders")

    Rel(engineer, labeling_framework, "Starts tasks, runs review, approves rule changes", "CLI")
    Rel(scada, labeling_framework, "Provides historical time-series exports", "Manual export")
    Rel(labeling_framework, kb, "Reads maintenance reports, metadata, engineering rules", "Local disk / ChromaDB")
    Rel(labeling_framework, task_memory, "Reads/writes rules, profiles, confirmed examples", "Local disk")
    Rel(labeling_framework, anthropic, "Limited LLM calls: discovery, fact extraction, explanation, draft rules", "HTTPS")
```

## Boundary Notes

- **In scope (PoC):** discovery dialogue, TaskSpec management, signal sanitization, regime profiling, candidate detection, fact extraction, rule-based labeling, review UI, persistent TaskMemory, structured logging
- **Out of scope (PoC):** real-time monitoring, production database integration, peer comparison in critical path, autonomous labeling, multi-user access control
- **Security boundary:** raw active power arrays never leave the local process; only report text and compact structured context are sent to the LLM API
- **Reference domain:** belt-break detection on rod pumping units; the framework is extensible to other deviation types and installation families
