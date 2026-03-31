# C4 Component Diagram

Zooms into the core pipeline and limited LLM-assisted components.

```mermaid
C4Component
    title Core Pipeline + Limited LLM Components — Component View

    Container_Boundary(core, "Core Pipeline") {
        Component(pipeline_runner, "PipelineRunner", "Python class", "Top-level entry point: loads TaskSpec, runs all stages, writes final output")
        Component(task_manager, "TaskManager", "Python class", "Creates and loads TaskSpec; manages task versions")
        Component(state_mgr, "StateManager", "Python class", "Maintains RunState and supports resume")
        Component(input_normalizer, "InputNormalizer", "Python functions", "Maps raw CSV/DB extract to CanonicalTimeSeries")
        Component(signal_sanitizer, "SignalSanitizer", "Python functions", "Clamps impossible values, interpolates short gaps, produces QualityFlags")
        Component(global_profiler, "GlobalSeriesProfiler", "Python functions", "Detects change-points and stable regimes on full series")
        Component(profile_builder, "HistoricalProfileBuilder", "Python functions", "Loads or updates per-well history and fallback profiles")
        Component(candidate_detector, "CandidateEventDetector", "Python functions", "Finds new or atypical regimes relative to WellProfile")
        Component(local_analyzer, "LocalSegmentAnalyzer", "Python functions", "Extracts detailed features for candidate segments only")
        Component(structured_lookup, "StructuredLookup", "Python functions", "Exact retrieval of maintenance logs and equipment metadata")
        Component(rule_engine, "RuleEngine", "Python functions", "Applies explicit rules in priority order and returns RuleResult")
        Component(rule_registry, "RuleRegistry", "Python functions", "Loads active ruleset and runs regression checks before activation")
        Component(rule_trace, "RuleTraceBuilder", "Python functions", "Records rules evaluated, fired, blocked, and conflict status")
        Component(example_store, "ExampleStore", "Python functions", "Stores confirmed examples and retrieves similar cases for review UI")
    }

    Container_Boundary(llm_assist, "Limited LLM Assistance") {
        Component(discovery_agent, "DiscoveryAgent", "Python + LLM", "Structured interview to build TaskSpec")
        Component(fact_extractor, "ContextFactExtractor", "Python + LLM", "Parses free-text maintenance report into StructuredFacts")
        Component(rule_miner, "RuleMiner", "Python + LLM", "Suggests draft rules from repeated correction patterns")
        Component(explanation_agent, "ExplanationAgent", "Python + LLM", "Generates human-readable explanation from RuleTrace and ContextBundle")
        Component(llm_client, "LLMClient", "Anthropic SDK wrapper", "Single client for all allowed LLM calls with schema validation and retries")
    }

    Rel(pipeline_runner, task_manager, "Create/load TaskSpec")
    Rel(pipeline_runner, state_mgr, "Read/write RunState")
    Rel(pipeline_runner, input_normalizer, "Normalize input")
    Rel(pipeline_runner, signal_sanitizer, "Sanitize full series")
    Rel(pipeline_runner, global_profiler, "Profile regimes")
    Rel(pipeline_runner, profile_builder, "Load well history")
    Rel(pipeline_runner, candidate_detector, "Detect candidate events")
    Rel(pipeline_runner, local_analyzer, "Extract candidate-local features")
    Rel(pipeline_runner, structured_lookup, "Retrieve docs and metadata")
    Rel(pipeline_runner, fact_extractor, "Extract structured facts from matched reports")
    Rel(pipeline_runner, rule_engine, "Evaluate candidates")
    Rel(rule_engine, rule_registry, "Load active rules")
    Rel(rule_engine, rule_trace, "Build trace")
    Rel(pipeline_runner, example_store, "Fetch similar confirmed cases for review UI")
    Rel(discovery_agent, llm_client, "Execute LLM call")
    Rel(fact_extractor, llm_client, "Execute LLM call")
    Rel(rule_miner, llm_client, "Execute LLM call")
    Rel(explanation_agent, llm_client, "Execute LLM call")
```

## DiscoveryAgent Flow

```text
receive(task_id, existing_task_memory_or_none)
  ├── if TaskSpec exists and is complete:
  │     return TaskSpec
  ├── ask structured questions about:
  │     equipment, signals, unit of analysis, confounders, context sources
  ├── llm_client.call(...)
  ├── validate structured TaskClarification
  ├── bootstrap defaults from DomainAdapter if known family
  └── return TaskSpec
```

## Candidate Evaluation Flow

```text
receive(CandidateEvent, LocalFeatures, ContextBundle, WellProfile)
  ├── RuleRegistry loads active ruleset
  ├── RuleEngine evaluates:
  │     Priority 0 quality exclusions
  │     Priority 1 sensor issues
  │     Priority 2 confounders
  │     Priority 3 stable unusual regime
  │     Priority 4 true deviations
  │     Priority 5 fallback unknown
  ├── RuleTraceBuilder records:
  │     rules_evaluated, rules_fired, rules_blocked, winning_rule, conflict
  └── return RuleResult
```

## ContextFactExtractor Flow

```text
receive(MaintenanceDocument)
  ├── wrap raw report text in XML
  ├── llm_client.call(..., output_schema=StructuredFacts)
  ├── on schema failure: retry
  ├── on exhausted retries:
  │     return StructuredFacts(extraction_confidence="failed")
  └── return StructuredFacts
```
