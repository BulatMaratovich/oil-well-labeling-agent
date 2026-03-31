# Spec: Serving and Configuration

## Runtime Environment

| Parameter | Value |
|-----------|-------|
| Language | Python 3.11+ |
| Execution | Single process, synchronous pipeline |
| Compute | CPU only (no GPU required) |
| OS | Linux / macOS (Windows untested) |
| Memory | ~500 MB peak (NumPy arrays + ChromaDB in-process + ExampleStore embeddings) |
| Disk | ~200 MB per task (ChromaDB index + TaskMemory + ExampleStore embeddings + logs) |

## Project Structure

```text
oil-well-labeling-agent/
├── main.py                          ← CLI entry point
├── config/
│   ├── settings.py                  ← all configurable parameters
│   ├── settings.yaml                ← default values (committed, no secrets)
│   └── prompts/
│       ├── discovery_system.txt
│       ├── fact_extraction_system.txt
│       ├── explanation_system.txt
│       └── rule_draft_system.txt
├── core/
│   ├── task_manager.py
│   ├── canonical_schema.py
│   ├── pipeline_runner.py
│   ├── state_manager.py
│   └── policy_engine.py
├── signals/
│   ├── input_normalizer.py
│   ├── signal_sanitizer.py
│   ├── global_series_profiler.py
│   ├── historical_profile_builder.py
│   ├── candidate_event_detector.py
│   └── local_segment_analyzer.py
├── rules/
│   ├── rule_engine.py
│   ├── rule_registry.py
│   ├── rule_trace.py
│   └── rule_schemas.py
├── context/
│   ├── structured_lookup.py
│   ├── semantic_retriever.py
│   ├── context_fact_extractor.py
│   └── context_bundle.py
├── learning/
│   ├── example_store.py
│   ├── task_memory.py
│   └── rule_miner.py
├── agents/
│   ├── discovery_agent.py
│   └── explanation_agent.py
├── ui/
│   ├── discovery_cli.py
│   ├── review_ui.py
│   └── export_ui.py
├── observability/
│   ├── logger.py
│   └── metrics.py
├── adapters/
│   ├── domain_adapter_base.py
│   └── rod_pump_belt_break.py
├── data/
│   ├── chroma/                      ← ChromaDB index (gitignored)
│   ├── knowledge_base/              ← source JSON/MD files (committed)
│   ├── baselines/                   ← population fallback baselines (committed)
│   ├── tasks/
│   │   └── {task_id}/
│   │       ├── memory.json          ← TaskMemory: TaskSpec, ruleset, profiles, examples
│   │       ├── ruleset.json         ← versioned ruleset
│   │       ├── ruleset_history.json ← rule change log
│   │       ├── examples.json        ← ExampleStore records
│   │       └── example_embeddings.npy
│   └── outputs/                     ← exported labeled datasets (gitignored)
├── runs/                            ← per-run RunState + audit_log.jsonl (gitignored)
├── scripts/
│   ├── build_index.py               ← one-time KB indexing into ChromaDB
│   ├── build_baselines.py           ← build population fallback baselines
│   └── evaluate.py                  ← offline evaluation against ground truth
└── tests/
```

## Installation

```bash
pip install -r requirements.txt
# Key dependencies:
# anthropic>=0.25.0
# chromadb>=0.4.0
# sentence-transformers>=2.7.0
# ruptures>=1.1.9             (PELT change-point detection)
# numpy>=1.26.0
# scipy>=1.13.0
# scikit-learn>=1.4.0
# structlog>=24.0.0
# pydantic>=2.0.0
# rich>=13.0.0
# matplotlib>=3.8.0           (signal plots in review UI)
```

## Running the Pipeline

```bash
# 1. Set required environment variable
export ANTHROPIC_API_KEY="sk-ant-..."

# 2. Build knowledge base index (once, or when KB files change)
python scripts/build_index.py

# 3. Build population fallback baselines
python scripts/build_baselines.py --data data/knowledge_base/reference_normal.csv

# 4. Run discovery dialogue (new task only)
python main.py discover --task-id rod_pump_belt_break_v1

# 5. Run labeling pipeline
python main.py label --task-id rod_pump_belt_break_v1 --input data/inputs/wells_batch.csv

# 6. Resume interrupted run
python main.py resume --run-id 2024-01-15_143022

# 7. Open review queue only
python main.py review --run-id 2024-01-15_143022

# 8. Export confirmed labeled dataset
python main.py export --task-id rod_pump_belt_break_v1 --run-id 2024-01-15_143022
```

## Configuration Parameters

All parameters have defaults in `settings.yaml`. Override via environment variables or `--config` flag.

```yaml
# settings.yaml

model:
  discovery:       "claude-sonnet-4-6"
  fact_extraction: "claude-sonnet-4-6"
  explanation:     "claude-sonnet-4-6"
  rule_draft:      "claude-sonnet-4-6"

signal:
  dropout_threshold:          5.0      # kW; values below this count as dropout
  min_dropout_duration_s:    60        # minimum consecutive dropout to flag as span
  max_interpolation_gap_s:  300        # gaps shorter than this are interpolated

profiling:
  pelt_model:               "rbf"
  pelt_penalty:             10         # change-point penalty; higher = fewer change-points
  pelt_min_segment_size:    12         # minimum data points per regime
  regime_cluster_method:    "kmeans"
  regime_n_clusters:        5          # max regime types per asset

thresholds:
  anomaly_z_threshold:       2.5       # z-score for atypical amplitude detection
  data_quality_min:          0.3       # asset is skipped when missing_pct exceeds this threshold
  unusual_duration_p_low:    0.10      # below p10 duration → unusual duration candidate
  unusual_duration_p_high:   0.90      # above p90 duration → unusual duration candidate

retrieval:
  maintenance_window_days_before: 7
  maintenance_window_days_after:  1
  max_maintenance_docs:           5
  top_k_rule_candidates:         10    # ChromaDB n_results
  top_k_rules_reranked:           2    # rules passed to ContextBundle
  min_rule_similarity:            0.40
  max_doc_age_days:             730

fact_extraction:
  max_report_tokens:         1500      # truncation limit for report text
  min_confidence_for_rules:  "medium"  # below this: fact does not trigger confounder rules

examples:
  top_k_similar:              3
  min_example_similarity:     0.60
  embedding_model:            "sentence-transformers/all-MiniLM-L6-v2"

rule_engine:
  unknown_on_no_match:        true
  unknown_on_conflict:        true

timeouts:
  input_normalization_s:     10
  signal_sanitization_s:     15
  global_series_profiling_s: 30
  historical_profile_build_s: 10
  candidate_event_detection_s: 10
  local_segment_analysis_s:  10
  context_fact_extraction_s: 40
  rule_engine_s:              5
  asset_total_s:            180
  candidate_total_s:         90
  llm_call_s:                30

retries:
  llm_max_retries:            3
  llm_backoff_base_s:         2
  fact_extraction_schema_retries: 2
  profiling_retries:          1

paths:
  chroma_dir:           "./data/chroma"
  knowledge_base_dir:   "./data/knowledge_base"
  baselines_dir:        "./data/baselines"
  tasks_dir:            "./data/tasks"
  outputs_dir:          "./data/outputs"
  runs_dir:             "./runs"
  prompts_dir:          "./config/prompts"
  adapters_dir:         "./adapters"

profile:
  max_profile_age_days: 365
  population_fallback:  true
  per_well_min_regimes: 3     # minimum observed regimes to use per-well profile
```

## Secrets Management

| Secret | Location | Notes |
|--------|----------|-------|
| `ANTHROPIC_API_KEY` | Environment variable only | Never in code, config files, or logs |

No other secrets exist in the PoC.

## Model Versioning

Model IDs are pinned in `settings.yaml`. To update:
1. Change model ID in `settings.yaml`
2. Run fact extraction accuracy spot-check on 50 samples
3. Commit with spot-check results in commit message

## Dependency Pinning

All dependencies are pinned with exact versions in `requirements.txt` (generated via `pip freeze`).
