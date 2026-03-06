# Product Proposal: Agent System for Automatic Labeling of Oil Well Time-Series Data

## 1. Problem Motivation

### The Labeling Bottleneck in Industrial Time-Series Data

Supervised machine learning models for equipment diagnostics depend on labeled training data. In the oil and gas industry, the primary source of diagnostic information for rod pumping systems is the **wattmeterogram** — a time-series recording of the electric motor power consumption during the pumping cycle. Characteristic distortions in these curves correspond to specific equipment states: normal operation, rod break, belt break, idle motor, overload, and other failure modes.

Labeling these time series is difficult and expensive for several reasons:

- **Domain expertise requirement.** Interpreting wattmeterograms requires knowledge of pumping mechanics, electrical engineering, and field operations. Only a small number of specialists possess this combination of skills.
- **Context dependency.** The same waveform anomaly can indicate a genuine failure or a benign operational event (e.g., a planned shutdown, a load test, a sensor recalibration). Correct labeling requires cross-referencing the time series with maintenance logs, equipment history, and operational schedules.
- **Volume mismatch.** Modern SCADA systems collect wattmeterograms continuously across hundreds or thousands of wells. The volume of unlabeled data far exceeds the capacity of human annotators.
- **Label ambiguity.** Transitional states, compound failures, and degraded sensor quality create ambiguous segments where even experts may disagree. Without structured guidelines and evidence trails, labeling consistency suffers.
- **Retroactive analysis cost.** Historical archives of wattmeterograms are a valuable resource for model training, but retroactive labeling is rarely prioritized because of its high cost relative to immediate operational needs.

### Why Agent Systems Are Useful Here

The labeling task is not purely algorithmic — it requires integrating heterogeneous sources of evidence (signal features, maintenance context, engineering rules) and making judgment calls under uncertainty. This makes it poorly suited to a single monolithic model but well-suited to a **multi-agent architecture** where specialized agents handle different aspects of the reasoning:

- **Decomposition of complexity.** Each agent focuses on a well-defined subtask (feature extraction, anomaly detection, context retrieval, label generation, confidence estimation), making the system easier to develop, test, and debug.
- **Tool integration.** Signal processing and statistical analysis are best performed by deterministic tools (Python libraries, statistical tests). Contextual reasoning and evidence synthesis are better handled by LLM-based reasoning. An agent architecture naturally combines both.
- **Transparency and auditability.** Each agent produces intermediate outputs that can be inspected, logged, and reviewed. This creates an evidence trail that supports human validation and system improvement.
- **Iterative refinement.** Agent pipelines can be extended incrementally — adding new failure types, new context sources, or improved detection methods — without redesigning the entire system.

## 2. Project Goal

Build a proof-of-concept multi-agent system that **automatically suggests labels for anomalous segments of oil well wattmeterograms** using a combination of:

- Statistical signal analysis (feature extraction, anomaly detection)
- Contextual retrieval from a knowledge base of maintenance logs, equipment metadata, and engineering rules (RAG)
- LLM-based reasoning to synthesize evidence and generate label proposals with explanations and confidence scores

The system is designed as an **assistive tool**: it proposes labels that human engineers validate. The validated labels form training datasets for classical ML models used in production diagnostics.

## 3. Success Metrics

### Product Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Reduction in manual labeling time | 50% reduction compared to fully manual labeling | Timed comparison on a reference dataset |
| Label acceptance rate | ≥ 70% of auto-suggested labels accepted by engineers without modification | Tracked through the validation interface |
| Engineer satisfaction | Qualitative positive feedback on usefulness of explanations and confidence scores | Post-demo survey |

### Agent Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| Anomaly detection recall | ≥ 90% of known anomalous segments detected | Evaluated against a pre-labeled test set |
| Label confidence calibration | Predicted confidence correlates with actual accuracy (calibration error < 0.15) | Reliability diagram on the test set |
| Retrieval relevance | ≥ 80% of retrieved context documents rated as relevant by engineers | Manual evaluation of retrieval results |
| False positive reduction | ≥ 50% reduction in false positives compared to signal-only detection | Comparison with baseline anomaly detector |

### Technical Metrics

| Metric | Target | Measurement Method |
|--------|--------|--------------------|
| P95 end-to-end latency | < 60 seconds per segment | Measured across the full pipeline |
| Pipeline success rate | ≥ 95% of submitted segments processed without errors | Error tracking in logs |
| System availability during demo | No unrecoverable failures during demonstration | Monitored during demo session |

## 4. Use Case Scenarios

### Scenario 1: Normal Operation Labeling

**Input.** A wattmeterogram segment showing a stable, periodic power consumption pattern.

**Expected behavior.** The Time-Series Analysis Agent extracts features that fall within normal operating ranges. The Event Detection Agent finds no significant anomalies. The Labeling Agent assigns the label **"normal_operation"** with high confidence (> 0.9). The Review Agent passes the label without flagging it for human review.

**Value.** Bulk labeling of normal segments saves engineers from inspecting thousands of routine records.

### Scenario 2: Rod Break Detection

**Input.** A wattmeterogram showing a sudden drop in load-bearing phase amplitude, followed by an irregular low-power pattern.

**Expected behavior.** The Event Detection Agent flags the segment as anomalous (sudden amplitude change). The Context Retrieval Agent finds no scheduled maintenance for the well during that time period but retrieves engineering rules describing rod break signatures. The Labeling Agent proposes the label **"rod_break"** with moderate-to-high confidence (0.7–0.9) and generates an explanation citing the signal features and absence of maintenance context. The Review Agent passes the label to the engineer for validation with supporting evidence.

**Value.** The system provides not just the label but a structured explanation that helps the engineer make a faster, more informed decision.

### Scenario 3: False Anomaly Due to Maintenance Event

**Input.** A wattmeterogram showing an abrupt shutdown and restart pattern during a period when scheduled maintenance was performed.

**Expected behavior.** The Event Detection Agent flags the segment as anomalous (unexpected shutdown). The Context Retrieval Agent finds a matching maintenance log entry for the well and time period. The Labeling Agent recognizes the maintenance context and proposes the label **"planned_maintenance"** instead of a failure label. Confidence is moderate (0.6–0.8) because the temporal overlap requires judgment.

**Value.** This is the core differentiator of the system — reducing false positives by incorporating contextual knowledge that a signal-only detector cannot access.

### Scenario 4: Missing or Noisy Sensor Data

**Input.** A wattmeterogram with missing data points, sensor dropouts, or high-frequency noise.

**Expected behavior.** The Time-Series Analysis Agent detects data quality issues (gaps, noise level above threshold). The system applies preprocessing (interpolation or filtering) where appropriate and flags quality concerns. If the data quality is too low for reliable analysis, the Labeling Agent assigns the label **"insufficient_data_quality"** and the Review Agent flags the segment for mandatory human review with an explanation of the quality issues found.

**Value.** The system avoids producing unreliable labels from poor data, maintaining dataset quality and engineer trust.

### Scenario 5: Overload Condition

**Input.** A wattmeterogram showing sustained elevated power consumption above the nominal operating range.

**Expected behavior.** The Event Detection Agent identifies persistent above-threshold power consumption. The Context Retrieval Agent checks for operational changes (new pump settings, fluid property changes) and retrieves engineering rules for overload conditions. The Labeling Agent proposes the label **"overload"** with supporting evidence. The Review Agent assesses confidence and routes to human validation.

**Value.** Overload conditions may develop gradually, making them easy to miss in manual review of large datasets.

## 5. Constraints

### Technical Constraints

| Constraint | Implication |
|------------|-------------|
| **PoC-scale dataset** | The system will be demonstrated on a dataset of 100–500 wattmeterogram segments. It is not optimized for datasets of millions of records. |
| **CPU-based analysis** | All signal processing and statistical analysis will run on CPU. No GPU resources are required or assumed. |
| **Simulated knowledge base** | The RAG knowledge base will contain a curated set of representative maintenance logs, equipment specs, and engineering rules — not a full production database. |
| **Single LLM provider** | The PoC will use a single LLM API for all reasoning tasks. Multi-provider fallback is out of scope. |

### Operational Constraints

| Constraint | Implication |
|------------|-------------|
| **Small engineering team** | 2–3 engineers developing the system. Architecture must be modular enough for parallel development. |
| **Limited development time** | 1–2 weeks for implementation. The system must prioritize core pipeline functionality over polish. |
| **No production infrastructure** | The system will run locally or on a single cloud instance. No container orchestration, load balancing, or production monitoring. |

## 6. Architecture Sketch

The system is organized as a **multi-agent pipeline** with six specialized agents coordinated by an orchestrator. Each agent has a defined responsibility, a set of tools it can invoke, and a structured output format.

### Agent Descriptions

#### Orchestrator Agent

**Role.** Controls the end-to-end workflow. Receives the input data, dispatches tasks to downstream agents in sequence, handles errors and retries, and assembles the final output.

**Responsibilities:**
- Parse and validate the input CSV
- Manage pipeline state (which segments have been processed, current stage)
- Route data between agents
- Aggregate results into the final label report
- Handle agent failures gracefully (log errors, skip segments, report partial results)

**Tools:** Pipeline state management, error handling utilities.

#### Time-Series Analysis Agent

**Role.** Extracts numerical features from raw wattmeterogram data to produce a structured representation suitable for downstream reasoning.

**Responsibilities:**
- Segment the time series into individual pumping cycles
- Compute statistical features: mean, variance, skewness, kurtosis of power consumption per cycle
- Compute periodicity metrics: cycle duration stability, frequency spectrum characteristics
- Compute waveform shape descriptors: peak-to-trough ratio, rise/fall time, area under curve
- Flag data quality issues: missing values, noise level, sensor dropouts

**Tools:** Python signal processing libraries (NumPy, SciPy), statistical analysis functions. This agent operates entirely through deterministic tool execution — no LLM reasoning is required.

#### Event Detection Agent

**Role.** Identifies candidate anomalous segments based on the extracted features.

**Responsibilities:**
- Compare extracted features against baseline operating profiles
- Apply threshold-based and statistical anomaly detection methods (z-score, IQR, change-point detection)
- Classify deviations by type: amplitude anomaly, frequency anomaly, waveform distortion, data quality issue
- Output a ranked list of candidate anomalous segments with deviation scores

**Tools:** Statistical anomaly detection libraries, threshold configuration. This agent primarily uses deterministic tools with optional LLM reasoning for ambiguous edge cases.

#### Context Retrieval Agent

**Role.** Retrieves relevant engineering knowledge and maintenance context for each candidate anomalous segment using a Retrieval-Augmented Generation (RAG) approach.

**Responsibilities:**
- Formulate retrieval queries based on the well identifier, time window, and detected anomaly type
- Search the knowledge base for: maintenance logs matching the well and time period, equipment specifications for the well's pump configuration, engineering rules describing failure mode signatures
- Rank and filter retrieved documents by relevance
- Summarize relevant context into a structured format for the Labeling Agent

**Knowledge base contents:**
- Maintenance and repair logs (date, well ID, work performed, parts replaced)
- Equipment metadata (pump model, rod string configuration, motor specifications)
- Engineering rules and heuristics for failure mode identification

**Tools:** Vector store for semantic search, keyword-based retrieval, document parsing utilities.

#### Labeling Agent

**Role.** Generates proposed labels for each candidate segment by synthesizing signal-level evidence and retrieved context.

**Responsibilities:**
- Receive the feature summary from the Time-Series Analysis Agent, the anomaly report from the Event Detection Agent, and the context summary from the Context Retrieval Agent
- Reason over the combined evidence to determine the most likely equipment state
- Assign a label from a predefined taxonomy: `normal_operation`, `rod_break`, `belt_break`, `idle_motor`, `overload`, `planned_maintenance`, `sensor_issue`, `unknown`
- Generate a natural-language explanation citing specific evidence (signal features, maintenance records, engineering rules)
- Estimate a raw confidence score based on evidence strength and consistency

**Tools:** LLM reasoning (this is the primary LLM-intensive agent), structured output formatting.

#### Review Agent

**Role.** Estimates confidence in each proposed label and decides whether the label can be auto-accepted or requires mandatory human validation.

**Responsibilities:**
- Evaluate the Labeling Agent's reasoning for consistency and completeness
- Calibrate the confidence score against predefined thresholds
- Apply routing rules: high-confidence labels (> 0.85) are auto-accepted with audit logging; medium-confidence labels (0.5–0.85) are presented for human review with supporting evidence; low-confidence labels (< 0.5) are flagged as uncertain and require mandatory review
- Check for known failure modes: contradictory evidence, missing context, hallucination indicators

**Tools:** LLM reasoning for consistency checking, threshold configuration, routing logic.

### System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      Orchestrator Agent                         │
│                  (workflow control, state mgmt)                  │
└──────┬──────────────┬───────────────┬──────────────┬────────────┘
       │              │               │              │
       ▼              ▼               ▼              ▼
┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
│Time-Series│  │    Event     │  │ Context  │  │ Labeling │
│ Analysis  │──▶  Detection   │──▶Retrieval │──▶  Agent   │
│  Agent    │  │    Agent     │  │  Agent   │  │          │
│ [Tools]   │  │[Tools + LLM] │  │[RAG+LLM]│  │  [LLM]   │
└──────────┘  └──────────────┘  └──────────┘  └────┬─────┘
                                                    │
                                                    ▼
                                              ┌──────────┐
                                              │  Review  │
                                              │  Agent   │
                                              │  [LLM]   │
                                              └────┬─────┘
                                                   │
                                                   ▼
                                            ┌────────────┐
                                            │   Human    │
                                            │ Validation │
                                            └────────────┘
```

## 7. Data Flow

### Pipeline Stages

The data flows through the system in a sequential pipeline. Each stage transforms the data and passes a structured output to the next stage.

```
Raw CSV ──▶ Feature Extraction ──▶ Anomaly Detection ──▶ Context Retrieval ──▶ Label Proposal ──▶ Confidence Review ──▶ Human Validation
```

### Detailed Data Flow

| Stage | Input | Processing | Output | Execution Mode |
|-------|-------|------------|--------|----------------|
| **1. Ingestion** | Raw CSV file (timestamps, power values, well ID) | Validate format, parse columns, segment into cycles | Structured time-series segments | Deterministic (Python) |
| **2. Feature Extraction** | Time-series segments | Statistical and waveform analysis | Feature vectors per segment (20–30 numerical features) | Deterministic (NumPy, SciPy) |
| **3. Anomaly Detection** | Feature vectors + baseline profiles | Statistical tests, threshold comparison, change-point detection | Ranked list of candidate anomalous segments with deviation scores | Deterministic (Python) with optional LLM for edge cases |
| **4. Context Retrieval** | Anomaly candidates (well ID, time window, anomaly type) | Query knowledge base, rank results, summarize context | Structured context summaries per candidate | RAG (vector search + LLM summarization) |
| **5. Label Proposal** | Feature summary + anomaly report + context summary | Multi-evidence reasoning, label assignment, explanation generation | Proposed labels with explanations and raw confidence scores | LLM reasoning |
| **6. Confidence Review** | Proposed labels with explanations | Consistency checking, confidence calibration, routing | Final labels with calibrated confidence and routing decision (auto-accept / human-review / mandatory-review) | LLM reasoning + rule-based thresholds |
| **7. Human Validation** | Labeled segments with explanations and confidence | Engineer review and correction | Validated labels for dataset export | Human decision |

### Tool vs. LLM Boundary

The system explicitly separates **deterministic tool execution** from **LLM-based reasoning**:

- **Stages 1–3** (Ingestion, Feature Extraction, Anomaly Detection) are primarily **tool-based**. They use Python libraries for signal processing and statistical analysis. Results are reproducible and verifiable. The Event Detection Agent may optionally invoke LLM reasoning for ambiguous edge cases, but the default path is fully deterministic.

- **Stage 4** (Context Retrieval) uses a **hybrid approach**. Document retrieval relies on vector similarity search (a tool). Summarization and relevance filtering use LLM reasoning.

- **Stages 5–6** (Label Proposal, Confidence Review) are primarily **LLM-based**. These stages require synthesizing heterogeneous evidence and making judgment calls that benefit from natural language reasoning. Structured output schemas constrain the LLM's responses to the predefined label taxonomy.

- **Stage 7** (Human Validation) is **human-driven** and outside the agent system.

### State Management

The Orchestrator Agent maintains a **pipeline state object** that tracks:

- Which segments have been processed and their current stage
- Intermediate outputs from each agent (for auditability and debugging)
- Error states and retry counts
- Final aggregated results

This state enables the system to resume processing after failures, generate audit logs, and provide progress reporting.
