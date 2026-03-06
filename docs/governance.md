# Governance: Risk Management, Safety, and Data Handling

This document defines the risk management framework, logging policies, data handling rules, and safety controls for the Agent System for Automatic Labeling of Oil Well Time-Series Data.

## 1. Risk Register

The following table identifies key risks associated with the system, assessed by probability (likelihood of occurrence), impact (severity of consequences), detectability (how easily the risk can be identified before causing harm), and proposed mitigations.

| Risk | Probability | Impact | Detection | Mitigation | Residual Risk |
|------|-------------|--------|-----------|------------|---------------|
| **Incorrect labeling of failures** — The Labeling Agent assigns a wrong failure category to an anomalous segment (e.g., labels a rod break as an overload). | Medium | High — Incorrect labels propagate into training datasets, degrading downstream ML model performance. If undetected, this can lead to missed failures in production. | Medium — Detectable through human validation, but only if the engineer disagrees with the proposed label. Subtle misclassifications may go unnoticed. | (1) Mandatory human validation for all labels below the high-confidence threshold. (2) Structured explanations with cited evidence, enabling engineers to verify reasoning. (3) Periodic audits of accepted labels against ground truth. | Low — With human-in-the-loop validation, most incorrect labels are caught before entering the training dataset. Residual risk exists for plausible but incorrect labels that pass review. |
| **Hallucinated explanations** — The LLM generates plausible-sounding but factually incorrect explanations to justify a label (e.g., citing a maintenance event that did not occur). | Medium | High — Hallucinated explanations can mislead engineers into accepting incorrect labels. They undermine trust in the system. | Low — Hallucinations are difficult to detect automatically because they are designed to sound plausible. Detection relies on engineers verifying claims against source records. | (1) Explanations must cite specific retrieved documents with identifiers that can be traced back to the knowledge base. (2) The Review Agent checks whether cited documents exist and are relevant. (3) Explanations that reference undocumented events are flagged. | Medium — Citation verification reduces but does not eliminate hallucination risk. LLMs can still generate misleading characterizations of real documents. |
| **Prompt injection via uploaded metadata** — An attacker embeds malicious instructions in CSV metadata fields, well names, or maintenance log entries that are processed by the LLM agents. | Low | High — Successful injection could cause the system to produce arbitrary labels, leak information from the knowledge base, or bypass safety controls. | Low — Prompt injections can be subtle and may not trigger obvious errors. Detection requires systematic input sanitization and output monitoring. | (1) All external text inputs (CSV headers, metadata fields, retrieved documents) are sanitized before being included in LLM prompts. (2) LLM outputs are validated against the predefined label taxonomy — free-form outputs are rejected. (3) Agent permissions are restricted to read-only access to the knowledge base. | Low — With input sanitization and output validation, the attack surface is significantly reduced. Residual risk exists for novel injection techniques. |
| **Low-quality sensor data** — Wattmeterogram data contains excessive noise, missing values, sensor dropouts, or calibration drift that prevents reliable analysis. | High | Medium — Low-quality data leads to unreliable features, spurious anomaly detections, and low-confidence labels. If not handled, it wastes engineer review time. | High — Data quality issues are detectable through statistical checks in the Time-Series Analysis Agent (missing value counts, noise level metrics, range validation). | (1) The Time-Series Analysis Agent performs explicit data quality checks before feature extraction. (2) Segments with quality scores below a minimum threshold are labeled as `insufficient_data_quality` and excluded from automated labeling. (3) Data quality metrics are included in the output for transparency. | Low — Quality gating prevents unreliable segments from entering the labeling pipeline. Engineers are informed about excluded segments and can inspect them manually. |
| **Knowledge base staleness** — The RAG knowledge base contains outdated maintenance logs or equipment metadata that no longer reflects the current state of the field. | Medium | Medium — Stale context can lead to incorrect label proposals (e.g., failing to recognize a recently repaired well as operational). | Medium — Detectable if timestamps in retrieved documents are significantly older than the query time window. | (1) Retrieved documents include timestamps that are displayed to the engineer. (2) The Context Retrieval Agent deprioritizes documents older than a configurable threshold. (3) The system logs cases where no recent context is available. | Medium — In a PoC setting with a curated knowledge base, staleness is controlled. In production, this risk would require an automated data ingestion pipeline. |
| **Pipeline failure or timeout** — An agent in the pipeline crashes, hangs, or exceeds the latency budget, blocking processing of subsequent segments. | Medium | Low — In a PoC context, pipeline failures delay the demo but do not cause data loss. In production, this would have higher impact. | High — Failures are detectable through timeouts, error codes, and health checks. | (1) The Orchestrator Agent implements per-agent timeouts. (2) Failed segments are logged and skipped, allowing the pipeline to continue with remaining segments. (3) Partial results are reported rather than failing silently. | Low — Graceful degradation ensures that individual failures do not halt the entire pipeline. |

## 2. Logging Policy

All system activity is logged to support auditability, debugging, and system improvement. Logs are structured (JSON format) and organized by pipeline run.

### What Is Logged

#### Input Logging
- **Raw input metadata:** File name, file size, number of rows, column names, well identifiers, time range. Raw data values are not logged in full to avoid excessive storage and potential data sensitivity concerns — only summary statistics are recorded.
- **Input validation results:** Parse errors, missing columns, data type mismatches, quality check outcomes.

#### Intermediate Agent Output Logging
- **Time-Series Analysis Agent:** Extracted feature vectors (numerical summaries per segment), data quality scores, processing time.
- **Event Detection Agent:** List of candidate anomalous segments, deviation scores, anomaly type classifications, detection method used.
- **Context Retrieval Agent:** Retrieval queries issued, document IDs returned, relevance scores, number of documents retrieved, context summaries generated.
- **Labeling Agent:** Proposed label, explanation text, raw confidence score, evidence citations, reasoning trace.
- **Review Agent:** Calibrated confidence score, routing decision (auto-accept / human-review / mandatory-review), consistency check results, flagged issues.

#### Final Decision Logging
- **Proposed label and confidence** for each segment.
- **Human validation decision:** accepted, modified (with new label), or rejected.
- **Final label** after human validation.
- **Timestamp and engineer identifier** for the validation action.

### Log Retention and Access

- Logs are stored locally during the PoC. In a production system, they would be stored in a centralized logging service with access controls.
- Logs do not contain raw production data values — only derived features, labels, and metadata.
- Log access is restricted to the development team during the PoC phase.

## 3. Data Handling Policy

### Industrial Data Sensitivity

Oil well production data, including wattmeterograms, maintenance logs, and equipment metadata, may constitute **commercially sensitive information**. Production volumes, equipment configurations, and failure histories can reveal operational details that oil and gas operators consider proprietary.

### Data Handling Rules for the PoC

| Rule | Description |
|------|-------------|
| **Data minimization** | The system processes only the data required for the labeling task. Raw wattmeterogram values are not persisted beyond the pipeline session unless explicitly exported by the user. |
| **No external transmission** | Production data is not transmitted to external services other than the configured LLM API. The LLM API is used only for reasoning tasks — raw time-series values are not sent to the LLM. Only extracted features, anomaly summaries, and context documents are included in LLM prompts. |
| **Knowledge base isolation** | The RAG knowledge base is stored locally and is not exposed through any external interface. |
| **Synthetic data option** | For demonstration and testing, the system supports synthetic wattmeterogram data to avoid the need for real production data. |
| **Export controls** | Labeled datasets are exported only through an explicit user action. The export includes labels, confidence scores, and explanations — not raw data. |

### Production Deployment Considerations (Out of Scope for PoC)

In a production deployment, the data handling policy would need to address:
- Encryption at rest and in transit
- Role-based access control for different user types
- Data residency requirements (on-premises vs. cloud)
- Retention and deletion policies
- Compliance with operator-specific data governance frameworks

## 4. Safety Controls

### Confidence Thresholds

The Review Agent applies a three-tier confidence routing system:

| Confidence Range | Routing Decision | Rationale |
|-----------------|------------------|-----------|
| **High (> 0.85)** | Auto-accepted with audit logging | Strong, consistent evidence from both signal analysis and contextual retrieval. Low risk of error. |
| **Medium (0.5 – 0.85)** | Presented for human review with supporting evidence | Evidence is suggestive but not conclusive. Engineer judgment is needed to confirm. |
| **Low (< 0.5)** | Flagged for mandatory human review | Insufficient or contradictory evidence. The system cannot make a reliable recommendation. |

Confidence thresholds are configurable. For the initial PoC, conservative thresholds are recommended to maximize human oversight until the system's accuracy is validated.

### Human-in-the-Loop Validation

Human validation is the primary safety mechanism:

- **All labels are reviewable.** Even auto-accepted labels are logged and can be audited retroactively.
- **Engineers can override any label.** The system's proposal is a suggestion, not a decision.
- **Explanations support informed review.** Each label is accompanied by a natural-language explanation citing specific signal features, retrieved context documents, and engineering rules. This enables engineers to evaluate the reasoning, not just the conclusion.
- **Rejection feedback loop.** When an engineer rejects or modifies a label, the original proposal and the correction are both logged. This data can be used to improve the system in future iterations.

### Tool Execution Constraints

Agent tool use is restricted to prevent unintended side effects:

| Constraint | Enforcement |
|------------|-------------|
| **Read-only data access** | Agents can read from the knowledge base and input data but cannot modify, delete, or create records in external systems. |
| **No network access beyond LLM API** | Agents cannot make arbitrary HTTP requests. Network access is limited to the configured LLM API endpoint and the local knowledge base. |
| **No file system writes outside designated output directory** | The Orchestrator Agent restricts file writes to a designated output directory. Agents cannot modify system files, configuration, or other project files. |
| **Execution timeouts** | Each agent has a configurable timeout (default: 30 seconds for tool agents, 60 seconds for LLM agents). Timed-out operations are logged and treated as failures. |
| **No code execution from external input** | The system does not execute code generated by the LLM or extracted from input data. All tool invocations use predefined, parameterized functions. |

## 5. Prompt Injection Defense

### Threat Model

The system processes external text from multiple sources:

- **CSV file contents** — column headers, well identifiers, metadata fields
- **Maintenance logs** — free-text descriptions of repair work, operator notes
- **Equipment metadata** — model names, configuration parameters, textual annotations
- **Engineering rules** — natural-language descriptions of failure mode signatures

Any of these text sources could contain adversarial content designed to manipulate the LLM agents' behavior — intentionally (by a malicious actor) or accidentally (by poorly formatted data).

### Defense Measures

#### Input Sanitization

All external text is sanitized before inclusion in LLM prompts:

- **Character filtering.** Control characters, unusual Unicode sequences, and escape sequences are stripped or replaced.
- **Length limits.** Text fields are truncated to predefined maximum lengths to prevent prompt stuffing.
- **Instruction delimiter enforcement.** External text is enclosed in clearly delimited blocks (e.g., XML-style tags) within the prompt, separated from system instructions. The LLM is explicitly instructed to treat delimited content as data, not as instructions.

#### Output Validation

LLM outputs are validated against expected schemas:

- **Label taxonomy enforcement.** The Labeling Agent's output must contain a label from the predefined set (`normal_operation`, `rod_break`, `belt_break`, `idle_motor`, `overload`, `planned_maintenance`, `sensor_issue`, `unknown`). Any output that does not match is rejected and logged.
- **Confidence range validation.** Confidence scores must be numeric values between 0.0 and 1.0. Out-of-range values are rejected.
- **Explanation structure validation.** Explanations must follow a predefined structure (evidence summary, cited documents, conclusion). Unstructured or anomalous outputs are flagged for review.

#### Architectural Separation

- **No agent-to-agent prompt passthrough.** External text retrieved by the Context Retrieval Agent is summarized before being passed to the Labeling Agent. Raw retrieved documents are not inserted directly into downstream prompts.
- **System prompts are immutable.** Agent system prompts are defined in code and cannot be modified by input data or intermediate outputs.
- **Least privilege.** Each agent has access only to the tools and data required for its specific role. The Context Retrieval Agent cannot invoke labeling functions; the Labeling Agent cannot issue retrieval queries.
