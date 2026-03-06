# Agent System for Automatic Labeling of Oil Well Time-Series Data

## Problem Description

Machine learning models for equipment failure detection in oil production require large volumes of accurately labeled time-series data. In practice, acquiring such labels is one of the most significant bottlenecks in deploying predictive diagnostics.

Oil wells equipped with rod pumping systems (pumpjacks) produce **wattmeterograms** — time-series recordings of electric motor power consumption. The shape of these curves carries diagnostic information: a healthy pump produces a characteristic periodic pattern, while equipment failures such as rod breaks, belt breaks, idle motor states, and overload conditions distort the waveform in identifiable ways.

Today, labeling these curves is performed manually by production engineers. This process is:

- **Slow.** A single engineer may need several minutes per segment to inspect, cross-reference maintenance logs, and assign a label.
- **Expensive.** Domain experts capable of interpreting wattmeterograms are scarce and their time is costly.
- **Inconsistent.** Different engineers may assign different labels to the same ambiguous segment, especially under time pressure.
- **Incomplete.** Large archives of historical data remain unlabeled because retroactive annotation is impractical.

Naive anomaly detection applied directly to the raw signal produces an unacceptable number of **false positives**. A statistical deviation in the wattmeterogram may reflect a genuine equipment failure — or it may be explained by a planned maintenance event, a sensor calibration issue, or a known operational mode change. Without access to contextual knowledge (maintenance records, equipment metadata, engineering heuristics), an automated system cannot distinguish between these cases.

This project addresses the problem by building a **multi-agent system** that combines signal analysis with contextual reasoning to automatically propose labels for segments of wattmeterogram data. The system does not replace human judgment — it accelerates the labeling process by presenting engineers with pre-annotated segments, supporting explanations, and confidence estimates that they can accept, modify, or reject.

## Target Users

| Role | How They Use the System |
|------|------------------------|
| **Production Engineers** | Validate proposed labels against their operational knowledge; provide feedback to improve system accuracy. |
| **Equipment Diagnostics Specialists** | Use the system to accelerate retrospective analysis of historical wattmeterogram archives; identify patterns across wells and time periods. |
| **Data Scientists** | Consume the validated labeled datasets to train and evaluate classical ML models for failure detection and predictive maintenance. |

## PoC Demonstration

The proof-of-concept will demonstrate the following end-to-end workflow:

1. **Data ingestion.** The user uploads a wattmeterogram CSV file containing time-series power consumption data from one or more wells.
2. **Feature extraction.** The Time-Series Analysis Agent processes the raw signal, computing statistical features, periodicity metrics, and waveform descriptors.
3. **Anomaly detection.** The Event Detection Agent identifies candidate segments where the signal deviates from expected operating patterns.
4. **Context retrieval.** The Context Retrieval Agent queries a knowledge base of maintenance logs, equipment specifications, and engineering rules to gather evidence relevant to each candidate segment.
5. **Label proposal.** The Labeling Agent synthesizes signal-level evidence and retrieved context to propose a failure category label (e.g., "rod break", "belt break", "idle motor", "overload", "normal operation") for each segment.
6. **Confidence scoring and review.** The Review Agent estimates confidence in each proposed label and flags low-confidence cases for mandatory human review.
7. **Human validation.** The engineer reviews proposed labels through a simple interface, accepting, modifying, or rejecting each suggestion. Validated labels are exported as a structured dataset.

## Out of Scope

The following capabilities are explicitly **not** part of this PoC:

- **Real-time monitoring.** The system processes historical data in batch mode. It does not connect to live SCADA systems or provide real-time alerting.
- **Full-scale industrial deployment.** The PoC is designed for demonstration on a limited dataset. It does not address the scalability, reliability, or integration requirements of a production deployment.
- **Automatic control of equipment.** The system produces labels and recommendations only. It does not issue commands to well controllers, pumps, or any physical equipment.
- **Replacement of human experts.** The system is an assistive tool. All proposed labels require human validation before being used for model training.
- **Support for other sensor types.** The PoC focuses exclusively on wattmeterograms. Dynamograms, pressure curves, and other sensor modalities are not addressed.

## Repository Structure

```
oil-well-labeling-agent/
├── README.md
└── docs/
    ├── product-proposal.md
    └── governance.md
```

## License

This project is developed as a proof-of-concept for academic and research purposes.
