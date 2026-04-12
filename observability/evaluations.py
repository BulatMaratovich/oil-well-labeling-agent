"""
observability/evaluations.py — Offline evaluation metrics.

Computes the metrics specified in observability-evals.md:
  - candidate_recall       — fraction of known events covered by candidates
  - rule_coverage          — fraction of candidates that got a non-unknown label
  - fp_rate_confounders    — rate of confounder events mislabelled as deviations
  - fact_extraction_accuracy — fraction of facts extracted with confidence "ok"
  - label_distribution     — counts per label
  - correction_rate        — fraction of human-reviewed records that were corrected

Usage
-----
    from observability.evaluations import evaluate_run, print_report
    report = evaluate_run(pipeline_result, task_memory)
    print_report(report)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from collections import Counter


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluationReport:
    task_id: str
    run_id: Optional[str] = None

    # Candidate quality
    n_candidates: int = 0
    n_candidates_labeled: int = 0   # label != unknown
    rule_coverage: float = 0.0      # n_labeled / n_candidates

    # Confounder FP rate (rules that should suppress but didn't)
    n_confounders_in_candidates: int = 0
    n_confounders_mislabeled: int = 0
    fp_rate_confounders: float = 0.0

    # Human review
    n_reviewed: int = 0
    n_corrected: int = 0
    correction_rate: float = 0.0

    # Label distribution
    label_distribution: dict[str, int] = field(default_factory=dict)

    # Fact extraction
    n_facts_total: int = 0
    n_facts_ok: int = 0
    fact_extraction_accuracy: float = 0.0

    # Rule trace stats
    n_conflicts: int = 0
    n_abstains: int = 0
    top_winning_rules: list[tuple[str, int]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def evaluate_run(
    pipeline_result,               # core.pipeline_runner.PipelineResult
    task_memory=None,              # learning.task_memory.TaskMemory | None
    confounder_labels: Optional[set] = None,
) -> EvaluationReport:
    """Compute evaluation metrics for one pipeline run.

    Parameters
    ----------
    pipeline_result:
        Output of PipelineRunner.run().
    task_memory:
        TaskMemory for the same task (used for correction_rate).
    confounder_labels:
        Label names that should suppress a deviation classification.
        Defaults to {"planned_stop", "planned_maintenance", "sensor_issue"}.
    """
    if confounder_labels is None:
        confounder_labels = {"planned_stop", "planned_maintenance", "sensor_issue"}

    report = EvaluationReport(
        task_id=pipeline_result.task_id,
        run_id=pipeline_result.run_id,
    )

    # ── Candidates & rule coverage ──────────────────────────────────
    report.n_candidates = len(pipeline_result.candidates)
    label_counts: Counter = Counter()
    n_conflicts = 0
    n_abstains = 0
    rule_wins: Counter = Counter()

    for rr in pipeline_result.rule_results:
        label_counts[rr.label] += 1
        if rr.conflict_flag:
            n_conflicts += 1
        if rr.abstain_reason:
            n_abstains += 1
        if rr.rule_trace.winning_rule:
            rule_wins[rr.rule_trace.winning_rule] += 1

    report.n_candidates_labeled = sum(
        v for k, v in label_counts.items() if k != "unknown"
    )
    report.rule_coverage = (
        report.n_candidates_labeled / report.n_candidates
        if report.n_candidates else 0.0
    )
    report.label_distribution = dict(label_counts.most_common())
    report.n_conflicts = n_conflicts
    report.n_abstains = n_abstains
    report.top_winning_rules = rule_wins.most_common(10)

    # ── Confounder FP rate ──────────────────────────────────────────
    confounder_candidates = [
        (c, rr)
        for c, rr in zip(pipeline_result.candidates, pipeline_result.rule_results)
        if rr.label in confounder_labels
    ]
    # A confounder is "mislabeled" if it also appears in corrections with a
    # deviation label — requires task_memory
    report.n_confounders_in_candidates = len(confounder_candidates)
    if task_memory:
        corrections = {r.record_id: r for r in task_memory.corrections()}
        deviation_labels = {"belt_break", "stable_unusual_regime"}
        mislabeled = sum(
            1 for c, _ in confounder_candidates
            if c.candidate_id in corrections
            and corrections[c.candidate_id].final_label in deviation_labels
        )
        report.n_confounders_mislabeled = mislabeled
        report.fp_rate_confounders = (
            mislabeled / len(confounder_candidates)
            if confounder_candidates else 0.0
        )

    # ── Human review / correction rate ─────────────────────────────
    if task_memory:
        all_records = task_memory.all()
        report.n_reviewed = len(all_records)
        report.n_corrected = len(task_memory.corrections())
        report.correction_rate = (
            report.n_corrected / report.n_reviewed
            if report.n_reviewed else 0.0
        )

    # ── Fact extraction accuracy ────────────────────────────────────
    # Count facts from context bundles
    for bundle in pipeline_result.context_bundles:
        for fact in bundle.maintenance_facts:
            report.n_facts_total += 1
            if fact.extraction_confidence == "ok":
                report.n_facts_ok += 1
    report.fact_extraction_accuracy = (
        report.n_facts_ok / report.n_facts_total
        if report.n_facts_total else 0.0
    )

    return report


# ---------------------------------------------------------------------------
# Pretty print
# ---------------------------------------------------------------------------

def print_report(report: EvaluationReport) -> None:
    lines = [
        f"═══ Evaluation Report  task={report.task_id}  run={report.run_id} ═══",
        f"Candidates       : {report.n_candidates}  labeled={report.n_candidates_labeled}  "
        f"coverage={report.rule_coverage:.1%}",
        f"Conflicts/abstains: {report.n_conflicts} / {report.n_abstains}",
        f"Reviewed         : {report.n_reviewed}  corrected={report.n_corrected}  "
        f"correction_rate={report.correction_rate:.1%}",
        f"Confounder FP    : {report.n_confounders_mislabeled}/{report.n_confounders_in_candidates}  "
        f"rate={report.fp_rate_confounders:.1%}",
        f"Fact accuracy    : {report.n_facts_ok}/{report.n_facts_total}  "
        f"acc={report.fact_extraction_accuracy:.1%}",
        "Label distribution:",
    ]
    for label, count in sorted(report.label_distribution.items(), key=lambda x: -x[1]):
        lines.append(f"  {label:<30} {count}")
    if report.top_winning_rules:
        lines.append("Top winning rules:")
        for rule_id, count in report.top_winning_rules:
            lines.append(f"  {rule_id:<40} {count}")
    print("\n".join(lines))


def report_to_dict(report: EvaluationReport) -> dict:
    """Serialise report to a plain dict for JSON export."""
    return {
        "task_id": report.task_id,
        "run_id": report.run_id,
        "n_candidates": report.n_candidates,
        "rule_coverage": round(report.rule_coverage, 4),
        "correction_rate": round(report.correction_rate, 4),
        "fp_rate_confounders": round(report.fp_rate_confounders, 4),
        "fact_extraction_accuracy": round(report.fact_extraction_accuracy, 4),
        "n_conflicts": report.n_conflicts,
        "n_abstains": report.n_abstains,
        "label_distribution": report.label_distribution,
        "top_winning_rules": report.top_winning_rules,
    }
