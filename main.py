"""
Universal Time-Series Labeling Agent — Demo

Demonstrates the full pipeline:
  1. Build device profiles from synthetic historical data
  2. Seed the knowledge base with labeled examples from known devices
  3. Run the pipeline on a new device (cold-start) with an anomalous window
  4. Show the label proposal, reasoning, and routing decision

Requires: ANTHROPIC_API_KEY environment variable
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from demo.generate_data import build_dataset
from src.models.schemas import TimeSeriesWindow
from src.orchestrator import LabelingPipeline, load_scenario
from src.tools.profile_store import ProfileStore
from src.tools.signal_processing import compute_window_stats, compute_relative_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

DB_PATH = "data/profiles.db"
BELT_BREAK_SCENARIO = "scenarios/belt_break.yaml"


def print_separator(title: str = "") -> None:
    width = 70
    if title:
        print(f"\n{'─' * 3} {title} {'─' * (width - len(title) - 5)}")
    else:
        print("─" * width)


def build_knowledge_base(pipeline: LabelingPipeline, dataset: dict) -> None:
    """
    Step 1: Build device profiles and seed the labeled example store.

    In production this would happen continuously as data streams in.
    Here we do it in bulk from synthetic historical data.
    """
    print_separator("Building Device Knowledge Base")

    for device_id, data in dataset["devices"].items():
        cfg = data["config"]
        ts_normal, vals_normal = data["normal"]
        ts_break, vals_break = data["belt_break"]
        ts_stop, vals_stop = data["planned_stop"]

        # Build operating profile from normal data
        pipeline.store.update_profile_samples(
            device_id=device_id,
            new_samples=vals_normal,
        )

        # Seed belt break example
        pipeline.seed_example(
            device_id=device_id,
            scenario_id="belt_break",
            label="belt_break",
            timestamps=ts_break,
            values=vals_break,
            description=(
                f"Motor at idle after belt break. "
                f"Power: {sum(vals_break)/len(vals_break):.2f} kW "
                f"vs normal {cfg['operating_mean']:.2f} kW "
                f"({100*cfg['idle_fraction']:.0f}% of normal). "
                f"Very stable signal."
            ),
        )

        # Seed planned stop example
        pipeline.seed_example(
            device_id=device_id,
            scenario_id="belt_break",
            label="planned_stop",
            timestamps=ts_stop,
            values=vals_stop,
            description=(
                f"Motor fully stopped. "
                f"Power: {sum(vals_stop)/len(vals_stop):.3f} kW "
                f"(near-zero, only control electronics). "
                f"Stop level vs belt-break idle: {cfg['stop_level']:.3f} vs "
                f"{cfg['operating_mean']*cfg['idle_fraction']:.3f} kW."
            ),
        )

        print(
            f"  ✓ {device_id}: profile built, examples seeded "
            f"(op={cfg['operating_mean']:.2f} kW, "
            f"idle={cfg['operating_mean']*cfg['idle_fraction']:.2f} kW, "
            f"stop={cfg['stop_level']:.3f} kW)"
        )

    total_devices = len(dataset["devices"])
    print(f"\n  Knowledge base ready: {total_devices} devices with labeled examples")


def run_belt_break_demo(pipeline: LabelingPipeline, dataset: dict) -> None:
    """
    Step 2: Run the pipeline on a new device experiencing a belt break.

    This device has no prior belt break examples — the agent must generalize
    from similar devices in the knowledge base.
    """
    demo = dataset["demo"]
    device_id = demo["device_id"]
    cfg = demo["config"]
    ts_normal, vals_normal = demo["normal"]
    pre_ts, pre_vals = demo["pre_window"]
    ts_break, vals_break = demo["belt_break_window"]
    ts_stop, vals_stop = demo["planned_stop_window"]

    print_separator(f"Demo Device: {device_id}")
    print(f"  Device description: {cfg['description']}")
    print(f"  Historical data: {len(vals_normal)} normal-operation samples")
    print(f"  No prior belt break examples for this device")

    # Build profile for demo device from normal operation history
    pipeline.store.update_profile_samples(
        device_id=device_id,
        new_samples=vals_normal,
    )

    scenario = load_scenario(BELT_BREAK_SCENARIO)

    # ── Case 1: Belt Break ──────────────────────────────────────────────
    print_separator("Case 1: Actual Belt Break")

    actual_idle = sum(vals_break) / len(vals_break)
    print(f"  Window stats: mean={actual_idle:.3f} kW, "
          f"std={sum((v - actual_idle)**2 for v in vals_break)**0.5/len(vals_break):.3f} kW")
    print(f"  Normal operating level: {cfg['operating_mean']:.3f} kW")
    print(f"  Level ratio: {actual_idle/cfg['operating_mean']:.2f}x of normal")
    print(f"  Samples in window: {len(vals_break)}")

    window_break = TimeSeriesWindow(
        device_id=device_id,
        timestamps=ts_break,
        values=vals_break,
        scenario_id="belt_break",
        pre_timestamps=list(pre_ts),
        pre_values=list(pre_vals),
    )

    print("\n  Running labeling pipeline (Claude is reasoning)...")
    result_break = pipeline.run(window=window_break, scenario=scenario)

    _print_result(result_break)

    # ── Case 2: Planned Stop (should NOT be labeled as belt break) ──────
    print_separator("Case 2: Planned Stop (false positive check)")

    actual_stop = sum(vals_stop) / len(vals_stop)
    print(f"  Window stats: mean={actual_stop:.4f} kW, near-zero")
    print(f"  Normal operating level: {cfg['operating_mean']:.3f} kW")
    print(f"  Level ratio: {actual_stop/cfg['operating_mean']:.3f}x of normal")

    window_stop = TimeSeriesWindow(
        device_id=device_id,
        timestamps=ts_stop,
        values=vals_stop,
        scenario_id="belt_break",
        pre_timestamps=list(pre_ts),
        pre_values=list(pre_vals),
    )

    print("\n  Running labeling pipeline (Claude is reasoning)...")
    result_stop = pipeline.run(window=window_stop, scenario=scenario)

    _print_result(result_stop)

    # ── Summary ─────────────────────────────────────────────────────────
    print_separator("Summary")
    print(f"  Belt break case:  label='{result_break.proposal.label}' "
          f"confidence={result_break.proposal.confidence:.2f} "
          f"routing={result_break.proposal.routing.value}")
    print(f"  Planned stop case: label='{result_stop.proposal.label}' "
          f"confidence={result_stop.proposal.confidence:.2f} "
          f"routing={result_stop.proposal.routing.value}")


def _print_result(result) -> None:
    p = result.proposal
    print(f"\n  LABEL: '{p.label}'")
    print(f"  CONFIDENCE: {p.confidence:.2f}")
    print(f"  ROUTING: {p.routing.value}")
    print(f"\n  EXPLANATION:")
    # Wrap explanation for readability
    words = p.explanation.split()
    line = "    "
    for word in words:
        if len(line) + len(word) > 72:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(f"\n  EVIDENCE:")
    for ev in p.evidence:
        print(f"    • {ev}")
    print(f"\n  PHYSICAL PLAUSIBILITY:")
    words = p.physical_plausibility.split()
    line = "    "
    for word in words:
        if len(line) + len(word) > 72:
            print(line)
            line = "    " + word + " "
        else:
            line += word + " "
    if line.strip():
        print(line)
    print(f"\n  Similar devices used: {result.similar_devices}")
    print(f"  Processing time: {result.processing_time_s:.1f}s")
    print(f"\n  Features: level_ratio={result.features_summary['level_ratio']:.3f}, "
          f"amplitude_ratio={result.features_summary['amplitude_ratio']:.3f}")


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY environment variable not set")
        sys.exit(1)

    # Clean up old database for fresh demo
    db_path = Path(DB_PATH)
    if db_path.exists():
        db_path.unlink()
        logger.info("Removed old database for fresh demo")

    print("\n" + "=" * 70)
    print("  UNIVERSAL TIME-SERIES LABELING AGENT")
    print("  Scenario: Belt Break Detection on Electric Motor Drives")
    print("=" * 70)

    print("\nGenerating synthetic dataset...")
    dataset = build_dataset(seed=42)
    print(f"  {len(dataset['devices'])} known devices + 1 demo device")

    pipeline = LabelingPipeline(db_path=DB_PATH, similar_devices_k=5)

    try:
        # Phase 1: build knowledge base
        build_knowledge_base(pipeline, dataset)

        # Phase 2: run demo on new device
        run_belt_break_demo(pipeline, dataset)

    finally:
        pipeline.close()

    print("\n" + "=" * 70)
    print("  Demo complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
