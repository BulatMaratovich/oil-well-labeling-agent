"""
signals/candidate_event_detector.py — Stage 5: Candidate Event Detector.

Compares the GlobalRegimeSequence against the WellProfile baselines and
emits CandidateEvent objects for segments that deviate from expected behaviour.

Deviation types (from canonical schema):
  - "novel_regime"         — regime type not seen in baseline
  - "atypical_amplitude"   — power level outside baseline p10/p90 band
  - "unusual_duration"     — segment duration >> typical_duration_h
  - "abrupt_transition"    — mean power delta at transition > threshold
  - "full_series_review"   — no regime structure; emit one event for whole series
"""
from __future__ import annotations

import hashlib
from typing import Optional

from core.canonical_schema import (
    CandidateEvent,
    DateRange,
    Regime,
    RegimeBaseline,
    RegimeSequence,
    WellProfile,
)
from core.task_manager import TaskSpec


# Multiplier on typical_duration_h to flag "unusual_duration"
_DURATION_MULTIPLIER = 3.0
# Fraction of (p90 - p10) used as abrupt-transition threshold
_TRANSITION_FRACTION = 0.5


def detect(
    regime_sequence: RegimeSequence,
    well_profile: WellProfile,
    task_spec: TaskSpec,
    *,
    max_candidates: int = 20,
) -> list[CandidateEvent]:
    """Return candidate events from *regime_sequence* relative to *well_profile*.

    Parameters
    ----------
    regime_sequence:
        Output of the Global Series Profiler for one asset.
    well_profile:
        Historical baselines for the same asset.
    task_spec:
        Used for ``primary_deviation`` text (embedded in ``context_query``).
    max_candidates:
        Upper bound on returned candidates (highest-score first).
    """
    regimes = regime_sequence.regimes
    asset_id = regime_sequence.asset_id

    # ------------------------------------------------------------------
    # Special case: no regime structure → one whole-series candidate
    # ------------------------------------------------------------------
    if regime_sequence.no_regime_structure or not regimes:
        if not regimes:
            return []
        r0, r_last = regimes[0], regimes[-1]
        return [
            CandidateEvent(
                candidate_id=_make_id(asset_id, r0.start, "full"),
                asset_id=asset_id,
                segment=DateRange(start=r0.start, end=r_last.end),
                deviation_type="full_series_review",
                deviation_score=1.0,
                context_query=(
                    f"{task_spec.primary_deviation} — полная серия, "
                    f"структура режимов не обнаружена"
                ),
                series_name=regime_sequence.signal_name,
                flags=["no_regime_structure"],
            )
        ]

    baseline_map = {b.regime_type: b for b in well_profile.baseline_regimes}

    # ------------------------------------------------------------------
    # Cross-type fallback for first-run data
    #
    # With only a single pass of data PELT segments the series into N
    # regime types each appearing once.  Building per-type baselines from
    # one observation means p10 == p90 == mean, so every regime looks
    # "normal" relative to its own history — no candidates are generated.
    #
    # Fix: when all baselines have observation_count <= 1, identify the
    # *dominant* operating regime (longest cumulative duration) as the
    # "normal" state and compare all other regimes against it.
    # ------------------------------------------------------------------
    all_single_obs = all(b.observation_count <= 1 for b in well_profile.baseline_regimes)
    dominant_baseline: Optional[RegimeBaseline] = None
    dominant_type: Optional[str] = None
    if all_single_obs and well_profile.baseline_regimes:
        duration_by_type: dict[str, float] = {}
        for r in regimes:
            duration_by_type[r.regime_type] = duration_by_type.get(r.regime_type, 0.0) + r.duration_h
        if duration_by_type:
            dominant_type = max(duration_by_type, key=lambda t: duration_by_type[t])
            dominant_baseline = baseline_map.get(dominant_type)

    candidates: list[CandidateEvent] = []

    for i, regime in enumerate(regimes):
        baseline = baseline_map.get(regime.regime_type)
        preceding = regimes[i - 1].regime_type if i > 0 else None
        following = regimes[i + 1].regime_type if i < len(regimes) - 1 else None

        # For single-run data: compare non-dominant regimes against the
        # dominant baseline, but only when they differ meaningfully
        # (>= 10 % relative deviation in mean power).
        effective_baseline = baseline
        if (
            all_single_obs
            and dominant_baseline is not None
            and dominant_type is not None
            and regime.regime_type != dominant_type
            and regime.mean_power is not None
            and dominant_baseline.mean_power is not None
        ):
            dom_mean = dominant_baseline.mean_power
            rel_diff = abs(regime.mean_power - dom_mean) / max(abs(dom_mean), 1.0)
            if rel_diff >= 0.10:
                effective_baseline = dominant_baseline

        events_for_regime = _evaluate_regime(
            regime=regime,
            baseline=effective_baseline,
            asset_id=asset_id,
            preceding=preceding,
            following=following,
            primary_deviation=task_spec.primary_deviation,
            series_name=regime_sequence.signal_name,
        )
        candidates.extend(events_for_regime)

    candidates.sort(key=lambda c: c.deviation_score, reverse=True)
    return candidates[:max_candidates]


# ---------------------------------------------------------------------------
# Per-regime evaluation
# ---------------------------------------------------------------------------

def _evaluate_regime(
    regime: Regime,
    baseline: Optional[RegimeBaseline],
    asset_id: str,
    preceding: Optional[str],
    following: Optional[str],
    primary_deviation: str,
    series_name: Optional[str],
) -> list[CandidateEvent]:
    events: list[CandidateEvent] = []

    # 1. Novel regime type
    if baseline is None:
        events.append(CandidateEvent(
            candidate_id=_make_id(asset_id, regime.start, "novel"),
            asset_id=asset_id,
            segment=DateRange(start=regime.start, end=regime.end),
            deviation_type="novel_regime",
            deviation_score=1.0,
            context_query=_query(primary_deviation, regime, "novel_regime"),
            series_name=series_name,
            preceding_regime_type=preceding,
            following_regime_type=following,
            flags=["novel_regime_type"],
        ))
        return events

    # 2. Atypical amplitude
    if regime.mean_power is not None and baseline.p10_power is not None:
        # When the baseline has a single observation p10 == p90 == mean, so the
        # raw band is 0.  Use at least 10 % of the baseline mean as the band to
        # avoid division-by-near-zero producing artificially huge scores and to
        # keep scores comparable across data sets with different scales.
        band = max(
            baseline.p90_power - baseline.p10_power,
            abs(baseline.mean_power) * 0.10,
            1.0,
        )
        if regime.mean_power < baseline.p10_power:
            score = min((baseline.p10_power - regime.mean_power) / band, 5.0)
            events.append(CandidateEvent(
                candidate_id=_make_id(asset_id, regime.start, "amp_low"),
                asset_id=asset_id,
                segment=DateRange(start=regime.start, end=regime.end),
                deviation_type="atypical_amplitude",
                deviation_score=round(score, 4),
                context_query=_query(primary_deviation, regime, "low_amplitude"),
                series_name=series_name,
                preceding_regime_type=preceding,
                following_regime_type=following,
            ))
        elif regime.mean_power > baseline.p90_power:
            score = min((regime.mean_power - baseline.p90_power) / band, 5.0)
            events.append(CandidateEvent(
                candidate_id=_make_id(asset_id, regime.start, "amp_high"),
                asset_id=asset_id,
                segment=DateRange(start=regime.start, end=regime.end),
                deviation_type="atypical_amplitude",
                deviation_score=round(score, 4),
                context_query=_query(primary_deviation, regime, "high_amplitude"),
                series_name=series_name,
                preceding_regime_type=preceding,
                following_regime_type=following,
            ))

    # 3. Unusual duration
    if baseline.typical_duration_h > 0:
        ratio = regime.duration_h / baseline.typical_duration_h
        if ratio >= _DURATION_MULTIPLIER:
            score = min(ratio / _DURATION_MULTIPLIER, 5.0)
            events.append(CandidateEvent(
                candidate_id=_make_id(asset_id, regime.start, "dur"),
                asset_id=asset_id,
                segment=DateRange(start=regime.start, end=regime.end),
                deviation_type="unusual_duration",
                deviation_score=round(score, 4),
                context_query=_query(primary_deviation, regime, "unusual_duration"),
                series_name=series_name,
                preceding_regime_type=preceding,
                following_regime_type=following,
            ))

    return events


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id(asset_id: str, start: object, suffix: str) -> str:
    key = f"{asset_id}:{start}:{suffix}"
    return "cand_" + hashlib.md5(key.encode()).hexdigest()[:10]


def _query(primary_deviation: str, regime: Regime, deviation_kind: str) -> str:
    return (
        f"{primary_deviation} — {deviation_kind} "
        f"в режиме {regime.regime_type} "
        f"({regime.start.date()} … {regime.end.date()})"
    )
