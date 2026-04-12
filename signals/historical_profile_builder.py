"""
signals/historical_profile_builder.py — Stage 4: Historical Profile Builder.

Aggregates a list of RegimeSequence objects (e.g. from multiple runs or
multiple wells) into per-well WellProfile baselines.

Baseline strategy: "per_well_history" — statistics computed from all regimes
observed for a given well over the full history.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np

from core.canonical_schema import (
    DateRange,
    Regime,
    RegimeBaseline,
    RegimeSequence,
    WellProfile,
)


def build_profile(
    sequences: list[RegimeSequence],
    *,
    existing_profile: Optional[WellProfile] = None,
    population_fallback: bool = True,
) -> WellProfile:
    """Build (or update) a :class:`~core.canonical_schema.WellProfile` from
    one or more :class:`~core.canonical_schema.RegimeSequence` objects.

    Parameters
    ----------
    sequences:
        All regime sequences for the same ``asset_id``.  Must share a single
        asset_id (the first non-empty one is used).
    existing_profile:
        If provided, its baselines are merged with the new observations.
    population_fallback:
        When ``True`` and the well has fewer than 3 regimes, the profile is
        flagged with ``"no_well_history"`` but still returned rather than
        raising an error.
    """
    if not sequences:
        raise ValueError("sequences must not be empty")

    asset_id = next(
        (s.asset_id for s in sequences if s.regimes),
        sequences[0].asset_id,
    )

    all_regimes: list[Regime] = [r for s in sequences for r in s.regimes]

    # ------------------------------------------------------------------
    # Group by regime_type
    # ------------------------------------------------------------------
    by_type: dict[str, list[Regime]] = {}
    for r in all_regimes:
        by_type.setdefault(r.regime_type, []).append(r)

    flags: list[str] = []
    if len(all_regimes) < 3:
        flags.append("no_well_history")
        if not population_fallback:
            raise RuntimeError(
                f"Well '{asset_id}' has fewer than 3 observed regimes; "
                "set population_fallback=True to allow a partial profile."
            )

    # ------------------------------------------------------------------
    # Build baselines
    # ------------------------------------------------------------------
    new_baselines: list[RegimeBaseline] = []
    for regime_type, regimes in by_type.items():
        powers = np.array(
            [r.mean_power for r in regimes if r.mean_power is not None],
            dtype=float,
        )
        durations = np.array(
            [r.duration_h for r in regimes],
            dtype=float,
        )
        if len(powers) == 0:
            continue
        new_baselines.append(
            RegimeBaseline(
                regime_type=regime_type,
                mean_power=float(np.mean(powers)),
                std_power=float(np.std(powers, ddof=1)) if len(powers) > 1 else 0.0,
                p10_power=float(np.percentile(powers, 10)),
                p90_power=float(np.percentile(powers, 90)),
                typical_duration_h=float(np.median(durations)) if len(durations) else 0.0,
                observation_count=len(powers),
            )
        )

    # ------------------------------------------------------------------
    # Merge with existing profile (if any)
    # ------------------------------------------------------------------
    if existing_profile:
        existing_by_type = {b.regime_type: b for b in existing_profile.baseline_regimes}
        for nb in new_baselines:
            if nb.regime_type not in existing_by_type:
                existing_by_type[nb.regime_type] = nb
            else:
                existing_by_type[nb.regime_type] = _merge_baselines(
                    existing_by_type[nb.regime_type], nb
                )
        merged_baselines = list(existing_by_type.values())
        first_seen = existing_profile.first_seen
        existing_flags = existing_profile.flags
    else:
        merged_baselines = new_baselines
        first_seen = _earliest_regime_time(all_regimes)
        existing_flags = []

    all_flags = list(dict.fromkeys(existing_flags + flags))

    return WellProfile(
        well_id=asset_id,
        baseline_regimes=merged_baselines,
        profile_source="well_history" if "no_well_history" not in all_flags else "population_fallback",
        first_seen=first_seen,
        last_updated=datetime.now(tz=timezone.utc),
        flags=all_flags,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _merge_baselines(old: RegimeBaseline, new: RegimeBaseline) -> RegimeBaseline:
    """Combine two baselines using a weighted running mean."""
    n_old = old.observation_count
    n_new = new.observation_count
    n_total = n_old + n_new
    mean = (old.mean_power * n_old + new.mean_power * n_new) / n_total
    # Pooled std
    std = _pooled_std(old.mean_power, old.std_power, n_old, new.mean_power, new.std_power, n_new)
    return RegimeBaseline(
        regime_type=old.regime_type,
        mean_power=mean,
        std_power=std,
        p10_power=min(old.p10_power, new.p10_power),
        p90_power=max(old.p90_power, new.p90_power),
        typical_duration_h=(old.typical_duration_h * n_old + new.typical_duration_h * n_new) / n_total,
        observation_count=n_total,
    )


def _pooled_std(m1: float, s1: float, n1: int, m2: float, s2: float, n2: int) -> float:
    """Two-sample pooled standard deviation."""
    if n1 + n2 <= 1:
        return 0.0
    # Combined variance via parallel formula
    combined_var = (
        (n1 - 1) * s1**2 + (n2 - 1) * s2**2
        + n1 * n2 / (n1 + n2) * (m1 - m2) ** 2
    ) / (n1 + n2 - 1)
    return float(np.sqrt(max(combined_var, 0.0)))


def _earliest_regime_time(regimes: list[Regime]) -> Optional[datetime]:
    if not regimes:
        return None
    return min(r.start for r in regimes)
