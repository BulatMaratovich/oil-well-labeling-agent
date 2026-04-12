"""
rules/starter_ruleset.py — Starter ruleset.

Provides a pre-built RuleRegistry with rules covering all 4 priority tiers.
Each rule is a pure Python callable — no ML, no LLM, no I/O.

Tier 1 — Sensor / data quality exclusions
Tier 2 — Confounder exclusions (planned stops, maintenance)
Tier 3 — Stable unusual regime (persistent shift, not a failure)
Tier 4 — True deviation / belt-break class

Usage
-----
    from rules.starter_ruleset import build_registry
    registry = build_registry()
"""
from __future__ import annotations

from rules.rule_schemas import Rule, RuleInput, RuleRegistry

# ---------------------------------------------------------------------------
# Helper accessors (keep conditions readable)
# ---------------------------------------------------------------------------

def _f(inp: RuleInput):
    """Shorthand: local features (may be None)."""
    return inp.features


def _ctx(inp: RuleInput):
    """Shorthand: context bundle (may be None)."""
    return inp.context


def _facts(inp: RuleInput):
    """All maintenance facts from context (empty list if no context)."""
    ctx = _ctx(inp)
    if ctx is None:
        return []
    return ctx.maintenance_facts or []


def _has_event_type(inp: RuleInput, *event_types: str) -> bool:
    """True if any maintenance fact matches one of *event_types*."""
    target = set(event_types)
    return any(f.event_type in target for f in _facts(inp))


# ---------------------------------------------------------------------------
# Tier 1 — Sensor / data-quality exclusions
# ---------------------------------------------------------------------------

RULE_SENSOR_ALL_ZERO = Rule(
    rule_id="sensor_all_zero",
    label="sensor_issue",
    priority=1,
    description="Сигнал равен нулю на всём интервале кандидата (dropout или обрыв датчика).",
    condition=lambda inp: (
        _f(inp) is not None
        and _f(inp).zero_fraction is not None
        and _f(inp).zero_fraction >= 0.95
    ),
    condition_params={"zero_fraction_threshold": 0.95},
    rationale="Если ≥95 % значений равны нулю — скорее всего проблема датчика, не реальный режим.",
)

RULE_SENSOR_HIGH_NOISE = Rule(
    rule_id="sensor_high_noise",
    label="sensor_issue",
    priority=1,
    description="Уровень шума сигнала аномально высокий (std/|mean| > 2.0).",
    condition=lambda inp: (
        _f(inp) is not None
        and _f(inp).power_mean is not None
        and _f(inp).power_std is not None
        and abs(_f(inp).power_mean) > 1e-6
        and (_f(inp).power_std / abs(_f(inp).power_mean)) > 2.0
    ),
    condition_params={"noise_ratio_threshold": 2.0},
    rationale="Экстремальный шум указывает на неисправность или помехи датчика.",
)

RULE_SENSOR_NOVEL_EXTREME = Rule(
    rule_id="sensor_novel_extreme",
    label="sensor_issue",
    priority=1,
    description="Новый режим с нулевой или отрицательной мощностью → скорее всего датчик.",
    condition=lambda inp: (
        inp.candidate.deviation_type == "novel_regime"
        and _f(inp) is not None
        and _f(inp).power_mean is not None
        and _f(inp).power_mean <= 0.0
    ),
    rationale="Отрицательные или нулевые значения мощности физически невозможны для ESP/штанга.",
)

# ---------------------------------------------------------------------------
# Tier 2 — Confounder exclusions
# ---------------------------------------------------------------------------

RULE_PLANNED_STOP_FACT = Rule(
    rule_id="planned_stop_from_fact",
    label="planned_stop",
    priority=2,
    description="В контексте есть запись о плановой остановке в период кандидата.",
    condition=lambda inp: _has_event_type(inp, "planned_stop"),
    rationale="Если ММ-документ подтверждает плановую остановку — это не отклонение.",
)

RULE_PLANNED_MAINTENANCE_FACT = Rule(
    rule_id="planned_maintenance_from_fact",
    label="planned_maintenance",
    priority=2,
    description="В контексте есть запись о техническом обслуживании в период кандидата.",
    condition=lambda inp: _has_event_type(inp, "equipment_service", "planned_maintenance"),
    rationale="Плановое ТО объясняет изменение режима без признаков аварии.",
)

RULE_BELT_REPLACEMENT_FACT = Rule(
    rule_id="belt_replacement_from_fact",
    label="planned_maintenance",
    priority=2,
    description="В контексте есть запись о замене ремня — это плановое обслуживание.",
    condition=lambda inp: _has_event_type(inp, "belt_replacement"),
    rationale="Замена ремня — штатная операция, не аварийный обрыв.",
)

RULE_LOW_POWER_STOP = Rule(
    rule_id="low_power_stop",
    label="planned_stop",
    priority=2,
    description="Мощность близка к нулю на >80 % интервала без признаков шума.",
    condition=lambda inp: (
        _f(inp) is not None
        and _f(inp).zero_fraction is not None
        and 0.50 <= _f(inp).zero_fraction < 0.95
        and (
            _f(inp).power_std is None
            or _f(inp).power_mean is None
            or abs(_f(inp).power_mean) < 1e-6
            or (_f(inp).power_std / max(abs(_f(inp).power_mean), 1e-6)) < 2.0
        )
    ),
    condition_params={"zero_fraction_min": 0.50, "zero_fraction_max": 0.95},
    rationale="Пониженная до нуля мощность без шума — вероятнее всего плановая остановка.",
)

# ---------------------------------------------------------------------------
# Tier 3 — Stable unusual regime
# ---------------------------------------------------------------------------

RULE_STABLE_SHIFT = Rule(
    rule_id="stable_unusual_regime",
    label="stable_unusual_regime",
    priority=3,
    description=(
        "Мощность стабильно отклонена от базовой линии (score > 0.3), "
        "но внутренний шум низкий (std/mean < 0.3) — устойчивый нестандартный режим."
    ),
    condition=lambda inp: (
        inp.candidate.deviation_score > 0.3
        and inp.candidate.deviation_type in ("atypical_amplitude", "novel_regime")
        and _f(inp) is not None
        and _f(inp).power_mean is not None
        and _f(inp).power_std is not None
        and abs(_f(inp).power_mean) > 1e-6
        and (_f(inp).power_std / abs(_f(inp).power_mean)) < 0.30
    ),
    condition_params={
        "deviation_score_min": 0.3,
        "internal_noise_max": 0.30,
    },
    rationale=(
        "Стабильный сдвиг без внутреннего шума — скорее изменение нагрузки или "
        "нового регулярного режима, а не аварийный отказ."
    ),
)

RULE_LONG_UNUSUAL_DURATION = Rule(
    rule_id="long_unusual_duration",
    label="stable_unusual_regime",
    priority=3,
    description="Аномально долгий интервал (unusual_duration) без резкого перехода.",
    condition=lambda inp: (
        inp.candidate.deviation_type == "unusual_duration"
        and (
            _f(inp) is None
            or _f(inp).transition_sharpness is None
            or _f(inp).transition_sharpness < 0.2
        )
    ),
    rationale=(
        "Долгие сегменты без резкого перехода — признак стабильного нестандартного "
        "режима, а не внезапного отказа."
    ),
)

# ---------------------------------------------------------------------------
# Tier 4 — True deviation / belt-break
# ---------------------------------------------------------------------------

RULE_ABRUPT_DROP = Rule(
    rule_id="belt_break_abrupt_drop",
    label="belt_break",
    priority=4,
    description=(
        "Резкое падение мощности: transition_sharpness высокий и "
        "средняя мощность сильно ниже ожидаемой (score > 1.0)."
    ),
    condition=lambda inp: (
        _f(inp) is not None
        and _f(inp).transition_sharpness is not None
        and _f(inp).transition_sharpness >= 0.4
        and inp.candidate.deviation_score >= 1.0
        and inp.candidate.deviation_type in ("atypical_amplitude", "novel_regime", "abrupt_transition")
        and _f(inp).power_mean is not None
        and _f(inp).power_mean < _f(inp).power_p90  # redundant guard
    ),
    condition_params={
        "transition_sharpness_min": 0.4,
        "deviation_score_min": 1.0,
    },
    rationale=(
        "Резкое падение + высокий deviation score — классический паттерн "
        "обрыва ремня или внезапного останова оборудования."
    ),
)

RULE_HIGH_DEVIATION_GENERIC = Rule(
    rule_id="high_deviation_generic",
    label="belt_break",
    priority=4,
    description="Очень высокий deviation_score (> 2.0) без других объяснений.",
    condition=lambda inp: inp.candidate.deviation_score > 2.0,
    condition_params={"deviation_score_threshold": 2.0},
    rationale="Экстремальное отклонение от базовой линии — признак реального события.",
)

RULE_FULL_SERIES_REVIEW = Rule(
    rule_id="full_series_review_unknown",
    label="unknown",
    priority=4,
    description="Нет структуры режимов — вся серия требует ручного просмотра.",
    condition=lambda inp: inp.candidate.deviation_type == "full_series_review",
    rationale="Без сегментации невозможно применить правила — передать инженеру.",
)

# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_registry() -> RuleRegistry:
    """Return a RuleRegistry pre-loaded with the starter ruleset."""
    registry = RuleRegistry()
    registry.register_all([
        # Tier 1
        RULE_SENSOR_ALL_ZERO,
        RULE_SENSOR_HIGH_NOISE,
        RULE_SENSOR_NOVEL_EXTREME,
        # Tier 2
        RULE_PLANNED_STOP_FACT,
        RULE_PLANNED_MAINTENANCE_FACT,
        RULE_BELT_REPLACEMENT_FACT,
        RULE_LOW_POWER_STOP,
        # Tier 3
        RULE_STABLE_SHIFT,
        RULE_LONG_UNUSUAL_DURATION,
        # Tier 4
        RULE_ABRUPT_DROP,
        RULE_HIGH_DEVIATION_GENERIC,
        RULE_FULL_SERIES_REVIEW,
    ])
    return registry
