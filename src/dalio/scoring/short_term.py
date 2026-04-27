"""Rule-based short-term debt cycle stage classifier.

Maps observed indicators to one of Dalio's four short-term cycle stages
(or "transition" if the top two stages are too close to call):

  1. Expansion         — positive growth, contained inflation, falling/stable unemployment
  2. Inflationary peak — elevated CPI with central bank tightening
  3. Recession         — negative growth or sharply rising unemployment
  4. Reflation         — central bank cutting, inflation moderating

Rules are intentionally transparent and inspectable: every classification carries
the full list of rule "votes" that produced it, with reasons.

Threshold rationale:
- GDP > 2.5 = clearly expansionary; 1-2.5 = mild/late-cycle; <0 = recession.
- CPI < 2 = below target (Reflation territory); 2-3 = at/near target (Expansion);
  3-4 = elevated; >4 = clear inflationary peak.
- Policy-rate 6m change captures the central bank's stance: ±0.5pp is the noise floor.
- Yield curve (10y-2y) acts as a *modifier* not a primary stage signal:
  inversion adds weight to Recession; steepness adds weight to Expansion.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.scoring.thresholds import DEFAULT_THRESHOLDS, Thresholds
from dalio.storage.db import Observation

STAGE_LABELS: dict[int, str] = {
    1: "Expansion",
    2: "Inflationary peak",
    3: "Recession",
    4: "Reflation",
    0: "Transition",
}

# Indicators required to compute the full feature set.
SHORT_TERM_INDICATORS: tuple[str, ...] = (
    "real_gdp_yoy",
    "cpi_yoy",
    "unemployment_rate",
    "policy_rate",
    "yield_10y",
    "yield_2y",
)


@dataclass(frozen=True)
class StageVote:
    stage: int
    weight: float
    reason: str


@dataclass(frozen=True)
class ShortTermFeatures:
    """Snapshot of indicators + selected lags for one country, with as-of dates."""
    country: str
    real_gdp_yoy: float | None = None
    cpi_yoy: float | None = None
    cpi_yoy_3m_ago: float | None = None
    unemployment_rate: float | None = None
    unemployment_3m_ago: float | None = None
    policy_rate: float | None = None
    policy_rate_6m_ago: float | None = None
    yield_10y: float | None = None
    yield_2y: float | None = None
    indicator_dates: dict[str, date] = field(default_factory=dict)

    @property
    def cpi_change_3m(self) -> float | None:
        if self.cpi_yoy is None or self.cpi_yoy_3m_ago is None:
            return None
        return self.cpi_yoy - self.cpi_yoy_3m_ago

    @property
    def unemployment_change_3m(self) -> float | None:
        if self.unemployment_rate is None or self.unemployment_3m_ago is None:
            return None
        return self.unemployment_rate - self.unemployment_3m_ago

    @property
    def policy_rate_change_6m(self) -> float | None:
        if self.policy_rate is None or self.policy_rate_6m_ago is None:
            return None
        return self.policy_rate - self.policy_rate_6m_ago

    @property
    def yield_curve_slope(self) -> float | None:
        if self.yield_10y is None or self.yield_2y is None:
            return None
        return self.yield_10y - self.yield_2y

    @property
    def as_of(self) -> date | None:
        if not self.indicator_dates:
            return None
        return max(self.indicator_dates.values())


@dataclass(frozen=True)
class Classification:
    country: str
    stage: int
    stage_label: str
    confidence: float
    votes: tuple[StageVote, ...]
    features: ShortTermFeatures


def _value_at_or_before(
    session: Session, country: str, indicator: str, target: date,
) -> tuple[float, date] | None:
    row = session.execute(
        select(Observation)
        .where(
            Observation.country == country,
            Observation.indicator == indicator,
            Observation.date <= target,
        )
        .order_by(Observation.date.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return (float(row.value), row.date)


def _latest_at(
    session: Session, country: str, indicator: str, as_of: date,
) -> tuple[float, date] | None:
    """Latest observation at or before `as_of`. When `as_of=date.today()` this
    is equivalent to fetching the latest live value. Used for both live and
    historical feature extraction (Slice 12 backtest)."""
    return _value_at_or_before(session, country, indicator, as_of)


def extract_features(
    session: Session, country: str, as_of: date | None = None,
) -> ShortTermFeatures:
    """Pull values + 3m/6m lags from DB into a feature snapshot.

    If `as_of` is provided, all "latest" lookups are capped at that date —
    used by `replay.py` to backtest classifications historically. Default
    behavior (`as_of=None`) is unchanged.
    """
    cap = as_of if as_of is not None else date.today()
    fields: dict[str, float | None] = {}
    indicator_dates: dict[str, date] = {}

    for ind in SHORT_TERM_INDICATORS:
        latest = _latest_at(session, country, ind, cap)
        if latest is None:
            fields[ind] = None
            continue
        value, when = latest
        fields[ind] = value
        indicator_dates[ind] = when

    # Lagged features (only computed when we have the latest value to anchor against)
    cpi_3m_ago = None
    if (cpi_anchor := indicator_dates.get("cpi_yoy")) is not None:
        lag = _value_at_or_before(session, country, "cpi_yoy", cpi_anchor - timedelta(days=90))
        cpi_3m_ago = lag[0] if lag else None

    unemp_3m_ago = None
    if (unemp_anchor := indicator_dates.get("unemployment_rate")) is not None:
        lag = _value_at_or_before(
            session, country, "unemployment_rate", unemp_anchor - timedelta(days=90)
        )
        unemp_3m_ago = lag[0] if lag else None

    policy_6m_ago = None
    if (policy_anchor := indicator_dates.get("policy_rate")) is not None:
        lag = _value_at_or_before(
            session, country, "policy_rate", policy_anchor - timedelta(days=180)
        )
        policy_6m_ago = lag[0] if lag else None

    return ShortTermFeatures(
        country=country,
        real_gdp_yoy=fields.get("real_gdp_yoy"),
        cpi_yoy=fields.get("cpi_yoy"),
        cpi_yoy_3m_ago=cpi_3m_ago,
        unemployment_rate=fields.get("unemployment_rate"),
        unemployment_3m_ago=unemp_3m_ago,
        policy_rate=fields.get("policy_rate"),
        policy_rate_6m_ago=policy_6m_ago,
        yield_10y=fields.get("yield_10y"),
        yield_2y=fields.get("yield_2y"),
        indicator_dates=indicator_dates,
    )


def _vote_recession(f: ShortTermFeatures) -> list[StageVote]:
    votes = []
    if f.real_gdp_yoy is not None and f.real_gdp_yoy < 0:
        votes.append(StageVote(3, 1.0, f"Real GDP YoY {f.real_gdp_yoy:.2f}% is negative"))
    # Sahm rule: 3m unemployment rise > 0.5pp has been a near-perfect recession signal
    # historically. Below that, treat as early-warning noise.
    if f.unemployment_change_3m is not None:
        if f.unemployment_change_3m > 0.5:
            votes.append(StageVote(
                3, 1.0,
                f"Unemployment +{f.unemployment_change_3m:.2f}pp over 3m (Sahm-rule territory)",
            ))
        elif f.unemployment_change_3m > 0.3:
            votes.append(StageVote(
                3, 0.5,
                f"Unemployment rising +{f.unemployment_change_3m:.2f}pp over 3m",
            ))
    if f.yield_curve_slope is not None and f.yield_curve_slope < 0:
        votes.append(StageVote(
            3, 0.4,
            f"Yield curve inverted ({f.yield_curve_slope:.2f}pp 10y-2y)",
        ))
    return votes


def _vote_inflationary_peak(f: ShortTermFeatures, t: Thresholds) -> list[StageVote]:
    votes = []
    cpi = f.cpi_yoy
    if cpi is not None and cpi > t.cpi_peak:
        votes.append(StageVote(
            2, 1.0,
            f"CPI {cpi:.2f}% > {t.cpi_peak:.1f}% (clearly elevated)",
        ))
    elif cpi is not None and cpi > t.cpi_elevated:
        weight = 0.5
        reason = f"CPI {cpi:.2f}% above {t.cpi_elevated:.1f}%"
        if f.policy_rate_change_6m is not None and f.policy_rate_change_6m > 0.5:
            weight = 0.9
            reason += f" with CB tightening (+{f.policy_rate_change_6m:.2f}pp/6m)"
        votes.append(StageVote(2, weight, reason))
    # Acceleration vote uses cpi_elevated - 0.5 as the floor (was a hard 2.5)
    accel_floor = max(t.cpi_elevated - 0.5, 2.0)
    if f.cpi_change_3m is not None and f.cpi_change_3m > 0.5 and cpi is not None and cpi > accel_floor:
        votes.append(StageVote(
            2, 0.4,
            f"CPI accelerating (+{f.cpi_change_3m:.2f}pp over 3m at {cpi:.2f}%)",
        ))
    return votes


def _vote_reflation(f: ShortTermFeatures) -> list[StageVote]:
    votes = []
    rate_chg = f.policy_rate_change_6m
    cutting_strong = rate_chg is not None and rate_chg < -0.5
    cutting_moderate = rate_chg is not None and -0.5 <= rate_chg < -0.25

    if cutting_strong and f.cpi_yoy is not None and f.cpi_yoy < 4:
        votes.append(StageVote(
            4, 1.0,
            f"CB cutting strongly ({rate_chg:.2f}pp/6m), CPI moderating to {f.cpi_yoy:.2f}%",
        ))
    elif cutting_strong:
        votes.append(StageVote(4, 0.6, f"CB cutting strongly ({rate_chg:.2f}pp/6m)"))
    elif cutting_moderate and f.cpi_yoy is not None and f.cpi_yoy < 4:
        votes.append(StageVote(
            4, 0.6,
            f"CB cutting moderately ({rate_chg:.2f}pp/6m) with CPI {f.cpi_yoy:.2f}%",
        ))
    elif cutting_moderate:
        votes.append(StageVote(4, 0.4, f"CB cutting moderately ({rate_chg:.2f}pp/6m)"))

    if f.cpi_change_3m is not None and f.cpi_change_3m < -0.3:
        votes.append(StageVote(
            4, 0.4,
            f"CPI decelerating ({f.cpi_change_3m:.2f}pp over 3m)",
        ))
    return votes


def _vote_expansion(f: ShortTermFeatures) -> list[StageVote]:
    votes = []
    gdp_ok = f.real_gdp_yoy is not None and f.real_gdp_yoy > 1.5
    cpi_ok = f.cpi_yoy is not None and f.cpi_yoy < 3
    unemp_falling = (
        f.unemployment_change_3m is not None and f.unemployment_change_3m <= 0.1
    )
    unemp_rising = (
        f.unemployment_change_3m is not None and f.unemployment_change_3m > 0.3
    )

    if gdp_ok and cpi_ok and unemp_falling:
        votes.append(StageVote(
            1, 1.0,
            f"GDP {f.real_gdp_yoy:.2f}%, CPI {f.cpi_yoy:.2f}%, unemployment stable/falling",
        ))
    # Lenient vote — but only if unemployment isn't actively rising (would contradict Expansion)
    elif gdp_ok and cpi_ok and not unemp_rising:
        votes.append(StageVote(
            1, 0.6,
            f"GDP {f.real_gdp_yoy:.2f}% with CPI {f.cpi_yoy:.2f}% < 3%",
        ))
    if f.yield_curve_slope is not None and f.yield_curve_slope > 1:
        votes.append(StageVote(
            1, 0.3,
            f"Yield curve steep ({f.yield_curve_slope:.2f}pp) — early-cycle signal",
        ))
    return votes


def classify_features(
    features: ShortTermFeatures,
    thresholds: Thresholds | None = None,
) -> Classification:
    """Apply all rule families to features, return classification."""
    t = thresholds if thresholds is not None else DEFAULT_THRESHOLDS
    votes: list[StageVote] = []
    votes.extend(_vote_recession(features))
    votes.extend(_vote_inflationary_peak(features, t))
    votes.extend(_vote_reflation(features))
    votes.extend(_vote_expansion(features))

    if not votes:
        return Classification(
            country=features.country,
            stage=0,
            stage_label=STAGE_LABELS[0] + " (insufficient data)",
            confidence=0.0,
            votes=tuple(votes),
            features=features,
        )

    weights: dict[int, float] = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}
    for v in votes:
        weights[v.stage] += v.weight

    total = sum(weights.values())
    sorted_stages = sorted(weights.items(), key=lambda kv: -kv[1])
    top_stage, top_weight = sorted_stages[0]
    second_stage, second_weight = sorted_stages[1] if len(sorted_stages) > 1 else (0, 0.0)

    # Saturating confidence — adds an implicit "uncertainty floor" so a single weak
    # rule firing doesn't read as 100% confidence. Floor=1.0 gives:
    #   top=0.5 alone → 33%; top=2.0 alone → 67%; top=3.0 alone → 75%
    confidence = top_weight / (total + 1.0)

    # Top two stages within 0.3 weight units → transition. Honest output.
    if (top_weight - second_weight) < 0.3 and second_weight > 0:
        label = (
            f"{STAGE_LABELS[0]} ({STAGE_LABELS[top_stage]} ↔ {STAGE_LABELS[second_stage]})"
        )
        return Classification(
            country=features.country,
            stage=0,
            stage_label=label,
            confidence=confidence,
            votes=tuple(votes),
            features=features,
        )

    return Classification(
        country=features.country,
        stage=top_stage,
        stage_label=STAGE_LABELS[top_stage],
        confidence=confidence,
        votes=tuple(votes),
        features=features,
    )


def classify(
    session: Session, country: str,
    thresholds: Thresholds | None = None,
    as_of: date | None = None,
) -> Classification:
    """Convenience: extract features + classify in one call.

    If `thresholds` is None, the per-country calibration is loaded
    automatically; pass DEFAULT_THRESHOLDS explicitly to use the global
    defaults.

    If `as_of` is provided, classify as of that historical date — used
    by `replay.py` to walk the regime path through history.
    """
    features = extract_features(session, country, as_of=as_of)
    if thresholds is None:
        from dalio.scoring.calibration import compute_country_thresholds
        thresholds = compute_country_thresholds(session, country)
    return classify_features(features, thresholds)
