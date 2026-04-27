"""Rule-based long-term debt cycle phase classifier.

Maps observed indicators to one of Dalio's 7 long-term cycle phases. Because
the long-term cycle moves over decades, the classifier focuses on regime
identification rather than precise phase pinpointing — modern developed
economies have all left "Phase 1 (Sound money)" decades ago, so the practical
question is *where in phases 2–6 does each country sit, and is anyone slipping
toward Phase 7 (Reset)*.

Indicators consumed (all from BIS Total Credit + DSR, plus FRED yield/CPI for
real rates):
  total_credit_pct_gdp      — total non-financial debt / GDP (BIS sector C)
  total_credit_5y_change_pp — 5-year change in the above
  debt_service_ratio        — private non-fin DSR (BIS WS_DSR)
  gov_debt_pct_gdp          — government debt / GDP (BIS sector G)
  real_rate_10y             — 10y nominal yield − CPI YoY (computed)

Output is a phase label + confidence + the rule votes that produced it. The
classifier never asserts Phase 7 (Reset) automatically — that's a regime
discontinuity, not something to infer from gradual indicators.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.storage.db import Observation

PHASE_LABELS: dict[int, str] = {
    1: "Sound money",
    2: "Debt outpaces income",
    3: "Bubble",
    4: "Top — peak debt service",
    5: "Deleveraging",
    6: "Reflation / financial repression",
    7: "Reset",
    0: "Transition",
}

LONG_TERM_INDICATORS: tuple[str, ...] = (
    "total_credit_pct_gdp",
    "gov_debt_pct_gdp",
    "private_nonfin_pct_gdp",
    "hh_debt_pct_gdp",
    "corp_debt_pct_gdp",
    "debt_service_ratio",
)


@dataclass(frozen=True)
class PhaseVote:
    phase: int
    weight: float
    reason: str


@dataclass(frozen=True)
class LongTermFeatures:
    country: str
    total_credit_pct_gdp: float | None = None
    total_credit_5y_ago: float | None = None
    gov_debt_pct_gdp: float | None = None
    private_nonfin_pct_gdp: float | None = None
    hh_debt_pct_gdp: float | None = None
    corp_debt_pct_gdp: float | None = None
    debt_service_ratio: float | None = None
    debt_service_5y_ago: float | None = None
    yield_10y: float | None = None
    cpi_yoy: float | None = None
    indicator_dates: dict[str, date] = field(default_factory=dict)

    @property
    def total_credit_5y_change_pp(self) -> float | None:
        if self.total_credit_pct_gdp is None or self.total_credit_5y_ago is None:
            return None
        return self.total_credit_pct_gdp - self.total_credit_5y_ago

    @property
    def dsr_5y_change_pp(self) -> float | None:
        if self.debt_service_ratio is None or self.debt_service_5y_ago is None:
            return None
        return self.debt_service_ratio - self.debt_service_5y_ago

    @property
    def real_rate_10y(self) -> float | None:
        if self.yield_10y is None or self.cpi_yoy is None:
            return None
        return self.yield_10y - self.cpi_yoy

    @property
    def as_of(self) -> date | None:
        if not self.indicator_dates:
            return None
        return max(self.indicator_dates.values())


@dataclass(frozen=True)
class PhaseClassification:
    country: str
    phase: int
    phase_label: str
    confidence: float
    votes: tuple[PhaseVote, ...]
    features: LongTermFeatures


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


def _latest(
    session: Session, country: str, indicator: str,
) -> tuple[float, date] | None:
    row = session.execute(
        select(Observation)
        .where(Observation.country == country, Observation.indicator == indicator)
        .order_by(Observation.date.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        return None
    return (float(row.value), row.date)


def extract_features(session: Session, country: str) -> LongTermFeatures:
    """Pull latest values + 5-year lags for the long-term debt cycle features."""
    fields_out: dict[str, float | None] = {}
    indicator_dates: dict[str, date] = {}

    for ind in LONG_TERM_INDICATORS:
        latest = _latest(session, country, ind)
        if latest is None:
            fields_out[ind] = None
            continue
        value, when = latest
        fields_out[ind] = value
        indicator_dates[ind] = when

    # 5-year lag for total credit and DSR
    tc_5y_ago = None
    if (anchor := indicator_dates.get("total_credit_pct_gdp")) is not None:
        lag = _value_at_or_before(
            session, country, "total_credit_pct_gdp", anchor - timedelta(days=365 * 5),
        )
        tc_5y_ago = lag[0] if lag else None

    dsr_5y_ago = None
    if (anchor := indicator_dates.get("debt_service_ratio")) is not None:
        lag = _value_at_or_before(
            session, country, "debt_service_ratio", anchor - timedelta(days=365 * 5),
        )
        dsr_5y_ago = lag[0] if lag else None

    # Real rate inputs from FRED data (already in DB if FRED ETL has been run)
    yield_lookup = _latest(session, country, "yield_10y")
    cpi_lookup = _latest(session, country, "cpi_yoy")
    yield_10y = yield_lookup[0] if yield_lookup else None
    cpi_yoy = cpi_lookup[0] if cpi_lookup else None
    if yield_lookup:
        indicator_dates.setdefault("yield_10y", yield_lookup[1])
    if cpi_lookup:
        indicator_dates.setdefault("cpi_yoy", cpi_lookup[1])

    return LongTermFeatures(
        country=country,
        total_credit_pct_gdp=fields_out.get("total_credit_pct_gdp"),
        total_credit_5y_ago=tc_5y_ago,
        gov_debt_pct_gdp=fields_out.get("gov_debt_pct_gdp"),
        private_nonfin_pct_gdp=fields_out.get("private_nonfin_pct_gdp"),
        hh_debt_pct_gdp=fields_out.get("hh_debt_pct_gdp"),
        corp_debt_pct_gdp=fields_out.get("corp_debt_pct_gdp"),
        debt_service_ratio=fields_out.get("debt_service_ratio"),
        debt_service_5y_ago=dsr_5y_ago,
        yield_10y=yield_10y,
        cpi_yoy=cpi_yoy,
        indicator_dates=indicator_dates,
    )


def _vote_sound_money(f: LongTermFeatures) -> list[PhaseVote]:
    if f.total_credit_pct_gdp is not None and f.total_credit_pct_gdp < 100:
        return [PhaseVote(
            1, 1.0,
            f"Total non-financial debt {f.total_credit_pct_gdp:.0f}% < 100% (low-leverage regime)",
        )]
    return []


def _vote_debt_outpaces(f: LongTermFeatures) -> list[PhaseVote]:
    votes = []
    tc = f.total_credit_pct_gdp
    chg5 = f.total_credit_5y_change_pp
    if tc is not None and 100 <= tc < 200 and chg5 is not None and chg5 > 5:
        votes.append(PhaseVote(
            2, 1.0,
            f"Debt {tc:.0f}% rising +{chg5:.0f}pp/5y (productive credit expansion territory)",
        ))
    elif tc is not None and 100 <= tc < 200:
        votes.append(PhaseVote(
            2, 0.4,
            f"Debt {tc:.0f}% in 100–200% range (mid-cycle leverage)",
        ))
    return votes


def _vote_bubble(f: LongTermFeatures) -> list[PhaseVote]:
    votes = []
    tc = f.total_credit_pct_gdp
    chg5 = f.total_credit_5y_change_pp
    if tc is not None and 200 <= tc < 280 and chg5 is not None and chg5 > 15:
        votes.append(PhaseVote(
            3, 1.0,
            f"Debt {tc:.0f}% with rapid expansion (+{chg5:.0f}pp/5y)",
        ))
    elif tc is not None and 220 <= tc < 280:
        votes.append(PhaseVote(
            3, 0.5,
            f"Debt {tc:.0f}% in late-cycle leverage zone",
        ))
    return votes


def _vote_top(f: LongTermFeatures) -> list[PhaseVote]:
    votes = []
    tc = f.total_credit_pct_gdp
    dsr = f.debt_service_ratio
    # Phase 4a: extreme debt level (≥280% of GDP)
    if tc is not None and tc >= 280:
        weight = 0.8 if (dsr is not None and dsr > 17) else 0.6
        notes = f"Debt {tc:.0f}% in extreme zone (≥280%)"
        if dsr is not None and dsr > 17:
            notes += f"; DSR {dsr:.1f}% stretched"
        votes.append(PhaseVote(4, weight, notes))
    # Phase 4b: stretched DSR with elevated debt — peak debt service burden
    if dsr is not None and dsr > 18 and tc is not None and tc > 220:
        votes.append(PhaseVote(
            4, 0.7,
            f"DSR {dsr:.1f}% > 18% on debt {tc:.0f}% (peak debt-service burden)",
        ))
    return votes


def _vote_deleveraging(f: LongTermFeatures) -> list[PhaseVote]:
    """Phase 5 = 'ugly' deleveraging: debt contraction AND distress (Dalio's term).

    Distinguished from Phase 6 by the *mechanism*: Phase 5 is real distress
    (defaults, recession, painful deleveraging), Phase 6 is inflation-led debt
    erosion. Both involve falling debt/GDP — context separates them.
    """
    votes = []
    tc = f.total_credit_pct_gdp
    chg5 = f.total_credit_5y_change_pp
    dsr = f.debt_service_ratio
    cpi = f.cpi_yoy
    if (
        chg5 is not None and chg5 < -10
        and tc is not None and tc > 150
        and dsr is not None and dsr > 18
        and cpi is not None and cpi < 3
    ):
        votes.append(PhaseVote(
            5, 1.0,
            f"Debt falling ({chg5:+.0f}pp/5y) at elevated level {tc:.0f}% "
            f"with distressed DSR {dsr:.1f}% (low inflation rules out beautiful)",
        ))
    if dsr is not None and dsr > 22:
        votes.append(PhaseVote(
            5, 0.7,
            f"DSR {dsr:.1f}% extreme — household/corp distress likely",
        ))
    return votes


def _vote_reflation_repression(f: LongTermFeatures) -> list[PhaseVote]:
    """Phase 6 = 'beautiful' deleveraging or financial repression.

    Two pathways:
    - Negative real rates (financial repression — savers lose, debtors gain)
    - Debt/GDP falling AND moderate-to-high inflation (inflation-eroded debt)
    """
    votes = []
    rr = f.real_rate_10y
    tc = f.total_credit_pct_gdp
    chg5 = f.total_credit_5y_change_pp
    cpi = f.cpi_yoy

    if rr is not None and rr < -1 and tc is not None and tc > 150:
        weight = 1.0 if rr < -2 else 0.7
        votes.append(PhaseVote(
            6, weight,
            f"Real 10y rate {rr:+.1f}% with debt {tc:.0f}% — financial repression",
        ))

    # Beautiful deleveraging: inflation-eroded debt ratio
    if (
        chg5 is not None and chg5 < -10
        and tc is not None and tc > 150
        and cpi is not None and cpi > 3
    ):
        votes.append(PhaseVote(
            6, 0.7,
            f"Debt {tc:.0f}% falling ({chg5:+.0f}pp/5y) with CPI {cpi:.1f}% "
            f"— beautiful deleveraging (inflation-eroded ratio)",
        ))
    return votes


def classify_features(features: LongTermFeatures) -> PhaseClassification:
    votes: list[PhaseVote] = []
    votes.extend(_vote_sound_money(features))
    votes.extend(_vote_debt_outpaces(features))
    votes.extend(_vote_bubble(features))
    votes.extend(_vote_top(features))
    votes.extend(_vote_deleveraging(features))
    votes.extend(_vote_reflation_repression(features))

    if not votes:
        return PhaseClassification(
            country=features.country,
            phase=0,
            phase_label=f"{PHASE_LABELS[0]} (insufficient data)",
            confidence=0.0,
            votes=tuple(votes),
            features=features,
        )

    weights: dict[int, float] = {p: 0.0 for p in (1, 2, 3, 4, 5, 6)}
    for v in votes:
        weights[v.phase] += v.weight
    total = sum(weights.values())
    sorted_phases = sorted(weights.items(), key=lambda kv: -kv[1])
    top_phase, top_w = sorted_phases[0]
    second_phase, second_w = sorted_phases[1] if len(sorted_phases) > 1 else (0, 0.0)

    # Saturating confidence (matches short-term classifier convention)
    confidence = top_w / (total + 1.0)

    if (top_w - second_w) < 0.3 and second_w > 0:
        return PhaseClassification(
            country=features.country,
            phase=0,
            phase_label=(
                f"{PHASE_LABELS[0]} ({PHASE_LABELS[top_phase]} ↔ {PHASE_LABELS[second_phase]})"
            ),
            confidence=confidence,
            votes=tuple(votes),
            features=features,
        )

    return PhaseClassification(
        country=features.country,
        phase=top_phase,
        phase_label=PHASE_LABELS[top_phase],
        confidence=confidence,
        votes=tuple(votes),
        features=features,
    )


def classify(session: Session, country: str) -> PhaseClassification:
    features = extract_features(session, country)
    return classify_features(features)
