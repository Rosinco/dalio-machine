"""View-model layer for the Streamlit dashboard.

Pulls together short-term + long-term + allocation classifications into the
shapes the UI wants — keeps streamlit_app.py focused on rendering.

The Eurozone is one logical "country" in our basket but renders as its 19
member states on the world map. EUROZONE_ISO3 holds the expansion list for
the choropleth.
"""
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.orm import Session

from dalio.countries import COUNTRIES, Country
from dalio.scoring.allocation import AllocationView, AssetTilt, compute_tilts
from dalio.scoring.long_term import (
    PhaseClassification,
)
from dalio.scoring.long_term import (
    classify as classify_long_term,
)
from dalio.scoring.short_term import (
    Classification,
)
from dalio.scoring.short_term import (
    classify as classify_short_term,
)

# ─── Concept-level explanations ────────────────────────────────────────────

EUROZONE_ISO3: tuple[str, ...] = (
    "AUT", "BEL", "CYP", "EST", "FIN", "FRA", "DEU", "GRC", "IRL", "ITA",
    "LVA", "LTU", "LUX", "MLT", "NLD", "PRT", "SVK", "SVN", "ESP", "HRV",
)


def expand_iso3_for_map(iso3: str) -> tuple[str, ...]:
    """Plot-time expansion: EU → 19 eurozone members for choropleth coloring."""
    if iso3 == "EMU":
        return EUROZONE_ISO3
    return (iso3,)


def map_iso3_to_country_iso2(iso3: str) -> str | None:
    """Reverse mapping for click handling: any eurozone iso3 → 'EU' in basket."""
    if iso3 in EUROZONE_ISO3:
        return "EU"
    iso3_to_iso2 = {"USA": "US", "CHN": "CN", "GBR": "UK", "JPN": "JP",
                    "SWE": "SE", "IND": "IN", "BRA": "BR"}
    return iso3_to_iso2.get(iso3)


PHASE_EXPLANATIONS: dict[int, str] = {
    1: "Sound money — low debt, high real rates, no fragility. Pre-1971-style.",
    2: "Debt outpaces income — credit growing faster than GDP, but still productive.",
    3: "Bubble — speculation builds, asset prices unhinge from cash flows. Be cautious.",
    4: "Top — debt-service burden peaks. Marginal borrower stops. Defensive posture.",
    5: "Ugly deleveraging — defaults rise, debt > income. Pain phase. Flight to safety.",
    6: "Beautiful deleveraging — central bank prints/inflates the debt away. Avoid long bonds, hold real assets.",
    7: "Reset — currency event, regime change. Out of model.",
    0: "Transition — cycle indicators conflict; classifier on a boundary.",
}

STAGE_EXPLANATIONS: dict[int, str] = {
    1: "Expansion — growth↑, inflation contained, unemployment falling. Risk-on environment.",
    2: "Inflationary peak — overheating: growth↑, inflation↑, central bank tightening. Inflation hedges win.",
    3: "Recession — growth↓, unemployment up, credit contracting. Long bonds rally; equities sell off.",
    4: "Reflation — central bank cutting, inflation moderating, recovery emerging. Mixed signals.",
    0: "Transition — short-term indicators conflict; usually a stage boundary.",
}

INDICATOR_EXPLANATIONS: dict[str, str] = {
    "policy_rate": "Central bank's target rate. Sets the floor for everything else.",
    "cpi_yoy": "Year-over-year price change. Above 3% = sticky inflation.",
    "unemployment_rate": "Headline jobless rate. Rising sharply (+0.5pp/3m) = recession (Sahm rule).",
    "yield_10y": "10-year govt borrowing cost. Reflects growth + inflation expectations.",
    "yield_2y": "2-year govt borrowing cost. Closer to the policy rate.",
    "real_gdp_yoy": "Actual economic growth, inflation-adjusted. Negative = recession.",
    "total_credit_pct_gdp": "Total non-financial debt as % of annual output. >250% = late cycle, >280% = extreme.",
    "gov_debt_pct_gdp": "Government debt alone. Drives fiscal sustainability concerns.",
    "private_nonfin_pct_gdp": "Households + non-financial corporates. Cyclical exposure.",
    "hh_debt_pct_gdp": "Household debt. High = vulnerable to mortgage-rate shocks.",
    "corp_debt_pct_gdp": "Non-financial corporate debt. High = cyclical risk.",
    "debt_service_ratio": "Interest + amortisation / income. >18% = stretched, >22% = distress.",
}


# ─── Per-country view ───────────────────────────────────────────────────────


@dataclass(frozen=True)
class CountryView:
    country: Country
    short_term: Classification
    long_term: PhaseClassification
    allocation: AllocationView


# Map ISO2 → currency code used by the home-currency overlay (Slice 14).
# Eurozone is treated as a single EUR view; Sweden is SEK; etc.
_HOME_COUNTRY_FOR_CURRENCY: dict[str, str] = {
    "USD": "US",
    "SEK": "SE",
    "EUR": "EU",
    "GBP": "UK",
    "JPY": "JP",
    "CNY": "CN",
}


def compute_country_view(
    session: Session,
    iso2: str,
    *,
    home_currency: str = "USD",
) -> CountryView:
    """Build the per-country view, optionally with a home-currency overlay
    on the allocation tilts. The home country's `real_rate_10y` is looked
    up from its own LongTermFeatures so `compute_tilts` itself stays DB-free.
    """
    country = next(c for c in COUNTRIES if c.iso2 == iso2)
    st = classify_short_term(session, iso2)
    lt = classify_long_term(session, iso2)

    home_real_rate = None
    if home_currency != "USD":
        home_iso2 = _HOME_COUNTRY_FOR_CURRENCY.get(home_currency)
        if home_iso2 and home_iso2 != iso2:
            home_lt = classify_long_term(session, home_iso2)
            home_real_rate = home_lt.features.real_rate_10y
        elif home_iso2 == iso2:
            # Viewing your own country in your own currency — overlay is no-op
            home_real_rate = lt.features.real_rate_10y

    alloc = compute_tilts(
        st, lt,
        home_currency=home_currency,
        home_real_rate_10y=home_real_rate,
    )
    return CountryView(country=country, short_term=st, long_term=lt, allocation=alloc)


def top_tilts(allocation: AllocationView, n: int = 3) -> list[AssetTilt]:
    """Return the N largest-magnitude tilts for the simplified summary card."""
    return sorted(allocation.tilts, key=lambda t: -abs(t.tilt))[:n]


# ─── World-map view ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CountryMapPoint:
    iso2: str
    iso3: str
    name: str
    has_data: bool
    long_term_phase: int
    long_term_label: str
    short_term_stage: int
    short_term_label: str
    caution_level: str          # "low" | "moderate" | "elevated" | "high"
    total_debt_pct_gdp: float | None
    debt_service_ratio: float | None
    cpi_yoy: float | None
    real_rate_10y: float | None
    hover_text: str


def _hover_for(view: CountryView) -> str:
    """Plain-language country card for the choropleth hover tooltip.

    Uses HTML <br> for line breaks (Plotly hover supports limited HTML).
    """
    f = view.long_term.features
    lines = [
        f"<b>{view.country.name}</b>",
        f"Phase: {view.long_term.phase_label} ({view.long_term.confidence:.0%})",
        f"Stage: {view.short_term.stage_label} ({view.short_term.confidence:.0%})",
    ]
    if f.total_credit_pct_gdp is not None:
        lines.append(f"Total debt/GDP: {f.total_credit_pct_gdp:.0f}%")
    if f.debt_service_ratio is not None:
        lines.append(f"DSR: {f.debt_service_ratio:.1f}%")
    rr = f.real_rate_10y
    if rr is not None:
        lines.append(f"Real rate: {rr:+.1f}%")
    lines.append(f"Caution: {view.allocation.caution_level}")
    return "<br>".join(lines)


def compute_world_view(session: Session) -> list[CountryMapPoint]:
    """Build map points for every country in the basket — including those
    without data (rendered greyed-out)."""
    from sqlalchemy import select

    from dalio.storage.db import Observation

    points: list[CountryMapPoint] = []
    for country in COUNTRIES:
        has_data = session.execute(
            select(Observation.id)
            .where(Observation.country == country.iso2)
            .limit(1)
        ).scalar_one_or_none() is not None

        if not has_data:
            points.append(CountryMapPoint(
                iso2=country.iso2,
                iso3=country.iso3,
                name=country.name,
                has_data=False,
                long_term_phase=0,
                long_term_label="No data",
                short_term_stage=0,
                short_term_label="No data",
                caution_level="moderate",
                total_debt_pct_gdp=None,
                debt_service_ratio=None,
                cpi_yoy=None,
                real_rate_10y=None,
                hover_text=f"<b>{country.name}</b><br>No data — slice 3 (Tier-2) pending",
            ))
            continue

        view = compute_country_view(session, country.iso2)
        f = view.long_term.features
        points.append(CountryMapPoint(
            iso2=country.iso2,
            iso3=country.iso3,
            name=country.name,
            has_data=True,
            long_term_phase=view.long_term.phase,
            long_term_label=view.long_term.phase_label,
            short_term_stage=view.short_term.stage,
            short_term_label=view.short_term.stage_label,
            caution_level=view.allocation.caution_level,
            total_debt_pct_gdp=f.total_credit_pct_gdp,
            debt_service_ratio=f.debt_service_ratio,
            cpi_yoy=f.cpi_yoy,
            real_rate_10y=f.real_rate_10y,
            hover_text=_hover_for(view),
        ))
    return points
