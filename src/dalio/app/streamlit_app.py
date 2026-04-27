"""Streamlit dashboard — slice 1 short-term debt cycle.

Country-selector layout: stage classification card + reasoning, indicator metric
cards with sparklines, yield-curve banner, history explorer.

Multi-country ready — countries without data render a "no data yet" placeholder
so slice 2/3 fan-out is a data-only change.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import COUNTRIES, Tier, get_country
from dalio.scoring.long_term import PHASE_LABELS, PhaseClassification
from dalio.scoring.long_term import classify as classify_long_term
from dalio.scoring.short_term import (
    SHORT_TERM_INDICATORS,
    STAGE_LABELS,
    Classification,
)
from dalio.scoring.short_term import classify as classify_short_term
from dalio.storage.db import Observation, make_engine, make_session_factory

STAGE_EMOJI: dict[int, str] = {
    1: "🟢",  # Expansion
    2: "🟠",  # Inflationary peak
    3: "🔴",  # Recession
    4: "🔵",  # Reflation
    0: "⚪",  # Transition / insufficient data
}

PHASE_EMOJI: dict[int, str] = {
    1: "🟢",  # Sound money
    2: "🟡",  # Debt outpaces income
    3: "🟠",  # Bubble
    4: "🔴",  # Top
    5: "🟣",  # Deleveraging
    6: "🔵",  # Reflation/repression
    7: "⚫",  # Reset
    0: "⚪",  # Transition
}


@st.cache_resource
def _engine():
    return make_engine()


def _open_session() -> Session:
    return make_session_factory(_engine())()


def _load_history(session: Session, country: str, indicator: str) -> pd.DataFrame:
    rows = session.execute(
        select(Observation)
        .where(Observation.country == country, Observation.indicator == indicator)
        .order_by(Observation.date.asc())
    ).scalars().all()
    if not rows:
        return pd.DataFrame(columns=["date", "value"])
    return pd.DataFrame([{"date": r.date, "value": r.value} for r in rows])


def _has_data(session: Session, country: str) -> bool:
    return session.execute(
        select(Observation.id).where(Observation.country == country).limit(1)
    ).scalar_one_or_none() is not None


def _render_stage_card(c: Classification) -> None:
    emoji = STAGE_EMOJI[c.stage]
    header = f"### Cycle stage: {emoji} **{c.stage_label}**"
    detail = f"Confidence **{c.confidence:.0%}**  •  As of {c.features.as_of}"

    body = f"{header}\n\n{detail}"
    if c.stage == 1:
        st.success(body)
    elif c.stage == 2:
        st.warning(body)
    elif c.stage == 3:
        st.error(body)
    else:
        st.info(body)

    if c.votes:
        with st.expander("Rule reasoning — why this stage?"):
            for v in c.votes:
                stage_label = STAGE_LABELS[v.stage]
                st.markdown(
                    f"- **{stage_label}** (weight `{v.weight:.2f}`) — {v.reason}"
                )
    else:
        st.warning("No rules fired — likely missing indicators in DB.")


def _fmt_pct(v: float | None, suffix: str = "%") -> str:
    return f"{v:.2f}{suffix}" if v is not None else "n/a"


def _fmt_delta(v: float | None, suffix: str = "pp") -> str | None:
    return f"{v:+.2f}{suffix}" if v is not None else None


def _sparkline(session: Session, country: str, indicator: str, tail: int) -> None:
    hist = _load_history(session, country, indicator)
    if hist.empty:
        return
    st.line_chart(hist.tail(tail).set_index("date")["value"], height=80)


def _render_indicator_grid(session: Session, country: str, c: Classification) -> None:
    f = c.features

    # Row 1: rates
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Policy rate",
            _fmt_pct(f.policy_rate),
            delta=_fmt_delta(f.policy_rate_change_6m) and f"{_fmt_delta(f.policy_rate_change_6m)} / 6m",
            delta_color="off",
        )
        _sparkline(session, country, "policy_rate", tail=750)
    with cols[1]:
        st.metric("10-year yield", _fmt_pct(f.yield_10y))
        _sparkline(session, country, "yield_10y", tail=750)
    with cols[2]:
        st.metric("2-year yield", _fmt_pct(f.yield_2y))
        _sparkline(session, country, "yield_2y", tail=750)

    # Yield-curve banner
    if f.yield_curve_slope is not None:
        slope = f.yield_curve_slope
        slope_msg = f"**Yield curve (10y − 2y): {slope:+.2f}pp**"
        if slope < 0:
            st.error(f"⚠️ {slope_msg} — inverted, historical recession leading indicator")
        elif slope < 0.5:
            st.warning(f"{slope_msg} — flat, late-cycle signal")
        else:
            st.info(f"{slope_msg} — normal positive")

    # Row 2: macro
    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "CPI YoY",
            _fmt_pct(f.cpi_yoy),
            delta=_fmt_delta(f.cpi_change_3m) and f"{_fmt_delta(f.cpi_change_3m)} / 3m",
            delta_color="inverse",
        )
        _sparkline(session, country, "cpi_yoy", tail=60)
    with cols[1]:
        st.metric(
            "Unemployment",
            _fmt_pct(f.unemployment_rate),
            delta=_fmt_delta(f.unemployment_change_3m) and f"{_fmt_delta(f.unemployment_change_3m)} / 3m",
            delta_color="inverse",
        )
        _sparkline(session, country, "unemployment_rate", tail=60)
    with cols[2]:
        st.metric("Real GDP YoY", _fmt_pct(f.real_gdp_yoy))
        _sparkline(session, country, "real_gdp_yoy", tail=40)


LONG_TERM_INDICATORS = (
    "total_credit_pct_gdp",
    "gov_debt_pct_gdp",
    "private_nonfin_pct_gdp",
    "hh_debt_pct_gdp",
    "corp_debt_pct_gdp",
    "debt_service_ratio",
)

ALL_INDICATORS = SHORT_TERM_INDICATORS + LONG_TERM_INDICATORS


def _render_long_term_card(c: PhaseClassification) -> None:
    emoji = PHASE_EMOJI[c.phase]
    header = f"### Long-term cycle phase: {emoji} **{c.phase_label}**"
    detail = f"Confidence **{c.confidence:.0%}**  •  As of {c.features.as_of}"
    body = f"{header}\n\n{detail}"

    if c.phase == 1:
        st.success(body)
    elif c.phase == 2:
        st.info(body)
    elif c.phase == 3:
        st.warning(body)
    elif c.phase in (4, 5):
        st.error(body)
    elif c.phase == 6:
        st.info(body)
    else:
        st.info(body)

    if c.votes:
        with st.expander("Rule reasoning — why this phase?"):
            for v in c.votes:
                st.markdown(
                    f"- **{PHASE_LABELS[v.phase]}** (weight `{v.weight:.2f}`) — {v.reason}"
                )


def _render_long_term_indicators(session: Session, country: str, c: PhaseClassification) -> None:
    f = c.features

    cols = st.columns(3)
    with cols[0]:
        delta = None
        if f.total_credit_5y_change_pp is not None:
            delta = f"{f.total_credit_5y_change_pp:+.0f}pp / 5y"
        st.metric(
            "Total non-fin debt / GDP",
            _fmt_pct(f.total_credit_pct_gdp),
            delta=delta,
            delta_color="inverse",
        )
        _sparkline(session, country, "total_credit_pct_gdp", tail=80)
    with cols[1]:
        st.metric("Government debt / GDP", _fmt_pct(f.gov_debt_pct_gdp))
        _sparkline(session, country, "gov_debt_pct_gdp", tail=80)
    with cols[2]:
        delta_dsr = None
        if f.dsr_5y_change_pp is not None:
            delta_dsr = f"{f.dsr_5y_change_pp:+.1f}pp / 5y"
        st.metric(
            "Debt service ratio (private)",
            _fmt_pct(f.debt_service_ratio),
            delta=delta_dsr,
            delta_color="inverse",
        )
        _sparkline(session, country, "debt_service_ratio", tail=80)

    cols = st.columns(3)
    with cols[0]:
        st.metric("Households / GDP", _fmt_pct(f.hh_debt_pct_gdp))
        _sparkline(session, country, "hh_debt_pct_gdp", tail=80)
    with cols[1]:
        st.metric("Non-fin corps / GDP", _fmt_pct(f.corp_debt_pct_gdp))
        _sparkline(session, country, "corp_debt_pct_gdp", tail=80)
    with cols[2]:
        rr = f.real_rate_10y
        rr_str = _fmt_pct(rr)
        st.metric("Real 10y rate (10y − CPI)", rr_str)
        if rr is not None:
            if rr < -1:
                st.caption("⚠️ Financial repression — real-rate punishes savers")
            elif rr < 0:
                st.caption("Mildly negative — gentle debt erosion")
            else:
                st.caption("Positive — savers compensated")


def _render_history_explorer(session: Session, country: str) -> None:
    st.subheader("History explorer")
    available = [
        ind for ind in ALL_INDICATORS
        if session.execute(
            select(Observation.id)
            .where(Observation.country == country, Observation.indicator == ind)
            .limit(1)
        ).scalar_one_or_none() is not None
    ]
    if not available:
        st.info("No history available.")
        return

    indicator = st.selectbox("Indicator", available)
    hist = _load_history(session, country, indicator)
    st.line_chart(hist.set_index("date")["value"], height=320)
    st.caption(
        f"{len(hist):,} observations from "
        f"{hist['date'].min()} to {hist['date'].max()}"
    )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Dalio Machine", layout="wide")

    st.title("📊 dalio-machine")
    st.caption(
        "Macro-cycle dashboard built on Ray Dalio's economic machine framework. "
        "**Decision-support for allocation tilts, not market-timing signals.**"
    )

    # Country picker
    with _open_session() as s:
        countries_with_data = {c.iso2 for c in COUNTRIES if _has_data(s, c.iso2)}

    def _format(iso2: str) -> str:
        c = get_country(iso2)
        suffix = "" if iso2 in countries_with_data else "  (no data yet)"
        tier_tag = "" if c.tier == Tier.TIER_1 else "  · Tier 2"
        return f"{c.name}{tier_tag}{suffix}"

    selected = st.sidebar.selectbox(
        "Country",
        [c.iso2 for c in COUNTRIES],
        format_func=_format,
    )
    country = get_country(selected)

    if country.tier == Tier.TIER_2:
        st.sidebar.warning(
            f"**Tier 2** — coverage for {country.name} is thinner. "
            "Stage classifications carry lower implicit confidence."
        )

    st.sidebar.divider()
    st.sidebar.markdown(
        f"**{country.name}**  \n"
        f"Currency: `{country.currency}`  \n"
        f"Central bank: {country.central_bank}"
    )

    if selected not in countries_with_data:
        st.warning(
            f"No data for **{country.name}** yet. ETL fan-out is "
            "slice 2 (Tier 1 — CN, EU, UK, JP, SE) and slice 3 (Tier 2 — IN, BR)."
        )
        st.info(
            "FRED has limited international coverage — slice 2 will use "
            "BIS / IMF / OECD SDMX APIs for cross-country indicators."
        )
        return

    st.subheader(f"Short-term debt cycle — {country.name}")
    with _open_session() as s:
        st_classification = classify_short_term(s, selected)
    _render_stage_card(st_classification)

    st.divider()
    st.subheader("Short-term indicators")
    with _open_session() as s:
        _render_indicator_grid(s, selected, st_classification)

    st.divider()
    st.subheader(f"Long-term debt cycle — {country.name}")
    with _open_session() as s:
        lt_classification = classify_long_term(s, selected)
    _render_long_term_card(lt_classification)

    st.divider()
    st.subheader("Long-term indicators (BIS Total Credit + DSR)")
    with _open_session() as s:
        _render_long_term_indicators(s, selected, lt_classification)

    st.divider()
    with _open_session() as s:
        _render_history_explorer(s, selected)


if __name__ == "__main__":
    main()
