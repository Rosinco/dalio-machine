"""Streamlit dashboard for dalio-machine.

Layout (top → bottom):
  1. World map — clickable choropleth, color-coded by selected metric
  2. Country headline + plain-language quick read
  3. Three concept cards: long-term phase · short-term stage · top-3 tilts
  4. "How these connect" expander — framework explainer
  5. Detail expanders (collapsed by default):
        Short-term indicators · Long-term debt indicators · All tilts · History

Design priorities (Pocock-style discipline):
  - Map is the primary navigation. Sidebar selectbox is a fallback.
  - Every chart and metric carries a one-line explanation in plain language.
  - Detail is on demand: defaults are clean, expanders reveal the rest.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.app.views import (
    INDICATOR_EXPLANATIONS,
    PHASE_EXPLANATIONS,
    STAGE_EXPLANATIONS,
    CountryMapPoint,
    CountryView,
    compute_country_view,
    compute_world_view,
    expand_iso3_for_map,
    map_iso3_to_country_iso2,
    top_tilts,
)
from dalio.countries import COUNTRIES, Tier, get_country
from dalio.scoring.allocation import AllocationView
from dalio.scoring.long_term import PHASE_LABELS, PhaseClassification
from dalio.scoring.short_term import (
    SHORT_TERM_INDICATORS,
    STAGE_LABELS,
    Classification,
)
from dalio.storage.db import Observation, make_engine, make_session_factory

# ─── Visual constants ───────────────────────────────────────────────────────


STAGE_EMOJI: dict[int, str] = {
    1: "🟢", 2: "🟠", 3: "🔴", 4: "🔵", 0: "⚪",
}

PHASE_EMOJI: dict[int, str] = {
    1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🟣", 6: "🔵", 7: "⚫", 0: "⚪",
}

# Colors keyed by phase for the choropleth (red=danger, green=safe, blue=repression).
PHASE_HEX: dict[int, str] = {
    1: "#16a34a",   # green — sound money
    2: "#65a30d",   # yellow-green — debt outpacing
    3: "#eab308",   # yellow — bubble
    4: "#dc2626",   # red — top
    5: "#7c3aed",   # purple — deleveraging
    6: "#2563eb",   # blue — reflation/repression
    7: "#171717",   # black — reset
    0: "#a3a3a3",   # grey — transition
}

CAUTION_HEX: dict[str, str] = {
    "low": "#16a34a",
    "moderate": "#eab308",
    "elevated": "#f97316",
    "high": "#dc2626",
}

LONG_TERM_INDICATORS = (
    "total_credit_pct_gdp",
    "gov_debt_pct_gdp",
    "private_nonfin_pct_gdp",
    "hh_debt_pct_gdp",
    "corp_debt_pct_gdp",
    "debt_service_ratio",
)
ALL_INDICATORS = SHORT_TERM_INDICATORS + LONG_TERM_INDICATORS


# ─── Database helpers ──────────────────────────────────────────────────────


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


def _fmt_pct(v: float | None, suffix: str = "%") -> str:
    return f"{v:.2f}{suffix}" if v is not None else "n/a"


def _fmt_delta(v: float | None, suffix: str = "pp") -> str | None:
    return f"{v:+.2f}{suffix}" if v is not None else None


def _sparkline(session: Session, country: str, indicator: str, tail: int) -> None:
    hist = _load_history(session, country, indicator)
    if hist.empty:
        return
    st.line_chart(hist.tail(tail).set_index("date")["value"], height=80)


# ─── World map ─────────────────────────────────────────────────────────────


def _build_choropleth(
    points: list[CountryMapPoint], metric: str, selected_iso2: str | None,
) -> go.Figure:
    """Build a Plotly choropleth keyed by country ISO-3.

    metric ∈ {phase, stage, caution, total_debt}.
    Eurozone is expanded to its 19 members so the EU shows as a coherent block.
    """
    locations: list[str] = []
    z_values: list[float] = []
    colors: list[str] = []
    hover_texts: list[str] = []

    for p in points:
        if not p.has_data:
            color = "#e5e5e5"
            z = -1.0  # reserved zone in the discrete colorscale
        elif metric == "phase":
            color = PHASE_HEX.get(p.long_term_phase, "#a3a3a3")
            z = float(p.long_term_phase)
        elif metric == "stage":
            color = PHASE_HEX.get(p.short_term_stage, "#a3a3a3")  # reuse palette
            z = float(p.short_term_stage)
        elif metric == "caution":
            color = CAUTION_HEX.get(p.caution_level, "#a3a3a3")
            z = {"low": 1, "moderate": 2, "elevated": 3, "high": 4}.get(p.caution_level, 0)
        elif metric == "total_debt":
            d = p.total_debt_pct_gdp
            color = "#a3a3a3" if d is None else (
                "#16a34a" if d < 100
                else "#65a30d" if d < 200
                else "#eab308" if d < 250
                else "#f97316" if d < 300
                else "#dc2626"
            )
            z = d if d is not None else 0
        else:
            color = "#a3a3a3"
            z = 0

        for iso3 in expand_iso3_for_map(p.iso3):
            locations.append(iso3)
            z_values.append(z)
            colors.append(color)
            hover_texts.append(p.hover_text)

    # Custom discrete colorscale via per-cell color array
    fig = go.Figure(go.Choropleth(
        locations=locations,
        z=z_values,
        text=hover_texts,
        hoverinfo="text",
        locationmode="ISO-3",
        marker_line_color="#ffffff",
        marker_line_width=0.5,
        colorscale=[[0, "#a3a3a3"], [1, "#a3a3a3"]],  # base (unused — overridden)
        showscale=False,
    ))
    # Apply per-location color via a second invisible trace approach? Plotly's
    # Choropleth doesn't support per-cell colors directly via marker.color; we
    # need to use a color *scale* mapped through z. Do that:
    if metric in ("phase", "stage"):
        # Discrete categorical scale. z=-1 reserved for "no data" — visibly
        # distinct from z=0 "transition" (countries with data but on a boundary).
        fig.update_traces(
            colorscale=[
                [0.000, "#e5e5e5"],  # -1: no data (very light grey)
                [0.125, "#fbbf24"],  #  0: transition (amber — "look closer")
                [0.250, "#16a34a"],  #  1: sound / expansion
                [0.375, "#65a30d"],  #  2: outpacing / inflationary
                [0.500, "#eab308"],  #  3: bubble / recession
                [0.625, "#dc2626"],  #  4: top / reflation
                [0.750, "#7c3aed"],  #  5: deleveraging
                [0.875, "#2563eb"],  #  6: repression
                [1.000, "#171717"],  #  7: reset
            ],
            zmin=-1, zmax=7,
        )
    elif metric == "caution":
        fig.update_traces(
            colorscale=[
                [0.0, "#a3a3a3"],
                [0.25, "#16a34a"],
                [0.5, "#eab308"],
                [0.75, "#f97316"],
                [1.0, "#dc2626"],
            ],
            zmin=0, zmax=4,
        )
    else:  # total_debt continuous
        fig.update_traces(
            colorscale="RdYlGn_r",
            zmin=50, zmax=400,
        )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
            showland=True,
            landcolor="#f5f5f5",
            oceancolor="#ffffff",
            showocean=True,
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=420,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_world_map(points: list[CountryMapPoint], selected_iso2: str | None) -> str | None:
    """Render the choropleth + metric selector. Returns iso2 of clicked country
    if user clicked a location, else None."""
    metric_label, metric_key = st.radio(
        "Color the map by:",
        options=[
            ("Long-term phase", "phase"),
            ("Short-term stage", "stage"),
            ("Caution level", "caution"),
            ("Total debt / GDP", "total_debt"),
        ],
        format_func=lambda x: x[0],
        horizontal=True,
        key="map_metric",
    ) or ("Long-term phase", "phase")

    fig = _build_choropleth(points, metric_key, selected_iso2)
    selection = st.plotly_chart(
        fig,
        width="stretch",
        on_select="rerun",
        selection_mode=("points",),
        key="world_map_chart",
    )

    # Legend / explainer below map
    if metric_key == "phase":
        st.caption(
            "**🟢** Sound money &nbsp; **🟢** Outpacing &nbsp; **🟡** Bubble &nbsp; "
            "**🔴** Top (peak burden) &nbsp; **🟣** Deleveraging (distress) &nbsp; "
            "**🔵** Reflation/repression &nbsp; **🟠** Transition (on a boundary) &nbsp; "
            "**⚪** No data"
        )
    elif metric_key == "stage":
        st.caption(
            "**🟢** Expansion &nbsp; **🟢** Inflationary peak &nbsp; "
            "**🔴** Recession &nbsp; **🔵** Reflation &nbsp; "
            "**🟠** Transition &nbsp; **⚪** No data"
        )
    elif metric_key == "caution":
        st.caption(
            "🟢 Low &nbsp; 🟡 Moderate &nbsp; 🟠 Elevated &nbsp; 🔴 High — derived from long-term phase"
        )
    else:
        st.caption("Continuous scale from low debt (green) to extreme (red ≥300% of GDP).")

    # Click handling
    if selection and getattr(selection, "selection", None):
        pts = selection.selection.get("points", []) if isinstance(selection.selection, dict) else []
        if pts:
            clicked_iso3 = pts[0].get("location")
            if clicked_iso3:
                return map_iso3_to_country_iso2(clicked_iso3)
    return None


# ─── Simplified country summary ─────────────────────────────────────────────


def _render_quick_read(view: CountryView) -> None:
    """One-paragraph plain-language read of the current regime."""
    f_lt = view.long_term.features
    debt = f"{f_lt.total_credit_pct_gdp:.0f}%" if f_lt.total_credit_pct_gdp else "n/a"
    cpi = f"{f_lt.cpi_yoy:.1f}%" if f_lt.cpi_yoy else "n/a"
    rr = f"{f_lt.real_rate_10y:+.1f}%" if f_lt.real_rate_10y is not None else "n/a"
    dsr = f"{f_lt.debt_service_ratio:.1f}%" if f_lt.debt_service_ratio else "n/a"

    st.markdown(
        f"### {view.country.name} — quick read\n"
        f"Total non-fin debt **{debt}** of GDP &nbsp;·&nbsp; "
        f"CPI YoY **{cpi}** &nbsp;·&nbsp; "
        f"Real 10y rate **{rr}** &nbsp;·&nbsp; "
        f"Debt-service ratio **{dsr}**"
    )


def _render_summary_cards(view: CountryView) -> None:
    """Three concept cards: long-term phase, short-term stage, top-3 tilts."""
    cols = st.columns(3)

    # Card 1: Long-term phase
    with cols[0]:
        emoji = PHASE_EMOJI.get(view.long_term.phase, "⚪")
        st.markdown(f"#### {emoji} Long-term phase")
        st.markdown(f"**{view.long_term.phase_label}**")
        st.caption(PHASE_EXPLANATIONS.get(view.long_term.phase, ""))
        st.progress(min(view.long_term.confidence, 1.0), text=f"Confidence: {view.long_term.confidence:.0%}")

    # Card 2: Short-term stage
    with cols[1]:
        emoji = STAGE_EMOJI.get(view.short_term.stage, "⚪")
        st.markdown(f"#### {emoji} Short-term stage")
        st.markdown(f"**{view.short_term.stage_label}**")
        st.caption(STAGE_EXPLANATIONS.get(view.short_term.stage, ""))
        st.progress(min(view.short_term.confidence, 1.0), text=f"Confidence: {view.short_term.confidence:.0%}")

    # Card 3: Top tilts
    with cols[2]:
        caution_emoji = {
            "low": "🟢", "moderate": "🟡",
            "elevated": "🟠", "high": "🔴",
        }.get(view.allocation.caution_level, "⚪")
        st.markdown(f"#### {caution_emoji} Allocation tilt")
        st.markdown(f"**Caution: {view.allocation.caution_level}**")
        for t in top_tilts(view.allocation, n=3):
            label = t.label.split(" (")[0]  # drop parenthetical
            st.markdown(f"&nbsp;&nbsp;{t.direction}&nbsp; **{label}** &nbsp;`{t.tilt:+.2f}`")
        st.caption("Tilts = deviations from a default diversified base. Not buy/sell signals.")


# ─── Framework note ─────────────────────────────────────────────────────────


def _render_framework_note() -> None:
    with st.expander("📚 How these pieces fit together (read once)"):
        st.markdown("""
**The dashboard tracks two cycles + an allocation overlay:**

1. **Short-term cycle** (5–8 years) — the business cycle. Driven by the central
   bank: tighten → recession → cut → recovery → tighten again. Affects
   *equities vs bonds* most.

2. **Long-term debt cycle** (50–75 years) — the slow build-up and resolution
   of total debt. Modern developed economies have all left "Sound money" and
   sit somewhere between Phase 3 (Bubble) and Phase 6 (Reflation). Affects
   *bonds vs gold/TIPS* most.

3. **Allocation tilts** combine both layers. Each cycle position implies
   asset preferences from Dalio's Growth × Inflation matrix. Tilts are
   *deviations* from a default diversified base — never absolute weights.

**Key relationships to read across the indicators:**

- **Yield curve (10y − 2y)**: inverts before recessions. When negative,
  Stage 3 (Recession) is likely 6–18 months out.
- **Debt service ratio (DSR)**: above 18% = stretched, above 22% = household
  distress (Sweden right now). High DSR + falling debt + low inflation =
  Phase 5 (ugly deleveraging). High DSR + high inflation = Phase 6 (beautiful).
- **Real 10y rate**: positive = savers compensated. Negative = financial
  repression — long nominal bonds are a bad place to be; gold and TIPS win.
- **Sahm rule**: when unemployment rises +0.5pp over 3 months, a recession
  has historically followed every time since 1970.
        """)


# ─── Detail panels (existing) ──────────────────────────────────────────────


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
        with st.expander("Rule reasoning"):
            for v in c.votes:
                st.markdown(
                    f"- **{STAGE_LABELS[v.stage]}** (weight `{v.weight:.2f}`) — {v.reason}"
                )


def _render_indicator_grid(session: Session, country: str, c: Classification) -> None:
    f = c.features

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "Policy rate",
            _fmt_pct(f.policy_rate),
            delta=_fmt_delta(f.policy_rate_change_6m) and f"{_fmt_delta(f.policy_rate_change_6m)} / 6m",
            delta_color="off",
            help=INDICATOR_EXPLANATIONS.get("policy_rate"),
        )
        _sparkline(session, country, "policy_rate", tail=750)
    with cols[1]:
        st.metric("10-year yield", _fmt_pct(f.yield_10y),
                  help=INDICATOR_EXPLANATIONS.get("yield_10y"))
        _sparkline(session, country, "yield_10y", tail=750)
    with cols[2]:
        st.metric("2-year yield", _fmt_pct(f.yield_2y),
                  help=INDICATOR_EXPLANATIONS.get("yield_2y"))
        _sparkline(session, country, "yield_2y", tail=750)

    if f.yield_curve_slope is not None:
        slope = f.yield_curve_slope
        slope_msg = f"**Yield curve (10y − 2y): {slope:+.2f}pp**"
        if slope < 0:
            st.error(f"⚠️ {slope_msg} — inverted, historical recession leading indicator")
        elif slope < 0.5:
            st.warning(f"{slope_msg} — flat, late-cycle signal")
        else:
            st.info(f"{slope_msg} — normal positive")

    cols = st.columns(3)
    with cols[0]:
        st.metric(
            "CPI YoY",
            _fmt_pct(f.cpi_yoy),
            delta=_fmt_delta(f.cpi_change_3m) and f"{_fmt_delta(f.cpi_change_3m)} / 3m",
            delta_color="inverse",
            help=INDICATOR_EXPLANATIONS.get("cpi_yoy"),
        )
        _sparkline(session, country, "cpi_yoy", tail=60)
    with cols[1]:
        st.metric(
            "Unemployment",
            _fmt_pct(f.unemployment_rate),
            delta=_fmt_delta(f.unemployment_change_3m) and f"{_fmt_delta(f.unemployment_change_3m)} / 3m",
            delta_color="inverse",
            help=INDICATOR_EXPLANATIONS.get("unemployment_rate"),
        )
        _sparkline(session, country, "unemployment_rate", tail=60)
    with cols[2]:
        st.metric("Real GDP YoY", _fmt_pct(f.real_gdp_yoy),
                  help=INDICATOR_EXPLANATIONS.get("real_gdp_yoy"))
        _sparkline(session, country, "real_gdp_yoy", tail=40)


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
    else:
        st.info(body)

    if c.votes:
        with st.expander("Rule reasoning"):
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
            delta=delta, delta_color="inverse",
            help=INDICATOR_EXPLANATIONS.get("total_credit_pct_gdp"),
        )
        _sparkline(session, country, "total_credit_pct_gdp", tail=80)
    with cols[1]:
        st.metric("Government debt / GDP", _fmt_pct(f.gov_debt_pct_gdp),
                  help=INDICATOR_EXPLANATIONS.get("gov_debt_pct_gdp"))
        _sparkline(session, country, "gov_debt_pct_gdp", tail=80)
    with cols[2]:
        delta_dsr = None
        if f.dsr_5y_change_pp is not None:
            delta_dsr = f"{f.dsr_5y_change_pp:+.1f}pp / 5y"
        st.metric(
            "Debt service ratio (private)",
            _fmt_pct(f.debt_service_ratio),
            delta=delta_dsr, delta_color="inverse",
            help=INDICATOR_EXPLANATIONS.get("debt_service_ratio"),
        )
        _sparkline(session, country, "debt_service_ratio", tail=80)

    cols = st.columns(3)
    with cols[0]:
        st.metric("Households / GDP", _fmt_pct(f.hh_debt_pct_gdp),
                  help=INDICATOR_EXPLANATIONS.get("hh_debt_pct_gdp"))
        _sparkline(session, country, "hh_debt_pct_gdp", tail=80)
    with cols[1]:
        st.metric("Non-fin corps / GDP", _fmt_pct(f.corp_debt_pct_gdp),
                  help=INDICATOR_EXPLANATIONS.get("corp_debt_pct_gdp"))
        _sparkline(session, country, "corp_debt_pct_gdp", tail=80)
    with cols[2]:
        rr = f.real_rate_10y
        st.metric("Real 10y rate", _fmt_pct(rr),
                  help="10y nominal yield − CPI YoY. Negative = financial repression.")
        if rr is not None:
            if rr < -1:
                st.caption("⚠️ Financial repression — savers punished")
            elif rr < 0:
                st.caption("Mildly negative — gentle debt erosion")
            else:
                st.caption("Positive — savers compensated")


def _render_allocation(view: AllocationView) -> None:
    caution = view.caution_level
    caution_color = {
        "low": "🟢", "moderate": "🟡", "elevated": "🟠", "high": "🔴",
    }.get(caution, "⚪")

    body = f"**Caution level: {caution_color} {caution.upper()}**\n\n{view.summary}"
    if caution in ("elevated", "high"):
        st.warning(body)
    else:
        st.info(body)

    sorted_tilts = sorted(view.tilts, key=lambda t: -abs(t.tilt))
    rows = [{
        "Direction": t.direction,
        "Asset class": t.label,
        "Tilt": f"{t.tilt:+.2f}",
        "Reasoning": " · ".join(t.reasons) if t.reasons else "—",
    } for t in sorted_tilts]
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.caption(
        "💡 **Tilts are deviations from a default diversified base portfolio, "
        "not absolute weights.** Magnitudes confidence-weighted: low-confidence "
        "regimes produce small tilts. Decision-support, not market timing."
    )


def _render_history_explorer(session: Session, country: str) -> None:
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

    indicator = st.selectbox(
        "Indicator",
        available,
        help="Pick any tracked indicator to see its full historical series.",
    )
    explanation = INDICATOR_EXPLANATIONS.get(indicator, "")
    if explanation:
        st.caption(f"_{explanation}_")
    hist = _load_history(session, country, indicator)
    st.line_chart(hist.set_index("date")["value"], height=320)
    st.caption(
        f"{len(hist):,} observations from "
        f"{hist['date'].min()} to {hist['date'].max()}"
    )


# ─── Main entry ────────────────────────────────────────────────────────────


def _resolve_selected_country() -> str:
    """Determine which country to display, with sidebar selectbox + map click
    both writing to st.session_state.country."""
    if "country" not in st.session_state:
        st.session_state.country = "US"
    return st.session_state.country


def main() -> None:
    load_dotenv()
    st.set_page_config(
        page_title="Dalio Machine",
        layout="wide",
        page_icon="📊",
    )

    st.title("📊 dalio-machine")
    st.caption(
        "Macro-cycle dashboard built on Ray Dalio's economic-machine framework. "
        "**Decision-support for allocation tilts, not market-timing signals.**"
    )

    with _open_session() as s:
        countries_with_data = {c.iso2 for c in COUNTRIES if _has_data(s, c.iso2)}
        points = compute_world_view(s)

    # ─── Sidebar (country selector) ────────────────────────────────
    def _format(iso2: str) -> str:
        c = get_country(iso2)
        suffix = "" if iso2 in countries_with_data else "  (no data)"
        tier_tag = "" if c.tier == Tier.TIER_1 else "  · Tier 2"
        return f"{c.name}{tier_tag}{suffix}"

    if "country" not in st.session_state:
        st.session_state.country = "US"
    selected = st.sidebar.selectbox(
        "Country",
        [c.iso2 for c in COUNTRIES],
        format_func=_format,
        key="country",
    )
    country = get_country(selected)

    if country.tier == Tier.TIER_2:
        st.sidebar.warning(
            f"**Tier 2** — coverage thinner for {country.name}. "
            "Lower implicit confidence."
        )
    st.sidebar.divider()
    st.sidebar.markdown(
        f"**{country.name}**  \n"
        f"Currency: `{country.currency}`  \n"
        f"Central bank: {country.central_bank}"
    )

    # ─── World map ────────────────────────────────────────────────
    st.subheader("Global cycle overview")
    clicked_iso2 = _render_world_map(points, selected_iso2=selected)
    if clicked_iso2 and clicked_iso2 != selected:
        st.session_state.country = clicked_iso2
        st.rerun()

    st.divider()

    # ─── Country detail ───────────────────────────────────────────
    if selected not in countries_with_data:
        st.warning(
            f"No data for **{country.name}** yet. "
            "Slice 3 (Tier-2 fan-out: IN, BR) is the next step."
        )
        return

    with _open_session() as s:
        view = compute_country_view(s, selected)

    _render_quick_read(view)
    st.markdown("&nbsp;")
    _render_summary_cards(view)
    st.markdown("&nbsp;")
    _render_framework_note()

    st.divider()

    # ─── Detail expanders ─────────────────────────────────────────
    with st.expander("🟦 Short-term cycle indicators"):
        _render_stage_card(view.short_term)
        st.markdown("##### Indicators")
        with _open_session() as s:
            _render_indicator_grid(s, selected, view.short_term)

    with st.expander("🟥 Long-term debt cycle indicators (BIS data)"):
        _render_long_term_card(view.long_term)
        st.markdown("##### Sector debt + DSR")
        with _open_session() as s:
            _render_long_term_indicators(s, selected, view.long_term)

    with st.expander("📊 All asset-allocation tilts (full table)"):
        _render_allocation(view.allocation)

    with st.expander("📈 History explorer (any indicator, full series)"), _open_session() as s:
        _render_history_explorer(s, selected)


if __name__ == "__main__":
    main()
