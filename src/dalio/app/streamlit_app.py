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

from datetime import date

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

# Editorial palette — muted, painterly. Each phase reads as a sentiment.
# Sage / olive = calm. Ochre / terracotta = warning. Oxblood / slate = distress / regime change.
PHASE_HEX: dict[int, str] = {
    1: "#506e58",   # sage — sound money
    2: "#708060",   # olive — debt outpacing
    3: "#b8893a",   # ochre — bubble
    4: "#a14a3a",   # terracotta — top
    5: "#6b3c4a",   # oxblood — deleveraging
    6: "#3a587a",   # slate blue — reflation/repression
    7: "#2a2a2a",   # charcoal — reset
    0: "#c69e3f",   # warm amber — transition (look closer)
}

CAUTION_HEX: dict[str, str] = {
    "low": "#506e58",
    "moderate": "#c69e3f",
    "elevated": "#b8893a",
    "high": "#a14a3a",
}

# Page palette — used both in CSS and to style plotly figures.
INK = "#0c1f3f"
INK_MUTED = "#5a5852"
PAPER = "#faf6ef"
PAPER_ELEV = "#f1ebde"
LAND = "#e8e0cb"
RULE = "#cdc6b6"
RUST = "#b94e23"
NO_DATA = "#d9d4c5"


def _inject_design_css() -> None:
    """Editorial macro-research aesthetic. Cream parchment, navy ink,
    Source Serif 4 / Inter Tight / JetBrains Mono. Hairline rules over
    drop-shadows, kicker labels over emoji.

    Called once at the top of main(). Streamlit-specific selectors target
    the rendered DOM (data-testid attributes) so future Streamlit upgrades
    may need updating here.
    """
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,800;1,8..60,400;1,8..60,600&family=Inter+Tight:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

:root {
  --bg: #faf6ef;
  --bg-elev: #f1ebde;
  --ink: #0c1f3f;
  --ink-muted: #5a5852;
  --rule: #cdc6b6;
  --rust: #b94e23;
  --display: 'Source Serif 4', 'Source Serif Pro', Georgia, serif;
  --body: 'Inter Tight', 'Inter', system-ui, sans-serif;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
}

html, body, .stApp,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"] {
  background: var(--bg) !important;
  color: var(--ink) !important;
  font-family: var(--body) !important;
}
[data-testid="stHeader"] { box-shadow: none !important; }

.block-container {
  padding-top: 2.5rem !important;
  padding-bottom: 4rem !important;
  max-width: 1240px !important;
}

[data-testid="stSidebar"] {
  background: var(--bg-elev) !important;
  border-right: 1px solid var(--rule) !important;
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label {
  font-family: var(--body) !important;
  font-size: 0.85rem !important;
  color: var(--ink) !important;
}

h1, h2, h3, h4, h5, h6 {
  font-family: var(--display) !important;
  color: var(--ink) !important;
  letter-spacing: -0.01em !important;
  font-weight: 700 !important;
}
h1 { font-size: 3rem !important; font-weight: 800 !important; letter-spacing: -0.025em !important; line-height: 1.05 !important; }
h2 { font-size: 1.55rem !important; }
h3 { font-size: 1.25rem !important; }
h4 { font-size: 1.05rem !important; }

p, li, label,
[data-testid="stMarkdownContainer"] p,
[data-testid="stCaptionContainer"] {
  font-family: var(--body) !important;
  color: var(--ink) !important;
  line-height: 1.55;
}

[data-testid="stCaptionContainer"],
.stCaption,
small {
  color: var(--ink-muted) !important;
  font-size: 0.85rem !important;
}

code, kbd, pre {
  font-family: var(--mono) !important;
  background: var(--bg-elev) !important;
  color: var(--ink) !important;
  border: 1px solid var(--rule) !important;
  padding: 0 0.3rem !important;
  border-radius: 0 !important;
}

/* ─── Custom blocks ──────────────────────────────────────────────── */

.masthead {
  border-top: 4px solid var(--ink);
  border-bottom: 1px solid var(--rule);
  padding: 0.9rem 0 1.4rem;
  margin-bottom: 2rem;
}
.masthead .meta {
  font-family: var(--body);
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--ink-muted);
  display: flex;
  justify-content: space-between;
  border-bottom: 1px solid var(--rule);
  padding-bottom: 0.5rem;
  margin-bottom: 1rem;
}
.masthead .title {
  font-family: var(--display);
  font-weight: 800;
  font-size: 3.4rem;
  letter-spacing: -0.03em;
  line-height: 0.95;
  margin: 0;
  color: var(--ink);
}
.masthead .subtitle {
  font-family: var(--display);
  font-style: italic;
  font-weight: 400;
  font-size: 1.1rem;
  color: var(--ink-muted);
  margin: 0.7rem 0 0;
  max-width: 56rem;
}

.kicker {
  font-family: var(--body);
  text-transform: uppercase;
  font-size: 0.7rem;
  letter-spacing: 0.2em;
  font-weight: 700;
  color: var(--rust);
  margin: 2.4rem 0 0.3rem;
  border-top: 1.5px solid var(--ink);
  padding-top: 0.55rem;
  display: block;
}
.kicker.no-rule { border-top: none; padding-top: 0; }
.section-title {
  font-family: var(--display) !important;
  font-weight: 700 !important;
  font-size: 1.7rem !important;
  margin: 0 0 0.3rem 0 !important;
  letter-spacing: -0.015em !important;
}
.section-lede {
  font-family: var(--display);
  font-style: italic;
  color: var(--ink-muted);
  font-size: 1rem;
  margin: 0 0 1rem 0;
  max-width: 50rem;
}

.pullquote {
  font-family: var(--display);
  font-size: 1.15rem;
  line-height: 1.55;
  color: var(--ink);
  border-left: 3px solid var(--rust);
  padding: 0.4rem 0 0.4rem 1.1rem;
  margin: 0.4rem 0 1.6rem;
  font-weight: 400;
}
.pullquote .num {
  font-family: var(--mono);
  font-weight: 500;
  color: var(--ink);
  background: transparent !important;
  border: none !important;
  padding: 0 !important;
}

.card {
  border-top: 1.5px solid var(--ink);
  padding: 0.85rem 1rem 0.5rem 0;
  height: 100%;
}
.card .eyebrow {
  font-family: var(--body);
  text-transform: uppercase;
  font-size: 0.65rem;
  letter-spacing: 0.18em;
  font-weight: 700;
  color: var(--ink-muted);
  margin: 0 0 0.6rem;
}
.card .eyebrow .accent { color: var(--rust); }
.card .title {
  font-family: var(--display);
  font-weight: 700;
  font-size: 1.55rem;
  line-height: 1.1;
  margin: 0 0 0.45rem;
  color: var(--ink);
  letter-spacing: -0.012em;
}
.card .deck {
  font-family: var(--body);
  font-size: 0.92rem;
  color: var(--ink-muted);
  margin: 0 0 0.9rem;
  line-height: 1.5;
}
.card .tilt-row {
  font-family: var(--body);
  font-size: 0.95rem;
  display: flex;
  gap: 0.6rem;
  align-items: baseline;
  border-bottom: 1px dashed var(--rule);
  padding: 0.35rem 0;
}
.card .tilt-row:last-child { border-bottom: none; }
.card .tilt-arrow {
  font-weight: 700;
  min-width: 1.9rem;
  color: var(--ink);
  font-family: var(--body);
}
.card .tilt-label { flex: 1; color: var(--ink); }
.card .tilt-num {
  font-family: var(--mono);
  font-size: 0.85rem;
  color: var(--ink-muted);
}

.confidence-track {
  height: 3px;
  background: var(--rule);
  margin: 1rem 0 0.35rem;
  position: relative;
}
.confidence-fill {
  display: block;
  height: 100%;
  background: var(--ink);
}
.confidence-meta {
  font-family: var(--body);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.16em;
  color: var(--ink-muted);
  display: flex;
  justify-content: space-between;
}

.legend-row {
  font-family: var(--body);
  font-size: 0.78rem;
  color: var(--ink-muted);
  display: flex;
  flex-wrap: wrap;
  gap: 1.1rem 1.6rem;
  margin: 0.4rem 0 0.4rem;
}
.legend-row .swatch {
  display: inline-block;
  width: 0.78rem;
  height: 0.78rem;
  margin-right: 0.45rem;
  vertical-align: -0.1rem;
  border: 1px solid rgba(0,0,0,0.15);
}

/* ─── Streamlit element overrides ──────────────────────────────── */

[data-testid="stMetric"] {
  background: transparent !important;
  border: none !important;
  border-top: 1px solid var(--rule) !important;
  padding: 0.55rem 0 0.4rem !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
  font-family: var(--body) !important;
  font-size: 0.7rem !important;
  text-transform: uppercase;
  letter-spacing: 0.14em !important;
  color: var(--ink-muted) !important;
  font-weight: 600 !important;
}
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-weight: 500 !important;
  font-size: 1.6rem !important;
  color: var(--ink) !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 0.78rem !important;
}

[data-testid="stExpander"] {
  border: 1px solid var(--rule) !important;
  border-radius: 0 !important;
  background: transparent !important;
  margin-bottom: 0.7rem !important;
  box-shadow: none !important;
}
[data-testid="stExpander"] details > summary,
[data-testid="stExpander"] summary {
  font-family: var(--body) !important;
  font-weight: 600 !important;
  color: var(--ink) !important;
  font-size: 0.95rem !important;
  padding: 0.6rem 1rem !important;
}
[data-testid="stExpander"] [data-testid="stMarkdownContainer"] {
  padding: 0 0.4rem;
}

hr, [data-testid="stDivider"] {
  border: 0 !important;
  border-top: 1px solid var(--rule) !important;
  margin: 1.6rem 0 !important;
}

[data-testid="stRadio"] label,
[data-testid="stRadio"] p {
  font-family: var(--body) !important;
  font-size: 0.82rem !important;
  color: var(--ink) !important;
}

div[data-baseweb="select"] > div,
div[data-baseweb="select"] {
  background: var(--bg) !important;
  border: 1px solid var(--rule) !important;
  border-radius: 0 !important;
  font-family: var(--body) !important;
  color: var(--ink) !important;
}

[data-testid="stProgress"] > div > div {
  background: var(--rule) !important;
  border-radius: 0 !important;
}
[data-testid="stProgress"] > div > div > div {
  background: var(--ink) !important;
  border-radius: 0 !important;
}

/* Tone-reset Streamlit's status alerts to editorial */
.stAlert {
  border-radius: 0 !important;
  border-left: 3px solid var(--rust) !important;
  background: var(--bg-elev) !important;
  color: var(--ink) !important;
}

/* Buttons */
button[kind="secondary"], .stButton > button {
  font-family: var(--body) !important;
  border-radius: 0 !important;
  border: 1px solid var(--ink) !important;
  background: transparent !important;
  color: var(--ink) !important;
  text-transform: uppercase;
  letter-spacing: 0.1em;
  font-size: 0.78rem !important;
  font-weight: 600 !important;
}
</style>
        """,
        unsafe_allow_html=True,
    )

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
            color = NO_DATA
            z = -1.0  # reserved zone in the discrete colorscale
        elif metric == "phase":
            color = PHASE_HEX.get(p.long_term_phase, INK_MUTED)
            z = float(p.long_term_phase)
        elif metric == "stage":
            color = PHASE_HEX.get(p.short_term_stage, INK_MUTED)
            z = float(p.short_term_stage)
        elif metric == "caution":
            color = CAUTION_HEX.get(p.caution_level, INK_MUTED)
            z = {"low": 1, "moderate": 2, "elevated": 3, "high": 4}.get(p.caution_level, 0)
        elif metric == "total_debt":
            d = p.total_debt_pct_gdp
            color = NO_DATA if d is None else (
                "#506e58" if d < 100
                else "#708060" if d < 200
                else "#b8893a" if d < 250
                else "#a14a3a" if d < 300
                else "#6b3c4a"
            )
            z = d if d is not None else 0
        else:
            color = INK_MUTED
            z = 0

        for iso3 in expand_iso3_for_map(p.iso3):
            locations.append(iso3)
            z_values.append(z)
            colors.append(color)
            hover_texts.append(p.hover_text)

    fig = go.Figure(go.Choropleth(
        locations=locations,
        z=z_values,
        text=hover_texts,
        hoverinfo="text",
        locationmode="ISO-3",
        marker_line_color=PAPER,
        marker_line_width=0.6,
        colorscale=[[0, INK_MUTED], [1, INK_MUTED]],  # base (overridden below)
        showscale=False,
    ))
    if metric in ("phase", "stage"):
        # Discrete categorical scale. z=-1 reserved for "no data" — visibly
        # distinct from z=0 "transition" (countries with data but on a boundary).
        fig.update_traces(
            colorscale=[
                [0.000, NO_DATA],     # -1: no data
                [0.125, PHASE_HEX[0]],# 0: transition (amber)
                [0.250, PHASE_HEX[1]],# 1: sound / expansion (sage)
                [0.375, PHASE_HEX[2]],# 2: outpacing / inflationary (olive)
                [0.500, PHASE_HEX[3]],# 3: bubble / recession (ochre)
                [0.625, PHASE_HEX[4]],# 4: top / reflation (terracotta)
                [0.750, PHASE_HEX[5]],# 5: deleveraging (oxblood)
                [0.875, PHASE_HEX[6]],# 6: repression (slate)
                [1.000, PHASE_HEX[7]],# 7: reset (charcoal)
            ],
            zmin=-1, zmax=7,
        )
    elif metric == "caution":
        fig.update_traces(
            colorscale=[
                [0.0, NO_DATA],
                [0.25, CAUTION_HEX["low"]],
                [0.5, CAUTION_HEX["moderate"]],
                [0.75, CAUTION_HEX["elevated"]],
                [1.0, CAUTION_HEX["high"]],
            ],
            zmin=0, zmax=4,
        )
    else:  # total_debt continuous — sage to oxblood, painterly
        fig.update_traces(
            colorscale=[
                [0.0, "#506e58"],
                [0.35, "#708060"],
                [0.55, "#b8893a"],
                [0.78, "#a14a3a"],
                [1.0, "#6b3c4a"],
            ],
            zmin=50, zmax=400,
        )

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type="natural earth",
            showland=True,
            landcolor=LAND,
            oceancolor=PAPER,
            showocean=True,
            showcountries=True,
            countrycolor=RULE,
            countrywidth=0.4,
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family="Inter Tight, system-ui, sans-serif", color=INK, size=12),
        hoverlabel=dict(
            bgcolor=PAPER_ELEV,
            bordercolor=INK,
            font=dict(family="Inter Tight, system-ui, sans-serif", color=INK, size=12),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=460,
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

    # Editorial legend below map — swatches with text, hairline-separated
    if metric_key == "phase":
        legend_items = [
            (PHASE_HEX[1], "Sound money"),
            (PHASE_HEX[2], "Outpacing"),
            (PHASE_HEX[3], "Bubble"),
            (PHASE_HEX[4], "Top (peak burden)"),
            (PHASE_HEX[5], "Deleveraging (distress)"),
            (PHASE_HEX[6], "Reflation / repression"),
            (PHASE_HEX[0], "Transition"),
            (NO_DATA, "No data"),
        ]
    elif metric_key == "stage":
        legend_items = [
            (PHASE_HEX[1], "Expansion"),
            (PHASE_HEX[2], "Inflationary peak"),
            (PHASE_HEX[4], "Recession"),
            (PHASE_HEX[6], "Reflation"),
            (PHASE_HEX[0], "Transition"),
            (NO_DATA, "No data"),
        ]
    elif metric_key == "caution":
        legend_items = [
            (CAUTION_HEX["low"], "Low"),
            (CAUTION_HEX["moderate"], "Moderate"),
            (CAUTION_HEX["elevated"], "Elevated"),
            (CAUTION_HEX["high"], "High"),
        ]
    else:
        legend_items = [
            ("#506e58", "< 100%"),
            ("#708060", "100–200%"),
            ("#b8893a", "200–250%"),
            ("#a14a3a", "250–300%"),
            ("#6b3c4a", "≥ 300%"),
        ]
    swatches = "".join(
        f'<span><i class="swatch" style="background:{c}"></i>{label}</span>'
        for c, label in legend_items
    )
    st.markdown(f'<div class="legend-row">{swatches}</div>', unsafe_allow_html=True)

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
    """Pull-quote-styled one-line plain read of the regime."""
    f_lt = view.long_term.features
    debt = f"{f_lt.total_credit_pct_gdp:.0f}%" if f_lt.total_credit_pct_gdp else "n/a"
    cpi = f"{f_lt.cpi_yoy:.1f}%" if f_lt.cpi_yoy else "n/a"
    rr = f"{f_lt.real_rate_10y:+.1f}%" if f_lt.real_rate_10y is not None else "n/a"
    dsr = f"{f_lt.debt_service_ratio:.1f}%" if f_lt.debt_service_ratio else "n/a"

    st.markdown(
        f"""
<div class="pullquote">
Total non-financial debt <span class="num">{debt}</span> of GDP.
CPI year-on-year <span class="num">{cpi}</span>.
Real 10-year rate <span class="num">{rr}</span>.
Private-sector debt-service ratio <span class="num">{dsr}</span>.
</div>
        """,
        unsafe_allow_html=True,
    )

    # Cross-regime real-yield note — surfaces only when the multiplier is firing.
    regime = view.allocation.real_yield_regime
    if regime == "repression":
        st.markdown(
            f"<p class='section-lede' style='margin-top:-0.6rem'>"
            f"Real rate <span class='num' style='color:#0c1f3f;font-family:JetBrains Mono,monospace'>{rr}</span> "
            f"is biasing every phase's tilts toward gold, commodities, and real estate, "
            f"away from long nominal bonds — financial repression regime.</p>",
            unsafe_allow_html=True,
        )
    elif regime == "mild repression":
        st.markdown(
            "<p class='section-lede' style='margin-top:-0.6rem'>"
            "Real rate is mildly negative — tilts have a small extra lean to "
            "gold and away from long nominal bonds, on top of the phase-specific tilts.</p>",
            unsafe_allow_html=True,
        )


def _confidence_block(label: str, confidence: float) -> str:
    pct = max(0.0, min(confidence, 1.0)) * 100
    return (
        f'<div class="confidence-track"><span class="confidence-fill" '
        f'style="width:{pct:.1f}%"></span></div>'
        f'<div class="confidence-meta"><span>{label}</span>'
        f'<span>{pct:.0f}%</span></div>'
    )


def _phase_swatch(color: str) -> str:
    return (
        f'<i class="swatch" style="background:{color};display:inline-block;'
        f'width:0.6rem;height:0.6rem;margin-right:0.45rem;'
        f'vertical-align:0.05rem;border:1px solid rgba(0,0,0,0.15);"></i>'
    )


def _render_summary_cards(view: CountryView) -> None:
    """Three editorial frames: long-term phase, short-term stage, allocation."""
    cols = st.columns(3, gap="large")

    # Card 1 — Long-term phase
    with cols[0]:
        phase = view.long_term.phase
        swatch = _phase_swatch(PHASE_HEX.get(phase, INK_MUTED))
        st.markdown(
            f"""
<div class="card">
  <div class="eyebrow">Long-term cycle <span class="accent">·</span> Phase {phase} of 7</div>
  <div class="title">{swatch}{view.long_term.phase_label}</div>
  <div class="deck">{PHASE_EXPLANATIONS.get(phase, "")}</div>
  {_confidence_block("Classifier confidence", view.long_term.confidence)}
</div>
            """,
            unsafe_allow_html=True,
        )

    # Card 2 — Short-term stage
    with cols[1]:
        stage = view.short_term.stage
        swatch = _phase_swatch(PHASE_HEX.get(stage, INK_MUTED))
        st.markdown(
            f"""
<div class="card">
  <div class="eyebrow">Short-term cycle <span class="accent">·</span> Stage {stage} of 4</div>
  <div class="title">{swatch}{view.short_term.stage_label}</div>
  <div class="deck">{STAGE_EXPLANATIONS.get(stage, "")}</div>
  {_confidence_block("Classifier confidence", view.short_term.confidence)}
</div>
            """,
            unsafe_allow_html=True,
        )

    # Card 3 — Allocation tilts
    with cols[2]:
        caution = view.allocation.caution_level
        swatch = _phase_swatch(CAUTION_HEX.get(caution, INK_MUTED))
        rows = "".join(
            f'<div class="tilt-row">'
            f'<span class="tilt-arrow">{t.direction}</span>'
            f'<span class="tilt-label">{t.label.split(" (")[0]}</span>'
            f'<span class="tilt-num">{t.tilt:+.2f}</span>'
            f'</div>'
            for t in top_tilts(view.allocation, n=3)
        )
        st.markdown(
            f"""
<div class="card">
  <div class="eyebrow">Positioning <span class="accent">·</span> Caution {caution}</div>
  <div class="title">{swatch}Top tilts</div>
  <div class="deck">Three largest-magnitude deviations from a default diversified base — not buy/sell signals.</div>
  {rows}
</div>
            """,
            unsafe_allow_html=True,
        )


# ─── Framework note ─────────────────────────────────────────────────────────


def _render_framework_note() -> None:
    with st.expander("How these pieces fit together (read once)"):
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
    _inject_design_css()

    today = date.today().strftime("%a %d %b %Y").upper()
    st.markdown(
        f"""
<div class="masthead">
  <div class="meta">
    <span>Vol. I &nbsp;·&nbsp; {today}</span>
    <span>A regime lens, not an oracle</span>
  </div>
  <h1 class="title">Dalio Machine</h1>
  <p class="subtitle">
    A macro-cycle research lens for the late-cycle world — short-term &amp;
    long-term debt cycles, mapped to allocation tilts. Decision-support for
    a diversified base portfolio, not market-timing signals.
  </p>
</div>
        """,
        unsafe_allow_html=True,
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
    st.markdown(
        '<span class="kicker">The lay of the land</span>'
        '<h2 class="section-title">Global cycle overview</h2>'
        '<p class="section-lede">Click any country to switch the brief below. '
        'The Eurozone shows as a single bloc; click any member to select it.</p>',
        unsafe_allow_html=True,
    )
    clicked_iso2 = _render_world_map(points, selected_iso2=selected)
    if clicked_iso2 and clicked_iso2 != selected:
        st.session_state.country = clicked_iso2
        st.rerun()

    # ─── Country detail ───────────────────────────────────────────
    if selected not in countries_with_data:
        st.warning(
            f"No data for **{country.name}** yet. "
            "Slice 3 (Tier-2 fan-out: IN, BR) is the next step."
        )
        return

    with _open_session() as s:
        view = compute_country_view(s, selected)

    st.markdown(
        f'<span class="kicker">Country brief</span>'
        f'<h2 class="section-title">{country.name}</h2>'
        f'<p class="section-lede">Two cycles, mapped to a positioning view. '
        f'The numbers below are inputs to a decision, not the decision itself.</p>',
        unsafe_allow_html=True,
    )
    _render_quick_read(view)
    _render_summary_cards(view)
    st.markdown("<div style='height:1.4rem'></div>", unsafe_allow_html=True)
    _render_framework_note()

    st.markdown(
        '<span class="kicker">Underlying readings</span>'
        '<h2 class="section-title">Indicators &amp; reasoning</h2>'
        '<p class="section-lede">Open any panel to see the indicators feeding '
        'each classification, the rule weights, and full historical series.</p>',
        unsafe_allow_html=True,
    )

    # ─── Detail expanders ─────────────────────────────────────────
    with st.expander("Short-term cycle indicators"):
        _render_stage_card(view.short_term)
        st.markdown("##### Indicators")
        with _open_session() as s:
            _render_indicator_grid(s, selected, view.short_term)

    with st.expander("Long-term debt cycle indicators (BIS data)"):
        _render_long_term_card(view.long_term)
        st.markdown("##### Sector debt + DSR")
        with _open_session() as s:
            _render_long_term_indicators(s, selected, view.long_term)

    with st.expander("All asset-allocation tilts (full table)"):
        _render_allocation(view.allocation)

    with st.expander("History explorer (any indicator, full series)"), _open_session() as s:
        _render_history_explorer(s, selected)


if __name__ == "__main__":
    main()
