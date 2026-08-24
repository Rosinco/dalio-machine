"""Editorial design tokens shared by every page and chart.

Single source for the cream/navy palette, the two validated sequential ramps
used by the World Fundamentals Map, small HTML helpers, the plotly base
layout, and the player centroids used for marker overlays.

Palette rule (dataviz validation on #faf6ef): the muted phase family cannot
carry categorical identity, so fundamentals categories are never encoded by
hue — only "how good" (navy ramp) and "how exposed" (rust ramp) get colour.
"""
from __future__ import annotations

from typing import Final

# ─── Page palette ─────────────────────────────────────────────────────────────

INK: Final = "#0c1f3f"
INK_MUTED: Final = "#5a5852"
PAPER: Final = "#faf6ef"
PAPER_ELEV: Final = "#f1ebde"
LAND: Final = "#e8e0cb"
RULE: Final = "#cdc6b6"
RUST: Final = "#b94e23"
NO_DATA: Final = "#d9d4c5"

# Editorial phase palette — muted, painterly. Each phase reads as a sentiment.
# Sage / olive = calm. Ochre / terracotta = warning. Oxblood / slate = distress / regime change.
PHASE_HEX: Final[dict[int, str]] = {
    1: "#506e58",   # sage — sound money
    2: "#708060",   # olive — debt outpacing
    3: "#b8893a",   # ochre — bubble
    4: "#a14a3a",   # terracotta — top
    5: "#6b3c4a",   # oxblood — deleveraging
    6: "#3a587a",   # slate blue — reflation/repression
    7: "#2a2a2a",   # charcoal — reset
    0: "#c69e3f",   # warm amber — transition (look closer)
}

CAUTION_HEX: Final[dict[str, str]] = {
    "low": "#506e58",
    "moderate": "#c69e3f",
    "elevated": "#b8893a",
    "high": "#a14a3a",
}

# ─── Fundamentals ramps (validated ordinal on cream) ─────────────────────────

# Quintile bins, darker = better. Bottom bin is a cool blue-grey so it separates
# from the warm LAND / NO_DATA tones by hue as well as lightness.
PCT_RAMP: Final[tuple[str, ...]] = ("#98a7bf", "#7389a7", "#4f688c", "#2f4a70", "#0c1f3f")
PCT_BIN_LABELS: Final[tuple[str, ...]] = ("Bottom 20 %", "20–40 %", "40–60 %", "60–80 %", "Top 20 %")

# Exposure overlay: 1 · 2 · 3 · 4+ pressure chains naming the country.
EXPOSURE_RAMP: Final[tuple[str, ...]] = ("#cf9474", "#c4754c", "#b94e23", "#8a3412")

FONT_BODY: Final = "Inter Tight, system-ui, sans-serif"
FONT_DISPLAY: Final = "Source Serif 4, serif"
FONT_MONO: Final = "JetBrains Mono, ui-monospace, monospace"

# Approximate label anchors (lat, lon) for marker overlays on the choropleth.
PLAYER_CENTROIDS: Final[dict[str, tuple[float, float]]] = {
    "US": (39.8, -98.6), "CN": (35.9, 104.2), "EU": (50.1, 9.0), "UK": (54.0, -2.5),
    "JP": (36.2, 138.3), "SE": (62.0, 15.0), "IN": (22.6, 79.0), "BR": (-10.8, -52.9),
    "DE": (51.2, 10.4), "FR": (46.6, 2.5), "IT": (42.8, 12.6), "ES": (40.3, -3.7),
    "NL": (52.2, 5.3), "CA": (56.0, -106.0), "RU": (61.5, 90.0), "KR": (36.5, 127.8),
    "AU": (-25.3, 133.8), "MX": (23.6, -102.5), "ID": (-2.5, 118.0), "SA": (24.0, 45.0),
    "TR": (39.0, 35.2), "CH": (46.8, 8.2),
}


# ─── HTML helpers ─────────────────────────────────────────────────────────────


def confidence_block(label: str, confidence: float) -> str:
    """Hairline confidence bar + meta line (used in cards and country headers)."""
    pct = max(0.0, min(confidence, 1.0)) * 100
    return (
        f'<div class="confidence-track"><span class="confidence-fill" '
        f'style="width:{pct:.1f}%"></span></div>'
        f'<div class="confidence-meta"><span>{label}</span>'
        f'<span>{pct:.0f}%</span></div>'
    )


def phase_swatch(color: str) -> str:
    return (
        f'<i class="swatch" style="background:{color};display:inline-block;'
        f'width:0.6rem;height:0.6rem;margin-right:0.45rem;'
        f'vertical-align:0.05rem;border:1px solid rgba(0,0,0,0.15);"></i>'
    )


def legend_row(items: list[tuple[str, str]]) -> str:
    """Editorial swatch legend: ``[(hex, label), ...]`` → one ``.legend-row`` div."""
    swatches = "".join(
        f'<span><i class="swatch" style="background:{c}"></i>{label}</span>'
        for c, label in items
    )
    return f'<div class="legend-row">{swatches}</div>'


# ─── Plotly ───────────────────────────────────────────────────────────────────


def plotly_base_layout(height: int) -> dict:
    """The font / hover / transparent-background block every chart repeats."""
    return dict(
        height=height,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=PAPER,
        font=dict(family=FONT_BODY, color=INK, size=12),
        hoverlabel=dict(
            bgcolor=PAPER_ELEV,
            bordercolor=INK,
            font=dict(family=FONT_BODY, color=INK, size=12),
        ),
    )


def geo_layout() -> dict:
    """Natural-earth globe styled to the page: cream ocean, parchment land."""
    return dict(
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
    )
