"""Bridgewater-style growth × inflation 2×2 tilt grid.

Slice 15: replaces the stage-keyed `SHORT_TERM_TILTS` dict with a 2×2
grid that matches Dalio's actual All Weather construction. The four
quadrants are independent regimes; tilts are hand-tuned per quadrant.

The four canonical quadrants:
  GROWTH_UP_INFL_DOWN  — Goldilocks expansion (Stage 1)
  GROWTH_UP_INFL_UP    — Overheating / Inflationary peak (Stage 2)
  GROWTH_DOWN_INFL_DOWN — Deflationary recession (Stage 3)
  GROWTH_DOWN_INFL_UP   — Stagflation (was conflated with Stage 2 in the old
                           4-stage classifier, but is a distinct regime —
                           recession + inflation = nominal bonds in their
                           worst-possible environment)

Stage 4 Reflation does NOT map cleanly to a single G×I quadrant. By the
classifier's own definition (CB cutting + CPI moderating, growth often
still soft) it spans the G↓I↓ → G↑I↓ transition. Implementation choice
documented in the plan: keep Reflation as a fifth canonical entry
(REFLATION) outside the strict 2×2 — the resolved tilt for Stage 4 is
neither a single quadrant nor a clean blend of two, it's its own thing
(the existing hand-tuned tilts captured this; we preserve them verbatim).

Round-trip preservation is the test invariant: stages 1–3 derive from
the corresponding quadrant within 1e-6; stage 4 derives from REFLATION.

The dashboard's "Growth × Inflation grid" panel shows the country's
current (real_gdp_yoy, cpi_yoy) point on the 2×2 with quadrant labels.
"""
from __future__ import annotations

from enum import Enum


class GridQuadrant(Enum):
    """The 2×2 of growth × inflation, plus a documented hybrid for Reflation."""
    GROWTH_UP_INFL_DOWN = "G↑I↓"
    GROWTH_UP_INFL_UP = "G↑I↑"
    GROWTH_DOWN_INFL_DOWN = "G↓I↓"
    GROWTH_DOWN_INFL_UP = "G↓I↑"
    REFLATION = "G↓→↑ I↓"  # documented hybrid, see module docstring


# Canonical asset tilts per quadrant. Stages 1, 2, 3 values are preserved
# verbatim from the old SHORT_TERM_TILTS dict (round-trip preservation).
# G↓I↑ (stagflation) is new — was implicitly conflated with Inflationary peak.
# REFLATION values are the old Stage 4 tilts (CB pivot context).
GRID_TILTS: dict[GridQuadrant, dict[str, float]] = {
    GridQuadrant.GROWTH_UP_INFL_DOWN: {  # Stage 1 Expansion (Goldilocks)
        "equities": +1.0,
        "credit": +0.5,
        "real_estate": +0.5,
        "long_bonds": -0.3,
        "gold": -0.3,
        "commodities": -0.2,
    },
    GridQuadrant.GROWTH_UP_INFL_UP: {  # Stage 2 Inflationary peak (Overheating)
        "commodities": +1.0,
        "gold": +0.7,
        "tips": +1.0,
        "real_estate": +0.3,
        "long_bonds": -1.2,
        "credit": -0.5,
        "equities": -0.3,
    },
    GridQuadrant.GROWTH_DOWN_INFL_DOWN: {  # Stage 3 Recession (Deflationary)
        "long_bonds": +1.2,
        "short_bonds": +0.5,
        "gold": +0.3,
        "equities": -1.0,
        "credit": -1.0,
        "commodities": -0.8,
        "real_estate": -0.5,
    },
    GridQuadrant.GROWTH_DOWN_INFL_UP: {  # NEW: Stagflation (worst nominal-bond regime)
        "gold": +1.2,
        "commodities": +1.0,
        "tips": +1.0,
        "short_bonds": +0.5,
        "real_estate": +0.3,
        "long_bonds": -1.7,  # the worst possible regime for long nominal bonds
        "credit": -1.0,
        "equities": -0.7,
    },
    GridQuadrant.REFLATION: {  # Stage 4 Reflation (CB pivot, growth recovering)
        "equities": +0.5,
        "gold": +0.7,
        "tips": +0.5,
        "real_estate": +0.3,
        "credit": +0.3,
        "short_bonds": -0.3,
    },
}


def quadrant_for_features(
    real_gdp_yoy: float | None,
    cpi_yoy: float | None,
    gdp_trend: float = 2.0,
    cpi_trend: float = 2.0,
) -> GridQuadrant:
    """Return the 2×2 quadrant for a (gdp, cpi) point.

    Defaults split at gdp=2% (rough developed-economy trend) and cpi=2%
    (most central-bank target). Pass per-country trends from a calibration
    layer for sharper boundaries.

    Does NOT return REFLATION — that's the Stage 4 *classifier* output, not a
    grid quadrant. The grid panel in the dashboard shows where the country
    sits in the 2×2 regardless of whether the classifier has labeled it
    Reflation.

    None inputs default to "above trend" so missing data lands the country
    in the more conservative quadrant rather than silently becoming G↓I↓.
    """
    growth_up = real_gdp_yoy is None or real_gdp_yoy >= gdp_trend
    infl_up = cpi_yoy is None or cpi_yoy >= cpi_trend
    if growth_up:
        return GridQuadrant.GROWTH_UP_INFL_UP if infl_up else GridQuadrant.GROWTH_UP_INFL_DOWN
    return GridQuadrant.GROWTH_DOWN_INFL_UP if infl_up else GridQuadrant.GROWTH_DOWN_INFL_DOWN


# Stage→quadrant mapping for the round-trip preservation test.
STAGE_TO_QUADRANT: dict[int, GridQuadrant] = {
    1: GridQuadrant.GROWTH_UP_INFL_DOWN,
    2: GridQuadrant.GROWTH_UP_INFL_UP,
    3: GridQuadrant.GROWTH_DOWN_INFL_DOWN,
    4: GridQuadrant.REFLATION,
}
