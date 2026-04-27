"""Tests for the growth × inflation tilt grid (Slice 15)."""
import pytest

from dalio.scoring.allocation import SHORT_TERM_TILTS
from dalio.scoring.grid import (
    GRID_TILTS,
    STAGE_TO_QUADRANT,
    GridQuadrant,
    quadrant_for_features,
)

# ─── Round-trip preservation ───────────────────────────────────────────────


def test_short_term_tilts_derived_from_grid():
    """Each stage's tilts must equal the corresponding quadrant's tilts
    within 1e-6 (round-trip preservation invariant from the plan)."""
    for stage, quadrant in STAGE_TO_QUADRANT.items():
        stage_tilts = SHORT_TERM_TILTS[stage]
        grid_tilts = GRID_TILTS[quadrant]
        assert stage_tilts.keys() == grid_tilts.keys(), (
            f"Stage {stage} keys mismatch: {set(stage_tilts) ^ set(grid_tilts)}"
        )
        for asset, value in grid_tilts.items():
            assert stage_tilts[asset] == pytest.approx(value, abs=1e-6), (
                f"Stage {stage} {asset}: {stage_tilts[asset]} != {value}"
            )


def test_short_term_tilts_transition_stage_empty():
    """Stage 0 (transition / insufficient data) has no derived tilts."""
    assert SHORT_TERM_TILTS[0] == {}


# ─── quadrant_for_features ─────────────────────────────────────────────────


def test_quadrant_clear_above_trend_low_inflation():
    """Strong growth + low inflation = G↑I↓ (Goldilocks)."""
    q = quadrant_for_features(real_gdp_yoy=3.0, cpi_yoy=1.5)
    assert q == GridQuadrant.GROWTH_UP_INFL_DOWN


def test_quadrant_clear_above_trend_high_inflation():
    """Strong growth + high inflation = G↑I↑ (overheating)."""
    q = quadrant_for_features(real_gdp_yoy=3.0, cpi_yoy=4.0)
    assert q == GridQuadrant.GROWTH_UP_INFL_UP


def test_quadrant_clear_below_trend_low_inflation():
    """Negative growth + low inflation = G↓I↓ (deflationary recession)."""
    q = quadrant_for_features(real_gdp_yoy=-1.0, cpi_yoy=1.0)
    assert q == GridQuadrant.GROWTH_DOWN_INFL_DOWN


def test_quadrant_clear_below_trend_high_inflation():
    """Negative growth + high inflation = G↓I↑ (stagflation)."""
    q = quadrant_for_features(real_gdp_yoy=-1.0, cpi_yoy=5.0)
    assert q == GridQuadrant.GROWTH_DOWN_INFL_UP


def test_quadrant_boundaries_at_default_thresholds():
    """At exactly gdp=2 and cpi=2, both >= conditions fire → G↑I↑."""
    q = quadrant_for_features(real_gdp_yoy=2.0, cpi_yoy=2.0)
    assert q == GridQuadrant.GROWTH_UP_INFL_UP


def test_quadrant_just_below_growth_trend():
    """gdp=1.99, cpi=2.01 → G↓I↑ (stagflation territory by trend default)."""
    q = quadrant_for_features(real_gdp_yoy=1.99, cpi_yoy=2.01)
    assert q == GridQuadrant.GROWTH_DOWN_INFL_UP


def test_quadrant_none_inputs_default_to_above_trend():
    """Missing data → conservative G↑I↑ (worst-case, not G↓I↓)."""
    q = quadrant_for_features(real_gdp_yoy=None, cpi_yoy=None)
    assert q == GridQuadrant.GROWTH_UP_INFL_UP


def test_quadrant_custom_trends():
    """Custom trend args change the boundary."""
    # gdp=4 looks high, but with gdp_trend=5 it's "below trend"
    q = quadrant_for_features(real_gdp_yoy=4.0, cpi_yoy=1.0, gdp_trend=5.0, cpi_trend=2.0)
    assert q == GridQuadrant.GROWTH_DOWN_INFL_DOWN


def test_quadrant_for_features_never_returns_reflation():
    """REFLATION is a stage label, not a grid quadrant — should never be returned."""
    points = [(2.5, 1.0), (3.0, 4.0), (-1.0, 1.0), (-1.0, 5.0), (None, None)]
    for gdp, cpi in points:
        q = quadrant_for_features(real_gdp_yoy=gdp, cpi_yoy=cpi)
        assert q != GridQuadrant.REFLATION


# ─── Stagflation regime (new in Slice 15) ─────────────────────────────────


def test_stagflation_quadrant_has_distinct_tilts():
    """G↓I↑ (stagflation) should have different tilts from any other quadrant —
    it was implicitly conflated with Inflationary peak (G↑I↑) before."""
    stagflation = GRID_TILTS[GridQuadrant.GROWTH_DOWN_INFL_UP]
    for q, tilts in GRID_TILTS.items():
        if q == GridQuadrant.GROWTH_DOWN_INFL_UP:
            continue
        assert stagflation != tilts, f"Stagflation tilts duplicate {q.name}"


def test_stagflation_avoids_long_bonds_more_than_inflationary_peak():
    """Long nominal bonds in stagflation are worse than in plain overheating
    (recession adds insolvency risk on top of inflation eroding nominal value)."""
    stagflation_lb = GRID_TILTS[GridQuadrant.GROWTH_DOWN_INFL_UP]["long_bonds"]
    inflationary_lb = GRID_TILTS[GridQuadrant.GROWTH_UP_INFL_UP]["long_bonds"]
    assert stagflation_lb < inflationary_lb


def test_stagflation_loads_real_assets():
    """Gold + commodities + TIPS should all be positive in stagflation."""
    stagflation = GRID_TILTS[GridQuadrant.GROWTH_DOWN_INFL_UP]
    assert stagflation.get("gold", 0) > 0
    assert stagflation.get("commodities", 0) > 0
    assert stagflation.get("tips", 0) > 0
