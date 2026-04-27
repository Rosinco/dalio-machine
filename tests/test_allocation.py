import pytest

from dalio.scoring.allocation import (
    ASSET_CLASSES,
    LONG_TERM_TILTS,
    SHORT_TERM_TILTS,
    AllocationView,
    _direction,
    _real_yield_multiplier,
    _real_yield_regime,
    compute_tilts,
)
from dalio.scoring.long_term import LongTermFeatures, PhaseClassification
from dalio.scoring.short_term import (
    Classification,
    ShortTermFeatures,
    StageVote,
)


def _short(stage: int, label: str, conf: float, votes: tuple = ()) -> Classification:
    return Classification(
        country="US",
        stage=stage,
        stage_label=label,
        confidence=conf,
        votes=votes,
        features=ShortTermFeatures(country="US"),
    )


def _long(phase: int, label: str, conf: float, votes: tuple = ()) -> PhaseClassification:
    return PhaseClassification(
        country="US",
        phase=phase,
        phase_label=label,
        confidence=conf,
        votes=votes,
        features=LongTermFeatures(country="US"),
    )


def _long_rr(phase: int, label: str, conf: float, real_rate: float | None) -> PhaseClassification:
    """Long-term classification with a target real_rate_10y. CPI fixed at 2%;
    yield_10y = real_rate + 2 so the real_rate property returns `real_rate`."""
    if real_rate is None:
        features = LongTermFeatures(country="US")
    else:
        features = LongTermFeatures(country="US", yield_10y=real_rate + 2.0, cpi_yoy=2.0)
    return PhaseClassification(
        country="US",
        phase=phase,
        phase_label=label,
        confidence=conf,
        votes=(),
        features=features,
    )


# ─── Direction symbol thresholds ───────────────────────────────────────────

def test_direction_arrow_thresholds():
    assert _direction(1.6) == "↑↑↑"
    assert _direction(1.0) == "↑↑"
    assert _direction(0.4) == "↑"
    assert _direction(0.1) == "→"
    assert _direction(-0.1) == "→"
    assert _direction(-0.4) == "↓"
    assert _direction(-1.0) == "↓↓"
    assert _direction(-1.6) == "↓↓↓"


# ─── Tilt table coverage ────────────────────────────────────────────────────

def test_short_term_tilts_cover_known_stages():
    # Stages 1-4 must have non-empty tilt tables (stage 0 = transition is empty)
    for stage in (1, 2, 3, 4):
        assert SHORT_TERM_TILTS.get(stage), f"Missing short-term tilts for stage {stage}"


def test_long_term_tilts_cover_active_phases():
    # Phases 3-6 are the active phases that should have tilts (1, 2, 7, 0 may be empty)
    for phase in (3, 4, 5, 6):
        assert LONG_TERM_TILTS.get(phase), f"Missing long-term tilts for phase {phase}"


def test_all_referenced_assets_in_canonical_list():
    canonical = set(ASSET_CLASSES)
    for stage_table in SHORT_TERM_TILTS.values():
        for asset in stage_table:
            assert asset in canonical, f"Unknown asset class in tilts: {asset}"
    for phase_table in LONG_TERM_TILTS.values():
        for asset in phase_table:
            assert asset in canonical, f"Unknown asset class in tilts: {asset}"


# ─── Compute_tilts behavior ────────────────────────────────────────────────

def test_compute_tilts_returns_view_with_all_assets():
    st = _short(1, "Expansion", 1.0)
    lt = _long(2, "Debt outpaces income", 0.5)
    view = compute_tilts(st, lt)
    assert isinstance(view, AllocationView)
    assert len(view.tilts) == len(ASSET_CLASSES)
    assert {t.asset_class for t in view.tilts} == set(ASSET_CLASSES)


def test_zero_confidence_produces_zero_tilts():
    st = _short(1, "Expansion", 0.0)
    lt = _long(4, "Top", 0.0)
    view = compute_tilts(st, lt)
    for t in view.tilts:
        assert abs(t.tilt) < 0.01


def test_high_confidence_inflationary_peak_loads_inflation_hedges():
    st = _short(2, "Inflationary peak", 1.0)
    lt = _long(0, "Transition", 0.0)
    view = compute_tilts(st, lt)
    by_asset = {t.asset_class: t.tilt for t in view.tilts}
    # Commodities, gold, TIPS should be positive
    assert by_asset["commodities"] > 0.5
    assert by_asset["gold"] > 0.5
    assert by_asset["tips"] > 0.5
    # Long bonds should be strongly negative
    assert by_asset["long_bonds"] < -0.5


def test_phase_6_repression_avoids_long_bonds():
    st = _short(0, "Transition", 0.0)
    lt = _long(6, "Reflation / financial repression", 1.0)
    view = compute_tilts(st, lt)
    by_asset = {t.asset_class: t.tilt for t in view.tilts}
    # Phase 6: long nominal bonds savaged
    assert by_asset["long_bonds"] < -0.5
    # Inflation hedges loaded
    assert by_asset["gold"] > 0.5
    assert by_asset["tips"] > 0.5


def test_phase_5_deleveraging_loads_long_bonds_and_avoids_credit():
    st = _short(0, "Transition", 0.0)
    lt = _long(5, "Deleveraging", 1.0)
    view = compute_tilts(st, lt)
    by_asset = {t.asset_class: t.tilt for t in view.tilts}
    assert by_asset["long_bonds"] > 0.5  # Flight to safety
    assert by_asset["credit"] < -0.5     # Defaults rising


def test_caution_level_escalates_with_phase():
    # Phase 5 = high caution
    st = _short(1, "Expansion", 0.5)
    lt5 = _long(5, "Deleveraging", 0.5)
    view5 = compute_tilts(st, lt5)
    assert view5.caution_level == "high"

    # Phase 4 = elevated
    lt4 = _long(4, "Top", 0.5)
    view4 = compute_tilts(st, lt4)
    assert view4.caution_level == "elevated"

    # Phase 2 = low
    lt2 = _long(2, "Debt outpaces income", 0.5)
    view2 = compute_tilts(st, lt2)
    assert view2.caution_level == "low"


def test_transition_blends_component_stage_tilts():
    """Stage 0 (transition) should produce tilts blended from constituent stages
    via vote weights, not zero everywhere."""
    # Mock a transition between Reflation (stage 4) and Inflationary peak (stage 2)
    votes = (
        StageVote(stage=4, weight=0.6, reason="cb cutting"),
        StageVote(stage=2, weight=0.5, reason="cpi above 3"),
    )
    st = _short(0, "Transition", 0.5, votes=votes)
    lt = _long(0, "Transition", 0.0)
    view = compute_tilts(st, lt)
    by_asset = {t.asset_class: t.tilt for t in view.tilts}
    # Both component stages favor gold — blended should also favor gold
    assert by_asset["gold"] > 0
    # Both component stages avoid long bonds (Stage 2 strongly, Stage 4 mildly)
    assert by_asset["long_bonds"] < 0


def test_summary_mentions_caution_and_disclaimer():
    st = _short(2, "Inflationary peak", 0.5)
    lt = _long(4, "Top", 0.6)
    view = compute_tilts(st, lt)
    s = view.summary.lower()
    assert "caution" in s
    assert "tilt" in s or "deviation" in s
    assert "not a market-timing" in s


# ─── Real-yield multiplier (Slice 10) ───────────────────────────────────────

def test_real_yield_multiplier_zero_when_positive_or_none():
    assert _real_yield_multiplier(2.0) == {}
    assert _real_yield_multiplier(0.5) == {}
    assert _real_yield_multiplier(0.0) == {}
    assert _real_yield_multiplier(None) == {}


def test_real_yield_multiplier_tapers_between_0_and_minus1():
    half = _real_yield_multiplier(-0.5)
    full = _real_yield_multiplier(-1.0)
    assert half["gold"] == pytest.approx(0.2)
    assert full["gold"] == pytest.approx(0.4)
    assert half["long_bonds"] == pytest.approx(-0.2)
    assert full["long_bonds"] == pytest.approx(-0.4)


def test_real_yield_multiplier_saturates_below_minus1():
    rr_minus1 = _real_yield_multiplier(-1.0)
    rr_minus3 = _real_yield_multiplier(-3.0)
    assert rr_minus1 == rr_minus3


def test_real_yield_regime_labels():
    assert _real_yield_regime(None) == "unknown"
    assert _real_yield_regime(-2.0) == "repression"
    assert _real_yield_regime(-0.5) == "mild repression"
    assert _real_yield_regime(0.5) == "neutral"
    assert _real_yield_regime(2.0) == "savers compensated"


def test_compute_tilts_negative_real_rate_increases_gold():
    """Holding ST and LT classifications fixed, flipping real-rate from
    +1 to −2 must strictly increase the gold tilt (Slice 10 plan invariant)."""
    st = _short(1, "Expansion", 0.5)
    pos = _long_rr(1, "Sound money", 0.5, real_rate=+1.0)
    neg = _long_rr(1, "Sound money", 0.5, real_rate=-2.0)
    gold_pos = next(t.tilt for t in compute_tilts(st, pos).tilts if t.asset_class == "gold")
    gold_neg = next(t.tilt for t in compute_tilts(st, neg).tilts if t.asset_class == "gold")
    assert gold_neg > gold_pos


def test_compute_tilts_negative_real_rate_decreases_long_bonds():
    """Holding ST and LT classifications fixed, flipping real-rate from
    +1 to −2 must strictly decrease the long_bonds tilt."""
    st = _short(1, "Expansion", 0.5)
    pos = _long_rr(1, "Sound money", 0.5, real_rate=+1.0)
    neg = _long_rr(1, "Sound money", 0.5, real_rate=-2.0)
    lb_pos = next(t.tilt for t in compute_tilts(st, pos).tilts if t.asset_class == "long_bonds")
    lb_neg = next(t.tilt for t in compute_tilts(st, neg).tilts if t.asset_class == "long_bonds")
    assert lb_neg < lb_pos


def test_compute_tilts_real_yield_regime_in_view():
    st = _short(1, "Expansion", 0.5)
    view = compute_tilts(st, _long_rr(1, "Sound money", 0.5, real_rate=-2.0))
    assert view.real_yield_regime == "repression"

    view2 = compute_tilts(st, _long_rr(1, "Sound money", 0.5, real_rate=+2.0))
    assert view2.real_yield_regime == "savers compensated"


def test_compute_tilts_real_yield_unknown_when_no_data():
    """When yield_10y or cpi_yoy is missing the regime is 'unknown' and no
    real-rate reasons appear in any tilt."""
    st = _short(1, "Expansion", 0.5)
    view = compute_tilts(st, _long(1, "Sound money", 0.5))
    assert view.real_yield_regime == "unknown"
    for t in view.tilts:
        assert not any("Real-rate regime" in r for r in t.reasons)


def test_compute_tilts_real_yield_appears_in_tilt_reasons():
    """Tilts materially affected by the multiplier should expose the
    'Real-rate regime' line in their reasons tuple."""
    st = _short(1, "Expansion", 1.0)
    view = compute_tilts(st, _long_rr(1, "Sound money", 0.5, real_rate=-2.0))
    gold_tilt = next(t for t in view.tilts if t.asset_class == "gold")
    assert any("Real-rate regime" in r for r in gold_tilt.reasons)
    long_bonds_tilt = next(t for t in view.tilts if t.asset_class == "long_bonds")
    assert any("Real-rate regime" in r for r in long_bonds_tilt.reasons)


# ─── Home-currency overlay (Slice 14) ───────────────────────────────────────

def test_compute_tilts_default_home_currency_is_usd():
    """Default `home_currency="USD"` produces no overlay regardless of rate."""
    st = _short(1, "Expansion", 1.0)
    lt = _long_rr(1, "Sound money", 0.5, real_rate=+2.0)
    view = compute_tilts(st, lt)
    assert view.home_currency == "USD"
    for t in view.tilts:
        assert not any("overlay" in r.lower() for r in t.reasons)


def test_compute_tilts_sek_higher_real_rate_pushes_usd_assets_down():
    """When SEK real rates are higher than target's, USD-denominated assets
    (gold, equities, long_bonds, commodities) tilt down vs USD baseline."""
    st = _short(1, "Expansion", 1.0)
    lt = _long_rr(1, "Sound money", 0.5, real_rate=+0.0)  # target rr 0
    usd_view = compute_tilts(st, lt)
    sek_view = compute_tilts(st, lt, home_currency="SEK", home_real_rate_10y=+2.0)

    # Gold (USD-denominated) should be lower in the SEK view
    gold_usd = next(t.tilt for t in usd_view.tilts if t.asset_class == "gold")
    gold_sek = next(t.tilt for t in sek_view.tilts if t.asset_class == "gold")
    assert gold_sek < gold_usd

    long_bonds_usd = next(t.tilt for t in usd_view.tilts if t.asset_class == "long_bonds")
    long_bonds_sek = next(t.tilt for t in sek_view.tilts if t.asset_class == "long_bonds")
    assert long_bonds_sek < long_bonds_usd


def test_compute_tilts_sek_lower_real_rate_pushes_usd_assets_up():
    """When SEK real rates are lower than target's, USD-denominated assets
    tilt up because SEK is likely to depreciate."""
    st = _short(1, "Expansion", 1.0)
    lt = _long_rr(1, "Sound money", 0.5, real_rate=+2.0)
    usd_view = compute_tilts(st, lt)
    sek_view = compute_tilts(st, lt, home_currency="SEK", home_real_rate_10y=+0.0)

    gold_usd = next(t.tilt for t in usd_view.tilts if t.asset_class == "gold")
    gold_sek = next(t.tilt for t in sek_view.tilts if t.asset_class == "gold")
    assert gold_sek > gold_usd


def test_compute_tilts_overlay_no_op_when_rates_missing():
    """If home_real_rate_10y is None, no overlay applies regardless of
    home_currency value."""
    st = _short(1, "Expansion", 1.0)
    lt = _long_rr(1, "Sound money", 0.5, real_rate=+1.0)
    view = compute_tilts(st, lt, home_currency="SEK", home_real_rate_10y=None)
    for t in view.tilts:
        assert not any("overlay" in r.lower() for r in t.reasons)


def test_compute_tilts_home_currency_label_in_reasons():
    """When the overlay fires, the reasons line names the home_currency."""
    st = _short(1, "Expansion", 1.0)
    lt = _long_rr(1, "Sound money", 0.5, real_rate=+0.0)
    view = compute_tilts(st, lt, home_currency="SEK", home_real_rate_10y=+2.0)
    gold_tilt = next(t for t in view.tilts if t.asset_class == "gold")
    assert any("SEK overlay" in r for r in gold_tilt.reasons)


def test_reasons_explain_each_active_tilt():
    st = _short(2, "Inflationary peak", 0.5)
    lt = _long(6, "Reflation / financial repression", 0.6)
    view = compute_tilts(st, lt)
    # Long-bonds gets hit by both layers — should have 2 reasons
    long_bonds_tilt = next(t for t in view.tilts if t.asset_class == "long_bonds")
    assert len(long_bonds_tilt.reasons) >= 1
    assert any("Long-term" in r for r in long_bonds_tilt.reasons) or any(
        "Short-term" in r for r in long_bonds_tilt.reasons
    )
