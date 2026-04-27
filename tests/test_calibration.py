"""Tests for per-country threshold calibration (Slice 11)."""
from datetime import date, timedelta

import numpy as np
import pytest

from dalio.scoring.calibration import (
    MIN_OBSERVATIONS,
    compute_country_quantiles,
    compute_country_thresholds,
    threshold_deltas,
)
from dalio.scoring.long_term import LongTermFeatures, classify_features
from dalio.scoring.thresholds import DEFAULT_THRESHOLDS, Thresholds
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed(session, country: str, indicator: str, values: list[float]) -> None:
    """Seed a country's indicator with `len(values)` quarterly observations
    starting from 2010-01-01."""
    base = date(2010, 1, 1)
    for i, v in enumerate(values):
        d = base + timedelta(days=int(i * 91.25))
        session.add(Observation(
            country=country, indicator=indicator, date=d,
            value=float(v), source="TEST", series_id="X",
        ))


# ─── compute_country_quantiles ──────────────────────────────────────────────

def test_quantiles_returns_none_below_min(session_factory):
    with session_factory() as s:
        _seed(s, "US", "debt_service_ratio", [10.0] * (MIN_OBSERVATIONS - 1))
        s.commit()
    with session_factory() as s:
        assert compute_country_quantiles(s, "US", "debt_service_ratio") is None


def test_quantiles_returns_dict_at_min(session_factory):
    """At MIN_OBSERVATIONS exactly, quantiles are computed."""
    values = list(range(1, MIN_OBSERVATIONS + 1))  # 1..40
    with session_factory() as s:
        _seed(s, "US", "debt_service_ratio", values)
        s.commit()
    with session_factory() as s:
        q = compute_country_quantiles(s, "US", "debt_service_ratio")
    assert q is not None
    # Sanity-check: q50 ≈ median of 1..40 (=20.5)
    assert q["q50"] == pytest.approx(20.5, abs=0.5)
    assert q["q75"] == pytest.approx(30.25, abs=0.5)
    assert q["q90"] == pytest.approx(36.1, abs=0.5)
    assert q["q95"] == pytest.approx(38.05, abs=0.5)


def test_quantiles_unknown_indicator_returns_none(session_factory):
    with session_factory() as s:
        _seed(s, "US", "debt_service_ratio", [1.0] * MIN_OBSERVATIONS)
        s.commit()
    with session_factory() as s:
        assert compute_country_quantiles(s, "US", "nonexistent_indicator") is None


# ─── compute_country_thresholds ─────────────────────────────────────────────

def test_thresholds_falls_back_to_defaults_with_no_data(session_factory):
    with session_factory() as s:
        t = compute_country_thresholds(s, "US")
    assert t == DEFAULT_THRESHOLDS


def test_thresholds_partial_data_partial_fallback(session_factory):
    """Country with only DSR history gets DSR-derived thresholds but
    debt and CPI thresholds stay at default."""
    rng = np.random.default_rng(42)
    dsr_values = list(rng.normal(loc=15, scale=4, size=MIN_OBSERVATIONS))
    with session_factory() as s:
        _seed(s, "SE", "debt_service_ratio", dsr_values)
        s.commit()
    with session_factory() as s:
        t = compute_country_thresholds(s, "SE")
    # DSR fields are derived from the data — should differ from defaults
    assert t.dsr_stretched != DEFAULT_THRESHOLDS.dsr_stretched
    # Debt and CPI fields stay at defaults
    assert t.debt_extreme == DEFAULT_THRESHOLDS.debt_extreme
    assert t.cpi_peak == DEFAULT_THRESHOLDS.cpi_peak


def test_thresholds_cpi_floor_applies(session_factory):
    """Japan-style chronically-low inflation should not produce a sub-2.5%
    'elevated' threshold. The floor enforces sanity."""
    low_cpi = [0.5] * MIN_OBSERVATIONS  # all 0.5%
    with session_factory() as s:
        _seed(s, "JP", "cpi_yoy", low_cpi)
        s.commit()
    with session_factory() as s:
        t = compute_country_thresholds(s, "JP")
    assert t.cpi_elevated == 2.5  # floor
    assert t.cpi_peak == 3.5      # floor


def test_thresholds_se_dsr_higher_than_us_default(session_factory):
    """Sweden's historical DSR distribution sits above the US-derived 18%
    distress floor. Per-country calibration should reflect that."""
    se_dsr = [20.0, 21.0, 22.0, 23.0] * (MIN_OBSERVATIONS // 4)
    with session_factory() as s:
        _seed(s, "SE", "debt_service_ratio", se_dsr)
        s.commit()
    with session_factory() as s:
        t = compute_country_thresholds(s, "SE")
    # SE's q90 should be above 22 → distress threshold rises with the data
    assert t.dsr_distress > DEFAULT_THRESHOLDS.dsr_distress


# ─── threshold_deltas ───────────────────────────────────────────────────────

def test_threshold_deltas_zero_for_default_country(session_factory):
    with session_factory() as s:
        t = compute_country_thresholds(s, "US")
    deltas = threshold_deltas(t)
    for _name, (default, country, delta) in deltas.items():
        assert default == country
        assert delta == 0.0


# ─── End-to-end: classifier uses the threshold ─────────────────────────────

def test_classifier_uses_provided_thresholds():
    """If we pass a custom Thresholds with much higher debt_extreme, a
    country with debt 280% no longer triggers Phase 4."""
    f = LongTermFeatures(
        country="US",
        total_credit_pct_gdp=280.0,
        total_credit_5y_ago=275.0,
        debt_service_ratio=15.0,  # below stretched
        yield_10y=4.0,
        cpi_yoy=2.0,
    )
    # Default thresholds → Phase 4 (debt_extreme=280)
    c_default = classify_features(f)
    assert c_default.phase == 4 or any(v.phase == 4 for v in c_default.votes)

    # Higher per-country thresholds → no Phase 4 vote (Japan-like)
    higher = Thresholds(debt_extreme=350.0, debt_late_cycle_low=280.0)
    c_higher = classify_features(f, thresholds=higher)
    # Phase 4 should NOT be the top phase here (debt now in late-cycle, not extreme)
    if c_higher.votes:
        phase_4_votes = [v for v in c_higher.votes if v.phase == 4]
        # No Phase 4 votes since debt is below the higher debt_extreme
        assert all(v.weight == 0 for v in phase_4_votes) or len(phase_4_votes) == 0


def test_classifier_back_compat_when_thresholds_none():
    """classify_features(features) with no thresholds arg uses defaults —
    must produce the same result as classify_features(features, DEFAULT_THRESHOLDS)."""
    f = LongTermFeatures(
        country="US",
        total_credit_pct_gdp=290.0,
        total_credit_5y_ago=270.0,
        debt_service_ratio=20.0,
        yield_10y=4.0,
        cpi_yoy=2.0,
    )
    c_implicit = classify_features(f)
    c_explicit = classify_features(f, thresholds=DEFAULT_THRESHOLDS)
    assert c_implicit.phase == c_explicit.phase
    assert c_implicit.confidence == c_explicit.confidence
