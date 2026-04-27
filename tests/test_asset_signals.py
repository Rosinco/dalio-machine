"""Tests for asset-price signals (Slice 16)."""
from datetime import date, timedelta

import numpy as np
import pytest

from dalio.scoring.asset_signals import (
    MIN_OBSERVATIONS,
    AssetSignals,
    compute_asset_signals,
)
from dalio.scoring.long_term import LongTermFeatures, classify_features
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed_hy(session, country: str, values: list[float]) -> None:
    """Seed `len(values)` daily observations of hy_spread ending today."""
    today = date.today()
    base = today - timedelta(days=len(values) - 1)
    for i, v in enumerate(values):
        session.add(Observation(
            country=country, indicator="hy_spread",
            date=base + timedelta(days=i),
            value=float(v), source="TEST", series_id="X",
        ))


# ─── compute_asset_signals ──────────────────────────────────────────────────


def test_asset_signals_us_no_data_returns_empty(session_factory):
    with session_factory() as s:
        sigs = compute_asset_signals(s, "US")
    assert sigs == AssetSignals(country="US")


def test_asset_signals_non_us_always_empty(session_factory):
    """Non-US countries return empty signals — FRED HY series is US-only."""
    with session_factory() as s:
        # Even if we seed data for SE, the function should ignore it
        _seed_hy(s, "SE", [4.0] * MIN_OBSERVATIONS)
        s.commit()
    with session_factory() as s:
        sigs = compute_asset_signals(s, "SE")
    assert sigs.hy_spread_latest is None
    assert sigs.hy_spread_z is None


def test_asset_signals_below_min_observations_returns_none(session_factory):
    with session_factory() as s:
        _seed_hy(s, "US", [4.0] * (MIN_OBSERVATIONS - 1))
        s.commit()
    with session_factory() as s:
        sigs = compute_asset_signals(s, "US")
    assert sigs.hy_spread_z is None


def test_asset_signals_at_min_observations_computes_z(session_factory):
    rng = np.random.default_rng(42)
    values = list(rng.normal(loc=4.0, scale=1.0, size=MIN_OBSERVATIONS))
    with session_factory() as s:
        _seed_hy(s, "US", values)
        s.commit()
    with session_factory() as s:
        sigs = compute_asset_signals(s, "US")
    assert sigs.hy_spread_latest == pytest.approx(values[-1])
    assert sigs.hy_spread_z is not None
    # Last value drawn from N(4,1) → z roughly within ±3
    assert -3.0 < sigs.hy_spread_z < 3.0


def test_asset_signals_tight_spread_z_negative(session_factory):
    """If the latest value is well below the historical mean, z is negative."""
    history = [5.0] * (MIN_OBSERVATIONS - 1) + [2.0]  # latest is 3pp below mean
    with session_factory() as s:
        _seed_hy(s, "US", history)
        s.commit()
    with session_factory() as s:
        sigs = compute_asset_signals(s, "US")
    assert sigs.hy_spread_z is not None
    assert sigs.hy_spread_z < -1.0  # very negative


# ─── Classifier integration ─────────────────────────────────────────────────


def test_classifier_hy_z_below_minus15_adds_bubble_vote():
    """HY spread z < -1.5 (unusually tight = bubble complacency) bumps Phase 3."""
    f = LongTermFeatures(
        country="US",
        total_credit_pct_gdp=180.0,  # mid-cycle, no other Phase 3 trigger
        hy_spread_z=-2.0,
    )
    c = classify_features(f)
    phase3_votes = [v for v in c.votes if v.phase == 3]
    assert any("HY credit spread" in v.reason for v in phase3_votes)


def test_classifier_hy_z_above_2_adds_top_vote():
    """HY spread z > 2 (unusually wide = distress repricing) bumps Phase 4."""
    f = LongTermFeatures(
        country="US",
        total_credit_pct_gdp=200.0,
        hy_spread_z=2.5,
    )
    c = classify_features(f)
    phase4_votes = [v for v in c.votes if v.phase == 4]
    assert any("HY credit spread" in v.reason for v in phase4_votes)


def test_classifier_hy_z_neutral_does_not_add_votes():
    """HY spread z near 0 doesn't trigger any asset-price votes."""
    f = LongTermFeatures(
        country="US",
        total_credit_pct_gdp=180.0,
        hy_spread_z=+0.2,
    )
    c = classify_features(f)
    assert not any("HY credit spread" in v.reason for v in c.votes)


def test_classifier_hy_z_none_does_not_add_votes():
    """When hy_spread_z is None (non-US, sparse data), no asset-price votes."""
    f = LongTermFeatures(
        country="SE",
        total_credit_pct_gdp=180.0,
        hy_spread_z=None,
    )
    c = classify_features(f)
    assert not any("HY credit spread" in v.reason for v in c.votes)
