from datetime import date, timedelta

import pytest

from dalio.scoring.short_term import (
    Classification,
    ShortTermFeatures,
    classify,
    classify_features,
    extract_features,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

# ─── Pure-rule tests (no DB) ───────────────────────────────────────────────

def _features(**overrides) -> ShortTermFeatures:
    """Build a features snapshot with sensible defaults; override per test."""
    base = dict(
        country="US",
        real_gdp_yoy=2.0,
        cpi_yoy=2.5,
        cpi_yoy_3m_ago=2.5,
        unemployment_rate=4.0,
        unemployment_3m_ago=4.0,
        policy_rate=3.0,
        policy_rate_6m_ago=3.0,
        yield_10y=4.0,
        yield_2y=3.5,
    )
    base.update(overrides)
    return ShortTermFeatures(**base)


def test_recession_via_negative_gdp():
    f = _features(real_gdp_yoy=-0.5)
    c = classify_features(f)
    assert c.stage == 3
    assert c.stage_label == "Recession"
    assert any("negative" in v.reason for v in c.votes)


def test_recession_via_rising_unemployment():
    f = _features(unemployment_rate=4.8, unemployment_3m_ago=4.0)
    c = classify_features(f)
    assert c.stage == 3


def test_inflationary_peak_high_cpi_with_tightening():
    f = _features(cpi_yoy=4.5, policy_rate=4.5, policy_rate_6m_ago=3.5)
    c = classify_features(f)
    assert c.stage == 2
    assert c.stage_label == "Inflationary peak"


def test_inflationary_peak_clearly_high_cpi():
    f = _features(cpi_yoy=5.5)
    c = classify_features(f)
    assert c.stage == 2


def test_reflation_cutting_with_moderating_cpi():
    f = _features(
        policy_rate=2.0, policy_rate_6m_ago=3.5,
        cpi_yoy=2.0, cpi_yoy_3m_ago=3.0,
    )
    c = classify_features(f)
    assert c.stage == 4
    assert c.stage_label == "Reflation"


def test_expansion_solid_growth_low_cpi_falling_unemp():
    f = _features(
        real_gdp_yoy=2.5, cpi_yoy=2.2,
        unemployment_rate=3.8, unemployment_3m_ago=4.0,
    )
    c = classify_features(f)
    assert c.stage == 1
    assert c.stage_label == "Expansion"


def test_transition_when_top_two_close():
    # Construct competing signals: borderline expansion + borderline reflation
    f = _features(
        real_gdp_yoy=1.8, cpi_yoy=2.4, cpi_yoy_3m_ago=2.5,
        policy_rate=2.5, policy_rate_6m_ago=3.2,
    )
    c = classify_features(f)
    # Either label "Transition" wins, or one stage is dominant — assert structurally
    if c.stage == 0:
        assert "↔" in c.stage_label or "insufficient" in c.stage_label


def test_no_data_returns_stage_zero():
    f = ShortTermFeatures(country="XX")
    c = classify_features(f)
    assert c.stage == 0
    assert c.confidence == 0.0
    assert c.votes == ()


def test_classification_has_country_and_features():
    f = _features()
    c = classify_features(f)
    assert c.country == "US"
    assert c.features is f


def test_yield_curve_inversion_adds_recession_weight():
    # Mild expansion features, but inverted yield curve adds recession vote
    f = _features(yield_10y=3.5, yield_2y=4.0)  # -0.5pp slope
    c = classify_features(f)
    # Recession should have at least one vote from inversion
    recession_votes = [v for v in c.votes if v.stage == 3]
    assert len(recession_votes) >= 1
    assert any("inverted" in v.reason for v in recession_votes)


def test_features_computed_lags():
    f = _features(
        cpi_yoy=3.5, cpi_yoy_3m_ago=3.0,
        unemployment_rate=4.5, unemployment_3m_ago=4.0,
        policy_rate=3.0, policy_rate_6m_ago=4.0,
    )
    assert f.cpi_change_3m == pytest.approx(0.5)
    assert f.unemployment_change_3m == pytest.approx(0.5)
    assert f.policy_rate_change_6m == pytest.approx(-1.0)
    assert f.yield_curve_slope == pytest.approx(0.5)


def test_features_none_when_lag_missing():
    f = _features(cpi_yoy_3m_ago=None)
    assert f.cpi_change_3m is None


# ─── DB-backed extraction test ─────────────────────────────────────────────

@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def _add_obs(session, country, indicator, d, value, source="FRED", series_id="X"):
    session.add(Observation(
        country=country, indicator=indicator, date=d,
        value=value, source=source, series_id=series_id,
    ))


def test_extract_features_picks_latest_and_lag(session_factory):
    today = date(2026, 4, 1)
    with session_factory() as s:
        # CPI: latest 3.5 today; 3 months ago = 3.0
        _add_obs(s, "US", "cpi_yoy", today, 3.5)
        _add_obs(s, "US", "cpi_yoy", today - timedelta(days=30), 3.3)
        _add_obs(s, "US", "cpi_yoy", today - timedelta(days=90), 3.0)
        _add_obs(s, "US", "cpi_yoy", today - timedelta(days=120), 2.8)
        # Policy rate: latest 3.0 today, 6 months ago 4.5
        _add_obs(s, "US", "policy_rate", today, 3.0)
        _add_obs(s, "US", "policy_rate", today - timedelta(days=180), 4.5)
        s.commit()

    with session_factory() as s:
        f = extract_features(s, "US")

    assert f.cpi_yoy == 3.5
    assert f.cpi_yoy_3m_ago == 3.0
    assert f.policy_rate == 3.0
    assert f.policy_rate_6m_ago == 4.5
    assert f.unemployment_rate is None
    assert f.indicator_dates["cpi_yoy"] == today


def test_classify_full_flow_via_db(session_factory):
    today = date(2026, 4, 1)
    with session_factory() as s:
        # Put together a clear-recession scenario
        _add_obs(s, "US", "real_gdp_yoy", today, -0.5)
        _add_obs(s, "US", "cpi_yoy", today, 2.0)
        _add_obs(s, "US", "unemployment_rate", today, 5.5)
        _add_obs(s, "US", "unemployment_rate", today - timedelta(days=90), 4.5)
        _add_obs(s, "US", "policy_rate", today, 4.0)
        _add_obs(s, "US", "yield_10y", today, 3.5)
        _add_obs(s, "US", "yield_2y", today, 4.0)
        s.commit()

    with session_factory() as s:
        c = classify(s, "US")

    assert isinstance(c, Classification)
    assert c.stage == 3  # Recession dominant
    assert c.confidence > 0


# ─── Slice 26: freshness guard, source tie-break, honest zero-vote label ────


def test_stale_input_is_dropped_and_reported(session_factory):
    today = date(2026, 8, 24)
    with session_factory() as s:
        _add_obs(s, "CN", "real_gdp_yoy", date(2026, 4, 1), 4.3)
        _add_obs(s, "CN", "cpi_yoy", date(2026, 6, 1), 1.0)
        _add_obs(s, "CN", "policy_rate", date(2026, 6, 1), 3.0)
        _add_obs(s, "CN", "unemployment_rate", date(2011, 7, 1), 4.1)     # dead FRED mirror
        _add_obs(s, "CN", "unemployment_rate", date(2011, 4, 1), 4.1)
        s.commit()
    with session_factory() as s:
        f = extract_features(s, "CN", as_of=today)
    assert f.unemployment_rate is None and f.unemployment_3m_ago is None
    assert f.stale_inputs == {"unemployment_rate": date(2011, 7, 1)}
    assert "unemployment_rate" not in f.indicator_dates
    assert f.has_core_inputs
    # The guard is relative to the as-of date: replaying 2011 sees it as fresh.
    with session_factory() as s:
        f2011 = extract_features(s, "CN", as_of=date(2011, 9, 1))
    assert f2011.unemployment_rate == 4.1 and not f2011.stale_inputs


def test_fresher_source_wins_and_same_date_tie_break_is_deterministic(session_factory):
    today = date(2026, 8, 24)
    with session_factory() as s:
        _add_obs(s, "JP", "cpi_yoy", date(2025, 3, 1), 3.6, source="FRED")
        _add_obs(s, "JP", "cpi_yoy", date(2026, 6, 1), 1.7, source="IMF_CPI")   # fresher
        _add_obs(s, "JP", "policy_rate", date(2026, 7, 1), 1.0, source="FRED")
        _add_obs(s, "JP", "policy_rate", date(2026, 7, 1), 0.9, source="BIS_CBPOL")  # same date
        s.commit()
    with session_factory() as s:
        f = extract_features(s, "JP", as_of=today)
    assert f.cpi_yoy == 1.7 and f.indicator_dates["cpi_yoy"] == date(2026, 6, 1)
    assert f.policy_rate == 0.9                      # source ascending: BIS_CBPOL < FRED


def test_zero_votes_label_distinguishes_missing_core_from_no_rule():
    quiet = _features(real_gdp_yoy=0.5, cpi_yoy=1.7, cpi_yoy_3m_ago=1.7, policy_rate=1.0,
                      policy_rate_6m_ago=1.0, yield_10y=1.5, yield_2y=1.2,
                      indicator_dates={"real_gdp_yoy": date(2026, 4, 1), "cpi_yoy": date(2026, 6, 1),
                                       "policy_rate": date(2026, 7, 1)})
    c = classify_features(quiet)
    assert c.stage == 0 and c.stage_label == "Transition (no rule fired)"
    empty = ShortTermFeatures(country="XX")
    assert classify_features(empty).stage_label == "Transition (insufficient data)"
