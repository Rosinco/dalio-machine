from datetime import date

import pytest

from dalio.scoring.long_term import (
    LongTermFeatures,
    PhaseClassification,
    classify_features,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


def _features(**overrides) -> LongTermFeatures:
    base = dict(
        country="US",
        total_credit_pct_gdp=200.0,
        total_credit_5y_ago=180.0,
        gov_debt_pct_gdp=80.0,
        private_nonfin_pct_gdp=120.0,
        hh_debt_pct_gdp=60.0,
        corp_debt_pct_gdp=60.0,
        debt_service_ratio=14.0,
        debt_service_5y_ago=13.0,
        yield_10y=4.0,
        cpi_yoy=3.0,
    )
    base.update(overrides)
    return LongTermFeatures(**base)


def test_sound_money_at_low_leverage():
    f = _features(total_credit_pct_gdp=80.0, total_credit_5y_ago=70.0)
    c = classify_features(f)
    assert c.phase == 1
    assert c.phase_label == "Sound money"


def test_top_at_extreme_debt():
    # JP-like: 357% with moderate DSR
    f = _features(total_credit_pct_gdp=357.0, total_credit_5y_ago=370.0,
                  debt_service_ratio=15.3)
    c = classify_features(f)
    assert c.phase == 4


def test_top_at_extreme_dsr_with_elevated_debt():
    # Sweden-like debt + extreme DSR — Top fires alongside Deleveraging
    f = _features(total_credit_pct_gdp=270.0, total_credit_5y_ago=300.0,
                  debt_service_ratio=23.0, cpi_yoy=2.0)
    c = classify_features(f)
    # Multiple phases fire — accept either Top or Deleveraging as winner
    assert c.phase in (4, 5)


def test_beautiful_deleveraging_fires_phase_6():
    # UK-like: debt fell sharply with high inflation
    f = _features(total_credit_pct_gdp=219.0, total_credit_5y_ago=305.0,
                  cpi_yoy=3.4, yield_10y=4.7)
    c = classify_features(f)
    assert c.phase == 6
    assert any("beautiful" in v.reason for v in c.votes)


def test_financial_repression_at_negative_real_rates():
    f = _features(total_credit_pct_gdp=200.0, total_credit_5y_ago=190.0,
                  yield_10y=1.0, cpi_yoy=4.0)  # real rate = -3%
    c = classify_features(f)
    assert c.phase == 6
    assert any("financial repression" in v.reason for v in c.votes)


def test_ugly_deleveraging_distinguished_from_beautiful():
    # Debt falling AND high DSR AND low inflation → ugly (Phase 5)
    f = _features(total_credit_pct_gdp=200.0, total_credit_5y_ago=215.0,
                  debt_service_ratio=20.0, cpi_yoy=2.0)
    c = classify_features(f)
    assert c.phase == 5


def test_no_data_returns_phase_zero():
    f = LongTermFeatures(country="XX")
    c = classify_features(f)
    assert c.phase == 0
    assert c.confidence == 0.0


def test_real_rate_computed_from_yield_minus_cpi():
    f = _features(yield_10y=4.5, cpi_yoy=3.5)
    assert f.real_rate_10y == pytest.approx(1.0)


def test_real_rate_none_when_inputs_missing():
    f = _features(cpi_yoy=None)
    assert f.real_rate_10y is None


def test_5y_change_computed():
    f = _features(total_credit_pct_gdp=250.0, total_credit_5y_ago=200.0)
    assert f.total_credit_5y_change_pp == pytest.approx(50.0)


def test_classification_carries_features():
    f = _features()
    c = classify_features(f)
    assert isinstance(c, PhaseClassification)
    assert c.country == "US"
    assert c.features is f


# ─── DB-backed extraction test ─────────────────────────────────────────────

@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def test_extract_features_pulls_total_credit_and_lag(session_factory):
    from datetime import timedelta

    from dalio.scoring.long_term import extract_features

    today = date(2025, 7, 1)
    five_years_ago = today - timedelta(days=365 * 5)
    with session_factory() as s:
        s.add(Observation(
            country="US", indicator="total_credit_pct_gdp",
            date=today, value=250.0, source="BIS_TC", series_id="X",
        ))
        s.add(Observation(
            country="US", indicator="total_credit_pct_gdp",
            date=five_years_ago, value=290.0, source="BIS_TC", series_id="X",
        ))
        s.commit()

    with session_factory() as s:
        f = extract_features(s, "US")

    assert f.total_credit_pct_gdp == 250.0
    assert f.total_credit_5y_ago == 290.0
    assert f.total_credit_5y_change_pp == pytest.approx(-40.0)
