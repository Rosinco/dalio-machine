"""No future economic-date observations may influence historical classifiers."""

from datetime import date, timedelta

import pytest

from dalio.scoring import calibration, long_term, short_term
from dalio.scoring.asset_signals import MIN_OBSERVATIONS, compute_asset_signals
from dalio.scoring.thresholds import DEFAULT_THRESHOLDS
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "historical-asof.db")
    init_db(engine)
    return make_session_factory(engine)


@pytest.mark.parametrize("classifier", [short_term.classify, long_term.classify])
def test_direct_historical_classifier_forwards_cutoff_to_calibration(
    session_factory,
    monkeypatch,
    classifier,
):
    cutoff = date(2020, 6, 30)
    calls = []

    def fake_thresholds(session, country, as_of=None):
        calls.append((session, country, as_of))
        return DEFAULT_THRESHOLDS

    monkeypatch.setattr(calibration, "compute_country_thresholds", fake_thresholds)
    with session_factory() as session:
        classifier(session, "US", as_of=cutoff)

    assert [(country, as_of) for _session, country, as_of in calls] == [("US", cutoff)]


def test_asset_signal_window_is_anchored_and_capped_at_historical_date(session_factory):
    cutoff = date(2020, 6, 30)
    start = cutoff - timedelta(days=MIN_OBSERVATIONS - 1)
    with session_factory() as session:
        for offset in range(MIN_OBSERVATIONS):
            session.add(Observation(
                country="US",
                indicator="hy_spread",
                date=start + timedelta(days=offset),
                value=float(offset + 1),
                source="TEST",
                series_id="HY",
            ))
        for offset in range(1, MIN_OBSERVATIONS + 1):
            session.add(Observation(
                country="US",
                indicator="hy_spread",
                date=cutoff + timedelta(days=offset),
                value=999.0,
                source="TEST",
                series_id="HY",
            ))
        session.commit()

    with session_factory() as session:
        historical = compute_asset_signals(session, "US", as_of=cutoff)
        full = compute_asset_signals(
            session,
            "US",
            as_of=cutoff + timedelta(days=MIN_OBSERVATIONS),
        )

    assert historical.hy_spread_latest == float(MIN_OBSERVATIONS)
    assert full.hy_spread_latest == 999.0


def test_long_term_feature_extraction_passes_its_cutoff_to_asset_signals(
    session_factory,
    monkeypatch,
):
    cutoff = date(2020, 6, 30)
    seen = []

    def fake_asset_signals(session, country, as_of=None):
        from dalio.scoring.asset_signals import AssetSignals

        seen.append((session, country, as_of))
        return AssetSignals(country=country, hy_spread_latest=4.0, hy_spread_z=0.0)

    monkeypatch.setattr(
        "dalio.scoring.asset_signals.compute_asset_signals",
        fake_asset_signals,
    )
    with session_factory() as session:
        features = long_term.extract_features(session, "US", as_of=cutoff)

    assert [(country, as_of) for _session, country, as_of in seen] == [("US", cutoff)]
    assert features.hy_spread == 4.0
