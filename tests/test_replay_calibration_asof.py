from datetime import date, timedelta
from types import SimpleNamespace

import pytest
from dateutil.relativedelta import relativedelta

from dalio.scoring import replay
from dalio.scoring.calibration import (
    MIN_OBSERVATIONS,
    compute_country_quantiles,
    compute_country_thresholds,
)
from dalio.scoring.short_term import classify as classify_short_term
from dalio.scoring.thresholds import Thresholds
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "replay-asof.db")
    init_db(engine)
    return make_session_factory(engine)


def _add_observation(session, indicator: str, observed_on: date, value: float) -> None:
    session.add(
        Observation(
            country="US",
            indicator=indicator,
            date=observed_on,
            value=value,
            source="TEST",
            series_id="X",
        )
    )


def test_quantiles_and_thresholds_accept_an_economic_date_cutoff(session_factory):
    early_start = date(2000, 1, 1)
    future_start = date(2030, 1, 1)
    with session_factory() as session:
        for offset in range(MIN_OBSERVATIONS):
            _add_observation(
                session,
                "debt_service_ratio",
                early_start + timedelta(days=offset),
                10.0,
            )
            _add_observation(
                session,
                "debt_service_ratio",
                future_start + timedelta(days=offset),
                100.0,
            )
        session.commit()

    cutoff = early_start + timedelta(days=MIN_OBSERVATIONS - 1)
    before_minimum = cutoff - timedelta(days=1)
    with session_factory() as session:
        expanding = compute_country_quantiles(
            session,
            "US",
            "debt_service_ratio",
            as_of=cutoff,
        )
        full = compute_country_quantiles(session, "US", "debt_service_ratio")
        too_short = compute_country_quantiles(
            session,
            "US",
            "debt_service_ratio",
            as_of=before_minimum,
        )
        expanding_thresholds = compute_country_thresholds(
            session,
            "US",
            as_of=cutoff,
        )
        default_call_thresholds = compute_country_thresholds(session, "US")

    assert expanding is not None and expanding["q90"] == 10.0
    assert full is not None and full["q90"] == 100.0
    assert too_short is None
    assert expanding_thresholds.dsr_distress == 10.0
    assert default_call_thresholds.dsr_distress == 100.0


def test_replay_classification_does_not_use_future_values_for_calibration(session_factory):
    cursor = date(2020, 1, 1)
    with session_factory() as session:
        for months_before in range(MIN_OBSERVATIONS - 1, 0, -1):
            _add_observation(
                session,
                "cpi_yoy",
                cursor - relativedelta(months=months_before),
                1.0,
            )
        _add_observation(session, "cpi_yoy", cursor, 5.0)
        for months_after in range(1, MIN_OBSERVATIONS + 1):
            _add_observation(
                session,
                "cpi_yoy",
                cursor + relativedelta(months=months_after),
                100.0,
            )
        session.commit()

    with session_factory() as session:
        full_history_thresholds = compute_country_thresholds(session, "US")
        leaky_result = classify_short_term(
            session,
            "US",
            thresholds=full_history_thresholds,
            as_of=cursor,
        )
        result = replay.replay_classifications(session, "US", cursor, cursor)

    assert full_history_thresholds.cpi_peak == 100.0
    assert leaky_result.stage == 0
    assert result.iloc[0]["st_stage"] == 2


def test_replay_computes_one_threshold_set_per_cursor_and_shares_it(monkeypatch):
    first = date(2020, 1, 1)
    second = date(2020, 4, 1)
    thresholds_by_cursor: dict[date, Thresholds] = {}
    threshold_calls: list[tuple[str, date]] = []
    classifier_calls: list[tuple[str, date, Thresholds]] = []

    def fake_thresholds(_session, country, as_of=None):
        threshold_calls.append((country, as_of))
        thresholds = Thresholds(cpi_peak=float(as_of.month))
        thresholds_by_cursor[as_of] = thresholds
        return thresholds

    def fake_short_term(_session, _country, thresholds=None, as_of=None):
        classifier_calls.append(("short", as_of, thresholds))
        return SimpleNamespace(stage=1, stage_label="Expansion", confidence=0.5)

    def fake_long_term(_session, _country, thresholds=None, as_of=None):
        classifier_calls.append(("long", as_of, thresholds))
        return SimpleNamespace(phase=2, phase_label="Debt outpaces", confidence=0.4)

    monkeypatch.setattr(replay, "compute_country_thresholds", fake_thresholds)
    monkeypatch.setattr(replay, "classify_short_term", fake_short_term)
    monkeypatch.setattr(replay, "classify_long_term", fake_long_term)

    result = replay.replay_classifications(object(), "US", first, second, step="Q")

    assert list(result["date"]) == [first, second]
    assert threshold_calls == [("US", first), ("US", second)]
    assert [(kind, cursor) for kind, cursor, _thresholds in classifier_calls] == [
        ("short", first),
        ("long", first),
        ("short", second),
        ("long", second),
    ]
    assert all(
        thresholds is thresholds_by_cursor[cursor] for _kind, cursor, thresholds in classifier_calls
    )
