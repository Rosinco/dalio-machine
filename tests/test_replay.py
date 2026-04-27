"""Tests for the historical regime replay (Slice 12)."""
from datetime import date

import pandas as pd
import pytest

from dalio.scoring.replay import replay_classifications
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "test.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed_recession(session, target_date: date) -> None:
    """Seed enough US data for an unambiguous Recession classification at
    target_date: GDP < 0, unemployment rising sharply (Sahm rule)."""
    # Unemployment trajectory: 4 → 5.5 over the 3 months before target_date
    obs = [
        ("unemployment_rate", target_date.replace(day=1).replace(month=max(1, target_date.month - 4)), 4.0),
        ("unemployment_rate", target_date, 5.5),
        ("real_gdp_yoy", target_date, -1.5),
        ("policy_rate", target_date, 4.0),
        ("cpi_yoy", target_date, 2.0),
    ]
    for ind, d, v in obs:
        session.add(Observation(
            country="US", indicator=ind, date=d,
            value=float(v), source="TEST", series_id="X",
        ))


def test_replay_returns_empty_when_no_data(session_factory):
    with session_factory() as s:
        df = replay_classifications(s, "US", date(2008, 1, 1), date(2008, 12, 31), "Q")
    # No data → all rows still produced, but classifications are "insufficient data"
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # 4 quarters
    assert all(df["st_stage"] == 0)
    assert all(df["lt_phase"] == 0)


def test_replay_quarterly_step_count(session_factory):
    """Quarterly step from 2007-01-01 to 2010-01-01 (4 years) → 13 rows."""
    with session_factory() as s:
        df = replay_classifications(s, "US", date(2007, 1, 1), date(2010, 1, 1), "Q")
    assert len(df) == 13


def test_replay_monthly_step_count(session_factory):
    """Monthly step from 2020-01-01 to 2020-06-01 → 6 rows (Jan, Feb, ..., Jun)."""
    with session_factory() as s:
        df = replay_classifications(s, "US", date(2020, 1, 1), date(2020, 6, 1), "M")
    assert len(df) == 6


def test_replay_columns_present(session_factory):
    with session_factory() as s:
        df = replay_classifications(s, "US", date(2020, 1, 1), date(2020, 4, 1), "Q")
    expected = {"date", "st_stage", "st_label", "st_confidence",
                "lt_phase", "lt_label", "lt_confidence"}
    assert set(df.columns) == expected


def test_replay_recession_classification(session_factory):
    """Seeded GDP < 0 + Sahm-rule unemployment → Stage 3 Recession at target."""
    target = date(2009, 6, 30)
    with session_factory() as s:
        _seed_recession(s, target)
        s.commit()
    with session_factory() as s:
        df = replay_classifications(
            s, "US",
            start=date(2009, 6, 30),
            end=date(2009, 6, 30),
            step="Q",
        )
    # Single row at target — should be Stage 3 Recession
    assert len(df) == 1
    assert df.iloc[0]["st_stage"] == 3
    assert "Recession" in df.iloc[0]["st_label"]


def test_replay_as_of_caps_lookups(session_factory):
    """Recession seeded at 2009-06-30 must NOT bleed back into the 2008-Q1 row."""
    target = date(2009, 6, 30)
    with session_factory() as s:
        _seed_recession(s, target)
        s.commit()
    with session_factory() as s:
        df = replay_classifications(
            s, "US",
            start=date(2008, 1, 1),
            end=date(2009, 12, 31),
            step="Q",
        )
    # 2008-Q1 row should NOT see the 2009-06 recession data → Stage 0 (no data)
    early = df[df["date"] == date(2008, 1, 1)].iloc[0]
    assert early["st_stage"] == 0
    # The 2009-Q3 row should see the recession data
    late = df[df["date"] == date(2009, 10, 1)].iloc[0]
    assert late["st_stage"] == 3 or "Recession" in late["st_label"]
