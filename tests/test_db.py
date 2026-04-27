from datetime import date

import pytest
from sqlalchemy.exc import IntegrityError

from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    db_path = tmp_path / "test.db"
    engine = make_engine(db_path)
    init_db(engine)
    return make_session_factory(engine)


def test_insert_and_query(session_factory):
    with session_factory() as s:
        s.add(Observation(
            country="US",
            indicator="policy_rate",
            date=date(2026, 4, 1),
            value=4.33,
            source="FRED",
            series_id="DFF",
        ))
        s.commit()

    with session_factory() as s:
        rows = s.query(Observation).all()
        assert len(rows) == 1
        assert rows[0].country == "US"
        assert rows[0].indicator == "policy_rate"
        assert rows[0].value == 4.33


def test_unique_constraint_prevents_dupes(session_factory):
    with session_factory() as s:
        s.add(Observation(
            country="US", indicator="policy_rate", date=date(2026, 4, 1),
            value=4.33, source="FRED", series_id="DFF",
        ))
        s.commit()

    with session_factory() as s:  # noqa: SIM117
        with pytest.raises(IntegrityError):
            s.add(Observation(
                country="US", indicator="policy_rate", date=date(2026, 4, 1),
                value=4.50, source="FRED", series_id="DFF",
            ))
            s.commit()


def test_different_sources_can_coexist(session_factory):
    with session_factory() as s:
        s.add(Observation(
            country="US", indicator="policy_rate", date=date(2026, 4, 1),
            value=4.33, source="FRED", series_id="DFF",
        ))
        s.add(Observation(
            country="US", indicator="policy_rate", date=date(2026, 4, 1),
            value=4.33, source="BIS", series_id="US_PR",
        ))
        s.commit()
        rows = s.query(Observation).all()
        assert len(rows) == 2
