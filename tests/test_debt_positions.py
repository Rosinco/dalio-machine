"""Typed, point-in-time sovereign debt-holder positions."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import func, select

from dalio.storage.db import DataRelease, DebtHolderPosition, make_engine, make_session_factory
from dalio.storage.debt import ingest_debt_holder_snapshot, load_debt_holder_vintage
from dalio.storage.releases import ReleaseMeta


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "country": "SE",
            "date": date(2026, 1, 1),
            "issuer_sector_code": "S1311",
            "issuer_sector_label": "Central government",
            "instrument_code": "FL3200",
            "instrument_label": "Long-term debt securities",
            "holder_sector_code": holder,
            "holder_sector_label": {
                "S121": "Central bank",
                "S129": "Pension funds",
                "S2": "Rest of the world",
            }[holder],
            "measure_code": "FM0103AS",
            "measure_label": "Balances",
            "unit": "SEK million",
            "value": value,
            "source": "SCB_FIN_ACCTS",
            "series_id": "TAB1203/S1311/FL3200/FM0103AS",
        }
        for holder, value in rows
    ])


def _meta(available_at: datetime, retrieved_at: datetime | None = None) -> ReleaseMeta:
    return ReleaseMeta(
        partition_key="facts:SCB_FIN_ACCTS:TAB1203:SE:government-debt-holders",
        source_family="SCB_FIN_ACCTS",
        available_at=available_at,
        retrieved_at=retrieved_at or available_at,
        source_url="https://api.scb.se/official-query",
    )


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "debt.db")
    from dalio.storage.db import init_db

    init_db(engine)
    return make_session_factory(engine)


def test_debt_holder_revisions_and_omissions_are_point_in_time_honest(session_factory):
    with session_factory() as session:
        first = ingest_debt_holder_snapshot(
            session,
            _frame([("S121", 100.0), ("S129", 80.0)]),
            _meta(_at(2)),
        )
        second = ingest_debt_holder_snapshot(
            session,
            _frame([("S121", 90.0), ("S2", 120.0)]),
            _meta(_at(3)),
        )

        before = load_debt_holder_vintage(session, _at(2))
        after = load_debt_holder_vintage(session, _at(3))

    assert first.created and first.row_count == 2
    assert second.created and second.row_count == 2
    assert dict(zip(before["holder_sector_code"], before["value"], strict=True)) == {
        "S121": 100.0,
        "S129": 80.0,
    }
    assert dict(zip(after["holder_sector_code"], after["value"], strict=True)) == {
        "S121": 90.0,
        "S2": 120.0,
    }


def test_debt_holder_snapshot_is_content_idempotent(session_factory):
    frame = _frame([("S121", 100.0), ("S129", 80.0)])
    with session_factory() as session:
        first = ingest_debt_holder_snapshot(session, frame, _meta(_at(2)))
        again = ingest_debt_holder_snapshot(session, frame.iloc[::-1], _meta(_at(3)))
        releases = session.scalar(select(func.count()).select_from(DataRelease))
        positions = session.scalar(select(func.count()).select_from(DebtHolderPosition))

    assert first.created is True
    assert again.created is False
    assert again.release_id == first.release_id
    assert releases == 1
    assert positions == 2


def test_late_older_debt_release_does_not_win_current_vintage(session_factory):
    with session_factory() as session:
        ingest_debt_holder_snapshot(session, _frame([("S121", 90.0)]), _meta(_at(3)))
        late_old = ingest_debt_holder_snapshot(
            session,
            _frame([("S121", 100.0)]),
            _meta(_at(2), retrieved_at=_at(4)),
        )
        current = load_debt_holder_vintage(session, _at(4))

    assert late_old.created is True
    assert list(current["value"]) == [90.0]


def test_debt_holder_filters_apply_after_release_ranking(session_factory):
    with session_factory() as session:
        ingest_debt_holder_snapshot(
            session,
            _frame([("S121", 100.0), ("S129", 80.0)]),
            _meta(_at(2)),
        )
        ingest_debt_holder_snapshot(session, _frame([("S121", 90.0)]), _meta(_at(3)))
        omitted = load_debt_holder_vintage(
            session,
            _at(3),
            holder_sector_codes=("S129",),
        )

    assert omitted.empty


def test_debt_holder_snapshot_rejects_empty_duplicate_and_nonfinite(session_factory):
    with session_factory() as session:
        with pytest.raises(ValueError, match="empty"):
            ingest_debt_holder_snapshot(session, _frame([]), _meta(_at(2)))

        duplicate = pd.concat([_frame([("S121", 100.0)]), _frame([("S121", 101.0)])])
        with pytest.raises(ValueError, match="duplicate"):
            ingest_debt_holder_snapshot(session, duplicate, _meta(_at(2)))

        nonfinite = _frame([("S121", float("nan"))])
        with pytest.raises(ValueError, match="finite"):
            ingest_debt_holder_snapshot(session, nonfinite, _meta(_at(2)))

        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0


def test_debt_holder_cells_are_immutable(session_factory):
    with session_factory() as session:
        ingest_debt_holder_snapshot(session, _frame([("S121", 100.0)]), _meta(_at(2)))
        position = session.scalar(select(DebtHolderPosition))
        position.value = 99.0
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
