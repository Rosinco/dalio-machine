"""Typed, immutable bilateral position snapshots."""

from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import func, select

from dalio.storage.db import CrossBorderPosition, DataRelease, init_db, make_engine
from dalio.storage.positions import (
    ingest_cross_border_snapshot,
    load_cross_border_vintage,
    make_position_partition_key,
)
from dalio.storage.releases import ReleaseMeta


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _frame(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    counterpart_codes = {"US": "USA", "UK": "GBR", "WLD": "G001"}
    return pd.DataFrame(
        [
            {
                "dataset": "PIP",
                "reporter_country": "SE",
                "reporter_code": "SWE",
                "counterpart_country": counterpart,
                "counterpart_code": counterpart_codes[counterpart],
                "date": observed_on,
                "direction": "outward_assets",
                "accounting_basis": "assets",
                "instrument_code": "portfolio_total",
                "instrument_label": "Total portfolio investment",
                "frequency": "A",
                "value": value,
                "unit": "USD",
                "source": "IMF_PIP",
                "native_indicator": "P_TOTINV_P_USD",
                "reporter_sector_code": "S1",
                "counterpart_sector_code": "S1",
                "derivation_type": None,
                "series_id": (f"PIP/SWE.A.P_TOTINV_P_USD.S1.S1.{counterpart_codes[counterpart]}.A"),
                "status": "observed",
            }
            for counterpart, observed_on, value in rows
        ]
    )


def _meta(available_at: datetime, retrieved_at: datetime | None = None) -> ReleaseMeta:
    return ReleaseMeta(
        partition_key=make_position_partition_key("IMF_PIP", "SE", "P_TOTINV_P_USD", "A"),
        source_family="IMF_PIP",
        available_at=available_at,
        retrieved_at=retrieved_at or available_at,
        source_url="https://api.imf.org/official-query",
    )


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "positions.db")
    init_db(engine)
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


def test_position_revisions_and_omissions_are_point_in_time_honest(session_factory):
    with session_factory() as session:
        first = ingest_cross_border_snapshot(
            session,
            _frame(
                [
                    ("US", date(2024, 1, 1), 100.0),
                    ("UK", date(2024, 1, 1), 80.0),
                ]
            ),
            _meta(_at(2)),
        )
        second = ingest_cross_border_snapshot(
            session,
            _frame(
                [
                    ("US", date(2024, 1, 1), 90.0),
                    ("WLD", date(2024, 1, 1), 190.0),
                ]
            ),
            _meta(_at(3)),
        )

        before = load_cross_border_vintage(session, _at(2))
        after = load_cross_border_vintage(session, _at(3))

    assert first.created and first.row_count == 2
    assert second.created and second.row_count == 2
    assert dict(zip(before["counterpart_country"], before["value"], strict=True)) == {
        "UK": 80.0,
        "US": 100.0,
    }
    assert dict(zip(after["counterpart_country"], after["value"], strict=True)) == {
        "US": 90.0,
        "WLD": 190.0,
    }


def test_position_selection_precedes_row_filters_and_late_old_release_loses(
    session_factory,
):
    with session_factory() as session:
        ingest_cross_border_snapshot(
            session,
            _frame(
                [
                    ("US", date(2024, 1, 1), 100.0),
                    ("UK", date(2024, 1, 1), 80.0),
                ]
            ),
            _meta(_at(2)),
        )
        ingest_cross_border_snapshot(
            session,
            _frame([("US", date(2024, 1, 1), 90.0)]),
            _meta(_at(3)),
        )
        late_old = ingest_cross_border_snapshot(
            session,
            _frame([("US", date(2024, 1, 1), 95.0)]),
            _meta(_at(2), retrieved_at=_at(4)),
        )

        omitted = load_cross_border_vintage(session, _at(4), counterpart_countries=("UK",))
        current = load_cross_border_vintage(
            session,
            _at(4),
            counterpart_countries=("US",),
            frequencies=("A",),
        )

    assert late_old.created is True
    assert omitted.empty
    assert list(current["value"]) == [90.0]


def test_position_snapshot_is_idempotent_and_validated(session_factory):
    frame = _frame(
        [
            ("US", date(2024, 1, 1), 100.0),
            ("UK", date(2024, 1, 1), 80.0),
        ]
    )
    with session_factory() as session:
        first = ingest_cross_border_snapshot(session, frame, _meta(_at(2)))
        again = ingest_cross_border_snapshot(session, frame.iloc[::-1], _meta(_at(3)))

        with pytest.raises(ValueError, match="empty"):
            ingest_cross_border_snapshot(session, frame.iloc[0:0], _meta(_at(3)))

        duplicate = pd.concat([frame.iloc[[0]], frame.iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate"):
            ingest_cross_border_snapshot(session, duplicate, _meta(_at(3)))

        nonfinite = frame.iloc[[0]].copy()
        nonfinite["value"] = float("inf")
        with pytest.raises(ValueError, match="finite"):
            ingest_cross_border_snapshot(session, nonfinite, _meta(_at(3)))

        releases = session.scalar(select(func.count()).select_from(DataRelease))
        positions = session.scalar(select(func.count()).select_from(CrossBorderPosition))

    assert first.created is True
    assert again.created is False
    assert again.release_id == first.release_id
    assert releases == 1
    assert positions == 2


def test_position_rows_are_immutable(session_factory):
    with session_factory() as session:
        ingest_cross_border_snapshot(
            session,
            _frame([("US", date(2024, 1, 1), 100.0)]),
            _meta(_at(2)),
        )
        position = session.scalar(select(CrossBorderPosition))
        position.value = 99.0
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
