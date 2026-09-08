"""SCB debt-holder pipeline writes one complete multidimensional release."""

from datetime import UTC, date, datetime

import pandas as pd
from sqlalchemy import func, select

from dalio.data_sources.scb_financial_accounts import (
    SCB_GOVERNMENT_DEBT_HOLDERS,
    SOURCE_SCB_FINANCIAL_ACCOUNTS,
)
from dalio.pipelines.fetch_debt_holders import partition_key_for, run_pipeline
from dalio.storage.db import DataRelease, DebtHolderPosition, make_engine, make_session_factory
from dalio.storage.debt import load_debt_holder_vintage


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _frame(rows: list[tuple[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "country": "SE",
            "date": date(2026, 1, 1),
            "issuer_sector_code": "S1311",
            "issuer_sector_label": "Central government",
            "instrument_code": "FL3000",
            "instrument_label": "Debt securities",
            "holder_sector_code": holder,
            "holder_sector_label": {"S121": "Central bank", "S2": "Rest of the world"}[holder],
            "measure_code": "FM0103AS",
            "measure_label": "Balances",
            "unit": "SEK million",
            "value": value,
            "source": SOURCE_SCB_FINANCIAL_ACCOUNTS,
            "series_id": f"TAB1203/S1311/FL3000/{holder}/FM0103AS",
            "status": "observed",
        }
        for holder, value in rows
    ])


class FakeSource:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame
        self.calls = []

    def fetch(self, spec=SCB_GOVERNMENT_DEBT_HOLDERS, use_cache=True):
        self.calls.append((spec, use_cache))
        return self.frame.copy()


def test_pipeline_records_complete_release_and_preserves_old_vintage(tmp_path):
    engine = make_engine(tmp_path / "debt.db")
    source = FakeSource(_frame([("S121", 100.0), ("S2", 120.0)]))
    first = run_pipeline(source=source, use_cache=False, engine=engine, retrieved_at=_at(2))
    source.frame = _frame([("S121", 90.0)])
    second = run_pipeline(source=source, engine=engine, retrieved_at=_at(3))

    assert source.calls == [
        (SCB_GOVERNMENT_DEBT_HOLDERS, False),
        (SCB_GOVERNMENT_DEBT_HOLDERS, True),
    ]
    assert first["created"] is True and first["rows"] == 2
    assert second["created"] is True and second["rows"] == 1

    factory = make_session_factory(engine)
    with factory() as session:
        old = load_debt_holder_vintage(session, _at(2))
        new = load_debt_holder_vintage(session, _at(3))
        releases = session.scalars(select(DataRelease).order_by(DataRelease.id)).all()
        position_count = session.scalar(select(func.count()).select_from(DebtHolderPosition))

    assert set(old["holder_sector_code"]) == {"S121", "S2"}
    assert list(new["holder_sector_code"]) == ["S121"]
    assert position_count == 3
    assert len(releases) == 2
    assert all(release.partition_key == partition_key_for() for release in releases)
    assert all(release.source_url == SCB_GOVERNMENT_DEBT_HOLDERS.url for release in releases)


def test_pipeline_fails_closed_on_empty_response(tmp_path):
    engine = make_engine(tmp_path / "debt.db")
    source = FakeSource(_frame([("S121", 100.0)]))
    run_pipeline(source=source, engine=engine, retrieved_at=_at(2))
    source.frame = pd.DataFrame(columns=_frame([]).columns)

    failed = run_pipeline(source=source, engine=engine, retrieved_at=_at(3))

    factory = make_session_factory(engine)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1
        assert session.scalar(select(func.count()).select_from(DebtHolderPosition)) == 1
    assert "empty" in failed["error"]
