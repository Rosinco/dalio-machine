"""Auditable allocator facts retain source artifacts and point-in-time history."""

import hashlib
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import func, select

from dalio.storage.allocators import (
    AllocatorReleaseMeta,
    ingest_allocator_snapshot,
    load_allocator_vintage,
)
from dalio.storage.db import AllocatorFact, DataRelease, init_db, make_engine, make_session_factory

PDF = b"%PDF-1.4\ntrusted allocator fixture\n%%EOF\n"


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _frame(rows: list[tuple[str, float | None, float | None]]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "fund": "AP2",
            "as_of_date": date(2026, 6, 30),
            "period_start": None,
            "period_end": None,
            "record_type": "asset_allocation",
            "item_code": item,
            "reported_amount": amount,
            "reported_unit": "SEK_bn" if amount is not None else None,
            "amount_sek_mn": amount * 1_000 if amount is not None else None,
            "exposure_pct": exposure,
            "basis": "actual_portfolio_exposure",
            "row_role": "detail",
            "physical_page": 4,
            "table_heading": "Asset-class exposure",
            "extraction_status": "model_visual_check",
            "quality_flag": "none",
            "notes": None,
        }
        for item, amount, exposure in rows
    ])


def _meta(available_at: datetime, retrieved_at: datetime | None = None) -> AllocatorReleaseMeta:
    return AllocatorReleaseMeta(
        partition_key="allocator:AP2:H1",
        source_family="AP_FUNDS",
        fund="AP2",
        report_date=date(2026, 6, 30),
        title="AP2 Half-Year Report 2026",
        available_at=available_at,
        retrieved_at=retrieved_at or available_at,
        source_url="https://ap2.se/report.pdf",
        official_domains=("ap2.se",),
        expected_sha256=hashlib.sha256(PDF).hexdigest(),
        parser_name="manual-table-transcription",
        parser_version="1",
    )


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "allocators.db")
    init_db(engine)
    return make_session_factory(engine)


def test_allocator_artifact_rows_and_vintage_are_auditable(session_factory, tmp_path):
    first_frame = _frame([("listed_equities", 100.0, 40.0), ("fixed_income", 90.0, 36.0)])
    second_frame = _frame([("listed_equities", 110.0, 42.0)])
    with session_factory() as session:
        first = ingest_allocator_snapshot(
            session, first_frame, PDF, _meta(_at(8)), blob_root=tmp_path / "artifacts"
        )
        second = ingest_allocator_snapshot(
            session, second_frame, PDF, _meta(_at(9)), blob_root=tmp_path / "artifacts"
        )
        old = load_allocator_vintage(session, _at(8))
        new = load_allocator_vintage(session, _at(9))

    assert first.created and second.created
    assert first.blob_path == second.blob_path
    assert first.blob_path.read_bytes() == PDF
    assert set(old["item_code"]) == {"listed_equities", "fixed_income"}
    assert list(new["item_code"]) == ["listed_equities"]
    assert set(new["artifact_sha256"]) == {hashlib.sha256(PDF).hexdigest()}
    assert set(new["source_url"]) == {"https://ap2.se/report.pdf"}


def test_allocator_snapshot_is_idempotent_when_rows_reorder(session_factory, tmp_path):
    frame = _frame([("listed_equities", 100.0, 40.0), ("fixed_income", 90.0, 36.0)])
    with session_factory() as session:
        first = ingest_allocator_snapshot(
            session, frame, PDF, _meta(_at(8)), blob_root=tmp_path / "artifacts"
        )
        again = ingest_allocator_snapshot(
            session, frame.iloc[::-1], PDF, _meta(_at(8)), blob_root=tmp_path / "artifacts"
        )
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        fact_count = session.scalar(select(func.count()).select_from(AllocatorFact))

    assert first.created is True
    assert again.created is False
    assert again.release_id == first.release_id
    assert release_count == 1
    assert fact_count == 2


def test_allocator_filters_happen_after_release_ranking(session_factory, tmp_path):
    with session_factory() as session:
        ingest_allocator_snapshot(
            session,
            _frame([("listed_equities", 100.0, 40.0), ("fixed_income", 90.0, 36.0)]),
            PDF,
            _meta(_at(8)),
            blob_root=tmp_path / "artifacts",
        )
        ingest_allocator_snapshot(
            session,
            _frame([("listed_equities", 110.0, 42.0)]),
            PDF,
            _meta(_at(9)),
            blob_root=tmp_path / "artifacts",
        )
        omitted = load_allocator_vintage(session, _at(9), item_codes=("fixed_income",))

    assert omitted.empty


def test_allocator_snapshot_rejects_untrusted_or_inconsistent_input(session_factory, tmp_path):
    with session_factory() as session:
        with pytest.raises(ValueError, match="official domain"):
            ingest_allocator_snapshot(
                session,
                _frame([("listed_equities", 100.0, 40.0)]),
                PDF,
                AllocatorReleaseMeta(**{**_meta(_at(8)).__dict__, "source_url": "https://evil.test/x"}),
                blob_root=tmp_path / "artifacts",
            )
        with pytest.raises(ValueError, match="sha256"):
            ingest_allocator_snapshot(
                session,
                _frame([("listed_equities", 100.0, 40.0)]),
                PDF + b"changed",
                _meta(_at(8)),
                blob_root=tmp_path / "artifacts",
            )
        with pytest.raises(ValueError, match="empty"):
            ingest_allocator_snapshot(
                session, _frame([]), PDF, _meta(_at(8)), blob_root=tmp_path / "artifacts"
            )
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0


def test_allocator_fact_rows_are_immutable(session_factory, tmp_path):
    with session_factory() as session:
        ingest_allocator_snapshot(
            session,
            _frame([("listed_equities", 100.0, 40.0)]),
            PDF,
            _meta(_at(8)),
            blob_root=tmp_path / "artifacts",
        )
        fact = session.scalar(select(AllocatorFact))
        fact.exposure_pct = 41.0
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
