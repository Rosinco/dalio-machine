"""Contracts for the all-company financial companion pack."""

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("financial_model", SCRIPTS / "financial_model.py")
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def row(**changes):
    return {
        "year": 2025,
        "period": 5,
        "report_start_date": "2025-01-01",
        "report_end_date": "2025-12-31",
        "report_date": "2026-02-01",
        "currency": "EUR",
        "currency_ratio": 2.0,
        "revenues": 200.0,
        "gross_income": 60.0,
        "operating_income": 20.0,
        "total_assets": 160.0,
        "total_equity": 80.0,
        "net_debt": 20.0,
        "cash_flow_from_operating_activities": 0.0,
        **changes,
    }


def test_report_codec_retains_raw_units_zero_and_source():
    packed, issue = model.pack_report(row(), "2026-08-10", "latest-a", annual=True)
    assert issue is None
    assert packed[:8] == [2025, 5, "2025-01-01", "2025-12-31", "2026-02-01", "EUR", 2.0, "latest-a"]
    assert packed[8 + model.COLUMNS.index("revenues")] == 200
    assert packed[8 + model.COLUMNS.index("cash_flow_from_operating_activities")] == 0
    assert packed[8 + model.COLUMNS.index("free_cash_flow")] is None


@pytest.mark.parametrize(
    "changes",
    [
        {"report_date": "2699-03-22"},
        {"report_date": "1899-12-30"},
        {"report_end_date": "2026-12-31"},
        {"report_start_date": None},
        {"currency": None},
        {"period": 1},
        {"report_date": "2026-02-30"},
    ],
)
def test_invalid_newest_metadata_is_withheld_with_reason(changes):
    packed, issue = model.pack_report(row(**changes), "2026-08-10", "latest-a", annual=True)
    assert packed is None and issue["reason"]
    assert issue["source_id"] == "latest-a"


def test_missing_conversion_is_retained_without_guessing_and_gaps_are_counted():
    first, _ = model.pack_report(
        row(year=2023, currency_ratio=0), "2026-08-10", "latest-a", annual=True
    )
    last, _ = model.pack_report(row(), "2026-08-10", "latest-a", annual=True)
    coverage = model.period_coverage([first, last], annual=True)
    assert coverage == {
        "count": 2,
        "first": 2023,
        "last": 2025,
        "last_period": 5,
        "end": "2025-12-31",
        "published": "2026-02-01",
        "gaps": 1,
        "unavailable": 1,
    }


def test_unknown_publication_date_is_kept_and_quarters_have_independent_slots():
    a, issue = model.pack_report(
        row(period=1, report_date=None), "2026-08-10", "latest-q", annual=False
    )
    b, _ = model.pack_report(row(period=3), "2026-08-10", "latest-q", annual=False)
    assert issue is None and a[4] is None
    assert model.period_coverage([a, b], annual=False)["gaps"] == 1


def test_pack_is_self_contained_and_selected_payloads_are_checksummed(tmp_path):
    import gzip
    import hashlib
    import json
    import sqlite3

    payload = {
        "id": "102",
        "annual": [model.pack_report(row(), "2026-08-10", "latest-a", annual=True)[0]],
        "quarterly": [],
        "withheld": [],
    }
    path = tmp_path / "pack.sqlite"
    conn = model.create_database(path)
    coverage = model.store_company(conn, payload)
    conn.commit()
    blob, digest = conn.execute(
        "SELECT payload, sha256 FROM companies WHERE id=?", ("102",)
    ).fetchone()
    assert json.loads(gzip.decompress(blob)) == payload
    assert hashlib.sha256(gzip.decompress(blob)).hexdigest() == digest
    assert coverage["annual"]["count"] == 1
    assert (
        sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        .execute("SELECT count(*) FROM companies")
        .fetchone()[0]
        == 1
    )
    with pytest.raises(sqlite3.IntegrityError):
        model.store_company(conn, payload)
    conn.close()
