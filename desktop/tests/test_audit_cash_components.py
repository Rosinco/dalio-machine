"""Meaningful accounting/data-boundary checks; no network or real vendor data."""

import gzip
import hashlib
import importlib.util
import json
import sqlite3
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "cash_components", Path(__file__).parents[1] / "scripts/audit-cash-components.py"
)
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def raw(**overrides):
    return {
        "cash_flow_from_operating_activities": 100,
        "cash_flow_from_investing_activities": -40,
        "cash_flow_from_financing_activities": -30,
        "cash_flow_for_the_year": 32,
        "free_cash_flow": 70,
        **overrides,
    }


def test_signed_components_are_not_capex_or_owner_cash():
    result = audit.component_evidence(raw(), 2, "EUR")
    assert result["native"]["operating"] == 50
    assert result["native"]["investing"] == -20
    assert result["derived"] == {
        "operatingPlusInvesting": 30,
        "providerFcfMinusOperatingAndInvesting": 5,
        "operatingMinusProviderFcf": 15,
        "componentSum": 15,
        "netCashMinusComponentSum": 1,
    }
    assert result["fcfComparison"] == "differs"
    assert result["netCashComparison"] == "differs"
    assert "capex" not in result["derived"]


def test_missing_component_is_not_zero_and_does_not_erase_other_cash():
    result = audit.component_evidence(raw(cash_flow_from_investing_activities=None), 1, "SEK")
    assert result["native"]["investing"] is None
    assert result["native"]["providerFcf"] == 70
    assert result["derived"]["operatingPlusInvesting"] is None
    assert result["derived"]["operatingMinusProviderFcf"] == 30
    assert result["fcfComparison"] == "unavailable"
    assert result["netCashComparison"] == "unavailable"


def test_zero_is_observed_and_matching_arithmetic_does_not_verify_definition():
    result = audit.component_evidence(dict.fromkeys(audit.FIELDS.values(), 0), 1, "SEK")
    assert result["fcfComparison"] == "matches"
    assert result["netCashComparison"] == "matches"
    assert result["native"]["providerFcf"] == 0
    assert result["definitionVerified"] is False


@pytest.mark.parametrize("ratio", [None, 0, -1, float("nan"), float("inf"), True])
def test_missing_or_invalid_conversion_preserves_raw_and_withholds_native(ratio):
    result = audit.component_evidence(raw(), ratio, "EUR")
    assert result["raw"]["operating"] == 100
    assert all(value is None for value in result["native"].values())
    assert all(value is None for value in result["derived"].values())
    assert result["fcfComparison"] == "unavailable"


def test_missing_currency_prevents_unlabelled_native_amounts():
    result = audit.component_evidence(raw(), 1, None)
    assert result["native"]["operating"] is None


def test_rounding_tolerance_is_symmetric_and_scale_sensitive():
    assert audit.comparison(100, 100.000001) == "matches"
    assert audit.comparison(100.000001, 100) == "matches"
    assert audit.comparison(0, 0.000002) == "differs"
    assert audit.comparison(100_000_000, 100_000_000.5) == "matches"
    assert audit.comparison(None, 0) == "unavailable"


def test_placeholder_flag_requires_all_statement_amounts_zero_or_missing():
    assert audit.placeholder({"cash": 0, "revenue": None})
    assert not audit.placeholder({"cash": 0, "revenue": 15})
    assert not audit.placeholder({"cash": None, "revenue": None})


def test_cross_vintage_native_comparison_handles_different_saved_fx():
    old = audit.component_evidence(raw(), 2, "EUR")
    new = audit.component_evidence({k: v * 3 for k, v in raw().items()}, 6, "EUR")
    result = audit.compare_vintages(old, new, "EUR", "EUR")
    assert result["status"] == "comparable"
    assert result["changedFields"] == []


def test_vintage_fcf_change_with_stable_components_remains_unexplained():
    old = audit.component_evidence(raw(), 1, "SEK")
    new = audit.component_evidence(raw(free_cash_flow=-20), 1, "SEK")
    result = audit.compare_vintages(old, new, "SEK", "SEK")
    assert result["fcfChanged"] is True
    assert result["fcfSignChanged"] is True
    assert result["otherFourCashFieldsUnchanged"] is True
    assert result["cause"] == "unconfirmed"


def test_cross_vintage_currency_change_is_unscored():
    evidence = audit.component_evidence(raw(), 1, "USD")
    assert audit.compare_vintages(evidence, evidence, "USD", "EUR")["status"] == "currency_changed"


def fixture_pack(tmp_path):
    columns = [*audit.FIELDS.values(), "revenues"]
    reports = [
        [2024, 5, "2024-01-01", "2024-12-31", "2025-02-01", "SEK", 1, "source", 100, -40, -20, 40, 60, 500],
        [2025, 5, "2025-01-01", "2025-12-31", None, "SEK", 1, "source", 120, -50, -30, 40, None, 600],
    ]
    companies = [{"id": "1", "annual": reports, "withheld": []}, {"id": "2", "annual": [], "withheld": []}]
    payloads = [(company["id"], json.dumps(company).encode()) for company in companies]
    index = {"columns": columns, "sources": [{"id": "source", "as_of": "2026-08-10", "sha256": "a" * 64, "path": "saved.parquet"}],
             "summary": {"annual": 2}, "companies": {ins_id: {"sha256": hashlib.sha256(raw).hexdigest()} for ins_id, raw in payloads}}
    pack = tmp_path / "pack.sqlite"
    with sqlite3.connect(pack) as db:
        db.executescript("CREATE TABLE metadata(key TEXT,payload BLOB); CREATE TABLE companies(id TEXT,payload BLOB,sha256 TEXT);")
        db.execute("INSERT INTO metadata VALUES('index',?)", (gzip.compress(json.dumps(index).encode()),))
        db.executemany("INSERT INTO companies VALUES(?,?,?)", [(ins_id, gzip.compress(raw), hashlib.sha256(raw).hexdigest()) for ins_id, raw in payloads])
    listings = {ins_id: {"name": ins_id, "isin": None, "source_as_of": "2026-08-10"} for ins_id in ["1", "2"]}
    taxonomy = {"catalogue": {"listings": listings, "as_of": "2026-08-10"}, "classifications": {
        "1": {"sector_id": "2", "branch_id": "20"}, "2": {"sector_id": None, "branch_id": None}}}
    return pack, taxonomy


def test_latest_source_availability_does_not_backfill_missing_cash_and_keeps_empty_listing(tmp_path):
    pack, taxonomy = fixture_pack(tmp_path)
    before = audit.digest(pack)
    _, result = audit.audit_pack(pack, taxonomy, tmp_path)
    assert audit.digest(pack) == before
    assert result["cohort"]["listings"] == 2
    assert result["cohort"]["noAnnualRows"] == 1
    assert result["summaries"]["allAnnualRows"]["rows"] == 2
    rows = [json.loads(line) for line in gzip.decompress((tmp_path / "latest-company-cash-components.jsonl.gz").read_bytes()).splitlines()]
    assert rows[0]["latest"]["year"] == 2025
    assert rows[0]["latest"]["cash"]["native"]["providerFcf"] is None
    assert rows[0]["latest"]["cash"]["derived"]["operatingPlusInvesting"] == 70
    assert rows[1]["latest"] is None


def test_tampered_company_identity_fails_closed(tmp_path):
    pack, taxonomy = fixture_pack(tmp_path)
    with sqlite3.connect(pack) as db:
        db.execute("UPDATE companies SET sha256=? WHERE id='1'", ("0" * 64,))
    with pytest.raises(ValueError, match="identity mismatch"):
        audit.audit_pack(pack, taxonomy, tmp_path)
