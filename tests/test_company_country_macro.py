"""Official-source country acquisition must stay outside the scoring universe."""

import json
from datetime import UTC, datetime
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from dalio.countries import COUNTRIES, RANKING_POPULATION
from dalio.data_sources.company_country_macro import (
    collect_bundle,
    company_country_manifest,
    load_bundle,
    parse_imf,
    parse_worldbank,
)
from dalio.data_sources.imf_datamapper import IMF_FUNDAMENTALS
from dalio.data_sources.worldbank import WB_FUNDAMENTALS
from dalio.pipelines.fetch_company_country_macro import ingest_bundle
from dalio.storage.db import DataRelease, Observation, make_engine

NOW = datetime(2026, 9, 10, 19, tzinfo=UTC)
SPEC = WB_FUNDAMENTALS[1]


def wb_page(*, page=1, pages=1, total=2, value=31.0, country="BEL"):
    return [{"page": page, "pages": pages, "per_page": 1000, "total": total,
             "sourceid": "2", "lastupdated": "2026-07-01"}, [
        {"indicator": {"id": SPEC.wb_code}, "countryiso3code": country,
         "date": "2024", "value": value, "obs_status": "", "unit": ""},
        {"indicator": {"id": SPEC.wb_code}, "countryiso3code": country,
         "date": "2025", "value": None, "obs_status": "", "unit": ""},
    ]]


def metadata():
    return [{"page": 1, "pages": 1, "total": 1}, [{
        "id": SPEC.wb_code, "name": "Old age dependency ratio",
        "source": {"id": "2", "value": "World Development Indicators"},
        "sourceNote": "Older dependents per working-age population.",
        "sourceOrganization": "United Nations Population Division", "unit": "%",
    }]]


def fake_client():
    client = Mock()
    def response(url, **kwargs):
        payload = wb_page() if "/country/" in url else metadata()
        result = Mock(status_code=200, content=json.dumps(payload).encode(), headers={})
        result.raise_for_status.return_value = None
        return result
    client.get.side_effect = response
    return client


def bundle(tmp_path):
    return collect_bundle(artifact_dir=tmp_path / "artifacts", countries=("BE",),
                          families=("wb",), indicators=(SPEC.indicator,),
                          client=fake_client(), retrieved_at=NOW)


def test_country_manifest_does_not_change_ranking_population():
    before = COUNTRIES, RANKING_POPULATION
    manifest = company_country_manifest()
    rows = manifest["countries"]
    assert len(rows) == 19
    assert sum(row["listing_count"] for row in rows) == 19140
    assert next(row for row in rows if row["listing_iso2"] == "GB")["country"] == "UK"
    assert manifest["source"]["sha256"] == (
        "cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff")
    assert before == (COUNTRIES, RANKING_POPULATION)


def test_wb_preserves_missing_zero_and_native_status():
    frame, missing = parse_worldbank([wb_page(value=0)], SPEC, {"BEL": "BE"}, 2026)
    assert list(frame.value) == [0]
    assert list(frame.country) == ["BE"]
    assert missing["BE"]["null_years"] == [2025]
    assert missing["BE"]["not_returned_years"][-1] == 2026


@pytest.mark.parametrize("mutation", ["message", "wrong_series", "wrong_country", "nan",
                                      "duplicate", "missing_page", "wrong_total"])
def test_worldbank_rejects_source_errors_and_incomplete_pagination(mutation):
    pages = [wb_page()]
    if mutation == "message":
        pages = [[{"message": [{"id": "120", "value": "Invalid value"}]}]]
    elif mutation == "wrong_series":
        pages[0][1][0]["indicator"]["id"] = "OTHER"
    elif mutation == "wrong_country":
        pages[0][1][0]["countryiso3code"] = "USA"
    elif mutation == "nan":
        pages[0][1][0]["value"] = float("nan")
    elif mutation == "duplicate":
        pages[0][1][1]["date"] = "2024"
    elif mutation == "missing_page":
        pages[0][0]["pages"] = 2
    else:
        pages[0][0]["total"] = 3
    with pytest.raises(ValueError):
        parse_worldbank(pages, SPEC, {"BEL": "BE"}, 2026)


def test_imf_separates_fiscal_monitor_and_estimate_status():
    spec = IMF_FUNDAMENTALS[3]
    values = {"values": {spec.imf_code: {"BEL": {"2025": -1, "2026": 0, "2027": None}}}}
    meta = {"dataset": "FM", "source": "Fiscal Monitor (April 2026)",
            "unit": "Percent of GDP", "label": "Primary net lending/borrowing"}
    frame, missing = parse_imf(values, meta, spec, {"BEL": "BE"}, 2026)
    assert list(frame.source) == ["IMF_FISCAL_MONITOR", "IMF_FISCAL_MONITOR_FCST"]
    assert list(frame.status) == ["estimate_or_outturn", "forecast_calendar_convention"]
    assert missing["BE"]["null_years"] == [2027]
    with pytest.raises(ValueError, match="dataset"):
        parse_imf(values, {**meta, "dataset": "WEO"}, spec, {"BEL": "BE"}, 2026)


def test_bundle_roundtrip_and_atomic_ingest(tmp_path):
    path = bundle(tmp_path)
    prepared = load_bundle(path)
    assert prepared["summary"]["source_error_partitions"] == 0
    assert prepared["summary"]["ready_partitions"] == 1
    engine = make_engine(tmp_path / "staging.db")
    result = ingest_bundle(path, engine=engine)
    assert result["observation_count"] == 1
    assert ingest_bundle(path, engine=engine)["created_releases"] == 0
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Observation)) == 1
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1


def test_tampered_artifact_rejected_before_database_created(tmp_path):
    path = bundle(tmp_path)
    data = json.loads(path.read_text())
    artifact = data["requests"][0]
    from pathlib import Path
    Path(artifact["path"]).write_text("{}")
    database = tmp_path / "untouched.db"
    with pytest.raises(ValueError, match="hash"):
        ingest_bundle(path, engine=make_engine(database))
    assert not database.exists()


def test_source_error_is_not_reported_as_absent_country_data(tmp_path):
    client = fake_client()
    client.get.side_effect = RuntimeError("publisher unavailable")
    path = collect_bundle(artifact_dir=tmp_path, countries=("BE",), families=("wb",),
                          indicators=(SPEC.indicator,), client=client, retrieved_at=NOW)
    result = load_bundle(path)
    assert result["summary"]["source_error_partitions"] == 1
    assert result["summary"]["missing_partitions"] == 0
    assert result["summary"]["ready_partitions"] == 0


def test_contraction_aborts_without_deleting_prior_history(tmp_path):
    path = bundle(tmp_path)
    engine = make_engine(tmp_path / "existing.db")
    ingest_bundle(path, engine=engine)
    with Session(engine) as session:
        from datetime import date
        session.add(Observation(country="BE", indicator=SPEC.indicator,
            source="WORLD_BANK", series_id=SPEC.wb_code, date=date(2023, 12, 31), value=30))
        session.commit()
    with pytest.raises(ValueError, match="contraction"):
        ingest_bundle(path, engine=engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Observation)) == 2
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1


def test_late_partition_failure_rolls_back_entire_batch(tmp_path, monkeypatch):
    client = fake_client()
    get = client.get.side_effect
    def both_countries(url, **kwargs):
        result = get(url, **kwargs)
        if "/country/" in url:
            data = wb_page(total=4)
            data[1] += wb_page(country="DNK")[1]
            result.content = json.dumps(data).encode()
        return result
    client.get.side_effect = both_countries
    path = collect_bundle(artifact_dir=tmp_path / "artifacts", countries=("BE", "DK"),
                          families=("wb",), indicators=(SPEC.indicator,),
                          client=client, retrieved_at=NOW)
    import dalio.pipelines.fetch_company_country_macro as pipeline
    original = pipeline.ingest_release_snapshot
    calls = 0
    def failing(session, frame, meta):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("Late second-country failure")
        return original(session, frame, meta)
    monkeypatch.setattr(pipeline, "ingest_release_snapshot", failing)
    engine = make_engine(tmp_path / "atomic.db")
    with pytest.raises(RuntimeError, match="Late"):
        ingest_bundle(path, engine=engine)
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(Observation)) == 0
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0
