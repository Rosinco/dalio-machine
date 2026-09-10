"""Point-in-time assessment inputs come from verified immutable releases only."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock

import pytest
from sqlalchemy import event, select
from sqlalchemy.orm import Session

from dalio.assessments.evidence import load_evidence
from dalio.data_sources.company_country_macro import canonical, collect_bundle, digest, load_bundle
from dalio.data_sources.imf_datamapper import IMF_FUNDAMENTALS
from dalio.pipelines.fetch_company_country_macro import ingest_bundle
from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
    make_engine,
)
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseMeta,
    ingest_release_snapshot,
    make_partition_key,
)
from tests.test_company_country_macro import NOW, SPEC, fake_client, wb_page


def capture(tmp_path, *, when=NOW, value=31.0, countries=("BE",)):
    client = fake_client()
    original = client.get.side_effect
    def response(url, **kwargs):
        result = original(url, **kwargs)
        if "/country/" in url:
            payload = wb_page(value=value, total=2 * len(countries))
            if "DK" in countries:
                payload[1] += wb_page(value=value + 1, country="DNK")[1]
            result.content = json.dumps(payload).encode()
        return result
    client.get.side_effect = response
    return collect_bundle(artifact_dir=tmp_path, countries=countries, families=("wb",),
                          indicators=(SPEC.indicator,), client=client, clock=lambda: when)


def database(tmp_path):
    path = capture(tmp_path / "capture")
    engine = make_engine(tmp_path / "evidence.db")
    ingest_bundle(path, engine=engine)
    return engine, path


def imf_capture(tmp_path, *, spec=IMF_FUNDAMENTALS[0]):
    metadata = {"indicators": {spec.imf_code: {
        "dataset": "FM" if spec == IMF_FUNDAMENTALS[3] else "WEO",
        "source": "Official April 2026 vintage", "unit": "Percent",
        "last-modified": "2026-04-08 12:00:00"}}}
    values = {"values": {spec.imf_code: {"BEL": {"2025": 1.0, "2026": 2.0, "2031": 3.0}}}}
    client = Mock()
    def response(url, **kwargs):
        body = metadata if url.endswith("/indicators") else values
        return Mock(status_code=200, content=json.dumps(body).encode(), headers={},
                    url=url, history=[])
    client.get.side_effect = response
    return collect_bundle(artifact_dir=tmp_path, countries=("BE",), families=("imf",),
                          indicators=(spec.indicator,), client=client, clock=lambda: NOW)


def test_missing_database_is_not_created_and_has_all_361_explicit_gaps(tmp_path):
    path = tmp_path / "absent.db"
    result = load_evidence(make_engine(path), as_known_at=NOW)
    assert not path.exists()
    assert len(result["countries"]) == 19
    assert sum(len(country["gaps"]) for country in result["countries"]) == 361
    assert all(not country["series"] for country in result["countries"])
    assert {gap["reason"] for country in result["countries"] for gap in country["gaps"]} == {
        "missing_tables"}


def test_empty_existing_database_is_read_only_and_does_not_create_tables(tmp_path):
    engine = make_engine(tmp_path / "empty.db")
    with engine.connect() as connection:
        assert connection.exec_driver_sql("SELECT count(*) FROM sqlite_master").scalar_one() == 0
    statements = []
    event.listen(engine, "before_cursor_execute", lambda conn, cur, sql, params, ctx, many:
                 statements.append(sql))
    result = load_evidence(engine, as_known_at=NOW, countries=["GB", "UK"])
    assert len(result["countries"]) == 1
    assert result["countries"][0]["country"] == "UK"
    assert result["countries"][0]["listing_iso2"] == "GB"
    assert all(not sql.lstrip().upper().startswith(("CREATE", "INSERT", "UPDATE", "DELETE", "DROP"))
               for sql in statements)


@pytest.mark.parametrize("cutoff", [datetime(2026, 9, 10), "2026-09-10"])
def test_naive_or_non_datetime_cutoff_rejected_before_reading_database(tmp_path, cutoff):
    path = tmp_path / "absent.db"
    with pytest.raises(ValueError, match="timezone-aware"):
        load_evidence(make_engine(path), as_known_at=cutoff)
    assert not path.exists()


@pytest.mark.parametrize("countries", [[], ["CN"], ["invalid"]])
def test_country_selection_is_explicit_and_bounded(tmp_path, countries):
    with pytest.raises(ValueError):
        load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW, countries=countries)


def test_cutoff_excludes_not_yet_available_release(tmp_path):
    engine, _ = database(tmp_path)
    result = load_evidence(engine, as_known_at=NOW - timedelta(microseconds=1), countries=["BE"])
    assert not result["countries"][0]["series"]
    assert {gap["reason"] for gap in result["countries"][0]["gaps"]} == {"no_release_as_known_at"}


def test_latest_known_release_precedes_fact_selection_and_ignores_current_projection(tmp_path):
    engine, _ = database(tmp_path)
    later = capture(tmp_path / "later", when=NOW + timedelta(days=1), value=44.0)
    ingest_bundle(later, engine=engine)
    with Session(engine) as session:
        session.scalar(select(Observation)).value = 999
        session.commit()
    prior = load_evidence(engine, as_known_at=NOW, countries=["BE"])
    latest = load_evidence(engine, as_known_at=NOW + timedelta(days=1), countries=["BE"])
    assert prior["countries"][0]["series"][0]["observations"][0]["value"] == 31.0
    assert latest["countries"][0]["series"][0]["observations"][0]["value"] == 44.0
    assert prior["countries"][0]["series"][0]["release_id"] != latest["countries"][0]["series"][0]["release_id"]


def test_future_forecasts_retain_native_status_and_source(tmp_path):
    path = imf_capture(tmp_path / "imf")
    engine = make_engine(tmp_path / "imf.db")
    ingest_bundle(path, engine=engine)
    result = load_evidence(engine, as_known_at=NOW, countries=["BE"])
    series = result["countries"][0]["series"][0]
    assert series["family"] == "imf"
    assert series["source"] == "IMF_WEO"
    assert [row["year"] for row in series["observations"]] == [2025, 2026, 2031]
    assert series["observations"][-1]["status"] == "forecast_calendar_convention"
    assert series["observations"][-1]["source"] == "IMF_WEO_FCST"
    assert series["observations"][0]["status"] == "estimate_or_outturn"


def test_fiscal_monitor_is_kept_distinct_from_weo(tmp_path):
    path = imf_capture(tmp_path / "imf", spec=IMF_FUNDAMENTALS[3])
    engine = make_engine(tmp_path / "imf.db")
    ingest_bundle(path, engine=engine)
    series = load_evidence(engine, as_known_at=NOW, countries=["BE"])["countries"][0]["series"][0]
    assert series["source"] == "IMF_FISCAL_MONITOR"
    assert series["publisher_metadata"]["dataset"] == "FM"


def test_latest_legacy_contract_is_a_gap_without_falling_back_to_prior_valid_release(tmp_path):
    engine, path = database(tmp_path)
    frame = load_bundle(path)["ready"][0]["frame"].copy()
    with Session(engine) as session:
        ingest_release_snapshot(session, frame, ReleaseMeta(
            partition_key=make_partition_key("WORLD_BANK", SPEC.wb_code, "BE", SPEC.indicator),
            source_family="WORLD_BANK", available_at=NOW + timedelta(days=1),
            retrieved_at=NOW + timedelta(days=1),
            projection=ProjectionScope("BE", SPEC.indicator, ("WORLD_BANK",))))
    result = load_evidence(engine, as_known_at=NOW + timedelta(days=1), countries=["BE"])
    assert not result["countries"][0]["series"]
    gap = next(gap for gap in result["countries"][0]["gaps"] if gap["indicator"] == SPEC.indicator)
    assert gap["reason"] == "unverified_release_contract"


def test_corrupt_scalar_outside_assessment_period_fails_closed(tmp_path):
    engine, _ = database(tmp_path)
    with Session(engine) as session:
        release = session.scalar(select(DataRelease))
        from datetime import date
        session.add(ReleaseObservation(release_id=release.id, country="BE", indicator=SPEC.indicator,
            date=date(1980, 12, 31), value=999, source="WORLD_BANK", series_id=SPEC.wb_code,
            status="published_statistic"))
        session.commit()
    with pytest.raises(ValueError, match="integrity"):
        load_evidence(engine, as_known_at=NOW, countries=["BE"])


def test_tampered_artifact_fails_closed(tmp_path):
    engine, path = database(tmp_path)
    request = json.loads(path.read_bytes())["requests"][1]
    Path(request["path"]).write_text("{}")
    with pytest.raises(ValueError):
        load_evidence(engine, as_known_at=NOW, countries=["BE"])


def test_later_corrupt_capture_does_not_replace_an_earlier_known_cutoff(tmp_path):
    engine, _ = database(tmp_path)
    later = capture(tmp_path / "later", when=NOW + timedelta(days=1), value=44.0)
    ingest_bundle(later, engine=engine)
    request = json.loads(later.read_bytes())["requests"][1]
    Path(request["path"]).write_text("{}")
    prior = load_evidence(engine, as_known_at=NOW, countries=["BE"])
    assert prior["countries"][0]["series"][0]["observations"][0]["value"] == 31.0
    with pytest.raises(ValueError):
        load_evidence(engine, as_known_at=NOW + timedelta(days=1), countries=["BE"])


def test_valid_evidence_loading_leaves_database_and_artifacts_unchanged(tmp_path):
    engine, path = database(tmp_path)
    db_path = Path(engine.url.database)
    tracked = [db_path, path, *(Path(request["path"])
                for request in json.loads(path.read_bytes())["requests"])]
    before = {file: (digest(file.read_bytes()), file.stat().st_mtime_ns) for file in tracked}
    statements = []
    event.listen(engine, "before_cursor_execute", lambda conn, cur, sql, params, ctx, many:
                 statements.append(sql))
    load_evidence(engine, as_known_at=NOW, countries=["BE"])
    assert before == {file: (digest(file.read_bytes()), file.stat().st_mtime_ns) for file in tracked}
    assert all("FROM observations" not in sql for sql in statements)
    assert all(not sql.lstrip().upper().startswith(("CREATE", "INSERT", "UPDATE", "DELETE", "DROP"))
               for sql in statements)


def test_shared_bundle_verified_once_and_all_retained_paths_are_protected(tmp_path, monkeypatch):
    import dalio.pipelines.fetch_company_country_macro as acquisition
    path = capture(tmp_path / "both", countries=("BE", "DK"))
    engine = make_engine(tmp_path / "both.db")
    ingest_bundle(path, engine=engine)
    loader = Mock(wraps=acquisition.load_bundle)
    monkeypatch.setattr(acquisition, "load_bundle", loader)
    result = load_evidence(engine, as_known_at=NOW, countries=["BE", "DK"])
    assert loader.call_count == 1
    with Session(engine) as session:
        paths = {str(Path(artifact.artifact_path).resolve())
                 for artifact in session.scalars(select(DataReleaseArtifact))}
    assert paths <= set(result["protected_artifact_paths"])
    assert str(path.resolve()) in result["protected_artifact_paths"]
    assert all(request["path"] in result["protected_artifact_paths"]
               for request in json.loads(path.read_bytes())["requests"])


def test_json_serializable_digest_is_stable_and_normalizes_timezone(tmp_path):
    engine, _ = database(tmp_path)
    first = load_evidence(engine, as_known_at=NOW, countries=["BE"])
    equivalent = NOW.astimezone(timezone(timedelta(hours=2)))
    second = load_evidence(engine, as_known_at=equivalent, countries=["BE"])
    assert first == second
    assert json.loads(json.dumps(first)) == first
    body = {key: value for key, value in first.items() if key != "evidence_digest"}
    assert digest(canonical(body)) == first["evidence_digest"]
