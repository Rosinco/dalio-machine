"""Known-at supplement selection and honest legacy scalar inventory."""

import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.monitoring import acquisition, evidence
from dalio.storage.db import DataRelease, ReleaseObservation, init_db, make_engine
from dalio.storage.releases import ReleaseMeta, ingest_release_snapshot, make_partition_key
from tests.test_sweden_monitoring_acquisition import NOW, Client, scb_contract  # noqa: F401


def captured(root, when, **kwargs):
    return acquisition.collect_bundle(artifact_root=root, client=Client(**kwargs),
                                      clock=lambda: when, pacing_seconds=0)


def legacy(engine, *, when=NOW, value=1.75):
    init_db(engine)
    frame = pd.DataFrame([dict(country="SE", indicator="policy_rate", source="RIKSBANK_SWEA",
        series_id="SECBREPOEFF", date=date(2026, 9, 9), value=value)])
    with Session(engine) as session:
        return ingest_release_snapshot(session, frame, ReleaseMeta(
            partition_key=make_partition_key("RIKSBANK_SWEA", "SECBREPOEFF", "SE", "policy_rate"),
            source_family="RIKSBANK_SWEA", available_at=when, retrieved_at=when,
            source_url=f"https://api.riksbank.se/swea/v1/Observations/SECBREPOEFF/2005-01-01/{when.date()}"),
            replace_current=False)


def test_missing_database_and_bundle_produce_five_gaps_without_creating_database(tmp_path):
    path = tmp_path / "missing.db"
    result = evidence.load_evidence(make_engine(path), as_known_at=NOW)
    assert not path.exists()
    assert result["series"] == [] and result["national_debt_context"] == []
    assert {gap["indicator"] for gap in result["gaps"]} == set(acquisition.INDICATORS)
    assert result["country"] == "SE" and result["as_known_at"] == NOW.isoformat()
    assert result["evidence_digest"] == acquisition.digest(acquisition.canonical(
        {key: value for key, value in result.items() if key != "evidence_digest"}))


def test_latest_batch_failure_never_resurrects_older_series(tmp_path):
    old = captured(tmp_path / "old", NOW - timedelta(hours=1))
    new = captured(tmp_path / "new", NOW, fail="SECBREPOEFF")
    result = evidence.load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW,
                                    bundle_paths=[old, new])
    assert "policy_rate" not in {series["indicator"] for series in result["series"]}
    assert any(gap["indicator"] == "policy_rate" and gap["reason"] == "source_error" for gap in result["gaps"])
    assert all(series["bundle_sha256"] == new.stem for series in result["series"])
    assert str(old) not in result["protected_artifact_paths"]


def test_exact_clock_excludes_future_corrupt_raw_before_validation(tmp_path):
    old = captured(tmp_path / "old", NOW - timedelta(seconds=1))
    new = captured(tmp_path / "future", NOW + timedelta(seconds=1))
    request = json.loads(new.read_bytes())["signals"][0]["requests"][0]
    Path(request["path"]).write_bytes(b"future corruption")
    engine = make_engine(tmp_path / "absent.db")
    result = evidence.load_evidence(engine, as_known_at=NOW, bundle_paths=[new, old])
    assert len(result["series"]) == 5
    assert all(series["bundle_sha256"] == old.stem for series in result["series"])
    with pytest.raises(ValueError, match="hash"):
        evidence.load_evidence(engine, as_known_at=NOW + timedelta(seconds=1), bundle_paths=[new, old])


def test_same_completion_clock_different_bundles_is_ambiguous(tmp_path):
    first = captured(tmp_path / "first", NOW)
    second = captured(tmp_path / "second", NOW, fail="SECBREPOEFF")
    with pytest.raises(ValueError, match="completion clock"):
        evidence.load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW,
                               bundle_paths=[first, second])


def test_reader_rejects_bundle_symlink_before_selection(tmp_path):
    path = captured(tmp_path / "capture", NOW)
    alias = tmp_path / path.name
    alias.symlink_to(path)
    with pytest.raises(ValueError, match="symlink"):
        evidence.load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW, bundle_paths=[alias])


def test_legacy_hash_verified_but_no_values_exposed_or_cache_claimed(tmp_path):
    path = tmp_path / "legacy.db"
    engine = make_engine(path)
    old = legacy(engine, when=NOW - timedelta(hours=1))
    legacy(engine, when=NOW + timedelta(hours=1), value=9)
    before = (path.read_bytes(), path.stat().st_mtime_ns)
    result = evidence.load_evidence(engine, as_known_at=NOW)
    assert result["series"] == []
    row = next(row for row in result["legacy_inventory"] if row["indicator"] == "policy_rate")
    assert row["release_id"] == old.release_id
    assert row["proof"] == "scalar_rows_verified_raw_response_not_bound"
    assert "observations" not in row and "value" not in row
    assert row["row_count"] == 1
    assert (path.read_bytes(), path.stat().st_mtime_ns) == before


def test_corrupt_latest_legacy_full_rows_fail_closed_before_inventory_filter(tmp_path):
    engine = make_engine(tmp_path / "legacy.db")
    legacy(engine)
    with Session(engine) as session:
        release = session.scalar(select(DataRelease))
        session.add(ReleaseObservation(release_id=release.id, country="SE", indicator="unrelated",
            source="RIKSBANK_SWEA", series_id="SECBREPOEFF", date=date(2026, 9, 9), value=99,
            status="observed"))
        session.commit()
    with pytest.raises(ValueError, match="row|hash"):
        evidence.load_evidence(engine, as_known_at=NOW)


def test_native_loader_exact_cutoff_and_protected_paths_preserved(tmp_path, monkeypatch):
    calls = []
    def native(engine, *, as_known_at, countries):
        calls.append((as_known_at, countries))
        return {"contexts": {"SE": [{"metric": "native_test_fact", "value": 7}]},
                "protected_artifact_paths": [str(tmp_path / "native.pdf")]}
    monkeypatch.setattr(evidence, "load_national_debt_context", native)
    result = evidence.load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW)
    assert calls == [(NOW, ["SE"])]
    assert result["national_debt_context"] == [{"metric": "native_test_fact", "value": 7}]
    assert str(tmp_path / "native.pdf") in result["protected_artifact_paths"]


def test_naive_cutoff_rejected(tmp_path):
    with pytest.raises(ValueError, match="timezone"):
        evidence.load_evidence(make_engine(tmp_path / "absent.db"), as_known_at=NOW.replace(tzinfo=None))
