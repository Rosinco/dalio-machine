"""Offline end-to-end contracts for the bounded refinancing history package."""

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from sqlalchemy import func, inspect, select, text
from sqlalchemy.orm import Session

from dalio.data_sources.ecb_refinancing import (
    ECB_GOV_REDEMPTIONS_1_12M,
    ECB_REFINANCING_SERIES,
    EcbRefinancingSource,
)
from dalio.data_sources.eurostat_refinancing import (
    EUROSTAT_REFINANCING_SERIES,
    EurostatRefinancingSource,
    build_eurostat_refinancing_url,
)
from dalio.pipelines.fetch_sovereign_refinancing import prepare_batch, run_pipeline
from dalio.storage import refinancing
from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
    init_db,
    make_engine,
)
from dalio.storage.inventory import build_observatory_inventory
from dalio.storage.refinancing import audit_refinancing, ingest_refinancing_batch
from tests.test_ecb_refinancing import _complete_observations, _csv_text, _response
from tests.test_eurostat_refinancing import _body, _payload

RUN_AT = datetime(2026, 9, 10, 12, tzinfo=UTC)


@pytest.fixture
def inputs(tmp_path):
    bodies = {}
    for spec in EUROSTAT_REFINANCING_SERIES:
        years = tuple(range(spec.verified_start_year, spec.verified_through_year + 1))
        values = {str(i): float(i + 1) for i in range(len(years))}
        # One genuine zero and one missing cell exercise the provenance contract.
        values["0"] = 0.0
        values.pop("1")
        bodies[build_eurostat_refinancing_url(spec)] = _body(
            _payload(spec, years=years, values=values, statuses={"1": ":", "2": "p"})
        )
    for spec in ECB_REFINANCING_SERIES:
        bodies[spec.url] = _csv_text(
            spec, _complete_observations(spec, overrides={"2010-01": (".", "L")})
        )
    client = MagicMock()
    client.get.side_effect = lambda url, **kwargs: _response(bodies[url])
    return {
        "eurostat_source": EurostatRefinancingSource(
            client=client,
            cache_dir=tmp_path / "cache/eurostat",
            artifact_dir=tmp_path / "evidence/eurostat",
            attempts=1,
        ),
        "ecb_source": EcbRefinancingSource(
            client=client,
            cache_dir=tmp_path / "cache/ecb",
            artifact_dir=tmp_path / "evidence/ecb",
        ),
        "artifact_dir": tmp_path / "evidence/catalogue",
        "retrieved_at": RUN_AT,
        "use_cache": False,
    }


def _counts(engine):
    with Session(engine) as session:
        return tuple(
            session.scalar(select(func.count()).select_from(model))
            for model in (
                Observation,
                DataRelease,
                ReleaseObservation,
                DataReleaseArtifact,
            )
        )


def test_all_31_releases_round_trip_long_native_ids_and_evidence(inputs, tmp_path):
    engine = make_engine(tmp_path / "database.sqlite")
    init_db(engine)
    with Session(engine) as session:
        session.add(
            Observation(
                country="US",
                indicator="existing",
                date=date(2020, 1, 1),
                value=42,
                source="OTHER",
                series_id="untouched",
            )
        )
        session.commit()
    batch = prepare_batch(**inputs)
    result = ingest_refinancing_batch(batch, engine=engine)
    count = sum(len(item.frame) for item in batch)
    assert result["created_releases"] == 31
    assert _counts(engine) == (count + 1, 31, count, 31 * 4)
    long_id = ECB_GOV_REDEMPTIONS_1_12M.native_series_id
    assert len(long_id) == 65
    with Session(engine) as session:
        for model in (Observation, ReleaseObservation):
            assert (
                session.scalar(select(model.series_id).where(model.series_id == long_id)) == long_id
            )
    audit = audit_refinancing(engine, as_of=RUN_AT.date())
    assert audit["ready_partitions"] == 31
    assert audit["expected_partitions"] == 48
    assert audit["harmonized_expected"] == 31
    assert audit["national_native_planned"] == 17
    assert audit["observation_count"] == count
    assert sum(row["missing_periods"] for row in audit["partitions"] if row["ready"]) == 31
    assert all(row["issues"] == [] for row in audit["partitions"] if row["ready"])
    repeat = prepare_batch(**(inputs | {"retrieved_at": RUN_AT + timedelta(hours=1)}))
    assert ingest_refinancing_batch(repeat, engine=engine)["created_releases"] == 0
    assert _counts(engine) == (count + 1, 31, count, 31 * 4)


def test_fetch_failure_does_not_initialize_or_mutate_database(inputs, tmp_path):
    engine = make_engine(tmp_path / "never-created.sqlite")
    inputs["ecb_source"].fetch = MagicMock(side_effect=ValueError("publisher unavailable"))
    with pytest.raises(ValueError, match="publisher unavailable"):
        run_pipeline(engine=engine, **inputs)
    assert not (tmp_path / "never-created.sqlite").exists()


@pytest.mark.parametrize(
    "damage", ["frame", "source", "native", "missing", "catalogue", "clock", "subset"]
)
def test_preflight_rejects_mutated_batch_before_db_writes(inputs, tmp_path, damage):
    batch = prepare_batch(**inputs)
    first = batch[0]
    if damage == "frame":
        first.frame.loc[0, "value"] = 99999
    elif damage == "clock":
        first = replace(first, meta=replace(first.meta, published_at=RUN_AT + timedelta(days=1)))
        batch = (first, *batch[1:])
    elif damage == "subset":
        batch = batch[:-1]
    else:
        role = {
            "source": "source_response",
            "native": "native_payload",
            "missing": "missingness_ledger",
            "catalogue": "catalogue_manifest",
        }[damage]
        artifact = next(item for item in first.meta.artifacts if item.role == role)
        Path(artifact.artifact_path).write_bytes(b"tampered")
    engine = make_engine(tmp_path / "rejected.sqlite")
    with pytest.raises(ValueError):
        ingest_refinancing_batch(batch, engine=engine)
    assert not (tmp_path / "rejected.sqlite").exists()


def test_late_partition_failure_rolls_back_entire_batch(inputs, tmp_path, monkeypatch):
    batch = prepare_batch(**inputs)
    engine = make_engine(tmp_path / "atomic.sqlite")
    init_db(engine)
    original = refinancing.ingest_release_snapshot
    calls = 0

    def fail_last(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 31:
            raise RuntimeError("last partition failed after inner commit")
        return result

    monkeypatch.setattr(refinancing, "ingest_release_snapshot", fail_last)
    with pytest.raises(RuntimeError, match="last partition failed"):
        ingest_refinancing_batch(batch, engine=engine)
    assert calls == 31
    assert _counts(engine) == (0, 0, 0, 0)


def test_audit_detects_current_projection_and_artifact_tampering(inputs, tmp_path):
    batch = prepare_batch(**inputs)
    engine = make_engine(tmp_path / "audit.sqlite")
    ingest_refinancing_batch(batch, engine=engine)
    with engine.begin() as connection:
        connection.execute(text("UPDATE observations SET value = 999 WHERE id = 1"))
    broken_source = next(a for a in batch[-1].meta.artifacts if a.role == "source_response")
    Path(broken_source.artifact_path).write_bytes(b"changed after ingest")
    audit = audit_refinancing(engine, as_of=RUN_AT.date())
    assert audit["ready_partitions"] == 29
    assert any("projection" in " ".join(row["issues"]) for row in audit["partitions"])
    assert any("hash" in " ".join(row["issues"]) for row in audit["partitions"])


def test_inventory_empty_database_preserves_fixed_denominator_without_ddl(tmp_path):
    engine = make_engine(tmp_path / "empty.sqlite")
    inventory = build_observatory_inventory(engine, as_of=RUN_AT.date())
    audit = inventory["sovereign_refinancing"]
    assert audit["ready_partitions"] == 0
    assert audit["expected_partitions"] == 48
    assert audit["national_native_planned"] == 17
    assert inventory["readiness"]["sovereign_refinancing"] == "empty"
    assert inspect(engine).get_table_names() == []


def test_legacy_sqlite_varchar_64_preserves_full_ecb_id(inputs, tmp_path):
    engine = make_engine(tmp_path / "legacy.sqlite")
    with engine.begin() as connection:
        connection.exec_driver_sql("""CREATE TABLE observations (
            id INTEGER PRIMARY KEY, country VARCHAR(8) NOT NULL,
            indicator VARCHAR(64) NOT NULL, date DATE NOT NULL, value FLOAT NOT NULL,
            source VARCHAR(32) NOT NULL, series_id VARCHAR(64) NOT NULL,
            fetched_at DATETIME NOT NULL,
            UNIQUE(country, indicator, date, source))""")
    ingest_refinancing_batch(prepare_batch(**inputs), engine=engine)
    with Session(engine) as session:
        assert session.scalar(select(func.max(func.length(Observation.series_id)))) == 65
    assert audit_refinancing(engine, as_of=RUN_AT.date())["ready_partitions"] == 31


def _replace_eurostat_response(inputs, batch, mutate):
    first = batch[0]
    artifact = next(a for a in first.meta.artifacts if a.role == "source_response")
    payload = json.loads(Path(artifact.artifact_path).read_bytes())
    mutate(payload)
    client = inputs["eurostat_source"]._client
    original = client.get.side_effect
    target = first.binding.partition.source_url
    client.get.side_effect = lambda url, **kwargs: (
        _response(_body(payload)) if url == target else original(url, **kwargs)
    )


def test_revision_appends_only_changed_partition_and_keeps_old_vintage(inputs, tmp_path):
    batch = prepare_batch(**inputs)
    engine = make_engine(tmp_path / "revision.sqlite")
    ingest_refinancing_batch(batch, engine=engine)
    _replace_eurostat_response(inputs, batch, lambda payload: payload["value"].update({"0": 12.5}))
    revised = prepare_batch(**(inputs | {"retrieved_at": RUN_AT + timedelta(hours=1)}))
    assert ingest_refinancing_batch(revised, engine=engine)["created_releases"] == 1
    with Session(engine) as session:
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 32
        assert (
            session.scalar(
                select(ReleaseObservation.value).where(
                    ReleaseObservation.release_id == 1,
                    ReleaseObservation.date == batch[0].frame.iloc[0]["date"],
                )
            )
            == 0.0
        )
    assert audit_refinancing(engine, as_of=RUN_AT.date())["ready_partitions"] == 31


def test_valid_publisher_removal_fails_closed_without_partial_refresh(inputs, tmp_path):
    batch = prepare_batch(**inputs)
    engine = make_engine(tmp_path / "contraction.sqlite")
    ingest_refinancing_batch(batch, engine=engine)
    before = _counts(engine)
    _replace_eurostat_response(inputs, batch, lambda payload: payload["value"].pop("2"))
    contracted = prepare_batch(**(inputs | {"retrieved_at": RUN_AT + timedelta(hours=1)}))
    with pytest.raises(ValueError, match="History contraction"):
        ingest_refinancing_batch(contracted, engine=engine)
    assert _counts(engine) == before
    assert audit_refinancing(engine, as_of=RUN_AT.date())["ready_partitions"] == 31


def test_future_publisher_timestamp_is_rejected_before_ingestion(inputs):
    batch = prepare_batch(**inputs)
    _replace_eurostat_response(
        inputs, batch, lambda payload: payload.update(updated="2027-01-01T00:00:00Z")
    )
    with pytest.raises(ValueError, match="Future source update"):
        prepare_batch(**inputs)


def test_verified_staging_batch_can_be_promoted_without_network(inputs, tmp_path):
    batch = prepare_batch(**inputs)
    staging = make_engine(tmp_path / "staging.sqlite")
    ingest_refinancing_batch(batch, engine=staging)
    promoted = refinancing.load_stored_refinancing_batch(staging, as_of=RUN_AT.date())
    assert [item.meta for item in promoted] == [item.meta for item in batch]
    live = make_engine(tmp_path / "live.sqlite")
    ingest_refinancing_batch(promoted, engine=live)
    assert audit_refinancing(live, as_of=RUN_AT.date())["ready_partitions"] == 31
    assert _counts(staging) == _counts(live)
    with staging.begin() as connection:
        connection.execute(text("UPDATE observations SET value = -1 WHERE id = 1"))
    with pytest.raises(ValueError, match="31.*verified"):
        refinancing.load_stored_refinancing_batch(staging, as_of=RUN_AT.date())


def test_brief_uses_common_periods_and_keeps_comparator_and_gaps_separate(inputs, tmp_path):
    from dalio.pipelines.build_refinancing_brief import build_brief, render_markdown

    batch = prepare_batch(**inputs)
    engine = make_engine(tmp_path / "brief.sqlite")
    ingest_refinancing_batch(batch, engine=engine)
    brief = build_brief(engine, as_of=RUN_AT.date())
    assert brief["common_year"] == 2025
    assert [row["country"] for row in brief["countries"]] == ["DE", "FR", "IT", "ES", "SE"]
    assert all(row["due_le1y_share_pct"] == 100.0 for row in brief["countries"])
    assert brief["countries"][-1]["variable_rate_pct_gdp"] is None
    assert len(brief["euro_area_comparison"]) == 2
    assert len(brief["input_releases"]) == 31
    assert "2025" in render_markdown(brief)
    assert "17 national" in render_markdown(brief)
    assert "Sweden" in render_markdown(brief)
    assert "risk_score" not in brief
    with engine.begin() as connection:
        connection.execute(text("UPDATE observations SET value = -1 WHERE id = 1"))
    with pytest.raises(ValueError, match="31.*verified"):
        build_brief(engine, as_of=RUN_AT.date())
