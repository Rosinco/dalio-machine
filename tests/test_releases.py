"""Point-in-time release ledger: immutable snapshots and honest as-of queries."""

import hashlib
import json
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime

import pandas as pd
import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
    init_db,
    make_engine,
    release_recurrence_migration_backup_path,
)
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseArtifactMeta,
    ReleaseEventConflictError,
    ReleaseMeta,
    bootstrap_current_observations,
    ingest_release_snapshot,
    load_vintage_panel,
    release_history,
)


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _frame(
    values: list[tuple[date, float]],
    *,
    country: str = "US",
    indicator: str = "policy_rate",
    source: str = "FRED",
    series_id: str = "DFF",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": country,
                "indicator": indicator,
                "date": period,
                "value": value,
                "source": source,
                "series_id": series_id,
            }
            for period, value in values
        ]
    )


def _artifact_meta(
    tmp_path,
    *,
    role: str = "source_response",
    content: bytes = b"raw source response",
    native_payload: bytes = b"canonical native payload",
    provenance: dict[str, object] | None = None,
) -> ReleaseArtifactMeta:
    artifact_sha256 = hashlib.sha256(content).hexdigest()
    artifact_path = tmp_path / f"{role}-{artifact_sha256}.bin"
    artifact_path.write_bytes(content)
    provenance_json = json.dumps(
        provenance or {"records": []},
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return ReleaseArtifactMeta(
        role=role,
        artifact_sha256=artifact_sha256,
        artifact_path=artifact_path,
        native_payload_sha256=hashlib.sha256(native_payload).hexdigest(),
        missing_provenance_sha256=hashlib.sha256(provenance_json.encode("utf-8")).hexdigest(),
        provenance_json=provenance_json,
    )


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "releases.db")
    init_db(engine)
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


def _meta(
    partition: str,
    available_at: datetime,
    *,
    country: str = "US",
    indicator: str = "policy_rate",
    source: str = "FRED",
    retrieved_at: datetime | None = None,
    published_at: datetime | None = None,
    vintage_label: str | None = None,
    source_url: str | None = None,
    projection_sources: tuple[str, ...] | None = None,
    artifacts: tuple[ReleaseArtifactMeta, ...] = (),
) -> ReleaseMeta:
    return ReleaseMeta(
        partition_key=partition,
        source_family=source,
        published_at=published_at,
        available_at=available_at,
        retrieved_at=retrieved_at or available_at,
        vintage_label=vintage_label,
        source_url=source_url,
        projection=ProjectionScope(
            country=country,
            indicator=indicator,
            sources=projection_sources or (source,),
        ),
        artifacts=artifacts,
    )


def test_revisions_and_omissions_are_point_in_time_honest(session_factory):
    first = _frame(
        [
            (date(2025, 12, 1), 4.00),
            (date(2026, 1, 1), 4.10),
        ]
    )
    second = _frame([(date(2025, 12, 1), 4.25)])

    with session_factory() as session:
        one = ingest_release_snapshot(
            session,
            first,
            _meta("fred:DFF:US", _at(2026, 1, 10)),
        )
        two = ingest_release_snapshot(
            session,
            second,
            _meta("fred:DFF:US", _at(2026, 2, 10)),
        )
        assert one.created and one.projected and one.row_count == 2
        assert two.created and two.projected and two.row_count == 1

    with session_factory() as session:
        before = load_vintage_panel(session, _at(2026, 2, 9))
        after = load_vintage_panel(session, _at(2026, 2, 10))
        current = session.execute(select(Observation).order_by(Observation.date)).scalars().all()

    assert before[["date", "value"]].to_records(index=False).tolist() == [
        (date(2025, 12, 1), 4.00),
        (date(2026, 1, 1), 4.10),
    ]
    assert after[["date", "value"]].to_records(index=False).tolist() == [
        (date(2025, 12, 1), 4.25),
    ]
    assert [(row.date, row.value) for row in current] == [(date(2025, 12, 1), 4.25)]


@pytest.mark.parametrize(
    ("model", "attribute", "replacement"),
    [
        (DataRelease, "partition_key", "rewritten:partition"),
        (DataReleaseArtifact, "artifact_path", "/tmp/rewritten-artifact"),
        (ReleaseObservation, "value", 999.0),
    ],
)
def test_release_ledger_rows_are_immutable_through_the_orm(
    session_factory, tmp_path, model, attribute, replacement
):
    artifact = _artifact_meta(tmp_path)
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 10), artifacts=(artifact,)),
        )

    with session_factory() as session:
        row = session.scalar(select(model))
        setattr(row, attribute, replacement)
        with pytest.raises(ValueError, match="immutable"):
            session.commit()

    with session_factory() as session:
        row = session.scalar(select(model))
        session.delete(row)
        with pytest.raises(ValueError, match="immutable"):
            session.commit()


@pytest.mark.parametrize(
    "table_name",
    ["data_releases", "data_release_artifacts", "release_observations"],
)
@pytest.mark.parametrize("operation", ["update", "delete"])
def test_release_ledger_sql_triggers_reject_mutation(
    session_factory, tmp_path, table_name, operation
):
    artifact = _artifact_meta(tmp_path)
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 10), artifacts=(artifact,)),
        )

    statement = (
        f"UPDATE {table_name} SET id = id WHERE id = 1"
        if operation == "update"
        else f"DELETE FROM {table_name} WHERE id = 1"
    )
    with session_factory() as session, pytest.raises(IntegrityError, match="immutable"):
        session.execute(text(statement))
        session.commit()


def test_release_records_a_verified_three_role_artifact_manifest(session_factory, tmp_path):
    provenance = {
        "cadence_policy": "official_periods_only",
        "native_series_id": "OFR.FNYR-A",
        "records": [{"period": "2026-01-02", "status": "missing"}],
    }
    native_payload = b'[{"date":"2026-01-02","value":null}]'
    provenance_json = json.dumps(
        provenance,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    artifacts = (
        _artifact_meta(
            tmp_path,
            role="source_response",
            content=b'{"data":{"timeseries":[]}}',
            native_payload=native_payload,
            provenance=provenance,
        ),
        _artifact_meta(
            tmp_path,
            role="native_series_payload",
            content=native_payload,
            native_payload=native_payload,
            provenance=provenance,
        ),
        _artifact_meta(
            tmp_path,
            role="missingness_ledger",
            content=provenance_json.encode("utf-8"),
            native_payload=native_payload,
            provenance=provenance,
        ),
    )

    with session_factory() as session:
        result = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta(
                "ofr:FNYR-A:US",
                _at(2026, 1, 10),
                artifacts=tuple(reversed(artifacts)),
            ),
        )
        rows = (
            session.execute(select(DataReleaseArtifact).order_by(DataReleaseArtifact.role))
            .scalars()
            .all()
        )

    assert result.created is True
    assert {row.release_id for row in rows} == {result.release_id}
    assert [row.role for row in rows] == [
        "missingness_ledger",
        "native_series_payload",
        "source_response",
    ]
    expected = {artifact.role: artifact for artifact in artifacts}
    for row in rows:
        artifact = expected[row.role]
        assert row.artifact_sha256 == artifact.artifact_sha256
        assert row.artifact_path == str(artifact.artifact_path.resolve())
        assert row.native_payload_sha256 == artifact.native_payload_sha256
        assert row.missing_provenance_sha256 == artifact.missing_provenance_sha256
        assert row.provenance_json == provenance_json


def test_release_artifact_schema_enforces_fk_indexes_hashes_and_unique_role(
    session_factory, tmp_path
):
    artifact = _artifact_meta(tmp_path)
    with session_factory() as session:
        result = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 10), artifacts=(artifact,)),
        )
        engine = session.get_bind()

    with engine.connect() as connection:
        foreign_keys = connection.exec_driver_sql(
            "PRAGMA foreign_key_list('data_release_artifacts')"
        ).all()
        indexes = connection.exec_driver_sql("PRAGMA index_list('data_release_artifacts')").all()
        index_columns = {
            row[1]: tuple(
                index_row[2]
                for index_row in connection.exec_driver_sql(f"PRAGMA index_info('{row[1]}')").all()
            )
            for row in indexes
        }

    assert any(
        row[2] == "data_releases"
        and row[3] == "release_id"
        and row[4] == "id"
        and row[6] == "RESTRICT"
        for row in foreign_keys
    )
    assert any(row[2] == 1 and index_columns[row[1]] == ("release_id", "role") for row in indexes)
    assert {
        ("release_id",),
        ("artifact_sha256",),
        ("native_payload_sha256",),
        ("missing_provenance_sha256",),
    } <= set(index_columns.values())

    row_values = {
        "artifact_sha256": artifact.artifact_sha256,
        "artifact_path": str(artifact.artifact_path.resolve()),
        "native_payload_sha256": artifact.native_payload_sha256,
        "missing_provenance_sha256": artifact.missing_provenance_sha256,
        "provenance_json": artifact.provenance_json,
    }
    with session_factory() as session:
        session.add(
            DataReleaseArtifact(
                release_id=result.release_id,
                role="source_response",
                **row_values,
            )
        )
        with pytest.raises(IntegrityError, match="UNIQUE constraint failed"):
            session.commit()

    with session_factory() as session:
        session.add(
            DataReleaseArtifact(
                release_id=result.release_id + 999,
                role="orphan",
                **row_values,
            )
        )
        with pytest.raises(IntegrityError, match="FOREIGN KEY constraint failed"):
            session.commit()

    with session_factory() as session:
        session.add(
            DataReleaseArtifact(
                release_id=result.release_id,
                role="bad_hash",
                **{**row_values, "artifact_sha256": "z" * 64},
            )
        )
        with pytest.raises(IntegrityError, match="CHECK constraint failed"):
            session.commit()


def test_exact_event_retry_can_atomically_backfill_missing_artifact_roles(
    session_factory, tmp_path
):
    clock = _at(2026, 1, 10)
    artifacts = (
        _artifact_meta(tmp_path, role="source_response"),
        _artifact_meta(
            tmp_path,
            role="native_series_payload",
            content=b"native series",
            native_payload=b"native series",
        ),
    )
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", clock),
        )
        backfill = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", clock, artifacts=artifacts),
        )
        repeat = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", clock, artifacts=tuple(reversed(artifacts))),
        )
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        artifact_rows = (
            session.execute(select(DataReleaseArtifact).order_by(DataReleaseArtifact.role))
            .scalars()
            .all()
        )

    assert first.created is True
    assert backfill.created is False
    assert repeat.created is False
    assert {first.release_id, backfill.release_id, repeat.release_id} == {first.release_id}
    assert release_count == 1
    assert [row.role for row in artifact_rows] == [
        "native_series_payload",
        "source_response",
    ]


def test_exact_event_artifact_conflict_rolls_back_partial_backfill(session_factory, tmp_path):
    clock = _at(2026, 1, 10)
    original = _artifact_meta(tmp_path, content=b"artifact A")
    conflicting = _artifact_meta(tmp_path, content=b"artifact B")
    new_role = _artifact_meta(
        tmp_path,
        role="missingness_ledger",
        content=b'{"records":[]}',
    )
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", clock, artifacts=(original,)),
        )
        with pytest.raises(ReleaseEventConflictError, match="artifact conflict"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 4.0)]),
                _meta(
                    "fred:DFF:US",
                    clock,
                    artifacts=(new_role, conflicting),
                ),
            )
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        artifact_rows = session.execute(select(DataReleaseArtifact)).scalars().all()

    assert first.created is True
    assert release_count == 1
    assert [row.role for row in artifact_rows] == ["source_response"]
    assert artifact_rows[0].artifact_sha256 == original.artifact_sha256


def test_later_artifact_assertion_does_not_backdate_a_legacy_release(session_factory, tmp_path):
    frame = _frame([(date(2026, 1, 1), 4.0)])
    artifact = _artifact_meta(tmp_path)
    with session_factory() as session:
        legacy = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 10)),
        )
        backed = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 11), artifacts=(artifact,)),
        )
        repeat = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 12), artifacts=(artifact,)),
        )
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        artifact_rows = session.execute(select(DataReleaseArtifact)).scalars().all()

    assert legacy.created is True
    assert backed.created is True
    assert backed.release_id != legacy.release_id
    assert repeat.created is False
    assert repeat.release_id == backed.release_id
    assert release_count == 2
    assert [row.release_id for row in artifact_rows] == [backed.release_id]


def test_changed_artifact_can_recur_without_false_deduplication(session_factory, tmp_path):
    frame = _frame([(date(2026, 1, 1), 4.0)])
    artifact_a = _artifact_meta(tmp_path, content=b"artifact A")
    artifact_b = _artifact_meta(tmp_path, content=b"artifact B")
    with session_factory() as session:
        first_a = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 10), artifacts=(artifact_a,)),
        )
        middle_b = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 11), artifacts=(artifact_b,)),
        )
        final_a = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 12), artifacts=(artifact_a,)),
        )
        repeat_a = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 13), artifacts=(artifact_a,)),
        )
        artifact_rows = (
            session.execute(select(DataReleaseArtifact).order_by(DataReleaseArtifact.release_id))
            .scalars()
            .all()
        )

    assert first_a.created and middle_b.created and final_a.created
    assert len({first_a.release_id, middle_b.release_id, final_a.release_id}) == 3
    assert repeat_a.created is False
    assert repeat_a.release_id == final_a.release_id
    assert [row.artifact_sha256 for row in artifact_rows] == [
        artifact_a.artifact_sha256,
        artifact_b.artifact_sha256,
        artifact_a.artifact_sha256,
    ]


@pytest.mark.parametrize(
    "defect",
    [
        "artifact_hash",
        "native_hash",
        "missing_hash",
        "noncanonical_json",
        "missing_path",
        "duplicate_role",
    ],
)
def test_invalid_artifact_manifest_rolls_back_the_owned_transaction(
    session_factory, tmp_path, defect
):
    artifact = _artifact_meta(tmp_path)
    if defect == "artifact_hash":
        artifacts = (replace(artifact, artifact_sha256="0" * 64),)
    elif defect == "native_hash":
        artifacts = (replace(artifact, native_payload_sha256="invalid"),)
    elif defect == "missing_hash":
        artifacts = (replace(artifact, missing_provenance_sha256="0" * 64),)
    elif defect == "noncanonical_json":
        artifacts = (replace(artifact, provenance_json='{ "records": [] }'),)
    elif defect == "missing_path":
        artifacts = (replace(artifact, artifact_path=tmp_path / "missing.json"),)
    else:
        artifacts = (artifact, artifact)

    with session_factory() as session:
        session.add(
            Observation(
                country="SE",
                indicator="pending",
                date=date(2026, 1, 1),
                value=1.0,
                source="TEST",
                series_id="PENDING",
            )
        )
        with pytest.raises(ValueError):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 4.0)]),
                _meta(
                    "fred:DFF:US",
                    _at(2026, 1, 10),
                    artifacts=artifacts,
                ),
            )
        assert session.scalar(select(func.count()).select_from(Observation)) == 0
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 0
        assert session.scalar(select(func.count()).select_from(ReleaseObservation)) == 0
        assert session.scalar(select(func.count()).select_from(DataReleaseArtifact)) == 0


def test_deduplication_fails_closed_if_a_stored_artifact_disappears(session_factory, tmp_path):
    frame = _frame([(date(2026, 1, 1), 4.0)])
    artifact = _artifact_meta(tmp_path)
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 1, 10), artifacts=(artifact,)),
        )

    artifact.artifact_path.unlink()
    with session_factory() as session:
        with pytest.raises(ReleaseEventConflictError, match="integrity validation"):
            ingest_release_snapshot(
                session,
                frame,
                _meta("fred:DFF:US", _at(2026, 1, 11)),
            )
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1
        assert session.scalar(select(func.count()).select_from(DataReleaseArtifact)) == 1
        assert session.scalar(select(Observation.value)) == 4.0

    assert first.created is True


def test_available_clock_not_retrieval_clock_controls_as_of(session_factory):
    meta = ReleaseMeta(
        partition_key="fred:DFF:US",
        source_family="FRED",
        published_at=_at(2026, 1, 5),
        available_at=_at(2026, 1, 8),
        retrieved_at=_at(2026, 2, 1),
        projection=ProjectionScope("US", "policy_rate", ("FRED",)),
    )
    with session_factory() as session:
        ingest_release_snapshot(session, _frame([(date(2025, 12, 1), 4.0)]), meta)
        assert load_vintage_panel(session, _at(2026, 1, 7)).empty
        known = load_vintage_panel(session, _at(2026, 1, 8))

    assert known.iloc[0]["value"] == 4.0
    assert known.iloc[0]["available_at"].date() == date(2026, 1, 8)
    assert known.iloc[0]["retrieved_at"].date() == date(2026, 2, 1)


def test_same_snapshot_is_idempotent_even_when_row_order_changes(session_factory):
    frame = _frame(
        [
            (date(2026, 1, 1), 4.0),
            (date(2026, 2, 1), 3.9),
        ]
    )
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            frame,
            _meta("fred:DFF:US", _at(2026, 3, 5)),
        )
        again = ingest_release_snapshot(
            session,
            frame.iloc[::-1],
            _meta("fred:DFF:US", _at(2026, 3, 6)),
        )
        release_count = session.scalar(select(func.count()).select_from(DataRelease))
        row_count = session.scalar(select(func.count()).select_from(ReleaseObservation))

    assert first.created is True
    assert again.created is False
    assert again.release_id == first.release_id
    assert release_count == 1
    assert row_count == 2


def test_exact_event_clock_rejects_changed_content_or_provenance(session_factory):
    clock = _at(2026, 3, 5)
    original_meta = _meta(
        "fred:DFF:US",
        clock,
        vintage_label="H6-2026-03-05",
        source_url="https://example.test/original",
    )
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            original_meta,
        )
        exact_repeat = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            original_meta,
        )
        with pytest.raises(ReleaseEventConflictError, match="content_sha256"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 3.5)]),
                original_meta,
            )
        with pytest.raises(ReleaseEventConflictError, match="source_url"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 4.0)]),
                _meta(
                    "fred:DFF:US",
                    clock,
                    vintage_label="H6-2026-03-05",
                    source_url="https://example.test/repointed",
                ),
            )
        count = session.scalar(select(func.count()).select_from(DataRelease))

    assert first.created is True
    assert exact_repeat.created is False
    assert count == 1


def test_database_constraint_rejects_two_payloads_at_one_event_clock(session_factory):
    clock = _at(2026, 3, 5).replace(tzinfo=None)
    with session_factory() as session:
        session.add_all(
            [
                DataRelease(
                    partition_key="fred:DFF:US",
                    source_family="FRED",
                    available_at=clock,
                    retrieved_at=clock,
                    content_sha256="a" * 64,
                    row_count=1,
                ),
                DataRelease(
                    partition_key="fred:DFF:US",
                    source_family="FRED",
                    available_at=clock,
                    retrieved_at=clock,
                    content_sha256="b" * 64,
                    row_count=1,
                ),
            ]
        )
        with pytest.raises(IntegrityError, match="UNIQUE constraint failed"):
            session.commit()


def test_semantic_release_metadata_participates_in_deduplication(session_factory):
    frame = _frame([(date(2026, 1, 1), 4.0)])
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            frame,
            _meta(
                "fred:DFF:US",
                _at(2026, 3, 5),
                published_at=_at(2026, 3, 4),
                vintage_label="catalogue-sha256:one",
                source_url="https://example.test/first",
            ),
        )
        catalogue_change = ingest_release_snapshot(
            session,
            frame,
            _meta(
                "fred:DFF:US",
                _at(2026, 3, 6),
                published_at=_at(2026, 3, 4),
                vintage_label="catalogue-sha256:two",
                source_url="https://example.test/second",
            ),
        )
        url_churn = ingest_release_snapshot(
            session,
            frame,
            _meta(
                "fred:DFF:US",
                _at(2026, 3, 7),
                published_at=_at(2026, 3, 4),
                vintage_label="catalogue-sha256:two",
                source_url="https://mirror.example.test/second",
            ),
        )
        publication_change = ingest_release_snapshot(
            session,
            frame,
            _meta(
                "fred:DFF:US",
                _at(2026, 3, 8),
                published_at=_at(2026, 3, 8),
                vintage_label="catalogue-sha256:two",
            ),
        )

    assert first.created is True
    assert catalogue_change.created is True
    assert url_churn.created is False
    assert url_churn.release_id == catalogue_change.release_id
    assert publication_change.created is True


def test_content_can_recur_after_a_different_vintage(session_factory):
    """A -> B -> A is three real vintages; only consecutive A -> A deduplicates."""
    frame_a = _frame([(date(2026, 1, 1), 4.0)])
    frame_b = _frame([(date(2026, 1, 1), 3.5)])

    with session_factory() as session:
        first_a = ingest_release_snapshot(session, frame_a, _meta("fred:DFF:US", _at(2026, 1, 5)))
        middle_b = ingest_release_snapshot(session, frame_b, _meta("fred:DFF:US", _at(2026, 2, 5)))
        final_a = ingest_release_snapshot(session, frame_a, _meta("fred:DFF:US", _at(2026, 3, 5)))
        repeated_a = ingest_release_snapshot(
            session, frame_a, _meta("fred:DFF:US", _at(2026, 3, 6))
        )

        history = release_history(session, "fred:DFF:US")
        current = session.scalar(select(Observation.value))
        in_february = load_vintage_panel(session, _at(2026, 2, 20))
        in_march = load_vintage_panel(session, _at(2026, 3, 20))

    assert first_a.created and middle_b.created and final_a.created
    assert final_a.release_id != first_a.release_id
    assert repeated_a.created is False
    assert repeated_a.release_id == final_a.release_id
    assert len(history) == 3
    assert current == 4.0
    assert in_february.iloc[0]["value"] == 3.5
    assert in_march.iloc[0]["value"] == 4.0


def test_identical_older_backfill_is_not_hidden_by_a_later_release(session_factory):
    frame = _frame([(date(2026, 1, 1), 4.0)])
    with session_factory() as session:
        current = ingest_release_snapshot(session, frame, _meta("fred:DFF:US", _at(2026, 3, 5)))
        backfill = ingest_release_snapshot(
            session,
            frame,
            _meta(
                "fred:DFF:US",
                _at(2026, 1, 5),
                retrieved_at=_at(2026, 4, 5),
            ),
        )
        known_in_january = load_vintage_panel(session, _at(2026, 1, 20))
        history = release_history(session, "fred:DFF:US")

    assert current.created is True
    assert backfill.created is True
    assert backfill.projected is False
    assert len(history) == 2
    assert known_in_january.iloc[0]["value"] == 4.0


def test_late_backfill_deduplicates_against_its_chronological_predecessor(
    session_factory,
):
    frame_a = _frame([(date(2026, 1, 1), 4.0)])
    frame_b = _frame([(date(2026, 1, 1), 3.5)])
    with session_factory() as session:
        january = ingest_release_snapshot(session, frame_a, _meta("fred:DFF:US", _at(2026, 1, 5)))
        ingest_release_snapshot(session, frame_b, _meta("fred:DFF:US", _at(2026, 2, 5)))
        ingest_release_snapshot(session, frame_a, _meta("fred:DFF:US", _at(2026, 3, 5)))

        redundant_a = ingest_release_snapshot(
            session,
            frame_a,
            _meta(
                "fred:DFF:US",
                _at(2026, 1, 15),
                retrieved_at=_at(2026, 4, 1),
            ),
        )
        meaningful_a = ingest_release_snapshot(
            session,
            frame_a,
            _meta(
                "fred:DFF:US",
                _at(2026, 2, 15),
                retrieved_at=_at(2026, 4, 2),
            ),
        )
        history = release_history(session, "fred:DFF:US")

    assert redundant_a.created is False
    assert redundant_a.release_id == january.release_id
    assert meaningful_a.created is True
    assert meaningful_a.projected is False
    assert len(history) == 4


def test_partition_key_cannot_be_reused_for_another_scalar_or_family(session_factory):
    partition = "stable:partition"
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta(partition, _at(2026, 1, 5)),
        )

        with pytest.raises(ValueError, match="country identity"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 2.0)], country="SE"),
                _meta(partition, _at(2026, 2, 5), country="SE"),
            )
        with pytest.raises(ValueError, match="series_id identity"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 3.5)], series_id="OTHER"),
                _meta(partition, _at(2026, 2, 5)),
            )
        with pytest.raises(ValueError, match="belongs to source_family"):
            ingest_release_snapshot(
                session,
                _frame([(date(2026, 1, 1), 3.5)]),
                ReleaseMeta(
                    partition_key=partition,
                    source_family="FEDERAL_RESERVE",
                    available_at=_at(2026, 2, 5),
                    retrieved_at=_at(2026, 2, 5),
                    projection=ProjectionScope("US", "policy_rate", ("FEDERAL_RESERVE", "FRED")),
                ),
            )


def test_partition_scope_allows_weo_history_and_forecast_sources(session_factory):
    partition = "series:IMF_WEO:GGXWDG_NGDP:US:gov_debt_pct_gdp"
    history = _frame(
        [(date(2025, 12, 31), 120.0)],
        indicator="gov_debt_pct_gdp",
        source="IMF_WEO",
        series_id="GGXWDG_NGDP",
    )
    forecast = _frame(
        [(date(2026, 12, 31), 122.0)],
        indicator="gov_debt_pct_gdp",
        source="IMF_WEO_FCST",
        series_id="GGXWDG_NGDP",
    )
    scope = ("IMF_WEO", "IMF_WEO_FCST")
    with session_factory() as session:
        first = ingest_release_snapshot(
            session,
            pd.concat([history, forecast], ignore_index=True),
            _meta(
                partition,
                _at(2026, 1, 5),
                indicator="gov_debt_pct_gdp",
                source="IMF_WEO",
                projection_sources=scope,
            ),
        )
        revised_forecast = forecast.assign(value=123.0)
        second = ingest_release_snapshot(
            session,
            pd.concat([history, revised_forecast], ignore_index=True),
            _meta(
                partition,
                _at(2026, 2, 5),
                indicator="gov_debt_pct_gdp",
                source="IMF_WEO",
                projection_sources=scope,
            ),
        )
        with pytest.raises(ValueError, match="outside the projection scope"):
            ingest_release_snapshot(
                session,
                history,
                _meta(
                    partition,
                    _at(2026, 3, 5),
                    indicator="gov_debt_pct_gdp",
                    source="IMF_WEO",
                ),
            )

    assert first.created is True
    assert second.created is True


@pytest.mark.parametrize(
    ("constraint_name", "constraint_columns"),
    [
        (
            "uq_release_partition_content",
            "partition_key, content_sha256",
        ),
        (
            "uq_release_partition_content_clock",
            "partition_key, content_sha256, available_at, retrieved_at",
        ),
    ],
)
def test_init_db_migrates_legacy_release_uniqueness_without_losing_children(
    tmp_path,
    constraint_name,
    constraint_columns,
):
    engine = make_engine(tmp_path / "legacy-release-constraint.db")
    backup_path = release_recurrence_migration_backup_path(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            f"""
            CREATE TABLE data_releases (
                id INTEGER NOT NULL PRIMARY KEY,
                partition_key VARCHAR(256) NOT NULL,
                source_family VARCHAR(64) NOT NULL,
                published_at DATETIME,
                available_at DATETIME NOT NULL,
                retrieved_at DATETIME NOT NULL,
                vintage_label VARCHAR(128),
                source_url TEXT,
                content_sha256 VARCHAR(64) NOT NULL,
                row_count INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT {constraint_name}
                    UNIQUE ({constraint_columns})
            )
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TABLE release_observations (
                id INTEGER NOT NULL PRIMARY KEY,
                release_id INTEGER NOT NULL
                    REFERENCES data_releases (id) ON DELETE RESTRICT,
                country VARCHAR(8) NOT NULL,
                indicator VARCHAR(64) NOT NULL,
                date DATE NOT NULL,
                value FLOAT NOT NULL,
                source VARCHAR(32) NOT NULL,
                series_id VARCHAR(128) NOT NULL,
                status VARCHAR(24) NOT NULL
            )
            """
        )
        connection.exec_driver_sql(
            """
            INSERT INTO data_releases VALUES
            (7, 'fred:DFF:US', 'FRED', NULL, '2026-01-05 12:00:00',
             '2026-01-05 12:00:00', NULL, NULL, ?, 1, '2026-01-05 12:00:00')
            """,
            ("a" * 64,),
        )
        connection.exec_driver_sql(
            """
            INSERT INTO release_observations VALUES
            (9, 7, 'US', 'policy_rate', '2026-01-01', 4.0,
             'FRED', 'DFF', 'observed')
            """
        )
        connection.exec_driver_sql(
            "CREATE INDEX custom_release_vintage_index ON data_releases (vintage_label)"
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER custom_release_insert_trigger
            AFTER INSERT ON data_releases
            BEGIN
                SELECT 1;
            END
            """
        )

    init_db(engine)

    with engine.connect() as connection:
        table_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_releases'"
        ).scalar_one()
        release = connection.exec_driver_sql(
            "SELECT id, partition_key, content_sha256 FROM data_releases"
        ).one()
        child = connection.exec_driver_sql(
            "SELECT id, release_id, value FROM release_observations"
        ).one()
        foreign_key_errors = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
        integrity = connection.exec_driver_sql("PRAGMA integrity_check").all()
        indexes = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='data_releases'"
            )
        }
        triggers = {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type='trigger' AND tbl_name='data_releases'"
            )
        }

    with sqlite3.connect(backup_path) as backup:
        backup_sql = backup.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_releases'"
        ).fetchone()[0]
        backup_release = backup.execute(
            "SELECT id, partition_key, content_sha256 FROM data_releases"
        ).fetchone()
        backup_child = backup.execute(
            "SELECT id, release_id, value FROM release_observations"
        ).fetchone()

    assert "uq_release_partition_clock" in table_sql
    assert "content_sha256, available_at" not in table_sql
    assert release == (7, "fred:DFF:US", "a" * 64)
    assert child == (9, 7, 4.0)
    assert foreign_key_errors == []
    assert integrity == [("ok",)]
    assert backup_path.is_file()
    assert constraint_name in backup_sql
    assert backup_release == release
    assert backup_child == child
    assert "custom_release_vintage_index" in indexes
    assert {
        "custom_release_insert_trigger",
        "data_releases_reject_update",
        "data_releases_reject_delete",
    } <= triggers


def test_release_migration_fails_closed_on_duplicate_event_clocks(tmp_path):
    engine = make_engine(tmp_path / "duplicate-release-clocks.db")
    backup_path = release_recurrence_migration_backup_path(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE data_releases (
                id INTEGER NOT NULL PRIMARY KEY,
                partition_key VARCHAR(256) NOT NULL,
                source_family VARCHAR(64) NOT NULL,
                published_at DATETIME,
                available_at DATETIME NOT NULL,
                retrieved_at DATETIME NOT NULL,
                vintage_label VARCHAR(128),
                source_url TEXT,
                content_sha256 VARCHAR(64) NOT NULL,
                row_count INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                CONSTRAINT uq_release_partition_content_clock UNIQUE
                    (partition_key, content_sha256, available_at, retrieved_at)
            )
            """
        )
        for release_id, digest in ((1, "a" * 64), (2, "b" * 64)):
            connection.exec_driver_sql(
                """
                INSERT INTO data_releases VALUES
                (?, 'fred:DFF:US', 'FRED', NULL, '2026-01-05 12:00:00',
                 '2026-01-05 12:00:00', NULL, NULL, ?, 1,
                 '2026-01-05 12:00:00')
                """,
                (release_id, digest),
            )

    with pytest.raises(RuntimeError, match="duplicate event clocks"):
        init_db(engine)

    with engine.connect() as connection:
        count = connection.exec_driver_sql("SELECT COUNT(*) FROM data_releases").scalar_one()
        table_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='data_releases'"
        ).scalar_one()
    assert count == 2
    assert "uq_release_partition_content_clock" in table_sql
    assert not backup_path.exists()


def test_release_migration_rejects_an_unknown_uniqueness_layout(tmp_path):
    engine = make_engine(tmp_path / "unknown-release-layout.db")
    with engine.begin() as connection:
        connection.exec_driver_sql(
            """
            CREATE TABLE data_releases (
                id INTEGER NOT NULL PRIMARY KEY,
                partition_key VARCHAR(256) NOT NULL,
                source_family VARCHAR(64) NOT NULL,
                published_at DATETIME,
                available_at DATETIME NOT NULL,
                retrieved_at DATETIME NOT NULL,
                vintage_label VARCHAR(128),
                source_url TEXT,
                content_sha256 VARCHAR(64) NOT NULL,
                row_count INTEGER NOT NULL,
                created_at DATETIME NOT NULL,
                UNIQUE (partition_key, retrieved_at)
            )
            """
        )

    with pytest.raises(RuntimeError, match="unknown uniqueness layout"):
        init_db(engine)


def test_partition_update_does_not_touch_another_country(session_factory):
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 2.0)], country="SE", series_id="SE_RATE"),
            _meta(
                "fred:SE_RATE:SE",
                _at(2026, 1, 5),
                country="SE",
            ),
        )
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 5)),
        )
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 3.5)]),
            _meta("fred:DFF:US", _at(2026, 2, 5)),
        )
        rows = session.execute(
            select(Observation.country, Observation.value).order_by(Observation.country)
        ).all()

    assert rows == [("SE", 2.0), ("US", 3.5)]


def test_older_release_ingested_late_never_rolls_current_back(session_factory):
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 3.5)]),
            _meta("fred:DFF:US", _at(2026, 2, 5), retrieved_at=_at(2026, 2, 5)),
        )
        old = ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 5), retrieved_at=_at(2026, 3, 5)),
        )
        current = session.scalar(select(Observation.value))

    assert old.created is True
    assert old.projected is False
    assert current == 3.5


def test_empty_or_ambiguous_snapshot_fails_without_erasing_current(session_factory):
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 1, 5)),
        )
        with pytest.raises(ValueError, match="empty"):
            ingest_release_snapshot(
                session,
                _frame([]),
                _meta("fred:DFF:US", _at(2026, 2, 5)),
            )
        collision = pd.concat(
            [
                _frame([(date(2026, 1, 1), 3.5)]),
                _frame([(date(2026, 1, 1), 3.4)], series_id="OTHER"),
            ]
        )
        with pytest.raises(ValueError, match="duplicate observation key"):
            ingest_release_snapshot(
                session,
                collision,
                _meta("fred:DFF:US", _at(2026, 2, 5)),
            )
        assert session.scalar(select(Observation.value)) == 4.0
        assert session.scalar(select(func.count()).select_from(DataRelease)) == 1


def test_release_history_is_chronological(session_factory):
    with session_factory() as session:
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.0)]),
            _meta("fred:DFF:US", _at(2026, 2, 5)),
        )
        ingest_release_snapshot(
            session,
            _frame([(date(2026, 1, 1), 4.1)]),
            _meta("fred:DFF:US", _at(2026, 3, 5)),
        )
        history = release_history(session, "fred:DFF:US")

    assert [release.available_at.date() for release in history] == [
        date(2026, 2, 5),
        date(2026, 3, 5),
    ]


def test_bootstrap_is_explicit_idempotent_and_does_not_change_current(session_factory):
    with session_factory() as session:
        session.add(
            Observation(
                country="US",
                indicator="policy_rate",
                date=date(2026, 1, 1),
                value=4.0,
                source="FRED",
                series_id="DFF",
            )
        )
        session.commit()
        first = bootstrap_current_observations(session, available_at=_at(2026, 9, 1))
        second = bootstrap_current_observations(session, available_at=_at(2026, 9, 1))
        before = load_vintage_panel(session, _at(2026, 8, 31))
        on_date = load_vintage_panel(session, _at(2026, 9, 1))
        current = session.scalar(select(Observation.value))

    assert first == (1, 1)
    assert second == (0, 1)
    assert before.empty
    assert on_date.iloc[0]["value"] == 4.0
    assert current == 4.0


def test_bootstrap_keeps_weo_history_and_forecast_in_one_partition(session_factory):
    with session_factory() as session:
        session.add_all(
            [
                Observation(
                    country="US",
                    indicator="gov_debt_pct_gdp",
                    date=date(2025, 12, 31),
                    value=120.0,
                    source="IMF_WEO",
                    series_id="GGXWDG_NGDP",
                ),
                Observation(
                    country="US",
                    indicator="gov_debt_pct_gdp",
                    date=date(2026, 12, 31),
                    value=122.0,
                    source="IMF_WEO_FCST",
                    series_id="GGXWDG_NGDP",
                ),
            ]
        )
        session.commit()
        assert bootstrap_current_observations(
            session,
            available_at=_at(2026, 9, 1),
        ) == (1, 2)
        releases = session.execute(select(DataRelease)).scalars().all()
        rows = session.execute(select(ReleaseObservation)).scalars().all()

    assert len(releases) == 1
    assert releases[0].source_family == "IMF_WEO"
    assert releases[0].partition_key == ("series:IMF_WEO:GGXWDG_NGDP:US:gov_debt_pct_gdp")
    assert {row.status for row in rows} == {"observed", "forecast"}


def test_fred_pipeline_writes_complete_release_snapshots(tmp_path, monkeypatch):
    from dalio.data_sources.fred import FredSeriesSpec
    from dalio.pipelines.fetch_fred import run_pipeline

    db_path = tmp_path / "fred.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db_path))
    spec = FredSeriesSpec("policy_rate", "DFF", "US", frequency="M")

    class FakeFred:
        frame = _frame(
            [
                (date(2025, 12, 1), 4.0),
                (date(2026, 1, 1), 4.1),
            ]
        )

        def fetch(self, _spec):
            return self.frame.copy()

    source = FakeFred()
    first = run_pipeline((spec,), source=source, retrieved_at=_at(2026, 1, 10))
    source.frame = _frame([(date(2025, 12, 1), 4.25)])
    second = run_pipeline((spec,), source=source, retrieved_at=_at(2026, 2, 10))

    engine = make_engine(db_path)
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        releases = release_history(session, "series:FRED:DFF:US:policy_rate")
        old = load_vintage_panel(session, _at(2026, 2, 9))
        new = load_vintage_panel(session, _at(2026, 2, 10))
        current = session.execute(select(Observation.date, Observation.value)).all()

    assert first["US/policy_rate"]["inserted"] == 2
    assert second["US/policy_rate"]["inserted"] == 1
    assert second["US/policy_rate"]["removed"] == 1
    assert len(releases) == 2
    assert len(old) == 2
    assert new[["date", "value"]].to_records(index=False).tolist() == [
        (date(2025, 12, 1), 4.25),
    ]
    assert current == [(date(2025, 12, 1), 4.25)]
