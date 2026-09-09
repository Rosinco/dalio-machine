"""Ordered, immutable semantic scopes for mixed communication artifacts."""

from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_SOURCES,
)
from dalio.storage import db as db_module
from dalio.storage.communications import (
    CommunicationArtifactMeta,
    CommunicationEventMeta,
    CommunicationSectionScopeMeta,
    get_communication_artifact_section_scopes,
    record_communication_artifact_metadata,
)
from dalio.storage.db import (
    _COMMUNICATION_SCHEMA_V1_SHA256,
    _COMMUNICATION_SCHEMA_V1_TABLES,
    _COMMUNICATION_TRIGGER_V1_SHA256,
    COMMUNICATION_SCHEMA_SHA256,
    COMMUNICATION_SCHEMA_VERSION,
    COMMUNICATION_TRIGGER_SHA256,
    CommunicationArtifact,
    CommunicationArtifactContent,
    CommunicationArtifactSectionScopeSet,
    Organization,
    _communication_ddl_sha256,
    _create_communication_segment_finalization_guard,
    _create_immutable_table_triggers,
    communication_schema_v2_migration_backup_path,
    init_db,
    make_engine,
    make_session_factory,
)

SOURCES = {source.source_id: source for source in COMMUNICATION_SOURCES}
OBSERVED_AT = datetime(2026, 9, 9, 8, tzinfo=UTC)


def _event() -> CommunicationEventMeta:
    return CommunicationEventMeta(
        organization_id="ecb",
        event_key="ecb_2025_01_30",
        event_type="central_bank_press_conference",
        title="ECB monetary policy press conference",
        event_date=date(2025, 1, 30),
        metadata_known_at=OBSERVED_AT,
    )


def _artifact(**changes) -> CommunicationArtifactMeta:
    source = SOURCES["ecb_monetary_policy_press_conferences_en"]
    values = {
        "source_id": source.source_id,
        "catalogue_sha256": COMMUNICATION_CATALOGUE_SHA256,
        "artifact_key": "official_statement_with_q_and_a_en",
        "artifact_role": "q_and_a_transcript",
        "material_type": "questions_and_answers",
        "language": "en",
        "translation_status": "original",
        "mime_type": "text/html",
        "origin_type": "official_published_transcript",
        "provenance_tier": "official_published_transcript",
        "host_organization": source.host_organization,
        "publisher": source.publisher,
        "transcriber": None,
        "transcriber_attribution": "not_disclosed",
        "rights_status": source.rights_status,
        "acquisition_status": source.acquisition_status,
        "rights_checked_by": "human:test_reviewer",
        "rights_checked_at": OBSERVED_AT,
        "published_at": None,
        "available_at": OBSERVED_AT,
        "retrieved_at": OBSERVED_AT,
        "metadata_known_at": OBSERVED_AT,
        "landing_url": source.landing_url,
        "artifact_url": source.landing_url,
    }
    values.update(changes)
    return CommunicationArtifactMeta(**values)


def _mixed_scopes() -> tuple[CommunicationSectionScopeMeta, ...]:
    return (
        CommunicationSectionScopeMeta(
            section_ordinal=1,
            scope_key="prepared_remarks",
            artifact_role="prepared_remarks",
            material_type="monetary_policy_statement",
            origin_type="publisher_authored",
            provenance_tier="official_authored_text",
            transcriber=None,
            transcriber_attribution="not_applicable",
        ),
        CommunicationSectionScopeMeta(
            section_ordinal=2,
            scope_key="q_and_a",
            artifact_role="q_and_a_transcript",
            material_type="questions_and_answers",
            origin_type="official_published_transcript",
            provenance_tier="official_published_transcript",
            transcriber=None,
            transcriber_attribution="not_disclosed",
        ),
    )


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "communication-scopes.db")
    init_db(engine)
    return make_session_factory(engine)


def test_mixed_scope_set_is_atomic_and_content_free(session_factory):
    with session_factory() as session:
        mixed = record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(),
            section_scopes=_mixed_scopes(),
        )
        stored = session.get(CommunicationArtifactSectionScopeSet, mixed.artifact_id)
        assert get_communication_artifact_section_scopes(session, mixed.artifact_id) == (
            _mixed_scopes()
        )
        assert stored.scope_set_sha256 == (
            "85608fe71129dc45f6905b091e05aaf47035495dbf898421f763b844f84a4fa9"
        )
        assert session.scalar(select(func.count()).select_from(CommunicationArtifact)) == 1
        assert session.scalar(select(func.count()).select_from(CommunicationArtifactContent)) == 0


def test_scope_order_is_authoritative_and_repeat_retrieval_preserves_artifact_version(
    session_factory,
):
    with session_factory() as session:
        with pytest.raises(ValueError, match="dense and follow sequence order"):
            record_communication_artifact_metadata(
                session,
                _event(),
                _artifact(),
                section_scopes=tuple(reversed(_mixed_scopes())),
            )

        first = record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(),
            section_scopes=_mixed_scopes(),
        )
        later = OBSERVED_AT + timedelta(days=1)
        second = record_communication_artifact_metadata(
            session,
            _event(),
            replace(_artifact(), retrieved_at=later, metadata_known_at=later),
            section_scopes=_mixed_scopes(),
        )
        assert second.artifact_id == first.artifact_id
        assert second.supersedes_artifact_id is None
        assert second.artifact_version_sha256 == first.artifact_version_sha256


def test_scope_rows_are_immutable_and_raw_retrieval_requires_scope(session_factory):
    with session_factory() as session:
        result = record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(),
            section_scopes=_mixed_scopes(),
        )
        scope_set = session.get(CommunicationArtifactSectionScopeSet, result.artifact_id)
        scope_set.scope_count = 2
        with pytest.raises(ValueError, match="immutable"):
            session.flush()
        session.rollback()

    with session_factory() as session:
        result = record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(),
            section_scopes=_mixed_scopes(),
        )
        session.execute(text("DROP TRIGGER communication_artifact_section_scope_sets_reject_delete"))
        session.execute(
            text("DELETE FROM communication_artifact_section_scope_sets WHERE artifact_id = :id"),
            {"id": result.artifact_id},
        )
        session.execute(
            text("DROP TRIGGER communication_artifacts_reject_update")
        )
        session.execute(
            text(
                "UPDATE communication_artifacts SET rights_status = 'cleared', "
                "acquisition_status = 'manual_collection_ready' WHERE id = :id"
            ),
            {"id": result.artifact_id},
        )
        retrieval_id = session.scalar(
            text(
                "SELECT id FROM communication_artifact_retrievals "
                "WHERE artifact_id = :id"
            ),
            {"id": result.artifact_id},
        )
        session.add(
            CommunicationArtifactContent(
                artifact_id=result.artifact_id,
                retrieval_id=retrieval_id,
                content_sha256="1" * 64,
                size_bytes=1,
                blob_path="artifacts/communications/sha256/11/" + "1" * 64,
                captured_at=OBSERVED_AT.replace(tzinfo=None),
            )
        )
        with pytest.raises(IntegrityError, match="retrieval or rights policy"):
            session.flush()


def test_mixed_representation_source_requires_explicit_scope_sequence(session_factory):
    with session_factory() as session, pytest.raises(
        ValueError,
        match="section_scopes must be explicit for mixed-representation source",
    ):
        record_communication_artifact_metadata(session, _event(), _artifact())

    with session_factory() as session, pytest.raises(
        ValueError,
        match="must declare ordered prepared_remarks and q_and_a scopes",
    ):
        record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(),
            section_scopes=(replace(_mixed_scopes()[1], section_ordinal=1),),
        )


def _downgrade_empty_v2_to_v1(engine) -> None:
    with engine.begin() as connection:
        segment_indexes = [
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT sql FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'communication_segments' AND sql IS NOT NULL"
            )
        ]
        segment_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'communication_segments'"
        ).scalar_one().replace("\n\tsection_ordinal INTEGER NOT NULL, ", "")
        finalization_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'communication_extraction_finalizations'"
        ).scalar_one().replace(
            "canonicalization_version = 'communication_segments_json_v2'",
            "canonicalization_version = 'communication_segments_json_v1'",
        )
        contract_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'communication_schema_contract'"
        ).scalar_one().replace("schema_version = 2", "schema_version = 1")

        for trigger_name in (
            "communication_retrieval_matches_artifact",
            "communication_content_matches_retrieval",
            "communication_segments_match_scope_set",
            "communication_extraction_finalization_validate",
        ):
            connection.exec_driver_sql(f"DROP TRIGGER {trigger_name}")
        connection.exec_driver_sql("DROP TABLE communication_artifact_section_scope_sets")
        connection.exec_driver_sql("DROP TABLE communication_extraction_finalizations")
        connection.exec_driver_sql("DROP TABLE communication_segments")
        connection.exec_driver_sql("DROP TABLE communication_schema_contract")
        connection.exec_driver_sql(contract_sql)
        connection.exec_driver_sql(segment_sql)
        connection.exec_driver_sql(finalization_sql)
        for index_sql in segment_indexes:
            connection.exec_driver_sql(index_sql)
        _create_immutable_table_triggers(
            connection,
            (
                "communication_schema_contract",
                "communication_segments",
                "communication_extraction_finalizations",
            ),
        )
        _create_communication_segment_finalization_guard(connection)
        connection.exec_driver_sql(
            """
            CREATE TRIGGER communication_retrieval_matches_artifact
            BEFORE INSERT ON communication_artifact_retrievals
            WHEN NOT EXISTS (
                SELECT 1 FROM communication_artifacts AS artifact
                WHERE artifact.id = NEW.artifact_id
                  AND artifact.landing_url = NEW.landing_url
                  AND artifact.artifact_url = NEW.artifact_url
                  AND NEW.retrieved_at >= artifact.available_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'retrieval observation conflicts with artifact');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER communication_content_matches_retrieval
            BEFORE INSERT ON communication_artifact_contents
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_artifacts AS artifact
                JOIN communication_artifact_retrievals AS retrieval
                  ON retrieval.artifact_id = artifact.id
                WHERE artifact.id = NEW.artifact_id
                  AND retrieval.id = NEW.retrieval_id
                  AND artifact.rights_status IN ('cleared', 'internal_only')
                  AND NEW.captured_at >= retrieval.retrieved_at
                  AND NEW.captured_at >= retrieval.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'content capture conflicts with retrieval or rights policy');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER communication_extraction_finalization_validate
            BEFORE INSERT ON communication_extraction_finalizations
            WHEN NOT EXISTS (
                SELECT 1 FROM communication_extractions AS extraction
                WHERE extraction.id = NEW.extraction_id
                  AND NEW.finalized_at >= extraction.extracted_at
                  AND NEW.segment_count = (
                      SELECT COUNT(*) FROM communication_segments
                      WHERE extraction_id = NEW.extraction_id
                  )
                  AND 1 = (
                      SELECT MIN(ordinal) FROM communication_segments
                      WHERE extraction_id = NEW.extraction_id
                  )
                  AND NEW.segment_count = (
                      SELECT MAX(ordinal) FROM communication_segments
                      WHERE extraction_id = NEW.extraction_id
                  )
                  AND NEW.total_char_count = (
                      SELECT SUM(char_count) FROM communication_segments
                      WHERE extraction_id = NEW.extraction_id
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'communication extraction finalization does not match segments');
            END
            """
        )
        connection.exec_driver_sql(
            "INSERT INTO communication_schema_contract VALUES (?, ?, ?, ?, ?)",
            (
                "institutional_communications",
                1,
                _COMMUNICATION_SCHEMA_V1_SHA256,
                _COMMUNICATION_TRIGGER_V1_SHA256,
                "2026-09-09 08:00:00.000000",
            ),
        )
        assert _communication_ddl_sha256(
            connection, ("table", "index"), _COMMUNICATION_SCHEMA_V1_TABLES
        ) == _COMMUNICATION_SCHEMA_V1_SHA256
        assert _communication_ddl_sha256(
            connection, ("trigger",), _COMMUNICATION_SCHEMA_V1_TABLES
        ) == _COMMUNICATION_TRIGGER_V1_SHA256


def test_empty_v1_migrates_backup_first_and_replays_idempotently(tmp_path):
    engine = make_engine(tmp_path / "legacy.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    backup = communication_schema_v2_migration_backup_path(engine)

    init_db(engine)
    backup_bytes = backup.read_bytes()
    init_db(engine)

    assert backup.is_file()
    assert backup.read_bytes() == backup_bytes
    assert not [
        path
        for path in tmp_path.iterdir()
        if path.name.startswith(f".{backup.name}.")
    ]
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT schema_version, schema_sha256, trigger_sha256 "
            "FROM communication_schema_contract"
        ).one() == (
            COMMUNICATION_SCHEMA_VERSION,
            COMMUNICATION_SCHEMA_SHA256,
            COMMUNICATION_TRIGGER_SHA256,
        )
        segment_columns = {
            row[1]: row
            for row in connection.exec_driver_sql("PRAGMA table_info(communication_segments)")
        }
        assert segment_columns["section_ordinal"][3] == 1
        finalization_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'table' "
            "AND name = 'communication_extraction_finalizations'"
        ).scalar_one()
        assert "communication_segments_json_v1" not in finalization_sql
        assert "canonicalization_version = 'communication_segments_json_v2'" in finalization_sql


def test_populated_v1_refuses_automatic_semantic_migration(tmp_path):
    engine = make_engine(tmp_path / "populated-legacy.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    with engine.begin() as connection:
        connection.execute(Organization.__table__.insert().values(organization_id="legacy_bank"))

    with pytest.raises(RuntimeError, match="requires an explicit semantic migration"):
        init_db(engine)

    assert not communication_schema_v2_migration_backup_path(engine).exists()


def test_v1_migration_refuses_untracked_communication_tables(tmp_path):
    engine = make_engine(tmp_path / "untracked-legacy.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql(
            "CREATE TABLE communication_experimental (id INTEGER PRIMARY KEY)"
        )
        connection.exec_driver_sql("INSERT INTO communication_experimental VALUES (1)")

    with pytest.raises(RuntimeError, match="unknown institutional-communications tables"):
        init_db(engine)

    assert not communication_schema_v2_migration_backup_path(engine).exists()
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT COUNT(*) FROM communication_experimental"
        ).scalar_one() == 1


def test_v1_migration_refuses_backup_hard_linked_to_source(tmp_path):
    engine = make_engine(tmp_path / "hard-linked-backup.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    database = tmp_path / "hard-linked-backup.db"
    backup = communication_schema_v2_migration_backup_path(engine)
    backup.hardlink_to(database)
    assert backup.samefile(database)

    with pytest.raises(RuntimeError, match="distinct filesystem object"):
        init_db(engine)

    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT schema_version FROM communication_schema_contract"
        ).scalar_one() == 1
        assert "communication_artifact_section_scope_sets" not in {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }


def test_v1_migration_refuses_tampered_trigger_before_backup(tmp_path):
    engine = make_engine(tmp_path / "tampered-trigger.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    backup = communication_schema_v2_migration_backup_path(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER communication_event_successor_order")

    with pytest.raises(RuntimeError, match="v1 schema fingerprint mismatch"):
        init_db(engine)

    assert not backup.exists()
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT schema_version FROM communication_schema_contract"
        ).scalar_one() == 1


def test_v1_migration_rolls_back_after_backup_and_retries(
    tmp_path,
    monkeypatch,
):
    engine = make_engine(tmp_path / "rollback-retry.db")
    init_db(engine)
    _downgrade_empty_v2_to_v1(engine)
    backup = communication_schema_v2_migration_backup_path(engine)
    original = db_module._create_communication_section_scope_triggers

    def fail_after_rebuild_started(connection):
        raise RuntimeError("injected migration failure")

    monkeypatch.setattr(
        db_module,
        "_create_communication_section_scope_triggers",
        fail_after_rebuild_started,
    )
    with pytest.raises(RuntimeError, match="injected migration failure"):
        init_db(engine)

    assert backup.is_file()
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT schema_version FROM communication_schema_contract"
        ).scalar_one() == 1
        assert "communication_artifact_section_scope_sets" not in {
            row[0]
            for row in connection.exec_driver_sql(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }

    monkeypatch.setattr(
        db_module,
        "_create_communication_section_scope_triggers",
        original,
    )
    init_db(engine)
    with engine.connect() as connection:
        assert connection.exec_driver_sql(
            "SELECT schema_version FROM communication_schema_contract"
        ).scalar_one() == COMMUNICATION_SCHEMA_VERSION
