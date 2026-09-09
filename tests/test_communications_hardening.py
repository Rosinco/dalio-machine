"""Release-blocking invariants for the communications metadata ledger."""

from dataclasses import replace
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select, text
from sqlalchemy.exc import IntegrityError

from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_SOURCES,
)
from dalio.storage.communications import (
    CommodityCoverageMeta,
    CommunicationArtifactMeta,
    CommunicationEventMeta,
    CommunicationSectionScopeMeta,
    _artifact_version,
    append_catalogue_commodity_coverage,
    record_communication_artifact_metadata,
    record_communication_artifact_section_scopes,
)
from dalio.storage.db import (
    CommunicationArtifact,
    CommunicationArtifactContent,
    CommunicationArtifactRetrieval,
    CommunicationEvent,
    CommunicationExtraction,
    CommunicationSourcePolicySnapshot,
    Organization,
    OrganizationCommodityCoverage,
    init_db,
    make_engine,
    make_session_factory,
)

SOURCES = {source.source_id: source for source in COMMUNICATION_SOURCES}


def _at(day: int) -> datetime:
    return datetime(2026, 7, day, 12, tzinfo=UTC)


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "communications-hardening.db")
    init_db(engine)
    return make_session_factory(engine)


def _event(**changes) -> CommunicationEventMeta:
    values = {
        "organization_id": "federal_reserve",
        "event_key": "fomc_2026_07_29",
        "event_type": "central_bank_press_conference",
        "title": "FOMC press conference",
        "event_date": date(2026, 7, 29),
        "metadata_known_at": _at(30),
    }
    values.update(changes)
    return CommunicationEventMeta(**values)


def _artifact(**changes) -> CommunicationArtifactMeta:
    source = SOURCES["fed_fomc_press_conferences_en"]
    values = {
        "source_id": source.source_id,
        "catalogue_sha256": COMMUNICATION_CATALOGUE_SHA256,
        "artifact_key": "official_transcript_en",
        "artifact_role": "full_transcript",
        "material_type": "press_conference_transcript",
        "language": "en",
        "translation_status": "original",
        "mime_type": "application/pdf",
        "origin_type": "official_published_transcript",
        "provenance_tier": "official_published_transcript",
        "host_organization": source.host_organization,
        "publisher": source.publisher,
        "transcriber": None,
        "transcriber_attribution": "not_disclosed",
        "rights_status": source.rights_status,
        "acquisition_status": source.acquisition_status,
        "rights_checked_by": "human:test_reviewer",
        "rights_checked_at": _at(30),
        "available_at": _at(29),
        "retrieved_at": _at(30),
        "metadata_known_at": _at(30),
        "landing_url": source.landing_url,
        "artifact_url": "https://www.federalreserve.gov/mediacenter/files/transcript.pdf",
        "published_at": _at(29),
    }
    values.update(changes)
    return CommunicationArtifactMeta(**values)


def _naive_at(day: int) -> datetime:
    return _at(day).replace(tzinfo=None)


def _seed_cleared_artifact(session):
    """Build a self-contained reviewed-policy fixture for content-schema tests."""
    catalogue_sha256 = "c" * 64
    event_sha256 = "e" * 64
    landing_url = "https://www.federalreserve.gov/monetarypolicy/test.htm"
    session.add(Organization(organization_id="test_central_bank"))
    session.flush()
    session.add(
        CommunicationSourcePolicySnapshot(
            catalogue_sha256=catalogue_sha256,
            source_id="test_press_conferences_en",
            organization_id="test_central_bank",
            organization_name="Test Central Bank",
            organization_type="central_bank",
            jurisdiction="US",
            language="en",
            landing_url=landing_url,
            official_domains_json='["federalreserve.gov"]',
            host_organization="Test Central Bank",
            publisher="Test Central Bank",
            transcriber=None,
            transcriber_attribution="not_disclosed",
            material_types_json='["press_conference_transcript"]',
            commodity_families_json="[]",
            verified_archive_start_year=2020,
            coverage_note="Test-only reviewed source policy.",
            source_provenance_tier="official_published_transcript",
            rights_status="cleared",
            rights_basis_url="https://www.federalreserve.gov/aboutthefed/terms.htm",
            rights_note="Test-only rights basis.",
            acquisition_status="manual_collection_ready",
            acquisition_note="Test-only manual collection.",
            automated_collection_allowed=False,
            rights_checked_by="human:test_reviewer",
            rights_checked_at=_naive_at(29),
            catalogue_evaluated_at=_naive_at(30),
            policy_sha256="f" * 64,
        )
    )
    session.flush()
    event = CommunicationEvent(
        organization_id="test_central_bank",
        event_key="policy_event_2026_07_29",
        event_type="central_bank_press_conference",
        title="Test policy event",
        event_date=date(2026, 7, 29),
        event_started_at=None,
        reference_start=None,
        reference_end=None,
        metadata_known_at=_naive_at(30),
        event_version_sha256=event_sha256,
        supersedes_event_id=None,
    )
    session.add(event)
    session.flush()
    artifact = CommunicationArtifact(
        event_id=event.id,
        source_id="test_press_conferences_en",
        catalogue_sha256=catalogue_sha256,
        event_version_sha256=event_sha256,
        artifact_key="official_transcript_en",
        artifact_role="full_transcript",
        material_type="press_conference_transcript",
        language="en",
        translation_status="original",
        mime_type="application/pdf",
        origin_type="official_published_transcript",
        provenance_tier="official_published_transcript",
        rights_status="cleared",
        acquisition_status="manual_collection_ready",
        rights_basis_url="https://www.federalreserve.gov/aboutthefed/terms.htm",
        rights_note="Test-only rights basis.",
        rights_checked_by="human:test_reviewer",
        rights_checked_at=_naive_at(29),
        host_organization="Test Central Bank",
        publisher="Test Central Bank",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        published_at=_naive_at(29),
        available_at=_naive_at(29),
        retrieved_at=_naive_at(30),
        metadata_known_at=_naive_at(30),
        landing_url=landing_url,
        artifact_url=landing_url,
        artifact_version_sha256="a" * 64,
        supersedes_artifact_id=None,
    )
    scope = CommunicationSectionScopeMeta(
        section_ordinal=1,
        scope_key="full_transcript",
        artifact_role="full_transcript",
        material_type="press_conference_transcript",
        origin_type="official_published_transcript",
        provenance_tier="official_published_transcript",
        transcriber=None,
        transcriber_attribution="not_disclosed",
    )
    artifact.artifact_version_sha256 = _artifact_version(event_sha256, artifact, (scope,))
    session.add(artifact)
    session.flush()
    record_communication_artifact_section_scopes(
        session,
        artifact_id=artifact.id,
        section_scopes=(scope,),
        metadata_known_at=_at(30),
    )
    retrieval = CommunicationArtifactRetrieval(
        artifact_id=artifact.id,
        retrieved_at=_naive_at(30),
        metadata_known_at=_naive_at(30),
        landing_url=landing_url,
        artifact_url=landing_url,
    )
    session.add(retrieval)
    session.flush()
    return artifact, retrieval


def test_helper_flushes_validated_policy_but_caller_owns_rollback(session_factory):
    with session_factory() as session:
        record_communication_artifact_metadata(session, _event(), _artifact())
        assert session.scalar(select(func.count()).select_from(CommunicationArtifact)) == 1
        assert (
            session.scalar(select(func.count()).select_from(CommunicationSourcePolicySnapshot)) == 1
        )
        session.rollback()

    with session_factory() as session:
        assert session.scalar(select(func.count()).select_from(CommunicationArtifact)) == 0
        assert (
            session.scalar(select(func.count()).select_from(CommunicationSourcePolicySnapshot)) == 0
        )


def test_retrieval_recurrence_is_preserved_without_new_artifact_version(session_factory):
    with session_factory() as session:
        first = record_communication_artifact_metadata(session, _event(), _artifact())
        second = record_communication_artifact_metadata(
            session,
            _event(),
            replace(_artifact(), retrieved_at=_at(31), metadata_known_at=_at(31)),
        )
        retrieval_count = session.scalar(
            select(func.count()).select_from(CommunicationArtifactRetrieval)
        )

    assert second.artifact_id == first.artifact_id
    assert retrieval_count == 2


def test_event_and_artifact_semantic_reversion_appends_a_new_occurrence(session_factory):
    reverted_known_at = datetime(2026, 8, 1, 12, tzinfo=UTC)
    corrected_url = "https://www.federalreserve.gov/mediacenter/files/transcript-corrected.pdf"

    with session_factory() as session:
        original = record_communication_artifact_metadata(session, _event(), _artifact())
        corrected = record_communication_artifact_metadata(
            session,
            _event(title="Corrected FOMC press conference", metadata_known_at=_at(31)),
            replace(
                _artifact(),
                artifact_url=corrected_url,
                retrieved_at=_at(31),
                metadata_known_at=_at(31),
            ),
        )
        reverted = record_communication_artifact_metadata(
            session,
            _event(metadata_known_at=reverted_known_at),
            replace(
                _artifact(),
                retrieved_at=reverted_known_at,
                metadata_known_at=reverted_known_at,
            ),
        )
        events = session.scalars(
            select(CommunicationEvent).order_by(CommunicationEvent.metadata_known_at)
        ).all()
        artifacts = session.scalars(
            select(CommunicationArtifact).order_by(CommunicationArtifact.metadata_known_at)
        ).all()

    assert len(events) == 3
    assert len(artifacts) == 3
    assert reverted.event_id not in {original.event_id, corrected.event_id}
    assert reverted.artifact_id not in {original.artifact_id, corrected.artifact_id}
    assert reverted.event_version_sha256 == original.event_version_sha256
    assert reverted.artifact_version_sha256 == original.artifact_version_sha256
    assert reverted.supersedes_event_id == corrected.event_id
    assert reverted.supersedes_artifact_id == corrected.artifact_id


def test_same_generic_artifact_key_in_distinct_events_does_not_cross_lineage(
    session_factory,
):
    with session_factory() as session:
        first = record_communication_artifact_metadata(session, _event(), _artifact())
        second = record_communication_artifact_metadata(
            session,
            _event(
                event_key="fomc_2026_09_16",
                title="September FOMC press conference",
                event_date=date(2026, 9, 16),
                metadata_known_at=datetime(2026, 9, 17, tzinfo=UTC),
            ),
            replace(
                _artifact(),
                published_at=datetime(2026, 9, 16, tzinfo=UTC),
                available_at=datetime(2026, 9, 16, tzinfo=UTC),
                retrieved_at=datetime(2026, 9, 17, tzinfo=UTC),
                metadata_known_at=datetime(2026, 9, 17, tzinfo=UTC),
            ),
        )

    assert first.supersedes_artifact_id is None
    assert second.supersedes_artifact_id is None


@pytest.mark.parametrize(
    "changes",
    [
        {"material_type": "subtitles"},
        {"provenance_tier": "official_authored_text"},
        {
            "origin_type": "automatic_caption",
            "provenance_tier": "official_hosted_automatic_caption",
        },
        {"translation_status": "machine_translation"},
        {"host_organization": ""},
        {"transcriber": "Vendor", "transcriber_attribution": "not_disclosed"},
    ],
)
def test_artifact_fidelity_fields_fail_closed(session_factory, changes):
    with session_factory() as session, pytest.raises(ValueError):
        record_communication_artifact_metadata(session, _event(), _artifact(**changes))


def test_official_video_has_a_non_text_media_provenance(session_factory):
    with session_factory() as session:
        result = record_communication_artifact_metadata(
            session,
            _event(),
            _artifact(
                artifact_key="official_video_en",
                artifact_role="webcast_video",
                material_type="press_conference_video",
                mime_type="video/mp4",
                origin_type="official_published_media",
                provenance_tier="official_published_media",
                transcriber_attribution="not_applicable",
            ),
        )
        artifact = session.get(CommunicationArtifact, result.artifact_id)

    assert artifact.origin_type == "official_published_media"
    assert artifact.provenance_tier == "official_published_media"


def test_metadata_helper_rejects_timezone_ambiguous_clocks(session_factory):
    with session_factory() as session, pytest.raises(ValueError, match="timezone"):
        record_communication_artifact_metadata(
            session,
            _event(metadata_known_at=datetime(2026, 7, 30, 12)),
            _artifact(),
        )


def test_changed_coverage_requires_explicit_lineage(session_factory):
    source = SOURCES["bhp_financial_results_en"]
    original = CommodityCoverageMeta(
        source_id=source.source_id,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        organization_id=source.organization_id,
        coverage_key="bhp_base_metals_producer",
        commodity_family="base_metals",
        exposure_role="producer",
        effective_from=date(2002, 1, 1),
        effective_to=None,
        available_at=_at(1),
        retrieved_at=_at(2),
        metadata_known_at=_at(2),
        evidence_url=source.landing_url,
    )
    with session_factory() as session:
        first_id = append_catalogue_commodity_coverage(session, original)
        with pytest.raises(ValueError, match="supersedes_coverage_id"):
            append_catalogue_commodity_coverage(
                session,
                replace(original, effective_from=date(2003, 1, 1), metadata_known_at=_at(3)),
            )
        second_id = append_catalogue_commodity_coverage(
            session,
            replace(
                original,
                effective_from=date(2003, 1, 1),
                metadata_known_at=_at(3),
                supersedes_coverage_id=first_id,
            ),
        )
        rows = session.scalars(select(OrganizationCommodityCoverage)).all()

    assert second_id != first_id
    assert len({row.coverage_version_sha256 for row in rows}) == 2
    assert rows[-1].effective_from == date(2003, 1, 1)


def test_coverage_semantic_reversion_appends_a_new_occurrence(session_factory):
    source = SOURCES["bhp_financial_results_en"]
    original = CommodityCoverageMeta(
        source_id=source.source_id,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        organization_id=source.organization_id,
        coverage_key="bhp_base_metals_producer",
        commodity_family="base_metals",
        exposure_role="producer",
        effective_from=date(2002, 1, 1),
        effective_to=None,
        available_at=_at(1),
        retrieved_at=_at(2),
        metadata_known_at=_at(2),
        evidence_url=source.landing_url,
    )

    with session_factory() as session:
        original_id = append_catalogue_commodity_coverage(session, original)
        corrected_id = append_catalogue_commodity_coverage(
            session,
            replace(
                original,
                effective_from=date(2003, 1, 1),
                metadata_known_at=_at(3),
                supersedes_coverage_id=original_id,
            ),
        )
        reverted_id = append_catalogue_commodity_coverage(
            session,
            replace(
                original,
                metadata_known_at=_at(4),
                supersedes_coverage_id=corrected_id,
            ),
        )
        rows = session.scalars(
            select(OrganizationCommodityCoverage).order_by(
                OrganizationCommodityCoverage.metadata_known_at
            )
        ).all()

    assert len(rows) == 3
    assert len({original_id, corrected_id, reverted_id}) == 3
    assert rows[2].coverage_version_sha256 == rows[0].coverage_version_sha256
    assert rows[2].supersedes_exposure_id == corrected_id
    assert rows[2].effective_from == original.effective_from


def test_distinct_coverage_keys_cannot_create_overlapping_current_mappings(
    session_factory,
):
    source = SOURCES["bhp_financial_results_en"]
    original = CommodityCoverageMeta(
        source_id=source.source_id,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        organization_id=source.organization_id,
        coverage_key="bhp_base_metals_producer",
        commodity_family="base_metals",
        exposure_role="producer",
        effective_from=date(2002, 1, 1),
        effective_to=None,
        available_at=_at(1),
        retrieved_at=_at(2),
        metadata_known_at=_at(2),
        evidence_url=source.landing_url,
    )
    with session_factory() as session:
        coverage_id = append_catalogue_commodity_coverage(session, original)
        with pytest.raises(ValueError, match="overlaps another current lineage"):
            append_catalogue_commodity_coverage(
                session,
                replace(
                    original,
                    coverage_key="bhp_metals_producer_duplicate",
                    effective_from=date(2020, 1, 1),
                    metadata_known_at=_at(3),
                ),
            )
        stored = session.get(OrganizationCommodityCoverage, coverage_id)
        values = {
            column.name: getattr(stored, column.name)
            for column in OrganizationCommodityCoverage.__table__.columns
            if column.name not in {"id", "created_at"}
        }
        values.update(
            coverage_key="raw_overlapping_mapping",
            effective_from=date(2020, 1, 1),
            metadata_known_at=_naive_at(3),
            coverage_version_sha256="5" * 64,
            supersedes_exposure_id=None,
        )
        session.add(OrganizationCommodityCoverage(**values))
        with pytest.raises(IntegrityError, match="overlapping commodity coverage"):
            session.flush()


def test_finalization_contract_is_structurally_guarded(session_factory):
    with session_factory() as session:
        trigger_sql = session.execute(
            text(
                "SELECT group_concat(sql, ' ') FROM sqlite_master "
                "WHERE type = 'trigger' AND name IN "
                "('communication_extraction_finalization_validate', "
                "'communication_segments_reject_after_finalization')"
            )
        ).scalar_one()
    assert "COUNT(*)" in trigger_sql
    assert "MIN(ordinal)" in trigger_sql
    assert "MAX(ordinal)" in trigger_sql
    assert "SUM(char_count)" in trigger_sql
    assert "finalized communication extraction" in trigger_sql


def test_extraction_identity_allows_distinct_configs_and_runs():
    constraint = next(
        item
        for item in CommunicationExtraction.__table__.constraints
        if item.name == "uq_communication_extraction_version"
    )

    assert tuple(column.name for column in constraint.columns) == (
        "artifact_content_id",
        "run_key",
    )


def test_content_captures_append_when_same_official_url_changes_bytes(session_factory):
    with session_factory() as session:
        artifact, first_retrieval = _seed_cleared_artifact(session)
        first = CommunicationArtifactContent(
            artifact_id=artifact.id,
            retrieval_id=first_retrieval.id,
            content_sha256="1" * 64,
            size_bytes=3,
            blob_path=("artifacts/communications/sha256/11/" + "1" * 64),
            captured_at=_naive_at(30),
            supersedes_content_id=None,
        )
        session.add(first)
        session.flush()
        for run_key in ("manual_extract_first", "manual_extract_repeat"):
            session.add(
                CommunicationExtraction(
                    artifact_content_id=first.id,
                    run_key=run_key,
                    extractor_name="test_extractor",
                    extractor_version="1.0",
                    extractor_config_sha256="6" * 64,
                    extracted_at=_naive_at(31),
                    run_sha256="7" * 64,
                )
            )
        session.flush()
        second_retrieval = CommunicationArtifactRetrieval(
            artifact_id=artifact.id,
            retrieved_at=_naive_at(31),
            metadata_known_at=_naive_at(31),
            landing_url=artifact.landing_url,
            artifact_url=artifact.artifact_url,
        )
        session.add(second_retrieval)
        session.flush()
        second = CommunicationArtifactContent(
            artifact_id=artifact.id,
            retrieval_id=second_retrieval.id,
            content_sha256="2" * 64,
            size_bytes=4,
            blob_path=("artifacts/communications/sha256/22/" + "2" * 64),
            captured_at=_naive_at(31),
            supersedes_content_id=first.id,
        )
        session.add(second)
        session.flush()

        assert second.id != first.id
        assert second.supersedes_content_id == first.id
        assert second.artifact_id == first.artifact_id
        assert second.content_sha256 != first.content_sha256
        assert session.scalar(select(func.count()).select_from(CommunicationExtraction)) == 2


def test_content_capture_requires_a_storage_enabling_rights_policy(session_factory):
    with session_factory() as session:
        result = record_communication_artifact_metadata(session, _event(), _artifact())
        retrieval = session.scalar(
            select(CommunicationArtifactRetrieval).where(
                CommunicationArtifactRetrieval.artifact_id == result.artifact_id
            )
        )
        session.add(
            CommunicationArtifactContent(
                artifact_id=result.artifact_id,
                retrieval_id=retrieval.id,
                content_sha256="3" * 64,
                size_bytes=3,
                blob_path=("artifacts/communications/sha256/33/" + "3" * 64),
                captured_at=_naive_at(30),
                supersedes_content_id=None,
            )
        )
        with pytest.raises(IntegrityError, match="rights policy"):
            session.flush()


@pytest.mark.parametrize(
    "changes",
    [
        {
            "artifact_role": "annual_report",
            "material_type": "annual_report",
            "origin_type": "publisher_authored",
            "provenance_tier": "official_authored_text",
            "transcriber_attribution": "not_applicable",
        },
        {
            "artifact_url": "https://evil.example/transcript.pdf",
        },
    ],
)
def test_database_policy_trigger_rejects_raw_artifact_policy_bypasses(session_factory, changes):
    with session_factory() as session:
        result = record_communication_artifact_metadata(session, _event(), _artifact())
        original = session.get(CommunicationArtifact, result.artifact_id)
        values = {
            column.name: getattr(original, column.name)
            for column in CommunicationArtifact.__table__.columns
            if column.name not in {"id", "created_at"}
        }
        values.update(
            artifact_key="raw_policy_bypass",
            artifact_version_sha256="4" * 64,
            supersedes_artifact_id=None,
            **changes,
        )
        session.add(CommunicationArtifact(**values))

        with pytest.raises(IntegrityError, match="source policy"):
            session.flush()


def test_init_db_refreshes_custom_communication_trigger_bodies(tmp_path):
    engine = make_engine(tmp_path / "stale-trigger.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER communication_artifacts_match_policy")
        connection.exec_driver_sql(
            """
            CREATE TRIGGER communication_artifacts_match_policy
            BEFORE INSERT ON communication_artifacts
            BEGIN
                SELECT 1;
            END
            """
        )

    init_db(engine)
    with engine.connect() as connection:
        trigger_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' "
            "AND name = 'communication_artifacts_match_policy'"
        ).scalar_one()

    assert "json_each(policy.material_types_json)" in trigger_sql
    assert "NEW.artifact_url" in trigger_sql
    assert "policy.rights_note = NEW.rights_note" in trigger_sql


def test_init_db_refuses_an_unknown_existing_communication_table_layout(tmp_path):
    engine = make_engine(tmp_path / "old-communications-schema.db")
    with engine.begin() as connection:
        connection.exec_driver_sql("CREATE TABLE communication_events (id INTEGER PRIMARY KEY)")

    with pytest.raises(RuntimeError, match="institutional-communications schema"):
        init_db(engine)

    with engine.connect() as connection:
        organizations_exists = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'table' AND name = 'organizations'"
        ).scalar_one()
    assert organizations_exists == 0
