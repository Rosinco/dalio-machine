"""Immutable, rights-gated institutional communication metadata."""

from dataclasses import replace
from datetime import UTC, date, datetime
from types import MappingProxyType

import pytest
from sqlalchemy import func, select

import dalio.communications.catalogue as catalogue_module
from dalio.communications.catalogue import (
    CATALOGUE_SCHEMA_VERSION,
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_CATALOGUE_SNAPSHOTS,
    COMMUNICATION_SOURCES,
    communication_catalogue_snapshot,
)
from dalio.storage.communications import (
    CommodityCoverageMeta,
    CommunicationArtifactMeta,
    CommunicationEventMeta,
    append_catalogue_commodity_coverage,
    record_communication_artifact_metadata,
)
from dalio.storage.db import (
    CommunicationArtifact,
    CommunicationArtifactContent,
    CommunicationEvent,
    CommunicationSourcePolicySnapshot,
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
    engine = make_engine(tmp_path / "communications.db")
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
        "event_started_at": _at(29),
        "reference_start": date(2026, 7, 28),
        "reference_end": date(2026, 7, 29),
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
        "published_at": _at(29),
        "available_at": _at(29),
        "retrieved_at": _at(30),
        "metadata_known_at": _at(30),
        "landing_url": source.landing_url,
        "artifact_url": (
            "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20260729.pdf"
        ),
    }
    values.update(changes)
    return CommunicationArtifactMeta(**values)


def test_metadata_record_is_catalogue_bound_idempotent_and_has_artifact_clocks(
    session_factory,
):
    with session_factory() as session:
        first = record_communication_artifact_metadata(session, _event(), _artifact())
        again = record_communication_artifact_metadata(
            session,
            _event(),
            replace(
                _artifact(),
                retrieved_at=datetime(2026, 8, 1, tzinfo=UTC),
                metadata_known_at=datetime(2026, 8, 1, tzinfo=UTC),
            ),
        )
        artifact = session.get(CommunicationArtifact, first.artifact_id)
        event = session.get(CommunicationEvent, first.event_id)
        content_count = session.scalar(
            select(func.count()).select_from(CommunicationArtifactContent)
        )

    assert first.event_created and first.artifact_created
    assert again.event_id == first.event_id and again.artifact_id == first.artifact_id
    assert not again.event_created and not again.artifact_created
    assert artifact.catalogue_sha256 == COMMUNICATION_CATALOGUE_SHA256
    assert content_count == 0
    assert artifact.publisher == "Board of Governors of the Federal Reserve System"
    assert artifact.transcriber is None
    assert artifact.published_at == _at(29).replace(tzinfo=None)
    assert artifact.available_at == _at(29).replace(tzinfo=None)
    assert artifact.retrieved_at == _at(30).replace(tzinfo=None)
    assert not hasattr(event, "available_at")


def test_metadata_record_persists_the_exact_registered_catalogue_snapshot(
    session_factory,
    monkeypatch,
):
    current_source = SOURCES["fed_fomc_press_conferences_en"]
    historical_source = replace(
        current_source,
        source_id="fed_historical_press_conferences_en",
        publisher="Historical Board of Governors",
    )
    historical_evaluated_at = datetime(2025, 1, 15, 9, tzinfo=UTC)
    historical_sources = tuple(
        historical_source if source.source_id == current_source.source_id else source
        for source in COMMUNICATION_SOURCES
    )
    historical_snapshot = communication_catalogue_snapshot(
        historical_sources,
        schema_version=CATALOGUE_SCHEMA_VERSION,
        evaluated_at=historical_evaluated_at,
    )
    historical_hash = historical_snapshot.catalogue_sha256
    monkeypatch.setattr(
        catalogue_module,
        "COMMUNICATION_CATALOGUE_SNAPSHOTS",
        MappingProxyType(
            {
                **COMMUNICATION_CATALOGUE_SNAPSHOTS,
                historical_hash: historical_snapshot,
            }
        ),
    )
    historical_artifact = replace(
        _artifact(),
        source_id=historical_source.source_id,
        catalogue_sha256=historical_hash,
        publisher=historical_source.publisher,
    )
    coverage_source = SOURCES["bhp_financial_results_en"]
    historical_coverage = CommodityCoverageMeta(
        source_id=coverage_source.source_id,
        catalogue_sha256=historical_hash,
        organization_id=coverage_source.organization_id,
        coverage_key="bhp_historical_base_metals_producer",
        commodity_family="base_metals",
        exposure_role="producer",
        effective_from=date(2002, 1, 1),
        available_at=_at(1),
        retrieved_at=_at(2),
        metadata_known_at=_at(2),
        evidence_url=coverage_source.landing_url,
    )

    with session_factory() as session:
        result = record_communication_artifact_metadata(
            session,
            _event(),
            historical_artifact,
        )
        artifact = session.get(CommunicationArtifact, result.artifact_id)
        policy = session.get(
            CommunicationSourcePolicySnapshot,
            (historical_hash, historical_source.source_id),
        )
        coverage_id = append_catalogue_commodity_coverage(session, historical_coverage)
        coverage = session.get(OrganizationCommodityCoverage, coverage_id)
        coverage_policy = session.get(
            CommunicationSourcePolicySnapshot,
            (historical_hash, coverage_source.source_id),
        )

    assert artifact.catalogue_sha256 == historical_hash
    assert artifact.source_id == historical_source.source_id
    assert policy.catalogue_sha256 == historical_hash
    assert policy.catalogue_evaluated_at == historical_evaluated_at.replace(tzinfo=None)
    assert policy.source_id == historical_source.source_id
    assert policy.publisher == historical_source.publisher
    assert coverage.catalogue_sha256 == historical_hash
    assert coverage.source_id == coverage_source.source_id
    assert coverage_policy.catalogue_evaluated_at == historical_evaluated_at.replace(tzinfo=None)
    assert coverage_policy.coverage_note == coverage_source.coverage_note

    with (
        session_factory() as session,
        pytest.raises(
            ValueError,
            match="not in the bound communication catalogue snapshot",
        ),
    ):
        record_communication_artifact_metadata(
            session,
            _event(),
            replace(historical_artifact, catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256),
        )


def test_event_and_artifact_corrections_append_successor_versions(session_factory):
    with session_factory() as session:
        first = record_communication_artifact_metadata(session, _event(), _artifact())
        corrected = record_communication_artifact_metadata(
            session,
            _event(
                title="FOMC press conference — corrected title",
                metadata_known_at=_at(31),
            ),
            replace(
                _artifact(),
                artifact_url=(
                    "https://www.federalreserve.gov/mediacenter/files/"
                    "FOMCpresconf20260729-corrected.pdf"
                ),
                available_at=_at(30),
                retrieved_at=_at(31),
                metadata_known_at=_at(31),
            ),
        )
        events = (
            session.execute(select(CommunicationEvent).order_by(CommunicationEvent.id))
            .scalars()
            .all()
        )

    assert corrected.event_id != first.event_id
    assert corrected.supersedes_event_id == first.event_id
    assert corrected.supersedes_artifact_id == first.artifact_id
    assert corrected.event_version_sha256 != first.event_version_sha256
    assert len(events) == 2


def test_historical_artifact_keeps_its_period_publisher_not_current_archive_name(
    session_factory,
):
    source = SOURCES["bhp_financial_results_en"]
    event = CommunicationEventMeta(
        organization_id=source.organization_id,
        event_key="annual_results_2002",
        event_type="annual_results",
        title="BHP Billiton 2002 annual results",
        event_date=date(2002, 8, 19),
        metadata_known_at=_at(30),
        reference_start=date(2001, 7, 1),
        reference_end=date(2002, 6, 30),
    )
    artifact = CommunicationArtifactMeta(
        source_id=source.source_id,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        artifact_key="results_transcript_en",
        artifact_role="full_transcript",
        material_type="results_transcript",
        language="en",
        translation_status="original",
        mime_type="application/pdf",
        origin_type="official_published_transcript",
        provenance_tier="official_published_transcript",
        host_organization=source.host_organization,
        publisher="BHP Billiton Limited",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        rights_status=source.rights_status,
        acquisition_status=source.acquisition_status,
        rights_checked_by="human:test_reviewer",
        rights_checked_at=_at(30),
        published_at=datetime(2002, 8, 19, tzinfo=UTC),
        available_at=datetime(2002, 8, 19, tzinfo=UTC),
        retrieved_at=_at(30),
        metadata_known_at=_at(30),
        landing_url=source.landing_url,
        artifact_url=source.landing_url,
    )
    with session_factory() as session:
        result = record_communication_artifact_metadata(session, event, artifact)
        stored = session.get(CommunicationArtifact, result.artifact_id)

    assert stored.publisher == "BHP Billiton Limited"
    assert stored.host_organization == source.host_organization


@pytest.mark.parametrize(
    ("event", "artifact", "message"),
    [
        (
            _event(organization_id="ecb"),
            _artifact(),
            "does not belong",
        ),
        (
            _event(),
            replace(_artifact(), catalogue_sha256="0" * 64),
            "known communication catalogue snapshot",
        ),
        (
            _event(),
            replace(_artifact(), rights_status="cleared"),
            "rights_status",
        ),
        (
            _event(),
            replace(_artifact(), artifact_url="https://example.com/transcript.pdf"),
            "official domain",
        ),
        (
            _event(),
            replace(
                _artifact(),
                artifact_url="https://www.federalreserve.gov.:443/transcript.pdf",
            ),
            "explicit port|trailing-dot",
        ),
        (
            _event(),
            replace(
                _artifact(),
                artifact_url="https://www.federalreserve.gov/transcript.pdf\nignored",
            ),
            "control characters",
        ),
        (
            _event(event_date=_at(29)),
            _artifact(),
            "exact date",
        ),
    ],
)
def test_metadata_record_fails_closed_on_source_policy_conflicts(
    session_factory, event, artifact, message
):
    with session_factory() as session, pytest.raises(ValueError, match=message):
        record_communication_artifact_metadata(session, event, artifact)


def test_commodity_mapping_is_explicitly_selection_taxonomy(session_factory):
    source = SOURCES["bhp_financial_results_en"]
    meta = CommodityCoverageMeta(
        source_id=source.source_id,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        organization_id=source.organization_id,
        coverage_key="bhp_base_metals_producer",
        commodity_family="base_metals",
        exposure_role="producer",
        effective_from=date(2002, 1, 1),
        available_at=_at(1),
        retrieved_at=_at(2),
        metadata_known_at=_at(2),
        evidence_url=source.landing_url,
    )
    with session_factory() as session:
        coverage_id = append_catalogue_commodity_coverage(session, meta)
        again = append_catalogue_commodity_coverage(session, meta)
        coverage = session.get(OrganizationCommodityCoverage, coverage_id)

    assert again == coverage_id
    assert coverage.mapping_status == "selection_taxonomy"
    assert coverage.coverage_key == "bhp_base_metals_producer"
    assert coverage.reviewed_by is None and coverage.reviewed_at is None
    assert coverage.catalogue_sha256 == COMMUNICATION_CATALOGUE_SHA256


def test_immutable_rows_reject_in_place_changes(session_factory):
    with session_factory() as session:
        result = record_communication_artifact_metadata(session, _event(), _artifact())
        event = session.get(CommunicationEvent, result.event_id)
        event.title = "silently rewritten"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
