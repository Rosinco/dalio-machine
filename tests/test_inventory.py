"""Read-only coverage inventory for the macro observatory."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from types import MappingProxyType

import pytest
from sqlalchemy import insert, select
from sqlalchemy.exc import IntegrityError

from dalio.communications.catalogue import (
    CATALOGUE_SCHEMA_VERSION as COMMUNICATION_CATALOGUE_SCHEMA_VERSION,
)
from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_CATALOGUE_SNAPSHOTS,
    COMMUNICATION_SOURCES,
    CommunicationSourceSpec,
    communication_catalogue_snapshot,
)
from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.imf_bop import BOP_COUNTRIES, BOP_SERIES
from dalio.data_sources.imf_positions import (
    DIP_FREQUENCIES,
    DIP_SERIES,
    IMF_POSITION_COUNTRIES,
    PIP_FREQUENCIES,
    PIP_SERIES,
)
from dalio.data_sources.money_liquidity import (
    BOE_M4EX_QUARTERLY,
    FED_M2,
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    MONEY_LIQUIDITY_SERIES,
    money_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    OFR_SHADOW_LIQUIDITY_SERIES,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.data_sources.worldbank_commodities import (
    CATALOGUE_GENERATOR_VERSION,
    CATALOGUE_SCHEMA_VERSION,
    SOURCE_WORLD_BANK_COMMODITIES,
    canonical_series_id,
    pink_sheet_vintage_label,
)
from dalio.data_sources.worldbank_qpsd import QPSD_COUNTRIES, QPSD_SERIES
from dalio.storage import inventory as inventory_module
from dalio.storage.communications import (
    CommunicationArtifactMeta,
    CommunicationEventMeta,
    CommunicationSectionScopeMeta,
    record_communication_artifact_metadata,
)
from dalio.storage.db import (
    AllocatorFact,
    Claim,
    ClaimCitation,
    CommunicationArtifact,
    CommunicationArtifactContent,
    CommunicationArtifactRetrieval,
    CommunicationArtifactSectionScopeSet,
    CommunicationEvent,
    CommunicationExtraction,
    CommunicationExtractionFinalization,
    CommunicationSegment,
    CommunicationSourcePolicySnapshot,
    DataRelease,
    DataReleaseArtifact,
    DebtHolderPosition,
    DocumentExtraction,
    DocumentPage,
    Observation,
    Organization,
    ReleaseObservation,
    ReportDocument,
    init_db,
    make_engine,
    make_session_factory,
)
from dalio.storage.inventory import build_observatory_inventory, render_inventory_summary
from dalio.storage.releases import make_partition_key


def _release(**overrides) -> dict:
    values = {
        "partition_key": "test:partition",
        "source_family": "TEST",
        "published_at": None,
        "available_at": datetime(2026, 8, 1, tzinfo=UTC),
        "retrieved_at": datetime(2026, 8, 2, tzinfo=UTC),
        "vintage_label": None,
        "source_url": "https://example.test/data",
        "content_sha256": "1" * 64,
        "row_count": 1,
    }
    values.update(overrides)
    return values


def _monthly_dates(start: date, end: date) -> list[date]:
    dates = []
    cursor = start
    while cursor <= end:
        dates.append(cursor)
        cursor = (
            date(cursor.year + 1, 1, 1)
            if cursor.month == 12
            else date(cursor.year, cursor.month + 1, 1)
        )
    return dates


def _money_observations(dates: list[date], spec=FED_M2) -> list[dict]:
    return [
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": observed_on,
            "value": float(index),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
        for index, observed_on in enumerate(dates, start=1)
    ]


def _shadow_observations(spec, dates: list[date]) -> list[dict]:
    return [
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": observed_on,
            "value": float(index),
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        }
        for index, observed_on in enumerate(dates, start=1)
    ]


def _shadow_release(
    spec,
    prefix: str,
    digest: str,
    row_count: int,
    *,
    payload_tag: str | None = None,
) -> dict:
    vintage_label = f"{prefix}{digest}"
    if payload_tag is not None:
        vintage_label = f"{vintage_label};p:{payload_tag}"
    return _release(
        partition_key=make_partition_key(
            spec.source_family,
            spec.native_series_id,
            spec.country,
            spec.indicator,
        ),
        source_family=spec.source_family,
        vintage_label=vintage_label,
        content_sha256=spec.native_series_id.encode().hex().ljust(64, "0")[:64],
        row_count=row_count,
    )


def _canonical_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _register_test_communication_catalogue(monkeypatch):
    evaluated_at = datetime(2026, 1, 10, tzinfo=UTC)
    source = CommunicationSourceSpec(
        source_id="test_bank_letters",
        organization_id="test_bank",
        organization_name="Test Bank",
        organization_type="bank",
        jurisdiction="US",
        language="en",
        landing_url="https://example.test/letters",
        official_domains=("example.test",),
        host_organization="Test Bank",
        publisher="Test Bank",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("ceo_letter",),
        commodity_families=(),
        verified_archive_start_year=2000,
        coverage_note="Official annual letters.",
        provenance_tier="official_authored_text",
        rights_status="cleared",
        rights_basis_url="https://example.test/terms",
        rights_note="Test rights review permits this fixture.",
        acquisition_status="manual_collection_ready",
        acquisition_note="Manual test capture only.",
        automated_collection_allowed=False,
        rights_checked_by="human:test_reviewer",
        rights_checked_at=datetime(2026, 1, 1, tzinfo=UTC),
    )
    import dalio.communications.catalogue as catalogue_module

    monkeypatch.setattr(
        catalogue_module,
        "_APPROVED_OFFICIAL_DOMAINS",
        frozenset({*catalogue_module._APPROVED_OFFICIAL_DOMAINS, "example.test"}),
    )
    snapshot = communication_catalogue_snapshot(
        (source,),
        schema_version=COMMUNICATION_CATALOGUE_SCHEMA_VERSION,
        evaluated_at=evaluated_at,
    )
    monkeypatch.setattr(
        catalogue_module,
        "COMMUNICATION_CATALOGUE_SNAPSHOTS",
        MappingProxyType(
            {
                **COMMUNICATION_CATALOGUE_SNAPSHOTS,
                snapshot.catalogue_sha256: snapshot,
            }
        ),
    )
    return snapshot


def _trust_test_communication_catalogue(monkeypatch) -> str:
    snapshot = _register_test_communication_catalogue(monkeypatch)
    monkeypatch.setattr(
        inventory_module,
        "_communication_catalogue_summary",
        lambda: {
            "validation_status": "valid",
            "source_policy_count": 1,
            "organization_count": 1,
            "sha256": snapshot.catalogue_sha256,
            "error": None,
        },
    )
    return snapshot.catalogue_sha256


def _insert_communication_corpus(
    engine,
    *,
    finalize: bool,
    catalogue_sha256: str = "a" * 64,
    policy_publisher: str = "Test Bank",
    valid_text_hash: bool = True,
    valid_corpus_hash: bool = True,
    segment_kind: str = "letter",
    speaker_side: str = "publisher",
) -> Path:
    source_id = "test_bank_letters"
    organization_id = "test_bank"
    checked_at = datetime(2026, 1, 1)
    evaluated_at = datetime(2026, 1, 10)
    landing_url = "https://example.test/letters"
    rights_basis_url = "https://example.test/terms"
    policy_values = {
        "catalogue_sha256": catalogue_sha256,
        "source_id": source_id,
        "organization_id": organization_id,
        "organization_name": "Test Bank",
        "organization_type": "bank",
        "jurisdiction": "US",
        "language": "en",
        "landing_url": landing_url,
        "official_domains_json": '["example.test"]',
        "host_organization": "Test Bank",
        "publisher": policy_publisher,
        "transcriber": None,
        "transcriber_attribution": "not_applicable",
        "material_types_json": '["ceo_letter"]',
        "commodity_families_json": "[]",
        "verified_archive_start_year": 2000,
        "coverage_note": "Official annual letters.",
        "source_provenance_tier": "official_authored_text",
        "rights_status": "cleared",
        "rights_basis_url": rights_basis_url,
        "rights_note": "Test rights review permits this fixture.",
        "acquisition_status": "manual_collection_ready",
        "acquisition_note": "Manual test capture only.",
        "automated_collection_allowed": False,
        "rights_checked_by": "human:test_reviewer",
        "rights_checked_at": checked_at,
        "catalogue_evaluated_at": evaluated_at,
    }
    policy_payload = {
        "source_id": source_id,
        "organization_id": organization_id,
        "organization_name": "Test Bank",
        "organization_type": "bank",
        "jurisdiction": "US",
        "language": "en",
        "landing_url": landing_url,
        "official_domains": ["example.test"],
        "host_organization": "Test Bank",
        "publisher": policy_publisher,
        "transcriber": None,
        "transcriber_attribution": "not_applicable",
        "material_types": ["ceo_letter"],
        "commodity_families": [],
        "verified_archive_start_year": 2000,
        "coverage_note": "Official annual letters.",
        "provenance_tier": "official_authored_text",
        "rights_status": "cleared",
        "rights_basis_url": rights_basis_url,
        "rights_note": "Test rights review permits this fixture.",
        "acquisition_status": "manual_collection_ready",
        "acquisition_note": "Manual test capture only.",
        "automated_collection_allowed": False,
        "rights_checked_by": "human:test_reviewer",
        "rights_checked_at": "2026-01-01T00:00:00Z",
        "catalogue_sha256": catalogue_sha256,
        "catalogue_evaluated_at": "2026-01-10T00:00:00Z",
    }
    policy_values["policy_sha256"] = _canonical_sha256(policy_payload)

    event_semantic = {
        "organization_id": organization_id,
        "event_key": "annual_letter_2025",
        "event_type": "annual_report",
        "title": "2025 annual letter",
        "event_date": "2026-01-02",
        "event_started_at": None,
        "reference_start": "2025-01-01",
        "reference_end": "2025-12-31",
    }
    event_sha256 = _canonical_sha256(event_semantic)
    event_known_at = datetime(2026, 1, 2)
    artifact_published_at = datetime(2026, 1, 3)
    artifact_available_at = datetime(2026, 1, 3)
    artifact_retrieved_at = datetime(2026, 1, 4)
    artifact_known_at = datetime(2026, 1, 4)
    artifact_url = "https://example.test/letters/2025.txt"
    section_scope = {
        "section_ordinal": 1,
        "scope_key": "ceo_letter",
        "artifact_role": "ceo_letter",
        "material_type": "ceo_letter",
        "origin_type": "publisher_authored",
        "provenance_tier": "official_authored_text",
        "transcriber": None,
        "transcriber_attribution": "not_applicable",
    }
    artifact_semantic = {
        "event_version_sha256": event_sha256,
        "source_id": source_id,
        "catalogue_sha256": catalogue_sha256,
        "artifact_key": "official_letter_en",
        "artifact_role": "ceo_letter",
        "material_type": "ceo_letter",
        "language": "en",
        "translation_status": "original",
        "mime_type": "text/plain",
        "origin_type": "publisher_authored",
        "provenance_tier": "official_authored_text",
        "rights_status": "cleared",
        "acquisition_status": "manual_collection_ready",
        "rights_checked_by": "human:test_reviewer",
        "rights_checked_at": checked_at.isoformat(),
        "host_organization": "Test Bank",
        "publisher": "Test Bank",
        "transcriber": None,
        "transcriber_attribution": "not_applicable",
        "published_at": artifact_published_at.isoformat(),
        "available_at": artifact_available_at.isoformat(),
        "section_scopes": [section_scope],
        "landing_url": landing_url,
        "artifact_url": artifact_url,
    }
    artifact_sha256 = _canonical_sha256(artifact_semantic)

    content_bytes = "Försiktig återhämtning — inflationen avtar.".encode()
    content_sha256 = hashlib.sha256(content_bytes).hexdigest()
    relative_blob = Path("artifacts/communications/sha256") / content_sha256[:2] / content_sha256
    database_path = Path(engine.url.database)
    blob_path = database_path.parent / relative_blob
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    blob_path.write_bytes(content_bytes)

    segment_text = content_bytes.decode()
    segment = {
        "ordinal": 1,
        "section_ordinal": 1,
        "segment_kind": segment_kind,
        "speaker_name": "Test CEO",
        "speaker_role": "Chief Executive Officer",
        "speaker_side": speaker_side,
        "section_title": None,
        "text": segment_text,
        "page_start": 1,
        "page_end": 1,
        "paragraph_start": 1,
        "paragraph_end": 1,
        "start_ms": None,
        "end_ms": None,
    }
    corpus_sha256 = _canonical_sha256(
        {
            "canonicalization": "communication_segments_json_v2",
            "segments": [segment],
        }
    )

    with engine.begin() as connection:
        connection.execute(insert(Organization).values(organization_id=organization_id))
        connection.execute(insert(CommunicationSourcePolicySnapshot).values(**policy_values))
        event_id = connection.execute(
            insert(CommunicationEvent).values(
                organization_id=organization_id,
                event_key="annual_letter_2025",
                event_type="annual_report",
                title="2025 annual letter",
                event_date=date(2026, 1, 2),
                event_started_at=None,
                reference_start=date(2025, 1, 1),
                reference_end=date(2025, 12, 31),
                metadata_known_at=event_known_at,
                event_version_sha256=event_sha256,
            )
        ).inserted_primary_key[0]
        artifact_id = connection.execute(
            insert(CommunicationArtifact).values(
                event_id=event_id,
                source_id=source_id,
                catalogue_sha256=catalogue_sha256,
                event_version_sha256=event_sha256,
                artifact_key="official_letter_en",
                artifact_role="ceo_letter",
                material_type="ceo_letter",
                language="en",
                translation_status="original",
                mime_type="text/plain",
                origin_type="publisher_authored",
                provenance_tier="official_authored_text",
                rights_status="cleared",
                acquisition_status="manual_collection_ready",
                rights_basis_url=rights_basis_url,
                rights_note="Test rights review permits this fixture.",
                rights_checked_by="human:test_reviewer",
                rights_checked_at=checked_at,
                host_organization="Test Bank",
                publisher="Test Bank",
                transcriber=None,
                transcriber_attribution="not_applicable",
                published_at=artifact_published_at,
                available_at=artifact_available_at,
                retrieved_at=artifact_retrieved_at,
                metadata_known_at=artifact_known_at,
                landing_url=landing_url,
                artifact_url=artifact_url,
                artifact_version_sha256=artifact_sha256,
            )
        ).inserted_primary_key[0]
        scopes_json = json.dumps(
            [section_scope],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        scope_set_sha256 = _canonical_sha256(
            {
                "artifact_version_sha256": artifact_sha256,
                "canonicalization_version": ("communication_artifact_section_scopes_json_v1"),
                "section_scopes": [section_scope],
            }
        )
        connection.execute(
            insert(CommunicationArtifactSectionScopeSet).values(
                artifact_id=artifact_id,
                artifact_version_sha256=artifact_sha256,
                scope_count=1,
                scopes_json=scopes_json,
                scope_set_sha256=scope_set_sha256,
                metadata_known_at=artifact_known_at,
                canonicalization_version=("communication_artifact_section_scopes_json_v1"),
            )
        )
        retrieval_id = connection.execute(
            insert(CommunicationArtifactRetrieval).values(
                artifact_id=artifact_id,
                retrieved_at=artifact_retrieved_at,
                metadata_known_at=artifact_known_at,
                landing_url=landing_url,
                artifact_url=artifact_url,
            )
        ).inserted_primary_key[0]
        content_id = connection.execute(
            insert(CommunicationArtifactContent).values(
                artifact_id=artifact_id,
                retrieval_id=retrieval_id,
                content_sha256=content_sha256,
                size_bytes=len(content_bytes),
                blob_path=relative_blob.as_posix(),
                captured_at=datetime(2026, 1, 5),
            )
        ).inserted_primary_key[0]
        extraction_id = connection.execute(
            insert(CommunicationExtraction).values(
                artifact_content_id=content_id,
                run_key="test_run_1",
                extractor_name="test_plain_text",
                extractor_version="1",
                extractor_config_sha256="b" * 64,
                extracted_at=datetime(2026, 1, 6),
                run_sha256="c" * 64,
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(CommunicationSegment).values(
                extraction_id=extraction_id,
                **segment,
                text_sha256=(
                    hashlib.sha256(segment_text.encode()).hexdigest()
                    if valid_text_hash
                    else "0" * 64
                ),
                char_count=len(segment_text),
            )
        )
        if finalize:
            connection.execute(
                insert(CommunicationExtractionFinalization).values(
                    extraction_id=extraction_id,
                    finalized_at=datetime(2026, 1, 7),
                    segment_count=1,
                    total_char_count=len(segment_text),
                    corpus_sha256=corpus_sha256 if valid_corpus_hash else "0" * 64,
                    canonicalization_version="communication_segments_json_v2",
                )
            )
    return blob_path


def _tamper_immutable_communication_row(
    engine,
    *,
    table_name: str,
    operation: str,
    sql: str,
) -> None:
    trigger_name = f"{table_name}_reject_{operation}"
    with engine.begin() as connection:
        trigger_sql = connection.exec_driver_sql(
            "SELECT sql FROM sqlite_master WHERE type = 'trigger' AND name = ?",
            (trigger_name,),
        ).scalar_one()
        connection.exec_driver_sql(f"DROP TRIGGER {trigger_name}")
        connection.exec_driver_sql("PRAGMA ignore_check_constraints=ON")
        try:
            connection.exec_driver_sql(sql)
        finally:
            connection.exec_driver_sql("PRAGMA ignore_check_constraints=OFF")
            connection.exec_driver_sql(trigger_sql)


def _insert_test_release_artifacts(
    connection, release_id: int, spec, artifact_root
) -> dict[str, Path]:
    series_root = artifact_root / spec.native_series_id
    series_root.mkdir(parents=True, exist_ok=True)
    provenance_json = json.dumps(
        {"native_series_id": spec.native_series_id, "records": []},
        sort_keys=True,
        separators=(",", ":"),
    )
    missing_hash = hashlib.sha256(provenance_json.encode()).hexdigest()
    native_bytes = json.dumps(
        {"native_series_id": spec.native_series_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    native_hash = hashlib.sha256(native_bytes).hexdigest()
    native_path = series_root / "native.json"
    native_path.write_bytes(native_bytes)
    source_bytes = native_bytes if spec.source_family == "BIS_GLI" else b'{"bulk":"response"}'
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_path = series_root / "source.payload"
    source_path.write_bytes(source_bytes)
    artifacts = [("source_response", source_hash, source_path)]
    if spec.source_family != "BIS_GLI":
        missing_path = series_root / "missing.json"
        missing_path.write_text(provenance_json)
        artifacts.extend(
            [
                ("native_series_payload", native_hash, native_path),
                ("missingness_ledger", missing_hash, missing_path),
            ]
        )
    connection.execute(
        insert(DataReleaseArtifact),
        [
            {
                "release_id": release_id,
                "role": role,
                "artifact_sha256": artifact_hash,
                "artifact_path": str(path.resolve()),
                "native_payload_sha256": native_hash,
                "missing_provenance_sha256": missing_hash,
                "provenance_json": provenance_json,
            }
            for role, artifact_hash, path in artifacts
        ],
    )
    return {role: path for role, _artifact_hash, path in artifacts}


def test_empty_inventory_is_json_safe_and_distinguishes_absent_tables(tmp_path):
    engine = make_engine(tmp_path / "empty.db")
    init_db(engine)

    inventory = build_observatory_inventory(engine)

    assert json.loads(json.dumps(inventory)) == inventory
    assert inventory["schema_version"] == 1
    assert inventory["observations"]["row_count"] == 0
    assert inventory["releases"]["release_count"] == 0
    assert inventory["qpsd"]["expected_partitions"] == len(QPSD_COUNTRIES) * len(QPSD_SERIES)
    assert inventory["qpsd"]["stored_partitions"] == 0
    assert inventory["qpsd"]["not_reported_partitions"] == 0
    assert inventory["qpsd"]["missing_partitions"] == len(QPSD_COUNTRIES) * len(QPSD_SERIES)
    assert inventory["imf_bop"]["expected_partitions"] == len(BOP_COUNTRIES) * len(BOP_SERIES)
    assert inventory["imf_bop"]["missing_partitions"] == len(BOP_COUNTRIES) * len(BOP_SERIES)
    assert inventory["cross_border_positions"]["table_present"] is True
    assert inventory["cross_border_positions"]["row_count"] == 0
    assert inventory["cross_border_positions"]["current_row_count"] == 0
    expected_positions = len(IMF_POSITION_COUNTRIES) * (
        len(PIP_SERIES) * len(PIP_FREQUENCIES) + len(DIP_SERIES) * len(DIP_FREQUENCIES)
    )
    assert inventory["cross_border_positions"]["expected_partitions"] == expected_positions
    assert inventory["cross_border_positions"]["stored_partitions"] == 0
    assert inventory["cross_border_positions"]["missing_partitions"] == expected_positions
    assert inventory["cross_border_positions"]["gap"] == "empty"
    assert inventory["market_history"]["commodities"]["series_count"] == 0
    assert inventory["market_history"]["commodities"]["stored_series"] == 0
    assert inventory["market_history"]["commodities"]["ready_series"] == 0
    assert inventory["market_history"]["commodities"]["guard_failures"] == ["missing"]
    assert inventory["market_history"]["money_liquidity"]["expected_series"] == len(
        MONEY_LIQUIDITY_SERIES
    )
    assert inventory["market_history"]["money_liquidity"]["stored_series"] == 0
    assert inventory["reports"]["review_count"] == 0
    assert inventory["reports"]["review_outcomes"] == []
    communications = inventory["communications"]
    assert communications["catalogue_source_policy_count"] == len(COMMUNICATION_SOURCES)
    assert communications["catalogue_organization_count"] == len(
        {source.organization_id for source in COMMUNICATION_SOURCES}
    )
    assert communications["catalogue_sha256"] == COMMUNICATION_CATALOGUE_SHA256
    assert communications["archive_coverage_status"] == "not_measured"
    assert all(communications["table_presence"].values())
    assert communications["organization_count"] == 0
    assert communications["source_policy_snapshot_count"] == 0
    assert communications["event_count"] == 0
    assert communications["event_version_count"] == 0
    assert communications["artifact_version_count"] == 0
    assert communications["artifact_section_scope_set_count"] == 0
    assert communications["declared_section_scope_count"] == 0
    assert communications["multi_section_artifact_count"] == 0
    assert communications["artifact_retrieval_count"] == 0
    assert communications["artifact_content_count"] == 0
    assert communications["stored_content_artifact_count"] == 0
    assert communications["extraction_count"] == 0
    assert communications["finalized_extraction_count"] == 0
    assert communications["segment_count"] == 0
    assert communications["finalized_segment_count"] == 0
    assert communications["unfinalized_segment_count"] == 0
    assert communications["incomplete_extraction_count"] == 0
    assert communications["represented_source_count"] == 0
    assert communications["stored_catalogue_hashes"] == []
    assert communications["schema_contract_status"] == "valid"
    assert communications["trigger_check_status"] == "valid"
    assert communications["missing_required_triggers"] == []
    assert communications["invalid_required_triggers"] == []
    assert communications["foreign_key_violation_count"] == 0
    assert communications["integrity_failures"] == []
    assert communications["integrity_status"] == "valid"
    assert (
        inventory["market_history"]["money_liquidity"]["catalogue_semantic_sha256"]
        == money_liquidity_catalogue_sha256()
    )
    shadow = inventory["market_history"]["shadow_liquidity"]
    expected_shadow = len(BIS_GLOBAL_LIQUIDITY_SERIES) + len(OFR_SHADOW_LIQUIDITY_SERIES)
    assert shadow["expected_series"] == expected_shadow
    assert shadow["stored_series"] == 0
    assert shadow["ready_series"] == 0
    assert shadow["guard_failed_series"] == 0
    assert shadow["missing_series"] == expected_shadow
    assert shadow["payload_provenance_unrecorded"] == 0
    assert shadow["provider_catalogue_semantic_sha256"] == {
        "BIS_GLI": bis_global_liquidity_catalogue_sha256(),
        "OFR_STFM": ofr_shadow_liquidity_catalogue_sha256(),
    }
    assert all(item["status"] == "missing" for item in shadow["series"])
    assert inventory["readiness"]["commodity_history"] == "empty"
    assert inventory["readiness"]["money_liquidity"] == "empty"
    assert inventory["readiness"]["shadow_liquidity"] == "empty"
    cross_border_table = next(
        row for row in inventory["tables"] if row["name"] == "cross_border_positions"
    )
    assert cross_border_table == {
        "name": "cross_border_positions",
        "present": True,
        "row_count": 0,
    }
    assert inventory["readiness"]["report_evidence"] == "empty"
    assert inventory["readiness"]["institutional_communications"] == "empty"
    assert (
        f"Institutional communications: {len(COMMUNICATION_SOURCES)} source policies"
        in render_inventory_summary(inventory)
    )


def test_invalid_communications_catalogue_does_not_disable_numeric_inventory(tmp_path, monkeypatch):
    engine = make_engine(tmp_path / "invalid-communications-policy.db")
    init_db(engine)
    monkeypatch.setattr(
        inventory_module,
        "_communication_catalogue_summary",
        lambda: {
            "validation_status": "invalid",
            "source_policy_count": None,
            "organization_count": None,
            "sha256": None,
            "error": "ValueError: deliberately broken policy",
        },
    )

    inventory = build_observatory_inventory(engine)

    assert inventory["observations"]["row_count"] == 0
    assert inventory["communications"]["catalogue_validation_status"] == "invalid"
    assert inventory["communications"]["event_version_count"] == 0
    assert inventory["communications"]["artifact_content_count"] == 0
    assert inventory["communications"]["integrity_status"] == "valid"
    assert inventory["readiness"]["institutional_communications"] == "policy_invalid"


def test_current_catalogue_metadata_is_verified_but_not_text_ready(tmp_path):
    engine = make_engine(tmp_path / "communications-current-metadata.db")
    init_db(engine)
    source = next(
        item for item in COMMUNICATION_SOURCES if item.source_id == "fed_fomc_press_conferences_en"
    )
    known_at = datetime(2026, 9, 2, tzinfo=UTC)
    with make_session_factory(engine)() as session:
        record_communication_artifact_metadata(
            session,
            CommunicationEventMeta(
                organization_id=source.organization_id,
                event_key="fomc_2026_07_29",
                event_type="central_bank_press_conference",
                title="FOMC press conference",
                event_date=date(2026, 7, 29),
                metadata_known_at=known_at,
            ),
            CommunicationArtifactMeta(
                source_id=source.source_id,
                catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
                artifact_key="official_transcript_en",
                artifact_role="full_transcript",
                material_type="press_conference_transcript",
                language="en",
                translation_status="original",
                mime_type="application/pdf",
                origin_type="official_published_transcript",
                provenance_tier="official_published_transcript",
                host_organization=source.host_organization,
                publisher=source.publisher,
                transcriber=None,
                transcriber_attribution="not_disclosed",
                rights_status=source.rights_status,
                acquisition_status=source.acquisition_status,
                rights_checked_by="human:test_reviewer",
                rights_checked_at=known_at,
                published_at=datetime(2026, 7, 29, tzinfo=UTC),
                available_at=datetime(2026, 7, 29, tzinfo=UTC),
                retrieved_at=known_at,
                metadata_known_at=known_at,
                landing_url=source.landing_url,
                artifact_url=(
                    "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20260729.pdf"
                ),
            ),
        )
        session.commit()

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["source_policy_snapshot_count"] == 1
    assert communications["current_catalogue_snapshot_check_status"] == "valid"
    assert communications["untrusted_catalogue_hash_count"] == 0
    assert communications["event_version_sha256_mismatch_count"] == 0
    assert communications["artifact_version_sha256_mismatch_count"] == 0
    assert communications["artifact_section_scope_set_count"] == 1
    assert communications["missing_artifact_scope_set_count"] == 0
    assert communications["missing_base_artifact_retrieval_count"] == 0
    assert communications["artifact_content_count"] == 0
    assert communications["integrity_status"] == "valid"
    assert inventory["readiness"]["institutional_communications"] == "metadata_only"


def test_communication_inventory_reports_one_capture_multi_section_scope(tmp_path):
    engine = make_engine(tmp_path / "communications-multi-section.db")
    init_db(engine)
    source = next(
        item
        for item in COMMUNICATION_SOURCES
        if item.source_id == "ecb_monetary_policy_press_conferences_en"
    )
    known_at = datetime(2026, 9, 9, tzinfo=UTC)
    artifact_url = (
        "https://www.ecb.europa.eu/press/press_conference/"
        "monetary-policy-statement/2025/html/ecb.is250130~1f418aa0f4.en.html"
    )
    with make_session_factory(engine)() as session:
        record_communication_artifact_metadata(
            session,
            CommunicationEventMeta(
                organization_id="ecb",
                event_key="ecb_2025_01_30",
                event_type="monetary_policy_press_conference",
                title="Monetary policy statement (with Q&A)",
                event_date=date(2025, 1, 30),
                metadata_known_at=known_at,
            ),
            CommunicationArtifactMeta(
                source_id=source.source_id,
                catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
                artifact_key="official_statement_with_q_and_a_en",
                artifact_role="q_and_a_transcript",
                material_type="questions_and_answers",
                language="en",
                translation_status="original",
                mime_type="text/html",
                origin_type="official_published_transcript",
                provenance_tier="official_published_transcript",
                host_organization=source.host_organization,
                publisher=source.publisher,
                transcriber=None,
                transcriber_attribution="not_disclosed",
                rights_status=source.rights_status,
                acquisition_status=source.acquisition_status,
                rights_checked_by="human:test_reviewer",
                rights_checked_at=known_at,
                available_at=known_at,
                retrieved_at=known_at,
                metadata_known_at=known_at,
                landing_url=artifact_url,
                artifact_url=artifact_url,
            ),
            section_scopes=(
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
            ),
        )
        session.commit()

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["artifact_version_count"] == 1
    assert communications["artifact_section_scope_set_count"] == 1
    assert communications["declared_section_scope_count"] == 2
    assert communications["multi_section_artifact_count"] == 1
    assert communications["scope_set_semantic_mismatch_count"] == 0
    assert communications["scope_set_sha256_mismatch_count"] == 0
    assert communications["artifact_version_sha256_mismatch_count"] == 0
    assert communications["integrity_status"] == "valid"
    assert inventory["readiness"]["institutional_communications"] == "metadata_only"


def test_unfinalized_communication_segments_are_not_analysis_ready(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-unfinalized.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=False, catalogue_sha256=catalogue_sha256)

    communications = build_observatory_inventory(engine)["communications"]

    assert communications["artifact_content_count"] == 1
    assert communications["verified_archived_blob_count"] == 1
    assert communications["extraction_count"] == 1
    assert communications["finalized_extraction_count"] == 0
    assert communications["valid_finalized_extraction_count"] == 0
    assert communications["segment_count"] == 1
    assert communications["finalized_segment_count"] == 0
    assert communications["unfinalized_segment_count"] == 1
    assert communications["incomplete_extraction_count"] == 1
    assert communications["integrity_status"] == "valid"
    inventory = build_observatory_inventory(engine)
    assert inventory["readiness"]["institutional_communications"] == ("artifacts_unextracted")


def test_verified_finalized_communication_corpus_is_analysis_ready(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-finalized.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True, catalogue_sha256=catalogue_sha256)

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["source_policy_snapshot_count"] == 1
    assert communications["event_version_sha256_mismatch_count"] == 0
    assert communications["artifact_version_sha256_mismatch_count"] == 0
    assert communications["artifact_section_scope_set_count"] == 1
    assert communications["declared_section_scope_count"] == 1
    assert communications["multi_section_artifact_count"] == 0
    assert communications["scope_set_sha256_mismatch_count"] == 0
    assert communications["policy_binding_mismatch_count"] == 0
    assert communications["artifact_retrieval_count"] == 1
    assert communications["missing_base_artifact_retrieval_count"] == 0
    assert communications["artifact_content_count"] == 1
    assert communications["verified_archived_blob_count"] == 1
    assert communications["extraction_content_binding_mismatch_count"] == 0
    assert communications["finalized_extraction_count"] == 1
    assert communications["valid_finalized_extraction_count"] == 1
    assert communications["finalized_segment_count"] == 1
    assert communications["segment_text_sha256_mismatch_count"] == 0
    assert communications["segment_char_count_mismatch_count"] == 0
    assert communications["segment_section_binding_mismatch_count"] == 0
    assert communications["finalized_section_coverage_mismatch_count"] == 0
    assert communications["corpus_sha256_mismatch_count"] == 0
    assert communications["integrity_status"] == "valid"
    assert inventory["readiness"]["institutional_communications"] == "available"
    assert communications["integrity_assurance_scope"] == ("structural_byte_hash_reproducibility")
    assert communications["transcript_semantic_fidelity_status"] == "not_verified"
    assert communications["extractor_execution_trust_status"] == "not_verified"
    assert "do not prove transcript semantic fidelity" in communications["integrity_assurance_note"]


def test_registered_historical_communication_snapshot_remains_trusted(tmp_path, monkeypatch):
    snapshot = _register_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-registered-historical.db")
    init_db(engine)
    _insert_communication_corpus(
        engine,
        finalize=False,
        catalogue_sha256=snapshot.catalogue_sha256,
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["registered_catalogue_snapshot_check_status"] == "valid"
    assert communications["registered_catalogue_snapshot_mismatch_count"] == 0
    assert communications["current_catalogue_snapshot_check_status"] == "valid"
    assert communications["current_catalogue_snapshot_mismatch_count"] == 0
    assert communications["untrusted_catalogue_hash_count"] == 0
    assert communications["integrity_status"] == "valid"


def test_registered_historical_policy_must_match_its_exact_snapshot(tmp_path, monkeypatch):
    snapshot = _register_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-forged-historical-policy.db")
    init_db(engine)
    _insert_communication_corpus(
        engine,
        finalize=False,
        catalogue_sha256=snapshot.catalogue_sha256,
        policy_publisher="Forged Test Bank",
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["policy_snapshot_sha256_mismatch_count"] == 0
    assert communications["registered_catalogue_snapshot_check_status"] == "invalid"
    assert communications["registered_catalogue_snapshot_mismatch_count"] == 1
    assert communications["untrusted_catalogue_hash_count"] == 0
    assert "registered_catalogue_snapshot" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"


def test_communication_inventory_rejects_scope_clock_after_retrieval_and_capture(
    tmp_path,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-scope-clock.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_artifact_section_scope_sets",
        operation="update",
        sql=(
            "UPDATE communication_artifact_section_scope_sets "
            "SET metadata_known_at = '2030-01-01 00:00:00.000000'"
        ),
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["communication_clock_mismatch_count"] == 2
    assert communications["artifact_retrieval_mismatch_count"] == 1
    assert communications["content_retrieval_binding_mismatch_count"] == 1
    assert communications["valid_finalized_extraction_count"] == 0
    assert "communication_clocks" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_corpus_v2_hash_binds_section_ordinal():
    segment = {
        "ordinal": 1,
        "section_ordinal": 1,
        "segment_kind": "narrative",
        "speaker_name": None,
        "speaker_role": None,
        "speaker_side": "unknown",
        "section_title": None,
        "text": "same text",
        "page_start": None,
        "page_end": None,
        "paragraph_start": None,
        "paragraph_end": None,
        "start_ms": None,
        "end_ms": None,
    }
    moved = {**segment, "section_ordinal": 2}

    assert inventory_module._communication_corpus_sha256(
        [segment], "communication_segments_json_v2"
    ) != inventory_module._communication_corpus_sha256([moved], "communication_segments_json_v2")
    assert inventory_module._communication_corpus_sha256(
        [segment], "communication_segments_json_v1"
    ) == inventory_module._communication_corpus_sha256([moved], "communication_segments_json_v1")


def test_communication_inventory_rejects_missing_scope_declaration(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-missing-scope.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=False, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_artifact_section_scope_sets",
        operation="delete",
        sql="DELETE FROM communication_artifact_section_scope_sets",
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["artifact_section_scope_set_count"] == 0
    assert communications["missing_artifact_scope_set_count"] == 1
    assert "missing_artifact_scope_set" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_inventory_rejects_malformed_scope_json(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-malformed-scope.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=False, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_artifact_section_scope_sets",
        operation="update",
        sql=("UPDATE communication_artifact_section_scope_sets SET scopes_json = 'not-json'"),
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["scope_set_json_malformed_count"] == 1
    assert "scope_set_json" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_inventory_rejects_invalid_segment_scope_binding(
    tmp_path,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-bad-segment-scope.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=False, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_segments",
        operation="update",
        sql="UPDATE communication_segments SET section_ordinal = 2",
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["segment_section_binding_mismatch_count"] == 1
    assert "segment_section_binding" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_scope_rejects_incompatible_segment_kind(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-scope-kind-trigger.db")
    init_db(engine)

    with pytest.raises(IntegrityError, match="artifact section scopes"):
        _insert_communication_corpus(
            engine,
            finalize=False,
            catalogue_sha256=catalogue_sha256,
            segment_kind="q_and_a_question",
            speaker_side="external",
        )


def test_communication_inventory_detects_tampered_segment_kind_scope(
    tmp_path,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-scope-kind-inventory.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_segments",
        operation="update",
        sql=(
            "UPDATE communication_segments "
            "SET segment_kind = 'q_and_a_question', speaker_side = 'external'"
        ),
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["segment_section_binding_mismatch_count"] == 1
    assert communications["valid_finalized_extraction_count"] == 0
    assert "segment_section_binding" in communications["integrity_failures"]
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_inventory_detects_tampered_segment_speaker_semantics(
    tmp_path,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-segment-speaker.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_segments",
        operation="update",
        sql="UPDATE communication_segments SET speaker_side = 'external'",
    )
    with engine.connect() as connection:
        segment = dict(
            connection.execute(
                select(
                    *(
                        getattr(CommunicationSegment, field)
                        for field in inventory_module._COMMUNICATION_CANONICAL_SEGMENT_FIELDS_V2
                    )
                )
            )
            .mappings()
            .one()
        )
    corpus_sha256 = inventory_module._communication_corpus_sha256(
        [segment],
        "communication_segments_json_v2",
    )
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_extraction_finalizations",
        operation="update",
        sql=(
            f"UPDATE communication_extraction_finalizations SET corpus_sha256 = '{corpus_sha256}'"
        ),
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["segment_semantic_mismatch_count"] == 1
    assert communications["corpus_sha256_mismatch_count"] == 0
    assert communications["valid_finalized_extraction_count"] == 0
    assert "segment_semantics" in communications["integrity_failures"]
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_inventory_rejects_finalized_scope_without_substantive_text(
    tmp_path,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-empty-finalized-scope.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True, catalogue_sha256=catalogue_sha256)
    _tamper_immutable_communication_row(
        engine,
        table_name="communication_segments",
        operation="update",
        sql="UPDATE communication_segments SET segment_kind = 'heading'",
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["finalized_section_semantic_mismatch_count"] == 1
    assert "finalized_section_semantics" in communications["integrity_failures"]
    assert communications["valid_finalized_extraction_count"] == 0
    assert inventory["readiness"]["institutional_communications"] == "invalid"


@pytest.mark.parametrize(
    ("valid_text_hash", "valid_corpus_hash", "failure"),
    [
        (False, True, "segment_text_sha256"),
        (True, False, "corpus_sha256"),
    ],
)
def test_communication_hash_mismatch_fails_closed(
    tmp_path,
    valid_text_hash,
    valid_corpus_hash,
    failure,
    monkeypatch,
):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / f"communications-bad-{failure}.db")
    init_db(engine)
    _insert_communication_corpus(
        engine,
        finalize=True,
        catalogue_sha256=catalogue_sha256,
        valid_text_hash=valid_text_hash,
        valid_corpus_hash=valid_corpus_hash,
    )

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert failure in communications["integrity_failures"]
    assert communications["valid_finalized_extraction_count"] == 0
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_communication_inventory_rehashes_blob_and_trigger_contract(tmp_path, monkeypatch):
    catalogue_sha256 = _trust_test_communication_catalogue(monkeypatch)
    engine = make_engine(tmp_path / "communications-contract.db")
    init_db(engine)
    blob_path = _insert_communication_corpus(
        engine, finalize=True, catalogue_sha256=catalogue_sha256
    )
    blob_path.write_bytes(b"tampered bytes")

    tampered_blob = build_observatory_inventory(engine)

    assert tampered_blob["communications"]["archived_blob_sha256_mismatch_count"] == 1
    assert tampered_blob["readiness"]["institutional_communications"] == "invalid"

    other_engine = make_engine(tmp_path / "communications-weak-trigger.db")
    init_db(other_engine)
    with other_engine.begin() as connection:
        connection.exec_driver_sql("DROP TRIGGER communication_content_matches_retrieval")
        connection.exec_driver_sql(
            "CREATE TRIGGER communication_content_matches_retrieval "
            "BEFORE INSERT ON communication_artifact_contents BEGIN SELECT 1; END"
        )

    weak_trigger = build_observatory_inventory(other_engine)

    assert weak_trigger["communications"]["schema_contract_status"] == "invalid"
    assert (
        "communication_content_matches_retrieval"
        in weak_trigger["communications"]["invalid_required_triggers"]
    )
    assert weak_trigger["readiness"]["institutional_communications"] == "invalid"


def test_unknown_communication_catalogue_hash_is_not_self_authenticating(tmp_path):
    engine = make_engine(tmp_path / "communications-untrusted-catalogue.db")
    init_db(engine)
    _insert_communication_corpus(engine, finalize=True)

    inventory = build_observatory_inventory(engine)
    communications = inventory["communications"]

    assert communications["untrusted_catalogue_hash_count"] == 1
    assert communications["untrusted_catalogue_hashes"] == ["a" * 64]
    assert "untrusted_catalogue_hash" in communications["integrity_failures"]
    assert communications["integrity_status"] == "invalid"
    assert inventory["readiness"]["institutional_communications"] == "invalid"


def test_inventory_reports_structured_coverage_and_evidence_without_mutating(tmp_path):
    engine = make_engine(tmp_path / "populated.db")
    init_db(engine)
    qpsd_spec = QPSD_SERIES[0]
    bop_spec = BOP_SERIES[0]

    with engine.begin() as connection:
        qpsd_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="qpsd:se:total",
                    source_family="WORLD_BANK_QPSD",
                    source_url=(
                        "https://api.worldbank.org/v2/country/USA;SWE/indicator/"
                        f"{qpsd_spec.wb_code}?source=20"
                    ),
                    content_sha256="2" * 64,
                    row_count=2,
                )
            )
        ).inserted_primary_key[0]
        bop_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="bop:se:direct",
                    source_family="IMF_BOP",
                    content_sha256="3" * 64,
                )
            )
        ).inserted_primary_key[0]
        debt_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="debt:se",
                    source_family="SCB_FINANCIAL_ACCOUNTS",
                    content_sha256="4" * 64,
                )
            )
        ).inserted_primary_key[0]
        allocator_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key="allocator:ap2",
                    source_family="AP_FUNDS",
                    content_sha256="5" * 64,
                )
            )
        ).inserted_primary_key[0]

        connection.execute(
            insert(Observation),
            [
                {
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2025, 10, 1),
                    "value": 30.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                },
                {
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 31.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                },
                {
                    "country": "SE",
                    "indicator": bop_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 10.0,
                    "source": "IMF_BOP",
                    "series_id": bop_spec.series_id,
                },
            ],
        )
        connection.execute(
            insert(ReleaseObservation),
            [
                {
                    "release_id": qpsd_release,
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2025, 10, 1),
                    "value": 30.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                    "status": "observed",
                },
                {
                    "release_id": qpsd_release,
                    "country": "SE",
                    "indicator": qpsd_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 31.0,
                    "source": "WORLD_BANK_QPSD",
                    "series_id": qpsd_spec.series_id,
                    "status": "observed",
                },
                {
                    "release_id": bop_release,
                    "country": "SE",
                    "indicator": bop_spec.indicator,
                    "date": date(2026, 1, 1),
                    "value": 10.0,
                    "source": "IMF_BOP",
                    "series_id": bop_spec.series_id,
                    "status": "observed",
                },
            ],
        )
        connection.execute(
            insert(DebtHolderPosition).values(
                release_id=debt_release,
                country="SE",
                date=date(2026, 1, 1),
                issuer_sector_code="S1311",
                issuer_sector_label="Central government",
                instrument_code="FL3000",
                instrument_label="Debt securities",
                holder_sector_code="S122",
                holder_sector_label="Deposit-taking corporations",
                measure_code="FM0103AS",
                measure_label="Closing balance",
                unit="SEK million",
                value=100.0,
                source="SCB_FINANCIAL_ACCOUNTS",
                series_id="TAB1203",
                status="observed",
            )
        )
        connection.execute(
            insert(AllocatorFact).values(
                release_id=allocator_release,
                fact_key="6" * 64,
                fund="AP2",
                as_of_date=date(2026, 6, 30),
                period_start=None,
                period_end=None,
                record_type="asset_allocation",
                item_code="listed_equity",
                reported_amount=10.0,
                reported_unit="SEK_bn",
                amount_sek_mn=10_000.0,
                exposure_pct=20.0,
                basis="fund_capital",
                row_role="component",
                physical_page=4,
                table_heading="Asset allocation",
                extraction_status="model_visual_check",
                quality_flag="none",
                notes=None,
                artifact_sha256="7" * 64,
                artifact_path="/blobs/report.pdf",
                source_url="https://ap2.se/report.pdf",
                parser_name="direct-table-transcription",
                parser_version="1",
            )
        )

        document_id = connection.execute(
            insert(ReportDocument).values(
                source_id="riksbank",
                report_family="monetary_policy_report",
                issue_key="2026-06",
                publisher="Sveriges Riksbank",
                jurisdiction="SE",
                title="Monetary Policy Report",
                language="en",
                document_date=date(2026, 6, 18),
                published_at=datetime(2026, 6, 18, tzinfo=UTC),
                available_at=datetime(2026, 6, 18, tzinfo=UTC),
                retrieved_at=datetime(2026, 8, 2, tzinfo=UTC),
                landing_url="https://riksbank.se/report",
                artifact_url="https://riksbank.se/report.pdf",
                mime_type="application/pdf",
                content_sha256="8" * 64,
                size_bytes=100,
                page_count=1,
                blob_path="/blobs/riksbank.pdf",
            )
        ).inserted_primary_key[0]
        extraction_id = connection.execute(
            insert(DocumentExtraction).values(
                document_id=document_id,
                extractor_name="pdftotext",
                extractor_version="1",
                extracted_at=datetime(2026, 8, 2, tzinfo=UTC),
                status="success",
                extracted_page_count=1,
                corpus_sha256="9" * 64,
                error=None,
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(DocumentPage).values(
                extraction_id=extraction_id,
                pdf_page=1,
                printed_page_label="1",
                text="Policy is restrictive.",
                text_sha256="a" * 64,
                char_count=22,
            )
        )
        claim_id = connection.execute(
            insert(Claim).values(
                claim_type="publisher_statement",
                statement="Policy is restrictive.",
                attribution_document_id=document_id,
                claim_series_key=None,
                topic_key="monetary_policy",
                geographies_json='["SE"]',
                status="verified",
                available_at=datetime(2026, 6, 18, tzinfo=UTC),
                created_by="test",
            )
        ).inserted_primary_key[0]
        connection.execute(
            insert(ClaimCitation).values(
                claim_id=claim_id,
                extraction_id=extraction_id,
                pdf_page_start=1,
                pdf_page_end=1,
                printed_locator="p. 1",
                section_title="Summary",
                evidence_excerpt="Policy is restrictive.",
                excerpt_sha256="b" * 64,
                support_role="supports",
                locator_verified_at=datetime(2026, 8, 2, tzinfo=UTC),
            )
        )

        connection.exec_driver_sql(
            """
            INSERT INTO cross_border_positions
                (release_id, dataset, reporter_country, reporter_code,
                 counterpart_country, counterpart_code, date, direction,
                 accounting_basis, instrument_code, instrument_label, frequency,
                 unit, value, source, native_indicator, series_id, status)
            VALUES
                (?, 'PIP', 'SE', 'SWE', 'US', 'USA', '2025-01-01', 'outward_assets',
                 'assets', 'portfolio_total', 'Total portfolio investment', 'A',
                 'USD', 1.0, 'IMF_PIP', 'P_TOTINV_P_USD', 'x', 'observed'),
                (?, 'PIP', 'SE', 'SWE', 'DE', 'DEU', '2024-01-01', 'outward_assets',
                 'assets', 'portfolio_total', 'Total portfolio investment', 'A',
                 'USD', 2.0, 'IMF_PIP', 'P_TOTINV_P_USD', 'y', 'observed')
            """,
            (bop_release, bop_release),
        )

    before = {}
    with engine.connect() as connection:
        for table in (Observation, DataRelease, DebtHolderPosition, AllocatorFact):
            before[table.__tablename__] = len(connection.execute(select(table)).all())

    inventory = build_observatory_inventory(engine)

    with engine.connect() as connection:
        after = {
            table.__tablename__: len(connection.execute(select(table)).all())
            for table in (Observation, DataRelease, DebtHolderPosition, AllocatorFact)
        }
    assert after == before

    qpsd_partition = inventory["qpsd"]["stored"][0]
    assert qpsd_partition == {
        "country": "SE",
        "indicator": qpsd_spec.indicator,
        "row_count": 2,
        "first_date": "2025-10-01",
        "latest_date": "2026-01-01",
    }
    assert inventory["qpsd"]["stored_partitions"] == 1
    assert inventory["qpsd"]["not_reported_partitions"] == 1
    assert inventory["qpsd"]["not_reported"] == [
        {
            "country": "US",
            "indicator": qpsd_spec.indicator,
            "series_id": qpsd_spec.series_id,
            "status": "not_reported",
        }
    ]
    assert inventory["qpsd"]["missing_partitions"] == (len(QPSD_COUNTRIES) * len(QPSD_SERIES) - 2)
    assert inventory["imf_bop"]["stored_partitions"] == 1
    assert inventory["imf_bop"]["missing_partitions"] == (len(BOP_COUNTRIES) * len(BOP_SERIES) - 1)

    assert inventory["debt_holders"]["current_row_count"] == 1
    assert inventory["debt_holders"]["instruments"] == [
        {"code": "FL3000", "label": "Debt securities"}
    ]
    assert inventory["debt_holders"]["holder_sectors"] == [
        {"code": "S122", "label": "Deposit-taking corporations"}
    ]
    assert inventory["allocators"]["funds"] == [
        {
            "fund": "AP2",
            "row_count": 1,
            "first_as_of_date": "2026-06-30",
            "latest_as_of_date": "2026-06-30",
            "record_types": ["asset_allocation"],
            "artifact_count": 1,
            "quality_flags": ["none"],
        }
    ]
    assert inventory["reports"]["document_count"] == 1
    assert inventory["reports"]["page_count"] == 1
    assert inventory["reports"]["claim_count"] == 1
    assert inventory["reports"]["citation_count"] == 1
    assert inventory["reports"]["review_count"] == 0
    assert inventory["reports"]["review_outcomes"] == []
    assert inventory["reports"]["documents"] == [
        {
            "publisher": "Sveriges Riksbank",
            "report_family": "monetary_policy_report",
            "document_count": 1,
            "first_document_date": "2026-06-18",
            "latest_document_date": "2026-06-18",
            "page_count": 1,
        }
    ]

    cross_border = inventory["cross_border_positions"]
    assert cross_border["table_present"] is True
    assert cross_border["row_count"] == 2
    assert cross_border["datasets"] == ["PIP"]
    assert cross_border["stored_partitions"] == 1
    assert cross_border["expected_partitions"] == 294
    assert cross_border["missing_partitions"] == 293
    assert cross_border["by_dataset_reporter"] == [
        {
            "dataset": "PIP",
            "reporter_country": "SE",
            "row_count": 2,
            "counterpart_count": 2,
            "first_date": "2024-01-01",
            "latest_date": "2025-01-01",
        }
    ]
    assert inventory["readiness"] == {
        "current_observations": "available",
        "immutable_release_history": "available",
        "sovereign_debt_anatomy": "partial",
        "sovereign_refinancing": "empty",
        "cross_border_transactions": "partial",
        "debt_holder_positions": "available",
        "allocator_disclosures": "available",
        "report_evidence": "available",
        "institutional_communications": "empty",
        "bilateral_positions": "partial",
        "commodity_history": "empty",
        "money_liquidity": "empty",
        "shadow_liquidity": "empty",
    }
    summary = render_inventory_summary(inventory)
    assert "QPSD: 1/264 stored" in summary
    assert "IMF BOP transactions: 1/525 stored" in summary
    assert "Cross-border positions: 2 rows; 1/294 partitions" in summary


def test_inventory_surfaces_commodity_and_money_history_without_comparing_levels(tmp_path):
    engine = make_engine(tmp_path / "markets.db")
    init_db(engine)
    catalogue_hash = money_liquidity_catalogue_sha256()
    with engine.begin() as connection:
        money_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="9" * 64,
                    row_count=len(_monthly_dates(FED_M2.expected_start, date(2026, 7, 1))),
                )
            )
        )
        _insert_test_release_artifacts(
            connection,
            int(money_release.inserted_primary_key[0]),
            FED_M2,
            tmp_path / "money-artifacts",
        )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": "commodity_price_copper",
                    "date": date(2026, 7, 1),
                    "value": 9_000.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Copper",
                },
                {
                    "country": "WLD",
                    "indicator": "commodity_price_copper",
                    "date": date(2026, 8, 1),
                    "value": 9_100.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Copper",
                },
                {
                    "country": "WLD",
                    "indicator": "commodity_index_energy",
                    "date": date(2026, 8, 1),
                    "value": 120.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": "Energy",
                },
                *_money_observations(_monthly_dates(FED_M2.expected_start, date(2026, 7, 1))),
            ],
        )

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    commodities = inventory["market_history"]["commodities"]
    money = inventory["market_history"]["money_liquidity"]

    assert commodities["row_count"] == 3
    assert commodities["series_count"] == 2
    assert commodities["price_series_count"] == 1
    assert commodities["index_series_count"] == 1
    assert commodities["first_date"] == "2026-07-01"
    assert commodities["latest_date"] == "2026-08-01"
    assert commodities["stored_series"] == 2
    assert commodities["ready_series"] == 0
    assert commodities["status"] == "guard_failed"
    assert "expected_catalogue" in commodities["guard_failures"]
    assert "latest_release_group" in commodities["guard_failures"]
    assert money["stored_series"] == 1
    assert money["ready_series"] == 1
    assert money["guard_failed_series"] == 0
    assert money["missing_series"] == len(MONEY_LIQUIDITY_SERIES) - 1
    assert money["coverage_pct"] == round(100 / len(MONEY_LIQUIDITY_SERIES), 3)
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")
    assert fed["status"] == "ready"
    assert fed["storage_status"] == "stored"
    assert fed["guard_failures"] == []
    assert fed["unit"] == "USD billion"
    assert fed["frequency"] == "monthly"
    assert fed["perimeter"] == FED_M2.perimeter
    assert fed["definition_notes"] == FED_M2.definition_notes
    assert fed["parent_native_series_id"] is None
    assert fed["non_additive_groups"] == list(FED_M2.non_additive_groups)
    assert fed["catalogue_semantic_sha256"] == catalogue_hash
    assert fed["release_catalogue_semantic_sha256"] == catalogue_hash
    assert fed["catalogue_semantic_status"] == "match"
    assert fed["artifact_manifest_status"] == "valid"
    assert fed["artifact_manifest"]["roles"] == [
        "missingness_ledger",
        "native_series_payload",
        "source_response",
    ]
    assert money["catalogue_semantic_sha256"] == catalogue_hash
    assert money["catalogue_semantic_mismatches"] == 0
    assert money["catalogue_semantic_unrecorded"] == 0
    assert inventory["readiness"]["commodity_history"] == "empty"
    assert inventory["readiness"]["money_liquidity"] == "partial"
    summary = render_inventory_summary(inventory)
    assert "Commodity history: 0/87 expected series ready; 2 stored" in summary
    assert (
        f"Money/liquidity: 1/{len(MONEY_LIQUIDITY_SERIES)} pinned series ready; 1 stored"
    ) in summary
    assert f"catalogue {catalogue_hash[:12]}" in summary


def test_money_inventory_requires_an_intact_release_artifact_manifest(tmp_path):
    dates = _monthly_dates(FED_M2.expected_start, date(2026, 7, 1))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "money-corrupt-artifact.db")
    init_db(engine)
    with engine.begin() as connection:
        release_result = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="6" * 64,
                    row_count=len(dates),
                )
            )
        )
        paths = _insert_test_release_artifacts(
            connection,
            int(release_result.inserted_primary_key[0]),
            FED_M2,
            tmp_path / "money-artifacts",
        )
        connection.execute(insert(Observation), _money_observations(dates))

    paths["missingness_ledger"].write_bytes(b"corrupt")
    money = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")

    assert fed["status"] == "guard_failed"
    assert fed["artifact_manifest_status"] == "invalid"
    assert "missingness_ledger:artifact_hash_mismatch" in fed["artifact_manifest"]["failures"]
    assert "artifact_manifest_invalid" in fed["guard_failures"]
    assert money["ready_series"] == 0


def test_commodity_inventory_qualifies_complete_group_and_allows_expansion(tmp_path, monkeypatch):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    new_id = canonical_series_id("New benchmark", "Monthly Prices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 7, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 2)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 2)
    series = [
        (copper_id, "commodity_price_copper"),
        (energy_id, "commodity_index_energy"),
        (new_id, "commodity_price_new_benchmark"),
    ]
    months = [date(2026, 7, 1), date(2026, 8, 1)]
    group_label = pink_sheet_vintage_label("c" * 64)
    engine = make_engine(tmp_path / "commodity-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        for series_id, indicator in series:
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=group_label,
                        content_sha256=indicator.encode().hex().ljust(64, "0")[:64],
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": float(100 + month_index),
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator in series
                for month_index, observed_on in enumerate(months)
            ],
        )

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    commodities = inventory["market_history"]["commodities"]

    assert commodities["status"] == "ready"
    assert commodities["expected_series"] == 2
    assert commodities["stored_series"] == 3
    assert commodities["ready_series"] == 2
    assert commodities["extra_series"] == 1
    assert commodities["extra_series_ids"] == [new_id]
    assert commodities["stored_coverage_pct"] == 100.0
    assert commodities["coverage_pct"] == 100.0
    assert commodities["price_series_count"] == 2
    assert commodities["index_series_count"] == 1
    assert commodities["workbook_sha256"] == "c" * 64
    assert commodities["catalogue_schema_version"] == CATALOGUE_SCHEMA_VERSION
    assert commodities["catalogue_generator_version"] == CATALOGUE_GENERATOR_VERSION
    assert commodities["catalogue_semantic_status"] == "match"
    assert commodities["guard_failures"] == []
    assert inventory["readiness"]["commodity_history"] == "complete"


def test_commodity_inventory_requires_identical_release_group_clocks(tmp_path, monkeypatch):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 7, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 2)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 2)
    series = [
        (copper_id, "commodity_price_copper"),
        (energy_id, "commodity_index_energy"),
    ]
    months = [date(2026, 7, 1), date(2026, 8, 1)]
    group_label = pink_sheet_vintage_label("f" * 64)
    engine = make_engine(tmp_path / "commodity-mixed-clocks.db")
    init_db(engine)
    with engine.begin() as connection:
        for index, (series_id, indicator) in enumerate(series):
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=group_label,
                        retrieved_at=datetime(2026, 8, 2 + index, tzinfo=UTC),
                        content_sha256=str(index + 1) * 64,
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": 100.0,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator in series
                for observed_on in months
            ],
        )

    commodities = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "commodities"
    ]

    assert commodities["guard_failures"] == ["latest_release_group"]
    assert commodities["status"] == "guard_failed"
    assert commodities["ready_series"] == 0
    assert commodities["release_group_vintage_label"] is None


def test_commodity_inventory_surfaces_value_cadence_and_release_group_failures(
    tmp_path, monkeypatch
):
    copper_id = canonical_series_id("Copper", "Monthly Prices")
    energy_id = canonical_series_id("Energy", "Monthly Indices")
    monkeypatch.setattr(
        inventory_module,
        "EXPECTED_SERIES_IDS_BY_WORKSHEET",
        {
            "Monthly Prices": frozenset({copper_id}),
            "Monthly Indices": frozenset({energy_id}),
        },
    )
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_PRICE_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_INDEX_SERIES", 1)
    monkeypatch.setattr(inventory_module, "EXPECTED_COMMODITY_HISTORY_START", date(2026, 6, 1))
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_HISTORY_MONTHS", 3)
    monkeypatch.setattr(inventory_module, "MIN_COMMODITY_OBSERVATIONS_PER_SERIES", 2)
    monkeypatch.setattr(inventory_module, "MAX_COMMODITY_LATEST_LAG_MONTHS", 1)
    engine = make_engine(tmp_path / "commodity-bad.db")
    init_db(engine)
    series = [
        (copper_id, "commodity_price_gold", "d" * 64),
        (energy_id, "commodity_index_energy", "e" * 64),
    ]
    with engine.begin() as connection:
        for series_id, indicator, digest in series:
            connection.execute(
                insert(DataRelease).values(
                    **_release(
                        partition_key=make_partition_key(
                            SOURCE_WORLD_BANK_COMMODITIES,
                            series_id,
                            "WLD",
                            indicator,
                        ),
                        source_family=SOURCE_WORLD_BANK_COMMODITIES,
                        vintage_label=pink_sheet_vintage_label(digest),
                        content_sha256=digest,
                        row_count=2,
                    )
                )
            )
        connection.execute(
            insert(Observation),
            [
                {
                    "country": "WLD",
                    "indicator": indicator,
                    "date": observed_on,
                    "value": value,
                    "source": SOURCE_WORLD_BANK_COMMODITIES,
                    "series_id": series_id,
                }
                for series_id, indicator, _digest in series
                for observed_on, value in (
                    (date(2026, 6, 1), 100.0),
                    (date(2026, 8, 1), 0.0),
                )
            ],
        )

    commodities = build_observatory_inventory(engine, as_of=date(2026, 10, 8))["market_history"][
        "commodities"
    ]

    assert commodities["status"] == "guard_failed"
    assert commodities["stored_series"] == 2
    assert commodities["ready_series"] == 0
    assert commodities["nonpositive_rows"] == 2
    assert set(commodities["guard_failures"]) >= {
        "nonpositive_values",
        "canonical_indicator_mapping",
        "monthly_cadence",
        "minimum_history",
        "latest_lag",
        "latest_release_group",
    }


@pytest.mark.parametrize(
    ("dates", "vintage_label", "failure"),
    [
        (
            _monthly_dates(date(1959, 2, 1), date(2026, 7, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "expected_start",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2024, 12, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "minimum_observations",
        ),
        (
            [
                observed_on
                for observed_on in _monthly_dates(FED_M2.expected_start, date(2026, 7, 1))
                if observed_on != date(2000, 1, 1)
            ],
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "cadence",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2025, 12, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{money_liquidity_catalogue_sha256()}",
            "latest_lag",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2026, 7, 1)),
            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{'0' * 64}",
            "catalogue_semantic_mismatch",
        ),
        (
            _monthly_dates(FED_M2.expected_start, date(2026, 7, 1)),
            None,
            "catalogue_semantic_unrecorded",
        ),
    ],
)
def test_money_inventory_readiness_requires_every_history_guard(
    tmp_path,
    dates,
    vintage_label,
    failure,
):
    engine = make_engine(tmp_path / f"money-{failure}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        FED_M2.source_family,
                        FED_M2.native_series_id,
                        FED_M2.country,
                        FED_M2.indicator,
                    ),
                    source_family=FED_M2.source_family,
                    vintage_label=vintage_label,
                    content_sha256="8" * 64,
                    row_count=len(dates),
                )
            )
        )
        connection.execute(insert(Observation), _money_observations(dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    money = inventory["market_history"]["money_liquidity"]
    fed = next(item for item in money["series"] if item["series_id"] == "M2SL")

    assert fed["storage_status"] == "stored"
    assert fed["status"] == "guard_failed"
    assert failure in fed["guard_failures"]
    assert money["stored_series"] == 1
    assert money["ready_series"] == 0
    assert money["guard_failed_series"] == 1
    assert money["coverage_pct"] == 0.0
    assert inventory["readiness"]["money_liquidity"] == "empty"


def test_money_inventory_qualifies_quarterly_bridge_and_preserves_semantics(
    tmp_path,
    monkeypatch,
):
    spec = replace(
        BOE_M4EX_QUARTERLY,
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    dates = [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
    monkeypatch.setattr(inventory_module, "MONEY_LIQUIDITY_SERIES", (spec,))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "money-quarterly-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        money_release = connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        spec.source_family,
                        spec.native_series_id,
                        spec.country,
                        spec.indicator,
                    ),
                    source_family=spec.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="7" * 64,
                    row_count=len(dates),
                )
            )
        )
        _insert_test_release_artifacts(
            connection,
            int(money_release.inserted_primary_key[0]),
            spec,
            tmp_path / "money-artifacts",
        )
        connection.execute(insert(Observation), _money_observations(dates, spec))

    money = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]
    item = money["series"][0]

    assert item["status"] == "ready"
    assert item["guard_failures"] == []
    assert item["perimeter"] == spec.perimeter
    assert item["definition_notes"] == spec.definition_notes
    assert item["parent_native_series_id"] == spec.parent_native_series_id
    assert item["non_additive_groups"] == list(spec.non_additive_groups)
    assert item["artifact_manifest_status"] == "valid"


@pytest.mark.parametrize(
    "dates",
    [
        [date(2026, 1, 1), date(2026, 7, 1)],
        [date(2026, 1, 1), date(2026, 5, 1)],
    ],
)
def test_money_inventory_rejects_noncontinuous_or_nonstart_quarters(
    tmp_path,
    monkeypatch,
    dates,
):
    spec = replace(
        BOE_M4EX_QUARTERLY,
        expected_start=date(2026, 1, 1),
        minimum_observations=2,
        max_latest_lag_days=365,
    )
    monkeypatch.setattr(inventory_module, "MONEY_LIQUIDITY_SERIES", (spec,))
    catalogue_hash = money_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / f"money-quarterly-{dates[-1].month}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_release(
                    partition_key=make_partition_key(
                        spec.source_family,
                        spec.native_series_id,
                        spec.country,
                        spec.indicator,
                    ),
                    source_family=spec.source_family,
                    vintage_label=(f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"),
                    content_sha256="6" * 64,
                    row_count=len(dates),
                )
            )
        )
        connection.execute(insert(Observation), _money_observations(dates, spec))

    item = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "money_liquidity"
    ]["series"][0]

    assert item["status"] == "guard_failed"
    assert "cadence" in item["guard_failures"]


def test_shadow_liquidity_inventory_qualifies_bis_and_ofr_native_cadences(
    tmp_path,
    monkeypatch,
):
    bis_spec = replace(
        BIS_GLOBAL_LIQUIDITY_SERIES[0],
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_complete_monthly_spec = replace(
        next(
            spec
            for spec in OFR_SHADOW_LIQUIDITY_SERIES
            if spec.cadence_policy == "complete_monthly"
        ),
        expected_start=date(2026, 1, 31),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_sparse_monthly_spec = replace(
        next(
            spec
            for spec in OFR_SHADOW_LIQUIDITY_SERIES
            if spec.cadence_policy == "sparse_monthly" and spec.parent_native_series_id is not None
        ),
        expected_start=date(2026, 1, 31),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    ofr_daily_spec = replace(
        next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.frequency == "daily"),
        expected_start=date(2026, 8, 31),
        minimum_observations=3,
        max_latest_lag_days=30,
        minimum_weekday_coverage_ratio=0.5,
    )
    monkeypatch.setattr(inventory_module, "BIS_GLOBAL_LIQUIDITY_SERIES", (bis_spec,))
    monkeypatch.setattr(
        inventory_module,
        "OFR_SHADOW_LIQUIDITY_SERIES",
        (ofr_complete_monthly_spec, ofr_sparse_monthly_spec, ofr_daily_spec),
    )
    histories = {
        bis_spec: [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)],
        ofr_complete_monthly_spec: [
            date(2026, 1, 31),
            date(2026, 2, 28),
            date(2026, 3, 31),
        ],
        # Missing February is allowed; observed MMF dates must still be month-end.
        ofr_sparse_monthly_spec: [date(2026, 1, 31), date(2026, 3, 31), date(2026, 7, 31)],
        # Missing Tuesday/Thursday is allowed; stored repo observations stay weekdays.
        ofr_daily_spec: [date(2026, 8, 31), date(2026, 9, 2), date(2026, 9, 4)],
    }
    bis_hash = bis_global_liquidity_catalogue_sha256()
    ofr_hash = ofr_shadow_liquidity_catalogue_sha256()
    engine = make_engine(tmp_path / "shadow-ready.db")
    init_db(engine)
    with engine.begin() as connection:
        for spec, dates in histories.items():
            prefix, digest = (
                (BIS_GLI_CATALOGUE_VINTAGE_PREFIX, bis_hash)
                if spec.source_family == "BIS_GLI"
                else (OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX, ofr_hash)
            )
            release_result = connection.execute(
                insert(DataRelease).values(
                    **_shadow_release(
                        spec,
                        prefix,
                        digest,
                        len(dates),
                        payload_tag=("a" * 24 if spec.source_family == "OFR_STFM" else None),
                    )
                )
            )
            _insert_test_release_artifacts(
                connection,
                int(release_result.inserted_primary_key[0]),
                spec,
                tmp_path / "artifacts",
            )
            connection.execute(insert(Observation), _shadow_observations(spec, dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    shadow = inventory["market_history"]["shadow_liquidity"]

    assert shadow["expected_series"] == 4
    assert shadow["stored_series"] == 4
    assert shadow["ready_series"] == 4
    assert shadow["missing_series"] == 0
    assert shadow["guard_failed_series"] == 0
    assert shadow["coverage_pct"] == 100.0
    assert shadow["provider_catalogue_semantic_sha256"] == {
        "BIS_GLI": bis_hash,
        "OFR_STFM": ofr_hash,
    }
    assert shadow["providers"]["BIS_GLI"]["ready_series"] == 1
    assert shadow["providers"]["OFR_STFM"]["ready_series"] == 3
    assert inventory["readiness"]["shadow_liquidity"] == "complete"

    by_id = {item["native_series_id"]: item for item in shadow["series"]}
    bis_item = by_id[bis_spec.native_series_id]
    assert bis_item["status"] == "ready"
    assert bis_item["measure_kind"] == "credit_stock"
    assert bis_item["side"] == bis_spec.claim_side
    assert bis_item["claim_side"] == bis_spec.claim_side
    assert bis_item["from_sector"] == bis_spec.from_sector
    assert bis_item["to_sector"] == bis_spec.to_sector
    assert bis_item["instrument"] == bis_spec.instrument
    assert bis_item["aggregation_role"] == "currency_total"
    assert bis_item["parent_native_series_id"] is None
    assert bis_item["parents"] == []
    assert bis_item["non_additive_groups"] == [bis_spec.non_additive_group]
    assert bis_item["native_unit"] == bis_spec.native_unit
    assert bis_item["unit"] == bis_spec.unit
    assert bis_item["currency"] == "USD"
    assert bis_item["frequency"] == "quarterly"

    monthly_item = by_id[ofr_sparse_monthly_spec.native_series_id]
    assert monthly_item["status"] == "ready"
    assert monthly_item["measure_kind"] == ofr_sparse_monthly_spec.measure_kind
    assert monthly_item["side"] == ofr_sparse_monthly_spec.economic_side
    assert monthly_item["claim_side"] == ofr_sparse_monthly_spec.claim_side
    assert monthly_item["from_sector"] == ofr_sparse_monthly_spec.from_sector
    assert monthly_item["to_sector"] == ofr_sparse_monthly_spec.to_sector
    assert monthly_item["instrument"] == ofr_sparse_monthly_spec.instrument
    assert monthly_item["collateral_scope"] == (ofr_sparse_monthly_spec.collateral_scope)
    assert monthly_item["aggregation_role"] == "component"
    assert monthly_item["parent_native_series_id"] == (
        ofr_sparse_monthly_spec.parent_native_series_id
    )
    assert monthly_item["parents"] == [ofr_sparse_monthly_spec.parent_native_series_id]
    assert monthly_item["non_additive_groups"] == list(ofr_sparse_monthly_spec.non_additive_groups)
    assert monthly_item["required_notes_phrases"] == list(
        ofr_sparse_monthly_spec.required_notes_phrases
    )
    assert monthly_item["cadence_policy"] == "sparse_monthly"
    assert monthly_item["native_unit"] == ofr_sparse_monthly_spec.native_unit
    assert monthly_item["unit"] == ofr_sparse_monthly_spec.unit
    assert monthly_item["currency"] == ofr_sparse_monthly_spec.currency
    assert monthly_item["frequency"] == "monthly"
    assert monthly_item["release_payload_provenance_tag"] == "a" * 24

    summary = render_inventory_summary(inventory)
    assert "Shadow liquidity: 4/4 series ready; 4 stored" in summary
    assert f"BIS {bis_hash[:12]}" in summary
    assert f"OFR {ofr_hash[:12]}" in summary


@pytest.mark.parametrize(
    ("damage", "manifest_failure"),
    [
        ("corrupt", "source_response:artifact_hash_mismatch"),
        ("delete", "source_response:artifact_missing"),
    ],
)
def test_shadow_liquidity_inventory_rehashes_retained_artifacts(
    tmp_path,
    monkeypatch,
    damage,
    manifest_failure,
):
    spec = replace(
        BIS_GLOBAL_LIQUIDITY_SERIES[0],
        expected_start=date(2026, 1, 1),
        minimum_observations=3,
        max_latest_lag_days=365,
    )
    monkeypatch.setattr(inventory_module, "BIS_GLOBAL_LIQUIDITY_SERIES", (spec,))
    monkeypatch.setattr(inventory_module, "OFR_SHADOW_LIQUIDITY_SERIES", ())
    dates = [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)]
    engine = make_engine(tmp_path / f"artifact-{damage}.db")
    init_db(engine)
    with engine.begin() as connection:
        release_result = connection.execute(
            insert(DataRelease).values(
                **_shadow_release(
                    spec,
                    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
                    bis_global_liquidity_catalogue_sha256(),
                    len(dates),
                )
            )
        )
        paths = _insert_test_release_artifacts(
            connection,
            int(release_result.inserted_primary_key[0]),
            spec,
            tmp_path / "artifacts",
        )
        connection.execute(insert(Observation), _shadow_observations(spec, dates))

    source_path = paths["source_response"]
    if damage == "corrupt":
        source_path.write_bytes(b"corrupt")
    else:
        source_path.unlink()

    item = build_observatory_inventory(engine, as_of=date(2026, 9, 8))["market_history"][
        "shadow_liquidity"
    ]["series"][0]

    assert item["status"] == "guard_failed"
    assert item["artifact_manifest_status"] == "invalid"
    assert manifest_failure in item["artifact_manifest"]["failures"]
    assert "artifact_manifest_invalid" in item["guard_failures"]


_SMALL_BIS_SHADOW_SPEC = replace(
    BIS_GLOBAL_LIQUIDITY_SERIES[0],
    expected_start=date(2026, 1, 1),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_COMPLETE_MONTHLY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.cadence_policy == "complete_monthly"),
    expected_start=date(2026, 1, 31),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_SPARSE_MONTHLY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.cadence_policy == "sparse_monthly"),
    expected_start=date(2026, 1, 31),
    minimum_observations=2,
    max_latest_lag_days=365,
)
_SMALL_OFR_DAILY_SPEC = replace(
    next(spec for spec in OFR_SHADOW_LIQUIDITY_SERIES if spec.frequency == "daily"),
    expected_start=date(2026, 8, 31),
    minimum_observations=2,
    max_latest_lag_days=30,
)


@pytest.mark.parametrize(
    ("spec", "dates", "digest", "payload_tag", "failure"),
    [
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 4, 1), date(2026, 7, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "expected_start",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "minimum_observations",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1), date(2026, 7, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "quarterly_cadence",
        ),
        (
            replace(_SMALL_BIS_SHADOW_SPEC, max_latest_lag_days=30),
            [date(2026, 1, 1), date(2026, 4, 1)],
            bis_global_liquidity_catalogue_sha256(),
            None,
            "latest_lag",
        ),
        (
            _SMALL_BIS_SHADOW_SPEC,
            [date(2026, 1, 1), date(2026, 4, 1), date(2026, 7, 1)],
            "0" * 64,
            None,
            "catalogue_semantic_mismatch",
        ),
        (
            _SMALL_OFR_SPARSE_MONTHLY_SPEC,
            [date(2026, 1, 31), date(2026, 3, 30)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "calendar_month_end",
        ),
        (
            _SMALL_OFR_COMPLETE_MONTHLY_SPEC,
            [date(2026, 1, 31), date(2026, 3, 31)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "monthly_cadence",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 5)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "business_day_cadence",
        ),
        (
            replace(_SMALL_OFR_DAILY_SPEC, max_internal_gap_days=3),
            [date(2026, 8, 31), date(2026, 9, 4)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "a" * 24,
            "maximum_internal_gap",
        ),
        (
            replace(_SMALL_OFR_DAILY_SPEC, minimum_weekday_coverage_ratio=0.9),
            [date(2026, 8, 31), date(2026, 9, 2), date(2026, 9, 4)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "a" * 24,
            "minimum_weekday_coverage",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 2)],
            ofr_shadow_liquidity_catalogue_sha256(),
            "z" * 24,
            "payload_provenance_unrecorded",
        ),
        (
            _SMALL_OFR_DAILY_SPEC,
            [date(2026, 8, 31), date(2026, 9, 2)],
            ofr_shadow_liquidity_catalogue_sha256(),
            None,
            "payload_provenance_unrecorded",
        ),
    ],
)
def test_shadow_liquidity_inventory_rejects_guard_or_hash_failures(
    tmp_path,
    monkeypatch,
    spec,
    dates,
    digest,
    payload_tag,
    failure,
):
    is_bis = spec.source_family == "BIS_GLI"
    monkeypatch.setattr(
        inventory_module,
        "BIS_GLOBAL_LIQUIDITY_SERIES",
        (spec,) if is_bis else (),
    )
    monkeypatch.setattr(
        inventory_module,
        "OFR_SHADOW_LIQUIDITY_SERIES",
        () if is_bis else (spec,),
    )
    prefix = BIS_GLI_CATALOGUE_VINTAGE_PREFIX if is_bis else OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX
    engine = make_engine(tmp_path / f"shadow-{failure}.db")
    init_db(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(DataRelease).values(
                **_shadow_release(
                    spec,
                    prefix,
                    digest,
                    len(dates),
                    payload_tag=payload_tag,
                )
            )
        )
        connection.execute(insert(Observation), _shadow_observations(spec, dates))

    inventory = build_observatory_inventory(engine, as_of=date(2026, 9, 8))
    shadow = inventory["market_history"]["shadow_liquidity"]
    item = shadow["series"][0]

    assert item["storage_status"] == "stored"
    assert item["status"] == "guard_failed"
    assert failure in item["guard_failures"]
    assert shadow["stored_series"] == 1
    assert shadow["ready_series"] == 0
    assert shadow["guard_failed_series"] == 1
    assert shadow["coverage_pct"] == 0.0
    assert inventory["readiness"]["shadow_liquidity"] == "empty"


def test_inventory_handles_a_database_with_only_an_older_table(tmp_path):
    engine = make_engine(tmp_path / "old.db")
    Observation.__table__.create(engine)
    with engine.begin() as connection:
        connection.execute(
            insert(Observation).values(
                country="US",
                indicator="policy_rate",
                date=date(2026, 1, 1),
                value=3.0,
                source="FRED",
                series_id="DFF",
            )
        )

    inventory = build_observatory_inventory(engine)

    assert inventory["observations"]["row_count"] == 1
    assert inventory["releases"]["release_count"] is None
    assert inventory["debt_holders"]["table_present"] is False
    assert inventory["reports"]["document_count"] is None
    assert inventory["communications"]["event_version_count"] is None
    assert inventory["communications"]["artifact_version_count"] is None
    assert inventory["readiness"]["institutional_communications"] == "table_absent"
    assert inventory["cross_border_positions"]["table_present"] is False
    assert inventory["cross_border_positions"]["row_count"] is None
