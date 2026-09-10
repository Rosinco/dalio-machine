"""SQLite storage for current macro data and immutable evidence ledgers.

``observations`` remains the backwards-compatible latest-value projection used by
the existing classifiers and UI. ``data_releases`` + ``release_observations`` are
the append-only point-in-time ledger: one complete source-partition snapshot per
release, so a historical query can reproduce both revisions and omitted rows.
Institutional prose stays separate in immutable report, extraction, page, claim,
and citation tables protected by foreign keys and append-only database triggers.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
    inspect,
    text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Observation(Base):
    __tablename__ = "observations"

    id = Column(Integer, primary_key=True)
    country = Column(String(8), nullable=False, index=True)
    indicator = Column(String(64), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    value = Column(Float, nullable=False)
    source = Column(String(32), nullable=False)
    # Preserve full native keys (ECB GFS refinancing includes a 65-character
    # key), matching the immutable ledger. Existing SQLite VARCHAR(64) columns
    # do not enforce a length limit and need no destructive table migration.
    series_id = Column(String(128), nullable=False)
    fetched_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "country",
            "indicator",
            "date",
            "source",
            name="uq_obs_country_ind_date_src",
        ),
        Index("ix_obs_lookup", "country", "indicator", "date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Observation {self.country}/{self.indicator} "
            f"{self.date}={self.value} from {self.source}>"
        )


class DataRelease(Base):
    """An immutable, complete snapshot of one independently refreshed partition."""

    __tablename__ = "data_releases"

    id = Column(Integer, primary_key=True)
    partition_key = Column(String(256), nullable=False)
    source_family = Column(String(64), nullable=False, index=True)
    published_at = Column(DateTime, nullable=True)
    available_at = Column(DateTime, nullable=False)
    retrieved_at = Column(DateTime, nullable=False)
    vintage_label = Column(String(128), nullable=True)
    source_url = Column(Text, nullable=True)
    content_sha256 = Column(String(64), nullable=False)
    row_count = Column(Integer, nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "partition_key",
            "available_at",
            "retrieved_at",
            name="uq_release_partition_clock",
        ),
        Index(
            "ix_release_partition_available",
            "partition_key",
            "available_at",
            "retrieved_at",
        ),
    )


class DataReleaseArtifact(Base):
    """One immutable raw-artifact manifest entry attached to a data release."""

    __tablename__ = "data_release_artifacts"

    id = Column(Integer, primary_key=True)
    release_id = Column(
        Integer,
        ForeignKey("data_releases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    role = Column(String(64), nullable=False)
    artifact_sha256 = Column(String(64), nullable=False, index=True)
    artifact_path = Column(Text, nullable=False)
    native_payload_sha256 = Column(String(64), nullable=False, index=True)
    missing_provenance_sha256 = Column(String(64), nullable=False, index=True)
    provenance_json = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "release_id",
            "role",
            name="uq_release_artifact_role",
        ),
        CheckConstraint(
            "length(role) BETWEEN 1 AND 64 AND role = trim(role) AND role NOT GLOB '*[^a-z0-9_]*'",
            name="ck_release_artifact_role",
        ),
        CheckConstraint(
            "length(artifact_sha256) = 64 AND artifact_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_release_artifact_sha256",
        ),
        CheckConstraint(
            "length(native_payload_sha256) = 64 AND native_payload_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_release_artifact_native_payload_sha256",
        ),
        CheckConstraint(
            "length(missing_provenance_sha256) = 64 "
            "AND missing_provenance_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_release_artifact_missing_provenance_sha256",
        ),
        CheckConstraint(
            "length(trim(artifact_path)) > 0",
            name="ck_release_artifact_path",
        ),
        CheckConstraint(
            "length(trim(provenance_json)) > 0",
            name="ck_release_artifact_provenance_json",
        ),
    )


class ReleaseObservation(Base):
    """One observation inside a complete :class:`DataRelease` snapshot."""

    __tablename__ = "release_observations"

    id = Column(Integer, primary_key=True)
    release_id = Column(
        Integer,
        ForeignKey("data_releases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    country = Column(String(8), nullable=False)
    indicator = Column(String(64), nullable=False)
    date = Column(Date, nullable=False)
    value = Column(Float, nullable=False)
    source = Column(String(32), nullable=False)
    series_id = Column(String(128), nullable=False)
    status = Column(String(24), nullable=False, default="observed")

    __table_args__ = (
        UniqueConstraint(
            "release_id",
            "country",
            "indicator",
            "date",
            "source",
            "series_id",
            name="uq_release_observation_key",
        ),
        Index(
            "ix_release_obs_lookup",
            "country",
            "indicator",
            "date",
            "source",
        ),
    )


class DebtHolderPosition(Base):
    """One typed holder-sector cell in an immutable sovereign-debt release.

    Holder positions are intentionally not squeezed into ``observations``.  Their
    issuer, instrument, counterpart sector, valuation measure and unit are all
    economically meaningful dimensions and must remain independently queryable.
    """

    __tablename__ = "debt_holder_positions"

    id = Column(Integer, primary_key=True)
    release_id = Column(
        Integer,
        ForeignKey("data_releases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    country = Column(String(8), nullable=False)
    date = Column(Date, nullable=False)
    issuer_sector_code = Column(String(32), nullable=False)
    issuer_sector_label = Column(String(192), nullable=False)
    instrument_code = Column(String(32), nullable=False)
    instrument_label = Column(String(192), nullable=False)
    holder_sector_code = Column(String(32), nullable=False)
    holder_sector_label = Column(String(192), nullable=False)
    measure_code = Column(String(32), nullable=False)
    measure_label = Column(String(192), nullable=False)
    unit = Column(String(64), nullable=False)
    value = Column(Float, nullable=False)
    source = Column(String(64), nullable=False)
    series_id = Column(String(192), nullable=False)
    status = Column(String(24), nullable=False, default="observed")

    __table_args__ = (
        UniqueConstraint(
            "release_id",
            "country",
            "date",
            "issuer_sector_code",
            "instrument_code",
            "holder_sector_code",
            "measure_code",
            "source",
            "series_id",
            name="uq_debt_holder_release_cell",
        ),
        Index(
            "ix_debt_holder_lookup",
            "country",
            "date",
            "instrument_code",
            "holder_sector_code",
        ),
    )


class CrossBorderPosition(Base):
    """One bilateral portfolio/direct-investment position in an immutable release.

    Positions stay out of the scalar ``observations`` projection because the
    reporter, counterpart, direction, accounting basis, instrument, frequency,
    and native observation status are all economically meaningful dimensions.
    """

    __tablename__ = "cross_border_positions"

    id = Column(Integer, primary_key=True)
    release_id = Column(
        Integer,
        ForeignKey("data_releases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    dataset = Column(String(8), nullable=False)
    reporter_country = Column(String(8), nullable=False)
    reporter_code = Column(String(16), nullable=False)
    counterpart_country = Column(String(8), nullable=False)
    counterpart_code = Column(String(16), nullable=False)
    date = Column(Date, nullable=False)
    direction = Column(String(32), nullable=False)
    accounting_basis = Column(String(48), nullable=False)
    instrument_code = Column(String(48), nullable=False)
    instrument_label = Column(String(192), nullable=False)
    frequency = Column(String(8), nullable=False)
    value = Column(Float, nullable=False)
    unit = Column(String(32), nullable=False)
    source = Column(String(32), nullable=False)
    native_indicator = Column(String(96), nullable=False)
    reporter_sector_code = Column(String(32), nullable=True)
    counterpart_sector_code = Column(String(32), nullable=True)
    derivation_type = Column(String(32), nullable=True)
    series_id = Column(String(256), nullable=False)
    status = Column(String(24), nullable=False, default="observed")

    __table_args__ = (
        UniqueConstraint(
            "release_id",
            "dataset",
            "reporter_code",
            "counterpart_code",
            "date",
            "direction",
            "instrument_code",
            "frequency",
            "source",
            "series_id",
            name="uq_cross_border_release_cell",
        ),
        Index(
            "ix_cross_border_lookup",
            "dataset",
            "reporter_country",
            "counterpart_country",
            "date",
            "direction",
            "instrument_code",
            "frequency",
        ),
    )


class AllocatorFact(Base):
    """One page-located numeric disclosure from a large allocator report."""

    __tablename__ = "allocator_facts"

    id = Column(Integer, primary_key=True)
    release_id = Column(
        Integer,
        ForeignKey("data_releases.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    fact_key = Column(String(64), nullable=False)
    fund = Column(String(16), nullable=False)
    as_of_date = Column(Date, nullable=False)
    period_start = Column(Date, nullable=True)
    period_end = Column(Date, nullable=True)
    record_type = Column(String(48), nullable=False)
    item_code = Column(String(96), nullable=False)
    reported_amount = Column(Float, nullable=True)
    reported_unit = Column(String(24), nullable=True)
    amount_sek_mn = Column(Float, nullable=True)
    exposure_pct = Column(Float, nullable=True)
    basis = Column(String(64), nullable=False)
    row_role = Column(String(24), nullable=False)
    physical_page = Column(Integer, nullable=False)
    table_heading = Column(Text, nullable=False)
    extraction_status = Column(String(32), nullable=False)
    quality_flag = Column(String(48), nullable=False)
    notes = Column(Text, nullable=True)
    artifact_sha256 = Column(String(64), nullable=False, index=True)
    artifact_path = Column(Text, nullable=False)
    source_url = Column(Text, nullable=False)
    parser_name = Column(String(96), nullable=False)
    parser_version = Column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint("release_id", "fact_key", name="uq_allocator_release_fact"),
        Index("ix_allocator_lookup", "fund", "as_of_date", "record_type", "item_code"),
    )


class CommunicationSchemaContract(Base):
    """Pinned SQLite DDL contract for the institutional-communications ledger."""

    __tablename__ = "communication_schema_contract"

    contract_id = Column(String(64), primary_key=True)
    schema_version = Column(Integer, nullable=False)
    schema_sha256 = Column(String(64), nullable=False)
    trigger_sha256 = Column(String(64), nullable=False)
    installed_at = Column(DateTime, nullable=False)

    __table_args__ = (
        CheckConstraint(
            "contract_id = 'institutional_communications'",
            name="ck_communication_schema_contract_id",
        ),
        CheckConstraint(
            "schema_version = 2",
            name="ck_communication_schema_contract_version",
        ),
        CheckConstraint(
            "length(schema_sha256) = 64 "
            "AND schema_sha256 NOT GLOB '*[^0-9a-f]*' "
            "AND length(trigger_sha256) = 64 "
            "AND trigger_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_schema_contract_sha256",
        ),
    )


class Organization(Base):
    """Stable identity anchor; descriptive metadata remains catalogue-versioned."""

    __tablename__ = "organizations"

    organization_id = Column(String(96), primary_key=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint(
            "length(organization_id) BETWEEN 1 AND 96 "
            "AND organization_id = trim(organization_id) "
            "AND substr(organization_id, 1, 1) GLOB '[a-z]' "
            "AND organization_id NOT GLOB '*[^a-z0-9_]*'",
            name="ck_organization_id",
        ),
    )


class CommunicationSourcePolicySnapshot(Base):
    """Immutable copy of one validated source policy used by persisted metadata.

    The checked-in catalogue is executable policy, while this table makes the
    exact policy consulted for an old write auditable after the Python catalogue
    changes.  Rows inserted outside the validated helper are not a trust
    boundary; SQLite constraints only provide structural defence in depth.
    """

    __tablename__ = "communication_source_policy_snapshots"

    catalogue_sha256 = Column(String(64), primary_key=True)
    source_id = Column(String(96), primary_key=True)
    organization_id = Column(
        String(96),
        ForeignKey("organizations.organization_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    organization_name = Column(String(192), nullable=False)
    organization_type = Column(String(32), nullable=False)
    jurisdiction = Column(String(32), nullable=False)
    language = Column(String(16), nullable=False)
    landing_url = Column(Text, nullable=False)
    official_domains_json = Column(Text, nullable=False)
    host_organization = Column(String(192), nullable=False)
    publisher = Column(String(192), nullable=False)
    transcriber = Column(String(192), nullable=True)
    transcriber_attribution = Column(String(32), nullable=False)
    material_types_json = Column(Text, nullable=False)
    commodity_families_json = Column(Text, nullable=False)
    verified_archive_start_year = Column(Integer, nullable=True)
    coverage_note = Column(Text, nullable=False)
    source_provenance_tier = Column(String(32), nullable=False)
    rights_status = Column(String(32), nullable=False)
    rights_basis_url = Column(Text, nullable=True)
    rights_note = Column(Text, nullable=False)
    acquisition_status = Column(String(32), nullable=False)
    acquisition_note = Column(Text, nullable=False)
    automated_collection_allowed = Column(Boolean, nullable=False)
    rights_checked_by = Column(String(192), nullable=True)
    rights_checked_at = Column(DateTime, nullable=True)
    catalogue_evaluated_at = Column(DateTime, nullable=False)
    policy_sha256 = Column(String(64), nullable=False)
    persisted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint(
            "length(catalogue_sha256) = 64 AND catalogue_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_policy_catalogue_sha256",
        ),
        CheckConstraint(
            "length(source_id) BETWEEN 1 AND 96 AND source_id = trim(source_id) "
            "AND substr(source_id, 1, 1) GLOB '[a-z]' "
            "AND source_id NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_policy_source_id",
        ),
        CheckConstraint(
            "organization_type IN ('central_bank', 'bank', 'commodity_company')",
            name="ck_communication_policy_organization_type",
        ),
        CheckConstraint(
            "source_provenance_tier IN "
            "('official_archive_mixed', 'official_authored_text', "
            "'official_published_transcript', 'official_hosted_third_party')",
            name="ck_communication_policy_provenance",
        ),
        CheckConstraint(
            "transcriber_attribution IN "
            "('artifact_specific', 'named_third_party', 'not_applicable', "
            "'not_disclosed', 'publisher')",
            name="ck_communication_policy_transcriber_attribution",
        ),
        CheckConstraint(
            "(transcriber_attribution IN ('artifact_specific', 'not_applicable', "
            "'not_disclosed') AND transcriber IS NULL) OR "
            "(transcriber_attribution = 'publisher' AND transcriber = publisher) OR "
            "(transcriber_attribution = 'named_third_party' AND transcriber IS NOT NULL "
            "AND length(trim(transcriber)) > 0 AND transcriber <> publisher)",
            name="ck_communication_policy_transcriber",
        ),
        CheckConstraint(
            "rights_status IN "
            "('cleared', 'internal_only', 'permission_required', 'metadata_only', "
            "'rights_review_required')",
            name="ck_communication_policy_rights",
        ),
        CheckConstraint(
            "(rights_status = 'cleared' AND acquisition_status = 'manual_collection_ready') OR "
            "(rights_status = 'internal_only' AND acquisition_status = 'manual_internal_only') OR "
            "(rights_status = 'permission_required' "
            "AND acquisition_status = 'blocked_pending_permission') OR "
            "(rights_status = 'metadata_only' AND acquisition_status = 'metadata_only') OR "
            "(rights_status = 'rights_review_required' "
            "AND acquisition_status = 'manual_review_required')",
            name="ck_communication_policy_rights_acquisition",
        ),
        CheckConstraint(
            "rights_status NOT IN ('cleared', 'internal_only', 'metadata_only', "
            "'permission_required') OR rights_basis_url IS NOT NULL",
            name="ck_communication_policy_required_rights_basis",
        ),
        CheckConstraint(
            "rights_basis_url IS NULL OR "
            "(rights_basis_url = trim(rights_basis_url) "
            "AND substr(rights_basis_url, 1, 8) = 'https://' "
            "AND instr(rights_basis_url, ' ') = 0 "
            "AND instr(rights_basis_url, char(9)) = 0 "
            "AND instr(rights_basis_url, char(10)) = 0 "
            "AND instr(rights_basis_url, char(13)) = 0 "
            "AND instr(rights_basis_url, '\\') = 0 "
            "AND instr(substr(rights_basis_url, 9), '/') > 1)",
            name="ck_communication_policy_rights_basis_url",
        ),
        CheckConstraint(
            "landing_url = trim(landing_url) AND substr(landing_url, 1, 8) = 'https://' "
            "AND instr(landing_url, ' ') = 0 AND instr(landing_url, char(9)) = 0 "
            "AND instr(landing_url, char(10)) = 0 AND instr(landing_url, char(13)) = 0 "
            "AND instr(landing_url, '\\') = 0 "
            "AND instr(substr(landing_url, 9), '/') > 1",
            name="ck_communication_policy_landing_url",
        ),
        CheckConstraint(
            "length(trim(organization_name)) > 0 "
            "AND length(trim(jurisdiction)) > 0 AND length(trim(language)) > 0 "
            "AND length(trim(host_organization)) > 0 AND length(trim(publisher)) > 0 "
            "AND length(trim(coverage_note)) > 0 AND length(trim(rights_note)) > 0 "
            "AND length(trim(acquisition_note)) > 0 "
            "AND json_valid(official_domains_json) = 1 "
            "AND json_type(official_domains_json) = 'array' "
            "AND json_array_length(official_domains_json) > 0 "
            "AND json_valid(material_types_json) = 1 "
            "AND json_type(material_types_json) = 'array' "
            "AND json_array_length(material_types_json) > 0 "
            "AND json_valid(commodity_families_json) = 1 "
            "AND json_type(commodity_families_json) = 'array'",
            name="ck_communication_policy_required_text",
        ),
        CheckConstraint(
            "verified_archive_start_year IS NULL OR "
            "verified_archive_start_year BETWEEN 1900 AND 9999",
            name="ck_communication_policy_archive_year",
        ),
        CheckConstraint(
            "automated_collection_allowed = 0",
            name="ck_communication_policy_automation_disabled",
        ),
        CheckConstraint(
            "(rights_checked_by IS NULL AND rights_checked_at IS NULL) OR "
            "(rights_checked_by IS NOT NULL AND length(trim(rights_checked_by)) > 6 "
            "AND substr(rights_checked_by, 1, 6) = 'human:' "
            "AND rights_checked_at IS NOT NULL)",
            name="ck_communication_policy_rights_review",
        ),
        CheckConstraint(
            "length(policy_sha256) = 64 AND policy_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_policy_sha256",
        ),
    )


class OrganizationCommodityCoverage(Base):
    """Effective-dated selection taxonomy, not an analytical exposure claim."""

    __tablename__ = "organization_commodity_coverage"

    id = Column(Integer, primary_key=True)
    organization_id = Column(
        String(96),
        ForeignKey("organizations.organization_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    coverage_key = Column(String(128), nullable=False)
    commodity_family = Column(String(96), nullable=False, index=True)
    exposure_role = Column(String(24), nullable=False)
    mapping_status = Column(String(32), nullable=False, index=True)
    effective_from = Column(Date, nullable=False)
    effective_to = Column(Date, nullable=True)
    source_id = Column(String(96), nullable=False, index=True)
    catalogue_sha256 = Column(String(64), nullable=False, index=True)
    evidence_url = Column(Text, nullable=False)
    evidence_note = Column(Text, nullable=False)
    published_at = Column(DateTime, nullable=True)
    available_at = Column(DateTime, nullable=False, index=True)
    retrieved_at = Column(DateTime, nullable=False)
    metadata_known_at = Column(DateTime, nullable=False, index=True)
    coverage_version_sha256 = Column(String(64), nullable=False, index=True)
    supersedes_exposure_id = Column(
        Integer,
        ForeignKey("organization_commodity_coverage.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    reviewed_by = Column(String(192), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        ForeignKeyConstraint(
            ["catalogue_sha256", "source_id"],
            [
                "communication_source_policy_snapshots.catalogue_sha256",
                "communication_source_policy_snapshots.source_id",
            ],
            ondelete="RESTRICT",
            name="fk_organization_commodity_source_policy",
        ),
        UniqueConstraint(
            "organization_id",
            "coverage_key",
            "metadata_known_at",
            name="uq_organization_commodity_exposure_version",
        ),
        Index(
            "ix_organization_commodity_exposure_effective",
            "organization_id",
            "commodity_family",
            "effective_from",
            "effective_to",
        ),
        Index(
            "uq_organization_commodity_exposure_successor",
            "supersedes_exposure_id",
            unique=True,
            sqlite_where=text("supersedes_exposure_id IS NOT NULL"),
        ),
        Index(
            "uq_organization_commodity_exposure_root",
            "organization_id",
            "coverage_key",
            unique=True,
            sqlite_where=text("supersedes_exposure_id IS NULL"),
        ),
        CheckConstraint(
            "length(coverage_key) BETWEEN 1 AND 128 "
            "AND coverage_key = trim(coverage_key) "
            "AND substr(coverage_key, 1, 1) GLOB '[a-z]' "
            "AND coverage_key NOT GLOB '*[^a-z0-9_]*'",
            name="ck_organization_commodity_coverage_key",
        ),
        CheckConstraint(
            "length(commodity_family) BETWEEN 1 AND 96 "
            "AND commodity_family = trim(commodity_family) "
            "AND substr(commodity_family, 1, 1) GLOB '[a-z]' "
            "AND commodity_family NOT GLOB '*[^a-z0-9_]*'",
            name="ck_organization_commodity_family",
        ),
        CheckConstraint(
            "exposure_role IN ('producer', 'processor', 'trader', 'consumer', 'integrated')",
            name="ck_organization_commodity_exposure_role",
        ),
        CheckConstraint(
            "mapping_status IN ('selection_taxonomy', 'evidence_reviewed')",
            name="ck_organization_commodity_mapping_status",
        ),
        CheckConstraint(
            "length(source_id) BETWEEN 1 AND 96 "
            "AND source_id = trim(source_id) "
            "AND substr(source_id, 1, 1) GLOB '[a-z]' "
            "AND source_id NOT GLOB '*[^a-z0-9_]*'",
            name="ck_organization_commodity_source_id",
        ),
        CheckConstraint(
            "length(catalogue_sha256) = 64 AND catalogue_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_organization_commodity_catalogue_sha256",
        ),
        CheckConstraint(
            "effective_to IS NULL OR effective_to >= effective_from",
            name="ck_organization_commodity_effective_period",
        ),
        CheckConstraint(
            "published_at IS NULL OR available_at >= published_at",
            name="ck_organization_commodity_published_clock",
        ),
        CheckConstraint(
            "retrieved_at >= available_at AND metadata_known_at >= retrieved_at",
            name="ck_organization_commodity_retrieval_clock",
        ),
        CheckConstraint(
            "evidence_url = trim(evidence_url) "
            "AND substr(evidence_url, 1, 8) = 'https://' "
            "AND instr(evidence_url, ' ') = 0 AND instr(evidence_url, char(9)) = 0 "
            "AND instr(evidence_url, char(10)) = 0 "
            "AND instr(evidence_url, char(13)) = 0 "
            "AND instr(evidence_url, '\\') = 0 "
            "AND instr(substr(evidence_url, 9), '/') > 1 "
            "AND length(trim(evidence_note)) > 0",
            name="ck_organization_commodity_evidence",
        ),
        CheckConstraint(
            "length(coverage_version_sha256) = 64 "
            "AND coverage_version_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_organization_commodity_version_sha256",
        ),
        CheckConstraint(
            "supersedes_exposure_id IS NULL OR supersedes_exposure_id <> id",
            name="ck_organization_commodity_not_self_superseding",
        ),
        CheckConstraint(
            "(mapping_status = 'selection_taxonomy' "
            "AND reviewed_by IS NULL AND reviewed_at IS NULL) OR "
            "(mapping_status = 'evidence_reviewed' "
            "AND reviewed_by IS NOT NULL AND length(trim(reviewed_by)) > 6 "
            "AND substr(reviewed_by, 1, 6) = 'human:' AND reviewed_at IS NOT NULL)",
            name="ck_organization_commodity_review",
        ),
    )


class CommunicationEvent(Base):
    """One dated institutional communication, independent of its representations."""

    __tablename__ = "communication_events"

    id = Column(Integer, primary_key=True)
    organization_id = Column(
        String(96),
        ForeignKey("organizations.organization_id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    event_key = Column(String(128), nullable=False)
    event_type = Column(String(64), nullable=False, index=True)
    title = Column(Text, nullable=False)
    event_date = Column(Date, nullable=False, index=True)
    event_started_at = Column(DateTime, nullable=True)
    reference_start = Column(Date, nullable=True)
    reference_end = Column(Date, nullable=True)
    metadata_known_at = Column(DateTime, nullable=False, index=True)
    event_version_sha256 = Column(String(64), nullable=False, index=True)
    supersedes_event_id = Column(
        Integer,
        ForeignKey("communication_events.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "organization_id",
            "event_key",
            "metadata_known_at",
            name="uq_communication_event_version",
        ),
        Index(
            "uq_communication_event_successor",
            "supersedes_event_id",
            unique=True,
            sqlite_where=text("supersedes_event_id IS NOT NULL"),
        ),
        Index(
            "uq_communication_event_root",
            "organization_id",
            "event_key",
            unique=True,
            sqlite_where=text("supersedes_event_id IS NULL"),
        ),
        Index(
            "ix_communication_event_history",
            "organization_id",
            "event_type",
            "event_date",
        ),
        CheckConstraint(
            "length(event_key) BETWEEN 1 AND 128 "
            "AND event_key = trim(event_key) "
            "AND substr(event_key, 1, 1) GLOB '[a-z]' "
            "AND event_key NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_event_key",
        ),
        CheckConstraint(
            "length(event_type) BETWEEN 1 AND 64 "
            "AND event_type = trim(event_type) "
            "AND substr(event_type, 1, 1) GLOB '[a-z]' "
            "AND event_type NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_event_type",
        ),
        CheckConstraint(
            "length(trim(title)) > 0",
            name="ck_communication_event_required_text",
        ),
        CheckConstraint(
            "(reference_start IS NULL AND reference_end IS NULL) OR "
            "(reference_start IS NOT NULL AND reference_end IS NOT NULL "
            "AND reference_end >= reference_start)",
            name="ck_communication_event_reference_period",
        ),
        CheckConstraint(
            "length(event_version_sha256) = 64 AND event_version_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_event_version_sha256",
        ),
        CheckConstraint(
            "supersedes_event_id IS NULL OR supersedes_event_id <> id",
            name="ck_communication_event_not_self_superseding",
        ),
    )


class CommunicationArtifact(Base):
    """One immutable official, hosted, captioned, or locally transcribed artifact."""

    __tablename__ = "communication_artifacts"

    id = Column(Integer, primary_key=True)
    event_id = Column(
        Integer,
        ForeignKey("communication_events.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    source_id = Column(String(96), nullable=False, index=True)
    catalogue_sha256 = Column(String(64), nullable=False, index=True)
    event_version_sha256 = Column(String(64), nullable=False, index=True)
    artifact_key = Column(String(128), nullable=False)
    artifact_role = Column(String(32), nullable=False, index=True)
    material_type = Column(String(48), nullable=False, index=True)
    language = Column(String(16), nullable=False)
    translation_status = Column(String(32), nullable=False)
    mime_type = Column(String(96), nullable=False)
    origin_type = Column(String(32), nullable=False, index=True)
    provenance_tier = Column(String(32), nullable=False, index=True)
    rights_status = Column(String(32), nullable=False, index=True)
    acquisition_status = Column(String(32), nullable=False, index=True)
    rights_basis_url = Column(Text, nullable=True)
    rights_note = Column(Text, nullable=False)
    rights_checked_by = Column(String(192), nullable=False)
    rights_checked_at = Column(DateTime, nullable=False)
    host_organization = Column(String(192), nullable=False)
    publisher = Column(String(192), nullable=False)
    transcriber = Column(String(192), nullable=True)
    transcriber_attribution = Column(String(32), nullable=False)
    published_at = Column(DateTime, nullable=True)
    available_at = Column(DateTime, nullable=False, index=True)
    retrieved_at = Column(DateTime, nullable=False)
    metadata_known_at = Column(DateTime, nullable=False, index=True)
    landing_url = Column(Text, nullable=False)
    artifact_url = Column(Text, nullable=False)
    artifact_version_sha256 = Column(String(64), nullable=False, index=True)
    supersedes_artifact_id = Column(
        Integer,
        ForeignKey("communication_artifacts.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        ForeignKeyConstraint(
            ["catalogue_sha256", "source_id"],
            [
                "communication_source_policy_snapshots.catalogue_sha256",
                "communication_source_policy_snapshots.source_id",
            ],
            ondelete="RESTRICT",
            name="fk_communication_artifact_source_policy",
        ),
        UniqueConstraint(
            "event_id",
            "source_id",
            "artifact_key",
            "metadata_known_at",
            name="uq_communication_artifact_version",
        ),
        Index(
            "uq_communication_artifact_successor",
            "supersedes_artifact_id",
            unique=True,
            sqlite_where=text("supersedes_artifact_id IS NOT NULL"),
        ),
        Index(
            "ix_communication_artifact_history",
            "event_id",
            "artifact_role",
            "available_at",
        ),
        CheckConstraint(
            "length(source_id) BETWEEN 1 AND 96 "
            "AND source_id = trim(source_id) "
            "AND substr(source_id, 1, 1) GLOB '[a-z]' "
            "AND source_id NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_artifact_source_id",
        ),
        CheckConstraint(
            "length(catalogue_sha256) = 64 AND catalogue_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_artifact_catalogue_sha256",
        ),
        CheckConstraint(
            "length(artifact_key) BETWEEN 1 AND 128 "
            "AND artifact_key = trim(artifact_key) "
            "AND substr(artifact_key, 1, 1) GLOB '[a-z]' "
            "AND artifact_key NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_artifact_key",
        ),
        CheckConstraint(
            "artifact_role IN "
            "('prepared_remarks', 'q_and_a_transcript', 'full_transcript', "
            "'ceo_letter', 'chair_letter', 'annual_report', 'subtitles', "
            "'webcast_video')",
            name="ck_communication_artifact_role",
        ),
        CheckConstraint(
            "(artifact_role = 'prepared_remarks' AND material_type IN "
            "('financial_results', 'management_review', 'monetary_policy_statement', "
            "'speech_text')) OR "
            "(artifact_role = 'q_and_a_transcript' "
            "AND material_type = 'questions_and_answers') OR "
            "(artifact_role = 'full_transcript' AND material_type IN "
            "('press_conference_transcript', 'results_transcript')) OR "
            "(artifact_role = 'ceo_letter' AND material_type = 'ceo_letter') OR "
            "(artifact_role = 'chair_letter' "
            "AND material_type IN ('annual_report', 'management_review')) OR "
            "(artifact_role = 'annual_report' AND material_type = 'annual_report') OR "
            "(artifact_role = 'subtitles' AND material_type = 'subtitles') OR "
            "(artifact_role = 'webcast_video' "
            "AND material_type = 'press_conference_video')",
            name="ck_communication_artifact_role_material",
        ),
        CheckConstraint(
            "translation_status IN ('original', 'official_translation')",
            name="ck_communication_artifact_translation_status",
        ),
        CheckConstraint(
            "origin_type IN "
            "('publisher_authored', 'official_published_transcript', "
            "'official_published_media', 'official_hosted_vendor', 'official_caption', "
            "'automatic_caption', 'local_asr')",
            name="ck_communication_artifact_origin",
        ),
        CheckConstraint(
            "(origin_type = 'publisher_authored' "
            "AND provenance_tier = 'official_authored_text') OR "
            "(origin_type = 'official_published_transcript' "
            "AND provenance_tier = 'official_published_transcript') OR "
            "(origin_type = 'official_published_media' "
            "AND provenance_tier = 'official_published_media') OR "
            "(origin_type = 'official_hosted_vendor' "
            "AND provenance_tier = 'official_hosted_third_party') OR "
            "(origin_type = 'official_caption' "
            "AND provenance_tier = 'official_caption') OR "
            "(origin_type = 'automatic_caption' "
            "AND provenance_tier = 'official_hosted_automatic_caption') OR "
            "(origin_type = 'local_asr' AND provenance_tier = 'local_derived_asr')",
            name="ck_communication_artifact_provenance_tier",
        ),
        CheckConstraint(
            "(material_type IN ('annual_report', 'ceo_letter', 'financial_results', "
            "'management_review', 'monetary_policy_statement', 'speech_text') "
            "AND origin_type = 'publisher_authored') OR "
            "(material_type IN ('questions_and_answers', 'press_conference_transcript', "
            "'results_transcript') AND origin_type IN "
            "('official_published_transcript', 'official_hosted_vendor')) OR "
            "(material_type = 'subtitles' AND origin_type IN "
            "('official_caption', 'automatic_caption', 'local_asr')) OR "
            "(material_type = 'press_conference_video' "
            "AND origin_type = 'official_published_media')",
            name="ck_communication_artifact_material_origin",
        ),
        CheckConstraint(
            "rights_status IN "
            "('cleared', 'internal_only', 'permission_required', 'metadata_only', "
            "'rights_review_required')",
            name="ck_communication_artifact_rights",
        ),
        CheckConstraint(
            "acquisition_status IN "
            "('blocked_pending_permission', 'manual_collection_ready', "
            "'manual_internal_only', 'manual_review_required', 'metadata_only')",
            name="ck_communication_artifact_acquisition",
        ),
        CheckConstraint(
            "(rights_status = 'cleared' AND acquisition_status = 'manual_collection_ready') OR "
            "(rights_status = 'internal_only' "
            "AND acquisition_status = 'manual_internal_only') OR "
            "(rights_status = 'permission_required' "
            "AND acquisition_status = 'blocked_pending_permission') OR "
            "(rights_status = 'metadata_only' AND acquisition_status = 'metadata_only') OR "
            "(rights_status = 'rights_review_required' "
            "AND acquisition_status = 'manual_review_required')",
            name="ck_communication_artifact_rights_acquisition",
        ),
        CheckConstraint(
            "rights_status <> 'permission_required' OR rights_basis_url IS NOT NULL",
            name="ck_communication_artifact_required_rights_basis",
        ),
        CheckConstraint(
            "rights_basis_url IS NULL OR "
            "(rights_basis_url = trim(rights_basis_url) "
            "AND substr(rights_basis_url, 1, 8) = 'https://' "
            "AND instr(rights_basis_url, ' ') = 0 "
            "AND instr(rights_basis_url, char(9)) = 0 "
            "AND instr(rights_basis_url, char(10)) = 0 "
            "AND instr(rights_basis_url, char(13)) = 0 "
            "AND instr(rights_basis_url, '\\') = 0 "
            "AND instr(substr(rights_basis_url, 9), '/') > 1)",
            name="ck_communication_artifact_rights_basis_url",
        ),
        CheckConstraint(
            "length(trim(language)) > 0 AND length(trim(mime_type)) > 0 "
            "AND length(trim(host_organization)) > 0 AND length(trim(publisher)) > 0 "
            "AND length(trim(landing_url)) > 0 AND length(trim(artifact_url)) > 0",
            name="ck_communication_artifact_required_text",
        ),
        CheckConstraint(
            "transcriber_attribution IN "
            "('named_third_party', 'not_applicable', 'not_disclosed', 'publisher')",
            name="ck_communication_artifact_transcriber_attribution",
        ),
        CheckConstraint(
            "(origin_type IN ('publisher_authored', 'official_published_media') "
            "AND transcriber IS NULL "
            "AND transcriber_attribution = 'not_applicable') OR "
            "(origin_type IN ('official_hosted_vendor', 'automatic_caption', 'local_asr') "
            "AND transcriber IS NOT NULL AND length(trim(transcriber)) > 0 "
            "AND transcriber <> publisher "
            "AND transcriber_attribution = 'named_third_party') OR "
            "(origin_type IN ('official_published_transcript', 'official_caption') AND "
            "((transcriber IS NULL AND transcriber_attribution = 'not_disclosed') OR "
            "(transcriber = publisher AND transcriber_attribution = 'publisher') OR "
            "(transcriber IS NOT NULL AND length(trim(transcriber)) > 0 "
            "AND transcriber <> publisher "
            "AND transcriber_attribution = 'named_third_party')))",
            name="ck_communication_artifact_transcriber",
        ),
        CheckConstraint(
            "length(trim(rights_note)) > 0 "
            "AND length(trim(rights_checked_by)) > 6 "
            "AND substr(rights_checked_by, 1, 6) = 'human:'",
            name="ck_communication_artifact_rights_review",
        ),
        CheckConstraint(
            "published_at IS NULL OR available_at >= published_at",
            name="ck_communication_artifact_published_clock",
        ),
        CheckConstraint(
            "retrieved_at >= available_at AND metadata_known_at >= retrieved_at",
            name="ck_communication_artifact_retrieval_clock",
        ),
        CheckConstraint(
            "metadata_known_at >= rights_checked_at",
            name="ck_communication_artifact_rights_known_clock",
        ),
        CheckConstraint(
            "landing_url = trim(landing_url) AND artifact_url = trim(artifact_url) "
            "AND substr(landing_url, 1, 8) = 'https://' "
            "AND substr(artifact_url, 1, 8) = 'https://' "
            "AND instr(landing_url, ' ') = 0 AND instr(artifact_url, ' ') = 0 "
            "AND instr(landing_url, char(9)) = 0 AND instr(artifact_url, char(9)) = 0 "
            "AND instr(landing_url, char(10)) = 0 AND instr(artifact_url, char(10)) = 0 "
            "AND instr(landing_url, char(13)) = 0 AND instr(artifact_url, char(13)) = 0 "
            "AND instr(landing_url, '\\') = 0 AND instr(artifact_url, '\\') = 0 "
            "AND instr(substr(landing_url, 9), '/') > 1 "
            "AND instr(substr(artifact_url, 9), '/') > 1",
            name="ck_communication_artifact_urls",
        ),
        CheckConstraint(
            "supersedes_artifact_id IS NULL OR supersedes_artifact_id <> id",
            name="ck_communication_artifact_not_self_superseding",
        ),
        CheckConstraint(
            "length(event_version_sha256) = 64 AND event_version_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_artifact_event_version_sha256",
        ),
        CheckConstraint(
            "length(artifact_version_sha256) = 64 "
            "AND artifact_version_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_artifact_version_sha256",
        ),
    )


class CommunicationArtifactSectionScopeSet(Base):
    """Atomic, immutable declaration of the semantic sections in one artifact.

    The canonical JSON array keeps the ordered scope set indivisible: a mixed
    document can map one artifact (and therefore every immutable byte capture
    of it) to several semantic sections without creating duplicate artifacts or
    partially inserting child rows.  ``metadata_known_at`` records when this
    interpretation became available; changed interpretations require a new
    artifact version rather than mutation of this row.
    """

    __tablename__ = "communication_artifact_section_scope_sets"

    artifact_id = Column(
        Integer,
        ForeignKey("communication_artifacts.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    artifact_version_sha256 = Column(String(64), nullable=False, index=True)
    scope_count = Column(Integer, nullable=False)
    scopes_json = Column(Text, nullable=False)
    scope_set_sha256 = Column(String(64), nullable=False, index=True)
    metadata_known_at = Column(DateTime, nullable=False, index=True)
    canonicalization_version = Column(String(64), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        CheckConstraint(
            "scope_count > 0 AND json_valid(scopes_json) "
            "AND json_type(scopes_json) = 'array' "
            "AND json_array_length(scopes_json) = scope_count",
            name="ck_communication_section_scope_set_shape",
        ),
        CheckConstraint(
            "length(artifact_version_sha256) = 64 "
            "AND artifact_version_sha256 NOT GLOB '*[^0-9a-f]*' "
            "AND length(scope_set_sha256) = 64 "
            "AND scope_set_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_section_scope_set_sha256",
        ),
        CheckConstraint(
            "canonicalization_version = 'communication_artifact_section_scopes_json_v1'",
            name="ck_communication_section_scope_set_canonicalization",
        ),
    )


class CommunicationArtifactRetrieval(Base):
    """Append-only observation that an unchanged artifact link was revisited."""

    __tablename__ = "communication_artifact_retrievals"

    id = Column(Integer, primary_key=True)
    artifact_id = Column(
        Integer,
        ForeignKey("communication_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    retrieved_at = Column(DateTime, nullable=False, index=True)
    metadata_known_at = Column(DateTime, nullable=False, index=True)
    landing_url = Column(Text, nullable=False)
    artifact_url = Column(Text, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "artifact_id",
            "retrieved_at",
            name="uq_communication_artifact_retrieval_clock",
        ),
        CheckConstraint(
            "landing_url = trim(landing_url) AND artifact_url = trim(artifact_url) "
            "AND substr(landing_url, 1, 8) = 'https://' "
            "AND substr(artifact_url, 1, 8) = 'https://' "
            "AND instr(landing_url, ' ') = 0 AND instr(artifact_url, ' ') = 0 "
            "AND instr(landing_url, char(9)) = 0 AND instr(artifact_url, char(9)) = 0 "
            "AND instr(landing_url, char(10)) = 0 AND instr(artifact_url, char(10)) = 0 "
            "AND instr(landing_url, char(13)) = 0 AND instr(artifact_url, char(13)) = 0 "
            "AND instr(landing_url, '\\') = 0 AND instr(artifact_url, '\\') = 0 "
            "AND instr(substr(landing_url, 9), '/') > 1 "
            "AND instr(substr(artifact_url, 9), '/') > 1",
            name="ck_communication_artifact_retrieval_urls",
        ),
        CheckConstraint(
            "metadata_known_at >= retrieved_at",
            name="ck_communication_artifact_retrieval_metadata_clock",
        ),
    )


class CommunicationArtifactContent(Base):
    """One immutable byte capture made during an artifact retrieval.

    Keeping captures separate lets link metadata be discovered before rights are
    cleared and lets changed bytes at an unchanged official URL append without
    rewriting the artifact record.
    """

    __tablename__ = "communication_artifact_contents"

    id = Column(Integer, primary_key=True)
    artifact_id = Column(
        Integer,
        ForeignKey("communication_artifacts.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    retrieval_id = Column(
        Integer,
        ForeignKey("communication_artifact_retrievals.id", ondelete="RESTRICT"),
        nullable=False,
        unique=True,
    )
    content_sha256 = Column(String(64), nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False)
    blob_path = Column(Text, nullable=False)
    captured_at = Column(DateTime, nullable=False, index=True)
    supersedes_content_id = Column(
        Integer,
        ForeignKey("communication_artifact_contents.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        Index(
            "uq_communication_artifact_content_successor",
            "supersedes_content_id",
            unique=True,
            sqlite_where=text("supersedes_content_id IS NOT NULL"),
        ),
        Index(
            "uq_communication_artifact_content_root",
            "artifact_id",
            unique=True,
            sqlite_where=text("supersedes_content_id IS NULL"),
        ),
        CheckConstraint(
            "length(content_sha256) = 64 AND content_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_artifact_content_sha256",
        ),
        CheckConstraint(
            "size_bytes > 0",
            name="ck_communication_artifact_content_size",
        ),
        CheckConstraint(
            "blob_path = 'artifacts/communications/sha256/' "
            "|| substr(content_sha256, 1, 2) || '/' || content_sha256",
            name="ck_communication_artifact_content_path",
        ),
        CheckConstraint(
            "supersedes_content_id IS NULL OR supersedes_content_id <> id",
            name="ck_communication_artifact_content_not_self_superseding",
        ),
    )


class CommunicationExtraction(Base):
    """Immutable extraction anchor; it is not complete until finalized."""

    __tablename__ = "communication_extractions"

    id = Column(Integer, primary_key=True)
    artifact_content_id = Column(
        Integer,
        ForeignKey("communication_artifact_contents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    run_key = Column(String(128), nullable=False)
    extractor_name = Column(String(128), nullable=False)
    extractor_version = Column(String(128), nullable=False)
    extractor_config_sha256 = Column(String(64), nullable=False)
    extracted_at = Column(DateTime, nullable=False)
    run_sha256 = Column(String(64), nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "artifact_content_id",
            "run_key",
            name="uq_communication_extraction_version",
        ),
        CheckConstraint(
            "length(run_key) BETWEEN 1 AND 128 "
            "AND run_key = trim(run_key) "
            "AND substr(run_key, 1, 1) GLOB '[a-z]' "
            "AND run_key NOT GLOB '*[^a-z0-9_]*'",
            name="ck_communication_extraction_run_key",
        ),
        CheckConstraint(
            "length(trim(extractor_name)) > 0 AND length(trim(extractor_version)) > 0",
            name="ck_communication_extraction_required_text",
        ),
        CheckConstraint(
            "length(extractor_config_sha256) = 64 "
            "AND extractor_config_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_extraction_inputs_sha256",
        ),
        CheckConstraint(
            "length(run_sha256) = 64 AND run_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_extraction_run_sha256",
        ),
    )


class CommunicationSegment(Base):
    """Speaker- and locator-addressable text from one communication extraction."""

    __tablename__ = "communication_segments"

    extraction_id = Column(
        Integer,
        ForeignKey("communication_extractions.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    ordinal = Column(Integer, primary_key=True)
    section_ordinal = Column(Integer, nullable=False)
    segment_kind = Column(String(32), nullable=False, index=True)
    speaker_name = Column(String(192), nullable=True)
    speaker_role = Column(String(192), nullable=True)
    speaker_side = Column(String(16), nullable=False, index=True)
    section_title = Column(Text, nullable=True)
    text = Column(Text, nullable=False)
    text_sha256 = Column(String(64), nullable=False)
    char_count = Column(Integer, nullable=False)
    page_start = Column(Integer, nullable=True)
    page_end = Column(Integer, nullable=True)
    paragraph_start = Column(Integer, nullable=True)
    paragraph_end = Column(Integer, nullable=True)
    start_ms = Column(Integer, nullable=True)
    end_ms = Column(Integer, nullable=True)

    __table_args__ = (
        CheckConstraint("ordinal > 0", name="ck_communication_segment_ordinal"),
        CheckConstraint(
            "segment_kind IN "
            "('prepared_remarks', 'q_and_a_question', 'q_and_a_answer', "
            "'letter', 'narrative', 'heading', 'other')",
            name="ck_communication_segment_kind",
        ),
        CheckConstraint(
            "speaker_side IN ('publisher', 'external', 'moderator', 'unknown')",
            name="ck_communication_segment_speaker_side",
        ),
        CheckConstraint(
            "(segment_kind <> 'q_and_a_question' OR speaker_side <> 'publisher') "
            "AND (segment_kind <> 'q_and_a_answer' "
            "OR speaker_side IN ('publisher', 'unknown')) "
            "AND (segment_kind NOT IN ('prepared_remarks', 'letter') "
            "OR speaker_side = 'publisher')",
            name="ck_communication_segment_kind_side",
        ),
        CheckConstraint(
            "length(trim(text)) > 0 AND char_count = length(text)",
            name="ck_communication_segment_text",
        ),
        CheckConstraint(
            "length(text_sha256) = 64 AND text_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_segment_sha256",
        ),
        CheckConstraint(
            "(page_start IS NULL AND page_end IS NULL) OR "
            "(page_start IS NOT NULL AND page_end IS NOT NULL "
            "AND page_start > 0 AND page_end >= page_start)",
            name="ck_communication_segment_pages",
        ),
        CheckConstraint(
            "(paragraph_start IS NULL AND paragraph_end IS NULL) OR "
            "(paragraph_start IS NOT NULL AND paragraph_end IS NOT NULL "
            "AND paragraph_start > 0 AND paragraph_end >= paragraph_start)",
            name="ck_communication_segment_paragraphs",
        ),
        CheckConstraint(
            "(start_ms IS NULL AND end_ms IS NULL) OR "
            "(start_ms IS NOT NULL AND end_ms IS NOT NULL "
            "AND start_ms >= 0 AND end_ms > start_ms)",
            name="ck_communication_segment_timecodes",
        ),
    )


class CommunicationExtractionFinalization(Base):
    """Completion marker inserted only after all immutable segments exist."""

    __tablename__ = "communication_extraction_finalizations"

    extraction_id = Column(
        Integer,
        ForeignKey("communication_extractions.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    finalized_at = Column(DateTime, nullable=False)
    segment_count = Column(Integer, nullable=False)
    total_char_count = Column(Integer, nullable=False)
    corpus_sha256 = Column(String(64), nullable=False)
    canonicalization_version = Column(String(64), nullable=False)

    __table_args__ = (
        CheckConstraint(
            "segment_count > 0 AND total_char_count > 0",
            name="ck_communication_extraction_finalization_count",
        ),
        CheckConstraint(
            "length(corpus_sha256) = 64 AND corpus_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_communication_extraction_finalization_sha256",
        ),
        CheckConstraint(
            "canonicalization_version = 'communication_segments_json_v2'",
            name="ck_communication_extraction_canonicalization",
        ),
    )


class ReportDocument(Base):
    """Immutable bytes and publication provenance for one official report version."""

    __tablename__ = "report_documents"

    id = Column(Integer, primary_key=True)
    source_id = Column(String(96), nullable=False, index=True)
    report_family = Column(String(96), nullable=False, index=True)
    issue_key = Column(String(128), nullable=False)
    publisher = Column(String(192), nullable=False, index=True)
    jurisdiction = Column(String(32), nullable=False)
    title = Column(Text, nullable=False)
    language = Column(String(16), nullable=False)
    document_date = Column(Date, nullable=False)
    published_at = Column(DateTime, nullable=True)
    available_at = Column(DateTime, nullable=False, index=True)
    retrieved_at = Column(DateTime, nullable=False)
    landing_url = Column(Text, nullable=False)
    artifact_url = Column(Text, nullable=False)
    mime_type = Column(String(64), nullable=False)
    content_sha256 = Column(String(64), nullable=False, index=True)
    size_bytes = Column(Integer, nullable=False)
    page_count = Column(Integer, nullable=False)
    blob_path = Column(Text, nullable=False)
    supersedes_document_id = Column(
        Integer,
        ForeignKey("report_documents.id", ondelete="RESTRICT"),
        nullable=True,
    )
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))

    __table_args__ = (
        UniqueConstraint(
            "source_id",
            "issue_key",
            "content_sha256",
            name="uq_report_document_issue_content",
        ),
        Index(
            "ix_report_document_issue_available",
            "source_id",
            "issue_key",
            "available_at",
        ),
    )


class DocumentExtraction(Base):
    """One immutable, reproducible page-extraction run for a report document."""

    __tablename__ = "document_extractions"

    id = Column(Integer, primary_key=True)
    document_id = Column(
        Integer,
        ForeignKey("report_documents.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    extractor_name = Column(String(128), nullable=False)
    extractor_version = Column(String(128), nullable=False)
    extracted_at = Column(DateTime, nullable=False)
    status = Column(String(24), nullable=False)
    extracted_page_count = Column(Integer, nullable=False)
    corpus_sha256 = Column(String(64), nullable=True)
    error = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "extractor_name",
            "extractor_version",
            name="uq_document_extraction_version",
        ),
    )


class DocumentPage(Base):
    """Page-addressable text from one immutable extraction run."""

    __tablename__ = "document_pages"

    extraction_id = Column(
        Integer,
        ForeignKey("document_extractions.id", ondelete="RESTRICT"),
        primary_key=True,
    )
    pdf_page = Column(Integer, primary_key=True)
    printed_page_label = Column(String(64), nullable=True)
    text = Column(Text, nullable=False)
    text_sha256 = Column(String(64), nullable=False)
    char_count = Column(Integer, nullable=False)


class Claim(Base):
    """An atomic publisher statement or explicitly attributed Observatory inference."""

    __tablename__ = "claims"

    id = Column(Integer, primary_key=True)
    claim_type = Column(String(24), nullable=False, index=True)
    statement = Column(Text, nullable=False)
    attribution_document_id = Column(
        Integer,
        ForeignKey("report_documents.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    claim_series_key = Column(String(192), nullable=True, index=True)
    topic_key = Column(String(96), nullable=False, index=True)
    geographies_json = Column(Text, nullable=False)
    reference_start = Column(Date, nullable=True)
    reference_end = Column(Date, nullable=True)
    target_start = Column(Date, nullable=True)
    target_end = Column(Date, nullable=True)
    numeric_value = Column(Float, nullable=True)
    lower_bound = Column(Float, nullable=True)
    upper_bound = Column(Float, nullable=True)
    unit = Column(String(96), nullable=True)
    condition_text = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)
    status = Column(String(24), nullable=False, index=True)
    available_at = Column(DateTime, nullable=False, index=True)
    created_by = Column(String(192), nullable=False)
    created_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    reviewed_by = Column(String(192), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    review_of_claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )
    supersedes_claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=True,
        index=True,
    )

    __table_args__ = (
        Index("ix_claim_asof", "status", "available_at", "claim_type", "topic_key"),
        Index(
            "uq_claim_verified_review",
            "review_of_claim_id",
            unique=True,
            sqlite_where=text("status = 'verified' AND review_of_claim_id IS NOT NULL"),
        ),
        Index(
            "uq_claim_verified_successor",
            "supersedes_claim_id",
            unique=True,
            sqlite_where=text("status = 'verified' AND supersedes_claim_id IS NOT NULL"),
        ),
    )


class ClaimCitation(Base):
    """A physical-page locator and bounded supporting excerpt for one claim."""

    __tablename__ = "claim_citations"

    id = Column(Integer, primary_key=True)
    claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    extraction_id = Column(
        Integer,
        ForeignKey("document_extractions.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    pdf_page_start = Column(Integer, nullable=False)
    pdf_page_end = Column(Integer, nullable=False)
    printed_locator = Column(String(128), nullable=True)
    section_title = Column(Text, nullable=True)
    evidence_excerpt = Column(Text, nullable=False)
    excerpt_sha256 = Column(String(64), nullable=False)
    support_role = Column(String(24), nullable=False)
    locator_verified_at = Column(DateTime, nullable=False)
    semantic_verified_by = Column(String(192), nullable=True)
    semantic_verified_at = Column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "claim_id",
            "extraction_id",
            "pdf_page_start",
            "pdf_page_end",
            "excerpt_sha256",
            name="uq_claim_citation_locator",
        ),
    )


class ReportCandidateReview(Base):
    """Immutable human disposition of one hash-bound report candidate."""

    __tablename__ = "report_candidate_reviews"

    id = Column(Integer, primary_key=True)
    candidate_id = Column(String(64), nullable=False)
    packet_sha256 = Column(String(64), nullable=False, index=True)
    candidate_catalogue_sha256 = Column(String(64), nullable=False)
    candidate_json = Column(Text, nullable=False)
    outcome = Column(String(16), nullable=False, index=True)
    reviewer = Column(String(192), nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=False, index=True)
    reason_code = Column(String(32), nullable=True, index=True)
    review_note = Column(Text, nullable=False)
    checklist_json = Column(Text, nullable=False)
    request_sha256 = Column(String(64), nullable=False)
    recorded_at = Column(DateTime, nullable=False, default=lambda: datetime.now(UTC))
    original_draft_claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=False,
    )
    revised_draft_claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=True,
    )
    verified_claim_id = Column(
        Integer,
        ForeignKey("claims.id", ondelete="RESTRICT"),
        nullable=True,
    )

    __table_args__ = (
        UniqueConstraint("candidate_id", name="uq_report_candidate_review_candidate"),
        UniqueConstraint(
            "original_draft_claim_id",
            name="uq_report_candidate_review_original_draft",
        ),
        UniqueConstraint(
            "revised_draft_claim_id",
            name="uq_report_candidate_review_revised_draft",
        ),
        UniqueConstraint(
            "verified_claim_id",
            name="uq_report_candidate_review_verified_claim",
        ),
        CheckConstraint(
            "length(candidate_id) = 64 AND candidate_id NOT GLOB '*[^0-9a-f]*'",
            name="ck_report_candidate_review_candidate_sha256",
        ),
        CheckConstraint(
            "length(packet_sha256) = 64 AND packet_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_report_candidate_review_packet_sha256",
        ),
        CheckConstraint(
            "length(candidate_catalogue_sha256) = 64 "
            "AND candidate_catalogue_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_report_candidate_review_catalogue_sha256",
        ),
        CheckConstraint(
            "length(request_sha256) = 64 AND request_sha256 NOT GLOB '*[^0-9a-f]*'",
            name="ck_report_candidate_review_request_sha256",
        ),
        CheckConstraint(
            "json_valid(candidate_json) AND json_type(candidate_json) = 'object' "
            "AND candidate_json = trim(candidate_json)",
            name="ck_report_candidate_review_candidate_json",
        ),
        CheckConstraint(
            "json_valid(checklist_json) AND json_type(checklist_json) = 'object' "
            "AND checklist_json = trim(checklist_json)",
            name="ck_report_candidate_review_checklist_json",
        ),
        CheckConstraint(
            "outcome IN ('approve', 'revise', 'reject')",
            name="ck_report_candidate_review_outcome",
        ),
        CheckConstraint(
            "(outcome = 'approve' AND reason_code IS NULL) OR "
            "(outcome IN ('revise', 'reject') AND reason_code IN "
            "('unsupported', 'misattributed', 'wrong_type', 'wrong_scope', "
            "'wrong_period_or_unit', 'missing_condition', 'not_material', "
            "'duplicate', 'other'))",
            name="ck_report_candidate_review_reason_code",
        ),
        CheckConstraint(
            "reviewer = trim(reviewer) AND length(reviewer) > 6 "
            "AND substr(reviewer, 1, 6) = 'human:'",
            name="ck_report_candidate_review_human_reviewer",
        ),
        CheckConstraint(
            "review_note = trim(review_note) AND length(review_note) > 0",
            name="ck_report_candidate_review_note_shape",
        ),
        CheckConstraint(
            "(outcome = 'approve' AND revised_draft_claim_id IS NULL "
            "AND verified_claim_id IS NOT NULL) OR "
            "(outcome = 'revise' AND revised_draft_claim_id IS NOT NULL "
            "AND verified_claim_id IS NOT NULL) OR "
            "(outcome = 'reject' AND revised_draft_claim_id IS NULL "
            "AND verified_claim_id IS NULL)",
            name="ck_report_candidate_review_outcome_links",
        ),
        CheckConstraint(
            "revised_draft_claim_id IS NULL OR revised_draft_claim_id <> original_draft_claim_id",
            name="ck_report_candidate_review_distinct_revised_draft",
        ),
        CheckConstraint(
            "verified_claim_id IS NULL OR verified_claim_id <> original_draft_claim_id",
            name="ck_report_candidate_review_distinct_verified",
        ),
        CheckConstraint(
            "revised_draft_claim_id IS NULL OR verified_claim_id IS NULL "
            "OR revised_draft_claim_id <> verified_claim_id",
            name="ck_report_candidate_review_distinct_revision_verification",
        ),
        CheckConstraint(
            "recorded_at >= reviewed_at",
            name="ck_report_candidate_review_recorded_after_review",
        ),
    )


def _prevent_evidence_mutation(_mapper, _connection, target) -> None:
    raise ValueError(f"{type(target).__name__} rows are immutable; append a new version")


for _immutable_model in (
    DataRelease,
    DataReleaseArtifact,
    ReleaseObservation,
    DebtHolderPosition,
    CrossBorderPosition,
    AllocatorFact,
    CommunicationSchemaContract,
    Organization,
    CommunicationSourcePolicySnapshot,
    OrganizationCommodityCoverage,
    CommunicationEvent,
    CommunicationArtifact,
    CommunicationArtifactSectionScopeSet,
    CommunicationArtifactRetrieval,
    CommunicationArtifactContent,
    CommunicationExtraction,
    CommunicationSegment,
    CommunicationExtractionFinalization,
    ReportDocument,
    DocumentExtraction,
    DocumentPage,
    Claim,
    ClaimCitation,
    ReportCandidateReview,
):
    event.listen(_immutable_model, "before_update", _prevent_evidence_mutation)
    event.listen(_immutable_model, "before_delete", _prevent_evidence_mutation)


def get_db_path() -> Path:
    raw = os.environ.get("DALIO_DB_PATH", "data/dalio.db")
    p = Path(raw)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def make_engine(db_path: Path | None = None) -> Engine:
    path = db_path if db_path is not None else get_db_path()
    engine = create_engine(f"sqlite:///{path}", future=True)

    def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    event.listen(engine, "connect", _enable_foreign_keys)
    return engine


_LEGACY_RELEASE_UNIQUE = ("partition_key", "content_sha256")
_INTERIM_RELEASE_UNIQUE = (
    "partition_key",
    "content_sha256",
    "available_at",
    "retrieved_at",
)
_CURRENT_RELEASE_UNIQUE = ("partition_key", "available_at", "retrieved_at")
_RELEASE_MIGRATION_BACKUP_SUFFIX = ".before-release-clock-migration.sqlite3"


def release_recurrence_migration_backup_path(engine: Engine) -> Path:
    """Return the deterministic, adjacent backup path for the release migration."""

    if engine.dialect.name != "sqlite":
        raise RuntimeError("release recurrence migration only supports SQLite")
    database = engine.url.database
    if not database or database == ":memory:":
        raise RuntimeError("release recurrence migration needs a file-backed SQLite database")
    source = Path(database).expanduser().resolve()
    return source.with_name(f"{source.name}{_RELEASE_MIGRATION_BACKUP_SUFFIX}")


def _quote_sqlite_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _sqlite_read_only(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)


def _verify_exact_sqlite_backup(source_path: Path, backup_path: Path) -> None:
    """Fail unless ``backup_path`` is a complete logical copy of ``source_path``."""

    if source_path.samefile(backup_path):
        raise RuntimeError(
            "SQLite backup must be a distinct filesystem object from its source"
        )

    with (
        closing(_sqlite_read_only(source_path)) as source,
        closing(_sqlite_read_only(backup_path)) as backup,
    ):
        for label, connection in (("source", source), ("backup", backup)):
            integrity = connection.execute("PRAGMA integrity_check").fetchall()
            if integrity != [("ok",)]:
                raise RuntimeError(
                    f"release migration {label} failed SQLite integrity_check: {integrity[:3]!r}"
                )
            foreign_key_errors = connection.execute("PRAGMA foreign_key_check").fetchall()
            if foreign_key_errors:
                raise RuntimeError(
                    f"release migration {label} has foreign-key violations: "
                    f"{foreign_key_errors[:3]!r}"
                )

        schema_sql = "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        source_schema = source.execute(schema_sql).fetchall()
        backup_schema = backup.execute(schema_sql).fetchall()
        if backup_schema != source_schema:
            raise RuntimeError("release migration backup schema does not match the source")

        for pragma in ("application_id", "user_version"):
            source_value = source.execute(f"PRAGMA {pragma}").fetchone()
            backup_value = backup.execute(f"PRAGMA {pragma}").fetchone()
            if backup_value != source_value:
                raise RuntimeError(
                    f"release migration backup PRAGMA {pragma} does not match the source"
                )

        tables = [
            row[0]
            for row in source.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table_name in tables:
            quoted = _quote_sqlite_identifier(table_name)
            source_cursor = source.execute(f"SELECT * FROM {quoted}")
            backup_cursor = backup.execute(f"SELECT * FROM {quoted}")
            while True:
                source_rows = source_cursor.fetchmany(512)
                backup_rows = backup_cursor.fetchmany(512)
                if source_rows != backup_rows:
                    raise RuntimeError(
                        "release migration backup rows do not match the source "
                        f"for table {table_name!r}"
                    )
                if not source_rows:
                    break


def _create_or_verify_release_migration_backup(
    source_path: Path,
    backup_path: Path,
) -> None:
    """Create one online SQLite backup without overwriting prior recovery evidence."""

    if backup_path.exists():
        _verify_exact_sqlite_backup(source_path, backup_path)
        return

    backup_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{backup_path.name}.",
            suffix=".tmp",
            dir=backup_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        with (
            closing(_sqlite_read_only(source_path)) as source,
            closing(sqlite3.connect(temporary_path)) as destination,
        ):
            source.backup(destination)
        _verify_exact_sqlite_backup(source_path, temporary_path)
        try:
            os.link(temporary_path, backup_path)
        except FileExistsError:
            _verify_exact_sqlite_backup(source_path, backup_path)
        else:
            temporary_path.unlink()
        _verify_exact_sqlite_backup(source_path, backup_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
            Path(f"{temporary_path}-wal").unlink(missing_ok=True)
            Path(f"{temporary_path}-shm").unlink(missing_ok=True)


def create_verified_sqlite_backup(source_path: Path, backup_path: Path) -> Path:
    """Create or verify one exact, non-overwriting SQLite backup."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(backup_path).expanduser().resolve()
    if source == destination:
        raise ValueError("SQLite backup destination must differ from its source")
    if not source.is_file():
        raise FileNotFoundError(f"SQLite backup source does not exist: {source}")
    _create_or_verify_release_migration_backup(source, destination)
    return destination


def _release_unique_layout(cursor: sqlite3.Cursor) -> tuple[tuple[str, ...], ...]:
    layouts: list[tuple[str, ...]] = []
    for index_row in cursor.execute("PRAGMA index_list('data_releases')").fetchall():
        if not index_row[2]:
            continue
        index_name = _quote_sqlite_identifier(str(index_row[1]))
        columns = tuple(
            str(column_row[2])
            for column_row in cursor.execute(f"PRAGMA index_info({index_name})").fetchall()
        )
        layouts.append(columns)
    return tuple(sorted(layouts))


def _create_release_immutability_triggers(cursor: sqlite3.Cursor) -> None:
    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS data_releases_reject_update
        BEFORE UPDATE ON data_releases
        BEGIN
            SELECT RAISE(ABORT, 'data_releases rows are immutable');
        END
        """
    )
    cursor.execute(
        """
        CREATE TRIGGER IF NOT EXISTS data_releases_reject_delete
        BEFORE DELETE ON data_releases
        BEGIN
            SELECT RAISE(ABORT, 'data_releases rows are immutable');
        END
        """
    )


def _migrate_release_recurrence_constraint(engine: Engine) -> None:
    """Allow the same payload to recur after an intervening release.

    Early release-ledger databases made ``(partition_key, content_sha256)``
    unique.  That made a real A -> B -> A sequence impossible to record.  The
    replacement constraint makes the partition and both release clocks the
    identity of one source event. Consecutive identical refreshes are still
    deduplicated by ingestion, while a later recurrence can be appended as a
    distinct, immutable vintage.

    SQLite cannot drop a table constraint in place, so the migration rebuilds
    only the parent table while foreign-key enforcement is temporarily disabled.
    It first creates an adjacent online SQLite backup, copies every row, restores
    explicit indexes and triggers, and checks all child references before commit.
    """

    if engine.dialect.name != "sqlite":
        raise RuntimeError("release recurrence migration only supports SQLite")

    raw_connection = engine.raw_connection()
    cursor = raw_connection.cursor()
    try:
        table_sql = cursor.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'data_releases'"
        ).fetchone()
        if table_sql is None:
            return

        layout = _release_unique_layout(cursor)
        if layout == (_CURRENT_RELEASE_UNIQUE,):
            return
        supported = {(_LEGACY_RELEASE_UNIQUE,), (_INTERIM_RELEASE_UNIQUE,)}
        if layout not in supported:
            raise RuntimeError(
                "data_releases has an unknown uniqueness layout; refusing automatic "
                f"migration: {layout!r}"
            )

        cursor.execute("PRAGMA foreign_keys=OFF")
        cursor.execute("BEGIN IMMEDIATE")

        # Recheck after taking the write reservation so two initializers cannot
        # race the schema rebuild.
        layout = _release_unique_layout(cursor)
        if layout == (_CURRENT_RELEASE_UNIQUE,):
            raw_connection.rollback()
            return
        if layout not in supported:
            raise RuntimeError("data_releases uniqueness changed while awaiting the migration lock")

        duplicate_clocks = cursor.execute(
            """
            SELECT partition_key, available_at, retrieved_at, COUNT(*)
            FROM data_releases
            GROUP BY partition_key, available_at, retrieved_at
            HAVING COUNT(*) > 1
            LIMIT 3
            """
        ).fetchall()
        if duplicate_clocks:
            raise RuntimeError(
                "data_releases contains conflicting duplicate event clocks; "
                f"refusing automatic migration: {duplicate_clocks!r}"
            )

        source_path = Path(
            next(
                row[2]
                for row in cursor.execute("PRAGMA database_list").fetchall()
                if row[1] == "main"
            )
        ).resolve()
        backup_path = release_recurrence_migration_backup_path(engine)
        if source_path != Path(engine.url.database or "").expanduser().resolve():
            raise RuntimeError("SQLite engine path does not match its main database path")
        _create_or_verify_release_migration_backup(source_path, backup_path)

        unique_index_names = {
            str(index_row[1])
            for index_row in cursor.execute("PRAGMA index_list('data_releases')").fetchall()
            if index_row[2]
        }
        explicit_schema_objects = [
            schema_object
            for schema_object in cursor.execute(
                """
            SELECT type, name, sql
            FROM sqlite_master
            WHERE tbl_name = 'data_releases'
              AND type IN ('index', 'trigger')
              AND sql IS NOT NULL
            ORDER BY type, name
            """
            ).fetchall()
            if not (schema_object[0] == "index" and schema_object[1] in unique_index_names)
        ]
        old_row_count = cursor.execute("SELECT COUNT(*) FROM data_releases").fetchone()[0]
        cursor.execute(
            """
            CREATE TABLE data_releases__recurrence_migration (
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
                CONSTRAINT uq_release_partition_clock UNIQUE
                    (partition_key, available_at, retrieved_at)
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO data_releases__recurrence_migration
                (id, partition_key, source_family, published_at, available_at,
                 retrieved_at, vintage_label, source_url, content_sha256,
                 row_count, created_at)
            SELECT
                id, partition_key, source_family, published_at, available_at,
                retrieved_at, vintage_label, source_url, content_sha256,
                row_count, created_at
            FROM data_releases
            """
        )
        copied_row_count = cursor.execute(
            "SELECT COUNT(*) FROM data_releases__recurrence_migration"
        ).fetchone()[0]
        if copied_row_count != old_row_count:
            raise RuntimeError(
                "release recurrence migration copied an unexpected number of rows: "
                f"expected {old_row_count}, got {copied_row_count}"
            )
        copied_differences = cursor.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT * FROM data_releases
                EXCEPT
                SELECT * FROM data_releases__recurrence_migration
            )
            """
        ).fetchone()[0]
        if copied_differences:
            raise RuntimeError("release recurrence migration changed copied release rows")
        cursor.execute("DROP TABLE data_releases")
        cursor.execute("ALTER TABLE data_releases__recurrence_migration RENAME TO data_releases")
        for _object_type, _object_name, schema_sql in explicit_schema_objects:
            cursor.execute(schema_sql)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_data_releases_source_family "
            "ON data_releases (source_family)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS ix_release_partition_available "
            "ON data_releases (partition_key, available_at, retrieved_at)"
        )
        _create_release_immutability_triggers(cursor)
        violations = cursor.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise RuntimeError(
                f"release recurrence migration would break foreign keys: {violations[:3]!r}"
            )
        integrity = cursor.execute("PRAGMA integrity_check").fetchall()
        if integrity != [("ok",)]:
            raise RuntimeError(
                f"release recurrence migration failed SQLite integrity_check: {integrity[:3]!r}"
            )
        raw_connection.commit()
    except Exception:
        raw_connection.rollback()
        raise
    finally:
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
        finally:
            cursor.close()
            raw_connection.close()


_COMMUNICATION_SCHEMA_V1_TABLES = (
    "communication_schema_contract",
    "organizations",
    "communication_source_policy_snapshots",
    "organization_commodity_coverage",
    "communication_events",
    "communication_artifacts",
    "communication_artifact_retrievals",
    "communication_artifact_contents",
    "communication_extractions",
    "communication_segments",
    "communication_extraction_finalizations",
)
_COMMUNICATION_SCHEMA_TABLES = (
    *_COMMUNICATION_SCHEMA_V1_TABLES[:6],
    "communication_artifact_section_scope_sets",
    *_COMMUNICATION_SCHEMA_V1_TABLES[6:],
)
_COMMUNICATION_CUSTOM_TRIGGERS = (
    "communication_policy_snapshot_validate",
    "communication_artifacts_match_policy",
    "communication_coverage_matches_policy",
    "communication_event_successor_order",
    "communication_artifact_reject_duplicate_root",
    "communication_artifact_successor_order",
    "communication_coverage_successor_order",
    "communication_coverage_reject_overlapping_head",
    "communication_retrieval_matches_artifact",
    "communication_content_matches_retrieval",
    "communication_content_successor_order",
    "communication_extractions_require_content",
    "communication_section_scope_set_validate",
    "communication_segments_match_scope_set",
    "communication_segments_reject_after_finalization",
    "communication_extraction_finalization_validate",
)
_COMMUNICATION_SCHEMA_V1_VERSION = 1
_COMMUNICATION_SCHEMA_V1_SHA256 = (
    "1fe8247a31b767f933b3c6f8646ee8b2536da2655bc8fc087ef99fe5f68dedb3"
)
_COMMUNICATION_TRIGGER_V1_SHA256 = (
    "62d2f2c2c9b152aeede8d4d4c203f05c24aba5697b311e534956053f6a315007"
)
COMMUNICATION_SCHEMA_VERSION = 2
COMMUNICATION_SCHEMA_SHA256 = "57ea8e7e6de787569aeaaa4e06dd419b4b6ccbbb78458eec2c8b92320d2c3e15"
COMMUNICATION_TRIGGER_SHA256 = "f86163665f9d827d83018423137e8fa38900163735ea8aef5c1587531696504c"


def _communication_ddl_sha256(
    connection,
    object_types: tuple[str, ...],
    table_names: tuple[str, ...] = _COMMUNICATION_SCHEMA_TABLES,
) -> str:
    placeholders = ",".join("?" for _ in table_names)
    type_placeholders = ",".join("?" for _ in object_types)
    rows = connection.exec_driver_sql(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        f"WHERE tbl_name IN ({placeholders}) "
        f"AND type IN ({type_placeholders}) AND sql IS NOT NULL "
        "ORDER BY type, name, tbl_name",
        (*table_names, *object_types),
    ).all()
    canonical = [
        {
            "type": str(row[0]),
            "name": str(row[1]),
            "table": str(row[2]),
            "sql": " ".join(str(row[3]).split()),
        }
        for row in rows
    ]
    payload = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _create_immutable_table_triggers(connection, table_names: tuple[str, ...]) -> None:
    for table_name in table_names:
        connection.exec_driver_sql(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table_name}_reject_update
            BEFORE UPDATE ON {table_name}
            BEGIN
                SELECT RAISE(ABORT, '{table_name} rows are immutable');
            END
            """
        )
        connection.exec_driver_sql(
            f"""
            CREATE TRIGGER IF NOT EXISTS {table_name}_reject_delete
            BEFORE DELETE ON {table_name}
            BEGIN
                SELECT RAISE(ABORT, '{table_name} rows are immutable');
            END
            """
        )


def _create_communication_section_scope_triggers(connection) -> None:
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_section_scope_set_validate
        BEFORE INSERT ON communication_artifact_section_scope_sets
        WHEN NOT EXISTS (
            SELECT 1
            FROM communication_artifacts AS artifact
            WHERE artifact.id = NEW.artifact_id
              AND artifact.artifact_version_sha256 = NEW.artifact_version_sha256
              AND NEW.metadata_known_at >= artifact.metadata_known_at
        ) OR EXISTS (
            SELECT 1 FROM communication_artifact_contents
            WHERE artifact_id = NEW.artifact_id
        ) OR EXISTS (
            SELECT 1
            FROM communication_extractions AS extraction
            JOIN communication_artifact_contents AS content
              ON content.id = extraction.artifact_content_id
            WHERE content.artifact_id = NEW.artifact_id
        ) OR EXISTS (
            SELECT 1
            FROM communication_segments AS segment
            JOIN communication_extractions AS extraction
              ON extraction.id = segment.extraction_id
            JOIN communication_artifact_contents AS content
              ON content.id = extraction.artifact_content_id
            WHERE content.artifact_id = NEW.artifact_id
        ) OR EXISTS (
            SELECT 1 FROM json_each(NEW.scopes_json) AS scope
            WHERE scope.type <> 'object'
               OR (SELECT COUNT(*) FROM json_each(scope.value)) <> 8
               OR EXISTS (
                   SELECT 1 FROM json_each(scope.value) AS field
                   WHERE field.key NOT IN (
                       'section_ordinal', 'scope_key', 'artifact_role', 'material_type',
                       'origin_type', 'provenance_tier', 'transcriber',
                       'transcriber_attribution'
                   )
               )
               OR json_type(scope.value, '$.section_ordinal') IS NOT 'integer'
               OR json_extract(scope.value, '$.section_ordinal') < 1
               OR json_type(scope.value, '$.scope_key') IS NOT 'text'
               OR length(json_extract(scope.value, '$.scope_key')) NOT BETWEEN 1 AND 96
               OR json_extract(scope.value, '$.scope_key')
                    <> trim(json_extract(scope.value, '$.scope_key'))
               OR substr(json_extract(scope.value, '$.scope_key'), 1, 1) NOT GLOB '[a-z]'
               OR json_extract(scope.value, '$.scope_key') GLOB '*[^a-z0-9_]*'
               OR json_type(scope.value, '$.artifact_role') IS NOT 'text'
               OR json_type(scope.value, '$.material_type') IS NOT 'text'
               OR json_type(scope.value, '$.origin_type') IS NOT 'text'
               OR json_type(scope.value, '$.provenance_tier') IS NOT 'text'
               OR json_type(scope.value, '$.transcriber_attribution') IS NOT 'text'
               OR json_type(scope.value, '$.transcriber') IS NULL
               OR json_type(scope.value, '$.transcriber') NOT IN ('text', 'null')
               OR (
                   json_type(scope.value, '$.transcriber') = 'text'
                   AND length(trim(json_extract(scope.value, '$.transcriber'))) = 0
               )
               OR json_extract(scope.value, '$.section_ordinal')
                    <> CAST(scope.key AS INTEGER) + 1
        ) OR (
            SELECT COUNT(DISTINCT json_extract(scope.value, '$.scope_key'))
            FROM json_each(NEW.scopes_json) AS scope
        ) <> NEW.scope_count OR (
            SELECT COUNT(DISTINCT json_extract(scope.value, '$.section_ordinal'))
            FROM json_each(NEW.scopes_json) AS scope
        ) <> NEW.scope_count OR EXISTS (
            SELECT 1 FROM json_each(NEW.scopes_json) AS scope
            WHERE NOT (
                (json_extract(scope.value, '$.scope_key') = 'prepared_remarks'
                 AND json_extract(scope.value, '$.artifact_role') = 'prepared_remarks'
                 AND json_extract(scope.value, '$.material_type') IN (
                     'financial_results', 'management_review',
                     'monetary_policy_statement', 'speech_text'
                 )) OR
                (json_extract(scope.value, '$.scope_key') = 'q_and_a'
                 AND json_extract(scope.value, '$.artifact_role') = 'q_and_a_transcript'
                 AND json_extract(scope.value, '$.material_type') = 'questions_and_answers') OR
                (json_extract(scope.value, '$.scope_key') = 'full_transcript'
                 AND json_extract(scope.value, '$.artifact_role') = 'full_transcript'
                 AND json_extract(scope.value, '$.material_type') IN (
                     'press_conference_transcript', 'results_transcript'
                 )) OR
                (json_extract(scope.value, '$.scope_key') = 'ceo_letter'
                 AND json_extract(scope.value, '$.artifact_role') = 'ceo_letter'
                 AND json_extract(scope.value, '$.material_type') = 'ceo_letter') OR
                (json_extract(scope.value, '$.scope_key') = 'chair_letter'
                 AND json_extract(scope.value, '$.artifact_role') = 'chair_letter'
                 AND json_extract(scope.value, '$.material_type') IN (
                     'annual_report', 'management_review'
                 )) OR
                (json_extract(scope.value, '$.scope_key') = 'annual_report'
                 AND json_extract(scope.value, '$.artifact_role') = 'annual_report'
                 AND json_extract(scope.value, '$.material_type') = 'annual_report') OR
                (json_extract(scope.value, '$.scope_key') = 'subtitles'
                 AND json_extract(scope.value, '$.artifact_role') = 'subtitles'
                 AND json_extract(scope.value, '$.material_type') = 'subtitles') OR
                (json_extract(scope.value, '$.scope_key') = 'webcast_video'
                 AND json_extract(scope.value, '$.artifact_role') = 'webcast_video'
                 AND json_extract(scope.value, '$.material_type') = 'press_conference_video')
            )
        ) OR EXISTS (
            SELECT 1 FROM json_each(NEW.scopes_json) AS scope
            WHERE NOT (
                (json_extract(scope.value, '$.origin_type') = 'publisher_authored'
                 AND json_extract(scope.value, '$.provenance_tier')
                     = 'official_authored_text') OR
                (json_extract(scope.value, '$.origin_type') = 'official_published_transcript'
                 AND json_extract(scope.value, '$.provenance_tier')
                     = 'official_published_transcript') OR
                (json_extract(scope.value, '$.origin_type') = 'official_published_media'
                 AND json_extract(scope.value, '$.provenance_tier')
                     = 'official_published_media') OR
                (json_extract(scope.value, '$.origin_type') = 'official_hosted_vendor'
                 AND json_extract(scope.value, '$.provenance_tier')
                     = 'official_hosted_third_party') OR
                (json_extract(scope.value, '$.origin_type') = 'official_caption'
                 AND json_extract(scope.value, '$.provenance_tier') = 'official_caption') OR
                (json_extract(scope.value, '$.origin_type') = 'automatic_caption'
                 AND json_extract(scope.value, '$.provenance_tier')
                     = 'official_hosted_automatic_caption') OR
                (json_extract(scope.value, '$.origin_type') = 'local_asr'
                 AND json_extract(scope.value, '$.provenance_tier') = 'local_derived_asr')
            )
        ) OR EXISTS (
            SELECT 1 FROM json_each(NEW.scopes_json) AS scope
            WHERE NOT (
                (json_extract(scope.value, '$.material_type') IN (
                     'annual_report', 'ceo_letter', 'financial_results',
                     'management_review', 'monetary_policy_statement', 'speech_text'
                 ) AND json_extract(scope.value, '$.origin_type') = 'publisher_authored') OR
                (json_extract(scope.value, '$.material_type') IN (
                     'questions_and_answers', 'press_conference_transcript',
                     'results_transcript'
                 ) AND json_extract(scope.value, '$.origin_type') IN (
                     'official_published_transcript', 'official_hosted_vendor'
                 )) OR
                (json_extract(scope.value, '$.material_type') = 'subtitles'
                 AND json_extract(scope.value, '$.origin_type') IN (
                     'official_caption', 'automatic_caption', 'local_asr'
                 )) OR
                (json_extract(scope.value, '$.material_type') = 'press_conference_video'
                 AND json_extract(scope.value, '$.origin_type') = 'official_published_media')
            )
        ) OR EXISTS (
            SELECT 1
            FROM json_each(NEW.scopes_json) AS scope
            JOIN communication_artifacts AS artifact ON artifact.id = NEW.artifact_id
            WHERE NOT (
                (json_extract(scope.value, '$.origin_type') IN (
                     'publisher_authored', 'official_published_media'
                 ) AND json_extract(scope.value, '$.transcriber') IS NULL
                   AND json_extract(scope.value, '$.transcriber_attribution')
                       = 'not_applicable') OR
                (json_extract(scope.value, '$.origin_type') IN (
                     'official_hosted_vendor', 'automatic_caption', 'local_asr'
                 ) AND json_extract(scope.value, '$.transcriber') IS NOT NULL
                   AND length(trim(json_extract(scope.value, '$.transcriber'))) > 0
                   AND json_extract(scope.value, '$.transcriber') <> artifact.publisher
                   AND json_extract(scope.value, '$.transcriber_attribution')
                       = 'named_third_party') OR
                (json_extract(scope.value, '$.origin_type') IN (
                     'official_published_transcript', 'official_caption'
                 ) AND (
                    (json_extract(scope.value, '$.transcriber') IS NULL
                     AND json_extract(scope.value, '$.transcriber_attribution')
                         = 'not_disclosed') OR
                    (json_extract(scope.value, '$.transcriber') = artifact.publisher
                     AND json_extract(scope.value, '$.transcriber_attribution') = 'publisher') OR
                    (json_extract(scope.value, '$.transcriber') IS NOT NULL
                     AND length(trim(json_extract(scope.value, '$.transcriber'))) > 0
                     AND json_extract(scope.value, '$.transcriber') <> artifact.publisher
                     AND json_extract(scope.value, '$.transcriber_attribution')
                         = 'named_third_party')
                 ))
            )
        ) OR EXISTS (
            SELECT 1 FROM json_each(NEW.scopes_json) AS scope
            WHERE NOT EXISTS (
                SELECT 1
                FROM communication_artifacts AS artifact
                JOIN communication_source_policy_snapshots AS policy
                  ON policy.catalogue_sha256 = artifact.catalogue_sha256
                 AND policy.source_id = artifact.source_id
                JOIN json_each(policy.material_types_json) AS material
                WHERE artifact.id = NEW.artifact_id
                  AND material.value = json_extract(scope.value, '$.material_type')
            )
        ) OR NOT EXISTS (
            SELECT 1
            FROM json_each(NEW.scopes_json) AS scope
            JOIN communication_artifacts AS artifact ON artifact.id = NEW.artifact_id
            WHERE json_extract(scope.value, '$.artifact_role') = artifact.artifact_role
              AND json_extract(scope.value, '$.material_type') = artifact.material_type
              AND json_extract(scope.value, '$.origin_type') = artifact.origin_type
              AND json_extract(scope.value, '$.provenance_tier') = artifact.provenance_tier
              AND json_extract(scope.value, '$.transcriber') IS artifact.transcriber
              AND json_extract(scope.value, '$.transcriber_attribution')
                    = artifact.transcriber_attribution
        )
        BEGIN
            SELECT RAISE(ABORT, 'invalid communication artifact section-scope set');
        END
        """
    )
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_segments_match_scope_set
        BEFORE INSERT ON communication_segments
        WHEN (
            NEW.section_ordinal IS NULL AND EXISTS (
                SELECT 1
                FROM communication_artifact_section_scope_sets AS scope_set
                JOIN communication_artifact_contents AS content
                  ON content.artifact_id = scope_set.artifact_id
                JOIN communication_extractions AS extraction
                  ON extraction.artifact_content_id = content.id
                WHERE extraction.id = NEW.extraction_id
            )
        ) OR (
            NEW.section_ordinal IS NOT NULL AND NOT EXISTS (
                SELECT 1
                FROM communication_artifact_section_scope_sets AS scope_set
                JOIN communication_artifact_contents AS content
                  ON content.artifact_id = scope_set.artifact_id
                JOIN communication_extractions AS extraction
                  ON extraction.artifact_content_id = content.id
                JOIN json_each(scope_set.scopes_json) AS scope
                WHERE extraction.id = NEW.extraction_id
                  AND json_extract(scope.value, '$.section_ordinal') = NEW.section_ordinal
                  AND (
                      NEW.segment_kind IN ('heading', 'other') OR
                      (json_extract(scope.value, '$.scope_key') = 'prepared_remarks'
                       AND NEW.segment_kind = 'prepared_remarks') OR
                      (json_extract(scope.value, '$.scope_key') = 'q_and_a'
                       AND NEW.segment_kind IN ('q_and_a_question', 'q_and_a_answer')) OR
                      (json_extract(scope.value, '$.scope_key') IN (
                          'ceo_letter', 'chair_letter'
                       ) AND NEW.segment_kind = 'letter') OR
                      (json_extract(scope.value, '$.scope_key') = 'annual_report'
                       AND NEW.segment_kind IN ('letter', 'narrative')) OR
                      (json_extract(scope.value, '$.scope_key') IN (
                          'full_transcript', 'subtitles', 'webcast_video'
                       ) AND NEW.segment_kind IN (
                          'prepared_remarks', 'q_and_a_question',
                          'q_and_a_answer', 'narrative'
                       ))
                  )
            )
        ) OR EXISTS (
            SELECT 1 FROM communication_segments AS prior
            WHERE prior.extraction_id = NEW.extraction_id
              AND prior.section_ordinal IS NOT NULL
              AND (
                  (prior.ordinal < NEW.ordinal
                   AND prior.section_ordinal > NEW.section_ordinal) OR
                  (prior.ordinal > NEW.ordinal
                   AND prior.section_ordinal < NEW.section_ordinal)
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'communication segment conflicts with artifact section scopes');
        END
        """
    )


def _create_communication_artifact_use_triggers(connection) -> None:
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_retrieval_matches_artifact
        BEFORE INSERT ON communication_artifact_retrievals
        WHEN NOT EXISTS (
            SELECT 1
            FROM communication_artifacts AS artifact
            JOIN communication_artifact_section_scope_sets AS scope_set
              ON scope_set.artifact_id = artifact.id
            WHERE artifact.id = NEW.artifact_id
              AND scope_set.artifact_version_sha256 = artifact.artifact_version_sha256
              AND artifact.landing_url = NEW.landing_url
              AND artifact.artifact_url = NEW.artifact_url
              AND NEW.retrieved_at >= artifact.available_at
              AND NEW.metadata_known_at >= scope_set.metadata_known_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'retrieval observation conflicts with artifact');
        END
        """
    )
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_content_matches_retrieval
        BEFORE INSERT ON communication_artifact_contents
        WHEN NOT EXISTS (
            SELECT 1
            FROM communication_artifacts AS artifact
            JOIN communication_artifact_retrievals AS retrieval
              ON retrieval.artifact_id = artifact.id
            JOIN communication_artifact_section_scope_sets AS scope_set
              ON scope_set.artifact_id = artifact.id
            WHERE artifact.id = NEW.artifact_id
              AND retrieval.id = NEW.retrieval_id
              AND artifact.rights_status IN ('cleared', 'internal_only')
              AND NEW.captured_at >= retrieval.retrieved_at
              AND NEW.captured_at >= retrieval.metadata_known_at
              AND NEW.captured_at >= scope_set.metadata_known_at
        )
        BEGIN
            SELECT RAISE(ABORT, 'content capture conflicts with retrieval or rights policy');
        END
        """
    )


def _create_communication_segment_finalization_guard(connection) -> None:
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_segments_reject_after_finalization
        BEFORE INSERT ON communication_segments
        WHEN EXISTS (
            SELECT 1 FROM communication_extraction_finalizations
            WHERE extraction_id = NEW.extraction_id
        )
        BEGIN
            SELECT RAISE(ABORT, 'finalized communication extraction cannot gain segments');
        END
        """
    )


def _create_communication_finalization_trigger(connection) -> None:
    connection.exec_driver_sql(
        """
        CREATE TRIGGER IF NOT EXISTS communication_extraction_finalization_validate
        BEFORE INSERT ON communication_extraction_finalizations
        WHEN NOT EXISTS (
            SELECT 1
            FROM communication_extractions AS extraction
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
              AND NEW.canonicalization_version = 'communication_segments_json_v2'
              AND EXISTS (
                      SELECT 1
                      FROM communication_artifact_section_scope_sets AS scope_set
                      JOIN communication_artifact_contents AS content
                        ON content.artifact_id = scope_set.artifact_id
                      WHERE content.id = extraction.artifact_content_id
                        AND scope_set.scope_count = (
                            SELECT COUNT(DISTINCT section_ordinal)
                            FROM communication_segments
                            WHERE extraction_id = NEW.extraction_id
                              AND section_ordinal IS NOT NULL
                        )
                        AND 1 = (
                            SELECT MIN(section_ordinal)
                            FROM communication_segments
                            WHERE extraction_id = NEW.extraction_id
                        )
                        AND scope_set.scope_count = (
                            SELECT MAX(section_ordinal)
                            FROM communication_segments
                            WHERE extraction_id = NEW.extraction_id
                        )
                        AND NOT EXISTS (
                            SELECT 1
                            FROM communication_segments AS left_segment
                            JOIN communication_segments AS right_segment
                              ON right_segment.extraction_id = left_segment.extraction_id
                             AND right_segment.ordinal > left_segment.ordinal
                            WHERE left_segment.extraction_id = NEW.extraction_id
                              AND left_segment.section_ordinal > right_segment.section_ordinal
                        )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM communication_artifact_section_scope_sets AS scope_set
                  JOIN communication_artifact_contents AS content
                    ON content.artifact_id = scope_set.artifact_id
                  JOIN json_each(scope_set.scopes_json) AS scope
                  WHERE content.id = extraction.artifact_content_id
                    AND NOT EXISTS (
                        SELECT 1 FROM communication_segments AS segment
                        WHERE segment.extraction_id = NEW.extraction_id
                          AND segment.section_ordinal = json_extract(
                              scope.value, '$.section_ordinal'
                          )
                          AND segment.segment_kind NOT IN ('heading', 'other')
                    )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM communication_artifact_section_scope_sets AS scope_set
                  JOIN communication_artifact_contents AS content
                    ON content.artifact_id = scope_set.artifact_id
                  JOIN json_each(scope_set.scopes_json) AS scope
                  WHERE content.id = extraction.artifact_content_id
                    AND json_extract(scope.value, '$.scope_key') = 'prepared_remarks'
                    AND NOT EXISTS (
                        SELECT 1 FROM communication_segments AS segment
                        WHERE segment.extraction_id = NEW.extraction_id
                          AND segment.section_ordinal = json_extract(
                              scope.value, '$.section_ordinal'
                          )
                          AND segment.segment_kind = 'prepared_remarks'
                          AND segment.speaker_side = 'publisher'
                    )
              )
              AND NOT EXISTS (
                  SELECT 1
                  FROM communication_artifact_section_scope_sets AS scope_set
                  JOIN communication_artifact_contents AS content
                    ON content.artifact_id = scope_set.artifact_id
                  JOIN json_each(scope_set.scopes_json) AS scope
                  WHERE content.id = extraction.artifact_content_id
                    AND json_extract(scope.value, '$.scope_key') = 'q_and_a'
                    AND (
                        NOT EXISTS (
                            SELECT 1 FROM communication_segments AS segment
                            WHERE segment.extraction_id = NEW.extraction_id
                              AND segment.section_ordinal = json_extract(
                                  scope.value, '$.section_ordinal'
                              )
                              AND segment.segment_kind = 'q_and_a_question'
                        ) OR NOT EXISTS (
                            SELECT 1 FROM communication_segments AS segment
                            WHERE segment.extraction_id = NEW.extraction_id
                              AND segment.section_ordinal = json_extract(
                                  scope.value, '$.section_ordinal'
                              )
                              AND segment.segment_kind = 'q_and_a_answer'
                        )
                    )
              )
        )
        BEGIN
            SELECT RAISE(ABORT, 'communication extraction finalization does not match segments');
        END
        """
    )


_COMMUNICATION_SCHEMA_V2_MIGRATION_BACKUP_SUFFIX = (
    ".before-communication-schema-v2.sqlite3"
)


def communication_schema_v2_migration_backup_path(engine: Engine) -> Path:
    """Return the deterministic adjacent backup used by the v1-to-v2 migration."""

    if engine.dialect.name != "sqlite":
        raise RuntimeError("communication schema migration only supports SQLite")
    database = engine.url.database
    if not database or database == ":memory:":
        raise RuntimeError("communication schema migration needs a file-backed SQLite database")
    source = Path(database).expanduser().resolve()
    return source.with_name(f"{source.name}{_COMMUNICATION_SCHEMA_V2_MIGRATION_BACKUP_SUFFIX}")


def _communication_schema_generation(connection) -> str:
    table_names = {
        str(row[0])
        for row in connection.exec_driver_sql(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }
    if any(
        table_name.startswith("communication_")
        and table_name not in _COMMUNICATION_SCHEMA_TABLES
        for table_name in table_names
    ):
        return "unknown"
    relevant = table_names.intersection(_COMMUNICATION_SCHEMA_TABLES)
    if not relevant:
        return "absent"
    if relevant == set(_COMMUNICATION_SCHEMA_TABLES):
        segment_columns = {
            str(row[1])
            for row in connection.exec_driver_sql(
                "PRAGMA table_info('communication_segments')"
            )
        }
        return "v2" if "section_ordinal" in segment_columns else "unknown"
    if relevant == set(_COMMUNICATION_SCHEMA_V1_TABLES):
        segment_columns = {
            str(row[1])
            for row in connection.exec_driver_sql(
                "PRAGMA table_info('communication_segments')"
            )
        }
        return "v1" if "section_ordinal" not in segment_columns else "unknown"
    return "unknown"


def _verify_communication_v1_contract(connection) -> None:
    for table_name in _COMMUNICATION_SCHEMA_V1_TABLES:
        expected = {column.name for column in Base.metadata.tables[table_name].columns}
        if table_name == "communication_segments":
            expected.remove("section_ordinal")
        actual = {
            str(row[1])
            for row in connection.exec_driver_sql(f"PRAGMA table_info('{table_name}')")
        }
        if actual != expected:
            raise RuntimeError(
                "unknown institutional-communications v1 schema for "
                f"{table_name}: missing={sorted(expected - actual)!r}, "
                f"unexpected={sorted(actual - expected)!r}"
            )
    rows = connection.exec_driver_sql(
        "SELECT schema_version, schema_sha256, trigger_sha256 "
        "FROM communication_schema_contract "
        "WHERE contract_id = 'institutional_communications'"
    ).all()
    if len(rows) != 1:
        raise RuntimeError("institutional-communications v1 schema contract is missing")
    version, stored_schema_sha256, stored_trigger_sha256 = rows[0]
    actual_schema_sha256 = _communication_ddl_sha256(
        connection,
        ("table", "index"),
        _COMMUNICATION_SCHEMA_V1_TABLES,
    )
    actual_trigger_sha256 = _communication_ddl_sha256(
        connection,
        ("trigger",),
        _COMMUNICATION_SCHEMA_V1_TABLES,
    )
    if (
        int(version) != _COMMUNICATION_SCHEMA_V1_VERSION
        or str(stored_schema_sha256) != _COMMUNICATION_SCHEMA_V1_SHA256
        or actual_schema_sha256 != _COMMUNICATION_SCHEMA_V1_SHA256
        or str(stored_trigger_sha256) != _COMMUNICATION_TRIGGER_V1_SHA256
        or actual_trigger_sha256 != _COMMUNICATION_TRIGGER_V1_SHA256
    ):
        raise RuntimeError("institutional-communications v1 schema fingerprint mismatch")


def _migrate_communication_schema_v2(engine: Engine) -> None:
    """Upgrade an empty, exactly fingerprinted v1 ledger after a verified backup.

    V1 has no semantic section binding.  Automatically interpreting existing
    artifacts, captures, or segments would be unsafe, so only a structurally
    exact ledger with no non-contract communication rows is eligible.  Any
    populated v1 ledger fails closed for an explicit, evidence-aware migration.
    """

    if engine.dialect.name != "sqlite":
        raise RuntimeError("communication schema migration only supports SQLite")
    with engine.connect() as connection:
        generation = _communication_schema_generation(connection)
    if generation in {"absent", "v2", "unknown"}:
        return

    with engine.connect() as connection:
        connection.exec_driver_sql("BEGIN IMMEDIATE")
        try:
            generation = _communication_schema_generation(connection)
            if generation == "v2":
                connection.rollback()
                return
            if generation != "v1":
                raise RuntimeError(
                    "institutional-communications schema changed while awaiting migration lock"
                )
            _verify_communication_v1_contract(connection)
            populated = {
                table_name: int(
                    connection.exec_driver_sql(
                        f"SELECT COUNT(*) FROM {_quote_sqlite_identifier(table_name)}"
                    ).scalar_one()
                )
                for table_name in _COMMUNICATION_SCHEMA_V1_TABLES
                if table_name != "communication_schema_contract"
            }
            populated = {name: count for name, count in populated.items() if count}
            if populated:
                raise RuntimeError(
                    "populated institutional-communications v1 schema requires an explicit "
                    f"semantic migration; rows={populated!r}"
                )

            database_path = Path(
                next(
                    row[2]
                    for row in connection.exec_driver_sql("PRAGMA database_list")
                    if row[1] == "main"
                )
            ).resolve()
            expected_path = Path(engine.url.database or "").expanduser().resolve()
            if database_path != expected_path:
                raise RuntimeError("SQLite engine path does not match its main database path")
            create_verified_sqlite_backup(
                database_path,
                communication_schema_v2_migration_backup_path(engine),
            )

            for trigger_name in (
                "communication_schema_contract_reject_update",
                "communication_schema_contract_reject_delete",
                "communication_segments_reject_update",
                "communication_segments_reject_delete",
                "communication_segments_reject_after_finalization",
                "communication_segments_match_scope_set",
                "communication_retrieval_matches_artifact",
                "communication_content_matches_retrieval",
                "communication_extraction_finalizations_reject_update",
                "communication_extraction_finalizations_reject_delete",
                "communication_extraction_finalization_validate",
            ):
                connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger_name}")
            connection.exec_driver_sql("DROP TABLE communication_extraction_finalizations")
            connection.exec_driver_sql("DROP TABLE communication_segments")
            connection.exec_driver_sql(
                "ALTER TABLE communication_schema_contract "
                "RENAME TO communication_schema_contract__v1"
            )
            CommunicationSchemaContract.__table__.create(connection)
            CommunicationSegment.__table__.create(connection)
            CommunicationExtractionFinalization.__table__.create(connection)
            CommunicationArtifactSectionScopeSet.__table__.create(connection)
            connection.exec_driver_sql("DROP TABLE communication_schema_contract__v1")

            _create_immutable_table_triggers(
                connection,
                (
                    "communication_schema_contract",
                    "communication_artifact_section_scope_sets",
                    "communication_segments",
                    "communication_extraction_finalizations",
                ),
            )
            _create_communication_section_scope_triggers(connection)
            _create_communication_artifact_use_triggers(connection)
            _create_communication_segment_finalization_guard(connection)
            _create_communication_finalization_trigger(connection)

            schema_sha256 = _communication_ddl_sha256(connection, ("table", "index"))
            trigger_sha256 = _communication_ddl_sha256(connection, ("trigger",))
            if schema_sha256 != COMMUNICATION_SCHEMA_SHA256:
                raise RuntimeError(
                    "migrated institutional-communications schema fingerprint mismatch"
                )
            if trigger_sha256 != COMMUNICATION_TRIGGER_SHA256:
                raise RuntimeError(
                    "migrated institutional-communications trigger fingerprint mismatch"
                )
            connection.exec_driver_sql(
                "INSERT INTO communication_schema_contract "
                "(contract_id, schema_version, schema_sha256, trigger_sha256, installed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "institutional_communications",
                    COMMUNICATION_SCHEMA_VERSION,
                    schema_sha256,
                    trigger_sha256,
                    datetime.now(UTC)
                    .replace(tzinfo=None)
                    .isoformat(sep=" ", timespec="microseconds"),
                ),
            )
            violations = connection.exec_driver_sql("PRAGMA foreign_key_check").all()
            if violations:
                raise RuntimeError(
                    "communication schema migration would break foreign keys: "
                    f"{violations[:3]!r}"
                )
            integrity = connection.exec_driver_sql("PRAGMA integrity_check").all()
            if integrity != [("ok",)]:
                raise RuntimeError(
                    "communication schema migration failed SQLite integrity_check: "
                    f"{integrity[:3]!r}"
                )
            connection.commit()
        except Exception:
            connection.rollback()
            raise


def _preflight_communication_schema(engine: Engine) -> None:
    """Refuse an unknown pre-existing communications table layout.

    ``create_all`` is additive and would otherwise silently leave an older,
    weaker table definition in place.  A first install has no domain tables. An
    existing install must contain the complete, fingerprinted schema contract.
    """

    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    unexpected_communication = sorted(
        table_name
        for table_name in present
        if table_name.startswith("communication_")
        and table_name not in _COMMUNICATION_SCHEMA_TABLES
    )
    if unexpected_communication:
        raise RuntimeError(
            "unknown institutional-communications tables; "
            f"unexpected={unexpected_communication!r}"
        )
    present_communication = present.intersection(_COMMUNICATION_SCHEMA_TABLES)
    if not present_communication:
        return
    if present_communication != set(_COMMUNICATION_SCHEMA_TABLES):
        missing_tables = sorted(set(_COMMUNICATION_SCHEMA_TABLES) - present)
        raise RuntimeError(
            f"incomplete institutional-communications schema; missing tables={missing_tables!r}"
        )
    for table_name in _COMMUNICATION_SCHEMA_TABLES:
        expected = {column.name for column in Base.metadata.tables[table_name].columns}
        actual = {str(column["name"]) for column in inspector.get_columns(table_name)}
        if actual != expected:
            missing = sorted(expected - actual)
            unexpected = sorted(actual - expected)
            raise RuntimeError(
                "unknown institutional-communications schema for "
                f"{table_name}: missing={missing!r}, unexpected={unexpected!r}"
            )
    with engine.connect() as connection:
        rows = connection.exec_driver_sql(
            "SELECT schema_version, schema_sha256, trigger_sha256 "
            "FROM communication_schema_contract "
            "WHERE contract_id = 'institutional_communications'"
        ).all()
        if len(rows) != 1:
            raise RuntimeError("institutional-communications schema contract is missing")
        version, stored_schema_sha256, stored_trigger_sha256 = rows[0]
        actual_schema_sha256 = _communication_ddl_sha256(connection, ("table", "index"))
        if (
            int(version) != COMMUNICATION_SCHEMA_VERSION
            or str(stored_schema_sha256) != COMMUNICATION_SCHEMA_SHA256
            or actual_schema_sha256 != COMMUNICATION_SCHEMA_SHA256
            or str(stored_trigger_sha256) != COMMUNICATION_TRIGGER_SHA256
        ):
            raise RuntimeError("institutional-communications schema fingerprint mismatch")


def init_db(engine: Engine) -> None:
    _migrate_communication_schema_v2(engine)
    _preflight_communication_schema(engine)
    _migrate_release_recurrence_constraint(engine)
    Base.metadata.create_all(engine)
    immutable_tables = (
        "data_releases",
        "data_release_artifacts",
        "release_observations",
        "debt_holder_positions",
        "cross_border_positions",
        "allocator_facts",
        "communication_schema_contract",
        "organizations",
        "communication_source_policy_snapshots",
        "organization_commodity_coverage",
        "communication_events",
        "communication_artifacts",
        "communication_artifact_section_scope_sets",
        "communication_artifact_retrievals",
        "communication_artifact_contents",
        "communication_extractions",
        "communication_segments",
        "communication_extraction_finalizations",
        "report_documents",
        "document_extractions",
        "document_pages",
        "claims",
        "claim_citations",
        "report_candidate_reviews",
    )
    with engine.begin() as connection:
        for trigger_name in _COMMUNICATION_CUSTOM_TRIGGERS:
            connection.exec_driver_sql(f"DROP TRIGGER IF EXISTS {trigger_name}")
        _create_immutable_table_triggers(connection, immutable_tables)
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_policy_snapshot_validate
            BEFORE INSERT ON communication_source_policy_snapshots
            WHEN EXISTS (
                SELECT 1 FROM json_each(NEW.official_domains_json)
                WHERE type <> 'text' OR value <> lower(value) OR instr(value, '.') = 0
                   OR value LIKE '.%' OR value LIKE '%.' OR instr(value, '..') > 0
                   OR instr(value, '/') > 0 OR instr(value, ':') > 0
                   OR instr(value, '@') > 0 OR instr(value, '\\') > 0
                   OR instr(value, ' ') > 0
            ) OR EXISTS (
                SELECT 1 FROM json_each(NEW.material_types_json)
                WHERE type <> 'text' OR value NOT IN (
                    'annual_report', 'ceo_letter', 'financial_results',
                    'management_review', 'monetary_policy_statement',
                    'press_conference_transcript', 'press_conference_video',
                    'questions_and_answers', 'results_transcript', 'speech_text',
                    'subtitles'
                )
            ) OR EXISTS (
                SELECT 1 FROM json_each(NEW.commodity_families_json)
                WHERE type <> 'text' OR value NOT IN (
                    'agricultural_raw_materials', 'base_metals', 'energy',
                    'fertilizers', 'food_and_beverages', 'precious_metals'
                )
            ) OR (
                NEW.organization_type = 'commodity_company'
                AND json_array_length(NEW.commodity_families_json) = 0
            ) OR (
                NEW.organization_type <> 'commodity_company'
                AND json_array_length(NEW.commodity_families_json) <> 0
            ) OR NOT EXISTS (
                SELECT 1 FROM json_each(NEW.official_domains_json) AS domain
                WHERE lower(substr(
                    NEW.landing_url,
                    9,
                    instr(substr(NEW.landing_url, 9), '/') - 1
                )) = domain.value
                   OR lower(substr(
                       NEW.landing_url,
                       9,
                       instr(substr(NEW.landing_url, 9), '/') - 1
                   )) LIKE '%.' || domain.value
            ) OR (
                NEW.rights_basis_url IS NOT NULL AND NOT EXISTS (
                    SELECT 1 FROM json_each(NEW.official_domains_json) AS domain
                    WHERE lower(substr(
                        NEW.rights_basis_url,
                        9,
                        instr(substr(NEW.rights_basis_url, 9), '/') - 1
                    )) = domain.value
                       OR lower(substr(
                           NEW.rights_basis_url,
                           9,
                           instr(substr(NEW.rights_basis_url, 9), '/') - 1
                       )) LIKE '%.' || domain.value
                )
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid communication source policy snapshot');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_artifacts_match_policy
            BEFORE INSERT ON communication_artifacts
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_source_policy_snapshots AS policy
                JOIN communication_events AS event ON event.id = NEW.event_id
                WHERE policy.catalogue_sha256 = NEW.catalogue_sha256
                  AND policy.source_id = NEW.source_id
                  AND policy.organization_id = event.organization_id
                  AND policy.rights_status = NEW.rights_status
                  AND policy.acquisition_status = NEW.acquisition_status
                  AND policy.rights_basis_url IS NEW.rights_basis_url
                  AND policy.rights_note = NEW.rights_note
                  AND policy.language = NEW.language
                  AND event.event_version_sha256 = NEW.event_version_sha256
                  AND EXISTS (
                      SELECT 1 FROM json_each(policy.material_types_json)
                      WHERE value = NEW.material_type
                  )
                  AND EXISTS (
                      SELECT 1 FROM json_each(policy.official_domains_json) AS domain
                      WHERE lower(substr(
                          NEW.landing_url,
                          9,
                          instr(substr(NEW.landing_url, 9), '/') - 1
                      )) = domain.value
                         OR lower(substr(
                             NEW.landing_url,
                             9,
                             instr(substr(NEW.landing_url, 9), '/') - 1
                         )) LIKE '%.' || domain.value
                  )
                  AND EXISTS (
                      SELECT 1 FROM json_each(policy.official_domains_json) AS domain
                      WHERE lower(substr(
                          NEW.artifact_url,
                          9,
                          instr(substr(NEW.artifact_url, 9), '/') - 1
                      )) = domain.value
                         OR lower(substr(
                             NEW.artifact_url,
                             9,
                             instr(substr(NEW.artifact_url, 9), '/') - 1
                         )) LIKE '%.' || domain.value
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'communication artifact conflicts with source policy or event');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_coverage_matches_policy
            BEFORE INSERT ON organization_commodity_coverage
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_source_policy_snapshots AS policy
                WHERE policy.catalogue_sha256 = NEW.catalogue_sha256
                  AND policy.source_id = NEW.source_id
                  AND policy.organization_id = NEW.organization_id
                  AND policy.organization_type = 'commodity_company'
                  AND policy.coverage_note = NEW.evidence_note
                  AND EXISTS (
                      SELECT 1 FROM json_each(policy.commodity_families_json)
                      WHERE value = NEW.commodity_family
                  )
                  AND EXISTS (
                      SELECT 1 FROM json_each(policy.official_domains_json) AS domain
                      WHERE lower(substr(
                          NEW.evidence_url,
                          9,
                          instr(substr(NEW.evidence_url, 9), '/') - 1
                      )) = domain.value
                         OR lower(substr(
                             NEW.evidence_url,
                             9,
                             instr(substr(NEW.evidence_url, 9), '/') - 1
                         )) LIKE '%.' || domain.value
                  )
            )
            BEGIN
                SELECT RAISE(ABORT, 'commodity coverage conflicts with source policy');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_event_successor_order
            BEFORE INSERT ON communication_events
            WHEN NEW.supersedes_event_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM communication_events AS prior
                WHERE prior.id = NEW.supersedes_event_id
                  AND prior.organization_id = NEW.organization_id
                  AND prior.event_key = NEW.event_key
                  AND NEW.metadata_known_at > prior.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid or backward communication event successor');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_artifact_reject_duplicate_root
            BEFORE INSERT ON communication_artifacts
            WHEN NEW.supersedes_artifact_id IS NULL AND EXISTS (
                SELECT 1
                FROM communication_artifacts AS prior
                JOIN communication_events AS prior_event ON prior_event.id = prior.event_id
                JOIN communication_events AS new_event ON new_event.id = NEW.event_id
                WHERE prior.supersedes_artifact_id IS NULL
                  AND prior_event.organization_id = new_event.organization_id
                  AND prior_event.event_key = new_event.event_key
                  AND prior.source_id = NEW.source_id
                  AND prior.artifact_key = NEW.artifact_key
            )
            BEGIN
                SELECT RAISE(ABORT, 'duplicate communication artifact lineage root');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_artifact_successor_order
            BEFORE INSERT ON communication_artifacts
            WHEN NEW.supersedes_artifact_id IS NOT NULL AND NOT EXISTS (
                SELECT 1
                FROM communication_artifacts AS prior
                JOIN communication_events AS prior_event ON prior_event.id = prior.event_id
                JOIN communication_events AS new_event ON new_event.id = NEW.event_id
                WHERE prior.id = NEW.supersedes_artifact_id
                  AND prior_event.organization_id = new_event.organization_id
                  AND prior_event.event_key = new_event.event_key
                  AND prior.source_id = NEW.source_id
                  AND prior.artifact_key = NEW.artifact_key
                  AND NEW.metadata_known_at > prior.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid or backward communication artifact successor');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_coverage_successor_order
            BEFORE INSERT ON organization_commodity_coverage
            WHEN NEW.supersedes_exposure_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM organization_commodity_coverage AS prior
                WHERE prior.id = NEW.supersedes_exposure_id
                  AND prior.organization_id = NEW.organization_id
                  AND prior.coverage_key = NEW.coverage_key
                  AND NOT EXISTS (
                      SELECT 1 FROM organization_commodity_coverage AS successor
                      WHERE successor.supersedes_exposure_id = prior.id
                  )
                  AND NEW.metadata_known_at > prior.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid or backward commodity coverage successor');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_coverage_reject_overlapping_head
            BEFORE INSERT ON organization_commodity_coverage
            WHEN EXISTS (
                SELECT 1
                FROM organization_commodity_coverage AS prior
                WHERE prior.organization_id = NEW.organization_id
                  AND prior.commodity_family = NEW.commodity_family
                  AND prior.exposure_role = NEW.exposure_role
                  AND prior.id IS NOT NEW.supersedes_exposure_id
                  AND NOT EXISTS (
                      SELECT 1 FROM organization_commodity_coverage AS successor
                      WHERE successor.supersedes_exposure_id = prior.id
                  )
                  AND prior.effective_from <= COALESCE(NEW.effective_to, '9999-12-31')
                  AND NEW.effective_from <= COALESCE(prior.effective_to, '9999-12-31')
            )
            BEGIN
                SELECT RAISE(ABORT, 'overlapping commodity coverage lineage heads');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_retrieval_matches_artifact
            BEFORE INSERT ON communication_artifact_retrievals
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_artifacts AS artifact
                JOIN communication_artifact_section_scope_sets AS scope_set
                  ON scope_set.artifact_id = artifact.id
                WHERE artifact.id = NEW.artifact_id
                  AND scope_set.artifact_version_sha256 = artifact.artifact_version_sha256
                  AND artifact.landing_url = NEW.landing_url
                  AND artifact.artifact_url = NEW.artifact_url
                  AND NEW.retrieved_at >= artifact.available_at
                  AND NEW.metadata_known_at >= scope_set.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'retrieval observation conflicts with artifact');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_content_matches_retrieval
            BEFORE INSERT ON communication_artifact_contents
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_artifacts AS artifact
                JOIN communication_artifact_retrievals AS retrieval
                  ON retrieval.artifact_id = artifact.id
                JOIN communication_artifact_section_scope_sets AS scope_set
                  ON scope_set.artifact_id = artifact.id
                WHERE artifact.id = NEW.artifact_id
                  AND retrieval.id = NEW.retrieval_id
                  AND artifact.rights_status IN ('cleared', 'internal_only')
                  AND NEW.captured_at >= retrieval.retrieved_at
                  AND NEW.captured_at >= retrieval.metadata_known_at
                  AND NEW.captured_at >= scope_set.metadata_known_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'content capture conflicts with retrieval or rights policy');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_content_successor_order
            BEFORE INSERT ON communication_artifact_contents
            WHEN NEW.supersedes_content_id IS NOT NULL AND NOT EXISTS (
                SELECT 1 FROM communication_artifact_contents AS prior
                WHERE prior.id = NEW.supersedes_content_id
                  AND prior.artifact_id = NEW.artifact_id
                  AND NOT EXISTS (
                      SELECT 1 FROM communication_artifact_contents AS successor
                      WHERE successor.supersedes_content_id = prior.id
                  )
                  AND NEW.captured_at > prior.captured_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'invalid or backward content-capture successor');
            END
            """
        )
        connection.exec_driver_sql(
            """
            CREATE TRIGGER IF NOT EXISTS communication_extractions_require_content
            BEFORE INSERT ON communication_extractions
            WHEN NOT EXISTS (
                SELECT 1
                FROM communication_artifact_contents
                WHERE id = NEW.artifact_content_id
                  AND NEW.extracted_at >= captured_at
            )
            BEGIN
                SELECT RAISE(ABORT, 'communication extraction requires archived content');
            END
            """
        )
        _create_communication_section_scope_triggers(connection)
        _create_communication_segment_finalization_guard(connection)
        _create_communication_finalization_trigger(connection)
        schema_sha256 = _communication_ddl_sha256(connection, ("table", "index"))
        trigger_sha256 = _communication_ddl_sha256(connection, ("trigger",))
        if schema_sha256 != COMMUNICATION_SCHEMA_SHA256:
            raise RuntimeError("generated institutional-communications schema fingerprint mismatch")
        if trigger_sha256 != COMMUNICATION_TRIGGER_SHA256:
            raise RuntimeError(
                "generated institutional-communications trigger fingerprint mismatch"
            )
        existing_contract = connection.exec_driver_sql(
            "SELECT COUNT(*) FROM communication_schema_contract "
            "WHERE contract_id = 'institutional_communications'"
        ).scalar_one()
        if existing_contract == 0:
            connection.exec_driver_sql(
                "INSERT INTO communication_schema_contract "
                "(contract_id, schema_version, schema_sha256, trigger_sha256, installed_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    "institutional_communications",
                    COMMUNICATION_SCHEMA_VERSION,
                    schema_sha256,
                    trigger_sha256,
                    datetime.now(UTC)
                    .replace(tzinfo=None)
                    .isoformat(sep=" ", timespec="microseconds"),
                ),
            )


def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
