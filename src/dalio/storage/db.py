"""SQLite storage for current macro data and immutable evidence ledgers.

``observations`` remains the backwards-compatible latest-value projection used by
the existing classifiers and UI. ``data_releases`` + ``release_observations`` are
the append-only point-in-time ledger: one complete source-partition snapshot per
release, so a historical query can reproduce both revisions and omitted rows.
Institutional prose stays separate in immutable report, extraction, page, claim,
and citation tables protected by foreign keys and append-only database triggers.
"""

from __future__ import annotations

import os
import sqlite3
import tempfile
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    DateTime,
    Engine,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
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
    series_id = Column(String(64), nullable=False)
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
            temporary_path = None
        _verify_exact_sqlite_backup(source_path, backup_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


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


def init_db(engine: Engine) -> None:
    _migrate_release_recurrence_constraint(engine)
    Base.metadata.create_all(engine)
    immutable_tables = (
        "data_releases",
        "data_release_artifacts",
        "release_observations",
        "debt_holder_positions",
        "cross_border_positions",
        "allocator_facts",
        "report_documents",
        "document_extractions",
        "document_pages",
        "claims",
        "claim_citations",
        "report_candidate_reviews",
    )
    with engine.begin() as connection:
        for table_name in immutable_tables:
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


def make_session_factory(engine: Engine):
    return sessionmaker(bind=engine, expire_on_commit=False)
