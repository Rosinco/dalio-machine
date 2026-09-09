"""Read-only, JSON-friendly inventory of the observatory's stored evidence.

The inventory describes what can actually be analysed from a supplied database.
It never initialises or updates the schema.  In particular, an absent partition
is represented as a gap rather than an economic zero.  For voluntary QPSD
submissions, ``not_reported`` is used only when an official request URL retained
in the release ledger proves that the country/series was queried successfully;
all other absent cells remain ``missing``.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import ipaddress
import json
import math
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from sqlalchemy import Engine, MetaData, Table, create_engine, func, inspect, select
from sqlalchemy.engine import Connection

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    SOURCE_BIS_GLI,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.imf_bop import BOP_COUNTRIES, BOP_SERIES, BOP_SOURCE
from dalio.data_sources.imf_positions import (
    DIP_FREQUENCIES,
    DIP_SERIES,
    IMF_POSITION_COUNTRIES,
    PIP_FREQUENCIES,
    PIP_SERIES,
)
from dalio.data_sources.money_liquidity import (
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    MONEY_LIQUIDITY_SERIES,
    MoneyLiquiditySeries,
    money_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    OFR_SHADOW_LIQUIDITY_SERIES,
    SOURCE_OFR_STFM,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.data_sources.worldbank_commodities import (
    CATALOGUE_GENERATOR_VERSION,
    CATALOGUE_SCHEMA_VERSION,
    EXPECTED_PINK_SHEET_HISTORY_START,
    EXPECTED_PINK_SHEET_INDEX_SERIES_COUNT,
    EXPECTED_PINK_SHEET_PRICE_SERIES_COUNT,
    EXPECTED_PINK_SHEET_SERIES_IDS_BY_WORKSHEET,
    MAX_PINK_SHEET_LATEST_LAG_MONTHS,
    MIN_PINK_SHEET_HISTORY_MONTHS,
    MIN_PINK_SHEET_OBSERVATIONS_PER_SERIES,
    MONTHLY_INDICES_SHEET_NAME,
    MONTHLY_SHEET_NAME,
    PINK_SHEET_CATALOGUE_VINTAGE_TAG,
    PINK_SHEET_VINTAGE_PREFIX,
    SOURCE_WORLD_BANK_COMMODITIES,
    WORLD_CODE,
)
from dalio.data_sources.worldbank_qpsd import QPSD_COUNTRIES, QPSD_SERIES, QPSD_SOURCE
from dalio.storage.releases import make_partition_key

_EXPECTED_TABLES = (
    "allocator_facts",
    "claim_citations",
    "claims",
    "communication_schema_contract",
    "communication_artifacts",
    "communication_artifact_retrievals",
    "communication_artifact_contents",
    "communication_events",
    "communication_extractions",
    "communication_extraction_finalizations",
    "communication_segments",
    "communication_source_policy_snapshots",
    "cross_border_positions",
    "data_release_artifacts",
    "data_releases",
    "debt_holder_positions",
    "document_extractions",
    "document_pages",
    "observations",
    "organization_commodity_coverage",
    "organizations",
    "release_observations",
    "report_candidate_reviews",
    "report_documents",
)

_COMMUNICATION_TABLE_NAMES = (
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
_COMMUNICATION_REQUIRED_COLUMNS = {
    "communication_schema_contract": frozenset(
        {
            "contract_id",
            "schema_version",
            "schema_sha256",
            "trigger_sha256",
            "installed_at",
        }
    ),
    "organizations": frozenset({"organization_id"}),
    "communication_source_policy_snapshots": frozenset(
        {
            "catalogue_sha256",
            "source_id",
            "organization_id",
            "organization_name",
            "organization_type",
            "jurisdiction",
            "language",
            "landing_url",
            "official_domains_json",
            "host_organization",
            "publisher",
            "transcriber",
            "transcriber_attribution",
            "material_types_json",
            "commodity_families_json",
            "verified_archive_start_year",
            "coverage_note",
            "source_provenance_tier",
            "rights_status",
            "rights_basis_url",
            "rights_note",
            "acquisition_status",
            "acquisition_note",
            "automated_collection_allowed",
            "rights_checked_by",
            "rights_checked_at",
            "catalogue_evaluated_at",
            "policy_sha256",
        }
    ),
    "organization_commodity_coverage": frozenset(
        {
            "id",
            "organization_id",
            "coverage_key",
            "commodity_family",
            "exposure_role",
            "source_id",
            "catalogue_sha256",
            "mapping_status",
            "effective_from",
            "effective_to",
            "evidence_url",
            "evidence_note",
            "published_at",
            "available_at",
            "retrieved_at",
            "metadata_known_at",
            "coverage_version_sha256",
            "supersedes_exposure_id",
        }
    ),
    "communication_events": frozenset(
        {
            "id",
            "organization_id",
            "event_key",
            "event_type",
            "title",
            "event_date",
            "event_started_at",
            "reference_start",
            "reference_end",
            "metadata_known_at",
            "event_version_sha256",
            "supersedes_event_id",
        }
    ),
    "communication_artifacts": frozenset(
        {
            "id",
            "event_id",
            "source_id",
            "catalogue_sha256",
            "event_version_sha256",
            "artifact_key",
            "artifact_role",
            "material_type",
            "language",
            "translation_status",
            "mime_type",
            "origin_type",
            "provenance_tier",
            "rights_status",
            "rights_basis_url",
            "rights_note",
            "acquisition_status",
            "rights_checked_by",
            "rights_checked_at",
            "host_organization",
            "publisher",
            "transcriber",
            "transcriber_attribution",
            "published_at",
            "available_at",
            "retrieved_at",
            "metadata_known_at",
            "landing_url",
            "artifact_url",
            "artifact_version_sha256",
            "supersedes_artifact_id",
        }
    ),
    "communication_artifact_retrievals": frozenset(
        {
            "id",
            "artifact_id",
            "retrieved_at",
            "metadata_known_at",
            "landing_url",
            "artifact_url",
        }
    ),
    "communication_artifact_contents": frozenset(
        {
            "id",
            "artifact_id",
            "retrieval_id",
            "content_sha256",
            "size_bytes",
            "blob_path",
            "captured_at",
            "supersedes_content_id",
        }
    ),
    "communication_extractions": frozenset(
        {"id", "artifact_content_id", "run_key", "extracted_at"}
    ),
    "communication_segments": frozenset(
        {
            "extraction_id",
            "ordinal",
            "segment_kind",
            "speaker_name",
            "speaker_role",
            "speaker_side",
            "section_title",
            "text",
            "text_sha256",
            "char_count",
            "page_start",
            "page_end",
            "paragraph_start",
            "paragraph_end",
            "start_ms",
            "end_ms",
        }
    ),
    "communication_extraction_finalizations": frozenset(
        {
            "extraction_id",
            "finalized_at",
            "segment_count",
            "total_char_count",
            "corpus_sha256",
            "canonicalization_version",
        }
    ),
}
_COMMUNICATION_REQUIRED_TRIGGERS = frozenset(
    {
        *(f"{name}_reject_update" for name in _COMMUNICATION_TABLE_NAMES),
        *(f"{name}_reject_delete" for name in _COMMUNICATION_TABLE_NAMES),
        "communication_artifacts_match_policy",
        "communication_policy_snapshot_validate",
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
        "communication_segments_reject_after_finalization",
        "communication_extraction_finalization_validate",
    }
)
_COMMUNICATION_CUSTOM_TRIGGER_FRAGMENTS = {
    "communication_policy_snapshot_validate": (
        "json_each(new.official_domains_json)",
        "json_each(new.material_types_json)",
        "json_each(new.commodity_families_json)",
        "new.rights_basis_url",
    ),
    "communication_artifacts_match_policy": (
        "policy.rights_note = new.rights_note",
        "policy.language = new.language",
        "event.event_version_sha256 = new.event_version_sha256",
        "json_each(policy.material_types_json)",
        "json_each(policy.official_domains_json)",
    ),
    "communication_coverage_matches_policy": (
        "policy.organization_type = 'commodity_company'",
        "policy.coverage_note = new.evidence_note",
        "json_each(policy.commodity_families_json)",
        "json_each(policy.official_domains_json)",
    ),
    "communication_event_successor_order": (
        "prior.organization_id = new.organization_id",
        "prior.event_key = new.event_key",
        "new.metadata_known_at > prior.metadata_known_at",
    ),
    "communication_artifact_reject_duplicate_root": (
        "prior_event.organization_id = new_event.organization_id",
        "prior_event.event_key = new_event.event_key",
        "prior.source_id = new.source_id",
        "prior.artifact_key = new.artifact_key",
    ),
    "communication_artifact_successor_order": (
        "prior.id = new.supersedes_artifact_id",
        "prior_event.organization_id = new_event.organization_id",
        "prior_event.event_key = new_event.event_key",
        "new.metadata_known_at > prior.metadata_known_at",
    ),
    "communication_coverage_successor_order": (
        "prior.id = new.supersedes_exposure_id",
        "prior.coverage_key = new.coverage_key",
        "not exists",
        "new.metadata_known_at > prior.metadata_known_at",
    ),
    "communication_coverage_reject_overlapping_head": (
        "prior.commodity_family = new.commodity_family",
        "prior.exposure_role = new.exposure_role",
        "not exists",
        "prior.effective_from <= coalesce(new.effective_to, '9999-12-31')",
    ),
    "communication_retrieval_matches_artifact": (
        "artifact.id = new.artifact_id",
        "artifact.landing_url = new.landing_url",
        "artifact.artifact_url = new.artifact_url",
        "new.retrieved_at >= artifact.available_at",
    ),
    "communication_content_matches_retrieval": (
        "retrieval.artifact_id = artifact.id",
        "retrieval.id = new.retrieval_id",
        "artifact.rights_status in ('cleared', 'internal_only')",
        "new.captured_at >= retrieval.metadata_known_at",
    ),
    "communication_content_successor_order": (
        "prior.id = new.supersedes_content_id",
        "prior.artifact_id = new.artifact_id",
        "not exists",
        "new.captured_at > prior.captured_at",
    ),
    "communication_extractions_require_content": (
        "from communication_artifact_contents",
        "id = new.artifact_content_id",
        "new.extracted_at >= captured_at",
    ),
    "communication_segments_reject_after_finalization": (
        "from communication_extraction_finalizations",
        "extraction_id = new.extraction_id",
    ),
    "communication_extraction_finalization_validate": (
        "new.finalized_at >= extraction.extracted_at",
        "select count(*) from communication_segments",
        "select min(ordinal) from communication_segments",
        "select max(ordinal) from communication_segments",
        "select sum(char_count) from communication_segments",
    ),
}
_COMMUNICATION_CANONICALIZATION = "communication_segments_json_v1"
_COMMUNICATION_MATERIAL_TYPES = frozenset(
    {
        "annual_report",
        "ceo_letter",
        "financial_results",
        "management_review",
        "monetary_policy_statement",
        "press_conference_transcript",
        "press_conference_video",
        "questions_and_answers",
        "results_transcript",
        "speech_text",
        "subtitles",
    }
)
_COMMUNICATION_COMMODITY_FAMILIES = frozenset(
    {
        "agricultural_raw_materials",
        "base_metals",
        "energy",
        "fertilizers",
        "food_and_beverages",
        "precious_metals",
    }
)
_COMMUNICATION_TRANSCRIPT_MATERIAL_TYPES = frozenset(
    {
        "press_conference_transcript",
        "questions_and_answers",
        "results_transcript",
        "subtitles",
    }
)
_COMMUNICATION_RIGHTS_TO_ACQUISITION = {
    "cleared": "manual_collection_ready",
    "internal_only": "manual_internal_only",
    "metadata_only": "metadata_only",
    "permission_required": "blocked_pending_permission",
    "rights_review_required": "manual_review_required",
}
_COMMUNICATION_ROLE_MATERIALS = {
    "prepared_remarks": frozenset(
        {
            "financial_results",
            "management_review",
            "monetary_policy_statement",
            "speech_text",
        }
    ),
    "q_and_a_transcript": frozenset({"questions_and_answers"}),
    "full_transcript": frozenset({"press_conference_transcript", "results_transcript"}),
    "ceo_letter": frozenset({"ceo_letter"}),
    "chair_letter": frozenset({"annual_report", "management_review"}),
    "annual_report": frozenset({"annual_report"}),
    "subtitles": frozenset({"subtitles"}),
    "webcast_video": frozenset({"press_conference_video"}),
}
_COMMUNICATION_MATERIAL_ORIGINS = {
    "annual_report": frozenset({"publisher_authored"}),
    "ceo_letter": frozenset({"publisher_authored"}),
    "financial_results": frozenset({"publisher_authored"}),
    "management_review": frozenset({"publisher_authored"}),
    "monetary_policy_statement": frozenset({"publisher_authored"}),
    "speech_text": frozenset({"publisher_authored"}),
    "questions_and_answers": frozenset({"official_published_transcript", "official_hosted_vendor"}),
    "press_conference_transcript": frozenset(
        {"official_published_transcript", "official_hosted_vendor"}
    ),
    "results_transcript": frozenset({"official_published_transcript", "official_hosted_vendor"}),
    "subtitles": frozenset({"official_caption", "automatic_caption", "local_asr"}),
    "press_conference_video": frozenset({"official_published_media"}),
}
_COMMUNICATION_ORIGIN_PROVENANCE = {
    "publisher_authored": "official_authored_text",
    "official_published_transcript": "official_published_transcript",
    "official_published_media": "official_published_media",
    "official_hosted_vendor": "official_hosted_third_party",
    "official_caption": "official_caption",
    "automatic_caption": "official_hosted_automatic_caption",
    "local_asr": "local_derived_asr",
}
_COMMUNICATION_CANONICAL_SEGMENT_FIELDS = (
    "ordinal",
    "segment_kind",
    "speaker_name",
    "speaker_role",
    "speaker_side",
    "section_title",
    "text",
    "page_start",
    "page_end",
    "paragraph_start",
    "paragraph_end",
    "start_ms",
    "end_ms",
)

EXPECTED_SERIES_IDS_BY_WORKSHEET = EXPECTED_PINK_SHEET_SERIES_IDS_BY_WORKSHEET
EXPECTED_COMMODITY_PRICE_SERIES = EXPECTED_PINK_SHEET_PRICE_SERIES_COUNT
EXPECTED_COMMODITY_INDEX_SERIES = EXPECTED_PINK_SHEET_INDEX_SERIES_COUNT
EXPECTED_COMMODITY_HISTORY_START = EXPECTED_PINK_SHEET_HISTORY_START
MIN_COMMODITY_HISTORY_MONTHS = MIN_PINK_SHEET_HISTORY_MONTHS
MIN_COMMODITY_OBSERVATIONS_PER_SERIES = MIN_PINK_SHEET_OBSERVATIONS_PER_SERIES
MAX_COMMODITY_LATEST_LAG_MONTHS = MAX_PINK_SHEET_LATEST_LAG_MONTHS
_CANONICAL_COMMODITY_SERIES_ID_RE = re.compile(
    r"^monthly_(?:prices|indices):[a-z0-9]+(?:_[a-z0-9]+)*$"
)
_PINK_SHEET_VINTAGE_RE = re.compile(
    rf"^{re.escape(PINK_SHEET_VINTAGE_PREFIX)}"
    r"(?P<workbook_sha256>[0-9a-f]{64});catalogue:v"
    r"(?P<schema_version>[1-9][0-9]*)/(?P<generator_version>[A-Za-z0-9._-]+)$"
)
_LEGACY_PINK_SHEET_VINTAGE_RE = re.compile(
    rf"^{re.escape(PINK_SHEET_VINTAGE_PREFIX)}(?P<workbook_sha256>[0-9a-f]{{64}})$"
)


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return value


def _dict_rows(rows) -> list[dict[str, Any]]:
    return [{key: _json_value(value) for key, value in row._mapping.items()} for row in rows]


def _reflect(connection: Connection, names: set[str]) -> dict[str, Table]:
    metadata = MetaData()
    return {
        name: Table(
            name,
            metadata,
            autoload_with=connection,
            resolve_fks=False,
        )
        for name in sorted(names)
    }


def _count(connection: Connection, table: Table | None) -> int | None:
    if table is None:
        return None
    return int(connection.scalar(select(func.count()).select_from(table)) or 0)


def _communication_catalogue_summary() -> dict[str, Any]:
    """Keep a broken optional policy catalogue from disabling numeric inventory."""
    try:
        from dalio.communications.catalogue import (
            COMMUNICATION_CATALOGUE_SHA256,
            COMMUNICATION_SOURCES,
        )

        return {
            "validation_status": "valid",
            "source_policy_count": len(COMMUNICATION_SOURCES),
            "organization_count": len({source.organization_id for source in COMMUNICATION_SOURCES}),
            "sha256": COMMUNICATION_CATALOGUE_SHA256,
            "error": None,
        }
    except Exception as exc:
        return {
            "validation_status": "invalid",
            "source_policy_count": None,
            "organization_count": None,
            "sha256": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _latest_release_ids(releases: Table | None):
    if releases is None:
        return None
    ranked = select(
        releases.c.id.label("release_id"),
        func.row_number()
        .over(
            partition_by=releases.c.partition_key,
            order_by=(
                releases.c.available_at.desc(),
                releases.c.retrieved_at.desc(),
                releases.c.id.desc(),
            ),
        )
        .label("release_rank"),
    ).subquery()
    return select(ranked.c.release_id).where(ranked.c.release_rank == 1)


def _current_condition(table: Table, latest_release_ids):
    if latest_release_ids is None or "release_id" not in table.c:
        return None
    return table.c.release_id.in_(latest_release_ids)


def _observation_inventory(
    connection: Connection,
    observations: Table | None,
) -> dict[str, Any]:
    if observations is None:
        return {"row_count": None, "sources": [], "partitions": []}

    row_count = _count(connection, observations)
    source_rows = connection.execute(
        select(
            observations.c.source.label("source"),
            func.count().label("row_count"),
            func.count(func.distinct(observations.c.country)).label("country_count"),
            func.count(func.distinct(observations.c.indicator)).label("indicator_count"),
            func.min(observations.c.date).label("first_date"),
            func.max(observations.c.date).label("latest_date"),
        )
        .group_by(observations.c.source)
        .order_by(observations.c.source)
    ).all()
    partition_rows = connection.execute(
        select(
            observations.c.source.label("source"),
            observations.c.country.label("country"),
            observations.c.indicator.label("indicator"),
            func.count().label("row_count"),
            func.min(observations.c.date).label("first_date"),
            func.max(observations.c.date).label("latest_date"),
        )
        .group_by(observations.c.source, observations.c.country, observations.c.indicator)
        .order_by(observations.c.source, observations.c.country, observations.c.indicator)
    ).all()
    return {
        "row_count": row_count,
        "sources": _dict_rows(source_rows),
        "partitions": _dict_rows(partition_rows),
    }


def _release_inventory(
    connection: Connection,
    releases: Table | None,
    release_observations: Table | None,
) -> dict[str, Any]:
    if releases is None:
        return {
            "release_count": None,
            "observation_row_count": _count(connection, release_observations),
            "by_source": [],
        }

    by_source = _dict_rows(
        connection.execute(
            select(
                releases.c.source_family.label("source_family"),
                func.count().label("release_count"),
                func.count(func.distinct(releases.c.partition_key)).label("partition_count"),
                func.sum(releases.c.row_count).label("declared_row_count"),
                func.min(releases.c.available_at).label("first_available_at"),
                func.max(releases.c.available_at).label("latest_available_at"),
            )
            .group_by(releases.c.source_family)
            .order_by(releases.c.source_family)
        ).all()
    )
    stored_by_source: dict[str, int] = {}
    if release_observations is not None:
        stored_by_source = {
            str(row.source_family): int(row.stored_observation_rows)
            for row in connection.execute(
                select(
                    releases.c.source_family.label("source_family"),
                    func.count(release_observations.c.id).label("stored_observation_rows"),
                )
                .select_from(
                    releases.join(
                        release_observations,
                        release_observations.c.release_id == releases.c.id,
                    )
                )
                .group_by(releases.c.source_family)
            )
        }
    for row in by_source:
        row["stored_observation_rows"] = stored_by_source.get(row["source_family"], 0)
    return {
        "release_count": _count(connection, releases),
        "observation_row_count": _count(connection, release_observations),
        "by_source": by_source,
    }


def _commodity_expected_ids() -> set[str]:
    return set().union(*EXPECTED_SERIES_IDS_BY_WORKSHEET.values())


def _commodity_empty_inventory(*, as_of: date) -> dict[str, Any]:
    expected_series = len(_commodity_expected_ids())
    return {
        "source": SOURCE_WORLD_BANK_COMMODITIES,
        "status": "empty",
        "row_count": 0,
        "series_count": 0,
        "expected_series": expected_series,
        "stored_series": 0,
        "ready_series": 0,
        "guard_failed_series": 0,
        "missing_series": expected_series,
        "extra_series": 0,
        "stored_coverage_pct": 0.0,
        "coverage_pct": 0.0,
        "expected_price_series_count": EXPECTED_COMMODITY_PRICE_SERIES,
        "expected_index_series_count": EXPECTED_COMMODITY_INDEX_SERIES,
        "price_series_count": 0,
        "index_series_count": 0,
        "first_date": None,
        "latest_date": None,
        "latest_lag_months": None,
        "nonpositive_rows": 0,
        "nonfinite_rows": 0,
        "missing_expected_series_ids": sorted(_commodity_expected_ids()),
        "extra_series_ids": [],
        "workbook_sha256": None,
        "release_group_vintage_label": None,
        "release_group_available_at": None,
        "release_group_retrieved_at": None,
        "expected_catalogue_schema_version": CATALOGUE_SCHEMA_VERSION,
        "expected_catalogue_generator_version": CATALOGUE_GENERATOR_VERSION,
        "catalogue_schema_version": None,
        "catalogue_generator_version": None,
        "catalogue_semantic_status": None,
        "evaluated_as_of": as_of.isoformat(),
        "guard_failures": ["missing"],
        "series": [],
    }


def _as_inventory_date(value: object) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _commodity_inventory(
    connection: Connection,
    observations: Table,
    releases: Table | None,
    commodity_series: list[dict[str, Any]],
    *,
    as_of: date,
) -> dict[str, Any]:
    """Qualify the current Pink Sheet projection, not merely count its rows."""

    empty = _commodity_empty_inventory(as_of=as_of)
    if not commodity_series:
        return empty

    expected_by_worksheet = {
        worksheet: set(series_ids)
        for worksheet, series_ids in EXPECTED_SERIES_IDS_BY_WORKSHEET.items()
    }
    expected_ids = set().union(*expected_by_worksheet.values())
    actual_ids = {str(row["series_id"]) for row in commodity_series}
    stored_expected = actual_ids & expected_ids
    missing_expected = expected_ids - actual_ids
    extra_ids = actual_ids - expected_ids
    failures: list[str] = []

    def fail(reason: str) -> None:
        if reason not in failures:
            failures.append(reason)

    if missing_expected:
        fail("expected_catalogue")
    if (
        len(actual_ids) != len(commodity_series)
        or any(str(row["country"]) != WORLD_CODE for row in commodity_series)
        or any(
            _CANONICAL_COMMODITY_SERIES_ID_RE.fullmatch(str(row["series_id"])) is None
            for row in commodity_series
        )
    ):
        fail("canonical_series_identity")

    price_count = sum(
        str(row["indicator"]).startswith("commodity_price_") for row in commodity_series
    )
    index_count = sum(
        str(row["indicator"]).startswith("commodity_index_") for row in commodity_series
    )
    expected_monthly_indices = expected_by_worksheet.get(MONTHLY_INDICES_SHEET_NAME, set())
    natural_gas_index_id = "monthly_prices:natural_gas_index"

    def semantic_mismatch(row: dict[str, Any]) -> bool:
        series_id = str(row["series_id"])
        indicator = str(row["indicator"])
        is_index = indicator.startswith("commodity_index_")
        if not indicator.startswith(("commodity_price_", "commodity_index_")):
            return True
        if series_id in expected_ids:
            expected_index = (
                series_id in expected_monthly_indices or series_id == natural_gas_index_id
            )
            slug = series_id.split(":", 1)[1]
            expected_indicator = (
                f"commodity_index_{slug}" if expected_index else f"commodity_price_{slug}"
            )
            return indicator != expected_indicator
        if series_id.startswith("monthly_indices:"):
            return not is_index
        # A newly published Monthly Prices column may be a price or an index;
        # its canonical indicator preserves that semantic without pretending
        # the database alone contains its unit metadata.
        return False

    semantic_mismatches = [
        str(row["series_id"]) for row in commodity_series if semantic_mismatch(row)
    ]
    if semantic_mismatches:
        fail("canonical_indicator_mapping")
    if (
        price_count < EXPECTED_COMMODITY_PRICE_SERIES
        or index_count < EXPECTED_COMMODITY_INDEX_SERIES
    ):
        fail("semantic_series_counts")

    raw_rows = connection.execute(
        select(
            observations.c.country,
            observations.c.indicator,
            observations.c.series_id,
            observations.c.date,
            observations.c.value,
        )
        .where(observations.c.source == SOURCE_WORLD_BANK_COMMODITIES)
        .order_by(
            observations.c.series_id,
            observations.c.indicator,
            observations.c.date,
        )
    ).all()
    dates_by_series: dict[str, list[date]] = defaultdict(list)
    periods_by_worksheet: dict[str, set[date]] = defaultdict(set)
    nonpositive_rows = 0
    nonfinite_rows = 0
    for row in raw_rows:
        observed_on = _as_inventory_date(row.date)
        series_id = str(row.series_id)
        dates_by_series[series_id].append(observed_on)
        if series_id.startswith("monthly_prices:"):
            periods_by_worksheet[MONTHLY_SHEET_NAME].add(observed_on)
        elif series_id.startswith("monthly_indices:"):
            periods_by_worksheet[MONTHLY_INDICES_SHEET_NAME].add(observed_on)
        try:
            numeric = float(row.value)
        except (TypeError, ValueError):
            nonfinite_rows += 1
        else:
            nonpositive_rows += numeric <= 0
            nonfinite_rows += not math.isfinite(numeric)
    if nonpositive_rows:
        fail("nonpositive_values")
    if nonfinite_rows:
        fail("nonfinite_values")
    if any(
        len(dates) < MIN_COMMODITY_OBSERVATIONS_PER_SERIES for dates in dates_by_series.values()
    ):
        fail("minimum_observations")

    worksheet_lags: list[int] = []
    for worksheet in (MONTHLY_SHEET_NAME, MONTHLY_INDICES_SHEET_NAME):
        periods = sorted(periods_by_worksheet[worksheet])
        if not periods:
            fail("expected_start")
            fail("minimum_history")
            continue
        if periods[0] != EXPECTED_COMMODITY_HISTORY_START:
            fail("expected_start")
        ordinals = [period.year * 12 + period.month for period in periods]
        if any(period.day != 1 for period in periods) or any(
            right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)
        ):
            fail("monthly_cadence")
        if len(periods) < MIN_COMMODITY_HISTORY_MONTHS:
            fail("minimum_history")
        as_of_ordinal = as_of.year * 12 + as_of.month
        latest_ordinal = periods[-1].year * 12 + periods[-1].month
        lag_months = as_of_ordinal - latest_ordinal
        worksheet_lags.append(lag_months)
        if lag_months < 0:
            fail("future_observations")
        elif lag_months > MAX_COMMODITY_LATEST_LAG_MONTHS:
            fail("latest_lag")

    latest_by_partition: dict[str, dict[str, Any]] = {}
    if releases is not None:
        ranked = (
            select(
                releases.c.partition_key.label("partition_key"),
                releases.c.vintage_label.label("vintage_label"),
                releases.c.available_at.label("available_at"),
                releases.c.retrieved_at.label("retrieved_at"),
                releases.c.row_count.label("release_row_count"),
                func.row_number()
                .over(
                    partition_by=releases.c.partition_key,
                    order_by=(
                        releases.c.available_at.desc(),
                        releases.c.retrieved_at.desc(),
                        releases.c.id.desc(),
                    ),
                )
                .label("release_rank"),
            )
            .where(releases.c.source_family == SOURCE_WORLD_BANK_COMMODITIES)
            .subquery()
        )
        latest_by_partition = {
            str(row["partition_key"]): row
            for row in _execute_rows(
                connection,
                select(
                    ranked.c.partition_key,
                    ranked.c.vintage_label,
                    ranked.c.available_at,
                    ranked.c.retrieved_at,
                    ranked.c.release_row_count,
                ).where(ranked.c.release_rank == 1),
            )
        }

    release_groups: set[tuple[str, str, str]] = set()
    qualified_release_count = 0
    enriched_series: list[dict[str, Any]] = []
    for series in commodity_series:
        partition_key = make_partition_key(
            SOURCE_WORLD_BANK_COMMODITIES,
            str(series["series_id"]),
            str(series["country"]),
            str(series["indicator"]),
        )
        latest_release = latest_by_partition.get(partition_key)
        release_label = latest_release["vintage_label"] if latest_release else None
        release_row_count = int(latest_release["release_row_count"]) if latest_release else None
        release_available_at = latest_release["available_at"] if latest_release else None
        release_retrieved_at = latest_release["retrieved_at"] if latest_release else None
        release_matches_projection = bool(
            latest_release and release_row_count == int(series["row_count"])
        )
        if (
            not release_matches_projection
            or not isinstance(release_label, str)
            or not isinstance(release_available_at, str)
            or not isinstance(release_retrieved_at, str)
        ):
            fail("latest_release_group")
        else:
            qualified_release_count += 1
            release_groups.add((release_label, release_available_at, release_retrieved_at))
        enriched_series.append(
            {
                **series,
                "partition_key": partition_key,
                "latest_release_row_count": release_row_count,
                "latest_release_vintage_label": release_label,
                "latest_release_available_at": release_available_at,
                "latest_release_retrieved_at": release_retrieved_at,
                "release_matches_projection": release_matches_projection,
            }
        )
    if len(release_groups) != 1 or qualified_release_count != len(commodity_series):
        fail("latest_release_group")

    release_group = next(iter(release_groups)) if len(release_groups) == 1 else None
    release_group_label = release_group[0] if release_group is not None else None
    release_group_available_at = release_group[1] if release_group is not None else None
    release_group_retrieved_at = release_group[2] if release_group is not None else None
    workbook_sha256: str | None = None
    catalogue_schema_version: int | None = None
    catalogue_generator_version: str | None = None
    catalogue_semantic_status: str | None = None
    if release_group_label is not None:
        match = _PINK_SHEET_VINTAGE_RE.fullmatch(release_group_label)
        legacy_match = _LEGACY_PINK_SHEET_VINTAGE_RE.fullmatch(release_group_label)
        if match is not None:
            workbook_sha256 = match.group("workbook_sha256")
            catalogue_schema_version = int(match.group("schema_version"))
            catalogue_generator_version = match.group("generator_version")
            catalogue_semantic_status = (
                "match"
                if (
                    catalogue_schema_version == CATALOGUE_SCHEMA_VERSION
                    and catalogue_generator_version == CATALOGUE_GENERATOR_VERSION
                    and release_group_label.endswith(PINK_SHEET_CATALOGUE_VINTAGE_TAG)
                )
                else "mismatch"
            )
        elif legacy_match is not None:
            workbook_sha256 = legacy_match.group("workbook_sha256")
            catalogue_semantic_status = "unrecorded"
        else:
            catalogue_semantic_status = "unrecorded"
        if catalogue_semantic_status != "match":
            fail("catalogue_semantics")
    else:
        catalogue_semantic_status = "unrecorded" if release_groups else None

    status = "ready" if not failures else "guard_failed"
    expected_count = len(expected_ids)
    ready_series = expected_count if status == "ready" else 0
    all_dates = [_as_inventory_date(row.date) for row in raw_rows]
    latest_lag_months = max(worksheet_lags) if worksheet_lags else None
    return {
        **empty,
        "status": status,
        "row_count": len(raw_rows),
        "series_count": len(commodity_series),
        "expected_series": expected_count,
        "stored_series": len(actual_ids),
        "ready_series": ready_series,
        "guard_failed_series": len(actual_ids) if failures else 0,
        "missing_series": len(missing_expected),
        "extra_series": len(extra_ids),
        "stored_coverage_pct": _coverage_pct(len(stored_expected), expected_count),
        "coverage_pct": _coverage_pct(ready_series, expected_count),
        "price_series_count": price_count,
        "index_series_count": index_count,
        "first_date": min(all_dates).isoformat() if all_dates else None,
        "latest_date": max(all_dates).isoformat() if all_dates else None,
        "latest_lag_months": latest_lag_months,
        "nonpositive_rows": nonpositive_rows,
        "nonfinite_rows": nonfinite_rows,
        "missing_expected_series_ids": sorted(missing_expected),
        "extra_series_ids": sorted(extra_ids),
        "workbook_sha256": workbook_sha256,
        "release_group_vintage_label": release_group_label,
        "release_group_available_at": release_group_available_at,
        "release_group_retrieved_at": release_group_retrieved_at,
        "catalogue_schema_version": catalogue_schema_version,
        "catalogue_generator_version": catalogue_generator_version,
        "catalogue_semantic_status": catalogue_semantic_status,
        "guard_failures": failures,
        "series": enriched_series,
    }


def _shadow_catalogue_item(spec: Any, catalogue_hash: str) -> dict[str, Any]:
    """Return every catalogue field in a JSON-stable inventory record."""
    record = asdict(spec)
    for key, value in tuple(record.items()):
        if isinstance(value, date):
            record[key] = value.isoformat()
        elif isinstance(value, tuple):
            record[key] = list(value)

    if spec.source_family == SOURCE_BIS_GLI:
        record["side"] = spec.claim_side
        record["parent_native_series_id"] = None
        record["parents"] = []
        record["non_additive_groups"] = [spec.non_additive_group]
    else:
        record["side"] = spec.economic_side
        parent = spec.parent_native_series_id
        record["parents"] = [] if parent is None else [parent]

    record.update(
        {
            "source": spec.source_family,
            "series_id": spec.native_series_id,
            "url": spec.url,
            "partition_key": make_partition_key(
                spec.source_family,
                spec.native_series_id,
                spec.country,
                spec.indicator,
            ),
            "catalogue_semantic_sha256": catalogue_hash,
        }
    )
    return record


def _shadow_expected_catalogue() -> tuple[list[dict[str, Any]], dict[str, str]]:
    provider_hashes = {
        SOURCE_BIS_GLI: bis_global_liquidity_catalogue_sha256(),
        SOURCE_OFR_STFM: ofr_shadow_liquidity_catalogue_sha256(),
    }
    records = [
        _shadow_catalogue_item(spec, provider_hashes[spec.source_family])
        for spec in (*BIS_GLOBAL_LIQUIDITY_SERIES, *OFR_SHADOW_LIQUIDITY_SERIES)
    ]
    return records, provider_hashes


def _shadow_history_guard_failures(
    dates: list[date],
    spec: Any,
    *,
    as_of: date,
    catalogue_semantic_status: str | None,
    payload_provenance_status: str | None,
    artifact_manifest_status: str | None,
) -> list[str]:
    """Qualify one BIS/OFR projection under its source-native cadence."""
    failures: list[str] = []
    if not dates:
        return ["missing_observations"]
    if dates[0] != spec.expected_start:
        failures.append("expected_start")
    if len(dates) < spec.minimum_observations:
        failures.append("minimum_observations")
    if len(dates) != len(set(dates)):
        failures.append("duplicate_dates")

    if spec.source_family == SOURCE_BIS_GLI:
        quarter_ordinals = [
            observed_on.year * 4 + (observed_on.month - 1) // 3 for observed_on in dates
        ]
        if (
            spec.frequency != "quarterly"
            or any(
                observed_on.day != 1 or observed_on.month not in {1, 4, 7, 10}
                for observed_on in dates
            )
            or any(
                right - left != 1
                for left, right in zip(quarter_ordinals, quarter_ordinals[1:], strict=False)
            )
        ):
            failures.append("quarterly_cadence")
    elif spec.source_family == SOURCE_OFR_STFM and spec.frequency == "monthly":
        if any(
            observed_on.day != calendar.monthrange(observed_on.year, observed_on.month)[1]
            for observed_on in dates
        ):
            failures.append("calendar_month_end")
        if spec.cadence_policy == "complete_monthly":
            month_ordinals = [observed_on.year * 12 + observed_on.month for observed_on in dates]
            if any(
                right - left != 1
                for left, right in zip(month_ordinals, month_ordinals[1:], strict=False)
            ):
                failures.append("monthly_cadence")
        elif spec.cadence_policy != "sparse_monthly":
            failures.append("cadence_policy")
        # Sparse native months and explicit nulls remain missing; never zero-fill.
    elif spec.source_family == SOURCE_OFR_STFM and spec.frequency == "daily":
        if any(observed_on.weekday() >= 5 for observed_on in dates):
            failures.append("business_day_cadence")
        if spec.cadence_policy != "observed_business_days":
            failures.append("cadence_policy")
        if spec.max_internal_gap_days is None or spec.minimum_weekday_coverage_ratio is None:
            failures.append("daily_completeness_policy")
        else:
            gaps = [(right - left).days for left, right in zip(dates, dates[1:], strict=False)]
            if gaps and max(gaps) > spec.max_internal_gap_days:
                failures.append("maximum_internal_gap")
            expected_weekdays = sum(
                1
                for offset in range((dates[-1] - dates[0]).days + 1)
                if (dates[0] + timedelta(days=offset)).weekday() < 5
            )
            if len(dates) / expected_weekdays < spec.minimum_weekday_coverage_ratio:
                failures.append("minimum_weekday_coverage")
        # Holidays, no-trade days, and disclosure edits permit bounded gaps.
    else:
        failures.append("unsupported_frequency")

    latest_lag_days = (as_of - dates[-1]).days
    if latest_lag_days < 0:
        failures.append("future_latest_observation")
    elif latest_lag_days > spec.max_latest_lag_days:
        failures.append("latest_lag")
    if catalogue_semantic_status == "unrecorded":
        failures.append("catalogue_semantic_unrecorded")
    elif catalogue_semantic_status == "mismatch":
        failures.append("catalogue_semantic_mismatch")
    if spec.source_family == SOURCE_OFR_STFM and payload_provenance_status != "recorded":
        failures.append("payload_provenance_unrecorded")
    if artifact_manifest_status != "valid":
        failures.append("artifact_manifest_invalid")
    return failures


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_artifact_manifest(
    connection: Connection,
    artifacts: Table | None,
    *,
    release_id: int | None,
    derived_payloads: bool,
) -> dict[str, Any]:
    """Re-hash the latest release's retained files instead of trusting locators."""
    expected_roles = (
        {"source_response", "native_series_payload", "missingness_ledger"}
        if derived_payloads
        else {"source_response"}
    )
    if release_id is None:
        return {
            "status": "unrecorded",
            "artifact_count": 0,
            "roles": [],
            "artifact_sha256_by_role": {},
            "native_payload_sha256": None,
            "missing_provenance_sha256": None,
            "failures": ["release_missing"],
        }
    if artifacts is None:
        return {
            "status": "table_absent",
            "artifact_count": 0,
            "roles": [],
            "artifact_sha256_by_role": {},
            "native_payload_sha256": None,
            "missing_provenance_sha256": None,
            "failures": ["artifact_table_absent"],
        }

    rows = _execute_rows(
        connection,
        select(
            artifacts.c.role,
            artifacts.c.artifact_sha256,
            artifacts.c.artifact_path,
            artifacts.c.native_payload_sha256,
            artifacts.c.missing_provenance_sha256,
            artifacts.c.provenance_json,
        )
        .where(artifacts.c.release_id == release_id)
        .order_by(artifacts.c.role),
    )
    roles = {str(row["role"]) for row in rows}
    failures: list[str] = []
    missing_roles = sorted(expected_roles - roles)
    unexpected_roles = sorted(roles - expected_roles)
    if missing_roles:
        failures.append(f"missing_roles:{','.join(missing_roles)}")
    if unexpected_roles:
        failures.append(f"unexpected_roles:{','.join(unexpected_roles)}")

    native_hashes: set[str] = set()
    missing_hashes: set[str] = set()
    hashes_by_role: dict[str, str] = {}
    hex_characters = set("0123456789abcdef")
    for row in rows:
        role = str(row["role"])
        artifact_hash = str(row["artifact_sha256"])
        native_hash = str(row["native_payload_sha256"])
        missing_hash = str(row["missing_provenance_sha256"])
        hashes_by_role[role] = artifact_hash
        native_hashes.add(native_hash)
        missing_hashes.add(missing_hash)
        for label, value in (
            ("artifact", artifact_hash),
            ("native", native_hash),
            ("missing", missing_hash),
        ):
            if len(value) != 64 or set(value) - hex_characters:
                failures.append(f"{role}:{label}_sha256_invalid")

        try:
            provenance = json.loads(str(row["provenance_json"]))
            canonical = json.dumps(
                provenance,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            if canonical != row["provenance_json"]:
                failures.append(f"{role}:provenance_not_canonical")
            if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != missing_hash:
                failures.append(f"{role}:provenance_hash_mismatch")
        except (TypeError, ValueError, json.JSONDecodeError):
            failures.append(f"{role}:provenance_invalid")

        try:
            path = Path(str(row["artifact_path"]))
            if not path.is_absolute() or not path.is_file():
                failures.append(f"{role}:artifact_missing")
            elif _file_sha256(path) != artifact_hash:
                failures.append(f"{role}:artifact_hash_mismatch")
        except OSError:
            failures.append(f"{role}:artifact_unreadable")

    if len(native_hashes) > 1:
        failures.append("native_payload_hash_inconsistent")
    if len(missing_hashes) > 1:
        failures.append("missing_provenance_hash_inconsistent")
    if derived_payloads:
        if hashes_by_role.get("native_series_payload") not in native_hashes:
            failures.append("native_payload_artifact_mismatch")
        if hashes_by_role.get("missingness_ledger") not in missing_hashes:
            failures.append("missingness_ledger_artifact_mismatch")
    elif hashes_by_role.get("source_response") not in native_hashes:
        failures.append("source_response_native_payload_mismatch")

    return {
        "status": "valid" if not failures else "invalid",
        "artifact_count": len(rows),
        "roles": sorted(roles),
        "artifact_sha256_by_role": hashes_by_role,
        "native_payload_sha256": next(iter(native_hashes), None),
        "missing_provenance_sha256": next(iter(missing_hashes), None),
        "failures": failures,
    }


def _shadow_provider_inventory(
    series: list[dict[str, Any]],
    provider_hashes: dict[str, str],
) -> dict[str, dict[str, Any]]:
    providers: dict[str, dict[str, Any]] = {}
    for source, catalogue_hash in provider_hashes.items():
        selected = [item for item in series if item["source_family"] == source]
        stored = sum(item["storage_status"] == "stored" for item in selected)
        ready = sum(item["status"] == "ready" for item in selected)
        failed = sum(item["status"] == "guard_failed" for item in selected)
        providers[source] = {
            "expected_series": len(selected),
            "stored_series": stored,
            "ready_series": ready,
            "guard_failed_series": failed,
            "missing_series": len(selected) - stored,
            "stored_coverage_pct": _coverage_pct(stored, len(selected)),
            "coverage_pct": _coverage_pct(ready, len(selected)),
            "catalogue_semantic_sha256": catalogue_hash,
        }
    return providers


def _shadow_liquidity_inventory(
    connection: Connection,
    observations: Table | None,
    releases: Table | None,
    artifacts: Table | None,
    *,
    as_of: date,
) -> dict[str, Any]:
    """Inventory BIS offshore credit and OFR funding markets separately from money."""
    expected, provider_hashes = _shadow_expected_catalogue()
    specs = {
        (spec.source_family, spec.country, spec.indicator, spec.native_series_id): spec
        for spec in (*BIS_GLOBAL_LIQUIDITY_SERIES, *OFR_SHADOW_LIQUIDITY_SERIES)
    }
    if observations is None:
        series = [
            {
                **item,
                "status": "missing",
                "storage_status": "missing",
                "row_count": 0,
                "first_date": None,
                "latest_date": None,
                "latest_lag_days": None,
                "latest_release_vintage_label": None,
                "release_catalogue_semantic_sha256": None,
                "release_payload_provenance_tag": None,
                "payload_provenance_status": None,
                "artifact_manifest": None,
                "artifact_manifest_status": None,
                "catalogue_semantic_status": None,
                "guard_failures": ["missing"],
            }
            for item in expected
        ]
    else:
        sources = set(provider_hashes)
        grouped = _execute_rows(
            connection,
            select(
                observations.c.source.label("source"),
                observations.c.country.label("country"),
                observations.c.indicator.label("indicator"),
                observations.c.series_id.label("series_id"),
                func.count().label("row_count"),
                func.min(observations.c.date).label("first_date"),
                func.max(observations.c.date).label("latest_date"),
            )
            .where(observations.c.source.in_(sources))
            .group_by(
                observations.c.source,
                observations.c.country,
                observations.c.indicator,
                observations.c.series_id,
            )
            .order_by(
                observations.c.source,
                observations.c.country,
                observations.c.indicator,
                observations.c.series_id,
            ),
        )
        stored_by_key = {
            (row["source"], row["country"], row["indicator"], row["series_id"]): row
            for row in grouped
        }
        dates_by_key: dict[tuple[str, str, str, str], list[date]] = defaultdict(list)
        for row in connection.execute(
            select(
                observations.c.source,
                observations.c.country,
                observations.c.indicator,
                observations.c.series_id,
                observations.c.date,
            )
            .where(observations.c.source.in_(sources))
            .order_by(
                observations.c.source,
                observations.c.country,
                observations.c.indicator,
                observations.c.series_id,
                observations.c.date,
            )
        ):
            key = (
                str(row.source),
                str(row.country),
                str(row.indicator),
                str(row.series_id),
            )
            dates_by_key[key].append(_as_inventory_date(row.date))

        latest_releases: dict[str, dict[str, Any]] = {}
        partition_keys = tuple(item["partition_key"] for item in expected)
        if releases is not None and partition_keys:
            ranked = (
                select(
                    releases.c.id.label("release_id"),
                    releases.c.partition_key.label("partition_key"),
                    releases.c.vintage_label.label("vintage_label"),
                    func.row_number()
                    .over(
                        partition_by=releases.c.partition_key,
                        order_by=(
                            releases.c.available_at.desc(),
                            releases.c.retrieved_at.desc(),
                            releases.c.id.desc(),
                        ),
                    )
                    .label("release_rank"),
                )
                .where(releases.c.partition_key.in_(partition_keys))
                .subquery()
            )
            latest_releases = {
                str(row["partition_key"]): row
                for row in _execute_rows(
                    connection,
                    select(
                        ranked.c.release_id,
                        ranked.c.partition_key,
                        ranked.c.vintage_label,
                    ).where(ranked.c.release_rank == 1),
                )
            }

        series = []
        for item in expected:
            key = (
                item["source_family"],
                item["country"],
                item["indicator"],
                item["native_series_id"],
            )
            stored = stored_by_key.get(key)
            dates = dates_by_key.get(key, [])
            latest_release = latest_releases.get(item["partition_key"])
            vintage_label = latest_release["vintage_label"] if latest_release else None
            release_id = int(latest_release["release_id"]) if latest_release else None
            artifact_manifest = _release_artifact_manifest(
                connection,
                artifacts,
                release_id=release_id,
                derived_payloads=item["source_family"] == SOURCE_OFR_STFM,
            )
            prefix = (
                BIS_GLI_CATALOGUE_VINTAGE_PREFIX
                if item["source_family"] == SOURCE_BIS_GLI
                else OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX
            )
            release_hash = None
            payload_provenance_tag = None
            if isinstance(vintage_label, str):
                if item["source_family"] == SOURCE_BIS_GLI:
                    match = re.fullmatch(
                        rf"{re.escape(prefix)}(?P<catalogue>[0-9a-f]{{64}})",
                        vintage_label,
                    )
                else:
                    match = re.fullmatch(
                        rf"{re.escape(prefix)}(?P<catalogue>[0-9a-f]{{64}})"
                        r"(?P<suffix>.*)",
                        vintage_label,
                    )
                if match is not None:
                    release_hash = match.group("catalogue")
                    if item["source_family"] == SOURCE_OFR_STFM:
                        payload_match = re.fullmatch(
                            r";p:(?P<payload>[0-9a-f]{24})",
                            match.group("suffix"),
                        )
                        if payload_match is not None:
                            payload_provenance_tag = payload_match.group("payload")
            payload_provenance_status = (
                None
                if not stored or item["source_family"] != SOURCE_OFR_STFM
                else "recorded"
                if payload_provenance_tag is not None
                else "unrecorded"
            )
            semantic_status = (
                None
                if not stored
                else "unrecorded"
                if release_hash is None
                else "match"
                if release_hash == item["catalogue_semantic_sha256"]
                else "mismatch"
            )
            failures = (
                _shadow_history_guard_failures(
                    dates,
                    specs[key],
                    as_of=as_of,
                    catalogue_semantic_status=semantic_status,
                    payload_provenance_status=payload_provenance_status,
                    artifact_manifest_status=artifact_manifest["status"],
                )
                if stored
                else ["missing"]
            )
            series.append(
                {
                    **item,
                    "status": (
                        "missing" if not stored else "guard_failed" if failures else "ready"
                    ),
                    "storage_status": "stored" if stored else "missing",
                    "row_count": int(stored["row_count"]) if stored else 0,
                    "first_date": stored["first_date"] if stored else None,
                    "latest_date": stored["latest_date"] if stored else None,
                    "latest_lag_days": (as_of - dates[-1]).days if dates else None,
                    "latest_release_vintage_label": vintage_label,
                    "release_catalogue_semantic_sha256": release_hash,
                    "release_payload_provenance_tag": payload_provenance_tag,
                    "payload_provenance_status": payload_provenance_status,
                    "artifact_manifest": artifact_manifest,
                    "artifact_manifest_status": artifact_manifest["status"],
                    "catalogue_semantic_status": semantic_status,
                    "guard_failures": failures,
                }
            )

    expected_keys = {
        (item["source_family"], item["country"], item["indicator"], item["native_series_id"])
        for item in expected
    }
    actual_keys = (
        {(row["source"], row["country"], row["indicator"], row["series_id"]) for row in grouped}
        if observations is not None
        else set()
    )
    stored_count = sum(item["storage_status"] == "stored" for item in series)
    ready_count = sum(item["status"] == "ready" for item in series)
    guard_failed_count = sum(item["status"] == "guard_failed" for item in series)
    return {
        "expected_series": len(series),
        "stored_series": stored_count,
        "ready_series": ready_count,
        "guard_failed_series": guard_failed_count,
        "missing_series": len(series) - stored_count,
        "extra_series": len(actual_keys - expected_keys),
        "stored_coverage_pct": _coverage_pct(stored_count, len(series)),
        "coverage_pct": _coverage_pct(ready_count, len(series)),
        "evaluated_as_of": as_of.isoformat(),
        "provider_catalogue_semantic_sha256": provider_hashes,
        "providers": _shadow_provider_inventory(series, provider_hashes),
        "catalogue_semantic_mismatches": sum(
            item["catalogue_semantic_status"] == "mismatch" for item in series
        ),
        "catalogue_semantic_unrecorded": sum(
            item["catalogue_semantic_status"] == "unrecorded" for item in series
        ),
        "payload_provenance_unrecorded": sum(
            item["payload_provenance_status"] == "unrecorded" for item in series
        ),
        "artifact_manifest_invalid": sum(
            item["artifact_manifest_status"] not in {None, "valid"} for item in series
        ),
        "series": series,
        "classification_note": (
            "BIS offshore credit, OFR MMF assets, repo volumes, and repo rates are "
            "separate non-additive evidence; readiness never implies they can be summed"
        ),
    }


def _market_history_inventory(
    connection: Connection,
    observations: Table | None,
    releases: Table | None,
    release_artifacts: Table | None,
    *,
    as_of: date,
) -> dict[str, Any]:
    """Describe stored commodity and monetary histories without deriving signals."""
    empty_commodity = _commodity_empty_inventory(as_of=as_of)
    shadow_liquidity = _shadow_liquidity_inventory(
        connection,
        observations,
        releases,
        release_artifacts,
        as_of=as_of,
    )
    catalogue_hash = money_liquidity_catalogue_sha256()
    expected_money = [
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "source": spec.source_family,
            "series_id": spec.native_series_id,
            "currency": spec.currency,
            "unit": spec.unit,
            "native_unit": spec.native_unit,
            "unit_multiplier": spec.unit_multiplier,
            "frequency": spec.frequency,
            "adjustment": spec.adjustment,
            "research_role": spec.research_role,
            "observation_basis": spec.observation_basis,
            "perimeter": spec.perimeter,
            "definition_notes": spec.definition_notes,
            "parent_native_series_id": spec.parent_native_series_id,
            "non_additive_groups": list(spec.non_additive_groups),
            "publisher": spec.publisher,
            "delivery_service": spec.delivery_service,
            "title": spec.title,
            "expected_start": spec.expected_start.isoformat(),
            "minimum_observations": spec.minimum_observations,
            "max_latest_lag_days": spec.max_latest_lag_days,
            "partition_key": make_partition_key(
                spec.source_family,
                spec.native_series_id,
                spec.country,
                spec.indicator,
            ),
            "catalogue_semantic_sha256": catalogue_hash,
        }
        for spec in MONEY_LIQUIDITY_SERIES
    ]
    if observations is None:
        missing_series = [
            {
                **item,
                "status": "missing",
                "storage_status": "missing",
                "row_count": 0,
                "first_date": None,
                "latest_date": None,
                "latest_lag_days": None,
                "release_catalogue_semantic_sha256": None,
                "catalogue_semantic_status": None,
                "artifact_manifest": None,
                "artifact_manifest_status": None,
                "guard_failures": ["missing"],
            }
            for item in expected_money
        ]
        return {
            "commodities": empty_commodity,
            "shadow_liquidity": shadow_liquidity,
            "money_liquidity": {
                "expected_series": len(expected_money),
                "stored_series": 0,
                "ready_series": 0,
                "guard_failed_series": 0,
                "missing_series": len(expected_money),
                "coverage_pct": 0.0,
                "stored_coverage_pct": 0.0,
                "evaluated_as_of": as_of.isoformat(),
                "series": missing_series,
                "catalogue_semantic_sha256": catalogue_hash,
                "catalogue_semantic_mismatches": 0,
                "catalogue_semantic_unrecorded": 0,
                "artifact_manifest_invalid": 0,
            },
        }

    market_sources = {
        SOURCE_WORLD_BANK_COMMODITIES,
        *(spec.source_family for spec in MONEY_LIQUIDITY_SERIES),
    }
    rows = _execute_rows(
        connection,
        select(
            observations.c.source.label("source"),
            observations.c.country.label("country"),
            observations.c.indicator.label("indicator"),
            observations.c.series_id.label("series_id"),
            func.count().label("row_count"),
            func.min(observations.c.date).label("first_date"),
            func.max(observations.c.date).label("latest_date"),
        )
        .where(observations.c.source.in_(market_sources))
        .group_by(
            observations.c.source,
            observations.c.country,
            observations.c.indicator,
            observations.c.series_id,
        )
        .order_by(
            observations.c.source,
            observations.c.country,
            observations.c.indicator,
            observations.c.series_id,
        ),
    )

    commodity_series = [row for row in rows if row["source"] == SOURCE_WORLD_BANK_COMMODITIES]
    commodities = _commodity_inventory(
        connection,
        observations,
        releases,
        commodity_series,
        as_of=as_of,
    )

    stored_by_key = {
        (row["source"], row["country"], row["indicator"], row["series_id"]): row
        for row in rows
        if row["source"] != SOURCE_WORLD_BANK_COMMODITIES
    }
    money_sources = {spec.source_family for spec in MONEY_LIQUIDITY_SERIES}
    stored_dates_by_key: dict[tuple[str, str, str, str], list[date]] = defaultdict(list)
    for row in connection.execute(
        select(
            observations.c.source,
            observations.c.country,
            observations.c.indicator,
            observations.c.series_id,
            observations.c.date,
        )
        .where(observations.c.source.in_(money_sources))
        .order_by(
            observations.c.source,
            observations.c.country,
            observations.c.indicator,
            observations.c.series_id,
            observations.c.date,
        )
    ):
        observed_on = row.date
        if isinstance(observed_on, datetime):
            observed_on = observed_on.date()
        elif not isinstance(observed_on, date):
            observed_on = date.fromisoformat(str(observed_on))
        stored_dates_by_key[
            (str(row.source), str(row.country), str(row.indicator), str(row.series_id))
        ].append(observed_on)
    release_catalogue_hashes: dict[str, str | None] = {}
    latest_money_release_ids: dict[str, int] = {}
    partition_keys = tuple(item["partition_key"] for item in expected_money)
    if releases is not None and partition_keys:
        ranked = (
            select(
                releases.c.id.label("release_id"),
                releases.c.partition_key.label("partition_key"),
                releases.c.vintage_label.label("vintage_label"),
                func.row_number()
                .over(
                    partition_by=releases.c.partition_key,
                    order_by=(
                        releases.c.available_at.desc(),
                        releases.c.retrieved_at.desc(),
                        releases.c.id.desc(),
                    ),
                )
                .label("release_rank"),
            )
            .where(releases.c.partition_key.in_(partition_keys))
            .subquery()
        )
        for row in connection.execute(
            select(
                ranked.c.release_id,
                ranked.c.partition_key,
                ranked.c.vintage_label,
            ).where(ranked.c.release_rank == 1)
        ):
            label = row.vintage_label
            digest = None
            if isinstance(label, str) and label.startswith(
                MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX
            ):
                candidate = label.removeprefix(MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX)
                if len(candidate) == 64 and all(
                    character in "0123456789abcdef" for character in candidate
                ):
                    digest = candidate
            release_catalogue_hashes[str(row.partition_key)] = digest
            latest_money_release_ids[str(row.partition_key)] = int(row.release_id)

    money_series = []
    specs_by_series_id = {spec.native_series_id: spec for spec in MONEY_LIQUIDITY_SERIES}
    for item in expected_money:
        key = (
            item["source"],
            item["country"],
            item["indicator"],
            item["series_id"],
        )
        stored = stored_by_key.get(key)
        dates = stored_dates_by_key.get(key, [])
        release_catalogue_hash = release_catalogue_hashes.get(item["partition_key"])
        artifact_manifest = _release_artifact_manifest(
            connection,
            release_artifacts,
            release_id=latest_money_release_ids.get(item["partition_key"]),
            derived_payloads=True,
        )
        semantic_status = (
            None
            if not stored
            else "unrecorded"
            if release_catalogue_hash is None
            else "match"
            if release_catalogue_hash == catalogue_hash
            else "mismatch"
        )
        guard_failures = (
            _money_history_guard_failures(
                dates,
                specs_by_series_id[item["series_id"]],
                as_of=as_of,
                catalogue_semantic_status=semantic_status,
                artifact_manifest_status=artifact_manifest["status"],
            )
            if stored
            else ["missing"]
        )
        latest_lag_days = (as_of - dates[-1]).days if dates else None
        money_series.append(
            {
                **item,
                "status": (
                    "missing" if not stored else "guard_failed" if guard_failures else "ready"
                ),
                "storage_status": "stored" if stored else "missing",
                "row_count": int(stored["row_count"]) if stored else 0,
                "first_date": stored["first_date"] if stored else None,
                "latest_date": stored["latest_date"] if stored else None,
                "latest_lag_days": latest_lag_days,
                "release_catalogue_semantic_sha256": release_catalogue_hash,
                "catalogue_semantic_status": semantic_status,
                "artifact_manifest": artifact_manifest,
                "artifact_manifest_status": artifact_manifest["status"],
                "guard_failures": guard_failures,
            }
        )
    stored_money = sum(item["storage_status"] == "stored" for item in money_series)
    ready_money = sum(item["status"] == "ready" for item in money_series)
    guard_failed_money = sum(item["status"] == "guard_failed" for item in money_series)
    return {
        "commodities": commodities,
        "shadow_liquidity": shadow_liquidity,
        "money_liquidity": {
            "expected_series": len(expected_money),
            "stored_series": stored_money,
            "ready_series": ready_money,
            "guard_failed_series": guard_failed_money,
            "missing_series": len(expected_money) - stored_money,
            "coverage_pct": _coverage_pct(ready_money, len(expected_money)),
            "stored_coverage_pct": _coverage_pct(stored_money, len(expected_money)),
            "evaluated_as_of": as_of.isoformat(),
            "series": money_series,
            "catalogue_semantic_sha256": catalogue_hash,
            "catalogue_semantic_mismatches": sum(
                item["catalogue_semantic_status"] == "mismatch" for item in money_series
            ),
            "catalogue_semantic_unrecorded": sum(
                item["catalogue_semantic_status"] == "unrecorded" for item in money_series
            ),
            "artifact_manifest_invalid": sum(
                item["artifact_manifest_status"] not in {None, "valid"} for item in money_series
            ),
            "classification_note": (
                "local-currency levels and source-native frequencies remain separate; "
                "coverage does not imply cross-currency level comparability"
            ),
        },
    }


def _money_history_guard_failures(
    dates: list[date],
    spec: MoneyLiquiditySeries,
    *,
    as_of: date,
    catalogue_semantic_status: str | None,
    artifact_manifest_status: str | None,
) -> list[str]:
    """Return deterministic inventory failures for one present money series."""
    failures: list[str] = []
    if not dates:
        return ["missing_observations"]
    if dates[0] != spec.expected_start:
        failures.append("expected_start")
    if len(dates) < spec.minimum_observations:
        failures.append("minimum_observations")
    if len(dates) != len(set(dates)):
        failures.append("duplicate_dates")

    if spec.frequency == "monthly":
        ordinals = [observed_on.year * 12 + observed_on.month for observed_on in dates]
        if any(observed_on.day != 1 for observed_on in dates) or any(
            right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)
        ):
            failures.append("cadence")
    elif spec.frequency == "weekly":
        expected_weekday = spec.expected_start.weekday()
        if any(observed_on.weekday() != expected_weekday for observed_on in dates) or any(
            (right - left).days != 7 for left, right in zip(dates, dates[1:], strict=False)
        ):
            failures.append("cadence")
    elif spec.frequency == "quarterly":
        ordinals = [observed_on.year * 4 + (observed_on.month - 1) // 3 for observed_on in dates]
        if any(
            observed_on.day != 1 or observed_on.month not in {1, 4, 7, 10} for observed_on in dates
        ) or any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
            failures.append("cadence")
    else:
        failures.append("unsupported_frequency")

    latest_lag_days = (as_of - dates[-1]).days
    if latest_lag_days < 0:
        failures.append("future_latest_observation")
    elif latest_lag_days > spec.max_latest_lag_days:
        failures.append("latest_lag")
    if catalogue_semantic_status == "unrecorded":
        failures.append("catalogue_semantic_unrecorded")
    elif catalogue_semantic_status == "mismatch":
        failures.append("catalogue_semantic_mismatch")
    if artifact_manifest_status != "valid":
        failures.append("artifact_manifest_invalid")
    return failures


def _source_partitions(
    observation_inventory: dict[str, Any],
    source: str,
) -> list[dict[str, Any]]:
    return [
        {
            "country": row["country"],
            "indicator": row["indicator"],
            "row_count": row["row_count"],
            "first_date": row["first_date"],
            "latest_date": row["latest_date"],
        }
        for row in observation_inventory["partitions"]
        if row["source"] == source
    ]


def _qpsd_requested_pairs(
    connection: Connection,
    releases: Table | None,
) -> set[tuple[str, str]]:
    """Recover country/series requests evidenced by successful stored releases."""
    if releases is None:
        return set()
    urls = connection.scalars(
        select(releases.c.source_url)
        .where(
            releases.c.source_family == QPSD_SOURCE,
            releases.c.source_url.is_not(None),
        )
        .distinct()
    ).all()
    countries_by_wb = {
        str(country.wb_id).upper(): country.iso2 for country in QPSD_COUNTRIES if country.wb_id
    }
    indicators_by_code = {spec.wb_code: spec.indicator for spec in QPSD_SERIES}
    requested: set[tuple[str, str]] = set()
    for raw_url in urls:
        parts = [unquote(part) for part in urlparse(str(raw_url)).path.split("/") if part]
        try:
            country_index = parts.index("country") + 1
            indicator_index = parts.index("indicator") + 1
            country_codes = parts[country_index].split(";")
            indicator = indicators_by_code[parts[indicator_index]]
        except (KeyError, ValueError, IndexError):
            continue
        requested.update(
            (iso2, indicator)
            for code in country_codes
            if (iso2 := countries_by_wb.get(code.upper())) is not None
        )
    return requested


def _coverage_pct(stored: int, expected: int) -> float:
    return round(100 * stored / expected, 3) if expected else 100.0


def _qpsd_inventory(
    connection: Connection,
    observations: dict[str, Any],
    releases: Table | None,
) -> dict[str, Any]:
    stored = _source_partitions(observations, QPSD_SOURCE)
    stored_keys = {(row["country"], row["indicator"]) for row in stored}
    requested_keys = _qpsd_requested_pairs(connection, releases)
    specs = {spec.indicator: spec for spec in QPSD_SERIES}
    expected_keys = {
        (country.iso2, spec.indicator) for country in QPSD_COUNTRIES for spec in QPSD_SERIES
    }
    absent = expected_keys - stored_keys
    known_absent = absent & requested_keys
    unknown_absent = absent - requested_keys

    def gaps(keys: set[tuple[str, str]], status: str) -> list[dict[str, Any]]:
        return [
            {
                "country": country,
                "indicator": indicator,
                "series_id": specs[indicator].series_id,
                "status": status,
            }
            for country, indicator in sorted(keys)
        ]

    expected_count = len(expected_keys)
    return {
        "source": QPSD_SOURCE,
        "expected_countries": len(QPSD_COUNTRIES),
        "expected_indicators": len(QPSD_SERIES),
        "expected_partitions": expected_count,
        "stored_partitions": len(stored_keys),
        "not_reported_partitions": len(known_absent),
        "missing_partitions": len(unknown_absent),
        "coverage_pct": _coverage_pct(len(stored_keys), expected_count),
        "stored": stored,
        "not_reported": gaps(known_absent, "not_reported"),
        "missing": gaps(unknown_absent, "not_collected_or_unverified"),
        "classification_note": (
            "not_reported requires a successful official basket URL in the immutable "
            "release ledger; other absent cells remain unverified missing data"
        ),
    }


def _bop_inventory(observations: dict[str, Any]) -> dict[str, Any]:
    stored = _source_partitions(observations, BOP_SOURCE)
    stored_keys = {(row["country"], row["indicator"]) for row in stored}
    specs = {spec.indicator: spec for spec in BOP_SERIES}
    expected_keys = {
        (country.iso2, spec.indicator) for country in BOP_COUNTRIES for spec in BOP_SERIES
    }
    missing_keys = expected_keys - stored_keys
    missing = [
        {
            "country": country,
            "indicator": indicator,
            "series_id": specs[indicator].series_id,
            "status": "missing_or_not_reported",
        }
        for country, indicator in sorted(missing_keys)
    ]
    expected_count = len(expected_keys)
    return {
        "source": BOP_SOURCE,
        "expected_countries": len(BOP_COUNTRIES),
        "expected_indicators": len(BOP_SERIES),
        "expected_partitions": expected_count,
        "stored_partitions": len(stored_keys),
        "missing_partitions": len(missing_keys),
        "coverage_pct": _coverage_pct(len(stored_keys), expected_count),
        "stored": stored,
        "missing": missing,
        "classification_note": (
            "the current schema does not persist successful empty BOP responses, so an "
            "absent partition is missing-or-not-reported, never zero"
        ),
    }


def _execute_rows(connection: Connection, statement) -> list[dict[str, Any]]:
    return _dict_rows(connection.execute(statement).all())


def _debt_holder_inventory(
    connection: Connection,
    table: Table | None,
    latest_release_ids,
) -> dict[str, Any]:
    if table is None:
        return {
            "table_present": False,
            "row_count": None,
            "current_row_count": None,
            "countries": [],
            "first_date": None,
            "latest_date": None,
            "issuer_sectors": [],
            "instruments": [],
            "holder_sectors": [],
            "measures": [],
            "units": [],
        }
    condition = _current_condition(table, latest_release_ids)
    count_statement = select(func.count()).select_from(table)
    if condition is not None:
        count_statement = count_statement.where(condition)
    current_row_count = (
        int(connection.scalar(count_statement) or 0) if condition is not None else None
    )
    date_statement = select(
        func.min(table.c.date).label("first_date"),
        func.max(table.c.date).label("latest_date"),
    )
    countries_statement = select(
        table.c.country.label("country"),
        func.count().label("row_count"),
        func.min(table.c.date).label("first_date"),
        func.max(table.c.date).label("latest_date"),
    ).group_by(table.c.country)
    if condition is not None:
        date_statement = date_statement.where(condition)
        countries_statement = countries_statement.where(condition)
    date_row = connection.execute(date_statement).one()

    def coded(code: str, label: str) -> list[dict[str, Any]]:
        statement = select(table.c[code].label("code"), table.c[label].label("label")).distinct()
        if condition is not None:
            statement = statement.where(condition)
        return _execute_rows(connection, statement.order_by(table.c[code], table.c[label]))

    units_statement = select(table.c.unit).distinct().order_by(table.c.unit)
    if condition is not None:
        units_statement = units_statement.where(condition)
    return {
        "table_present": True,
        "row_count": _count(connection, table),
        "current_row_count": current_row_count,
        "countries": _execute_rows(connection, countries_statement.order_by(table.c.country)),
        "first_date": _json_value(date_row.first_date),
        "latest_date": _json_value(date_row.latest_date),
        "issuer_sectors": coded("issuer_sector_code", "issuer_sector_label"),
        "instruments": coded("instrument_code", "instrument_label"),
        "holder_sectors": coded("holder_sector_code", "holder_sector_label"),
        "measures": coded("measure_code", "measure_label"),
        "units": [str(value) for value in connection.scalars(units_statement).all()],
    }


def _allocator_inventory(
    connection: Connection,
    table: Table | None,
    latest_release_ids,
) -> dict[str, Any]:
    if table is None:
        return {
            "table_present": False,
            "row_count": None,
            "current_row_count": None,
            "funds": [],
        }
    condition = _current_condition(table, latest_release_ids)
    count_statement = select(func.count()).select_from(table)
    if condition is not None:
        count_statement = count_statement.where(condition)
    current_row_count = (
        int(connection.scalar(count_statement) or 0) if condition is not None else None
    )
    fund_statement = (
        select(
            table.c.fund.label("fund"),
            func.count().label("row_count"),
            func.min(table.c.as_of_date).label("first_as_of_date"),
            func.max(table.c.as_of_date).label("latest_as_of_date"),
            func.count(func.distinct(table.c.artifact_sha256)).label("artifact_count"),
        )
        .group_by(table.c.fund)
        .order_by(table.c.fund)
    )
    if condition is not None:
        fund_statement = fund_statement.where(condition)
    funds = _execute_rows(connection, fund_statement)

    record_types: dict[str, list[str]] = defaultdict(list)
    quality_flags: dict[str, list[str]] = defaultdict(list)
    for field, target in (("record_type", record_types), ("quality_flag", quality_flags)):
        statement = select(table.c.fund, table.c[field]).distinct()
        if condition is not None:
            statement = statement.where(condition)
        statement = statement.order_by(table.c.fund, table.c[field])
        for fund, value in connection.execute(statement):
            target[str(fund)].append(str(value))
    for fund in funds:
        fund["record_types"] = record_types[fund["fund"]]
        fund["quality_flags"] = quality_flags[fund["fund"]]
        # Keep the stable, human-facing field order used by JSON snapshots.
        artifact_count = fund.pop("artifact_count")
        record_type_values = fund.pop("record_types")
        quality_flag_values = fund.pop("quality_flags")
        fund["record_types"] = record_type_values
        fund["artifact_count"] = artifact_count
        fund["quality_flags"] = quality_flag_values
    return {
        "table_present": True,
        "row_count": _count(connection, table),
        "current_row_count": current_row_count,
        "funds": funds,
    }


def _reports_inventory(
    connection: Connection,
    tables: dict[str, Table],
) -> dict[str, Any]:
    documents = tables.get("report_documents")
    extractions = tables.get("document_extractions")
    pages = tables.get("document_pages")
    claims = tables.get("claims")
    citations = tables.get("claim_citations")
    reviews = tables.get("report_candidate_reviews")
    document_groups: list[dict[str, Any]] = []
    if documents is not None:
        document_groups = _execute_rows(
            connection,
            select(
                documents.c.publisher.label("publisher"),
                documents.c.report_family.label("report_family"),
                func.count().label("document_count"),
                func.min(documents.c.document_date).label("first_document_date"),
                func.max(documents.c.document_date).label("latest_document_date"),
                func.sum(documents.c.page_count).label("page_count"),
            )
            .group_by(documents.c.publisher, documents.c.report_family)
            .order_by(documents.c.publisher, documents.c.report_family),
        )

    def counts_by(table: Table | None, field: str) -> list[dict[str, Any]]:
        if table is None:
            return []
        return _execute_rows(
            connection,
            select(table.c[field].label(field), func.count().label("row_count"))
            .group_by(table.c[field])
            .order_by(table.c[field]),
        )

    return {
        "document_count": _count(connection, documents),
        "extraction_count": _count(connection, extractions),
        "page_count": _count(connection, pages),
        "claim_count": _count(connection, claims),
        "citation_count": _count(connection, citations),
        "review_count": _count(connection, reviews),
        "documents": document_groups,
        "extraction_statuses": counts_by(extractions, "status"),
        "claim_statuses": counts_by(claims, "status"),
        "claim_types": counts_by(claims, "claim_type"),
        "review_outcomes": counts_by(reviews, "outcome"),
    }


def _communication_corpus_sha256(segments: list[dict[str, Any]]) -> str:
    """Hash the exact, versioned communication-segment canonical form."""
    payload = {
        "canonicalization": _COMMUNICATION_CANONICALIZATION,
        "segments": [
            {field: segment[field] for field in _COMMUNICATION_CANONICAL_SEGMENT_FIELDS}
            for segment in segments
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _communication_policy_payload(row: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the exact semantic object hashed by the storage helper."""

    def json_string_list(field: str) -> list[str]:
        value = json.loads(row[field])
        if (
            not isinstance(value, list)
            or any(not isinstance(item, str) for item in value)
            or value != sorted(set(value))
        ):
            raise ValueError(f"{field} must be a sorted, duplicate-free string array")
        return value

    def utc_text(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, datetime):
            raise ValueError("policy clocks must be datetimes")
        normalized = value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)
        return normalized.isoformat().replace("+00:00", "Z")

    return {
        "source_id": row["source_id"],
        "organization_id": row["organization_id"],
        "organization_name": row["organization_name"],
        "organization_type": row["organization_type"],
        "jurisdiction": row["jurisdiction"],
        "language": row["language"],
        "landing_url": row["landing_url"],
        "official_domains": json_string_list("official_domains_json"),
        "host_organization": row["host_organization"],
        "publisher": row["publisher"],
        "transcriber": row["transcriber"],
        "transcriber_attribution": row["transcriber_attribution"],
        "material_types": json_string_list("material_types_json"),
        "commodity_families": json_string_list("commodity_families_json"),
        "verified_archive_start_year": row["verified_archive_start_year"],
        "coverage_note": row["coverage_note"],
        "provenance_tier": row["source_provenance_tier"],
        "rights_status": row["rights_status"],
        "rights_basis_url": row["rights_basis_url"],
        "rights_note": row["rights_note"],
        "acquisition_status": row["acquisition_status"],
        "acquisition_note": row["acquisition_note"],
        "automated_collection_allowed": bool(row["automated_collection_allowed"]),
        "rights_checked_by": row["rights_checked_by"],
        "rights_checked_at": utc_text(row["rights_checked_at"]),
        "catalogue_sha256": row["catalogue_sha256"],
        "catalogue_evaluated_at": utc_text(row["catalogue_evaluated_at"]),
    }


def _canonical_dict_sha256(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _communication_policy_semantically_valid(
    row: dict[str, Any],
    payload: dict[str, Any],
) -> bool:
    identifier = re.compile(r"^[a-z][a-z0-9_]*$")
    materials = set(payload["material_types"])
    has_transcript = bool(materials & _COMMUNICATION_TRANSCRIPT_MATERIAL_TYPES)
    attribution = row["transcriber_attribution"]
    transcriber = row["transcriber"]
    publisher = row["publisher"]
    provenance = row["source_provenance_tier"]
    reviewer = row["rights_checked_by"]
    checked_at = row["rights_checked_at"]
    evaluated_at = row["catalogue_evaluated_at"]
    rights_status = row["rights_status"]

    transcriber_valid = (
        (
            attribution in {"artifact_specific", "not_applicable", "not_disclosed"}
            and transcriber is None
        )
        or (attribution == "publisher" and transcriber == publisher)
        or (
            attribution == "named_third_party"
            and isinstance(transcriber, str)
            and bool(transcriber.strip())
            and transcriber != publisher
        )
    )
    provenance_valid = (
        provenance
        in {
            "official_archive_mixed",
            "official_authored_text",
            "official_published_transcript",
            "official_hosted_third_party",
        }
        and not (provenance == "official_authored_text" and has_transcript)
        and not (provenance == "official_published_transcript" and not has_transcript)
        and not (provenance == "official_hosted_third_party" and attribution != "named_third_party")
        and not (attribution == "named_third_party" and provenance != "official_hosted_third_party")
    )
    reviewer_pair = (reviewer is None) == (checked_at is None)
    reviewer_valid = reviewer_pair and (
        reviewer is None
        or (
            isinstance(reviewer, str)
            and reviewer.startswith("human:")
            and reviewer != "human:"
            and isinstance(checked_at, datetime)
            and isinstance(evaluated_at, datetime)
            and checked_at <= evaluated_at
        )
    )
    archive_year = row["verified_archive_start_year"]
    archive_year_valid = archive_year is None or (
        isinstance(archive_year, int)
        and not isinstance(archive_year, bool)
        and isinstance(evaluated_at, datetime)
        and 1900 <= archive_year <= evaluated_at.year
    )
    required_text_fields = (
        "organization_name",
        "host_organization",
        "publisher",
        "coverage_note",
        "rights_note",
        "acquisition_note",
    )
    return all(
        (
            isinstance(row["source_id"], str),
            identifier.fullmatch(row["source_id"]) is not None,
            len(row["source_id"]) <= 96,
            isinstance(row["organization_id"], str),
            identifier.fullmatch(row["organization_id"]) is not None,
            len(row["organization_id"]) <= 96,
            row["organization_type"] in {"central_bank", "bank", "commodity_company"},
            isinstance(row["jurisdiction"], str),
            re.fullmatch(r"[A-Z]{2,3}", row["jurisdiction"]) is not None,
            isinstance(row["language"], str),
            re.fullmatch(r"[a-z]{2,3}", row["language"]) is not None,
            all(
                isinstance(row[field], str) and bool(row[field].strip())
                for field in required_text_fields
            ),
            bool(materials),
            (has_transcript and attribution != "not_applicable")
            or (not has_transcript and attribution == "not_applicable"),
            transcriber_valid,
            provenance_valid,
            rights_status in _COMMUNICATION_RIGHTS_TO_ACQUISITION,
            row["acquisition_status"] == _COMMUNICATION_RIGHTS_TO_ACQUISITION.get(rights_status),
            row["automated_collection_allowed"] is False,
            reviewer_valid,
            rights_status not in {"cleared", "internal_only"} or reviewer is not None,
            rights_status
            not in {"cleared", "internal_only", "metadata_only", "permission_required"}
            or row["rights_basis_url"] is not None,
            isinstance(evaluated_at, datetime),
            archive_year_valid,
        )
    )


def _communication_artifact_semantically_valid(row: dict[str, Any]) -> bool:
    role_materials = _COMMUNICATION_ROLE_MATERIALS.get(row["artifact_role"], frozenset())
    material_origins = _COMMUNICATION_MATERIAL_ORIGINS.get(row["material_type"], frozenset())
    expected_provenance = _COMMUNICATION_ORIGIN_PROVENANCE.get(row["origin_type"])
    transcriber = row["transcriber"]
    attribution = row["transcriber_attribution"]
    publisher = row["publisher"]
    if row["origin_type"] in {"publisher_authored", "official_published_media"}:
        transcriber_valid = transcriber is None and attribution == "not_applicable"
    elif row["origin_type"] in {"official_hosted_vendor", "automatic_caption", "local_asr"}:
        transcriber_valid = (
            isinstance(transcriber, str)
            and bool(transcriber.strip())
            and transcriber != publisher
            and attribution == "named_third_party"
        )
    elif row["origin_type"] in {"official_published_transcript", "official_caption"}:
        transcriber_valid = (
            (transcriber is None and attribution == "not_disclosed")
            or (transcriber == publisher and attribution == "publisher")
            or (
                isinstance(transcriber, str)
                and bool(transcriber.strip())
                and transcriber != publisher
                and attribution == "named_third_party"
            )
        )
    else:
        transcriber_valid = False
    identifier = re.compile(r"^[a-z][a-z0-9_]*$")
    return all(
        (
            isinstance(row["artifact_key"], str),
            identifier.fullmatch(row["artifact_key"]) is not None,
            len(row["artifact_key"]) <= 128,
            row["material_type"] in role_materials,
            row["origin_type"] in material_origins,
            row["provenance_tier"] == expected_provenance,
            row["translation_status"] in {"original", "official_translation"},
            transcriber_valid,
            row["rights_status"] in _COMMUNICATION_RIGHTS_TO_ACQUISITION,
            row["acquisition_status"]
            == _COMMUNICATION_RIGHTS_TO_ACQUISITION.get(row["rights_status"]),
            isinstance(row["rights_checked_by"], str),
            row["rights_checked_by"].startswith("human:"),
            row["rights_checked_by"] != "human:",
            all(
                isinstance(row[field], str) and bool(row[field].strip())
                for field in (
                    "language",
                    "mime_type",
                    "host_organization",
                    "publisher",
                    "rights_note",
                )
            ),
        )
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _communication_ddl_sha256(
    connection: Connection,
    object_types: tuple[str, ...],
) -> str:
    table_placeholders = ",".join("?" for _ in _COMMUNICATION_TABLE_NAMES)
    type_placeholders = ",".join("?" for _ in object_types)
    rows = connection.exec_driver_sql(
        "SELECT type, name, tbl_name, sql FROM sqlite_master "
        f"WHERE tbl_name IN ({table_placeholders}) "
        f"AND type IN ({type_placeholders}) AND sql IS NOT NULL "
        "ORDER BY type, name, tbl_name",
        (*_COMMUNICATION_TABLE_NAMES, *object_types),
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
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _communication_url_matches_domains(value: Any, domains: list[str]) -> bool:
    if not isinstance(value, str) or not value or value != value.strip():
        return False
    if "\\" in value or any(ord(character) < 33 or ord(character) == 127 for character in value):
        return False
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        return False
    try:
        if parsed.port is not None:
            return False
    except ValueError:
        return False
    host = parsed.hostname.lower()
    if host.endswith(".") or host == "localhost":
        return False
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return False
    dns_label = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
    if any(dns_label.fullmatch(label) is None for label in host.split(".")):
        return False
    return any(host == domain or host.endswith(f".{domain}") for domain in domains)


def _communication_iso(value: Any) -> str:
    if not isinstance(value, (date, datetime)):
        raise ValueError("communication semantic clock must be a date or datetime")
    return value.isoformat()


def _communication_lineage_mismatch_count(
    rows: list[dict[str, Any]],
    *,
    id_field: str,
    predecessor_field: str,
    identity_fields: tuple[str, ...],
    clock_field: str,
) -> int:
    """Count broken roots, branches, identity/clock links, and disconnected cycles."""
    by_id = {int(row[id_field]): row for row in rows}
    by_identity: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_identity[tuple(row[field] for field in identity_fields)].append(row)

    mismatches = 0
    successor_counts: dict[int, int] = defaultdict(int)
    for row in rows:
        predecessor_id = row[predecessor_field]
        if predecessor_id is not None:
            successor_counts[int(predecessor_id)] += 1
    mismatches += sum(count > 1 for count in successor_counts.values())

    for identity, identity_rows in by_identity.items():
        roots = [row for row in identity_rows if row[predecessor_field] is None]
        mismatches += len(roots) != 1
        for row in identity_rows:
            predecessor_id = row[predecessor_field]
            if predecessor_id is None:
                continue
            predecessor = by_id.get(int(predecessor_id))
            try:
                forward_clock = (
                    predecessor is not None and row[clock_field] > predecessor[clock_field]
                )
            except TypeError:
                forward_clock = False
            if (
                predecessor is None
                or tuple(predecessor[field] for field in identity_fields) != identity
                or not forward_clock
            ):
                mismatches += 1

        for row in identity_rows:
            seen: set[int] = set()
            cursor = row
            while cursor[predecessor_field] is not None:
                cursor_id = int(cursor[id_field])
                if cursor_id in seen:
                    mismatches += 1
                    break
                seen.add(cursor_id)
                predecessor = by_id.get(int(cursor[predecessor_field]))
                if predecessor is None:
                    break
                cursor = predecessor
    return mismatches


def _communication_integrity_inventory(
    connection: Connection,
    tables: dict[str, Table],
    *,
    catalogue: dict[str, Any],
) -> dict[str, Any]:
    """Re-verify the stored communications corpus instead of trusting row headers."""
    table_presence = {name: name in tables for name in _COMMUNICATION_TABLE_NAMES}
    missing_columns = {
        name: sorted(required - set(tables[name].c.keys()))
        for name, required in _COMMUNICATION_REQUIRED_COLUMNS.items()
        if name in tables and not required.issubset(tables[name].c.keys())
    }

    trigger_check_status = "unsupported"
    missing_triggers = sorted(_COMMUNICATION_REQUIRED_TRIGGERS)
    invalid_triggers: list[str] = []
    trigger_check_error: str | None = None
    foreign_key_check_status = "unsupported"
    foreign_key_violations: list[dict[str, Any]] = []
    foreign_key_check_error: str | None = None
    if connection.dialect.name == "sqlite":
        try:
            stored_trigger_sql = {
                str(row[0]): str(row[1] or "")
                for row in connection.exec_driver_sql(
                    "SELECT name, sql FROM sqlite_master WHERE type = 'trigger'"
                )
            }
            stored_triggers = set(stored_trigger_sql)
            missing_triggers = sorted(_COMMUNICATION_REQUIRED_TRIGGERS - stored_triggers)
            for trigger_name in sorted(_COMMUNICATION_REQUIRED_TRIGGERS & stored_triggers):
                normalized_sql = re.sub(
                    r"\s+", " ", stored_trigger_sql[trigger_name].lower()
                ).strip()
                fragments = _COMMUNICATION_CUSTOM_TRIGGER_FRAGMENTS.get(trigger_name)
                if fragments is None:
                    operation = "update" if trigger_name.endswith("_reject_update") else "delete"
                    table_name = trigger_name.removesuffix(f"_reject_{operation}")
                    fragments = (
                        f"before {operation} on {table_name}",
                        "select raise(abort",
                    )
                if any(fragment not in normalized_sql for fragment in fragments):
                    invalid_triggers.append(trigger_name)
            trigger_check_status = (
                "valid" if not missing_triggers and not invalid_triggers else "invalid"
            )
        except Exception as exc:  # pragma: no cover - corrupt SQLite catalogue defence
            trigger_check_status = "error"
            trigger_check_error = f"{type(exc).__name__}: {exc}"
        try:
            for row in connection.exec_driver_sql("PRAGMA foreign_key_check"):
                if str(row[0]) not in _COMMUNICATION_TABLE_NAMES:
                    continue
                foreign_key_violations.append(
                    {
                        "table": str(row[0]),
                        "row_id": row[1],
                        "parent_table": str(row[2]),
                        "foreign_key_index": int(row[3]),
                    }
                )
            foreign_key_check_status = "valid" if not foreign_key_violations else "invalid"
        except Exception as exc:  # pragma: no cover - corrupt SQLite catalogue defence
            foreign_key_check_status = "error"
            foreign_key_check_error = f"{type(exc).__name__}: {exc}"

    result: dict[str, Any] = {
        "missing_required_columns": missing_columns,
        "trigger_check_status": trigger_check_status,
        "missing_required_triggers": missing_triggers,
        "invalid_required_triggers": invalid_triggers,
        "trigger_check_error": trigger_check_error,
        "foreign_key_check_status": foreign_key_check_status,
        "foreign_key_violation_count": len(foreign_key_violations),
        "foreign_key_violations": foreign_key_violations,
        "foreign_key_check_error": foreign_key_check_error,
        "schema_contract_status": "not_checked",
        "schema_contract_row_count": None,
        "schema_contract_version": None,
        "expected_schema_contract_version": None,
        "stored_schema_sha256": None,
        "stored_trigger_sha256": None,
        "actual_schema_sha256": None,
        "actual_trigger_sha256": None,
        "expected_schema_sha256": None,
        "expected_trigger_sha256": None,
        "policy_snapshot_json_malformed_count": None,
        "policy_snapshot_semantic_mismatch_count": None,
        "policy_snapshot_sha256_mismatch_count": None,
        "current_catalogue_snapshot_check_status": "not_checked",
        "current_catalogue_snapshot_mismatch_count": None,
        "untrusted_catalogue_hash_count": None,
        "untrusted_catalogue_hashes": [],
        "event_version_sha256_mismatch_count": None,
        "artifact_version_sha256_mismatch_count": None,
        "artifact_semantic_mismatch_count": None,
        "coverage_version_sha256_mismatch_count": None,
        "event_lineage_mismatch_count": None,
        "artifact_lineage_mismatch_count": None,
        "coverage_lineage_mismatch_count": None,
        "content_lineage_mismatch_count": None,
        "communication_clock_mismatch_count": None,
        "artifact_policy_binding_mismatch_count": None,
        "commodity_policy_binding_mismatch_count": None,
        "policy_binding_mismatch_count": None,
        "missing_base_artifact_retrieval_count": None,
        "orphan_artifact_retrieval_count": None,
        "artifact_retrieval_mismatch_count": None,
        "content_retrieval_binding_mismatch_count": None,
        "archived_blob_path_mismatch_count": None,
        "archived_blob_missing_count": None,
        "archived_blob_size_mismatch_count": None,
        "archived_blob_sha256_mismatch_count": None,
        "archived_blob_read_error_count": None,
        "verified_archived_blob_count": None,
        "extraction_content_binding_mismatch_count": None,
        "incomplete_extraction_count": None,
        "finalized_segment_count": None,
        "unfinalized_segment_count": None,
        "segment_text_sha256_mismatch_count": None,
        "segment_char_count_mismatch_count": None,
        "finalization_structure_mismatch_count": None,
        "unsupported_canonicalization_count": None,
        "corpus_sha256_mismatch_count": None,
        "canonicalization_error_count": None,
        "valid_finalized_extraction_count": None,
        "integrity_failures": [],
        "integrity_status": "table_absent",
    }
    if not all(table_presence.values()):
        return result
    if missing_columns:
        result["integrity_failures"] = ["required_columns"]
        result["integrity_status"] = "invalid"
        return result

    contract_status = "error"
    try:
        from dalio.storage.db import (
            COMMUNICATION_SCHEMA_SHA256,
            COMMUNICATION_SCHEMA_VERSION,
            COMMUNICATION_TRIGGER_SHA256,
        )

        contract = tables["communication_schema_contract"]
        contract_rows = (
            connection.execute(
                select(
                    contract.c.contract_id,
                    contract.c.schema_version,
                    contract.c.schema_sha256,
                    contract.c.trigger_sha256,
                )
            )
            .mappings()
            .all()
        )
        actual_schema_sha256 = _communication_ddl_sha256(connection, ("table", "index"))
        actual_trigger_sha256 = _communication_ddl_sha256(connection, ("trigger",))
        contract_row = dict(contract_rows[0]) if len(contract_rows) == 1 else None
        contract_status = (
            "valid"
            if (
                contract_row is not None
                and contract_row["contract_id"] == "institutional_communications"
                and int(contract_row["schema_version"]) == COMMUNICATION_SCHEMA_VERSION
                and contract_row["schema_sha256"] == COMMUNICATION_SCHEMA_SHA256
                and contract_row["trigger_sha256"] == COMMUNICATION_TRIGGER_SHA256
                and actual_schema_sha256 == COMMUNICATION_SCHEMA_SHA256
                and actual_trigger_sha256 == COMMUNICATION_TRIGGER_SHA256
            )
            else "invalid"
        )
        result.update(
            {
                "schema_contract_status": contract_status,
                "schema_contract_row_count": len(contract_rows),
                "schema_contract_version": (
                    contract_row["schema_version"] if contract_row is not None else None
                ),
                "expected_schema_contract_version": COMMUNICATION_SCHEMA_VERSION,
                "stored_schema_sha256": (
                    contract_row["schema_sha256"] if contract_row is not None else None
                ),
                "stored_trigger_sha256": (
                    contract_row["trigger_sha256"] if contract_row is not None else None
                ),
                "actual_schema_sha256": actual_schema_sha256,
                "actual_trigger_sha256": actual_trigger_sha256,
                "expected_schema_sha256": COMMUNICATION_SCHEMA_SHA256,
                "expected_trigger_sha256": COMMUNICATION_TRIGGER_SHA256,
            }
        )
    except Exception as exc:  # pragma: no cover - corrupt SQLite catalogue defence
        result["schema_contract_status"] = f"error: {type(exc).__name__}: {exc}"

    policies = tables["communication_source_policy_snapshots"]
    coverage = tables["organization_commodity_coverage"]
    events = tables["communication_events"]
    artifacts = tables["communication_artifacts"]
    extractions = tables["communication_extractions"]
    segments = tables["communication_segments"]
    finalizations = tables["communication_extraction_finalizations"]

    policy_rows = [
        dict(row)
        for row in connection.execute(
            select(
                *(
                    policies.c[field]
                    for field in sorted(
                        _COMMUNICATION_REQUIRED_COLUMNS["communication_source_policy_snapshots"]
                    )
                )
            )
        ).mappings()
    ]
    policy_by_key = {
        (str(row["catalogue_sha256"]), str(row["source_id"])): row for row in policy_rows
    }
    malformed_policy_json_count = 0
    policy_semantic_mismatch_count = 0
    policy_hash_mismatch_count = 0
    policy_materials: dict[tuple[str, str], list[str]] = {}
    policy_commodity_families: dict[tuple[str, str], list[str]] = {}
    policy_domains: dict[tuple[str, str], list[str]] = {}
    for policy in policy_rows:
        key = (str(policy["catalogue_sha256"]), str(policy["source_id"]))
        try:
            payload = _communication_policy_payload(policy)
            expected_hash = _canonical_dict_sha256(payload)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, UnicodeEncodeError):
            malformed_policy_json_count += 1
            continue
        policy_materials[key] = payload["material_types"]
        policy_commodity_families[key] = payload["commodity_families"]
        policy_domains[key] = payload["official_domains"]
        policy_hash_mismatch_count += expected_hash != policy["policy_sha256"]
        is_commodity_company = policy["organization_type"] == "commodity_company"
        if (
            not _communication_policy_semantically_valid(policy, payload)
            or not payload["official_domains"]
            or any(
                not _communication_url_matches_domains(f"https://{domain}/", [domain])
                for domain in payload["official_domains"]
            )
            or not _communication_url_matches_domains(
                policy["landing_url"], payload["official_domains"]
            )
            or (
                policy["rights_basis_url"] is not None
                and not _communication_url_matches_domains(
                    policy["rights_basis_url"], payload["official_domains"]
                )
            )
            or not set(payload["material_types"]).issubset(_COMMUNICATION_MATERIAL_TYPES)
            or not set(payload["commodity_families"]).issubset(_COMMUNICATION_COMMODITY_FAMILIES)
            or (is_commodity_company != bool(payload["commodity_families"]))
        ):
            policy_semantic_mismatch_count += 1

    current_snapshot_check_status = "not_checked"
    current_snapshot_mismatch_count: int | None = None
    if catalogue["validation_status"] == "valid":
        try:
            from dalio.communications.catalogue import COMMUNICATION_SOURCES
            from dalio.storage.communications import _source_policy_values

            expected_current = {
                (catalogue["sha256"], source.source_id): _source_policy_values(source)
                for source in COMMUNICATION_SOURCES
            }
            current_snapshot_mismatch_count = 0
            for key, policy in policy_by_key.items():
                if key[0] != catalogue["sha256"]:
                    continue
                expected = expected_current.get(key)
                if expected is None or any(
                    policy[field] != expected_value for field, expected_value in expected.items()
                ):
                    current_snapshot_mismatch_count += 1
            current_snapshot_check_status = (
                "valid" if current_snapshot_mismatch_count == 0 else "invalid"
            )
        except Exception as exc:  # pragma: no cover - optional catalogue defence
            current_snapshot_check_status = f"error: {type(exc).__name__}: {exc}"
    event_rows = [
        dict(row)
        for row in connection.execute(
            select(
                events.c.id,
                events.c.organization_id,
                events.c.event_key,
                events.c.event_type,
                events.c.title,
                events.c.event_date,
                events.c.event_started_at,
                events.c.reference_start,
                events.c.reference_end,
                events.c.metadata_known_at,
                events.c.event_version_sha256,
                events.c.supersedes_event_id,
            )
        ).mappings()
    ]
    event_by_id = {int(row["id"]): row for row in event_rows}
    event_hash_mismatch_count = 0
    for event in event_rows:
        try:
            expected_event_hash = _canonical_dict_sha256(
                {
                    "organization_id": event["organization_id"],
                    "event_key": event["event_key"],
                    "event_type": event["event_type"],
                    "title": event["title"],
                    "event_date": _communication_iso(event["event_date"]),
                    "event_started_at": (
                        _communication_iso(event["event_started_at"])
                        if event["event_started_at"] is not None
                        else None
                    ),
                    "reference_start": (
                        _communication_iso(event["reference_start"])
                        if event["reference_start"] is not None
                        else None
                    ),
                    "reference_end": (
                        _communication_iso(event["reference_end"])
                        if event["reference_end"] is not None
                        else None
                    ),
                }
            )
        except (TypeError, ValueError, UnicodeEncodeError):
            expected_event_hash = None
        event_hash_mismatch_count += expected_event_hash != event["event_version_sha256"]
    event_lineage_mismatch_count = _communication_lineage_mismatch_count(
        event_rows,
        id_field="id",
        predecessor_field="supersedes_event_id",
        identity_fields=("organization_id", "event_key"),
        clock_field="metadata_known_at",
    )
    artifact_rows = [
        dict(row)
        for row in connection.execute(
            select(
                artifacts.c.id,
                artifacts.c.event_id,
                artifacts.c.catalogue_sha256,
                artifacts.c.source_id,
                artifacts.c.event_version_sha256,
                artifacts.c.artifact_key,
                artifacts.c.artifact_role,
                artifacts.c.material_type,
                artifacts.c.language,
                artifacts.c.translation_status,
                artifacts.c.mime_type,
                artifacts.c.origin_type,
                artifacts.c.provenance_tier,
                artifacts.c.rights_status,
                artifacts.c.rights_basis_url,
                artifacts.c.rights_note,
                artifacts.c.acquisition_status,
                artifacts.c.rights_checked_by,
                artifacts.c.rights_checked_at,
                artifacts.c.host_organization,
                artifacts.c.publisher,
                artifacts.c.transcriber,
                artifacts.c.transcriber_attribution,
                artifacts.c.published_at,
                artifacts.c.available_at,
                artifacts.c.retrieved_at,
                artifacts.c.metadata_known_at,
                artifacts.c.landing_url,
                artifacts.c.artifact_url,
                artifacts.c.artifact_version_sha256,
                artifacts.c.supersedes_artifact_id,
            )
        ).mappings()
    ]
    artifact_by_id = {int(row["id"]): row for row in artifact_rows}

    retrieval_table = tables["communication_artifact_retrievals"]
    retrieval_rows = [
        dict(row)
        for row in connection.execute(
            select(
                retrieval_table.c.id,
                retrieval_table.c.artifact_id,
                retrieval_table.c.retrieved_at,
                retrieval_table.c.metadata_known_at,
                retrieval_table.c.landing_url,
                retrieval_table.c.artifact_url,
            )
        ).mappings()
    ]
    exact_retrieval_artifact_ids: set[int] = set()
    bad_retrieval_ids: set[int] = set()
    orphan_retrieval_count = 0
    retrieval_mismatch_count = 0
    clock_mismatch_count = 0
    for retrieval in retrieval_rows:
        artifact_id = int(retrieval["artifact_id"])
        artifact = artifact_by_id.get(artifact_id)
        if artifact is None:
            orphan_retrieval_count += 1
            bad_retrieval_ids.add(int(retrieval["id"]))
            continue
        if (
            retrieval["landing_url"] != artifact["landing_url"]
            or retrieval["artifact_url"] != artifact["artifact_url"]
            or retrieval["retrieved_at"] < artifact["available_at"]
            or retrieval["metadata_known_at"] < retrieval["retrieved_at"]
            or retrieval["metadata_known_at"] < artifact["metadata_known_at"]
        ):
            retrieval_mismatch_count += 1
            bad_retrieval_ids.add(int(retrieval["id"]))
            clock_mismatch_count += (
                retrieval["metadata_known_at"] < retrieval["retrieved_at"]
                or retrieval["metadata_known_at"] < artifact["metadata_known_at"]
                or retrieval["retrieved_at"] < artifact["available_at"]
            )
            continue
        if (
            retrieval["retrieved_at"] == artifact["retrieved_at"]
            and retrieval["metadata_known_at"] == artifact["metadata_known_at"]
        ):
            exact_retrieval_artifact_ids.add(artifact_id)
    missing_base_retrieval_count = len(set(artifact_by_id) - exact_retrieval_artifact_ids)

    artifact_policy_mismatches = 0
    artifact_hash_mismatch_count = 0
    artifact_semantic_mismatch_count = 0
    for artifact in artifact_rows:
        event = event_by_id.get(int(artifact["event_id"]))
        policy_key = (str(artifact["catalogue_sha256"]), str(artifact["source_id"]))
        policy = policy_by_key.get(policy_key)
        materials = policy_materials.get(policy_key, [])
        domains = policy_domains.get(policy_key, [])
        if (
            event is None
            or policy is None
            or policy["organization_id"] != event["organization_id"]
            or policy["rights_status"] != artifact["rights_status"]
            or policy["rights_basis_url"] != artifact["rights_basis_url"]
            or policy["rights_note"] != artifact["rights_note"]
            or policy["acquisition_status"] != artifact["acquisition_status"]
            or policy["language"] != artifact["language"]
            or artifact["material_type"] not in materials
            or not _communication_url_matches_domains(artifact["landing_url"], domains)
            or not _communication_url_matches_domains(artifact["artifact_url"], domains)
            or event["event_version_sha256"] != artifact["event_version_sha256"]
        ):
            artifact_policy_mismatches += 1
        artifact_semantic_mismatch_count += not _communication_artifact_semantically_valid(artifact)

        try:
            expected_artifact_hash = _canonical_dict_sha256(
                {
                    "event_version_sha256": artifact["event_version_sha256"],
                    "source_id": artifact["source_id"],
                    "catalogue_sha256": artifact["catalogue_sha256"],
                    "artifact_key": artifact["artifact_key"],
                    "artifact_role": artifact["artifact_role"],
                    "material_type": artifact["material_type"],
                    "language": artifact["language"],
                    "translation_status": artifact["translation_status"],
                    "mime_type": artifact["mime_type"],
                    "origin_type": artifact["origin_type"],
                    "provenance_tier": artifact["provenance_tier"],
                    "rights_status": artifact["rights_status"],
                    "acquisition_status": artifact["acquisition_status"],
                    "rights_checked_by": artifact["rights_checked_by"],
                    "rights_checked_at": _communication_iso(artifact["rights_checked_at"]),
                    "host_organization": artifact["host_organization"],
                    "publisher": artifact["publisher"],
                    "transcriber": artifact["transcriber"],
                    "transcriber_attribution": artifact["transcriber_attribution"],
                    "published_at": (
                        _communication_iso(artifact["published_at"])
                        if artifact["published_at"] is not None
                        else None
                    ),
                    "available_at": _communication_iso(artifact["available_at"]),
                    "landing_url": artifact["landing_url"],
                    "artifact_url": artifact["artifact_url"],
                }
            )
        except (TypeError, ValueError, UnicodeEncodeError):
            expected_artifact_hash = None
        artifact_hash_mismatch_count += (
            expected_artifact_hash != artifact["artifact_version_sha256"]
        )
        if (
            (event is not None and artifact["metadata_known_at"] < event["metadata_known_at"])
            or (
                artifact["published_at"] is not None
                and artifact["available_at"] < artifact["published_at"]
            )
            or artifact["retrieved_at"] < artifact["available_at"]
            or artifact["metadata_known_at"] < artifact["retrieved_at"]
            or artifact["metadata_known_at"] < artifact["rights_checked_at"]
        ):
            clock_mismatch_count += 1

    artifact_lineage_rows: list[dict[str, Any]] = []
    for artifact in artifact_rows:
        event = event_by_id.get(int(artifact["event_id"]))
        artifact_lineage_rows.append(
            {
                **artifact,
                "event_organization_id": (event["organization_id"] if event is not None else None),
                "event_key": event["event_key"] if event is not None else None,
            }
        )
    artifact_lineage_mismatch_count = _communication_lineage_mismatch_count(
        artifact_lineage_rows,
        id_field="id",
        predecessor_field="supersedes_artifact_id",
        identity_fields=(
            "event_organization_id",
            "event_key",
            "source_id",
            "artifact_key",
        ),
        clock_field="metadata_known_at",
    )

    content_table = tables["communication_artifact_contents"]
    content_rows = [
        dict(row)
        for row in connection.execute(
            select(
                *(
                    content_table.c[field]
                    for field in sorted(
                        _COMMUNICATION_REQUIRED_COLUMNS["communication_artifact_contents"]
                    )
                )
            )
        ).mappings()
    ]
    content_by_id = {int(row["id"]): row for row in content_rows}
    retrieval_by_id = {int(row["id"]): row for row in retrieval_rows}
    bad_content_bindings: set[int] = set()
    blob_path_mismatches = 0
    blob_missing_count = 0
    blob_size_mismatches = 0
    blob_hash_mismatches = 0
    blob_read_errors = 0
    verified_content_ids: set[int] = set()
    database = connection.engine.url.database
    if database in {None, "", ":memory:"} and connection.dialect.name == "sqlite":
        database = next(
            (
                str(row[2])
                for row in connection.exec_driver_sql("PRAGMA database_list")
                if str(row[1]) == "main" and str(row[2])
            ),
            None,
        )
    database_parent = (
        Path(database).resolve().parent if database not in {None, "", ":memory:"} else None
    )
    for content in content_rows:
        content_id = int(content["id"])
        artifact = artifact_by_id.get(int(content["artifact_id"]))
        retrieval = retrieval_by_id.get(int(content["retrieval_id"]))
        if (
            artifact is None
            or retrieval is None
            or int(retrieval["artifact_id"]) != int(content["artifact_id"])
            or int(content["retrieval_id"]) in bad_retrieval_ids
            or artifact["rights_status"] not in {"cleared", "internal_only"}
            or content["captured_at"] < retrieval["retrieved_at"]
            or content["captured_at"] < retrieval["metadata_known_at"]
        ):
            bad_content_bindings.add(content_id)
        if retrieval is not None and (
            content["captured_at"] < retrieval["retrieved_at"]
            or content["captured_at"] < retrieval["metadata_known_at"]
        ):
            clock_mismatch_count += 1

        digest = str(content["content_sha256"])
        relative_blob = Path(str(content["blob_path"]))
        parts = relative_blob.parts
        filename = parts[-1] if parts else ""
        path_has_content_address = (
            len(parts) == 5
            and parts[:3] == ("artifacts", "communications", "sha256")
            and parts[3] == digest[:2]
            and filename == digest
        )
        if database_parent is None or not path_has_content_address:
            blob_path_mismatches += 1
            continue
        candidate = (database_parent / relative_blob).resolve()
        try:
            candidate.relative_to(database_parent)
        except ValueError:
            blob_path_mismatches += 1
            continue
        if not candidate.is_file():
            blob_missing_count += 1
            continue
        try:
            actual_size = candidate.stat().st_size
            actual_hash = _file_sha256(candidate)
        except OSError:
            blob_read_errors += 1
            continue
        size_matches = actual_size == content["size_bytes"]
        hash_matches = actual_hash == content["content_sha256"]
        blob_size_mismatches += not size_matches
        blob_hash_mismatches += not hash_matches
        if size_matches and hash_matches:
            verified_content_ids.add(content_id)

    content_lineage_mismatch_count = _communication_lineage_mismatch_count(
        content_rows,
        id_field="id",
        predecessor_field="supersedes_content_id",
        identity_fields=("artifact_id",),
        clock_field="captured_at",
    )

    coverage_rows = [
        dict(row)
        for row in connection.execute(
            select(
                coverage.c.id,
                coverage.c.organization_id,
                coverage.c.coverage_key,
                coverage.c.commodity_family,
                coverage.c.exposure_role,
                coverage.c.mapping_status,
                coverage.c.effective_from,
                coverage.c.effective_to,
                coverage.c.source_id,
                coverage.c.catalogue_sha256,
                coverage.c.evidence_url,
                coverage.c.evidence_note,
                coverage.c.published_at,
                coverage.c.available_at,
                coverage.c.retrieved_at,
                coverage.c.metadata_known_at,
                coverage.c.coverage_version_sha256,
                coverage.c.supersedes_exposure_id,
            )
        ).mappings()
    ]
    coverage_policy_mismatches = 0
    coverage_hash_mismatch_count = 0
    for row in coverage_rows:
        policy_key = (str(row["catalogue_sha256"]), str(row["source_id"]))
        policy = policy_by_key.get(policy_key)
        if (
            policy is None
            or policy["organization_id"] != row["organization_id"]
            or policy["organization_type"] != "commodity_company"
            or row["commodity_family"] not in policy_commodity_families.get(policy_key, [])
            or row["evidence_note"] != policy["coverage_note"]
            or not _communication_url_matches_domains(
                row["evidence_url"], policy_domains.get(policy_key, [])
            )
        ):
            coverage_policy_mismatches += 1
        try:
            expected_coverage_hash = _canonical_dict_sha256(
                {
                    "organization_id": row["organization_id"],
                    "coverage_key": row["coverage_key"],
                    "commodity_family": row["commodity_family"],
                    "exposure_role": row["exposure_role"],
                    "mapping_status": row["mapping_status"],
                    "effective_from": _communication_iso(row["effective_from"]),
                    "effective_to": (
                        _communication_iso(row["effective_to"])
                        if row["effective_to"] is not None
                        else None
                    ),
                    "source_id": row["source_id"],
                    "catalogue_sha256": row["catalogue_sha256"],
                    "evidence_url": row["evidence_url"],
                    "evidence_note": row["evidence_note"],
                    "published_at": (
                        _communication_iso(row["published_at"])
                        if row["published_at"] is not None
                        else None
                    ),
                    "available_at": _communication_iso(row["available_at"]),
                    "retrieved_at": _communication_iso(row["retrieved_at"]),
                }
            )
        except (TypeError, ValueError, UnicodeEncodeError):
            expected_coverage_hash = None
        coverage_hash_mismatch_count += expected_coverage_hash != row["coverage_version_sha256"]
        if (
            (row["published_at"] is not None and row["available_at"] < row["published_at"])
            or row["retrieved_at"] < row["available_at"]
            or row["metadata_known_at"] < row["retrieved_at"]
        ):
            clock_mismatch_count += 1

    coverage_lineage_mismatch_count = _communication_lineage_mismatch_count(
        coverage_rows,
        id_field="id",
        predecessor_field="supersedes_exposure_id",
        identity_fields=("organization_id", "coverage_key"),
        clock_field="metadata_known_at",
    )
    superseded_coverage_ids = {
        int(row["supersedes_exposure_id"])
        for row in coverage_rows
        if row["supersedes_exposure_id"] is not None
    }
    coverage_heads: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        if int(row["id"]) not in superseded_coverage_ids:
            coverage_heads[
                (row["organization_id"], row["commodity_family"], row["exposure_role"])
            ].append(row)
    for heads in coverage_heads.values():
        for left_index, left in enumerate(heads):
            left_end = left["effective_to"] or date.max
            for right in heads[left_index + 1 :]:
                right_end = right["effective_to"] or date.max
                coverage_lineage_mismatch_count += (
                    left["effective_from"] <= right_end and right["effective_from"] <= left_end
                )

    stored_catalogue_hashes = {
        str(row["catalogue_sha256"]) for row in (*policy_rows, *artifact_rows, *coverage_rows)
    }
    untrusted_catalogue_hashes = sorted(stored_catalogue_hashes - {str(catalogue["sha256"])})

    extraction_rows = [
        dict(row)
        for row in connection.execute(
            select(
                extractions.c.id,
                extractions.c.artifact_content_id,
                extractions.c.extracted_at,
            )
        ).mappings()
    ]
    extraction_ids = {int(row["id"]) for row in extraction_rows}
    extraction_by_id = {int(row["id"]): row for row in extraction_rows}
    bad_extraction_bindings: set[int] = set()
    for extraction in extraction_rows:
        extraction_id = int(extraction["id"])
        content_id = int(extraction["artifact_content_id"])
        content = content_by_id.get(content_id)
        if (
            content is None
            or content_id in bad_content_bindings
            or content_id not in verified_content_ids
            or extraction["extracted_at"] < content["captured_at"]
        ):
            bad_extraction_bindings.add(extraction_id)
        if content is not None and extraction["extracted_at"] < content["captured_at"]:
            clock_mismatch_count += 1

    segment_rows = [
        dict(row)
        for row in connection.execute(
            select(
                *(
                    segments.c[field]
                    for field in ("extraction_id", *_COMMUNICATION_CANONICAL_SEGMENT_FIELDS)
                ),
                segments.c.text_sha256,
                segments.c.char_count,
            ).order_by(segments.c.extraction_id, segments.c.ordinal)
        ).mappings()
    ]
    segments_by_extraction: dict[int, list[dict[str, Any]]] = defaultdict(list)
    text_hash_mismatch_ids: set[int] = set()
    char_count_mismatch_ids: set[int] = set()
    text_hash_mismatch_count = 0
    char_count_mismatch_count = 0
    for segment in segment_rows:
        extraction_id = int(segment["extraction_id"])
        segments_by_extraction[extraction_id].append(segment)
        text_value = segment["text"]
        if isinstance(text_value, str):
            try:
                expected_text_hash = hashlib.sha256(text_value.encode("utf-8")).hexdigest()
            except UnicodeEncodeError:
                expected_text_hash = None
            expected_char_count = len(text_value)
        else:
            expected_text_hash = None
            expected_char_count = None
        if expected_text_hash != segment["text_sha256"]:
            text_hash_mismatch_count += 1
            text_hash_mismatch_ids.add(extraction_id)
        if expected_char_count != segment["char_count"]:
            char_count_mismatch_count += 1
            char_count_mismatch_ids.add(extraction_id)

    finalization_rows = [
        dict(row)
        for row in connection.execute(
            select(
                finalizations.c.extraction_id,
                finalizations.c.finalized_at,
                finalizations.c.segment_count,
                finalizations.c.total_char_count,
                finalizations.c.corpus_sha256,
                finalizations.c.canonicalization_version,
            )
        ).mappings()
    ]
    finalized_ids = {int(row["extraction_id"]) for row in finalization_rows}
    bad_structure_ids: set[int] = set()
    unsupported_canonicalization_ids: set[int] = set()
    corpus_hash_mismatch_ids: set[int] = set()
    canonicalization_error_ids: set[int] = set()
    for finalization in finalization_rows:
        extraction_id = int(finalization["extraction_id"])
        extraction = extraction_by_id.get(extraction_id)
        extraction_segments = segments_by_extraction.get(extraction_id, [])
        ordinals = [segment["ordinal"] for segment in extraction_segments]
        stored_char_counts = [segment["char_count"] for segment in extraction_segments]
        stored_char_sum = (
            sum(stored_char_counts)
            if all(isinstance(value, int) for value in stored_char_counts)
            else None
        )
        exact_char_sum = (
            sum(len(segment["text"]) for segment in extraction_segments)
            if all(isinstance(segment["text"], str) for segment in extraction_segments)
            else None
        )
        if (
            extraction_id not in extraction_ids
            or len(extraction_segments) != finalization["segment_count"]
            or ordinals != list(range(1, len(extraction_segments) + 1))
            or stored_char_sum != finalization["total_char_count"]
            or exact_char_sum != finalization["total_char_count"]
        ):
            bad_structure_ids.add(extraction_id)
        if extraction is not None and finalization["finalized_at"] < extraction["extracted_at"]:
            clock_mismatch_count += 1
            bad_structure_ids.add(extraction_id)
        if finalization["canonicalization_version"] != _COMMUNICATION_CANONICALIZATION:
            unsupported_canonicalization_ids.add(extraction_id)
            continue
        try:
            expected_corpus_hash = _communication_corpus_sha256(extraction_segments)
        except (KeyError, TypeError, ValueError, UnicodeEncodeError):
            canonicalization_error_ids.add(extraction_id)
            continue
        if expected_corpus_hash != finalization["corpus_sha256"]:
            corpus_hash_mismatch_ids.add(extraction_id)

    locally_invalid_finalizations = (
        bad_extraction_bindings
        | text_hash_mismatch_ids
        | char_count_mismatch_ids
        | bad_structure_ids
        | unsupported_canonicalization_ids
        | corpus_hash_mismatch_ids
        | canonicalization_error_ids
    )
    result.update(
        {
            "policy_snapshot_json_malformed_count": malformed_policy_json_count,
            "policy_snapshot_semantic_mismatch_count": policy_semantic_mismatch_count,
            "policy_snapshot_sha256_mismatch_count": policy_hash_mismatch_count,
            "current_catalogue_snapshot_check_status": current_snapshot_check_status,
            "current_catalogue_snapshot_mismatch_count": current_snapshot_mismatch_count,
            "untrusted_catalogue_hash_count": len(untrusted_catalogue_hashes),
            "untrusted_catalogue_hashes": untrusted_catalogue_hashes,
            "event_version_sha256_mismatch_count": event_hash_mismatch_count,
            "artifact_version_sha256_mismatch_count": artifact_hash_mismatch_count,
            "artifact_semantic_mismatch_count": artifact_semantic_mismatch_count,
            "coverage_version_sha256_mismatch_count": coverage_hash_mismatch_count,
            "event_lineage_mismatch_count": event_lineage_mismatch_count,
            "artifact_lineage_mismatch_count": artifact_lineage_mismatch_count,
            "coverage_lineage_mismatch_count": coverage_lineage_mismatch_count,
            "content_lineage_mismatch_count": content_lineage_mismatch_count,
            "communication_clock_mismatch_count": clock_mismatch_count,
            "artifact_policy_binding_mismatch_count": artifact_policy_mismatches,
            "commodity_policy_binding_mismatch_count": coverage_policy_mismatches,
            "policy_binding_mismatch_count": (
                artifact_policy_mismatches + coverage_policy_mismatches
            ),
            "missing_base_artifact_retrieval_count": missing_base_retrieval_count,
            "orphan_artifact_retrieval_count": orphan_retrieval_count,
            "artifact_retrieval_mismatch_count": retrieval_mismatch_count,
            "content_retrieval_binding_mismatch_count": len(bad_content_bindings),
            "archived_blob_path_mismatch_count": blob_path_mismatches,
            "archived_blob_missing_count": blob_missing_count,
            "archived_blob_size_mismatch_count": blob_size_mismatches,
            "archived_blob_sha256_mismatch_count": blob_hash_mismatches,
            "archived_blob_read_error_count": blob_read_errors,
            "verified_archived_blob_count": len(verified_content_ids),
            "extraction_content_binding_mismatch_count": len(bad_extraction_bindings),
            "incomplete_extraction_count": len(extraction_ids - finalized_ids),
            "finalized_segment_count": sum(
                len(segments_by_extraction.get(extraction_id, []))
                for extraction_id in finalized_ids
            ),
            "unfinalized_segment_count": sum(
                len(extraction_segments)
                for extraction_id, extraction_segments in segments_by_extraction.items()
                if extraction_id not in finalized_ids
            ),
            "segment_text_sha256_mismatch_count": text_hash_mismatch_count,
            "segment_char_count_mismatch_count": char_count_mismatch_count,
            "finalization_structure_mismatch_count": len(bad_structure_ids),
            "unsupported_canonicalization_count": len(unsupported_canonicalization_ids),
            "corpus_sha256_mismatch_count": len(corpus_hash_mismatch_ids),
            "canonicalization_error_count": len(canonicalization_error_ids),
            "valid_finalized_extraction_count": len(finalized_ids - locally_invalid_finalizations),
        }
    )

    failure_counts = (
        ("policy_snapshot_json", result["policy_snapshot_json_malformed_count"]),
        ("policy_snapshot_semantics", result["policy_snapshot_semantic_mismatch_count"]),
        ("policy_snapshot_sha256", result["policy_snapshot_sha256_mismatch_count"]),
        ("current_catalogue_snapshot", result["current_catalogue_snapshot_mismatch_count"]),
        ("untrusted_catalogue_hash", result["untrusted_catalogue_hash_count"]),
        ("event_version_sha256", result["event_version_sha256_mismatch_count"]),
        ("artifact_version_sha256", result["artifact_version_sha256_mismatch_count"]),
        ("artifact_semantics", result["artifact_semantic_mismatch_count"]),
        ("coverage_version_sha256", result["coverage_version_sha256_mismatch_count"]),
        ("event_lineage", result["event_lineage_mismatch_count"]),
        ("artifact_lineage", result["artifact_lineage_mismatch_count"]),
        ("coverage_lineage", result["coverage_lineage_mismatch_count"]),
        ("content_lineage", result["content_lineage_mismatch_count"]),
        ("communication_clocks", result["communication_clock_mismatch_count"]),
        ("policy_binding", result["policy_binding_mismatch_count"]),
        ("missing_base_artifact_retrieval", result["missing_base_artifact_retrieval_count"]),
        ("orphan_artifact_retrieval", result["orphan_artifact_retrieval_count"]),
        ("artifact_retrieval", result["artifact_retrieval_mismatch_count"]),
        ("content_retrieval_binding", result["content_retrieval_binding_mismatch_count"]),
        ("archived_blob_path", result["archived_blob_path_mismatch_count"]),
        ("archived_blob_missing", result["archived_blob_missing_count"]),
        ("archived_blob_size", result["archived_blob_size_mismatch_count"]),
        ("archived_blob_sha256", result["archived_blob_sha256_mismatch_count"]),
        ("archived_blob_read", result["archived_blob_read_error_count"]),
        ("extraction_content_binding", result["extraction_content_binding_mismatch_count"]),
        ("segment_text_sha256", result["segment_text_sha256_mismatch_count"]),
        ("segment_char_count", result["segment_char_count_mismatch_count"]),
        ("finalization_structure", result["finalization_structure_mismatch_count"]),
        ("canonicalization_version", result["unsupported_canonicalization_count"]),
        ("corpus_sha256", result["corpus_sha256_mismatch_count"]),
        ("canonicalization", result["canonicalization_error_count"]),
    )
    failures = [reason for reason, count in failure_counts if count]
    if trigger_check_status != "valid":
        failures.append("required_triggers")
    if foreign_key_check_status != "valid":
        failures.append("foreign_keys")
    if contract_status != "valid":
        failures.append("schema_contract")
    if catalogue["validation_status"] == "valid" and current_snapshot_check_status != "valid":
        failures.append("current_catalogue_snapshot_check")
    result["integrity_failures"] = failures
    result["integrity_status"] = "invalid" if failures else "valid"
    return result


def _communications_inventory(
    connection: Connection,
    tables: dict[str, Table],
) -> dict[str, Any]:
    """Describe collected communication evidence without implying archive completeness."""
    catalogue = _communication_catalogue_summary()
    organizations = tables.get("organizations")
    policies = tables.get("communication_source_policy_snapshots")
    coverage = tables.get("organization_commodity_coverage")
    events = tables.get("communication_events")
    artifacts = tables.get("communication_artifacts")
    retrievals = tables.get("communication_artifact_retrievals")
    contents = tables.get("communication_artifact_contents")
    extractions = tables.get("communication_extractions")
    segments = tables.get("communication_segments")
    finalizations = tables.get("communication_extraction_finalizations")

    def counts_by(table: Table | None, field: str) -> list[dict[str, Any]]:
        if table is None or field not in table.c:
            return []
        return _execute_rows(
            connection,
            select(table.c[field].label(field), func.count().label("row_count"))
            .group_by(table.c[field])
            .order_by(table.c[field]),
        )

    event_version_count = _count(connection, events)
    logical_event_count: int | None = None
    first_event_date = None
    latest_event_date = None
    if events is not None and {"organization_id", "event_key", "event_date"}.issubset(
        events.c.keys()
    ):
        logical_event_count = len(
            connection.execute(
                select(events.c.organization_id, events.c.event_key).distinct()
            ).all()
        )
        first_event_date, latest_event_date = connection.execute(
            select(func.min(events.c.event_date), func.max(events.c.event_date))
        ).one()

    artifact_version_count = _count(connection, artifacts)
    represented_source_count: int | None = None
    stored_content_artifact_count: int | None = None
    archived_artifact_count: int | None = None
    first_available_at = None
    latest_available_at = None
    catalogue_hashes: set[str] = set()
    if artifacts is not None:
        if "source_id" in artifacts.c:
            represented_source_count = int(
                connection.scalar(select(func.count(func.distinct(artifacts.c.source_id)))) or 0
            )
        if "available_at" in artifacts.c:
            first_available_at, latest_available_at = connection.execute(
                select(func.min(artifacts.c.available_at), func.max(artifacts.c.available_at))
            ).one()
    if contents is not None and "artifact_id" in contents.c:
        stored_content_artifact_count = int(
            connection.scalar(select(func.count(func.distinct(contents.c.artifact_id)))) or 0
        )
        archived_artifact_count = stored_content_artifact_count
    for table in (policies, coverage, artifacts):
        if table is not None and "catalogue_sha256" in table.c:
            catalogue_hashes.update(
                str(value)
                for value in connection.scalars(select(table.c.catalogue_sha256).distinct()).all()
                if value is not None
            )

    integrity = _communication_integrity_inventory(
        connection,
        tables,
        catalogue=catalogue,
    )
    return {
        "catalogue_validation_status": catalogue["validation_status"],
        "catalogue_source_policy_count": catalogue["source_policy_count"],
        "catalogue_organization_count": catalogue["organization_count"],
        "catalogue_sha256": catalogue["sha256"],
        "catalogue_error": catalogue["error"],
        "archive_coverage_status": "not_measured",
        "archive_coverage_note": (
            "No complete event universe is pinned yet; counts describe stored evidence only."
        ),
        "integrity_assurance_scope": "structural_byte_hash_reproducibility",
        "transcript_semantic_fidelity_status": "not_verified",
        "extractor_execution_trust_status": "not_verified",
        "integrity_assurance_note": (
            "Valid integrity and available readiness prove structural, byte, and hash "
            "reproducibility only; they do not prove transcript semantic fidelity or "
            "trusted extractor execution."
        ),
        "table_presence": {name: name in tables for name in _COMMUNICATION_TABLE_NAMES},
        "organization_count": _count(connection, organizations),
        "source_policy_snapshot_count": _count(connection, policies),
        "commodity_mapping_count": _count(connection, coverage),
        "event_count": logical_event_count,
        "event_version_count": event_version_count,
        "artifact_version_count": artifact_version_count,
        "artifact_retrieval_count": _count(connection, retrievals),
        "artifact_content_count": _count(connection, contents),
        "stored_content_artifact_count": stored_content_artifact_count,
        "archived_artifact_count": archived_artifact_count,
        "extraction_count": _count(connection, extractions),
        "finalized_extraction_count": _count(connection, finalizations),
        "segment_count": _count(connection, segments),
        "represented_source_count": represented_source_count,
        "stored_catalogue_hashes": sorted(catalogue_hashes),
        "first_event_date": _json_value(first_event_date),
        "latest_event_date": _json_value(latest_event_date),
        "first_available_at": _json_value(first_available_at),
        "latest_available_at": _json_value(latest_available_at),
        "events_by_type": counts_by(events, "event_type"),
        "artifacts_by_role": counts_by(artifacts, "artifact_role"),
        "artifacts_by_origin": counts_by(artifacts, "origin_type"),
        "artifacts_by_provenance": counts_by(artifacts, "provenance_tier"),
        "artifacts_by_rights": counts_by(artifacts, "rights_status"),
        "artifacts_by_acquisition": counts_by(artifacts, "acquisition_status"),
        "commodity_mappings_by_status": counts_by(coverage, "mapping_status"),
        "segments_by_kind": counts_by(segments, "segment_kind"),
        "segments_by_speaker_side": counts_by(segments, "speaker_side"),
        "finalizations_by_canonicalization": counts_by(finalizations, "canonicalization_version"),
        **integrity,
    }


def _cross_border_inventory(
    connection: Connection,
    table: Table | None,
    latest_release_ids,
) -> dict[str, Any]:
    if table is None:
        expected_count = len(IMF_POSITION_COUNTRIES) * (
            len(PIP_SERIES) * len(PIP_FREQUENCIES) + len(DIP_SERIES) * len(DIP_FREQUENCIES)
        )
        return {
            "table_present": False,
            "row_count": None,
            "current_row_count": None,
            "expected_partitions": expected_count,
            "stored_partitions": 0,
            "missing_partitions": expected_count,
            "coverage_pct": 0.0,
            "by_dataset": [],
            "missing": [],
            "datasets": [],
            "by_dataset_reporter": [],
            "gap": "table_not_present",
        }
    row_count = _count(connection, table)
    condition = _current_condition(table, latest_release_ids)
    count_statement = select(func.count()).select_from(table)
    if condition is not None:
        count_statement = count_statement.where(condition)
    current_row_count = (
        int(connection.scalar(count_statement) or 0) if condition is not None else row_count
    )
    datasets: list[str] = []
    if "dataset" in table.c:
        statement = select(table.c.dataset).distinct().order_by(table.c.dataset)
        if condition is not None:
            statement = statement.where(condition)
        datasets = [str(value) for value in connection.scalars(statement).all()]

    coverage: list[dict[str, Any]] = []
    needed = {"dataset", "reporter_country", "counterpart_country", "date"}
    if needed.issubset(table.c.keys()):
        statement = (
            select(
                table.c.dataset.label("dataset"),
                table.c.reporter_country.label("reporter_country"),
                func.count().label("row_count"),
                func.count(func.distinct(table.c.counterpart_country)).label("counterpart_count"),
                func.min(table.c.date).label("first_date"),
                func.max(table.c.date).label("latest_date"),
            )
            .group_by(table.c.dataset, table.c.reporter_country)
            .order_by(table.c.dataset, table.c.reporter_country)
        )
        if condition is not None:
            statement = statement.where(condition)
        coverage = _execute_rows(connection, statement)

    catalogue = {
        "PIP": (PIP_SERIES, PIP_FREQUENCIES),
        "DIP": (DIP_SERIES, DIP_FREQUENCIES),
    }
    expected: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for dataset, (specs, frequencies) in catalogue.items():
        for country in IMF_POSITION_COUNTRIES:
            for spec in specs:
                for frequency in frequencies:
                    expected[(dataset, country.iso2, spec.native_indicator, frequency)] = {
                        "dataset": dataset,
                        "reporter_country": country.iso2,
                        "indicator": spec.indicator,
                        "native_indicator": spec.native_indicator,
                        "frequency": frequency,
                        "status": "missing_or_not_reported",
                    }

    stored_keys: set[tuple[str, str, str, str]] = set()
    partition_fields = {"dataset", "reporter_country", "native_indicator", "frequency"}
    if partition_fields.issubset(table.c.keys()):
        statement = select(
            table.c.dataset,
            table.c.reporter_country,
            table.c.native_indicator,
            table.c.frequency,
        ).distinct()
        if condition is not None:
            statement = statement.where(condition)
        stored_keys = {
            (str(dataset), str(reporter), str(indicator), str(frequency))
            for dataset, reporter, indicator, frequency in connection.execute(statement)
        } & set(expected)
    missing_keys = set(expected) - stored_keys
    by_dataset = []
    for dataset in sorted(catalogue):
        dataset_expected = sum(key[0] == dataset for key in expected)
        dataset_stored = sum(key[0] == dataset for key in stored_keys)
        by_dataset.append(
            {
                "dataset": dataset,
                "expected_partitions": dataset_expected,
                "stored_partitions": dataset_stored,
                "missing_partitions": dataset_expected - dataset_stored,
                "coverage_pct": _coverage_pct(dataset_stored, dataset_expected),
            }
        )

    dimensions: dict[str, list[str]] = {}
    for field in (
        "direction",
        "accounting_basis",
        "instrument_code",
        "frequency",
        "unit",
        "source",
        "status",
    ):
        if field not in table.c:
            continue
        statement = select(table.c[field]).distinct().order_by(table.c[field])
        if condition is not None:
            statement = statement.where(condition)
        dimensions[field] = [str(value) for value in connection.scalars(statement).all()]
    instruments: list[dict[str, Any]] = []
    if {"instrument_code", "instrument_label"}.issubset(table.c.keys()):
        statement = select(
            table.c.instrument_code.label("code"),
            table.c.instrument_label.label("label"),
        ).distinct()
        if condition is not None:
            statement = statement.where(condition)
        instruments = _execute_rows(
            connection,
            statement.order_by(table.c.instrument_code, table.c.instrument_label),
        )
    return {
        "table_present": True,
        "row_count": row_count,
        "current_row_count": current_row_count,
        "expected_partitions": len(expected),
        "stored_partitions": len(stored_keys),
        "missing_partitions": len(missing_keys),
        "coverage_pct": _coverage_pct(len(stored_keys), len(expected)),
        "by_dataset": by_dataset,
        "missing": [expected[key] for key in sorted(missing_keys)],
        "datasets": datasets,
        "by_dataset_reporter": coverage,
        "dimensions": dimensions,
        "instruments": instruments,
        "gap": ("empty" if current_row_count == 0 else "partial" if missing_keys else None),
    }


def _availability(value: int | None) -> str:
    if value is None:
        return "table_absent"
    return "available" if value > 0 else "empty"


def _coverage_status(stored: int, expected: int) -> str:
    if stored == 0:
        return "empty"
    return "complete" if stored == expected else "partial"


def _report_readiness(reports: dict[str, Any]) -> str:
    counts = [
        reports["document_count"],
        reports["page_count"],
        reports["claim_count"],
        reports["citation_count"],
    ]
    if any(value is None for value in counts):
        return "table_absent"
    if all(value and value > 0 for value in counts):
        return "available"
    if any(value and value > 0 for value in counts):
        return "partial"
    return "empty"


def _communications_readiness(communications: dict[str, Any]) -> str:
    if communications["catalogue_validation_status"] != "valid":
        return "policy_invalid"
    if not all(communications["table_presence"].values()):
        return "table_absent"
    if communications["integrity_status"] != "valid":
        return "invalid"
    if communications["event_version_count"] == 0:
        return "empty"
    if communications["valid_finalized_extraction_count"] > 0:
        return "available"
    if communications["artifact_content_count"] > 0:
        return "artifacts_unextracted"
    return "metadata_only"


def build_observatory_inventory(
    engine: Engine,
    *,
    as_of: date | None = None,
) -> dict[str, Any]:
    """Inspect without writes; ``as_of`` makes recency qualification reproducible."""
    inventory_as_of = as_of or datetime.now(UTC).date()
    present = set(inspect(engine).get_table_names())
    all_names = present | set(_EXPECTED_TABLES)
    with engine.connect() as connection:
        tables = _reflect(connection, present)
        table_inventory = [
            {
                "name": name,
                "present": name in present,
                "row_count": _count(connection, tables.get(name)),
            }
            for name in sorted(all_names)
        ]
        observations = _observation_inventory(connection, tables.get("observations"))
        releases = _release_inventory(
            connection,
            tables.get("data_releases"),
            tables.get("release_observations"),
        )
        latest_release_ids = _latest_release_ids(tables.get("data_releases"))
        qpsd = _qpsd_inventory(
            connection,
            observations,
            tables.get("data_releases"),
        )
        bop = _bop_inventory(observations)
        debt_holders = _debt_holder_inventory(
            connection,
            tables.get("debt_holder_positions"),
            latest_release_ids,
        )
        allocators = _allocator_inventory(
            connection,
            tables.get("allocator_facts"),
            latest_release_ids,
        )
        reports = _reports_inventory(connection, tables)
        communications = _communications_inventory(connection, tables)
        cross_border = _cross_border_inventory(
            connection,
            tables.get("cross_border_positions"),
            latest_release_ids,
        )
        market_history = _market_history_inventory(
            connection,
            tables.get("observations"),
            tables.get("data_releases"),
            tables.get("data_release_artifacts"),
            as_of=inventory_as_of,
        )

    readiness = {
        "current_observations": _availability(observations["row_count"]),
        "immutable_release_history": _availability(releases["release_count"]),
        "sovereign_debt_anatomy": _coverage_status(
            qpsd["stored_partitions"], qpsd["expected_partitions"]
        ),
        "cross_border_transactions": _coverage_status(
            bop["stored_partitions"], bop["expected_partitions"]
        ),
        "debt_holder_positions": _availability(debt_holders["current_row_count"]),
        "allocator_disclosures": _availability(allocators["current_row_count"]),
        "report_evidence": _report_readiness(reports),
        "institutional_communications": _communications_readiness(communications),
        "bilateral_positions": (
            "table_absent"
            if not cross_border["table_present"]
            else _coverage_status(
                cross_border["stored_partitions"],
                cross_border["expected_partitions"],
            )
        ),
        "commodity_history": _coverage_status(
            market_history["commodities"]["ready_series"],
            market_history["commodities"]["expected_series"],
        ),
        "money_liquidity": _coverage_status(
            market_history["money_liquidity"]["ready_series"],
            market_history["money_liquidity"]["expected_series"],
        ),
        "shadow_liquidity": _coverage_status(
            market_history["shadow_liquidity"]["ready_series"],
            market_history["shadow_liquidity"]["expected_series"],
        ),
    }
    return {
        "schema_version": 1,
        "tables": table_inventory,
        "observations": observations,
        "releases": releases,
        "qpsd": qpsd,
        "imf_bop": bop,
        "debt_holders": debt_holders,
        "allocators": allocators,
        "reports": reports,
        "communications": communications,
        "cross_border_positions": cross_border,
        "market_history": market_history,
        "readiness": readiness,
    }


def render_inventory_summary(inventory: dict[str, Any]) -> str:
    """Render a compact deterministic summary; ``--json`` retains full detail."""
    observations = inventory["observations"]
    releases = inventory["releases"]
    qpsd = inventory["qpsd"]
    bop = inventory["imf_bop"]
    debt = inventory["debt_holders"]
    allocators = inventory["allocators"]
    reports = inventory["reports"]
    communications = inventory["communications"]
    positions = inventory["cross_border_positions"]
    market = inventory["market_history"]
    commodities = market["commodities"]
    money = market["money_liquidity"]
    shadow = market["shadow_liquidity"]

    def count_text(value: int | None) -> str:
        return "table absent" if value is None else f"{value:,}"

    if positions["table_present"]:
        position_line = (
            f"Cross-border positions: {positions['current_row_count']:,} rows; "
            f"{positions['stored_partitions']}/{positions['expected_partitions']} partitions"
        )
    else:
        position_line = "Cross-border positions: table not present"
    return "\n".join(
        (
            "Observatory data inventory",
            f"Current observations: {count_text(observations['row_count'])} rows",
            f"Immutable releases: {count_text(releases['release_count'])} releases; "
            f"{count_text(releases['observation_row_count'])} observation rows",
            f"QPSD: {qpsd['stored_partitions']}/{qpsd['expected_partitions']} stored; "
            f"{qpsd['not_reported_partitions']} evidenced not reported; "
            f"{qpsd['missing_partitions']} unverified missing",
            f"IMF BOP transactions: {bop['stored_partitions']}/{bop['expected_partitions']} stored",
            f"Debt-holder positions: {count_text(debt['current_row_count'])} current rows",
            f"Allocator disclosures: {count_text(allocators['current_row_count'])} current rows",
            f"Report evidence: {count_text(reports['document_count'])} documents; "
            f"{count_text(reports['page_count'])} pages; "
            f"{count_text(reports['claim_count'])} claims; "
            f"{count_text(reports['review_count'])} human decisions",
            "Institutional communications: "
            f"{communications['catalogue_source_policy_count']} source policies; "
            f"{communications['catalogue_organization_count']} organizations; "
            f"{count_text(communications['event_version_count'])} event versions; "
            f"{count_text(communications['artifact_version_count'])} artifact versions; "
            f"{count_text(communications['artifact_content_count'])} content captures; "
            f"{count_text(communications['finalized_extraction_count'])} finalized extractions; "
            f"{count_text(communications['finalized_segment_count'])} finalized segments; "
            f"readiness {inventory['readiness']['institutional_communications']}; "
            "semantic fidelity/extractor trust not verified; archive coverage not measured",
            position_line,
            f"Commodity history: {commodities['ready_series']}/"
            f"{commodities['expected_series']} expected series ready; "
            f"{commodities['stored_series']} stored; "
            f"{commodities['price_series_count']} prices + "
            f"{commodities['index_series_count']} indices",
            f"Money/liquidity: {money['ready_series']}/{money['expected_series']} "
            f"pinned series ready; {money['stored_series']} stored; "
            f"catalogue {money['catalogue_semantic_sha256'][:12]}",
            f"Shadow liquidity: {shadow['ready_series']}/{shadow['expected_series']} "
            f"series ready; {shadow['stored_series']} stored; "
            f"BIS {shadow['provider_catalogue_semantic_sha256'][SOURCE_BIS_GLI][:12]}; "
            f"OFR {shadow['provider_catalogue_semantic_sha256'][SOURCE_OFR_STFM][:12]}",
        )
    )


def _read_only_engine(path: Path) -> Engine:
    resolved = path.resolve()
    return create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{resolved}?mode=ro", uri=True),
        future=True,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit structured observatory data read-only.")
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(os.environ.get("DALIO_DB_PATH", "data/dalio.db")),
        help="SQLite database path (default: DALIO_DB_PATH or data/dalio.db).",
    )
    parser.add_argument("--json", action="store_true", help="Print the complete JSON inventory.")
    args = parser.parse_args(argv)
    if not args.db.is_file():
        parser.error(f"database does not exist: {args.db}")
    engine = _read_only_engine(args.db)
    inventory = build_observatory_inventory(engine)
    if args.json:
        print(json.dumps(inventory, indent=2, sort_keys=True))
    else:
        print(render_inventory_summary(inventory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
