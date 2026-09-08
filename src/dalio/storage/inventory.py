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
    "cross_border_positions",
    "data_release_artifacts",
    "data_releases",
    "debt_holder_positions",
    "document_extractions",
    "document_pages",
    "observations",
    "release_observations",
    "report_documents",
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
        "documents": document_groups,
        "extraction_statuses": counts_by(extractions, "status"),
        "claim_statuses": counts_by(claims, "status"),
        "claim_types": counts_by(claims, "claim_type"),
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
            f"{count_text(reports['claim_count'])} claims",
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
