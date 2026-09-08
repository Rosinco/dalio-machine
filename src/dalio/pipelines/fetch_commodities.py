"""Fetch the World Bank Pink Sheet monthly universe into immutable releases."""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import Counter
from datetime import UTC, date, datetime

import numpy as np
import pandas as pd
from sqlalchemy import Engine, delete, select
from sqlalchemy.orm import Session

from dalio.data_sources.worldbank_commodities import (
    EXPECTED_PINK_SHEET_HISTORY_START,
    EXPECTED_PINK_SHEET_MIN_INDEX_COLUMNS,
    EXPECTED_PINK_SHEET_MIN_PRICE_COLUMNS,
    EXPECTED_PINK_SHEET_SERIES_IDS_BY_WORKSHEET,
    MAX_PINK_SHEET_LATEST_LAG_MONTHS,
    MIN_PINK_SHEET_HISTORY_MONTHS,
    MIN_PINK_SHEET_OBSERVATIONS_PER_SERIES,
    MONTHLY_INDICES_SHEET_NAME,
    MONTHLY_SHEET_NAME,
    SOURCE_WORLD_BANK_COMMODITIES,
    WORLD_CODE,
    CommodityDataset,
    CommoditySeries,
    PinkSheetSource,
    canonical_indicator,
    canonical_series_id,
    pink_sheet_vintage_label,
)
from dalio.storage.db import Observation, init_db, make_engine
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseMeta,
    ingest_release_snapshot,
    latest_release,
    make_partition_key,
)

logger = logging.getLogger(__name__)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_INDEX_UNIT_RE = re.compile(r"^([12][0-9]{3})=100$")
_CELL_REFERENCE_RE = re.compile(r"^[A-Z]+[1-9][0-9]*$")
_REQUIRED_COLUMNS = {"country", "indicator", "date", "value", "source", "series_id"}

# Backwards-compatible local names keep the guard logic and its tests readable;
# the authoritative catalogue contract lives with the source adapter.
EXPECTED_MIN_PRICE_COLUMNS = EXPECTED_PINK_SHEET_MIN_PRICE_COLUMNS
EXPECTED_MIN_INDEX_COLUMNS = EXPECTED_PINK_SHEET_MIN_INDEX_COLUMNS
EXPECTED_HISTORY_START = EXPECTED_PINK_SHEET_HISTORY_START
MIN_HISTORY_MONTHS = MIN_PINK_SHEET_HISTORY_MONTHS
MIN_OBSERVATIONS_PER_SERIES = MIN_PINK_SHEET_OBSERVATIONS_PER_SERIES
MAX_LATEST_LAG_MONTHS = MAX_PINK_SHEET_LATEST_LAG_MONTHS
EXPECTED_SERIES_IDS_BY_WORKSHEET = EXPECTED_PINK_SHEET_SERIES_IDS_BY_WORKSHEET


def partition_key_for(series: CommoditySeries) -> str:
    """Stable independently replaceable WLD/native-benchmark partition."""

    return make_partition_key(
        SOURCE_WORLD_BANK_COMMODITIES,
        series.series_id,
        WORLD_CODE,
        series.indicator,
    )


def _month_number(value: date) -> int:
    return value.year * 12 + value.month


def _validate_dataset(
    dataset: CommodityDataset,
    *,
    as_of: date,
    allow_contraction: bool,
) -> pd.DataFrame:
    if not _SHA256_RE.fullmatch(dataset.workbook_sha256):
        raise ValueError("Pink Sheet workbook SHA-256 metadata is invalid")
    if not dataset.source_url.startswith("https://"):
        raise ValueError("Pink Sheet source URL must use HTTPS")
    if not dataset.catalogue:
        raise ValueError("Pink Sheet catalogue is empty")
    frame = dataset.observations
    if frame.empty:
        raise ValueError("Pink Sheet release snapshot is empty")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"Pink Sheet snapshot missing columns: {sorted(missing)}")

    catalogue_pairs: list[tuple[str, str]] = []
    worksheet_by_pair: dict[tuple[str, str], str] = {}
    for series in dataset.catalogue:
        strings = (
            series.indicator,
            series.series_id,
            series.benchmark,
            series.unit,
            series.frequency,
            series.price_basis,
            series.category,
            series.worksheet,
            series.series_kind,
        )
        if any(not isinstance(value, str) or not value.strip() for value in strings):
            raise ValueError("Pink Sheet catalogue contains empty metadata")
        expected_series_id = canonical_series_id(series.benchmark, series.worksheet)
        if series.series_id != expected_series_id:
            raise ValueError(
                "Pink Sheet catalogue series_id does not match its canonical benchmark alias"
            )
        expected_indicator = canonical_indicator(
            series.benchmark,
            series_kind=series.series_kind,
        )
        if series.indicator != expected_indicator:
            raise ValueError(
                "Pink Sheet catalogue indicator does not match its canonical benchmark alias"
            )
        if series.frequency != "monthly":
            raise ValueError("Pink Sheet catalogue contains a non-monthly series")
        if series.worksheet not in {MONTHLY_SHEET_NAME, MONTHLY_INDICES_SHEET_NAME}:
            raise ValueError("Pink Sheet catalogue contains an unsupported worksheet")
        if series.series_kind == "price":
            if series.worksheet != MONTHLY_SHEET_NAME:
                raise ValueError("Pink Sheet index worksheet contains a price series")
            if series.price_basis != "nominal_monthly_average":
                raise ValueError("Pink Sheet price catalogue basis is invalid")
            if _INDEX_UNIT_RE.fullmatch(series.unit):
                raise ValueError("Pink Sheet index-valued column is classified as a price")
        elif series.series_kind == "index":
            unit_match = _INDEX_UNIT_RE.fullmatch(series.unit)
            if unit_match is None:
                raise ValueError("Pink Sheet index catalogue unit must declare YYYY=100")
            if series.currency is not None:
                raise ValueError("Pink Sheet index catalogue must not declare a currency")
            expected_basis = f"nominal_usd_index_{unit_match.group(1)}_100"
            if series.price_basis != expected_basis:
                raise ValueError("Pink Sheet index catalogue basis disagrees with its unit")
        else:
            raise ValueError("Pink Sheet catalogue contains an unsupported series kind")
        pair = (series.indicator, series.series_id)
        catalogue_pairs.append(pair)
        worksheet_by_pair[pair] = series.worksheet
    if len(catalogue_pairs) != len(set(catalogue_pairs)):
        raise ValueError("Pink Sheet catalogue contains duplicate native series")
    if len({indicator for indicator, _series_id in catalogue_pairs}) != len(catalogue_pairs):
        raise ValueError("Pink Sheet catalogue contains duplicate indicators")
    if len({series_id for _indicator, series_id in catalogue_pairs}) != len(catalogue_pairs):
        raise ValueError("Pink Sheet catalogue contains duplicate native benchmark ids")

    known_benchmarks = {
        (series.worksheet, series.benchmark) for series in dataset.catalogue
    }
    quality_cells: set[tuple[str, str]] = set()
    for issue in dataset.quality_issues:
        if (issue.worksheet, issue.benchmark) not in known_benchmarks:
            raise ValueError("Pink Sheet quality issue does not match its catalogue")
        if not _CELL_REFERENCE_RE.fullmatch(issue.cell):
            raise ValueError("Pink Sheet quality issue has an invalid XLSX cell reference")
        quality_cell = (issue.worksheet, issue.cell)
        if quality_cell in quality_cells:
            raise ValueError("Pink Sheet quality metadata contains a duplicate cell")
        quality_cells.add(quality_cell)
        if issue.reason != "nonpositive_value":
            raise ValueError("Pink Sheet quality issue has an unsupported reason")
        if issue.date.day != 1 or issue.date > as_of:
            raise ValueError("Pink Sheet quality issue has an invalid observation date")
        if not math.isfinite(issue.raw_value) or issue.raw_value > 0:
            raise ValueError("Pink Sheet quality issue raw value is not nonpositive")

    work = frame.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    for column in ("country", "indicator", "source", "series_id"):
        if work[column].isna().any():
            raise ValueError(f"Pink Sheet snapshot {column} must not be missing")
        work[column] = work[column].astype(str).str.strip()
        if (work[column] == "").any():
            raise ValueError(f"Pink Sheet snapshot {column} must not be empty")
    if set(work["country"].drop_duplicates()) != {WORLD_CODE}:
        raise ValueError("Pink Sheet snapshot country must be WLD")
    if set(work["source"].drop_duplicates()) != {SOURCE_WORLD_BANK_COMMODITIES}:
        raise ValueError("Pink Sheet snapshot source does not match its catalogue")
    observed_pairs = set(
        work[["indicator", "series_id"]].drop_duplicates().itertuples(index=False, name=None)
    )
    if observed_pairs != set(catalogue_pairs):
        raise ValueError("Pink Sheet observations and catalogue native series disagree")
    try:
        parsed_dates = pd.to_datetime(work["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("Pink Sheet snapshot contains an invalid observation date") from exc
    if parsed_dates.isna().any():
        raise ValueError("Pink Sheet snapshot contains a missing observation date")
    work["date"] = parsed_dates.dt.date
    if work.duplicated(["indicator", "date", "source"], keep=False).any():
        raise ValueError("Pink Sheet snapshot contains duplicate monthly observations")
    if any(observed_on.day != 1 for observed_on in work["date"]):
        raise ValueError("Pink Sheet monthly observation dates must use month start")
    try:
        values = pd.to_numeric(work["value"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pink Sheet snapshot contains a non-numeric value") from exc
    if not np.isfinite(values).all():
        raise ValueError("Pink Sheet snapshot contains a non-finite value")
    if (values <= 0).any():
        raise ValueError("Pink Sheet snapshot contains a nonpositive price or index value")
    work["value"] = values

    catalogue_counts = Counter(series.worksheet for series in dataset.catalogue)
    expected_counts = {
        MONTHLY_SHEET_NAME: EXPECTED_MIN_PRICE_COLUMNS,
        MONTHLY_INDICES_SHEET_NAME: EXPECTED_MIN_INDEX_COLUMNS,
    }
    if not allow_contraction:
        for worksheet, minimum in expected_counts.items():
            actual = catalogue_counts[worksheet]
            if actual < minimum:
                raise ValueError(
                    f"Pink Sheet catalogue contracts {worksheet!r} to {actual} columns; "
                    f"expected at least {minimum}. Rerun with --allow-contraction only "
                    "after checking the publisher workbook"
                )
            actual_ids = {
                series.series_id
                for series in dataset.catalogue
                if series.worksheet == worksheet
            }
            missing_expected = sorted(
                EXPECTED_SERIES_IDS_BY_WORKSHEET[worksheet] - actual_ids
            )
            if missing_expected:
                raise ValueError(
                    f"Pink Sheet {worksheet} catalogue is missing expected series: "
                    f"{missing_expected}. Add a reviewed publisher-label alias or rerun "
                    "with --allow-contraction after verifying a genuine catalogue change"
                )

    pair_counts = Counter(
        work[["indicator", "series_id"]].itertuples(index=False, name=None)
    )
    if not allow_contraction:
        too_short = sorted(
            pair[1]
            for pair, count in pair_counts.items()
            if count < MIN_OBSERVATIONS_PER_SERIES
        )
        if too_short:
            raise ValueError(
                "Pink Sheet catalogue series fall below the minimum observation count "
                f"{MIN_OBSERVATIONS_PER_SERIES}: {too_short}"
            )

    as_of_month = date(as_of.year, as_of.month, 1)
    for worksheet in (MONTHLY_SHEET_NAME, MONTHLY_INDICES_SHEET_NAME):
        pairs = {
            pair for pair, pair_worksheet in worksheet_by_pair.items()
            if pair_worksheet == worksheet
        }
        if not pairs:
            continue
        periods = sorted(
            {
                row.date
                for row in work.itertuples(index=False)
                if (row.indicator, row.series_id) in pairs
            }
        )
        ordinals = [_month_number(period) for period in periods]
        if any(
            right - left != 1
            for left, right in zip(ordinals, ordinals[1:], strict=False)
        ):
            raise ValueError(f"Pink Sheet {worksheet} observations have a gap in monthly cadence")
        if periods[-1] > as_of_month:
            raise ValueError(
                f"Pink Sheet {worksheet} latest observation {periods[-1].isoformat()} "
                f"is after retrieval month {as_of_month.isoformat()}"
            )
        if not allow_contraction:
            if periods[0] != EXPECTED_HISTORY_START:
                raise ValueError(
                    f"Pink Sheet {worksheet} expected start "
                    f"{EXPECTED_HISTORY_START.isoformat()}, got {periods[0].isoformat()}"
                )
            if len(periods) < MIN_HISTORY_MONTHS:
                raise ValueError(
                    f"Pink Sheet {worksheet} has only {len(periods)} contiguous months; "
                    f"minimum is {MIN_HISTORY_MONTHS}"
                )
            lag_months = _month_number(as_of_month) - _month_number(periods[-1])
            if lag_months > MAX_LATEST_LAG_MONTHS:
                raise ValueError(
                    f"Pink Sheet {worksheet} latest month is stale by {lag_months} months; "
                    f"maximum is {MAX_LATEST_LAG_MONTHS}"
                )
    return work


def _stale_current_indicators(
    session: Session,
    dataset: CommodityDataset,
    frame: pd.DataFrame,
    *,
    allow_contraction: bool,
) -> set[str]:
    """Validate stored history/catalogue contraction and return retired projections."""

    incoming_indicators = {series.indicator for series in dataset.catalogue}
    current_indicators = set(
        session.scalars(
            select(Observation.indicator)
            .where(
                Observation.country == WORLD_CODE,
                Observation.source == SOURCE_WORLD_BANK_COMMODITIES,
            )
            .distinct()
        )
    )
    stale = current_indicators - incoming_indicators
    if stale and not allow_contraction:
        raise ValueError(
            "Pink Sheet catalogue contracts stored current series; missing indicators: "
            f"{sorted(stale)}. Add a reviewed label alias or rerun with "
            "--allow-contraction after verifying a genuine publisher removal"
        )

    incoming_counts = Counter(frame["indicator"])
    for series in dataset.catalogue:
        previous = latest_release(session, partition_key_for(series))
        incoming_count = incoming_counts[series.indicator]
        if (
            previous is not None
            and incoming_count < previous.row_count
            and not allow_contraction
        ):
            raise ValueError(
                f"Pink Sheet series {series.series_id} contracts complete history from "
                f"{previous.row_count} to {incoming_count} observations; rerun with "
                "--allow-contraction only after verifying the publisher removal"
            )
    return stale


def run_pipeline(
    *,
    source: PinkSheetSource | None = None,
    use_cache: bool = True,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
    allow_contraction: bool = False,
) -> dict[str, object]:
    """Atomically store every native series in one validated workbook release group."""

    adapter = source or PinkSheetSource()
    db_engine = engine or make_engine()
    init_db(db_engine)
    try:
        dataset = adapter.fetch(use_cache=use_cache)
        # Capture the conservative availability clock only after the workbook
        # has actually been obtained, unless a deterministic test clock is set.
        run_at = retrieved_at or datetime.now(UTC)
        frame = _validate_dataset(
            dataset,
            as_of=run_at.date(),
            allow_contraction=allow_contraction,
        )
    except Exception as exc:  # noqa: BLE001 - CLI reports fail-closed source errors
        logger.exception("Failed World Bank Pink Sheet refresh: %s", exc)
        return {"error": str(exc)}

    summary: dict[str, object] = {
        "source": SOURCE_WORLD_BANK_COMMODITIES,
        "rows": len(frame),
        "series_total": len(dataset.catalogue),
        "curated_series": sum(series.curated for series in dataset.catalogue),
        "workbook_sha256": dataset.workbook_sha256,
        "artifact_path": str(dataset.artifact_path) if dataset.artifact_path else None,
        "catalogue_path": str(dataset.catalogue_path) if dataset.catalogue_path else None,
        "quarantined_cells": len(dataset.quality_issues),
        "created_releases": 0,
        "inserted": 0,
        "skipped": 0,
        "removed": 0,
        "series": {},
    }
    series_summary: dict[str, dict[str, object]] = {}
    active_series: CommoditySeries | None = None
    try:
        # Helpers commit each partition for their ordinary standalone callers.
        # rollback_only joins those logical commits to this outer connection
        # transaction, so the database publishes all 87 partitions or none.
        with (
            db_engine.begin() as connection,
            Session(
                bind=connection,
                expire_on_commit=False,
                join_transaction_mode="rollback_only",
            ) as session,
        ):
                stale_indicators = _stale_current_indicators(
                    session,
                    dataset,
                    frame,
                    allow_contraction=allow_contraction,
                )
                if stale_indicators:
                    retired = session.execute(
                        delete(Observation).where(
                            Observation.country == WORLD_CODE,
                            Observation.source == SOURCE_WORLD_BANK_COMMODITIES,
                            Observation.indicator.in_(stale_indicators),
                        )
                    )
                    retired_rows = max(retired.rowcount or 0, 0)
                    summary["removed"] += retired_rows
                    summary["retired_series"] = sorted(stale_indicators)

                for series in dataset.catalogue:
                    active_series = series
                    snapshot = frame.loc[
                        (frame["indicator"] == series.indicator)
                        & (frame["series_id"] == series.series_id)
                    ].reset_index(drop=True)
                    result = ingest_release_snapshot(
                        session,
                        snapshot,
                        ReleaseMeta(
                            partition_key=partition_key_for(series),
                            source_family=SOURCE_WORLD_BANK_COMMODITIES,
                            available_at=run_at,
                            retrieved_at=run_at,
                            vintage_label=pink_sheet_vintage_label(
                                dataset.workbook_sha256
                            ),
                            source_url=dataset.source_url,
                            projection=ProjectionScope(
                                country=WORLD_CODE,
                                indicator=series.indicator,
                                sources=(SOURCE_WORLD_BANK_COMMODITIES,),
                            ),
                        ),
                    )
                    summary["created_releases"] += int(result.created)
                    summary["inserted"] += result.changed_rows
                    summary["skipped"] += result.unchanged_rows
                    summary["removed"] += result.removed_rows
                    series_summary[series.indicator] = {
                        "benchmark": series.benchmark,
                        "unit": series.unit,
                        "category": series.category,
                        "curated": series.curated,
                        "rows": result.row_count,
                        "release_id": result.release_id,
                        "created": result.created,
                        "inserted": result.changed_rows,
                        "skipped": result.unchanged_rows,
                        "removed": result.removed_rows,
                    }
    except Exception as exc:  # noqa: BLE001 - workbook transaction must fail closed
        failed_id = active_series.series_id if active_series is not None else None
        logger.exception("Rolled back complete Pink Sheet workbook at %s: %s", failed_id, exc)
        summary.update(
            {
                "error": str(exc),
                "rolled_back": True,
                "created_releases": 0,
                "inserted": 0,
                "skipped": 0,
                "removed": 0,
                "series": {},
            }
        )
        if failed_id is not None:
            summary["failed_series"] = 1
            summary["failed_series_id"] = failed_id
        return summary
    summary["series"] = series_summary
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch the complete World Bank Pink Sheet monthly commodity history."
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the rolling cache and download the official workbook.",
    )
    parser.add_argument(
        "--allow-contraction",
        action="store_true",
        help=(
            "Allow a reviewed publisher catalogue/history contraction and retire "
            "omitted current projections. Structural/date/value checks still apply."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    summary = run_pipeline(
        use_cache=not args.no_cache,
        allow_contraction=args.allow_contraction,
    )
    if "error" in summary:
        print(f"World Bank Pink Sheet failed: {summary['error']}")
        return 1
    print(
        "World Bank Pink Sheet: "
        f"{summary['rows']} observations / {summary['series_total']} native series; "
        f"{summary['created_releases']} new releases."
    )
    print(f"Archived workbook: {summary['artifact_path']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
