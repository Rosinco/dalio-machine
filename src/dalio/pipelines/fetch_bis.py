"""ETL pipeline: fetch BIS Total Credit + DSR → release ledger + SQLite."""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dalio.data_sources.bis import (
    ALL_DSR,
    ALL_TOTAL_CREDIT,
    TIER_1_DSR,
    TIER_1_TOTAL_CREDIT,
    BisSource,
    DsrSpec,
    TotalCreditSpec,
)
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.releases import (
    IngestResult,
    ProjectionScope,
    ReleaseMeta,
    ingest_release_snapshot,
    make_partition_key,
)

logger = logging.getLogger(__name__)


def _single_native_series(frame: pd.DataFrame, source_family: str) -> str:
    """Return the native BIS series id, rejecting incomplete/mixed partitions."""
    if frame.empty:
        raise ValueError("BIS returned an empty snapshot; release not recorded")

    missing = {"source", "series_id"} - set(frame.columns)
    if missing:
        raise ValueError(f"BIS snapshot missing partition columns: {sorted(missing)}")

    sources = frame["source"].drop_duplicates().tolist()
    if sources != [source_family]:
        raise ValueError(f"BIS snapshot must contain source {source_family!r}, got {sources!r}")

    series_ids = frame["series_id"].drop_duplicates().tolist()
    if len(series_ids) != 1 or pd.isna(series_ids[0]):
        raise ValueError("BIS snapshot must contain exactly one non-null native series_id")
    series_id = str(series_ids[0]).strip()
    if not series_id:
        raise ValueError("BIS snapshot native series_id must not be empty")
    return series_id


def _ingest_bis_release(
    session: Session,
    frame: pd.DataFrame,
    *,
    country: str,
    indicator: str,
    source_family: str,
    retrieved_at: datetime,
) -> tuple[IngestResult, str]:
    """Append one complete native-series snapshot and refresh its projection."""
    series_id = _single_native_series(frame, source_family)
    result = ingest_release_snapshot(
        session,
        frame,
        ReleaseMeta(
            partition_key=make_partition_key(
                source_family,
                series_id,
                country,
                indicator,
            ),
            source_family=source_family,
            # BIS does not expose a reliable release timestamp in these frames.
            # Retrieval time is therefore the conservative point-in-time clock.
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            projection=ProjectionScope(
                country=country,
                indicator=indicator,
                sources=(source_family,),
            ),
        ),
    )
    return result, series_id


def run_pipeline(
    tc_specs: tuple[TotalCreditSpec, ...] = TIER_1_TOTAL_CREDIT,
    dsr_specs: tuple[DsrSpec, ...] = TIER_1_DSR,
    source: BisSource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict]:
    src = source or BisSource()
    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for spec in tc_specs:
            key = f"{spec.country}/{spec.indicator}"
            try:
                df = src.fetch_total_credit(spec, use_cache=use_cache)
                result, series_id = _ingest_bis_release(
                    session,
                    df,
                    country=spec.country,
                    indicator=spec.indicator,
                    source_family="BIS_TC",
                    retrieved_at=run_at,
                )
                ins, skp = result.changed_rows, result.unchanged_rows
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(df),
                    "inserted": ins,
                    "skipped": skp,
                    "removed": result.removed_rows,
                    "release_id": result.release_id,
                    "release_created": result.created,
                    "series_id": series_id,
                }
                logger.info(
                    "Fetched %s: %d rows (%d new/updated, %d unchanged)",
                    key,
                    len(df),
                    ins,
                    skp,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed %s: %s", key, e)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "error": str(e),
                }

        for spec in dsr_specs:
            key = f"{spec.country}/{spec.indicator}"
            try:
                df = src.fetch_dsr(spec, use_cache=use_cache)
                result, series_id = _ingest_bis_release(
                    session,
                    df,
                    country=spec.country,
                    indicator=spec.indicator,
                    source_family="BIS_DSR",
                    retrieved_at=run_at,
                )
                ins, skp = result.changed_rows, result.unchanged_rows
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(df),
                    "inserted": ins,
                    "skipped": skp,
                    "removed": result.removed_rows,
                    "release_id": result.release_id,
                    "release_created": result.created,
                    "series_id": series_id,
                }
                logger.info(
                    "Fetched %s: %d rows (%d new/updated, %d unchanged)",
                    key,
                    len(df),
                    ins,
                    skp,
                )
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed %s: %s", key, e)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "error": str(e),
                }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch BIS Total Credit + DSR series for the long-term debt cycle "
            "and upsert into SQLite."
        )
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="Optional country codes (US CN EU UK JP SE). Default: all Tier-1.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass on-disk cache and force re-download.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if args.countries:
        wanted = {c.upper() for c in args.countries}
        tc_specs = tuple(s for s in ALL_TOTAL_CREDIT if s.country in wanted)
        dsr_specs = tuple(s for s in ALL_DSR if s.country in wanted)
        label = f"{', '.join(args.countries)} only"
    else:
        tc_specs = ALL_TOTAL_CREDIT
        dsr_specs = ALL_DSR
        label = f"all countries ({len({s.country for s in ALL_TOTAL_CREDIT})} countries)"

    print(f"dalio-fetch-bis — long-term debt cycle bundle ({label})")
    print(f"  ({len(tc_specs)} Total Credit + {len(dsr_specs)} DSR series)")

    summary = run_pipeline(tc_specs, dsr_specs, use_cache=not args.no_cache)

    by_country: dict[str, list] = {}
    for key, stats in summary.items():
        by_country.setdefault(stats["country"], []).append((key, stats))

    print("\nSummary:")
    failed = 0
    for country, items in sorted(by_country.items()):
        print(f"\n  [{country}]")
        for _key, stats in items:
            if "error" in stats:
                print(f"    ✗ {stats['indicator']:<24}  {stats['error'][:80]}")
                failed += 1
            else:
                print(
                    f"    ✓ {stats['indicator']:<24}  "
                    f"{stats['rows']:>4} rows ({stats['inserted']} new/updated)"
                )
    print()
    if failed:
        print(f"⚠️  {failed} series failed — see logs above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
