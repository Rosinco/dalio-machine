"""ETL pipeline: Riksbank SWEA -> immutable release ledger + SQLite."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dalio.data_sources.riksbank import (
    RIKSBANK_SERIES,
    RIKSBANK_SOURCE,
    RiksbankSeriesSpec,
    RiksbankSource,
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


def _validate_snapshot(frame: pd.DataFrame, spec: RiksbankSeriesSpec) -> None:
    if frame.empty:
        raise ValueError("Riksbank returned an empty snapshot; release not recorded")

    required = {"country", "indicator", "source", "series_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Riksbank snapshot missing partition columns: {sorted(missing)}")

    expected = {
        "country": spec.country,
        "indicator": spec.indicator,
        "source": RIKSBANK_SOURCE,
        "series_id": spec.series_id,
    }
    for column, expected_value in expected.items():
        actual = frame[column].drop_duplicates().tolist()
        if actual != [expected_value]:
            if column == "series_id":
                raise ValueError(
                    f"Riksbank snapshot expected native series {expected_value!r}, got {actual!r}"
                )
            raise ValueError(
                f"Riksbank snapshot expected {column} {expected_value!r}, got {actual!r}"
            )


def _ingest_riksbank_release(
    session: Session,
    frame: pd.DataFrame,
    *,
    spec: RiksbankSeriesSpec,
    retrieved_at: datetime,
    source_url: str,
) -> IngestResult:
    """Append one complete native SWEA-series snapshot and refresh its projection."""
    _validate_snapshot(frame, spec)
    return ingest_release_snapshot(
        session,
        frame,
        ReleaseMeta(
            partition_key=make_partition_key(
                RIKSBANK_SOURCE,
                spec.series_id,
                spec.country,
                spec.indicator,
            ),
            source_family=RIKSBANK_SOURCE,
            # SWEA observations do not carry a release timestamp. Retrieval is
            # therefore the conservative point-in-time availability clock.
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            source_url=source_url,
            projection=ProjectionScope(
                country=spec.country,
                indicator=spec.indicator,
                sources=(RIKSBANK_SOURCE,),
            ),
        ),
    )


def run_pipeline(
    specs: tuple[RiksbankSeriesSpec, ...] = RIKSBANK_SERIES,
    source: RiksbankSource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
    pacing_seconds: float | None = None,
) -> dict[str, dict]:
    """Fetch and independently ingest each requested native SWEA series.

    Every partition has a fixed canonical window: its catalogue history start
    through the retrieval date. A failure is isolated to that series. Empty or
    malformed frames never replace a previously valid current projection.
    """
    src = source or RiksbankSource()
    run_at = retrieved_at or datetime.now(UTC)
    end = run_at.date()

    delay = (
        float(src.recommended_pacing_seconds) if pacing_seconds is None else float(pacing_seconds)
    )
    if not math.isfinite(delay) or delay < 0:
        raise ValueError("pacing_seconds must be a finite non-negative number")

    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    summary: dict[str, dict] = {}

    with session_factory() as session:
        for index, spec in enumerate(specs):
            if index and delay:
                time.sleep(delay)

            key = f"{spec.country}/{spec.indicator}"
            try:
                series_start = spec.history_start
                frame = src.fetch(
                    spec,
                    from_date=series_start,
                    to_date=end,
                    use_cache=use_cache,
                )
                source_url = RiksbankSource.url_for(
                    spec,
                    from_date=series_start,
                    to_date=end,
                )
                result = _ingest_riksbank_release(
                    session,
                    frame,
                    spec=spec,
                    retrieved_at=run_at,
                    source_url=source_url,
                )
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(frame),
                    "inserted": result.changed_rows,
                    "skipped": result.unchanged_rows,
                    "removed": result.removed_rows,
                    "release_id": result.release_id,
                    "release_created": result.created,
                    "series_id": spec.series_id,
                }
                logger.info(
                    "Fetched %s (%s): %d rows (%d new/updated, %d unchanged)",
                    key,
                    spec.series_id,
                    len(frame),
                    result.changed_rows,
                    result.unchanged_rows,
                )
            except Exception as exc:  # noqa: BLE001 - isolate every native series
                logger.exception("Failed %s (%s): %s", key, spec.series_id, exc)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "series_id": spec.series_id,
                    "error": str(exc),
                }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch official Swedish rates and SEK exchange rates from Riksbank SWEA "
            "into the immutable release ledger."
        )
    )
    parser.add_argument(
        "indicators",
        nargs="*",
        help="Optional indicators (for example policy_rate yield_10y). Default: all eight.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk response cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    by_indicator = {spec.indicator: spec for spec in RIKSBANK_SERIES}
    if args.indicators:
        unknown = sorted(set(args.indicators) - set(by_indicator))
        if unknown:
            parser.error(f"unknown indicator(s): {', '.join(unknown)}")
        specs = tuple(by_indicator[indicator] for indicator in args.indicators)
    else:
        specs = RIKSBANK_SERIES

    print("dalio-fetch-riksbank — official Swedish rates and SEK FX")
    print(f"  {len(specs)} complete series histories through today")
    summary = run_pipeline(
        specs,
        use_cache=not args.no_cache,
    )

    print("\nSummary:")
    failed = 0
    for stats in summary.values():
        if "error" in stats:
            print(f"  ✗ {stats['indicator']:<16} {stats['series_id']:<12} {stats['error'][:80]}")
            failed += 1
        else:
            print(
                f"  ✓ {stats['indicator']:<16} {stats['series_id']:<12} "
                f"{stats['rows']:>5} rows ({stats['inserted']} new/updated)"
            )
    print()
    if failed:
        print(f"⚠️  {failed} series failed — see logs above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
