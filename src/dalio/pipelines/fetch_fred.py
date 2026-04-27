"""ETL pipeline: fetch FRED series → upsert into SQLite."""
from __future__ import annotations

import logging
import sys
from collections.abc import Iterable

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.data_sources.fred import US_SHORT_TERM_SERIES, FredSeriesSpec, FredSource
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)


def upsert_observations(session: Session, df: pd.DataFrame) -> tuple[int, int]:
    inserted = 0
    skipped = 0
    for row in df.itertuples(index=False):
        existing = session.execute(
            select(Observation).where(
                Observation.country == row.country,
                Observation.indicator == row.indicator,
                Observation.date == row.date,
                Observation.source == row.source,
            )
        ).scalar_one_or_none()
        if existing is not None:
            if existing.value != float(row.value):
                existing.value = float(row.value)
                inserted += 1
            else:
                skipped += 1
            continue
        session.add(Observation(
            country=row.country,
            indicator=row.indicator,
            date=row.date,
            value=float(row.value),
            source=row.source,
            series_id=row.series_id,
        ))
        inserted += 1
    session.commit()
    return inserted, skipped


def run_pipeline(
    specs: Iterable[FredSeriesSpec],
    source: FredSource | None = None,
) -> dict[str, dict]:
    src = source or FredSource()
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for spec in specs:
            key = f"{spec.country}/{spec.indicator}"
            try:
                df = src.fetch(spec)
                ins, skp = upsert_observations(session, df)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(df),
                    "inserted": ins,
                    "skipped": skp,
                    "series_id": spec.series_id,
                }
                logger.info(
                    "Fetched %s: %d rows (%d new/updated, %d unchanged)",
                    key, len(df), ins, skp,
                )
            except Exception as e:  # noqa: BLE001 — collect per-series, never crash batch
                logger.exception("Failed %s: %s", key, e)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "error": str(e),
                }
    return summary


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv()

    print("dalio-fetch-fred — fetching US short-term cycle bundle")
    print(f"  ({len(US_SHORT_TERM_SERIES)} series)")
    summary = run_pipeline(US_SHORT_TERM_SERIES)

    print("\nSummary:")
    failed = 0
    for key, stats in summary.items():
        if "error" in stats:
            print(f"  ✗ {key}: {stats['error']}")
            failed += 1
        else:
            print(
                f"  ✓ {key}: {stats['rows']:>6} rows "
                f"({stats['inserted']} new/updated, {stats['skipped']} unchanged)"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
