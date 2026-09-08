"""ETL pipeline: fetch FRED → immutable release + current SQLite projection."""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import insert, select, update
from sqlalchemy.orm import Session

from dalio.data_sources.fred import (
    TIER_1_SERIES,
    FredSeriesSpec,
    FredSource,
    specs_for_countries,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseMeta,
    ingest_release_snapshot,
    make_partition_key,
)

logger = logging.getLogger(__name__)


_KEY = ["country", "indicator", "date", "source"]
_CHUNK = 5000


def upsert_observations(session: Session, df: pd.DataFrame) -> tuple[int, int]:
    """Legacy-compatible set-based upsert into ``observations``.

    Kept for callers that depend on the original helper and its return contract.
    New provider pipelines must use ``ingest_release_snapshot`` so revisions and
    omissions remain reproducible rather than being overwritten here.

    Returns ``(inserted, skipped)`` where *inserted* counts new **and changed**
    rows and *skipped* counts rows whose value is unchanged — the same contract
    as the original row-by-row implementation for duplicate-free frames, but
    with one SELECT per batch and executemany INSERT/UPDATE instead of one
    SELECT per row (20k rows: ~8 s → sub-second).

    Duplicate keys inside ``df`` collapse to the last occurrence before
    counting, so ``inserted + skipped`` equals the number of **distinct** keys,
    not ``len(df)``.

    Atomic per call: on any failure the whole batch is rolled back (nothing
    partial is left pending for a later ``commit()`` to sweep in) and the
    exception is re-raised.
    """
    if df.empty:
        session.commit()
        return 0, 0

    try:
        work = df.loc[:, [*_KEY, "value", "series_id"]].copy()
        work["date"] = pd.to_datetime(work["date"]).dt.date
        work["value"] = work["value"].astype(float)
        work = work.drop_duplicates(subset=_KEY, keep="last")

        existing = pd.DataFrame(
            session.execute(
                select(
                    Observation.id, Observation.country, Observation.indicator,
                    Observation.date, Observation.source, Observation.value,
                ).where(
                    Observation.country.in_(work["country"].unique().tolist()),
                    Observation.indicator.in_(work["indicator"].unique().tolist()),
                    Observation.source.in_(work["source"].unique().tolist()),
                )
            ).all(),
            columns=["id", *_KEY[:3], "source", "old_value"],
        )

        if existing.empty:
            merged = work.assign(id=float("nan"), old_value=float("nan"))
        else:
            merged = work.merge(existing, on=_KEY, how="left")

        is_new = merged["id"].isna()
        is_changed = ~is_new & (merged["old_value"].astype(float) != merged["value"])

        new_rows = merged.loc[is_new, [*_KEY, "value", "series_id"]].to_dict("records")
        changed_rows = [
            {"id": int(i), "value": float(v)}
            for i, v in zip(merged.loc[is_changed, "id"], merged.loc[is_changed, "value"], strict=True)
        ]

        for start in range(0, len(new_rows), _CHUNK):
            session.execute(insert(Observation), new_rows[start:start + _CHUNK])
        for start in range(0, len(changed_rows), _CHUNK):
            session.execute(update(Observation), changed_rows[start:start + _CHUNK])
        session.commit()
    except Exception:
        session.rollback()
        raise

    inserted = len(new_rows) + len(changed_rows)
    skipped = int(len(merged) - inserted)
    return inserted, skipped


def run_pipeline(
    specs: Iterable[FredSeriesSpec],
    source: FredSource | None = None,
    *,
    retrieved_at: datetime | None = None,
) -> dict[str, dict]:
    src = source or FredSource()
    run_at = retrieved_at or datetime.now(UTC)
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for spec in specs:
            key = f"{spec.country}/{spec.indicator}"
            try:
                df = src.fetch(spec)
                result = ingest_release_snapshot(
                    session,
                    df,
                    ReleaseMeta(
                        partition_key=make_partition_key(
                            "FRED", spec.series_id, spec.country, spec.indicator,
                        ),
                        source_family="FRED",
                        available_at=run_at,
                        retrieved_at=run_at,
                        projection=ProjectionScope(
                            country=spec.country,
                            indicator=spec.indicator,
                            sources=("FRED",),
                        ),
                    ),
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
                    "series_id": spec.series_id,
                    "error": str(e),
                }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch FRED short-term cycle indicators and upsert into SQLite. "
            "Fetches all Tier-1 countries by default; pass country codes to subset."
        )
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="Optional country codes (US, CN, EU, UK, JP, SE). Default: all Tier-1.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    load_dotenv()

    specs = specs_for_countries(tuple(args.countries) if args.countries else None)
    label = (
        f"{', '.join(args.countries)} only" if args.countries
        else f"all Tier-1 ({len({s.country for s in TIER_1_SERIES})} countries)"
    )
    print(f"dalio-fetch-fred — short-term cycle bundle ({label})")
    print(f"  ({len(specs)} series)")

    summary = run_pipeline(specs)

    print("\nSummary:")
    failed = 0
    by_country: dict[str, list[tuple[str, dict]]] = {}
    for key, stats in summary.items():
        by_country.setdefault(stats["country"], []).append((key, stats))

    for country, items in sorted(by_country.items()):
        print(f"\n  [{country}]")
        for _key, stats in items:
            if "error" in stats:
                print(f"    ✗ {stats['indicator']:<20}  ({stats.get('series_id', '?')}): {stats['error'][:80]}")
                failed += 1
            else:
                print(
                    f"    ✓ {stats['indicator']:<20}  ({stats['series_id']}): "
                    f"{stats['rows']:>6} rows ({stats['inserted']} new/updated)"
                )
    print()
    if failed:
        print(f"⚠️  {failed} series failed — see logs above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
