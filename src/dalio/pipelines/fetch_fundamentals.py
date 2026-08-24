"""ETL pipeline: fetch World Fundamentals Map indicators → upsert into SQLite.

Sources are pulled per indicator for the whole basket in one paginated call
(World Bank). IMF WEO (slice 20) and the BIS extension (slice 22) plug into
the same ``run_pipeline`` via the ``sources`` tuple.

    dalio-fetch-fundamentals                # everything implemented
    dalio-fetch-fundamentals --only wb      # one source family
    dalio-fetch-fundamentals --countries SE KR
    dalio-fetch-fundamentals --no-cache
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from dotenv import load_dotenv

from dalio.countries import COUNTRIES, Country, get_country
from dalio.data_sources.worldbank import WB_FUNDAMENTALS, WbIndicatorSpec, WorldBankSource
from dalio.pipelines.fetch_fred import upsert_observations
from dalio.storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)

IMPLEMENTED_SOURCES: tuple[str, ...] = ("wb",)
PLANNED_SOURCES: dict[str, str] = {
    "imf": "IMF DataMapper (WEO history + forecasts) — slice 20",
    "bis": "BIS DSR + private credit for Tier-3 players — slice 22",
}


def run_pipeline(
    sources: Sequence[str] = IMPLEMENTED_SOURCES,
    countries: Sequence[Country] | None = None,
    use_cache: bool = True,
    wb_source: WorldBankSource | None = None,
    wb_specs: Sequence[WbIndicatorSpec] = WB_FUNDAMENTALS,
) -> dict[str, dict]:
    """Fetch every spec of every requested source and upsert. Returns a
    per-spec summary keyed ``"{source}/{indicator}"``; errors are collected,
    never raised, so one dead series cannot sink the batch."""
    basket = tuple(countries) if countries else COUNTRIES
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for source in sources:
            if source == "wb":
                wb = wb_source or WorldBankSource()
                for spec in wb_specs:
                    key = f"wb/{spec.indicator}"
                    try:
                        df = wb.fetch(spec, basket, use_cache=use_cache)
                        ins, skp = upsert_observations(session, df)
                        summary[key] = {
                            "source": "wb", "indicator": spec.indicator, "series_id": spec.wb_code,
                            "rows": len(df), "inserted": ins, "skipped": skp,
                            "countries": int(df["country"].nunique()) if not df.empty else 0,
                        }
                        logger.info("Fetched %s: %d rows / %d countries (%d new/updated)",
                                    key, len(df), summary[key]["countries"], ins)
                    except Exception as e:  # noqa: BLE001 — collect per-series
                        logger.exception("Failed %s: %s", key, e)
                        summary[key] = {"source": "wb", "indicator": spec.indicator,
                                        "series_id": spec.wb_code, "error": str(e)}
            elif source in PLANNED_SOURCES:
                summary[f"{source}/*"] = {"source": source, "indicator": "*",
                                          "error": f"not implemented yet: {PLANNED_SOURCES[source]}"}
            else:
                summary[f"{source}/*"] = {"source": source, "indicator": "*",
                                          "error": f"unknown source {source!r}"}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fetch World Fundamentals Map indicators for the 22-player basket."
    )
    parser.add_argument("--only", choices=[*IMPLEMENTED_SOURCES, *PLANNED_SOURCES],
                        nargs="*", help="Restrict to these source families.")
    parser.add_argument("--countries", nargs="*", help="ISO2 codes to subset (default: all).")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the on-disk cache.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    sources = tuple(args.only) if args.only else IMPLEMENTED_SOURCES
    countries = tuple(get_country(c) for c in args.countries) if args.countries else None
    print(f"dalio-fetch-fundamentals — sources {', '.join(sources)}; "
          f"{len(countries or COUNTRIES)} players")

    summary = run_pipeline(sources, countries, use_cache=not args.no_cache)

    print("\nSummary:")
    failed = 0
    for key, stats in summary.items():
        if "error" in stats:
            print(f"  ✗ {key:<28} {stats['error'][:90]}")
            failed += 1
        else:
            print(f"  ✓ {key:<28} {stats['rows']:>6} rows · {stats['countries']:>2} countries "
                  f"· {stats['inserted']} new/updated · {stats['skipped']} unchanged")
    print()
    if failed:
        print(f"⚠️  {failed} series failed — see logs above.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
