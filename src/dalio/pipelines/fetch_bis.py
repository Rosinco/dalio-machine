"""ETL pipeline: fetch BIS Total Credit + DSR → upsert into SQLite."""
from __future__ import annotations

import argparse
import logging
import sys

from dotenv import load_dotenv

from dalio.data_sources.bis import (
    ALL_DSR,
    ALL_TOTAL_CREDIT,
    TIER_1_DSR,
    TIER_1_TOTAL_CREDIT,
    BisSource,
    DsrSpec,
    TotalCreditSpec,
)
from dalio.pipelines.fetch_fred import upsert_observations
from dalio.storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)


def run_pipeline(
    tc_specs: tuple[TotalCreditSpec, ...] = TIER_1_TOTAL_CREDIT,
    dsr_specs: tuple[DsrSpec, ...] = TIER_1_DSR,
    source: BisSource | None = None,
    use_cache: bool = True,
) -> dict[str, dict]:
    src = source or BisSource()
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for spec in tc_specs:
            key = f"{spec.country}/{spec.indicator}"
            try:
                df = src.fetch_total_credit(spec, use_cache=use_cache)
                ins, skp = upsert_observations(session, df)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(df),
                    "inserted": ins,
                    "skipped": skp,
                }
                logger.info("Fetched %s: %d rows (%d new/updated)", key, len(df), ins)
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
                ins, skp = upsert_observations(session, df)
                summary[key] = {
                    "country": spec.country,
                    "indicator": spec.indicator,
                    "rows": len(df),
                    "inserted": ins,
                    "skipped": skp,
                }
                logger.info("Fetched %s: %d rows (%d new/updated)", key, len(df), ins)
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
