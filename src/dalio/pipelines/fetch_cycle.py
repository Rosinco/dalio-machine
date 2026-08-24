"""ETL pipeline (slice 26): fresh cycle inputs from keyless SDMX sources.

FRED's OECD-MEI mirrors for the non-US cycle countries are discontinued or a
year stale (JP CPI gone since 2021, IN policy rate frozen in 2022, CN
unemployment in 2011). This pipeline stores the same indicators from sources
that publish them currently — FRED rows are left in place; the classifiers
take the latest observation regardless of source.

    dalio-fetch-cycle                       # cpi cbpol qna lfs for the 8 cycle countries
    dalio-fetch-cycle --only cpi qna
    dalio-fetch-cycle --countries JP UK
    dalio-fetch-cycle --no-cache

| key   | indicator           | source     | covers                       |
|-------|---------------------|------------|------------------------------|
| cpi   | cpi_yoy             | IMF_CPI    | all but EU (no IMF entity)   |
| cbpol | policy_rate         | BIS_CBPOL  | all 8 (EU = XM)              |
| qna   | real_gdp_yoy        | OECD_QNA   | all 8 (EU = EA)              |
| lfs   | unemployment_rate   | OECD_LFS   | US EU UK JP SE — not CN/IN/BR |
"""
from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence

from dotenv import load_dotenv

from dalio.countries import CYCLE_COUNTRIES, Country, get_country
from dalio.data_sources.bis import BisSource, PolicyRateSpec
from dalio.data_sources.imf_cpi import ImfCpiSource
from dalio.data_sources.oecd import LFS_UNEMPLOYMENT, QNA_GDP_GROWTH, OecdSource
from dalio.pipelines.fetch_fred import upsert_observations
from dalio.storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)

FLOWS: tuple[str, ...] = ("cpi", "cbpol", "qna", "lfs")


def run_pipeline(
    flows: Sequence[str] = FLOWS,
    countries: Sequence[Country] = CYCLE_COUNTRIES,
    *,
    cpi_source: ImfCpiSource | None = None,
    bis_source: BisSource | None = None,
    oecd_source: OecdSource | None = None,
    use_cache: bool = True,
    engine=None,
) -> dict[str, dict]:
    """Fetch each requested flow and upsert. One failing flow (or one failing
    policy-rate country) never fails the batch — it lands in the summary."""
    unknown = set(flows) - set(FLOWS)
    if unknown:
        raise ValueError(f"Unknown flow(s): {sorted(unknown)}; choose from {FLOWS}")
    engine = engine or make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)
    summary: dict[str, dict] = {}

    def _store(session, key: str, df, indicator: str, source: str) -> None:
        ins, skp = upsert_observations(session, df)
        summary[key] = {"indicator": indicator, "source": source, "rows": len(df),
                        "inserted": ins, "skipped": skp,
                        "countries": sorted(set(df["country"])) if len(df) else []}
        logger.info("Fetched %s: %d rows (%d new/updated)", key, len(df), ins)

    with session_factory() as session:
        if "cpi" in flows:
            src = cpi_source or ImfCpiSource()
            try:
                _store(session, "cpi/cpi_yoy", src.fetch(countries, use_cache=use_cache),
                       "cpi_yoy", "IMF_CPI")
            except Exception as e:  # noqa: BLE001
                logger.exception("Failed cpi: %s", e)
                summary["cpi/cpi_yoy"] = {"indicator": "cpi_yoy", "source": "IMF_CPI", "error": str(e)}
        if "cbpol" in flows:
            src = bis_source or BisSource()
            for c in countries:
                key = f"cbpol/policy_rate/{c.iso2}"
                try:
                    _store(session, key, src.fetch_policy_rate(PolicyRateSpec(c.iso2), use_cache=use_cache),
                           "policy_rate", "BIS_CBPOL")
                except Exception as e:  # noqa: BLE001
                    logger.exception("Failed %s: %s", key, e)
                    summary[key] = {"indicator": "policy_rate", "source": "BIS_CBPOL", "error": str(e)}
        if "qna" in flows or "lfs" in flows:
            src = oecd_source or OecdSource()
            for flow in (QNA_GDP_GROWTH, LFS_UNEMPLOYMENT):
                if flow.name not in flows:
                    continue
                key = f"{flow.name}/{flow.indicator}"
                try:
                    _store(session, key, src.fetch(flow, countries, use_cache=use_cache),
                           flow.indicator, flow.source)
                except Exception as e:  # noqa: BLE001
                    logger.exception("Failed %s: %s", key, e)
                    summary[key] = {"indicator": flow.indicator, "source": flow.source, "error": str(e)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--only", nargs="+", choices=FLOWS, default=list(FLOWS),
                        help="Flows to run (default: all).")
    parser.add_argument("--countries", nargs="+", default=None,
                        help="ISO2 subset of the cycle basket (default: all 8).")
    parser.add_argument("--no-cache", action="store_true", help="Bypass the on-disk cache.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    countries = CYCLE_COUNTRIES
    if args.countries:
        wanted = {c.upper() for c in args.countries}
        countries = tuple(get_country(c) for c in wanted if get_country(c).has_cycle_wiring)
    print(f"dalio-fetch-cycle — flows {', '.join(args.only)}; "
          f"{len(countries)} countries: {' '.join(c.iso2 for c in countries)}")

    summary = run_pipeline(args.only, countries, use_cache=not args.no_cache)

    print("\nSummary:")
    failed = 0
    for key, stats in summary.items():
        if "error" in stats:
            failed += 1
            print(f"  ✗ {key:<28} {stats['error'][:90]}")
        else:
            print(f"  ✓ {key:<28} {stats['rows']:>5} rows ({stats['inserted']} new/updated) "
                  f"{' '.join(stats['countries'])}")
    if failed:
        print(f"\n⚠️  {failed} flow(s) failed — see logs above.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
