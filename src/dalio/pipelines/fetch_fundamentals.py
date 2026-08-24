"""ETL pipeline: fetch World Fundamentals Map indicators → upsert into SQLite.

Sources are pulled per indicator for the whole basket in one paginated call
(World Bank). IMF WEO (slice 20), the BIS extension (slice 22) and IMF IMTS
bilateral trade (slice 24) plug into the same ``run_pipeline`` via the
``sources`` tuple.

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
from sqlalchemy import delete

from dalio.countries import COUNTRIES, Country, get_country
from dalio.data_sources.bis import (
    TIER_3_DSR,
    TIER_3_PRIVATE_CREDIT,
    BisSource,
    DsrSpec,
    TotalCreditSpec,
)
from dalio.data_sources.imf_datamapper import (
    IMF_FUNDAMENTALS,
    SOURCE_FORECAST,
    ImfDataMapperSource,
    ImfSpec,
    derive_interest_burden,
)
from dalio.data_sources.imf_imts import IMTS_FLOWS, ImtsSource, ImtsSpec
from dalio.data_sources.worldbank import (
    WB_FUNDAMENTALS,
    WB_MEMBER_MEAN_INDICATORS,
    WB_WORLD_SHARES,
    WbIndicatorSpec,
    WorldBankSource,
    derive_member_mean,
    derive_world_share,
)
from dalio.pipelines.fetch_fred import upsert_observations
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)

IMPLEMENTED_SOURCES: tuple[str, ...] = ("wb", "imf", "bis", "imts")
PLANNED_SOURCES: dict[str, str] = {}


def delete_forecasts(session, indicator: str, iso2s: Sequence[str]) -> int:
    """A WEO vintage supersedes the previous one entirely: drop old forecast
    rows before inserting the new ones (history rows are upserted as usual)."""
    res = session.execute(
        delete(Observation).where(
            Observation.indicator == indicator,
            Observation.source == SOURCE_FORECAST,
            Observation.country.in_(list(iso2s)),
        )
    )
    session.commit()
    return int(res.rowcount or 0)


def run_pipeline(
    sources: Sequence[str] = IMPLEMENTED_SOURCES,
    countries: Sequence[Country] | None = None,
    use_cache: bool = True,
    wb_source: WorldBankSource | None = None,
    wb_specs: Sequence[WbIndicatorSpec] = WB_FUNDAMENTALS,
    imf_source: ImfDataMapperSource | None = None,
    imf_specs: Sequence[ImfSpec] = IMF_FUNDAMENTALS,
    bis_source: BisSource | None = None,
    bis_dsr_specs: Sequence[DsrSpec] = TIER_3_DSR,
    bis_credit_specs: Sequence[TotalCreditSpec] = TIER_3_PRIVATE_CREDIT,
    imts_source: ImtsSource | None = None,
    imts_specs: Sequence[ImtsSpec] = IMTS_FLOWS,
) -> dict[str, dict]:
    """Fetch every spec of every requested source and upsert. Returns a
    per-spec summary keyed ``"{source}/{indicator}"``; errors are collected,
    never raised, so one dead series cannot sink the batch."""
    basket = tuple(countries) if countries else COUNTRIES
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    world_shares = dict(WB_WORLD_SHARES)
    eu_members = [c.iso2 for c in basket if c.eu_member]
    has_eu = any(c.iso2 == "EU" for c in basket)

    def _store(key: str, df, series_id: str, indicator: str, source_family: str = "wb") -> None:
        ins, skp = upsert_observations(session, df)
        summary[key] = {
            "source": source_family, "indicator": indicator, "series_id": series_id,
            "rows": len(df), "inserted": ins, "skipped": skp,
            "countries": int(df["country"].nunique()) if not df.empty else 0,
        }
        logger.info("Stored %s: %d rows / %d countries (%d new/updated)",
                    key, len(df), summary[key]["countries"], ins)

    with session_factory() as session:
        for source in sources:
            if source == "imf":
                imf = imf_source or ImfDataMapperSource()
                fetched: dict[str, object] = {}
                iso2s = [c.iso2 for c in basket if c.imf_id]
                for spec in imf_specs:
                    key = f"imf/{spec.indicator}"
                    try:
                        df = imf.fetch(spec, basket, use_cache=use_cache)
                        delete_forecasts(session, spec.indicator, iso2s)
                        _store(key, df, spec.imf_code, spec.indicator, "imf")
                        fetched[spec.indicator] = df
                    except Exception as e:  # noqa: BLE001 — collect per-series
                        logger.exception("Failed %s: %s", key, e)
                        summary[key] = {"source": "imf", "indicator": spec.indicator,
                                        "series_id": spec.imf_code, "error": str(e)}
                if "primary_balance_pct_gdp" in fetched and "fiscal_balance_pct_gdp" in fetched:
                    try:
                        delete_forecasts(session, "interest_burden_pct_gdp", iso2s)
                        _store("imf/interest_burden_pct_gdp",
                               derive_interest_burden(fetched["primary_balance_pct_gdp"],
                                                      fetched["fiscal_balance_pct_gdp"]),
                               "primary-overall", "interest_burden_pct_gdp", "imf")
                    except Exception as e:  # noqa: BLE001
                        logger.exception("Failed imf/interest_burden_pct_gdp: %s", e)
                        summary["imf/interest_burden_pct_gdp"] = {
                            "source": "imf", "indicator": "interest_burden_pct_gdp",
                            "series_id": "primary-overall", "error": str(e)}
            elif source == "bis":
                bis = bis_source or BisSource()
                wanted = {c.iso2 for c in basket}
                for spec in [*bis_dsr_specs, *bis_credit_specs]:
                    if spec.country not in wanted:
                        continue
                    key = f"bis/{spec.country}/{spec.indicator}"
                    try:
                        df = (bis.fetch_dsr(spec, use_cache=use_cache) if isinstance(spec, DsrSpec)
                              else bis.fetch_total_credit(spec, use_cache=use_cache))
                        _store(key, df, spec.indicator, spec.indicator, "bis")
                    except Exception as e:  # noqa: BLE001 — collect per-series (SA/RU expected)
                        logger.warning("Failed %s: %s", key, e)
                        summary[key] = {"source": "bis", "indicator": spec.indicator,
                                        "series_id": spec.country, "error": str(e)}
            elif source == "imts":
                imts = imts_source or ImtsSource()
                for spec in imts_specs:
                    key = f"imts/{spec.indicator_prefix}"
                    try:
                        df = imts.fetch(spec, basket, use_cache=use_cache)
                        _store(key, df, spec.imts_code, spec.indicator_prefix + "_*", "imts")
                    except Exception as e:  # noqa: BLE001 — collect per-flow
                        logger.exception("Failed %s: %s", key, e)
                        summary[key] = {"source": "imts", "indicator": spec.indicator_prefix + "_*",
                                        "series_id": spec.imts_code, "error": str(e)}
            elif source == "wb":
                wb = wb_source or WorldBankSource()
                for spec in wb_specs:
                    key = f"wb/{spec.indicator}"
                    try:
                        df = wb.fetch(spec, basket, use_cache=use_cache)
                        _store(key, df, spec.wb_code, spec.indicator)
                        if spec.indicator in world_shares:
                            out = world_shares[spec.indicator]
                            _store(f"wb/{out}", derive_world_share(df, out), spec.wb_code + "÷WLD", out)
                        if has_eu and spec.indicator in WB_MEMBER_MEAN_INDICATORS and eu_members:
                            dkey = f"wb/{spec.indicator}:EU"
                            _store(dkey, derive_member_mean(df, eu_members, "EU"),
                                   spec.wb_code + ":member-mean", spec.indicator)
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
