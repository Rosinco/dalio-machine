"""ETL pipeline: fetch fundamentals → release ledger + current projection.

Sources are pulled per indicator for the whole basket in one paginated call
(World Bank). IMF WEO (slice 20), the BIS extension (slice 22), IMF IMTS
bilateral trade (slice 24) and the OEC complexity index (slice 23) plug into
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
from datetime import UTC, datetime

from dotenv import load_dotenv

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
    SOURCE_HISTORY,
    ImfDataMapperSource,
    ImfSpec,
    derive_interest_burden,
)
from dalio.data_sources.imf_imts import IMTS_FLOWS, ImtsSource, ImtsSpec
from dalio.data_sources.oec import INDICATOR_ECI, OEC_SERIES_ID, OecSource
from dalio.data_sources.worldbank import (
    WB_FUNDAMENTALS,
    WB_MEMBER_MEAN_INDICATORS,
    WB_WORLD_SHARES,
    WbIndicatorSpec,
    WorldBankSource,
    derive_member_mean,
    derive_world_share,
)
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseMeta,
    ingest_release_snapshot,
    make_partition_key,
)

logger = logging.getLogger(__name__)

IMPLEMENTED_SOURCES: tuple[str, ...] = ("wb", "imf", "bis", "imts", "oec")
PLANNED_SOURCES: dict[str, str] = {}


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
    oec_source: OecSource | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict]:
    """Fetch every spec into complete source-partition releases. Returns a
    per-spec summary keyed ``"{source}/{indicator}"``; errors are collected,
    never raised, so one dead series cannot sink the batch."""
    basket = tuple(countries) if countries else COUNTRIES
    run_at = retrieved_at or datetime.now(UTC)
    engine = make_engine()
    init_db(engine)
    session_factory = make_session_factory(engine)

    summary: dict[str, dict] = {}
    world_shares = dict(WB_WORLD_SHARES)
    eu_members = [c.iso2 for c in basket if c.eu_member]
    has_eu = any(c.iso2 == "EU" for c in basket)

    def _store_release(
        key: str,
        df,
        series_id: str,
        indicator: str,
        summary_source: str = "wb",
    ) -> None:
        """Store complete independently replaceable non-WEO partitions.

        Multi-country provider calls and IMTS partner flows are split on their
        actual country, indicator, source and native series id. An empty frame
        records the legacy zero-row summary but cannot erase a prior projection.
        """
        inserted = skipped = removed = created = 0
        release_ids: list[int] = []
        if not df.empty:
            partition_columns = ["country", "indicator", "source", "series_id"]
            missing = set(partition_columns) - set(df.columns)
            if missing:
                raise ValueError(f"release snapshot missing partition columns: {sorted(missing)}")
            if df[partition_columns].isna().any().any():
                raise ValueError("release snapshot partition fields must not be null")

            partitions = df[partition_columns].drop_duplicates()
            duplicate_scopes = partitions.duplicated(
                subset=["country", "indicator", "source"], keep=False,
            )
            if duplicate_scopes.any():
                raise ValueError(
                    "one current projection scope contains multiple native series ids"
                )

            for partition, snapshot in df.groupby(
                partition_columns, sort=True, dropna=False,
            ):
                country, native_indicator, native_source, native_series_id = map(
                    str, partition,
                )
                result = ingest_release_snapshot(
                    session,
                    snapshot.reset_index(drop=True),
                    ReleaseMeta(
                        partition_key=make_partition_key(
                            native_source,
                            native_series_id,
                            country,
                            native_indicator,
                        ),
                        source_family=native_source,
                        available_at=run_at,
                        retrieved_at=run_at,
                        projection=ProjectionScope(
                            country=country,
                            indicator=native_indicator,
                            sources=(native_source,),
                        ),
                    ),
                )
                inserted += result.changed_rows
                skipped += result.unchanged_rows
                removed += result.removed_rows
                created += int(result.created)
                release_ids.append(result.release_id)

        summary[key] = {
            "source": summary_source, "indicator": indicator, "series_id": series_id,
            "rows": len(df), "inserted": inserted, "skipped": skipped,
            "countries": int(df["country"].nunique()) if not df.empty else 0,
            "removed": removed,
            "release_ids": release_ids,
            "releases_created": created,
        }
        logger.info("Stored %s: %d rows / %d countries (%d new/updated)",
                    key, len(df), summary[key]["countries"], inserted)

    def _store_imf_release(
        key: str,
        df,
        series_id: str,
        indicator: str,
        expected_iso2s: Sequence[str],
    ) -> None:
        """Store a WEO response as complete country/indicator release snapshots.

        One DataMapper call returns both history and forecasts for many countries.
        Each country is an independently replaceable current projection, while its
        prior complete responses remain in the release ledger for point-in-time use.
        """
        if df.empty:
            raise ValueError("IMF WEO release snapshot is empty; current data left unchanged")

        inserted = skipped = removed = created = 0
        release_ids: list[int] = []
        for iso2, country_frame in df.groupby("country", sort=True):
            result = ingest_release_snapshot(
                session,
                country_frame.reset_index(drop=True),
                ReleaseMeta(
                    partition_key=make_partition_key(
                        SOURCE_HISTORY,
                        series_id,
                        str(iso2),
                        indicator,
                    ),
                    source_family=SOURCE_HISTORY,
                    available_at=run_at,
                    retrieved_at=run_at,
                    projection=ProjectionScope(
                        country=str(iso2),
                        indicator=indicator,
                        sources=(SOURCE_HISTORY, SOURCE_FORECAST),
                    ),
                ),
            )
            inserted += result.changed_rows
            skipped += result.unchanged_rows
            removed += result.removed_rows
            created += int(result.created)
            release_ids.append(result.release_id)

        present = {str(country) for country in df["country"].unique()}
        missing = sorted(set(expected_iso2s) - present)
        if missing:
            logger.warning(
                "IMF WEO %s returned no rows for %s; prior current partitions, if any, "
                "remain unchanged",
                indicator,
                ", ".join(missing),
            )

        summary[key] = {
            "source": "imf",
            "indicator": indicator,
            "series_id": series_id,
            "rows": len(df),
            "inserted": inserted,
            "skipped": skipped,
            "countries": len(present),
            "removed": removed,
            "release_ids": release_ids,
            "releases_created": created,
        }
        logger.info(
            "Stored %s: %d rows / %d countries (%d new/updated, %d removed)",
            key,
            len(df),
            len(present),
            inserted,
            removed,
        )

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
                        _store_imf_release(key, df, spec.imf_code, spec.indicator, iso2s)
                        fetched[spec.indicator] = df
                    except Exception as e:  # noqa: BLE001 — collect per-series
                        logger.exception("Failed %s: %s", key, e)
                        summary[key] = {"source": "imf", "indicator": spec.indicator,
                                        "series_id": spec.imf_code, "error": str(e)}
                if "primary_balance_pct_gdp" in fetched and "fiscal_balance_pct_gdp" in fetched:
                    try:
                        _store_imf_release(
                            "imf/interest_burden_pct_gdp",
                            derive_interest_burden(
                                fetched["primary_balance_pct_gdp"],
                                fetched["fiscal_balance_pct_gdp"],
                            ),
                            "primary-overall",
                            "interest_burden_pct_gdp",
                            iso2s,
                        )
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
                        _store_release(key, df, spec.indicator, spec.indicator, "bis")
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
                        _store_release(
                            key,
                            df,
                            spec.imts_code,
                            spec.indicator_prefix + "_*",
                            "imts",
                        )
                    except Exception as e:  # noqa: BLE001 — collect per-flow
                        logger.exception("Failed %s: %s", key, e)
                        summary[key] = {"source": "imts", "indicator": spec.indicator_prefix + "_*",
                                        "series_id": spec.imts_code, "error": str(e)}
            elif source == "oec":
                oec = oec_source or OecSource()
                key = f"oec/{INDICATOR_ECI}"
                try:
                    df = oec.fetch_eci(basket, use_cache=use_cache)
                    _store_release(key, df, OEC_SERIES_ID, INDICATOR_ECI, "oec")
                    if has_eu and eu_members:
                        _store_release(
                            f"{key}:EU",
                            derive_member_mean(df, eu_members, "EU"),
                            OEC_SERIES_ID + ":member-mean",
                            INDICATOR_ECI,
                            "oec",
                        )
                except Exception as e:  # noqa: BLE001
                    logger.exception("Failed %s: %s", key, e)
                    summary[key] = {"source": "oec", "indicator": INDICATOR_ECI,
                                    "series_id": OEC_SERIES_ID, "error": str(e)}
            elif source == "wb":
                wb = wb_source or WorldBankSource()
                for spec in wb_specs:
                    key = f"wb/{spec.indicator}"
                    try:
                        df = wb.fetch(spec, basket, use_cache=use_cache)
                        _store_release(key, df, spec.wb_code, spec.indicator)
                        if spec.indicator in world_shares:
                            out = world_shares[spec.indicator]
                            _store_release(
                                f"wb/{out}",
                                derive_world_share(df, out),
                                spec.wb_code + "÷WLD",
                                out,
                            )
                        if has_eu and spec.indicator in WB_MEMBER_MEAN_INDICATORS and eu_members:
                            dkey = f"wb/{spec.indicator}:EU"
                            _store_release(
                                dkey,
                                derive_member_mean(df, eu_members, "EU"),
                                spec.wb_code + ":member-mean",
                                spec.indicator,
                            )
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
