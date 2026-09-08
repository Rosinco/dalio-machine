"""Fetch World Bank QPSD sovereign-debt anatomy into immutable releases.

The World Bank's Quarterly Public Sector Debt database is voluntarily
reported.  A successful source response can therefore have no observations
for a requested country/series.  Such partitions are reported as
``not_reported`` and deliberately left untouched in both the immutable ledger
and the current projection.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dalio.countries import Country, get_country
from dalio.data_sources.worldbank_qpsd import (
    QPSD_COUNTRIES,
    QPSD_SERIES,
    QPSD_SOURCE,
    QpsdSeriesSpec,
    WorldBankQpsdSource,
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

_LONG_COLUMNS = {"country", "indicator", "date", "value", "source", "series_id"}


def _validate_batch(
    frame: pd.DataFrame,
    spec: QpsdSeriesSpec,
    countries: Sequence[Country],
) -> None:
    """Reject a malformed basket before any country partition is committed."""

    missing = _LONG_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"QPSD snapshot missing columns: {sorted(missing)}")
    if frame.empty:
        return

    partition_columns = ["country", "indicator", "source", "series_id"]
    if frame[partition_columns].isna().any().any():
        raise ValueError("QPSD snapshot partition fields must not be null")

    expected_countries = {country.iso2 for country in countries}
    returned_countries = set(frame["country"].astype(str))
    unexpected = returned_countries - expected_countries
    if unexpected:
        raise ValueError(f"QPSD snapshot contains unrequested countries: {sorted(unexpected)}")
    if set(frame["indicator"].astype(str)) != {spec.indicator}:
        raise ValueError(f"QPSD snapshot must contain only indicator {spec.indicator!r}")
    if set(frame["source"].astype(str)) != {QPSD_SOURCE}:
        raise ValueError(f"QPSD snapshot must contain only source {QPSD_SOURCE!r}")
    if set(frame["series_id"].astype(str)) != {spec.series_id}:
        raise ValueError(f"QPSD snapshot must contain only native series {spec.series_id!r}")

    parsed_dates = pd.to_datetime(frame["date"], errors="raise")
    valid_quarter_start = parsed_dates.dt.month.isin((1, 4, 7, 10)) & parsed_dates.dt.day.eq(1)
    if not valid_quarter_start.all():
        raise ValueError("QPSD snapshot dates must be quarter starts")

    if frame["value"].map(lambda value: isinstance(value, bool)).any():
        raise ValueError("QPSD snapshot values must be numeric, not boolean")
    numeric = pd.to_numeric(frame["value"], errors="raise")
    if not numeric.map(lambda value: math.isfinite(float(value))).all():
        raise ValueError("QPSD snapshot values must be finite")
    keys = frame.loc[:, ["country", "indicator", "source"]].copy()
    keys["date"] = parsed_dates.dt.date
    if keys.duplicated(subset=["country", "indicator", "date", "source"], keep=False).any():
        raise ValueError("QPSD snapshot contains duplicate observation keys")


def _ingest_country_release(
    session: Session,
    frame: pd.DataFrame,
    *,
    spec: QpsdSeriesSpec,
    country: Country,
    retrieved_at: datetime,
    source_url: str,
) -> IngestResult:
    """Append one complete country/native-series history and refresh its view."""

    if frame.empty:
        raise ValueError("QPSD country snapshot must not be empty")
    return ingest_release_snapshot(
        session,
        frame.reset_index(drop=True),
        ReleaseMeta(
            partition_key=make_partition_key(
                QPSD_SOURCE,
                spec.series_id,
                country.iso2,
                spec.indicator,
            ),
            source_family=QPSD_SOURCE,
            # QPSD pages do not expose a trustworthy release timestamp, so the
            # retrieval instant is the conservative point-in-time availability.
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            source_url=source_url,
            projection=ProjectionScope(
                country=country.iso2,
                indicator=spec.indicator,
                sources=(QPSD_SOURCE,),
            ),
        ),
    )


def _not_reported(spec: QpsdSeriesSpec, country: Country) -> dict[str, object]:
    return {
        "country": country.iso2,
        "indicator": spec.indicator,
        "series_id": spec.series_id,
        "status": "not_reported",
        "not_reported": True,
        "rows": 0,
        "inserted": 0,
        "skipped": 0,
        "removed": 0,
        "release_id": None,
        "release_created": False,
    }


def run_pipeline(
    specs: Sequence[QpsdSeriesSpec] = QPSD_SERIES,
    countries: Sequence[Country] = QPSD_COUNTRIES,
    source: WorldBankQpsdSource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict[str, object]]:
    """Fetch QPSD once per native series and store per-country histories.

    Every requested country receives a summary entry.  Missing voluntary
    submissions are identified as ``not_reported``; no empty release is written
    and any previously valid current projection remains in place.
    """

    basket = tuple(countries)
    catalogue = tuple(specs)
    if len({country.iso2 for country in basket}) != len(basket):
        raise ValueError("QPSD country basket contains duplicate ISO-2 codes")
    if len({spec.indicator for spec in catalogue}) != len(catalogue):
        raise ValueError("QPSD series selection contains duplicate indicators")
    if not basket or not catalogue:
        return {}

    adapter = source or WorldBankQpsdSource()
    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)

    summary: dict[str, dict[str, object]] = {}
    with session_factory() as session:
        for spec in catalogue:
            source_url = WorldBankQpsdSource.url_for(
                spec,
                tuple(str(country.wb_id or country.iso3) for country in basket),
            )
            try:
                frame = adapter.fetch(spec, basket, use_cache=use_cache)
                _validate_batch(frame, spec, basket)
            except Exception as exc:  # noqa: BLE001 - isolate native-series failures
                logger.exception("Failed QPSD native series %s: %s", spec.series_id, exc)
                for country in basket:
                    key = f"{country.iso2}/{spec.indicator}"
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "series_id": spec.series_id,
                        "status": "error",
                        "error": str(exc),
                    }
                continue

            for country in basket:
                key = f"{country.iso2}/{spec.indicator}"
                country_frame = frame.loc[frame["country"] == country.iso2].copy()
                if country_frame.empty:
                    summary[key] = _not_reported(spec, country)
                    logger.warning(
                        "QPSD %s has no reported observations for %s; prior data unchanged",
                        spec.series_id,
                        country.iso2,
                    )
                    continue

                try:
                    result = _ingest_country_release(
                        session,
                        country_frame,
                        spec=spec,
                        country=country,
                        retrieved_at=run_at,
                        source_url=source_url,
                    )
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "series_id": spec.series_id,
                        "status": "stored",
                        "not_reported": False,
                        "rows": len(country_frame),
                        "inserted": result.changed_rows,
                        "skipped": result.unchanged_rows,
                        "removed": result.removed_rows,
                        "release_id": result.release_id,
                        "release_created": result.created,
                    }
                except Exception as exc:  # noqa: BLE001 - isolate country partitions
                    logger.exception("Failed QPSD partition %s: %s", key, exc)
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "series_id": spec.series_id,
                        "status": "error",
                        "error": str(exc),
                    }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch World Bank QPSD central-government debt anatomy as complete "
            "quarterly country/native-series releases."
        )
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="Optional ISO-2 countries (for example SE US JP). Default: full basket.",
    )
    parser.add_argument(
        "--indicators",
        nargs="+",
        help="Optional internal indicator names. Default: all twelve QPSD series.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the response cache and force a fresh complete download.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    try:
        basket = (
            tuple(get_country(code) for code in args.countries)
            if args.countries
            else QPSD_COUNTRIES
        )
    except KeyError as exc:
        parser.error(str(exc))

    specs_by_indicator = {spec.indicator: spec for spec in QPSD_SERIES}
    if args.indicators:
        unknown = sorted(set(args.indicators) - set(specs_by_indicator))
        if unknown:
            parser.error(f"unknown QPSD indicators: {', '.join(unknown)}")
        specs = tuple(specs_by_indicator[indicator] for indicator in args.indicators)
    else:
        specs = QPSD_SERIES

    print(f"dalio-fetch-sovereign-debt — {len(specs)} QPSD series / {len(basket)} countries")
    summary = run_pipeline(
        specs,
        basket,
        use_cache=not args.no_cache,
    )
    failed = sum(stats.get("status") == "error" for stats in summary.values())
    absent = sum(stats.get("status") == "not_reported" for stats in summary.values())
    stored = sum(stats.get("status") == "stored" for stats in summary.values())
    print(f"Stored: {stored}; not reported by country: {absent}; failed: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
