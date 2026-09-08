"""ETL: IMF BOP financial-account transactions -> immutable release ledger."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from datetime import UTC, datetime

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import Engine
from sqlalchemy.orm import Session

from dalio.countries import Country, get_country
from dalio.data_sources.imf_bop import (
    BOP_COUNTRIES,
    BOP_HISTORY_START_YEAR,
    BOP_SERIES,
    BOP_SOURCE,
    BopSeriesSpec,
    ImfBopSource,
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


def _validate_partition(
    frame: pd.DataFrame,
    *,
    country: Country,
    spec: BopSeriesSpec,
) -> None:
    """Reject an incomplete or semantically mixed native-series snapshot."""
    if frame.empty:
        raise ValueError(
            f"IMF BOP returned an empty snapshot for {country.iso2}/{spec.indicator}; "
            "release not recorded"
        )

    required = {"country", "indicator", "source", "series_id", "status"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"IMF BOP snapshot missing partition columns: {sorted(missing)}")

    expected = {
        "country": country.iso2,
        "indicator": spec.indicator,
        "source": BOP_SOURCE,
        "series_id": spec.series_id,
    }
    for column, expected_value in expected.items():
        actual = frame[column].drop_duplicates().tolist()
        if actual != [expected_value]:
            if column == "series_id":
                raise ValueError(
                    f"IMF BOP snapshot expected native series {expected_value!r}, "
                    f"got {actual!r}"
                )
            raise ValueError(
                f"IMF BOP snapshot expected {column} {expected_value!r}, got {actual!r}"
            )


def _ingest_partition(
    session: Session,
    frame: pd.DataFrame,
    *,
    country: Country,
    spec: BopSeriesSpec,
    retrieved_at: datetime,
    source_url: str,
) -> IngestResult:
    _validate_partition(frame, country=country, spec=spec)
    return ingest_release_snapshot(
        session,
        frame,
        ReleaseMeta(
            partition_key=make_partition_key(
                BOP_SOURCE,
                spec.series_id,
                country.iso2,
                spec.indicator,
            ),
            source_family=BOP_SOURCE,
            # IMF exposes dataset update metadata, not a dependable timestamp
            # for each native series. Retrieval is the conservative knowledge
            # clock; later refreshes preserve any revisions in the ledger.
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            source_url=source_url,
            projection=ProjectionScope(
                country=country.iso2,
                indicator=spec.indicator,
                sources=(BOP_SOURCE,),
            ),
        ),
    )


def run_pipeline(
    countries: Sequence[Country] = BOP_COUNTRIES,
    specs: Sequence[BopSeriesSpec] = BOP_SERIES,
    source: ImfBopSource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict]:
    """Fetch one compact bundle, then ingest each native series independently.

    The history start is deliberately fixed.  Each country/indicator is a
    complete partition from 2000 onward, so an arbitrary shorter refresh can
    never delete valid older history.  A valid response can omit a series that
    the country does not publish; that is reported as ``not_reported`` and does
    not replace its previous current projection.  Malformed partitions fail
    closed.
    """
    selected_countries = tuple(country for country in countries if country.imf_id)
    selected_specs = tuple(specs)
    if not selected_countries or not selected_specs:
        return {}
    if len({country.iso2 for country in selected_countries}) != len(selected_countries):
        raise ValueError("IMF BOP country selection contains duplicates")
    if len({spec.indicator for spec in selected_specs}) != len(selected_specs):
        raise ValueError("IMF BOP series selection contains duplicate indicators")

    src = source or ImfBopSource()
    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    source_url = src.url_for(
        selected_countries,
        specs=selected_specs,
        start_year=BOP_HISTORY_START_YEAR,
    )

    try:
        bundle = src.fetch(
            selected_countries,
            specs=selected_specs,
            start_year=BOP_HISTORY_START_YEAR,
            use_cache=use_cache,
        )
        fetch_error: Exception | None = None
    except Exception as exc:  # noqa: BLE001 - report every expected partition
        logger.exception("Failed IMF BOP bundle: %s", exc)
        bundle = pd.DataFrame()
        fetch_error = exc

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for country in selected_countries:
            for spec in selected_specs:
                key = f"{country.iso2}/{spec.indicator}"
                if fetch_error is not None:
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "series_id": spec.series_id,
                        "error": str(fetch_error),
                    }
                    continue

                try:
                    partition = bundle[
                        (bundle["country"] == country.iso2)
                        & (bundle["indicator"] == spec.indicator)
                    ].copy()
                    if partition.empty:
                        summary[key] = {
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
                        logger.warning(
                            "IMF BOP %s has no published observations; prior data unchanged",
                            key,
                        )
                        continue
                    result = _ingest_partition(
                        session,
                        partition,
                        country=country,
                        spec=spec,
                        retrieved_at=run_at,
                        source_url=source_url,
                    )
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "status": "stored",
                        "not_reported": False,
                        "rows": len(partition),
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
                        len(partition),
                        result.changed_rows,
                        result.unchanged_rows,
                    )
                except Exception as exc:  # noqa: BLE001 - isolate every partition
                    logger.exception("Failed %s (%s): %s", key, spec.series_id, exc)
                    summary[key] = {
                        "country": country.iso2,
                        "indicator": spec.indicator,
                        "series_id": spec.series_id,
                        "error": str(exc),
                    }

    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch IMF BPM6 financial-account transactions into the immutable "
            "release ledger. Asset, liability and net entries stay separate."
        )
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="Optional internal country codes (for example SE US UK). Default: all mapped.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk IMF response cache.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()

    if args.countries:
        try:
            countries = tuple(get_country(code) for code in args.countries)
        except KeyError as exc:
            parser.error(str(exc))
    else:
        countries = BOP_COUNTRIES

    print("dalio-fetch-flows — IMF BOP financial-account transactions")
    print(
        f"  {len(countries)} countries × {len(BOP_SERIES)} explicit series, "
        f"complete quarterly history from {BOP_HISTORY_START_YEAR}"
    )
    summary = run_pipeline(countries=countries, use_cache=not args.no_cache)

    failed = sum("error" in stats for stats in summary.values())
    absent = sum(stats.get("status") == "not_reported" for stats in summary.values())
    succeeded = len(summary) - failed - absent
    print(
        f"\nSummary: {succeeded} partitions refreshed; "
        f"{absent} not reported; {failed} failed."
    )
    if failed:
        print("Failed country/series pairs were left unchanged; see logs.")
    if absent:
        print("Unpublished country/series pairs were left unchanged and not filled.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
