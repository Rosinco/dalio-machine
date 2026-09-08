"""ETL: IMF PIP/DIP bilateral stocks -> typed immutable position ledger."""

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
from dalio.data_sources.imf_positions import (
    DIP_FREQUENCIES,
    DIP_HISTORY_START_YEAR,
    DIP_SERIES,
    DIP_SOURCE,
    IMF_POSITION_COUNTERPARTS,
    IMF_POSITION_COUNTRIES,
    PIP_FREQUENCIES,
    PIP_HISTORY_START_YEAR,
    PIP_SERIES,
    PIP_SOURCE,
    ImfDipSource,
    ImfPipSource,
    PositionCounterpart,
    PositionSeriesSpec,
)
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.positions import (
    PositionIngestResult,
    ingest_cross_border_snapshot,
    make_position_partition_key,
)
from dalio.storage.releases import ReleaseMeta

logger = logging.getLogger(__name__)


def _summary_key(
    dataset: str,
    reporter: Country,
    spec: PositionSeriesSpec,
    frequency: str,
) -> str:
    return f"{dataset}/{reporter.iso2}/{spec.indicator}/{frequency}"


def _validate_partition(
    frame: pd.DataFrame,
    *,
    dataset: str,
    reporter: Country,
    spec: PositionSeriesSpec,
    frequency: str,
    counterparts: Sequence[PositionCounterpart],
) -> None:
    """Reject a partial-key or semantically mixed native position partition."""
    if frame.empty:
        raise ValueError("position partition must not be empty")
    required = {
        "dataset",
        "reporter_country",
        "reporter_code",
        "counterpart_country",
        "counterpart_code",
        "direction",
        "accounting_basis",
        "instrument_code",
        "frequency",
        "unit",
        "source",
        "native_indicator",
        "reporter_sector_code",
        "counterpart_sector_code",
        "derivation_type",
        "series_id",
        "status",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"position snapshot missing partition columns: {sorted(missing)}")

    source = PIP_SOURCE if dataset == "PIP" else DIP_SOURCE
    expected_scalars = {
        "dataset": dataset,
        "reporter_country": reporter.iso2,
        "reporter_code": str(reporter.imf_id),
        "direction": spec.direction,
        "accounting_basis": spec.accounting_basis,
        "instrument_code": spec.instrument_code,
        "frequency": frequency,
        "unit": "USD",
        "source": source,
        "native_indicator": spec.native_indicator,
    }
    for column, expected in expected_scalars.items():
        actual = frame[column].drop_duplicates().tolist()
        if actual != [expected]:
            raise ValueError(
                f"{dataset} position partition expected {column} {expected!r}, got {actual!r}"
            )

    allowed_counterparts = {
        (counterpart.internal_code, counterpart.imf_code) for counterpart in counterparts
    }
    actual_counterparts = set(
        frame[["counterpart_country", "counterpart_code"]].itertuples(index=False, name=None)
    )
    if not actual_counterparts <= allowed_counterparts:
        raise ValueError(f"{dataset} position partition contains an unexpected counterpart")

    if dataset == "PIP":
        if set(frame["reporter_sector_code"].dropna()) != {"S1"}:
            raise ValueError("PIP position partition must use reporter sector S1")
        if set(frame["counterpart_sector_code"].dropna()) != {"S1"}:
            raise ValueError("PIP position partition must use counterpart sector S1")
        if frame["derivation_type"].notna().any():
            raise ValueError("PIP position partition must not set a derivation type")
    else:
        if frame["reporter_sector_code"].notna().any():
            raise ValueError("DIP position partition must not set a reporter sector")
        if frame["counterpart_sector_code"].notna().any():
            raise ValueError("DIP position partition must not set a counterpart sector")
        if set(frame["derivation_type"].dropna()) != {"O"}:
            raise ValueError("DIP position partition must use reported DV_TYPE O")

    expected_series = frame.apply(
        lambda row: spec.series_id(str(reporter.imf_id), row["counterpart_code"], frequency),
        axis=1,
    )
    if not frame["series_id"].equals(expected_series):
        raise ValueError(f"{dataset} position partition contains an unexpected native series")


def _ingest_partition(
    session: Session,
    frame: pd.DataFrame,
    *,
    dataset: str,
    reporter: Country,
    spec: PositionSeriesSpec,
    frequency: str,
    counterparts: Sequence[PositionCounterpart],
    retrieved_at: datetime,
    source_url: str,
) -> PositionIngestResult:
    _validate_partition(
        frame,
        dataset=dataset,
        reporter=reporter,
        spec=spec,
        frequency=frequency,
        counterparts=counterparts,
    )
    source = PIP_SOURCE if dataset == "PIP" else DIP_SOURCE
    return ingest_cross_border_snapshot(
        session,
        frame,
        ReleaseMeta(
            partition_key=make_position_partition_key(
                source, reporter.iso2, spec.native_indicator, frequency
            ),
            source_family=source,
            # IMF dataset-level update metadata is not a dependable per-series
            # timestamp. Retrieval is therefore the conservative knowledge clock.
            available_at=retrieved_at,
            retrieved_at=retrieved_at,
            source_url=source_url,
        ),
    )


def _fetch_bundle(source, reporter, specs, frequencies, counterparts, start_year, use_cache):
    """Use the compact adapter API, with a small fake-friendly fallback."""
    if hasattr(source, "fetch_bundle"):
        return source.fetch_bundle(
            reporter,
            specs,
            frequencies,
            counterparts=counterparts,
            start_year=start_year,
            use_cache=use_cache,
        )
    frames = [
        source.fetch(
            reporter,
            spec,
            frequency,
            counterparts=counterparts,
            start_year=start_year,
            use_cache=use_cache,
        )
        for spec in specs
        for frequency in frequencies
    ]
    frames = [frame for frame in frames if not frame.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _bundle_url(source, reporter, specs, frequencies, counterparts, start_year):
    if hasattr(source, "bundle_url_for"):
        return source.bundle_url_for(
            reporter,
            specs,
            frequencies,
            counterparts=counterparts,
            start_year=start_year,
        )
    return source.url_for(
        reporter,
        specs[0],
        frequencies[0],
        counterparts=counterparts,
        start_year=start_year,
    )


def _validate_selection(
    reporters: tuple[Country, ...],
    counterparts: tuple[PositionCounterpart, ...],
    pip_specs: tuple[PositionSeriesSpec, ...],
    dip_specs: tuple[PositionSeriesSpec, ...],
    pip_frequencies: tuple[str, ...],
) -> None:
    if len({reporter.iso2 for reporter in reporters}) != len(reporters):
        raise ValueError("IMF position reporter selection contains duplicates")
    allowed_reporters = {country.iso2 for country in IMF_POSITION_COUNTRIES}
    if any(reporter.iso2 not in allowed_reporters for reporter in reporters):
        raise ValueError("IMF position reporters must be individual on-map countries")

    canonical_counterparts = [
        (item.internal_code, item.imf_code) for item in IMF_POSITION_COUNTERPARTS
    ]
    selected_counterparts = [(item.internal_code, item.imf_code) for item in counterparts]
    if selected_counterparts != canonical_counterparts:
        raise ValueError(
            "position ETL requires the complete canonical 21-country plus world counterpart basket"
        )
    if len({spec.native_indicator for spec in pip_specs}) != len(pip_specs):
        raise ValueError("PIP series selection contains duplicates")
    if len({spec.native_indicator for spec in dip_specs}) != len(dip_specs):
        raise ValueError("DIP series selection contains duplicates")
    if any(spec.dataset != "PIP" for spec in pip_specs):
        raise ValueError("PIP series selection contains a different-dataset spec")
    if any(spec.dataset != "DIP" for spec in dip_specs):
        raise ValueError("DIP series selection contains a different-dataset spec")
    if not pip_frequencies or not set(pip_frequencies) <= set(PIP_FREQUENCIES):
        raise ValueError("PIP frequencies must be a non-empty subset of A and S")
    if len(set(pip_frequencies)) != len(pip_frequencies):
        raise ValueError("PIP frequency selection contains duplicates")


def run_pipeline(
    reporters: Sequence[Country] = IMF_POSITION_COUNTRIES,
    counterparts: Sequence[PositionCounterpart] = IMF_POSITION_COUNTERPARTS,
    pip_specs: Sequence[PositionSeriesSpec] = PIP_SERIES,
    dip_specs: Sequence[PositionSeriesSpec] = DIP_SERIES,
    pip_frequencies: Sequence[str] = PIP_FREQUENCIES,
    pip_source: ImfPipSource | None = None,
    dip_source: ImfDipSource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict]:
    """Fetch two compact bundles per reporter and ingest stable partitions.

    Each reporter produces at most one PIP and one DIP HTTP request, but every
    reporter/indicator/frequency history is stored as its own complete release.
    An absent native partition is reported and left absent; it is never written
    as zero and never erases the last valid release after a transient empty reply.
    """
    selected_reporters = tuple(reporters)
    selected_counterparts = tuple(counterparts)
    selected_pip_specs = tuple(pip_specs)
    selected_dip_specs = tuple(dip_specs)
    selected_pip_frequencies = tuple(str(value).upper() for value in pip_frequencies)
    _validate_selection(
        selected_reporters,
        selected_counterparts,
        selected_pip_specs,
        selected_dip_specs,
        selected_pip_frequencies,
    )
    if not selected_reporters or (not selected_pip_specs and not selected_dip_specs):
        return {}

    sources = (
        (
            "PIP",
            pip_source or ImfPipSource(),
            selected_pip_specs,
            selected_pip_frequencies,
            PIP_HISTORY_START_YEAR,
        ),
        (
            "DIP",
            dip_source or ImfDipSource(),
            selected_dip_specs,
            DIP_FREQUENCIES,
            DIP_HISTORY_START_YEAR,
        ),
    )
    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    summary: dict[str, dict] = {}

    with session_factory() as session:
        for dataset, source, specs, frequencies, start_year in sources:
            if not specs:
                continue
            for reporter in selected_reporters:
                source_url = _bundle_url(
                    source,
                    reporter,
                    specs,
                    frequencies,
                    selected_counterparts,
                    start_year,
                )
                try:
                    bundle = _fetch_bundle(
                        source,
                        reporter,
                        specs,
                        frequencies,
                        selected_counterparts,
                        start_year,
                        use_cache,
                    )
                    fetch_error: Exception | None = None
                except Exception as exc:  # noqa: BLE001 - isolate reporter/dataset bundle
                    logger.exception("Failed IMF %s bundle for %s: %s", dataset, reporter.iso2, exc)
                    bundle = pd.DataFrame()
                    fetch_error = exc

                for spec in specs:
                    for frequency in frequencies:
                        key = _summary_key(dataset, reporter, spec, frequency)
                        base = {
                            "dataset": dataset,
                            "reporter": reporter.iso2,
                            "indicator": spec.indicator,
                            "native_indicator": spec.native_indicator,
                            "frequency": frequency,
                        }
                        if fetch_error is not None:
                            summary[key] = {**base, "error": str(fetch_error)}
                            continue
                        if bundle.empty:
                            partition = pd.DataFrame()
                        else:
                            partition = bundle[
                                (bundle["dataset"] == dataset)
                                & (bundle["reporter_country"] == reporter.iso2)
                                & (bundle["native_indicator"] == spec.native_indicator)
                                & (bundle["frequency"] == frequency)
                            ].copy()
                        if partition.empty:
                            summary[key] = {
                                **base,
                                "rows": 0,
                                "not_reported": True,
                                "release_created": False,
                            }
                            continue
                        try:
                            result = _ingest_partition(
                                session,
                                partition,
                                dataset=dataset,
                                reporter=reporter,
                                spec=spec,
                                frequency=frequency,
                                counterparts=selected_counterparts,
                                retrieved_at=run_at,
                                source_url=source_url,
                            )
                            summary[key] = {
                                **base,
                                "rows": len(partition),
                                "release_id": result.release_id,
                                "release_created": result.created,
                            }
                        except Exception as exc:  # noqa: BLE001 - isolate native partition
                            logger.exception("Failed %s: %s", key, exc)
                            summary[key] = {**base, "error": str(exc)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch IMF PIP portfolio-asset and DIP direct-investment bilateral "
            "position stocks into the typed immutable release ledger."
        )
    )
    parser.add_argument(
        "countries",
        nargs="*",
        help="Optional internal reporter codes (for example SE US UK). Default: all 21.",
    )
    parser.add_argument(
        "--dataset",
        choices=("both", "pip", "dip"),
        default="both",
        help="Limit the pull to one IMF dataset (default: both).",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass the on-disk IMF caches.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    if args.countries:
        try:
            reporters = tuple(get_country(code) for code in args.countries)
        except KeyError as exc:
            parser.error(str(exc))
    else:
        reporters = IMF_POSITION_COUNTRIES

    pip_specs = PIP_SERIES if args.dataset in {"both", "pip"} else ()
    dip_specs = DIP_SERIES if args.dataset in {"both", "dip"} else ()
    print("dalio-fetch-positions — IMF bilateral investment position stocks")
    print(
        f"  {len(reporters)} reporters; at most "
        f"{len(reporters) * int(bool(pip_specs)) + len(reporters) * int(bool(dip_specs))} "
        "bounded official API requests"
    )
    summary = run_pipeline(
        reporters=reporters,
        pip_specs=pip_specs,
        dip_specs=dip_specs,
        use_cache=not args.no_cache,
    )
    failed = sum("error" in item for item in summary.values())
    absent = sum(item.get("not_reported", False) for item in summary.values())
    loaded = len(summary) - failed - absent
    print(f"\nSummary: {loaded} partitions refreshed; {absent} not reported; {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
