"""Fetch official money/liquidity histories into the immutable scalar ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import Engine

from dalio.data_sources.money_liquidity import (
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    MONEY_LIQUIDITY_SERIES,
    MoneyLiquiditySeries,
    MoneyLiquiditySource,
    money_liquidity_catalogue_sha256,
)
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseArtifactMeta,
    ReleaseMeta,
    ingest_release_snapshot,
    latest_release,
    make_partition_key,
)

logger = logging.getLogger(__name__)
_CORE_COLUMNS = {"country", "indicator", "date", "value", "source", "series_id"}


def _sha256_attr(frame: pd.DataFrame, name: str, spec: MoneyLiquiditySeries) -> str:
    value = frame.attrs.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{spec.native_series_id} lacks a valid {name}; release not recorded")
    return value


def _release_artifacts(
    frame: pd.DataFrame,
    spec: MoneyLiquiditySeries,
) -> tuple[ReleaseArtifactMeta, ...]:
    """Bind exact publisher bytes and the non-imputing missingness ledger."""
    source_hash = _sha256_attr(frame, "source_artifact_sha256", spec)
    native_hash = _sha256_attr(frame, "native_payload_sha256", spec)
    missing_hash = _sha256_attr(frame, "missing_provenance_sha256", spec)
    provenance_json = frame.attrs.get("missing_provenance_json")
    if not isinstance(provenance_json, str):
        raise ValueError(
            f"{spec.native_series_id} lacks canonical missingness provenance; release not recorded"
        )
    try:
        provenance_payload = json.loads(provenance_json)
        canonical = json.dumps(
            provenance_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"{spec.native_series_id} has invalid missingness provenance") from exc
    if not isinstance(provenance_payload, dict):
        raise ValueError(f"{spec.native_series_id} missingness provenance must be a JSON object")
    if canonical != provenance_json:
        raise ValueError(f"{spec.native_series_id} missingness provenance is not canonical JSON")
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != missing_hash:
        raise ValueError(
            f"{spec.native_series_id} missingness provenance hash does not match its payload"
        )
    if provenance_payload.get("native_series_id") != spec.native_series_id:
        raise ValueError(
            f"{spec.native_series_id} missingness provenance has the wrong native identity"
        )
    missing_records = frame.attrs.get("missing_period_records")
    if not isinstance(missing_records, (tuple, list)) or provenance_payload.get("records") != list(
        missing_records
    ):
        raise ValueError(
            f"{spec.native_series_id} missingness ledger does not match frame provenance"
        )

    common = {
        "native_payload_sha256": native_hash,
        "missing_provenance_sha256": missing_hash,
        "provenance_json": canonical,
    }
    return (
        ReleaseArtifactMeta(
            role="source_response",
            artifact_sha256=source_hash,
            artifact_path=frame.attrs.get("source_artifact_path", ""),
            **common,
        ),
        ReleaseArtifactMeta(
            role="native_series_payload",
            artifact_sha256=native_hash,
            artifact_path=frame.attrs.get("native_payload_artifact_path", ""),
            **common,
        ),
        ReleaseArtifactMeta(
            role="missingness_ledger",
            artifact_sha256=missing_hash,
            artifact_path=frame.attrs.get("missing_provenance_artifact_path", ""),
            **common,
        ),
    )


def _partition_key(spec: MoneyLiquiditySeries) -> str:
    return make_partition_key(
        spec.source_family,
        spec.native_series_id,
        spec.country,
        spec.indicator,
    )


def _validate_partition(
    frame: pd.DataFrame,
    spec: MoneyLiquiditySeries,
    *,
    as_of: datetime,
) -> None:
    if frame.empty:
        raise ValueError(
            f"{spec.delivery_service} returned an empty snapshot; release not recorded"
        )
    missing = _CORE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"money/liquidity snapshot missing columns: {sorted(missing)}")

    expected = {
        "country": spec.country,
        "indicator": spec.indicator,
        "source": spec.source_family,
        "series_id": spec.native_series_id,
    }
    for column, wanted in expected.items():
        values = frame[column].drop_duplicates().tolist()
        if values != [wanted]:
            label = "native series" if column == "series_id" else column
            raise ValueError(
                f"money/liquidity snapshot {label} mismatch: expected {wanted!r}, got {values!r}"
            )

    try:
        parsed_dates = pd.to_datetime(frame["date"], errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError("money/liquidity snapshot has an invalid observation date") from exc
    if parsed_dates.isna().any():
        raise ValueError("money/liquidity snapshot has a missing observation date")
    dates = sorted(parsed_dates.dt.date.tolist())
    if len(dates) != len(set(dates)):
        raise ValueError("money/liquidity snapshot has duplicate observation dates")
    if dates[0] != spec.expected_start:
        raise ValueError(
            f"{spec.native_series_id} expected start {spec.expected_start.isoformat()}, "
            f"got {dates[0].isoformat()}"
        )
    if len(dates) < spec.minimum_observations:
        raise ValueError(
            f"{spec.native_series_id} has {len(dates)} observations; minimum observation "
            f"count is {spec.minimum_observations}"
        )

    try:
        values = pd.to_numeric(frame["value"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.native_series_id} has a non-numeric observation") from exc
    if not values.map(math.isfinite).all():
        raise ValueError(f"{spec.native_series_id} has a non-finite observation")
    if (values <= 0).any():
        raise ValueError(f"{spec.native_series_id} has a nonpositive monetary stock")

    if spec.frequency == "monthly":
        if any(observed_on.day != 1 for observed_on in dates):
            raise ValueError(f"{spec.native_series_id} monthly dates must use month start")
        ordinals = [observed_on.year * 12 + observed_on.month for observed_on in dates]
        if any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
            raise ValueError(f"{spec.native_series_id} has a gap in monthly cadence")
    elif spec.frequency == "quarterly":
        if any(
            observed_on.day != 1 or observed_on.month not in {1, 4, 7, 10} for observed_on in dates
        ):
            raise ValueError(f"{spec.native_series_id} quarterly dates must use quarter start")
        ordinals = [observed_on.year * 4 + (observed_on.month - 1) // 3 for observed_on in dates]
        if any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
            raise ValueError(f"{spec.native_series_id} has a gap in quarterly cadence")
    elif spec.frequency == "weekly":
        expected_weekday = spec.expected_start.weekday()
        if any(observed_on.weekday() != expected_weekday for observed_on in dates):
            raise ValueError(
                f"{spec.native_series_id} weekly dates do not share the expected weekday"
            )
        if any((right - left).days != 7 for left, right in zip(dates, dates[1:], strict=False)):
            raise ValueError(f"{spec.native_series_id} has a gap in weekly cadence")
    else:
        raise ValueError(f"unsupported money/liquidity frequency: {spec.frequency!r}")

    as_of_date = as_of.date()
    latest = dates[-1]
    latest_lag = (as_of_date - latest).days
    if latest_lag < 0:
        raise ValueError(
            f"{spec.native_series_id} latest observation {latest.isoformat()} is after "
            f"retrieval date {as_of_date.isoformat()}"
        )
    if latest_lag > spec.max_latest_lag_days:
        raise ValueError(
            f"{spec.native_series_id} latest observation is stale by {latest_lag} days; "
            f"maximum is {spec.max_latest_lag_days}"
        )


def run_pipeline(
    specs: Iterable[MoneyLiquiditySeries] = MONEY_LIQUIDITY_SERIES,
    source: MoneyLiquiditySource | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
    allow_contraction: bool = False,
) -> dict[str, dict]:
    """Record each native series independently; one failure never erases peers."""
    src = source or MoneyLiquiditySource()
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    catalogue_hash = money_liquidity_catalogue_sha256()

    summary: dict[str, dict] = {}
    with session_factory() as session:
        for spec in specs:
            key = f"{spec.country}/{spec.indicator}"
            catalogue_fields = {
                "country": spec.country,
                "indicator": spec.indicator,
                "currency": spec.currency,
                "unit": spec.unit,
                "native_unit": spec.native_unit,
                "unit_multiplier": spec.unit_multiplier,
                "frequency": spec.frequency,
                "adjustment": spec.adjustment,
                "observation_basis": spec.observation_basis,
                "publisher": spec.publisher,
                "delivery_service": spec.delivery_service,
                "series_id": spec.native_series_id,
                "expected_start": spec.expected_start.isoformat(),
                "minimum_observations": spec.minimum_observations,
                "max_latest_lag_days": spec.max_latest_lag_days,
                "research_role": spec.research_role,
                "perimeter": spec.perimeter,
                "definition_notes": spec.definition_notes,
                "parent_series_id": spec.parent_native_series_id,
                "non_additive_groups": list(spec.non_additive_groups),
                "catalogue_semantic_sha256": catalogue_hash,
            }
            try:
                frame = src.fetch(spec, use_cache=use_cache)
                # When no deterministic clock is injected, capture the instant
                # immediately after this partition was actually obtained.
                release_at = retrieved_at or datetime.now(UTC)
                _validate_partition(frame, spec, as_of=release_at)
                release_artifacts = _release_artifacts(frame, spec)
                partition_key = _partition_key(spec)
                previous = latest_release(session, partition_key)
                if (
                    previous is not None
                    and len(frame) < previous.row_count
                    and not allow_contraction
                ):
                    raise ValueError(
                        f"{spec.native_series_id} contracts complete history from "
                        f"{previous.row_count} to {len(frame)} observations; rerun with "
                        "--allow-contraction only after verifying the publisher removal"
                    )
                result = ingest_release_snapshot(
                    session,
                    frame,
                    ReleaseMeta(
                        partition_key=partition_key,
                        source_family=spec.source_family,
                        # Retrieval time remains the conservative availability
                        # clock. SCB's dataset-update timestamp, when present,
                        # is retained separately as publication provenance.
                        published_at=frame.attrs.get("published_at"),
                        available_at=release_at,
                        retrieved_at=release_at,
                        vintage_label=(
                            f"{MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX}{catalogue_hash}"
                        ),
                        source_url=spec.url,
                        projection=ProjectionScope(
                            country=spec.country,
                            indicator=spec.indicator,
                            sources=(spec.source_family,),
                        ),
                        artifacts=release_artifacts,
                    ),
                )
                summary[key] = {
                    **catalogue_fields,
                    "rows": len(frame),
                    "inserted": result.changed_rows,
                    "skipped": result.unchanged_rows,
                    "removed": result.removed_rows,
                    "release_id": result.release_id,
                    "release_created": result.created,
                    "source_artifact_sha256": frame.attrs["source_artifact_sha256"],
                    "source_artifact_path": frame.attrs["source_artifact_path"],
                    "native_payload_sha256": frame.attrs["native_payload_sha256"],
                    "missing_provenance_sha256": frame.attrs["missing_provenance_sha256"],
                    "missing_native_periods": len(frame.attrs.get("missing_period_records", ())),
                    "artifact_roles": [artifact.role for artifact in release_artifacts],
                }
                logger.info(
                    "Fetched %s (%s): %d rows (%d new/updated, %d unchanged)",
                    key,
                    spec.native_series_id,
                    len(frame),
                    result.changed_rows,
                    result.unchanged_rows,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate native partitions
                session.rollback()
                logger.exception("Failed %s (%s): %s", key, spec.native_series_id, exc)
                summary[key] = {**catalogue_fields, "error": str(exc)}
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch pinned native broad-money and central-bank-assets histories "
            "from official endpoints."
        )
    )
    parser.add_argument(
        "series_ids",
        nargs="*",
        help="Optional exact native series IDs. Default: the complete pinned catalogue.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Bypass the on-disk HTTP cache.",
    )
    parser.add_argument(
        "--allow-contraction",
        action="store_true",
        help=(
            "Allow a verified publisher snapshot to contain fewer observations than "
            "the latest stored release (unsafe unless the removal was checked)."
        ),
    )
    args = parser.parse_args()

    by_id = {spec.native_series_id: spec for spec in MONEY_LIQUIDITY_SERIES}
    unknown = sorted(set(args.series_ids) - set(by_id))
    if unknown:
        parser.error(f"unknown native series IDs: {', '.join(unknown)}")
    specs = tuple(by_id[series_id] for series_id in args.series_ids) if args.series_ids else None

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected = specs or MONEY_LIQUIDITY_SERIES
    summary = run_pipeline(
        selected,
        use_cache=not args.no_cache,
        allow_contraction=args.allow_contraction,
    )
    failed = 0
    for key, stats in summary.items():
        if "error" in stats:
            print(f"✗ {key}: {stats['error']}")
            failed += 1
        else:
            print(
                f"✓ {key} ({stats['series_id']}): {stats['rows']} rows, "
                f"{stats['unit']}, {stats['frequency']}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
