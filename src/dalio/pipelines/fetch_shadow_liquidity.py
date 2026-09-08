"""Fetch the first non-additive shadow-liquidity evidence layer.

BIS offshore credit and OFR money-market/repo observations are deliberately
stored as independent native-series releases.  This pipeline never combines
currencies, totals and components, stocks and flows, or rates and quantities.
"""

from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, select

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    SOURCE_BIS_GLI,
    BisGlobalLiquiditySeries,
    BisGlobalLiquiditySource,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    OFR_SHADOW_LIQUIDITY_SERIES,
    SOURCE_OFR_STFM,
    OfrShadowLiquiditySeries,
    OfrShadowLiquiditySource,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.storage.db import ReleaseObservation, init_db, make_engine, make_session_factory
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

ShadowLiquiditySeries = BisGlobalLiquiditySeries | OfrShadowLiquiditySeries
SHADOW_LIQUIDITY_SERIES: tuple[ShadowLiquiditySeries, ...] = (
    *BIS_GLOBAL_LIQUIDITY_SERIES,
    *OFR_SHADOW_LIQUIDITY_SERIES,
)
SHADOW_CATALOGUE_SCHEMA_VERSION = 1


def _semantic_records(specs: Iterable[ShadowLiquiditySeries]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for spec in sorted(specs, key=lambda item: item.native_series_id):
        record = asdict(spec)
        record["expected_start"] = spec.expected_start.isoformat()
        records.append(record)
    return records


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def archive_shadow_catalogues(artifact_dir: Path | None = None) -> dict[str, Path]:
    """Persist reconstructable, content-addressed semantic catalogues."""
    root = artifact_dir or Path(
        os.environ.get(
            "DALIO_SHADOW_LIQUIDITY_ARTIFACTS",
            "data/artifacts/liquidity_frontier",
        )
    )
    catalogues: tuple[tuple[str, tuple[ShadowLiquiditySeries, ...], str], ...] = (
        (
            SOURCE_BIS_GLI,
            BIS_GLOBAL_LIQUIDITY_SERIES,
            bis_global_liquidity_catalogue_sha256(),
        ),
        (
            SOURCE_OFR_STFM,
            OFR_SHADOW_LIQUIDITY_SERIES,
            ofr_shadow_liquidity_catalogue_sha256(),
        ),
    )
    archived: dict[str, Path] = {}
    for source_family, specs, semantic_hash in catalogues:
        payload = {
            "schema_version": SHADOW_CATALOGUE_SCHEMA_VERSION,
            "source_family": source_family,
            "catalogue_semantic_sha256": semantic_hash,
            "series": _semantic_records(specs),
        }
        content = (
            json.dumps(
                payload,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        content_hash = hashlib.sha256(content).hexdigest()
        destination = (
            root
            / "catalogues"
            / source_family.lower()
            / content_hash[:2]
            / f"{content_hash}.catalogue-v{SHADOW_CATALOGUE_SCHEMA_VERSION}.json"
        )
        if destination.exists():
            if destination.read_bytes() != content:
                raise ValueError(f"shadow-liquidity catalogue archive is corrupt: {destination}")
        else:
            _atomic_write(destination, content)
        archived[source_family] = destination
    return archived


def _partition_key(spec: ShadowLiquiditySeries) -> str:
    return make_partition_key(
        spec.source_family,
        spec.native_series_id,
        spec.country,
        spec.indicator,
    )


def _catalogue_identity(spec: ShadowLiquiditySeries) -> tuple[str, str]:
    if isinstance(spec, BisGlobalLiquiditySeries):
        return (
            BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
            bis_global_liquidity_catalogue_sha256(),
        )
    if isinstance(spec, OfrShadowLiquiditySeries):
        return (
            OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
            ofr_shadow_liquidity_catalogue_sha256(),
        )
    raise ValueError(f"unsupported shadow-liquidity specification: {type(spec)!r}")


def _sha256_attr(frame: pd.DataFrame, name: str, spec: ShadowLiquiditySeries) -> str:
    value = frame.attrs.get(name)
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{spec.native_series_id} lacks a valid {name}; release not recorded")
    return value


def _vintage_label(
    spec: ShadowLiquiditySeries,
    frame: pd.DataFrame,
    prefix: str,
    catalogue_hash: str,
) -> tuple[str, str, str, str, str]:
    if isinstance(spec, OfrShadowLiquiditySeries):
        missing_records = frame.attrs.get("missing_period_records")
        if not isinstance(missing_records, (tuple, list)) or any(
            not isinstance(record, dict) for record in missing_records
        ):
            raise ValueError(
                f"{spec.native_series_id} lacks canonical missing-period records; "
                "release not recorded"
            )
        cadence_policy = spec.cadence_policy
        missing_value_policy = spec.missing_value_policy
    else:
        missing_records = ()
        cadence_policy = "complete_quarterly"
        missing_value_policy = (
            "BIS scalar observations must remain complete by reported quarter; "
            "never interpolate or infer zero"
        )
    payload_hash = _sha256_attr(frame, "native_payload_sha256", spec)
    artifact_hash = _sha256_attr(frame, "source_artifact_sha256", spec)
    missing_payload = {
        "native_series_id": spec.native_series_id,
        "cadence_policy": cadence_policy,
        "missing_value_policy": missing_value_policy,
        "records": missing_records,
    }
    provenance_json = json.dumps(
        missing_payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    expected_missing_hash = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
    if isinstance(spec, OfrShadowLiquiditySeries) and (
        _sha256_attr(frame, "missing_provenance_sha256", spec) != expected_missing_hash
    ):
        raise ValueError(
            f"{spec.native_series_id} missing-period provenance hash does not match its records"
        )
    missing_hash = expected_missing_hash
    provenance_hash = hashlib.sha256(
        f"{artifact_hash}:{payload_hash}:{missing_hash}".encode("ascii")
    ).hexdigest()
    # The release content hash already binds observed values. This 96-bit tag
    # additionally binds the exact native response and its null/disclosure map
    # while keeping the existing compact vintage-label schema.
    vintage = f"{prefix}{catalogue_hash}"
    if isinstance(spec, OfrShadowLiquiditySeries):
        vintage = f"{vintage};p:{provenance_hash[:24]}"
    return (
        vintage,
        artifact_hash,
        payload_hash,
        missing_hash,
        provenance_json,
    )


def _release_artifacts(
    spec: ShadowLiquiditySeries,
    frame: pd.DataFrame,
    *,
    source_artifact_hash: str,
    native_payload_hash: str,
    missing_provenance_hash: str,
    provenance_json: str,
) -> tuple[ReleaseArtifactMeta, ...]:
    """Build a byte-verified manifest for one native-series release."""

    common = {
        "native_payload_sha256": native_payload_hash,
        "missing_provenance_sha256": missing_provenance_hash,
        "provenance_json": provenance_json,
    }
    artifacts = [
        ReleaseArtifactMeta(
            role="source_response",
            artifact_sha256=source_artifact_hash,
            artifact_path=frame.attrs.get("source_artifact_path", ""),
            **common,
        )
    ]
    if isinstance(spec, OfrShadowLiquiditySeries):
        artifacts.extend(
            (
                ReleaseArtifactMeta(
                    role="native_series_payload",
                    artifact_sha256=native_payload_hash,
                    artifact_path=frame.attrs.get("native_payload_artifact_path", ""),
                    **common,
                ),
                ReleaseArtifactMeta(
                    role="missingness_ledger",
                    artifact_sha256=missing_provenance_hash,
                    artifact_path=frame.attrs.get("missing_provenance_artifact_path", ""),
                    **common,
                ),
            )
        )
    return tuple(artifacts)


def _coerce_dates(values: Iterable[object], *, label: str) -> list[date]:
    parsed: list[date] = []
    for value in values:
        if isinstance(value, datetime):
            parsed.append(value.date())
        elif isinstance(value, date):
            parsed.append(value)
        else:
            try:
                timestamp = pd.to_datetime(value, errors="raise")
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{label} contains an invalid date: {value!r}") from exc
            if pd.isna(timestamp):
                raise ValueError(f"{label} contains a missing date")
            parsed.append(timestamp.date())
    return parsed


def _ofr_native_dates(
    frame: pd.DataFrame,
    spec: OfrShadowLiquiditySeries,
) -> list[date]:
    if "native_periods" not in frame.attrs:
        raise ValueError(
            f"{spec.native_series_id} lacks native-period provenance; release not recorded"
        )
    native_dates = _coerce_dates(
        frame.attrs["native_periods"],
        label=f"{spec.native_series_id} native periods",
    )
    if not native_dates:
        raise ValueError(f"{spec.native_series_id} has no native periods")
    if native_dates != sorted(native_dates) or len(native_dates) != len(set(native_dates)):
        raise ValueError(f"{spec.native_series_id} native periods must be unique and increasing")

    disclosure_raw = frame.attrs.get("disclosure_edit_dates", ())
    null_raw = frame.attrs.get("unclassified_null_dates", ())
    disclosure_dates = _coerce_dates(
        disclosure_raw,
        label=f"{spec.native_series_id} disclosure edits",
    )
    null_dates = _coerce_dates(
        null_raw,
        label=f"{spec.native_series_id} null periods",
    )
    if spec.requires_disclosure_subseries and "disclosure_edit_dates" not in frame.attrs:
        raise ValueError(
            f"{spec.native_series_id} lacks disclosure-edit provenance; release not recorded"
        )
    native_set = set(native_dates)
    outside = (set(null_dates) | set(disclosure_dates)) - native_set
    if outside:
        raise ValueError(
            f"{spec.native_series_id} missing-value provenance falls outside native periods"
        )
    return native_dates


def _validate_partition(
    frame: pd.DataFrame,
    spec: ShadowLiquiditySeries,
    *,
    as_of: datetime,
) -> None:
    if frame.empty:
        raise ValueError(
            f"{spec.delivery_service} returned an empty snapshot; release not recorded"
        )
    missing = _CORE_COLUMNS - set(frame.columns)
    if missing:
        raise ValueError(f"shadow-liquidity snapshot missing columns: {sorted(missing)}")

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
                f"shadow-liquidity snapshot {label} mismatch: expected {wanted!r}, got {values!r}"
            )

    observed_dates = _coerce_dates(
        frame["date"].tolist(),
        label=f"{spec.native_series_id} observations",
    )
    if len(observed_dates) != len(set(observed_dates)):
        raise ValueError(f"{spec.native_series_id} has duplicate observation dates")
    dates = (
        _ofr_native_dates(frame, spec)
        if isinstance(spec, OfrShadowLiquiditySeries)
        else sorted(observed_dates)
    )
    if isinstance(spec, OfrShadowLiquiditySeries) and not set(observed_dates) <= set(dates):
        raise ValueError(
            f"{spec.native_series_id} observations fall outside its native-period ledger"
        )
    if dates[0] != spec.expected_start:
        raise ValueError(
            f"{spec.native_series_id} expected start {spec.expected_start.isoformat()}, "
            f"got {dates[0].isoformat()}"
        )
    if len(frame) < spec.minimum_observations:
        raise ValueError(
            f"{spec.native_series_id} has {len(frame)} observations; minimum observation "
            f"count is {spec.minimum_observations}"
        )

    try:
        values = pd.to_numeric(frame["value"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{spec.native_series_id} has a non-numeric observation") from exc
    if not values.map(math.isfinite).all():
        raise ValueError(f"{spec.native_series_id} has a non-finite observation")
    if getattr(spec, "measure_kind", None) != "volume_weighted_mean_rate" and (values < 0).any():
        raise ValueError(f"{spec.native_series_id} has a negative stock or volume")

    if spec.frequency == "quarterly":
        if any(item.day != 1 or item.month not in {1, 4, 7, 10} for item in dates):
            raise ValueError(f"{spec.native_series_id} dates must use quarter start")
        ordinals = [item.year * 4 + (item.month - 1) // 3 for item in dates]
        if any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
            raise ValueError(f"{spec.native_series_id} has a gap in quarterly cadence")
    elif spec.frequency == "monthly":
        if any(item.day != calendar.monthrange(item.year, item.month)[1] for item in dates):
            raise ValueError(f"{spec.native_series_id} monthly dates must be month end")
        if getattr(spec, "cadence_policy", "sparse_monthly") == "complete_monthly":
            ordinals = [item.year * 12 + item.month for item in dates]
            if any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
                raise ValueError(f"{spec.native_series_id} has a gap in monthly cadence")
            observed_ordinals = sorted(item.year * 12 + item.month for item in observed_dates)
            if set(observed_dates) != set(dates) or any(
                right - left != 1
                for left, right in zip(observed_ordinals, observed_ordinals[1:], strict=False)
            ):
                raise ValueError(
                    f"{spec.native_series_id} complete monthly history has a missing "
                    "numeric observation"
                )
        # Sparse OFR counterparty histories may publish explicit null months.
        # They remain missing and are never converted to zero or interpolated.
    elif spec.frequency == "daily":
        if any(item.weekday() >= 5 for item in dates):
            raise ValueError(f"{spec.native_series_id} daily dates must be weekdays")
        # Holidays, no-trading days and confidentiality edits are legitimate gaps.
        assert isinstance(spec, OfrShadowLiquiditySeries)
        if spec.max_internal_gap_days is None or spec.minimum_weekday_coverage_ratio is None:
            raise ValueError(f"{spec.native_series_id} lacks daily completeness guards")
        gaps = [(right - left).days for left, right in zip(dates, dates[1:], strict=False)]
        if gaps and max(gaps) > spec.max_internal_gap_days:
            raise ValueError(
                f"{spec.native_series_id} has an internal daily gap of {max(gaps)} "
                f"days; maximum is {spec.max_internal_gap_days}"
            )
        expected_weekdays = sum(
            1
            for offset in range((dates[-1] - dates[0]).days + 1)
            if (dates[0] + timedelta(days=offset)).weekday() < 5
        )
        coverage = len(dates) / expected_weekdays
        if coverage < spec.minimum_weekday_coverage_ratio:
            raise ValueError(
                f"{spec.native_series_id} weekday coverage is {coverage:.3f}; minimum "
                f"is {spec.minimum_weekday_coverage_ratio:.3f}"
            )
    else:
        raise ValueError(f"unsupported shadow-liquidity frequency: {spec.frequency!r}")

    as_of_date = as_of.date()
    latest_native = dates[-1]
    native_lag = (as_of_date - latest_native).days
    if native_lag < 0:
        raise ValueError(
            f"{spec.native_series_id} latest native period {latest_native.isoformat()} is after "
            f"retrieval date {as_of_date.isoformat()}"
        )
    if native_lag > spec.max_latest_lag_days:
        raise ValueError(
            f"{spec.native_series_id} latest native period is stale by {native_lag} "
            f"days; maximum is {spec.max_latest_lag_days}"
        )
    latest_observed = max(observed_dates)
    observed_lag = (as_of_date - latest_observed).days
    if observed_lag < 0:
        raise ValueError(
            f"{spec.native_series_id} latest observation {latest_observed.isoformat()} is "
            f"after retrieval date {as_of_date.isoformat()}"
        )
    if observed_lag > spec.max_latest_lag_days:
        raise ValueError(
            f"{spec.native_series_id} latest numeric observation is stale by "
            f"{observed_lag} days; maximum is {spec.max_latest_lag_days}"
        )


def _catalogue_fields(spec: ShadowLiquiditySeries, digest: str) -> dict[str, Any]:
    fields: dict[str, Any] = {
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
        "source_family": spec.source_family,
        "series_id": spec.native_series_id,
        "title": spec.title,
        "expected_start": spec.expected_start.isoformat(),
        "minimum_observations": spec.minimum_observations,
        "max_latest_lag_days": spec.max_latest_lag_days,
        "measure_kind": spec.measure_kind,
        "aggregation_role": spec.aggregation_role,
        "catalogue_semantic_sha256": digest,
    }
    if isinstance(spec, BisGlobalLiquiditySeries):
        fields.update(
            {
                "economic_side": spec.claim_side,
                "from_sector": spec.from_sector,
                "to_sector": spec.to_sector,
                "instrument": spec.instrument,
                "parent_series_id": None,
                "component_axis": "currency",
                "venue": None,
                "non_additive_groups": [spec.non_additive_group],
                "research_role": "primary",
            }
        )
    else:
        fields.update(
            {
                "dataset": spec.dataset,
                "economic_side": spec.economic_side,
                "claim_side": getattr(spec, "claim_side", None),
                "from_sector": getattr(spec, "from_sector", None),
                "to_sector": getattr(spec, "to_sector", None),
                "instrument": getattr(spec, "instrument", None),
                "collateral_scope": getattr(spec, "collateral_scope", None),
                "cadence_policy": getattr(spec, "cadence_policy", None),
                "max_internal_gap_days": spec.max_internal_gap_days,
                "minimum_weekday_coverage_ratio": spec.minimum_weekday_coverage_ratio,
                "native_date_alignment_group": spec.native_date_alignment_group,
                "parent_series_id": spec.parent_native_series_id,
                "component_axis": spec.component_axis,
                "venue": spec.venue,
                "non_additive_groups": list(spec.non_additive_groups),
                "missing_value_policy": spec.missing_value_policy,
                "research_role": spec.research_role,
            }
        )
    return fields


def run_pipeline(
    specs: Iterable[ShadowLiquiditySeries] = SHADOW_LIQUIDITY_SERIES,
    sources: Mapping[str, Any] | None = None,
    use_cache: bool = True,
    *,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
    allow_contraction: bool = False,
) -> dict[str, dict[str, Any]]:
    """Store each native frontier series independently and isolate failures."""
    selected = tuple(specs)
    source_map: dict[str, Any] = dict(sources or {})
    families = {spec.source_family for spec in selected}
    if SOURCE_BIS_GLI in families and SOURCE_BIS_GLI not in source_map:
        source_map[SOURCE_BIS_GLI] = BisGlobalLiquiditySource()
    if SOURCE_OFR_STFM in families and SOURCE_OFR_STFM not in source_map:
        source_map[SOURCE_OFR_STFM] = OfrShadowLiquiditySource()

    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)

    # OFR supports a single exact multi-series request. If either transport or
    # parsing fails, retry native series independently so one bad partition is
    # visible without suppressing all valid peers.
    ofr_specs = tuple(spec for spec in selected if isinstance(spec, OfrShadowLiquiditySeries))
    bulk_ofr_frames: dict[str, pd.DataFrame] | None = None
    bulk_ofr_retrieved_at: datetime | None = None
    if ofr_specs:
        ofr_source = source_map.get(SOURCE_OFR_STFM)
        fetch_many = getattr(ofr_source, "fetch_many", None)
        if callable(fetch_many):
            try:
                bulk_ofr_frames = fetch_many(ofr_specs, use_cache=use_cache)
                bulk_ofr_retrieved_at = retrieved_at or datetime.now(UTC)
            except Exception as exc:  # noqa: BLE001 -- intentionally isolate fallback
                logger.warning("OFR bulk request failed; retrying each series: %s", exc)

    summary: dict[str, dict[str, Any]] = {}
    with session_factory() as session:
        for spec in selected:
            key = f"{spec.source_family}/{spec.country}/{spec.indicator}"
            try:
                prefix, catalogue_hash = _catalogue_identity(spec)
                fields = _catalogue_fields(spec, catalogue_hash)
            except Exception as exc:  # noqa: BLE001 -- isolate malformed catalogue entry
                summary[key] = {"series_id": spec.native_series_id, "error": str(exc)}
                continue
            try:
                if isinstance(spec, OfrShadowLiquiditySeries) and bulk_ofr_frames is not None:
                    frame = bulk_ofr_frames[spec.native_series_id]
                    release_at = bulk_ofr_retrieved_at
                else:
                    source = source_map.get(spec.source_family)
                    if source is None:
                        raise ValueError(
                            f"no shadow-liquidity source configured for {spec.source_family}"
                        )
                    frame = source.fetch(spec, use_cache=use_cache)
                    release_at = retrieved_at or datetime.now(UTC)
                if release_at is None:  # defensive: only possible with a broken bulk source
                    raise ValueError("retrieval clock was not captured")
                _validate_partition(frame, spec, as_of=release_at)

                partition_key = _partition_key(spec)
                previous = latest_release(session, partition_key)
                if previous is not None and not allow_contraction:
                    previous_dates = set(
                        session.scalars(
                            select(ReleaseObservation.date).where(
                                ReleaseObservation.release_id == previous.id
                            )
                        )
                    )
                    incoming_dates = set(
                        _coerce_dates(
                            frame["date"].tolist(),
                            label=f"{spec.native_series_id} observations",
                        )
                    )
                    removed_dates = previous_dates - incoming_dates
                    if removed_dates or len(frame) < previous.row_count:
                        first_removed = min(removed_dates).isoformat() if removed_dates else None
                        detail = (
                            f"; first removed date {first_removed}"
                            if first_removed is not None
                            else ""
                        )
                        raise ValueError(
                            f"{spec.native_series_id} contracts complete observed history from "
                            f"{previous.row_count} to {len(frame)} rows{detail}; rerun with "
                            "--allow-contraction only after verifying the publisher removal "
                            "or confidentiality edit"
                        )

                (
                    vintage_label,
                    source_artifact_hash,
                    native_payload_hash,
                    missing_provenance_hash,
                    provenance_json,
                ) = _vintage_label(spec, frame, prefix, catalogue_hash)

                release_artifacts = _release_artifacts(
                    spec,
                    frame,
                    source_artifact_hash=source_artifact_hash,
                    native_payload_hash=native_payload_hash,
                    missing_provenance_hash=missing_provenance_hash,
                    provenance_json=provenance_json,
                )

                result = ingest_release_snapshot(
                    session,
                    frame,
                    ReleaseMeta(
                        partition_key=partition_key,
                        source_family=spec.source_family,
                        published_at=(
                            frame.attrs.get("published_at")
                            or frame.attrs.get("publisher_last_updated_at")
                        ),
                        available_at=release_at,
                        retrieved_at=release_at,
                        vintage_label=vintage_label,
                        source_url=frame.attrs.get("source_url", spec.url),
                        projection=ProjectionScope(
                            country=spec.country,
                            indicator=spec.indicator,
                            sources=(spec.source_family,),
                        ),
                        artifacts=release_artifacts,
                    ),
                )
                summary[key] = {
                    **fields,
                    "rows": len(frame),
                    "native_period_count": len(frame.attrs.get("native_periods", frame)),
                    "missing_native_periods": (len(frame.attrs.get("missing_period_records", ()))),
                    "native_payload_sha256": native_payload_hash,
                    "missing_provenance_sha256": missing_provenance_hash,
                    "source_artifact_sha256": source_artifact_hash,
                    "source_artifact_path": frame.attrs.get("source_artifact_path"),
                    "artifact_roles": [artifact.role for artifact in release_artifacts],
                    "inserted": result.changed_rows,
                    "skipped": result.unchanged_rows,
                    "removed": result.removed_rows,
                    "release_id": result.release_id,
                    "release_created": result.created,
                }
                logger.info(
                    "Fetched %s (%s): %d observed rows (%d new/updated, %d unchanged)",
                    key,
                    spec.native_series_id,
                    len(frame),
                    result.changed_rows,
                    result.unchanged_rows,
                )
            except Exception as exc:  # noqa: BLE001 -- isolate native partitions
                session.rollback()
                logger.exception("Failed %s (%s): %s", key, spec.native_series_id, exc)
                summary[key] = {**fields, "error": str(exc)}
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch pinned BIS offshore-credit and OFR MMF/repo histories as "
            "separate, explicitly non-additive releases."
        )
    )
    parser.add_argument(
        "series_ids",
        nargs="*",
        help="Optional exact native series IDs. Default: the complete pinned catalogue.",
    )
    parser.add_argument("--no-cache", action="store_true", help="Bypass HTTP caches.")
    parser.add_argument(
        "--allow-contraction",
        action="store_true",
        help=(
            "Allow a verified publisher snapshot to contain fewer observed rows than "
            "the latest stored release."
        ),
    )
    args = parser.parse_args(argv)

    by_id = {spec.native_series_id: spec for spec in SHADOW_LIQUIDITY_SERIES}
    unknown = sorted(set(args.series_ids) - set(by_id))
    if unknown:
        parser.error(f"unknown native series IDs: {', '.join(unknown)}")
    selected = (
        tuple(by_id[series_id] for series_id in args.series_ids)
        if args.series_ids
        else SHADOW_LIQUIDITY_SERIES
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    catalogue_paths = archive_shadow_catalogues()
    for source_family, path in catalogue_paths.items():
        logger.info("Archived %s semantic catalogue at %s", source_family, path)
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
                f"✓ {key} ({stats['series_id']}): {stats['rows']} observed rows, "
                f"{stats['unit']}, {stats['frequency']}"
            )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
