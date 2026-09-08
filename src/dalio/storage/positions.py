"""Immutable cross-border position snapshots and point-in-time queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import quote

import numpy as np
import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from dalio.storage.db import CrossBorderPosition, DataRelease
from dalio.storage.releases import ReleaseMeta

_CORE_COLUMNS = [
    "dataset",
    "reporter_country",
    "reporter_code",
    "counterpart_country",
    "counterpart_code",
    "date",
    "direction",
    "accounting_basis",
    "instrument_code",
    "instrument_label",
    "frequency",
    "value",
    "unit",
    "source",
    "native_indicator",
    "reporter_sector_code",
    "counterpart_sector_code",
    "derivation_type",
    "series_id",
]
_REQUIRED_STRING_COLUMNS = [
    column
    for column in _CORE_COLUMNS
    if column
    not in {
        "date",
        "value",
        "reporter_sector_code",
        "counterpart_sector_code",
        "derivation_type",
    }
]
_OPTIONAL_STRING_COLUMNS = [
    "reporter_sector_code",
    "counterpart_sector_code",
    "derivation_type",
]
_CELL_KEY = [
    "dataset",
    "reporter_code",
    "counterpart_code",
    "date",
    "direction",
    "instrument_code",
    "frequency",
    "source",
    "series_id",
]
_OUTPUT_COLUMNS = [
    "release_id",
    "partition_key",
    "source_family",
    "published_at",
    "available_at",
    "retrieved_at",
    "vintage_label",
    *_CORE_COLUMNS,
    "status",
]


@dataclass(frozen=True)
class PositionIngestResult:
    release_id: int
    created: bool
    row_count: int


def make_position_partition_key(
    source_family: str,
    reporter_country: str,
    native_indicator: str,
    frequency: str,
) -> str:
    """Build a stable complete-partition key for one native position series."""
    parts = (source_family, reporter_country, native_indicator, frequency)
    if any(not str(part).strip() for part in parts):
        raise ValueError("position partition key fields must not be empty")
    return "positions:" + ":".join(quote(str(part).strip(), safe="") for part in parts)


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _normalize_meta(meta: ReleaseMeta) -> tuple[datetime | None, datetime, datetime]:
    if meta.projection is not None:
        raise ValueError("cross-border position releases do not use an observations projection")
    if not meta.partition_key.strip() or not meta.source_family.strip():
        raise ValueError("release partition and source family must not be empty")
    published_at = _utc_naive(meta.published_at) if meta.published_at else None
    available_at = _utc_naive(meta.available_at)
    retrieved_at = _utc_naive(meta.retrieved_at)
    if published_at is not None and available_at < published_at:
        raise ValueError("available_at cannot be earlier than published_at")
    if retrieved_at < available_at:
        raise ValueError("retrieved_at cannot be earlier than available_at")
    return published_at, available_at, retrieved_at


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("cross-border position release snapshot must not be empty")
    missing = set(_CORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"cross-border position snapshot missing columns: {sorted(missing)}")

    columns = [*_CORE_COLUMNS, *(["status"] if "status" in frame.columns else [])]
    work = frame.loc[:, columns].copy()
    if work[_REQUIRED_STRING_COLUMNS].isna().any().any():
        raise ValueError("cross-border position string dimensions must not be null")
    for column in _REQUIRED_STRING_COLUMNS:
        work[column] = work[column].astype(str).str.strip()
        if (work[column] == "").any():
            raise ValueError(f"cross-border position {column} must not be empty")

    for column in _OPTIONAL_STRING_COLUMNS:
        supplied = work[column].notna()
        work.loc[supplied, column] = work.loc[supplied, column].astype(str).str.strip()
        if (work.loc[supplied, column] == "").any():
            raise ValueError(f"cross-border position {column} must be null or non-empty")
        work.loc[~supplied, column] = None

    work["dataset"] = work["dataset"].str.upper()
    work["frequency"] = work["frequency"].str.upper()
    work["unit"] = work["unit"].str.upper()
    if not set(work["dataset"]) <= {"PIP", "DIP"}:
        raise ValueError("cross-border position dataset must be PIP or DIP")
    if not set(work["frequency"]) <= {"A", "S"}:
        raise ValueError("cross-border position frequency must be A or S")
    if not (work["unit"] == "USD").all():
        raise ValueError("cross-border positions must use unscaled USD")

    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.date
    work["value"] = pd.to_numeric(work["value"], errors="raise").astype(float)
    if not np.isfinite(work["value"]).all():
        raise ValueError("cross-border position values must be finite")
    work.loc[work["value"] == 0.0, "value"] = 0.0

    if "status" not in work:
        work["status"] = "observed"
    else:
        if work["status"].isna().any():
            raise ValueError("cross-border position status must not be null")
        work["status"] = work["status"].astype(str).str.strip().str.lower()
        if (work["status"] == "").any():
            raise ValueError("cross-border position status must not be empty")
        if work["status"].str.len().gt(24).any():
            raise ValueError("cross-border position status is too long")

    if work.duplicated(subset=_CELL_KEY, keep=False).any():
        raise ValueError("cross-border position snapshot contains a duplicate dimensional cell")
    return work.sort_values(_CELL_KEY, kind="stable").reset_index(drop=True)


def _content_hash(frame: pd.DataFrame) -> str:
    records: list[dict] = []
    for row in frame.itertuples(index=False):
        record = {
            column: getattr(row, column)
            for column in _CORE_COLUMNS
            if column not in {"date", "value"}
        }
        record["date"] = row.date.isoformat()
        record["value"] = format(float(row.value), ".17g")
        record["status"] = row.status
        records.append(record)
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ingest_cross_border_snapshot(
    session: Session,
    frame: pd.DataFrame,
    meta: ReleaseMeta,
) -> PositionIngestResult:
    """Append one complete bilateral-position partition; identical content deduplicates."""
    work = _canonicalize(frame)
    published_at, available_at, retrieved_at = _normalize_meta(meta)
    digest = _content_hash(work)

    try:
        existing = session.execute(
            select(DataRelease).where(
                DataRelease.partition_key == meta.partition_key.strip(),
                DataRelease.content_sha256 == digest,
            )
        ).scalar_one_or_none()
        if existing is not None:
            session.commit()
            return PositionIngestResult(existing.id, False, existing.row_count)

        release = DataRelease(
            partition_key=meta.partition_key.strip(),
            source_family=meta.source_family.strip(),
            published_at=published_at,
            available_at=available_at,
            retrieved_at=retrieved_at,
            vintage_label=meta.vintage_label,
            source_url=meta.source_url,
            content_sha256=digest,
            row_count=len(work),
        )
        session.add(release)
        session.flush()
        rows = work.to_dict("records")
        for row in rows:
            row["release_id"] = release.id
        session.execute(insert(CrossBorderPosition), rows)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return PositionIngestResult(release.id, True, len(work))


def load_cross_border_vintage(
    session: Session,
    as_known_at: datetime,
    *,
    partition_keys: tuple[str, ...] | None = None,
    datasets: tuple[str, ...] | None = None,
    reporter_countries: tuple[str, ...] | None = None,
    counterpart_countries: tuple[str, ...] | None = None,
    directions: tuple[str, ...] | None = None,
    instrument_codes: tuple[str, ...] | None = None,
    frequencies: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load the latest complete position release per partition known at a cutoff.

    Release ranking deliberately happens before dimensional filters. Therefore,
    an omitted bilateral cell cannot be resurrected from an older IMF response.
    """
    cutoff = _utc_naive(as_known_at)
    ranked_stmt = select(
        DataRelease.id.label("release_id"),
        func.row_number()
        .over(
            partition_by=DataRelease.partition_key,
            order_by=(
                DataRelease.available_at.desc(),
                DataRelease.retrieved_at.desc(),
                DataRelease.id.desc(),
            ),
        )
        .label("release_rank"),
    ).where(DataRelease.available_at <= cutoff)
    if partition_keys:
        ranked_stmt = ranked_stmt.where(DataRelease.partition_key.in_(partition_keys))
    ranked = ranked_stmt.subquery()

    fact_columns = [getattr(CrossBorderPosition, column) for column in _CORE_COLUMNS]
    stmt = (
        select(
            CrossBorderPosition.release_id,
            DataRelease.partition_key,
            DataRelease.source_family,
            DataRelease.published_at,
            DataRelease.available_at,
            DataRelease.retrieved_at,
            DataRelease.vintage_label,
            *fact_columns,
            CrossBorderPosition.status,
        )
        .select_from(CrossBorderPosition)
        .join(ranked, ranked.c.release_id == CrossBorderPosition.release_id)
        .join(DataRelease, DataRelease.id == CrossBorderPosition.release_id)
        .where(ranked.c.release_rank == 1)
    )
    if datasets:
        stmt = stmt.where(CrossBorderPosition.dataset.in_(datasets))
    if reporter_countries:
        stmt = stmt.where(CrossBorderPosition.reporter_country.in_(reporter_countries))
    if counterpart_countries:
        stmt = stmt.where(CrossBorderPosition.counterpart_country.in_(counterpart_countries))
    if directions:
        stmt = stmt.where(CrossBorderPosition.direction.in_(directions))
    if instrument_codes:
        stmt = stmt.where(CrossBorderPosition.instrument_code.in_(instrument_codes))
    if frequencies:
        stmt = stmt.where(CrossBorderPosition.frequency.in_(frequencies))
    stmt = stmt.order_by(
        CrossBorderPosition.dataset,
        CrossBorderPosition.reporter_country,
        CrossBorderPosition.counterpart_country,
        CrossBorderPosition.direction,
        CrossBorderPosition.instrument_code,
        CrossBorderPosition.frequency,
        CrossBorderPosition.date,
    )
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
