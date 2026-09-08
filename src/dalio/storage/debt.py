"""Immutable sovereign debt-holder snapshots and point-in-time queries."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from dalio.storage.db import DataRelease, DebtHolderPosition
from dalio.storage.releases import ReleaseMeta

_CORE_COLUMNS = [
    "country",
    "date",
    "issuer_sector_code",
    "issuer_sector_label",
    "instrument_code",
    "instrument_label",
    "holder_sector_code",
    "holder_sector_label",
    "measure_code",
    "measure_label",
    "unit",
    "value",
    "source",
    "series_id",
]
_STRING_COLUMNS = [column for column in _CORE_COLUMNS if column not in {"date", "value"}]
_CELL_KEY = [
    "country",
    "date",
    "issuer_sector_code",
    "instrument_code",
    "holder_sector_code",
    "measure_code",
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
class DebtIngestResult:
    release_id: int
    created: bool
    row_count: int


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _normalize_meta(meta: ReleaseMeta) -> tuple[datetime | None, datetime, datetime]:
    if meta.projection is not None:
        raise ValueError("debt-holder releases do not use an observations projection")
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
        raise ValueError("debt-holder release snapshot must not be empty")
    missing = set(_CORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"debt-holder snapshot missing columns: {sorted(missing)}")

    columns = [*_CORE_COLUMNS, *(["status"] if "status" in frame.columns else [])]
    work = frame.loc[:, columns].copy()
    if work[_STRING_COLUMNS].isna().any().any():
        raise ValueError("debt-holder snapshot string fields must not be null")
    for column in _STRING_COLUMNS:
        work[column] = work[column].astype(str).str.strip()
        if (work[column] == "").any():
            raise ValueError(f"debt-holder snapshot {column} must not be empty")

    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.date
    work["value"] = pd.to_numeric(work["value"], errors="raise").astype(float)
    if not np.isfinite(work["value"]).all():
        raise ValueError("debt-holder snapshot values must be finite")
    work.loc[work["value"] == 0.0, "value"] = 0.0

    if "status" not in work:
        work["status"] = "observed"
    else:
        if work["status"].isna().any():
            raise ValueError("debt-holder snapshot status must not be null")
        work["status"] = work["status"].astype(str).str.strip().str.lower()
        if (work["status"] == "").any():
            raise ValueError("debt-holder snapshot status must not be empty")

    if work.duplicated(subset=_CELL_KEY, keep=False).any():
        raise ValueError("debt-holder snapshot contains a duplicate dimensional cell")
    return work.sort_values(_CELL_KEY, kind="stable").reset_index(drop=True)


def _content_hash(frame: pd.DataFrame) -> str:
    records = []
    for row in frame.itertuples(index=False):
        record = {column: getattr(row, column) for column in _CORE_COLUMNS if column != "value"}
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


def ingest_debt_holder_snapshot(
    session: Session,
    frame: pd.DataFrame,
    meta: ReleaseMeta,
) -> DebtIngestResult:
    """Append one complete holder-position partition; identical content deduplicates."""
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
            return DebtIngestResult(existing.id, False, existing.row_count)

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
        session.execute(insert(DebtHolderPosition), rows)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return DebtIngestResult(release.id, True, len(work))


def load_debt_holder_vintage(
    session: Session,
    as_known_at: datetime,
    *,
    partition_keys: tuple[str, ...] | None = None,
    countries: tuple[str, ...] | None = None,
    instrument_codes: tuple[str, ...] | None = None,
    holder_sector_codes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Load latest complete holder release per partition known at the cutoff.

    Release ranking precedes cell filters, preventing an omitted cell from being
    resurrected from an older source response.
    """
    cutoff = _utc_naive(as_known_at)
    ranked_stmt = select(
        DataRelease.id.label("release_id"),
        func.row_number().over(
            partition_by=DataRelease.partition_key,
            order_by=(
                DataRelease.available_at.desc(),
                DataRelease.retrieved_at.desc(),
                DataRelease.id.desc(),
            ),
        ).label("release_rank"),
    ).where(DataRelease.available_at <= cutoff)
    if partition_keys:
        ranked_stmt = ranked_stmt.where(DataRelease.partition_key.in_(partition_keys))
    ranked = ranked_stmt.subquery()

    fact_columns = [getattr(DebtHolderPosition, column) for column in _CORE_COLUMNS]
    stmt = (
        select(
            DebtHolderPosition.release_id,
            DataRelease.partition_key,
            DataRelease.source_family,
            DataRelease.published_at,
            DataRelease.available_at,
            DataRelease.retrieved_at,
            DataRelease.vintage_label,
            *fact_columns,
            DebtHolderPosition.status,
        )
        .select_from(DebtHolderPosition)
        .join(ranked, ranked.c.release_id == DebtHolderPosition.release_id)
        .join(DataRelease, DataRelease.id == DebtHolderPosition.release_id)
        .where(ranked.c.release_rank == 1)
    )
    if countries:
        stmt = stmt.where(DebtHolderPosition.country.in_(countries))
    if instrument_codes:
        stmt = stmt.where(DebtHolderPosition.instrument_code.in_(instrument_codes))
    if holder_sector_codes:
        stmt = stmt.where(DebtHolderPosition.holder_sector_code.in_(holder_sector_codes))
    stmt = stmt.order_by(
        DebtHolderPosition.country,
        DebtHolderPosition.date,
        DebtHolderPosition.instrument_code,
        DebtHolderPosition.holder_sector_code,
    )
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
