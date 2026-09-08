"""Immutable, page-located disclosures from systemically important allocators."""

from __future__ import annotations

import hashlib
import json
import math
import re
from contextlib import suppress
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
from sqlalchemy import func, insert, select
from sqlalchemy.orm import Session

from dalio.storage.db import AllocatorFact, DataRelease

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CORE_COLUMNS = [
    "fund",
    "as_of_date",
    "period_start",
    "period_end",
    "record_type",
    "item_code",
    "reported_amount",
    "reported_unit",
    "amount_sek_mn",
    "exposure_pct",
    "basis",
    "row_role",
    "physical_page",
    "table_heading",
    "extraction_status",
    "quality_flag",
    "notes",
]
_REQUIRED_STRINGS = [
    "fund",
    "record_type",
    "item_code",
    "basis",
    "row_role",
    "table_heading",
    "extraction_status",
    "quality_flag",
]
_SEMANTIC_KEY = [
    "fund",
    "as_of_date",
    "period_start",
    "period_end",
    "record_type",
    "item_code",
    "basis",
    "row_role",
    "physical_page",
    "table_heading",
]
_STORED_COLUMNS = [
    "fact_key",
    *_CORE_COLUMNS,
    "artifact_sha256",
    "artifact_path",
    "source_url",
    "parser_name",
    "parser_version",
]
_OUTPUT_COLUMNS = [
    "release_id",
    "partition_key",
    "source_family",
    "published_at",
    "available_at",
    "retrieved_at",
    "vintage_label",
    *_STORED_COLUMNS,
]


@dataclass(frozen=True)
class AllocatorReleaseMeta:
    partition_key: str
    source_family: str
    fund: str
    report_date: date
    title: str
    available_at: datetime
    retrieved_at: datetime
    source_url: str
    official_domains: tuple[str, ...]
    expected_sha256: str
    parser_name: str
    parser_version: str
    published_at: datetime | None = None


@dataclass(frozen=True)
class AllocatorIngestResult:
    release_id: int
    created: bool
    row_count: int
    artifact_sha256: str
    blob_path: Path


def _required(value: str, field: str) -> str:
    clean = str(value).strip()
    if not clean:
        raise ValueError(f"{field} must not be empty")
    return clean


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _normalize_meta(
    meta: AllocatorReleaseMeta,
) -> tuple[datetime | None, datetime, datetime, tuple[str, ...]]:
    _required(meta.partition_key, "partition_key")
    _required(meta.source_family, "source_family")
    _required(meta.fund, "fund")
    _required(meta.title, "title")
    _required(meta.parser_name, "parser_name")
    _required(meta.parser_version, "parser_version")
    if not isinstance(meta.report_date, date):
        raise ValueError("report_date must be a date")
    if not _SHA256_RE.fullmatch(meta.expected_sha256):
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

    published_at = _utc_naive(meta.published_at) if meta.published_at else None
    available_at = _utc_naive(meta.available_at)
    retrieved_at = _utc_naive(meta.retrieved_at)
    if published_at is not None and available_at < published_at:
        raise ValueError("available_at cannot be earlier than published_at")
    if retrieved_at < available_at:
        raise ValueError("retrieved_at cannot be earlier than available_at")

    domains = tuple(domain.strip().lower().strip(".") for domain in meta.official_domains)
    if not domains or any(not domain or "/" in domain or ":" in domain for domain in domains):
        raise ValueError("official domains must be non-empty host names")
    parsed = urlparse(meta.source_url)
    host = (parsed.hostname or "").lower().strip(".")
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise ValueError("source_url must be an HTTPS URL on an official domain")
    if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
        raise ValueError("source_url is outside the official domain allowlist")
    return published_at, available_at, retrieved_at, domains


def _nullable_date(value: object, field: str) -> date | None:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="raise")
    if isinstance(parsed, pd.Timestamp):
        return parsed.date()
    raise ValueError(f"{field} must be a date")


def _nullable_float(value: object, field: str) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be finite")
    return 0.0 if number == 0.0 else number


def _canonicalize(frame: pd.DataFrame, meta: AllocatorReleaseMeta) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("allocator release snapshot must not be empty")
    missing = set(_CORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"allocator snapshot missing columns: {sorted(missing)}")
    work = frame.loc[:, _CORE_COLUMNS].copy()

    for column in _REQUIRED_STRINGS:
        if work[column].isna().any():
            raise ValueError(f"allocator snapshot {column} must not be null")
        work[column] = work[column].astype(str).str.strip()
        if (work[column] == "").any():
            raise ValueError(f"allocator snapshot {column} must not be empty")
    if set(work["fund"]) != {meta.fund}:
        raise ValueError("allocator snapshot fund does not match release metadata")

    work["as_of_date"] = work["as_of_date"].map(lambda value: _nullable_date(value, "as_of_date"))
    if work["as_of_date"].isna().any():
        raise ValueError("allocator snapshot as_of_date must not be null")
    for column in ("period_start", "period_end"):
        work[column] = work[column].map(lambda value, c=column: _nullable_date(value, c))
    one_sided_period = work["period_start"].isna() != work["period_end"].isna()
    if one_sided_period.any():
        raise ValueError("allocator period_start and period_end must both be set or both be null")
    for row in work.itertuples(index=False):
        if row.period_start is not None and row.period_end < row.period_start:
            raise ValueError("allocator period_end cannot be earlier than period_start")

    for column in ("reported_amount", "amount_sek_mn", "exposure_pct"):
        normalized = work[column].map(lambda value, c=column: _nullable_float(value, c))
        # A numeric pandas Series coerces ``None`` back to NaN.  Keep an object
        # dtype so the row-level unit checks, release JSON digest and database
        # insert all retain SQL-null semantics for exposure-only disclosures.
        work[column] = normalized.astype(object).where(normalized.notna(), None)
    if (work["reported_amount"].isna() & work["exposure_pct"].isna()).any():
        raise ValueError("each allocator fact needs a reported amount or exposure percentage")

    units: list[str | None] = []
    for row in work.itertuples(index=False):
        unit = (
            None
            if row.reported_unit is None or pd.isna(row.reported_unit)
            else str(row.reported_unit).strip()
        )
        if row.reported_amount is None:
            if unit:
                raise ValueError("reported_unit requires a reported_amount")
            if row.amount_sek_mn is not None:
                raise ValueError("amount_sek_mn requires a reported_amount")
        else:
            if unit not in {"SEK_mn", "SEK_bn"}:
                raise ValueError("reported amounts must use SEK_mn or SEK_bn")
            if row.amount_sek_mn is None:
                raise ValueError("reported amounts need amount_sek_mn normalization")
            expected = row.reported_amount if unit == "SEK_mn" else row.reported_amount * 1_000
            if not math.isclose(row.amount_sek_mn, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError("amount_sek_mn conflicts with the reported amount and unit")
        units.append(unit)
    work["reported_unit"] = units

    pages = pd.to_numeric(work["physical_page"], errors="raise")
    if (pages % 1 != 0).any() or (pages < 1).any():
        raise ValueError("physical_page must be a positive integer")
    work["physical_page"] = pages.astype(int)
    work["notes"] = work["notes"].map(
        lambda value: None if value is None or pd.isna(value) else str(value).strip() or None
    )

    semantic_payloads = work[_SEMANTIC_KEY].apply(
        lambda row: json.dumps(
            {
                key: value.isoformat() if isinstance(value, date) else value
                for key, value in row.items()
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        axis=1,
    )
    work.insert(
        0,
        "fact_key",
        semantic_payloads.map(lambda payload: hashlib.sha256(payload.encode()).hexdigest()),
    )
    if work["fact_key"].duplicated(keep=False).any():
        raise ValueError("allocator snapshot contains a duplicate semantic fact")
    return work.sort_values("fact_key", kind="stable").reset_index(drop=True)


def _release_digest(
    frame: pd.DataFrame,
    artifact_sha256: str,
    parser_name: str,
    parser_version: str,
) -> str:
    records = []
    for row in frame.itertuples(index=False):
        record = {}
        for column in ["fact_key", *_CORE_COLUMNS]:
            value = getattr(row, column)
            if isinstance(value, date):
                value = value.isoformat()
            record[column] = value
        records.append(record)
    payload = json.dumps(
        {
            "artifact_sha256": artifact_sha256,
            "parser_name": parser_name,
            "parser_version": parser_version,
            "rows": records,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _blob_path(blob_root: Path, digest: str) -> Path:
    return Path(blob_root).resolve() / "sha256" / digest[:2] / f"{digest}.pdf"


def _store_blob(path: Path, pdf_bytes: bytes, digest: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(pdf_bytes)
    except FileExistsError:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"allocator artifact has wrong sha256: {path}") from None
        return False
    return True


def ingest_allocator_snapshot(
    session: Session,
    frame: pd.DataFrame,
    pdf_bytes: bytes,
    meta: AllocatorReleaseMeta,
    *,
    blob_root: Path,
) -> AllocatorIngestResult:
    """Archive one official PDF and append its complete typed fact release."""
    published_at, available_at, retrieved_at, _domains = _normalize_meta(meta)
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("allocator artifact must have a PDF %PDF- header")
    artifact_digest = hashlib.sha256(pdf_bytes).hexdigest()
    if artifact_digest != meta.expected_sha256:
        raise ValueError(
            f"allocator artifact sha256 mismatch: expected {meta.expected_sha256}, "
            f"calculated {artifact_digest}"
        )
    work = _canonicalize(frame, meta)
    parser_name = _required(meta.parser_name, "parser_name")
    parser_version = _required(meta.parser_version, "parser_version")
    digest = _release_digest(work, artifact_digest, parser_name, parser_version)
    artifact_path = _blob_path(blob_root, artifact_digest)

    existing = session.execute(
        select(DataRelease).where(
            DataRelease.partition_key == meta.partition_key.strip(),
            DataRelease.content_sha256 == digest,
        )
    ).scalar_one_or_none()
    if existing is not None:
        if not artifact_path.is_file() or hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest() != (artifact_digest):
            raise ValueError("stored allocator artifact is missing or corrupt")
        session.commit()
        return AllocatorIngestResult(
            existing.id, False, existing.row_count, artifact_digest, artifact_path
        )

    blob_created = False
    try:
        blob_created = _store_blob(artifact_path, pdf_bytes, artifact_digest)
        release = DataRelease(
            partition_key=meta.partition_key.strip(),
            source_family=meta.source_family.strip(),
            published_at=published_at,
            available_at=available_at,
            retrieved_at=retrieved_at,
            vintage_label=meta.title.strip(),
            source_url=meta.source_url.strip(),
            content_sha256=digest,
            row_count=len(work),
        )
        session.add(release)
        session.flush()
        rows = work.to_dict("records")
        for row in rows:
            row.update(
                release_id=release.id,
                artifact_sha256=artifact_digest,
                artifact_path=str(artifact_path),
                source_url=meta.source_url.strip(),
                parser_name=parser_name,
                parser_version=parser_version,
            )
        session.execute(insert(AllocatorFact), rows)
        session.commit()
    except Exception:
        session.rollback()
        if blob_created:
            with suppress(OSError):
                artifact_path.unlink(missing_ok=True)
        raise

    return AllocatorIngestResult(release.id, True, len(work), artifact_digest, artifact_path)


def load_allocator_vintage(
    session: Session,
    as_known_at: datetime,
    *,
    partition_keys: tuple[str, ...] | None = None,
    funds: tuple[str, ...] | None = None,
    record_types: tuple[str, ...] | None = None,
    item_codes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return the latest complete allocator release per partition at a cutoff."""
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

    stored_columns = [getattr(AllocatorFact, column) for column in _STORED_COLUMNS]
    stmt = (
        select(
            AllocatorFact.release_id,
            DataRelease.partition_key,
            DataRelease.source_family,
            DataRelease.published_at,
            DataRelease.available_at,
            DataRelease.retrieved_at,
            DataRelease.vintage_label,
            *stored_columns,
        )
        .select_from(AllocatorFact)
        .join(ranked, ranked.c.release_id == AllocatorFact.release_id)
        .join(DataRelease, DataRelease.id == AllocatorFact.release_id)
        .where(ranked.c.release_rank == 1)
    )
    if funds:
        stmt = stmt.where(AllocatorFact.fund.in_(funds))
    if record_types:
        stmt = stmt.where(AllocatorFact.record_type.in_(record_types))
    if item_codes:
        stmt = stmt.where(AllocatorFact.item_code.in_(item_codes))
    stmt = stmt.order_by(
        AllocatorFact.fund,
        AllocatorFact.as_of_date,
        AllocatorFact.record_type,
        AllocatorFact.item_code,
    )
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)
