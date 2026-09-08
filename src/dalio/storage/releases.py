"""Immutable source snapshots and point-in-time observation queries.

The mutable :class:`~dalio.storage.db.Observation` table remains the fast,
backwards-compatible latest projection. This module stores each provider refresh
as a complete snapshot of one stable partition and can reconstruct the latest
release that was available at an earlier instant.

A complete snapshot is deliberate. If a new forecast vintage omits a previously
published year, selecting the newest release makes that row disappear. A
change-only ledger would need tombstones and can otherwise resurrect the stale
forecast during replay. Consecutive snapshots with the same observations and
semantic release identity deduplicate, but content that returns after an
intervening vintage is recorded again (A -> B -> A).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from sqlalchemy import and_, delete, func, insert, or_, select
from sqlalchemy.orm import Session

from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
)

_CORE_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_CURRENT_KEY = ["country", "indicator", "date", "source"]
_STRING_COLUMNS = ["country", "indicator", "source", "series_id"]
_OUTPUT_COLUMNS = [
    "release_id",
    "partition_key",
    "source_family",
    "published_at",
    "available_at",
    "retrieved_at",
    "vintage_label",
    "country",
    "indicator",
    "date",
    "value",
    "source",
    "series_id",
    "status",
]


@dataclass(frozen=True)
class ProjectionScope:
    """Rows in ``observations`` replaced by one complete partition snapshot."""

    country: str
    indicator: str
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ReleaseArtifactMeta:
    """A retained raw artifact and the canonical native-provenance it binds.

    ``artifact_sha256`` hashes the complete file at ``artifact_path``.
    ``provenance_json`` must already use the project's canonical JSON encoding;
    its digest must equal ``missing_provenance_sha256``.
    """

    role: str
    artifact_sha256: str
    artifact_path: str | Path
    native_payload_sha256: str
    missing_provenance_sha256: str
    provenance_json: str


@dataclass(frozen=True)
class ReleaseMeta:
    """Provenance and clocks for a source release.

    ``available_at`` is the point-in-time clock used by historical queries.
    ``retrieved_at`` is when this project obtained the release. When the true
    publication/availability time is unknown, callers must conservatively use
    retrieval time for ``available_at`` instead of backdating it.
    """

    partition_key: str
    source_family: str
    available_at: datetime
    retrieved_at: datetime
    published_at: datetime | None = None
    vintage_label: str | None = None
    source_url: str | None = None
    projection: ProjectionScope | None = None
    artifacts: tuple[ReleaseArtifactMeta, ...] = ()


@dataclass(frozen=True)
class IngestResult:
    release_id: int
    created: bool
    projected: bool
    row_count: int
    changed_rows: int = 0
    unchanged_rows: int = 0
    removed_rows: int = 0


class ReleaseEventConflictError(ValueError):
    """One partition/event clock was reused with different content or provenance."""


@dataclass(frozen=True)
class _NormalizedReleaseArtifact:
    role: str
    artifact_sha256: str
    artifact_path: str
    native_payload_sha256: str
    missing_provenance_sha256: str
    provenance_json: str


def _utc_naive(value: datetime) -> datetime:
    """Normalize timestamps for SQLite while treating naive input as UTC."""
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _validated_sha256(value: object, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"artifact {field} must be a lowercase SHA-256 hex digest")
    return value


def _canonical_provenance_json(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("artifact provenance_json must be a string")
    try:
        payload = json.loads(value)
        canonical = json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("artifact provenance_json must contain valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("artifact provenance_json must encode a JSON object")
    if value != canonical:
        raise ValueError("artifact provenance_json must use canonical JSON encoding")
    return canonical


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as artifact:
            for chunk in iter(lambda: artifact.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ValueError(f"artifact_path cannot be read: {path}") from exc
    return digest.hexdigest()


def _normalize_release_artifact(
    artifact: ReleaseArtifactMeta,
) -> _NormalizedReleaseArtifact:
    if not isinstance(artifact, ReleaseArtifactMeta):
        raise ValueError("ReleaseMeta.artifacts must contain ReleaseArtifactMeta values")

    role = artifact.role
    if (
        not isinstance(role, str)
        or not 1 <= len(role) <= 64
        or role != role.strip()
        or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in role)
    ):
        raise ValueError(
            "artifact role must be 1-64 lowercase ASCII letters, digits, or underscores"
        )

    artifact_sha256 = _validated_sha256(artifact.artifact_sha256, "artifact_sha256")
    native_payload_sha256 = _validated_sha256(
        artifact.native_payload_sha256, "native_payload_sha256"
    )
    missing_provenance_sha256 = _validated_sha256(
        artifact.missing_provenance_sha256, "missing_provenance_sha256"
    )
    provenance_json = _canonical_provenance_json(artifact.provenance_json)
    actual_provenance_sha256 = hashlib.sha256(provenance_json.encode("utf-8")).hexdigest()
    if actual_provenance_sha256 != missing_provenance_sha256:
        raise ValueError("artifact missing_provenance_sha256 does not hash provenance_json")

    try:
        artifact_path = Path(artifact.artifact_path).expanduser().resolve(strict=True)
    except (OSError, TypeError) as exc:
        raise ValueError(f"artifact_path does not exist: {artifact.artifact_path}") from exc
    if not artifact_path.is_file():
        raise ValueError(f"artifact_path is not a regular file: {artifact_path}")
    actual_artifact_sha256 = _sha256_file(artifact_path)
    if actual_artifact_sha256 != artifact_sha256:
        raise ValueError(f"artifact_sha256 does not hash the complete file at {artifact_path}")
    if role == "native_series_payload" and artifact_sha256 != native_payload_sha256:
        raise ValueError("native_series_payload artifact_sha256 must equal native_payload_sha256")
    if role == "missingness_ledger" and (artifact_sha256 != missing_provenance_sha256):
        raise ValueError("missingness_ledger artifact_sha256 must equal missing_provenance_sha256")

    return _NormalizedReleaseArtifact(
        role=role,
        artifact_sha256=artifact_sha256,
        artifact_path=str(artifact_path),
        native_payload_sha256=native_payload_sha256,
        missing_provenance_sha256=missing_provenance_sha256,
        provenance_json=provenance_json,
    )


def _normalize_release_artifacts(
    artifacts: tuple[ReleaseArtifactMeta, ...],
) -> tuple[_NormalizedReleaseArtifact, ...]:
    try:
        normalized = tuple(_normalize_release_artifact(artifact) for artifact in artifacts)
    except TypeError as exc:
        raise ValueError("ReleaseMeta.artifacts must be an iterable") from exc
    roles = [artifact.role for artifact in normalized]
    if len(roles) != len(set(roles)):
        raise ValueError("ReleaseMeta.artifacts contains a duplicate artifact role")
    return tuple(sorted(normalized, key=lambda artifact: artifact.role))


def _normalize_meta(meta: ReleaseMeta) -> ReleaseMeta:
    partition_key = meta.partition_key.strip()
    source_family = meta.source_family.strip()
    if not partition_key:
        raise ValueError("partition_key must not be empty")
    if not source_family:
        raise ValueError("source_family must not be empty")

    available_at = _utc_naive(meta.available_at)
    retrieved_at = _utc_naive(meta.retrieved_at)
    published_at = _utc_naive(meta.published_at) if meta.published_at else None
    if published_at is not None and available_at < published_at:
        raise ValueError("available_at cannot be earlier than published_at")
    if retrieved_at < available_at:
        raise ValueError("retrieved_at cannot be earlier than available_at")

    projection = meta.projection
    if projection is not None:
        sources = tuple(source.strip() for source in projection.sources)
        if not projection.country.strip() or not projection.indicator.strip() or not all(sources):
            raise ValueError("projection scope fields must not be empty")
        if not sources:
            raise ValueError("projection scope needs at least one source")
        projection = ProjectionScope(
            projection.country.strip(),
            projection.indicator.strip(),
            sources,
        )
        if source_family not in projection.sources:
            raise ValueError("projection sources must include source_family")

    vintage_label = meta.vintage_label.strip() if meta.vintage_label else None
    source_url = meta.source_url.strip() if meta.source_url else None

    return replace(
        meta,
        partition_key=partition_key,
        source_family=source_family,
        published_at=published_at,
        available_at=available_at,
        retrieved_at=retrieved_at,
        vintage_label=vintage_label,
        source_url=source_url,
        projection=projection,
    )


def _canonicalize(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        raise ValueError("release snapshot must not be empty")
    missing = set(_CORE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"release snapshot missing columns: {sorted(missing)}")

    columns = [*_CORE_COLUMNS, *(["status"] if "status" in frame.columns else [])]
    work = frame.loc[:, columns].copy()
    if work[_STRING_COLUMNS].isna().any().any():
        raise ValueError("release snapshot string fields must not be null")
    for column in _STRING_COLUMNS:
        work[column] = work[column].astype(str).str.strip()
        if (work[column] == "").any():
            raise ValueError(f"release snapshot {column} must not be empty")

    work["date"] = pd.to_datetime(work["date"], errors="raise").dt.date
    work["value"] = pd.to_numeric(work["value"], errors="raise").astype(float)
    if not np.isfinite(work["value"]).all():
        raise ValueError("release snapshot values must be finite")
    work.loc[work["value"] == 0.0, "value"] = 0.0  # normalize negative zero

    if "status" not in work:
        work["status"] = np.where(
            work["source"].str.endswith("_FCST"),
            "forecast",
            "observed",
        )
    else:
        if work["status"].isna().any():
            raise ValueError("release snapshot status must not be null")
        work["status"] = work["status"].astype(str).str.strip().str.lower()
        if (work["status"] == "").any():
            raise ValueError("release snapshot status must not be empty")

    if work.duplicated(subset=_CURRENT_KEY, keep=False).any():
        raise ValueError(
            "release snapshot contains a duplicate observation key "
            "(country, indicator, date, source)"
        )
    return work.sort_values(
        [*_CURRENT_KEY, "series_id"],
        kind="stable",
    ).reset_index(drop=True)


def _content_hash(frame: pd.DataFrame) -> str:
    records = [
        {
            "country": row.country,
            "indicator": row.indicator,
            "date": row.date.isoformat(),
            "value": format(float(row.value), ".17g"),
            "source": row.source,
            "series_id": row.series_id,
            "status": row.status,
        }
        for row in frame.itertuples(index=False)
    ]
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_projection(frame: pd.DataFrame, scope: ProjectionScope) -> None:
    if set(frame["country"]) != {scope.country}:
        raise ValueError("release snapshot does not match projection country")
    if set(frame["indicator"]) != {scope.indicator}:
        raise ValueError("release snapshot does not match projection indicator")
    unexpected = set(frame["source"]) - set(scope.sources)
    if unexpected:
        raise ValueError(f"release snapshot has sources outside projection scope: {unexpected}")


def _single_identity_value(frame: pd.DataFrame, column: str) -> str:
    values = set(frame[column])
    if len(values) != 1:
        raise ValueError(
            f"release snapshot must contain exactly one scalar {column}, got {sorted(values)!r}"
        )
    return next(iter(values))


def _validate_existing_partition_identity(
    session: Session,
    frame: pd.DataFrame,
    meta: ReleaseMeta,
) -> None:
    """Prevent a stable partition key from being reused for another scalar."""

    country = _single_identity_value(frame, "country")
    indicator = _single_identity_value(frame, "indicator")
    series_id = _single_identity_value(frame, "series_id")
    sources = set(frame["source"])

    releases = session.execute(
        select(DataRelease.id, DataRelease.source_family).where(
            DataRelease.partition_key == meta.partition_key
        )
    ).all()
    if not releases:
        return

    source_families = {row.source_family for row in releases}
    if source_families != {meta.source_family}:
        raise ValueError(
            f"partition {meta.partition_key!r} belongs to source_family "
            f"{sorted(source_families)!r}, not {meta.source_family!r}"
        )

    stored = session.execute(
        select(
            ReleaseObservation.country,
            ReleaseObservation.indicator,
            ReleaseObservation.source,
            ReleaseObservation.series_id,
        )
        .join(DataRelease, DataRelease.id == ReleaseObservation.release_id)
        .where(DataRelease.partition_key == meta.partition_key)
        .distinct()
    ).all()
    if not stored:
        raise ValueError(
            f"partition {meta.partition_key!r} has releases without stored observations"
        )

    expected = {
        "country": {row.country for row in stored},
        "indicator": {row.indicator for row in stored},
        "series_id": {row.series_id for row in stored},
    }
    incoming = {
        "country": {country},
        "indicator": {indicator},
        "series_id": {series_id},
    }
    for field in expected:
        if expected[field] != incoming[field]:
            raise ValueError(
                f"partition {meta.partition_key!r} {field} identity is "
                f"{sorted(expected[field])!r}, not {sorted(incoming[field])!r}"
            )

    stored_sources = {row.source for row in stored}
    if meta.projection is not None:
        outside_scope = stored_sources - set(meta.projection.sources)
        if outside_scope:
            raise ValueError(
                f"partition {meta.partition_key!r} has stored sources outside the "
                f"projection scope: {sorted(outside_scope)!r}"
            )
    elif stored_sources != sources:
        raise ValueError(
            f"partition {meta.partition_key!r} source identity is "
            f"{sorted(stored_sources)!r}, not {sorted(sources)!r}"
        )


def _same_release_fingerprint(
    release: DataRelease,
    content_sha256: str,
    meta: ReleaseMeta,
) -> bool:
    """Compare content and semantic identity, excluding a replaceable URL locator."""

    return (
        release.content_sha256 == content_sha256
        and release.published_at == meta.published_at
        and release.vintage_label == meta.vintage_label
    )


def _exact_event_conflicts(
    release: DataRelease,
    content_sha256: str,
    row_count: int,
    meta: ReleaseMeta,
) -> list[str]:
    conflicts: list[str] = []
    if release.content_sha256 != content_sha256:
        conflicts.append("content_sha256")
    if release.row_count != row_count:
        conflicts.append("row_count")
    if release.source_family != meta.source_family:
        conflicts.append("source_family")
    if release.published_at != meta.published_at:
        conflicts.append("published_at")
    if release.vintage_label != meta.vintage_label:
        conflicts.append("vintage_label")
    if release.source_url != meta.source_url:
        conflicts.append("source_url")
    return conflicts


def _artifact_identity(
    artifact: _NormalizedReleaseArtifact,
) -> tuple[str, str, str, str]:
    """Return semantic identity; the verified local path is only a locator."""

    return (
        artifact.artifact_sha256,
        artifact.native_payload_sha256,
        artifact.missing_provenance_sha256,
        artifact.provenance_json,
    )


def _stored_release_artifacts(
    session: Session,
    release_id: int,
) -> dict[str, _NormalizedReleaseArtifact]:
    rows = (
        session.execute(
            select(DataReleaseArtifact)
            .where(DataReleaseArtifact.release_id == release_id)
            .order_by(DataReleaseArtifact.role)
        )
        .scalars()
        .all()
    )
    stored: dict[str, _NormalizedReleaseArtifact] = {}
    for row in rows:
        try:
            normalized = _normalize_release_artifact(
                ReleaseArtifactMeta(
                    role=row.role,
                    artifact_sha256=row.artifact_sha256,
                    artifact_path=row.artifact_path,
                    native_payload_sha256=row.native_payload_sha256,
                    missing_provenance_sha256=row.missing_provenance_sha256,
                    provenance_json=row.provenance_json,
                )
            )
        except ValueError as exc:
            raise ReleaseEventConflictError(
                f"stored artifact manifest for release {release_id}, role "
                f"{row.role!r}, failed integrity validation: {exc}"
            ) from exc
        if normalized.artifact_path != row.artifact_path:
            raise ReleaseEventConflictError(
                f"stored artifact path for release {release_id}, role {row.role!r}, "
                "is not canonical"
            )
        if normalized.role in stored:
            raise ReleaseEventConflictError(
                f"stored artifact manifest for release {release_id} has duplicate "
                f"role {normalized.role!r}"
            )
        stored[normalized.role] = normalized
    return stored


def _artifact_manifest_delta(
    stored: dict[str, _NormalizedReleaseArtifact],
    incoming: tuple[_NormalizedReleaseArtifact, ...],
) -> tuple[tuple[_NormalizedReleaseArtifact, ...], tuple[str, ...]]:
    missing: list[_NormalizedReleaseArtifact] = []
    conflicts: list[str] = []
    for artifact in incoming:
        prior = stored.get(artifact.role)
        if prior is None:
            missing.append(artifact)
        elif _artifact_identity(prior) != _artifact_identity(artifact):
            conflicts.append(artifact.role)
    return tuple(missing), tuple(conflicts)


def _insert_release_artifacts(
    session: Session,
    release_id: int,
    artifacts: tuple[_NormalizedReleaseArtifact, ...],
) -> None:
    if not artifacts:
        return
    session.execute(
        insert(DataReleaseArtifact),
        [
            {
                "release_id": release_id,
                "role": artifact.role,
                "artifact_sha256": artifact.artifact_sha256,
                "artifact_path": artifact.artifact_path,
                "native_payload_sha256": artifact.native_payload_sha256,
                "missing_provenance_sha256": artifact.missing_provenance_sha256,
                "provenance_json": artifact.provenance_json,
            }
            for artifact in artifacts
        ],
    )


def _backfill_exact_event_artifacts(
    session: Session,
    release_id: int,
    incoming: tuple[_NormalizedReleaseArtifact, ...],
) -> None:
    stored = _stored_release_artifacts(session, release_id)
    missing, conflicts = _artifact_manifest_delta(stored, incoming)
    if conflicts:
        raise ReleaseEventConflictError(
            f"release event artifact conflict for release {release_id}: differing "
            f"roles {', '.join(conflicts)}"
        )
    _insert_release_artifacts(session, release_id, missing)


def _predecessor_artifacts_match(
    session: Session,
    release_id: int,
    incoming: tuple[_NormalizedReleaseArtifact, ...],
) -> bool:
    stored = _stored_release_artifacts(session, release_id)
    missing, conflicts = _artifact_manifest_delta(stored, incoming)
    # A later retrieval cannot prove that newly supplied bytes belonged to an
    # older release clock. Only an exact-clock retry may append missing roles.
    return not missing and not conflicts


def _projection_diff(
    session: Session,
    frame: pd.DataFrame,
    scope: ProjectionScope,
) -> tuple[int, int, int]:
    current = session.execute(
        select(
            Observation.country,
            Observation.indicator,
            Observation.date,
            Observation.source,
            Observation.value,
            Observation.series_id,
        ).where(
            Observation.country == scope.country,
            Observation.indicator == scope.indicator,
            Observation.source.in_(scope.sources),
        )
    ).all()
    old = {
        (row.country, row.indicator, row.date, row.source): (float(row.value), row.series_id)
        for row in current
    }
    new = {
        (row.country, row.indicator, row.date, row.source): (float(row.value), row.series_id)
        for row in frame.itertuples(index=False)
    }
    unchanged = sum(old.get(key) == value for key, value in new.items())
    changed = len(new) - unchanged
    removed = len(set(old) - set(new))
    return changed, unchanged, removed


def _replace_projection(
    session: Session,
    frame: pd.DataFrame,
    scope: ProjectionScope,
    fetched_at: datetime,
) -> tuple[int, int, int]:
    changed, unchanged, removed = _projection_diff(session, frame, scope)
    session.execute(
        delete(Observation).where(
            Observation.country == scope.country,
            Observation.indicator == scope.indicator,
            Observation.source.in_(scope.sources),
        )
    )
    rows = frame.loc[:, _CORE_COLUMNS].to_dict("records")
    for row in rows:
        row["fetched_at"] = fetched_at
    session.execute(insert(Observation), rows)
    return changed, unchanged, removed


def ingest_release_snapshot(
    session: Session,
    frame: pd.DataFrame,
    meta: ReleaseMeta,
    *,
    replace_current: bool = True,
) -> IngestResult:
    """Append one complete release and optionally refresh its current projection.

    The append and projection replacement commit atomically. Consecutive
    identical content with the same vintage label and publication time is
    idempotent; A -> B -> A remains three releases. A historical release ingested
    after a newer one is retained for replay but never rolls the current
    projection back. Empty snapshots fail closed, before any current rows can be
    deleted. This function owns the supplied session transaction and commits or
    rolls it back; bind the session to an outer connection transaction when a
    larger batch must be atomic.
    """
    try:
        normalized = _normalize_meta(meta)
        artifacts = _normalize_release_artifacts(normalized.artifacts)
        work = _canonicalize(frame)
        if replace_current:
            if normalized.projection is None:
                raise ValueError("replace_current requires an explicit projection scope")
            _validate_projection(work, normalized.projection)
        digest = _content_hash(work)

        _validate_existing_partition_identity(session, work, normalized)

        exact_existing = session.execute(
            select(DataRelease).where(
                DataRelease.partition_key == normalized.partition_key,
                DataRelease.available_at == normalized.available_at,
                DataRelease.retrieved_at == normalized.retrieved_at,
            )
        ).scalar_one_or_none()
        if exact_existing is not None:
            conflicts = _exact_event_conflicts(
                exact_existing,
                digest,
                len(work),
                normalized,
            )
            if conflicts:
                raise ReleaseEventConflictError(
                    f"release event clock conflict for partition "
                    f"{normalized.partition_key!r} at "
                    f"({normalized.available_at.isoformat()}, "
                    f"{normalized.retrieved_at.isoformat()}): differing "
                    f"{', '.join(conflicts)}"
                )
            _backfill_exact_event_artifacts(
                session,
                exact_existing.id,
                artifacts,
            )
            session.commit()
            return IngestResult(
                release_id=exact_existing.id,
                created=False,
                projected=False,
                row_count=exact_existing.row_count,
                unchanged_rows=exact_existing.row_count,
            )

        predecessor = session.execute(
            select(DataRelease)
            .where(
                DataRelease.partition_key == normalized.partition_key,
                or_(
                    DataRelease.available_at < normalized.available_at,
                    and_(
                        DataRelease.available_at == normalized.available_at,
                        DataRelease.retrieved_at < normalized.retrieved_at,
                    ),
                ),
            )
            .order_by(
                DataRelease.available_at.desc(),
                DataRelease.retrieved_at.desc(),
                DataRelease.id.desc(),
            )
            .limit(1)
        ).scalar_one_or_none()
        # Compare to the incoming vintage's chronological predecessor rather
        # than the globally latest release. This handles both ordinary polls
        # and late-arriving backfills without erasing an A -> B -> A reversion.
        if (
            predecessor is not None
            and _same_release_fingerprint(predecessor, digest, normalized)
            and _predecessor_artifacts_match(session, predecessor.id, artifacts)
        ):
            session.commit()
            return IngestResult(
                release_id=predecessor.id,
                created=False,
                projected=False,
                row_count=predecessor.row_count,
                unchanged_rows=predecessor.row_count,
            )

        release = DataRelease(
            partition_key=normalized.partition_key,
            source_family=normalized.source_family,
            published_at=normalized.published_at,
            available_at=normalized.available_at,
            retrieved_at=normalized.retrieved_at,
            vintage_label=normalized.vintage_label,
            source_url=normalized.source_url,
            content_sha256=digest,
            row_count=len(work),
        )
        session.add(release)
        session.flush()
        _insert_release_artifacts(session, release.id, artifacts)
        release_rows = work.to_dict("records")
        for row in release_rows:
            row["release_id"] = release.id
        session.execute(insert(ReleaseObservation), release_rows)

        latest_id = session.scalar(
            select(DataRelease.id)
            .where(DataRelease.partition_key == normalized.partition_key)
            .order_by(
                DataRelease.available_at.desc(),
                DataRelease.retrieved_at.desc(),
                DataRelease.id.desc(),
            )
            .limit(1)
        )
        projected = bool(replace_current and latest_id == release.id)
        changed = unchanged = removed = 0
        if projected:
            assert normalized.projection is not None
            changed, unchanged, removed = _replace_projection(
                session,
                work,
                normalized.projection,
                normalized.retrieved_at,
            )
        session.commit()
    except Exception:
        session.rollback()
        raise

    return IngestResult(
        release_id=release.id,
        created=True,
        projected=projected,
        row_count=len(work),
        changed_rows=changed,
        unchanged_rows=unchanged,
        removed_rows=removed,
    )


def latest_release(
    session: Session,
    partition_key: str,
    *,
    as_known_at: datetime | None = None,
) -> DataRelease | None:
    stmt = select(DataRelease).where(DataRelease.partition_key == partition_key)
    if as_known_at is not None:
        stmt = stmt.where(DataRelease.available_at <= _utc_naive(as_known_at))
    return session.execute(
        stmt.order_by(
            DataRelease.available_at.desc(),
            DataRelease.retrieved_at.desc(),
            DataRelease.id.desc(),
        ).limit(1)
    ).scalar_one_or_none()


def release_history(session: Session, partition_key: str) -> list[DataRelease]:
    return list(
        session.execute(
            select(DataRelease)
            .where(DataRelease.partition_key == partition_key)
            .order_by(
                DataRelease.available_at,
                DataRelease.retrieved_at,
                DataRelease.id,
            )
        ).scalars()
    )


def load_vintage_panel(
    session: Session,
    as_known_at: datetime,
    *,
    countries: tuple[str, ...] | None = None,
    indicators: tuple[str, ...] | None = None,
    sources: tuple[str, ...] | None = None,
    partition_keys: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Return each partition's complete latest release available at ``as_known_at``.

    Release ranking happens *before* observation filters. This is essential: a
    row omitted from the newest release must not fall back to an older release.
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

    stmt = (
        select(
            ReleaseObservation.release_id,
            DataRelease.partition_key,
            DataRelease.source_family,
            DataRelease.published_at,
            DataRelease.available_at,
            DataRelease.retrieved_at,
            DataRelease.vintage_label,
            ReleaseObservation.country,
            ReleaseObservation.indicator,
            ReleaseObservation.date,
            ReleaseObservation.value,
            ReleaseObservation.source,
            ReleaseObservation.series_id,
            ReleaseObservation.status,
        )
        .select_from(ReleaseObservation)
        .join(ranked, ranked.c.release_id == ReleaseObservation.release_id)
        .join(DataRelease, DataRelease.id == ReleaseObservation.release_id)
        .where(ranked.c.release_rank == 1)
    )
    if countries:
        stmt = stmt.where(ReleaseObservation.country.in_(countries))
    if indicators:
        stmt = stmt.where(ReleaseObservation.indicator.in_(indicators))
    if sources:
        stmt = stmt.where(ReleaseObservation.source.in_(sources))
    stmt = stmt.order_by(
        ReleaseObservation.country,
        ReleaseObservation.indicator,
        ReleaseObservation.date,
        ReleaseObservation.source,
    )
    rows = session.execute(stmt).mappings().all()
    if not rows:
        return pd.DataFrame(columns=_OUTPUT_COLUMNS)
    return pd.DataFrame(rows, columns=_OUTPUT_COLUMNS)


def make_partition_key(
    source_family: str,
    series_id: str,
    country: str,
    indicator: str,
) -> str:
    """Build the default stable partition key for one country/native series."""
    parts = (source_family, series_id, country, indicator)
    return "series:" + ":".join(quote(str(part), safe="") for part in parts)


def bootstrap_current_observations(
    session: Session,
    *,
    available_at: datetime,
    retrieved_at: datetime | None = None,
) -> tuple[int, int]:
    """Conservatively seed the release ledger from the legacy current table.

    Existing rows did not retain publication vintages, so the caller must choose
    an explicit ``available_at``. The function does not backdate from observation
    periods or mutate ``observations``. It is safe to run repeatedly.

    Returns ``(created_releases, covered_observation_rows)``.
    """
    rows = (
        session.execute(
            select(
                Observation.country,
                Observation.indicator,
                Observation.date,
                Observation.value,
                Observation.source,
                Observation.series_id,
            ).order_by(
                Observation.source,
                Observation.series_id,
                Observation.country,
                Observation.indicator,
                Observation.date,
            )
        )
        .mappings()
        .all()
    )
    if not rows:
        return 0, 0

    frame = pd.DataFrame(rows, columns=_CORE_COLUMNS)
    frame["source_family"] = frame["source"].str.removesuffix("_FCST")
    retrieved = retrieved_at or datetime.now(UTC)
    created = 0
    group_columns = ["source_family", "series_id", "country", "indicator"]
    for (family, series_id, country, indicator), group in frame.groupby(
        group_columns,
        sort=True,
        dropna=False,
    ):
        snapshot = group.loc[:, _CORE_COLUMNS]
        result = ingest_release_snapshot(
            session,
            snapshot,
            ReleaseMeta(
                partition_key=make_partition_key(family, series_id, country, indicator),
                source_family=family,
                available_at=available_at,
                retrieved_at=retrieved,
                vintage_label="legacy-current-table bootstrap",
            ),
            replace_current=False,
        )
        created += int(result.created)
    return created, len(frame)
