"""Immutable, source-verified debt-office tables with their native dimensions."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

from sqlalchemy import Engine, func, insert, inspect, select
from sqlalchemy.orm import Session

from dalio.storage.db import DataRelease, DataReleaseArtifact, NationalDebtFact, init_db
from dalio.storage.releases import (
    ReleaseArtifactMeta,
    _insert_release_artifacts,
    _normalize_release_artifacts,
    _stored_release_artifacts,
    latest_release,
)

SOURCE = "RIKSGALDEN_DEBT"
PREFIX = "national-debt:"
STREAMS = {"se_central_government_debt_monthly_report", "se_central_government_funding_plan"}
FACT_FIELDS = (
    "fact_type",
    "metric",
    "value",
    "unit",
    "period_start",
    "period_end",
    "status",
    "dimensions",
    "source_locator",
    "native_label",
    "native_value",
)
STORED_FIELDS = (
    "fact_key",
    "country",
    *(field if field != "dimensions" else "dimensions_json" for field in FACT_FIELDS),
)
ROLES = {"source_response", "native_payload", "missingness_ledger", "catalogue_manifest"}


def _bytes(value: object) -> bytes:
    def encode(item):
        if isinstance(item, date | datetime):
            return item.isoformat()
        raise TypeError(f"Unsupported native evidence type {type(item).__name__}")

    return json.dumps(
        value,
        default=encode,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _reparse_document(document):
    from dalio.data_sources.riksgalden_debt import reparse_document

    return reparse_document(document)


def _construct_document(fields):
    from dalio.data_sources.riksgalden_debt import NationalDebtDocument

    return NationalDebtDocument(**fields)


def _key(document) -> str:
    return PREFIX + document.stream_id + ":" + quote(document.snapshot_key, safe="")


def _descriptor(document) -> dict:
    return {
        "schema_version": "national-debt-evidence-v1",
        "stream_id": document.stream_id,
        "snapshot_key": document.snapshot_key,
        "country": document.country,
        "source_url": document.source_url,
        "published_at": _utc(document.published_at).isoformat(),
        "reference_date": document.reference_date.isoformat(),
        "source_sha256": _sha(document.source_bytes),
        "metadata": document.metadata,
    }


def _rows(document) -> list[dict]:
    rows = []
    seen = set()
    for fact in document.facts:
        if set(fact) != set(FACT_FIELDS):
            raise ValueError("Native debt fact has unexpected or missing fields")
        for field in ("fact_type", "metric", "unit", "source_locator", "native_label"):
            if not isinstance(fact[field], str) or not fact[field].strip():
                raise ValueError(f"Native debt {field} must be a nonempty string")
        if not isinstance(fact["native_value"], str) or not isinstance(fact["dimensions"], dict):
            raise ValueError("Native debt lexeme/dimensions malformed")
        if fact["status"] not in {"observed", "forecast", "not_reported"}:
            raise ValueError("Native debt status must distinguish outcomes, forecasts and missing")
        value = fact["value"]
        if (value is None) != (fact["status"] == "not_reported"):
            raise ValueError("Native missingness must remain explicit")
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError("Native debt numeric value must be finite")
        if any(type(fact[field]) is not date for field in ("period_start", "period_end")):
            raise ValueError("Native debt periods must be dates")
        if fact["period_start"] > fact["period_end"]:
            raise ValueError("Native debt period starts after it ends")
        identity = {
            field: fact[field]
            for field in FACT_FIELDS
            if field not in {"value", "native_value", "status"}
        }
        key = _sha(_bytes(identity))
        if key in seen:
            raise ValueError("Duplicate native debt cell identity")
        seen.add(key)
        row = {"fact_key": key, "country": document.country, **fact}
        row["value"] = float(value) if value is not None else None
        row["dimensions_json"] = _bytes(row.pop("dimensions")).decode()
        rows.append(row)
    if not rows:
        raise ValueError("Native debt document has no numeric/missing facts")
    return sorted(rows, key=lambda row: row["fact_key"])


def _components(document, retrieved_at: datetime) -> tuple[Any, dict[str, bytes]]:
    if document.country != "SE" or document.stream_id not in STREAMS:
        raise ValueError("Unsupported national debt source contract")
    if not document.snapshot_key or len(_key(document)) > 256:
        raise ValueError("Invalid national debt snapshot identity")
    parsed_url = urlparse(document.source_url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.hostname != "www.riksgalden.se"
        or parsed_url.username
        or parsed_url.password
        or parsed_url.fragment
    ):
        raise ValueError("National debt source must be the original official publisher")
    run_at = _utc(retrieved_at)
    if _utc(document.published_at) > run_at or document.reference_date > run_at.date():
        raise ValueError("National debt source clock is after retrieval")
    if document.reference_date > _utc(document.published_at).date():
        raise ValueError("National debt reference date is after publication")
    if not isinstance(document.source_bytes, bytes) or not document.source_bytes:
        raise ValueError("National debt source bytes are absent")
    reparsed = _reparse_document(document)
    if _bytes(_descriptor(reparsed)) != _bytes(_descriptor(document)) or _bytes(
        reparsed.facts
    ) != _bytes(document.facts):
        raise ValueError("Native debt document differs from reparsed official source")
    rows = _rows(reparsed)
    if any(
        row["status"] == "observed" and row["period_end"] > _utc(document.published_at).date()
        for row in rows
    ):
        raise ValueError("Observed national debt fact has a future period")
    native = {"schema_version": "national-debt-facts-v1", "facts": list(reparsed.facts)}
    missing = {
        "schema_version": "national-debt-missingness-v1",
        "policy": "source missing cells remain missing; forecasts remain separate from outcomes",
        "records": [fact for fact in reparsed.facts if fact["value"] is None],
    }
    return reparsed, {
        "source_response": reparsed.source_bytes,
        "native_payload": _bytes(native),
        "missingness_ledger": _bytes(missing),
        "catalogue_manifest": _bytes(_descriptor(reparsed)),
    }


@dataclass(frozen=True)
class PreparedNativePartition:
    document: Any
    retrieved_at: datetime
    artifacts: tuple[ReleaseArtifactMeta, ...]


def _artifact_metas(
    components: dict[str, bytes], paths: dict[str, Path]
) -> tuple[ReleaseArtifactMeta, ...]:
    return tuple(
        ReleaseArtifactMeta(
            role=role,
            artifact_sha256=_sha(components[role]),
            artifact_path=str(paths[role].resolve()),
            native_payload_sha256=_sha(components["native_payload"]),
            missing_provenance_sha256=_sha(components["missingness_ledger"]),
            provenance_json=components["missingness_ledger"].decode(),
        )
        for role in sorted(ROLES)
    )


def prepare_native_batch(
    documents, *, artifact_root: Path, retrieved_at: datetime | None = None
) -> tuple[PreparedNativePartition, ...]:
    """Validate the whole source batch before retaining files; never open a DB."""
    run_at = _utc(retrieved_at or datetime.now(UTC))
    validated = [_components(doc, run_at) for doc in documents]
    if not validated or len({_key(doc) for doc, _ in validated}) != len(validated):
        raise ValueError("Native batch must contain nonempty, unique snapshot identities")
    batch = []
    for doc, components in validated:
        paths = {}
        for role, body in components.items():
            digest = _sha(body)
            extension = "json"
            if role == "source_response":
                extension = (
                    "pdf"
                    if body.startswith(b"%PDF-")
                    else "xlsx"
                    if body.startswith(b"PK")
                    else "bin"
                )
            path = artifact_root.resolve() / role / digest[:2] / f"{digest}.{extension}"
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with path.open("xb") as handle:
                    handle.write(body)
            except FileExistsError:
                pass
            if path.read_bytes() != body:
                raise ValueError(f"Native {role} artifact hash mismatch")
            paths[role] = path
        batch.append(PreparedNativePartition(doc, run_at, _artifact_metas(components, paths)))
    return tuple(batch)


def _checked(item: PreparedNativePartition) -> PreparedNativePartition:
    doc, components = _components(item.document, item.retrieved_at)
    if len(item.artifacts) != 4 or {a.role for a in item.artifacts} != ROLES:
        raise ValueError("Native debt requires exactly four evidence roles")
    paths = {a.role: Path(a.artifact_path) for a in item.artifacts}
    for role, path in paths.items():
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Missing native debt {role} artifact") from exc
        if body != components[role]:
            raise ValueError(f"Native debt {role} hash mismatch")
    expected = _artifact_metas(components, paths)
    if tuple(sorted(item.artifacts, key=lambda a: a.role)) != expected:
        raise ValueError("Native debt artifact manifest mismatch")
    _normalize_release_artifacts(item.artifacts)
    return PreparedNativePartition(doc, _utc(item.retrieved_at), expected)


def _digest(item: PreparedNativePartition) -> str:
    return _sha(_bytes({a.role: a.artifact_sha256 for a in item.artifacts}))


def _ingest_one(session: Session, item: PreparedNativePartition) -> dict:
    doc = item.document
    key = _key(doc)
    digest = _digest(item)
    prior = latest_release(session, key)
    rows = _rows(doc)
    if prior:
        if _utc(prior.available_at) > item.retrieved_at:
            raise ValueError("Retrograde native debt snapshot")
        # An unchanged incoming digest cannot bless an augmented or damaged
        # existing snapshot: verify metadata, all four roles and every row.
        _restore_release(session, prior, item.retrieved_at.date())
        if prior.content_sha256 == digest:
            return {"release_id": prior.id, "created": False, "row_count": prior.row_count}
        if _utc(prior.retrieved_at) == item.retrieved_at:
            raise ValueError("Conflicting native debt content at an existing release clock")
        previous_keys = set(
            session.scalars(
                select(NationalDebtFact.fact_key).where(NationalDebtFact.release_id == prior.id)
            )
        )
        if previous_keys - {row["fact_key"] for row in rows}:
            raise ValueError(
                "Native snapshot revision removes previously reported cells; review source contract"
            )
    release = DataRelease(
        partition_key=key,
        source_family=SOURCE,
        published_at=_utc(doc.published_at).replace(tzinfo=None),
        available_at=item.retrieved_at.replace(tzinfo=None),
        retrieved_at=item.retrieved_at.replace(tzinfo=None),
        source_url=doc.source_url,
        vintage_label=f"{doc.snapshot_key}:native-debt-v1",
        content_sha256=digest,
        row_count=len(rows),
    )
    session.add(release)
    session.flush()
    session.execute(insert(NationalDebtFact), [dict(release_id=release.id, **row) for row in rows])
    _insert_release_artifacts(session, release.id, _normalize_release_artifacts(item.artifacts))
    return {"release_id": release.id, "created": True, "row_count": len(rows)}


def ingest_native_batch(batch, *, engine: Engine) -> dict:
    """Preflight and publish one complete selection atomically, without projection."""
    checked = tuple(_checked(item) for item in batch)
    if not checked or len({_key(item.document) for item in checked}) != len(checked):
        raise ValueError("Native debt batch must contain unique snapshots")
    init_db(engine)
    with (
        engine.begin() as connection,
        Session(bind=connection, join_transaction_mode="rollback_only") as session,
    ):
        results = [_ingest_one(session, item) for item in checked]
    return {
        "created_releases": sum(row["created"] for row in results),
        "fact_count": sum(row["row_count"] for row in results),
        "snapshots": len(results),
        "releases": results,
    }


def _latest_native_releases(session: Session, as_of: date) -> list[DataRelease]:
    cutoff = datetime.combine(as_of, time.max, UTC)
    keys = session.scalars(
        select(DataRelease.partition_key)
        .where(DataRelease.partition_key.startswith(PREFIX))
        .distinct()
    ).all()
    return [
        release
        for key in sorted(keys)
        if (release := latest_release(session, key, as_known_at=cutoff)) is not None
    ]


def _restore_release(
    session: Session, release: DataRelease, as_of: date
) -> PreparedNativePartition:
    if _utc(release.available_at) > datetime.combine(as_of, time.max, UTC):
        raise ValueError("Native debt availability is after audit cutoff")
    stored = _stored_release_artifacts(session, release.id)
    if set(stored) != ROLES:
        raise ValueError("Native debt snapshot is missing evidence roles")
    descriptor = json.loads(Path(stored["catalogue_manifest"].artifact_path).read_bytes())
    fields = {
        key: descriptor[key]
        for key in ("stream_id", "snapshot_key", "country", "source_url", "metadata")
    }
    fields.update(
        published_at=datetime.fromisoformat(descriptor["published_at"]),
        reference_date=date.fromisoformat(descriptor["reference_date"]),
        source_bytes=Path(stored["source_response"].artifact_path).read_bytes(),
        facts=(),
    )
    document = _reparse_document(_construct_document(fields))
    artifacts = tuple(ReleaseArtifactMeta(**asdict(stored[role])) for role in sorted(stored))
    item = _checked(PreparedNativePartition(document, _utc(release.retrieved_at), artifacts))
    if (
        release.partition_key != _key(document)
        or release.source_family != SOURCE
        or release.source_url != document.source_url
        or release.content_sha256 != _digest(item)
        or _utc(release.available_at) != item.retrieved_at
        or _utc(release.published_at) != _utc(document.published_at)
        or release.vintage_label != f"{document.snapshot_key}:native-debt-v1"
    ):
        raise ValueError("Native debt release metadata/content hash mismatch")
    rows = [
        dict(row)
        for row in session.execute(
            select(*(getattr(NationalDebtFact, field) for field in STORED_FIELDS))
            .where(NationalDebtFact.release_id == release.id)
            .order_by(NationalDebtFact.fact_key)
        ).mappings()
    ]
    expected = _rows(document)
    if release.row_count != len(expected) or _bytes(rows) != _bytes(expected):
        raise ValueError("Native debt rows differ from retained source cells")
    return item


def audit_national_debt(engine: Engine, *, as_of: date | None = None) -> dict:
    """Inspect native releases without creating tables or trusting stored totals."""
    on = as_of or datetime.now(UTC).date()
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    models = (DataRelease, DataReleaseArtifact, NationalDebtFact)
    schema_ready = all(
        model.__tablename__ in present
        and set(model.__table__.columns.keys())
        <= {column["name"] for column in inspector.get_columns(model.__tablename__)}
        for model in models
    )
    rows = []
    historical = 0
    if schema_ready:
        with Session(engine) as session:
            historical = session.scalar(
                select(func.count())
                .select_from(DataRelease)
                .where(DataRelease.partition_key.startswith(PREFIX))
            )
            for release in _latest_native_releases(session, on):
                row = {
                    "release_id": release.id,
                    "partition_key": release.partition_key,
                    "ready": False,
                    "issues": [],
                }
                try:
                    item = _restore_release(session, release, on)
                    doc = item.document
                    row.update(
                        ready=True,
                        stream_id=doc.stream_id,
                        snapshot_key=doc.snapshot_key,
                        country=doc.country,
                        source_url=doc.source_url,
                        reference_date=doc.reference_date.isoformat(),
                        published_at=_utc(doc.published_at).isoformat(),
                        available_at=item.retrieved_at.isoformat(),
                        fact_count=len(doc.facts),
                        observed_facts=sum(f["status"] == "observed" for f in doc.facts),
                        forecast_facts=sum(f["status"] == "forecast" for f in doc.facts),
                        missing_facts=sum(f["status"] == "not_reported" for f in doc.facts),
                        source_sha256=_sha(doc.source_bytes),
                    )
                except (ValueError, TypeError, KeyError, OSError, AttributeError) as exc:
                    row["issues"].append(str(exc))
                rows.append(row)
    streams = []
    for stream_id in sorted(STREAMS):
        subset = [row for row in rows if row.get("stream_id") == stream_id]
        # Match failed releases by their stable partition key as well, so a
        # corrupted snapshot cannot disappear from the readiness denominator.
        all_rows = [
            row for row in rows if row["partition_key"].startswith(PREFIX + stream_id + ":")
        ]
        streams.append(
            {
                "stream_id": stream_id,
                "stored_snapshots": len(all_rows),
                "ready_snapshots": sum(row["ready"] for row in all_rows),
                "ready": bool(all_rows) and all(row["ready"] for row in all_rows),
                "first_reference_date": min(
                    (row["reference_date"] for row in subset), default=None
                ),
                "latest_reference_date": max(
                    (row["reference_date"] for row in subset), default=None
                ),
            }
        )
    return {
        "schema_version": 1,
        "as_of": on.isoformat(),
        "table_present": schema_ready,
        "historical_releases": historical,
        "stored_snapshots": len(rows),
        "ready_snapshots": sum(row["ready"] for row in rows),
        "fact_count": sum(row.get("fact_count", 0) for row in rows if row["ready"]),
        "streams": streams,
        "snapshots": rows,
    }


def load_native_batch(
    engine: Engine, *, as_of: date | None = None
) -> tuple[PreparedNativePartition, ...]:
    """Reconstruct all stored native snapshots from verified evidence, offline."""
    on = as_of or datetime.now(UTC).date()
    audit = audit_national_debt(engine, as_of=on)
    if not audit["stored_snapshots"] or audit["ready_snapshots"] != audit["stored_snapshots"]:
        raise ValueError("Native promotion requires every stored snapshot verified")
    with Session(engine) as session:
        batch = [
            _restore_release(session, release, on)
            for release in _latest_native_releases(session, on)
        ]
    return tuple(batch)
