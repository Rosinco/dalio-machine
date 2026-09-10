"""Atomic refinancing releases and read-only verification against retained evidence.

Readiness means complete source-contract and provenance checks for this fixed
package. It does not imply a sovereign risk score or current market conditions.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from dalio.data_sources.ecb_refinancing import (
    ECB_REFINANCING_SERIES,
    SOURCE_ECB_GFS,
    EcbRefinancingSeries,
    ecb_refinancing_catalogue_sha256,
    parse_ecb_refinancing_csv,
)
from dalio.data_sources.eurostat_refinancing import (
    EUROSTAT_REFINANCING_SERIES,
    EUROSTAT_REFINANCING_SOURCE,
    EurostatRefinancingSeries,
    build_eurostat_refinancing_url,
    parse_eurostat_refinancing_json,
)
from dalio.data_sources.sovereign_refinancing import (
    SOVEREIGN_REFINANCING_VINTAGE_PREFIX,
    SovereignRefinancingPartition,
    load_checked_sovereign_refinancing_manifest,
)
from dalio.storage.db import (
    DataRelease,
    DataReleaseArtifact,
    Observation,
    ReleaseObservation,
    init_db,
)
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseArtifactMeta,
    ReleaseMeta,
    _canonicalize,
    _content_hash,
    ingest_release_snapshot,
    latest_release,
    make_partition_key,
)

COLUMNS = ["country", "indicator", "date", "value", "source", "series_id", "status"]
ARTIFACT_FIELDS = {
    "source_response": ("source_artifact_path", "source_artifact_sha256"),
    "native_payload": ("native_payload_artifact_path", "native_payload_sha256"),
    "missingness_ledger": ("missing_provenance_artifact_path", "missing_provenance_sha256"),
}


def _json_bytes(payload: object) -> bytes:
    def encode(value: object) -> str:
        if isinstance(value, date | datetime):
            return value.isoformat()
        raise TypeError(f"Unsupported catalogue value {type(value).__name__}")

    return json.dumps(
        payload,
        default=encode,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


@dataclass(frozen=True)
class RefinancingBinding:
    partition: SovereignRefinancingPartition
    spec: EurostatRefinancingSeries | EcbRefinancingSeries

    @property
    def series_id(self) -> str:
        return (
            self.spec.series_id
            if isinstance(self.spec, EurostatRefinancingSeries)
            else self.spec.native_series_id
        )

    @property
    def source(self) -> str:
        return (
            EUROSTAT_REFINANCING_SOURCE
            if isinstance(self.spec, EurostatRefinancingSeries)
            else SOURCE_ECB_GFS
        )

    @property
    def partition_key(self) -> str:
        return make_partition_key(
            self.source, self.series_id, self.spec.country, self.spec.indicator
        )


@dataclass(frozen=True)
class PreparedRefinancingPartition:
    binding: RefinancingBinding
    frame: pd.DataFrame
    meta: ReleaseMeta


def refinancing_bindings() -> tuple[RefinancingBinding, ...]:
    """Require an exact bijection between 31 pinned adapters and the manifest."""
    manifest = load_checked_sovereign_refinancing_manifest()
    by_identity = {}
    for spec in (*EUROSTAT_REFINANCING_SERIES, *ECB_REFINANCING_SERIES):
        if isinstance(spec, EurostatRefinancingSeries):
            native_id = "|".join([spec.dataset, *(code for _, code in spec.dimension_codes)])
            url = build_eurostat_refinancing_url(spec)
            entity = spec.country
        else:
            native_id, url = spec.native_series_id, spec.url
            # The denominator spells out fixed composition; the compatible
            # eight-character country field uses the pinned adapter code EA21.
            entity = "EA21_FIXED" if spec.country == "EA21" else spec.country
        key = (entity, native_id, url)
        if key in by_identity:
            raise ValueError("Duplicate refinancing adapter identity")
        by_identity[key] = spec
    bindings = []
    for partition in manifest.partitions:
        if partition.phase != "harmonized_scalar":
            continue
        key = (partition.entity, partition.native_identity, partition.source_url)
        if key not in by_identity:
            raise ValueError(f"No exact adapter for {partition.partition_id}")
        bindings.append(RefinancingBinding(partition, by_identity.pop(key)))
    if by_identity or len(bindings) != 31:
        raise ValueError("Refinancing requires exactly all 31 harmonized partitions")
    return tuple(bindings)


def refinancing_catalogue_bytes() -> bytes:
    """Bind scope, full source contracts, native units and verified history floors."""
    return _json_bytes(
        {
            "schema_version": "sovereign-refinancing-ingest-v1",
            "manifest": asdict(load_checked_sovereign_refinancing_manifest()),
            "bindings": [asdict(binding) for binding in refinancing_bindings()],
            "ecb_catalogue_sha256": ecb_refinancing_catalogue_sha256(),
            "availability_policy": "retrieval_time; source update is not historical availability",
            "status_policy": "release status lowercased; native status retained in source evidence",
            "missing_value_policy": "publisher missing cells retained, never imputed or zero-filled",
        }
    )


def write_refinancing_catalogue(artifact_dir: Path) -> Path:
    body = refinancing_catalogue_bytes()
    digest = _sha256(body)
    path = artifact_dir.resolve() / digest[:2] / f"{digest}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(body)
    except FileExistsError:
        pass
    if path.read_bytes() != body:
        raise ValueError(f"Refinancing catalogue hash mismatch: {path}")
    return path


def _same_frame(
    actual: pd.DataFrame, expected: pd.DataFrame, *, label: str, with_status: bool = True
) -> None:
    columns = COLUMNS if with_status else COLUMNS[:-1]
    try:
        left = actual.loc[:, columns].copy()
        right = expected.loc[:, columns].copy()
        for frame in (left, right):
            frame.attrs = {}
            frame["date"] = pd.to_datetime(frame["date"]).dt.date
            if with_status:
                frame["status"] = frame["status"].str.lower()
        pd.testing.assert_frame_equal(
            left.sort_values("date").reset_index(drop=True),
            right.sort_values("date").reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} differs from retained source observations") from exc


def _reparse(binding: RefinancingBinding, body: bytes) -> pd.DataFrame:
    if isinstance(binding.spec, EurostatRefinancingSeries):
        return parse_eurostat_refinancing_json(body, binding.spec)
    return parse_ecb_refinancing_csv(body.decode("utf-8-sig"), binding.spec)


def validate_partition(
    binding: RefinancingBinding,
    frame: pd.DataFrame,
    *,
    retrieved_at: datetime,
    catalogue_path: Path,
) -> PreparedRefinancingPartition:
    """Reparse exact source bytes and independently bind all four artifact roles."""
    run_at = _utc(retrieved_at)
    if frame.attrs.get("source_url") != binding.partition.source_url:
        raise ValueError(f"Source URL mismatch for {binding.partition.partition_id}")
    bodies, paths = {}, {}
    for role, (path_field, hash_field) in ARTIFACT_FIELDS.items():
        if not frame.attrs.get(path_field) or not frame.attrs.get(hash_field):
            raise ValueError(f"Missing {role} evidence for {binding.series_id}")
        path = Path(frame.attrs[path_field]).resolve()
        try:
            body = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"Unreadable {role} evidence: {path}") from exc
        if _sha256(body) != frame.attrs[hash_field]:
            raise ValueError(f"{role} hash mismatch for {binding.series_id}")
        bodies[role], paths[role] = body, path
    parsed = _reparse(binding, bodies["source_response"])
    _same_frame(frame, parsed, label="Prepared frame")
    for field in ("native_payload_sha256", "missing_provenance_sha256", "missing_provenance_json"):
        if frame.attrs.get(field) != parsed.attrs[field]:
            raise ValueError(f"Reparsed {field} mismatch for {binding.series_id}")
    if bodies["missingness_ledger"] != parsed.attrs["missing_provenance_json"].encode("utf-8"):
        raise ValueError(f"Non-canonical missingness ledger for {binding.series_id}")
    source_updated = parsed.attrs.get("source_updated_at")
    if frame.attrs.get("source_updated_at") != source_updated:
        raise ValueError(f"Source update clock mismatch for {binding.series_id}")
    if source_updated is not None and _utc(source_updated) > run_at:
        raise ValueError(f"Future source update clock for {binding.series_id}")
    periods = list(parsed["date"])
    for record in parsed.attrs["missing_period_records"]:
        native = record["native_period"]
        periods.append(date.fromisoformat(native + ("-01-01" if len(native) == 4 else "-01")))
    if max(periods) > run_at.date():
        raise ValueError(f"Future reference period for {binding.series_id}")
    try:
        catalogue = catalogue_path.read_bytes()
    except OSError as exc:
        raise ValueError("Unreadable refinancing catalogue evidence") from exc
    if catalogue != refinancing_catalogue_bytes():
        raise ValueError("Refinancing catalogue hash/contract mismatch")
    bodies["catalogue_manifest"] = catalogue
    paths["catalogue_manifest"] = catalogue_path.resolve()
    artifacts = tuple(
        ReleaseArtifactMeta(
            role=role,
            artifact_sha256=_sha256(body),
            artifact_path=str(paths[role]),
            native_payload_sha256=parsed.attrs["native_payload_sha256"],
            missing_provenance_sha256=parsed.attrs["missing_provenance_sha256"],
            provenance_json=parsed.attrs["missing_provenance_json"],
        )
        for role, body in bodies.items()
    )
    # Use the freshly reparsed frame, retaining only verified fetch metadata.
    parsed.attrs.update(
        {field: frame.attrs[field] for fields in ARTIFACT_FIELDS.values() for field in fields}
    )
    parsed.attrs["source_url"] = binding.partition.source_url
    meta = ReleaseMeta(
        partition_key=binding.partition_key,
        source_family=binding.source,
        published_at=source_updated,
        available_at=run_at,
        retrieved_at=run_at,
        source_url=binding.partition.source_url,
        vintage_label=SOVEREIGN_REFINANCING_VINTAGE_PREFIX + _sha256(catalogue),
        projection=ProjectionScope(binding.spec.country, binding.spec.indicator, (binding.source,)),
        artifacts=artifacts,
    )
    return PreparedRefinancingPartition(binding, parsed, meta)


def _checked_batch(
    batch: tuple[PreparedRefinancingPartition, ...],
) -> tuple[PreparedRefinancingPartition, ...]:
    expected = refinancing_bindings()
    if tuple(item.binding for item in batch) != expected:
        raise ValueError("Batch must contain exactly all 31 pinned partitions in manifest order")
    if len({_utc(item.meta.retrieved_at) for item in batch}) != 1:
        raise ValueError("Refinancing batch must share one conservative retrieval clock")
    checked = []
    for item in batch:
        catalogues = [a for a in item.meta.artifacts if a.role == "catalogue_manifest"]
        if len(catalogues) != 1:
            raise ValueError("Missing or duplicate catalogue artifact")
        verified = validate_partition(
            item.binding,
            item.frame,
            retrieved_at=item.meta.retrieved_at,
            catalogue_path=Path(catalogues[0].artifact_path),
        )
        if item.meta != verified.meta:
            raise ValueError(f"Release metadata mismatch for {item.binding.series_id}")
        checked.append(verified)
    return tuple(checked)


def ingest_refinancing_batch(
    batch: tuple[PreparedRefinancingPartition, ...], *, engine: Engine
) -> dict[str, Any]:
    """Preflight before schema initialization; publish all 31 partitions or none."""
    checked = _checked_batch(batch)
    init_db(engine)
    results = []
    with (
        engine.begin() as connection,
        Session(
            bind=connection,
            expire_on_commit=False,
            join_transaction_mode="rollback_only",
        ) as session,
    ):
        # Validate every replacement against both the immutable ledger and the
        # compatible current projection before any partition is changed.
        for item in checked:
            previous = latest_release(session, item.meta.partition_key)
            if previous and _utc(previous.available_at) > item.meta.available_at:
                raise ValueError(f"Retrograde batch for {item.binding.series_id}")
            old_dates = set(
                session.scalars(
                    select(Observation.date).where(
                        Observation.country == item.binding.spec.country,
                        Observation.indicator == item.binding.spec.indicator,
                        Observation.source == item.binding.source,
                    )
                )
            )
            if previous:
                old_dates.update(
                    session.scalars(
                        select(ReleaseObservation.date).where(
                            ReleaseObservation.release_id == previous.id,
                        )
                    )
                )
            omitted = old_dates - set(item.frame["date"])
            if omitted:
                raise ValueError(
                    f"History contraction for {item.binding.series_id}: "
                    f"{len(omitted)} formerly finite periods now absent; "
                    "review publisher removal before changing the contract"
                )
        for item in checked:
            result = ingest_release_snapshot(session, item.frame, item.meta)
            results.append({"partition_id": item.binding.partition.partition_id, **asdict(result)})
    return {
        "created_releases": sum(row["created"] for row in results),
        "observation_count": sum(row["row_count"] for row in results),
        "harmonized_partitions": 31,
        "expected_partitions": 48,
        "national_native_planned": 17,
        "retrieved_at": checked[0].meta.retrieved_at.isoformat(),
        "catalogue_sha256": _sha256(refinancing_catalogue_bytes()),
        "partitions": results,
    }


def load_stored_refinancing_batch(
    engine: Engine, *, as_of: date | None = None
) -> tuple[PreparedRefinancingPartition, ...]:
    """Reconstruct an audited batch offline for staging-to-live promotion.

    All source bytes and original source-update clocks are retained. For mixed
    deduplicated vintages, use the latest retrieval time: the whole batch was
    demonstrably available by that time, never before it.
    """
    audit = audit_refinancing(engine, as_of=as_of)
    if audit["ready_partitions"] != 31:
        raise ValueError("Promotion requires all 31 harmonized partitions verified")
    rows = {row["partition_id"]: row for row in audit["partitions"] if row["ready"]}
    run_at = max(_utc(datetime.fromisoformat(row["retrieved_at"])) for row in rows.values())
    batch = []
    for binding in refinancing_bindings():
        row = rows[binding.partition.partition_id]
        artifacts = {artifact["role"]: artifact for artifact in row["artifacts"]}
        frame = _reparse(binding, Path(artifacts["source_response"]["artifact_path"]).read_bytes())
        for role, (path_field, hash_field) in ARTIFACT_FIELDS.items():
            frame.attrs[path_field] = artifacts[role]["artifact_path"]
            frame.attrs[hash_field] = artifacts[role]["artifact_sha256"]
        frame.attrs["source_url"] = binding.partition.source_url
        batch.append(
            validate_partition(
                binding,
                frame,
                retrieved_at=run_at,
                catalogue_path=Path(artifacts["catalogue_manifest"]["artifact_path"]),
            )
        )
    return _checked_batch(tuple(batch))


def _audit_partition(session: Session, binding: RefinancingBinding, as_of: date) -> dict[str, Any]:
    release = latest_release(session, binding.partition_key)
    if release is None:
        return {"stored": False, "ready": False, "issues": ["release missing"]}
    row = {
        "stored": True,
        "ready": False,
        "release_id": release.id,
        "retrieved_at": release.retrieved_at.isoformat(),
        "published_at": release.published_at.isoformat() if release.published_at else None,
        "issues": [],
    }
    try:
        if _utc(release.available_at) > datetime.combine(as_of, time.max, UTC):
            raise ValueError("Release availability is after audit as-of date")
        records = session.scalars(
            select(DataReleaseArtifact).where(
                DataReleaseArtifact.release_id == release.id,
            )
        ).all()
        by_role = {record.role: record for record in records}
        if len(records) != 4 or set(by_role) != {*ARTIFACT_FIELDS, "catalogue_manifest"}:
            raise ValueError("Release requires exactly four evidence artifact roles")
        source = by_role["source_response"]
        source_bytes = Path(source.artifact_path).read_bytes()
        if _sha256(source_bytes) != source.artifact_sha256:
            raise ValueError("source_response hash mismatch")
        parsed = _reparse(binding, source_bytes)
        for role, (path_field, hash_field) in ARTIFACT_FIELDS.items():
            parsed.attrs[path_field] = by_role[role].artifact_path
            parsed.attrs[hash_field] = by_role[role].artifact_sha256
        parsed.attrs["source_url"] = release.source_url
        verified = validate_partition(
            binding,
            parsed,
            retrieved_at=_utc(release.retrieved_at),
            catalogue_path=Path(by_role["catalogue_manifest"].artifact_path),
        )
        for expected in verified.meta.artifacts:
            actual = by_role[expected.role]
            if any(getattr(actual, key) != value for key, value in asdict(expected).items()):
                raise ValueError(f"{expected.role} evidence manifest mismatch")
        if (
            release.source_family != verified.meta.source_family
            or release.vintage_label != verified.meta.vintage_label
            or _utc(release.available_at) != verified.meta.available_at
            or (None if release.published_at is None else _utc(release.published_at))
            != verified.meta.published_at
            or release.row_count != len(parsed)
            or release.content_sha256 != _content_hash(_canonicalize(parsed))
        ):
            raise ValueError("Release metadata/content hash differs from source evidence")
        ledger = pd.DataFrame(
            session.execute(
                select(*(getattr(ReleaseObservation, field) for field in COLUMNS)).where(
                    ReleaseObservation.release_id == release.id
                )
            ).all(),
            columns=COLUMNS,
        )
        _same_frame(ledger, parsed, label="Immutable ledger")
        current = pd.DataFrame(
            session.execute(
                select(*(getattr(Observation, field) for field in COLUMNS[:-1])).where(
                    Observation.country == binding.spec.country,
                    Observation.indicator == binding.spec.indicator,
                    Observation.source == binding.source,
                )
            ).all(),
            columns=COLUMNS[:-1],
        )
        _same_frame(current, parsed, label="Current projection", with_status=False)
        latest = parsed.sort_values("date").iloc[-1]
        row.update(
            ready=True,
            observation_count=len(parsed),
            first_date=min(parsed["date"]).isoformat(),
            latest_date=latest["date"].isoformat(),
            latest_value=float(latest["value"]),
            latest_status=latest["status"],
            missing_periods=len(parsed.attrs["missing_period_records"]),
            artifacts=[asdict(artifact) for artifact in verified.meta.artifacts],
        )
    except (ValueError, OSError, TypeError, KeyError) as exc:
        row["issues"].append(str(exc))
    return row


def audit_refinancing(engine: Engine, *, as_of: date | None = None) -> dict[str, Any]:
    """Read without DDL, re-hashing and re-parsing latest evidence for all 31 series."""
    manifest = load_checked_sovereign_refinancing_manifest()
    bindings = {item.partition.partition_id: item for item in refinancing_bindings()}
    inspector = inspect(engine)
    present = set(inspector.get_table_names())
    models = (Observation, DataRelease, ReleaseObservation, DataReleaseArtifact)
    schema_ready = all(
        model.__tablename__ in present
        and set(model.__table__.columns.keys())
        <= {column["name"] for column in inspector.get_columns(model.__tablename__)}
        for model in models
    )
    audit_on = as_of or datetime.now(UTC).date()
    rows = []
    with Session(engine) as session:
        for partition in manifest.partitions:
            row = {
                **asdict(partition),
                "ready": False,
                "stored": False,
                "observation_count": 0,
                "missing_periods": 0,
                "issues": [],
            }
            binding = bindings.get(partition.partition_id)
            if binding:
                row.update(
                    indicator=binding.spec.indicator,
                    series_id=binding.series_id,
                    country=binding.spec.country,
                    unit=binding.spec.unit,
                    source=binding.source,
                )
                if schema_ready:
                    row.update(_audit_partition(session, binding, audit_on))
                else:
                    row["issues"] = ["required database tables/columns absent"]
            else:
                row["issues"] = ["national-native collection planned"]
            rows.append(row)
    return {
        "schema_version": 1,
        "as_of": audit_on.isoformat(),
        "manifest_sha256": manifest.semantic_sha256,
        "catalogue_sha256": _sha256(refinancing_catalogue_bytes()),
        "expected_partitions": 48,
        "harmonized_expected": 31,
        "national_native_planned": 17,
        "stored_partitions": sum(row["stored"] for row in rows),
        "ready_partitions": sum(row["ready"] for row in rows),
        "observation_count": sum(row["observation_count"] for row in rows if row["ready"]),
        "readiness_basis": "retained source contracts and projection integrity; not a risk or freshness score",
        "partitions": rows,
    }
