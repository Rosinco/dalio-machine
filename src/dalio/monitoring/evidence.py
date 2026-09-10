"""Read-only Swedish monitoring evidence at an exact known-at instant.

Only the newest eligible complete supplement supplies scalar calculation
facts. A failed attempt in that batch remains a gap. Earlier unbound scalar
releases are integrity-checked inventory, never retroactively raw-verified by
a mutable cache. Original national debt documents retain their separate scope.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from dalio.assessments.debt_evidence import load_national_debt_context
from dalio.data_sources.riksbank import RIKSBANK_SERIES, RIKSBANK_SOURCE
from dalio.monitoring.acquisition import INDICATORS, canonical, digest, load_bundle, utc
from dalio.storage.db import DataRelease, DataReleaseArtifact, ReleaseObservation
from dalio.storage.releases import (
    _canonicalize,
    _content_hash,
    _stored_release_artifacts,
    latest_release,
    make_partition_key,
)


def _stored_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    return (value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)).isoformat()


def _tables(engine: Engine) -> set[str]:
    database = engine.url.database
    if (engine.dialect.name == "sqlite" and database and database != ":memory:"
            and not database.startswith("file:") and not Path(database).exists()):
        return set()
    return set(inspect(engine).get_table_names())


def _legacy_inventory(engine: Engine, cutoff: datetime) -> tuple[list, set]:
    """The two existing SWEA partitions have scalar provenance, not raw proof."""
    tables = _tables(engine)
    inventory, protected = [], set()
    if not {DataRelease.__tablename__, ReleaseObservation.__tablename__} <= tables:
        return inventory, protected
    with Session(engine, autoflush=False) as session:
        for spec in (s for s in RIKSBANK_SERIES if s.indicator in INDICATORS[3:]):
            key = make_partition_key(RIKSBANK_SOURCE, spec.series_id, "SE", spec.indicator)
            release = latest_release(session, key, as_known_at=cutoff)
            if release is None:
                continue
            fields = ("country", "indicator", "date", "value", "source", "series_id", "status")
            rows = session.execute(select(*(getattr(ReleaseObservation, field) for field in fields))
                .where(ReleaseObservation.release_id == release.id)).mappings().all()
            frame = _canonicalize(pd.DataFrame([dict(row) for row in rows], columns=fields))
            if len(frame) != release.row_count or _content_hash(frame) != release.content_sha256:
                raise ValueError("Legacy monitoring release full-row hash/count mismatch")
            for field, expected in {"country": "SE", "indicator": spec.indicator,
                                    "source": RIKSBANK_SOURCE, "series_id": spec.series_id}.items():
                if frame[field].unique().tolist() != [expected]:
                    raise ValueError("Legacy monitoring row differs from exact source partition")
            if release.source_family != RIKSBANK_SOURCE:
                raise ValueError("Legacy monitoring release source family mismatch")
            artifacts = (_stored_release_artifacts(session, release.id)
                         if DataReleaseArtifact.__tablename__ in tables else {})
            protected.update(artifact.artifact_path for artifact in artifacts.values())
            inventory.append(dict(indicator=spec.indicator, source=RIKSBANK_SOURCE,
                series_id=spec.series_id, release_id=int(release.id), partition_key=key,
                available_at=_stored_time(release.available_at), retrieved_at=_stored_time(release.retrieved_at),
                published_at=_stored_time(release.published_at), source_url=release.source_url,
                row_count=len(frame), first_period=frame.date.min().isoformat(),
                last_period=frame.date.max().isoformat(), content_sha256=release.content_sha256,
                proof=("scalar_rows_verified_unrecognized_raw_contract" if artifacts
                       else "scalar_rows_verified_raw_response_not_bound"),
                artifacts=[dict(role=a.role, sha256=a.artifact_sha256, path=a.artifact_path)
                           for a in artifacts.values()]))
    return inventory, protected


def _select_bundle(paths, cutoff: datetime) -> Path | None:
    if isinstance(paths, str | Path):
        raise ValueError("bundle_paths must be a sequence of immutable bundle paths")
    candidates, originals = [], set()
    for value in paths:
        path = Path(value).expanduser()
        if path.is_symlink():
            raise ValueError("Monitoring bundle symlink is not permitted")
        originals.add(path.resolve())
    for path in originals:
        # Read just the envelope clock before choosing a release. Future raw
        # corruption must not invalidate a report for an earlier known-at time.
        header = json.loads(path.read_bytes())
        available = utc(header["available_at"])
        if available <= cutoff:
            candidates.append((available, path))
    if not candidates:
        return None
    newest = max(clock for clock, path in candidates)
    latest = sorted(path for clock, path in candidates if clock == newest)
    if len({path.read_bytes() for path in latest}) != 1:
        raise ValueError("Distinct monitoring bundles share the latest completion clock")
    return latest[0]


def load_evidence(engine: Engine, *, as_known_at: datetime, bundle_paths=()) -> dict:
    """Return JSON-safe verified facts, gaps and explicit legacy inventory.

    The digest hashes canonical JSON of every other top-level field (sorted
    UTF-8 keys, compact separators, no NaN). No current observation projection,
    DDL, database mutation, source acquisition or artifact writing occurs here.
    ``bundle_paths`` is the caller's explicit local immutable capture history.
    """
    if not isinstance(as_known_at, datetime):
        raise ValueError("as_known_at must be a timezone-aware datetime")
    cutoff = utc(as_known_at)
    result = dict(country="SE", as_known_at=cutoff.isoformat(), series=[], gaps=[],
                  national_debt_context=[], legacy_inventory=[])
    inventory, protected = _legacy_inventory(engine, cutoff)
    result["legacy_inventory"] = inventory
    selected = _select_bundle(bundle_paths, cutoff)
    if selected is None:
        result["gaps"] = [dict(indicator=indicator, reason="no_bundle_as_known_at")
                          for indicator in INDICATORS]
    else:
        bundle = load_bundle(selected)
        result["series"], result["gaps"] = bundle["series"], bundle["gaps"]
        protected.update(bundle["protected_artifact_paths"])
    native = load_national_debt_context(engine, as_known_at=cutoff, countries=["SE"])
    result["national_debt_context"] = native["contexts"]["SE"]
    protected.update(native["protected_artifact_paths"])
    result["protected_artifact_paths"] = sorted(protected)
    result["evidence_digest"] = digest(canonical(result))
    return result
