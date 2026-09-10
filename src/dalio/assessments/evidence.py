"""Read-only, point-in-time country evidence for assessment logic.

``load_evidence(engine, as_known_at=aware_datetime, countries=None)`` returns
JSON-serializable country manifest rows with verified ``series`` and explicit
``gaps``. Each of the 19 source series is selected by release availability at
the cutoff before contract or observation filtering. Forecast observations keep
their recorded source/status; no current projection is read or initialized.

The top-level ``evidence_digest`` is SHA-256 of canonical JSON for every other
top-level field: sorted keys, UTF-8, compact separators, and no nonfinite values.
``protected_artifact_paths`` lists the evidence and catalogue files that callers
must not overwrite with generated assessment output.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from dalio.assessments.debt_evidence import load_national_debt_context
from dalio.data_sources.company_country_macro import (
    MANIFEST_PATH,
    METHOD,
    canonical,
    company_country_manifest,
    digest,
    selected_countries,
)
from dalio.data_sources.imf_datamapper import IMF_FUNDAMENTALS, IMF_PRIMARY_BALANCE_CODE
from dalio.data_sources.worldbank import WB_FUNDAMENTALS
from dalio.pipelines.fetch_company_country_macro import _verify_prior_release
from dalio.storage.db import DataRelease, DataReleaseArtifact, ReleaseObservation
from dalio.storage.releases import latest_release, make_partition_key

_REQUIRED_TABLES = {
    DataRelease.__tablename__, DataReleaseArtifact.__tablename__, ReleaseObservation.__tablename__,
}


def _expected_series() -> tuple[dict, ...]:
    expected = [dict(indicator=spec.indicator, family="wb", source=spec.source_label,
                     series_id=spec.wb_code) for spec in WB_FUNDAMENTALS]
    expected.extend(dict(indicator=spec.indicator, family="imf",
        source="IMF_FISCAL_MONITOR" if spec.imf_code == IMF_PRIMARY_BALANCE_CODE else "IMF_WEO",
        series_id=spec.imf_code) for spec in IMF_FUNDAMENTALS)
    return tuple(expected)


def _has_release_tables(engine: Engine) -> bool:
    # Connecting to an ordinary nonexistent SQLite path creates a database file,
    # even for SELECT. Avoid that mutation; URI/custom-creator engines preserve
    # their caller-supplied connection policy.
    database = engine.url.database
    if (engine.dialect.name == "sqlite" and database and database != ":memory:"
            and not database.startswith("file:") and not Path(database).exists()):
        return False
    return set(inspect(engine).get_table_names()) >= _REQUIRED_TABLES


def _stored_time(value: datetime | None) -> str | None:
    if value is None:
        return None
    # Release storage normalizes timestamps to naive UTC.
    return value.replace(tzinfo=UTC).isoformat() if value.tzinfo is None else value.astimezone(UTC).isoformat()


def _gap(spec: dict, reason: str) -> dict:
    return dict(indicator=spec["indicator"], family=spec["family"], reason=reason)


def _protect_bundles(bundle_cache: dict, protected: set[str]) -> None:
    """Protect every retained response in a validated bundle, including failed requests.

    Those requests may not appear among a country's successful series, but the
    bundle loader still verifies their hashes when reconstructing its evidence.
    """
    for path, sha256 in bundle_cache:
        bundle_path = Path(path)
        body = bundle_path.read_bytes()
        if digest(body) != sha256:
            raise ValueError("Acquisition bundle changed after evidence verification")
        protected.add(str(bundle_path.resolve()))
        for request in json.loads(body)["requests"]:
            protected.add(str(Path(request["path"]).resolve()))


def load_evidence(
    engine: Engine, *, as_known_at: datetime, countries: list[str] | None = None,
) -> dict:
    """Select and verify immutable evidence without DDL or database writes.

    Timezone-aware cutoffs are normalized to UTC. ``None`` selects all 19
    checked listing countries; an explicit selection accepts GB/UK aliases and
    preserves manifest order. Empty or unknown selections are rejected.

    Gap reasons are ``missing_tables``, ``no_release_as_known_at`` and
    ``unverified_release_contract``. A malformed selected v2 release raises an
    exception rather than falling back to an older release or current facts.
    """
    if (not isinstance(as_known_at, datetime) or as_known_at.tzinfo is None
            or as_known_at.utcoffset() is None):
        raise ValueError("as_known_at must be a timezone-aware datetime")
    cutoff = as_known_at.astimezone(UTC)
    if countries is not None and (not isinstance(countries, list) or not countries):
        raise ValueError("countries must be a nonempty list or None for all listing countries")
    manifest = company_country_manifest()
    selection = selected_countries(countries)
    expected = _expected_series()
    result = dict(as_known_at=cutoff.isoformat(), countries=[])
    protected = {str(MANIFEST_PATH.resolve()), str(Path(manifest["source"]["path"]).resolve())}
    country_results = [{**country, "series": [], "gaps": [], "national_debt_context": []}
                       for country in selection]
    result["countries"] = country_results
    if not _has_release_tables(engine):
        for country in country_results:
            country["gaps"] = [_gap(spec, "missing_tables") for spec in expected]
    else:
        bundle_cache = {}
        with Session(engine, autoflush=False) as session:
            for country in country_results:
                for spec in expected:
                    key = make_partition_key(spec["source"], spec["series_id"],
                                             country["country"], spec["indicator"])
                    # Selection precedes evidence-contract validation. A latest
                    # legacy or corrupt release must not reveal older v2 facts.
                    release = latest_release(session, key, as_known_at=cutoff)
                    if release is None:
                        country["gaps"].append(_gap(spec, "no_release_as_known_at"))
                        continue
                    if not (release.vintage_label or "").startswith(METHOD + ":"):
                        country["gaps"].append(_gap(spec, "unverified_release_contract"))
                        continue
                    _verify_prior_release(session, release, bundle_cache=bundle_cache)
                    artifacts = list(session.scalars(select(DataReleaseArtifact).where(
                        DataReleaseArtifact.release_id == release.id).order_by(DataReleaseArtifact.role)))
                    bundle_artifact = next(artifact for artifact in artifacts
                                           if artifact.role == "acquisition_bundle")
                    _, item = bundle_cache[(bundle_artifact.artifact_path,
                                            bundle_artifact.artifact_sha256)][key]
                    report = item["report"]
                    country["series"].append(dict(**spec, release_id=int(release.id),
                        partition_key=key, available_at=_stored_time(release.available_at),
                        retrieved_at=_stored_time(release.retrieved_at),
                        published_at=_stored_time(release.published_at), source_url=release.source_url,
                        publisher_metadata=report["publisher_metadata"],
                        artifacts=[dict(role=artifact.role, sha256=artifact.artifact_sha256,
                                        path=artifact.artifact_path) for artifact in artifacts],
                        observations=[dict(year=int(row.date.year), date=row.date.isoformat(),
                                           value=float(row.value), status=row.status, source=row.source)
                                      for row in item["frame"].itertuples(index=False)],
                        missingness=report["missingness"]))
                    protected.update(str(Path(artifact.artifact_path).resolve()) for artifact in artifacts)
        _protect_bundles(bundle_cache, protected)
    native = load_national_debt_context(engine, as_known_at=cutoff,
                                       countries=[country["country"] for country in country_results])
    for country in country_results:
        country["national_debt_context"] = native["contexts"][country["country"]]
    protected.update(native["protected_artifact_paths"])
    result["protected_artifact_paths"] = sorted(protected)
    result["evidence_digest"] = digest(canonical(result))
    return result
