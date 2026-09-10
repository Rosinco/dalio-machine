"""Optional original debt-office context, separate from harmonized macro facts.

Only the currently supported Swedish native contracts are used. Releases are
selected at the exact known-at instant for each snapshot before choosing the
latest monthly reference and funding publication. The retained source is then
fully restored and checked before any fact is filtered for the assessment.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import unquote

from sqlalchemy import Engine, inspect, select
from sqlalchemy.orm import Session

from dalio.storage import national_debt
from dalio.storage.db import DataRelease, DataReleaseArtifact, NationalDebtFact
from dalio.storage.releases import latest_release

MONTHLY_STREAM = "se_central_government_debt_monthly_report"
FUNDING_STREAM = "se_central_government_funding_plan"


def _utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _schema_present(engine: Engine) -> bool:
    database = engine.url.database
    if (engine.dialect.name == "sqlite" and database and database != ":memory:"
            and not database.startswith("file:") and not Path(database).exists()):
        return False
    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    return all(model.__tablename__ in tables and set(model.__table__.columns.keys()) <= {
        column["name"] for column in inspector.get_columns(model.__tablename__)}
        for model in (DataRelease, DataReleaseArtifact, NationalDebtFact))


def _selected_releases(session: Session, cutoff: datetime) -> list[DataRelease]:
    keys = session.scalars(select(DataRelease.partition_key).where(
        DataRelease.partition_key.startswith(national_debt.PREFIX)).distinct()).all()
    monthly, funding = [], []
    for key in sorted(keys):
        if not any(key.startswith(national_debt.PREFIX + stream + ":")
                   for stream in (MONTHLY_STREAM, FUNDING_STREAM)):
            continue
        # Do not request the end-of-day latest release and filter it afterward:
        # an afternoon revision must not hide an eligible morning release.
        release = latest_release(session, key, as_known_at=cutoff)
        if release is None:
            continue
        if key.startswith(national_debt.PREFIX + MONTHLY_STREAM + ":"):
            reference = date.fromisoformat(unquote(key.rsplit(":", 1)[1]))
            if reference <= cutoff.date():
                monthly.append((reference, release))
        else:
            if release.published_at is None:
                raise ValueError("Eligible national funding release lacks its publication clock")
            funding.append(release)
    selected = []
    if monthly:
        selected.append(max(monthly, key=lambda pair: (
            pair[0], _utc(pair[1].available_at), _utc(pair[1].retrieved_at), pair[1].id))[1])
    if funding:
        selected.append(max(funding, key=lambda release: (
            _utc(release.published_at), _utc(release.available_at),
            _utc(release.retrieved_at), release.id)))
    return selected


def _retained(row: dict, stream: str, snapshot: str, cutoff: datetime) -> bool:
    if row["value"] is None:
        return False
    dims = json.loads(row["dimensions_json"])
    if stream == MONTHLY_STREAM:
        if row["status"] != "observed" or row["period_end"] > cutoff.date():
            return False
        if row["metric"] == "central_gov_gross_debt_sek":
            return row["fact_type"] == "debt_stock" and dims.get("scope") == "central_government"
        return (row["metric"] == "average_time_to_refixing" and row["fact_type"] == "portfolio_risk"
                and dims.get("debt_class") == "total"
                and dims.get("aggregation") in {"monthly_mean", "reference_date"})
    return (stream == FUNDING_STREAM and row["fact_type"] == "funding_flow"
            and row["metric"] in {"gross_borrowing_requirement", "redemptions"}
            and row["source_locator"].startswith("XLSX F10!")
            and dims.get("forecast_vintage") == snapshot
            and row["status"] == "forecast" and row["period_end"].year >= cutoff.year)


def load_national_debt_context(
    engine: Engine, *, as_known_at: datetime, countries: list[str],
) -> dict:
    """Return ``contexts`` by internal country code and protected artifact paths.

    Missing native schema/releases yield empty lists; corrupt selected evidence
    raises. No current observation projection, DDL or database writes are used.
    Fact references are ``native:r<release_id>:<stored fact_key>``. Units, scope,
    aggregation, forecast vintage and native source labels are never flattened.
    """
    if (not isinstance(as_known_at, datetime) or as_known_at.tzinfo is None
            or as_known_at.utcoffset() is None):
        raise ValueError("as_known_at must be a timezone-aware datetime")
    cutoff = as_known_at.astimezone(UTC)
    contexts = {country: [] for country in countries}
    protected = set()
    result = {"contexts": contexts, "protected_artifact_paths": []}
    if "SE" not in contexts or not _schema_present(engine):
        return result
    with Session(engine, autoflush=False) as session:
        for release in _selected_releases(session, cutoff):
            item = national_debt._restore_release(session, release, cutoff.date())
            document = item.document
            if document.country != "SE" or document.reference_date > cutoff.date():
                raise ValueError("Selected native debt source has an unexpected country/reference date")
            artifacts = [dict(role=artifact.role, sha256=artifact.artifact_sha256,
                              path=artifact.artifact_path) for artifact in item.artifacts]
            protected.update(str(Path(artifact.artifact_path).resolve()) for artifact in item.artifacts)
            metadata = {**document.metadata, "stream_id": document.stream_id,
                        "snapshot_key": document.snapshot_key,
                        "reference_date": document.reference_date.isoformat()}
            for row in national_debt._rows(document):
                if not _retained(row, document.stream_id, document.snapshot_key, cutoff):
                    continue
                contexts["SE"].append(dict(metric=row["metric"], value=float(row["value"]),
                    unit=row["unit"], period_start=row["period_start"].isoformat(),
                    period_end=row["period_end"].isoformat(), year=int(row["period_end"].year),
                    status=row["status"], dimensions=json.loads(row["dimensions_json"]),
                    source_locator=row["source_locator"], native_label=row["native_label"],
                    source=national_debt.SOURCE, source_url=document.source_url,
                    release_id=int(release.id), series_id=document.stream_id,
                    partition_key=release.partition_key, available_at=_utc(release.available_at).isoformat(),
                    retrieved_at=_utc(release.retrieved_at).isoformat(),
                    published_at=_utc(release.published_at).isoformat(), publisher_metadata=metadata,
                    artifacts=artifacts, evidence_ref=f"native:r{release.id}:{row['fact_key']}"))
    contexts["SE"].sort(key=lambda fact: (fact["period_end"], fact["metric"], fact["evidence_ref"]))
    result["protected_artifact_paths"] = sorted(protected)
    return result
