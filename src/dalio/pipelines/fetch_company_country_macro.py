"""Collect/replay official macro histories for saved company listing countries.

Use an explicit staging database first. --from-bundle promotes the same verified
bytes offline; source errors and missing histories never erase earlier records.
This entry point does not run scores or modify the ranking country population.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from datetime import UTC
from pathlib import Path

from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from dalio.data_sources.company_country_macro import (
    METHOD,
    archive,
    canonical,
    collect_bundle,
    digest,
    load_bundle,
)
from dalio.storage.db import Observation, ReleaseObservation, init_db, make_engine
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseArtifactMeta,
    ReleaseMeta,
    ingest_release_snapshot,
    latest_release,
    make_partition_key,
)


def ingest_bundle(path: Path, *, engine: Engine) -> dict:
    """Verify raw evidence before opening the target; replace valid partitions atomically."""
    batch = load_bundle(path)
    artifact_root = path.resolve().parent.parent.parent / "normalized"
    prepared = []
    for item in batch["ready"]:
        frame, report = item["frame"], item["report"]
        native = canonical(dict(method=METHOD, country=report["country"],
            indicator=report["indicator"], publisher_metadata=report["publisher_metadata"],
            source_response_sha256=[ref["sha256"] for ref in item["refs"]],
            observations=[{**row, "date": row["date"].isoformat()}
                          for row in frame.to_dict("records")]))
        provenance = canonical(dict(method=METHOD, country=report["country"],
                                   series_id=item["series_id"], **report["missingness"]))
        evidence = [("native_series_payload", archive(native, artifact_root), digest(native)),
                    ("missingness_ledger", archive(provenance, artifact_root), digest(provenance)),
                    ("acquisition_bundle", batch["bundle_path"], batch["bundle_sha256"])]
        evidence += [("publisher_metadata" if index == 0 else f"source_response_{index:03d}",
                      Path(ref["path"]), ref["sha256"])
                     for index, ref in enumerate(item["refs"])]
        artifacts = tuple(ReleaseArtifactMeta(role=role, artifact_sha256=sha,
            artifact_path=str(artifact), native_payload_sha256=digest(native),
            missing_provenance_sha256=digest(provenance), provenance_json=provenance.decode())
            for role, artifact, sha in evidence)
        meta = ReleaseMeta(
            partition_key=make_partition_key(item["sources"][0], item["series_id"],
                                             report["country"], report["indicator"]),
            source_family=item["sources"][0], available_at=batch["retrieved_at"],
            retrieved_at=batch["retrieved_at"], source_url=item["source_url"],
            vintage_label=METHOD + ":" + digest(canonical(report["publisher_metadata"])),
            projection=ProjectionScope(report["country"], report["indicator"], item["sources"]),
            artifacts=artifacts)
        prepared.append((frame, meta))
    if not prepared:
        return {**batch["summary"], "created_releases": 0, "releases": []}
    init_db(engine)
    results = []
    with (engine.begin() as connection,
          Session(bind=connection, expire_on_commit=False,
                  join_transaction_mode="rollback_only") as session):
        for frame, meta in prepared:
            previous = latest_release(session, meta.partition_key)
            if previous and previous.available_at.replace(tzinfo=UTC) > meta.available_at:
                raise ValueError(f"Retrograde country acquisition: {meta.partition_key}")
            old_dates = set(session.scalars(select(Observation.date).where(
                Observation.country == meta.projection.country,
                Observation.indicator == meta.projection.indicator,
                Observation.source.in_(meta.projection.sources))))
            if previous:
                old_dates.update(session.scalars(select(ReleaseObservation.date).where(
                    ReleaseObservation.release_id == previous.id)))
            omitted = old_dates - set(frame.date)
            if omitted:
                raise ValueError(f"History contraction for {meta.partition_key}: "
                                 f"{len(omitted)} formerly finite periods now absent")
        for frame, meta in prepared:
            result = ingest_release_snapshot(session, frame, meta)
            results.append(dict(partition_key=meta.partition_key, **asdict(result)))
    return {**batch["summary"], "created_releases": sum(row["created"] for row in results),
            "releases": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Explicit SQLite staging/target DB")
    parser.add_argument("--evidence-root", type=Path,
                        default=Path("data/artifacts/company_country_macro"))
    parser.add_argument("--from-bundle", type=Path, help="Replay archived evidence offline")
    parser.add_argument("--countries", nargs="+", help="Listing country ISO2 codes; GB/UK accepted")
    parser.add_argument("--only", nargs="+", choices=("wb", "imf"), default=("wb", "imf"))
    parser.add_argument("--indicators", nargs="+")
    parser.add_argument("--report", type=Path, help="Write full dated coverage and provenance JSON")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.from_bundle and (args.countries or args.indicators or tuple(args.only) != ("wb", "imf")):
            parser.error("Bundle replay uses its recorded selection; do not add collection filters")
        path = args.from_bundle or collect_bundle(artifact_dir=args.evidence_root,
            countries=args.countries, families=tuple(args.only), indicators=args.indicators)
        result = ingest_bundle(path, engine=make_engine(args.db))
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({key: value for key, value in result.items()
                          if key not in {"partitions", "releases"}}, indent=2))
        return 2 if result["source_error_partitions"] else 0
    except Exception:
        logging.exception("Company-country acquisition failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
