"""Publish an offline Sweden/Norway/Denmark/Finland monitoring comparison."""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, date, datetime, time
from pathlib import Path

from sqlalchemy import create_engine

from dalio.assessments.core import build_snapshot as build_assessment
from dalio.assessments.evidence import load_evidence as load_assessment
from dalio.assessments.render import render_country as render_context_country
from dalio.monitoring.evidence import load_evidence as load_sweden
from dalio.nordic_monitoring.acquisition import load_evidence, utc
from dalio.nordic_monitoring.core import COUNTRIES, build_snapshot, validate_snapshot
from dalio.nordic_monitoring.render import render_country, render_index
from dalio.pipelines.build_country_assessments import (
    _check_output,
    _json_bytes,
    _protected_paths,
    _publish_directory,
    _update_latest,
)

DEFAULT_ARTIFACT_ROOT = Path("data/artifacts/nordic_monitoring")
DEFAULT_SWEDEN_ROOT = Path("data/artifacts/sweden_monitoring")
DEFAULT_OUTPUT_DIR = Path("data/snapshots/nordic_monitoring")


def load_inputs(engine, *, as_known_at, as_of, bundle_paths, sweden_bundle_paths):
    nordic = load_evidence(bundle_paths=bundle_paths, as_known_at=as_known_at)
    sweden = load_sweden(engine, bundle_paths=sweden_bundle_paths, as_known_at=as_known_at)
    assessment = build_assessment(load_assessment(engine, as_known_at=as_known_at, countries=list(COUNTRIES)), as_of=as_of)
    return nordic, sweden, assessment


def export_monitoring(*, db_path: Path, as_of=None, as_known_at=None, bundle_paths=None,
                      sweden_bundle_paths=None, artifact_root=DEFAULT_ARTIFACT_ROOT,
                      sweden_artifact_root=DEFAULT_SWEDEN_ROOT, output_dir=DEFAULT_OUTPUT_DIR,
                      update_latest=True) -> Path:
    db_path, output_dir = Path(db_path), Path(output_dir)
    if not db_path.is_file():
        raise ValueError(f"Source database does not exist: {db_path}")
    now = datetime.now(UTC)
    as_of = as_of or now.date()
    cutoff = utc(as_known_at) if as_known_at is not None else min(now, datetime.combine(as_of, time.max, UTC))
    if as_of > cutoff.date():
        raise ValueError("Nordic assessment date is after the UTC knowledge cutoff")
    paths = list(bundle_paths) if bundle_paths is not None else sorted((Path(artifact_root) / "bundles").glob("*.json"))
    se_paths = (list(sweden_bundle_paths) if sweden_bundle_paths is not None else
                sorted((Path(sweden_artifact_root) / "bundles").glob("*.json")))
    if len({Path(p).resolve() for p in paths + se_paths}) != len(paths + se_paths):
        raise ValueError("Duplicate Nordic/Sweden bundle paths")
    protected = {db_path.resolve(), *(Path(p).resolve() for p in paths + se_paths)}
    _check_output(output_dir, protected)
    engine = create_engine("sqlite://", creator=lambda: sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True))
    try:
        inputs = load_inputs(engine, as_known_at=cutoff, as_of=as_of, bundle_paths=paths, sweden_bundle_paths=se_paths)
        snapshot = build_snapshot(*inputs, as_of=as_of)
    finally:
        engine.dispose()
    validate_snapshot(snapshot)
    protected |= _protected_paths(snapshot)
    files = {"snapshot.json": _json_bytes(snapshot), "index.md": render_index(snapshot).encode("utf-8")}
    contexts = {c["country"]: c for c in snapshot["country_assessment"]["countries"]}
    for code in COUNTRIES:
        files[f"{code}.md"] = render_country(snapshot, code).encode("utf-8")
        files[f"context/{code}.md"] = render_context_country(contexts[code], snapshot["country_assessment"]).encode("utf-8")
    directory = output_dir / snapshot["snapshot_sha256"]
    _check_output(output_dir, protected, [directory, output_dir / "LATEST.json", *(directory / name for name in files)])
    if update_latest and (output_dir / "LATEST.json").is_symlink():
        raise ValueError("LATEST.json must not be a symlink")
    result = _publish_directory(output_dir, snapshot["snapshot_sha256"], files)
    if update_latest:
        _update_latest(output_dir, snapshot)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument("--known-at", type=datetime.fromisoformat)
    parser.add_argument("--bundle", action="append", type=Path)
    parser.add_argument("--sweden-bundle", action="append", type=Path)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--sweden-artifact-root", type=Path, default=DEFAULT_SWEDEN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-latest", action="store_true")
    args = parser.parse_args(argv)
    directory = export_monitoring(db_path=args.db, as_of=args.as_of, as_known_at=args.known_at,
        bundle_paths=args.bundle, sweden_bundle_paths=args.sweden_bundle, artifact_root=args.artifact_root,
        sweden_artifact_root=args.sweden_artifact_root, output_dir=args.output_dir, update_latest=not args.no_latest)
    print(json.dumps({"directory": str(directory.resolve()), "snapshot_sha256": directory.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
