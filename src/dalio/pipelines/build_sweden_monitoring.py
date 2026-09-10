"""Export the fixed Sweden monitoring pilot from original evidence, offline."""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import UTC, date, datetime, time
from pathlib import Path

from sqlalchemy import create_engine

from dalio.monitoring.core import build_monitoring, validate_monitoring
from dalio.pipelines.build_country_assessments import (
    _check_output,
    _json_bytes,
    _protected_paths,
    _publish_directory,
    _update_latest,
)

logger = logging.getLogger(__name__)
DEFAULT_ARTIFACT_ROOT = Path("data/artifacts/sweden_monitoring")
DEFAULT_OUTPUT_DIR = Path("data/snapshots/sweden_monitoring")


def load_evidence(engine, *, as_known_at, bundle_paths):
    from dalio.monitoring.evidence import load_evidence as load

    return load(engine, as_known_at=as_known_at, bundle_paths=bundle_paths)


def load_assessment(engine, *, as_known_at, as_of):
    from dalio.assessments.core import build_snapshot
    from dalio.assessments.evidence import load_evidence as load

    return build_snapshot(load(engine, as_known_at=as_known_at, countries=["SE"]), as_of=as_of)


def render_monitoring(snapshot):
    from dalio.monitoring.render import render_monitoring as render

    return render(snapshot)


def export_monitoring(
    *, db_path: Path, as_of: date | None = None, as_known_at: datetime | None = None,
    bundle_paths: list[Path] | None = None, artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR, update_latest: bool = True,
) -> Path:
    db_path, output_dir = Path(db_path), Path(output_dir)
    if not db_path.is_file():
        raise ValueError(f"Source database does not exist: {db_path}")
    now = datetime.now(UTC)
    as_of = as_of or now.date()
    if as_known_at is None:
        as_known_at = min(now, datetime.combine(as_of, time.max, UTC))
    if as_known_at.tzinfo is None or as_known_at.utcoffset() is None:
        raise ValueError("Monitoring knowledge cutoff needs an explicit timezone")
    as_known_at = as_known_at.astimezone(UTC)
    if as_of > as_known_at.date():
        raise ValueError("Monitoring date is after the UTC knowledge cutoff")
    paths = ([Path(path) for path in bundle_paths] if bundle_paths is not None else
             sorted((Path(artifact_root) / "bundles").glob("*.json")))
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Duplicate monitoring bundle paths")
    protected = {db_path.resolve(), *(path.resolve() for path in paths)}
    _check_output(output_dir, protected)
    engine = create_engine(
        "sqlite://", creator=lambda: sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True),
    )
    try:
        evidence = load_evidence(engine, as_known_at=as_known_at, bundle_paths=paths)
        assessment = load_assessment(engine, as_known_at=as_known_at, as_of=as_of)
        snapshot = build_monitoring(evidence, assessment, as_of=as_of)
    finally:
        engine.dispose()
    validate_monitoring(snapshot)
    protected |= _protected_paths(evidence) | _protected_paths(snapshot)
    files = {"snapshot.json": _json_bytes(snapshot),
             "SE.md": render_monitoring(snapshot).encode("utf-8")}
    digest = snapshot["snapshot_sha256"]
    directory = output_dir / digest
    _check_output(output_dir, protected,
                  [directory, output_dir / "LATEST.json", *(directory / name for name in files)])
    if update_latest and (output_dir / "LATEST.json").is_symlink():
        raise ValueError("LATEST.json must not be a symbolic link")
    result = _publish_directory(output_dir, digest, files)
    if update_latest:
        _update_latest(output_dir, snapshot)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat, help="UTC assessment date")
    parser.add_argument("--known-at", type=datetime.fromisoformat, help="Exact timezone-aware cutoff")
    parser.add_argument("--bundle", action="append", type=Path,
                        help="Explicit supplement bundle; repeat for known-at selection")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT,
                        help="Discover retained bundles here when --bundle is omitted")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--no-latest", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        directory = export_monitoring(db_path=args.db, as_of=args.as_of, as_known_at=args.known_at,
                                      bundle_paths=args.bundle, artifact_root=args.artifact_root,
                                      output_dir=args.output_dir, update_latest=not args.no_latest)
    except Exception as exc:  # noqa: BLE001 - leave existing complete output intact
        logger.exception("Sweden monitoring export failed: %s", exc)
        return 1
    print(json.dumps({"directory": str(directory.resolve()), "snapshot_sha256": directory.name}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
