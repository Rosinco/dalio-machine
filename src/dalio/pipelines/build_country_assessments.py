"""Build deterministic offline country assessments from a read-only SQLite source."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import sqlite3
import tempfile
from datetime import UTC, date, datetime, time
from pathlib import Path

from sqlalchemy import create_engine

from dalio.assessments.render import country_code, render_country, render_index

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT_DIR = Path("data/snapshots/country_assessments")


def load_evidence(engine, *, as_known_at: datetime, countries: list[str] | None):
    from dalio.assessments.evidence import load_evidence as load

    return load(engine, as_known_at=as_known_at, countries=countries)


def build_snapshot(evidence: dict, *, as_of: date) -> dict:
    from dalio.assessments.core import build_snapshot as build

    return build(evidence, as_of=as_of)


def _json_bytes(value, *, compact=False) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":") if compact else None,
        indent=None if compact else 2,
    ).encode("utf-8") + (b"" if compact else b"\n")


def _protected_paths(value) -> set[Path]:
    paths = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "protected_artifact_paths":
                paths.update(Path(path).resolve() for path in child)
            elif key in {"artifact_path", "source_artifact_path", "evidence_path"} and child:
                paths.add(Path(child).resolve())
            else:
                paths.update(_protected_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.update(_protected_paths(child))
    return paths


def _check_output(output_dir: Path, protected: set[Path], prospective=()) -> None:
    root = output_dir.resolve()
    targets = {root, *(Path(path).resolve() for path in prospective)}
    for path in protected:
        for target in targets:
            if target == path or path in target.parents:
                raise ValueError(
                    "Assessment output would overwrite or enter a protected database/evidence path"
                )
        # Content-addressed evidence lives under an artifacts namespace. Keep
        # reports outside that namespace even when the individual files differ.
        artifact_root = next(
            (parent for parent in path.parents if parent.name == "artifacts"), None
        )
        if artifact_root is not None and (root == artifact_root or artifact_root in root.parents):
            raise ValueError(
                "Assessment output cannot be placed in the protected artifacts namespace"
            )
    if output_dir.exists() and not output_dir.is_dir():
        raise ValueError("Assessment output must be a directory")


def _write_file(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(body)


def _compare_existing(directory: Path, files: dict[str, bytes]) -> None:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("The existing immutable snapshot path is not a real directory")
    actual = {}
    expected_directories = {
        str(parent) for name in files for parent in Path(name).parents if str(parent) != "."
    }
    actual_directories = set()
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError("The existing immutable snapshot contains a symbolic link")
        if path.is_file():
            actual[path.relative_to(directory).as_posix()] = path.read_bytes()
        elif path.is_dir():
            actual_directories.add(path.relative_to(directory).as_posix())
        else:
            raise ValueError("The existing immutable snapshot contains an unexpected special file")
    if actual != files or actual_directories != expected_directories:
        raise ValueError(
            "The existing immutable snapshot differs from the complete rendered content"
        )


def _publish_directory(output_dir: Path, digest: str, files: dict[str, bytes]) -> Path:
    directory = output_dir / digest
    if directory.exists() or directory.is_symlink():
        _compare_existing(directory, files)
        return directory
    output_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".build-", dir=output_dir))
    try:
        for name, body in sorted(files.items()):
            _write_file(stage / name, body)
        _compare_existing(stage, files)
        if directory.exists() or directory.is_symlink():
            _compare_existing(directory, files)
        else:
            try:
                stage.rename(directory)
            except OSError:
                if not directory.exists():
                    raise
                # Another publisher may have completed the identical immutable
                # snapshot between the existence check and directory rename.
                _compare_existing(directory, files)
        return directory
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def _update_latest(output_dir: Path, snapshot: dict) -> None:
    latest = output_dir / "LATEST.json"
    if latest.is_symlink():
        raise ValueError("LATEST.json must not be a symbolic link")
    payload = _json_bytes(
        {
            "snapshot_sha256": snapshot["snapshot_sha256"],
            "directory": snapshot["snapshot_sha256"],
            "as_of": snapshot["as_of"],
            "as_known_at": snapshot["as_known_at"],
        }
    )
    if latest.is_file() and latest.read_bytes() == payload:
        return
    descriptor, name = tempfile.mkstemp(prefix=".latest-", dir=output_dir)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_bytes(payload)
        temporary.replace(latest)
    finally:
        temporary.unlink(missing_ok=True)


def export_assessments(
    *,
    db_path: Path,
    as_of: date | None = None,
    as_known_at: datetime | None = None,
    countries: list[str] | None = None,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    update_latest: bool = True,
) -> Path:
    db_path, output_dir = Path(db_path), Path(output_dir)
    if not db_path.is_file():
        raise ValueError(f"Source database does not exist: {db_path}")
    now = datetime.now(UTC)
    as_of = as_of or now.date()
    if as_known_at is None:
        as_known_at = min(now, datetime.combine(as_of, time.max, UTC))
    elif as_known_at.tzinfo is None or as_known_at.utcoffset() is None:
        raise ValueError("Knowledge cutoff must include an explicit timezone")
    as_known_at = as_known_at.astimezone(UTC)
    if countries is not None:
        countries = [
            "UK" if code.strip().upper() == "GB" else code.strip().upper() for code in countries
        ]
        if not countries or any(not re.fullmatch(r"[A-Z]{2}", code) for code in countries):
            raise ValueError("Country selection must contain two-letter country codes")
        if len(countries) != len(set(countries)):
            raise ValueError("Duplicate country selection after GB/UK normalization")
    protected = {db_path.resolve()}
    _check_output(output_dir, protected)
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True),
    )
    try:
        evidence = load_evidence(engine, as_known_at=as_known_at, countries=countries)
        snapshot = build_snapshot(evidence, as_of=as_of)
    finally:
        engine.dispose()
    digest = snapshot.get("snapshot_sha256")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("Assessment snapshot must have a valid content hash")
    canonical = _json_bytes(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}, compact=True
    )
    if hashlib.sha256(canonical).hexdigest() != digest:
        raise ValueError("Assessment snapshot content hash does not match its contents")
    protected |= _protected_paths(evidence) | _protected_paths(snapshot)
    files = {
        "snapshot.json": _json_bytes(snapshot),
        "index.md": render_index(snapshot).encode("utf-8"),
    }
    if not isinstance(snapshot.get("countries"), list) or not snapshot["countries"]:
        raise ValueError("Assessment snapshot has no countries")
    for country in snapshot["countries"]:
        name = f"countries/{country_code(country)}.md"
        if name in files:
            raise ValueError("Duplicate normalized country assessment filename")
        files[name] = render_country(country, snapshot).encode("utf-8")
    directory = output_dir / digest
    _check_output(
        output_dir,
        protected,
        [directory, output_dir / "LATEST.json", *(directory / name for name in files)],
    )
    if update_latest and (output_dir / "LATEST.json").is_symlink():
        raise ValueError("LATEST.json must not be a symbolic link")
    result = _publish_directory(output_dir, digest, files)
    if update_latest:
        _update_latest(output_dir, snapshot)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--as-of", type=date.fromisoformat)
    parser.add_argument(
        "--known-at", type=datetime.fromisoformat, help="Timezone-aware ISO timestamp"
    )
    parser.add_argument(
        "--countries", nargs="+", help="Country codes separated by spaces or commas"
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--no-latest", action="store_true", help="Keep the existing LATEST.json pointer"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    countries = (
        [code for group in args.countries for code in group.split(",")] if args.countries else None
    )
    try:
        directory = export_assessments(
            db_path=args.db,
            as_of=args.as_of,
            as_known_at=args.known_at,
            countries=countries,
            output_dir=args.output_dir,
            update_latest=not args.no_latest,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed without publishing partial reports
        logger.exception("Country assessment export failed: %s", exc)
        return 1
    print(
        json.dumps(
            {"directory": str(directory.resolve()), "snapshot_sha256": directory.name}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
