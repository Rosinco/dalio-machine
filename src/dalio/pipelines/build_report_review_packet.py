"""Build a deterministic, read-only packet of unverified report claim drafts.

The checked candidate catalogue and the immutable report ledger are the inputs.
This command never promotes a candidate to the claims table; it only writes
regenerable JSON and Markdown review aids beneath ``data/review`` by default.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import create_engine

from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import (
    build_review_packet,
    load_candidate_catalogue,
    render_review_markdown,
)
from dalio.storage.db import make_session_factory

LATEST_JSON = "report_claims_latest.json"
LATEST_MARKDOWN = "report_claims_latest.md"


def _read_only_engine(path: Path):
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"database does not exist: {resolved}")
    return create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(f"file:{resolved}?mode=ro", uri=True),
        future=True,
    )


def _atomic_write(path: Path, payload: str, *, immutable: bool = False) -> Path:
    encoded = payload.encode("utf-8")
    if immutable and path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"refusing to overwrite different hash-addressed output: {path}")
        return path

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _packet_date(packet: dict[str, Any]) -> str:
    known_at = packet.get("as_known_at")
    if not isinstance(known_at, str) or len(known_at) < 10:
        raise ValueError("review packet lacks a canonical as_known_at timestamp")
    return known_at[:10]


def run(
    *,
    db_path: Path,
    catalogue_path: Path,
    output_dir: Path,
    require_all_sources: bool = True,
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    """Validate the catalogue against a read-only ledger and publish review aids."""

    catalogue = load_candidate_catalogue(catalogue_path)
    engine = _read_only_engine(db_path)
    try:
        factory = make_session_factory(engine)
        with factory() as session:
            packet = build_review_packet(session, catalogue)
            if session.new or session.dirty or session.deleted:
                raise RuntimeError("report review packet builder attempted to mutate its session")
    finally:
        engine.dispose()

    if require_all_sources:
        expected = {source.source_id for source in REPORT_SOURCES if source.enabled}
        present = {document["source_id"] for document in packet["documents"]}
        missing = sorted(expected - present)
        unexpected = sorted(present - expected)
        if missing or unexpected:
            details = []
            if missing:
                details.append(f"missing {', '.join(missing)}")
            if unexpected:
                details.append(f"unexpected {', '.join(unexpected)}")
            raise RuntimeError(
                "report review packet requires every enabled official family: " + "; ".join(details)
            )

    json_payload = (
        json.dumps(packet, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    markdown = render_review_markdown(packet)
    if not markdown.endswith("\n"):
        markdown += "\n"

    packet_hash = packet.get("packet_sha256")
    if not isinstance(packet_hash, str) or len(packet_hash) != 64:
        raise ValueError("review packet lacks a full SHA-256 fingerprint")
    dated_stem = f"report_claims_{_packet_date(packet)}_{packet_hash[:16]}"
    dated_json = output_dir / f"{dated_stem}.json"
    dated_markdown = output_dir / f"{dated_stem}.md"

    # Establish the immutable copies first. Moving aliases only after both
    # succeed avoids publishing a latest packet without its reproducible pair.
    paths = (
        _atomic_write(dated_json, json_payload, immutable=True),
        _atomic_write(dated_markdown, markdown, immutable=True),
        _atomic_write(output_dir / LATEST_JSON, json_payload),
        _atomic_write(output_dir / LATEST_MARKDOWN, markdown),
    )
    return packet, paths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a read-only human-review packet of page-cited, unverified model drafts."
        )
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="SQLite database (default: DALIO_DB_PATH or data/dalio.db).",
    )
    parser.add_argument(
        "--catalogue",
        type=Path,
        default=None,
        help=(
            "Checked candidate catalogue (default: DALIO_REPORT_CANDIDATES or "
            "data/reference/report_claim_candidates.json)."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Generated output directory (default: DALIO_REPORT_REVIEW_DIR or data/review).",
    )
    args = parser.parse_args(argv)
    load_dotenv()

    db_path = args.db or Path(os.environ.get("DALIO_DB_PATH", "data/dalio.db"))
    catalogue_path = args.catalogue or Path(
        os.environ.get(
            "DALIO_REPORT_CANDIDATES",
            "data/reference/report_claim_candidates.json",
        )
    )
    output_dir = args.out_dir or Path(os.environ.get("DALIO_REPORT_REVIEW_DIR", "data/review"))

    try:
        packet, paths = run(
            db_path=db_path,
            catalogue_path=catalogue_path,
            output_dir=output_dir,
        )
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"dalio-report-review-packet: {exc}", file=sys.stderr)
        return 2

    print(
        "Report review packet "
        f"known at {packet['as_known_at']}: "
        f"{packet['candidate_count']} unverified model drafts across "
        f"{packet['document_count']} documents"
    )
    for path in paths:
        print(f"  wrote {path}")
    print("  verified conclusions: 0 (human review required)")
    print("  database writes: none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
