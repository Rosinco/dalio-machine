"""Ingest checked AP-fund disclosures and their official PDF artifacts."""

from __future__ import annotations

import argparse
import hashlib
import logging
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv
from sqlalchemy import Engine

from dalio.data_sources.ap_funds import (
    AP_FUND_PARSER_NAME,
    AP_FUND_PARSER_VERSION,
    AP_FUNDS_H1_2026_REFERENCE,
    SOURCE_AP_FUNDS,
    ApFundReference,
    ApFundRelease,
    load_ap_fund_reference,
)
from dalio.storage.allocators import AllocatorReleaseMeta, ingest_allocator_snapshot
from dalio.storage.db import init_db, make_engine, make_session_factory

logger = logging.getLogger(__name__)

DEFAULT_ALLOCATOR_BLOB_ROOT = Path("data/artifacts/allocators")


def partition_key_for(fund: str, report_date: date) -> str:
    """Stable identity of one fund's complete half-year report transcription."""
    clean_fund = fund.strip()
    if not clean_fund:
        raise ValueError("fund must not be empty")
    if not isinstance(report_date, date):
        raise ValueError("report_date must be a date")
    return f"allocator:{quote(clean_fund, safe='')}:half-year:{report_date.isoformat()}"


def _select_releases(
    reference: ApFundReference,
    funds: Sequence[str] | None,
) -> tuple[ApFundRelease, ...]:
    if funds is None:
        return reference.releases
    requested = tuple(fund.strip() for fund in funds)
    if not requested or any(not fund for fund in requested):
        raise ValueError("fund selection must not be empty")
    if len(requested) != len(set(requested)):
        raise ValueError("fund selection contains duplicates")
    by_fund = {release.fund: release for release in reference.releases}
    unknown = sorted(set(requested) - set(by_fund))
    if unknown:
        raise ValueError(f"unknown AP-fund releases: {', '.join(unknown)}")
    return tuple(by_fund[fund] for fund in requested)


def _resolve_artifact_paths(
    releases: Sequence[ApFundRelease],
    *,
    artifacts: Mapping[str, Path | str] | None,
    artifact_dir: Path | None,
) -> dict[str, Path]:
    if (artifacts is None) == (artifact_dir is None):
        raise ValueError("provide either artifacts or artifact_dir, but not both")
    expected = {release.fund for release in releases}
    if artifacts is not None:
        if any(not isinstance(fund, str) or not fund.strip() for fund in artifacts):
            raise ValueError("artifact mapping keys must be non-empty fund names")
        normalized = {fund.strip(): Path(path) for fund, path in artifacts.items()}
        if len(normalized) != len(artifacts):
            raise ValueError("artifact mappings contain duplicate normalized fund names")
        missing = sorted(expected - set(normalized))
        unexpected = sorted(set(normalized) - expected)
        if missing:
            raise ValueError(f"missing artifact mappings: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"unexpected artifact mappings: {', '.join(unexpected)}")
        return normalized

    base = Path(artifact_dir)
    return {release.fund: base / release.artifact_filename for release in releases}


def _preflight_artifacts(
    releases: Sequence[ApFundRelease], paths: Mapping[str, Path]
) -> dict[str, bytes]:
    """Read and verify every input before creating a database or storing a release."""
    verified: dict[str, bytes] = {}
    for release in releases:
        path = paths[release.fund]
        if not path.is_file():
            raise ValueError(f"{release.fund} artifact does not exist or is not a file: {path}")
        pdf_bytes = path.read_bytes()
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(f"{release.fund} artifact is not a PDF: {path}")
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if digest != release.sha256:
            raise ValueError(
                f"{release.fund} artifact sha256 mismatch: expected {release.sha256}, "
                f"calculated {digest}"
            )
        verified[release.fund] = pdf_bytes
    return verified


def run_pipeline(
    *,
    reference_path: Path = AP_FUNDS_H1_2026_REFERENCE,
    artifacts: Mapping[str, Path | str] | None = None,
    artifact_dir: Path | None = None,
    funds: Sequence[str] | None = None,
    blob_root: Path = DEFAULT_ALLOCATOR_BLOB_ROOT,
    engine: Engine | None = None,
    retrieved_at: datetime | None = None,
) -> dict[str, dict[str, object]]:
    """Preflight and append selected checked AP-fund report releases.

    ``artifacts`` must map exactly the selected funds to local official PDF
    files.  Alternatively, ``artifact_dir`` uses each release's deterministic
    filename (for example ``ap2-h1-2026.pdf``).  Every artifact is verified
    before the database is initialized, preventing a late missing/hash-mismatch
    error from leaving a partially ingested batch.
    """
    reference = load_ap_fund_reference(reference_path)
    selected = _select_releases(reference, funds)
    paths = _resolve_artifact_paths(selected, artifacts=artifacts, artifact_dir=artifact_dir)
    pdfs = _preflight_artifacts(selected, paths)

    run_at = retrieved_at or datetime.now(UTC)
    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    summary: dict[str, dict[str, object]] = {}
    with session_factory() as session:
        for release in selected:
            result = ingest_allocator_snapshot(
                session,
                release.facts,
                pdfs[release.fund],
                AllocatorReleaseMeta(
                    partition_key=partition_key_for(release.fund, release.report_date),
                    source_family=SOURCE_AP_FUNDS,
                    fund=release.fund,
                    report_date=release.report_date,
                    title=release.title,
                    available_at=run_at,
                    retrieved_at=run_at,
                    source_url=release.source_url,
                    official_domains=release.official_domains,
                    expected_sha256=release.sha256,
                    parser_name=AP_FUND_PARSER_NAME,
                    parser_version=AP_FUND_PARSER_VERSION,
                ),
                blob_root=blob_root,
            )
            summary[release.fund] = {
                "fund": release.fund,
                "rows": result.row_count,
                "release_id": result.release_id,
                "created": result.created,
                "artifact_sha256": result.artifact_sha256,
                "blob_path": str(result.blob_path),
            }
            logger.info(
                "Stored %s allocator report: %d rows (release %d, created=%s)",
                release.fund,
                result.row_count,
                result.release_id,
                result.created,
            )
    return summary


def _parse_artifact_assignments(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        fund, separator, raw_path = value.partition("=")
        fund = fund.strip()
        raw_path = raw_path.strip()
        if not separator or not fund or not raw_path:
            raise ValueError("--artifact must use FUND=PATH (for example AP2=/tmp/ap2.pdf)")
        if fund in result:
            raise ValueError(f"duplicate --artifact assignment for {fund}")
        result[fund] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest checked AP-fund facts with their official PDF artifacts."
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=AP_FUNDS_H1_2026_REFERENCE,
        help="Versioned AP-fund reference JSON.",
    )
    artifact_group = parser.add_mutually_exclusive_group(required=True)
    artifact_group.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory containing deterministic names such as ap2-h1-2026.pdf.",
    )
    artifact_group.add_argument(
        "--artifact",
        action="append",
        metavar="FUND=PATH",
        help="Exact fund-to-PDF mapping; repeat once for every selected fund.",
    )
    parser.add_argument(
        "--fund",
        action="append",
        help="Ingest only this fund; repeat to select multiple funds. Default: all.",
    )
    parser.add_argument(
        "--blob-root",
        type=Path,
        default=DEFAULT_ALLOCATOR_BLOB_ROOT,
        help="Content-addressed archive root.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    load_dotenv()
    try:
        artifact_mapping = (
            _parse_artifact_assignments(args.artifact) if args.artifact is not None else None
        )
        summary = run_pipeline(
            reference_path=args.reference,
            artifacts=artifact_mapping,
            artifact_dir=args.artifact_dir,
            funds=tuple(args.fund) if args.fund else None,
            blob_root=args.blob_root,
        )
    except Exception as exc:  # noqa: BLE001 - CLI returns a clear non-zero failure
        logger.exception("AP-fund ingestion failed: %s", exc)
        print(f"AP-fund ingestion failed: {exc}", file=sys.stderr)
        return 1

    created = sum(bool(result["created"]) for result in summary.values())
    rows = sum(int(result["rows"]) for result in summary.values())
    print(
        f"AP-fund reports: {len(summary)} artifact-verified; "
        f"{rows} model-checked facts; {created} new releases."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
