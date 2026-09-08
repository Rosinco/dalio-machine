"""Ingest checked-in official-report metadata and local PDF artifacts.

This command intentionally has no downloader. Acquisition and metadata review
remain separate from the immutable ingestion boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import Engine

from dalio.reports.catalogue import ReportIssue, load_issue_catalogue
from dalio.reports.poppler import PopplerPageExtractor
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.reports import PageExtractor, PageText, ingest_report

DEFAULT_CATALOGUE_PATH = Path("data/reference/report_issues.json")
DEFAULT_BLOB_ROOT = Path("data/artifacts/reports")


def issue_identity(source_id: str, issue_key: str) -> str:
    return f"{source_id}:{issue_key}"


@dataclass(frozen=True)
class _PreparedIssue:
    issue: ReportIssue
    pdf_bytes: bytes
    pages: tuple[PageText, ...]


@dataclass(frozen=True)
class _PreparedExtractor:
    name: str
    version: str
    content_sha256: str
    pages: tuple[PageText, ...]

    def extract(self, pdf_bytes: bytes) -> tuple[PageText, ...]:
        if hashlib.sha256(pdf_bytes).hexdigest() != self.content_sha256:
            raise ValueError("prepared report bytes changed before ingestion")
        return self.pages


def _select_issues(
    catalogue: tuple[ReportIssue, ...], issue_ids: tuple[str, ...] | None
) -> tuple[ReportIssue, ...]:
    if issue_ids is None:
        return catalogue
    if not issue_ids:
        raise ValueError("issue_ids must not be empty when supplied")
    if len(set(issue_ids)) != len(issue_ids):
        raise ValueError("issue_ids contains duplicates")
    by_identity = {issue.identity: issue for issue in catalogue}
    unknown = sorted(set(issue_ids) - set(by_identity))
    if unknown:
        raise ValueError(f"unknown issue selection: {', '.join(unknown)}")
    requested = set(issue_ids)
    return tuple(issue for issue in catalogue if issue.identity in requested)


def _artifact_paths(
    issues: tuple[ReportIssue, ...],
    *,
    artifacts: Mapping[str, Path] | None,
    artifact_dir: Path | None,
) -> dict[str, Path]:
    if (artifacts is None) == (artifact_dir is None):
        raise ValueError("provide either artifacts or artifact_dir, but not both")
    identities = {issue.identity for issue in issues}
    if artifacts is not None:
        supplied = set(artifacts)
        missing = sorted(identities - supplied)
        unexpected = sorted(supplied - identities)
        if missing:
            raise ValueError(f"missing artifact mappings: {', '.join(missing)}")
        if unexpected:
            raise ValueError(f"unexpected artifact mappings: {', '.join(unexpected)}")
        return {identity: Path(artifacts[identity]) for identity in identities}
    assert artifact_dir is not None
    root = Path(artifact_dir)
    return {issue.identity: root / issue.artifact_filename for issue in issues}


def _canonical_preflight_pages(
    issue: ReportIssue, pages: Sequence[PageText]
) -> tuple[PageText, ...]:
    work = tuple(pages)
    if not all(isinstance(page, PageText) for page in work):
        raise ValueError(f"{issue.identity} extractor must return PageText rows")
    numbers = [page.pdf_page for page in work]
    expected = list(range(1, len(work) + 1))
    if numbers != expected:
        raise ValueError(
            f"{issue.identity} extractor pages are not a complete 1-based physical sequence"
        )
    if len(work) != issue.page_count:
        raise ValueError(
            f"{issue.identity} physical page count mismatch: "
            f"catalogue={issue.page_count}, extracted={len(work)}"
        )
    return work


def _preflight(
    issues: tuple[ReportIssue, ...],
    paths: Mapping[str, Path],
    extractor: PageExtractor,
) -> tuple[_PreparedIssue, ...]:
    prepared = []
    for issue in issues:
        path = paths[issue.identity]
        try:
            pdf_bytes = path.read_bytes()
        except OSError as exc:
            raise ValueError(f"could not read {issue.identity} artifact {path}: {exc}") from exc
        if not pdf_bytes.startswith(b"%PDF-"):
            raise ValueError(f"{issue.identity} artifact does not have a PDF magic header")
        digest = hashlib.sha256(pdf_bytes).hexdigest()
        if digest != issue.sha256:
            raise ValueError(
                f"{issue.identity} artifact sha256 mismatch: "
                f"expected {issue.sha256}, calculated {digest}"
            )
        pages = _canonical_preflight_pages(issue, extractor.extract(pdf_bytes))
        prepared.append(_PreparedIssue(issue=issue, pdf_bytes=pdf_bytes, pages=pages))
    return tuple(prepared)


def run_pipeline(
    *,
    catalogue_path: Path = DEFAULT_CATALOGUE_PATH,
    artifacts: Mapping[str, Path] | None = None,
    artifact_dir: Path | None = None,
    issue_ids: tuple[str, ...] | None = None,
    blob_root: Path = DEFAULT_BLOB_ROOT,
    engine: Engine | None = None,
    extractor: PageExtractor | None = None,
) -> dict[str, dict[str, object]]:
    """Preflight and ingest one or all local catalogue artifacts.

    Catalogue structure, selections, artifact mappings, all hashes, and every
    physical page count are checked before the database is initialized.
    """
    catalogue = load_issue_catalogue(Path(catalogue_path))
    selected = _select_issues(catalogue, issue_ids)
    paths = _artifact_paths(selected, artifacts=artifacts, artifact_dir=artifact_dir)
    page_extractor = extractor or PopplerPageExtractor()
    extractor_name = str(page_extractor.name).strip()
    extractor_version = str(page_extractor.version).strip()
    if not extractor_name or not extractor_version:
        raise ValueError("extractor name and version must not be empty")
    prepared = _preflight(selected, paths, page_extractor)

    db_engine = engine or make_engine()
    init_db(db_engine)
    session_factory = make_session_factory(db_engine)
    summary: dict[str, dict[str, object]] = {}
    for item in prepared:
        cached_extractor = _PreparedExtractor(
            name=extractor_name,
            version=extractor_version,
            content_sha256=item.issue.sha256,
            pages=item.pages,
        )
        with session_factory() as session:
            result = ingest_report(
                session,
                item.pdf_bytes,
                item.issue.to_report_meta(),
                extractor=cached_extractor,
                blob_root=Path(blob_root),
            )
        summary[item.issue.identity] = {
            "source_id": item.issue.source_id,
            "issue_key": item.issue.issue_key,
            "document_id": result.document_id,
            "extraction_id": result.extraction_id,
            "created": result.created,
            "extraction_created": result.extraction_created,
            "content_sha256": result.content_sha256,
            "page_count": result.page_count,
            "blob_path": str(result.blob_path),
        }
    return summary


def _artifact_mapping(values: Sequence[str], parser: argparse.ArgumentParser) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        identity, separator, raw_path = value.partition("=")
        if not separator or not identity.strip() or not raw_path.strip():
            parser.error("--artifact must use SOURCE_ID:ISSUE_KEY=PATH")
        if identity in result:
            parser.error(f"duplicate --artifact identity: {identity}")
        result[identity] = Path(raw_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ingest checked-in official report issues from local PDF artifacts."
    )
    parser.add_argument("--catalogue", type=Path, default=DEFAULT_CATALOGUE_PATH)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--artifact-dir", type=Path)
    source.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="SOURCE_ID:ISSUE_KEY=PATH",
    )
    parser.add_argument("--issue", action="append", dest="issue_ids")
    parser.add_argument("--db", type=Path, help="SQLite database path; defaults to DALIO_DB_PATH.")
    parser.add_argument("--blob-root", type=Path, default=DEFAULT_BLOB_ROOT)
    parser.add_argument("--pdftotext", default="pdftotext", help="Path to Poppler pdftotext.")
    args = parser.parse_args(argv)

    artifacts = _artifact_mapping(args.artifact, parser) if args.artifact else None
    try:
        summary = run_pipeline(
            catalogue_path=args.catalogue,
            artifacts=artifacts,
            artifact_dir=args.artifact_dir,
            issue_ids=tuple(args.issue_ids) if args.issue_ids else None,
            blob_root=args.blob_root,
            engine=make_engine(args.db) if args.db else None,
            extractor=PopplerPageExtractor(executable=args.pdftotext),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Official-report ingestion failed: {exc}", file=sys.stderr)
        return 1

    for identity, result in summary.items():
        action = "created" if result["created"] else "already stored"
        print(
            f"{identity}: {result['page_count']} pages; document {result['document_id']} ({action})"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
