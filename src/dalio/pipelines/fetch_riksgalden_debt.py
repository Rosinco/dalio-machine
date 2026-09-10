"""Fetch nine original Riksgälden documents or replay their verified evidence offline."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import requests
from sqlalchemy import create_engine

from dalio.data_sources.riksgalden_debt import (
    FUNDING_PLAN_2026_1,
    RIKSGALDEN_DOCUMENT_SPECS,
    parse_funding_workbook,
    parse_monthly_report,
)
from dalio.data_sources.sdmx_csv import DEFAULT_USER_AGENT
from dalio.storage.db import make_engine
from dalio.storage.national_debt import (
    PreparedNativePartition,
    ingest_native_batch,
    load_native_batch,
    prepare_native_batch,
)

logger = logging.getLogger(__name__)
DEFAULT_EVIDENCE_ROOT = Path("data/artifacts/debt_refinancing/riksgalden")
MAX_DOCUMENT_BYTES = 10_000_000


def _check_complete_batch(batch) -> tuple[PreparedNativePartition, ...]:
    batch = tuple(batch)
    expected = {
        (
            spec.stream_id,
            spec.snapshot_key,
            "SE",
            spec.source_url,
            spec.published_at,
            spec.reference_date,
        )
        for spec in RIKSGALDEN_DOCUMENT_SPECS
    }
    actual = {
        (
            item.document.stream_id,
            item.document.snapshot_key,
            item.document.country,
            item.document.source_url,
            item.document.published_at,
            item.document.reference_date,
        )
        for item in batch
    }
    if len(batch) != len(expected) or actual != expected:
        raise ValueError("Riksgalden refresh requires all nine exact original document identities")
    return batch


def prepare_batch(
    *,
    client=None,
    artifact_root: Path = DEFAULT_EVIDENCE_ROOT,
    retrieved_at: datetime | None = None,
) -> tuple[PreparedNativePartition, ...]:
    """Acquire and preflight every document before opening any target database."""
    owned_client = client is None
    client = requests.Session() if client is None else client
    documents = []
    try:
        for position, spec in enumerate(RIKSGALDEN_DOCUMENT_SPECS, 1):
            logger.info(
                "Fetching original Riksgalden document %s/9: %s", position, spec.snapshot_key
            )
            response = client.get(
                spec.source_url,
                timeout=(10, 45),
                allow_redirects=False,
                headers={"User-Agent": DEFAULT_USER_AGENT},
            )
            actual_url = urlsplit(response.url)
            if (
                response.status_code != 200
                or response.url != spec.source_url
                or actual_url.scheme != "https"
                or actual_url.netloc != "www.riksgalden.se"
                or actual_url.username
                or actual_url.password
            ):
                raise ValueError("Original Riksgalden response failed status/URL/redirect checks")
            is_pdf = spec != FUNDING_PLAN_2026_1
            allowed_types = {"application/octet-stream"}
            allowed_types.add(
                "application/pdf"
                if is_pdf
                else "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
            content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
            if content_type not in allowed_types:
                raise ValueError("Unexpected original Riksgalden response content type")
            body = response.content
            if (
                not isinstance(body, bytes)
                or not body
                or len(body) > MAX_DOCUMENT_BYTES
                or not body.startswith(b"%PDF-" if is_pdf else b"PK\x03\x04")
            ):
                raise ValueError("Unexpected or oversized original Riksgalden document bytes")
            parser = parse_monthly_report if is_pdf else parse_funding_workbook
            documents.append(parser(body, spec=spec))
        # The receipt clock follows the complete acquisition, never the period or
        # a publication timestamp. Offline replay retains this original clock.
        receipt = retrieved_at or datetime.now(UTC)
        batch = prepare_native_batch(documents, artifact_root=artifact_root, retrieved_at=receipt)
        return _check_complete_batch(batch)
    finally:
        if owned_client:
            client.close()


def _report_allowed(
    report_path: Path | None, *, db_path: Path, source_path: Path | None = None, batch=()
) -> None:
    if report_path is None:
        return
    protected = {db_path.resolve()}
    if source_path is not None:
        protected.add(source_path.resolve())
    protected.update(
        Path(artifact.artifact_path).resolve() for item in batch for artifact in item.artifacts
    )
    if report_path.resolve() in protected:
        raise ValueError("Receipt report cannot overwrite a database or source evidence artifact")


def _receipt(batch, result: dict) -> dict:
    return {
        **result,
        "schema_version": "riksgalden-acquisition-receipt-v1",
        "report_generated_at": datetime.now(UTC).isoformat(),
        "observed_facts": sum(
            fact["status"] == "observed" for item in batch for fact in item.document.facts
        ),
        "forecast_facts": sum(
            fact["status"] == "forecast" for item in batch for fact in item.document.facts
        ),
        "missing_facts": sum(
            fact["status"] == "not_reported" for item in batch for fact in item.document.facts
        ),
        "sources": [
            {
                "stream_id": item.document.stream_id,
                "snapshot_key": item.document.snapshot_key,
                "source_url": item.document.source_url,
                "reference_date": item.document.reference_date.isoformat(),
                "published_at": item.document.published_at.isoformat(),
                "retrieved_at": item.retrieved_at.isoformat(),
                "fact_count": len(item.document.facts),
                "source_sha256": hashlib.sha256(item.document.source_bytes).hexdigest(),
            }
            for item in batch
        ],
    }


def _publish(batch, *, db_path: Path, report_path: Path | None = None) -> dict:
    batch = _check_complete_batch(batch)
    _report_allowed(report_path, db_path=db_path, batch=batch)
    engine = make_engine(db_path)
    try:
        result = ingest_native_batch(batch, engine=engine)
    finally:
        engine.dispose()
    receipt = _receipt(batch, result)
    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            dir=report_path.parent,
            prefix=".riksgalden-receipt-",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(receipt, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
        try:
            temporary_path.replace(report_path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return receipt


def run_pipeline(
    *,
    db_path: Path,
    report_path: Path | None = None,
    **kwargs,
) -> dict:
    _report_allowed(report_path, db_path=db_path)
    batch = prepare_batch(**kwargs)
    return _publish(batch, db_path=db_path, report_path=report_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True, help="Explicit target SQLite database")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_EVIDENCE_ROOT)
    parser.add_argument(
        "--from-db", type=Path, help="Replay all nine verified staging documents offline"
    )
    parser.add_argument(
        "--report", type=Path, help="Write a JSON acquisition and ingestion receipt"
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        _report_allowed(args.report, db_path=args.db, source_path=args.from_db)
        if args.from_db is not None:
            if not args.from_db.is_file():
                raise ValueError(f"Staging database does not exist: {args.from_db}")
            source_engine = create_engine(
                "sqlite://",
                creator=lambda: sqlite3.connect(
                    args.from_db.resolve().as_uri() + "?mode=ro", uri=True
                ),
            )
            try:
                batch = load_native_batch(source_engine)
            finally:
                source_engine.dispose()
            result = _publish(batch, db_path=args.db, report_path=args.report)
        else:
            result = run_pipeline(
                db_path=args.db, artifact_root=args.evidence_root, report_path=args.report
            )
    except Exception as exc:  # noqa: BLE001 - explicit fail-closed acquisition CLI
        logger.exception("Riksgalden batch failed: %s", exc)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
