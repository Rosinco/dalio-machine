"""Read-only publication boundary for official-report human-review packets."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import select

from dalio.pipelines.build_report_review_packet import (
    LATEST_JSON,
    LATEST_MARKDOWN,
    main,
    run,
)
from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import REPORT_REVIEW_METHODOLOGY_VERSION
from dalio.storage.db import (
    DocumentExtraction,
    ReportDocument,
    init_db,
    make_engine,
    make_session_factory,
)
from dalio.storage.reports import PageText, ReportMeta, ingest_report


class FakeExtractor:
    name = "fixture-pages"
    version = "1"

    def extract(self, _pdf_bytes: bytes) -> tuple[PageText, ...]:
        return (
            PageText(
                1,
                "Financial con-\nditions remain tight as inflation risks persist.",
                printed_page_label="1",
            ),
            PageText(2, "A second physical page with policy context.", "2"),
        )


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _fixture_inputs(tmp_path: Path):
    db_path = tmp_path / "reports.db"
    engine = make_engine(db_path)
    init_db(engine)
    factory = make_session_factory(engine)
    source = REPORT_SOURCES[0]
    pdf = b"%PDF-1.7\nreview packet fixture\n%%EOF\n"
    with factory() as session:
        result = ingest_report(
            session,
            pdf,
            ReportMeta(
                source_id=source.source_id,
                report_family=source.report_family,
                issue_key="2026-06",
                publisher=source.publisher,
                jurisdiction=source.jurisdiction,
                title="Monetary Policy Report June 2026",
                language=source.language,
                document_date=_at(2026, 6, 18).date(),
                published_at=None,
                available_at=_at(2026, 6, 18),
                retrieved_at=_at(2026, 9, 8),
                landing_url=source.landing_url,
                artifact_url="https://www.riksbank.se/report-2026-06.pdf",
                allowed_domains=source.official_domains,
                expected_sha256=hashlib.sha256(pdf).hexdigest(),
            ),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "archive",
        )
        document = session.get(ReportDocument, result.document_id)
        extraction = session.scalar(
            select(DocumentExtraction).where(DocumentExtraction.document_id == document.id)
        )
        candidate = {
            "source_id": document.source_id,
            "issue_key": document.issue_key,
            "document_sha256": document.content_sha256,
            "extraction_name": extraction.extractor_name,
            "extraction_version": extraction.extractor_version,
            "extraction_corpus_sha256": extraction.corpus_sha256,
            "claim_type": "judgment",
            "statement": "The publisher judges financial conditions to remain tight.",
            "topic_key": "financial_conditions",
            "geographies": ["SE"],
            "claim_series_key": None,
            "reference_start": None,
            "reference_end": None,
            "target_start": None,
            "target_end": None,
            "numeric_value": None,
            "lower_bound": None,
            "upper_bound": None,
            "unit": None,
            "condition_text": "As assessed in this report edition.",
            "horizon": "1-3y",
            "importance_rationale": "Model-proposed transmission constraint for review.",
            "citations": [
                {
                    "pdf_page_start": 1,
                    "pdf_page_end": 1,
                    "printed_locator": "p. 1",
                    "section_title": "Overview",
                    "evidence_excerpt": (
                        "Financial conditions remain tight as inflation risks persist."
                    ),
                    "support_role": "direct",
                }
            ],
        }
    engine.dispose()
    catalogue_path = tmp_path / "report_claim_candidates.json"
    catalogue_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "methodology_version": REPORT_REVIEW_METHODOLOGY_VERSION,
                "created_by": "model:fixture-v1",
                "created_at": "2099-01-02T12:00:00Z",
                "as_known_at": "2026-12-31T23:59:59Z",
                "candidates": [candidate],
            }
        ),
        encoding="utf-8",
    )
    return db_path, catalogue_path


def test_run_publishes_exact_latest_and_hash_addressed_pairs_without_db_writes(tmp_path):
    db_path, catalogue_path = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "review"
    before_db_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    packet, paths = run(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        require_all_sources=False,
    )

    after_db_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()
    hash16 = packet["packet_sha256"][:16]
    dated_stem = f"report_claims_2026-12-31_{hash16}"
    expected = (
        output_dir / f"{dated_stem}.json",
        output_dir / f"{dated_stem}.md",
        output_dir / LATEST_JSON,
        output_dir / LATEST_MARKDOWN,
    )
    assert paths == expected
    assert all(path.is_file() for path in paths)
    assert paths[0].read_bytes() == paths[2].read_bytes()
    assert paths[1].read_bytes() == paths[3].read_bytes()
    assert json.loads(paths[0].read_text(encoding="utf-8")) == packet
    assert str(tmp_path) not in paths[0].read_text(encoding="utf-8")
    assert str(tmp_path) not in paths[1].read_text(encoding="utf-8")
    assert before_db_sha == after_db_sha


def test_repeat_run_is_content_and_path_idempotent(tmp_path):
    db_path, catalogue_path = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "review"
    first_packet, first_paths = run(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        require_all_sources=False,
    )
    first_contents = {path: path.read_bytes() for path in first_paths}

    second_packet, second_paths = run(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        require_all_sources=False,
    )

    assert second_packet == first_packet
    assert second_paths == first_paths
    assert {path: path.read_bytes() for path in second_paths} == first_contents
    assert set(output_dir.iterdir()) == set(first_paths)


def test_default_run_requires_every_enabled_official_report_family(tmp_path):
    db_path, catalogue_path = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "review"

    with pytest.raises(RuntimeError, match="requires every enabled official family"):
        run(
            db_path=db_path,
            catalogue_path=catalogue_path,
            output_dir=output_dir,
        )

    assert not output_dir.exists()


def test_run_refuses_different_bytes_at_hash_addressed_name_before_moving_aliases(tmp_path):
    db_path, catalogue_path = _fixture_inputs(tmp_path)
    output_dir = tmp_path / "review"
    _packet, paths = run(
        db_path=db_path,
        catalogue_path=catalogue_path,
        output_dir=output_dir,
        require_all_sources=False,
    )
    latest_json_before = paths[2].read_bytes()
    latest_markdown_before = paths[3].read_bytes()
    paths[0].write_text("different bytes\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="refusing to overwrite different hash-addressed"):
        run(
            db_path=db_path,
            catalogue_path=catalogue_path,
            output_dir=output_dir,
            require_all_sources=False,
        )

    assert paths[2].read_bytes() == latest_json_before
    assert paths[3].read_bytes() == latest_markdown_before


def test_cli_returns_error_for_missing_database(tmp_path, capsys):
    _db_path, catalogue_path = _fixture_inputs(tmp_path)
    missing_db = tmp_path / "missing.db"

    result = main(
        [
            "--db",
            str(missing_db),
            "--catalogue",
            str(catalogue_path),
            "--out-dir",
            str(tmp_path / "review"),
        ]
    )

    captured = capsys.readouterr()
    assert result == 2
    assert "database does not exist" in captured.err
    assert str(missing_db.resolve()) in captured.err
    assert not (tmp_path / "review").exists()
