"""Unverified report candidates become a deterministic, read-only human review packet."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from sqlalchemy import func, select

from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import (
    MAX_REVIEW_CANDIDATES_PER_DOCUMENT,
    REPORT_REVIEW_METHODOLOGY_VERSION,
    UNVERIFIED_DRAFT_LABEL,
    build_review_packet,
    candidate_fingerprint,
    load_candidate_catalogue,
    render_review_markdown,
    resolve_candidate_evidence,
    validate_candidate_catalogue,
)
from dalio.storage.db import (
    Claim,
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

    def __init__(self, pages: tuple[PageText, ...]) -> None:
        self.pages = pages

    def extract(self, _pdf_bytes: bytes) -> tuple[PageText, ...]:
        return self.pages


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _source(source_id: str):
    return next(spec for spec in REPORT_SOURCES if spec.source_id == source_id)


def _ingest(
    session,
    tmp_path: Path,
    *,
    source_id: str = "riksbank_mpr_en",
    issue_key: str = "2026-06",
    available_at: datetime | None = None,
    retrieved_at: datetime | None = None,
    page_text: str = "Financial con-\nditions remain tight as inflation risks persist.",
):
    source = _source(source_id)
    public_at = available_at or _at(2026, 6, 18)
    retrieved = retrieved_at or _at(2026, 9, 8)
    pdf = f"%PDF-1.7\n{source_id}:{issue_key}\n%%EOF\n".encode()
    return ingest_report(
        session,
        pdf,
        ReportMeta(
            source_id=source.source_id,
            report_family=source.report_family,
            issue_key=issue_key,
            publisher=source.publisher,
            jurisdiction=source.jurisdiction,
            title=f"{source.report_family} {issue_key}",
            language=source.language,
            document_date=public_at.date(),
            published_at=None,
            available_at=public_at,
            retrieved_at=retrieved,
            landing_url=source.landing_url,
            artifact_url=f"https://www.{source.official_domains[0]}/{issue_key}.pdf",
            allowed_domains=source.official_domains,
            expected_sha256=hashlib.sha256(pdf).hexdigest(),
        ),
        extractor=FakeExtractor(
            (
                PageText(1, page_text, printed_page_label="1"),
                PageText(2, "A second physical page with policy context.", "2"),
            )
        ),
        blob_root=tmp_path / "archive",
    )


def _candidate_payload(
    session,
    document_id: int,
    *,
    statement: str = "The publisher judges financial conditions to remain tight.",
    topic_key: str = "financial_conditions",
    excerpt: str = "Financial conditions remain tight as inflation risks persist.",
) -> dict[str, object]:
    document = session.get(ReportDocument, document_id)
    extraction = session.scalar(
        select(DocumentExtraction).where(DocumentExtraction.document_id == document_id)
    )
    return {
        "source_id": document.source_id,
        "issue_key": document.issue_key,
        "document_sha256": document.content_sha256,
        "extraction_name": extraction.extractor_name,
        "extraction_version": extraction.extractor_version,
        "extraction_corpus_sha256": extraction.corpus_sha256,
        "claim_type": "judgment",
        "statement": statement,
        "topic_key": topic_key,
        "geographies": [document.jurisdiction],
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
        "importance_rationale": (
            "Model-proposed monetary transmission constraint for human review."
        ),
        "citations": [
            {
                "pdf_page_start": 1,
                "pdf_page_end": 1,
                "printed_locator": "p. 1",
                "section_title": "Overview",
                "evidence_excerpt": excerpt,
                "support_role": "direct",
            }
        ],
    }


def _catalogue_payload(candidates: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "methodology_version": REPORT_REVIEW_METHODOLOGY_VERSION,
        "created_by": "model:fixture-v1",
        "created_at": "2099-01-02T12:00:00Z",
        "as_known_at": "2026-12-31T23:59:59Z",
        "candidates": candidates,
    }


def _write_catalogue(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "candidates.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


@pytest.fixture
def report_store(tmp_path):
    db_path = tmp_path / "reports.db"
    engine = make_engine(db_path)
    init_db(engine)
    factory = make_session_factory(engine)
    return db_path, factory


def test_load_candidate_catalogue_is_schema_exact_and_fingerprints_content(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        raw_candidate = _candidate_payload(session, report.document_id)
    catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload([raw_candidate]))
    )

    assert catalogue.schema_version == 1
    assert catalogue.created_by == "model:fixture-v1"
    assert catalogue.created_at.tzinfo is UTC
    assert catalogue.as_known_at.tzinfo is UTC
    assert len(catalogue.candidates) == 1
    digest = candidate_fingerprint(catalogue.candidates[0])
    assert len(digest) == 64
    assert digest == candidate_fingerprint(catalogue.candidates[0])


@pytest.mark.parametrize("forbidden", ["status", "approval", "score", "probability", "action"])
def test_catalogue_rejects_forbidden_or_unknown_candidate_fields(report_store, tmp_path, forbidden):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        candidate = _candidate_payload(session, report.document_id)
    candidate[forbidden] = None

    with pytest.raises(ValueError, match="unknown fields"):
        load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([candidate])))


def test_catalogue_rejects_inference_human_creator_and_invalid_numeric_contract(
    report_store, tmp_path
):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        base = _candidate_payload(session, report.document_id)

    inference = {**base, "claim_type": "inference"}
    with pytest.raises(ValueError, match="publisher claim"):
        load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([inference])))

    human_payload = _catalogue_payload([base])
    human_payload["created_by"] = "human:analyst"
    with pytest.raises(ValueError, match="model:<id/version>"):
        load_candidate_catalogue(_write_catalogue(tmp_path, human_payload))

    invalid_number = {**base, "numeric_value": 2.0, "unit": None}
    with pytest.raises(ValueError, match="unit"):
        load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([invalid_number])))


def test_validate_resolves_latest_hash_bound_evidence_and_build_is_read_only(
    report_store, tmp_path
):
    db_path, factory = report_store
    with factory() as session:
        _ingest(
            session,
            tmp_path,
            issue_key="2026-03",
            available_at=_at(2026, 3, 19),
        )
        latest = _ingest(session, tmp_path, issue_key="2026-06")
        imf = _ingest(
            session,
            tmp_path,
            source_id="imf_weo_en",
            issue_key="2026-07",
            available_at=_at(2026, 7, 8),
            page_text="Global growth is projected to slow before recovering in 2027.",
        )
        candidates = [
            _candidate_payload(session, latest.document_id),
            _candidate_payload(
                session,
                imf.document_id,
                statement="The IMF projects global growth to slow before recovering.",
                topic_key="growth",
                excerpt="Global growth is projected to slow before recovering in 2027.",
            ),
        ]
    catalogue = load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload(candidates)))
    before_sha = hashlib.sha256(db_path.read_bytes()).hexdigest()

    with factory() as session:
        before_claims = session.scalar(select(func.count()).select_from(Claim))
        resolved = validate_candidate_catalogue(session, catalogue)
        packet = build_review_packet(session, catalogue)
        repeated = build_review_packet(session, catalogue)
        after_claims = session.scalar(select(func.count()).select_from(Claim))
        assert not session.new and not session.dirty and not session.deleted

    assert [item.candidate.source_id for item in resolved] == [
        "riksbank_mpr_en",
        "imf_weo_en",
    ]
    assert before_claims == after_claims == 0
    assert hashlib.sha256(db_path.read_bytes()).hexdigest() == before_sha
    assert packet == repeated
    assert packet["packet_sha256"] == repeated["packet_sha256"]
    assert packet["candidate_count"] == 2
    assert packet["document_count"] == 2
    assert "conditions remain tight" in json.dumps(packet)
    serialized = json.dumps(packet)
    assert str(tmp_path) not in serialized
    for document in packet["documents"]:
        digest = document["content_sha256"]
        assert document["archive_reference"] == (f"reports/sha256/{digest[:2]}/{digest}.pdf")
    assert all(
        candidate["draft_label"] == UNVERIFIED_DRAFT_LABEL
        for document in packet["documents"]
        for candidate in document["candidates"]
    )

    def keys(value):
        if isinstance(value, dict):
            yield from value
            for nested in value.values():
                yield from keys(nested)
        elif isinstance(value, list):
            for nested in value:
                yield from keys(nested)

    assert {"status", "approval", "score", "probability", "action"}.isdisjoint(keys(packet))


def test_validation_rejects_prior_issue_and_missing_latest_source(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        old = _ingest(
            session,
            tmp_path,
            issue_key="2026-03",
            available_at=_at(2026, 3, 19),
        )
        latest = _ingest(session, tmp_path, issue_key="2026-06")
        imf = _ingest(
            session,
            tmp_path,
            source_id="imf_weo_en",
            issue_key="2026-07",
            available_at=_at(2026, 7, 8),
            page_text="Global growth is projected to slow before recovering in 2027.",
        )
        old_raw = _candidate_payload(session, old.document_id)
        latest_raw = _candidate_payload(session, latest.document_id)
        imf_raw = _candidate_payload(
            session,
            imf.document_id,
            statement="The IMF projects global growth to slow before recovering.",
            topic_key="growth",
            excerpt="Global growth is projected to slow before recovering in 2027.",
        )

    prior_catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload([old_raw, imf_raw]))
    )
    with factory() as session, pytest.raises(ValueError, match="latest eligible"):
        validate_candidate_catalogue(session, prior_catalogue)

    missing_catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload([latest_raw]))
    )
    with factory() as session, pytest.raises(ValueError, match="missing candidate source"):
        validate_candidate_catalogue(session, missing_catalogue)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"document_sha256": "0" * 64}, "latest eligible"),
        ({"extraction_name": "unknown"}, "extraction"),
        ({"extraction_corpus_sha256": "0" * 64}, "extraction"),
        ({"topic_key": "sovereign_debt"}, "topic"),
    ],
)
def test_resolution_rejects_provenance_and_policy_mismatches(
    report_store, tmp_path, change, message
):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        raw = {**_candidate_payload(session, report.document_id), **change}
    catalogue = load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([raw])))

    with factory() as session, pytest.raises(ValueError, match=message):
        validate_candidate_catalogue(session, catalogue)


def test_resolution_rejects_wrong_page_excerpt_and_tampered_blob(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        base = _candidate_payload(session, report.document_id)

    wrong_page = json.loads(json.dumps(base))
    wrong_page["citations"][0]["pdf_page_start"] = 3
    wrong_page["citations"][0]["pdf_page_end"] = 3
    catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload([wrong_page]))
    )
    with factory() as session, pytest.raises(ValueError, match="outside|page range"):
        validate_candidate_catalogue(session, catalogue)

    wrong_excerpt = json.loads(json.dumps(base))
    wrong_excerpt["citations"][0]["evidence_excerpt"] = "Words absent from the report."
    catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload([wrong_excerpt]))
    )
    with factory() as session, pytest.raises(ValueError, match="excerpt"):
        validate_candidate_catalogue(session, catalogue)

    with factory() as session:
        document = session.get(ReportDocument, report.document_id)
        blob_path = Path(document.blob_path)
    blob_path.write_bytes(b"%PDF-1.7\ntampered\n%%EOF\n")
    catalogue = load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([base])))
    with factory() as session, pytest.raises(ValueError, match="archived PDF"):
        resolve_candidate_evidence(session, catalogue.candidates[0], catalogue=catalogue)


def test_validation_enforces_bounded_candidates_per_document(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        base = _candidate_payload(session, report.document_id)
    raw_candidates = [
        {**base, "statement": f"Distinct model-proposed statement number {number}."}
        for number in range(MAX_REVIEW_CANDIDATES_PER_DOCUMENT + 1)
    ]
    catalogue = load_candidate_catalogue(
        _write_catalogue(tmp_path, _catalogue_payload(raw_candidates))
    )

    with factory() as session, pytest.raises(ValueError, match="at most"):
        validate_candidate_catalogue(session, catalogue)


def test_cutoff_selects_the_latest_public_issue_not_a_future_document(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        current = _ingest(session, tmp_path, issue_key="2026-06")
        _ingest(
            session,
            tmp_path,
            issue_key="2027-01",
            available_at=_at(2027, 1, 10),
            retrieved_at=_at(2027, 1, 11),
        )
        raw = _candidate_payload(session, current.document_id)
    catalogue = load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([raw])))

    with factory() as session:
        resolved = validate_candidate_catalogue(session, catalogue)

    assert resolved[0].document.issue_key == "2026-06"
    assert resolved[0].document.available_at == _at(2026, 6, 18)


def test_markdown_is_an_unverified_review_aid_with_original_evidence(report_store, tmp_path):
    _db_path, factory = report_store
    with factory() as session:
        report = _ingest(session, tmp_path)
        raw = _candidate_payload(session, report.document_id)
    catalogue = load_candidate_catalogue(_write_catalogue(tmp_path, _catalogue_payload([raw])))
    with factory() as session:
        markdown = render_review_markdown(build_review_packet(session, catalogue))

    assert "UNVERIFIED MODEL DRAFTS" in markdown
    assert f"### {UNVERIFIED_DRAFT_LABEL} —" in markdown
    assert "The publisher judges financial conditions" in markdown
    assert "PDF page 1" in markdown
    assert "Financial conditions remain tight" in markdown
    assert "Model selection rationale" in markdown
    assert "- [ ] Attribution is fair" in markdown
    assert "Outcome: approve / revise / reject" in markdown
    assert "fixture-pages 1" in markdown
    assert "Archived PDF reference: `reports/sha256/" in markdown
    assert str(tmp_path) not in markdown
