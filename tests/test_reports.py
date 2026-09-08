"""Immutable report evidence, page citations, and point-in-time claim queries."""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from dalio.reports.manifest import REPORT_SOURCES
from dalio.storage.db import (
    Claim,
    ClaimCitation,
    DocumentExtraction,
    DocumentPage,
    ReportDocument,
    init_db,
    make_engine,
)
from dalio.storage.reports import (
    CitationDraft,
    ClaimDraft,
    PageText,
    ReportMeta,
    ingest_report,
    load_claims,
    propose_claims,
    verify_claim,
)


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=UTC)


def _pdf(label: str = "one") -> bytes:
    return f"%PDF-1.7\nfixture {label}\n%%EOF\n".encode()


class FakeExtractor:
    name = "fixture"
    version = "1"

    def __init__(self, pages: tuple[PageText, ...] | None = None):
        self.pages = pages or (
            PageText(1, "Inflation was 2.0 percent in 2025."),
            PageText(2, "Growth is ex-\npected to reach 1.5 percent in 2027.", "7"),
        )

    def extract(self, _pdf_bytes: bytes) -> tuple[PageText, ...]:
        return self.pages


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "reports.db")
    init_db(engine)
    from sqlalchemy.orm import sessionmaker

    return sessionmaker(bind=engine, expire_on_commit=False)


def _meta(
    *,
    source_id: str = "riksbank_mpr_en",
    publisher: str | None = None,
    issue_key: str = "2026-06",
    published_at: datetime | None = None,
    available_at: datetime | None = None,
    retrieved_at: datetime | None = None,
    expected_sha256: str | None = None,
) -> ReportMeta:
    available = available_at or _at(2026, 6, 18)
    source = next(spec for spec in REPORT_SOURCES if spec.source_id == source_id)
    artifact_host = source.official_domains[0]
    return ReportMeta(
        source_id=source_id,
        report_family=source.report_family,
        issue_key=issue_key,
        publisher=publisher or source.publisher,
        jurisdiction=source.jurisdiction,
        title=f"Monetary Policy Report {issue_key}",
        language=source.language,
        document_date=date.fromisoformat(f"{issue_key}-01"),
        published_at=published_at,
        available_at=available,
        retrieved_at=retrieved_at or available,
        landing_url=source.landing_url,
        artifact_url=f"https://www.{artifact_host}/reports/{issue_key}.pdf",
        allowed_domains=source.official_domains,
        expected_sha256=expected_sha256,
    )


def _ingest(session: Session, tmp_path, **meta_kwargs):
    return ingest_report(
        session,
        _pdf(meta_kwargs.get("issue_key", "one")),
        _meta(**meta_kwargs),
        extractor=FakeExtractor(),
        blob_root=tmp_path / "blobs",
    )


def test_ingest_report_archives_hash_pages_and_provenance(session_factory, tmp_path):
    pdf = _pdf()
    expected = hashlib.sha256(pdf).hexdigest()
    with session_factory() as session:
        result = ingest_report(
            session,
            pdf,
            _meta(expected_sha256=expected),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        document = session.get(ReportDocument, result.document_id)
        extraction = session.get(DocumentExtraction, result.extraction_id)
        pages = (
            session.execute(
                select(DocumentPage)
                .where(DocumentPage.extraction_id == result.extraction_id)
                .order_by(DocumentPage.pdf_page)
            )
            .scalars()
            .all()
        )

    assert result.created is True
    assert result.content_sha256 == expected
    assert result.blob_path == tmp_path / "blobs" / "sha256" / expected[:2] / f"{expected}.pdf"
    assert result.blob_path.read_bytes() == pdf
    assert document.content_sha256 == expected and document.page_count == 2
    assert extraction.status == "complete" and extraction.extracted_page_count == 2
    assert [(page.pdf_page, page.printed_page_label) for page in pages] == [(1, None), (2, "7")]


@pytest.mark.parametrize(
    ("pdf", "meta", "extractor", "match"),
    [
        (b"not a pdf", _meta(), FakeExtractor(), "PDF"),
        (
            _pdf(),
            _meta(expected_sha256="0" * 64),
            FakeExtractor(),
            "sha256",
        ),
        (
            _pdf(),
            _meta(retrieved_at=_at(2026, 6, 17)),
            FakeExtractor(),
            "retrieved_at",
        ),
        (
            _pdf(),
            _meta(published_at=_at(2026, 6, 19)),
            FakeExtractor(),
            "published_at",
        ),
        (
            _pdf(),
            ReportMeta(
                **{
                    **_meta().__dict__,
                    "landing_url": "http://www.riksbank.se/monetary-policy/",
                }
            ),
            FakeExtractor(),
            "HTTPS",
        ),
        (
            _pdf(),
            ReportMeta(
                **{
                    **_meta().__dict__,
                    "artifact_url": "https://evil.example/report.pdf",
                }
            ),
            FakeExtractor(),
            "allowlisted",
        ),
        (
            _pdf(),
            _meta(),
            FakeExtractor((PageText(1, "one"), PageText(3, "three"))),
            "complete 1-based",
        ),
    ],
)
def test_ingest_rejects_unsafe_or_incomplete_evidence_without_rows(
    session_factory, tmp_path, pdf, meta, extractor, match
):
    with session_factory() as session:
        with pytest.raises(ValueError, match=match):
            ingest_report(session, pdf, meta, extractor=extractor, blob_root=tmp_path / "blobs")
        assert session.scalar(select(func.count()).select_from(ReportDocument)) == 0
        assert session.scalar(select(func.count()).select_from(DocumentExtraction)) == 0


def test_report_ingest_is_idempotent_and_corrections_append_superseding_version(
    session_factory, tmp_path
):
    with session_factory() as session:
        first = ingest_report(
            session,
            _pdf("original"),
            _meta(retrieved_at=_at(2026, 6, 19)),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        again = ingest_report(
            session,
            _pdf("original"),
            _meta(retrieved_at=_at(2026, 6, 20)),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        corrected = ingest_report(
            session,
            _pdf("corrected"),
            _meta(available_at=_at(2026, 6, 21), retrieved_at=_at(2026, 6, 21)),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        documents = (
            session.execute(select(ReportDocument).order_by(ReportDocument.id)).scalars().all()
        )

    assert again.created is False and again.document_id == first.document_id
    assert corrected.created is True and corrected.document_id != first.document_id
    assert corrected.supersedes_document_id == first.document_id
    assert len(documents) == 2
    assert documents[0].supersedes_document_id is None


def test_manifest_identity_and_same_byte_metadata_are_authoritative(session_factory, tmp_path):
    pdf = _pdf("identity")
    with session_factory() as session:
        first = ingest_report(
            session,
            pdf,
            _meta(),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        conflicting_title = ReportMeta(
            **{
                **_meta(retrieved_at=_at(2026, 6, 20)).__dict__,
                "title": "A conflicting title for identical bytes",
            }
        )
        with pytest.raises(ValueError, match="conflicting immutable metadata"):
            ingest_report(
                session,
                pdf,
                conflicting_title,
                extractor=FakeExtractor(),
                blob_root=tmp_path / "blobs",
            )
        wrong_publisher = ReportMeta(
            **{
                **_meta(issue_key="2026-07").__dict__,
                "publisher": "Not the Riksbank",
            }
        )
        with pytest.raises(ValueError, match="publisher conflicts"):
            ingest_report(
                session,
                _pdf("publisher"),
                wrong_publisher,
                extractor=FakeExtractor(),
                blob_root=tmp_path / "blobs",
            )
        document_count = session.scalar(select(func.count()).select_from(ReportDocument))

    assert first.created is True
    assert document_count == 1


def test_existing_document_uses_stored_blob_and_versions_extractions(session_factory, tmp_path):
    pdf = _pdf("same")
    original_root = tmp_path / "original"
    unused_root = tmp_path / "must-not-be-used"
    with session_factory() as session:
        first = ingest_report(
            session,
            pdf,
            _meta(),
            extractor=FakeExtractor(),
            blob_root=original_root,
        )

        class ExtractorV2(FakeExtractor):
            version = "2"

        second_extraction = ingest_report(
            session,
            pdf,
            _meta(retrieved_at=_at(2026, 6, 20)),
            extractor=ExtractorV2(),
            blob_root=unused_root,
        )
        repeated = ingest_report(
            session,
            pdf,
            _meta(retrieved_at=_at(2026, 6, 21)),
            extractor=ExtractorV2(),
            blob_root=unused_root,
        )
        extraction_count = session.scalar(select(func.count()).select_from(DocumentExtraction))

        with pytest.raises(ValueError, match="same extractor name/version"):
            ingest_report(
                session,
                pdf,
                _meta(retrieved_at=_at(2026, 6, 22)),
                extractor=FakeExtractor(
                    (
                        PageText(1, "Drifted text."),
                        PageText(2, "Growth is expected to reach 1.5 percent in 2027."),
                    )
                ),
                blob_root=unused_root,
            )

    assert first.created is True and first.extraction_created is True
    assert second_extraction.created is False and second_extraction.extraction_created is True
    assert second_extraction.extraction_id != first.extraction_id
    assert repeated.extraction_id == second_extraction.extraction_id
    assert repeated.extraction_created is False
    assert repeated.blob_path == first.blob_path
    assert extraction_count == 2
    assert not unused_root.exists()


def test_existing_document_fails_loud_if_its_archived_blob_is_missing(session_factory, tmp_path):
    pdf = _pdf("missing")
    with session_factory() as session:
        report = ingest_report(
            session,
            pdf,
            _meta(),
            extractor=FakeExtractor(),
            blob_root=tmp_path / "blobs",
        )
        report.blob_path.unlink()
        with pytest.raises(ValueError, match="blob is missing"):
            ingest_report(
                session,
                pdf,
                _meta(retrieved_at=_at(2026, 6, 20)),
                extractor=FakeExtractor(),
                blob_root=tmp_path / "other",
            )
        with pytest.raises(ValueError, match="blob is missing"):
            propose_claims(
                session,
                [
                    ClaimDraft(
                        claim_type="fact",
                        statement="Inflation was 2.0 percent.",
                        attribution_document_id=report.document_id,
                        topic_key="inflation",
                        geographies=("SE",),
                        citations=(
                            CitationDraft(
                                report.extraction_id,
                                1,
                                1,
                                "Inflation was 2.0 percent",
                                "direct",
                            ),
                        ),
                    )
                ],
                created_by="human:analyst",
                created_at=_at(2026, 6, 20),
            )


def test_new_blob_is_cleaned_up_if_database_commit_fails(session_factory, tmp_path, monkeypatch):
    pdf = _pdf("commit-failure")
    digest = hashlib.sha256(pdf).hexdigest()
    blob_path = tmp_path / "blobs" / "sha256" / digest[:2] / f"{digest}.pdf"
    with session_factory() as session:

        def fail_commit():
            raise RuntimeError("forced commit failure")

        monkeypatch.setattr(session, "commit", fail_commit)
        with pytest.raises(RuntimeError, match="forced commit failure"):
            ingest_report(
                session,
                pdf,
                _meta(),
                extractor=FakeExtractor(),
                blob_root=tmp_path / "blobs",
            )
        assert session.scalar(select(func.count()).select_from(ReportDocument)) == 0
    assert not blob_path.exists()


def test_claim_excerpt_matching_normalizes_whitespace_and_line_hyphenation(
    session_factory, tmp_path
):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        claim_id = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="forecast",
                    statement="Growth is expected to reach 1.5 percent in 2027.",
                    attribution_document_id=report.document_id,
                    topic_key="growth",
                    geographies=("SE",),
                    target_end=date(2027, 12, 31),
                    citations=(
                        CitationDraft(
                            report.extraction_id,
                            2,
                            2,
                            "Growth is expected to reach 1.5 percent in 2027.",
                            "direct",
                        ),
                    ),
                )
            ],
            created_by="model:test",
            created_at=_at(2026, 6, 20),
        )[0]
        citation = session.scalar(select(ClaimCitation).where(ClaimCitation.claim_id == claim_id))
        locator_verified = citation.locator_verified_at is not None
        semantic_reviewer = citation.semantic_verified_by

        with pytest.raises(ValueError, match="excerpt"):
            propose_claims(
                session,
                [
                    ClaimDraft(
                        claim_type="fact",
                        statement="A number not in the report.",
                        attribution_document_id=report.document_id,
                        topic_key="inflation",
                        geographies=("SE",),
                        citations=(
                            CitationDraft(
                                report.extraction_id, 1, 1, "Inflation was 9 percent.", "direct"
                            ),
                        ),
                    )
                ],
                created_by="human:analyst",
                created_at=_at(2026, 6, 20),
            )

    assert locator_verified is True
    assert semantic_reviewer is None


def test_verification_appends_reviewed_publisher_claim_without_mutating_draft(
    session_factory, tmp_path
):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        draft_id = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="fact",
                    statement="Inflation was 2.0 percent in 2025.",
                    attribution_document_id=report.document_id,
                    topic_key="inflation",
                    geographies=("SE",),
                    reference_end=date(2025, 12, 31),
                    numeric_value=2.0,
                    unit="percent",
                    citations=(
                        CitationDraft(
                            report.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent in 2025.",
                            "direct",
                        ),
                    ),
                )
            ],
            created_by="model:extractor-v1",
            created_at=_at(2026, 6, 20),
        )[0]
        verified_id = verify_claim(
            session, draft_id, reviewer="human:adam", reviewed_at=_at(2026, 6, 21)
        )
        draft = session.get(Claim, draft_id)
        verified = session.get(Claim, verified_id)
        citations = (
            session.execute(select(ClaimCitation).order_by(ClaimCitation.claim_id)).scalars().all()
        )

    assert verified_id != draft_id
    assert draft.status == "draft" and draft.reviewed_by is None
    assert verified.status == "verified" and verified.review_of_claim_id == draft_id
    assert verified.supersedes_claim_id is None
    assert verified.reviewed_by == "human:adam"
    assert citations[0].semantic_verified_by is None
    assert citations[1].semantic_verified_by == "human:adam"


def test_model_cannot_self_review_or_act_as_semantic_reviewer(session_factory, tmp_path):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        draft_id = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="fact",
                    statement="Inflation was 2.0 percent.",
                    attribution_document_id=report.document_id,
                    topic_key="inflation",
                    geographies=("SE",),
                    citations=(
                        CitationDraft(
                            report.extraction_id, 1, 1, "Inflation was 2.0 percent", "direct"
                        ),
                    ),
                )
            ],
            created_by="model:test",
            created_at=_at(2026, 6, 20),
        )[0]
        with pytest.raises(ValueError, match="human:<id>"):
            verify_claim(session, draft_id, reviewer="model:test", reviewed_at=_at(2026, 6, 21))
        assert session.scalar(select(func.count()).select_from(Claim)) == 1


def test_claim_types_and_forecast_horizon_are_strict(session_factory, tmp_path):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        base = dict(
            statement="Inflation was 2.0 percent.",
            attribution_document_id=report.document_id,
            topic_key="inflation",
            geographies=("SE",),
            citations=(
                CitationDraft(report.extraction_id, 1, 1, "Inflation was 2.0 percent", "direct"),
            ),
        )
        for invalid_type in ("opinion", "FACT"):
            with pytest.raises(ValueError, match="claim_type"):
                propose_claims(
                    session,
                    [ClaimDraft(claim_type=invalid_type, **base)],
                    created_by="human:analyst",
                    created_at=_at(2026, 6, 20),
                )
        with pytest.raises(ValueError, match="target_end"):
            propose_claims(
                session,
                [ClaimDraft(claim_type="forecast", **base)],
                created_by="human:analyst",
                created_at=_at(2026, 6, 20),
            )
        with pytest.raises(ValueError, match="future"):
            propose_claims(
                session,
                [ClaimDraft(claim_type="forecast", target_end=date(2025, 12, 31), **base)],
                created_by="human:analyst",
                created_at=_at(2026, 6, 20),
            )


def test_actor_identity_and_numeric_range_contracts_are_strict(session_factory, tmp_path):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        base = dict(
            claim_type="fact",
            statement="Inflation was 2.0 percent.",
            attribution_document_id=report.document_id,
            topic_key="inflation",
            geographies=("SE",),
            citations=(
                CitationDraft(report.extraction_id, 1, 1, "Inflation was 2.0 percent", "direct"),
            ),
        )
        with pytest.raises(ValueError, match="human:<id> or model"):
            propose_claims(
                session,
                [ClaimDraft(**base)],
                created_by="anonymous",
                created_at=_at(2026, 6, 20),
            )
        with pytest.raises(ValueError, match="supplied together"):
            propose_claims(
                session,
                [ClaimDraft(**base, lower_bound=1.0, unit="percent")],
                created_by="human:analyst",
                created_at=_at(2026, 6, 20),
            )
        draft_id = propose_claims(
            session,
            [ClaimDraft(**base)],
            created_by="human:analyst",
            created_at=_at(2026, 6, 20),
        )[0]
        with pytest.raises(ValueError, match="human:<id>"):
            verify_claim(session, draft_id, reviewer="adam", reviewed_at=_at(2026, 6, 21))


def test_judgment_remains_a_distinct_publisher_claim_type(session_factory, tmp_path):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        draft_id = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="judgment",
                    statement="The reported inflation level was judged material.",
                    attribution_document_id=report.document_id,
                    topic_key="inflation",
                    geographies=("SE",),
                    citations=(
                        CitationDraft(
                            report.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent in 2025.",
                            "direct",
                        ),
                    ),
                )
            ],
            created_by="human:analyst",
            created_at=_at(2026, 6, 20),
        )[0]
        verified_id = verify_claim(
            session, draft_id, reviewer="human:adam", reviewed_at=_at(2026, 6, 21)
        )
        views = load_claims(
            session,
            as_known_at=_at(2026, 6, 18),
            claim_types=("judgment",),
        )

    assert [view.id for view in views] == [verified_id]
    assert views[0].claim_type == "judgment"


def test_verified_inference_requires_reasoning_and_two_independent_publishers(
    session_factory, tmp_path
):
    with session_factory() as session:
        riksbank = _ingest(session, tmp_path)
        imf = _ingest(
            session,
            tmp_path,
            source_id="imf_weo_en",
            publisher="International Monetary Fund",
            issue_key="2026-04",
            available_at=_at(2026, 4, 22),
            retrieved_at=_at(2026, 4, 23),
        )
        one_source = ClaimDraft(
            claim_type="inference",
            statement="Disinflation may create room for easier policy.",
            topic_key="policy",
            geographies=("SE",),
            reasoning="Lower reported inflation reduces one constraint on policy.",
            citations=(
                CitationDraft(
                    riksbank.extraction_id,
                    1,
                    1,
                    "Inflation was 2.0 percent in 2025.",
                    "supporting",
                ),
            ),
        )
        draft_id = propose_claims(
            session,
            [one_source],
            created_by="human:analyst",
            created_at=_at(2026, 6, 20),
        )[0]
        with pytest.raises(ValueError, match="two independent publishers"):
            verify_claim(session, draft_id, reviewer="human:adam", reviewed_at=_at(2026, 6, 21))

        supported_id = propose_claims(
            session,
            [
                ClaimDraft(
                    **{
                        **one_source.__dict__,
                        "citations": (
                            *one_source.citations,
                            CitationDraft(
                                imf.extraction_id,
                                1,
                                1,
                                "Inflation was 2.0 percent in 2025.",
                                "supporting",
                            ),
                        ),
                    }
                )
            ],
            created_by="model:synthesizer-v1",
            created_at=_at(2026, 6, 20),
        )[0]
        verified_id = verify_claim(
            session, supported_id, reviewer="human:adam", reviewed_at=_at(2026, 6, 21)
        )
        verified = session.get(Claim, verified_id)

    assert verified.status == "verified"
    assert verified.attribution_document_id is None
    assert verified.reasoning


def test_two_documents_from_one_publisher_do_not_verify_an_inference(session_factory, tmp_path):
    with session_factory() as session:
        first = _ingest(
            session,
            tmp_path,
            issue_key="2026-01",
            available_at=_at(2026, 1, 10),
            retrieved_at=_at(2026, 1, 11),
        )
        second = _ingest(
            session,
            tmp_path,
            issue_key="2026-06",
            available_at=_at(2026, 6, 18),
            retrieved_at=_at(2026, 6, 19),
        )
        draft_id = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="inference",
                    statement="Two vintages suggest disinflation.",
                    topic_key="inflation",
                    geographies=("SE",),
                    reasoning="Both reports contain the same observed direction.",
                    citations=(
                        CitationDraft(
                            first.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent",
                            "supporting",
                        ),
                        CitationDraft(
                            second.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent",
                            "supporting",
                        ),
                    ),
                )
            ],
            created_by="model:test",
            created_at=_at(2026, 6, 20),
        )[0]
        with pytest.raises(ValueError, match="two independent publishers"):
            verify_claim(session, draft_id, reviewer="human:adam", reviewed_at=_at(2026, 6, 21))


def test_load_claims_is_point_in_time_honest_for_claim_and_every_document(
    session_factory, tmp_path
):
    with session_factory() as session:
        old = _ingest(
            session,
            tmp_path,
            issue_key="2026-01",
            available_at=_at(2026, 1, 10),
            retrieved_at=_at(2026, 3, 1),
        )
        new = _ingest(
            session,
            tmp_path,
            source_id="imf_weo_en",
            publisher="International Monetary Fund",
            issue_key="2026-04",
            available_at=_at(2026, 4, 20),
            retrieved_at=_at(2026, 4, 21),
        )
        old_draft = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="fact",
                    statement="Inflation was 2.0 percent.",
                    attribution_document_id=old.document_id,
                    topic_key="inflation",
                    geographies=("SE",),
                    reference_end=date(2025, 12, 31),
                    numeric_value=2.0,
                    unit="percent",
                    condition_text="As reported in this release.",
                    citations=(
                        CitationDraft(
                            old.extraction_id, 1, 1, "Inflation was 2.0 percent", "direct"
                        ),
                    ),
                )
            ],
            created_by="model:test",
            created_at=_at(2026, 5, 1),
        )[0]
        old_verified = verify_claim(
            session, old_draft, reviewer="human:adam", reviewed_at=_at(2026, 5, 2)
        )
        inference_draft = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="inference",
                    statement="Two publishers now corroborate disinflation.",
                    topic_key="inflation",
                    geographies=("SE", "WLD"),
                    reasoning="The independent reports point in the same direction.",
                    citations=(
                        CitationDraft(
                            old.extraction_id, 1, 1, "Inflation was 2.0 percent", "supporting"
                        ),
                        CitationDraft(
                            new.extraction_id, 1, 1, "Inflation was 2.0 percent", "supporting"
                        ),
                    ),
                )
            ],
            created_by="human:analyst",
            created_at=_at(2026, 4, 22),
        )[0]
        inference_verified = verify_claim(
            session,
            inference_draft,
            reviewer="human:adam",
            reviewed_at=_at(2026, 4, 23),
        )

        january = load_claims(session, as_known_at=_at(2026, 1, 15))
        april_21 = load_claims(session, as_known_at=_at(2026, 4, 21))
        april_22 = load_claims(session, as_known_at=_at(2026, 4, 22))
        april_23 = load_claims(session, as_known_at=_at(2026, 4, 23))

    assert [claim.id for claim in january] == [old_verified]
    assert [claim.id for claim in april_21] == [old_verified]
    assert [claim.id for claim in april_22] == [old_verified]
    assert {claim.id for claim in april_23} == {old_verified, inference_verified}
    assert all(claim.status == "verified" for claim in april_23)
    assert all(citation.pdf_page_start >= 1 for claim in april_23 for citation in claim.citations)
    reconstructed = january[0]
    assert reconstructed.numeric_value == 2.0 and reconstructed.unit == "percent"
    assert reconstructed.reference_end == date(2025, 12, 31)
    assert reconstructed.condition_text == "As reported in this release."
    assert reconstructed.created_by == "model:test"
    assert reconstructed.created_at == _at(2026, 5, 2)
    assert reconstructed.reviewed_by == "human:adam"
    assert reconstructed.reviewed_at == _at(2026, 5, 2)
    source = reconstructed.citations[0]
    assert source.report_title and source.artifact_url.startswith("https://")
    assert source.blob_path.is_file()
    assert source.document_published_at is None
    assert source.document_available_at == _at(2026, 1, 10)
    assert source.document_retrieved_at == _at(2026, 3, 1)
    assert source.extractor_name == "fixture" and source.extractor_version == "1"
    assert source.extracted_at.tzinfo is UTC


def test_claim_supersession_preserves_history_and_current_view(session_factory, tmp_path):
    with session_factory() as session:
        january_report = _ingest(
            session,
            tmp_path,
            issue_key="2026-01",
            available_at=_at(2026, 1, 10),
            retrieved_at=_at(2026, 1, 11),
        )
        old_draft = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="judgment",
                    statement="January assessment.",
                    attribution_document_id=january_report.document_id,
                    claim_series_key="se.inflation.assessment",
                    topic_key="inflation",
                    geographies=("SE",),
                    citations=(
                        CitationDraft(
                            january_report.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent",
                            "direct",
                        ),
                    ),
                )
            ],
            created_by="human:analyst",
            created_at=_at(2026, 1, 12),
        )[0]
        old_verified = verify_claim(
            session, old_draft, reviewer="human:adam", reviewed_at=_at(2026, 1, 13)
        )

        june_report = _ingest(
            session,
            tmp_path,
            issue_key="2026-06",
            available_at=_at(2026, 6, 18),
            retrieved_at=_at(2026, 6, 19),
        )
        july_support = _ingest(
            session,
            tmp_path,
            source_id="imf_weo_en",
            publisher="International Monetary Fund",
            issue_key="2026-07",
            available_at=_at(2026, 7, 1),
            retrieved_at=_at(2026, 7, 2),
        )
        new_draft = propose_claims(
            session,
            [
                ClaimDraft(
                    claim_type="judgment",
                    statement="June assessment supersedes January.",
                    attribution_document_id=june_report.document_id,
                    claim_series_key="se.inflation.assessment",
                    topic_key="inflation",
                    geographies=("SE",),
                    citations=(
                        CitationDraft(
                            june_report.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent",
                            "direct",
                        ),
                        CitationDraft(
                            july_support.extraction_id,
                            1,
                            1,
                            "Inflation was 2.0 percent",
                            "supporting",
                        ),
                    ),
                    supersedes_claim_id=old_verified,
                )
            ],
            created_by="human:analyst",
            created_at=_at(2026, 7, 2),
        )[0]
        new_verified = verify_claim(
            session, new_draft, reviewer="human:adam", reviewed_at=_at(2026, 7, 3)
        )

        before = load_claims(session, as_known_at=_at(2026, 6, 30))
        after = load_claims(session, as_known_at=_at(2026, 7, 1))
        history = load_claims(
            session,
            as_known_at=_at(2026, 7, 1),
            include_superseded=True,
        )

    assert [claim.id for claim in before] == [old_verified]
    assert [claim.id for claim in after] == [new_verified]
    assert {claim.id for claim in history} == {old_verified, new_verified}
    assert after[0].supersedes_claim_id == old_verified
    assert after[0].review_of_claim_id == new_draft


def test_evidence_rows_are_immutable_through_the_orm(session_factory, tmp_path):
    with session_factory() as session:
        report = _ingest(session, tmp_path)
        assert session.connection().exec_driver_sql("PRAGMA foreign_keys").scalar_one() == 1
        document = session.get(ReportDocument, report.document_id)
        document.title = "Rewritten history"
        with pytest.raises(ValueError, match="immutable"):
            session.commit()
        session.rollback()
        assert session.get(ReportDocument, report.document_id).title != "Rewritten history"

        with pytest.raises(IntegrityError, match="immutable"):
            session.execute(
                ReportDocument.__table__.update()
                .where(ReportDocument.id == report.document_id)
                .values(title="Bulk rewrite")
            )
        session.rollback()

        session.add(
            DocumentExtraction(
                document_id=999_999,
                extractor_name="fixture",
                extractor_version="orphan",
                extracted_at=_at(2026, 6, 20),
                status="complete",
                extracted_page_count=1,
                corpus_sha256="0" * 64,
            )
        )
        with pytest.raises(IntegrityError, match="FOREIGN KEY"):
            session.commit()
