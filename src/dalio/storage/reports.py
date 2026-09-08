"""Immutable institutional reports, page evidence, and reviewed atomic claims.

The numeric release ledger answers which values were publicly available at a
point in time.  This module applies the same clock discipline to prose reports
without treating an extracted or model-proposed statement as verified evidence.
Raw PDF bytes, extraction runs, pages, claim drafts, and reviewed claims are all
append-only.  Verification creates a successor claim instead of rewriting its
draft.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Sequence
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.reports.manifest import REPORT_SOURCES
from dalio.storage.db import (
    Claim,
    ClaimCitation,
    DocumentExtraction,
    DocumentPage,
    ReportDocument,
)

CLAIM_TYPES = frozenset({"fact", "forecast", "judgment", "inference"})
SUPPORT_ROLES = frozenset({"direct", "supporting", "contradicting"})
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_MAX_EXCERPT_CHARS = 1_000
_REPORT_SOURCES_BY_ID = {source.source_id: source for source in REPORT_SOURCES}


@dataclass(frozen=True)
class ReportMeta:
    source_id: str
    report_family: str
    issue_key: str
    publisher: str
    jurisdiction: str
    title: str
    language: str
    document_date: date
    available_at: datetime
    retrieved_at: datetime
    landing_url: str
    artifact_url: str
    allowed_domains: tuple[str, ...]
    published_at: datetime | None = None
    expected_sha256: str | None = None


@dataclass(frozen=True)
class PageText:
    pdf_page: int
    text: str
    printed_page_label: str | None = None


class PageExtractor(Protocol):
    """Injected extractor responsible for returning every physical PDF page, 1-based.

    The ledger verifies a gap-free sequence and deterministic output for an
    extractor name/version. Independent PDF page-count parsing belongs in the
    ingestion adapter so this storage core needs no PDF dependency.
    """

    name: str
    version: str

    def extract(self, pdf_bytes: bytes) -> Sequence[PageText]: ...


@dataclass(frozen=True)
class ReportIngestResult:
    document_id: int
    extraction_id: int
    created: bool
    content_sha256: str
    page_count: int
    blob_path: Path
    supersedes_document_id: int | None = None
    extraction_created: bool = False


@dataclass(frozen=True)
class CitationDraft:
    extraction_id: int
    pdf_page_start: int
    pdf_page_end: int
    evidence_excerpt: str
    support_role: str
    printed_locator: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ClaimDraft:
    claim_type: str
    statement: str
    topic_key: str
    geographies: tuple[str, ...]
    citations: tuple[CitationDraft, ...]
    attribution_document_id: int | None = None
    claim_series_key: str | None = None
    reference_start: date | None = None
    reference_end: date | None = None
    target_start: date | None = None
    target_end: date | None = None
    numeric_value: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str | None = None
    condition_text: str | None = None
    reasoning: str | None = None
    supersedes_claim_id: int | None = None


@dataclass(frozen=True)
class CitationView:
    document_id: int
    extraction_id: int
    pdf_page_start: int
    pdf_page_end: int
    printed_locator: str | None
    section_title: str | None
    evidence_excerpt: str
    support_role: str
    publisher: str
    source_id: str
    report_title: str
    issue_key: str
    document_date: date
    document_published_at: datetime | None
    document_available_at: datetime
    document_retrieved_at: datetime
    extractor_name: str
    extractor_version: str
    extracted_at: datetime
    landing_url: str
    artifact_url: str
    blob_path: Path
    content_sha256: str


@dataclass(frozen=True)
class ClaimView:
    id: int
    claim_type: str
    statement: str
    topic_key: str
    geographies: tuple[str, ...]
    status: str
    available_at: datetime
    attribution_document_id: int | None
    publisher: str | None
    source_id: str | None
    claim_series_key: str | None
    reference_start: date | None
    reference_end: date | None
    target_start: date | None
    target_end: date | None
    numeric_value: float | None
    lower_bound: float | None
    upper_bound: float | None
    unit: str | None
    condition_text: str | None
    reasoning: str | None
    created_by: str
    created_at: datetime
    reviewed_by: str | None
    reviewed_at: datetime | None
    review_of_claim_id: int | None
    supersedes_claim_id: int | None
    citations: tuple[CitationView, ...]


def _utc_naive(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value
    return value.astimezone(UTC).replace(tzinfo=None)


def _utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _required(value: str, field: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field} must not be empty")
    return clean


def _actor(value: str, field: str, *, human_only: bool = False) -> str:
    clean = _required(value, field)
    prefixes = ("human:",) if human_only else ("human:", "model:")
    if not clean.startswith(prefixes) or clean.endswith(":"):
        expected = "human:<id>" if human_only else "human:<id> or model:<id/version>"
        raise ValueError(f"{field} must be {expected}")
    return clean


def _normalize_meta(meta: ReportMeta) -> ReportMeta:
    available_at = _utc_naive(meta.available_at)
    retrieved_at = _utc_naive(meta.retrieved_at)
    published_at = _utc_naive(meta.published_at) if meta.published_at else None
    if published_at is not None and available_at < published_at:
        raise ValueError("available_at cannot be earlier than published_at")
    if retrieved_at < available_at:
        raise ValueError("retrieved_at cannot be earlier than available_at")
    if not isinstance(meta.document_date, date):
        raise ValueError("document_date must be a date")

    domains = tuple(
        domain.strip().lower().strip(".") for domain in meta.allowed_domains if domain.strip()
    )
    if not domains:
        raise ValueError("allowed_domains needs at least one official domain")
    for domain in domains:
        if "/" in domain or ":" in domain or not domain:
            raise ValueError(f"invalid allowlisted domain: {domain!r}")

    expected = meta.expected_sha256.lower() if meta.expected_sha256 else None
    if expected is not None and _SHA256_RE.fullmatch(expected) is None:
        raise ValueError("expected_sha256 must be 64 lowercase hexadecimal characters")

    normalized = replace(
        meta,
        source_id=_required(meta.source_id, "source_id"),
        report_family=_required(meta.report_family, "report_family"),
        issue_key=_required(meta.issue_key, "issue_key"),
        publisher=_required(meta.publisher, "publisher"),
        jurisdiction=_required(meta.jurisdiction, "jurisdiction"),
        title=_required(meta.title, "title"),
        language=_required(meta.language, "language").lower(),
        published_at=published_at,
        available_at=available_at,
        retrieved_at=retrieved_at,
        landing_url=_required(meta.landing_url, "landing_url"),
        artifact_url=_required(meta.artifact_url, "artifact_url"),
        allowed_domains=domains,
        expected_sha256=expected,
    )
    source = _REPORT_SOURCES_BY_ID.get(normalized.source_id)
    if source is None:
        raise ValueError(
            f"source_id is not in the official report manifest: {normalized.source_id}"
        )
    manifest_domains = tuple(domain.lower() for domain in source.official_domains)
    _validate_official_url(normalized.landing_url, manifest_domains, "landing_url")
    _validate_official_url(normalized.artifact_url, manifest_domains, "artifact_url")
    if normalized.publisher != source.publisher:
        raise ValueError("publisher conflicts with the official report manifest")
    if normalized.report_family != source.report_family:
        raise ValueError("report_family conflicts with the official report manifest")
    if normalized.jurisdiction != source.jurisdiction:
        raise ValueError("jurisdiction conflicts with the official report manifest")
    if normalized.language != source.language:
        raise ValueError("language conflicts with the official report manifest")
    if normalized.landing_url != source.landing_url:
        raise ValueError("landing_url conflicts with the official report manifest")
    if normalized.allowed_domains != manifest_domains:
        raise ValueError("allowed_domains conflicts with the official report manifest")
    return normalized


def _validate_official_url(url: str, domains: tuple[str, ...], field: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"{field} must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} must not contain user credentials")
    host = parsed.hostname.lower().strip(".")
    if not any(host == domain or host.endswith(f".{domain}") for domain in domains):
        raise ValueError(f"{field} host {host!r} is not allowlisted")


def _content_blob_path(blob_root: Path, digest: str) -> Path:
    return blob_root / "sha256" / digest[:2] / f"{digest}.pdf"


def _store_blob(blob_path: Path, pdf_bytes: bytes, digest: str) -> bool:
    """Store bytes once and return whether this call created the blob."""
    blob_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with blob_path.open("xb") as handle:
            handle.write(pdf_bytes)
    except FileExistsError:
        if hashlib.sha256(blob_path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"content-addressed blob has wrong sha256: {blob_path}") from None
        return False
    except Exception:
        with suppress(OSError):
            blob_path.unlink(missing_ok=True)
        raise
    return True


def _verify_stored_blob(blob_path: Path, digest: str) -> None:
    if blob_path.name != f"{digest}.pdf" or blob_path.parent.name != digest[:2]:
        raise ValueError("stored report blob_path is not content-addressed by its sha256")
    if not blob_path.is_file():
        raise ValueError(f"stored report blob is missing: {blob_path}")
    if hashlib.sha256(blob_path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"stored report blob has wrong sha256: {blob_path}")


def _canonical_pages(pages: Sequence[PageText]) -> tuple[PageText, ...]:
    work = tuple(pages)
    if not work:
        raise ValueError("extractor returned no pages")
    if not all(isinstance(page, PageText) for page in work):
        raise ValueError("extractor must return PageText rows")
    numbers = [page.pdf_page for page in work]
    expected = list(range(1, len(work) + 1))
    if numbers != expected:
        raise ValueError(f"extractor pages must be a complete 1-based sequence; got {numbers!r}")
    if any(not isinstance(page.text, str) for page in work):
        raise ValueError("page text must be a string")
    return work


def _pages_digest(pages: tuple[PageText, ...]) -> str:
    payload = json.dumps(
        [
            {
                "pdf_page": page.pdf_page,
                "printed_page_label": page.printed_page_label,
                "text": page.text,
            }
            for page in pages
        ],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _append_extraction(
    session: Session,
    *,
    document_id: int,
    extractor_name: str,
    extractor_version: str,
    pages: tuple[PageText, ...],
    corpus_sha256: str,
) -> DocumentExtraction:
    extraction = DocumentExtraction(
        document_id=document_id,
        extractor_name=extractor_name,
        extractor_version=extractor_version,
        extracted_at=datetime.now(UTC).replace(tzinfo=None),
        status="complete",
        extracted_page_count=len(pages),
        corpus_sha256=corpus_sha256,
        error=None,
    )
    session.add(extraction)
    session.flush()
    for page in pages:
        session.add(
            DocumentPage(
                extraction_id=extraction.id,
                pdf_page=page.pdf_page,
                printed_page_label=(
                    page.printed_page_label.strip() if page.printed_page_label else None
                ),
                text=page.text,
                text_sha256=hashlib.sha256(page.text.encode()).hexdigest(),
                char_count=len(page.text),
            )
        )
    session.flush()
    return extraction


def _assert_idempotent_metadata(existing: ReportDocument, meta: ReportMeta) -> None:
    expected = {
        "report_family": meta.report_family,
        "publisher": meta.publisher,
        "jurisdiction": meta.jurisdiction,
        "title": meta.title,
        "language": meta.language,
        "document_date": meta.document_date,
        "published_at": meta.published_at,
        "available_at": meta.available_at,
        "landing_url": meta.landing_url,
        "artifact_url": meta.artifact_url,
    }
    conflicts = [field for field, value in expected.items() if getattr(existing, field) != value]
    if conflicts:
        raise ValueError(
            "identical report bytes were submitted with conflicting immutable metadata: "
            + ", ".join(conflicts)
        )


def ingest_report(
    session: Session,
    pdf_bytes: bytes,
    meta: ReportMeta,
    *,
    extractor: PageExtractor,
    blob_root: Path,
) -> ReportIngestResult:
    """Archive and extract one complete official PDF report atomically.

    Identical bytes for the same logical issue are idempotent. A later version
    of that issue appends a document linked to the prior version; it never edits
    or removes the historical document.
    """
    normalized = _normalize_meta(meta)
    if not isinstance(pdf_bytes, bytes) or not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("report bytes must have a PDF %PDF- magic header")
    digest = hashlib.sha256(pdf_bytes).hexdigest()
    if normalized.expected_sha256 is not None and digest != normalized.expected_sha256:
        raise ValueError(
            f"report sha256 mismatch: expected {normalized.expected_sha256}, calculated {digest}"
        )
    extractor_name = _required(str(extractor.name), "extractor.name")
    extractor_version = _required(str(extractor.version), "extractor.version")
    pages = _canonical_pages(extractor.extract(pdf_bytes))
    corpus_sha256 = _pages_digest(pages)

    existing = session.execute(
        select(ReportDocument).where(
            ReportDocument.source_id == normalized.source_id,
            ReportDocument.issue_key == normalized.issue_key,
            ReportDocument.content_sha256 == digest,
        )
    ).scalar_one_or_none()
    if existing is not None:
        _assert_idempotent_metadata(existing, normalized)
        blob_path = Path(existing.blob_path)
        _verify_stored_blob(blob_path, digest)
        extraction = session.execute(
            select(DocumentExtraction).where(
                DocumentExtraction.document_id == existing.id,
                DocumentExtraction.extractor_name == extractor_name,
                DocumentExtraction.extractor_version == extractor_version,
            )
        ).scalar_one_or_none()
        extraction_created = False
        if extraction is not None:
            if extraction.status != "complete" or extraction.corpus_sha256 != corpus_sha256:
                raise ValueError(
                    "same extractor name/version produced different text for this document"
                )
        else:
            if len(pages) != existing.page_count:
                raise ValueError(
                    "new extractor version returned a different physical PDF page count"
                )
            try:
                extraction = _append_extraction(
                    session,
                    document_id=existing.id,
                    extractor_name=extractor_name,
                    extractor_version=extractor_version,
                    pages=pages,
                    corpus_sha256=corpus_sha256,
                )
                session.commit()
                extraction_created = True
            except Exception:
                session.rollback()
                raise
        return ReportIngestResult(
            document_id=existing.id,
            extraction_id=extraction.id,
            created=False,
            content_sha256=digest,
            page_count=existing.page_count,
            blob_path=blob_path,
            supersedes_document_id=existing.supersedes_document_id,
            extraction_created=extraction_created,
        )

    blob_path = _content_blob_path(Path(blob_root).resolve(), digest)

    previous = session.execute(
        select(ReportDocument)
        .where(
            ReportDocument.source_id == normalized.source_id,
            ReportDocument.issue_key == normalized.issue_key,
            ReportDocument.available_at <= normalized.available_at,
        )
        .order_by(
            ReportDocument.available_at.desc(),
            ReportDocument.retrieved_at.desc(),
            ReportDocument.id.desc(),
        )
        .limit(1)
    ).scalar_one_or_none()
    supersedes_id = previous.id if previous is not None else None

    blob_created = False
    try:
        document = ReportDocument(
            source_id=normalized.source_id,
            report_family=normalized.report_family,
            issue_key=normalized.issue_key,
            publisher=normalized.publisher,
            jurisdiction=normalized.jurisdiction,
            title=normalized.title,
            language=normalized.language,
            document_date=normalized.document_date,
            published_at=normalized.published_at,
            available_at=normalized.available_at,
            retrieved_at=normalized.retrieved_at,
            landing_url=normalized.landing_url,
            artifact_url=normalized.artifact_url,
            mime_type="application/pdf",
            content_sha256=digest,
            size_bytes=len(pdf_bytes),
            page_count=len(pages),
            blob_path=str(blob_path),
            supersedes_document_id=supersedes_id,
            created_at=datetime.now(UTC).replace(tzinfo=None),
        )
        session.add(document)
        session.flush()
        extraction = _append_extraction(
            session,
            document_id=document.id,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
            pages=pages,
            corpus_sha256=corpus_sha256,
        )
        blob_created = _store_blob(blob_path, pdf_bytes, digest)
        session.commit()
    except Exception:
        session.rollback()
        if blob_created:
            with suppress(OSError):
                blob_path.unlink(missing_ok=True)
        raise

    return ReportIngestResult(
        document_id=document.id,
        extraction_id=extraction.id,
        created=True,
        content_sha256=digest,
        page_count=len(pages),
        blob_path=blob_path,
        supersedes_document_id=supersedes_id,
        extraction_created=True,
    )


def _normalize_evidence_text(value: str) -> str:
    work = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    work = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", work)
    return re.sub(r"\s+", " ", work).strip().casefold()


@dataclass(frozen=True)
class _CitationEvidence:
    draft: CitationDraft
    extraction: DocumentExtraction
    document: ReportDocument
    excerpt: str
    excerpt_sha256: str


def _citation_evidence(session: Session, draft: CitationDraft) -> _CitationEvidence:
    if draft.support_role not in SUPPORT_ROLES:
        raise ValueError(f"support_role must be one of {sorted(SUPPORT_ROLES)}")
    if draft.pdf_page_start < 1 or draft.pdf_page_end < draft.pdf_page_start:
        raise ValueError("citation PDF page range must be positive and ordered")
    excerpt = draft.evidence_excerpt.strip()
    if not excerpt or len(excerpt) > _MAX_EXCERPT_CHARS:
        raise ValueError(f"evidence excerpt must contain 1-{_MAX_EXCERPT_CHARS} characters")

    extraction = session.get(DocumentExtraction, draft.extraction_id)
    if extraction is None or extraction.status != "complete":
        raise ValueError("citation requires a complete extraction")
    document = session.get(ReportDocument, extraction.document_id)
    if document is None:
        raise ValueError("citation extraction has no report document")
    _verify_stored_blob(Path(document.blob_path), document.content_sha256)
    if draft.pdf_page_end > document.page_count:
        raise ValueError("citation page is outside the report")
    pages = (
        session.execute(
            select(DocumentPage)
            .where(
                DocumentPage.extraction_id == extraction.id,
                DocumentPage.pdf_page >= draft.pdf_page_start,
                DocumentPage.pdf_page <= draft.pdf_page_end,
            )
            .order_by(DocumentPage.pdf_page)
        )
        .scalars()
        .all()
    )
    expected_count = draft.pdf_page_end - draft.pdf_page_start + 1
    if len(pages) != expected_count:
        raise ValueError("citation page range is incomplete")
    normalized_excerpt = _normalize_evidence_text(excerpt)
    normalized_pages = _normalize_evidence_text("\n".join(page.text for page in pages))
    if normalized_excerpt not in normalized_pages:
        raise ValueError("evidence excerpt was not found on the cited PDF page range")
    return _CitationEvidence(
        draft=draft,
        extraction=extraction,
        document=document,
        excerpt=excerpt,
        excerpt_sha256=hashlib.sha256(normalized_excerpt.encode()).hexdigest(),
    )


def _validate_period(start: date | None, end: date | None, label: str) -> None:
    if start is not None and end is not None and start > end:
        raise ValueError(f"{label}_start cannot be later than {label}_end")


def _validate_numbers(draft: ClaimDraft) -> None:
    values = (draft.numeric_value, draft.lower_bound, draft.upper_bound)
    if any(value is not None and not math.isfinite(float(value)) for value in values):
        raise ValueError("claim numeric fields must be finite")
    if any(value is not None for value in values) and not (draft.unit and draft.unit.strip()):
        raise ValueError("numeric claim fields require a unit")
    if (draft.lower_bound is None) != (draft.upper_bound is None):
        raise ValueError("lower_bound and upper_bound must be supplied together")
    if (
        draft.lower_bound is not None
        and draft.upper_bound is not None
        and draft.lower_bound > draft.upper_bound
    ):
        raise ValueError("lower_bound cannot exceed upper_bound")


def _validate_claim_draft(
    session: Session,
    draft: ClaimDraft,
    *,
    created_at: datetime,
) -> tuple[ReportDocument | None, tuple[_CitationEvidence, ...], datetime]:
    if draft.claim_type not in CLAIM_TYPES:
        raise ValueError(f"claim_type must be one of {sorted(CLAIM_TYPES)}")
    _required(draft.statement, "statement")
    _required(draft.topic_key, "topic_key")
    if not draft.geographies or any(not geography.strip() for geography in draft.geographies):
        raise ValueError("geographies needs at least one non-empty geography")
    if not draft.citations:
        raise ValueError("claim needs at least one page citation")
    _validate_period(draft.reference_start, draft.reference_end, "reference")
    _validate_period(draft.target_start, draft.target_end, "target")
    _validate_numbers(draft)

    evidence = tuple(_citation_evidence(session, citation) for citation in draft.citations)
    if any(created_at < item.document.retrieved_at for item in evidence):
        raise ValueError("claim created_at cannot precede retrieval of a cited document")
    attribution: ReportDocument | None = None
    if draft.claim_type == "inference":
        if draft.attribution_document_id is not None:
            raise ValueError("inference must be attributed to the Observatory, not a document")
        if not draft.reasoning or not draft.reasoning.strip():
            raise ValueError("inference requires explicit reasoning")
        if any(item.draft.support_role == "direct" for item in evidence):
            raise ValueError("inference citations must be supporting or contradicting, not direct")
        supporting = [item for item in evidence if item.draft.support_role == "supporting"]
        if not supporting:
            raise ValueError("inference needs at least one supporting citation")
        available_at = max(created_at, *(item.document.available_at for item in evidence))
    else:
        if draft.attribution_document_id is None:
            raise ValueError(f"{draft.claim_type} requires an attribution_document_id")
        attribution = session.get(ReportDocument, draft.attribution_document_id)
        if attribution is None:
            raise ValueError("attribution_document_id does not exist")
        direct = [
            item
            for item in evidence
            if item.draft.support_role == "direct" and item.document.id == attribution.id
        ]
        if not direct:
            raise ValueError("publisher claim needs a direct citation to its attribution document")
        available_at = max(
            attribution.available_at,
            *(item.document.available_at for item in evidence),
        )

    if draft.supersedes_claim_id is not None:
        prior = session.get(Claim, draft.supersedes_claim_id)
        if prior is None or prior.status != "verified":
            raise ValueError("supersedes_claim_id must identify a verified claim")
        if draft.claim_type != prior.claim_type:
            raise ValueError("a successor claim must keep the prior claim_type")
        if draft.topic_key.strip() != prior.topic_key:
            raise ValueError("a successor claim must keep the prior topic_key")
        if not draft.claim_series_key or draft.claim_series_key != prior.claim_series_key:
            raise ValueError("a successor claim must keep a non-empty claim_series_key")
        if available_at < prior.available_at:
            raise ValueError("a successor claim cannot become available before the prior claim")

    if draft.claim_type == "forecast":
        if draft.target_end is None:
            raise ValueError("forecast requires target_end")
        assert attribution is not None
        if draft.target_end <= attribution.available_at.date():
            raise ValueError("forecast target_end must be in the future at report availability")
    elif draft.claim_type == "fact" and (
        draft.target_start is not None or draft.target_end is not None
    ):
        raise ValueError("fact must not carry a future target period; split it from the forecast")

    return attribution, evidence, available_at


def propose_claims(
    session: Session,
    drafts: Sequence[ClaimDraft],
    *,
    created_by: str,
    created_at: datetime | None = None,
) -> list[int]:
    """Append structurally valid claim drafts and mechanically located citations."""
    creator = _actor(created_by, "created_by")
    proposed_at = _utc_naive(created_at or datetime.now(UTC))
    ids: list[int] = []
    try:
        for draft in drafts:
            _attribution, evidence, available_at = _validate_claim_draft(
                session, draft, created_at=proposed_at
            )
            claim = Claim(
                claim_type=draft.claim_type,
                statement=draft.statement.strip(),
                attribution_document_id=draft.attribution_document_id,
                claim_series_key=(
                    draft.claim_series_key.strip() if draft.claim_series_key else None
                ),
                topic_key=draft.topic_key.strip(),
                geographies_json=json.dumps(
                    list(dict.fromkeys(geography.strip() for geography in draft.geographies)),
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
                reference_start=draft.reference_start,
                reference_end=draft.reference_end,
                target_start=draft.target_start,
                target_end=draft.target_end,
                numeric_value=draft.numeric_value,
                lower_bound=draft.lower_bound,
                upper_bound=draft.upper_bound,
                unit=draft.unit.strip() if draft.unit else None,
                condition_text=draft.condition_text.strip() if draft.condition_text else None,
                reasoning=draft.reasoning.strip() if draft.reasoning else None,
                status="draft",
                available_at=available_at,
                created_by=creator,
                created_at=proposed_at,
                reviewed_by=None,
                reviewed_at=None,
                review_of_claim_id=None,
                supersedes_claim_id=draft.supersedes_claim_id,
            )
            session.add(claim)
            session.flush()
            for item in evidence:
                session.add(
                    ClaimCitation(
                        claim_id=claim.id,
                        extraction_id=item.extraction.id,
                        pdf_page_start=item.draft.pdf_page_start,
                        pdf_page_end=item.draft.pdf_page_end,
                        printed_locator=(
                            item.draft.printed_locator.strip()
                            if item.draft.printed_locator
                            else None
                        ),
                        section_title=(
                            item.draft.section_title.strip() if item.draft.section_title else None
                        ),
                        evidence_excerpt=item.excerpt,
                        excerpt_sha256=item.excerpt_sha256,
                        support_role=item.draft.support_role,
                        locator_verified_at=proposed_at,
                        semantic_verified_by=None,
                        semantic_verified_at=None,
                    )
                )
            session.flush()
            ids.append(claim.id)
        session.commit()
    except Exception:
        session.rollback()
        raise
    return ids


def _claim_citations(session: Session, claim_id: int) -> list[ClaimCitation]:
    return list(
        session.execute(
            select(ClaimCitation)
            .where(ClaimCitation.claim_id == claim_id)
            .order_by(ClaimCitation.id)
        ).scalars()
    )


def _citation_resolution(
    session: Session, citation: ClaimCitation
) -> tuple[DocumentExtraction, ReportDocument]:
    extraction = session.get(DocumentExtraction, citation.extraction_id)
    if extraction is None or extraction.status != "complete":
        raise ValueError("claim citation no longer resolves to a complete extraction")
    document = session.get(ReportDocument, extraction.document_id)
    if document is None:
        raise ValueError("claim citation no longer resolves to a report document")
    _verify_stored_blob(Path(document.blob_path), document.content_sha256)
    return extraction, document


def _citation_document(session: Session, citation: ClaimCitation) -> ReportDocument:
    return _citation_resolution(session, citation)[1]


def verify_claim(
    session: Session,
    claim_id: int,
    *,
    reviewer: str,
    reviewed_at: datetime | None = None,
) -> int:
    """Append a human-reviewed successor to a draft; never mutate the draft."""
    reviewer_name = _actor(reviewer, "reviewer", human_only=True)
    checked_at = _utc_naive(reviewed_at or datetime.now(UTC))
    draft = session.get(Claim, claim_id)
    if draft is None:
        raise ValueError(f"claim {claim_id} does not exist")
    if draft.status != "draft":
        raise ValueError("only a draft claim can be verified")
    if checked_at < draft.created_at:
        raise ValueError("reviewed_at cannot be earlier than claim created_at")

    existing = session.execute(
        select(Claim).where(
            Claim.review_of_claim_id == draft.id,
            Claim.status == "verified",
        )
    ).scalar_one_or_none()
    if existing is not None:
        return existing.id
    if draft.supersedes_claim_id is not None:
        competing = session.execute(
            select(Claim).where(
                Claim.supersedes_claim_id == draft.supersedes_claim_id,
                Claim.status == "verified",
            )
        ).scalar_one_or_none()
        if competing is not None:
            raise ValueError("the prior claim already has a verified successor")

    citations = _claim_citations(session, draft.id)
    if not citations or any(citation.locator_verified_at is None for citation in citations):
        raise ValueError("verified claim requires mechanically located citations")
    resolved = [(citation, _citation_document(session, citation)) for citation in citations]

    if draft.claim_type == "inference":
        if not draft.reasoning or not draft.reasoning.strip():
            raise ValueError("verified inference requires explicit reasoning")
        supporting = [pair for pair in resolved if pair[0].support_role == "supporting"]
        documents = {document.id for _citation, document in supporting}
        publishers = {document.publisher.casefold() for _citation, document in supporting}
        if len(documents) < 2 or len(publishers) < 2:
            raise ValueError("verified inference requires two independent publishers and documents")
        available_at = max(
            checked_at,
            draft.available_at,
            *(document.available_at for _citation, document in resolved),
        )
    else:
        if draft.attribution_document_id is None:
            raise ValueError("verified publisher claim requires document attribution")
        direct = [
            pair
            for pair in resolved
            if pair[0].support_role == "direct" and pair[1].id == draft.attribution_document_id
        ]
        if not direct:
            raise ValueError(
                "verified publisher claim requires a semantically reviewed direct citation"
            )
        available_at = draft.available_at

    try:
        verified = Claim(
            claim_type=draft.claim_type,
            statement=draft.statement,
            attribution_document_id=draft.attribution_document_id,
            claim_series_key=draft.claim_series_key,
            topic_key=draft.topic_key,
            geographies_json=draft.geographies_json,
            reference_start=draft.reference_start,
            reference_end=draft.reference_end,
            target_start=draft.target_start,
            target_end=draft.target_end,
            numeric_value=draft.numeric_value,
            lower_bound=draft.lower_bound,
            upper_bound=draft.upper_bound,
            unit=draft.unit,
            condition_text=draft.condition_text,
            reasoning=draft.reasoning,
            status="verified",
            available_at=available_at,
            created_by=draft.created_by,
            created_at=checked_at,
            reviewed_by=reviewer_name,
            reviewed_at=checked_at,
            review_of_claim_id=draft.id,
            supersedes_claim_id=draft.supersedes_claim_id,
        )
        session.add(verified)
        session.flush()
        for citation in citations:
            session.add(
                ClaimCitation(
                    claim_id=verified.id,
                    extraction_id=citation.extraction_id,
                    pdf_page_start=citation.pdf_page_start,
                    pdf_page_end=citation.pdf_page_end,
                    printed_locator=citation.printed_locator,
                    section_title=citation.section_title,
                    evidence_excerpt=citation.evidence_excerpt,
                    excerpt_sha256=citation.excerpt_sha256,
                    support_role=citation.support_role,
                    locator_verified_at=citation.locator_verified_at,
                    semantic_verified_by=reviewer_name,
                    semantic_verified_at=checked_at,
                )
            )
        session.flush()
        session.commit()
    except Exception:
        session.rollback()
        raise
    return verified.id


def load_claims(
    session: Session,
    *,
    as_known_at: datetime,
    source_ids: tuple[str, ...] | None = None,
    topics: tuple[str, ...] | None = None,
    claim_types: tuple[str, ...] | None = None,
    verified_only: bool = True,
    include_superseded: bool = False,
) -> list[ClaimView]:
    """Return claims whose claim clock and every supporting document pass the cutoff."""
    cutoff = _utc_naive(as_known_at)
    if claim_types and not set(claim_types).issubset(CLAIM_TYPES):
        raise ValueError(f"claim_types must be drawn from {sorted(CLAIM_TYPES)}")
    stmt = select(Claim).where(Claim.available_at <= cutoff)
    if verified_only:
        stmt = stmt.where(Claim.status == "verified")
    if topics:
        stmt = stmt.where(Claim.topic_key.in_(topics))
    if claim_types:
        stmt = stmt.where(Claim.claim_type.in_(claim_types))
    claims = session.execute(stmt.order_by(Claim.available_at, Claim.id)).scalars().all()
    allowed_sources = set(source_ids) if source_ids else None
    views: list[ClaimView] = []
    for claim in claims:
        if (
            claim.claim_type == "inference"
            and claim.status == "verified"
            and (claim.reviewed_at is None or claim.reviewed_at > cutoff)
        ):
            continue
        attribution = (
            session.get(ReportDocument, claim.attribution_document_id)
            if claim.attribution_document_id is not None
            else None
        )
        if attribution is not None and attribution.available_at > cutoff:
            continue
        citation_views: list[CitationView] = []
        citation_sources: set[str] = set()
        unavailable = False
        for citation in _claim_citations(session, claim.id):
            extraction, document = _citation_resolution(session, citation)
            if document.available_at > cutoff:
                unavailable = True
                break
            citation_sources.add(document.source_id)
            citation_views.append(
                CitationView(
                    document_id=document.id,
                    extraction_id=citation.extraction_id,
                    pdf_page_start=citation.pdf_page_start,
                    pdf_page_end=citation.pdf_page_end,
                    printed_locator=citation.printed_locator,
                    section_title=citation.section_title,
                    evidence_excerpt=citation.evidence_excerpt,
                    support_role=citation.support_role,
                    publisher=document.publisher,
                    source_id=document.source_id,
                    report_title=document.title,
                    issue_key=document.issue_key,
                    document_date=document.document_date,
                    document_published_at=(
                        _utc_aware(document.published_at) if document.published_at else None
                    ),
                    document_available_at=_utc_aware(document.available_at),
                    document_retrieved_at=_utc_aware(document.retrieved_at),
                    extractor_name=extraction.extractor_name,
                    extractor_version=extraction.extractor_version,
                    extracted_at=_utc_aware(extraction.extracted_at),
                    landing_url=document.landing_url,
                    artifact_url=document.artifact_url,
                    blob_path=Path(document.blob_path),
                    content_sha256=document.content_sha256,
                )
            )
        if unavailable:
            continue
        if allowed_sources is not None:
            represented = {attribution.source_id} if attribution is not None else citation_sources
            if represented.isdisjoint(allowed_sources):
                continue
        views.append(
            ClaimView(
                id=claim.id,
                claim_type=claim.claim_type,
                statement=claim.statement,
                topic_key=claim.topic_key,
                geographies=tuple(json.loads(claim.geographies_json)),
                status=claim.status,
                available_at=_utc_aware(claim.available_at),
                attribution_document_id=claim.attribution_document_id,
                publisher=attribution.publisher if attribution is not None else None,
                source_id=attribution.source_id if attribution is not None else None,
                claim_series_key=claim.claim_series_key,
                reference_start=claim.reference_start,
                reference_end=claim.reference_end,
                target_start=claim.target_start,
                target_end=claim.target_end,
                numeric_value=claim.numeric_value,
                lower_bound=claim.lower_bound,
                upper_bound=claim.upper_bound,
                unit=claim.unit,
                condition_text=claim.condition_text,
                reasoning=claim.reasoning,
                created_by=claim.created_by,
                created_at=_utc_aware(claim.created_at),
                reviewed_by=claim.reviewed_by,
                reviewed_at=_utc_aware(claim.reviewed_at) if claim.reviewed_at else None,
                review_of_claim_id=claim.review_of_claim_id,
                supersedes_claim_id=claim.supersedes_claim_id,
                citations=tuple(citation_views),
            )
        )
    if include_superseded:
        return views
    superseded_ids = {
        claim.supersedes_claim_id
        for claim in views
        if claim.status == "verified" and claim.supersedes_claim_id is not None
    }
    return [claim for claim in views if claim.id not in superseded_ids]
