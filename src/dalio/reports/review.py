"""Strict, read-only preparation of model-drafted report claims for human review.

The candidate catalogue is not an evidence verdict.  It deliberately cannot
carry review state, approval, probabilities, scores, or investor instructions.
This module resolves each proposal back to immutable official PDF bytes and a
complete page extraction, mechanically checks the excerpt, and builds a
deterministic packet that a named human can assess through the separate claim
review workflow.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.reports.manifest import REPORT_SOURCES, ReportSourceSpec
from dalio.storage.db import DocumentExtraction, DocumentPage, ReportDocument

REPORT_REVIEW_SCHEMA_VERSION = 1
REPORT_REVIEW_METHODOLOGY_VERSION = "report-review-candidates-v1"
MAX_REVIEW_CANDIDATES_PER_DOCUMENT = 4
UNVERIFIED_DRAFT_LABEL = "UNVERIFIED MODEL DRAFT"
PUBLISHER_CLAIM_TYPES = frozenset({"fact", "forecast", "judgment"})
REVIEW_HORIZONS = frozenset({"0-1y", "1-3y", "3-5y", "5y+"})

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ISSUE_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_GEOGRAPHY_RE = re.compile(r"^[A-Z]{2,3}$")
_MAX_EXCERPT_CHARS = 1_000
_MAX_TEXT_CHARS = 2_000
_CONTEXT_SIDE_CHARS = 350
_SOURCES_BY_ID = {source.source_id: source for source in REPORT_SOURCES}
_SOURCE_ORDER = {source.source_id: index for index, source in enumerate(REPORT_SOURCES)}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "methodology_version",
        "created_by",
        "created_at",
        "as_known_at",
        "candidates",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "source_id",
        "issue_key",
        "document_sha256",
        "extraction_name",
        "extraction_version",
        "extraction_corpus_sha256",
        "claim_type",
        "statement",
        "topic_key",
        "geographies",
        "claim_series_key",
        "reference_start",
        "reference_end",
        "target_start",
        "target_end",
        "numeric_value",
        "lower_bound",
        "upper_bound",
        "unit",
        "condition_text",
        "horizon",
        "importance_rationale",
        "citations",
    }
)
_CITATION_FIELDS = frozenset(
    {
        "pdf_page_start",
        "pdf_page_end",
        "printed_locator",
        "section_title",
        "evidence_excerpt",
        "support_role",
    }
)


@dataclass(frozen=True)
class CandidateCitation:
    pdf_page_start: int
    pdf_page_end: int
    evidence_excerpt: str
    support_role: str
    printed_locator: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class ClaimCandidate:
    source_id: str
    issue_key: str
    document_sha256: str
    extraction_name: str
    extraction_version: str
    extraction_corpus_sha256: str
    claim_type: str
    statement: str
    topic_key: str
    geographies: tuple[str, ...]
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
    horizon: str
    importance_rationale: str
    citations: tuple[CandidateCitation, ...]

    @property
    def candidate_id(self) -> str:
        return candidate_fingerprint(self)


@dataclass(frozen=True)
class CandidateCatalogue:
    schema_version: int
    methodology_version: str
    created_by: str
    created_at: datetime
    as_known_at: datetime
    candidates: tuple[ClaimCandidate, ...]


@dataclass(frozen=True)
class ResolvedDocument:
    # Internal ledger identifiers are intentionally omitted from published
    # packets.  The separate human decision writer uses them only after it has
    # rebuilt and revalidated the packet against the same database.
    document_id: int
    extraction_id: int
    source_id: str
    issue_key: str
    publisher: str
    report_family: str
    jurisdiction: str
    title: str
    document_date: date
    published_at: datetime | None
    available_at: datetime
    retrieved_at: datetime
    landing_url: str
    artifact_url: str
    content_sha256: str
    page_count: int
    blob_path: Path
    extraction_name: str
    extraction_version: str
    extraction_corpus_sha256: str
    extracted_at: datetime


@dataclass(frozen=True)
class ResolvedCitation:
    citation: CandidateCitation
    page_text_sha256: tuple[str, ...]
    context_excerpt: str


@dataclass(frozen=True)
class ResolvedCandidate:
    candidate: ClaimCandidate
    document: ResolvedDocument
    citations: tuple[ResolvedCitation, ...]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _exact_fields(payload: dict[str, object], expected: frozenset[str], label: str) -> None:
    supplied = set(payload)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ValueError(f"{label} has missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _required_string(value: object, field: str, *, maximum: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not contain leading or trailing whitespace")
    if len(value) > maximum:
        raise ValueError(f"{field} must contain at most {maximum} characters")
    return value


def _optional_string(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_string(value, field)


def _hash(value: object, field: str) -> str:
    digest = _required_string(value, field, maximum=64)
    if _SHA256_RE.fullmatch(digest) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return digest


def _parse_datetime(value: object, field: str) -> datetime:
    raw = _required_string(value, field, maximum=64)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _parse_date(value: object, field: str) -> date | None:
    if value is None:
        return None
    raw = _required_string(value, field, maximum=10)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date or null") from exc
    if raw != parsed.isoformat():
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date or null")
    return parsed


def _number(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number or null")
    return result


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _parse_citation(payload: object, index: int, candidate_index: int) -> CandidateCitation:
    label = f"candidates[{candidate_index}].citations[{index}]"
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    _exact_fields(payload, _CITATION_FIELDS, label)
    page_start = _positive_int(payload["pdf_page_start"], f"{label}.pdf_page_start")
    page_end = _positive_int(payload["pdf_page_end"], f"{label}.pdf_page_end")
    if page_end < page_start:
        raise ValueError(f"{label} PDF page range must be ordered")
    support_role = _required_string(payload["support_role"], f"{label}.support_role")
    if support_role != "direct":
        raise ValueError(f"{label}.support_role must be direct for a publisher claim")
    return CandidateCitation(
        pdf_page_start=page_start,
        pdf_page_end=page_end,
        printed_locator=_optional_string(payload["printed_locator"], f"{label}.printed_locator"),
        section_title=_optional_string(payload["section_title"], f"{label}.section_title"),
        evidence_excerpt=_required_string(
            payload["evidence_excerpt"],
            f"{label}.evidence_excerpt",
            maximum=_MAX_EXCERPT_CHARS,
        ),
        support_role=support_role,
    )


def _parse_candidate(payload: object, index: int) -> ClaimCandidate:
    label = f"candidates[{index}]"
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be an object")
    _exact_fields(payload, _CANDIDATE_FIELDS, label)
    source_id = _required_string(payload["source_id"], f"{label}.source_id")
    if source_id not in _SOURCES_BY_ID:
        raise ValueError(f"{label}.source_id is not in the official report manifest")
    issue_key = _required_string(payload["issue_key"], f"{label}.issue_key")
    if _ISSUE_KEY_RE.fullmatch(issue_key) is None:
        raise ValueError(f"{label}.issue_key contains unsupported characters")
    claim_type = _required_string(payload["claim_type"], f"{label}.claim_type")
    if claim_type not in PUBLISHER_CLAIM_TYPES:
        raise ValueError(
            f"{label}.claim_type must be a publisher claim: "
            f"{', '.join(sorted(PUBLISHER_CLAIM_TYPES))}"
        )
    raw_geographies = payload["geographies"]
    if not isinstance(raw_geographies, list) or not raw_geographies:
        raise ValueError(f"{label}.geographies must be a non-empty array")
    geographies = tuple(
        _required_string(value, f"{label}.geographies[{geo_index}]", maximum=3)
        for geo_index, value in enumerate(raw_geographies)
    )
    if len(set(geographies)) != len(geographies):
        raise ValueError(f"{label}.geographies must not contain duplicates")
    if any(_GEOGRAPHY_RE.fullmatch(value) is None for value in geographies):
        raise ValueError(f"{label}.geographies must use uppercase 2-3 letter codes")
    raw_citations = payload["citations"]
    if not isinstance(raw_citations, list) or not raw_citations:
        raise ValueError(f"{label}.citations must be a non-empty array")
    citations = tuple(
        _parse_citation(item, citation_index, index)
        for citation_index, item in enumerate(raw_citations)
    )
    reference_start = _parse_date(payload["reference_start"], f"{label}.reference_start")
    reference_end = _parse_date(payload["reference_end"], f"{label}.reference_end")
    target_start = _parse_date(payload["target_start"], f"{label}.target_start")
    target_end = _parse_date(payload["target_end"], f"{label}.target_end")
    if (
        reference_start is not None
        and reference_end is not None
        and reference_start > reference_end
    ):
        raise ValueError(f"{label}.reference_start cannot be later than reference_end")
    if target_start is not None and target_end is not None and target_start > target_end:
        raise ValueError(f"{label}.target_start cannot be later than target_end")
    if claim_type == "forecast" and target_end is None:
        raise ValueError(f"{label} forecast requires target_end")
    if claim_type == "fact" and (target_start is not None or target_end is not None):
        raise ValueError(f"{label} fact must not carry a future target period")
    numeric_value = _number(payload["numeric_value"], f"{label}.numeric_value")
    lower_bound = _number(payload["lower_bound"], f"{label}.lower_bound")
    upper_bound = _number(payload["upper_bound"], f"{label}.upper_bound")
    unit = _optional_string(payload["unit"], f"{label}.unit")
    if (
        any(value is not None for value in (numeric_value, lower_bound, upper_bound))
        and unit is None
    ):
        raise ValueError(f"{label} numeric fields require a unit")
    if unit is not None and all(
        value is None for value in (numeric_value, lower_bound, upper_bound)
    ):
        raise ValueError(f"{label}.unit requires at least one numeric field")
    if (lower_bound is None) != (upper_bound is None):
        raise ValueError(f"{label}.lower_bound and upper_bound must be supplied together")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError(f"{label}.lower_bound cannot exceed upper_bound")
    horizon = _required_string(payload["horizon"], f"{label}.horizon", maximum=8)
    if horizon not in REVIEW_HORIZONS:
        raise ValueError(f"{label}.horizon must be one of {sorted(REVIEW_HORIZONS)}")
    return ClaimCandidate(
        source_id=source_id,
        issue_key=issue_key,
        document_sha256=_hash(payload["document_sha256"], f"{label}.document_sha256"),
        extraction_name=_required_string(payload["extraction_name"], f"{label}.extraction_name"),
        extraction_version=_required_string(
            payload["extraction_version"], f"{label}.extraction_version"
        ),
        extraction_corpus_sha256=_hash(
            payload["extraction_corpus_sha256"], f"{label}.extraction_corpus_sha256"
        ),
        claim_type=claim_type,
        statement=_required_string(payload["statement"], f"{label}.statement"),
        topic_key=_required_string(payload["topic_key"], f"{label}.topic_key"),
        geographies=geographies,
        claim_series_key=_optional_string(payload["claim_series_key"], f"{label}.claim_series_key"),
        reference_start=reference_start,
        reference_end=reference_end,
        target_start=target_start,
        target_end=target_end,
        numeric_value=numeric_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        condition_text=_optional_string(payload["condition_text"], f"{label}.condition_text"),
        horizon=horizon,
        importance_rationale=_required_string(
            payload["importance_rationale"], f"{label}.importance_rationale"
        ),
        citations=citations,
    )


def load_candidate_catalogue(path: Path) -> CandidateCatalogue:
    """Load the exact candidate schema without touching the report database."""
    source_path = Path(path)
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"), object_pairs_hook=_unique_json_object
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read report candidate catalogue {source_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("report candidate catalogue must be a JSON object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "report candidate catalogue")
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != REPORT_REVIEW_SCHEMA_VERSION
    ):
        raise ValueError(
            f"report candidate catalogue schema_version must be {REPORT_REVIEW_SCHEMA_VERSION}"
        )
    methodology = _required_string(payload["methodology_version"], "methodology_version")
    if methodology != REPORT_REVIEW_METHODOLOGY_VERSION:
        raise ValueError(f"methodology_version must be {REPORT_REVIEW_METHODOLOGY_VERSION}")
    creator = _required_string(payload["created_by"], "created_by")
    if not creator.startswith("model:") or creator == "model:":
        raise ValueError("created_by must be model:<id/version>")
    created_at = _parse_datetime(payload["created_at"], "created_at")
    as_known_at = _parse_datetime(payload["as_known_at"], "as_known_at")
    if created_at < as_known_at:
        raise ValueError("created_at cannot be earlier than as_known_at")
    raw_candidates = payload["candidates"]
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidates must be a non-empty array")
    candidates = tuple(_parse_candidate(item, index) for index, item in enumerate(raw_candidates))
    fingerprints = [candidate_fingerprint(candidate) for candidate in candidates]
    if len(fingerprints) != len(set(fingerprints)):
        raise ValueError("candidate catalogue contains duplicate candidate fingerprints")
    return CandidateCatalogue(
        schema_version=schema_version,
        methodology_version=methodology,
        created_by=creator,
        created_at=created_at,
        as_known_at=as_known_at,
        candidates=candidates,
    )


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _candidate_payload(candidate: ClaimCandidate) -> dict[str, object]:
    payload = asdict(candidate)
    for key in ("reference_start", "reference_end", "target_start", "target_end"):
        value = payload[key]
        payload[key] = value.isoformat() if value is not None else None
    payload["geographies"] = list(candidate.geographies)
    payload["citations"] = [asdict(citation) for citation in candidate.citations]
    return payload


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def candidate_fingerprint(candidate: ClaimCandidate) -> str:
    """Return the stable semantic-and-evidence identifier for one proposal."""
    if not isinstance(candidate, ClaimCandidate):
        raise TypeError("candidate must be a ClaimCandidate")
    return hashlib.sha256(_canonical_json(_candidate_payload(candidate))).hexdigest()


def _utc_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _normalize_evidence_text(value: str, *, casefold: bool = True) -> str:
    work = unicodedata.normalize("NFKC", value).replace("\u00ad", "")
    work = re.sub(r"(?<=\w)-[ \t]*\r?\n[ \t]*(?=\w)", "", work)
    work = re.sub(r"\s+", " ", work).strip()
    return work.casefold() if casefold else work


def _pages_digest(pages: tuple[DocumentPage, ...]) -> str:
    payload = [
        {
            "pdf_page": page.pdf_page,
            "printed_page_label": page.printed_page_label,
            "text": page.text,
        }
        for page in pages
    ]
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _context_excerpt(page_text: str, excerpt: str) -> str:
    display = _normalize_evidence_text(page_text, casefold=False)
    needle = _normalize_evidence_text(excerpt)
    location = display.casefold().find(needle)
    if location < 0:
        return display[: 2 * _CONTEXT_SIDE_CHARS]
    start = max(0, location - _CONTEXT_SIDE_CHARS)
    end = min(len(display), location + len(needle) + _CONTEXT_SIDE_CHARS)
    prefix = "…" if start else ""
    suffix = "…" if end < len(display) else ""
    return f"{prefix}{display[start:end]}{suffix}"


def _latest_eligible_documents(
    session: Session, catalogue: CandidateCatalogue
) -> dict[str, ReportDocument]:
    cutoff = catalogue.as_known_at.astimezone(UTC).replace(tzinfo=None)
    created = catalogue.created_at.astimezone(UTC).replace(tzinfo=None)
    rows = session.execute(
        select(ReportDocument)
        .where(
            ReportDocument.source_id.in_(tuple(_SOURCES_BY_ID)),
            ReportDocument.available_at <= cutoff,
            ReportDocument.retrieved_at <= created,
        )
        .order_by(
            ReportDocument.source_id,
            ReportDocument.available_at.desc(),
            ReportDocument.document_date.desc(),
            ReportDocument.retrieved_at.desc(),
            ReportDocument.id.desc(),
        )
    ).scalars()
    latest: dict[str, ReportDocument] = {}
    for document in rows:
        latest.setdefault(document.source_id, document)
    return latest


def _verify_blob(document: ReportDocument) -> Path:
    blob_path = Path(document.blob_path)
    if not blob_path.is_file():
        raise ValueError(f"archived PDF is missing: {blob_path}")
    if hashlib.sha256(blob_path.read_bytes()).hexdigest() != document.content_sha256:
        raise ValueError(f"archived PDF sha256 does not match document: {blob_path}")
    if blob_path.name != f"{document.content_sha256}.pdf":
        raise ValueError("archived PDF path is not content-addressed by document sha256")
    if blob_path.parent.name != document.content_sha256[:2]:
        raise ValueError("archived PDF path has an invalid content-addressed parent")
    return blob_path


def _logical_archive_reference(content_sha256: str) -> str:
    """Return a checkout-independent reference without exposing the physical blob root."""
    return f"reports/sha256/{content_sha256[:2]}/{content_sha256}.pdf"


def _verify_manifest_identity(document: ReportDocument, source: ReportSourceSpec) -> None:
    expected = {
        "publisher": source.publisher,
        "report_family": source.report_family,
        "jurisdiction": source.jurisdiction,
        "language": source.language,
        "landing_url": source.landing_url,
        "mime_type": source.mime_type,
    }
    conflicts = [field for field, value in expected.items() if getattr(document, field) != value]
    if conflicts:
        raise ValueError(
            "stored document conflicts with the official report manifest: " + ", ".join(conflicts)
        )
    parsed = urlsplit(document.artifact_url)
    host = parsed.hostname.lower().strip(".") if parsed.hostname else ""
    if parsed.scheme.lower() != "https" or not any(
        host == domain or host.endswith(f".{domain}") for domain in source.official_domains
    ):
        raise ValueError("stored document artifact URL is outside its official domain allowlist")


def resolve_candidate_evidence(
    session: Session,
    candidate: ClaimCandidate,
    *,
    catalogue: CandidateCatalogue,
) -> ResolvedCandidate:
    """Resolve and mechanically verify one unverified proposal without writing rows."""
    if candidate not in catalogue.candidates:
        raise ValueError("candidate is not part of the supplied catalogue")
    with session.no_autoflush:
        latest = _latest_eligible_documents(session, catalogue).get(candidate.source_id)
        if latest is None:
            raise ValueError(f"no eligible official document for {candidate.source_id}")
        identity = (latest.issue_key, latest.content_sha256)
        if identity != (candidate.issue_key, candidate.document_sha256):
            raise ValueError(
                f"candidate does not bind the latest eligible document for {candidate.source_id}"
            )
        source = _SOURCES_BY_ID[candidate.source_id]
        _verify_manifest_identity(latest, source)
        if candidate.topic_key not in source.topic_allowlist:
            raise ValueError(
                f"candidate topic {candidate.topic_key!r} is outside the official source allowlist"
            )
        if (
            candidate.claim_type == "forecast"
            and candidate.target_end is not None
            and candidate.target_end <= latest.available_at.date()
        ):
            raise ValueError("candidate forecast target_end must follow report availability")
        blob_path = _verify_blob(latest)
        extraction = session.execute(
            select(DocumentExtraction).where(
                DocumentExtraction.document_id == latest.id,
                DocumentExtraction.extractor_name == candidate.extraction_name,
                DocumentExtraction.extractor_version == candidate.extraction_version,
                DocumentExtraction.corpus_sha256 == candidate.extraction_corpus_sha256,
            )
        ).scalar_one_or_none()
        if extraction is None or extraction.status != "complete":
            raise ValueError("candidate extraction identity is missing or not complete")
        if catalogue.created_at < _utc_aware(extraction.extracted_at):
            raise ValueError("candidate created_at cannot precede its extraction")
        pages = tuple(
            session.execute(
                select(DocumentPage)
                .where(DocumentPage.extraction_id == extraction.id)
                .order_by(DocumentPage.pdf_page)
            ).scalars()
        )
        expected_pages = list(range(1, latest.page_count + 1))
        if [page.pdf_page for page in pages] != expected_pages:
            raise ValueError("candidate extraction does not contain every physical page")
        if extraction.extracted_page_count != latest.page_count:
            raise ValueError("candidate extraction page count conflicts with its document")
        for page in pages:
            if page.char_count != len(page.text):
                raise ValueError("candidate extraction has an invalid page character count")
            if hashlib.sha256(page.text.encode()).hexdigest() != page.text_sha256:
                raise ValueError("candidate extraction has an invalid page text hash")
        if _pages_digest(pages) != extraction.corpus_sha256:
            raise ValueError("candidate extraction corpus sha256 does not match its pages")

        resolved_citations: list[ResolvedCitation] = []
        by_number = {page.pdf_page: page for page in pages}
        for citation in candidate.citations:
            if citation.pdf_page_end > latest.page_count:
                raise ValueError("candidate citation is outside the report physical page range")
            cited_pages = tuple(
                by_number[number]
                for number in range(citation.pdf_page_start, citation.pdf_page_end + 1)
                if number in by_number
            )
            expected_count = citation.pdf_page_end - citation.pdf_page_start + 1
            if len(cited_pages) != expected_count:
                raise ValueError("candidate citation page range is incomplete")
            joined_text = "\n".join(page.text for page in cited_pages)
            if _normalize_evidence_text(citation.evidence_excerpt) not in _normalize_evidence_text(
                joined_text
            ):
                raise ValueError("candidate evidence excerpt was not found on the cited PDF pages")
            resolved_citations.append(
                ResolvedCitation(
                    citation=citation,
                    page_text_sha256=tuple(page.text_sha256 for page in cited_pages),
                    context_excerpt=_context_excerpt(joined_text, citation.evidence_excerpt),
                )
            )

    return ResolvedCandidate(
        candidate=candidate,
        document=ResolvedDocument(
            document_id=latest.id,
            extraction_id=extraction.id,
            source_id=latest.source_id,
            issue_key=latest.issue_key,
            publisher=latest.publisher,
            report_family=latest.report_family,
            jurisdiction=latest.jurisdiction,
            title=latest.title,
            document_date=latest.document_date,
            published_at=_utc_aware(latest.published_at) if latest.published_at else None,
            available_at=_utc_aware(latest.available_at),
            retrieved_at=_utc_aware(latest.retrieved_at),
            landing_url=latest.landing_url,
            artifact_url=latest.artifact_url,
            content_sha256=latest.content_sha256,
            page_count=latest.page_count,
            blob_path=blob_path,
            extraction_name=extraction.extractor_name,
            extraction_version=extraction.extractor_version,
            extraction_corpus_sha256=extraction.corpus_sha256,
            extracted_at=_utc_aware(extraction.extracted_at),
        ),
        citations=tuple(resolved_citations),
    )


def validate_candidate_catalogue(
    session: Session, catalogue: CandidateCatalogue
) -> tuple[ResolvedCandidate, ...]:
    """Fail closed unless the catalogue exactly covers each locally eligible source."""
    if not isinstance(catalogue, CandidateCatalogue):
        raise TypeError("catalogue must be a CandidateCatalogue")
    with session.no_autoflush:
        latest = _latest_eligible_documents(session, catalogue)
        expected_sources = set(latest)
        candidate_sources = {candidate.source_id for candidate in catalogue.candidates}
        missing = sorted(expected_sources - candidate_sources)
        unexpected = sorted(candidate_sources - expected_sources)
        if missing:
            raise ValueError(
                f"missing candidate source for latest eligible document: {', '.join(missing)}"
            )
        if unexpected:
            raise ValueError(
                f"candidate source has no latest eligible document: {', '.join(unexpected)}"
            )
        counts = Counter(candidate.source_id for candidate in catalogue.candidates)
        for source_id, count in counts.items():
            source = _SOURCES_BY_ID[source_id]
            limit = min(source.max_claims, MAX_REVIEW_CANDIDATES_PER_DOCUMENT)
            if count > limit:
                raise ValueError(
                    f"{source_id} has {count} candidates; at most {limit} are allowed per document"
                )
        normalized_statements = [
            (candidate.source_id, _normalize_evidence_text(candidate.statement))
            for candidate in catalogue.candidates
        ]
        if len(normalized_statements) != len(set(normalized_statements)):
            raise ValueError("candidate catalogue repeats a statement within one document")
        resolved = [
            resolve_candidate_evidence(session, candidate, catalogue=catalogue)
            for candidate in catalogue.candidates
        ]
    return tuple(
        sorted(
            resolved,
            key=lambda item: (
                _SOURCE_ORDER[item.candidate.source_id],
                item.candidate.candidate_id,
            ),
        )
    )


def _catalogue_sha256(catalogue: CandidateCatalogue) -> str:
    payload = {
        "schema_version": catalogue.schema_version,
        "methodology_version": catalogue.methodology_version,
        "created_by": catalogue.created_by,
        "created_at": _iso_datetime(catalogue.created_at),
        "as_known_at": _iso_datetime(catalogue.as_known_at),
        "candidates": sorted(
            (_candidate_payload(candidate) for candidate in catalogue.candidates),
            key=lambda item: _canonical_json(item),
        ),
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _packet_candidate(item: ResolvedCandidate) -> dict[str, object]:
    candidate = item.candidate
    payload = _candidate_payload(candidate)
    payload["candidate_id"] = candidate.candidate_id
    payload["draft_label"] = UNVERIFIED_DRAFT_LABEL
    payload["citations"] = [
        {
            **asdict(citation.citation),
            "page_text_sha256": list(citation.page_text_sha256),
            "context_excerpt": citation.context_excerpt,
        }
        for citation in item.citations
    ]
    return payload


def build_review_packet(session: Session, catalogue: CandidateCatalogue) -> dict[str, object]:
    """Build a deterministic review aid; this API performs no database writes."""
    resolved = validate_candidate_catalogue(session, catalogue)
    documents: list[dict[str, object]] = []
    for source_id in sorted({item.candidate.source_id for item in resolved}, key=_SOURCE_ORDER.get):
        source_items = [item for item in resolved if item.candidate.source_id == source_id]
        document = source_items[0].document
        documents.append(
            {
                "source_id": document.source_id,
                "issue_key": document.issue_key,
                "publisher": document.publisher,
                "report_family": document.report_family,
                "jurisdiction": document.jurisdiction,
                "title": document.title,
                "document_date": document.document_date.isoformat(),
                "published_at": _iso_datetime(document.published_at),
                "available_at": _iso_datetime(document.available_at),
                "retrieved_at": _iso_datetime(document.retrieved_at),
                "landing_url": document.landing_url,
                "artifact_url": document.artifact_url,
                "content_sha256": document.content_sha256,
                "page_count": document.page_count,
                "archive_reference": _logical_archive_reference(document.content_sha256),
                "extraction": {
                    "name": document.extraction_name,
                    "version": document.extraction_version,
                    "corpus_sha256": document.extraction_corpus_sha256,
                    "extracted_at": _iso_datetime(document.extracted_at),
                    "page_count": document.page_count,
                },
                "candidates": [_packet_candidate(item) for item in source_items],
            }
        )
    packet: dict[str, object] = {
        "schema_version": REPORT_REVIEW_SCHEMA_VERSION,
        "packet_kind": "unverified_report_claim_human_review",
        "notice": (
            "UNVERIFIED MODEL DRAFTS. Mechanical provenance and locator checks do not "
            "establish semantic support, importance, or suitability for risk analysis."
        ),
        "methodology_version": catalogue.methodology_version,
        "candidate_catalogue_sha256": _catalogue_sha256(catalogue),
        "proposed_by": catalogue.created_by,
        "proposed_at": _iso_datetime(catalogue.created_at),
        "as_known_at": _iso_datetime(catalogue.as_known_at),
        "document_count": len(documents),
        "candidate_count": len(resolved),
        "review_questions": [
            "Is the attribution a fair reading of the cited original page?",
            "Are type, scope, periods, units, and material conditions correct?",
            "Is this proposition important enough for macro risk analysis?",
        ],
        "documents": documents,
    }
    packet["packet_sha256"] = hashlib.sha256(_canonical_json(packet)).hexdigest()
    return packet


def _markdown_quote(value: str) -> str:
    return "\n".join(f"> {line}" if line else ">" for line in value.splitlines())


def _period_text(candidate: dict[str, object]) -> str:
    fields = (
        ("Reference", candidate["reference_start"], candidate["reference_end"]),
        ("Target", candidate["target_start"], candidate["target_end"]),
    )
    pieces = []
    for label, start, end in fields:
        if start is not None or end is not None:
            pieces.append(f"{label}: {start or 'open'} to {end or 'open'}")
    return " · ".join(pieces) if pieces else "No explicit reference or target period proposed"


def render_review_markdown(packet: dict[str, object]) -> str:
    """Render a plain-language review sheet from :func:`build_review_packet`."""
    if packet.get("packet_kind") != "unverified_report_claim_human_review":
        raise ValueError("unsupported report review packet")
    lines = [
        "# Official-report claim review",
        "",
        "**UNVERIFIED MODEL DRAFTS — nothing below is an approved conclusion.**",
        "",
        str(packet["notice"]),
        "",
        f"Public-information cutoff: `{packet['as_known_at']}`  ",
        f"Proposal created: `{packet['proposed_at']}` by `{packet['proposed_by']}`  ",
        f"Packet SHA-256: `{packet['packet_sha256']}`",
        "",
    ]
    for document in packet["documents"]:
        lines.extend(
            [
                f"## {document['publisher']} — {document['title']}",
                "",
                f"Issue `{document['source_id']}:{document['issue_key']}` · "
                f"document date `{document['document_date']}` · "
                f"publicly available `{document['available_at']}` · "
                f"retrieved `{document['retrieved_at']}`",
                "",
                f"Official artifact: [{document['artifact_url']}]({document['artifact_url']})  ",
                f"Archived PDF reference: `{document['archive_reference']}`  ",
                f"PDF SHA-256: `{document['content_sha256']}`  ",
                f"Extraction: `{document['extraction']['name']} "
                f"{document['extraction']['version']}` · corpus "
                f"`{document['extraction']['corpus_sha256']}`",
                "",
            ]
        )
        for candidate in document["candidates"]:
            lines.extend(
                [
                    f"### {candidate['draft_label']} — `{candidate['candidate_id']}`",
                    "",
                    f"Type: `{candidate['claim_type']}` · topic: `{candidate['topic_key']}` · "
                    f"horizon: `{candidate['horizon']}` · geographies: "
                    f"`{', '.join(candidate['geographies'])}`",
                    "",
                    "Proposed statement:",
                    "",
                    _markdown_quote(candidate["statement"]),
                    "",
                    f"Conditions: {candidate['condition_text'] or 'none proposed'}  ",
                    f"Periods: {_period_text(candidate)}",
                    "",
                    f"Model selection rationale: {candidate['importance_rationale']}",
                    "",
                ]
            )
            for citation in candidate["citations"]:
                page_label = (
                    f"PDF page {citation['pdf_page_start']}"
                    if citation["pdf_page_start"] == citation["pdf_page_end"]
                    else f"PDF pages {citation['pdf_page_start']}–{citation['pdf_page_end']}"
                )
                lines.extend(
                    [
                        f"Evidence — {page_label}",
                        "",
                        _markdown_quote(citation["evidence_excerpt"]),
                        "",
                        "Surrounding extracted context (review aid only):",
                        "",
                        _markdown_quote(citation["context_excerpt"]),
                        "",
                    ]
                )
            lines.extend(
                [
                    "- [ ] Attribution is fair",
                    "- [ ] Claim type is correct",
                    "- [ ] Scope, periods, units, and conditions are correct",
                    "- [ ] Important enough for macro-risk analysis",
                    "",
                    "Outcome: approve / revise / reject — ____________________",
                    "",
                ]
            )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "CandidateCatalogue",
    "CandidateCitation",
    "ClaimCandidate",
    "MAX_REVIEW_CANDIDATES_PER_DOCUMENT",
    "PUBLISHER_CLAIM_TYPES",
    "REPORT_REVIEW_METHODOLOGY_VERSION",
    "REPORT_REVIEW_SCHEMA_VERSION",
    "REVIEW_HORIZONS",
    "ResolvedCandidate",
    "ResolvedCitation",
    "ResolvedDocument",
    "UNVERIFIED_DRAFT_LABEL",
    "build_review_packet",
    "candidate_fingerprint",
    "load_candidate_catalogue",
    "render_review_markdown",
    "resolve_candidate_evidence",
    "validate_candidate_catalogue",
]
