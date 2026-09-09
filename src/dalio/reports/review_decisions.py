"""Strict human-decision contract for provenance-bound report candidates.

The editable decision file deliberately contains neither a reviewer identity nor
a review clock.  Those values belong to the operator-confirmed write boundary.  A
decision file is useful only when it still matches a packet rebuilt from the
checked catalogue and immutable report ledger.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.reports.manifest import REPORT_SOURCES
from dalio.reports.review import (
    PUBLISHER_CLAIM_TYPES,
    CandidateCatalogue,
    ClaimCandidate,
    ResolvedCandidate,
    build_review_packet,
    validate_candidate_catalogue,
)
from dalio.storage.db import ReportCandidateReview
from dalio.storage.reports import (
    CitationDraft,
    ClaimDraft,
    _append_claims,
    _append_verified_claim,
)

REPORT_DECISION_SCHEMA_VERSION = 1
REPORT_DECISION_KIND = "official_report_claim_human_review_decisions"
DECISION_OUTCOMES = frozenset({"approve", "revise", "reject"})
DECISION_REASON_CODES = frozenset(
    {
        "unsupported",
        "misattributed",
        "wrong_type",
        "wrong_scope",
        "wrong_period_or_unit",
        "missing_condition",
        "not_material",
        "duplicate",
        "other",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GEOGRAPHY_RE = re.compile(r"^[A-Z]{2,3}$")
_HUMAN_REVIEWER_RE = re.compile(r"^human:[A-Za-z0-9][A-Za-z0-9._@-]{0,127}$")
_MAX_TEXT_CHARS = 2_000
_SOURCES_BY_ID = {source.source_id: source for source in REPORT_SOURCES}

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "decision_kind",
        "methodology_version",
        "packet_sha256",
        "candidate_catalogue_sha256",
        "decisions",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "candidate_id",
        "source_id",
        "issue_key",
        "proposed_statement",
        "outcome",
        "attestations",
        "reason_code",
        "review_note",
        "revision",
    }
)
_ATTESTATION_FIELDS = frozenset(
    {
        "attribution_fair",
        "type_correct",
        "scope_periods_units_conditions_correct",
        "important_for_macro_risk",
    }
)
_REVISION_FIELDS = frozenset(
    {
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
    }
)


@dataclass(frozen=True)
class DecisionAttestations:
    attribution_fair: bool | None
    type_correct: bool | None
    scope_periods_units_conditions_correct: bool | None
    important_for_macro_risk: bool | None

    def values(self) -> tuple[bool | None, ...]:
        return (
            self.attribution_fair,
            self.type_correct,
            self.scope_periods_units_conditions_correct,
            self.important_for_macro_risk,
        )


@dataclass(frozen=True)
class SemanticRevision:
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


@dataclass(frozen=True)
class CandidateDecision:
    candidate_id: str
    source_id: str
    issue_key: str
    proposed_statement: str
    outcome: str | None
    attestations: DecisionAttestations
    reason_code: str | None
    review_note: str | None
    revision: SemanticRevision | None


@dataclass(frozen=True)
class ReviewDecisionFile:
    schema_version: int
    decision_kind: str
    methodology_version: str
    packet_sha256: str
    candidate_catalogue_sha256: str
    decisions: tuple[CandidateDecision, ...]


@dataclass(frozen=True)
class ValidatedDecision:
    decision: CandidateDecision
    resolved: ResolvedCandidate


@dataclass(frozen=True)
class ValidatedDecisionPlan:
    decision_file: ReviewDecisionFile
    packet: dict[str, object]
    decisions: tuple[ValidatedDecision, ...]


@dataclass(frozen=True)
class AppliedDecision:
    candidate_id: str
    outcome: str
    review_id: int
    original_draft_claim_id: int
    revised_draft_claim_id: int | None
    verified_claim_id: int | None
    replayed: bool


@dataclass(frozen=True)
class ReviewApplicationResult:
    packet_sha256: str
    results: tuple[AppliedDecision, ...]

    @property
    def replayed(self) -> bool:
        return bool(self.results) and all(result.replayed for result in self.results)


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant: {value}")


def _exact_fields(payload: dict[str, object], expected: frozenset[str], label: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing:
        raise ValueError(f"{label} has missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")


def _required_string(value: object, field: str, *, maximum: int = _MAX_TEXT_CHARS) -> str:
    if not isinstance(value, str) or not value:
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


def _sha256(value: object, field: str) -> str:
    result = _required_string(value, field, maximum=64)
    if _SHA256_RE.fullmatch(result) is None:
        raise ValueError(f"{field} must be 64 lowercase hexadecimal characters")
    return result


def _parse_date(value: object, field: str) -> date | None:
    if value is None:
        return None
    raw = _required_string(value, field, maximum=10)
    try:
        result = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date or null") from exc
    if raw != result.isoformat():
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date or null")
    return result


def _number(value: object, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be a finite number or null")
    return result


def _parse_attestations(value: object, label: str) -> DecisionAttestations:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_fields(value, _ATTESTATION_FIELDS, label)
    for key, item in value.items():
        if item is not None and not isinstance(item, bool):
            raise ValueError(f"{label}.{key} must be true, false, or null")
    return DecisionAttestations(**value)


def _parse_revision(value: object, label: str) -> SemanticRevision | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object or null")
    _exact_fields(value, _REVISION_FIELDS, label)
    claim_type = _required_string(value["claim_type"], f"{label}.claim_type", maximum=24)
    if claim_type not in PUBLISHER_CLAIM_TYPES:
        raise ValueError(
            f"{label}.claim_type must be one of {', '.join(sorted(PUBLISHER_CLAIM_TYPES))}"
        )
    raw_geographies = value["geographies"]
    if not isinstance(raw_geographies, list) or not raw_geographies:
        raise ValueError(f"{label}.geographies must be a non-empty array")
    geographies = tuple(
        _required_string(item, f"{label}.geographies[{index}]", maximum=3)
        for index, item in enumerate(raw_geographies)
    )
    if len(set(geographies)) != len(geographies):
        raise ValueError(f"{label}.geographies must not contain duplicates")
    if any(_GEOGRAPHY_RE.fullmatch(item) is None for item in geographies):
        raise ValueError(f"{label}.geographies must use uppercase 2-3 letter codes")
    reference_start = _parse_date(value["reference_start"], f"{label}.reference_start")
    reference_end = _parse_date(value["reference_end"], f"{label}.reference_end")
    target_start = _parse_date(value["target_start"], f"{label}.target_start")
    target_end = _parse_date(value["target_end"], f"{label}.target_end")
    if (
        reference_start is not None
        and reference_end is not None
        and reference_start > reference_end
    ):
        raise ValueError(f"{label}.reference_start cannot be later than reference_end")
    if target_start is not None and target_end is not None and target_start > target_end:
        raise ValueError(f"{label}.target_start cannot be later than target_end")
    numeric_value = _number(value["numeric_value"], f"{label}.numeric_value")
    lower_bound = _number(value["lower_bound"], f"{label}.lower_bound")
    upper_bound = _number(value["upper_bound"], f"{label}.upper_bound")
    if (lower_bound is None) != (upper_bound is None):
        raise ValueError(f"{label} lower_bound and upper_bound must be supplied together")
    if lower_bound is not None and upper_bound is not None and lower_bound > upper_bound:
        raise ValueError(f"{label}.lower_bound cannot exceed upper_bound")
    unit = _optional_string(value["unit"], f"{label}.unit")
    if any(item is not None for item in (numeric_value, lower_bound, upper_bound)) and unit is None:
        raise ValueError(f"{label} numeric fields require a unit")
    if claim_type == "forecast" and target_end is None:
        raise ValueError(f"{label} forecast requires target_end")
    if claim_type == "fact" and (target_start is not None or target_end is not None):
        raise ValueError(f"{label} fact cannot carry a future target period")
    return SemanticRevision(
        claim_type=claim_type,
        statement=_required_string(value["statement"], f"{label}.statement"),
        topic_key=_required_string(value["topic_key"], f"{label}.topic_key", maximum=96),
        geographies=geographies,
        claim_series_key=_optional_string(value["claim_series_key"], f"{label}.claim_series_key"),
        reference_start=reference_start,
        reference_end=reference_end,
        target_start=target_start,
        target_end=target_end,
        numeric_value=numeric_value,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        unit=unit,
        condition_text=_optional_string(value["condition_text"], f"{label}.condition_text"),
    )


def _parse_decision(value: object, index: int) -> CandidateDecision:
    label = f"decisions[{index}]"
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    _exact_fields(value, _DECISION_FIELDS, label)
    outcome = value["outcome"]
    if outcome is not None:
        outcome = _required_string(outcome, f"{label}.outcome", maximum=16)
        if outcome not in DECISION_OUTCOMES:
            raise ValueError(f"{label}.outcome must be approve, revise, reject, or null")
    reason_code = _optional_string(value["reason_code"], f"{label}.reason_code")
    if reason_code is not None and reason_code not in DECISION_REASON_CODES:
        raise ValueError(
            f"{label}.reason_code must be one of {', '.join(sorted(DECISION_REASON_CODES))}"
        )
    result = CandidateDecision(
        candidate_id=_sha256(value["candidate_id"], f"{label}.candidate_id"),
        source_id=_required_string(value["source_id"], f"{label}.source_id", maximum=96),
        issue_key=_required_string(value["issue_key"], f"{label}.issue_key", maximum=128),
        proposed_statement=_required_string(
            value["proposed_statement"], f"{label}.proposed_statement"
        ),
        outcome=outcome,
        attestations=_parse_attestations(value["attestations"], f"{label}.attestations"),
        reason_code=reason_code,
        review_note=_optional_string(value["review_note"], f"{label}.review_note"),
        revision=_parse_revision(value["revision"], f"{label}.revision"),
    )
    _validate_outcome_contract(result, label)
    return result


def _validate_outcome_contract(decision: CandidateDecision, label: str) -> None:
    attestations = decision.attestations.values()
    if decision.outcome is None:
        if any(item is not None for item in attestations):
            raise ValueError(f"{label} pending decision must leave all attestations null")
        if any(
            item is not None
            for item in (decision.reason_code, decision.review_note, decision.revision)
        ):
            raise ValueError(f"{label} pending decision must not contain review content")
        return
    if any(item is None for item in attestations):
        raise ValueError(f"{label} completed decision requires all four attestations")
    if decision.review_note is None:
        raise ValueError(f"{label} completed decision requires a non-empty review_note")
    if decision.outcome == "approve":
        if not all(attestations):
            raise ValueError(f"{label} approval requires all four attestations to be true")
        if decision.reason_code is not None or decision.revision is not None:
            raise ValueError(f"{label} approval must not contain reason_code or revision")
    elif decision.outcome == "revise":
        if not all(attestations):
            raise ValueError(f"{label} revision requires all four attestations to be true")
        if decision.reason_code is None or decision.revision is None:
            raise ValueError(f"{label} revision requires reason_code and complete revision")
    else:
        if all(attestations):
            raise ValueError(f"{label} rejection requires at least one false attestation")
        if decision.reason_code is None:
            raise ValueError(f"{label} rejection requires reason_code")
        if decision.revision is not None:
            raise ValueError(f"{label} rejection must not contain revision")


def parse_decision_payload(payload: object) -> ReviewDecisionFile:
    """Parse a schema-exact editable decision payload."""
    if not isinstance(payload, dict):
        raise ValueError("decision file must contain one JSON object")
    _exact_fields(payload, _TOP_LEVEL_FIELDS, "decision file")
    schema_version = payload["schema_version"]
    if type(schema_version) is not int or schema_version != REPORT_DECISION_SCHEMA_VERSION:
        raise ValueError(f"decision file schema_version must be {REPORT_DECISION_SCHEMA_VERSION}")
    decision_kind = _required_string(payload["decision_kind"], "decision_kind", maximum=96)
    if decision_kind != REPORT_DECISION_KIND:
        raise ValueError(f"decision_kind must be {REPORT_DECISION_KIND!r}")
    raw_decisions = payload["decisions"]
    if not isinstance(raw_decisions, list) or not raw_decisions:
        raise ValueError("decisions must be a non-empty array")
    decisions = tuple(_parse_decision(value, index) for index, value in enumerate(raw_decisions))
    ids = [decision.candidate_id for decision in decisions]
    if len(ids) != len(set(ids)):
        raise ValueError("decision file repeats a candidate_id")
    return ReviewDecisionFile(
        schema_version=schema_version,
        decision_kind=decision_kind,
        methodology_version=_required_string(
            payload["methodology_version"], "methodology_version", maximum=96
        ),
        packet_sha256=_sha256(payload["packet_sha256"], "packet_sha256"),
        candidate_catalogue_sha256=_sha256(
            payload["candidate_catalogue_sha256"], "candidate_catalogue_sha256"
        ),
        decisions=decisions,
    )


def load_decision_file(path: Path) -> ReviewDecisionFile:
    """Load UTF-8 JSON while rejecting duplicate keys and non-finite constants."""
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"cannot read decision file {path}: {exc}") from exc
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid decision JSON in {path}: {exc.msg}") from exc
    return parse_decision_payload(payload)


def build_decision_template(session: Session, catalogue: CandidateCatalogue) -> dict[str, object]:
    """Return an editable, packet-bound template without reviewer attribution."""
    packet = build_review_packet(session, catalogue)
    decisions = []
    for document in packet["documents"]:
        for candidate in document["candidates"]:
            decisions.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "source_id": candidate["source_id"],
                    "issue_key": candidate["issue_key"],
                    "proposed_statement": candidate["statement"],
                    "outcome": None,
                    "attestations": {field: None for field in sorted(_ATTESTATION_FIELDS)},
                    "reason_code": None,
                    "review_note": None,
                    "revision": None,
                }
            )
    return {
        "schema_version": REPORT_DECISION_SCHEMA_VERSION,
        "decision_kind": REPORT_DECISION_KIND,
        "methodology_version": packet["methodology_version"],
        "packet_sha256": packet["packet_sha256"],
        "candidate_catalogue_sha256": packet["candidate_catalogue_sha256"],
        "decisions": decisions,
    }


def _candidate_semantics(candidate: ClaimCandidate) -> dict[str, object]:
    result: dict[str, object] = {}
    for field in sorted(_REVISION_FIELDS):
        value = getattr(candidate, field)
        if isinstance(value, date):
            value = value.isoformat()
        elif isinstance(value, tuple):
            value = list(value)
        result[field] = value
    return result


def _original_candidate_payload(candidate: ClaimCandidate) -> dict[str, object]:
    """Return the complete canonical catalogue candidate retained for audit."""
    fields = (
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
    )
    payload: dict[str, object] = {}
    for field in fields:
        value = getattr(candidate, field)
        if isinstance(value, date):
            value = value.isoformat()
        elif field == "geographies":
            value = list(value)
        elif field == "citations":
            value = [asdict(citation) for citation in value]
        payload[field] = value
    payload["candidate_id"] = candidate.candidate_id
    return payload


def _revision_semantics(revision: SemanticRevision) -> dict[str, object]:
    result = asdict(revision)
    for field in ("reference_start", "reference_end", "target_start", "target_end"):
        value = result[field]
        result[field] = value.isoformat() if value is not None else None
    result["geographies"] = list(revision.geographies)
    return {field: result[field] for field in sorted(_REVISION_FIELDS)}


def validate_review_decisions(
    session: Session,
    catalogue: CandidateCatalogue,
    decision_file: ReviewDecisionFile | dict[str, object],
    *,
    require_complete: bool = False,
) -> ValidatedDecisionPlan:
    """Rebuild the packet and bind every editable field back to its candidate."""
    parsed = (
        decision_file
        if isinstance(decision_file, ReviewDecisionFile)
        else parse_decision_payload(decision_file)
    )
    packet = build_review_packet(session, catalogue)
    resolved = validate_candidate_catalogue(session, catalogue)
    if parsed.methodology_version != packet["methodology_version"]:
        raise ValueError("decision methodology_version does not match rebuilt packet")
    if parsed.packet_sha256 != packet["packet_sha256"]:
        raise ValueError("decision packet_sha256 does not match rebuilt packet")
    if parsed.candidate_catalogue_sha256 != packet["candidate_catalogue_sha256"]:
        raise ValueError("decision candidate_catalogue_sha256 does not match rebuilt packet")

    by_id = {item.candidate.candidate_id: item for item in resolved}
    supplied_ids = {decision.candidate_id for decision in parsed.decisions}
    expected_ids = set(by_id)
    if supplied_ids != expected_ids:
        missing = sorted(expected_ids - supplied_ids)
        unexpected = sorted(supplied_ids - expected_ids)
        details = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError(
            "decision file must contain exactly every packet candidate: " + "; ".join(details)
        )

    validated = []
    for decision in parsed.decisions:
        item = by_id[decision.candidate_id]
        candidate = item.candidate
        if (decision.source_id, decision.issue_key, decision.proposed_statement) != (
            candidate.source_id,
            candidate.issue_key,
            candidate.statement,
        ):
            raise ValueError(
                f"decision identity fields do not match candidate {decision.candidate_id}"
            )
        if require_complete and decision.outcome is None:
            raise ValueError(f"candidate {decision.candidate_id} has not been decided")
        if decision.revision is not None:
            source = _SOURCES_BY_ID[candidate.source_id]
            if decision.revision.topic_key not in source.topic_allowlist:
                raise ValueError(
                    f"revision topic {decision.revision.topic_key!r} is outside "
                    f"{candidate.source_id} policy"
                )
            if _revision_semantics(decision.revision) == _candidate_semantics(candidate):
                raise ValueError(f"candidate {decision.candidate_id} revision is a no-op")
            if (
                decision.revision.claim_type == "forecast"
                and decision.revision.target_end is not None
                and decision.revision.target_end <= item.document.available_at.date()
            ):
                raise ValueError(
                    f"candidate {decision.candidate_id} revision forecast target_end "
                    "must follow report availability"
                )
        validated.append(ValidatedDecision(decision=decision, resolved=item))
    return ValidatedDecisionPlan(parsed, packet, tuple(validated))


def _citation_drafts(item: ResolvedCandidate) -> tuple[CitationDraft, ...]:
    return tuple(
        CitationDraft(
            extraction_id=item.document.extraction_id,
            pdf_page_start=resolved.citation.pdf_page_start,
            pdf_page_end=resolved.citation.pdf_page_end,
            evidence_excerpt=resolved.citation.evidence_excerpt,
            support_role=resolved.citation.support_role,
            printed_locator=resolved.citation.printed_locator,
            section_title=resolved.citation.section_title,
        )
        for resolved in item.citations
    )


def _original_claim_draft(item: ResolvedCandidate) -> ClaimDraft:
    candidate = item.candidate
    return ClaimDraft(
        claim_type=candidate.claim_type,
        statement=candidate.statement,
        topic_key=candidate.topic_key,
        geographies=candidate.geographies,
        citations=_citation_drafts(item),
        attribution_document_id=item.document.document_id,
        claim_series_key=candidate.claim_series_key,
        reference_start=candidate.reference_start,
        reference_end=candidate.reference_end,
        target_start=candidate.target_start,
        target_end=candidate.target_end,
        numeric_value=candidate.numeric_value,
        lower_bound=candidate.lower_bound,
        upper_bound=candidate.upper_bound,
        unit=candidate.unit,
        condition_text=candidate.condition_text,
    )


def _revised_claim_draft(item: ResolvedCandidate, revision: SemanticRevision) -> ClaimDraft:
    return ClaimDraft(
        claim_type=revision.claim_type,
        statement=revision.statement,
        topic_key=revision.topic_key,
        geographies=revision.geographies,
        citations=_citation_drafts(item),
        attribution_document_id=item.document.document_id,
        claim_series_key=revision.claim_series_key,
        reference_start=revision.reference_start,
        reference_end=revision.reference_end,
        target_start=revision.target_start,
        target_end=revision.target_end,
        numeric_value=revision.numeric_value,
        lower_bound=revision.lower_bound,
        upper_bound=revision.upper_bound,
        unit=revision.unit,
        condition_text=revision.condition_text,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _decision_payload(decision: CandidateDecision) -> dict[str, object]:
    payload = asdict(decision)
    payload["attestations"] = asdict(decision.attestations)
    payload["revision"] = (
        _revision_semantics(decision.revision) if decision.revision is not None else None
    )
    return payload


def decision_file_sha256(
    decision_file: ReviewDecisionFile | dict[str, object],
) -> str:
    """Fingerprint the complete semantic decision document canonically."""
    parsed = (
        decision_file
        if isinstance(decision_file, ReviewDecisionFile)
        else parse_decision_payload(decision_file)
    )
    payload = {
        "schema_version": parsed.schema_version,
        "decision_kind": parsed.decision_kind,
        "methodology_version": parsed.methodology_version,
        "packet_sha256": parsed.packet_sha256,
        "candidate_catalogue_sha256": parsed.candidate_catalogue_sha256,
        "decisions": [_decision_payload(decision) for decision in parsed.decisions],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _request_sha256(
    plan: ValidatedDecisionPlan,
    decision: CandidateDecision,
    *,
    reviewer: str,
) -> str:
    payload = {
        "schema_version": REPORT_DECISION_SCHEMA_VERSION,
        "packet_sha256": plan.decision_file.packet_sha256,
        "candidate_catalogue_sha256": plan.decision_file.candidate_catalogue_sha256,
        "decision": _decision_payload(decision),
        "reviewer": reviewer,
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _reviewer(value: str) -> str:
    if not isinstance(value, str) or _HUMAN_REVIEWER_RE.fullmatch(value) is None:
        raise ValueError("reviewer must be a concrete human:<id> identity")
    return value


def _reviewed_at(value: datetime, proposal: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("reviewed_at must be a timezone-aware datetime")
    result = value.astimezone(UTC)
    if result < proposal.astimezone(UTC):
        raise ValueError("reviewed_at cannot be earlier than proposal")
    return result


def preflight_review_decisions(
    session: Session,
    plan: ValidatedDecisionPlan,
    *,
    reviewer: str,
) -> ReviewApplicationResult | None:
    """Return an exact replay receipt, reject conflicts, or permit a new batch."""
    reviewer_name = _reviewer(reviewer)
    if session.new or session.dirty or session.deleted:
        raise ValueError("review session must not contain unrelated pending changes")
    if any(item.decision.outcome is None for item in plan.decisions):
        raise ValueError("review preflight requires a complete decision plan")

    existing_by_candidate = {
        row.candidate_id: row
        for row in session.execute(
            select(ReportCandidateReview).where(
                ReportCandidateReview.candidate_id.in_(
                    item.decision.candidate_id for item in plan.decisions
                )
            )
        ).scalars()
    }
    if not existing_by_candidate:
        return None

    requests = {
        item.decision.candidate_id: _request_sha256(
            plan,
            item.decision,
            reviewer=reviewer_name,
        )
        for item in plan.decisions
    }
    if len(existing_by_candidate) != len(plan.decisions):
        raise ValueError("decision set conflicts with a partially applied packet")
    for candidate_id, row in existing_by_candidate.items():
        if row.request_sha256 != requests[candidate_id]:
            raise ValueError(f"candidate {candidate_id} already has a different review")
    return ReviewApplicationResult(
        packet_sha256=plan.decision_file.packet_sha256,
        results=tuple(
            AppliedDecision(
                candidate_id=item.decision.candidate_id,
                outcome=item.decision.outcome or "",
                review_id=existing_by_candidate[item.decision.candidate_id].id,
                original_draft_claim_id=(
                    existing_by_candidate[item.decision.candidate_id].original_draft_claim_id
                ),
                revised_draft_claim_id=(
                    existing_by_candidate[item.decision.candidate_id].revised_draft_claim_id
                ),
                verified_claim_id=(
                    existing_by_candidate[item.decision.candidate_id].verified_claim_id
                ),
                replayed=True,
            )
            for item in plan.decisions
        ),
    )


def apply_review_decisions(
    session: Session,
    catalogue: CandidateCatalogue,
    decision_file: ReviewDecisionFile | dict[str, object],
    *,
    reviewer: str,
    reviewed_at: datetime,
) -> ReviewApplicationResult:
    """Apply a complete human decision set atomically, or replay it exactly."""
    reviewer_name = _reviewer(reviewer)
    checked_at = _reviewed_at(reviewed_at, catalogue.created_at)
    if session.new or session.dirty or session.deleted:
        raise ValueError("review session must not contain unrelated pending changes")
    try:
        plan = validate_review_decisions(
            session,
            catalogue,
            decision_file,
            require_complete=True,
        )
        replay = preflight_review_decisions(
            session,
            plan,
            reviewer=reviewer_name,
        )
        if replay is not None:
            session.rollback()
            return replay
        requests = {
            item.decision.candidate_id: _request_sha256(
                plan,
                item.decision,
                reviewer=reviewer_name,
            )
            for item in plan.decisions
        }

        results: list[AppliedDecision] = []
        for item in plan.decisions:
            decision = item.decision
            assert decision.outcome is not None
            original_id = _append_claims(
                session,
                [_original_claim_draft(item.resolved)],
                created_by=catalogue.created_by,
                created_at=catalogue.created_at,
            )[0]
            revised_id = None
            verified_id = None
            if decision.outcome == "approve":
                verified_id, _created = _append_verified_claim(
                    session,
                    original_id,
                    reviewer=reviewer_name,
                    reviewed_at=checked_at,
                )
            elif decision.outcome == "revise":
                assert decision.revision is not None
                revised_id = _append_claims(
                    session,
                    [_revised_claim_draft(item.resolved, decision.revision)],
                    created_by=reviewer_name,
                    created_at=checked_at,
                )[0]
                verified_id, _created = _append_verified_claim(
                    session,
                    revised_id,
                    reviewer=reviewer_name,
                    reviewed_at=checked_at,
                )
            row = ReportCandidateReview(
                candidate_id=decision.candidate_id,
                request_sha256=requests[decision.candidate_id],
                packet_sha256=plan.decision_file.packet_sha256,
                candidate_catalogue_sha256=plan.decision_file.candidate_catalogue_sha256,
                outcome=decision.outcome,
                checklist_json=_canonical_json(asdict(decision.attestations)),
                reason_code=decision.reason_code,
                review_note=decision.review_note,
                candidate_json=_canonical_json(
                    _original_candidate_payload(item.resolved.candidate)
                ),
                reviewer=reviewer_name,
                reviewed_at=checked_at,
                recorded_at=checked_at,
                original_draft_claim_id=original_id,
                revised_draft_claim_id=revised_id,
                verified_claim_id=verified_id,
            )
            session.add(row)
            session.flush()
            results.append(
                AppliedDecision(
                    candidate_id=decision.candidate_id,
                    outcome=decision.outcome,
                    review_id=row.id,
                    original_draft_claim_id=original_id,
                    revised_draft_claim_id=revised_id,
                    verified_claim_id=verified_id,
                    replayed=False,
                )
            )
        session.commit()
    except Exception:
        session.rollback()
        raise
    return ReviewApplicationResult(plan.decision_file.packet_sha256, tuple(results))


__all__ = [
    "AppliedDecision",
    "CandidateDecision",
    "DECISION_OUTCOMES",
    "DECISION_REASON_CODES",
    "DecisionAttestations",
    "REPORT_DECISION_KIND",
    "REPORT_DECISION_SCHEMA_VERSION",
    "ReviewApplicationResult",
    "ReviewDecisionFile",
    "SemanticRevision",
    "ValidatedDecision",
    "ValidatedDecisionPlan",
    "apply_review_decisions",
    "build_decision_template",
    "decision_file_sha256",
    "load_decision_file",
    "parse_decision_payload",
    "preflight_review_decisions",
    "validate_review_decisions",
]
