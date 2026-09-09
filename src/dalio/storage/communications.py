"""Rights-gated metadata storage for institutional communications.

This first slice records exact event and artifact metadata only.  The checked-in
source catalogue currently authorizes no content acquisition, so this module
fails closed instead of accepting transcript, caption, audio, or report bytes.
The database schema already reserves immutable artifact/extraction/segment rows
for a later acquisition slice after an exact source policy is cleared.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, date, datetime
from urllib.parse import urlsplit

from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.communications.catalogue import (
    CommunicationCatalogueSnapshot,
    CommunicationSourceSpec,
    resolve_communication_catalogue_snapshot,
)
from dalio.storage.db import (
    CommunicationArtifact,
    CommunicationArtifactContent,
    CommunicationArtifactRetrieval,
    CommunicationArtifactSectionScopeSet,
    CommunicationEvent,
    CommunicationSourcePolicySnapshot,
    Organization,
    OrganizationCommodityCoverage,
)

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_ARTIFACT_ROLES = frozenset(
    {
        "prepared_remarks",
        "q_and_a_transcript",
        "full_transcript",
        "ceo_letter",
        "chair_letter",
        "annual_report",
        "subtitles",
        "webcast_video",
    }
)
_ORIGIN_TYPES = frozenset(
    {
        "publisher_authored",
        "official_published_transcript",
        "official_published_media",
        "official_hosted_vendor",
        "official_caption",
        "automatic_caption",
        "local_asr",
    }
)
_TRANSLATION_STATUSES = frozenset({"original", "official_translation"})
_TRANSCRIBER_ATTRIBUTIONS = frozenset(
    {"artifact_specific", "named_third_party", "not_applicable", "not_disclosed", "publisher"}
)
_EXPOSURE_ROLES = frozenset({"producer", "processor", "trader", "consumer", "integrated"})
_ROLE_MATERIALS = {
    "prepared_remarks": {
        "financial_results",
        "management_review",
        "monetary_policy_statement",
        "speech_text",
    },
    "q_and_a_transcript": {"questions_and_answers"},
    "full_transcript": {"press_conference_transcript", "results_transcript"},
    "ceo_letter": {"ceo_letter"},
    "chair_letter": {"annual_report", "management_review"},
    "annual_report": {"annual_report"},
    "subtitles": {"subtitles"},
    "webcast_video": {"press_conference_video"},
}
_ORIGIN_PROVENANCE = {
    "publisher_authored": "official_authored_text",
    "official_published_transcript": "official_published_transcript",
    "official_published_media": "official_published_media",
    "official_hosted_vendor": "official_hosted_third_party",
    "official_caption": "official_caption",
    "automatic_caption": "official_hosted_automatic_caption",
    "local_asr": "local_derived_asr",
}
_MATERIAL_ORIGINS = {
    "annual_report": frozenset({"publisher_authored"}),
    "ceo_letter": frozenset({"publisher_authored"}),
    "financial_results": frozenset({"publisher_authored"}),
    "management_review": frozenset({"publisher_authored"}),
    "monetary_policy_statement": frozenset({"publisher_authored"}),
    "speech_text": frozenset({"publisher_authored"}),
    "questions_and_answers": frozenset({"official_published_transcript", "official_hosted_vendor"}),
    "press_conference_transcript": frozenset(
        {"official_published_transcript", "official_hosted_vendor"}
    ),
    "results_transcript": frozenset({"official_published_transcript", "official_hosted_vendor"}),
    "subtitles": frozenset({"official_caption", "automatic_caption", "local_asr"}),
    "press_conference_video": frozenset({"official_published_media"}),
}
_ROLE_SCOPE_KEYS = {
    "prepared_remarks": "prepared_remarks",
    "q_and_a_transcript": "q_and_a",
    "full_transcript": "full_transcript",
    "ceo_letter": "ceo_letter",
    "chair_letter": "chair_letter",
    "annual_report": "annual_report",
    "subtitles": "subtitles",
    "webcast_video": "webcast_video",
}
_SECTION_SCOPE_CANONICALIZATION = "communication_artifact_section_scopes_json_v1"
_EXPLICIT_SECTION_SCOPE_SOURCE_IDS = frozenset({"ecb_monetary_policy_press_conferences_en"})


@dataclass(frozen=True)
class CommunicationEventMeta:
    organization_id: str
    event_key: str
    event_type: str
    title: str
    event_date: date
    metadata_known_at: datetime
    event_started_at: datetime | None = None
    reference_start: date | None = None
    reference_end: date | None = None


@dataclass(frozen=True)
class CommunicationArtifactMeta:
    source_id: str
    catalogue_sha256: str
    artifact_key: str
    artifact_role: str
    material_type: str
    language: str
    translation_status: str
    mime_type: str
    origin_type: str
    provenance_tier: str
    host_organization: str
    publisher: str
    transcriber: str | None
    transcriber_attribution: str
    rights_status: str
    acquisition_status: str
    rights_checked_by: str
    rights_checked_at: datetime
    available_at: datetime
    retrieved_at: datetime
    metadata_known_at: datetime
    landing_url: str
    artifact_url: str
    published_at: datetime | None = None


@dataclass(frozen=True)
class CommunicationSectionScopeMeta:
    section_ordinal: int
    scope_key: str
    artifact_role: str
    material_type: str
    origin_type: str
    provenance_tier: str
    transcriber: str | None
    transcriber_attribution: str


@dataclass(frozen=True)
class CommodityCoverageMeta:
    source_id: str
    catalogue_sha256: str
    organization_id: str
    coverage_key: str
    commodity_family: str
    exposure_role: str
    effective_from: date
    available_at: datetime
    retrieved_at: datetime
    metadata_known_at: datetime
    evidence_url: str
    effective_to: date | None = None
    published_at: datetime | None = None
    supersedes_coverage_id: int | None = None


@dataclass(frozen=True)
class CommunicationMetadataResult:
    event_id: int
    artifact_id: int
    event_created: bool
    artifact_created: bool
    event_version_sha256: str
    artifact_version_sha256: str
    supersedes_event_id: int | None
    supersedes_artifact_id: int | None


@dataclass(frozen=True)
class CommunicationSectionScopeSetResult:
    artifact_id: int
    scope_count: int
    scope_set_sha256: str
    created: bool


def _required(value: str, field: str) -> str:
    clean = value.strip()
    if not clean:
        raise ValueError(f"{field} must not be empty")
    return clean


def _identifier(value: str, field: str, *, max_length: int = 96) -> str:
    clean = _required(value, field)
    if len(clean) > max_length or _ID.fullmatch(clean) is None:
        raise ValueError(f"{field} must be lowercase snake_case")
    return clean


def _choice(value: str, allowed: frozenset[str], field: str) -> str:
    clean = _required(value, field)
    if clean not in allowed:
        raise ValueError(f"{field} must be one of {sorted(allowed)}")
    return clean


def _utc_naive(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("communication timestamps must include a timezone")
    return value.astimezone(UTC).replace(tzinfo=None)


def _clocks(
    published_at: datetime | None,
    available_at: datetime,
    retrieved_at: datetime,
    metadata_known_at: datetime,
) -> tuple[datetime | None, datetime, datetime, datetime]:
    published = _utc_naive(published_at) if published_at is not None else None
    available = _utc_naive(available_at)
    retrieved = _utc_naive(retrieved_at)
    known = _utc_naive(metadata_known_at)
    if published is not None and available < published:
        raise ValueError("available_at cannot be earlier than published_at")
    if retrieved < available:
        raise ValueError("retrieved_at cannot be earlier than available_at")
    if known < retrieved:
        raise ValueError("metadata_known_at cannot be earlier than retrieved_at")
    return published, available, retrieved, known


def _official_url(url: str, source: CommunicationSourceSpec, field: str) -> str:
    clean = _required(url, field)
    if any(ord(character) < 33 or ord(character) == 127 for character in clean):
        raise ValueError(f"{field} must not contain whitespace or control characters")
    if "\\" in clean:
        raise ValueError(f"{field} must not contain backslashes")
    parsed = urlsplit(clean)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{field} must be an HTTPS URL")
    if parsed.username or parsed.password:
        raise ValueError(f"{field} must not contain credentials")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port") from exc
    if port is not None:
        raise ValueError(f"{field} must not contain an explicit port")
    host = parsed.hostname.lower()
    if host.endswith("."):
        raise ValueError(f"{field} must not use a trailing-dot hostname")
    if host == "localhost":
        raise ValueError(f"{field} must not use localhost")
    if any(_DNS_LABEL.fullmatch(label) is None for label in host.split(".")):
        raise ValueError(f"{field} has an invalid DNS hostname")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{field} must not use an IP address")
    if not any(host == domain or host.endswith(f".{domain}") for domain in source.official_domains):
        raise ValueError(f"{field} must stay on a source-catalogue official domain")
    return clean


def _source(
    source_id: str,
    catalogue_sha256: str,
) -> tuple[CommunicationSourceSpec, CommunicationCatalogueSnapshot]:
    source_key = _identifier(source_id, "source_id")
    if _SHA256.fullmatch(catalogue_sha256) is None:
        raise ValueError("catalogue_sha256 must be 64 lowercase hexadecimal characters")
    catalogue = resolve_communication_catalogue_snapshot(catalogue_sha256)
    source = next(
        (candidate for candidate in catalogue.sources if candidate.source_id == source_key),
        None,
    )
    if source is None:
        raise ValueError(
            f"source_id is not in the bound communication catalogue snapshot: {source_key}"
        )
    return source, catalogue


def _canonical_sha256(value: dict) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _section_scope_dict(scope: CommunicationSectionScopeMeta) -> dict[str, object]:
    return {
        "section_ordinal": scope.section_ordinal,
        "scope_key": scope.scope_key,
        "artifact_role": scope.artifact_role,
        "material_type": scope.material_type,
        "origin_type": scope.origin_type,
        "provenance_tier": scope.provenance_tier,
        "transcriber": scope.transcriber,
        "transcriber_attribution": scope.transcriber_attribution,
    }


def _section_scopes_json(scopes: tuple[CommunicationSectionScopeMeta, ...]) -> str:
    return json.dumps(
        [_section_scope_dict(scope) for scope in scopes],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _section_scope_set_sha256(
    artifact_version_sha256: str,
    scopes: tuple[CommunicationSectionScopeMeta, ...],
) -> str:
    return _canonical_sha256(
        {
            "artifact_version_sha256": artifact_version_sha256,
            "canonicalization_version": _SECTION_SCOPE_CANONICALIZATION,
            "section_scopes": [_section_scope_dict(scope) for scope in scopes],
        }
    )


def _normalize_scope_transcriber(
    *,
    origin_type: str,
    publisher: str,
    transcriber: str | None,
    transcriber_attribution: str,
) -> tuple[str | None, str]:
    attribution = _choice(
        transcriber_attribution,
        _TRANSCRIBER_ATTRIBUTIONS,
        "section scope transcriber_attribution",
    )
    if attribution == "artifact_specific":
        raise ValueError("section scope transcriber attribution must be resolved")
    normalized = _required(transcriber, "section scope transcriber") if transcriber else None
    if origin_type in {"publisher_authored", "official_published_media"}:
        if normalized is not None or attribution != "not_applicable":
            raise ValueError("publisher-authored scope and official media have no transcriber")
    elif origin_type in {"official_hosted_vendor", "automatic_caption", "local_asr"}:
        if normalized is None or normalized == publisher or attribution != "named_third_party":
            raise ValueError("vendor, automatic-caption and ASR scopes need a named third party")
    elif attribution == "not_disclosed":
        if normalized is not None:
            raise ValueError("not_disclosed section scope requires an unset transcriber")
    elif attribution == "publisher":
        if normalized != publisher:
            raise ValueError("publisher section-scope transcriber must equal artifact publisher")
    elif attribution == "named_third_party":
        if normalized is None or normalized == publisher:
            raise ValueError("named third-party scope transcriber must differ from publisher")
    else:
        raise ValueError("section scope origin and transcriber attribution are incompatible")
    return normalized, attribution


def _normalize_section_scopes(
    scopes: Sequence[CommunicationSectionScopeMeta],
    artifact: CommunicationArtifactMeta | CommunicationArtifact,
    allowed_material_types: frozenset[str] | set[str],
) -> tuple[CommunicationSectionScopeMeta, ...]:
    if isinstance(scopes, (str, bytes)) or not isinstance(scopes, Sequence):
        raise ValueError("section_scopes must be an ordered sequence")
    if not scopes or len(scopes) > 32:
        raise ValueError("section_scopes must contain between 1 and 32 entries")
    normalized: list[CommunicationSectionScopeMeta] = []
    seen_keys: set[str] = set()
    for expected_ordinal, value in enumerate(scopes, start=1):
        if not isinstance(value, CommunicationSectionScopeMeta):
            raise ValueError("section_scopes entries must be CommunicationSectionScopeMeta")
        if (
            isinstance(value.section_ordinal, bool)
            or not isinstance(value.section_ordinal, int)
            or value.section_ordinal != expected_ordinal
        ):
            raise ValueError("section scope ordinals must be dense and follow sequence order")
        role = _choice(value.artifact_role, _ARTIFACT_ROLES, "section scope artifact_role")
        scope_key = _identifier(value.scope_key, "section scope_key")
        if scope_key != _ROLE_SCOPE_KEYS[role]:
            raise ValueError("section scope_key and artifact_role are incompatible")
        if scope_key in seen_keys:
            raise ValueError("section scope keys must be unique within an artifact")
        seen_keys.add(scope_key)
        material_type = _identifier(
            value.material_type,
            "section scope material_type",
            max_length=48,
        )
        if material_type not in _ROLE_MATERIALS[role]:
            raise ValueError("section scope artifact_role and material_type are incompatible")
        if material_type not in allowed_material_types:
            raise ValueError("section scope material_type is not declared by the source policy")
        origin_type = _choice(value.origin_type, _ORIGIN_TYPES, "section scope origin_type")
        provenance_tier = _required(value.provenance_tier, "section scope provenance_tier")
        if provenance_tier != _ORIGIN_PROVENANCE[origin_type]:
            raise ValueError("section scope origin_type and provenance_tier are incompatible")
        if origin_type not in _MATERIAL_ORIGINS[material_type]:
            raise ValueError("section scope material_type and origin_type are incompatible")
        if origin_type == "local_asr":
            raise ValueError("local ASR scopes need the later content-acquisition path")
        transcriber, attribution = _normalize_scope_transcriber(
            origin_type=origin_type,
            publisher=artifact.publisher,
            transcriber=value.transcriber,
            transcriber_attribution=value.transcriber_attribution,
        )
        normalized.append(
            CommunicationSectionScopeMeta(
                section_ordinal=expected_ordinal,
                scope_key=scope_key,
                artifact_role=role,
                material_type=material_type,
                origin_type=origin_type,
                provenance_tier=provenance_tier,
                transcriber=transcriber,
                transcriber_attribution=attribution,
            )
        )
    artifact_semantics = (
        artifact.artifact_role,
        artifact.material_type,
        artifact.origin_type,
        artifact.provenance_tier,
        artifact.transcriber,
        artifact.transcriber_attribution,
    )
    if not any(
        (
            scope.artifact_role,
            scope.material_type,
            scope.origin_type,
            scope.provenance_tier,
            scope.transcriber,
            scope.transcriber_attribution,
        )
        == artifact_semantics
        for scope in normalized
    ):
        raise ValueError("one section scope must anchor the artifact's primary semantics")
    return tuple(normalized)


def _default_section_scopes(
    artifact: CommunicationArtifactMeta | CommunicationArtifact,
) -> tuple[CommunicationSectionScopeMeta, ...]:
    return (
        CommunicationSectionScopeMeta(
            section_ordinal=1,
            scope_key=_ROLE_SCOPE_KEYS[artifact.artifact_role],
            artifact_role=artifact.artifact_role,
            material_type=artifact.material_type,
            origin_type=artifact.origin_type,
            provenance_tier=artifact.provenance_tier,
            transcriber=artifact.transcriber,
            transcriber_attribution=artifact.transcriber_attribution,
        ),
    )


def _exact_date(value: date, field: str) -> date:
    if isinstance(value, datetime) or not isinstance(value, date):
        raise ValueError(f"{field} must be an exact date, not a datetime")
    return value


def _normalize_event(meta: CommunicationEventMeta) -> CommunicationEventMeta:
    event_date = _exact_date(meta.event_date, "event_date")
    reference_start = (
        _exact_date(meta.reference_start, "reference_start")
        if meta.reference_start is not None
        else None
    )
    reference_end = (
        _exact_date(meta.reference_end, "reference_end") if meta.reference_end is not None else None
    )
    if (meta.reference_start is None) != (meta.reference_end is None):
        raise ValueError("reference_start and reference_end must be provided together")
    if meta.reference_start is not None and meta.reference_end < meta.reference_start:
        raise ValueError("reference_end cannot be earlier than reference_start")
    return replace(
        meta,
        organization_id=_identifier(meta.organization_id, "organization_id"),
        event_key=_identifier(meta.event_key, "event_key", max_length=128),
        event_type=_identifier(meta.event_type, "event_type", max_length=64),
        title=_required(meta.title, "title"),
        event_date=event_date,
        metadata_known_at=_utc_naive(meta.metadata_known_at),
        event_started_at=(
            _utc_naive(meta.event_started_at) if meta.event_started_at is not None else None
        ),
        reference_start=reference_start,
        reference_end=reference_end,
    )


def _event_version(meta: CommunicationEventMeta) -> str:
    return _canonical_sha256(
        {
            "organization_id": meta.organization_id,
            "event_key": meta.event_key,
            "event_type": meta.event_type,
            "title": meta.title,
            "event_date": meta.event_date.isoformat(),
            "event_started_at": meta.event_started_at.isoformat()
            if meta.event_started_at
            else None,
            "reference_start": meta.reference_start.isoformat() if meta.reference_start else None,
            "reference_end": meta.reference_end.isoformat() if meta.reference_end else None,
        }
    )


def _ensure_organization(session: Session, organization_id: str) -> None:
    if session.get(Organization, organization_id) is None:
        session.add(Organization(organization_id=organization_id))
        session.flush()


def _source_policy_values(
    source: CommunicationSourceSpec,
    catalogue: CommunicationCatalogueSnapshot,
) -> dict[str, object]:
    """Return the exact validated policy snapshot persisted for a source."""
    payload = asdict(source)
    for field in ("commodity_families", "material_types", "official_domains"):
        payload[field] = sorted(payload[field])
    checked_at = source.rights_checked_at
    checked_at_text = (
        checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
        if checked_at is not None
        else None
    )
    catalogue_evaluated_at = _utc_naive(catalogue.evaluated_at)
    catalogue_evaluated_at_text = (
        catalogue.evaluated_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
    )
    payload["rights_checked_at"] = checked_at_text
    payload["catalogue_sha256"] = catalogue.catalogue_sha256
    payload["catalogue_evaluated_at"] = catalogue_evaluated_at_text
    policy_sha256 = _canonical_sha256(payload)
    return {
        "catalogue_sha256": catalogue.catalogue_sha256,
        "source_id": source.source_id,
        "organization_id": source.organization_id,
        "organization_name": source.organization_name,
        "organization_type": source.organization_type,
        "jurisdiction": source.jurisdiction,
        "language": source.language,
        "landing_url": source.landing_url,
        "official_domains_json": json.dumps(sorted(source.official_domains), separators=(",", ":")),
        "host_organization": source.host_organization,
        "publisher": source.publisher,
        "transcriber": source.transcriber,
        "transcriber_attribution": source.transcriber_attribution,
        "material_types_json": json.dumps(sorted(source.material_types), separators=(",", ":")),
        "commodity_families_json": json.dumps(
            sorted(source.commodity_families), separators=(",", ":")
        ),
        "verified_archive_start_year": source.verified_archive_start_year,
        "coverage_note": source.coverage_note,
        "source_provenance_tier": source.provenance_tier,
        "rights_status": source.rights_status,
        "rights_basis_url": source.rights_basis_url,
        "rights_note": source.rights_note,
        "acquisition_status": source.acquisition_status,
        "acquisition_note": source.acquisition_note,
        "automated_collection_allowed": source.automated_collection_allowed,
        "rights_checked_by": source.rights_checked_by,
        "rights_checked_at": (
            _utc_naive(source.rights_checked_at) if source.rights_checked_at is not None else None
        ),
        "catalogue_evaluated_at": catalogue_evaluated_at,
        "policy_sha256": policy_sha256,
    }


def _ensure_source_policy_snapshot(
    session: Session,
    source: CommunicationSourceSpec,
    catalogue: CommunicationCatalogueSnapshot,
) -> CommunicationSourcePolicySnapshot:
    """Flush a validated policy snapshot; the caller owns the transaction."""
    values = _source_policy_values(source, catalogue)
    key = (catalogue.catalogue_sha256, source.source_id)
    existing = session.get(CommunicationSourcePolicySnapshot, key)
    if existing is not None:
        for field, expected in values.items():
            if getattr(existing, field) != expected:
                raise ValueError(f"persisted source policy snapshot differs at {field}")
        return existing
    snapshot = CommunicationSourcePolicySnapshot(**values)
    session.add(snapshot)
    session.flush()
    return snapshot


def _append_event(
    session: Session,
    meta: CommunicationEventMeta,
) -> tuple[CommunicationEvent, bool]:
    version = _event_version(meta)
    prior = (
        session.execute(
            select(CommunicationEvent)
            .where(
                CommunicationEvent.organization_id == meta.organization_id,
                CommunicationEvent.event_key == meta.event_key,
            )
            .order_by(CommunicationEvent.metadata_known_at.desc(), CommunicationEvent.id.desc())
        )
        .scalars()
        .first()
    )
    if prior is not None and prior.event_version_sha256 == version:
        if meta.metadata_known_at < prior.metadata_known_at:
            raise ValueError("event metadata_known_at cannot move backward")
        return prior, False
    if prior is not None and meta.metadata_known_at <= prior.metadata_known_at:
        raise ValueError("an event correction must have a later metadata_known_at")
    event = CommunicationEvent(
        organization_id=meta.organization_id,
        event_key=meta.event_key,
        event_type=meta.event_type,
        title=meta.title,
        event_date=meta.event_date,
        event_started_at=meta.event_started_at,
        reference_start=meta.reference_start,
        reference_end=meta.reference_end,
        metadata_known_at=meta.metadata_known_at,
        event_version_sha256=version,
        supersedes_event_id=prior.id if prior is not None else None,
    )
    session.add(event)
    session.flush()
    return event, True


def _normalize_artifact(
    meta: CommunicationArtifactMeta,
    source: CommunicationSourceSpec,
) -> CommunicationArtifactMeta:
    published, available, retrieved, known = _clocks(
        meta.published_at,
        meta.available_at,
        meta.retrieved_at,
        meta.metadata_known_at,
    )
    role = _choice(meta.artifact_role, _ARTIFACT_ROLES, "artifact_role")
    material_type = _identifier(meta.material_type, "material_type", max_length=48)
    if material_type not in _ROLE_MATERIALS[role]:
        raise ValueError("artifact_role and material_type are incompatible")
    if material_type not in source.material_types:
        raise ValueError(f"material_type is not declared by source {source.source_id}")
    if meta.language != source.language:
        raise ValueError("artifact language conflicts with the source catalogue")
    if meta.rights_status != source.rights_status:
        raise ValueError("artifact rights_status conflicts with the source catalogue")
    if meta.acquisition_status != source.acquisition_status:
        raise ValueError("artifact acquisition_status conflicts with the source catalogue")
    transcriber_attribution = _choice(
        meta.transcriber_attribution,
        _TRANSCRIBER_ATTRIBUTIONS,
        "transcriber_attribution",
    )
    if transcriber_attribution == "artifact_specific":
        raise ValueError("artifact_specific must be resolved before artifact persistence")
    transcriber = _required(meta.transcriber, "transcriber") if meta.transcriber else None
    publisher = _required(meta.publisher, "publisher")
    host_organization = _required(meta.host_organization, "host_organization")
    origin_type = _choice(meta.origin_type, _ORIGIN_TYPES, "origin_type")
    provenance_tier = _required(meta.provenance_tier, "provenance_tier")
    if provenance_tier != _ORIGIN_PROVENANCE[origin_type]:
        raise ValueError("origin_type and provenance_tier are incompatible")
    if origin_type not in _MATERIAL_ORIGINS[material_type]:
        raise ValueError("material_type and origin_type are incompatible")
    if origin_type in {"publisher_authored", "official_published_media"}:
        if transcriber is not None or transcriber_attribution != "not_applicable":
            raise ValueError("publisher-authored text and official media have no transcriber")
    elif origin_type in {"official_hosted_vendor", "automatic_caption", "local_asr"}:
        if (
            transcriber is None
            or transcriber == publisher
            or transcriber_attribution != "named_third_party"
        ):
            raise ValueError("vendor, automatic-caption and ASR origins need a named third party")
    elif transcriber_attribution == "not_disclosed":
        if transcriber is not None:
            raise ValueError("not_disclosed requires an unset transcriber")
    elif transcriber_attribution == "publisher":
        if transcriber != publisher:
            raise ValueError("publisher transcriber must equal the artifact publisher")
    elif transcriber_attribution == "named_third_party":
        if transcriber is None or transcriber == publisher:
            raise ValueError("named_third_party must differ from the publisher")
    else:
        raise ValueError("origin_type and transcriber attribution are incompatible")
    if origin_type == "local_asr":
        raise ValueError("local ASR metadata needs the later content-acquisition path")
    rights_checked_at = _utc_naive(meta.rights_checked_at)
    if known < rights_checked_at:
        raise ValueError("metadata_known_at cannot be earlier than rights_checked_at")
    return replace(
        meta,
        source_id=source.source_id,
        artifact_key=_identifier(meta.artifact_key, "artifact_key", max_length=128),
        artifact_role=role,
        material_type=material_type,
        language=meta.language.lower(),
        translation_status=_choice(
            meta.translation_status, _TRANSLATION_STATUSES, "translation_status"
        ),
        mime_type=_required(meta.mime_type, "mime_type").lower(),
        origin_type=origin_type,
        provenance_tier=provenance_tier,
        host_organization=host_organization,
        publisher=publisher,
        transcriber=transcriber,
        transcriber_attribution=transcriber_attribution,
        rights_checked_by=_human(meta.rights_checked_by),
        rights_checked_at=rights_checked_at,
        published_at=published,
        available_at=available,
        retrieved_at=retrieved,
        metadata_known_at=known,
        landing_url=_official_url(meta.landing_url, source, "landing_url"),
        artifact_url=_official_url(meta.artifact_url, source, "artifact_url"),
    )


def _human(value: str) -> str:
    clean = _required(value, "rights_checked_by")
    if not clean.startswith("human:") or clean == "human:":
        raise ValueError("rights_checked_by must be human:<id>")
    return clean


def _artifact_version(
    event_version_sha256: str,
    meta: CommunicationArtifactMeta | CommunicationArtifact,
    section_scopes: tuple[CommunicationSectionScopeMeta, ...],
) -> str:
    return _canonical_sha256(
        {
            "event_version_sha256": event_version_sha256,
            "source_id": meta.source_id,
            "catalogue_sha256": meta.catalogue_sha256,
            "artifact_key": meta.artifact_key,
            "artifact_role": meta.artifact_role,
            "material_type": meta.material_type,
            "language": meta.language,
            "translation_status": meta.translation_status,
            "mime_type": meta.mime_type,
            "origin_type": meta.origin_type,
            "provenance_tier": meta.provenance_tier,
            "rights_status": meta.rights_status,
            "acquisition_status": meta.acquisition_status,
            "rights_checked_by": meta.rights_checked_by,
            "rights_checked_at": meta.rights_checked_at.isoformat(),
            "host_organization": meta.host_organization,
            "publisher": meta.publisher,
            "transcriber": meta.transcriber,
            "transcriber_attribution": meta.transcriber_attribution,
            "published_at": meta.published_at.isoformat() if meta.published_at else None,
            "available_at": meta.available_at.isoformat(),
            "section_scopes": [_section_scope_dict(scope) for scope in section_scopes],
            # A repeat retrieval does not create another public artifact version.
            "landing_url": meta.landing_url,
            "artifact_url": meta.artifact_url,
        }
    )


def _policy_material_types(
    session: Session,
    artifact: CommunicationArtifact,
) -> frozenset[str]:
    policy = session.get(
        CommunicationSourcePolicySnapshot,
        (artifact.catalogue_sha256, artifact.source_id),
    )
    if policy is None:
        raise ValueError("artifact source policy snapshot is missing")
    try:
        values = json.loads(policy.material_types_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact source policy material_types_json is invalid") from exc
    if (
        not isinstance(values, list)
        or not values
        or any(not isinstance(value, str) for value in values)
        or values != sorted(set(values))
    ):
        raise ValueError("artifact source policy material_types_json is not canonical")
    return frozenset(values)


def _validated_stored_scope_set(
    session: Session,
    artifact: CommunicationArtifact,
    stored: CommunicationArtifactSectionScopeSet,
) -> tuple[CommunicationSectionScopeMeta, ...]:
    try:
        payload = json.loads(stored.scopes_json)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("stored communication section scopes are invalid JSON") from exc
    if not isinstance(payload, list):
        raise ValueError("stored communication section scopes must be a JSON array")
    exact_fields = {
        "section_ordinal",
        "scope_key",
        "artifact_role",
        "material_type",
        "origin_type",
        "provenance_tier",
        "transcriber",
        "transcriber_attribution",
    }
    if any(not isinstance(value, dict) or set(value) != exact_fields for value in payload):
        raise ValueError("stored communication section scopes have an unknown shape")
    try:
        decoded = tuple(CommunicationSectionScopeMeta(**value) for value in payload)
    except TypeError as exc:
        raise ValueError("stored communication section scopes have invalid fields") from exc
    scopes = _normalize_section_scopes(
        decoded,
        artifact,
        _policy_material_types(session, artifact),
    )
    canonical_json = _section_scopes_json(scopes)
    expected_artifact_version = _artifact_version(
        artifact.event_version_sha256,
        artifact,
        scopes,
    )
    expected_scope_hash = _section_scope_set_sha256(expected_artifact_version, scopes)
    if (
        stored.artifact_id != artifact.id
        or stored.artifact_version_sha256 != artifact.artifact_version_sha256
        or expected_artifact_version != artifact.artifact_version_sha256
        or stored.scope_count != len(scopes)
        or stored.scopes_json != canonical_json
        or stored.scope_set_sha256 != expected_scope_hash
        or stored.canonicalization_version != _SECTION_SCOPE_CANONICALIZATION
        or stored.metadata_known_at < artifact.metadata_known_at
    ):
        raise ValueError("stored communication section-scope set fails semantic validation")
    return scopes


def record_communication_artifact_section_scopes(
    session: Session,
    *,
    artifact_id: int,
    section_scopes: Sequence[CommunicationSectionScopeMeta],
    metadata_known_at: datetime,
) -> CommunicationSectionScopeSetResult:
    """Flush one complete immutable semantic scope set; never capture content."""

    if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id < 1:
        raise ValueError("artifact_id must be a positive integer")
    known = _utc_naive(metadata_known_at)
    artifact = session.get(CommunicationArtifact, artifact_id)
    if artifact is None:
        raise ValueError("artifact_id does not exist")
    scopes = _normalize_section_scopes(
        section_scopes,
        artifact,
        _policy_material_types(session, artifact),
    )
    expected_artifact_version = _artifact_version(
        artifact.event_version_sha256,
        artifact,
        scopes,
    )
    if expected_artifact_version != artifact.artifact_version_sha256:
        raise ValueError("section scopes do not match the immutable artifact version")
    if known < artifact.metadata_known_at:
        raise ValueError("section-scope metadata cannot be known before artifact metadata")
    canonical_json = _section_scopes_json(scopes)
    scope_hash = _section_scope_set_sha256(artifact.artifact_version_sha256, scopes)
    existing = session.get(CommunicationArtifactSectionScopeSet, artifact_id)
    if existing is not None:
        persisted = _validated_stored_scope_set(session, artifact, existing)
        if known < existing.metadata_known_at:
            raise ValueError("section-scope metadata_known_at cannot move backward")
        if persisted != scopes or existing.scope_set_sha256 != scope_hash:
            raise ValueError(
                "artifact already has a different immutable section-scope set; "
                "append an artifact version"
            )
        return CommunicationSectionScopeSetResult(
            artifact_id=artifact.id,
            scope_count=len(scopes),
            scope_set_sha256=scope_hash,
            created=False,
        )
    has_content = session.scalar(
        select(CommunicationArtifactContent.id)
        .where(CommunicationArtifactContent.artifact_id == artifact.id)
        .limit(1)
    )
    if has_content is not None:
        raise ValueError("section scopes must be declared before artifact content")
    session.add(
        CommunicationArtifactSectionScopeSet(
            artifact_id=artifact.id,
            artifact_version_sha256=artifact.artifact_version_sha256,
            scope_count=len(scopes),
            scopes_json=canonical_json,
            scope_set_sha256=scope_hash,
            metadata_known_at=known,
            canonicalization_version=_SECTION_SCOPE_CANONICALIZATION,
        )
    )
    session.flush()
    return CommunicationSectionScopeSetResult(
        artifact_id=artifact.id,
        scope_count=len(scopes),
        scope_set_sha256=scope_hash,
        created=True,
    )


def get_communication_artifact_section_scopes(
    session: Session,
    artifact_id: int,
) -> tuple[CommunicationSectionScopeMeta, ...]:
    """Return a revalidated ordered scope declaration for a v2 artifact."""

    if isinstance(artifact_id, bool) or not isinstance(artifact_id, int) or artifact_id < 1:
        raise ValueError("artifact_id must be a positive integer")
    artifact = session.get(CommunicationArtifact, artifact_id)
    if artifact is None:
        raise ValueError("artifact_id does not exist")
    stored = session.get(CommunicationArtifactSectionScopeSet, artifact_id)
    if stored is None:
        raise ValueError("artifact section-scope set is missing")
    return _validated_stored_scope_set(session, artifact, stored)


def record_communication_artifact_metadata(
    session: Session,
    event_meta: CommunicationEventMeta,
    artifact_meta: CommunicationArtifactMeta,
    *,
    section_scopes: Sequence[CommunicationSectionScopeMeta] | None = None,
) -> CommunicationMetadataResult:
    """Flush catalogue-bound link metadata without acquiring source content.

    This helper never commits or rolls back.  The caller owns the surrounding
    transaction and must commit all flushed rows together or roll them back.
    """
    source, catalogue = _source(artifact_meta.source_id, artifact_meta.catalogue_sha256)
    event_input = _normalize_event(event_meta)
    if event_input.organization_id != source.organization_id:
        raise ValueError("source_id does not belong to the event organization")
    artifact_input = _normalize_artifact(artifact_meta, source)
    if section_scopes is None and source.source_id in _EXPLICIT_SECTION_SCOPE_SOURCE_IDS:
        raise ValueError(
            f"section_scopes must be explicit for mixed-representation source {source.source_id}"
        )
    normalized_scopes = _normalize_section_scopes(
        section_scopes if section_scopes is not None else _default_section_scopes(artifact_input),
        artifact_input,
        source.material_types,
    )
    if (
        source.source_id in _EXPLICIT_SECTION_SCOPE_SOURCE_IDS
        and artifact_input.artifact_role == "q_and_a_transcript"
        and artifact_input.material_type == "questions_and_answers"
        and tuple(scope.scope_key for scope in normalized_scopes) != ("prepared_remarks", "q_and_a")
    ):
        raise ValueError(
            "mixed ECB Q&A representation must declare ordered prepared_remarks and q_and_a scopes"
        )
    _ensure_organization(session, source.organization_id)
    _ensure_source_policy_snapshot(session, source, catalogue)
    event, event_created = _append_event(session, event_input)
    if artifact_input.metadata_known_at < event.metadata_known_at:
        raise ValueError("artifact metadata cannot be known before its event metadata")
    version = _artifact_version(event.event_version_sha256, artifact_input, normalized_scopes)
    prior = (
        session.execute(
            select(CommunicationArtifact)
            .join(
                CommunicationEvent,
                CommunicationEvent.id == CommunicationArtifact.event_id,
            )
            .where(
                CommunicationEvent.organization_id == event.organization_id,
                CommunicationEvent.event_key == event.event_key,
                CommunicationArtifact.source_id == source.source_id,
                CommunicationArtifact.artifact_key == artifact_input.artifact_key,
            )
            .order_by(
                CommunicationArtifact.metadata_known_at.desc(),
                CommunicationArtifact.id.desc(),
            )
        )
        .scalars()
        .first()
    )
    if prior is not None and prior.artifact_version_sha256 == version:
        if artifact_input.metadata_known_at < prior.metadata_known_at:
            raise ValueError("artifact metadata_known_at cannot move backward")
        artifact = prior
        artifact_created = False
    else:
        if prior is not None and artifact_input.metadata_known_at <= prior.metadata_known_at:
            raise ValueError("an artifact correction must have a later metadata_known_at")
        artifact = CommunicationArtifact(
            event_id=event.id,
            source_id=source.source_id,
            catalogue_sha256=artifact_input.catalogue_sha256,
            event_version_sha256=event.event_version_sha256,
            artifact_key=artifact_input.artifact_key,
            artifact_role=artifact_input.artifact_role,
            material_type=artifact_input.material_type,
            language=artifact_input.language,
            translation_status=artifact_input.translation_status,
            mime_type=artifact_input.mime_type,
            origin_type=artifact_input.origin_type,
            provenance_tier=artifact_input.provenance_tier,
            rights_status=source.rights_status,
            acquisition_status=source.acquisition_status,
            rights_basis_url=source.rights_basis_url,
            rights_note=source.rights_note,
            rights_checked_by=artifact_input.rights_checked_by,
            rights_checked_at=artifact_input.rights_checked_at,
            host_organization=artifact_input.host_organization,
            publisher=artifact_input.publisher,
            transcriber=artifact_input.transcriber,
            transcriber_attribution=artifact_input.transcriber_attribution,
            published_at=artifact_input.published_at,
            available_at=artifact_input.available_at,
            retrieved_at=artifact_input.retrieved_at,
            metadata_known_at=artifact_input.metadata_known_at,
            landing_url=artifact_input.landing_url,
            artifact_url=artifact_input.artifact_url,
            artifact_version_sha256=version,
            supersedes_artifact_id=prior.id if prior is not None else None,
        )
        session.add(artifact)
        session.flush()
        artifact_created = True
    record_communication_artifact_section_scopes(
        session,
        artifact_id=artifact.id,
        section_scopes=normalized_scopes,
        metadata_known_at=artifact_input.metadata_known_at.replace(tzinfo=UTC),
    )
    retrieval = session.execute(
        select(CommunicationArtifactRetrieval).where(
            CommunicationArtifactRetrieval.artifact_id == artifact.id,
            CommunicationArtifactRetrieval.retrieved_at == artifact_input.retrieved_at,
        )
    ).scalar_one_or_none()
    if retrieval is not None and artifact_input.metadata_known_at < retrieval.metadata_known_at:
        raise ValueError("retrieval metadata_known_at cannot move backward")
    if retrieval is None:
        session.add(
            CommunicationArtifactRetrieval(
                artifact_id=artifact.id,
                retrieved_at=artifact_input.retrieved_at,
                metadata_known_at=artifact_input.metadata_known_at,
                landing_url=artifact_input.landing_url,
                artifact_url=artifact_input.artifact_url,
            )
        )
        session.flush()
    return CommunicationMetadataResult(
        event_id=event.id,
        artifact_id=artifact.id,
        event_created=event_created,
        artifact_created=artifact_created,
        event_version_sha256=event.event_version_sha256,
        artifact_version_sha256=artifact.artifact_version_sha256,
        supersedes_event_id=event.supersedes_event_id,
        supersedes_artifact_id=artifact.supersedes_artifact_id,
    )


def append_catalogue_commodity_coverage(
    session: Session,
    meta: CommodityCoverageMeta,
) -> int:
    """Flush an effective-dated selection-taxonomy version.

    Changed semantics require an explicit predecessor.  The caller owns commit
    and rollback; this helper only validates, appends, and flushes.
    """
    source, catalogue = _source(meta.source_id, meta.catalogue_sha256)
    organization_id = _identifier(meta.organization_id, "organization_id")
    coverage_key = _identifier(meta.coverage_key, "coverage_key", max_length=128)
    family = _identifier(meta.commodity_family, "commodity_family")
    role = _choice(meta.exposure_role, _EXPOSURE_ROLES, "exposure_role")
    if organization_id != source.organization_id:
        raise ValueError("source_id does not belong to the coverage organization")
    if family not in source.commodity_families:
        raise ValueError("commodity_family is not declared by the source catalogue")
    effective_from = _exact_date(meta.effective_from, "effective_from")
    effective_to = (
        _exact_date(meta.effective_to, "effective_to") if meta.effective_to is not None else None
    )
    if effective_to is not None and effective_to < effective_from:
        raise ValueError("effective_to cannot be earlier than effective_from")
    published, available, retrieved, known = _clocks(
        meta.published_at,
        meta.available_at,
        meta.retrieved_at,
        meta.metadata_known_at,
    )
    evidence_url = _official_url(meta.evidence_url, source, "evidence_url")
    semantic = {
        "organization_id": organization_id,
        "coverage_key": coverage_key,
        "commodity_family": family,
        "exposure_role": role,
        "mapping_status": "selection_taxonomy",
        "effective_from": effective_from.isoformat(),
        "effective_to": effective_to.isoformat() if effective_to else None,
        "source_id": source.source_id,
        "catalogue_sha256": meta.catalogue_sha256,
        "evidence_url": evidence_url,
        "evidence_note": source.coverage_note,
        "published_at": published.isoformat() if published else None,
        "available_at": available.isoformat(),
        # Coverage has no separate recurrence ledger; retain a changed retrieval
        # as an explicit version rather than silently discarding it.
        "retrieved_at": retrieved.isoformat(),
    }
    coverage_version_sha256 = _canonical_sha256(semantic)
    _ensure_organization(session, organization_id)
    _ensure_source_policy_snapshot(session, source, catalogue)
    head = (
        session.execute(
            select(OrganizationCommodityCoverage)
            .where(
                OrganizationCommodityCoverage.organization_id == organization_id,
                OrganizationCommodityCoverage.coverage_key == coverage_key,
            )
            .order_by(
                OrganizationCommodityCoverage.metadata_known_at.desc(),
                OrganizationCommodityCoverage.id.desc(),
            )
        )
        .scalars()
        .first()
    )
    if head is not None and head.coverage_version_sha256 == coverage_version_sha256:
        if known < head.metadata_known_at:
            raise ValueError("coverage metadata_known_at cannot move backward")
        return head.id
    prior = None
    if meta.supersedes_coverage_id is not None:
        prior = session.get(OrganizationCommodityCoverage, meta.supersedes_coverage_id)
        if prior is None:
            raise ValueError("supersedes_coverage_id does not exist")
        if (
            prior.organization_id,
            prior.coverage_key,
        ) != (organization_id, coverage_key):
            raise ValueError("coverage successor must keep organization and coverage_key")
        if head is None or prior.id != head.id:
            raise ValueError("supersedes_coverage_id must identify the current coverage head")
        successor = session.execute(
            select(OrganizationCommodityCoverage.id).where(
                OrganizationCommodityCoverage.supersedes_exposure_id == prior.id
            )
        ).scalar_one_or_none()
        if successor is not None:
            raise ValueError("supersedes_coverage_id already has a successor")
        if known <= prior.metadata_known_at:
            raise ValueError("a coverage correction must have a later metadata_known_at")
    else:
        if head is not None:
            raise ValueError("changed coverage requires supersedes_coverage_id")
    superseded_ids = {
        int(value)
        for value in session.scalars(
            select(OrganizationCommodityCoverage.supersedes_exposure_id).where(
                OrganizationCommodityCoverage.supersedes_exposure_id.is_not(None)
            )
        )
    }
    other_heads = session.scalars(
        select(OrganizationCommodityCoverage).where(
            OrganizationCommodityCoverage.organization_id == organization_id,
            OrganizationCommodityCoverage.commodity_family == family,
            OrganizationCommodityCoverage.exposure_role == role,
        )
    ).all()
    requested_end = effective_to or date.max
    for head in other_heads:
        if head.id in superseded_ids or (prior is not None and head.id == prior.id):
            continue
        head_end = head.effective_to or date.max
        if head.effective_from <= requested_end and effective_from <= head_end:
            raise ValueError(
                "coverage period overlaps another current lineage for this family and role"
            )
    coverage = OrganizationCommodityCoverage(
        organization_id=organization_id,
        coverage_key=coverage_key,
        commodity_family=family,
        exposure_role=role,
        mapping_status="selection_taxonomy",
        effective_from=effective_from,
        effective_to=effective_to,
        source_id=source.source_id,
        catalogue_sha256=meta.catalogue_sha256,
        evidence_url=evidence_url,
        evidence_note=source.coverage_note,
        published_at=published,
        available_at=available,
        retrieved_at=retrieved,
        metadata_known_at=known,
        coverage_version_sha256=coverage_version_sha256,
        supersedes_exposure_id=prior.id if prior is not None else None,
        reviewed_by=None,
        reviewed_at=None,
    )
    session.add(coverage)
    session.flush()
    return coverage.id


__all__ = [
    "CommodityCoverageMeta",
    "CommunicationArtifactMeta",
    "CommunicationEventMeta",
    "CommunicationMetadataResult",
    "CommunicationSectionScopeMeta",
    "CommunicationSectionScopeSetResult",
    "append_catalogue_commodity_coverage",
    "get_communication_artifact_section_scopes",
    "record_communication_artifact_section_scopes",
    "record_communication_artifact_metadata",
]
