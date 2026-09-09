"""Strict, offline manifest for the Fed/ECB communications metadata pilot.

The manifest is a pre-ingest observation record.  It deliberately contains no
source bytes, content digest, rights decision, or permission to collect content.
Exact event denominators and one selected representation per event make coverage
gaps measurable before any text analysis is attempted.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import urlsplit

from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_SOURCES,
    CommunicationSourceSpec,
)

PILOT_MANIFEST_SCHEMA_VERSION = 1
PILOT_METHODOLOGY_VERSION = "communication-metadata-pilot-v2"

_PILOT_ORGANIZATIONS = frozenset({"federal_reserve", "ecb"})
_PILOT_START_DATE = date(2025, 1, 1)
_PILOT_END_DATE = date(2025, 12, 31)
_EXPECTED_EVENT_KEYS = {
    "federal_reserve": frozenset(
        {
            "fomc_2025_01_29",
            "fomc_2025_03_19",
            "fomc_2025_05_07",
            "fomc_2025_06_18",
            "fomc_2025_07_30",
            "fomc_2025_09_17",
            "fomc_2025_10_29",
            "fomc_2025_12_10",
        }
    ),
    "ecb": frozenset(
        {
            "ecb_2025_01_30",
            "ecb_2025_03_06",
            "ecb_2025_04_17",
            "ecb_2025_06_05",
            "ecb_2025_07_24",
            "ecb_2025_09_11",
            "ecb_2025_10_30",
            "ecb_2025_12_18",
        }
    ),
}
_EXPECTED_DENOMINATOR_URLS = {
    "federal_reserve": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    "ecb": "https://www.ecb.europa.eu/press/tvservices/webcast/html/index.en.html",
}
_EXPECTED_RIGHTS_BASIS_URLS = {
    "federal_reserve": "https://www.federalreserve.gov/disclaimer.htm",
    "ecb": "https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html",
}
_SOURCES_BY_ID = {source.source_id: source for source in COMMUNICATION_SOURCES}
_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTOR_RE = re.compile(r"^(?:agent|human|model):[a-z0-9][a-z0-9_.-]*$")
_DNS_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MIME_RE = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "methodology_version",
        "catalogue_sha256",
        "pilot_id",
        "created_by",
        "created_at",
        "as_known_at",
        "scope",
        "events",
    }
)
_SCOPE_FIELDS = frozenset({"start_date", "end_date", "organizations"})
_DENOMINATOR_FIELDS = frozenset(
    {
        "organization_id",
        "denominator_source_url",
        "checked_by",
        "checked_at",
        "event_keys",
        "representation_spec",
        "rights_review",
    }
)
_REPRESENTATION_SPEC_FIELDS = frozenset(
    {"representation_key", "source_id", "artifact_role", "material_type", "section_coverage"}
)
_RIGHTS_REVIEW_FIELDS = frozenset({"status", "basis_url", "summary", "questions"})
_EVENT_FIELDS = frozenset(
    {
        "event_key",
        "organization_id",
        "event_type",
        "title",
        "event_date",
        "event_started_at",
        "metadata_known_at",
        "representation",
    }
)
_REPRESENTATION_FIELDS = frozenset(
    {
        "representation_key",
        "availability_status",
        "checked_at",
        "status_evidence_url",
        "status_note",
        "section_coverage",
        "candidate",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "artifact_key",
        "source_id",
        "artifact_role",
        "material_type",
        "language",
        "translation_status",
        "mime_type",
        "origin_type",
        "provenance_tier",
        "host_organization",
        "publisher",
        "transcriber",
        "transcriber_attribution",
        "published_at",
        "available_at",
        "retrieved_at",
        "metadata_known_at",
        "landing_url",
        "artifact_url",
    }
)

# This module is intentionally a narrowly defined pilot, not a general-purpose
# communications manifest language.  Changing a selected representation is a
# methodology change and therefore requires a new methodology version.
_EXPECTED_REPRESENTATIONS = {
    "federal_reserve": {
        "representation_key": "official_transcript_en",
        "source_id": "fed_fomc_press_conferences_en",
        "artifact_role": "full_transcript",
        "material_type": "press_conference_transcript",
        "section_coverage": ("full_transcript",),
        "mime_type": "application/pdf",
    },
    "ecb": {
        "representation_key": "official_statement_with_q_and_a_en",
        "source_id": "ecb_monetary_policy_press_conferences_en",
        "artifact_role": "q_and_a_transcript",
        "material_type": "questions_and_answers",
        "section_coverage": ("prepared_remarks", "q_and_a"),
        "mime_type": "text/html",
    },
}

_EXPECTED_EVENT_LOCATORS = {
    "fomc_2025_01_29": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250129.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250129.pdf",
    ),
    "fomc_2025_03_19": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250319.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250319.pdf",
    ),
    "fomc_2025_05_07": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250507.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250507.pdf",
    ),
    "fomc_2025_06_18": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250618.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250618.pdf",
    ),
    "fomc_2025_07_30": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250730.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250730.pdf",
    ),
    "fomc_2025_09_17": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250917.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250917.pdf",
    ),
    "fomc_2025_10_29": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20251029.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20251029.pdf",
    ),
    "fomc_2025_12_10": (
        "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20251210.htm",
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20251210.pdf",
    ),
    "ecb_2025_01_30": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250130~1f418aa0f4.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250130~1f418aa0f4.en.html",
    ),
    "ecb_2025_03_06": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250306~4307bd0941.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250306~4307bd0941.en.html",
    ),
    "ecb_2025_04_17": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250417~091c625eb6.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250417~091c625eb6.en.html",
    ),
    "ecb_2025_06_05": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250605~f00a36ef2b.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250605~f00a36ef2b.en.html",
    ),
    "ecb_2025_07_24": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250724~a66e730494.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250724~a66e730494.en.html",
    ),
    "ecb_2025_09_11": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250911~a13675b834.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is250911~a13675b834.en.html",
    ),
    "ecb_2025_10_30": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is251030~4f74dde15e.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is251030~4f74dde15e.en.html",
    ),
    "ecb_2025_12_18": (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is251218~3a10402adb.en.html",
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/"
        "html/ecb.is251218~3a10402adb.en.html",
    ),
}


@dataclass(frozen=True)
class RightsReview:
    status: str
    basis_url: str
    summary: str
    questions: tuple[str, ...]


@dataclass(frozen=True)
class RepresentationSpec:
    representation_key: str
    source_id: str
    artifact_role: str
    material_type: str
    section_coverage: tuple[str, ...]


@dataclass(frozen=True)
class OrganizationDenominator:
    organization_id: str
    denominator_source_url: str
    checked_by: str
    checked_at: datetime
    event_keys: tuple[str, ...]
    representation_spec: RepresentationSpec
    rights_review: RightsReview


@dataclass(frozen=True)
class PilotScope:
    start_date: date
    end_date: date
    organizations: tuple[OrganizationDenominator, ...]


@dataclass(frozen=True)
class ArtifactCandidate:
    artifact_key: str
    source_id: str
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
    published_at: datetime | None
    available_at: datetime
    retrieved_at: datetime
    metadata_known_at: datetime
    landing_url: str
    artifact_url: str


@dataclass(frozen=True)
class CandidateRepresentation:
    representation_key: str
    availability_status: str
    checked_at: datetime
    status_evidence_url: str
    status_note: str
    section_coverage: tuple[str, ...]
    candidate: ArtifactCandidate


@dataclass(frozen=True)
class PilotEvent:
    event_key: str
    organization_id: str
    event_type: str
    title: str
    event_date: date
    event_started_at: datetime | None
    metadata_known_at: datetime
    representation: CandidateRepresentation


@dataclass(frozen=True)
class CommunicationPilotManifest:
    schema_version: int
    methodology_version: str
    catalogue_sha256: str
    pilot_id: str
    created_by: str
    created_at: datetime
    as_known_at: datetime
    scope: PilotScope
    events: tuple[PilotEvent, ...]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant: {value}")


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _array(value: object, field: str, *, nonempty: bool = True) -> list[object]:
    if not isinstance(value, list) or (nonempty and not value):
        suffix = "a non-empty array" if nonempty else "an array"
        raise ValueError(f"{field} must be {suffix}")
    return value


def _exact_fields(payload: dict[str, object], expected: frozenset[str], field: str) -> None:
    supplied = set(payload)
    missing = sorted(expected - supplied)
    unknown = sorted(supplied - expected)
    if missing:
        raise ValueError(f"{field} has missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")


def _string(value: object, field: str, *, max_length: int = 2_000) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{field} must not have leading or trailing whitespace")
    if len(value) > max_length:
        raise ValueError(f"{field} is longer than {max_length} characters")
    return value


def _nullable_string(value: object, field: str, *, max_length: int = 192) -> str | None:
    if value is None:
        return None
    return _string(value, field, max_length=max_length)


def _identifier(value: object, field: str, *, max_length: int = 128) -> str:
    raw = _string(value, field, max_length=max_length)
    if _ID_RE.fullmatch(raw) is None:
        raise ValueError(f"{field} must be lowercase snake_case")
    return raw


def _actor(value: object, field: str) -> str:
    raw = _string(value, field, max_length=192)
    if _ACTOR_RE.fullmatch(raw) is None:
        raise ValueError(f"{field} must be a namespaced agent:, model:, or human: identifier")
    return raw


def _date(value: object, field: str) -> date:
    raw = _string(value, field, max_length=40)
    try:
        parsed = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date") from exc
    if raw != parsed.isoformat():
        raise ValueError(f"{field} must be an ISO YYYY-MM-DD date")
    return parsed


def _datetime(value: object, field: str, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    raw = _string(value, field, max_length=40)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return parsed.astimezone(UTC)


def _distinct_strings(
    value: object,
    field: str,
    *,
    identifiers: bool = False,
) -> tuple[str, ...]:
    items = _array(value, field)
    parsed = tuple(
        _identifier(item, f"{field}[{index}]")
        if identifiers
        else _string(item, f"{field}[{index}]")
        for index, item in enumerate(items)
    )
    if len(parsed) != len(set(parsed)):
        raise ValueError(f"{field} must not contain duplicates")
    return parsed


def _official_url(value: object, source: CommunicationSourceSpec, field: str) -> str:
    raw = _string(value, field, max_length=4_096)
    if any(ord(character) <= 32 or ord(character) == 127 for character in raw):
        raise ValueError(f"{field} contains whitespace or control characters")
    if "\\" in raw:
        raise ValueError(f"{field} must not contain a backslash")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid URL authority") from exc
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError(f"{field} must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{field} must not contain credentials")
    if parsed.hostname.endswith("."):
        raise ValueError(f"{field} must not use a trailing-dot hostname")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{field} has an invalid port") from exc
    if port is not None:
        raise ValueError(f"{field} must not contain an explicit port")
    host = parsed.hostname.lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{field} must not use an IP address")
    if host == "localhost" or "." not in host:
        raise ValueError(f"{field} must not use a local hostname")
    if len(host) > 253 or any(_DNS_LABEL_RE.fullmatch(part) is None for part in host.split(".")):
        raise ValueError(f"{field} has an invalid DNS hostname")
    if not any(host == domain or host.endswith(f".{domain}") for domain in source.official_domains):
        raise ValueError(f"{field} is outside the source official-domain allowlist")
    return raw


def _source(source_id: str, organization_id: str, field: str) -> CommunicationSourceSpec:
    source = _SOURCES_BY_ID.get(source_id)
    if source is None:
        raise ValueError(f"{field} is not in the communication source catalogue")
    if source.organization_id != organization_id:
        raise ValueError(f"{field} does not belong to {organization_id}")
    if source.rights_status != "rights_review_required":
        raise ValueError(f"{field} is not pending a source rights review")
    if source.acquisition_status != "manual_review_required":
        raise ValueError(f"{field} is not gated for manual review")
    if source.automated_collection_allowed:
        raise ValueError(f"{field} unexpectedly allows automated collection")
    return source


def _parse_representation_spec(
    value: object,
    *,
    organization_id: str,
    field: str,
) -> tuple[RepresentationSpec, CommunicationSourceSpec]:
    payload = _object(value, field)
    _exact_fields(payload, _REPRESENTATION_SPEC_FIELDS, field)
    spec = RepresentationSpec(
        representation_key=_identifier(
            payload["representation_key"], f"{field}.representation_key"
        ),
        source_id=_identifier(payload["source_id"], f"{field}.source_id", max_length=96),
        artifact_role=_identifier(
            payload["artifact_role"], f"{field}.artifact_role", max_length=32
        ),
        material_type=_identifier(
            payload["material_type"], f"{field}.material_type", max_length=48
        ),
        section_coverage=_distinct_strings(
            payload["section_coverage"], f"{field}.section_coverage", identifiers=True
        ),
    )
    expected = _EXPECTED_REPRESENTATIONS[organization_id]
    supplied = {
        "representation_key": spec.representation_key,
        "source_id": spec.source_id,
        "artifact_role": spec.artifact_role,
        "material_type": spec.material_type,
        "section_coverage": tuple(spec.section_coverage),
    }
    expected_without_derived = {key: item for key, item in expected.items() if key != "mime_type"}
    if supplied != expected_without_derived:
        raise ValueError(
            f"{field} does not match the selected {organization_id} pilot representation"
        )
    source = _source(spec.source_id, organization_id, f"{field}.source_id")
    if spec.material_type not in source.material_types:
        raise ValueError(f"{field}.material_type is not declared by its source")
    return spec, source


def _parse_rights_review(
    value: object,
    *,
    source: CommunicationSourceSpec,
    field: str,
) -> RightsReview:
    payload = _object(value, field)
    _exact_fields(payload, _RIGHTS_REVIEW_FIELDS, field)
    status = _string(payload["status"], f"{field}.status", max_length=32)
    if status != "pending":
        raise ValueError(f"{field}.status must remain pending")
    questions = _distinct_strings(payload["questions"], f"{field}.questions")
    return RightsReview(
        status=status,
        basis_url=_official_url(payload["basis_url"], source, f"{field}.basis_url"),
        summary=_string(payload["summary"], f"{field}.summary"),
        questions=questions,
    )


def _parse_denominator(value: object, index: int) -> OrganizationDenominator:
    field = f"scope.organizations[{index}]"
    payload = _object(value, field)
    _exact_fields(payload, _DENOMINATOR_FIELDS, field)
    organization_id = _identifier(
        payload["organization_id"], f"{field}.organization_id", max_length=96
    )
    if organization_id not in _PILOT_ORGANIZATIONS:
        raise ValueError(f"{field}.organization_id is outside the Fed/ECB pilot")
    spec, source = _parse_representation_spec(
        payload["representation_spec"],
        organization_id=organization_id,
        field=f"{field}.representation_spec",
    )
    checked_at = _datetime(payload["checked_at"], f"{field}.checked_at")
    assert checked_at is not None
    event_keys = _distinct_strings(payload["event_keys"], f"{field}.event_keys", identifiers=True)
    expected_event_keys = _EXPECTED_EVENT_KEYS[organization_id]
    if frozenset(event_keys) != expected_event_keys:
        missing = sorted(expected_event_keys - set(event_keys))
        unexpected = sorted(set(event_keys) - expected_event_keys)
        detail = []
        if missing:
            detail.append("missing " + ", ".join(missing))
        if unexpected:
            detail.append("unexpected " + ", ".join(unexpected))
        raise ValueError(
            f"{field}.event_keys does not match the fixed pilot denominator: " + "; ".join(detail)
        )
    denominator_source_url = _official_url(
        payload["denominator_source_url"], source, f"{field}.denominator_source_url"
    )
    if denominator_source_url != _EXPECTED_DENOMINATOR_URLS[organization_id]:
        raise ValueError(f"{field}.denominator_source_url is not the selected official archive")
    rights_review = _parse_rights_review(
        payload["rights_review"], source=source, field=f"{field}.rights_review"
    )
    if rights_review.basis_url != _EXPECTED_RIGHTS_BASIS_URLS[organization_id]:
        raise ValueError(f"{field}.rights_review.basis_url is not the selected official notice")
    return OrganizationDenominator(
        organization_id=organization_id,
        denominator_source_url=denominator_source_url,
        checked_by=_actor(payload["checked_by"], f"{field}.checked_by"),
        checked_at=checked_at,
        event_keys=event_keys,
        representation_spec=spec,
        rights_review=rights_review,
    )


def _parse_scope(value: object) -> PilotScope:
    payload = _object(value, "scope")
    _exact_fields(payload, _SCOPE_FIELDS, "scope")
    start_date = _date(payload["start_date"], "scope.start_date")
    end_date = _date(payload["end_date"], "scope.end_date")
    if end_date < start_date:
        raise ValueError("scope.end_date cannot be earlier than scope.start_date")
    if (start_date, end_date) != (_PILOT_START_DATE, _PILOT_END_DATE):
        raise ValueError("scope must be exactly the closed 2025 calendar year")
    organizations = tuple(
        _parse_denominator(item, index)
        for index, item in enumerate(_array(payload["organizations"], "scope.organizations"))
    )
    organization_ids = [item.organization_id for item in organizations]
    if len(organization_ids) != len(set(organization_ids)):
        raise ValueError("scope.organizations contains a duplicate organization_id")
    if set(organization_ids) != _PILOT_ORGANIZATIONS:
        raise ValueError("scope.organizations must contain exactly federal_reserve and ecb")
    return PilotScope(start_date=start_date, end_date=end_date, organizations=organizations)


def _parse_candidate(
    value: object,
    *,
    event_key: str,
    organization_id: str,
    spec: RepresentationSpec,
    representation_checked_at: datetime,
    event_date: date,
    field: str,
) -> ArtifactCandidate:
    payload = _object(value, field)
    _exact_fields(payload, _CANDIDATE_FIELDS, field)
    source_id = _identifier(payload["source_id"], f"{field}.source_id", max_length=96)
    source = _source(source_id, organization_id, f"{field}.source_id")
    artifact_role = _identifier(payload["artifact_role"], f"{field}.artifact_role", max_length=32)
    material_type = _identifier(payload["material_type"], f"{field}.material_type", max_length=48)
    if (source_id, artifact_role, material_type) != (
        spec.source_id,
        spec.artifact_role,
        spec.material_type,
    ):
        raise ValueError(f"{field} conflicts with the tracked representation specification")
    if material_type not in source.material_types:
        raise ValueError(f"{field}.material_type is not declared by its source")
    language = _string(payload["language"], f"{field}.language", max_length=16)
    if language != source.language:
        raise ValueError(f"{field}.language conflicts with its source catalogue")
    translation_status = _string(
        payload["translation_status"], f"{field}.translation_status", max_length=32
    )
    if translation_status != "original":
        raise ValueError(f"{field}.translation_status must be original in this pilot")
    mime_type = _string(payload["mime_type"], f"{field}.mime_type", max_length=96)
    if _MIME_RE.fullmatch(mime_type) is None:
        raise ValueError(f"{field}.mime_type is invalid")
    if mime_type != _EXPECTED_REPRESENTATIONS[organization_id]["mime_type"]:
        raise ValueError(f"{field}.mime_type conflicts with the selected representation")
    origin_type = _identifier(payload["origin_type"], f"{field}.origin_type", max_length=32)
    provenance_tier = _identifier(
        payload["provenance_tier"], f"{field}.provenance_tier", max_length=32
    )
    if (origin_type, provenance_tier) != (
        "official_published_transcript",
        "official_published_transcript",
    ):
        raise ValueError(f"{field} must be an official_published_transcript in this narrow pilot")
    transcriber = _nullable_string(payload["transcriber"], f"{field}.transcriber")
    transcriber_attribution = _string(
        payload["transcriber_attribution"],
        f"{field}.transcriber_attribution",
        max_length=32,
    )
    publisher = _string(payload["publisher"], f"{field}.publisher", max_length=192)
    host_organization = _string(
        payload["host_organization"], f"{field}.host_organization", max_length=192
    )
    if host_organization != source.host_organization or publisher != source.publisher:
        raise ValueError(f"{field} host and publisher must match the current source catalogue")
    if transcriber is not None or transcriber_attribution != "not_disclosed":
        raise ValueError(f"{field} must keep transcriber null and attribution not_disclosed")
    published_at = _datetime(payload["published_at"], f"{field}.published_at", nullable=True)
    available_at = _datetime(payload["available_at"], f"{field}.available_at")
    retrieved_at = _datetime(payload["retrieved_at"], f"{field}.retrieved_at")
    metadata_known_at = _datetime(payload["metadata_known_at"], f"{field}.metadata_known_at")
    assert available_at is not None and retrieved_at is not None and metadata_known_at is not None
    if published_at is not None and available_at < published_at:
        raise ValueError(f"{field}.available_at cannot precede published_at")
    if retrieved_at < available_at:
        raise ValueError(f"{field}.retrieved_at cannot precede available_at")
    if metadata_known_at < retrieved_at:
        raise ValueError(f"{field}.metadata_known_at cannot precede retrieved_at")
    if metadata_known_at > representation_checked_at:
        raise ValueError(f"{field}.metadata_known_at cannot follow representation checked_at")
    if available_at.date() < event_date:
        raise ValueError(f"{field}.available_at cannot precede the event date")
    if published_at is None and not (
        available_at == retrieved_at == metadata_known_at == representation_checked_at
    ):
        raise ValueError(
            f"{field} must use the observation time for all public clocks when published_at "
            "is unknown"
        )
    landing_url = _official_url(payload["landing_url"], source, f"{field}.landing_url")
    artifact_url = _official_url(payload["artifact_url"], source, f"{field}.artifact_url")
    expected_landing_url, expected_artifact_url = _EXPECTED_EVENT_LOCATORS[event_key]
    if (landing_url, artifact_url) != (expected_landing_url, expected_artifact_url):
        raise ValueError(f"{field} does not match the exact selected locator pair for {event_key}")
    return ArtifactCandidate(
        artifact_key=_identifier(payload["artifact_key"], f"{field}.artifact_key"),
        source_id=source_id,
        artifact_role=artifact_role,
        material_type=material_type,
        language=language,
        translation_status=translation_status,
        mime_type=mime_type,
        origin_type=origin_type,
        provenance_tier=provenance_tier,
        host_organization=host_organization,
        publisher=publisher,
        transcriber=transcriber,
        transcriber_attribution=transcriber_attribution,
        published_at=published_at,
        available_at=available_at,
        retrieved_at=retrieved_at,
        metadata_known_at=metadata_known_at,
        landing_url=landing_url,
        artifact_url=artifact_url,
    )


def _parse_representation(
    value: object,
    *,
    event_key: str,
    organization_id: str,
    spec: RepresentationSpec,
    event_date: date,
    event_metadata_known_at: datetime,
    field: str,
) -> CandidateRepresentation:
    payload = _object(value, field)
    _exact_fields(payload, _REPRESENTATION_FIELDS, field)
    representation_key = _identifier(payload["representation_key"], f"{field}.representation_key")
    if representation_key != spec.representation_key:
        raise ValueError(f"{field}.representation_key conflicts with its denominator")
    availability_status = _string(
        payload["availability_status"], f"{field}.availability_status", max_length=32
    )
    if availability_status != "available":
        raise ValueError(f"{field}.availability_status must be available in this pilot")
    checked_at = _datetime(payload["checked_at"], f"{field}.checked_at")
    assert checked_at is not None
    if checked_at < event_metadata_known_at:
        raise ValueError(f"{field}.checked_at cannot precede event metadata_known_at")
    section_coverage = _distinct_strings(
        payload["section_coverage"], f"{field}.section_coverage", identifiers=True
    )
    if tuple(section_coverage) != tuple(spec.section_coverage):
        raise ValueError(f"{field}.section_coverage conflicts with its denominator")
    source = _source(spec.source_id, organization_id, f"{field}.source_id")
    status_evidence_url = _official_url(
        payload["status_evidence_url"], source, f"{field}.status_evidence_url"
    )
    candidate = _parse_candidate(
        payload["candidate"],
        event_key=event_key,
        organization_id=organization_id,
        spec=spec,
        representation_checked_at=checked_at,
        event_date=event_date,
        field=f"{field}.candidate",
    )
    if status_evidence_url != candidate.landing_url:
        raise ValueError(f"{field}.status_evidence_url must equal the candidate landing_url")
    return CandidateRepresentation(
        representation_key=representation_key,
        availability_status=availability_status,
        checked_at=checked_at,
        status_evidence_url=status_evidence_url,
        status_note=_string(payload["status_note"], f"{field}.status_note"),
        section_coverage=section_coverage,
        candidate=candidate,
    )


def _parse_event(
    value: object,
    index: int,
    denominators: dict[str, OrganizationDenominator],
) -> PilotEvent:
    field = f"events[{index}]"
    payload = _object(value, field)
    _exact_fields(payload, _EVENT_FIELDS, field)
    organization_id = _identifier(
        payload["organization_id"], f"{field}.organization_id", max_length=96
    )
    denominator = denominators.get(organization_id)
    if denominator is None:
        raise ValueError(f"{field}.organization_id has no pilot denominator")
    event_key = _identifier(payload["event_key"], f"{field}.event_key")
    if event_key not in denominator.event_keys:
        raise ValueError(f"{field}.event_key is outside its fixed organization denominator")
    event_date = _date(payload["event_date"], f"{field}.event_date")
    expected_date = date(
        int(event_key[-10:-6]),
        int(event_key[-5:-3]),
        int(event_key[-2:]),
    )
    if event_date != expected_date:
        raise ValueError(f"{field}.event_date conflicts with event_key")
    event_started_at = _datetime(
        payload["event_started_at"], f"{field}.event_started_at", nullable=True
    )
    metadata_known_at = _datetime(payload["metadata_known_at"], f"{field}.metadata_known_at")
    assert metadata_known_at is not None
    if metadata_known_at.date() < event_date:
        raise ValueError(f"{field}.metadata_known_at cannot precede event_date")
    if event_started_at is not None and event_started_at > metadata_known_at:
        raise ValueError(f"{field}.event_started_at cannot follow metadata_known_at")
    if event_started_at is not None and event_started_at.date() != event_date:
        raise ValueError(f"{field}.event_started_at must fall on event_date in UTC")
    event_type = _identifier(payload["event_type"], f"{field}.event_type", max_length=64)
    if event_type != "monetary_policy_press_conference":
        raise ValueError(f"{field}.event_type is outside the pilot methodology")
    return PilotEvent(
        event_key=event_key,
        organization_id=organization_id,
        event_type=event_type,
        title=_string(payload["title"], f"{field}.title"),
        event_date=event_date,
        event_started_at=event_started_at,
        metadata_known_at=metadata_known_at,
        representation=_parse_representation(
            payload["representation"],
            event_key=event_key,
            organization_id=organization_id,
            spec=denominator.representation_spec,
            event_date=event_date,
            event_metadata_known_at=metadata_known_at,
            field=f"{field}.representation",
        ),
    )


def _validate_manifest_relationships(manifest: CommunicationPilotManifest) -> None:
    if manifest.created_at < manifest.as_known_at:
        raise ValueError("created_at cannot precede as_known_at")
    if manifest.scope.end_date >= manifest.as_known_at.date():
        raise ValueError("scope must be closed before as_known_at")
    denominators = {
        denominator.organization_id: denominator for denominator in manifest.scope.organizations
    }
    identities = [(event.organization_id, event.event_key) for event in manifest.events]
    if len(identities) != len(set(identities)):
        raise ValueError("events contains a duplicate organization/event identity")
    artifact_keys = [event.representation.candidate.artifact_key for event in manifest.events]
    if len(artifact_keys) != len(set(artifact_keys)):
        raise ValueError("events contains a duplicate candidate artifact_key")
    for organization_id, denominator in denominators.items():
        if denominator.checked_at > manifest.as_known_at:
            raise ValueError(f"{organization_id} denominator checked_at follows as_known_at")
        if denominator.checked_at.date() <= manifest.scope.end_date:
            raise ValueError(
                f"{organization_id} denominator was not checked after the closed scope"
            )
        expected = set(denominator.event_keys)
        actual = {
            event.event_key for event in manifest.events if event.organization_id == organization_id
        }
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            detail = []
            if missing:
                detail.append("missing " + ", ".join(missing))
            if unexpected:
                detail.append("unexpected " + ", ".join(unexpected))
            raise ValueError(f"{organization_id} event denominator mismatch: {'; '.join(detail)}")
    for event in manifest.events:
        if not manifest.scope.start_date <= event.event_date <= manifest.scope.end_date:
            raise ValueError(f"{event.event_key} event_date is outside the pilot scope")
        if event.metadata_known_at > manifest.as_known_at:
            raise ValueError(f"{event.event_key} metadata_known_at follows as_known_at")
        representation = event.representation
        if representation.checked_at > manifest.as_known_at:
            raise ValueError(f"{event.event_key} representation checked_at follows as_known_at")
        denominator = denominators[event.organization_id]
        if representation.checked_at > denominator.checked_at:
            raise ValueError(
                f"{event.event_key} representation checked_at follows denominator checked_at"
            )


def load_pilot_manifest(path: Path) -> CommunicationPilotManifest:
    """Load and validate one schema-exact, metadata-only pilot manifest."""
    source_path = Path(path)
    try:
        payload = json.loads(
            source_path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"could not read communication pilot manifest {source_path}: {exc}"
        ) from exc
    root = _object(payload, "communication pilot manifest")
    _exact_fields(root, _TOP_LEVEL_FIELDS, "communication pilot manifest")
    schema_version = root["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != PILOT_MANIFEST_SCHEMA_VERSION
    ):
        raise ValueError(f"schema_version must be exactly {PILOT_MANIFEST_SCHEMA_VERSION}")
    methodology_version = _string(root["methodology_version"], "methodology_version", max_length=96)
    if methodology_version != PILOT_METHODOLOGY_VERSION:
        raise ValueError(f"methodology_version must be {PILOT_METHODOLOGY_VERSION}")
    catalogue_sha256 = _string(root["catalogue_sha256"], "catalogue_sha256", max_length=64)
    if catalogue_sha256 != COMMUNICATION_CATALOGUE_SHA256:
        raise ValueError("catalogue_sha256 must bind the current communication source catalogue")
    scope = _parse_scope(root["scope"])
    denominators = {item.organization_id: item for item in scope.organizations}
    events = tuple(
        _parse_event(item, index, denominators)
        for index, item in enumerate(_array(root["events"], "events"))
    )
    created_at = _datetime(root["created_at"], "created_at")
    as_known_at = _datetime(root["as_known_at"], "as_known_at")
    assert created_at is not None and as_known_at is not None
    manifest = CommunicationPilotManifest(
        schema_version=schema_version,
        methodology_version=methodology_version,
        catalogue_sha256=catalogue_sha256,
        pilot_id=_identifier(root["pilot_id"], "pilot_id"),
        created_by=_actor(root["created_by"], "created_by"),
        created_at=created_at,
        as_known_at=as_known_at,
        scope=scope,
        events=events,
    )
    _validate_manifest_relationships(manifest)
    return manifest


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("manifest datetimes must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _candidate_payload(candidate: ArtifactCandidate) -> dict[str, object]:
    return {
        "artifact_key": candidate.artifact_key,
        "source_id": candidate.source_id,
        "artifact_role": candidate.artifact_role,
        "material_type": candidate.material_type,
        "language": candidate.language,
        "translation_status": candidate.translation_status,
        "mime_type": candidate.mime_type,
        "origin_type": candidate.origin_type,
        "provenance_tier": candidate.provenance_tier,
        "host_organization": candidate.host_organization,
        "publisher": candidate.publisher,
        "transcriber": candidate.transcriber,
        "transcriber_attribution": candidate.transcriber_attribution,
        "published_at": _iso_datetime(candidate.published_at),
        "available_at": _iso_datetime(candidate.available_at),
        "retrieved_at": _iso_datetime(candidate.retrieved_at),
        "metadata_known_at": _iso_datetime(candidate.metadata_known_at),
        "landing_url": candidate.landing_url,
        "artifact_url": candidate.artifact_url,
    }


def _manifest_payload(manifest: CommunicationPilotManifest) -> dict[str, object]:
    organizations = []
    for denominator in sorted(manifest.scope.organizations, key=lambda item: item.organization_id):
        spec = denominator.representation_spec
        organizations.append(
            {
                "organization_id": denominator.organization_id,
                "denominator_source_url": denominator.denominator_source_url,
                "checked_by": denominator.checked_by,
                "checked_at": _iso_datetime(denominator.checked_at),
                "event_keys": sorted(denominator.event_keys),
                "representation_spec": {
                    "representation_key": spec.representation_key,
                    "source_id": spec.source_id,
                    "artifact_role": spec.artifact_role,
                    "material_type": spec.material_type,
                    "section_coverage": list(spec.section_coverage),
                },
                "rights_review": {
                    "status": denominator.rights_review.status,
                    "basis_url": denominator.rights_review.basis_url,
                    "summary": denominator.rights_review.summary,
                    "questions": sorted(denominator.rights_review.questions),
                },
            }
        )
    events = []
    for event in sorted(manifest.events, key=lambda item: (item.organization_id, item.event_key)):
        representation = event.representation
        events.append(
            {
                "event_key": event.event_key,
                "organization_id": event.organization_id,
                "event_type": event.event_type,
                "title": event.title,
                "event_date": event.event_date.isoformat(),
                "event_started_at": _iso_datetime(event.event_started_at),
                "metadata_known_at": _iso_datetime(event.metadata_known_at),
                "representation": {
                    "representation_key": representation.representation_key,
                    "availability_status": representation.availability_status,
                    "checked_at": _iso_datetime(representation.checked_at),
                    "status_evidence_url": representation.status_evidence_url,
                    "status_note": representation.status_note,
                    "section_coverage": list(representation.section_coverage),
                    "candidate": _candidate_payload(representation.candidate),
                },
            }
        )
    return {
        "schema_version": manifest.schema_version,
        "methodology_version": manifest.methodology_version,
        "catalogue_sha256": manifest.catalogue_sha256,
        "pilot_id": manifest.pilot_id,
        "created_by": manifest.created_by,
        "created_at": _iso_datetime(manifest.created_at),
        "as_known_at": _iso_datetime(manifest.as_known_at),
        "scope": {
            "start_date": manifest.scope.start_date.isoformat(),
            "end_date": manifest.scope.end_date.isoformat(),
            "organizations": organizations,
        },
        "events": events,
    }


def pilot_manifest_sha256(manifest: CommunicationPilotManifest) -> str:
    """Return the deterministic semantic hash for a validated manifest."""
    if not isinstance(manifest, CommunicationPilotManifest):
        raise TypeError("manifest must be a CommunicationPilotManifest")
    canonical = json.dumps(
        _manifest_payload(manifest),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


# Descriptive aliases make call sites self-documenting while preserving the
# short API used by the review-packet pipeline.
load_communication_pilot_manifest = load_pilot_manifest
communication_pilot_manifest_sha256 = pilot_manifest_sha256


__all__ = [
    "PILOT_MANIFEST_SCHEMA_VERSION",
    "PILOT_METHODOLOGY_VERSION",
    "ArtifactCandidate",
    "CandidateRepresentation",
    "CommunicationPilotManifest",
    "OrganizationDenominator",
    "PilotEvent",
    "PilotScope",
    "RepresentationSpec",
    "RightsReview",
    "communication_pilot_manifest_sha256",
    "load_communication_pilot_manifest",
    "load_pilot_manifest",
    "pilot_manifest_sha256",
]
