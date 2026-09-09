"""Strict metadata-only manifests for one institution and one closed year.

This is deliberately separate from the fixed Fed/ECB pilot contract.  It can
describe several representation families for the same closed event denominator
without downloading any of them.  A locator is a link observation, not source
content, a rights decision, or evidence that the representation is faithful.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from dalio.communications.catalogue import (
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_SOURCES,
    CommunicationSourceSpec,
)

INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION = 1
INSTITUTION_YEAR_METHODOLOGY_VERSION = "communication-institution-year-metadata-v1"
BOE_2025_MANIFEST_ID = "boe_2025_mpr_press_conferences"
BOE_2025_MANIFEST_SHA256 = "4bba6c8415de46718a5ae6906d0d09e9041f7be1903dbeef7be4f40223717eb2"

_SOURCES_BY_ID = {source.source_id: source for source in COMMUNICATION_SOURCES}
_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_ACTOR = re.compile(r"^(?:agent|model):[a-z0-9][a-z0-9_.-]*$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MIME = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_PLATFORM_MEDIA_ID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")

_TOP_FIELDS = frozenset(
    {
        "schema_version",
        "methodology_version",
        "catalogue_sha256",
        "manifest_id",
        "created_by",
        "created_at",
        "as_known_at",
        "content_capture_authorized",
        "scope",
        "events",
    }
)
_SCOPE_FIELDS = frozenset(
    {
        "organization_id",
        "year",
        "start_date",
        "end_date",
        "event_type",
        "denominator_source_url",
        "inclusion_rule",
        "exclusions",
        "checked_by",
        "checked_at",
        "event_keys",
        "representation_specs",
    }
)
_SPEC_FIELDS = frozenset(
    {
        "representation_key",
        "source_id",
        "artifact_role",
        "material_type",
        "section_coverage",
        "completeness_basis",
        "rights_status",
        "acquisition_status",
        "automated_collection_allowed",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "event_key",
        "organization_id",
        "event_type",
        "title",
        "event_date",
        "metadata_known_at",
        "representations",
    }
)
_OBSERVATION_FIELDS = frozenset(
    {
        "representation_key",
        "availability_status",
        "checked_at",
        "status_evidence_url",
        "status_note",
        "locator",
    }
)
_LOCATOR_FIELDS = frozenset(
    {
        "locator_key",
        "source_id",
        "locator_kind",
        "locator_url",
        "platform_media_id",
        "mime_type",
        "language",
        "translation_status",
        "host_organization",
        "publisher",
        "transcriber",
        "transcriber_attribution",
        "origin_type",
        "provenance_tier",
        "published_at",
        "observed_at",
    }
)

_ROLE_MATERIALS = {
    "full_transcript": "press_conference_transcript",
    "webcast_video": "press_conference_video",
    "subtitles": "subtitles",
}
_ROLE_SCOPES = {
    "full_transcript": ("full_transcript",),
    "webcast_video": ("webcast_video",),
    "subtitles": ("subtitles",),
}
_COMPLETENESS_BASES = frozenset(
    {"exact_direct_artifact_url", "official_page_media_locator", "exact_caption_track_url"}
)
_ROLE_COMPLETENESS_BASIS = {
    "full_transcript": "exact_direct_artifact_url",
    "webcast_video": "official_page_media_locator",
    "subtitles": "exact_caption_track_url",
}
_AVAILABILITY_STATUSES = frozenset(
    {
        "direct_artifact_link",
        "external_platform_link",
        "embedded_platform_id_only",
        "not_verified",
    }
)
_LOCATOR_KINDS = frozenset(
    {
        "official_direct_artifact",
        "external_platform_page",
        "external_platform_id",
        "exact_caption_track",
    }
)


@dataclass(frozen=True)
class InstitutionYearRepresentationSpec:
    representation_key: str
    source_id: str
    artifact_role: str
    material_type: str
    section_coverage: tuple[str, ...]
    completeness_basis: str
    rights_status: str
    acquisition_status: str
    automated_collection_allowed: bool


@dataclass(frozen=True)
class InstitutionYearLocator:
    locator_key: str
    source_id: str
    locator_kind: str
    locator_url: str | None
    platform_media_id: str | None
    mime_type: str | None
    language: str
    translation_status: str
    host_organization: str
    publisher: str
    transcriber: str | None
    transcriber_attribution: str
    origin_type: str
    provenance_tier: str
    published_at: datetime | None
    observed_at: datetime


@dataclass(frozen=True)
class InstitutionYearRepresentationObservation:
    representation_key: str
    availability_status: str
    checked_at: datetime
    status_evidence_url: str
    status_note: str
    locator: InstitutionYearLocator | None


@dataclass(frozen=True)
class InstitutionYearEvent:
    event_key: str
    organization_id: str
    event_type: str
    title: str
    event_date: date
    metadata_known_at: datetime
    representations: tuple[InstitutionYearRepresentationObservation, ...]


@dataclass(frozen=True)
class InstitutionYearScope:
    organization_id: str
    year: int
    start_date: date
    end_date: date
    event_type: str
    denominator_source_url: str
    inclusion_rule: str
    exclusions: tuple[str, ...]
    checked_by: str
    checked_at: datetime
    event_keys: tuple[str, ...]
    representation_specs: tuple[InstitutionYearRepresentationSpec, ...]


@dataclass(frozen=True)
class InstitutionYearManifest:
    schema_version: int
    methodology_version: str
    catalogue_sha256: str
    manifest_id: str
    created_by: str
    created_at: datetime
    as_known_at: datetime
    content_capture_authorized: bool
    scope: InstitutionYearScope
    events: tuple[InstitutionYearEvent, ...]


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"unsupported JSON numeric constant: {value}")


def _object(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be an object")
    return value


def _array(value: object, field: str, *, nonempty: bool = True) -> list[object]:
    if not isinstance(value, list) or (nonempty and not value):
        qualifier = "a non-empty array" if nonempty else "an array"
        raise ValueError(f"{field} must be {qualifier}")
    return value


def _exact(payload: dict[str, object], expected: frozenset[str], field: str) -> None:
    missing = sorted(expected - set(payload))
    unknown = sorted(set(payload) - expected)
    if missing:
        raise ValueError(f"{field} has missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{field} has unknown fields: {', '.join(unknown)}")


def _string(value: object, field: str, *, max_length: int = 2_000) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{field} must be a non-empty trimmed string")
    if len(value) > max_length:
        raise ValueError(f"{field} is longer than {max_length} characters")
    return value


def _nullable_string(value: object, field: str, *, max_length: int = 192) -> str | None:
    if value is None:
        return None
    return _string(value, field, max_length=max_length)


def _identifier(value: object, field: str, *, max_length: int = 128) -> str:
    result = _string(value, field, max_length=max_length)
    if _ID.fullmatch(result) is None:
        raise ValueError(f"{field} must be lowercase snake_case")
    return result


def _actor(value: object, field: str) -> str:
    result = _string(value, field, max_length=192)
    if _ACTOR.fullmatch(result) is None:
        raise ValueError(f"{field} must be an agent: or model: identifier")
    return result


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _date(value: object, field: str) -> date:
    raw = _string(value, field, max_length=10)
    try:
        result = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO date") from exc
    if result.isoformat() != raw:
        raise ValueError(f"{field} must be an ISO date")
    return result


def _datetime(value: object, field: str, *, nullable: bool = False) -> datetime | None:
    if value is None and nullable:
        return None
    raw = _string(value, field, max_length=40)
    try:
        result = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be an ISO datetime") from exc
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone")
    return result.astimezone(UTC)


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _host(url: str, field: str) -> str:
    if "\\" in url or any(character.isspace() or ord(character) < 32 for character in url):
        raise ValueError(f"{field} cannot contain whitespace or control characters")
    parts = urlsplit(url)
    if (
        parts.scheme != "https"
        or parts.username
        or parts.password
        or parts.port
        or parts.fragment
    ):
        raise ValueError(f"{field} must be a simple HTTPS URL")
    raw_hostname = parts.hostname or ""
    if raw_hostname.endswith("."):
        raise ValueError(f"{field} cannot use a trailing-dot host")
    hostname = raw_hostname.lower()
    if not hostname or len(hostname) > 253:
        raise ValueError(f"{field} has an invalid host")
    try:
        ipaddress.ip_address(hostname)
    except ValueError:
        pass
    else:
        raise ValueError(f"{field} cannot use an IP-address host")
    if any(_DNS_LABEL.fullmatch(label) is None for label in hostname.split(".")):
        raise ValueError(f"{field} has an invalid host")
    if not parts.path.startswith("/") or "\\" in parts.path:
        raise ValueError(f"{field} has an invalid path")
    return hostname


def _official_url(value: object, source: CommunicationSourceSpec, field: str) -> str:
    url = _string(value, field)
    host = _host(url, field)
    if not any(host == domain or host.endswith(f".{domain}") for domain in source.official_domains):
        raise ValueError(f"{field} must use an official domain for {source.source_id}")
    return url


def _source(source_id: str, organization_id: str, field: str) -> CommunicationSourceSpec:
    source = _SOURCES_BY_ID.get(source_id)
    if source is None:
        raise ValueError(f"{field} is not in the communication source catalogue")
    if source.organization_id != organization_id:
        raise ValueError(f"{field} does not belong to {organization_id}")
    return source


def _parse_spec(
    value: object, organization_id: str, index: int
) -> InstitutionYearRepresentationSpec:
    field = f"scope.representation_specs[{index}]"
    payload = _object(value, field)
    _exact(payload, _SPEC_FIELDS, field)
    representation_key = _identifier(payload["representation_key"], f"{field}.representation_key")
    source_id = _identifier(payload["source_id"], f"{field}.source_id")
    source = _source(source_id, organization_id, f"{field}.source_id")
    artifact_role = _identifier(payload["artifact_role"], f"{field}.artifact_role")
    material_type = _identifier(payload["material_type"], f"{field}.material_type")
    if _ROLE_MATERIALS.get(artifact_role) != material_type:
        raise ValueError(f"{field} has incompatible artifact_role and material_type")
    if material_type not in source.material_types:
        raise ValueError(f"{field}.material_type is not declared by its source policy")
    section_coverage = tuple(
        _identifier(item, f"{field}.section_coverage[{item_index}]")
        for item_index, item in enumerate(
            _array(payload["section_coverage"], f"{field}.section_coverage")
        )
    )
    if section_coverage != _ROLE_SCOPES[artifact_role]:
        raise ValueError(f"{field}.section_coverage conflicts with artifact_role")
    completeness_basis = _string(
        payload["completeness_basis"], f"{field}.completeness_basis", max_length=64
    )
    if completeness_basis not in _COMPLETENESS_BASES:
        raise ValueError(f"{field}.completeness_basis is unsupported")
    if completeness_basis != _ROLE_COMPLETENESS_BASIS[artifact_role]:
        raise ValueError(f"{field}.completeness_basis conflicts with artifact_role")
    if payload["rights_status"] != source.rights_status:
        raise ValueError(f"{field}.rights_status conflicts with its source policy")
    if payload["acquisition_status"] != source.acquisition_status:
        raise ValueError(f"{field}.acquisition_status conflicts with its source policy")
    if payload["automated_collection_allowed"] is not False or source.automated_collection_allowed:
        raise ValueError(f"{field} must keep automated collection disabled")
    if source.rights_status in {"cleared", "internal_only"}:
        raise ValueError(
            f"{field} cannot encode a cleared source in this pending metadata manifest"
        )
    return InstitutionYearRepresentationSpec(
        representation_key=representation_key,
        source_id=source_id,
        artifact_role=artifact_role,
        material_type=material_type,
        section_coverage=section_coverage,
        completeness_basis=completeness_basis,
        rights_status=source.rights_status,
        acquisition_status=source.acquisition_status,
        automated_collection_allowed=False,
    )


def _parse_locator(
    value: object,
    *,
    organization_id: str,
    spec: InstitutionYearRepresentationSpec,
    checked_at: datetime,
    field: str,
) -> InstitutionYearLocator:
    payload = _object(value, field)
    _exact(payload, _LOCATOR_FIELDS, field)
    source_id = _identifier(payload["source_id"], f"{field}.source_id")
    if source_id != spec.source_id:
        raise ValueError(f"{field}.source_id conflicts with its representation specification")
    source = _source(source_id, organization_id, f"{field}.source_id")
    locator_kind = _string(payload["locator_kind"], f"{field}.locator_kind", max_length=64)
    if locator_kind not in _LOCATOR_KINDS:
        raise ValueError(f"{field}.locator_kind is unsupported")
    locator_url = _nullable_string(payload["locator_url"], f"{field}.locator_url", max_length=2_000)
    platform_media_id = _nullable_string(
        payload["platform_media_id"], f"{field}.platform_media_id", max_length=64
    )
    if platform_media_id is not None and _PLATFORM_MEDIA_ID.fullmatch(platform_media_id) is None:
        raise ValueError(f"{field}.platform_media_id is invalid")
    mime_type = _nullable_string(payload["mime_type"], f"{field}.mime_type", max_length=128)
    if mime_type is not None and _MIME.fullmatch(mime_type) is None:
        raise ValueError(f"{field}.mime_type is invalid")
    host_organization = _string(
        payload["host_organization"], f"{field}.host_organization", max_length=192
    )
    publisher = _string(payload["publisher"], f"{field}.publisher", max_length=192)
    transcriber = _nullable_string(payload["transcriber"], f"{field}.transcriber")
    transcriber_attribution = _string(
        payload["transcriber_attribution"], f"{field}.transcriber_attribution", max_length=64
    )
    origin_type = _string(payload["origin_type"], f"{field}.origin_type", max_length=64)
    provenance_tier = _string(payload["provenance_tier"], f"{field}.provenance_tier", max_length=64)
    language = _string(payload["language"], f"{field}.language", max_length=16)
    if language != source.language:
        raise ValueError(f"{field}.language conflicts with its source policy")
    if payload["translation_status"] != "original":
        raise ValueError(f"{field}.translation_status must be original")
    published_at = _datetime(payload["published_at"], f"{field}.published_at", nullable=True)
    observed_at = _datetime(payload["observed_at"], f"{field}.observed_at")
    assert observed_at is not None
    if published_at is not None and published_at > observed_at:
        raise ValueError(f"{field}.published_at follows observed_at")
    if observed_at != checked_at:
        raise ValueError(f"{field}.observed_at must equal its representation check clock")

    if locator_kind == "official_direct_artifact":
        if locator_url is None or platform_media_id is not None or mime_type != "application/pdf":
            raise ValueError(f"{field} has an invalid official direct-artifact locator")
        _official_url(locator_url, source, f"{field}.locator_url")
        if spec.artifact_role != "full_transcript":
            raise ValueError(f"{field} direct PDF locator must be the transcript representation")
        expected = (
            source.host_organization,
            source.publisher,
            None,
            "not_disclosed",
            "official_published_transcript",
            "official_published_transcript",
        )
    elif locator_kind in {"external_platform_page", "external_platform_id"}:
        if spec.artifact_role != "webcast_video" or platform_media_id is None:
            raise ValueError(f"{field} external locator must be the video representation")
        if locator_kind == "external_platform_page":
            if locator_url is None or mime_type != "text/html":
                raise ValueError(f"{field} external platform page needs an HTML URL")
            host = _host(locator_url, f"{field}.locator_url")
            if host not in {"youtube.com", "www.youtube.com"}:
                raise ValueError(f"{field}.locator_url must use the observed YouTube platform")
            parts = urlsplit(locator_url)
            if parts.path != f"/live/{platform_media_id}" or parse_qs(parts.query) != {
                "feature": ["share"]
            }:
                raise ValueError(f"{field}.locator_url does not match its platform media id")
        elif locator_url is not None or mime_type is not None:
            raise ValueError(f"{field} ID-only locator cannot invent a URL or MIME type")
        expected = (
            "YouTube",
            source.publisher,
            None,
            "not_applicable",
            "official_linked_platform_media",
            "official_linked_external_platform",
        )
    else:
        if spec.artifact_role != "subtitles":
            raise ValueError(f"{field} exact caption locator must be the subtitle representation")
        if locator_url is None or platform_media_id is None:
            raise ValueError(f"{field} exact caption locator needs a URL and platform media id")
        if mime_type not in {"text/vtt", "text/xml", "application/xml"}:
            raise ValueError(f"{field} exact caption locator has an unsupported MIME type")
        parts = urlsplit(locator_url)
        host = _host(locator_url, f"{field}.locator_url")
        if host not in {"youtube.com", "www.youtube.com"}:
            raise ValueError(f"{field}.locator_url must use the observed YouTube platform")
        query = parse_qs(parts.query, keep_blank_values=True)
        if "tlang" in query:
            raise ValueError(
                f"{field}.locator_url cannot request a translated track declared original"
            )
        if (
            parts.path != "/api/timedtext"
            or query.get("v") != [platform_media_id]
            or query.get("lang") != [language]
        ):
            raise ValueError(f"{field}.locator_url does not match its platform media id")
        if transcriber is None:
            raise ValueError(f"{field} exact caption locator must name its caption producer")
        if origin_type not in {"official_caption", "automatic_caption"}:
            raise ValueError(f"{field}.origin_type does not classify caption origin")
        expected = (
            "YouTube",
            source.publisher,
            transcriber,
            "artifact_specific",
            origin_type,
            "official_hosted_third_party",
        )
    actual = (
        host_organization,
        publisher,
        transcriber,
        transcriber_attribution,
        origin_type,
        provenance_tier,
    )
    if actual != expected:
        raise ValueError(f"{field} has inconsistent host, publisher, or provenance metadata")
    return InstitutionYearLocator(
        locator_key=_identifier(payload["locator_key"], f"{field}.locator_key"),
        source_id=source_id,
        locator_kind=locator_kind,
        locator_url=locator_url,
        platform_media_id=platform_media_id,
        mime_type=mime_type,
        language=language,
        translation_status="original",
        host_organization=host_organization,
        publisher=publisher,
        transcriber=transcriber,
        transcriber_attribution=transcriber_attribution,
        origin_type=origin_type,
        provenance_tier=provenance_tier,
        published_at=published_at,
        observed_at=observed_at,
    )


def _parse_observation(
    value: object,
    *,
    organization_id: str,
    specs: dict[str, InstitutionYearRepresentationSpec],
    event_known_at: datetime,
    index: int,
    field_prefix: str,
) -> InstitutionYearRepresentationObservation:
    field = f"{field_prefix}.representations[{index}]"
    payload = _object(value, field)
    _exact(payload, _OBSERVATION_FIELDS, field)
    representation_key = _identifier(payload["representation_key"], f"{field}.representation_key")
    spec = specs.get(representation_key)
    if spec is None:
        raise ValueError(f"{field}.representation_key is not declared by the scope")
    availability_status = _string(
        payload["availability_status"], f"{field}.availability_status", max_length=64
    )
    if availability_status not in _AVAILABILITY_STATUSES:
        raise ValueError(f"{field}.availability_status is unsupported")
    checked_at = _datetime(payload["checked_at"], f"{field}.checked_at")
    assert checked_at is not None
    if checked_at > event_known_at:
        raise ValueError(f"{field}.checked_at follows event metadata_known_at")
    source = _source(spec.source_id, organization_id, f"{field}.representation_key")
    evidence_url = _official_url(
        payload["status_evidence_url"], source, f"{field}.status_evidence_url"
    )
    status_note = _string(payload["status_note"], f"{field}.status_note")
    locator_value = payload["locator"]
    if availability_status == "not_verified":
        if locator_value is not None:
            raise ValueError(f"{field} not_verified must be a locator-free observation")
        locator = None
    else:
        if locator_value is None:
            raise ValueError(f"{field} available locator status requires locator metadata")
        locator = _parse_locator(
            locator_value,
            organization_id=organization_id,
            spec=spec,
            checked_at=checked_at,
            field=f"{field}.locator",
        )
        expected_status = {
            "official_direct_artifact": "direct_artifact_link",
            "external_platform_page": "external_platform_link",
            "external_platform_id": "embedded_platform_id_only",
            "exact_caption_track": "external_platform_link",
        }[locator.locator_kind]
        if availability_status != expected_status:
            raise ValueError(f"{field}.availability_status conflicts with locator_kind")
    return InstitutionYearRepresentationObservation(
        representation_key=representation_key,
        availability_status=availability_status,
        checked_at=checked_at,
        status_evidence_url=evidence_url,
        status_note=status_note,
        locator=locator,
    )


def _parse_scope(value: object) -> InstitutionYearScope:
    field = "scope"
    payload = _object(value, field)
    _exact(payload, _SCOPE_FIELDS, field)
    organization_id = _identifier(payload["organization_id"], "scope.organization_id")
    year = _integer(payload["year"], "scope.year", minimum=1900)
    start_date = _date(payload["start_date"], "scope.start_date")
    end_date = _date(payload["end_date"], "scope.end_date")
    if (start_date, end_date) != (date(year, 1, 1), date(year, 12, 31)):
        raise ValueError("scope must cover exactly one calendar year")
    event_type = _identifier(payload["event_type"], "scope.event_type")
    raw_specs = _array(payload["representation_specs"], "scope.representation_specs")
    specs = tuple(_parse_spec(item, organization_id, index) for index, item in enumerate(raw_specs))
    spec_keys = [spec.representation_key for spec in specs]
    if len(spec_keys) != len(set(spec_keys)):
        raise ValueError("scope.representation_specs contains duplicate representation keys")
    sources = [
        _source(source_id, organization_id, "scope.representation_specs.source_id")
        for source_id in {spec.source_id for spec in specs}
    ]
    denominator_source_url = _string(
        payload["denominator_source_url"], "scope.denominator_source_url"
    )
    if not any(
        _host(denominator_source_url, "scope.denominator_source_url") == domain
        or _host(denominator_source_url, "scope.denominator_source_url").endswith(f".{domain}")
        for source in sources
        for domain in source.official_domains
    ):
        raise ValueError("scope.denominator_source_url must use an official source domain")
    exclusions = tuple(
        _string(item, f"scope.exclusions[{index}]")
        for index, item in enumerate(_array(payload["exclusions"], "scope.exclusions"))
    )
    if len(exclusions) != len(set(exclusions)):
        raise ValueError("scope.exclusions contains duplicates")
    event_keys = tuple(
        _identifier(item, f"scope.event_keys[{index}]")
        for index, item in enumerate(_array(payload["event_keys"], "scope.event_keys"))
    )
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("scope.event_keys contains duplicates")
    checked_at = _datetime(payload["checked_at"], "scope.checked_at")
    assert checked_at is not None
    if checked_at.date() <= end_date:
        raise ValueError("scope.checked_at must follow the closed denominator year")
    return InstitutionYearScope(
        organization_id=organization_id,
        year=year,
        start_date=start_date,
        end_date=end_date,
        event_type=event_type,
        denominator_source_url=denominator_source_url,
        inclusion_rule=_string(payload["inclusion_rule"], "scope.inclusion_rule"),
        exclusions=exclusions,
        checked_by=_actor(payload["checked_by"], "scope.checked_by"),
        checked_at=checked_at,
        event_keys=event_keys,
        representation_specs=specs,
    )


def _parse_event(value: object, scope: InstitutionYearScope, index: int) -> InstitutionYearEvent:
    field = f"events[{index}]"
    payload = _object(value, field)
    _exact(payload, _EVENT_FIELDS, field)
    event_key = _identifier(payload["event_key"], f"{field}.event_key")
    if event_key not in scope.event_keys:
        raise ValueError(f"{field}.event_key is outside the closed denominator")
    if (
        payload["organization_id"] != scope.organization_id
        or payload["event_type"] != scope.event_type
    ):
        raise ValueError(f"{field} conflicts with its scope identity")
    event_date = _date(payload["event_date"], f"{field}.event_date")
    if not scope.start_date <= event_date <= scope.end_date:
        raise ValueError(f"{field}.event_date is outside the closed year")
    metadata_known_at = _datetime(payload["metadata_known_at"], f"{field}.metadata_known_at")
    assert metadata_known_at is not None
    if metadata_known_at.date() <= scope.end_date or metadata_known_at > scope.checked_at:
        raise ValueError(f"{field}.metadata_known_at has an invalid observation clock")
    specs = {spec.representation_key: spec for spec in scope.representation_specs}
    observations = tuple(
        _parse_observation(
            item,
            organization_id=scope.organization_id,
            specs=specs,
            event_known_at=metadata_known_at,
            index=observation_index,
            field_prefix=field,
        )
        for observation_index, item in enumerate(
            _array(payload["representations"], f"{field}.representations")
        )
    )
    observed_keys = [item.representation_key for item in observations]
    if len(observed_keys) != len(set(observed_keys)):
        raise ValueError(f"{field}.representations contains duplicate keys")
    if set(observed_keys) != set(specs):
        raise ValueError(f"{field}.representations must cover every declared representation")
    video_ids = {
        observation.locator.platform_media_id
        for observation in observations
        if specs[observation.representation_key].artifact_role == "webcast_video"
        and observation.locator is not None
        and observation.locator.platform_media_id is not None
    }
    for observation in observations:
        locator = observation.locator
        if (
            locator is not None
            and locator.locator_kind == "exact_caption_track"
            and locator.platform_media_id not in video_ids
        ):
            raise ValueError(
                f"{field} exact caption track does not match a video locator for this event"
            )
    return InstitutionYearEvent(
        event_key=event_key,
        organization_id=scope.organization_id,
        event_type=scope.event_type,
        title=_string(payload["title"], f"{field}.title", max_length=500),
        event_date=event_date,
        metadata_known_at=metadata_known_at,
        representations=observations,
    )


def load_institution_year_manifest(path: Path) -> InstitutionYearManifest:
    """Load and validate a metadata-only, one-institution/year manifest."""
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read institution-year manifest {path}: {exc}") from exc
    root = _object(payload, "institution-year manifest")
    _exact(root, _TOP_FIELDS, "institution-year manifest")
    schema_version = _integer(root["schema_version"], "schema_version", minimum=1)
    if schema_version != INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported institution-year manifest schema_version")
    if root["methodology_version"] != INSTITUTION_YEAR_METHODOLOGY_VERSION:
        raise ValueError("unsupported institution-year methodology_version")
    if root["catalogue_sha256"] != COMMUNICATION_CATALOGUE_SHA256:
        raise ValueError("manifest must bind the current communication source catalogue")
    if root["content_capture_authorized"] is not False:
        raise ValueError("institution-year metadata cannot authorize content capture")
    scope = _parse_scope(root["scope"])
    created_at = _datetime(root["created_at"], "created_at")
    as_known_at = _datetime(root["as_known_at"], "as_known_at")
    assert created_at is not None and as_known_at is not None
    if created_at > as_known_at or scope.checked_at > as_known_at:
        raise ValueError("manifest clocks follow as_known_at")
    events = tuple(
        _parse_event(item, scope, index)
        for index, item in enumerate(_array(root["events"], "events"))
    )
    event_keys = [event.event_key for event in events]
    if len(event_keys) != len(set(event_keys)):
        raise ValueError("events contains duplicate event keys")
    if set(event_keys) != set(scope.event_keys):
        raise ValueError("events do not reconcile with the closed denominator")
    locator_keys = [
        observation.locator.locator_key
        for event in events
        for observation in event.representations
        if observation.locator is not None
    ]
    if len(locator_keys) != len(set(locator_keys)):
        raise ValueError("manifest contains duplicate locator keys")
    for event in events:
        if event.metadata_known_at > as_known_at:
            raise ValueError(f"{event.event_key} metadata_known_at follows as_known_at")
    return InstitutionYearManifest(
        schema_version=INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION,
        methodology_version=INSTITUTION_YEAR_METHODOLOGY_VERSION,
        catalogue_sha256=COMMUNICATION_CATALOGUE_SHA256,
        manifest_id=_identifier(root["manifest_id"], "manifest_id"),
        created_by=_actor(root["created_by"], "created_by"),
        created_at=created_at,
        as_known_at=as_known_at,
        content_capture_authorized=False,
        scope=scope,
        events=events,
    )


def _semantic_payload(manifest: InstitutionYearManifest) -> dict[str, object]:
    payload = asdict(manifest)
    payload["created_at"] = _iso_datetime(manifest.created_at)
    payload["as_known_at"] = _iso_datetime(manifest.as_known_at)
    scope = payload["scope"]
    assert isinstance(scope, dict)
    scope["start_date"] = manifest.scope.start_date.isoformat()
    scope["end_date"] = manifest.scope.end_date.isoformat()
    scope["checked_at"] = _iso_datetime(manifest.scope.checked_at)
    scope["event_keys"] = sorted(scope["event_keys"])
    scope["exclusions"] = sorted(scope["exclusions"])
    scope["representation_specs"] = sorted(
        scope["representation_specs"], key=lambda item: item["representation_key"]
    )
    for spec in scope["representation_specs"]:
        spec["section_coverage"] = list(spec["section_coverage"])
    events = list(payload["events"])
    payload["events"] = events
    events.sort(key=lambda item: item["event_key"])
    by_event = {event.event_key: event for event in manifest.events}
    for event_payload in events:
        event = by_event[event_payload["event_key"]]
        event_payload["event_date"] = event.event_date.isoformat()
        event_payload["metadata_known_at"] = _iso_datetime(event.metadata_known_at)
        event_payload["representations"] = sorted(
            event_payload["representations"], key=lambda item: item["representation_key"]
        )
        by_representation = {item.representation_key: item for item in event.representations}
        for observation_payload in event_payload["representations"]:
            observation = by_representation[observation_payload["representation_key"]]
            observation_payload["checked_at"] = _iso_datetime(observation.checked_at)
            if observation.locator is not None:
                locator_payload = observation_payload["locator"]
                locator_payload["published_at"] = _iso_datetime(observation.locator.published_at)
                locator_payload["observed_at"] = _iso_datetime(observation.locator.observed_at)
    return payload


def institution_year_manifest_sha256(manifest: InstitutionYearManifest) -> str:
    """Return the canonical semantic fingerprint for a validated manifest."""
    encoded = json.dumps(
        _semantic_payload(manifest),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_checked_boe_2025_manifest(path: Path) -> InstitutionYearManifest:
    """Load the checked BoE cohort and require its pinned semantic identity."""
    manifest = load_institution_year_manifest(path)
    if manifest.manifest_id != BOE_2025_MANIFEST_ID:
        raise ValueError("checked BoE manifest has the wrong manifest_id")
    if institution_year_manifest_sha256(manifest) != BOE_2025_MANIFEST_SHA256:
        raise ValueError("checked BoE manifest does not match its pinned semantic SHA-256")
    return manifest


__all__ = [
    "INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION",
    "INSTITUTION_YEAR_METHODOLOGY_VERSION",
    "BOE_2025_MANIFEST_ID",
    "BOE_2025_MANIFEST_SHA256",
    "InstitutionYearEvent",
    "InstitutionYearLocator",
    "InstitutionYearManifest",
    "InstitutionYearRepresentationObservation",
    "InstitutionYearRepresentationSpec",
    "InstitutionYearScope",
    "institution_year_manifest_sha256",
    "load_checked_boe_2025_manifest",
    "load_institution_year_manifest",
]
