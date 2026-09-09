"""Deterministic, offline rights-review packet for the communications pilot.

The packet is a review aid, not a rights decision and not collection authority.
It consumes only the already validated pilot manifest: it performs no network,
database, or communication-content reads.  Every selected representation stays
behind the source-level human rights gate.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from collections import Counter, defaultdict
from datetime import UTC, date, datetime
from typing import Any

from dalio.communications.pilot_manifest import (
    PILOT_MANIFEST_SCHEMA_VERSION,
    CommunicationPilotManifest,
    pilot_manifest_sha256,
)

COMMUNICATION_RIGHTS_REVIEW_SCHEMA_VERSION = 1
COMMUNICATION_RIGHTS_REVIEW_METHODOLOGY_VERSION = "communication-rights-review-v1"
UNVERIFIED_RIGHTS_REVIEW_LABEL = "UNVERIFIED RIGHTS REVIEW"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

EXPECTED_ORGANIZATION_EVENT_COUNTS = {
    "ecb": 8,
    "federal_reserve": 8,
}
EXPECTED_EVENT_COUNT = sum(EXPECTED_ORGANIZATION_EVENT_COUNTS.values())
EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT = EXPECTED_EVENT_COUNT

ECB_MIXED_SECTION_BLOCKER = (
    "ECB selected pages combine the monetary-policy statement and questions-and-answers "
    "sections. The current one-role artifact schema cannot faithfully classify that mixed "
    "page. Define a single-capture, multi-section mapping before recording durable database "
    "metadata; do not create duplicate byte captures. This packet does not authorize the page "
    "for content capture."
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("rights-review clocks must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _iso_date(value: date) -> str:
    return value.isoformat()


def _representation_spec_payload(spec: Any) -> dict[str, object]:
    return {
        "representation_key": spec.representation_key,
        "source_id": spec.source_id,
        "artifact_role": spec.artifact_role,
        "material_type": spec.material_type,
        "section_coverage": sorted(spec.section_coverage),
    }


def _rights_review_payload(review: Any) -> dict[str, object]:
    return {
        "status": review.status,
        "basis_url": review.basis_url,
        "summary": review.summary,
        "questions": sorted(review.questions),
    }


def _candidate_payload(candidate: Any) -> dict[str, object]:
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


def _event_payload(event: Any) -> dict[str, object]:
    representation = event.representation
    candidate = representation.candidate
    if representation.availability_status != "available" or candidate is None:
        raise ValueError(
            f"{event.event_key} must have one available chosen representation candidate"
        )
    return {
        "event_key": event.event_key,
        "organization_id": event.organization_id,
        "event_type": event.event_type,
        "title": event.title,
        "event_date": _iso_date(event.event_date),
        "event_started_at": _iso_datetime(event.event_started_at),
        "metadata_known_at": _iso_datetime(event.metadata_known_at),
        "chosen_representation": {
            "representation_key": representation.representation_key,
            "availability_status": representation.availability_status,
            "checked_at": _iso_datetime(representation.checked_at),
            "status_evidence_url": representation.status_evidence_url,
            "status_note": representation.status_note,
            "section_coverage": sorted(representation.section_coverage),
            "candidate": _candidate_payload(candidate),
        },
    }


def _validate_and_reconcile(
    manifest: CommunicationPilotManifest,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if manifest.schema_version != PILOT_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported communication pilot manifest schema version")

    denominators = tuple(manifest.scope.organizations)
    denominator_ids = [item.organization_id for item in denominators]
    duplicates = sorted(
        organization_id for organization_id, count in Counter(denominator_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"duplicate organization denominators: {', '.join(duplicates)}")

    expected_organizations = set(EXPECTED_ORGANIZATION_EVENT_COUNTS)
    actual_organizations = set(denominator_ids)
    if actual_organizations != expected_organizations:
        missing = sorted(expected_organizations - actual_organizations)
        unexpected = sorted(actual_organizations - expected_organizations)
        details: list[str] = []
        if missing:
            details.append(f"missing {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected {', '.join(unexpected)}")
        raise ValueError("pilot organization denominator mismatch: " + "; ".join(details))

    events = tuple(manifest.events)
    if len(events) != EXPECTED_EVENT_COUNT:
        raise ValueError(
            f"communication pilot requires exactly {EXPECTED_EVENT_COUNT} events; "
            f"found {len(events)}"
        )
    event_key_counts = Counter(event.event_key for event in events)
    duplicate_event_keys = sorted(key for key, count in event_key_counts.items() if count > 1)
    if duplicate_event_keys:
        raise ValueError(f"duplicate pilot event keys: {', '.join(duplicate_event_keys)}")

    events_by_organization: dict[str, list[Any]] = defaultdict(list)
    for event in events:
        if event.organization_id not in expected_organizations:
            raise ValueError(
                f"event {event.event_key} has unexpected organization {event.organization_id}"
            )
        events_by_organization[event.organization_id].append(event)

    organization_rows: list[dict[str, object]] = []
    for denominator in sorted(denominators, key=lambda item: item.organization_id):
        organization_id = denominator.organization_id
        expected_count = EXPECTED_ORGANIZATION_EVENT_COUNTS[organization_id]
        expected_keys = tuple(denominator.event_keys)
        if len(expected_keys) != len(set(expected_keys)):
            raise ValueError(f"{organization_id} denominator contains duplicate event keys")
        if len(expected_keys) != expected_count:
            raise ValueError(
                f"{organization_id} denominator requires exactly {expected_count} event keys; "
                f"found {len(expected_keys)}"
            )

        organization_events = events_by_organization[organization_id]
        actual_keys = {event.event_key for event in organization_events}
        expected_key_set = set(expected_keys)
        missing_keys = sorted(expected_key_set - actual_keys)
        unexpected_keys = sorted(actual_keys - expected_key_set)
        if missing_keys or unexpected_keys or len(organization_events) != expected_count:
            details = []
            if missing_keys:
                details.append(f"missing {', '.join(missing_keys)}")
            if unexpected_keys:
                details.append(f"unexpected {', '.join(unexpected_keys)}")
            if len(organization_events) != expected_count:
                details.append(f"found {len(organization_events)} events")
            raise ValueError(
                f"{organization_id} event denominator does not reconcile: " + "; ".join(details)
            )

        spec = denominator.representation_spec
        for event in organization_events:
            representation = event.representation
            candidate = representation.candidate
            if representation.availability_status != "available" or candidate is None:
                raise ValueError(
                    f"{event.event_key} must have one available chosen representation candidate"
                )
            if representation.representation_key != spec.representation_key:
                raise ValueError(
                    f"{event.event_key} representation key does not match its denominator"
                )
            if set(representation.section_coverage) != set(spec.section_coverage):
                raise ValueError(
                    f"{event.event_key} section coverage does not match its denominator"
                )
            for field in ("source_id", "artifact_role", "material_type"):
                if getattr(candidate, field) != getattr(spec, field):
                    raise ValueError(
                        f"{event.event_key} candidate {field} does not match its denominator"
                    )
        if denominator.rights_review.status != "pending":
            raise ValueError(
                f"{organization_id} source rights status must remain pending in this packet"
            )

        organization_rows.append(
            {
                "organization_id": organization_id,
                "denominator_source_url": denominator.denominator_source_url,
                "denominator_checked_by": denominator.checked_by,
                "denominator_checked_at": _iso_datetime(denominator.checked_at),
                "expected_event_count": expected_count,
                "manifest_event_count": len(organization_events),
                "available_chosen_representation_count": len(organization_events),
                "missing_event_keys": missing_keys,
                "unexpected_event_keys": unexpected_keys,
                "event_keys": sorted(expected_keys),
                "representation_spec": _representation_spec_payload(spec),
                "rights_review": _rights_review_payload(denominator.rights_review),
            }
        )

    event_rows = [
        _event_payload(event)
        for event in sorted(
            events,
            key=lambda item: (item.organization_id, item.event_date, item.event_key),
        )
    ]
    return organization_rows, event_rows


def rights_review_packet_sha256(packet_without_hash: dict[str, object]) -> str:
    """Return the canonical packet fingerprint; the self-hash must be absent."""
    if "packet_sha256" in packet_without_hash:
        raise ValueError("packet_sha256 must be excluded while computing its own hash")
    return hashlib.sha256(_canonical_json(packet_without_hash)).hexdigest()


def validate_rights_review_packet_sha256(packet: dict[str, object]) -> str:
    """Verify and return a packet's canonical self-excluding fingerprint."""
    if not isinstance(packet, dict):
        raise TypeError("rights-review packet must be a dictionary")
    supplied = packet.get("packet_sha256")
    if not isinstance(supplied, str) or _SHA256_RE.fullmatch(supplied) is None:
        raise ValueError("rights-review packet lacks a full lowercase SHA-256 fingerprint")
    payload = {key: value for key, value in packet.items() if key != "packet_sha256"}
    expected = rights_review_packet_sha256(payload)
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("rights-review packet_sha256 does not match its canonical payload")
    return supplied


def build_rights_review_packet(
    manifest: CommunicationPilotManifest,
) -> dict[str, object]:
    """Build the fail-closed human review aid without external or content I/O."""
    organizations, events = _validate_and_reconcile(manifest)
    available_count = sum(
        int(row["available_chosen_representation_count"]) for row in organizations
    )
    if available_count != EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT:
        raise ValueError(
            "available chosen representations do not reconcile with the closed pilot denominator"
        )

    packet: dict[str, object] = {
        "schema_version": COMMUNICATION_RIGHTS_REVIEW_SCHEMA_VERSION,
        "packet_kind": "unverified_communication_rights_review",
        "notice": (
            "UNVERIFIED RIGHTS REVIEW — metadata discovery is not a rights decision, and "
            "nothing in this packet authorizes downloading, storing, extracting, or "
            "transcribing communication content."
        ),
        "review_label": UNVERIFIED_RIGHTS_REVIEW_LABEL,
        "content_capture_authorized": False,
        "verified_rights_decisions": 0,
        "methodology_version": COMMUNICATION_RIGHTS_REVIEW_METHODOLOGY_VERSION,
        "pilot_methodology_version": manifest.methodology_version,
        "pilot_manifest_sha256": pilot_manifest_sha256(manifest),
        "catalogue_sha256": manifest.catalogue_sha256,
        "pilot_id": manifest.pilot_id,
        "manifest_created_by": manifest.created_by,
        "manifest_created_at": _iso_datetime(manifest.created_at),
        "as_known_at": _iso_datetime(manifest.as_known_at),
        "scope": {
            "start_date": _iso_date(manifest.scope.start_date),
            "end_date": _iso_date(manifest.scope.end_date),
        },
        "organization_count": len(organizations),
        "expected_event_count": EXPECTED_EVENT_COUNT,
        "event_count": len(events),
        "expected_available_chosen_representation_count": (
            EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT
        ),
        "available_chosen_representation_count": available_count,
        "organization_reconciliation": organizations,
        "durable_metadata_blockers": [
            {
                "organization_id": "ecb",
                "status": "multi_section_mapping_required",
                "message": ECB_MIXED_SECTION_BLOCKER,
            }
        ],
        "review_questions": [
            "Does the cited source-level rights basis apply to each exact selected artifact?",
            "Does any selected page or file contain third-party text, media, captions, or marks?",
            "Which representations, if any, may be retained, extracted, and used internally?",
            "What attribution, access-frequency, retention, or redistribution conditions apply?",
        ],
        "events": events,
    }
    packet["packet_sha256"] = rights_review_packet_sha256(packet)
    validate_rights_review_packet_sha256(packet)
    return packet


def _markdown_value(value: object) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _markdown_sequence(values: list[object]) -> str:
    return ", ".join(str(value) for value in values)


def render_rights_review_markdown(packet: dict[str, object]) -> str:
    """Render a deterministic, visibly unverified human review sheet."""
    validate_rights_review_packet_sha256(packet)
    if packet.get("packet_kind") != "unverified_communication_rights_review":
        raise ValueError("unsupported communication rights-review packet")
    if packet.get("content_capture_authorized") is not False:
        raise ValueError("rights-review packet must not authorize content capture")
    if packet.get("verified_rights_decisions") != 0:
        raise ValueError("unverified rights-review packet cannot contain verified decisions")

    lines = [
        "# Institutional-communications rights review",
        "",
        "**UNVERIFIED RIGHTS REVIEW — NO CONTENT CAPTURE IS AUTHORIZED.**",
        "",
        str(packet["notice"]),
        "",
        f"Pilot: `{packet['pilot_id']}`  ",
        f"Public-information cutoff: `{packet['as_known_at']}`  ",
        f"Pilot manifest SHA-256: `{packet['pilot_manifest_sha256']}`  ",
        f"Packet SHA-256: `{packet['packet_sha256']}`",
        "",
        f"Content capture authorized: `{_markdown_value(packet['content_capture_authorized'])}`  ",
        f"Verified rights decisions: `{packet['verified_rights_decisions']}`  ",
        f"Closed denominator: `{packet['event_count']}/{packet['expected_event_count']}` events  ",
        "Available chosen representations: "
        f"`{packet['available_chosen_representation_count']}/"
        f"{packet['expected_available_chosen_representation_count']}`",
        "",
        "## Denominator and source-level rights gates",
        "",
    ]

    for organization in packet["organization_reconciliation"]:
        review = organization["rights_review"]
        spec = organization["representation_spec"]
        lines.extend(
            [
                f"### `{organization['organization_id']}`",
                "",
                f"Events: `{organization['manifest_event_count']}/"
                f"{organization['expected_event_count']}` · available chosen representations: "
                f"`{organization['available_chosen_representation_count']}`  ",
                f"Official denominator: <{organization['denominator_source_url']}>  ",
                f"Denominator checked: `{organization['denominator_checked_at']}` by "
                f"`{organization['denominator_checked_by']}`  ",
                f"Chosen representation: `{spec['representation_key']}` · source "
                f"`{spec['source_id']}` · `{spec['artifact_role']}` / "
                f"`{spec['material_type']}` · section coverage "
                f"`{_markdown_sequence(spec['section_coverage'])}`",
                "",
                f"Rights status: **{review['status']}**  ",
                "Rights-basis candidate: "
                + (f"<{review['basis_url']}>" if review["basis_url"] else "not supplied"),
                "",
                str(review["summary"]),
                "",
                "Pending source-level questions:",
                "",
            ]
        )
        lines.extend(f"- [ ] {question}" for question in review["questions"])
        lines.append("")

    lines.extend(
        [
            "## Durable-metadata blocker",
            "",
            f"**{ECB_MIXED_SECTION_BLOCKER}**",
            "",
            "## Selected artifact candidates",
            "",
        ]
    )
    for event in packet["events"]:
        representation = event["chosen_representation"]
        candidate = representation["candidate"]
        lines.extend(
            [
                f"### {event['title']}",
                "",
                f"Event: `{event['event_key']}` · organization "
                f"`{event['organization_id']}` · date `{event['event_date']}` · "
                f"start `{_markdown_value(event['event_started_at'])}`  ",
                f"Event metadata known: `{event['metadata_known_at']}`  ",
                f"Availability: `{representation['availability_status']}` · checked "
                f"`{representation['checked_at']}`  ",
                f"Status evidence: <{representation['status_evidence_url']}>  ",
                f"Coverage: `{_markdown_sequence(representation['section_coverage'])}` — "
                f"{representation['status_note']}",
                "",
                f"Candidate: `{candidate['artifact_key']}` · source `{candidate['source_id']}` · "
                f"`{candidate['artifact_role']}` / `{candidate['material_type']}` · "
                f"`{candidate['mime_type']}`  ",
                f"Language: `{candidate['language']}` · translation "
                f"`{candidate['translation_status']}`  ",
                f"Official landing: <{candidate['landing_url']}>  ",
                f"Candidate artifact: <{candidate['artifact_url']}>  ",
                "Clocks — published: "
                f"`{_markdown_value(candidate['published_at'])}` · available: "
                f"`{candidate['available_at']}` · retrieved: `{candidate['retrieved_at']}` · "
                f"metadata known: `{candidate['metadata_known_at']}`",
                "",
                f"Provenance: `{candidate['origin_type']}` / `{candidate['provenance_tier']}`  ",
                f"Host: {candidate['host_organization']}  ",
                f"Publisher: {candidate['publisher']}  ",
                "Transcriber: "
                f"{_markdown_value(candidate['transcriber'])} "
                f"(`{candidate['transcriber_attribution']}`)",
                "",
                "- [ ] Exact-artifact rights basis verified by a named human",
                "- [ ] Third-party material and attribution conditions checked",
                "- [ ] Permitted capture, retention, extraction, and use recorded separately",
                "",
            ]
        )

    lines.extend(
        [
            "## Gate remains closed",
            "",
            "This packet contains zero verified rights decisions. It must not be treated as "
            "permission to download, archive, extract, transcribe, or analyze source content.",
            "",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "COMMUNICATION_RIGHTS_REVIEW_METHODOLOGY_VERSION",
    "COMMUNICATION_RIGHTS_REVIEW_SCHEMA_VERSION",
    "ECB_MIXED_SECTION_BLOCKER",
    "EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT",
    "EXPECTED_EVENT_COUNT",
    "EXPECTED_ORGANIZATION_EVENT_COUNTS",
    "UNVERIFIED_RIGHTS_REVIEW_LABEL",
    "build_rights_review_packet",
    "render_rights_review_markdown",
    "rights_review_packet_sha256",
    "validate_rights_review_packet_sha256",
]
