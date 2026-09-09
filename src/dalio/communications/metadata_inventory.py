"""Deterministic representation coverage for metadata-only communication cohorts."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
from datetime import UTC, date, datetime

from dalio.communications.institution_year_manifest import (
    InstitutionYearManifest,
    institution_year_manifest_sha256,
)

COMMUNICATION_METADATA_INVENTORY_SCHEMA_VERSION = 1
COMMUNICATION_METADATA_INVENTORY_METHODOLOGY_VERSION = "communication-metadata-inventory-v1"

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _iso_datetime(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("inventory clocks must include a timezone")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def communication_metadata_inventory_sha256(payload_without_hash: dict[str, object]) -> str:
    """Return the canonical self-excluding inventory fingerprint."""
    if "inventory_sha256" in payload_without_hash:
        raise ValueError("inventory_sha256 must be excluded while computing its own hash")
    return hashlib.sha256(_canonical_json(payload_without_hash)).hexdigest()


def validate_communication_metadata_inventory_sha256(inventory: dict[str, object]) -> str:
    """Verify and return a metadata inventory's self-excluding fingerprint."""
    if not isinstance(inventory, dict):
        raise TypeError("communication metadata inventory must be a dictionary")
    supplied = inventory.get("inventory_sha256")
    if not isinstance(supplied, str) or _SHA256.fullmatch(supplied) is None:
        raise ValueError("communication metadata inventory lacks a full SHA-256")
    expected = communication_metadata_inventory_sha256(
        {key: value for key, value in inventory.items() if key != "inventory_sha256"}
    )
    if not hmac.compare_digest(supplied, expected):
        raise ValueError("inventory_sha256 does not match the canonical payload")
    return supplied


def _coverage_status(
    *,
    completeness_basis: str,
    denominator_count: int,
    located_count: int,
    exact_url_count: int,
) -> str:
    numerator = (
        located_count if completeness_basis == "official_page_media_locator" else exact_url_count
    )
    if numerator == 0:
        return "not_verified"
    if numerator == denominator_count:
        return "complete"
    return "partial"


def _locator_payload(locator: object) -> dict[str, object] | None:
    if locator is None:
        return None
    return {
        "locator_key": locator.locator_key,
        "source_id": locator.source_id,
        "locator_kind": locator.locator_kind,
        "locator_url": locator.locator_url,
        "platform_media_id": locator.platform_media_id,
        "mime_type": locator.mime_type,
        "language": locator.language,
        "translation_status": locator.translation_status,
        "host_organization": locator.host_organization,
        "publisher": locator.publisher,
        "transcriber": locator.transcriber,
        "transcriber_attribution": locator.transcriber_attribution,
        "origin_type": locator.origin_type,
        "provenance_tier": locator.provenance_tier,
        "published_at": _iso_datetime(locator.published_at),
        "observed_at": _iso_datetime(locator.observed_at),
    }


def build_representation_inventory(
    manifest: InstitutionYearManifest,
) -> dict[str, object]:
    """Build an offline link inventory without database, network, or content reads."""
    denominator_count = len(manifest.scope.event_keys)
    events = sorted(manifest.events, key=lambda item: (item.event_date, item.event_key))
    spec_by_key = {spec.representation_key: spec for spec in manifest.scope.representation_specs}
    coverage: list[dict[str, object]] = []
    for representation_key, spec in sorted(spec_by_key.items()):
        observations = [
            next(
                observation
                for observation in event.representations
                if observation.representation_key == representation_key
            )
            for event in events
        ]
        status_counts = {
            status: sum(item.availability_status == status for item in observations)
            for status in (
                "direct_artifact_link",
                "external_platform_link",
                "embedded_platform_id_only",
                "official_replay_page_link",
                "not_verified",
            )
        }
        located_count = sum(item.locator is not None for item in observations)
        exact_url_count = sum(
            item.locator is not None and item.locator.locator_url is not None
            for item in observations
        )
        row: dict[str, object] = {
            "representation_key": representation_key,
            "source_id": spec.source_id,
            "artifact_role": spec.artifact_role,
            "material_type": spec.material_type,
            "section_coverage": list(spec.section_coverage),
            "completeness_basis": spec.completeness_basis,
            "denominator_count": denominator_count,
            "observation_count": len(observations),
            "located_count": located_count,
            "exact_url_count": exact_url_count,
            "direct_artifact_link_count": status_counts["direct_artifact_link"],
            "external_platform_link_count": status_counts["external_platform_link"],
            "embedded_platform_id_only_count": status_counts["embedded_platform_id_only"],
            "not_verified_count": status_counts["not_verified"],
            "coverage_status": _coverage_status(
                completeness_basis=spec.completeness_basis,
                denominator_count=denominator_count,
                located_count=located_count,
                exact_url_count=exact_url_count,
            ),
            "rights_status": spec.rights_status,
            "acquisition_status": spec.acquisition_status,
            "automated_collection_allowed": spec.automated_collection_allowed,
        }
        if status_counts["official_replay_page_link"]:
            # Add the new counter only where the status is present. Historical
            # BoE inventories predate this vocabulary and remain byte-identical.
            row["official_replay_page_link_count"] = status_counts["official_replay_page_link"]
        coverage.append(row)

    event_rows = []
    for event in events:
        event_rows.append(
            {
                "event_key": event.event_key,
                "title": event.title,
                "event_date": event.event_date.isoformat(),
                "metadata_known_at": _iso_datetime(event.metadata_known_at),
                "representations": [
                    {
                        "representation_key": observation.representation_key,
                        "availability_status": observation.availability_status,
                        "checked_at": _iso_datetime(observation.checked_at),
                        "status_evidence_url": observation.status_evidence_url,
                        "status_note": observation.status_note,
                        "locator": _locator_payload(observation.locator),
                    }
                    for observation in sorted(
                        event.representations, key=lambda item: item.representation_key
                    )
                ],
            }
        )

    inventory: dict[str, object] = {
        "schema_version": COMMUNICATION_METADATA_INVENTORY_SCHEMA_VERSION,
        "methodology_version": COMMUNICATION_METADATA_INVENTORY_METHODOLOGY_VERSION,
        "inventory_kind": "communication_link_metadata_inventory",
        "notice": (
            "METADATA ONLY — link and platform-ID observations do not authorize content "
            "capture, prove transcript or caption fidelity, or make a communication corpus "
            "analysis-ready."
        ),
        "content_capture_authorized": False,
        "verified_rights_decisions": 0,
        "manifest_id": manifest.manifest_id,
        "manifest_schema_version": manifest.schema_version,
        "manifest_methodology_version": manifest.methodology_version,
        "manifest_sha256": institution_year_manifest_sha256(manifest),
        "catalogue_sha256": manifest.catalogue_sha256,
        "created_by": manifest.created_by,
        "created_at": _iso_datetime(manifest.created_at),
        "as_known_at": _iso_datetime(manifest.as_known_at),
        "organization_id": manifest.scope.organization_id,
        "year": manifest.scope.year,
        "scope": {
            "start_date": manifest.scope.start_date.isoformat(),
            "end_date": manifest.scope.end_date.isoformat(),
            "event_type": manifest.scope.event_type,
            "denominator_source_url": manifest.scope.denominator_source_url,
            "inclusion_rule": manifest.scope.inclusion_rule,
            "exclusions": sorted(manifest.scope.exclusions),
            "checked_by": manifest.scope.checked_by,
            "checked_at": _iso_datetime(manifest.scope.checked_at),
        },
        "expected_event_count": denominator_count,
        "event_count": len(events),
        "event_denominator_status": (
            "complete" if len(events) == denominator_count else "incomplete"
        ),
        "representation_coverage": coverage,
        "events": event_rows,
    }
    inventory["inventory_sha256"] = communication_metadata_inventory_sha256(inventory)
    validate_communication_metadata_inventory_sha256(inventory)
    return inventory


def _markdown_value(value: object) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def render_representation_inventory_markdown(inventory: dict[str, object]) -> str:
    """Render the deterministic metadata coverage sheet."""
    validate_communication_metadata_inventory_sha256(inventory)
    if inventory.get("schema_version") != COMMUNICATION_METADATA_INVENTORY_SCHEMA_VERSION:
        raise ValueError("unsupported communication metadata inventory schema")
    if inventory.get("inventory_kind") != "communication_link_metadata_inventory":
        raise ValueError("unsupported communication metadata inventory kind")
    if inventory.get("content_capture_authorized") is not False:
        raise ValueError("metadata inventory cannot authorize content capture")
    if inventory.get("verified_rights_decisions") != 0:
        raise ValueError("metadata inventory cannot contain rights decisions")

    scope = inventory["scope"]
    if inventory.get("organization_id") == "bank_of_england":
        hosting_boundary = (
            "- External platform hosting is distinct from Bank of England publication and "
            "does not establish caption origin, producer, language or time coverage."
        )
    else:
        hosting_boundary = (
            "- A first-party replay-page locator identifies an event page, not captured media, "
            "and does not establish caption origin, producer, language or time coverage."
        )
    lines = [
        "# Institutional-communications metadata inventory",
        "",
        "**METADATA ONLY — NO CONTENT CAPTURE IS AUTHORIZED.**",
        "",
        str(inventory["notice"]),
        "",
        f"Manifest: `{inventory['manifest_id']}`  ",
        f"Manifest SHA-256: `{inventory['manifest_sha256']}`  ",
        f"Institution/year: `{inventory['organization_id']}` / `{inventory['year']}`  ",
        f"Public-information cutoff: `{inventory['as_known_at']}`  ",
        f"Inventory SHA-256: `{inventory['inventory_sha256']}`  ",
        f"Closed event denominator: `{inventory['event_count']}/{inventory['expected_event_count']}`",
        "",
        "## Representation-specific coverage",
        "",
        "| Representation | Basis | Observed | Located | Exact URL | Not verified | Status |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for row in inventory["representation_coverage"]:
        lines.append(
            f"| `{row['representation_key']}` | `{row['completeness_basis']}` | "
            f"{row['observation_count']}/{row['denominator_count']} | "
            f"{row['located_count']}/{row['denominator_count']} | "
            f"{row['exact_url_count']}/{row['denominator_count']} | "
            f"{row['not_verified_count']} | `{row['coverage_status']}` |"
        )
    lines.extend(
        [
            "",
            "A video locator is not a caption locator. An embedded platform ID counts only "
            "toward the official-page media-locator measure; it does not count as an exact URL.",
            "",
            "## Closed denominator",
            "",
            f"Official source: <{scope['denominator_source_url']}>  ",
            f"Inclusion rule: {scope['inclusion_rule']}  ",
            f"Checked: `{scope['checked_at']}` by `{scope['checked_by']}`",
            "",
            "## Events and link observations",
            "",
        ]
    )
    for event in inventory["events"]:
        lines.extend([f"### {event['event_date']} — {event['title']}", ""])
        for observation in event["representations"]:
            locator = observation["locator"]
            if locator is None:
                location = "no exact locator"
            elif locator["locator_url"] is not None:
                location = f"<{locator['locator_url']}>"
            else:
                location = f"platform ID `{locator['platform_media_id']}`; no exact URL"
            lines.append(
                f"- `{observation['representation_key']}` — "
                f"`{observation['availability_status']}` — {location}"
            )
        lines.append("")
    lines.extend(
        [
            "## Boundaries",
            "",
            "- All representation rights remain source-gated; automation is disabled.",
            "- No source bytes, transcript text, captions, extraction, claims, scores, "
            "forecasts or portfolio conclusions are present.",
            hosting_boundary,
            "- `not_verified` is an unresolved metadata state, not evidence that a caption "
            "track is absent.",
            "",
        ]
    )
    return "\n".join(lines)


def inventory_publication_date(inventory: dict[str, object]) -> date:
    """Return the UTC date used for deterministic output naming."""
    raw = inventory.get("as_known_at")
    if not isinstance(raw, str):
        raise ValueError("inventory as_known_at is missing")
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).date()


__all__ = [
    "COMMUNICATION_METADATA_INVENTORY_METHODOLOGY_VERSION",
    "COMMUNICATION_METADATA_INVENTORY_SCHEMA_VERSION",
    "build_representation_inventory",
    "communication_metadata_inventory_sha256",
    "inventory_publication_date",
    "render_representation_inventory_markdown",
    "validate_communication_metadata_inventory_sha256",
]
