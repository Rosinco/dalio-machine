"""Offline, fail-closed review boundary for communication-source rights."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from dalio.communications.pilot_manifest import load_pilot_manifest
from dalio.communications.rights_review import (
    ECB_MULTI_SECTION_MAPPING_NOTE,
    EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT,
    EXPECTED_EVENT_COUNT,
    UNVERIFIED_RIGHTS_REVIEW_LABEL,
    build_rights_review_packet,
    render_rights_review_markdown,
    rights_review_packet_sha256,
)

_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _ROOT / "data/reference/communication_pilot_events.json"


@pytest.fixture
def pilot_manifest():
    return load_pilot_manifest(_MANIFEST_PATH)


def _all_keys(value):
    if isinstance(value, dict):
        yield from value
        for nested in value.values():
            yield from _all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _all_keys(nested)


def test_packet_is_deterministic_hash_bound_and_loudly_fail_closed(pilot_manifest):
    packet = build_rights_review_packet(pilot_manifest)
    repeated = build_rights_review_packet(pilot_manifest)

    assert packet == repeated
    assert packet["review_label"] == UNVERIFIED_RIGHTS_REVIEW_LABEL
    assert packet["notice"].startswith("UNVERIFIED RIGHTS REVIEW")
    assert packet["content_capture_authorized"] is False
    assert packet["verified_rights_decisions"] == 0
    assert packet["expected_event_count"] == EXPECTED_EVENT_COUNT == 16
    assert packet["event_count"] == 16
    assert (
        packet["expected_available_chosen_representation_count"]
        == EXPECTED_AVAILABLE_CHOSEN_REPRESENTATION_COUNT
        == 16
    )
    assert packet["available_chosen_representation_count"] == 16
    assert packet["packet_sha256"] == (
        "f841b5875835f9ef1d469c6484386b88d8f129fee72d1ff62e12cd59768cdd31"
    )

    without_self_hash = {key: value for key, value in packet.items() if key != "packet_sha256"}
    assert packet["packet_sha256"] == rights_review_packet_sha256(without_self_hash)
    with pytest.raises(ValueError, match="must be excluded"):
        rights_review_packet_sha256(packet)

    forbidden_content_fields = {
        "archive_reference",
        "blob_path",
        "content_sha256",
        "extraction",
        "segments",
    }
    assert forbidden_content_fields.isdisjoint(_all_keys(packet))


def test_markdown_refuses_a_stale_or_malformed_packet_self_hash(pilot_manifest):
    packet = build_rights_review_packet(pilot_manifest)
    packet["event_count"] = 15

    with pytest.raises(ValueError, match="does not match its canonical payload"):
        render_rights_review_markdown(packet)

    packet["packet_sha256"] = "not-a-sha256"
    with pytest.raises(ValueError, match="full lowercase SHA-256"):
        render_rights_review_markdown(packet)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("durable_metadata_blockers", ["stale"], "blockers must be empty"),
        ("durable_metadata_notes", [], "notes have an invalid shape"),
    ],
)
def test_markdown_refuses_self_consistent_invalid_durable_metadata_shape(
    pilot_manifest,
    field,
    value,
    message,
):
    packet = build_rights_review_packet(pilot_manifest)
    packet[field] = value
    packet.pop("packet_sha256")
    packet["packet_sha256"] = rights_review_packet_sha256(packet)

    with pytest.raises(ValueError, match=message):
        render_rights_review_markdown(packet)


def test_packet_reconciles_eight_events_and_candidates_per_organization(pilot_manifest):
    packet = build_rights_review_packet(pilot_manifest)

    reconciliation = {
        item["organization_id"]: item for item in packet["organization_reconciliation"]
    }
    assert set(reconciliation) == {"ecb", "federal_reserve"}
    for item in reconciliation.values():
        assert item["expected_event_count"] == 8
        assert item["manifest_event_count"] == 8
        assert item["available_chosen_representation_count"] == 8
        assert item["missing_event_keys"] == []
        assert item["unexpected_event_keys"] == []
        assert item["rights_review"]["status"] == "pending"
        assert item["rights_review"]["basis_url"].startswith("https://")
        assert item["rights_review"]["questions"]

    by_organization = {
        organization_id: [
            event for event in packet["events"] if event["organization_id"] == organization_id
        ]
        for organization_id in reconciliation
    }
    assert {key: len(value) for key, value in by_organization.items()} == {
        "ecb": 8,
        "federal_reserve": 8,
    }
    for event in packet["events"]:
        representation = event["chosen_representation"]
        candidate = representation["candidate"]
        assert representation["availability_status"] == "available"
        assert representation["status_evidence_url"].startswith("https://")
        assert candidate["landing_url"].startswith("https://")
        assert candidate["artifact_url"].startswith("https://")
        assert candidate["available_at"].endswith("Z")
        assert candidate["retrieved_at"].endswith("Z")
        assert candidate["metadata_known_at"].endswith("Z")
        assert candidate["origin_type"]
        assert candidate["provenance_tier"]
        assert candidate["host_organization"]
        assert candidate["publisher"]
        assert candidate["transcriber_attribution"]


def test_semantically_set_like_manifest_order_does_not_change_packet(pilot_manifest):
    reordered_organizations = []
    for denominator in reversed(pilot_manifest.scope.organizations):
        reordered_organizations.append(
            replace(
                denominator,
                event_keys=tuple(reversed(denominator.event_keys)),
                rights_review=replace(
                    denominator.rights_review,
                    questions=tuple(reversed(denominator.rights_review.questions)),
                ),
            )
        )
    reordered_events = tuple(reversed(pilot_manifest.events))
    reordered = replace(
        pilot_manifest,
        scope=replace(
            pilot_manifest.scope,
            organizations=tuple(reordered_organizations),
        ),
        events=reordered_events,
    )

    assert build_rights_review_packet(reordered) == build_rights_review_packet(pilot_manifest)


def test_section_scope_order_is_semantic(pilot_manifest):
    ecb = next(
        item for item in pilot_manifest.scope.organizations if item.organization_id == "ecb"
    )
    reordered_denominators = tuple(
        replace(
            item,
            representation_spec=replace(
                item.representation_spec,
                section_coverage=tuple(reversed(item.representation_spec.section_coverage)),
            ),
        )
        if item.organization_id == "ecb"
        else item
        for item in pilot_manifest.scope.organizations
    )
    reordered_events = tuple(
        replace(
            event,
            representation=replace(
                event.representation,
                section_coverage=tuple(reversed(event.representation.section_coverage)),
            ),
        )
        if event.organization_id == "ecb"
        else event
        for event in pilot_manifest.events
    )
    reordered = replace(
        pilot_manifest,
        scope=replace(pilot_manifest.scope, organizations=reordered_denominators),
        events=reordered_events,
    )

    assert ecb.representation_spec.section_coverage == ("prepared_remarks", "q_and_a")
    assert build_rights_review_packet(reordered) != build_rights_review_packet(pilot_manifest)


def test_markdown_shows_pending_basis_candidates_provenance_and_ecb_mapping(pilot_manifest):
    packet = build_rights_review_packet(pilot_manifest)
    markdown = render_rights_review_markdown(packet)

    assert packet["durable_metadata_blockers"] == []
    assert packet["durable_metadata_notes"] == [
        {
            "organization_id": "ecb",
            "status": "ordered_multi_section_mapping_available",
            "message": ECB_MULTI_SECTION_MAPPING_NOTE,
        }
    ]
    assert "**UNVERIFIED RIGHTS REVIEW — NO CONTENT CAPTURE IS AUTHORIZED.**" in markdown
    assert "Content capture authorized: `false`" in markdown
    assert "Verified rights decisions: `0`" in markdown
    assert "Closed denominator: `16/16` events" in markdown
    assert ECB_MULTI_SECTION_MAPPING_NOTE in markdown
    assert "ordered, provenance-specific" in markdown
    assert "resolves the structural mapping blocker only" in markdown
    assert "Pending source-level questions:" in markdown
    assert "https://www.federalreserve.gov/disclaimer.htm" in markdown
    assert "https://www.ecb.europa.eu/services/using-our-site/disclaimer" in markdown
    assert "Provenance:" in markdown
    assert "Clocks — published:" in markdown
    for event in packet["events"]:
        assert event["event_key"] in markdown
        assert event["chosen_representation"]["candidate"]["artifact_url"] in markdown


def test_builder_fails_closed_on_incomplete_or_nonpending_manifest(pilot_manifest):
    incomplete = replace(pilot_manifest, events=pilot_manifest.events[:-1])
    with pytest.raises(ValueError, match="exactly 16 events"):
        build_rights_review_packet(incomplete)

    first_event = pilot_manifest.events[0]
    unavailable = replace(
        pilot_manifest,
        events=(
            replace(
                first_event,
                representation=replace(
                    first_event.representation,
                    availability_status="missing",
                ),
            ),
            *pilot_manifest.events[1:],
        ),
    )
    with pytest.raises(ValueError, match="available chosen representation"):
        build_rights_review_packet(unavailable)

    wrong_coverage = replace(
        pilot_manifest,
        events=(
            replace(
                first_event,
                representation=replace(
                    first_event.representation,
                    section_coverage=("prepared_remarks",),
                ),
            ),
            *pilot_manifest.events[1:],
        ),
    )
    with pytest.raises(ValueError, match="section coverage"):
        build_rights_review_packet(wrong_coverage)

    first_denominator = pilot_manifest.scope.organizations[0]
    decided = replace(
        pilot_manifest,
        scope=replace(
            pilot_manifest.scope,
            organizations=(
                replace(
                    first_denominator,
                    rights_review=replace(first_denominator.rights_review, status="cleared"),
                ),
                *pilot_manifest.scope.organizations[1:],
            ),
        ),
    )
    with pytest.raises(ValueError, match="must remain pending"):
        build_rights_review_packet(decided)


def test_builder_packet_contains_no_local_filesystem_path(pilot_manifest):
    packet = build_rights_review_packet(pilot_manifest)
    serialized = json.dumps(packet, ensure_ascii=False, sort_keys=True)
    markdown = render_rights_review_markdown(packet)

    assert str(_ROOT) not in serialized
    assert str(_ROOT) not in markdown
    assert "\\wsl.localhost" not in serialized
    assert "\\wsl.localhost" not in markdown
