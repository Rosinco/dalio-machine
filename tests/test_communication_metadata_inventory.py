"""Representation-specific coverage for metadata-only communication cohorts."""

from __future__ import annotations

from pathlib import Path

import pytest

from dalio.communications.institution_year_manifest import load_checked_boe_2025_manifest
from dalio.communications.metadata_inventory import (
    build_representation_inventory,
    render_representation_inventory_markdown,
    validate_communication_metadata_inventory_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "reference" / "communication_boe_2025_events.json"


def _inventory() -> dict[str, object]:
    return build_representation_inventory(load_checked_boe_2025_manifest(MANIFEST))


def _coverage(inventory: dict[str, object]) -> dict[str, dict[str, object]]:
    return {row["representation_key"]: row for row in inventory["representation_coverage"]}


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(map(_keys, value.values())))
    if isinstance(value, list):
        return set().union(*(map(_keys, value)), set())
    return set()


def test_inventory_reports_each_representation_against_the_same_closed_denominator():
    inventory = _inventory()

    assert inventory["event_count"] == inventory["expected_event_count"] == 4
    assert inventory["event_denominator_status"] == "complete"
    assert inventory["content_capture_authorized"] is False
    assert inventory["verified_rights_decisions"] == 0
    assert inventory["manifest_sha256"] == (
        "4bba6c8415de46718a5ae6906d0d09e9041f7be1903dbeef7be4f40223717eb2"
    )
    assert validate_communication_metadata_inventory_sha256(inventory) == (
        "97c6116f1c0b94a7d9f6b01c8bebbdff1bc13b84b0e760665b4af24b870cc230"
    )

    coverage = _coverage(inventory)
    transcript = coverage["official_transcript_en"]
    assert transcript["observation_count"] == 4
    assert transcript["located_count"] == 4
    assert transcript["exact_url_count"] == 4
    assert transcript["direct_artifact_link_count"] == 4
    assert transcript["coverage_status"] == "complete"

    video = coverage["official_page_video_locator"]
    assert video["observation_count"] == 4
    assert video["located_count"] == 4
    assert video["exact_url_count"] == 3
    assert video["external_platform_link_count"] == 3
    assert video["embedded_platform_id_only_count"] == 1
    assert video["coverage_status"] == "complete"

    captions = coverage["exact_caption_track_en"]
    assert captions["observation_count"] == 4
    assert captions["located_count"] == 0
    assert captions["exact_url_count"] == 0
    assert captions["not_verified_count"] == 4
    assert captions["coverage_status"] == "not_verified"


def test_inventory_contains_link_metadata_but_no_content_or_decisions():
    inventory = _inventory()
    forbidden = {
        "content",
        "content_sha256",
        "blob_path",
        "extraction",
        "segments",
        "claims",
        "reviewed_by",
        "decision",
    }
    assert not forbidden.intersection(_keys(inventory))
    assert all(
        row["rights_status"] == "rights_review_required"
        and row["acquisition_status"] == "manual_review_required"
        and row["automated_collection_allowed"] is False
        for row in inventory["representation_coverage"]
    )


def test_inventory_preserves_may_id_only_and_caption_unknown_states():
    inventory = _inventory()
    may = next(row for row in inventory["events"] if row["event_key"] == "boe_mpr_2025_05_08")
    observations = {row["representation_key"]: row for row in may["representations"]}
    video = observations["official_page_video_locator"]
    assert video["availability_status"] == "embedded_platform_id_only"
    assert video["locator"]["platform_media_id"] == "DAEab7yDUmE"
    assert video["locator"]["locator_url"] is None
    caption = observations["exact_caption_track_en"]
    assert caption["availability_status"] == "not_verified"
    assert caption["locator"] is None


def test_inventory_hash_rejects_tampering():
    inventory = _inventory()
    inventory["event_count"] = 3
    with pytest.raises(ValueError, match="does not match"):
        validate_communication_metadata_inventory_sha256(inventory)


def test_markdown_keeps_counts_and_boundaries_explicit():
    markdown = render_representation_inventory_markdown(_inventory())

    assert "Closed event denominator: `4/4`" in markdown
    assert "`official_transcript_en`" in markdown
    assert "4/4 | 4/4 | 4/4 | 0 | `complete`" in markdown
    assert "`official_page_video_locator`" in markdown
    assert "4/4 | 4/4 | 3/4 | 0 | `complete`" in markdown
    assert "`exact_caption_track_en`" in markdown
    assert "4/4 | 0/4 | 0/4 | 4 | `not_verified`" in markdown
    assert "platform ID `DAEab7yDUmE`; no exact URL" in markdown
    assert "video locator is not a caption locator" in markdown.lower()
    assert "NO CONTENT CAPTURE IS AUTHORIZED" in markdown
