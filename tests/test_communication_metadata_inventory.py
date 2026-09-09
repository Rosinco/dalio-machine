"""Representation-specific coverage for metadata-only communication cohorts."""

from __future__ import annotations

from pathlib import Path

import pytest

from dalio.communications.institution_year_manifest import (
    load_checked_boe_2025_manifest,
    load_checked_rba_2025_manifest,
    load_checked_riksbank_2025_manifest,
)
from dalio.communications.metadata_inventory import (
    build_representation_inventory,
    render_representation_inventory_markdown,
    validate_communication_metadata_inventory_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "reference" / "communication_boe_2025_events.json"
RBA_MANIFEST = ROOT / "data" / "reference" / "communication_rba_2025_events.json"
RIKSBANK_MANIFEST = ROOT / "data" / "reference" / "communication_riksbank_2025_events.json"


def _inventory() -> dict[str, object]:
    return build_representation_inventory(load_checked_boe_2025_manifest(MANIFEST))


def _riksbank_inventory() -> dict[str, object]:
    return build_representation_inventory(load_checked_riksbank_2025_manifest(RIKSBANK_MANIFEST))


def _rba_inventory() -> dict[str, object]:
    return build_representation_inventory(load_checked_rba_2025_manifest(RBA_MANIFEST))


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


def test_riksbank_inventory_reports_replay_and_slide_links_without_content():
    inventory = _riksbank_inventory()

    assert inventory["event_count"] == inventory["expected_event_count"] == 8
    assert inventory["event_denominator_status"] == "complete"
    assert inventory["content_capture_authorized"] is False
    assert inventory["verified_rights_decisions"] == 0
    assert inventory["manifest_sha256"] == (
        "34f610043e933f74ca71ae56a9d29129fc92e2dd1ea83cc932293581ac4a2468"
    )
    assert validate_communication_metadata_inventory_sha256(inventory) == (
        "43fd289a37e28f6f8345b43d52c056cb562f87412f5f2294fcf930dda2705f3a"
    )

    coverage = _coverage(inventory)
    for representation_key in ("official_replay_page_sv", "official_slides_sv"):
        row = coverage[representation_key]
        assert row["observation_count"] == 8
        assert row["located_count"] == 8
        assert row["exact_url_count"] == 8
        assert row["not_verified_count"] == 0
        assert row["coverage_status"] == "complete"
    assert coverage["official_replay_page_sv"]["official_replay_page_link_count"] == 8
    assert coverage["official_slides_sv"]["direct_artifact_link_count"] == 8

    for representation_key in ("official_transcript_sv", "exact_caption_track_sv"):
        row = coverage[representation_key]
        assert row["observation_count"] == 8
        assert row["located_count"] == 0
        assert row["exact_url_count"] == 0
        assert row["not_verified_count"] == 8
        assert row["coverage_status"] == "not_verified"

    for row in coverage.values():
        assert (
            sum(
                row[key]
                for key in (
                    "direct_artifact_link_count",
                    "external_platform_link_count",
                    "embedded_platform_id_only_count",
                    "not_verified_count",
                )
            )
            + row.get("official_replay_page_link_count", 0)
            == row["observation_count"]
        )

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


def test_riksbank_markdown_uses_the_generic_replay_page_boundary():
    markdown = render_representation_inventory_markdown(_riksbank_inventory())

    assert "Closed event denominator: `8/8`" in markdown
    assert "`official_replay_page_sv`" in markdown
    assert "`official_slides_sv`" in markdown
    assert "8/8 | 8/8 | 8/8 | 0 | `complete`" in markdown
    assert "`official_transcript_sv`" in markdown
    assert "`exact_caption_track_sv`" in markdown
    assert "8/8 | 0/8 | 0/8 | 8 | `not_verified`" in markdown
    assert "first-party replay-page locator" in markdown
    assert "Bank of England publication" not in markdown
    assert "NO CONTENT CAPTURE IS AUTHORIZED" in markdown


def test_rba_inventory_reports_transcript_and_external_video_links_without_content():
    inventory = _rba_inventory()

    assert inventory["event_count"] == inventory["expected_event_count"] == 8
    assert inventory["event_denominator_status"] == "complete"
    assert inventory["content_capture_authorized"] is False
    assert inventory["verified_rights_decisions"] == 0
    assert inventory["manifest_sha256"] == (
        "00d6016fdf83288ceb28483760f1adac6415a9d18eea3e45bc5a21278f85aeab"
    )
    assert validate_communication_metadata_inventory_sha256(inventory) == (
        "82d5024c65e927e8007e970d9eaede1e25a194b31caa314a9927b77a01ab1309"
    )

    coverage = _coverage(inventory)
    transcript = coverage["official_transcript_en"]
    assert transcript["observation_count"] == 8
    assert transcript["located_count"] == 8
    assert transcript["exact_url_count"] == 8
    assert transcript["direct_artifact_link_count"] == 8
    assert transcript["coverage_status"] == "complete"

    video = coverage["official_page_video_locator"]
    assert video["observation_count"] == 8
    assert video["located_count"] == 8
    assert video["exact_url_count"] == 8
    assert video["external_platform_link_count"] == 8
    assert video["coverage_status"] == "complete"

    captions = coverage["exact_caption_track_en"]
    assert captions["observation_count"] == 8
    assert captions["located_count"] == 0
    assert captions["exact_url_count"] == 0
    assert captions["not_verified_count"] == 8
    assert captions["coverage_status"] == "not_verified"

    for event in inventory["events"]:
        observations = {row["representation_key"]: row for row in event["representations"]}
        transcript_locator = observations["official_transcript_en"]["locator"]
        assert transcript_locator["locator_kind"] == "official_direct_artifact"
        assert transcript_locator["mime_type"] == "text/html"
        assert transcript_locator["locator_url"].startswith("https://www.rba.gov.au/")
        video_locator = observations["official_page_video_locator"]["locator"]
        assert video_locator["locator_kind"] == "external_platform_page"
        assert video_locator["mime_type"] == "text/html"
        assert video_locator["locator_url"] == (
            f"https://youtu.be/{video_locator['platform_media_id']}"
        )
        assert observations["exact_caption_track_en"]["locator"] is None

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


def test_rba_markdown_uses_the_external_platform_hosting_boundary():
    markdown = render_representation_inventory_markdown(_rba_inventory())

    assert "Closed event denominator: `8/8`" in markdown
    assert "`official_transcript_en`" in markdown
    assert "`official_page_video_locator`" in markdown
    assert "8/8 | 8/8 | 8/8 | 0 | `complete`" in markdown
    assert "`exact_caption_track_en`" in markdown
    assert "8/8 | 0/8 | 0/8 | 8 | `not_verified`" in markdown
    assert "External platform hosting is distinct from first-party institutional publication" in (
        markdown
    )
    assert "first-party replay-page locator" not in markdown
    assert "Bank of England publication" not in markdown
    assert "NO CONTENT CAPTURE IS AUTHORIZED" in markdown
