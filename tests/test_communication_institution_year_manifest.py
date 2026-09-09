"""Closed one-institution/year communication metadata manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dalio.communications.institution_year_manifest import (
    BOE_2025_MANIFEST_SHA256,
    INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION,
    INSTITUTION_YEAR_METHODOLOGY_VERSION,
    institution_year_manifest_sha256,
    load_checked_boe_2025_manifest,
    load_institution_year_manifest,
)
from dalio.communications.metadata_inventory import build_representation_inventory

ROOT = Path(__file__).resolve().parents[1]
CHECKED_MANIFEST = ROOT / "data" / "reference" / "communication_boe_2025_events.json"
EVENT_KEYS = {
    "boe_mpr_2025_02_06",
    "boe_mpr_2025_05_08",
    "boe_mpr_2025_08_07",
    "boe_mpr_2025_11_06",
}
REPRESENTATION_KEYS = {
    "official_transcript_en",
    "official_page_video_locator",
    "exact_caption_track_en",
}


def _payload() -> dict[str, object]:
    return json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _scope(payload: dict[str, object]) -> dict[str, object]:
    scope = payload["scope"]
    assert isinstance(scope, dict)
    return scope


def _events(payload: dict[str, object]) -> list[dict[str, object]]:
    events = payload["events"]
    assert isinstance(events, list)
    return events


def _representations(event: dict[str, object]) -> list[dict[str, object]]:
    representations = event["representations"]
    assert isinstance(representations, list)
    return representations


def _representation(event: dict[str, object], key: str) -> dict[str, object]:
    return next(item for item in _representations(event) if item["representation_key"] == key)


def test_checked_boe_manifest_is_closed_metadata_only_and_hash_pinned():
    before = CHECKED_MANIFEST.read_bytes()
    manifest = load_checked_boe_2025_manifest(CHECKED_MANIFEST)

    assert manifest.schema_version == INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION == 1
    assert manifest.methodology_version == INSTITUTION_YEAR_METHODOLOGY_VERSION
    assert manifest.scope.organization_id == "bank_of_england"
    assert manifest.scope.year == 2025
    assert set(manifest.scope.event_keys) == EVENT_KEYS
    assert {event.event_key for event in manifest.events} == EVENT_KEYS
    assert len(manifest.events) == 4
    assert manifest.content_capture_authorized is False
    assert institution_year_manifest_sha256(manifest) == BOE_2025_MANIFEST_SHA256
    assert CHECKED_MANIFEST.read_bytes() == before

    specs = {spec.representation_key: spec for spec in manifest.scope.representation_specs}
    assert set(specs) == REPRESENTATION_KEYS
    assert all(spec.rights_status == "rights_review_required" for spec in specs.values())
    assert all(spec.acquisition_status == "manual_review_required" for spec in specs.values())
    assert all(spec.automated_collection_allowed is False for spec in specs.values())

    statuses = {
        key: [
            next(item for item in event.representations if item.representation_key == key)
            for event in manifest.events
        ]
        for key in REPRESENTATION_KEYS
    }
    assert all(
        item.availability_status == "direct_artifact_link"
        for item in statuses["official_transcript_en"]
    )
    assert [item.availability_status for item in statuses["official_page_video_locator"]].count(
        "external_platform_link"
    ) == 3
    assert [item.availability_status for item in statuses["official_page_video_locator"]].count(
        "embedded_platform_id_only"
    ) == 1
    assert all(
        item.availability_status == "not_verified" for item in statuses["exact_caption_track_en"]
    )
    assert all(item.locator is None for item in statuses["exact_caption_track_en"])
    assert not hasattr(manifest, "content")


def test_semantic_hash_ignores_record_order_but_preserves_section_order(tmp_path):
    original = load_institution_year_manifest(CHECKED_MANIFEST)
    reordered = _payload()
    _events(reordered).reverse()
    scope = _scope(reordered)
    scope["event_keys"].reverse()
    scope["exclusions"].reverse()
    scope["representation_specs"].reverse()
    for event in _events(reordered):
        _representations(event).reverse()
    assert institution_year_manifest_sha256(
        load_institution_year_manifest(_write(tmp_path, reordered))
    ) == institution_year_manifest_sha256(original)

    changed = _payload()
    specs = _scope(changed)["representation_specs"]
    assert isinstance(specs, list)
    transcript = next(
        item for item in specs if item["representation_key"] == "official_transcript_en"
    )
    transcript["section_coverage"] = ["full_transcript", "subtitles"]
    with pytest.raises(ValueError, match="section_coverage"):
        load_institution_year_manifest(_write(tmp_path, changed))


def test_duplicate_json_fields_and_forbidden_content_fields_are_rejected(tmp_path):
    raw = CHECKED_MANIFEST.read_text(encoding="utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON field"):
        load_institution_year_manifest(path)

    for forbidden in ("content", "content_sha256", "reviewed_by", "decision"):
        payload = _payload()
        _representation(_events(payload)[0], "official_transcript_en")[forbidden] = "forbidden"
        with pytest.raises(ValueError, match="unknown fields"):
            load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    payload["verified_rights_decisions"] = 0
    with pytest.raises(ValueError, match="unknown fields"):
        load_institution_year_manifest(_write(tmp_path, payload))


@pytest.mark.parametrize("invalid", [True, 1.0, False, 0.0])
def test_schema_version_is_type_strict(tmp_path, invalid):
    payload = _payload()
    payload["schema_version"] = invalid
    with pytest.raises(ValueError, match="schema_version"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_denominator_missing_duplicate_and_cross_year_events_are_rejected(tmp_path):
    missing = _payload()
    _events(missing).pop()
    with pytest.raises(ValueError, match="do not reconcile"):
        load_institution_year_manifest(_write(tmp_path, missing))

    duplicate = _payload()
    _events(duplicate).append(_events(duplicate)[0])
    with pytest.raises(ValueError, match="duplicate event keys"):
        load_institution_year_manifest(_write(tmp_path, duplicate))

    wrong_year = _payload()
    _events(wrong_year)[0]["event_date"] = "2024-02-06"
    with pytest.raises(ValueError, match="outside the closed year"):
        load_institution_year_manifest(_write(tmp_path, wrong_year))


def test_every_event_must_observe_each_representation_exactly_once(tmp_path):
    missing = _payload()
    _representations(_events(missing)[0]).pop()
    with pytest.raises(ValueError, match="cover every declared representation"):
        load_institution_year_manifest(_write(tmp_path, missing))

    duplicate = _payload()
    first = _representations(_events(duplicate)[0])[0]
    _representations(_events(duplicate)[0]).append(first)
    with pytest.raises(ValueError, match="duplicate keys"):
        load_institution_year_manifest(_write(tmp_path, duplicate))


def test_video_locator_and_caption_uncertainty_cannot_be_conflated(tmp_path):
    payload = _payload()
    may = next(event for event in _events(payload) if event["event_key"] == "boe_mpr_2025_05_08")
    video = _representation(may, "official_page_video_locator")
    locator = video["locator"]
    assert isinstance(locator, dict)
    assert locator["platform_media_id"] == "DAEab7yDUmE"
    assert locator["locator_url"] is None
    assert video["availability_status"] == "embedded_platform_id_only"

    locator["locator_url"] = "https://youtube.com/live/DAEab7yDUmE?feature=share"
    with pytest.raises(ValueError, match="ID-only locator cannot invent"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    caption = _representation(_events(payload)[0], "exact_caption_track_en")
    caption["availability_status"] = "external_platform_link"
    caption["locator"] = _representation(_events(payload)[0], "official_page_video_locator")[
        "locator"
    ]
    with pytest.raises(ValueError, match="conflicts with its representation specification"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_unverified_transcript_is_allowed_and_reduces_coverage(tmp_path):
    payload = _payload()
    transcript = _representation(_events(payload)[0], "official_transcript_en")
    transcript["availability_status"] = "not_verified"
    transcript["locator"] = None

    inventory = build_representation_inventory(
        load_institution_year_manifest(_write(tmp_path, payload))
    )
    coverage = next(
        row
        for row in inventory["representation_coverage"]
        if row["representation_key"] == "official_transcript_en"
    )
    assert coverage["located_count"] == 3
    assert coverage["exact_url_count"] == 3
    assert coverage["not_verified_count"] == 1
    assert coverage["coverage_status"] == "partial"


def test_exact_caption_track_requires_resolved_platform_provenance(tmp_path):
    payload = _payload()
    event = _events(payload)[0]
    video_locator = _representation(event, "official_page_video_locator")["locator"]
    assert isinstance(video_locator, dict)
    video_id = video_locator["platform_media_id"]
    caption = _representation(event, "exact_caption_track_en")
    caption["availability_status"] = "external_platform_link"
    caption["locator"] = {
        "locator_key": "boe_mpr_caption_2025_02_06",
        "source_id": "boe_monetary_policy_press_conference_subtitles_en",
        "locator_kind": "exact_caption_track",
        "locator_url": f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en",
        "platform_media_id": video_id,
        "mime_type": "text/vtt",
        "language": "en",
        "translation_status": "original",
        "host_organization": "YouTube",
        "publisher": "Bank of England",
        "transcriber": "YouTube",
        "transcriber_attribution": "artifact_specific",
        "origin_type": "automatic_caption",
        "provenance_tier": "official_hosted_third_party",
        "published_at": None,
        "observed_at": caption["checked_at"],
    }

    inventory = build_representation_inventory(
        load_institution_year_manifest(_write(tmp_path, payload))
    )
    coverage = next(
        row
        for row in inventory["representation_coverage"]
        if row["representation_key"] == "exact_caption_track_en"
    )
    assert coverage["located_count"] == 1
    assert coverage["exact_url_count"] == 1
    assert coverage["coverage_status"] == "partial"

    mismatched = _payload()
    mismatched_event = _events(mismatched)[0]
    mismatched_caption = _representation(mismatched_event, "exact_caption_track_en")
    mismatched_caption["availability_status"] = "external_platform_link"
    mismatched_caption["locator"] = dict(caption["locator"])
    mismatched_caption["locator"]["locator_url"] = (
        "https://www.youtube.com/api/timedtext?v=DAEab7yDUmE&lang=en"
    )
    mismatched_caption["locator"]["platform_media_id"] = "DAEab7yDUmE"
    with pytest.raises(ValueError, match="does not match a video locator for this event"):
        load_institution_year_manifest(_write(tmp_path, mismatched))

    translated = _payload()
    translated_event = _events(translated)[0]
    translated_caption = _representation(translated_event, "exact_caption_track_en")
    translated_caption["availability_status"] = "external_platform_link"
    translated_caption["locator"] = dict(caption["locator"])
    translated_caption["locator"]["locator_url"] += "&tlang=sv"
    with pytest.raises(ValueError, match="translated track declared original"):
        load_institution_year_manifest(_write(tmp_path, translated))


def test_external_and_official_locator_domains_are_separate(tmp_path):
    payload = _payload()
    february = _events(payload)[0]
    transcript = _representation(february, "official_transcript_en")["locator"]
    assert isinstance(transcript, dict)
    transcript["locator_url"] = "https://example.com/transcript.pdf"
    with pytest.raises(ValueError, match="official domain"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    video = _representation(_events(payload)[0], "official_page_video_locator")["locator"]
    assert isinstance(video, dict)
    video["locator_url"] = "https://vimeo.com/live/EIVyXeKjIF8?feature=share"
    with pytest.raises(ValueError, match="YouTube"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    _representation(_events(payload)[0], "official_transcript_en")["status_evidence_url"] = (
        "https://www.bankofengland.co.uk./monetary-policy-report/2025/february-2025"
    )
    with pytest.raises(ValueError, match="trailing-dot"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    _scope(payload)["denominator_source_url"] = (
        "https://www.bankofengland.co.uk/monetary-policy-report/2025/\nfebruary"
    )
    with pytest.raises(ValueError, match="whitespace or control"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_completeness_basis_is_bound_to_representation_role(tmp_path):
    payload = _payload()
    specs = _scope(payload)["representation_specs"]
    assert isinstance(specs, list)
    transcript = next(
        item for item in specs if item["representation_key"] == "official_transcript_en"
    )
    transcript["completeness_basis"] = "exact_caption_track_url"
    with pytest.raises(ValueError, match="completeness_basis conflicts"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_catalogue_binding_source_semantics_and_clocks_are_fail_closed(tmp_path):
    payload = _payload()
    payload["catalogue_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="current communication source catalogue"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    specs = _scope(payload)["representation_specs"]
    assert isinstance(specs, list)
    specs[0]["source_id"] = "fed_fomc_press_conferences_en"
    with pytest.raises(ValueError, match="does not belong"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _payload()
    _events(payload)[0]["metadata_known_at"] = "2026-09-09T11:21:48Z"
    with pytest.raises(ValueError, match="checked_at follows"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_checked_loader_rejects_a_structurally_valid_semantic_change(tmp_path):
    payload = _payload()
    _events(payload)[0]["title"] = "Changed but structurally valid title"
    path = _write(tmp_path, payload)
    changed = load_institution_year_manifest(path)
    assert institution_year_manifest_sha256(changed) != BOE_2025_MANIFEST_SHA256
    with pytest.raises(ValueError, match="pinned semantic SHA-256"):
        load_checked_boe_2025_manifest(path)


def test_checked_manifest_exact_event_page_transcript_and_video_id_mappings():
    manifest = load_institution_year_manifest(CHECKED_MANIFEST)
    expected = {
        "boe_mpr_2025_02_06": (
            "2025-02-06",
            "february",
            "EIVyXeKjIF8",
            "https://youtube.com/live/EIVyXeKjIF8?feature=share",
        ),
        "boe_mpr_2025_05_08": ("2025-05-08", "may", "DAEab7yDUmE", None),
        "boe_mpr_2025_08_07": (
            "2025-08-07",
            "august",
            "v7obVnDNs50",
            "https://youtube.com/live/v7obVnDNs50?feature=share",
        ),
        "boe_mpr_2025_11_06": (
            "2025-11-06",
            "november",
            "9p6HuF1biEA",
            "https://youtube.com/live/9p6HuF1biEA?feature=share",
        ),
    }
    for event in manifest.events:
        event_date, month, video_id, video_url = expected[event.event_key]
        assert event.event_date.isoformat() == event_date
        observations = {item.representation_key: item for item in event.representations}
        transcript = observations["official_transcript_en"]
        video = observations["official_page_video_locator"]
        caption = observations["exact_caption_track_en"]
        official_page = f"https://www.bankofengland.co.uk/monetary-policy-report/2025/{month}-2025"
        assert transcript.status_evidence_url == video.status_evidence_url == official_page
        assert caption.status_evidence_url == official_page
        assert transcript.locator is not None
        assert transcript.locator.locator_url == (
            "https://www.bankofengland.co.uk/-/media/boe/files/monetary-policy-report/"
            f"2025/{month}/mpr-press-conference-transcript-{month}-2025.pdf"
        )
        assert video.locator is not None
        assert video.locator.platform_media_id == video_id
        assert video.locator.locator_url == video_url
        assert caption.availability_status == "not_verified"
        assert caption.locator is None
