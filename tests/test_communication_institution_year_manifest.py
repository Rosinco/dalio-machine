"""Closed one-institution/year communication metadata manifests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from urllib.parse import urlsplit

import pytest

import dalio.communications.catalogue as catalogue_module
from dalio.communications.catalogue import (
    CATALOGUE_EVALUATED_AT,
    CATALOGUE_SCHEMA_VERSION,
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_CATALOGUE_SNAPSHOTS,
    COMMUNICATION_SOURCES,
    communication_catalogue_snapshot,
    resolve_communication_catalogue_snapshot,
)
from dalio.communications.institution_year_manifest import (
    BOE_2025_MANIFEST_SHA256,
    INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION,
    INSTITUTION_YEAR_METHODOLOGY_VERSION,
    RIKSBANK_2025_MANIFEST_ID,
    RIKSBANK_2025_MANIFEST_SHA256,
    institution_year_manifest_sha256,
    load_checked_boe_2025_manifest,
    load_checked_riksbank_2025_manifest,
    load_institution_year_manifest,
)
from dalio.communications.metadata_inventory import build_representation_inventory

ROOT = Path(__file__).resolve().parents[1]
CHECKED_MANIFEST = ROOT / "data" / "reference" / "communication_boe_2025_events.json"
RIKSBANK_CHECKED_MANIFEST = ROOT / "data" / "reference" / "communication_riksbank_2025_events.json"
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
RIKSBANK_EVENT_KEYS = {
    "riksbank_mpr_2025_01_29",
    "riksbank_mpr_2025_03_20",
    "riksbank_mpr_2025_05_08",
    "riksbank_mpr_2025_06_18",
    "riksbank_mpr_2025_08_20",
    "riksbank_mpr_2025_09_23",
    "riksbank_mpr_2025_11_05",
    "riksbank_mpr_2025_12_18",
}
RIKSBANK_REPRESENTATION_KEYS = {
    "official_transcript_sv",
    "official_replay_page_sv",
    "official_slides_sv",
    "exact_caption_track_sv",
}


def _payload() -> dict[str, object]:
    return json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))


def _riksbank_payload() -> dict[str, object]:
    checked_at = "2026-09-09T13:30:00Z"
    event_url = (
        "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/2025/"
        "presstraff-om-det-penningpolitiska-beslutet-i-januari-2025/"
    )
    decision_url = (
        "https://www.riksbank.se/sv/penningpolitik/penningpolitisk-rapport/2025/"
        "penningpolitiskt-beslut-januari-2025/"
    )
    main_source = "riksbank_monetary_policy_press_conferences_sv"
    subtitle_source = "riksbank_monetary_policy_press_conference_subtitles_sv"
    return {
        "schema_version": INSTITUTION_YEAR_MANIFEST_SCHEMA_VERSION,
        "methodology_version": INSTITUTION_YEAR_METHODOLOGY_VERSION,
        "catalogue_sha256": COMMUNICATION_CATALOGUE_SHA256,
        "manifest_id": "riksbank_2025_monetary_policy_press_conferences",
        "created_by": "agent:test",
        "created_at": checked_at,
        "as_known_at": checked_at,
        "content_capture_authorized": False,
        "scope": {
            "organization_id": "sveriges_riksbank",
            "year": 2025,
            "start_date": "2025-01-01",
            "end_date": "2025-12-31",
            "event_type": "monetary_policy_press_conference",
            "denominator_source_url": (
                "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/"
                "?category=28&year=2025&page=1"
            ),
            "inclusion_rule": "Test fixture: one official 2025 monetary-policy replay page.",
            "exclusions": ["All events outside this one-event schema fixture."],
            "checked_by": "agent:test",
            "checked_at": checked_at,
            "event_keys": ["riksbank_mpr_2025_01_29"],
            "representation_specs": [
                {
                    "representation_key": "official_transcript_sv",
                    "source_id": main_source,
                    "artifact_role": "full_transcript",
                    "material_type": "press_conference_transcript",
                    "section_coverage": ["full_transcript"],
                    "completeness_basis": "exact_direct_artifact_url",
                    "rights_status": "rights_review_required",
                    "acquisition_status": "manual_review_required",
                    "automated_collection_allowed": False,
                },
                {
                    "representation_key": "official_replay_page_sv",
                    "source_id": main_source,
                    "artifact_role": "webcast_video",
                    "material_type": "press_conference_video",
                    "section_coverage": ["webcast_video"],
                    "completeness_basis": "official_page_media_locator",
                    "rights_status": "rights_review_required",
                    "acquisition_status": "manual_review_required",
                    "automated_collection_allowed": False,
                },
                {
                    "representation_key": "official_slides_sv",
                    "source_id": main_source,
                    "artifact_role": "presentation_slides",
                    "material_type": "press_conference_slides",
                    "section_coverage": ["presentation_slides"],
                    "completeness_basis": "exact_direct_artifact_url",
                    "rights_status": "rights_review_required",
                    "acquisition_status": "manual_review_required",
                    "automated_collection_allowed": False,
                },
                {
                    "representation_key": "exact_caption_track_sv",
                    "source_id": subtitle_source,
                    "artifact_role": "subtitles",
                    "material_type": "subtitles",
                    "section_coverage": ["subtitles"],
                    "completeness_basis": "exact_caption_track_url",
                    "rights_status": "rights_review_required",
                    "acquisition_status": "manual_review_required",
                    "automated_collection_allowed": False,
                },
            ],
        },
        "events": [
            {
                "event_key": "riksbank_mpr_2025_01_29",
                "organization_id": "sveriges_riksbank",
                "event_type": "monetary_policy_press_conference",
                "title": "Pressträff om det penningpolitiska beslutet i januari 2025",
                "event_date": "2025-01-29",
                "metadata_known_at": checked_at,
                "representations": [
                    {
                        "representation_key": "official_transcript_sv",
                        "availability_status": "not_verified",
                        "checked_at": checked_at,
                        "status_evidence_url": event_url,
                        "status_note": "No first-party transcript locator is verified.",
                        "locator": None,
                    },
                    {
                        "representation_key": "official_replay_page_sv",
                        "availability_status": "official_replay_page_link",
                        "checked_at": checked_at,
                        "status_evidence_url": event_url,
                        "status_note": "The official page presents the event replay.",
                        "locator": {
                            "locator_key": "riksbank_mpr_2025_01_29_replay_page_sv",
                            "source_id": main_source,
                            "locator_kind": "official_replay_page",
                            "locator_url": event_url,
                            "platform_media_id": None,
                            "mime_type": "text/html",
                            "language": "sv",
                            "translation_status": "original",
                            "host_organization": "Sveriges Riksbank",
                            "publisher": "Sveriges Riksbank",
                            "transcriber": None,
                            "transcriber_attribution": "not_applicable",
                            "origin_type": "official_replay_page",
                            "provenance_tier": "official_archive_mixed",
                            "published_at": None,
                            "observed_at": checked_at,
                        },
                    },
                    {
                        "representation_key": "official_slides_sv",
                        "availability_status": "direct_artifact_link",
                        "checked_at": checked_at,
                        "status_evidence_url": decision_url,
                        "status_note": "The official decision page links the Swedish slide PDF.",
                        "locator": {
                            "locator_key": "riksbank_mpr_2025_01_29_slides_sv",
                            "source_id": main_source,
                            "locator_kind": "official_direct_artifact",
                            "locator_url": (
                                "https://www.riksbank.se/globalassets/media/rapporter/ppr/"
                                "bilder-fran-presstraffen/2025/250129/"
                                "bilder-fran-presstraffen-den-29-januari-2025.pdf"
                            ),
                            "platform_media_id": None,
                            "mime_type": "application/pdf",
                            "language": "sv",
                            "translation_status": "original",
                            "host_organization": "Sveriges Riksbank",
                            "publisher": "Sveriges Riksbank",
                            "transcriber": None,
                            "transcriber_attribution": "not_applicable",
                            "origin_type": "official_published_slides",
                            "provenance_tier": "official_authored_text",
                            "published_at": None,
                            "observed_at": checked_at,
                        },
                    },
                    {
                        "representation_key": "exact_caption_track_sv",
                        "availability_status": "not_verified",
                        "checked_at": checked_at,
                        "status_evidence_url": event_url,
                        "status_note": "No exact Swedish caption track is verified.",
                        "locator": None,
                    },
                ],
            }
        ],
    }


def _checked_riksbank_payload() -> dict[str, object]:
    return json.loads(RIKSBANK_CHECKED_MANIFEST.read_text(encoding="utf-8"))


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
    assert manifest.catalogue_sha256 == (
        "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    )
    assert manifest.catalogue_sha256 != COMMUNICATION_CATALOGUE_SHA256
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


def test_checked_riksbank_manifest_is_closed_metadata_only_and_hash_pinned():
    before = RIKSBANK_CHECKED_MANIFEST.read_bytes()
    boe_before = CHECKED_MANIFEST.read_bytes()
    manifest = load_checked_riksbank_2025_manifest(RIKSBANK_CHECKED_MANIFEST)

    assert manifest.manifest_id == RIKSBANK_2025_MANIFEST_ID
    assert manifest.scope.organization_id == "sveriges_riksbank"
    assert manifest.scope.year == 2025
    assert manifest.scope.start_date.isoformat() == "2025-01-01"
    assert manifest.scope.end_date.isoformat() == "2025-12-31"
    assert manifest.scope.denominator_source_url == (
        "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/"
        "?category=28&year=2025&page=1"
    )
    assert (
        "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/"
        "?category=28&year=2025&page=2"
    ) in manifest.scope.inclusion_rule
    assert "include the eight entries titled as monetary-policy-decision press conferences" in (
        manifest.scope.inclusion_rule
    )
    assert set(manifest.scope.exclusions) == {
        "the Payments Report press conference dated 10 March 2025",
        "the Financial Stability Report press conferences dated 28 May and 13 November 2025",
        "speeches and seminars",
        (
            "reports, updates, minutes, votes and releases as representations distinct from "
            "the selected press-conference replay pages and slides"
        ),
    }
    assert manifest.created_at.isoformat() == "2026-09-09T13:37:45+00:00"
    assert manifest.as_known_at == manifest.created_at
    assert manifest.scope.checked_at == manifest.created_at
    assert manifest.content_capture_authorized is False
    assert manifest.catalogue_sha256 == (
        "ad499a7129238d70be8c7243c62252b3a1f11628a69d2df4ecf14c41d33ef1b0"
    )
    assert set(manifest.scope.event_keys) == RIKSBANK_EVENT_KEYS
    assert {event.event_key for event in manifest.events} == RIKSBANK_EVENT_KEYS
    assert len(manifest.events) == 8
    assert all(
        {item.representation_key for item in event.representations} == RIKSBANK_REPRESENTATION_KEYS
        for event in manifest.events
    )
    assert institution_year_manifest_sha256(manifest) == RIKSBANK_2025_MANIFEST_SHA256
    assert RIKSBANK_2025_MANIFEST_SHA256 == (
        "34f610043e933f74ca71ae56a9d29129fc92e2dd1ea83cc932293581ac4a2468"
    )
    assert RIKSBANK_CHECKED_MANIFEST.read_bytes() == before

    boe = load_checked_boe_2025_manifest(CHECKED_MANIFEST)
    assert institution_year_manifest_sha256(boe) == BOE_2025_MANIFEST_SHA256
    assert BOE_2025_MANIFEST_SHA256 == (
        "4bba6c8415de46718a5ae6906d0d09e9041f7be1903dbeef7be4f40223717eb2"
    )
    assert CHECKED_MANIFEST.read_bytes() == boe_before


def test_checked_riksbank_exact_event_replay_and_slide_mappings():
    manifest = load_checked_riksbank_2025_manifest(RIKSBANK_CHECKED_MANIFEST)
    expected = {
        "riksbank_mpr_2025_01_29": (
            "2025-01-29",
            "januari-2025",
            "januari",
            "250129",
            "29-januari",
        ),
        "riksbank_mpr_2025_03_20": (
            "2025-03-20",
            "mars-2025",
            "mars",
            "250320",
            "20-mars",
        ),
        "riksbank_mpr_2025_05_08": ("2025-05-08", "maj", "maj", "250508", "8-maj"),
        "riksbank_mpr_2025_06_18": (
            "2025-06-18",
            "juni-2025",
            "juni",
            "250618",
            "18-juni",
        ),
        "riksbank_mpr_2025_08_20": (
            "2025-08-20",
            "augusti-2025",
            "augusti",
            "250820",
            "20-augusti",
        ),
        "riksbank_mpr_2025_09_23": (
            "2025-09-23",
            "september-2025",
            "september",
            "250923",
            "23-september",
        ),
        "riksbank_mpr_2025_11_05": (
            "2025-11-05",
            "november-2025",
            "november",
            "251105",
            "5-november",
        ),
        "riksbank_mpr_2025_12_18": (
            "2025-12-18",
            "december-2025",
            "december",
            "251218",
            "18-december",
        ),
    }

    for event in manifest.events:
        event_date, replay_slug, decision_month, asset_date, slide_date = expected[event.event_key]
        replay_url = (
            "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/2025/"
            f"presstraff-om-det-penningpolitiska-beslutet-i-{replay_slug}/"
        )
        slide_evidence_url = (
            "https://www.riksbank.se/sv/penningpolitik/penningpolitisk-rapport/2025/"
            f"penningpolitiskt-beslut-{decision_month}-2025/"
        )
        slide_url = (
            "https://www.riksbank.se/globalassets/media/rapporter/ppr/"
            f"bilder-fran-presstraffen/2025/{asset_date}/"
            f"bilder-fran-presstraffen-den-{slide_date}-2025.pdf"
        )
        observations = {item.representation_key: item for item in event.representations}
        transcript = observations["official_transcript_sv"]
        replay = observations["official_replay_page_sv"]
        slides = observations["official_slides_sv"]
        captions = observations["exact_caption_track_sv"]

        assert event.event_date.isoformat() == event_date
        assert transcript.availability_status == "not_verified"
        assert transcript.status_evidence_url == replay_url
        assert transcript.locator is None
        assert replay.availability_status == "official_replay_page_link"
        assert replay.status_evidence_url == replay_url
        assert replay.locator is not None
        assert replay.locator.locator_kind == "official_replay_page"
        assert replay.locator.locator_url == replay_url
        assert replay.locator.platform_media_id is None
        assert replay.locator.mime_type == "text/html"
        assert replay.locator.transcriber is None
        assert replay.locator.transcriber_attribution == "not_applicable"
        assert replay.locator.origin_type == "official_replay_page"
        assert replay.locator.provenance_tier == "official_archive_mixed"
        assert slides.availability_status == "direct_artifact_link"
        assert slides.status_evidence_url == slide_evidence_url
        assert "decision page links the exact Swedish" in slides.status_note
        assert slides.locator is not None
        assert slides.locator.locator_kind == "official_direct_artifact"
        assert slides.locator.locator_url == slide_url
        assert slides.locator.platform_media_id is None
        assert slides.locator.mime_type == "application/pdf"
        assert slides.locator.language == "sv"
        assert slides.locator.translation_status == "original"
        assert slides.locator.transcriber is None
        assert slides.locator.transcriber_attribution == "not_applicable"
        assert slides.locator.origin_type == "official_published_slides"
        assert slides.locator.provenance_tier == "official_authored_text"
        assert captions.availability_status == "not_verified"
        assert captions.status_evidence_url == replay_url
        assert captions.locator is None


def test_checked_riksbank_manifest_has_only_first_party_urls_and_no_content_fields():
    payload = _checked_riksbank_payload()
    serialized = json.dumps(payload, ensure_ascii=False).lower()
    assert "qcnl.tv" not in serialized
    assert "youtube.com" not in serialized
    assert "youtu.be" not in serialized
    assert payload["content_capture_authorized"] is False

    forbidden_fields = {
        "content",
        "content_sha256",
        "captured_at",
        "capture_status",
        "source_bytes",
        "transcript_text",
        "caption_text",
    }
    observed_keys: set[str] = set()
    observed_urls: list[str] = []

    def visit(value: object) -> None:
        if isinstance(value, dict):
            observed_keys.update(value)
            for key, item in value.items():
                if key.endswith("_url") and isinstance(item, str):
                    observed_urls.append(item)
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    assert observed_keys.isdisjoint(forbidden_fields)
    assert observed_urls
    assert {urlsplit(url).hostname for url in observed_urls} == {"www.riksbank.se"}


@pytest.mark.parametrize(
    ("representation_key", "replacement_url"),
    [
        (
            "official_replay_page_sv",
            "https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/2025/"
            "presstraff-om-det-penningpolitiska-beslutet-i-mars-2025/",
        ),
        (
            "official_slides_sv",
            "https://www.riksbank.se/globalassets/media/rapporter/ppr/"
            "bilder-fran-presstraffen/2025/250320/"
            "bilder-fran-presstraffen-den-20-mars-2025.pdf",
        ),
    ],
)
def test_checked_riksbank_loader_rejects_same_domain_locator_tampering(
    tmp_path, representation_key, replacement_url
):
    payload = _checked_riksbank_payload()
    observation = _representation(_events(payload)[0], representation_key)
    locator = observation["locator"]
    assert isinstance(locator, dict)
    locator["locator_url"] = replacement_url
    path = _write(tmp_path, payload)

    assert load_institution_year_manifest(path).manifest_id == RIKSBANK_2025_MANIFEST_ID
    with pytest.raises(ValueError, match="pinned semantic SHA-256"):
        load_checked_riksbank_2025_manifest(path)


def test_checked_riksbank_loader_rejects_wrong_cohort_and_added_content(tmp_path):
    with pytest.raises(ValueError, match="wrong manifest_id"):
        load_checked_riksbank_2025_manifest(CHECKED_MANIFEST)

    payload = _checked_riksbank_payload()
    _representation(_events(payload)[0], "official_transcript_sv")["content"] = "forbidden"
    with pytest.raises(ValueError, match="unknown fields"):
        load_checked_riksbank_2025_manifest(_write(tmp_path, payload))


def test_riksbank_roles_accept_first_party_replay_page_and_slide_pdf(tmp_path):
    manifest = load_institution_year_manifest(_write(tmp_path, _riksbank_payload()))

    assert manifest.scope.organization_id == "sveriges_riksbank"
    specs = {spec.representation_key: spec for spec in manifest.scope.representation_specs}
    assert specs["official_slides_sv"].artifact_role == "presentation_slides"
    assert specs["official_slides_sv"].material_type == "press_conference_slides"
    observations = {item.representation_key: item for item in manifest.events[0].representations}
    replay = observations["official_replay_page_sv"]
    assert replay.availability_status == "official_replay_page_link"
    assert replay.locator is not None
    assert replay.locator.locator_kind == "official_replay_page"
    assert replay.locator.platform_media_id is None
    assert replay.locator.locator_url is not None
    assert "riksbank.se" in replay.locator.locator_url
    slides = observations["official_slides_sv"]
    assert slides.locator is not None
    assert slides.locator.origin_type == "official_published_slides"
    assert observations["official_transcript_sv"].locator is None
    assert observations["exact_caption_track_sv"].locator is None


def test_riksbank_replay_locator_rejects_external_player_metadata(tmp_path):
    payload = _riksbank_payload()
    replay = _representation(_events(payload)[0], "official_replay_page_sv")
    locator = replay["locator"]
    assert isinstance(locator, dict)
    locator["locator_url"] = "https://qcnl.tv/e/vendor-player-id"
    with pytest.raises(ValueError, match="official domain"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _riksbank_payload()
    replay = _representation(_events(payload)[0], "official_replay_page_sv")
    locator = replay["locator"]
    assert isinstance(locator, dict)
    locator["platform_media_id"] = "vendor_player_id"
    with pytest.raises(ValueError, match="invalid official replay-page locator"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_riksbank_replay_status_and_role_are_bound(tmp_path):
    payload = _riksbank_payload()
    replay = _representation(_events(payload)[0], "official_replay_page_sv")
    replay["availability_status"] = "external_platform_link"
    with pytest.raises(ValueError, match="availability_status conflicts"):
        load_institution_year_manifest(_write(tmp_path, payload))

    payload = _riksbank_payload()
    slides = _representation(_events(payload)[0], "official_slides_sv")
    locator = slides["locator"]
    assert isinstance(locator, dict)
    locator["locator_kind"] = "official_replay_page"
    locator["mime_type"] = "text/html"
    with pytest.raises(ValueError, match="invalid official replay-page locator"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_riksbank_sources_are_unavailable_in_the_frozen_catalogue_snapshot(tmp_path):
    payload = _riksbank_payload()
    payload["catalogue_sha256"] = "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    with pytest.raises(ValueError, match="not in the communication source catalogue"):
        load_institution_year_manifest(_write(tmp_path, payload))


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
    with pytest.raises(ValueError, match="known communication catalogue snapshot"):
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


def test_loader_uses_only_sources_from_the_bound_catalogue_snapshot(tmp_path, monkeypatch):
    frozen_hash = "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    frozen = resolve_communication_catalogue_snapshot(frozen_hash)
    base = next(
        source
        for source in frozen.sources
        if source.source_id == "boe_monetary_policy_press_conferences_en"
    )
    added = replace(base, source_id="boe_future_press_conferences_en")
    expanded_sources = (*frozen.sources, added)
    expanded_snapshot = communication_catalogue_snapshot(
        expanded_sources,
        schema_version=frozen.schema_version,
        evaluated_at=frozen.evaluated_at,
    )
    expanded_hash = expanded_snapshot.catalogue_sha256
    monkeypatch.setattr(
        catalogue_module,
        "COMMUNICATION_CATALOGUE_SNAPSHOTS",
        MappingProxyType({**COMMUNICATION_CATALOGUE_SNAPSHOTS, expanded_hash: expanded_snapshot}),
    )

    payload = _payload()
    payload["catalogue_sha256"] = expanded_hash
    for spec in _scope(payload)["representation_specs"]:
        if spec["source_id"] == base.source_id:
            spec["source_id"] = added.source_id
    for event in _events(payload):
        for observation in _representations(event):
            locator = observation["locator"]
            if isinstance(locator, dict) and locator["source_id"] == base.source_id:
                locator["source_id"] = added.source_id

    assert (
        load_institution_year_manifest(_write(tmp_path, payload)).catalogue_sha256 == expanded_hash
    )
    payload["catalogue_sha256"] = frozen_hash
    with pytest.raises(ValueError, match="not in the communication source catalogue"):
        load_institution_year_manifest(_write(tmp_path, payload))


def test_institution_year_manifest_cannot_predate_its_catalogue_snapshot(tmp_path, monkeypatch):
    future_snapshot = communication_catalogue_snapshot(
        COMMUNICATION_SOURCES,
        schema_version=CATALOGUE_SCHEMA_VERSION,
        evaluated_at=CATALOGUE_EVALUATED_AT + timedelta(days=1),
    )
    monkeypatch.setattr(
        catalogue_module,
        "COMMUNICATION_CATALOGUE_SNAPSHOTS",
        MappingProxyType(
            {
                **COMMUNICATION_CATALOGUE_SNAPSHOTS,
                future_snapshot.catalogue_sha256: future_snapshot,
            }
        ),
    )
    payload = _payload()
    payload["catalogue_sha256"] = future_snapshot.catalogue_sha256

    with pytest.raises(ValueError, match="clocks precede.*catalogue evaluation time"):
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
