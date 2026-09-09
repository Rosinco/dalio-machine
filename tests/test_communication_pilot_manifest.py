"""Strict pre-ingest contract for the metadata-only Fed/ECB pilot."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType

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
from dalio.communications.pilot_manifest import (
    PILOT_MANIFEST_SCHEMA_VERSION,
    PILOT_METHODOLOGY_VERSION,
    load_pilot_manifest,
    pilot_manifest_sha256,
)

ROOT = Path(__file__).resolve().parents[1]
CHECKED_MANIFEST = ROOT / "data" / "reference" / "communication_pilot_events.json"

FED_EVENT_KEYS = {
    "fomc_2025_01_29",
    "fomc_2025_03_19",
    "fomc_2025_05_07",
    "fomc_2025_06_18",
    "fomc_2025_07_30",
    "fomc_2025_09_17",
    "fomc_2025_10_29",
    "fomc_2025_12_10",
}
ECB_EVENT_KEYS = {
    "ecb_2025_01_30",
    "ecb_2025_03_06",
    "ecb_2025_04_17",
    "ecb_2025_06_05",
    "ecb_2025_07_24",
    "ecb_2025_09_11",
    "ecb_2025_10_30",
    "ecb_2025_12_18",
}


def _payload() -> dict[str, object]:
    return json.loads(CHECKED_MANIFEST.read_text(encoding="utf-8"))


def _write(tmp_path: Path, payload: dict[str, object]) -> Path:
    path = tmp_path / "pilot.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _organizations(payload: dict[str, object]) -> list[dict[str, object]]:
    scope = payload["scope"]
    assert isinstance(scope, dict)
    organizations = scope["organizations"]
    assert isinstance(organizations, list)
    return organizations


def _events(payload: dict[str, object]) -> list[dict[str, object]]:
    events = payload["events"]
    assert isinstance(events, list)
    return events


def _candidate(event: dict[str, object]) -> dict[str, object]:
    representation = event["representation"]
    assert isinstance(representation, dict)
    candidate = representation["candidate"]
    assert isinstance(candidate, dict)
    return candidate


def _event(payload: dict[str, object], organization_id: str) -> dict[str, object]:
    return next(item for item in _events(payload) if item["organization_id"] == organization_id)


def test_checked_manifest_has_exact_closed_denominators_and_only_metadata():
    before = CHECKED_MANIFEST.read_bytes()
    manifest = load_pilot_manifest(CHECKED_MANIFEST)

    assert manifest.schema_version == PILOT_MANIFEST_SCHEMA_VERSION == 1
    assert manifest.methodology_version == PILOT_METHODOLOGY_VERSION
    assert manifest.catalogue_sha256 == (
        "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    )
    assert manifest.catalogue_sha256 != COMMUNICATION_CATALOGUE_SHA256
    assert manifest.scope.start_date.isoformat() == "2025-01-01"
    assert manifest.scope.end_date.isoformat() == "2025-12-31"
    assert len(manifest.events) == 16
    assert CHECKED_MANIFEST.read_bytes() == before

    denominators = {
        denominator.organization_id: denominator for denominator in manifest.scope.organizations
    }
    assert set(denominators["federal_reserve"].event_keys) == FED_EVENT_KEYS
    assert set(denominators["ecb"].event_keys) == ECB_EVENT_KEYS
    assert all(
        denominator.rights_review.status == "pending" for denominator in denominators.values()
    )
    assert all(denominator.rights_review.questions for denominator in denominators.values())

    fed = [event for event in manifest.events if event.organization_id == "federal_reserve"]
    ecb = [event for event in manifest.events if event.organization_id == "ecb"]
    assert {event.event_key for event in fed} == FED_EVENT_KEYS
    assert {event.event_key for event in ecb} == ECB_EVENT_KEYS
    assert all(event.representation.section_coverage == ("full_transcript",) for event in fed)
    assert all(
        set(event.representation.section_coverage) == {"prepared_remarks", "q_and_a"}
        for event in ecb
    )
    for event in manifest.events:
        representation = event.representation
        candidate = representation.candidate
        assert representation.availability_status == "available"
        assert candidate.published_at is None
        assert (
            candidate.available_at
            == candidate.retrieved_at
            == candidate.metadata_known_at
            == representation.checked_at
        )
        assert not hasattr(candidate, "content_sha256")
        assert not hasattr(candidate, "rights_status")


def test_checked_manifest_semantic_hash_is_pinned():
    manifest = load_pilot_manifest(CHECKED_MANIFEST)

    assert pilot_manifest_sha256(manifest) == (
        "91848b6f144e4f2c46e821c3e24cbcf67be7d101e5781f5819dd6bcd10c87670"
    )


def test_semantic_hash_ignores_set_like_and_record_order_but_detects_changes(tmp_path):
    original = load_pilot_manifest(CHECKED_MANIFEST)
    reordered = _payload()
    organizations = _organizations(reordered)
    organizations.reverse()
    for denominator in organizations:
        event_keys = denominator["event_keys"]
        rights_review = denominator["rights_review"]
        assert isinstance(event_keys, list) and isinstance(rights_review, dict)
        event_keys.reverse()
        questions = rights_review["questions"]
        assert isinstance(questions, list)
        questions.reverse()
    _events(reordered).reverse()
    reordered_manifest = load_pilot_manifest(_write(tmp_path, reordered))
    assert pilot_manifest_sha256(reordered_manifest) == pilot_manifest_sha256(original)

    changed = _payload()
    _events(changed)[0]["title"] = "Semantically changed title"
    changed_manifest = load_pilot_manifest(_write(tmp_path, changed))
    assert pilot_manifest_sha256(changed_manifest) != pilot_manifest_sha256(original)


def test_loader_rejects_duplicate_json_fields(tmp_path):
    raw = CHECKED_MANIFEST.read_text(encoding="utf-8").replace(
        '"schema_version": 1,',
        '"schema_version": 1,\n  "schema_version": 1,',
        1,
    )
    path = tmp_path / "duplicate.json"
    path.write_text(raw, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate JSON field: schema_version"):
        load_pilot_manifest(path)


@pytest.mark.parametrize("forbidden_field", ["content_sha256", "content", "rights_status"])
def test_candidate_rejects_content_and_rights_decision_fields(tmp_path, forbidden_field):
    payload = _payload()
    _candidate(_events(payload)[0])[forbidden_field] = "not-allowed"

    with pytest.raises(ValueError, match="unknown fields"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_pending_rights_record_rejects_a_decision_or_reviewer(tmp_path):
    for forbidden_field in ("decision", "reviewed_by"):
        payload = _payload()
        review = _organizations(payload)[0]["rights_review"]
        assert isinstance(review, dict)
        review[forbidden_field] = "human:analyst"
        with pytest.raises(ValueError, match="unknown fields"):
            load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    review = _organizations(payload)[0]["rights_review"]
    assert isinstance(review, dict)
    review["status"] = "cleared"
    with pytest.raises(ValueError, match="must remain pending"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_loader_binds_known_source_catalogue_snapshot_and_selected_representation(tmp_path):
    payload = _payload()
    payload["catalogue_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="known communication catalogue snapshot"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    spec = _organizations(payload)[0]["representation_spec"]
    assert isinstance(spec, dict)
    spec["source_id"] = "fed_fomc_press_conference_subtitles_en"
    with pytest.raises(ValueError, match="selected federal_reserve pilot representation"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    _candidate(_events(payload)[0])["material_type"] = "press_conference_video"
    with pytest.raises(ValueError, match="tracked representation specification"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    spec = _organizations(payload)[1]["representation_spec"]
    assert isinstance(spec, dict)
    spec["section_coverage"] = ["q_and_a"]
    with pytest.raises(ValueError, match="selected ecb pilot representation"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    spec = _organizations(payload)[1]["representation_spec"]
    assert isinstance(spec, dict)
    spec["section_coverage"] = ["q_and_a", "prepared_remarks"]
    with pytest.raises(ValueError, match="selected ecb pilot representation"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_pilot_loader_uses_source_semantics_from_the_bound_snapshot(tmp_path, monkeypatch):
    frozen_hash = "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    frozen = resolve_communication_catalogue_snapshot(frozen_hash)
    current_fed = next(
        source for source in frozen.sources if source.source_id == "fed_fomc_press_conferences_en"
    )
    historic_publisher = "Historical Board of Governors"
    historic_fed = replace(current_fed, publisher=historic_publisher)
    historic_sources = tuple(
        historic_fed if source.source_id == current_fed.source_id else source
        for source in frozen.sources
    )
    historic_snapshot = communication_catalogue_snapshot(
        historic_sources,
        schema_version=frozen.schema_version,
        evaluated_at=frozen.evaluated_at,
    )
    historic_hash = historic_snapshot.catalogue_sha256
    monkeypatch.setattr(
        catalogue_module,
        "COMMUNICATION_CATALOGUE_SNAPSHOTS",
        MappingProxyType({**COMMUNICATION_CATALOGUE_SNAPSHOTS, historic_hash: historic_snapshot}),
    )

    payload = _payload()
    payload["catalogue_sha256"] = historic_hash
    for event in _events(payload):
        if event["organization_id"] == "federal_reserve":
            _candidate(event)["publisher"] = historic_publisher
    assert load_pilot_manifest(_write(tmp_path, payload)).catalogue_sha256 == historic_hash

    payload["catalogue_sha256"] = frozen_hash
    with pytest.raises(ValueError, match="bound catalogue snapshot"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_pilot_manifest_cannot_predate_its_bound_catalogue_snapshot(tmp_path, monkeypatch):
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
        load_pilot_manifest(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host_organization", "Federal Reserve vendor", "host and publisher"),
        ("publisher", "Unknown publisher", "host and publisher"),
        ("origin_type", "official_hosted_vendor", "official_published_transcript"),
        ("provenance_tier", "official_hosted_third_party", "official_published_transcript"),
        ("transcriber", "Vendor", "transcriber null"),
        ("transcriber_attribution", "publisher", "not_disclosed"),
    ],
)
def test_candidate_provenance_is_narrow_and_catalogue_bound(tmp_path, field, value, message):
    payload = _payload()
    _candidate(_events(payload)[0])[field] = value

    with pytest.raises(ValueError, match=message):
        load_pilot_manifest(_write(tmp_path, payload))


def test_denominators_cannot_be_redefined_by_the_manifest(tmp_path):
    payload = _payload()
    event_keys = _organizations(payload)[0]["event_keys"]
    assert isinstance(event_keys, list)
    event_keys.pop()
    with pytest.raises(ValueError, match="fixed pilot denominator"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    _events(payload).pop()
    with pytest.raises(ValueError, match="event denominator mismatch"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    _events(payload).append(dict(_events(payload)[0]))
    with pytest.raises(ValueError, match="duplicate organization/event identity"):
        load_pilot_manifest(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("start_date", "end_date"),
    [("2024-01-01", "2025-12-31"), ("2025-01-01", "2026-01-01")],
)
def test_scope_is_the_exact_closed_2025_window(tmp_path, start_date, end_date):
    payload = _payload()
    scope = payload["scope"]
    assert isinstance(scope, dict)
    scope["start_date"] = start_date
    scope["end_date"] = end_date

    with pytest.raises(ValueError, match="exactly the closed 2025 calendar year"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_each_event_has_exactly_one_available_selected_representation(tmp_path):
    payload = _payload()
    first = _events(payload)[0]
    first["representation"] = [first["representation"], first["representation"]]
    with pytest.raises(ValueError, match="representation must be an object"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    representation = _event(payload, "ecb")["representation"]
    assert isinstance(representation, dict)
    representation["availability_status"] = "missing"
    with pytest.raises(ValueError, match="must be available"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    representation = _event(payload, "ecb")["representation"]
    assert isinstance(representation, dict)
    representation["section_coverage"] = ["q_and_a"]
    with pytest.raises(ValueError, match="section_coverage conflicts"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    representation = _event(payload, "ecb")["representation"]
    assert isinstance(representation, dict)
    representation["section_coverage"] = ["q_and_a", "prepared_remarks"]
    with pytest.raises(ValueError, match="section_coverage conflicts"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_unknown_publication_time_uses_one_conservative_observation_clock(tmp_path):
    payload = _payload()
    _candidate(_events(payload)[0])["available_at"] = "2026-09-09T08:00:00Z"

    with pytest.raises(ValueError, match="observation time for all public clocks"):
        load_pilot_manifest(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("retrieved_at", "2026-09-09T08:00:00Z", "cannot precede available_at"),
        ("metadata_known_at", "2026-09-09T08:07:00Z", "cannot follow representation"),
        ("published_at", "2026-09-09T08:07:00Z", "cannot precede published_at"),
        ("available_at", "2025-01-01T00:00:00", "include a timezone"),
    ],
)
def test_candidate_clock_chain_fails_closed(tmp_path, field, value, message):
    payload = _payload()
    _candidate(_events(payload)[0])[field] = value

    with pytest.raises(ValueError, match=message):
        load_pilot_manifest(_write(tmp_path, payload))


def test_exact_dates_reject_datetimes(tmp_path):
    payload = _payload()
    _events(payload)[0]["event_date"] = "2025-01-29T00:00:00Z"

    with pytest.raises(ValueError, match="ISO YYYY-MM-DD date"):
        load_pilot_manifest(_write(tmp_path, payload))


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("http://www.federalreserve.gov/file.pdf", "HTTPS"),
        ("https://federalreserve.gov.evil.example/file.pdf", "official-domain"),
        ("https://user@www.federalreserve.gov/file.pdf", "credentials"),
        ("https://www.federalreserve.gov:443/file.pdf", "explicit port"),
        ("https://www.federalreserve.gov:8443/file.pdf", "explicit port"),
        ("https://127.0.0.1/file.pdf", "IP address"),
        ("https://localhost/file.pdf", "local hostname"),
        ("https://www.federalreserve.gov./file.pdf", "trailing-dot"),
        ("https://www.federalreserve.gov\\@evil.example/file.pdf", "backslash"),
        ("https://www.federalreserve.gov/file\n.pdf", "control"),
    ],
)
def test_urls_reject_ambiguous_or_nonofficial_authorities(tmp_path, url, message):
    payload = _payload()
    _candidate(_events(payload)[0])["artifact_url"] = url

    with pytest.raises(ValueError, match=message):
        load_pilot_manifest(_write(tmp_path, payload))


def test_fed_event_landing_rejects_same_domain_substitution(tmp_path):
    payload = _payload()
    fed_event = _event(payload, "federal_reserve")
    fed_representation = fed_event["representation"]
    assert isinstance(fed_representation, dict)
    substituted_landing = "https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250319.htm"
    fed_representation["status_evidence_url"] = substituted_landing
    _candidate(fed_event)["landing_url"] = substituted_landing
    with pytest.raises(ValueError, match="exact selected locator pair"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_fed_transcript_rejects_same_domain_fabricated_pdf(tmp_path):
    payload = _payload()
    fed_event = _event(payload, "federal_reserve")
    _candidate(fed_event)["artifact_url"] = (
        "https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250129-copy.pdf"
    )
    with pytest.raises(ValueError, match="exact selected locator pair"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_ecb_page_rejects_same_domain_fabricated_hash_slug(tmp_path):
    payload = _payload()
    ecb_event = _event(payload, "ecb")
    ecb_representation = ecb_event["representation"]
    assert isinstance(ecb_representation, dict)
    fabricated_ecb_page = (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/"
        "2025/html/ecb.is250130~0000000000.en.html"
    )
    ecb_representation["status_evidence_url"] = fabricated_ecb_page
    ecb_candidate = _candidate(ecb_event)
    ecb_candidate["landing_url"] = fabricated_ecb_page
    ecb_candidate["artifact_url"] = fabricated_ecb_page
    with pytest.raises(ValueError, match="exact selected locator pair"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_actor_and_pending_review_questions_are_strict(tmp_path):
    payload = _payload()
    payload["created_by"] = "anonymous"
    with pytest.raises(ValueError, match="namespaced"):
        load_pilot_manifest(_write(tmp_path, payload))

    payload = _payload()
    review = _organizations(payload)[0]["rights_review"]
    assert isinstance(review, dict)
    review["questions"] = []
    with pytest.raises(ValueError, match="non-empty array"):
        load_pilot_manifest(_write(tmp_path, payload))


def test_semantic_hash_rejects_the_wrong_object_type():
    with pytest.raises(TypeError, match="CommunicationPilotManifest"):
        pilot_manifest_sha256({})  # type: ignore[arg-type]
