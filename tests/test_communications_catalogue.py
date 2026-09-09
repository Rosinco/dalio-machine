"""Rights-aware, metadata-only institutional communications source catalogue."""

from dataclasses import FrozenInstanceError, replace
from datetime import UTC, datetime
from urllib.parse import urlsplit

import pytest

from dalio.communications.catalogue import (
    ACQUISITION_STATUSES,
    CATALOGUE_SCHEMA_VERSION,
    COMMUNICATION_CATALOGUE_SHA256,
    COMMUNICATION_SOURCES,
    REQUIRED_COMMODITY_FAMILIES,
    RIGHTS_STATUSES,
    CommunicationSourceSpec,
    communication_catalogue_sha256,
    validate_communication_sources,
    validate_pilot_coverage,
)

EXPECTED_SOURCE_IDS = {
    # Central-bank communication history.
    "fed_fomc_press_conferences_en",
    "fed_fomc_press_conference_subtitles_en",
    "ecb_monetary_policy_press_conferences_en",
    "boe_monetary_policy_press_conferences_en",
    "boe_monetary_policy_press_conference_subtitles_en",
    "rba_speeches_en",
    # A geographically balanced bank-letter baseline.
    "jpmorgan_chase_annual_reports_en",
    "hsbc_group_reporting_en",
    "deutsche_bank_annual_reports_en",
    "mufg_annual_reports_en",
    "ubs_annual_reports_en",
    "citigroup_annual_reports_en",
    "bank_of_china_reports_en",
    # One or more first-party archives for every commodity family in scope.
    "shell_annual_reports_en",
    "bhp_financial_results_en",
    "barrick_annual_reports_en",
    "nutrien_financial_reporting_en",
    "bunge_annual_reports_en",
    "west_fraser_reports_en",
    "wilmar_annual_reports_en",
}


def _example_spec() -> CommunicationSourceSpec:
    return CommunicationSourceSpec(
        source_id="example_annual_reports_en",
        organization_id="example_company",
        organization_name="Example Company",
        organization_type="commodity_company",
        jurisdiction="US",
        language="en",
        landing_url="https://www.jpmorganchase.com/example-reports",
        official_domains=("jpmorganchase.com",),
        host_organization="Example Company",
        publisher="Example Company",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "ceo_letter"),
        commodity_families=("energy",),
        verified_archive_start_year=2000,
        coverage_note="Annual reports are verified from 2000.",
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note="Review the terms and each selected artifact before capture.",
        acquisition_status="manual_review_required",
        acquisition_note="No network collector is authorized by this metadata record.",
        automated_collection_allowed=False,
    )


def test_catalogue_contains_the_verified_balanced_source_set():
    assert {source.source_id for source in COMMUNICATION_SOURCES} == EXPECTED_SOURCE_IDS
    assert len(COMMUNICATION_SOURCES) == len(EXPECTED_SOURCE_IDS)

    by_type: dict[str, list[CommunicationSourceSpec]] = {}
    for source in COMMUNICATION_SOURCES:
        by_type.setdefault(source.organization_type, []).append(source)

    assert len(by_type["central_bank"]) == 6
    assert len(by_type["bank"]) == 7
    assert len(by_type["commodity_company"]) == 7
    assert {
        family for source in by_type["commodity_company"] for family in source.commodity_families
    } == REQUIRED_COMMODITY_FAMILIES

    expected_start_years = {
        "fed_fomc_press_conferences_en": 2011,
        "fed_fomc_press_conference_subtitles_en": 2011,
        "ecb_monetary_policy_press_conferences_en": 1998,
        "boe_monetary_policy_press_conferences_en": 2015,
        "boe_monetary_policy_press_conference_subtitles_en": 2015,
        "rba_speeches_en": 2024,
        "jpmorgan_chase_annual_reports_en": 2003,
        "hsbc_group_reporting_en": 2004,
        "deutsche_bank_annual_reports_en": 2007,
        "mufg_annual_reports_en": 2006,
        "ubs_annual_reports_en": 1998,
        "citigroup_annual_reports_en": 2016,
        "bank_of_china_reports_en": 2006,
        "shell_annual_reports_en": 2001,
        "bhp_financial_results_en": 2002,
        "barrick_annual_reports_en": 1998,
        "nutrien_financial_reporting_en": 2018,
        "bunge_annual_reports_en": 2008,
        "west_fraser_reports_en": None,
        "wilmar_annual_reports_en": 2006,
    }
    assert {
        source.source_id: source.verified_archive_start_year for source in COMMUNICATION_SOURCES
    } == expected_start_years
    assert all(source.coverage_note for source in COMMUNICATION_SOURCES)
    barrick = next(
        source
        for source in COMMUNICATION_SOURCES
        if source.source_id == "barrick_annual_reports_en"
    )
    assert barrick.organization_id == "barrick_gold"  # stable identity across the legal rename
    assert barrick.organization_name == "Barrick Mining Corporation"
    assert "Barrick Gold Corporation" in barrick.coverage_note


def test_subtitles_are_an_explicit_lower_fidelity_source_pathway():
    subtitle_sources = [
        source for source in COMMUNICATION_SOURCES if "subtitles" in source.material_types
    ]

    assert {source.source_id for source in subtitle_sources} == {
        "fed_fomc_press_conference_subtitles_en",
        "boe_monetary_policy_press_conference_subtitles_en",
    }
    assert all(source.has_transcript_material for source in subtitle_sources)
    assert all(source.transcriber_attribution == "artifact_specific" for source in subtitle_sources)
    assert all(source.provenance_tier == "official_archive_mixed" for source in subtitle_sources)
    assert all(source.acquisition_status == "manual_review_required" for source in subtitle_sources)


def test_catalogue_fingerprint_is_canonical_and_change_sensitive():
    assert CATALOGUE_SCHEMA_VERSION == 2
    assert COMMUNICATION_CATALOGUE_SHA256 == (
        "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
    )
    assert communication_catalogue_sha256(tuple(reversed(COMMUNICATION_SOURCES))) == (
        COMMUNICATION_CATALOGUE_SHA256
    )
    changed = (replace(COMMUNICATION_SOURCES[0], rights_note="Reviewed later."),) + (
        COMMUNICATION_SOURCES[1:]
    )
    assert communication_catalogue_sha256(changed) != COMMUNICATION_CATALOGUE_SHA256


def test_catalogue_is_metadata_only_and_keeps_provenance_roles_separate():
    for source in COMMUNICATION_SOURCES:
        assert source.host_organization
        assert source.publisher
        assert source.transcriber_attribution
        assert source.automated_collection_allowed is False
        assert source.rights_status in RIGHTS_STATUSES
        assert source.acquisition_status in ACQUISITION_STATUSES
        assert urlsplit(source.landing_url).scheme == "https"
        assert source.rights_note
        assert source.acquisition_note

    transcript_sources = {
        source.source_id: source
        for source in COMMUNICATION_SOURCES
        if source.has_transcript_material
    }
    assert transcript_sources["fed_fomc_press_conferences_en"].transcriber is None
    assert (
        transcript_sources["fed_fomc_press_conferences_en"].transcriber_attribution
        == "not_disclosed"
    )
    assert transcript_sources["bhp_financial_results_en"].transcriber_attribution == (
        "artifact_specific"
    )
    barrick = next(
        source
        for source in COMMUNICATION_SOURCES
        if source.source_id == "barrick_annual_reports_en"
    )
    assert barrick.organization_name == "Barrick Mining Corporation"
    assert barrick.host_organization == "Barrick Mining Corporation"
    assert barrick.publisher == "Barrick Mining Corporation"
    assert "Barrick Gold Corporation" in barrick.coverage_note


def test_rights_gates_are_conservative_and_machine_enforced():
    for source in COMMUNICATION_SOURCES:
        if source.rights_status == "rights_review_required":
            assert source.acquisition_status == "manual_review_required"
        elif source.rights_status == "permission_required":
            assert source.acquisition_status == "blocked_pending_permission"
        elif source.rights_status == "metadata_only":
            assert source.acquisition_status == "metadata_only"
        else:  # pragma: no cover - forces a deliberate test update for a new policy
            raise AssertionError(f"unexpected checked-in rights status: {source.rights_status}")

    metadata_only = {
        source.source_id
        for source in COMMUNICATION_SOURCES
        if source.rights_status == "metadata_only"
    }
    assert metadata_only == {"ubs_annual_reports_en", "citigroup_annual_reports_en"}
    assert all(
        source.rights_basis_url is not None
        for source in COMMUNICATION_SOURCES
        if source.rights_status in {"metadata_only", "permission_required"}
    )


def test_source_specs_are_frozen():
    with pytest.raises(FrozenInstanceError):
        COMMUNICATION_SOURCES[0].publisher = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("domain", "url"),
    [
        ("com", "https://evil.com/reports"),
        ("co.uk", "https://co.uk/reports"),
        ("localhost", "https://localhost/reports"),
        ("127.0.0.1", "https://127.0.0.1/reports"),
        ("federalreserve.gov.evil.com", "https://federalreserve.gov.evil.com/reports"),
    ],
)
def test_domain_allowlist_rejects_public_suffix_local_ip_and_unverified_roots(domain, url):
    with pytest.raises(ValueError, match="official domain"):
        validate_communication_sources(
            (replace(_example_spec(), official_domains=(domain,), landing_url=url),)
        )


@pytest.mark.parametrize(
    ("url", "message"),
    [
        ("https://www.jpmorganchase.com:8443/reports", "port"),
        ("https://www.jpmorganchase.com:bad/reports", "port"),
        ("https://www.jpmorganchase.com:99999/reports", "port"),
        ("https://user@www.jpmorganchase.com/reports", "credentials"),
        ("https://user:secret@www.jpmorganchase.com/reports", "credentials"),
        ("https://@www.jpmorganchase.com/reports", "credentials"),
        ("https://127.0.0.1/reports", "IP address"),
        ("https://localhost/reports", "local name"),
        ("https://www.jpmorganchase.com/re\nports", "control"),
        ("https://www.jpmorganchase.com/re\x00ports", "control"),
        ("https://www.jpmorganchase.com\\@evil.com/reports", "backslash"),
        ("https://www.jpmorganchase.com./reports", "trailing dot"),
        ("https://jpmorganchase.com.evil.com/reports", "official domain"),
    ],
)
def test_url_validation_rejects_ambiguous_or_deceptive_authorities(url, message):
    with pytest.raises(ValueError, match=message):
        validate_communication_sources((replace(_example_spec(), landing_url=url),))


def test_catalogue_hash_canonicalizes_set_like_tuple_order():
    example = replace(
        _example_spec(),
        material_types=("annual_report", "ceo_letter", "management_review"),
        commodity_families=("energy", "food_and_beverages"),
    )
    reordered = replace(
        example,
        material_types=tuple(reversed(example.material_types)),
        commodity_families=tuple(reversed(example.commodity_families)),
    )

    assert communication_catalogue_sha256((example,)) == communication_catalogue_sha256(
        (reordered,)
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"source_id": "Bad-ID"}, "source_id"),
        ({"source_id": "a" * 97}, "source_id"),
        ({"organization_id": "Bad ID"}, "organization_id"),
        ({"organization_id": "a" * 97}, "organization_id"),
        ({"organization_type": "insurer"}, "organization_type"),
        ({"jurisdiction": "United States"}, "jurisdiction"),
        ({"language": "EN"}, "language"),
        ({"landing_url": "http://investors.example.com/reports"}, "HTTPS"),
        ({"landing_url": "https://example.net/reports"}, "official domain"),
        ({"official_domains": ()}, "official domain"),
        ({"host_organization": ""}, "host_organization"),
        ({"publisher": ""}, "publisher"),
        ({"material_types": ()}, "material_types"),
        ({"material_types": ("unknown",)}, "material type"),
        ({"commodity_families": ("unknown",)}, "commodity family"),
        ({"verified_archive_start_year": True}, "verified_archive_start_year"),
        ({"verified_archive_start_year": 1899}, "verified_archive_start_year"),
        ({"coverage_note": ""}, "coverage_note"),
        ({"provenance_tier": "blog_copy"}, "provenance_tier"),
        ({"rights_status": "probably_ok"}, "rights_status"),
        ({"acquisition_status": "download_now"}, "acquisition_status"),
        ({"automated_collection_allowed": True}, "automated collection"),
        ({"rights_note": ""}, "rights_note"),
        ({"acquisition_note": ""}, "acquisition_note"),
    ],
)
def test_validation_rejects_unsafe_or_ambiguous_source_metadata(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_communication_sources((replace(_example_spec(), **changes),))


def test_validation_rejects_role_rights_and_commodity_mismatches():
    example = _example_spec()
    with pytest.raises(ValueError, match="Duplicate communication source id"):
        validate_communication_sources((example, example))

    with pytest.raises(ValueError, match="commodity families"):
        validate_communication_sources(
            (replace(example, organization_type="bank", commodity_families=("energy",)),)
        )
    with pytest.raises(ValueError, match="needs at least one commodity family"):
        validate_communication_sources((replace(example, commodity_families=()),))
    with pytest.raises(ValueError, match="rights/acquisition"):
        validate_communication_sources(
            (replace(example, acquisition_status="blocked_pending_permission"),)
        )
    with pytest.raises(ValueError, match="transcriber"):
        validate_communication_sources(
            (
                replace(
                    example,
                    material_types=("press_conference_transcript",),
                    transcriber_attribution="not_applicable",
                ),
            )
        )
    with pytest.raises(ValueError, match="named_third_party"):
        validate_communication_sources(
            (
                replace(
                    example,
                    material_types=("press_conference_transcript",),
                    transcriber="Transcript Vendor",
                    transcriber_attribution="not_disclosed",
                ),
            )
        )
    with pytest.raises(ValueError, match="provenance_tier"):
        validate_communication_sources(
            (
                replace(
                    example,
                    material_types=("results_transcript",),
                    provenance_tier="official_authored_text",
                    transcriber="Transcript Vendor",
                    transcriber_attribution="named_third_party",
                ),
            )
        )


def test_rights_basis_url_is_https_and_stays_on_an_official_domain():
    example = _example_spec()
    with pytest.raises(ValueError, match="rights_basis_url.*HTTPS"):
        validate_communication_sources(
            (replace(example, rights_basis_url="http://jpmorganchase.com/terms"),)
        )
    with pytest.raises(ValueError, match="rights_basis_url.*official domain"):
        validate_communication_sources(
            (replace(example, rights_basis_url="https://example.net/terms"),)
        )
    with pytest.raises(ValueError, match="requires a rights_basis_url"):
        validate_communication_sources(
            (
                replace(
                    example,
                    rights_status="permission_required",
                    acquisition_status="blocked_pending_permission",
                ),
            )
        )


@pytest.mark.parametrize(
    ("rights_status", "acquisition_status"),
    [
        ("cleared", "manual_collection_ready"),
        ("internal_only", "manual_internal_only"),
    ],
)
def test_storage_enabling_rights_states_require_documented_human_review(
    rights_status, acquisition_status
):
    example = replace(
        _example_spec(),
        rights_status=rights_status,
        acquisition_status=acquisition_status,
        rights_basis_url="https://www.jpmorganchase.com/legal/terms-and-conditions",
    )
    with pytest.raises(ValueError, match="documented human rights review"):
        validate_communication_sources((example,))
    with pytest.raises(ValueError, match="human:<id>"):
        validate_communication_sources(
            (
                replace(
                    example,
                    rights_checked_by="model:reviewer",
                    rights_checked_at=datetime(2026, 9, 9, tzinfo=UTC),
                ),
            )
        )
    with pytest.raises(ValueError, match="timezone"):
        validate_communication_sources(
            (
                replace(
                    example,
                    rights_checked_by="human:legal-reviewer",
                    rights_checked_at=datetime(2026, 9, 9),
                ),
            )
        )

    reviewed = replace(
        example,
        rights_checked_by="human:legal-reviewer",
        rights_checked_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="supplied together"):
        validate_communication_sources((replace(reviewed, rights_checked_at=None),))
    with pytest.raises(ValueError, match="evaluation time"):
        validate_communication_sources(
            (replace(reviewed, rights_checked_at=datetime(2026, 9, 10, tzinfo=UTC)),)
        )
    with pytest.raises(ValueError, match="requires a rights_basis_url"):
        validate_communication_sources((replace(reviewed, rights_basis_url=None),))

    validate_communication_sources((reviewed,))


def test_pilot_coverage_rejects_missing_institution_class_or_commodity_family():
    validate_pilot_coverage(COMMUNICATION_SOURCES)

    without_banks = tuple(
        source for source in COMMUNICATION_SOURCES if source.organization_type != "bank"
    )
    with pytest.raises(ValueError, match="bank"):
        validate_pilot_coverage(without_banks)

    without_energy = tuple(
        source for source in COMMUNICATION_SOURCES if "energy" not in source.commodity_families
    )
    with pytest.raises(ValueError, match="energy"):
        validate_pilot_coverage(without_energy)
