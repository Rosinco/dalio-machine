"""Static report-source manifest contract; no documents are fetched."""

from dataclasses import FrozenInstanceError, replace
from urllib.parse import urlsplit

import pytest

from dalio.reports.manifest import (
    REPORT_SOURCES,
    ReportSourceSpec,
    validate_report_sources,
)

EXPECTED_SOURCES = {
    "riksbank_mpr_en": (
        "https://www.riksbank.se/en-gb/monetary-policy/monetary-policy-report/"
        "monetary-policy-reports-and-updates/",
        ("riksbank.se",),
    ),
    "ecb_staff_projections_en": (
        "https://www.ecb.europa.eu/press/projections/html/index.en.html",
        ("ecb.europa.eu",),
    ),
    "fed_mpr_en": (
        "https://www.federalreserve.gov/monetarypolicy/publications/mpr_default.htm",
        ("federalreserve.gov",),
    ),
    "imf_weo_en": (
        "https://www.imf.org/en/publications/weo",
        ("imf.org",),
    ),
    "bis_aer_en": (
        "https://www.bis.org/publications/aer",
        ("bis.org",),
    ),
}


def test_manifest_contains_exactly_the_five_verified_official_landings():
    assert {spec.source_id for spec in REPORT_SOURCES} == set(EXPECTED_SOURCES)
    assert len(REPORT_SOURCES) == len(EXPECTED_SOURCES)

    for spec in REPORT_SOURCES:
        expected_url, expected_domains = EXPECTED_SOURCES[spec.source_id]
        assert spec.landing_url == expected_url
        assert spec.official_domains == expected_domains
        assert urlsplit(spec.landing_url).scheme == "https"
        assert spec.mime_type == "application/pdf"
        assert spec.issue_rule == "latest_plus_previous"
        assert spec.jurisdiction in {"SE", "EU", "US", "WLD"}
        assert spec.language == "en"
        assert spec.topic_allowlist
        assert spec.max_claims == 8
        assert spec.enabled is True


def test_report_source_specs_are_frozen():
    with pytest.raises(FrozenInstanceError):
        REPORT_SOURCES[0].enabled = False  # type: ignore[misc]


def test_validation_rejects_duplicate_ids_and_unknown_issue_rules():
    first = REPORT_SOURCES[0]
    with pytest.raises(ValueError, match="Duplicate report source id"):
        validate_report_sources((first, first))

    with pytest.raises(ValueError, match="issue rule"):
        validate_report_sources((replace(first, issue_rule="all_issues"),))


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"landing_url": "http://www.riksbank.se/reports"}, "HTTPS"),
        ({"landing_url": "https://example.com/reports"}, "official domain"),
        ({"official_domains": ()}, "official domain"),
        ({"jurisdiction": "Sweden"}, "jurisdiction"),
        ({"language": "EN"}, "language"),
        ({"mime_type": "text/html"}, "application/pdf"),
        ({"topic_allowlist": ()}, "topic allowlist"),
        ({"max_claims": 7}, "max_claims"),
        ({"enabled": False}, "enabled"),
    ],
)
def test_validation_rejects_unsafe_or_out_of_contract_specs(changes, message):
    with pytest.raises(ValueError, match=message):
        validate_report_sources((replace(REPORT_SOURCES[0], **changes),))


def test_validator_accepts_an_independent_well_formed_spec():
    spec = ReportSourceSpec(
        source_id="example_report_en",
        publisher="Example institution",
        report_family="Example report",
        jurisdiction="WLD",
        language="en",
        landing_url="https://reports.example.org/current",
        official_domains=("example.org",),
        topic_allowlist=("growth",),
    )

    validate_report_sources((spec,))
