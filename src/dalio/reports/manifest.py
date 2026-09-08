"""Static allowlisted entry points for the first official report families.

This module is metadata only. It does not discover, fetch, or parse documents.
Landing pages are deliberately used instead of issue-specific PDF URLs so a
future discovery layer can retain both the latest and immediately prior issue.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

PDF_MIME_TYPE = "application/pdf"
LATEST_PLUS_PREVIOUS = "latest_plus_previous"
ALLOWED_ISSUE_RULES = frozenset({LATEST_PLUS_PREVIOUS})
MAX_CLAIMS_PER_DOCUMENT = 8

_SOURCE_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_DOMAIN = re.compile(r"^[a-z0-9](?:[a-z0-9.-]*[a-z0-9])?$")


@dataclass(frozen=True)
class ReportSourceSpec:
    """Policy and provenance boundary for one recurring official report."""

    source_id: str
    publisher: str
    report_family: str
    jurisdiction: str
    language: str
    landing_url: str
    official_domains: tuple[str, ...]
    topic_allowlist: tuple[str, ...]
    mime_type: str = PDF_MIME_TYPE
    issue_rule: str = LATEST_PLUS_PREVIOUS
    max_claims: int = MAX_CLAIMS_PER_DOCUMENT
    enabled: bool = True


REPORT_SOURCES: tuple[ReportSourceSpec, ...] = (
    ReportSourceSpec(
        source_id="riksbank_mpr_en",
        publisher="Sveriges Riksbank",
        report_family="Monetary Policy Report",
        jurisdiction="SE",
        language="en",
        landing_url=(
            "https://www.riksbank.se/en-gb/monetary-policy/monetary-policy-report/"
            "monetary-policy-reports-and-updates/"
        ),
        official_domains=("riksbank.se",),
        topic_allowlist=(
            "growth",
            "inflation",
            "labour_market",
            "monetary_policy",
            "exchange_rate",
            "financial_conditions",
            "risks",
        ),
    ),
    ReportSourceSpec(
        source_id="ecb_staff_projections_en",
        publisher="European Central Bank",
        report_family="ECB and Eurosystem staff macroeconomic projections",
        jurisdiction="EU",
        language="en",
        landing_url="https://www.ecb.europa.eu/press/projections/html/index.en.html",
        official_domains=("ecb.europa.eu",),
        topic_allowlist=(
            "growth",
            "inflation",
            "labour_market",
            "monetary_policy",
            "trade",
            "financial_conditions",
            "risks",
        ),
    ),
    ReportSourceSpec(
        source_id="fed_mpr_en",
        publisher="Board of Governors of the Federal Reserve System",
        report_family="Monetary Policy Report",
        jurisdiction="US",
        language="en",
        landing_url="https://www.federalreserve.gov/monetarypolicy/publications/mpr_default.htm",
        official_domains=("federalreserve.gov",),
        topic_allowlist=(
            "growth",
            "inflation",
            "labour_market",
            "monetary_policy",
            "financial_conditions",
            "financial_stability",
            "risks",
        ),
    ),
    ReportSourceSpec(
        source_id="imf_weo_en",
        publisher="International Monetary Fund",
        report_family="World Economic Outlook",
        jurisdiction="WLD",
        language="en",
        landing_url="https://www.imf.org/en/publications/weo",
        official_domains=("imf.org",),
        topic_allowlist=(
            "growth",
            "inflation",
            "fiscal_policy",
            "sovereign_debt",
            "trade",
            "capital_flows",
            "financial_conditions",
            "risks",
        ),
    ),
    ReportSourceSpec(
        source_id="bis_aer_en",
        publisher="Bank for International Settlements",
        report_family="Annual Economic Report",
        jurisdiction="WLD",
        language="en",
        landing_url="https://www.bis.org/publications/aer",
        official_domains=("bis.org",),
        topic_allowlist=(
            "growth",
            "inflation",
            "monetary_policy",
            "fiscal_policy",
            "sovereign_debt",
            "financial_stability",
            "banking",
            "capital_flows",
            "risks",
        ),
    ),
)


def validate_report_sources(sources: Sequence[ReportSourceSpec]) -> None:
    """Reject ambiguous or unsafe report-source metadata."""
    seen_ids: set[str] = set()
    for spec in sources:
        if not _SOURCE_ID.fullmatch(spec.source_id):
            raise ValueError(f"Invalid report source id: {spec.source_id!r}")
        if spec.source_id in seen_ids:
            raise ValueError(f"Duplicate report source id: {spec.source_id}")
        seen_ids.add(spec.source_id)

        if spec.issue_rule not in ALLOWED_ISSUE_RULES:
            raise ValueError(f"Unsupported issue rule for {spec.source_id}: {spec.issue_rule!r}")
        if not re.fullmatch(r"[A-Z]{2,3}", spec.jurisdiction):
            raise ValueError(f"{spec.source_id} has an invalid jurisdiction")
        if not re.fullmatch(r"[a-z]{2,3}", spec.language):
            raise ValueError(f"{spec.source_id} has an invalid language")

        parsed = urlsplit(spec.landing_url)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError(f"{spec.source_id} landing URL must use HTTPS")

        if not spec.official_domains:
            raise ValueError(f"{spec.source_id} needs an explicit official domain allowlist")
        domains = tuple(domain.lower() for domain in spec.official_domains)
        if any(not _DOMAIN.fullmatch(domain) for domain in domains):
            raise ValueError(f"{spec.source_id} has an invalid official domain allowlist")
        hostname = parsed.hostname.lower()
        if not any(hostname == domain or hostname.endswith(f".{domain}") for domain in domains):
            raise ValueError(
                f"{spec.source_id} landing URL is outside its official domain allowlist"
            )

        if spec.mime_type != PDF_MIME_TYPE:
            raise ValueError(f"{spec.source_id} mime_type must be application/pdf")
        if not spec.topic_allowlist or any(not topic.strip() for topic in spec.topic_allowlist):
            raise ValueError(f"{spec.source_id} needs a nonempty topic allowlist")
        if spec.max_claims != MAX_CLAIMS_PER_DOCUMENT:
            raise ValueError(f"{spec.source_id} max_claims must be 8")
        if spec.enabled is not True:
            raise ValueError(f"{spec.source_id} must be enabled")


validate_report_sources(REPORT_SOURCES)


__all__ = [
    "ALLOWED_ISSUE_RULES",
    "LATEST_PLUS_PREVIOUS",
    "MAX_CLAIMS_PER_DOCUMENT",
    "PDF_MIME_TYPE",
    "REPORT_SOURCES",
    "ReportSourceSpec",
    "validate_report_sources",
]
