"""Static source catalogue for institutional communications research.

The records in this module are discovery metadata, not download permission and
not evidence artifacts.  They identify first-party archive entry points while
keeping the website host, text publisher, and transcript producer as separate
provenance roles.  Every checked-in source disables automated collection and is
held behind an explicit rights/acquisition gate.

Artifact-level dates, hashes, speakers, revisions, and transcriber attribution
belong in a later issue catalogue after an operator has selected and inspected
an exact document.  This module deliberately performs no network I/O.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from urllib.parse import urlsplit

ORGANIZATION_TYPES = frozenset({"central_bank", "bank", "commodity_company"})
CATALOGUE_SCHEMA_VERSION = 2
CATALOGUE_EVALUATED_AT = datetime(2026, 9, 9, 13, 26, 26, tzinfo=UTC)

MATERIAL_TYPES = frozenset(
    {
        "annual_report",
        "ceo_letter",
        "financial_results",
        "management_review",
        "monetary_policy_statement",
        "press_conference_slides",
        "press_conference_transcript",
        "press_conference_video",
        "questions_and_answers",
        "results_transcript",
        "speech_text",
        "subtitles",
    }
)
# ``subtitles`` is the storage-aligned representation kind for both subtitle
# and caption tracks.  Artifact metadata must separately classify its origin as
# official_caption, automatic_caption, or local_asr; the source catalogue cannot
# infer that fidelity from an embedded video.
TRANSCRIPT_MATERIAL_TYPES = frozenset(
    {
        "press_conference_transcript",
        "questions_and_answers",
        "results_transcript",
        "subtitles",
    }
)

REQUIRED_COMMODITY_FAMILIES = frozenset(
    {
        "agricultural_raw_materials",
        "base_metals",
        "energy",
        "fertilizers",
        "food_and_beverages",
        "precious_metals",
    }
)

PROVENANCE_TIERS = frozenset(
    {
        "official_archive_mixed",
        "official_authored_text",
        "official_published_transcript",
        "official_hosted_third_party",
    }
)
TRANSCRIBER_ATTRIBUTIONS = frozenset(
    {
        "artifact_specific",
        "named_third_party",
        "not_applicable",
        "not_disclosed",
        "publisher",
    }
)

# A rights label is deliberately not inferred from an official-looking host.
# ``cleared`` and ``internal_only`` are supported for future, documented legal
# review, but no current source is assigned either status without that review.
RIGHTS_STATUSES = frozenset(
    {
        "cleared",
        "internal_only",
        "metadata_only",
        "permission_required",
        "rights_review_required",
    }
)
ACQUISITION_STATUSES = frozenset(
    {
        "blocked_pending_permission",
        "manual_collection_ready",
        "manual_internal_only",
        "manual_review_required",
        "metadata_only",
    }
)
_RIGHTS_TO_ACQUISITION = {
    "cleared": "manual_collection_ready",
    "internal_only": "manual_internal_only",
    "metadata_only": "metadata_only",
    "permission_required": "blocked_pending_permission",
    "rights_review_required": "manual_review_required",
}

_ID = re.compile(r"^[a-z][a-z0-9_]*$")
_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_MAX_ID_LENGTH = 96
_APPROVED_OFFICIAL_DOMAINS = frozenset(
    {
        "bankofengland.co.uk",
        "barrick.com",
        "bhp.com",
        "boc.cn",
        "bunge.com",
        "citigroup.com",
        "db.com",
        "ecb.europa.eu",
        "federalreserve.gov",
        "hsbc.com",
        "jpmorganchase.com",
        "mufg.jp",
        "nutrien.com",
        "rba.gov.au",
        "riksbank.se",
        "shell.com",
        "ubs.com",
        "westfraser.com",
        "wilmar-international.com",
    }
)

_RIGHTS_REVIEW_NOTE = (
    "No reusable corpus licence has been verified; review the terms and each exact artifact "
    "before capture."
)
_PERMISSION_NOTE = (
    "Systematic content collection is blocked until permission or a documented legal basis "
    "is recorded."
)
_METADATA_NOTE = (
    "Retain only source metadata and links; do not archive or transcribe source content under "
    "this catalogue status."
)
_MANUAL_REVIEW_NOTE = (
    "No network collector is authorized; an operator must review rights and select exact artifacts."
)
_BLOCKED_NOTE = "Do not acquire content while the source remains on the permission hold."
_LINK_ONLY_NOTE = "Keep the archive landing link only; content acquisition is disabled."


@dataclass(frozen=True)
class CommunicationSourceSpec:
    """One verified first-party landing point with an explicit collection gate.

    ``host_organization`` is the operator of the verified current landing page,
    and ``publisher`` is only the expected current-source default.  Neither may
    be copied onto an old artifact without checking the name printed on that
    artifact; historical publishers and hosts belong in artifact metadata.

    ``transcriber is None`` never means that the publisher transcribed an event.
    Its meaning is controlled by ``transcriber_attribution``: for example,
    ``not_disclosed`` means an official transcript is present but its producer is
    not named, while ``artifact_specific`` requires inspection of each issue.
    """

    source_id: str
    organization_id: str
    organization_name: str
    organization_type: str
    jurisdiction: str
    language: str
    landing_url: str
    official_domains: tuple[str, ...]
    host_organization: str
    publisher: str
    transcriber: str | None
    transcriber_attribution: str
    material_types: tuple[str, ...]
    commodity_families: tuple[str, ...]
    verified_archive_start_year: int | None
    coverage_note: str
    provenance_tier: str
    rights_status: str
    rights_basis_url: str | None
    rights_note: str
    acquisition_status: str
    acquisition_note: str
    automated_collection_allowed: bool = False
    rights_checked_by: str | None = None
    rights_checked_at: datetime | None = None

    @property
    def has_transcript_material(self) -> bool:
        return bool(TRANSCRIPT_MATERIAL_TYPES.intersection(self.material_types))


@dataclass(frozen=True)
class CommunicationCatalogueSnapshot:
    """One immutable source-policy catalogue vintage addressed by its semantic hash."""

    catalogue_sha256: str
    schema_version: int
    evaluated_at: datetime
    sources: tuple[CommunicationSourceSpec, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.catalogue_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.catalogue_sha256) is None
        ):
            raise ValueError("catalogue_sha256 must be a lowercase SHA-256")
        schema_version = _catalogue_schema_version(self.schema_version)
        evaluated_at = _catalogue_evaluated_at(self.evaluated_at)
        if not isinstance(self.sources, tuple) or not self.sources:
            raise ValueError("catalogue snapshot sources must be a non-empty tuple")
        expected_sha256 = _communication_catalogue_sha256(
            self.sources,
            schema_version=schema_version,
            evaluated_at=evaluated_at,
        )
        if self.catalogue_sha256 != expected_sha256:
            raise ValueError("catalogue_sha256 does not match the snapshot semantics")
        object.__setattr__(self, "evaluated_at", evaluated_at)


_COMMUNICATION_SOURCES_2026_09_09: tuple[CommunicationSourceSpec, ...] = (
    # Central banks: official policy text and event archives.  These records do
    # not assume that captions, embedded media, and transcript text share rights.
    CommunicationSourceSpec(
        source_id="fed_fomc_press_conferences_en",
        organization_id="federal_reserve",
        organization_name="Board of Governors of the Federal Reserve System",
        organization_type="central_bank",
        jurisdiction="US",
        language="en",
        landing_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        official_domains=("federalreserve.gov",),
        host_organization="Board of Governors of the Federal Reserve System",
        publisher="Board of Governors of the Federal Reserve System",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        material_types=("press_conference_transcript", "press_conference_video"),
        commodity_families=(),
        verified_archive_start_year=2011,
        coverage_note=(
            "Official post-meeting press conferences begin in 2011; separately released, "
            "delayed FOMC meeting transcripts are outside this source family."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "Board-authored material may have a public-domain basis, but third-party and "
            "embedded-media notices must be checked on each artifact."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="fed_fomc_press_conference_subtitles_en",
        organization_id="federal_reserve",
        organization_name="Board of Governors of the Federal Reserve System",
        organization_type="central_bank",
        jurisdiction="US",
        language="en",
        landing_url="https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
        official_domains=("federalreserve.gov",),
        host_organization="Board of Governors of the Federal Reserve System",
        publisher="Board of Governors of the Federal Reserve System",
        transcriber=None,
        transcriber_attribution="artifact_specific",
        material_types=("subtitles",),
        commodity_families=(),
        verified_archive_start_year=2011,
        coverage_note=(
            "This is a subtitle-discovery pathway for press-conference media from 2011, not "
            "a claim that every event has a caption track.  Each track's media host, caption "
            "producer, automatic/official origin and time coverage require artifact review."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "Embedded-media and caption rights can differ from Board-authored page text and "
            "must be reviewed for each exact track."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="ecb_monetary_policy_press_conferences_en",
        organization_id="ecb",
        organization_name="European Central Bank",
        organization_type="central_bank",
        jurisdiction="EU",
        language="en",
        landing_url=(
            "https://www.ecb.europa.eu/press/press_conference/"
            "monetary-policy-statement/html/index.en.html"
        ),
        official_domains=("ecb.europa.eu",),
        host_organization="European Central Bank",
        publisher="European Central Bank",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        material_types=("monetary_policy_statement", "questions_and_answers"),
        commodity_families=(),
        verified_archive_start_year=1998,
        coverage_note=(
            "The statement archive reaches 1998; statement and Q&A availability must still "
            "be checked for each individual event."
        ),
        provenance_tier="official_published_transcript",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "ECB pages commonly state source-attribution terms, but the notice must be "
            "verified for every selected text and media artifact."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="boe_monetary_policy_press_conference_subtitles_en",
        organization_id="bank_of_england",
        organization_name="Bank of England",
        organization_type="central_bank",
        jurisdiction="GB",
        language="en",
        landing_url="https://www.bankofengland.co.uk/press-conferences",
        official_domains=("bankofengland.co.uk",),
        host_organization="Bank of England",
        publisher="Bank of England",
        transcriber=None,
        transcriber_attribution="artifact_specific",
        material_types=("subtitles",),
        commodity_families=(),
        verified_archive_start_year=2015,
        coverage_note=(
            "This is a subtitle-discovery pathway for archived press-conference video, not a "
            "claim of caption continuity.  The actual media host, caption producer, origin and "
            "time coverage require artifact review."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "Video-platform caption rights can differ from Bank page and transcript rights; "
            "review every exact subtitle track before capture."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="boe_monetary_policy_press_conferences_en",
        organization_id="bank_of_england",
        organization_name="Bank of England",
        organization_type="central_bank",
        jurisdiction="GB",
        language="en",
        landing_url="https://www.bankofengland.co.uk/press-conferences",
        official_domains=("bankofengland.co.uk",),
        host_organization="Bank of England",
        publisher="Bank of England",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        material_types=("press_conference_transcript", "press_conference_video"),
        commodity_families=(),
        verified_archive_start_year=2015,
        coverage_note=(
            "The verified press-conference archive reaches 2015; transcript availability "
            "varies by event and must not be inferred from video coverage."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="rba_speeches_en",
        organization_id="reserve_bank_of_australia",
        organization_name="Reserve Bank of Australia",
        organization_type="central_bank",
        jurisdiction="AU",
        language="en",
        landing_url="https://www.rba.gov.au/speeches/2024/",
        official_domains=("rba.gov.au",),
        host_organization="Reserve Bank of Australia",
        publisher="Reserve Bank of Australia",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("speech_text",),
        commodity_families=(),
        verified_archive_start_year=2024,
        coverage_note=(
            "This verified landing covers the 2024 speech archive only; older annual archive "
            "entry points require separate verification."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    # Banks: issuer-authored annual reporting is the initial canonical text.
    # Earnings-call transcripts can be catalogued later only after their actual
    # publisher/transcriber and licence have been established per artifact.
    CommunicationSourceSpec(
        source_id="jpmorgan_chase_annual_reports_en",
        organization_id="jpmorgan_chase",
        organization_name="JPMorgan Chase & Co.",
        organization_type="bank",
        jurisdiction="US",
        language="en",
        landing_url="https://www.jpmorganchase.com/ir/annual-report",
        official_domains=("jpmorganchase.com",),
        host_organization="JPMorgan Chase & Co.",
        publisher="JPMorgan Chase & Co.",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "ceo_letter"),
        commodity_families=(),
        verified_archive_start_year=2003,
        coverage_note=(
            "Annual reports and shareholder letters are verified from 2003; this source does "
            "not assert an earnings-call transcript history."
        ),
        provenance_tier="official_authored_text",
        rights_status="permission_required",
        rights_basis_url="https://www.jpmorganchase.com/legal/terms-and-conditions",
        rights_note=_PERMISSION_NOTE,
        acquisition_status="blocked_pending_permission",
        acquisition_note=_BLOCKED_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="hsbc_group_reporting_en",
        organization_id="hsbc",
        organization_name="HSBC Holdings plc",
        organization_type="bank",
        jurisdiction="GB",
        language="en",
        landing_url=(
            "https://www.hsbc.com/investors/results-and-announcements/all-reporting/group"
        ),
        official_domains=("hsbc.com",),
        host_organization="HSBC Holdings plc",
        publisher="HSBC Holdings plc",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=2004,
        coverage_note=(
            "Group annual reporting is verified from 2004; management-letter presence and "
            "location are artifact-specific."
        ),
        provenance_tier="official_authored_text",
        rights_status="permission_required",
        rights_basis_url="https://www.hsbc.com/terms-and-conditions",
        rights_note=_PERMISSION_NOTE,
        acquisition_status="blocked_pending_permission",
        acquisition_note=_BLOCKED_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="deutsche_bank_annual_reports_en",
        organization_id="deutsche_bank",
        organization_name="Deutsche Bank AG",
        organization_type="bank",
        jurisdiction="DE",
        language="en",
        landing_url=("https://investor-relations.db.com/reports-and-events/annual-reports/index"),
        official_domains=("db.com",),
        host_organization="Deutsche Bank AG",
        publisher="Deutsche Bank AG",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=2007,
        coverage_note=(
            "Annual reports are verified from 2007; no call-transcript continuity is claimed."
        ),
        provenance_tier="official_authored_text",
        rights_status="permission_required",
        rights_basis_url="https://www.db.com/legal-resources/index?language_id=1",
        rights_note=_PERMISSION_NOTE,
        acquisition_status="blocked_pending_permission",
        acquisition_note=_BLOCKED_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="mufg_annual_reports_en",
        organization_id="mufg",
        organization_name="Mitsubishi UFJ Financial Group, Inc.",
        organization_type="bank",
        jurisdiction="JP",
        language="en",
        landing_url=("https://www.mufg.jp/english/ir/report/annual_report/backnumber/index.html"),
        official_domains=("mufg.jp",),
        host_organization="Mitsubishi UFJ Financial Group, Inc.",
        publisher="Mitsubishi UFJ Financial Group, Inc.",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=2006,
        coverage_note=(
            "English annual reports are verified from 2006; management-review naming varies "
            "across report vintages."
        ),
        provenance_tier="official_authored_text",
        rights_status="permission_required",
        rights_basis_url="https://www.mufg.jp/english/conditions/index.html",
        rights_note=_PERMISSION_NOTE,
        acquisition_status="blocked_pending_permission",
        acquisition_note=_BLOCKED_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="ubs_annual_reports_en",
        organization_id="ubs",
        organization_name="UBS Group AG",
        organization_type="bank",
        jurisdiction="CH",
        language="en",
        landing_url=(
            "https://www.ubs.com/global/en/investor-relations/financial-information/"
            "annual-reporting/ar-archive.html"
        ),
        official_domains=("ubs.com",),
        host_organization="UBS Group AG",
        publisher="UBS Group AG",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=1998,
        coverage_note=(
            "Annual reporting is verified from 1998; predecessor and post-merger entity scope "
            "must be preserved per artifact."
        ),
        provenance_tier="official_authored_text",
        rights_status="metadata_only",
        rights_basis_url="https://www.ubs.com/global/en/legal/disclaimer.html",
        rights_note=_METADATA_NOTE,
        acquisition_status="metadata_only",
        acquisition_note=_LINK_ONLY_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="citigroup_annual_reports_en",
        organization_id="citigroup",
        organization_name="Citigroup Inc.",
        organization_type="bank",
        jurisdiction="US",
        language="en",
        landing_url=(
            "https://www.citigroup.com/global/investors/annual-reports-and-proxy-statements"
        ),
        official_domains=("citigroup.com",),
        host_organization="Citigroup Inc.",
        publisher="Citigroup Inc.",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=2016,
        coverage_note=(
            "The current annual-report landing archive is verified from 2016; earlier filings "
            "require a separately catalogued source."
        ),
        provenance_tier="official_authored_text",
        rights_status="metadata_only",
        rights_basis_url="https://www.citigroup.com/global/terms",
        rights_note=_METADATA_NOTE,
        acquisition_status="metadata_only",
        acquisition_note=_LINK_ONLY_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="bank_of_china_reports_en",
        organization_id="bank_of_china",
        organization_name="Bank of China Limited",
        organization_type="bank",
        jurisdiction="CN",
        language="en",
        landing_url="https://www.boc.cn/en/investor/ir3/",
        official_domains=("boc.cn",),
        host_organization="Bank of China Limited",
        publisher="Bank of China Limited",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=(),
        verified_archive_start_year=2006,
        coverage_note=(
            "English investor reports are verified from 2006; translated and original-language "
            "artifacts must remain distinct."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    # Commodity issuers: family balance prevents document-rich diversified
    # companies from becoming multiple votes in later narrative comparisons.
    CommunicationSourceSpec(
        source_id="shell_annual_reports_en",
        organization_id="shell",
        organization_name="Shell plc",
        organization_type="commodity_company",
        jurisdiction="GB",
        language="en",
        landing_url=(
            "https://www.shell.com/investors/results-and-reporting/annual-report-archive.html"
        ),
        official_domains=("shell.com",),
        host_organization="Shell plc",
        publisher="Shell plc",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=("energy",),
        verified_archive_start_year=2001,
        coverage_note=(
            "Annual reports are verified from 2001; this source does not claim continuous "
            "results-call transcript coverage."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="bhp_financial_results_en",
        organization_id="bhp",
        organization_name="BHP Group Limited",
        organization_type="commodity_company",
        jurisdiction="AU",
        language="en",
        landing_url="https://www.bhp.com/financial-results",
        official_domains=("bhp.com",),
        host_organization="BHP Group Limited",
        publisher="BHP Group Limited",
        transcriber=None,
        transcriber_attribution="artifact_specific",
        material_types=("financial_results", "management_review", "results_transcript"),
        commodity_families=("base_metals", "energy"),
        verified_archive_start_year=2002,
        coverage_note=(
            "Financial-result material is verified from 2002; transcript presence and producer "
            "must be established separately for every result event."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="barrick_annual_reports_en",
        organization_id="barrick_gold",
        organization_name="Barrick Mining Corporation",
        organization_type="commodity_company",
        jurisdiction="CA",
        language="en",
        landing_url="https://www.barrick.com/English/investors/annual-report/default.aspx",
        official_domains=("barrick.com",),
        host_organization="Barrick Mining Corporation",
        publisher="Barrick Mining Corporation",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "ceo_letter"),
        commodity_families=("precious_metals",),
        verified_archive_start_year=1998,
        coverage_note=(
            "The current archive is operated by Barrick Mining Corporation and reaches 1998. "
            "Historical reports were published under Barrick Gold Corporation; preserve the "
            "legal name printed on each artifact rather than applying the current name backward."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="nutrien_financial_reporting_en",
        organization_id="nutrien",
        organization_name="Nutrien Ltd.",
        organization_type="commodity_company",
        jurisdiction="CA",
        language="en",
        landing_url="https://www.nutrien.com/investors/financial-reporting",
        official_domains=("nutrien.com",),
        host_organization="Nutrien Ltd.",
        publisher="Nutrien Ltd.",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=("fertilizers",),
        verified_archive_start_year=2018,
        coverage_note=(
            "Nutrien annual reporting is verified from 2018; predecessor-company reports are "
            "outside this organization source."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="bunge_annual_reports_en",
        organization_id="bunge",
        organization_name="Bunge",
        organization_type="commodity_company",
        jurisdiction="CH",
        language="en",
        landing_url="https://investors.bunge.com/financial-information/annual-reports",
        official_domains=("bunge.com",),
        host_organization="Bunge",
        publisher="Bunge",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=("food_and_beverages",),
        verified_archive_start_year=2008,
        coverage_note=(
            "Annual reports are verified from 2008; earnings-call transcripts require their "
            "own artifact-level publisher and transcriber checks."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="west_fraser_reports_en",
        organization_id="west_fraser",
        organization_name="West Fraser Timber Co. Ltd.",
        organization_type="commodity_company",
        jurisdiction="CA",
        language="en",
        landing_url="https://www.westfraser.com/reports-presentations-filings-archive",
        official_domains=("westfraser.com",),
        host_organization="West Fraser Timber Co. Ltd.",
        publisher="West Fraser Timber Co. Ltd.",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=("agricultural_raw_materials",),
        verified_archive_start_year=None,
        coverage_note=(
            "The verified page is a rolling five-year archive; no earlier continuous start year "
            "is asserted."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="wilmar_annual_reports_en",
        organization_id="wilmar_international",
        organization_name="Wilmar International Limited",
        organization_type="commodity_company",
        jurisdiction="SG",
        language="en",
        landing_url="https://ir-media.wilmar-international.com/annual-reports",
        official_domains=("wilmar-international.com",),
        host_organization="Wilmar International Limited",
        publisher="Wilmar International Limited",
        transcriber=None,
        transcriber_attribution="not_applicable",
        material_types=("annual_report", "management_review"),
        commodity_families=("agricultural_raw_materials", "food_and_beverages"),
        verified_archive_start_year=2006,
        coverage_note=(
            "Annual reports are verified from 2006; commodity-family exposure must be dated "
            "from each report rather than projected backward."
        ),
        provenance_tier="official_authored_text",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=_RIGHTS_REVIEW_NOTE,
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
)

_RIKSBANK_COMMUNICATION_SOURCES_2026_09_09: tuple[CommunicationSourceSpec, ...] = (
    CommunicationSourceSpec(
        source_id="riksbank_monetary_policy_press_conferences_sv",
        organization_id="sveriges_riksbank",
        organization_name="Sveriges Riksbank",
        organization_type="central_bank",
        jurisdiction="SE",
        language="sv",
        landing_url="https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/",
        official_domains=("riksbank.se",),
        host_organization="Sveriges Riksbank",
        publisher="Sveriges Riksbank",
        transcriber=None,
        transcriber_attribution="not_disclosed",
        material_types=(
            "press_conference_transcript",
            "press_conference_video",
            "press_conference_slides",
        ),
        commodity_families=(),
        verified_archive_start_year=2025,
        coverage_note=(
            "The verified 2025 Riksbanken Play pages provide first-party replay pages and "
            "links to presentation slides. Transcript availability is not established and "
            "must be observed independently; embedded-player locators are outside this policy."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "Riksbanken Play page, slide-document and embedded-media rights can differ; review "
            "each exact representation before capture."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
    CommunicationSourceSpec(
        source_id="riksbank_monetary_policy_press_conference_subtitles_sv",
        organization_id="sveriges_riksbank",
        organization_name="Sveriges Riksbank",
        organization_type="central_bank",
        jurisdiction="SE",
        language="sv",
        landing_url="https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/",
        official_domains=("riksbank.se",),
        host_organization="Sveriges Riksbank",
        publisher="Sveriges Riksbank",
        transcriber=None,
        transcriber_attribution="artifact_specific",
        material_types=("subtitles",),
        commodity_families=(),
        verified_archive_start_year=2025,
        coverage_note=(
            "The 2025 replay pages are a subtitle-discovery pathway only. No caption-track "
            "continuity, producer, origin, language or time coverage is asserted."
        ),
        provenance_tier="official_archive_mixed",
        rights_status="rights_review_required",
        rights_basis_url=None,
        rights_note=(
            "Replay-platform caption rights can differ from Riksbank page and document rights; "
            "review every exact subtitle track before capture."
        ),
        acquisition_status="manual_review_required",
        acquisition_note=_MANUAL_REVIEW_NOTE,
    ),
)

# Public current-vintage alias.  The original tuple remains separately named and
# hash-pinned so manifests bound to that policy vintage retain their semantics.
COMMUNICATION_SOURCES = (
    *_COMMUNICATION_SOURCES_2026_09_09,
    *_RIKSBANK_COMMUNICATION_SOURCES_2026_09_09,
)


def _required_string(value: object, field: str, source_id: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{source_id} {field} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{source_id} {field} must not contain surrounding whitespace")
    return value


def _validate_dns_name(value: object, *, field: str, source_id: str) -> str:
    """Validate one pre-reviewed DNS policy root without resolving it."""
    raw = _required_string(value, field, source_id)
    if any(ord(character) <= 32 or ord(character) == 127 for character in raw):
        raise ValueError(f"{source_id} {field} contains control or whitespace characters")
    if raw != raw.lower() or raw.startswith(".") or raw.endswith(".") or ".." in raw:
        raise ValueError(f"{source_id} {field} is not a canonical official domain")
    try:
        ipaddress.ip_address(raw)
    except ValueError:
        pass
    else:
        raise ValueError(f"{source_id} {field} cannot be an IP address")
    labels = raw.split(".")
    if raw == "localhost" or len(labels) < 2 or len(raw) > 253:
        raise ValueError(
            f"{source_id} {field} must be a registrable official domain, not a local name"
        )
    if any(_DNS_LABEL.fullmatch(label) is None for label in labels):
        raise ValueError(f"{source_id} {field} is not a valid DNS domain")
    if raw not in _APPROVED_OFFICIAL_DOMAINS:
        raise ValueError(
            f"{source_id} {field} is not a verified registrable official domain; "
            "public suffixes and unreviewed roots are rejected"
        )
    return raw


def _validate_url(
    value: object,
    *,
    field: str,
    source_id: str,
    official_domains: tuple[str, ...],
) -> str:
    raw = _required_string(value, field, source_id)
    if any(ord(character) <= 32 or ord(character) == 127 for character in raw):
        raise ValueError(f"{source_id} {field} contains control or whitespace characters")
    if "\\" in raw:
        raise ValueError(f"{source_id} {field} must not contain a backslash")
    try:
        parsed = urlsplit(raw)
    except ValueError as exc:
        raise ValueError(f"{source_id} {field} has an invalid URL authority") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname:
        raise ValueError(f"{source_id} {field} must be an HTTPS URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{source_id} {field} must not contain credentials")
    if parsed.hostname.endswith("."):
        raise ValueError(f"{source_id} {field} hostname must not have a trailing dot")
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError(f"{source_id} {field} has an invalid port") from exc
    if port not in {None, 443}:
        raise ValueError(f"{source_id} {field} must not use a nonstandard HTTPS port")
    host = parsed.hostname.lower()
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ValueError(f"{source_id} {field} hostname must not be an IP address")
    if host == "localhost" or "." not in host:
        raise ValueError(f"{source_id} {field} hostname must not be a local name")
    if len(host) > 253 or any(_DNS_LABEL.fullmatch(label) is None for label in host.split(".")):
        raise ValueError(f"{source_id} {field} hostname is not a valid DNS name")
    if not any(host == domain or host.endswith(f".{domain}") for domain in official_domains):
        raise ValueError(f"{source_id} {field} is outside its official domain allowlist")
    return raw


def _validate_distinct_tuple(
    values: tuple[str, ...],
    *,
    field: str,
    source_id: str,
    allowed: frozenset[str],
    allow_empty: bool,
) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise ValueError(f"{source_id} {field} must be a non-empty tuple")
    if any(not isinstance(value, str) or value not in allowed for value in values):
        singular = {
            "commodity_families": "commodity family",
            "material_types": "material type",
        }.get(field, field)
        raise ValueError(f"{source_id} has an unsupported {singular}")
    if len(values) != len(set(values)):
        raise ValueError(f"{source_id} {field} contains duplicates")


def _catalogue_schema_version(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("catalogue schema_version must be an integer >= 1")
    return value


def _catalogue_evaluated_at(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("catalogue evaluated_at must be a timezone-aware datetime")
    return value.astimezone(UTC)


def validate_communication_sources(
    sources: Sequence[CommunicationSourceSpec],
    *,
    evaluated_at: datetime = CATALOGUE_EVALUATED_AT,
) -> None:
    """Reject ambiguous provenance, unsafe URLs, and permissive collection metadata."""
    evaluation_time = _catalogue_evaluated_at(evaluated_at)
    seen_source_ids: set[str] = set()
    organization_identity: dict[str, tuple[str, str, str]] = {}

    for spec in sources:
        if not _ID.fullmatch(spec.source_id) or len(spec.source_id) > _MAX_ID_LENGTH:
            raise ValueError(f"Invalid communication source_id: {spec.source_id!r}")
        if spec.source_id in seen_source_ids:
            raise ValueError(f"Duplicate communication source id: {spec.source_id}")
        seen_source_ids.add(spec.source_id)

        if not _ID.fullmatch(spec.organization_id) or len(spec.organization_id) > _MAX_ID_LENGTH:
            raise ValueError(f"{spec.source_id} has an invalid organization_id")
        organization_name = _required_string(
            spec.organization_name, "organization_name", spec.source_id
        )
        if spec.organization_type not in ORGANIZATION_TYPES:
            raise ValueError(f"{spec.source_id} has an unsupported organization_type")
        if not re.fullmatch(r"[A-Z]{2,3}", spec.jurisdiction):
            raise ValueError(f"{spec.source_id} has an invalid jurisdiction")
        if not re.fullmatch(r"[a-z]{2,3}", spec.language):
            raise ValueError(f"{spec.source_id} has an invalid language")

        identity = (organization_name, spec.organization_type, spec.jurisdiction)
        previous_identity = organization_identity.setdefault(spec.organization_id, identity)
        if previous_identity != identity:
            raise ValueError(f"{spec.organization_id} has inconsistent organization identity")

        if not spec.official_domains:
            raise ValueError(f"{spec.source_id} needs an official domain allowlist")
        if not isinstance(spec.official_domains, tuple):
            raise ValueError(f"{spec.source_id} official domains must be a tuple")
        for domain in spec.official_domains:
            _validate_dns_name(
                domain,
                field="official domain",
                source_id=spec.source_id,
            )
        if len(spec.official_domains) != len(set(spec.official_domains)):
            raise ValueError(f"{spec.source_id} official domain allowlist contains duplicates")
        _validate_url(
            spec.landing_url,
            field="landing_url",
            source_id=spec.source_id,
            official_domains=spec.official_domains,
        )

        _required_string(spec.host_organization, "host_organization", spec.source_id)
        publisher = _required_string(spec.publisher, "publisher", spec.source_id)
        if spec.transcriber is not None:
            _required_string(spec.transcriber, "transcriber", spec.source_id)
        if spec.transcriber_attribution not in TRANSCRIBER_ATTRIBUTIONS:
            raise ValueError(f"{spec.source_id} has an unsupported transcriber attribution")

        _validate_distinct_tuple(
            spec.material_types,
            field="material_types",
            source_id=spec.source_id,
            allowed=MATERIAL_TYPES,
            allow_empty=False,
        )
        has_transcript = bool(TRANSCRIPT_MATERIAL_TYPES.intersection(spec.material_types))
        if has_transcript and spec.transcriber_attribution == "not_applicable":
            raise ValueError(f"{spec.source_id} transcript material needs transcriber provenance")
        if not has_transcript and spec.transcriber_attribution != "not_applicable":
            raise ValueError(
                f"{spec.source_id} has transcriber metadata without transcript material"
            )
        if spec.transcriber_attribution == "named_third_party" and spec.transcriber is None:
            raise ValueError(f"{spec.source_id} named_third_party requires a transcriber")
        if spec.transcriber is not None and spec.transcriber_attribution not in {
            "named_third_party",
            "publisher",
        }:
            raise ValueError(
                f"{spec.source_id} a named transcriber requires named_third_party or publisher"
            )
        if spec.transcriber_attribution == "publisher" and spec.transcriber != publisher:
            raise ValueError(f"{spec.source_id} publisher transcriber must equal publisher")
        if spec.transcriber_attribution in {"artifact_specific", "not_disclosed"} and (
            spec.transcriber is not None
        ):
            raise ValueError(f"{spec.source_id} transcriber must remain unset at source level")
        if spec.provenance_tier == "official_authored_text" and has_transcript:
            raise ValueError(
                f"{spec.source_id} provenance_tier official_authored_text cannot label "
                "transcript or subtitle material"
            )
        if spec.provenance_tier == "official_published_transcript" and not has_transcript:
            raise ValueError(
                f"{spec.source_id} provenance_tier official_published_transcript needs "
                "transcript material"
            )
        if (
            spec.provenance_tier == "official_hosted_third_party"
            and spec.transcriber_attribution != "named_third_party"
        ):
            raise ValueError(
                f"{spec.source_id} provenance_tier official_hosted_third_party needs a named "
                "third-party transcriber"
            )
        if (
            spec.transcriber_attribution == "named_third_party"
            and spec.provenance_tier != "official_hosted_third_party"
        ):
            raise ValueError(
                f"{spec.source_id} named vendor text needs provenance_tier "
                "official_hosted_third_party"
            )

        _validate_distinct_tuple(
            spec.commodity_families,
            field="commodity_families",
            source_id=spec.source_id,
            allowed=REQUIRED_COMMODITY_FAMILIES,
            allow_empty=True,
        )
        if spec.organization_type == "commodity_company" and not spec.commodity_families:
            raise ValueError(f"{spec.source_id} needs at least one commodity family")
        if spec.organization_type != "commodity_company" and spec.commodity_families:
            raise ValueError(f"{spec.source_id} non-commodity organization has commodity families")

        if spec.verified_archive_start_year is not None and (
            isinstance(spec.verified_archive_start_year, bool)
            or not isinstance(spec.verified_archive_start_year, int)
            or not 1900 <= spec.verified_archive_start_year <= evaluation_time.year
        ):
            raise ValueError(f"{spec.source_id} has an invalid verified_archive_start_year")
        _required_string(spec.coverage_note, "coverage_note", spec.source_id)

        if spec.provenance_tier not in PROVENANCE_TIERS:
            raise ValueError(f"{spec.source_id} has an unsupported provenance_tier")
        if spec.rights_status not in RIGHTS_STATUSES:
            raise ValueError(f"{spec.source_id} has an unsupported rights_status")
        if spec.acquisition_status not in ACQUISITION_STATUSES:
            raise ValueError(f"{spec.source_id} has an unsupported acquisition_status")
        expected_acquisition = _RIGHTS_TO_ACQUISITION[spec.rights_status]
        if spec.acquisition_status != expected_acquisition:
            raise ValueError(f"{spec.source_id} has an unsafe rights/acquisition combination")
        if spec.automated_collection_allowed is not False:
            raise ValueError(f"{spec.source_id} automated collection must remain disabled")
        _required_string(spec.rights_note, "rights_note", spec.source_id)
        _required_string(spec.acquisition_note, "acquisition_note", spec.source_id)
        has_rights_reviewer = spec.rights_checked_by is not None
        has_rights_review_time = spec.rights_checked_at is not None
        if has_rights_reviewer != has_rights_review_time:
            raise ValueError(
                f"{spec.source_id} rights_checked_by and rights_checked_at must be supplied together"
            )
        if has_rights_reviewer:
            reviewer = _required_string(
                spec.rights_checked_by,
                "rights_checked_by",
                spec.source_id,
            )
            if not reviewer.startswith("human:") or reviewer == "human:":
                raise ValueError(f"{spec.source_id} rights_checked_by must be human:<id>")
            assert spec.rights_checked_at is not None
            if (
                not isinstance(spec.rights_checked_at, datetime)
                or spec.rights_checked_at.tzinfo is None
                or spec.rights_checked_at.utcoffset() is None
            ):
                raise ValueError(f"{spec.source_id} rights_checked_at must include a timezone")
            if spec.rights_checked_at.astimezone(UTC) > evaluation_time:
                raise ValueError(
                    f"{spec.source_id} rights_checked_at cannot be later than the catalogue "
                    "evaluation time"
                )
        if spec.rights_basis_url is not None:
            _validate_url(
                spec.rights_basis_url,
                field="rights_basis_url",
                source_id=spec.source_id,
                official_domains=spec.official_domains,
            )
        if spec.rights_status in {
            "cleared",
            "internal_only",
            "metadata_only",
            "permission_required",
        } and (spec.rights_basis_url is None):
            raise ValueError(f"{spec.source_id} rights_status requires a rights_basis_url")
        if spec.rights_status in {"cleared", "internal_only"} and not has_rights_reviewer:
            raise ValueError(
                f"{spec.source_id} storage-enabling rights status requires a documented human "
                "rights review"
            )


def validate_pilot_coverage(sources: Sequence[CommunicationSourceSpec]) -> None:
    """Require all institution classes and each commodity family in the pilot universe."""
    organization_types = {source.organization_type for source in sources}
    for required_type in sorted(ORGANIZATION_TYPES):
        if required_type not in organization_types:
            raise ValueError(
                f"communications pilot is missing {required_type.replace('_', ' ')} sources"
            )

    commodity_families = {
        family
        for source in sources
        if source.organization_type == "commodity_company"
        for family in source.commodity_families
    }
    missing_families = sorted(REQUIRED_COMMODITY_FAMILIES - commodity_families)
    if missing_families:
        raise ValueError(
            "communications pilot is missing commodity families: " + ", ".join(missing_families)
        )


def _communication_catalogue_sha256(
    sources: Sequence[CommunicationSourceSpec],
    *,
    schema_version: int,
    evaluated_at: datetime,
) -> str:
    validated_schema_version = _catalogue_schema_version(schema_version)
    validated_evaluated_at = _catalogue_evaluated_at(evaluated_at)
    validate_communication_sources(sources, evaluated_at=validated_evaluated_at)
    canonical_sources: list[dict[str, object]] = []
    for source in sorted(sources, key=lambda candidate: candidate.source_id):
        source_payload = asdict(source)
        for field in ("commodity_families", "material_types", "official_domains"):
            source_payload[field] = sorted(source_payload[field])
        checked_at = source.rights_checked_at
        source_payload["rights_checked_at"] = (
            checked_at.astimezone(UTC).isoformat().replace("+00:00", "Z")
            if checked_at is not None
            else None
        )
        canonical_sources.append(source_payload)
    payload = {
        "evaluated_at": validated_evaluated_at.isoformat().replace("+00:00", "Z"),
        "schema_version": validated_schema_version,
        "sources": canonical_sources,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def communication_catalogue_sha256(sources: Sequence[CommunicationSourceSpec]) -> str:
    """Return a current-vintage hash independent of source and tuple ordering."""
    return _communication_catalogue_sha256(
        sources,
        schema_version=CATALOGUE_SCHEMA_VERSION,
        evaluated_at=CATALOGUE_EVALUATED_AT,
    )


def communication_catalogue_snapshot(
    sources: Sequence[CommunicationSourceSpec],
    *,
    schema_version: int,
    evaluated_at: datetime,
) -> CommunicationCatalogueSnapshot:
    """Return a validated immutable catalogue snapshot with a bound semantic hash."""
    frozen_sources = tuple(sources)
    validated_schema_version = _catalogue_schema_version(schema_version)
    validated_evaluated_at = _catalogue_evaluated_at(evaluated_at)
    return CommunicationCatalogueSnapshot(
        catalogue_sha256=_communication_catalogue_sha256(
            frozen_sources,
            schema_version=validated_schema_version,
            evaluated_at=validated_evaluated_at,
        ),
        schema_version=validated_schema_version,
        evaluated_at=validated_evaluated_at,
        sources=frozen_sources,
    )


validate_communication_sources(COMMUNICATION_SOURCES)
validate_pilot_coverage(COMMUNICATION_SOURCES)
_FROZEN_2026_09_09_CATALOGUE_SHA256 = (
    "67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6"
)
_FROZEN_2026_09_09_CATALOGUE = communication_catalogue_snapshot(
    _COMMUNICATION_SOURCES_2026_09_09,
    schema_version=2,
    evaluated_at=datetime(2026, 9, 9, 7, 0, 0, tzinfo=UTC),
)
if _FROZEN_2026_09_09_CATALOGUE.catalogue_sha256 != _FROZEN_2026_09_09_CATALOGUE_SHA256:
    raise RuntimeError("the frozen 2026-09-09 communication catalogue snapshot changed")

_CURRENT_COMMUNICATION_CATALOGUE = communication_catalogue_snapshot(
    COMMUNICATION_SOURCES,
    schema_version=CATALOGUE_SCHEMA_VERSION,
    evaluated_at=CATALOGUE_EVALUATED_AT,
)
COMMUNICATION_CATALOGUE_SHA256 = _CURRENT_COMMUNICATION_CATALOGUE.catalogue_sha256
COMMUNICATION_CATALOGUE_SNAPSHOTS: Mapping[str, CommunicationCatalogueSnapshot] = MappingProxyType(
    {
        _FROZEN_2026_09_09_CATALOGUE.catalogue_sha256: _FROZEN_2026_09_09_CATALOGUE,
        _CURRENT_COMMUNICATION_CATALOGUE.catalogue_sha256: _CURRENT_COMMUNICATION_CATALOGUE,
    }
)


def resolve_communication_catalogue_snapshot(
    catalogue_sha256: str,
) -> CommunicationCatalogueSnapshot:
    """Resolve one exact policy vintage; unknown hashes never fall back to current policy."""
    if (
        not isinstance(catalogue_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", catalogue_sha256) is None
    ):
        raise ValueError("catalogue_sha256 must be a lowercase SHA-256")
    snapshot = COMMUNICATION_CATALOGUE_SNAPSHOTS.get(catalogue_sha256)
    if snapshot is None:
        raise ValueError("catalogue_sha256 must bind a known communication catalogue snapshot")
    if snapshot.catalogue_sha256 != catalogue_sha256:
        raise RuntimeError("communication catalogue snapshot registry key does not match its value")
    return snapshot


__all__ = [
    "ACQUISITION_STATUSES",
    "CATALOGUE_EVALUATED_AT",
    "CATALOGUE_SCHEMA_VERSION",
    "COMMUNICATION_CATALOGUE_SHA256",
    "COMMUNICATION_CATALOGUE_SNAPSHOTS",
    "COMMUNICATION_SOURCES",
    "MATERIAL_TYPES",
    "ORGANIZATION_TYPES",
    "PROVENANCE_TIERS",
    "REQUIRED_COMMODITY_FAMILIES",
    "RIGHTS_STATUSES",
    "TRANSCRIBER_ATTRIBUTIONS",
    "CommunicationCatalogueSnapshot",
    "CommunicationSourceSpec",
    "communication_catalogue_snapshot",
    "communication_catalogue_sha256",
    "resolve_communication_catalogue_snapshot",
    "validate_communication_sources",
    "validate_pilot_coverage",
]
