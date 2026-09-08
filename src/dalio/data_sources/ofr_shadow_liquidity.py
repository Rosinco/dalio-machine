"""First-party OFR shadow-liquidity histories and semantic safeguards.

The Office of Financial Research (OFR) Short-term Funding Monitor (STFM)
publishes useful evidence about two parts of the dollar shadow-money system:

* month-end assets held by U.S. money market mutual funds; and
* daily rates and volumes in selected cleared and tri-party repo venues.

These are related indicators, not pieces of one balance-sheet total.  The
catalogue therefore records economic side, measure kind, parent relationship,
and overlapping ``non_additive_groups``.  A downstream calculation must make
an explicit transformation instead of blindly adding a reported total to its
components, adding rates to volumes, or presenting selected venues as the
whole U.S. repo market.

The adapter requests complete native histories from OFR API v1.  It does not
ask the API to resample, aggregate, or remove nulls.  Missing values never
become zero.  For repo series, OFR's distinct ``disclosure_edits`` subseries is
validated and retained as DataFrame provenance; other absent periods remain
ambiguous because OFR says they can also reflect days with no trading.
"""

from __future__ import annotations

import calendar
import hashlib
import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Literal
from urllib.parse import urlencode

import pandas as pd
import requests

from dalio.data_sources.sdmx_csv import (
    DEFAULT_USER_AGENT,
    CachedTextFetcher,
    HttpClient,
    set_default_headers,
)

SOURCE_OFR_STFM = "OFR_STFM"
OFR_API_BASE_URL = "https://data.financialresearch.gov/v1/series/multifull"
OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX = "ofr-shadow-catalogue-sha256:"
DEFAULT_TIMEOUT = 60.0

_LONG_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]
_MMF_RELEASE_LONG_NAME = "OFR U.S. Money Market Fund Data Release"
_MMF_RELEASE_SHORT_NAME = "U.S. Money Market Funds"
_MMF_RELEASE_HREF = "/short-term-funding-monitor/datasets/mmf/"
_REPO_RELEASE_LONG_NAME = "OFR U.S. Repo Markets Data Release"
_REPO_RELEASE_SHORT_NAME = "U.S. Repo Markets"
_REPO_RELEASE_HREF = "/short-term-funding-monitor/datasets/repo/"

MMF_TOTAL_INSTRUMENT_GROUP = "mmf_total_with_instrument_components"
MMF_REPO_COUNTERPARTY_GROUP = "mmf_repo_total_with_counterparty_components"
MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP = "mmf_repo_exposures_and_repo_venue_quantities_overlap"
REPO_SELECTED_VENUES_GROUP = "repo_selected_venues_not_market_total"
REPO_RATE_GROUP = "repo_volume_weighted_rates_not_additive"
REPO_DVP_MEASURES_GROUP = "repo_dvp_unlike_measures"
REPO_GCF_MEASURES_GROUP = "repo_gcf_unlike_measures"
REPO_TRIPARTY_MEASURES_GROUP = "repo_triparty_ex_fed_unlike_measures"

EconomicSide = Literal["asset_side", "market_activity"]
MeasureKind = Literal["outstanding_stock", "transaction_volume", "volume_weighted_mean_rate"]
AggregationRole = Literal["reported_total", "component"]
CadencePolicy = Literal["complete_monthly", "sparse_monthly", "observed_business_days"]


@dataclass(frozen=True)
class OfrShadowLiquiditySeries:
    """Pinned native identity and interpretation for one OFR STFM series."""

    indicator: str
    country: str
    currency: str | None
    unit: str
    native_unit: str
    unit_multiplier: int
    frequency: Literal["monthly", "daily"]
    adjustment: str
    observation_basis: str
    publisher: str
    delivery_service: str
    source_family: str
    dataset: Literal["mmf", "repo"]
    native_series_id: str
    title: str
    native_description: str
    native_subtype: str
    native_subsetting: str
    native_vintage: str
    native_vintage_approach: str
    release_long_name: str
    release_short_name: str
    release_href: str
    expected_start: date
    minimum_observations: int
    max_latest_lag_days: int
    economic_side: EconomicSide
    claim_side: str
    from_sector: str
    to_sector: str
    instrument: str
    collateral_scope: str
    measure_kind: MeasureKind
    aggregation_role: AggregationRole
    parent_native_series_id: str | None
    component_axis: str | None
    venue: str | None
    non_additive_groups: tuple[str, ...]
    cadence_policy: CadencePolicy
    max_internal_gap_days: int | None
    minimum_weekday_coverage_ratio: float | None
    native_date_alignment_group: str | None
    missing_value_policy: str
    requires_disclosure_subseries: bool
    required_notes_phrases: tuple[str, ...] = ()
    research_role: Literal["primary", "diagnostic"] = "primary"

    @property
    def url(self) -> str:
        """Return an untransformed complete-history API URL for this series."""
        return build_ofr_multifull_url((self,))


def _mmf_spec(
    *,
    indicator: str,
    native_series_id: str,
    title: str,
    native_description: str,
    native_subsetting: str,
    expected_start: date,
    minimum_observations: int,
    aggregation_role: AggregationRole,
    parent_native_series_id: str | None,
    component_axis: str | None,
    non_additive_groups: tuple[str, ...],
    to_sector: str,
    instrument: str,
    collateral_scope: str,
    cadence_policy: Literal["complete_monthly", "sparse_monthly"] = "complete_monthly",
    research_role: Literal["primary", "diagnostic"] = "primary",
) -> OfrShadowLiquiditySeries:
    return OfrShadowLiquiditySeries(
        indicator=indicator,
        country="US",
        currency="USD",
        unit="USD",
        native_unit="USD",
        unit_multiplier=1,
        frequency="monthly",
        adjustment="not seasonally adjusted",
        observation_basis="month-end reported investment stock; native calendar date",
        publisher="Office of Financial Research, U.S. Department of the Treasury",
        delivery_service="OFR Short-term Funding Monitor API v1",
        source_family=SOURCE_OFR_STFM,
        dataset="mmf",
        native_series_id=native_series_id,
        title=title,
        native_description=native_description,
        native_subtype="Outstanding Volume",
        native_subsetting=native_subsetting,
        native_vintage="Monthly Revisions - Complete Series",
        native_vintage_approach="Monthly Revisions - Complete Series",
        release_long_name=_MMF_RELEASE_LONG_NAME,
        release_short_name=_MMF_RELEASE_SHORT_NAME,
        release_href=_MMF_RELEASE_HREF,
        expected_start=expected_start,
        minimum_observations=minimum_observations,
        max_latest_lag_days=100,
        economic_side="asset_side",
        claim_side="holder_asset",
        from_sector="us_money_market_mutual_funds",
        to_sector=to_sector,
        instrument=instrument,
        collateral_scope=collateral_scope,
        measure_kind="outstanding_stock",
        aggregation_role=aggregation_role,
        parent_native_series_id=parent_native_series_id,
        component_axis=component_axis,
        venue=None,
        non_additive_groups=non_additive_groups,
        cadence_policy=cadence_policy,
        max_internal_gap_days=None,
        minimum_weekday_coverage_ratio=None,
        native_date_alignment_group=None,
        missing_value_policy=(
            "OFR-null or omitted periods remain missing; never infer or impute zero"
        ),
        requires_disclosure_subseries=False,
        research_role=research_role,
    )


_REPO_MISSING_NOTES = (
    "missing values in this series represent observation periods in which either no "
    "trading took place",
    "disclosure edits were applied to protect business-confidential information",
)


def _repo_spec(
    *,
    indicator: str,
    native_series_id: str,
    title: str,
    native_description: str,
    native_subtype: str,
    expected_start: date,
    minimum_observations: int,
    measure_kind: MeasureKind,
    venue: str,
    venue_group: str,
) -> OfrShadowLiquiditySeries:
    is_rate = measure_kind == "volume_weighted_mean_rate"
    groups = [REPO_SELECTED_VENUES_GROUP, venue_group]
    if is_rate:
        groups.append(REPO_RATE_GROUP)
    else:
        groups.append(MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP)
    return OfrShadowLiquiditySeries(
        indicator=indicator,
        country="US",
        currency=None if is_rate else "USD",
        unit="percent" if is_rate else "USD",
        native_unit="Percent" if is_rate else "USD",
        unit_multiplier=1,
        frequency="daily",
        adjustment="not seasonally adjusted",
        observation_basis=(
            "single-day volume-weighted mean rate; native business date"
            if is_rate
            else "single-day reported repo volume; native business date"
        ),
        publisher="Office of Financial Research, U.S. Department of the Treasury",
        delivery_service="OFR Short-term Funding Monitor API v1",
        source_family=SOURCE_OFR_STFM,
        dataset="repo",
        native_series_id=native_series_id,
        title=title,
        native_description=native_description,
        native_subtype=native_subtype,
        native_subsetting="Total",
        native_vintage="Preliminary",
        native_vintage_approach="Preliminary",
        release_long_name=_REPO_RELEASE_LONG_NAME,
        release_short_name=_REPO_RELEASE_SHORT_NAME,
        release_href=_REPO_RELEASE_HREF,
        expected_start=expected_start,
        minimum_observations=minimum_observations,
        max_latest_lag_days=10,
        economic_side="market_activity",
        claim_side="market_activity_not_a_claim",
        from_sector="not_available",
        to_sector="not_available",
        instrument="repurchase_agreement",
        collateral_scope="all_reported_collateral",
        measure_kind=measure_kind,
        aggregation_role="reported_total",
        parent_native_series_id=None,
        component_axis=None,
        venue=venue,
        non_additive_groups=tuple(groups),
        cadence_policy="observed_business_days",
        max_internal_gap_days=(14 if venue == "tri_party_excluding_federal_reserve" else 7),
        minimum_weekday_coverage_ratio=(
            0.92 if venue == "tri_party_excluding_federal_reserve" else 0.93
        ),
        native_date_alignment_group=f"repo_native_dates:{venue}",
        missing_value_policy=(
            "Missing can mean no trading or confidentiality protection; preserve OFR "
            "disclosure-edit dates separately and never infer zero"
        ),
        requires_disclosure_subseries=True,
        required_notes_phrases=_REPO_MISSING_NOTES,
    )


OFR_MMF_TOTAL_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_total_investments_stock",
    native_series_id="MMF-MMF_TOT-M",
    title="Money Market Mutual Fund Investments: Total",
    native_description="Money market fund investments in all securities",
    native_subsetting="None",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="reported_total",
    parent_native_series_id=None,
    component_axis=None,
    non_additive_groups=(MMF_TOTAL_INSTRUMENT_GROUP,),
    to_sector="multiple_security_issuers_and_repo_counterparties",
    instrument="mixed_short_term_securities_and_repurchase_agreements",
    collateral_scope="mixed_not_available",
)

OFR_MMF_REPO_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_repo_investments_stock",
    native_series_id="MMF-MMF_RP_TOT-M",
    title="Money Market Mutual Fund Investments in Repurchase Agreements",
    native_description=("Money market funds' outstanding volume in all repurchase agreements"),
    native_subsetting="None",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_TOT-M",
    component_axis="instrument",
    non_additive_groups=(
        MMF_TOTAL_INSTRUMENT_GROUP,
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="multiple_repo_counterparties",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
)

OFR_MMF_TREASURY_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_us_treasury_investments_stock",
    native_series_id="MMF-MMF_T_TOT-M",
    title="Money Market Mutual Fund Investments in U.S. Treasury Securities",
    native_description="Money market fund investments in U.S. Treasury securities",
    native_subsetting="Assets",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_TOT-M",
    component_axis="instrument",
    non_additive_groups=(MMF_TOTAL_INSTRUMENT_GROUP,),
    to_sector="us_federal_government",
    instrument="us_treasury_security",
    collateral_scope="not_applicable",
)

OFR_MMF_AGENCY_GSE_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_agency_gse_investments_stock",
    native_series_id="MMF-MMF_AG_TOT-M",
    title="Money Market Mutual Fund Investments in Federal Agency and GSE Securities",
    native_description=("Money market fund investments in Federal Agency and GSE securities"),
    native_subsetting="Assets",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_TOT-M",
    component_axis="instrument",
    non_additive_groups=(MMF_TOTAL_INSTRUMENT_GROUP,),
    to_sector="us_federal_agencies_and_gses",
    instrument="federal_agency_and_gse_security",
    collateral_scope="not_applicable",
)

OFR_MMF_BANK_RELATED_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_bank_related_investments_stock",
    native_series_id="MMF-MMF_BRA_TOT-M",
    title="Money Market Mutual Fund Investments in Bank-Related Assets",
    native_description=(
        "Money market fund investments in bank-related assets, including financial "
        "commercial paper, certificates of deposit, and other deposits"
    ),
    native_subsetting="Assets",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_TOT-M",
    component_axis="instrument",
    non_additive_groups=(MMF_TOTAL_INSTRUMENT_GROUP,),
    to_sector="banks_and_other_financial_issuers",
    instrument="bank_related_short_term_asset",
    collateral_scope="not_applicable",
)

OFR_MMF_OTHER_ASSET_INVESTMENTS = _mmf_spec(
    indicator="shadow_mmf_other_asset_investments_stock",
    native_series_id="MMF-MMF_OA_TOT-M",
    title="Money Market Mutual Fund Investments in Other Assets",
    native_description=(
        "Money market fund investments in other assets, including municipal "
        "securities, commercial paper, asset-backed securities, and uncategorized "
        "assets"
    ),
    native_subsetting="Assets",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_TOT-M",
    component_axis="instrument",
    non_additive_groups=(MMF_TOTAL_INSTRUMENT_GROUP,),
    to_sector="municipal_and_other_security_issuers",
    instrument="municipal_commercial_paper_asset_backed_and_other_assets",
    collateral_scope="not_applicable",
)

OFR_MMF_REPO_WITH_FED = _mmf_spec(
    indicator="shadow_mmf_repo_federal_reserve_stock",
    native_series_id="MMF-MMF_RP_wFR-M",
    title=(
        "Money Market Mutual Fund Investments in Repurchase Agreements with the Federal Reserve"
    ),
    native_description=(
        "Money market funds' outstanding volume of repurchase agreements with the Federal Reserve"
    ),
    native_subsetting="Counterparty",
    expected_start=date(2011, 3, 31),
    minimum_observations=145,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_RP_TOT-M",
    component_axis="counterparty",
    non_additive_groups=(
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="federal_reserve",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
    cadence_policy="sparse_monthly",
    research_role="diagnostic",
)

OFR_MMF_REPO_CLEARED_FICC = _mmf_spec(
    indicator="shadow_mmf_repo_ficc_cleared_stock",
    native_series_id="MMF-MMF_RP_wFICC-M",
    title="Money Market Mutual Fund Investments in Repurchase Agreements Cleared by FICC",
    native_description=(
        "Money market funds' outstanding volume of repurchase agreements cleared by "
        "the Fixed Income Clearing Corporation"
    ),
    native_subsetting="Counterparty",
    expected_start=date(2014, 3, 31),
    minimum_observations=130,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_RP_TOT-M",
    component_axis="counterparty",
    non_additive_groups=(
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="ficc_cleared_repo_counterparties",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
    cadence_policy="sparse_monthly",
)

OFR_MMF_REPO_WITH_US_FINANCIALS = _mmf_spec(
    indicator="shadow_mmf_repo_us_financial_institutions_stock",
    native_series_id="MMF-MMF_RP_wDFI-M",
    title=(
        "Money Market Mutual Fund Investments in Repurchase Agreements with U.S. "
        "Financial Institutions"
    ),
    native_description=(
        "Money market funds' outstanding volume of repurchase agreements with U.S. "
        "financial institutions"
    ),
    native_subsetting="Counterparty",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_RP_TOT-M",
    component_axis="counterparty",
    non_additive_groups=(
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="us_financial_institutions",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
)

OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS = _mmf_spec(
    indicator="shadow_mmf_repo_foreign_financial_institutions_stock",
    native_series_id="MMF-MMF_RP_wFFI-M",
    title=(
        "Money Market Mutual Fund Investments in Repurchase Agreements with Foreign "
        "Financial Institutions"
    ),
    native_description=(
        "Money market funds' outstanding volume of repurchase agreements with foreign "
        "financial institutions"
    ),
    native_subsetting="Counterparty",
    expected_start=date(2010, 11, 30),
    minimum_observations=180,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_RP_TOT-M",
    component_axis="counterparty",
    non_additive_groups=(
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="foreign_financial_institutions",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
)

OFR_MMF_REPO_WITH_OTHER_COUNTERPARTIES = _mmf_spec(
    indicator="shadow_mmf_repo_other_counterparties_stock",
    native_series_id="MMF-MMF_RP_wOCP-M",
    title=(
        "Money Market Mutual Fund Investments in Repurchase Agreements with Other Counterparties"
    ),
    native_description=(
        "Money market funds' outstanding volume of repurchase agreements with other counterparties"
    ),
    native_subsetting="Counterparty",
    expected_start=date(2010, 12, 31),
    minimum_observations=100,
    aggregation_role="component",
    parent_native_series_id="MMF-MMF_RP_TOT-M",
    component_axis="counterparty",
    non_additive_groups=(
        MMF_REPO_COUNTERPARTY_GROUP,
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    ),
    to_sector="other_repo_counterparties",
    instrument="repurchase_agreement",
    collateral_scope="all_reported_collateral",
    cadence_policy="sparse_monthly",
)

OFR_REPO_DVP_AVERAGE_RATE = _repo_spec(
    indicator="repo_dvp_average_rate",
    native_series_id="REPO-DVP_AR_TOT-P",
    title="DVP Service Average Rate: Total (Preliminary)",
    native_description=(
        "Volume-weighted mean interest rate of all repurchase agreements starting on "
        "a given day in the Fixed Income Clearing Corporation's DVP Service"
    ),
    native_subtype="Interest Rate",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="volume_weighted_mean_rate",
    venue="ficc_dvp",
    venue_group=REPO_DVP_MEASURES_GROUP,
)

OFR_REPO_DVP_OUTSTANDING_VOLUME = _repo_spec(
    indicator="repo_dvp_outstanding_volume",
    native_series_id="REPO-DVP_OV_TOT-P",
    title="DVP Service Outstanding Volume: Total (Preliminary)",
    native_description=(
        "Outstanding volume of all repurchase agreements in the Fixed Income Clearing "
        "Corporation's DVP Service"
    ),
    native_subtype="Outstanding Volume",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="outstanding_stock",
    venue="ficc_dvp",
    venue_group=REPO_DVP_MEASURES_GROUP,
)

OFR_REPO_DVP_TRANSACTION_VOLUME = _repo_spec(
    indicator="repo_dvp_transaction_volume",
    native_series_id="REPO-DVP_TV_TOT-P",
    title="DVP Service Transaction Volume: Total (Preliminary)",
    native_description=(
        "Transaction volume of all repurchase agreements starting on a given day in "
        "the Fixed Income Clearing Corporation's DVP Service"
    ),
    native_subtype="Transaction Volume",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="transaction_volume",
    venue="ficc_dvp",
    venue_group=REPO_DVP_MEASURES_GROUP,
)

OFR_REPO_GCF_AVERAGE_RATE = _repo_spec(
    indicator="repo_gcf_average_rate",
    native_series_id="REPO-GCF_AR_TOT-P",
    title="GCF Repo Service Average Rate: Total (Preliminary)",
    native_description=(
        "Volume-weighted mean interest rate of all repurchase agreements starting on "
        "a given day in the Fixed Income Clearing Corporation's GCF Repo Service"
    ),
    native_subtype="Interest Rate",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="volume_weighted_mean_rate",
    venue="ficc_gcf",
    venue_group=REPO_GCF_MEASURES_GROUP,
)

OFR_REPO_GCF_OUTSTANDING_VOLUME = _repo_spec(
    indicator="repo_gcf_outstanding_volume",
    native_series_id="REPO-GCF_OV_TOT-P",
    title="GCF Repo Service Outstanding Volume: Total (Preliminary)",
    native_description=(
        "Outstanding volume of all repurchase agreements in the Fixed Income Clearing "
        "Corporation's GCF Repo Service"
    ),
    native_subtype="Outstanding Volume",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="outstanding_stock",
    venue="ficc_gcf",
    venue_group=REPO_GCF_MEASURES_GROUP,
)

OFR_REPO_GCF_TRANSACTION_VOLUME = _repo_spec(
    indicator="repo_gcf_transaction_volume",
    native_series_id="REPO-GCF_TV_TOT-P",
    title="GCF Repo Service Transaction Volume: Total (Preliminary)",
    native_description=(
        "Transaction volume of all repurchase agreements starting on a given day in "
        "the Fixed Income Clearing Corporation's GCF Repo Service"
    ),
    native_subtype="Transaction Volume",
    expected_start=date(2018, 5, 7),
    minimum_observations=2_000,
    measure_kind="transaction_volume",
    venue="ficc_gcf",
    venue_group=REPO_GCF_MEASURES_GROUP,
)

OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE = _repo_spec(
    indicator="repo_triparty_ex_fed_average_rate",
    native_series_id="REPO-TRIV1_AR_TOT-P",
    title="Tri-Party Average Rate, excluding Federal Reserve transactions: Total (Preliminary)",
    native_description=(
        "Volume-weighted mean interest rate of all repurchase agreements that were "
        "settled in tri-party repo, after excluding transactions with the Federal Reserve"
    ),
    native_subtype="Interest Rate",
    expected_start=date(2014, 8, 22),
    minimum_observations=2_900,
    measure_kind="volume_weighted_mean_rate",
    venue="tri_party_excluding_federal_reserve",
    venue_group=REPO_TRIPARTY_MEASURES_GROUP,
)

OFR_REPO_TRIPARTY_EX_FED_TRANSACTION_VOLUME = _repo_spec(
    indicator="repo_triparty_ex_fed_transaction_volume",
    native_series_id="REPO-TRIV1_TV_TOT-P",
    title=(
        "Tri-Party Transaction Volume, excluding Federal Reserve transactions: Total (Preliminary)"
    ),
    native_description=(
        "Transaction volume of all repurchase agreements starting on a given day that "
        "were settled in tri-party repo, after excluding transactions with the Federal "
        "Reserve"
    ),
    native_subtype="Transaction Volume",
    expected_start=date(2014, 8, 22),
    minimum_observations=2_900,
    measure_kind="transaction_volume",
    venue="tri_party_excluding_federal_reserve",
    venue_group=REPO_TRIPARTY_MEASURES_GROUP,
)


OFR_SHADOW_LIQUIDITY_SERIES: tuple[OfrShadowLiquiditySeries, ...] = (
    OFR_MMF_TOTAL_INVESTMENTS,
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_TREASURY_INVESTMENTS,
    OFR_MMF_AGENCY_GSE_INVESTMENTS,
    OFR_MMF_BANK_RELATED_INVESTMENTS,
    OFR_MMF_OTHER_ASSET_INVESTMENTS,
    OFR_MMF_REPO_WITH_FED,
    OFR_MMF_REPO_CLEARED_FICC,
    OFR_MMF_REPO_WITH_US_FINANCIALS,
    OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS,
    OFR_MMF_REPO_WITH_OTHER_COUNTERPARTIES,
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_REPO_DVP_OUTSTANDING_VOLUME,
    OFR_REPO_DVP_TRANSACTION_VOLUME,
    OFR_REPO_GCF_AVERAGE_RATE,
    OFR_REPO_GCF_OUTSTANDING_VOLUME,
    OFR_REPO_GCF_TRANSACTION_VOLUME,
    OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
    OFR_REPO_TRIPARTY_EX_FED_TRANSACTION_VOLUME,
)


def validate_ofr_shadow_catalogue(
    specs: Iterable[OfrShadowLiquiditySeries] = OFR_SHADOW_LIQUIDITY_SERIES,
) -> tuple[OfrShadowLiquiditySeries, ...]:
    """Validate identities, hierarchy, and non-additive semantic coverage."""
    selected = tuple(specs)
    if not selected:
        raise ValueError("OFR shadow-liquidity catalogue cannot be empty")
    ids = [spec.native_series_id for spec in selected]
    indicators = [spec.indicator for spec in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("OFR shadow-liquidity catalogue has duplicate native series IDs")
    if len(indicators) != len(set(indicators)):
        raise ValueError("OFR shadow-liquidity catalogue has duplicate indicators")

    by_id = {spec.native_series_id: spec for spec in selected}
    for spec in selected:
        if spec.aggregation_role == "reported_total":
            if spec.parent_native_series_id is not None or spec.component_axis is not None:
                raise ValueError(
                    f"reported total {spec.native_series_id} cannot declare a parent/component axis"
                )
        elif spec.aggregation_role == "component":
            if spec.parent_native_series_id is None or spec.component_axis is None:
                raise ValueError(
                    f"component {spec.native_series_id} must declare its parent and axis"
                )
            if spec.parent_native_series_id not in by_id:
                raise ValueError(
                    f"component {spec.native_series_id} has parent outside the catalogue: "
                    f"{spec.parent_native_series_id}"
                )
        else:  # pragma: no cover - Literal protects normal construction
            raise ValueError(f"unsupported aggregation role: {spec.aggregation_role!r}")
        if not spec.non_additive_groups:
            raise ValueError(
                f"{spec.native_series_id} must declare at least one non-additive group"
            )
        if spec.economic_side == "asset_side" and spec.dataset != "mmf":
            raise ValueError(f"asset-side series {spec.native_series_id} must use MMF data")
        if spec.economic_side == "market_activity" and spec.dataset != "repo":
            raise ValueError(f"market-activity series {spec.native_series_id} must use repo data")
        directional_fields = (
            spec.claim_side,
            spec.from_sector,
            spec.to_sector,
            spec.instrument,
            spec.collateral_scope,
        )
        if any(not isinstance(value, str) or not value.strip() for value in directional_fields):
            raise ValueError(f"{spec.native_series_id} has incomplete claim-direction semantics")
        if spec.economic_side == "asset_side":
            if (
                spec.claim_side != "holder_asset"
                or spec.from_sector != "us_money_market_mutual_funds"
            ):
                raise ValueError(
                    f"asset-side series {spec.native_series_id} has invalid claim direction"
                )
        elif (
            spec.claim_side != "market_activity_not_a_claim"
            or spec.from_sector != "not_available"
            or spec.to_sector != "not_available"
        ):
            raise ValueError(
                f"market-activity series {spec.native_series_id} must not invent sectors"
            )
        expected_cadences = {
            "mmf": {"complete_monthly", "sparse_monthly"},
            "repo": {"observed_business_days"},
        }
        if spec.cadence_policy not in expected_cadences[spec.dataset]:
            raise ValueError(
                f"{spec.native_series_id} has cadence policy incompatible with {spec.dataset}"
            )
        if spec.dataset == "mmf":
            if any(
                value is not None
                for value in (
                    spec.max_internal_gap_days,
                    spec.minimum_weekday_coverage_ratio,
                    spec.native_date_alignment_group,
                )
            ):
                raise ValueError(
                    f"monthly MMF series {spec.native_series_id} cannot declare daily guards"
                )
        elif (
            not isinstance(spec.max_internal_gap_days, int)
            or spec.max_internal_gap_days < 1
            or not isinstance(spec.minimum_weekday_coverage_ratio, int | float)
            or not 0 < float(spec.minimum_weekday_coverage_ratio) <= 1
            or not isinstance(spec.native_date_alignment_group, str)
            or not spec.native_date_alignment_group.strip()
        ):
            raise ValueError(
                f"daily repo series {spec.native_series_id} has incomplete density guards"
            )
        if spec.measure_kind == "volume_weighted_mean_rate":
            if spec.currency is not None or spec.unit != "percent":
                raise ValueError(f"rate series {spec.native_series_id} has invalid unit semantics")
        elif spec.currency != "USD" or spec.unit != "USD":
            raise ValueError(f"volume series {spec.native_series_id} has invalid unit semantics")

    group_counts = Counter(group for spec in selected for group in spec.non_additive_groups)
    singleton_groups = sorted(group for group, count in group_counts.items() if count < 2)
    if singleton_groups:
        raise ValueError(
            f"OFR non-additive groups must identify an actual collision: {singleton_groups!r}"
        )
    return selected


def overlapping_non_additive_groups(
    specs: Iterable[OfrShadowLiquiditySeries],
) -> tuple[str, ...]:
    """Return semantic groups that make a raw multi-series sum invalid."""
    counts = Counter(group for spec in specs for group in spec.non_additive_groups)
    return tuple(sorted(group for group, count in counts.items() if count > 1))


def assert_raw_sum_is_semantically_valid(
    specs: Iterable[OfrShadowLiquiditySeries],
) -> None:
    """Fail closed when selected raw OFR series share a non-additive group."""
    selected = tuple(specs)
    signatures = {(spec.currency, spec.unit, spec.measure_kind) for spec in selected}
    if len(signatures) > 1:
        raise ValueError(
            "raw OFR shadow-liquidity series cannot be summed across incompatible "
            "units or measure kinds"
        )
    conflicts = overlapping_non_additive_groups(selected)
    if conflicts:
        raise ValueError(
            "raw OFR shadow-liquidity series cannot be summed across non-additive "
            f"groups: {', '.join(conflicts)}"
        )


def build_ofr_multifull_url(specs: Iterable[OfrShadowLiquiditySeries]) -> str:
    """Build the exact native-history call; no server-side transformation flags."""
    selected = tuple(specs)
    if not selected:
        raise ValueError("at least one OFR series is required")
    ids = [spec.native_series_id for spec in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("OFR multifull request contains duplicate native series IDs")
    return f"{OFR_API_BASE_URL}?{urlencode({'mnemonics': ','.join(ids)}, safe=',')}"


def ofr_shadow_liquidity_catalogue_sha256(
    specs: Iterable[OfrShadowLiquiditySeries] = OFR_SHADOW_LIQUIDITY_SERIES,
) -> str:
    """Hash the complete pinned source and interpretation catalogue."""
    records: list[dict[str, object]] = []
    for spec in sorted(specs, key=lambda item: item.native_series_id):
        record = asdict(spec)
        record["expected_start"] = spec.expected_start.isoformat()
        records.append(record)
    payload = json.dumps(
        records,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OfrShadowLiquiditySource:
    """Fetch complete, unresampled histories from the official OFR STFM API."""

    def __init__(
        self,
        client: HttpClient | None = None,
        cache_dir: Path | None = None,
        cache_ttl_hours: float = 24.0,
        user_agent: str = DEFAULT_USER_AGENT,
        artifact_dir: Path | None = None,
    ) -> None:
        self._client = client or requests.Session()
        set_default_headers(self._client, user_agent, "application/json")
        resolved_cache = (
            cache_dir
            if cache_dir is not None
            else Path(
                os.environ.get(
                    "DALIO_SHADOW_LIQUIDITY_CACHE",
                    "data/cache/shadow_liquidity",
                )
            )
            / "ofr"
        )
        self._fetcher = CachedTextFetcher(
            self._client,
            resolved_cache,
            cache_ttl_hours,
            label="OFR Short-term Funding Monitor",
            timeout=DEFAULT_TIMEOUT,
            suffix=".json",
            forbidden_hint="official OFR public API refused the request",
        )
        self._artifact_dir = (
            artifact_dir
            if artifact_dir is not None
            else Path(
                os.environ.get(
                    "DALIO_SHADOW_LIQUIDITY_ARTIFACTS",
                    "data/artifacts/liquidity_frontier",
                )
            )
            / "ofr"
        )

    def fetch(
        self,
        spec: OfrShadowLiquiditySeries,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Fetch one series while using the same validated multifull contract."""
        return self.fetch_many((spec,), use_cache=use_cache)[spec.native_series_id]

    def fetch_many(
        self,
        specs: Iterable[OfrShadowLiquiditySeries] = OFR_SHADOW_LIQUIDITY_SERIES,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        selected = tuple(specs)
        url = build_ofr_multifull_url(selected)
        text = self._fetcher.fetch(url, use_cache=use_cache)
        frames = parse_ofr_multifull_json(text, selected)
        # Archive only after *every* requested series has passed the exact
        # schema, metadata, history, and value checks above.
        artifact_path, artifact_sha256 = _archive_validated_response(
            text,
            self._artifact_dir,
        )
        payload = json.loads(text)
        for spec in selected:
            frame = frames[spec.native_series_id]
            native_payload = _canonical_json_bytes(payload[spec.native_series_id])
            native_path, native_sha256 = _archive_canonical_bytes(
                native_payload,
                self._artifact_dir / "native-series",
            )
            missing_payload = {
                "native_series_id": spec.native_series_id,
                "cadence_policy": spec.cadence_policy,
                "missing_value_policy": spec.missing_value_policy,
                "records": frame.attrs["missing_period_records"],
            }
            missing_bytes = _canonical_json_bytes(missing_payload)
            missing_path, missing_sha256 = _archive_canonical_bytes(
                missing_bytes,
                self._artifact_dir / "missingness-ledgers",
            )
            if native_sha256 != frame.attrs["native_payload_sha256"]:
                raise ValueError(
                    f"OFR native payload archive hash mismatch for {spec.native_series_id}"
                )
            if missing_sha256 != frame.attrs["missing_provenance_sha256"]:
                raise ValueError(
                    f"OFR missingness archive hash mismatch for {spec.native_series_id}"
                )
            frame.attrs["source_url"] = url
            frame.attrs["source_artifact_path"] = str(artifact_path)
            frame.attrs["source_artifact_sha256"] = artifact_sha256
            frame.attrs["native_payload_artifact_path"] = str(native_path)
            frame.attrs["missing_provenance_artifact_path"] = str(missing_path)
        return frames


def _archive_validated_response(text: str, artifact_dir: Path) -> tuple[Path, str]:
    """Atomically retain the exact UTF-8 JSON bytes under their SHA-256."""
    payload = text.encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_dir / digest[:2] / f"{digest}.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"OFR artifact hash collision or corrupt archive: {destination}")
        return destination, digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination, digest


def _archive_canonical_bytes(payload: bytes, artifact_dir: Path) -> tuple[Path, str]:
    """Retain a validated derived JSON document under its exact byte hash."""
    digest = hashlib.sha256(payload).hexdigest()
    destination = artifact_dir / digest[:2] / f"{digest}.json"
    if destination.exists():
        if destination.read_bytes() != payload:
            raise ValueError(f"OFR artifact hash collision or corrupt archive: {destination}")
        return destination, digest

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{digest}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination, digest


def parse_ofr_multifull_json(
    text: str,
    specs: Iterable[OfrShadowLiquiditySeries],
) -> dict[str, pd.DataFrame]:
    """Parse an exact OFR ``multifull`` response into native-series frames."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("OFR returned an empty response body")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("OFR returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("OFR multifull response must be a JSON object")

    selected = tuple(specs)
    if not selected:
        raise ValueError("at least one OFR series is required")
    expected_ids = [spec.native_series_id for spec in selected]
    if len(expected_ids) != len(set(expected_ids)):
        raise ValueError("OFR parse request contains duplicate native series IDs")
    actual_ids = set(payload)
    if actual_ids != set(expected_ids):
        missing = sorted(set(expected_ids) - actual_ids)
        extra = sorted(actual_ids - set(expected_ids))
        raise ValueError(
            f"OFR multifull series identity mismatch; missing={missing!r}, extra={extra!r}"
        )

    frames = {
        spec.native_series_id: _parse_ofr_series(payload[spec.native_series_id], spec)
        for spec in selected
    }
    _validate_native_date_alignment(frames, selected)
    return frames


def _validate_native_date_alignment(
    frames: Mapping[str, pd.DataFrame],
    specs: Sequence[OfrShadowLiquiditySeries],
) -> None:
    """Require unlike measures from one venue to cover identical native dates."""
    by_group: dict[str, list[OfrShadowLiquiditySeries]] = {}
    for spec in specs:
        if spec.native_date_alignment_group is not None:
            by_group.setdefault(spec.native_date_alignment_group, []).append(spec)
    for group, members in by_group.items():
        if len(members) < 2:
            continue
        reference = tuple(frames[members[0].native_series_id].attrs["native_periods"])
        mismatches = [
            spec.native_series_id
            for spec in members[1:]
            if tuple(frames[spec.native_series_id].attrs["native_periods"]) != reference
        ]
        if mismatches:
            raise ValueError(
                f"OFR same-venue native-date alignment mismatch for {group}: "
                f"{[members[0].native_series_id, *mismatches]!r}"
            )


def _parse_ofr_series(
    raw: object,
    spec: OfrShadowLiquiditySeries,
) -> pd.DataFrame:
    if not isinstance(raw, dict):
        raise ValueError(f"OFR series {spec.native_series_id} must be a JSON object")
    try:
        timeseries = raw["timeseries"]
        metadata = raw["metadata"]
    except KeyError as exc:
        raise ValueError(f"OFR series {spec.native_series_id} is missing data or metadata") from exc
    if not isinstance(timeseries, dict) or not isinstance(metadata, dict):
        raise ValueError(f"OFR series {spec.native_series_id} has malformed data or metadata")

    _validate_metadata(metadata, spec)
    allowed_subseries = {"aggregation", "disclosure_edits"}
    unexpected = set(timeseries) - allowed_subseries
    if unexpected:
        raise ValueError(
            f"OFR series {spec.native_series_id} has unexpected subseries: {sorted(unexpected)!r}"
        )
    if "aggregation" not in timeseries:
        raise ValueError(f"OFR series {spec.native_series_id} lacks aggregation data")
    if spec.requires_disclosure_subseries and "disclosure_edits" not in timeseries:
        raise ValueError(f"OFR repo series {spec.native_series_id} lacks disclosure_edits data")

    aggregation = _parse_aggregation_pairs(timeseries["aggregation"], spec)
    disclosure_dates = _parse_disclosure_pairs(timeseries.get("disclosure_edits", []), spec)
    observed_by_date = {observed_on: value for observed_on, value in aggregation}
    conflicts = sorted(
        observed_on
        for observed_on in disclosure_dates
        if observed_by_date.get(observed_on) is not None
    )
    if conflicts:
        raise ValueError(
            f"OFR series {spec.native_series_id} marks observed values as disclosure edits: "
            f"{[value.isoformat() for value in conflicts]!r}"
        )

    native_dates = [observed_on for observed_on, _value in aggregation]
    all_native_dates = sorted(set(native_dates) | set(disclosure_dates))
    if not all_native_dates:
        raise ValueError(f"OFR series {spec.native_series_id} has no native periods")
    if all_native_dates[0] != spec.expected_start:
        raise ValueError(
            f"OFR series {spec.native_series_id} expected history start "
            f"{spec.expected_start.isoformat()}, got {all_native_dates[0].isoformat()}"
        )
    _validate_cadence(all_native_dates, spec)

    observed = [(observed_on, value) for observed_on, value in aggregation if value is not None]
    if len(observed) < spec.minimum_observations:
        raise ValueError(
            f"OFR series {spec.native_series_id} has {len(observed)} observations; "
            f"minimum is {spec.minimum_observations}"
        )

    last_updated_at = _parse_last_update(metadata, spec)
    if last_updated_at.date() < all_native_dates[-1]:
        raise ValueError(
            f"OFR series {spec.native_series_id} last-update precedes its latest period"
        )

    output = pd.DataFrame(
        {
            "country": spec.country,
            "indicator": spec.indicator,
            "date": [observed_on for observed_on, _value in observed],
            "value": [float(value) for _observed_on, value in observed],
            "source": spec.source_family,
            "series_id": spec.native_series_id,
        },
        columns=_LONG_COLUMNS,
    )
    if spec.native_vintage == "Preliminary":
        # Generic release ingestion otherwise defaults a missing status to
        # ``observed``. OFR's -P histories are explicitly preliminary.
        output["status"] = "preliminary"
    null_dates = {observed_on for observed_on, value in aggregation if value is None}
    missing_records = _missing_period_records(
        spec,
        all_native_dates=all_native_dates,
        null_dates=null_dates,
        disclosure_dates=set(disclosure_dates),
    )
    missing_payload = {
        "native_series_id": spec.native_series_id,
        "cadence_policy": spec.cadence_policy,
        "missing_value_policy": spec.missing_value_policy,
        "records": missing_records,
    }
    output.attrs.update(
        {
            # Disclosure dates can be absent from ``aggregation`` and therefore
            # belong in the canonical explicit native-period ledger.
            "native_periods": tuple(observed_on.isoformat() for observed_on in all_native_dates),
            "native_aggregation_periods": tuple(
                observed_on.isoformat() for observed_on in native_dates
            ),
            "native_period_format": "YYYY-MM-DD",
            "publisher_last_updated_at": last_updated_at,
            "native_vintage": spec.native_vintage,
            "native_dataset": spec.dataset,
            "release_href": spec.release_href,
            "disclosure_edit_dates": tuple(sorted(disclosure_dates)),
            "unclassified_null_dates": tuple(sorted(null_dates - set(disclosure_dates))),
            "missing_period_records": missing_records,
            "missing_provenance_sha256": _canonical_json_sha256(missing_payload),
            "native_payload_sha256": _canonical_json_sha256(raw),
            "missing_value_policy": spec.missing_value_policy,
            "non_additive_groups": spec.non_additive_groups,
            "cadence_policy": spec.cadence_policy,
        }
    )
    return output


def _validate_cadence(
    native_dates: Sequence[date],
    spec: OfrShadowLiquiditySeries,
) -> None:
    """Apply the series-specific completeness promise, never a blanket fill."""
    if spec.cadence_policy == "complete_monthly":
        ordinals = [observed_on.year * 12 + observed_on.month for observed_on in native_dates]
        if any(right - left != 1 for left, right in zip(ordinals, ordinals[1:], strict=False)):
            raise ValueError(
                f"OFR complete-monthly series {spec.native_series_id} has a cadence gap"
            )
    elif spec.cadence_policy == "sparse_monthly":
        # Sparse counterparty histories legitimately omit months. Absence is
        # retained below as provenance and is never interpreted as a zero.
        return
    elif spec.cadence_policy == "observed_business_days":
        # Holidays, no-trading days, and confidentiality edits make a complete
        # weekday sequence invalid as an expectation. Bounded gaps and density
        # still reject a silently truncated first snapshot.
        assert spec.max_internal_gap_days is not None
        assert spec.minimum_weekday_coverage_ratio is not None
        gaps = [
            (right - left).days for left, right in zip(native_dates, native_dates[1:], strict=False)
        ]
        if gaps and max(gaps) > spec.max_internal_gap_days:
            raise ValueError(
                f"OFR daily series {spec.native_series_id} has an internal gap of "
                f"{max(gaps)} days; maximum is {spec.max_internal_gap_days}"
            )
        expected_weekdays = sum(
            1
            for offset in range((native_dates[-1] - native_dates[0]).days + 1)
            if (native_dates[0] + timedelta(days=offset)).weekday() < 5
        )
        coverage = len(native_dates) / expected_weekdays
        if coverage < spec.minimum_weekday_coverage_ratio:
            raise ValueError(
                f"OFR daily series {spec.native_series_id} weekday coverage is "
                f"{coverage:.3f}; minimum is {spec.minimum_weekday_coverage_ratio:.3f}"
            )
    else:  # pragma: no cover - catalogue validation rejects this first
        raise ValueError(
            f"unsupported OFR cadence policy for {spec.native_series_id}: {spec.cadence_policy!r}"
        )


def _missing_period_records(
    spec: OfrShadowLiquiditySeries,
    *,
    all_native_dates: Sequence[date],
    null_dates: set[date],
    disclosure_dates: set[date],
) -> tuple[dict[str, str], ...]:
    """Build a canonical, non-imputing ledger of knowable missing provenance."""
    records: list[dict[str, str]] = []
    for observed_on in sorted(disclosure_dates):
        records.append(
            {
                "date": observed_on.isoformat(),
                "reason": "disclosure_edit",
                "evidence": "ofr_disclosure_edits_subseries",
            }
        )
    for observed_on in sorted(null_dates - disclosure_dates):
        records.append(
            {
                "date": observed_on.isoformat(),
                "reason": "publisher_null_not_zero",
                "evidence": "ofr_aggregation_null",
            }
        )

    if spec.cadence_policy == "sparse_monthly" and all_native_dates:
        explicit = set(all_native_dates)
        first = all_native_dates[0]
        last = all_native_dates[-1]
        for ordinal in range(
            first.year * 12 + first.month,
            last.year * 12 + last.month + 1,
        ):
            year, zero_based_month = divmod(ordinal - 1, 12)
            month = zero_based_month + 1
            month_end = date(year, month, calendar.monthrange(year, month)[1])
            if month_end not in explicit:
                records.append(
                    {
                        "date": month_end.isoformat(),
                        "reason": "publisher_omitted_month_not_zero",
                        "evidence": "derived_sparse_monthly_gap",
                    }
                )
    return tuple(sorted(records, key=lambda item: (item["date"], item["reason"])))


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("OFR payload cannot be represented as canonical JSON") from exc


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _validate_metadata(
    metadata: Mapping[str, object],
    spec: OfrShadowLiquiditySeries,
) -> None:
    try:
        mnemonic = metadata["mnemonic"]
        description = metadata["description"]
        schedule = metadata["schedule"]
        release = metadata["release"]
        unit = metadata["unit"]
        parents = metadata["parents"]
    except KeyError as exc:
        raise ValueError(
            f"OFR series {spec.native_series_id} is missing required metadata"
        ) from exc
    if not all(isinstance(value, dict) for value in (description, schedule, release, unit)):
        raise ValueError(f"OFR series {spec.native_series_id} has malformed metadata")
    if mnemonic != spec.native_series_id:
        raise ValueError(
            f"OFR native series mismatch: expected {spec.native_series_id!r}, got {mnemonic!r}"
        )

    expected_description = {
        "name": spec.title,
        "description": spec.native_description,
        "subtype": spec.native_subtype,
        "subsetting": spec.native_subsetting,
        "vintage": spec.native_vintage,
        "vintage_approach": spec.native_vintage_approach,
    }
    for field, wanted in expected_description.items():
        actual = description.get(field)
        if _normalized_text(actual) != _normalized_text(wanted):
            raise ValueError(
                f"OFR {field.replace('_', ' ')} mismatch for {spec.native_series_id}: {actual!r}"
            )
    notes = _normalized_text(description.get("notes", ""))
    for phrase in spec.required_notes_phrases:
        if _normalized_text(phrase) not in notes:
            raise ValueError(f"OFR missing-value notes mismatch for {spec.native_series_id}")

    expected_schedule = {
        "observation_period": "Single Day",
        "seasonal_adjustment": "None",
        "observation_frequency": "Monthly" if spec.frequency == "monthly" else "Daily",
        "start_date": spec.expected_start.isoformat(),
    }
    for field, wanted in expected_schedule.items():
        actual = schedule.get(field)
        if _normalized_text(actual) != _normalized_text(wanted):
            label = "history start" if field == "start_date" else field.replace("_", " ")
            raise ValueError(f"OFR {label} mismatch for {spec.native_series_id}: {actual!r}")

    expected_release = {
        "long_name": spec.release_long_name,
        "short_name": spec.release_short_name,
        "href": spec.release_href,
        "frequency": "Monthly" if spec.frequency == "monthly" else "Daily",
    }
    for field, wanted in expected_release.items():
        actual = release.get(field)
        if _normalized_text(actual) != _normalized_text(wanted):
            raise ValueError(
                f"OFR release {field.replace('_', ' ')} mismatch for "
                f"{spec.native_series_id}: {actual!r}"
            )

    expected_unit_type = "Rate" if spec.measure_kind == "volume_weighted_mean_rate" else "Volume"
    expected_unit = {
        "name": spec.native_unit,
        "type": expected_unit_type,
        "magnitude": 0,
        "display_magnitude": 0,
        "precision": 2,
    }
    for field, wanted in expected_unit.items():
        actual = unit.get(field)
        if actual != wanted:
            raise ValueError(
                f"OFR unit metadata mismatch for {spec.native_series_id}: {field}={actual!r}"
            )

    if not isinstance(parents, list) or any(not isinstance(value, str) for value in parents):
        raise ValueError(f"OFR parent metadata is malformed for {spec.native_series_id}")
    expected_parents = (
        [] if spec.parent_native_series_id is None else [spec.parent_native_series_id]
    )
    if parents != expected_parents:
        raise ValueError(
            f"OFR parent relationship mismatch for {spec.native_series_id}: {parents!r}"
        )

    # Parse here as well as after observations so malformed timestamps fail even
    # if another validation exits early in a future refactor.
    _parse_last_update(metadata, spec)


def _parse_last_update(
    metadata: Mapping[str, object],
    spec: OfrShadowLiquiditySeries,
) -> datetime:
    try:
        raw = metadata["schedule"]["last_update"]  # type: ignore[index]
        parsed = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"OFR last-update timestamp is invalid for {spec.native_series_id}"
        ) from exc
    # OFR's series page labels this timestamp UTC; the API returns no offset.
    return parsed.replace(tzinfo=UTC)


def _parse_aggregation_pairs(
    raw: object,
    spec: OfrShadowLiquiditySeries,
) -> list[tuple[date, float | None]]:
    if not isinstance(raw, list):
        raise ValueError(f"OFR aggregation data is malformed for {spec.native_series_id}")
    parsed: list[tuple[date, float | None]] = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"OFR aggregation entry is malformed for {spec.native_series_id}")
        observed_on = _parse_native_date(pair[0], spec)
        value = pair[1]
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(
                    f"OFR returned a non-numeric observation for {spec.native_series_id}: {value!r}"
                )
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(
                    f"OFR returned a non-finite observation for {spec.native_series_id}"
                )
            if spec.measure_kind != "volume_weighted_mean_rate" and value < 0:
                raise ValueError(f"OFR returned a negative volume for {spec.native_series_id}")
        parsed.append((observed_on, value))
    _validate_strict_date_order(
        [observed_on for observed_on, _value in parsed], spec, "aggregation"
    )
    return parsed


def _parse_disclosure_pairs(
    raw: object,
    spec: OfrShadowLiquiditySeries,
) -> tuple[date, ...]:
    if not isinstance(raw, list):
        raise ValueError(f"OFR disclosure_edits data is malformed for {spec.native_series_id}")
    parsed: list[date] = []
    for pair in raw:
        if not isinstance(pair, list) or len(pair) != 2 or pair[1] is not None:
            raise ValueError(
                f"OFR disclosure edit must be a date/null pair for {spec.native_series_id}"
            )
        parsed.append(_parse_native_date(pair[0], spec))
    _validate_strict_date_order(parsed, spec, "disclosure_edits")
    return tuple(parsed)


def _parse_native_date(raw: object, spec: OfrShadowLiquiditySeries) -> date:
    if not isinstance(raw, str):
        raise ValueError(f"OFR returned an invalid observation date for {spec.native_series_id}")
    try:
        observed_on = date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"OFR returned an invalid observation date for {spec.native_series_id}: {raw!r}"
        ) from exc
    if observed_on.isoformat() != raw:
        raise ValueError(
            f"OFR returned a non-canonical observation date for {spec.native_series_id}: {raw!r}"
        )
    if spec.frequency == "monthly":
        month_end = calendar.monthrange(observed_on.year, observed_on.month)[1]
        if observed_on.day != month_end:
            raise ValueError(
                f"OFR monthly series {spec.native_series_id} date is not calendar month end"
            )
    elif observed_on.weekday() >= 5:
        raise ValueError(f"OFR daily repo series {spec.native_series_id} contains a weekend date")
    return observed_on


def _validate_strict_date_order(
    dates: Sequence[date],
    spec: OfrShadowLiquiditySeries,
    label: str,
) -> None:
    if len(dates) != len(set(dates)):
        raise ValueError(f"OFR {label} has duplicate dates for {spec.native_series_id}")
    if list(dates) != sorted(dates):
        raise ValueError(
            f"OFR {label} dates are not strictly increasing for {spec.native_series_id}"
        )


def _normalized_text(value: object) -> str:
    return " ".join(str(value).strip().casefold().split())


# Fail during import if a future edit makes the shipped catalogue internally
# contradictory.  Parsing subsets remains supported by ``fetch``/``fetch_many``.
validate_ofr_shadow_catalogue()
