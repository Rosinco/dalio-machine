"""OFR Short-term Funding Monitor shadow-liquidity adapter tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dalio.data_sources.ofr_shadow_liquidity import (
    MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP,
    MMF_REPO_COUNTERPARTY_GROUP,
    MMF_TOTAL_INSTRUMENT_GROUP,
    OFR_MMF_AGENCY_GSE_INVESTMENTS,
    OFR_MMF_BANK_RELATED_INVESTMENTS,
    OFR_MMF_OTHER_ASSET_INVESTMENTS,
    OFR_MMF_REPO_CLEARED_FICC,
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_REPO_WITH_FED,
    OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS,
    OFR_MMF_REPO_WITH_OTHER_COUNTERPARTIES,
    OFR_MMF_REPO_WITH_US_FINANCIALS,
    OFR_MMF_TOTAL_INVESTMENTS,
    OFR_MMF_TREASURY_INVESTMENTS,
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_REPO_DVP_OUTSTANDING_VOLUME,
    OFR_REPO_DVP_TRANSACTION_VOLUME,
    OFR_REPO_GCF_AVERAGE_RATE,
    OFR_REPO_GCF_OUTSTANDING_VOLUME,
    OFR_REPO_GCF_TRANSACTION_VOLUME,
    OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
    OFR_REPO_TRIPARTY_EX_FED_TRANSACTION_VOLUME,
    OFR_SHADOW_LIQUIDITY_SERIES,
    REPO_SELECTED_VENUES_GROUP,
    OfrShadowLiquiditySource,
    assert_raw_sum_is_semantically_valid,
    build_ofr_multifull_url,
    ofr_shadow_liquidity_catalogue_sha256,
    overlapping_non_additive_groups,
    parse_ofr_multifull_json,
    validate_ofr_shadow_catalogue,
)


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _small(spec, *, expected_start: date):
    return replace(spec, expected_start=expected_start, minimum_observations=1)


def _entry(spec, points, *, disclosure_edits=None, last_update="2026-09-04 14:00:00"):
    notes = ""
    if spec.required_notes_phrases:
        notes = (
            "Missing values in this series represent observation periods in which "
            "either no trading took place or in which disclosure edits were applied "
            "to protect business-confidential information."
        )
    timeseries = {"aggregation": points}
    if spec.requires_disclosure_subseries:
        timeseries["disclosure_edits"] = disclosure_edits or []
    return {
        "timeseries": timeseries,
        "metadata": {
            "mnemonic": spec.native_series_id,
            "description": {
                "vintage_approach": spec.native_vintage_approach,
                "vintage": spec.native_vintage,
                "notes": notes,
                "name": spec.title,
                "subsetting": spec.native_subsetting,
                "subtype": spec.native_subtype,
                "description": spec.native_description,
            },
            "schedule": {
                "observation_period": "Single Day",
                "seasonal_adjustment": "None",
                "observation_frequency": ("Monthly" if spec.frequency == "monthly" else "Daily"),
                "start_date": spec.expected_start.isoformat(),
                "last_update": last_update,
            },
            "rights": {"description": ""},
            "parents": (
                [] if spec.parent_native_series_id is None else [spec.parent_native_series_id]
            ),
            "release": {
                "long_name": spec.release_long_name,
                "href": spec.release_href,
                "frequency": "Monthly" if spec.frequency == "monthly" else "Daily",
                "short_name": spec.release_short_name,
            },
            "children": [],
            "unit": {
                "display_magnitude": 0,
                "magnitude": 0,
                "type": ("Rate" if spec.measure_kind == "volume_weighted_mean_rate" else "Volume"),
                "name": spec.native_unit,
                "precision": 2,
            },
        },
    }


def _text(spec, points, **kwargs) -> str:
    return json.dumps({spec.native_series_id: _entry(spec, points, **kwargs)})


def test_catalogue_pins_selected_official_mmf_and_repo_series():
    assert OFR_SHADOW_LIQUIDITY_SERIES == (
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
    assert {spec.native_series_id for spec in OFR_SHADOW_LIQUIDITY_SERIES} == {
        "MMF-MMF_TOT-M",
        "MMF-MMF_RP_TOT-M",
        "MMF-MMF_T_TOT-M",
        "MMF-MMF_AG_TOT-M",
        "MMF-MMF_BRA_TOT-M",
        "MMF-MMF_OA_TOT-M",
        "MMF-MMF_RP_wFR-M",
        "MMF-MMF_RP_wFICC-M",
        "MMF-MMF_RP_wDFI-M",
        "MMF-MMF_RP_wFFI-M",
        "MMF-MMF_RP_wOCP-M",
        "REPO-DVP_AR_TOT-P",
        "REPO-DVP_OV_TOT-P",
        "REPO-DVP_TV_TOT-P",
        "REPO-GCF_AR_TOT-P",
        "REPO-GCF_OV_TOT-P",
        "REPO-GCF_TV_TOT-P",
        "REPO-TRIV1_AR_TOT-P",
        "REPO-TRIV1_TV_TOT-P",
    }
    assert len({spec.indicator for spec in OFR_SHADOW_LIQUIDITY_SERIES}) == 19
    assert validate_ofr_shadow_catalogue() == OFR_SHADOW_LIQUIDITY_SERIES


def test_catalogue_encodes_side_measure_parent_and_non_additivity():
    total = OFR_MMF_TOTAL_INVESTMENTS
    repo = OFR_MMF_REPO_INVESTMENTS
    counterparty = OFR_MMF_REPO_WITH_FOREIGN_FINANCIALS
    rate = OFR_REPO_DVP_AVERAGE_RATE
    volume = OFR_REPO_DVP_OUTSTANDING_VOLUME

    assert (total.economic_side, total.aggregation_role, total.parent_native_series_id) == (
        "asset_side",
        "reported_total",
        None,
    )
    assert (repo.aggregation_role, repo.parent_native_series_id, repo.component_axis) == (
        "component",
        total.native_series_id,
        "instrument",
    )
    assert (
        counterparty.parent_native_series_id,
        counterparty.component_axis,
    ) == (repo.native_series_id, "counterparty")
    assert MMF_TOTAL_INSTRUMENT_GROUP in total.non_additive_groups
    assert MMF_TOTAL_INSTRUMENT_GROUP in repo.non_additive_groups
    assert MMF_REPO_COUNTERPARTY_GROUP in repo.non_additive_groups
    assert MMF_REPO_COUNTERPARTY_GROUP in counterparty.non_additive_groups
    assert rate.economic_side == volume.economic_side == "market_activity"
    assert total.claim_side == "holder_asset"
    assert total.from_sector == "us_money_market_mutual_funds"
    assert total.to_sector == "multiple_security_issuers_and_repo_counterparties"
    assert counterparty.to_sector == "foreign_financial_institutions"
    assert counterparty.instrument == "repurchase_agreement"
    assert rate.claim_side == "market_activity_not_a_claim"
    assert rate.from_sector == rate.to_sector == "not_available"
    assert rate.collateral_scope == "all_reported_collateral"
    assert rate.measure_kind == "volume_weighted_mean_rate"
    assert volume.measure_kind == "outstanding_stock"
    assert rate.currency is None and rate.unit == "percent"
    assert volume.currency == "USD" and volume.unit == "USD"
    assert REPO_SELECTED_VENUES_GROUP in rate.non_additive_groups
    assert repo.cadence_policy == "complete_monthly"
    assert OFR_MMF_REPO_WITH_FED.cadence_policy == "sparse_monthly"
    assert OFR_MMF_REPO_CLEARED_FICC.cadence_policy == "sparse_monthly"
    assert counterparty.cadence_policy == "complete_monthly"
    assert rate.cadence_policy == "observed_business_days"
    assert rate.max_internal_gap_days == 7
    assert rate.minimum_weekday_coverage_ratio == 0.93
    assert rate.native_date_alignment_group == "repo_native_dates:ficc_dvp"
    assert OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE.max_internal_gap_days == 14
    assert OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE.minimum_weekday_coverage_ratio == 0.92
    assert MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP in repo.non_additive_groups
    assert (
        MMF_REPO_AND_REPO_VENUE_QUANTITIES_GROUP
        in OFR_REPO_DVP_OUTSTANDING_VOLUME.non_additive_groups
    )

    conflicts = overlapping_non_additive_groups((total, repo))
    assert conflicts == (MMF_TOTAL_INSTRUMENT_GROUP,)
    with pytest.raises(ValueError, match="cannot be summed"):
        assert_raw_sum_is_semantically_valid((total, repo))
    with pytest.raises(ValueError, match="incompatible units or measure kinds"):
        assert_raw_sum_is_semantically_valid((total, rate))
    with pytest.raises(ValueError, match="cannot be summed"):
        assert_raw_sum_is_semantically_valid((repo, OFR_REPO_DVP_OUTSTANDING_VOLUME))
    assert_raw_sum_is_semantically_valid((OFR_MMF_TREASURY_INVESTMENTS, counterparty))


def test_mmf_curated_decompositions_include_all_official_top_level_children():
    instrument_children = {
        spec.native_series_id
        for spec in OFR_SHADOW_LIQUIDITY_SERIES
        if spec.parent_native_series_id == "MMF-MMF_TOT-M" and spec.component_axis == "instrument"
    }
    counterparty_children = {
        spec.native_series_id
        for spec in OFR_SHADOW_LIQUIDITY_SERIES
        if spec.parent_native_series_id == "MMF-MMF_RP_TOT-M"
        and spec.component_axis == "counterparty"
    }
    assert instrument_children == {
        "MMF-MMF_T_TOT-M",
        "MMF-MMF_AG_TOT-M",
        "MMF-MMF_BRA_TOT-M",
        "MMF-MMF_RP_TOT-M",
        "MMF-MMF_OA_TOT-M",
    }
    assert counterparty_children == {
        "MMF-MMF_RP_wFICC-M",
        "MMF-MMF_RP_wFR-M",
        "MMF-MMF_RP_wDFI-M",
        "MMF-MMF_RP_wFFI-M",
        "MMF-MMF_RP_wOCP-M",
    }


def test_catalogue_hash_is_order_independent_and_covers_semantics():
    digest = ofr_shadow_liquidity_catalogue_sha256()
    assert len(digest) == 64
    assert digest == ofr_shadow_liquidity_catalogue_sha256(reversed(OFR_SHADOW_LIQUIDITY_SERIES))
    changed = (
        replace(
            OFR_MMF_TOTAL_INVESTMENTS,
            non_additive_groups=("different-semantic-group",),
        ),
        *OFR_SHADOW_LIQUIDITY_SERIES[1:],
    )
    assert digest != ofr_shadow_liquidity_catalogue_sha256(changed)


def test_build_url_uses_complete_native_data_without_resampling_flags():
    url = build_ofr_multifull_url((OFR_MMF_TOTAL_INVESTMENTS, OFR_REPO_DVP_AVERAGE_RATE))
    assert url == (
        "https://data.financialresearch.gov/v1/series/multifull?mnemonics="
        "MMF-MMF_TOT-M,REPO-DVP_AR_TOT-P"
    )
    assert "periodicity" not in url
    assert "remove_nulls" not in url
    with pytest.raises(ValueError, match="at least one"):
        build_ofr_multifull_url(())
    with pytest.raises(ValueError, match="duplicate"):
        build_ofr_multifull_url((OFR_MMF_TOTAL_INVESTMENTS, OFR_MMF_TOTAL_INVESTMENTS))


def test_parses_month_end_mmf_stock_in_native_dollars():
    spec = _small(OFR_MMF_TOTAL_INVESTMENTS, expected_start=date(2026, 1, 31))
    frames = parse_ofr_multifull_json(
        _text(
            spec,
            [["2026-01-31", 8_100_000_000_000.25], ["2026-02-28", None]],
        ),
        (spec,),
    )
    frame = frames[spec.native_series_id]

    assert list(frame.columns) == [
        "country",
        "indicator",
        "date",
        "value",
        "source",
        "series_id",
    ]
    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2026, 1, 31), 8_100_000_000_000.25)
    ]
    assert set(frame["country"]) == {"US"}
    assert set(frame["series_id"]) == {spec.native_series_id}
    assert frame.attrs["native_periods"] == ("2026-01-31", "2026-02-28")
    assert frame.attrs["unclassified_null_dates"] == (date(2026, 2, 28),)
    assert frame.attrs["disclosure_edit_dates"] == ()
    assert frame.attrs["publisher_last_updated_at"] == datetime(2026, 9, 4, 14, tzinfo=UTC)
    assert "never infer" in frame.attrs["missing_value_policy"]


def test_repo_disclosures_remain_missing_and_negative_rates_remain_valid():
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2018, 5, 7))
    text = _text(
        spec,
        [
            ["2018-05-07", None],
            ["2018-05-08", -0.02],
            ["2018-05-09", None],
            ["2018-05-10", 0.0],
        ],
        disclosure_edits=[["2018-05-07", None]],
    )

    frame = parse_ofr_multifull_json(text, (spec,))[spec.native_series_id]

    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2018, 5, 8), -0.02),
        (date(2018, 5, 10), 0.0),
    ]
    assert frame.attrs["disclosure_edit_dates"] == (date(2018, 5, 7),)
    assert frame.attrs["unclassified_null_dates"] == (date(2018, 5, 9),)
    assert set(frame["status"]) == {"preliminary"}
    assert frame.attrs["missing_period_records"] == (
        {
            "date": "2018-05-07",
            "reason": "disclosure_edit",
            "evidence": "ofr_disclosure_edits_subseries",
        },
        {
            "date": "2018-05-09",
            "reason": "publisher_null_not_zero",
            "evidence": "ofr_aggregation_null",
        },
    )
    assert len(frame.attrs["missing_provenance_sha256"]) == 64
    assert len(frame.attrs["native_payload_sha256"]) == 64
    assert "no trading" in frame.attrs["missing_value_policy"]
    assert "never infer zero" in frame.attrs["missing_value_policy"]


def test_disclosure_date_outside_aggregation_joins_native_period_ledger():
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2018, 5, 7))
    frame = parse_ofr_multifull_json(
        _text(
            spec,
            [["2018-05-08", 1.5]],
            disclosure_edits=[["2018-05-07", None]],
        ),
        (spec,),
    )[spec.native_series_id]

    assert frame.attrs["native_periods"] == ("2018-05-07", "2018-05-08")
    assert frame.attrs["native_aggregation_periods"] == ("2018-05-08",)
    assert frame.attrs["disclosure_edit_dates"] == (date(2018, 5, 7),)


def test_complete_monthly_and_sparse_monthly_have_distinct_gap_policies():
    complete = replace(
        OFR_MMF_TOTAL_INVESTMENTS,
        expected_start=date(2026, 1, 31),
        minimum_observations=2,
    )
    points = [["2026-01-31", 1.0], ["2026-03-31", 2.0]]
    with pytest.raises(ValueError, match="complete-monthly.*cadence gap"):
        parse_ofr_multifull_json(_text(complete, points), (complete,))

    sparse = replace(complete, cadence_policy="sparse_monthly")
    frame = parse_ofr_multifull_json(_text(sparse, points), (sparse,))[sparse.native_series_id]
    assert frame.attrs["missing_period_records"] == (
        {
            "date": "2026-02-28",
            "reason": "publisher_omitted_month_not_zero",
            "evidence": "derived_sparse_monthly_gap",
        },
    )


def test_daily_repo_first_snapshot_rejects_large_gap_or_low_density():
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2026, 1, 5))
    with pytest.raises(ValueError, match="internal gap"):
        parse_ofr_multifull_json(
            _text(spec, [["2026-01-05", 1.0], ["2026-01-20", 1.1]]),
            (spec,),
        )

    low_density = replace(spec, max_internal_gap_days=7)
    with pytest.raises(ValueError, match="weekday coverage"):
        parse_ofr_multifull_json(
            _text(
                low_density,
                [
                    ["2026-01-05", 1.0],
                    ["2026-01-09", 1.1],
                    ["2026-01-13", 1.2],
                ],
            ),
            (low_density,),
        )


def test_same_repo_venue_measures_require_identical_native_date_ledgers():
    rate = replace(
        _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2026, 1, 5)),
        minimum_weekday_coverage_ratio=0.5,
    )
    volume = replace(
        _small(OFR_REPO_DVP_TRANSACTION_VOLUME, expected_start=date(2026, 1, 5)),
        minimum_weekday_coverage_ratio=0.5,
    )
    text = json.dumps(
        {
            rate.native_series_id: _entry(
                rate,
                [["2026-01-05", 1.0], ["2026-01-06", 1.1]],
            ),
            volume.native_series_id: _entry(
                volume,
                [["2026-01-05", 10.0], ["2026-01-07", 11.0]],
            ),
        }
    )

    with pytest.raises(ValueError, match="same-venue native-date alignment mismatch"):
        parse_ofr_multifull_json(text, (rate, volume))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda entry: entry["metadata"].update(mnemonic="WRONG"), "native series"),
        (
            lambda entry: entry["metadata"]["description"].update(name="Wrong name"),
            "name mismatch",
        ),
        (
            lambda entry: entry["metadata"]["description"].update(description="Wrong definition"),
            "description mismatch",
        ),
        (
            lambda entry: entry["metadata"]["description"].update(subtype="Transaction Volume"),
            "subtype mismatch",
        ),
        (
            lambda entry: entry["metadata"]["description"].update(subsetting="Counterparty"),
            "subsetting mismatch",
        ),
        (
            lambda entry: entry["metadata"]["description"].update(vintage="Final"),
            "vintage mismatch",
        ),
        (
            lambda entry: entry["metadata"]["schedule"].update(observation_frequency="Weekly"),
            "observation frequency mismatch",
        ),
        (
            lambda entry: entry["metadata"]["schedule"].update(start_date="2018-05-08"),
            "history start mismatch",
        ),
        (
            lambda entry: entry["metadata"]["unit"].update(name="Basis Points"),
            "unit metadata mismatch",
        ),
        (
            lambda entry: entry["metadata"]["release"].update(long_name="Wrong"),
            "release long name mismatch",
        ),
        (
            lambda entry: entry["metadata"].update(parents=["SOME-PARENT"]),
            "parent relationship mismatch",
        ),
        (
            lambda entry: entry["metadata"]["description"].update(notes=""),
            "missing-value notes mismatch",
        ),
        (
            lambda entry: entry["metadata"]["schedule"].update(last_update="2026-09-04T14:00:00Z"),
            "last-update timestamp",
        ),
    ],
)
def test_rejects_ofr_metadata_semantic_drift(mutate, message):
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2018, 5, 7))
    entry = _entry(spec, [["2018-05-07", 1.5]], disclosure_edits=[])
    mutate(entry)
    text = json.dumps({spec.native_series_id: entry})

    with pytest.raises(ValueError, match=message):
        parse_ofr_multifull_json(text, (spec,))


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty response"),
        ("not json", "malformed JSON"),
        ("[]", "JSON object"),
        ('{"error": "unknown mnemonic"}', "series identity mismatch"),
    ],
)
def test_rejects_empty_malformed_or_wrong_root_payload(text, message):
    with pytest.raises(ValueError, match=message):
        parse_ofr_multifull_json(text, (OFR_MMF_TOTAL_INVESTMENTS,))


def test_multifull_response_must_have_exact_requested_series_set():
    spec = _small(OFR_MMF_TOTAL_INVESTMENTS, expected_start=date(2026, 1, 31))
    entry = _entry(spec, [["2026-01-31", 1.0]])
    with pytest.raises(ValueError, match="extra=.*UNREQUESTED"):
        parse_ofr_multifull_json(
            json.dumps({spec.native_series_id: entry, "UNREQUESTED": entry}),
            (spec,),
        )


@pytest.mark.parametrize(
    ("points", "message"),
    [
        ([["2026-01-30", 1.0]], "calendar month end"),
        ([["2026-01-31", "1.0"]], "non-numeric"),
        ([["2026-01-31", True]], "non-numeric"),
        ([["2026-01-31", float("inf")]], "non-finite"),
        (
            [["2026-02-28", 1.0], ["2026-01-31", 2.0]],
            "not strictly increasing",
        ),
        (
            [["2026-01-31", 1.0], ["2026-01-31", 2.0]],
            "duplicate dates",
        ),
    ],
)
def test_rejects_invalid_mmf_dates_and_values(points, message):
    spec = _small(OFR_MMF_TOTAL_INVESTMENTS, expected_start=date(2026, 1, 31))
    with pytest.raises(ValueError, match=message):
        parse_ofr_multifull_json(_text(spec, points), (spec,))


def test_rejects_history_start_contraction_and_stale_update_metadata():
    spec = replace(
        OFR_MMF_TOTAL_INVESTMENTS,
        expected_start=date(2026, 1, 31),
        minimum_observations=2,
    )
    with pytest.raises(ValueError, match="minimum is 2"):
        parse_ofr_multifull_json(
            _text(spec, [["2026-01-31", 1.0]]),
            (spec,),
        )

    one_row = replace(spec, minimum_observations=1)
    with pytest.raises(ValueError, match="expected history start"):
        parse_ofr_multifull_json(
            _text(one_row, [["2026-02-28", 1.0]]),
            (one_row,),
        )
    with pytest.raises(ValueError, match="last-update precedes"):
        parse_ofr_multifull_json(
            _text(
                one_row,
                [["2026-01-31", 1.0]],
                last_update="2026-01-30 12:00:00",
            ),
            (one_row,),
        )


def test_rejects_negative_volume_but_not_negative_rate():
    volume = _small(OFR_REPO_DVP_OUTSTANDING_VOLUME, expected_start=date(2018, 5, 7))
    with pytest.raises(ValueError, match="negative volume"):
        parse_ofr_multifull_json(
            _text(volume, [["2018-05-07", -1.0]], disclosure_edits=[]),
            (volume,),
        )


@pytest.mark.parametrize(
    ("points", "edits", "message"),
    [
        ([["2018-05-05", 1.0]], [], "weekend date"),
        ([["2018-05-07", 1.0]], [["2018-05-08", 0]], "date/null pair"),
        (
            [["2018-05-07", None], ["2018-05-08", 1.0]],
            [["2018-05-07", None], ["2018-05-07", None]],
            "duplicate dates",
        ),
        (
            [["2018-05-07", 1.0]],
            [["2018-05-07", None]],
            "marks observed values as disclosure edits",
        ),
    ],
)
def test_rejects_invalid_repo_date_or_disclosure_contract(points, edits, message):
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2018, 5, 7))
    with pytest.raises(ValueError, match=message):
        parse_ofr_multifull_json(
            _text(spec, points, disclosure_edits=edits),
            (spec,),
        )


def test_repo_requires_explicit_disclosure_subseries():
    spec = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2018, 5, 7))
    entry = _entry(spec, [["2018-05-07", 1.0]], disclosure_edits=[])
    del entry["timeseries"]["disclosure_edits"]
    with pytest.raises(ValueError, match="lacks disclosure_edits"):
        parse_ofr_multifull_json(
            json.dumps({spec.native_series_id: entry}),
            (spec,),
        )


def test_source_fetch_many_uses_one_official_call_and_ttl_cache(tmp_path):
    mmf = _small(OFR_MMF_TOTAL_INVESTMENTS, expected_start=date(2026, 1, 31))
    repo = _small(OFR_REPO_DVP_AVERAGE_RATE, expected_start=date(2026, 2, 2))
    payload = {
        mmf.native_series_id: _entry(mmf, [["2026-01-31", 10.0]]),
        repo.native_series_id: _entry(
            repo,
            [["2026-02-02", 3.5]],
            disclosure_edits=[],
        ),
    }
    payload_text = json.dumps(payload, indent=1)
    client = MagicMock()
    client.get.return_value = _response(payload_text)
    cache_dir = tmp_path / "cache"
    artifact_dir = tmp_path / "artifacts"
    source = OfrShadowLiquiditySource(
        client=client,
        cache_dir=cache_dir,
        artifact_dir=artifact_dir,
    )

    frames = source.fetch_many((mmf, repo), use_cache=False)
    cached = source.fetch_many((mmf, repo), use_cache=True)

    expected_url = build_ofr_multifull_url((mmf, repo))
    client.get.assert_called_once_with(expected_url, timeout=60.0)
    assert list(frames) == [mmf.native_series_id, repo.native_series_id]
    assert cached[repo.native_series_id].iloc[0]["value"] == 3.5
    assert frames[mmf.native_series_id].attrs["source_url"] == expected_url
    assert len(list(cache_dir.glob("*.json"))) == 1
    digest = hashlib.sha256(payload_text.encode("utf-8")).hexdigest()
    artifact = artifact_dir / digest[:2] / f"{digest}.json"
    assert artifact.read_bytes() == payload_text.encode("utf-8")
    assert frames[mmf.native_series_id].attrs["source_artifact_path"] == str(artifact)
    assert frames[repo.native_series_id].attrs["source_artifact_sha256"] == digest
    assert len(list(artifact_dir.rglob("*.json"))) == 5
    for frame in frames.values():
        native_path = Path(frame.attrs["native_payload_artifact_path"])
        missing_path = Path(frame.attrs["missing_provenance_artifact_path"])
        assert (
            hashlib.sha256(native_path.read_bytes()).hexdigest()
            == frame.attrs["native_payload_sha256"]
        )
        assert (
            hashlib.sha256(missing_path.read_bytes()).hexdigest()
            == frame.attrs["missing_provenance_sha256"]
        )


def test_invalid_payload_is_never_admitted_to_content_addressed_archive(tmp_path):
    client = MagicMock()
    client.get.return_value = _response('{"error":"upstream contract changed"}')
    artifact_dir = tmp_path / "artifacts"
    source = OfrShadowLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="series identity mismatch"):
        source.fetch(OFR_MMF_TOTAL_INVESTMENTS, use_cache=False)

    assert not artifact_dir.exists()


def test_shared_cache_environment_uses_provider_subdirectory(tmp_path, monkeypatch):
    spec = _small(OFR_MMF_TOTAL_INVESTMENTS, expected_start=date(2026, 1, 31))
    client = MagicMock()
    client.get.return_value = _response(_text(spec, [["2026-01-31", 1.0]]))
    cache_root = tmp_path / "shared-cache"
    monkeypatch.setenv("DALIO_SHADOW_LIQUIDITY_CACHE", str(cache_root))
    source = OfrShadowLiquiditySource(
        client=client,
        artifact_dir=tmp_path / "artifacts",
    )

    source.fetch(spec, use_cache=False)

    assert len(list((cache_root / "ofr").glob("*.json"))) == 1


def test_catalogue_validator_rejects_orphan_component_and_missing_groups():
    orphan = replace(
        OFR_MMF_REPO_INVESTMENTS,
        parent_native_series_id="NOT-IN-CATALOGUE",
    )
    specs = (OFR_MMF_TOTAL_INVESTMENTS, orphan)
    with pytest.raises(ValueError, match="parent outside"):
        validate_ofr_shadow_catalogue(specs)

    no_groups = replace(OFR_MMF_TOTAL_INVESTMENTS, non_additive_groups=())
    with pytest.raises(ValueError, match="at least one non-additive group"):
        validate_ofr_shadow_catalogue((no_groups,))


def test_catalogue_validator_rejects_invented_direction_and_wrong_cadence():
    invented_repo_sector = replace(
        OFR_REPO_DVP_AVERAGE_RATE,
        from_sector="dealers",
    )
    specs = tuple(
        invented_repo_sector if item is OFR_REPO_DVP_AVERAGE_RATE else item
        for item in OFR_SHADOW_LIQUIDITY_SERIES
    )
    with pytest.raises(ValueError, match="must not invent sectors"):
        validate_ofr_shadow_catalogue(specs)

    wrong_cadence = replace(
        OFR_MMF_TOTAL_INVESTMENTS,
        cadence_policy="observed_business_days",
    )
    specs = tuple(
        wrong_cadence if item is OFR_MMF_TOTAL_INVESTMENTS else item
        for item in OFR_SHADOW_LIQUIDITY_SERIES
    )
    with pytest.raises(ValueError, match="cadence policy incompatible"):
        validate_ofr_shadow_catalogue(specs)
