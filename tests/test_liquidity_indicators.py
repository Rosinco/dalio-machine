"""Derived liquidity diagnostics remain point-in-time, separate, and auditable."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_CATALOGUE_VINTAGE_PREFIX,
    BIS_GLI_USD,
    bis_global_liquidity_catalogue_sha256,
)
from dalio.data_sources.money_liquidity import (
    BOE_M4,
    BOE_M4EX,
    BOE_M4EX_QUARTERLY,
    BOJ_BROADLY_DEFINED_LIQUIDITY,
    BOJ_M3,
    ECB_M3,
    FED_M2,
    FED_TOTAL_ASSETS,
    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX,
    SCB_M3,
    money_liquidity_catalogue_sha256,
)
from dalio.data_sources.ofr_shadow_liquidity import (
    OFR_MMF_AGENCY_GSE_INVESTMENTS,
    OFR_MMF_BANK_RELATED_INVESTMENTS,
    OFR_MMF_OTHER_ASSET_INVESTMENTS,
    OFR_MMF_REPO_INVESTMENTS,
    OFR_MMF_TOTAL_INVESTMENTS,
    OFR_MMF_TREASURY_INVESTMENTS,
    OFR_REPO_DVP_AVERAGE_RATE,
    OFR_REPO_GCF_AVERAGE_RATE,
    OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
    OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX,
    ofr_shadow_liquidity_catalogue_sha256,
)
from dalio.indicators.liquidity import (
    EXCLUDED_MONEY_DIAGNOSTICS,
    METHODOLOGY_VERSION,
    PARAMETERS,
    PRIMARY_MONEY,
    REPO_RATE_SERIES,
    REPO_VOLUME_SERIES,
    LiquiditySemanticsError,
    build_liquidity_overview,
    load_liquidity_panel,
    methodology_record,
    methodology_sha256,
    mmf_overview,
    money_central_bank_gap,
    money_impulse,
    offshore_credit_impulse,
    repo_conditions,
    validate_liquidity_snapshot,
)
from dalio.pipelines import build_liquidity_brief
from dalio.storage.db import init_db, make_engine, make_session_factory
from dalio.storage.releases import (
    ProjectionScope,
    ReleaseArtifactMeta,
    ReleaseMeta,
    ingest_release_snapshot,
    make_partition_key,
)


def _rows(
    spec,
    dates: list[date],
    values: list[float],
    *,
    release_id: int = 1,
    status: str = "observed",
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_id": release_id,
                "partition_key": make_partition_key(
                    spec.source_family,
                    spec.native_series_id,
                    spec.country,
                    spec.indicator,
                ),
                "source_family": spec.source_family,
                "published_at": None,
                "available_at": datetime(2026, 9, 8),
                "retrieved_at": datetime(2026, 9, 8),
                "vintage_label": "test",
                "country": spec.country,
                "indicator": spec.indicator,
                "date": observed,
                "value": value,
                "source": spec.source_family,
                "series_id": spec.native_series_id,
                "status": status,
            }
            for observed, value in zip(dates, values, strict=True)
        ]
    )


def _policy_rows(
    dates: list[date],
    values: list[float],
    *,
    release_id: int = 99,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "release_id": release_id,
                "partition_key": make_partition_key("FRED", "DFF", "US", "policy_rate"),
                "source_family": "FRED",
                "published_at": None,
                "available_at": datetime(2026, 9, 8),
                "retrieved_at": datetime(2026, 9, 8),
                "vintage_label": None,
                "country": "US",
                "indicator": "policy_rate",
                "date": observed,
                "value": value,
                "source": "FRED",
                "series_id": "DFF",
                "status": "observed",
            }
            for observed, value in zip(dates, values, strict=True)
        ]
    )


def _monthly_dates(start: str, periods: int, *, month_end: bool = False) -> list[date]:
    index = pd.period_range(start=start, periods=periods, freq="M")
    if month_end:
        return [period.end_time.date() for period in index]
    return [period.start_time.date() for period in index]


def test_methodology_pins_primary_series_and_excludes_nested_diagnostics():
    assert METHODOLOGY_VERSION == "liquidity-diagnostics-v1"
    assert [spec.native_series_id for spec in PRIMARY_MONEY] == [
        "M2SL",
        "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E",
        "TAB6541/5LLM3a.1E.NEP.V.A/000007WQ",
        "RPMB53Q",
        "MAM1NABLBLMO",
    ]
    assert {spec.native_series_id for spec in EXCLUDED_MONEY_DIAGNOSTICS} == {
        BOE_M4EX_QUARTERLY.native_series_id,
        BOE_M4.native_series_id,
        BOJ_M3.native_series_id,
    }
    assert len(methodology_sha256()) == 64


def test_money_impulse_uses_exact_calendar_log_growth_and_acceleration():
    dates = _monthly_dates("2025-04", 16)
    values = [100.0 + index * 2.0 for index in range(16)]
    result = money_impulse(_rows(FED_M2, dates, values), FED_M2, as_of=date(2026, 9, 8))

    expected_growth = 100.0 * math.log(values[15] / values[3])
    prior_growth = 100.0 * math.log(values[12] / values[0])
    assert result["availability_status"] == "ready"
    assert result["annual_log_growth_pct"] == pytest.approx(expected_growth)
    assert result["acceleration_3m_pp"] == pytest.approx(expected_growth - prior_growth)
    assert result["input_dates"] == [
        "2025-04-01",
        "2025-07-01",
        "2026-04-01",
        "2026-07-01",
    ]


def test_money_impulse_does_not_bridge_or_fill_a_missing_month():
    dates = _monthly_dates("2025-04", 16)
    frame = _rows(FED_M2, dates, [100.0 + index for index in range(16)]).drop(index=7)
    result = money_impulse(frame, FED_M2, as_of=date(2026, 9, 8))
    assert result["availability_status"] == "unavailable"
    assert "16 calendar months" in result["missing_reason"]


def test_central_bank_gap_uses_last_weekly_anchor_before_month_end():
    money_dates = _monthly_dates("2025-04", 16)
    money_values = [1_000.0 * 1.005**index for index in range(16)]
    weekly_dates = [date(2025, 3, 26) + timedelta(days=7 * index) for index in range(71)]
    weekly_values = [5_000.0 * 1.001**index for index in range(71)]
    future = date(2026, 8, 5)
    weekly_dates.append(future)
    weekly_values.append(9_999.0)

    result = money_central_bank_gap(
        _rows(FED_M2, money_dates, money_values, release_id=1),
        _rows(FED_TOTAL_ASSETS, weekly_dates, weekly_values, release_id=2),
        FED_M2,
        FED_TOTAL_ASSETS,
        as_of=date(2026, 9, 8),
    )

    assert result["availability_status"] == "ready"
    assert future.isoformat() not in result["input_dates"]
    assert all(input_date <= "2026-07-31" for input_date in result["input_dates"])


def _mmf_panel(*, include_counterparties: bool = False) -> pd.DataFrame:
    dates = _monthly_dates("2025-04", 16, month_end=True)
    total_values = [360_000_000_000.0 * 1.01**index for index in range(16)]
    specs_and_shares = (
        (OFR_MMF_TOTAL_INVESTMENTS, 1.0),
        (OFR_MMF_REPO_INVESTMENTS, 0.30),
        (OFR_MMF_TREASURY_INVESTMENTS, 0.40),
        (OFR_MMF_AGENCY_GSE_INVESTMENTS, 0.15),
        (OFR_MMF_BANK_RELATED_INVESTMENTS, 0.10),
        (OFR_MMF_OTHER_ASSET_INVESTMENTS, 0.05),
    )
    frames = [
        _rows(
            spec,
            dates,
            [total * share for total in total_values],
            release_id=index,
        )
        for index, (spec, share) in enumerate(specs_and_shares, start=1)
    ]
    m2_dates = _monthly_dates("2025-04", 16)
    frames.append(
        _rows(
            FED_M2,
            m2_dates,
            [1_000.0 * 1.005**index for index in range(16)],
            release_id=20,
        )
    )
    if include_counterparties:
        from dalio.indicators.liquidity import MMF_REPO_COUNTERPARTIES

        for index, spec in enumerate(MMF_REPO_COUNTERPARTIES, start=30):
            frames.append(
                _rows(
                    spec, dates, [total * 0.30 * 0.20 for total in total_values], release_id=index
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_mmf_uses_explicit_parent_shares_and_does_not_zero_fill_missing_counterparties():
    panel = _mmf_panel()
    result = mmf_overview(panel, as_of=date(2026, 9, 8))
    assert result["availability_status"] == "ready"
    assert [row["share_of_total_pct"] for row in result["asset_allocation"]] == pytest.approx(
        [30.0, 40.0, 15.0, 10.0, 5.0]
    )
    assert result["mmf_minus_m2_growth_gap_pp"] > 0

    counterparties = result["published_repo_counterparty_categories"]
    assert {
        "availability_status",
        "period",
        "ratios",
        "interpretation_limit",
        "input_series_ids",
        "input_release_ids",
        "input_dates",
        "input_statuses",
    } <= counterparties.keys()
    assert counterparties["availability_status"] == "unavailable"
    assert counterparties["period"] is None
    assert counterparties["ratios"] == []

    overview = build_liquidity_overview(
        panel,
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
    )
    assert overview["coverage"]["mmf_headline_and_allocation"] == {
        "ready": 1,
        "expected": 1,
    }
    assert overview["coverage"]["mmf_published_repo_categories"] == {
        "ready": 0,
        "expected": 1,
    }
    markdown = build_liquidity_brief.render_markdown(overview)
    assert counterparties["missing_reason"] in markdown


def test_mmf_counterparties_are_each_divided_by_named_repo_parent():
    result = mmf_overview(_mmf_panel(include_counterparties=True), as_of=date(2026, 9, 8))
    counterparties = result["published_repo_counterparty_categories"]
    assert {
        "availability_status",
        "period",
        "ratios",
        "interpretation_limit",
        "input_series_ids",
        "input_release_ids",
        "input_dates",
        "input_statuses",
    } <= counterparties.keys()
    assert counterparties["availability_status"] == "ready"
    assert counterparties["period"] == "2026-07"
    assert [row["share_of_repo_pct"] for row in counterparties["ratios"]] == pytest.approx(
        [20.0] * 5
    )


def _repo_panel(
    *,
    drop_latest_triparty: bool = False,
    keep: int = 260,
    include_volumes: bool = False,
) -> pd.DataFrame:
    dates = [timestamp.date() for timestamp in pd.bdate_range("2025-01-02", periods=260)]
    frames = []
    for index, spec in enumerate(
        (
            OFR_REPO_DVP_AVERAGE_RATE,
            OFR_REPO_GCF_AVERAGE_RATE,
            OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE,
        ),
        start=1,
    ):
        selected_dates = dates[:keep]
        if drop_latest_triparty and spec is OFR_REPO_TRIPARTY_EX_FED_AVERAGE_RATE:
            selected_dates = selected_dates[:-1]
        values = [
            3.0 + 0.01 * index + 0.001 * (offset % (5 + index))
            for offset in range(len(selected_dates))
        ]
        frames.append(_rows(spec, selected_dates, values, release_id=index, status="preliminary"))
    frames.append(_policy_rows(dates[:keep], [0.0] * keep))
    if include_volumes:
        for index, spec in enumerate(REPO_VOLUME_SERIES, start=10):
            frames.append(
                _rows(
                    spec,
                    dates[:keep],
                    [100_000_000_000.0 + 10_000_000.0 * offset for offset in range(keep)],
                    release_id=index,
                    status="preliminary",
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_repo_conditions_use_only_common_dates_and_allow_a_zero_policy_rate():
    expected_date = pd.bdate_range("2025-01-02", periods=259)[-1].date()
    result = repo_conditions(
        _repo_panel(drop_latest_triparty=True),
        as_of=expected_date + timedelta(days=10),
    )
    assert result["availability_status"] == "ready"
    assert result["period_date"] == expected_date.isoformat()
    assert all(row["status"] == "preliminary" for row in result["venues"])
    assert result["fragmentation_5d_median_bp"] >= 0
    assert all(row["availability_status"] == "unavailable" for row in result["volume_context"])
    assert all("missing_reason" in row for row in result["volume_context"])

    overview = build_liquidity_overview(
        _repo_panel(drop_latest_triparty=True),
        as_of=expected_date + timedelta(days=10),
        as_known_at=datetime(2026, 1, 9, 21, tzinfo=UTC),
    )
    assert overview["coverage"]["repo_pricing"] == {"ready": 1, "expected": 1}
    assert overview["coverage"]["repo_activity_context"] == {"ready": 0, "expected": 5}


def test_repo_conditions_return_unavailable_instead_of_filling_sparse_dates():
    result = repo_conditions(_repo_panel(keep=256), as_of=date(2026, 1, 10))
    assert result["availability_status"] == "unavailable"
    assert "257 aligned" in result["missing_reason"]


def test_repo_conditions_publish_exact_smoothing_baseline_and_activity_lineage():
    dates = [timestamp.date() for timestamp in pd.bdate_range("2025-01-02", periods=257)]
    panel = _repo_panel(keep=257, include_volumes=True)
    result = repo_conditions(
        panel,
        as_of=dates[-1] + timedelta(days=3),
    )

    assert result["availability_status"] == "ready"
    assert result["smoothing_start_date"] == dates[-5].isoformat()
    assert result["smoothing_end_date"] == dates[-1].isoformat()
    assert result["smoothing_observations"] == 5
    assert result["baseline_observations"] == 252
    assert result["baseline_start_date"] in result["input_dates"]
    assert result["baseline_end_date"] in result["input_dates"]
    assert result["input_dates"] == [observed.isoformat() for observed in dates]
    assert set(result["input_series_ids"]) == {
        *(spec.native_series_id for spec in REPO_RATE_SERIES),
        "DFF",
    }
    assert all(row["availability_status"] == "ready" for row in result["volume_context"])
    for row in result["volume_context"]:
        assert row["context_start_date"] == dates[-20].isoformat()
        assert row["context_end_date"] == dates[-1].isoformat()
        assert row["context_observations"] == 20
        assert row["status"] == "preliminary"

    overview = build_liquidity_overview(
        panel,
        as_of=dates[-1] + timedelta(days=3),
        as_known_at=datetime.combine(
            dates[-1] + timedelta(days=3),
            datetime.min.time(),
            tzinfo=UTC,
        ),
    )
    markdown = build_liquidity_brief.render_markdown(overview)
    expected_median = result["volume_context"][0]["trailing_observation_median"] / 1_000_000_000
    assert f"{expected_median:,.1f} USD bn" in markdown


def test_offshore_credit_uses_quarterly_log_growth_without_currency_aggregation():
    periods = pd.period_range("2025Q1", periods=6, freq="Q")
    dates = [period.start_time.date() for period in periods]
    values = [100.0, 101.0, 102.0, 103.0, 110.0, 115.0]
    result = offshore_credit_impulse(
        _rows(BIS_GLI_USD, dates, values),
        BIS_GLI_USD,
        as_of=date(2026, 9, 8),
    )
    growth = 100.0 * math.log(values[-1] / values[1])
    previous = 100.0 * math.log(values[-2] / values[0])
    assert result["availability_status"] == "ready"
    assert result["annual_log_growth_pct"] == pytest.approx(growth)
    assert result["acceleration_1q_pp"] == pytest.approx(growth - previous)
    assert result["currency"] == "USD"


def test_money_summary_suppresses_cross_country_statistics_when_periods_are_mixed():
    specs = (FED_M2, ECB_M3, SCB_M3, BOE_M4EX, BOJ_BROADLY_DEFINED_LIQUIDITY)
    frames = []
    for release_id, spec in enumerate(specs, start=1):
        start = "2025-03" if spec is BOJ_BROADLY_DEFINED_LIQUIDITY else "2025-04"
        dates = _monthly_dates(start, 16)
        frames.append(
            _rows(
                spec,
                dates,
                [100.0 + release_id + 2.0 * index for index in range(16)],
                release_id=release_id,
            )
        )

    snapshot = build_liquidity_overview(
        pd.concat(frames, ignore_index=True),
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
    )
    assert snapshot["money_summary"]["ready"] == len(specs)
    assert (
        len(
            {
                row["period"]
                for row in snapshot["broad_money"]
                if row["availability_status"] == "ready"
            }
        )
        == 2
    )
    assert snapshot["money_summary"]["median_annual_log_growth_pct"] is None
    assert snapshot["money_summary"]["positive_growth_breadth"] is None
    assert snapshot["money_summary"]["accelerating_breadth"] is None


def test_overview_contract_has_no_score_or_cross_currency_total_and_rejects_nan():
    columns = [
        "release_id",
        "partition_key",
        "source_family",
        "published_at",
        "available_at",
        "retrieved_at",
        "vintage_label",
        "country",
        "indicator",
        "date",
        "value",
        "source",
        "series_id",
        "status",
    ]
    snapshot = build_liquidity_overview(
        pd.DataFrame(columns=columns),
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
    )
    assert "composite_score" not in snapshot
    assert "global_liquidity_total" not in snapshot
    validate_liquidity_snapshot(snapshot)

    bad = dict(snapshot, overall_score=np.nan)
    with pytest.raises(LiquiditySemanticsError, match="forbidden aggregate"):
        validate_liquidity_snapshot(bad)
    bad = dict(snapshot)
    bad["money_summary"] = {"value": np.nan}
    with pytest.raises(LiquiditySemanticsError, match="finite canonical JSON"):
        validate_liquidity_snapshot(bad)

    tampered = json.loads(json.dumps(snapshot))
    tampered["as_of"] = "1900-01-01"
    with pytest.raises(LiquiditySemanticsError, match="snapshot.*(hash|fingerprint)|fingerprint"):
        validate_liquidity_snapshot(tampered)


def test_methodology_record_is_json_and_pins_all_window_and_alignment_parameters():
    record = methodology_record()
    assert json.loads(json.dumps(record, allow_nan=False, sort_keys=True)) == record
    assert record["sha256"] == methodology_sha256()
    required_parameters = {
        "money_required_months": 16,
        "money_acceleration_months": 3,
        "central_bank_anchor_max_lag_days": 10,
        "mmf_required_months": 16,
        "mmf_repo_required_months": 13,
        "repo_smoothing_observations": 5,
        "repo_baseline_observations": 252,
        "repo_required_aligned_observations": 257,
        "repo_activity_window_observations": 20,
        "offshore_required_quarters": 6,
        "offshore_acceleration_quarters": 1,
        "money_summary_requires_common_period": True,
    }
    assert required_parameters.items() <= record["parameters"].items()
    changed = replace(PARAMETERS, repo_smoothing_observations=6)
    assert methodology_sha256(changed) != methodology_sha256()


def _money_snapshot(values: list[float]) -> pd.DataFrame:
    dates = _monthly_dates("2025-04", len(values))
    return pd.DataFrame(
        {
            "country": FED_M2.country,
            "indicator": FED_M2.indicator,
            "date": dates,
            "value": values,
            "source": FED_M2.source_family,
            "series_id": FED_M2.native_series_id,
        }
    )


def test_release_first_loader_respects_known_at_and_does_not_resurrect_omissions(tmp_path):
    engine = make_engine(tmp_path / "point-in-time.db")
    init_db(engine)
    sessions = make_session_factory(engine)
    first_clock = datetime(2026, 9, 8, 10, tzinfo=UTC)
    second_clock = datetime(2026, 9, 8, 12, tzinfo=UTC)
    vintage = MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()
    partition = make_partition_key(
        FED_M2.source_family,
        FED_M2.native_series_id,
        FED_M2.country,
        FED_M2.indicator,
    )
    first = _money_snapshot([100.0 + index for index in range(17)])
    second = first.drop(index=5).copy()
    with sessions() as session:
        for frame, clock in ((first, first_clock), (second, second_clock)):
            ingest_release_snapshot(
                session,
                frame,
                ReleaseMeta(
                    partition_key=partition,
                    source_family=FED_M2.source_family,
                    available_at=clock,
                    retrieved_at=clock,
                    vintage_label=vintage,
                    projection=ProjectionScope(
                        country=FED_M2.country,
                        indicator=FED_M2.indicator,
                        sources=(FED_M2.source_family,),
                    ),
                ),
            )

    with sessions() as session:
        before = load_liquidity_panel(
            session,
            as_known_at=first_clock - timedelta(seconds=1),
            through_date=date(2026, 7, 31),
        )
        first_view = load_liquidity_panel(
            session,
            as_known_at=first_clock + timedelta(seconds=1),
            through_date=date(2026, 7, 31),
        )
        second_view = load_liquidity_panel(
            session,
            as_known_at=second_clock + timedelta(seconds=1),
            through_date=date(2026, 7, 31),
        )
    assert before.empty
    assert len(first_view) == 16  # future August observation is capped after release selection
    omitted = first.iloc[5]["date"]
    assert omitted in set(first_view["date"])
    assert omitted not in set(second_view["date"])


@pytest.mark.parametrize(
    ("spec", "bad_vintage"),
    (
        (
            FED_M2,
            MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256() + "junk",
        ),
        (
            BIS_GLI_USD,
            BIS_GLI_CATALOGUE_VINTAGE_PREFIX + bis_global_liquidity_catalogue_sha256() + "junk",
        ),
        (
            OFR_MMF_TOTAL_INVESTMENTS,
            OFR_SHADOW_CATALOGUE_VINTAGE_PREFIX
            + ofr_shadow_liquidity_catalogue_sha256()
            + ";p:0123456789abcdef01234567junk",
        ),
    ),
)
def test_release_loader_rejects_a_valid_catalogue_digest_with_trailing_junk(
    tmp_path,
    spec,
    bad_vintage,
):
    engine = make_engine(tmp_path / "catalogue-junk.db")
    init_db(engine)
    sessions = make_session_factory(engine)
    clock = datetime(2026, 9, 8, 10, tzinfo=UTC)
    frame = pd.DataFrame(
        {
            "country": [spec.country],
            "indicator": [spec.indicator],
            "date": [date(2026, 7, 1)],
            "value": [100.0],
            "source": [spec.source_family],
            "series_id": [spec.native_series_id],
        }
    )
    with sessions() as session:
        ingest_release_snapshot(
            session,
            frame,
            ReleaseMeta(
                partition_key=make_partition_key(
                    spec.source_family,
                    spec.native_series_id,
                    spec.country,
                    spec.indicator,
                ),
                source_family=spec.source_family,
                available_at=clock,
                retrieved_at=clock,
                vintage_label=bad_vintage,
            ),
            replace_current=False,
        )

    with (
        sessions() as session,
        pytest.raises(
            LiquiditySemanticsError,
            match="catalogue does not match",
        ),
    ):
        load_liquidity_panel(
            session,
            as_known_at=clock + timedelta(seconds=1),
            through_date=date(2026, 7, 31),
        )


def test_artifact_loss_fails_closed_before_analysis(tmp_path):
    engine = make_engine(tmp_path / "artifact.db")
    init_db(engine)
    sessions = make_session_factory(engine)
    source = tmp_path / "source.csv"
    source.write_bytes(b"source")
    ledger = tmp_path / "missing.json"
    provenance = json.dumps({"missing": []}, sort_keys=True, separators=(",", ":"))
    ledger.write_text(provenance)
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    ledger_hash = hashlib.sha256(ledger.read_bytes()).hexdigest()
    clock = datetime(2026, 9, 8, 10, tzinfo=UTC)
    frame = _money_snapshot([100.0 + index for index in range(16)])
    with sessions() as session:
        ingest_release_snapshot(
            session,
            frame,
            ReleaseMeta(
                partition_key=make_partition_key(
                    FED_M2.source_family,
                    FED_M2.native_series_id,
                    FED_M2.country,
                    FED_M2.indicator,
                ),
                source_family=FED_M2.source_family,
                available_at=clock,
                retrieved_at=clock,
                vintage_label=(
                    MONEY_LIQUIDITY_CATALOGUE_VINTAGE_PREFIX + money_liquidity_catalogue_sha256()
                ),
                projection=ProjectionScope(
                    country=FED_M2.country,
                    indicator=FED_M2.indicator,
                    sources=(FED_M2.source_family,),
                ),
                artifacts=(
                    ReleaseArtifactMeta(
                        "source_response",
                        source_hash,
                        source,
                        source_hash,
                        ledger_hash,
                        provenance,
                    ),
                    ReleaseArtifactMeta(
                        "native_series_payload",
                        source_hash,
                        source,
                        source_hash,
                        ledger_hash,
                        provenance,
                    ),
                    ReleaseArtifactMeta(
                        "missingness_ledger",
                        ledger_hash,
                        ledger,
                        source_hash,
                        ledger_hash,
                        provenance,
                    ),
                ),
            ),
        )
    source.unlink()

    with sessions() as session, pytest.raises(LiquiditySemanticsError, match="artifact manifest"):
        from dalio.indicators.liquidity import build_liquidity_snapshot

        build_liquidity_snapshot(
            session,
            as_of=date(2026, 7, 31),
            as_known_at=clock + timedelta(seconds=1),
        )


def test_brief_output_is_deterministic_and_database_is_opened_read_only(tmp_path, monkeypatch):
    db_path = tmp_path / "brief.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute("create table sentinel(value integer)")
        connection.execute("insert into sentinel values (1)")
    before = db_path.read_bytes()
    empty = pd.DataFrame(
        columns=[
            "release_id",
            "partition_key",
            "source_family",
            "published_at",
            "available_at",
            "retrieved_at",
            "vintage_label",
            "country",
            "indicator",
            "date",
            "value",
            "source",
            "series_id",
            "status",
        ]
    )
    snapshot = build_liquidity_overview(
        empty,
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
    )
    monkeypatch.setattr(build_liquidity_brief, "build_liquidity_snapshot", lambda *a, **k: snapshot)
    out = tmp_path / "snapshots"
    first, paths = build_liquidity_brief.run(
        db_path=db_path,
        output_dir=out,
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
        require_complete=False,
    )
    first_bytes = [path.read_bytes() for path in paths]
    second, second_paths = build_liquidity_brief.run(
        db_path=db_path,
        output_dir=out,
        as_of=date(2026, 9, 8),
        as_known_at=datetime(2026, 9, 8, 21, tzinfo=UTC),
        require_complete=False,
    )
    assert first == second
    assert first_bytes == [path.read_bytes() for path in second_paths]
    assert db_path.read_bytes() == before
    assert "not a universal M5" in build_liquidity_brief.render_markdown(first)
