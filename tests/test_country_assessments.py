"""Behavioral guards for descriptive country assessments and conditional scenarios."""

from copy import deepcopy
from datetime import date

import pytest

from dalio.assessments.core import build_snapshot, content_hash


def evidence(*, missing=(), structural_year=2025):
    values = {
        "real_gdp_growth": (2.5, 1.5, 2.0),
        "gov_debt_pct_gdp": (80.0, 81.0, 85.0),
        "fiscal_balance_pct_gdp": (-3.0, -3.5, -2.0),
        "primary_balance_pct_gdp": (-1.0, -1.2, 0.5),
        "current_account_pct_gdp": (-2.0, -2.5, -1.0),
        "old_age_dependency": (30.0, 30.0, 30.0),
        "energy_net_imports_pct": (25.0, 25.0, 25.0),
        "rd_pct_gdp": (2.0, 2.0, 2.0),
    }
    series = []
    for number, (metric, points) in enumerate(values.items(), 1):
        if metric in missing:
            continue
        structural = number > 5
        source = "WORLD_BANK" if structural else (
            "IMF_FISCAL_MONITOR" if metric == "primary_balance_pct_gdp" else "IMF_WEO"
        )
        rows = [
            {"year": year, "date": f"{year}-12-31", "value": value,
             "source": source + ("_FCST" if year >= 2026 and not structural else ""),
             "status": "published_statistic" if structural else (
                 "forecast_calendar_convention" if year >= 2026 else "estimate_or_outturn")}
            for year, value in (
                [(structural_year, points[0])] if structural
                else zip((2025, 2026, 2031), points, strict=True)
            )
        ]
        series.append({
            "indicator": metric, "family": "wb" if structural else "imf",
            "source": source, "series_id": metric, "release_id": number,
            "partition_key": f"{source}:{metric}:SE", "published_at": None,
            "available_at": "2026-09-10T20:49:32+00:00",
            "retrieved_at": "2026-09-10T20:49:32+00:00",
            "source_url": f"https://example.test/{metric}",
            "publisher_metadata": {"source": "Test retained publisher vintage"},
            "artifacts": [], "observations": rows, "missingness": {},
        })
    result = {
        "as_known_at": "2026-09-10T23:59:59+00:00", "protected_artifact_paths": [],
        "countries": [{"country": "SE", "listing_iso2": "SE", "name": "Sweden",
                       "listing_count": 100, "series": series, "gaps": []}],
    }
    result["evidence_digest"] = content_hash(result)
    return result


def changed(bundle, mutation):
    mutation(bundle)
    bundle["evidence_digest"] = content_hash({k: v for k, v in bundle.items()
                                               if k != "evidence_digest"})
    return bundle


def build(bundle):
    return build_snapshot(bundle, as_of=date(2026, 9, 10))


def test_rate_slowdown_is_not_called_output_contraction():
    country = build(evidence())["countries"][0]
    finding = next(f for f in country["findings"] if f["id"] == "growth_path")
    assert "1.00 percentage points lower" in finding["text"]
    assert "positive" in finding["text"]
    assert "contraction" not in finding["text"]
    assert finding["kind"] == "derived_interpretation"


def test_debt_change_is_ratio_arithmetic_and_preserves_estimate_status():
    country = build(evidence())["countries"][0]
    assert country["baseline"]["gov_debt_pct_gdp"]["status"] == "estimate_or_outturn"
    finding = next(f for f in country["findings"] if f["id"] == "debt_path")
    assert "5.00 percentage points" in finding["text"]
    assert any("nominal GDP" in limit for limit in finding["limits"])


def test_missing_baseline_is_not_backfilled_or_zero_filled():
    def remove(bundle):
        s = bundle["countries"][0]["series"][0]
        s["observations"][0]["year"] = 2024
        s["observations"][0]["date"] = "2024-12-31"
    country = build(changed(evidence(), remove))["countries"][0]
    assert country["baseline"]["real_gdp_growth"] is None
    assert country["coverage"]["core_available"] == 4
    assert not any(f["id"] == "growth_path" for f in country["findings"])


def test_missing_forecast_endpoint_is_explicit_and_not_interpolated():
    def remove(bundle):
        bundle["countries"][0]["series"][1]["observations"].pop()
    country = build(changed(evidence(), remove))["countries"][0]
    assert country["projections"][-1]["metrics"]["gov_debt_pct_gdp"] is None
    assert not any(f["id"] == "debt_path" for f in country["findings"])
    assert any("2031" in gap and "debt" in gap.lower() for gap in country["gaps"])
    assert country["projections"][1]["metrics"]["real_gdp_growth"] is None


def test_structural_staleness_blocks_current_energy_scenario():
    country = build(evidence(structural_year=2015))["countries"][0]
    assert country["structural"]["energy_net_imports_pct"]["stale"] is True
    assert not any(f["id"] == "energy_exposure" for f in country["findings"])
    assert not any(s["id"] == "energy_import_shock" for s in country["scenarios"])
    assert any("2015" in gap for gap in country["gaps"])


def test_net_exporter_is_not_treated_as_energy_price_insulated():
    def modify(bundle):
        bundle["countries"][0]["series"][6]["observations"][0]["value"] = -30
    country = build(changed(evidence(), modify))["countries"][0]
    finding = next(f for f in country["findings"] if f["id"] == "energy_exposure")
    assert "net energy exporter" in finding["text"]
    assert any("price" in x for x in finding["limits"])
    assert not any(s["id"] == "energy_import_shock" for s in country["scenarios"])


def test_scenarios_are_conditional_with_evidence_and_company_checks():
    result = build(evidence())
    country = result["countries"][0]
    for scenario in country["scenarios"]:
        assert scenario["kind"] == "conditional_scenario"
        assert scenario["assumptions"] and scenario["pathway"]
        assert scenario["signposts"] and scenario["invalidators"]
        assert scenario["company_checks"] and scenario["limitations"]
        assert scenario["evidence_refs"]
        assert "probability" not in scenario
        assert set(scenario["evidence_refs"]) <= result["citations"].keys()
    assert all(f["kind"] == "derived_interpretation" for f in country["findings"])
    assert "risk_score" not in country


def test_no_cross_dataset_interest_expense_and_current_account_has_no_verdict():
    country = build(evidence())["countries"][0]
    assert "interest_burden" not in str(country)
    finding = next(f for f in country["findings"] if f["id"] == "external_balance")
    assert "deficit" in finding["text"]
    assert "bad" not in finding["text"]
    assert any("financing" in x for x in finding["limits"])


def test_insufficient_core_evidence_does_not_generate_standard_scenarios():
    core = ("real_gdp_growth", "gov_debt_pct_gdp", "fiscal_balance_pct_gdp",
            "primary_balance_pct_gdp", "current_account_pct_gdp")
    country = build(evidence(missing=core))["countries"][0]
    assert country["status"] == "insufficient_evidence"
    assert country["scenarios"] == []


def test_snapshot_is_deterministic_and_does_not_mutate_evidence():
    bundle = evidence()
    original = deepcopy(bundle)
    first = build(bundle)
    assert first == build(bundle)
    assert bundle == original
    assert first["snapshot_sha256"] == content_hash(
        {k: v for k, v in first.items() if k != "snapshot_sha256"})


def test_tampered_evidence_digest_and_impossible_clock_are_rejected():
    bundle = evidence()
    bundle["countries"][0]["series"][0]["observations"][0]["value"] = 99
    with pytest.raises(ValueError, match="digest"):
        build(bundle)
    with pytest.raises(ValueError, match="known"):
        build_snapshot(evidence(), as_of=date(2027, 1, 1))


def test_nonforecast_row_is_not_presented_as_publisher_projection():
    def modify(bundle):
        row = bundle["countries"][0]["series"][0]["observations"][1]
        row["status"] = "estimate_or_outturn"
        row["source"] = "IMF_WEO"
    country = build(changed(evidence(), modify))["countries"][0]
    assert country["projections"][0]["metrics"]["real_gdp_growth"] is None


def test_point_references_retain_release_and_measurement_identity():
    result = build(evidence())
    for point in result["citations"].values():
        assert point["country"] == "SE"
        assert point["year"] and point["release_id"] and point["series_id"]
        assert point["source_url"] and point["available_at"] and point["unit"]
    primary = result["countries"][0]["baseline"]["primary_balance_pct_gdp"]
    assert primary["source"] == "IMF_FISCAL_MONITOR"


def test_native_central_debt_never_replaces_general_government_ratio():
    def add_native(bundle):
        bundle["countries"][0]["national_debt_context"] = [{
            "metric": "central_gov_gross_debt_sek", "value": 1_249_558_872_203,
            "unit": "SEK", "year": 2026, "status": "observed",
            "period_start": "2026-08-31", "period_end": "2026-08-31",
            "evidence_ref": "native:r100:stock", "source": "RIKSGALDEN_DEBT",
            "source_url": "https://example.test/original.pdf", "release_id": 100,
            "available_at": "2026-09-10T20:59:01+00:00",
            "dimensions": {"scope": "central_government"},
        }]
    result = build(changed(evidence(), add_native))
    country = result["countries"][0]
    assert country["baseline"]["gov_debt_pct_gdp"]["value"] == 80
    assert country["national_debt_context"][0]["unit"] == "SEK"
    assert result["citations"]["native:r100:stock"]["country"] == "SE"


def test_future_native_outcome_cannot_enter_earlier_assessment_date():
    def add_native(bundle):
        bundle["countries"][0]["national_debt_context"] = [{
            "metric": "central_gov_gross_debt_sek", "value": 1.0,
            "unit": "SEK", "year": 2026, "status": "observed",
            "period_start": "2026-09-30", "period_end": "2026-09-30",
            "evidence_ref": "native:r100:stock",
            "available_at": "2026-09-10T20:59:01+00:00",
        }]
    result = build(changed(evidence(), add_native))
    assert result["countries"][0]["national_debt_context"] == []


def test_scenario_cites_horizon_years_and_names_missing_reference_years():
    def extend(bundle):
        series = bundle["countries"][0]["series"][0]
        row = deepcopy(series["observations"][1])
        row.update(year=2027, date="2027-12-31", value=1.8)
        series["observations"].append(row)
    result = build(changed(evidence(), extend))
    cases = {case["id"]: case for case in result["countries"][0]["scenarios"]}
    for case_id in ("demand_shortfall", "stronger_activity"):
        case = cases[case_id]
        assert "2027: 1.80%" in case["assumptions"][0]
        assert "2028: unavailable" in case["assumptions"][0]
        assert any(result["citations"][ref]["year"] == 2027 for ref in case["evidence_refs"])
    assert any("external-demand" in limit for limit in cases["demand_shortfall"]["limitations"])


def test_no_stronger_activity_case_without_any_forecast_in_its_window():
    cases = build(evidence())["countries"][0]["scenarios"]
    assert not any(case["id"] == "stronger_activity" for case in cases)
