"""Meaningful comparison, scope, missingness and cutoff guards for the pilot."""

from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

import pytest

from dalio.assessments.core import build_snapshot, content_hash
from dalio.monitoring.core import build_monitoring, validate_monitoring

KNOWN = "2026-09-10T23:00:00+00:00"
AS_OF = date(2026, 9, 10)


def assessment():
    series = []
    for i, indicator in enumerate(("real_gdp_growth", "gov_debt_pct_gdp",
                                   "fiscal_balance_pct_gdp", "primary_balance_pct_gdp",
                                   "current_account_pct_gdp"), 1):
        source = "IMF_FISCAL_MONITOR" if indicator == "primary_balance_pct_gdp" else "IMF_WEO"
        series.append({"indicator": indicator, "family": "imf", "source": source,
                       "series_id": indicator, "release_id": i,
                       "partition_key": f"{source}:{indicator}:SE", "published_at": None,
                       "available_at": KNOWN, "retrieved_at": KNOWN,
                       "source_url": f"https://example.test/{indicator}",
                       "publisher_metadata": {}, "artifacts": [], "missingness": {},
                       "observations": [{"year": year, "date": f"{year}-12-31",
                                         "value": 2.0, "source": source,
                                         "status": "estimate_or_outturn" if year == 2025
                                         else "forecast_calendar_convention"}
                                        for year in range(2025, 2032)]})
    result = {"as_known_at": KNOWN, "protected_artifact_paths": [],
              "countries": [{"country": "SE", "listing_iso2": "SE", "name": "Sweden",
                             "listing_count": 1050, "series": series, "gaps": []}]}
    result["evidence_digest"] = content_hash(result)
    return build_snapshot(result, as_of=AS_OF)


def point(period, value, *, status="observed"):
    if len(period) == 7:
        year, month = map(int, period.split("-"))
        end = date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)
    else:
        end = date.fromisoformat(period)
    return {"period": period, "date": end.isoformat(), "value": value, "status": status}


def data():
    series = []
    for key in ("industrial_production", "industrial_orders"):
        series.append({"indicator": key, "source": "SCB", "series_id": key,
                       "unit": "index, 2021=100", "frequency": "monthly",
                       "adjustment": "calendar_and_seasonally_adjusted",
                       "definition": "Industry total; seasonally adjusted constant-price index.",
                       "observations": [point(f"2026-{m:02d}", 100 if m < 5 else 110)
                                        for m in range(2, 8)]})
    series.append({"indicator": "corporate_new_lending_rate", "source": "SCB",
                   "series_id": "nfc-rate", "unit": "% per annum", "frequency": "monthly",
                   "definition": "Non-financial corporations; new and renegotiated agreements.",
                   "observations": [point("2026-04", 4.5), point("2026-07", 3.5)]})
    for key in ("policy_rate", "yield_10y"):
        series.append({"indicator": key, "source": "RIKSBANK_SWEA", "series_id": key,
                       "unit": "%", "frequency": "daily", "definition": key,
                       "observations": [point("2026-06-09", 3), point("2026-09-08", 2)]})
    for item in series:
        item.update(available_at=KNOWN, retrieved_at=KNOWN, published_at=None,
                    source_url="https://example.test/series", artifacts=[], publisher_metadata={})
    result = {"country": "SE", "as_known_at": KNOWN, "series": series, "gaps": [],
              "national_debt_context": [], "protected_artifact_paths": []}
    return seal(result)


def seal(value):
    value["evidence_digest"] = content_hash({k: v for k, v in value.items()
                                               if k != "evidence_digest"})
    return value


def build(value=None):
    return build_monitoring(data() if value is None else seal(value), assessment(), as_of=AS_OF)


def signal(result, key):
    return next(item for item in result["signals"] if item["indicator"] == key)


def test_industrial_momentum_uses_complete_adjacent_three_month_windows():
    item = signal(build(), "industrial_production")
    assert item["comparison"]["value"] == pytest.approx(10)
    assert item["comparison"]["unit"] == "%"
    assert len(item["comparison"]["evidence_refs"]) == 6
    assert item["direction"] == "up"
    assert "annual" in " ".join(item["limits"]).lower()
    assert "gdp" in " ".join(item["limits"]).lower()


def test_missing_month_is_not_interpolated_or_replaced_by_an_older_month():
    value = data()
    value["series"][0]["observations"].pop(2)
    item = signal(build(value), "industrial_production")
    assert item["latest"]["value"] == 110
    assert item["comparison"] is None
    assert "2026-04" in " ".join(item["gaps"])
    assert item["direction"] == "unavailable"


def test_latest_native_null_stays_unavailable():
    value = data()
    value["series"][0]["observations"][-1].update(value=None, status="not_reported")
    item = signal(build(value), "industrial_production")
    assert item["latest"]["value"] is None
    assert item["latest"]["period"] == "2026-07"
    assert item["status"] == "missing_latest"
    assert item["comparison"] is None


def test_unadjusted_indices_are_not_used_for_short_term_momentum():
    value = data()
    value["series"][0]["adjustment"] = "unadjusted"
    item = signal(build(value), "industrial_production")
    assert item["comparison"] is None
    assert any("adjust" in gap.lower() for gap in item["gaps"])


def test_lending_rate_change_is_percentage_points_and_scope_survives():
    item = signal(build(), "corporate_new_lending_rate")
    assert item["comparison"]["value"] == -1
    assert item["comparison"]["unit"] == "percentage points"
    assert "renegotiated" in item["definition"]
    assert any("composition" in text.lower() for text in item["limits"])


def test_lending_comparison_needs_exact_target_month():
    value = data()
    value["series"][2]["observations"][0] = point("2026-03", 4.5)
    item = signal(build(value), "corporate_new_lending_rate")
    assert item["comparison"] is None
    assert "2026-04" in " ".join(item["gaps"])


def test_daily_anchor_uses_only_date_at_or_before_target_with_tolerance():
    item = signal(build(), "yield_10y")
    assert item["comparison"]["value"] == -1
    assert item["comparison"]["anchor_date"] == "2026-06-09"
    value = data()
    value["series"][4]["observations"][0] = point("2026-06-11", 3)
    assert signal(build(value), "yield_10y")["comparison"] is None
    value["series"][4]["observations"][0] = point("2026-05-31", 3)
    assert signal(build(value), "yield_10y")["comparison"] is None


def test_stale_values_are_displayed_without_a_current_directional_reading():
    value = data()
    value["series"][3]["observations"][-1] = point("2026-08-01", 2)
    item = signal(build(value), "policy_rate")
    assert item["status"] == "stale"
    assert item["latest"]["value"] == 2
    assert item["direction"] == "unavailable"


def test_known_at_and_reference_dates_do_not_leak_future_values():
    value = data()
    value["series"][3]["observations"].append(point("2026-09-11", 9))
    assert signal(build(value), "policy_rate")["latest"]["date"] == "2026-09-08"
    value["series"][3]["available_at"] = "2026-09-11T00:00:00+00:00"
    with pytest.raises(ValueError, match="cutoff"):
        build(value)


def test_missing_series_does_not_read_legacy_values_from_gap_metadata():
    value = data()
    value["series"] = value["series"][:2]
    value["gaps"].append({"indicator": "policy_rate", "reason": "raw_response_not_bound",
                           "latest_value": 100})
    item = signal(build(value), "policy_rate")
    assert item["latest"] is None
    assert item["status"] == "unavailable"


def test_reject_duplicate_periods_nonfinite_and_forecast_monitoring_cells():
    for mutate in (
        lambda rows: rows.append(deepcopy(rows[-1])),
        lambda rows: rows[-1].update(value=float("inf")),
        lambda rows: rows[-1].update(status="forecast"),
    ):
        value = data()
        mutate(value["series"][0]["observations"])
        with pytest.raises(ValueError):
            build(value)


def test_snapshot_reproducible_source_immutable_and_citations_resolve():
    value = data()
    original = deepcopy(value)
    a, b = build(value), build(value)
    assert a == b
    assert value == original
    assert a["coverage"]["signals_with_comparison"] == 5
    assert "probability" not in a
    for item in a["signals"]:
        assert set(item["evidence_refs"]) <= a["citations"].keys()
        assert all(link["scenario_id"] in {s["id"] for s in a["scenarios"]}
                   for link in item["scenario_links"])
    a["signals"][0]["latest"]["value"] = 999
    with pytest.raises(ValueError, match="hash"):
        validate_monitoring(a)


def test_mismatched_assessment_clock_and_unknown_input_rejected():
    parent = assessment()
    parent["as_known_at"] = "2026-09-11T00:00:00+00:00"
    parent["snapshot_sha256"] = content_hash({k: v for k, v in parent.items()
                                              if k != "snapshot_sha256"})
    with pytest.raises(ValueError, match="cutoff"):
        build_monitoring(data(), parent, as_of=AS_OF)
    value = data()
    value["series"][0]["indicator"] = "unknown_input"
    with pytest.raises(ValueError, match="indicator"):
        build(value)


def test_rate_values_in_basis_points_cannot_be_mislabelled_percentage_points():
    value = data()
    value["series"][2]["unit"] = "basis points"
    with pytest.raises(ValueError, match="unit"):
        build(value)


def test_original_scb_fixtures_integrate_with_exact_units_periods_and_scope():
    from dalio.data_sources.scb_monitoring import SCB_MONITORING_SPECS, parse_scb_input

    value = data()
    root = Path(__file__).parent / "fixtures/scb_monitoring"
    for index, spec in enumerate(SCB_MONITORING_SPECS):
        parsed = parse_scb_input(spec.key, (root / f"{spec.key}_metadata.json").read_bytes(),
                                 (root / f"{spec.key}_response.json").read_bytes(), spec.request)
        parsed.update(available_at=KNOWN, retrieved_at=KNOWN, published_at=None, artifacts=[])
        value["series"][index] = parsed
    result = build(value)
    assert signal(result, "industrial_production")["comparison"]["value"] == pytest.approx(1.3285024155)
    assert signal(result, "industrial_orders")["comparison"]["value"] == pytest.approx(9.7779250912)
    rate = signal(result, "corporate_new_lending_rate")
    assert rate["latest"]["period"] == "2026-07"
    assert rate["latest"]["value"] == 3.5912
    assert rate["comparison"]["value"] == pytest.approx(0.0805)
    assert rate["source"]["published_at"] is None
    assert rate["source"]["publisher_updated_at"] is not None
    assert "SEK" in rate["definition"] and "renegotiated" in rate["definition"]
