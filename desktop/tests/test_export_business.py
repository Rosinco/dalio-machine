"""Financial contracts for the selective offline export; no API or source writes."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "business_model", Path(__file__).parents[1] / "scripts" / "business_model.py"
)
model = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model)


def row(**changes):
    return {
        "year": 2025,
        "period": 5,
        "report_start_date": "2025-01-01",
        "report_end_date": "2025-12-31",
        "report_date": "2026-01-30",
        "currency": "EUR",
        "currency_ratio": 2,
        "revenues": 200,
        "operating_income": 20,
        "total_equity": 80,
        "net_debt": 20,
        "total_assets": 160,
        "cash_flow_from_operating_activities": 0,
        "free_cash_flow": None,
        **changes,
    }


def test_native_currency_divides_by_ratio_and_preserves_zero_missing():
    report = model.report(row(), "2026-08-10")
    assert report["values"]["revenues"] == 100
    assert report["values"]["operating_margin"] == 10
    assert report["values"]["return_on_capital"] == 20
    assert report["values"]["cash_flow_from_operating_activities"] == 0
    assert report["values"]["free_cash_flow"] is None
    assert report["raw"]["revenues"] == 200


@pytest.mark.parametrize("ratio", [None, 0, -1, float("nan")])
def test_invalid_fx_is_missing_not_a_fallback(ratio):
    result = model.report(row(currency_ratio=ratio), "2026-08-10")
    assert all(v is None for v in result["values"].values())


def test_negative_equity_and_zero_revenue_do_not_create_ratios():
    values = model.report(row(total_equity=-40, net_debt=20, revenues=0), "2026-08-10")["values"]
    assert values["revenues"] == 0
    assert values["return_on_capital"] is None
    assert values["net_debt_to_equity"] is None
    assert values["operating_margin"] is None


def test_quarters_are_not_annualized_and_future_reports_are_rejected():
    result = model.report(row(period=1, report_end_date="2025-03-31"), "2026-08-10")
    assert result["period"] == 1
    assert result["values"]["return_on_capital"] is None
    with pytest.raises(ValueError, match="after snapshot"):
        model.report(row(report_date="2026-08-11"), "2026-08-10")


def test_duplicates_and_period_type_mix_are_rejected():
    with pytest.raises(ValueError, match="Duplicate"):
        model.reports([row(), row()], "2026-08-10", annual=True)
    with pytest.raises(ValueError, match="period"):
        model.reports([row(period=4)], "2026-08-10", annual=True)


def test_light_catalogue_uses_common_period_and_excludes_histories_and_prose():
    annual = model.reports(
        [row(), row(year=2024, report_start_date="2024-01-01", report_end_date="2024-12-31")],
        "2026-08-10",
        annual=True,
    )
    raw = {
        "version": 1,
        "companies": {
            "1": {
                "annual": annual,
                "quarterly": [],
                "research": [{"text": "archive"}],
                "name": "One",
            },
            "2": {"annual": deepcopy(annual[:1]), "quarterly": [], "research": [], "name": "Two"},
        },
        "research": [{"text": "branch archive"}],
    }
    index = model.business_index(raw)
    assert index["common_year"] == 2024
    assert index["companies"]["1"]["comparison"]["year"] == 2024
    assert "annual" not in index["companies"]["1"]
    assert "research" not in index
    raw["companies"]["2"]["annual"][0]["end"] = "2024-11-30"
    assert model.business_index(raw)["common_year"] is None
