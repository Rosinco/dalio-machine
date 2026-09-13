"""Fixed within-branch refinement checks independent of company outcomes."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "within_branch", Path(__file__).parents[1] / "scripts/cash-flow-segmentation-within-branch.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def frame():
    value = pd.DataFrame({
        "company_id": list("abcdefgh"), "training_history_fingerprint": list("ABCDEFGH"),
        "sector": ["Energy"] * 8, "branch": ["Energy / Oil"] * 6 + ["Energy / Tiny"] * 2,
        "tangible_assets_sales": ["low"] * 3 + ["high"] * 3 + ["low"] * 2,
        "tbv_ebit": ["under 3 years"] * 8, "ebit_margin": ["5–20%"] * 8,
        "cash_residual_dispersion": ["low"] * 8, "error": [1., 2., 3., 4., 5., 6., 10., 20.],
        "signed_error": [1., 2., 3., 4., 5., 6., 10., 20.], "target_year": [2020] * 8,
        "calibration_ok": [True] * 8,
    })
    return MODULE.add_refinements(value)


def test_sparse_and_missing_refinement_use_branch_then_sector_then_global():
    helper = MODULE.load_helpers()
    factors, _ = MODULE.fit_calibration(frame(), helper, minimum_rows=3, minimum_ids=3, minimum_histories=3)
    target = frame().iloc[:4].copy()
    target["sector"] = ["Energy", "Energy", "Energy", "Other"]
    target["branch"] = ["Energy / Oil", "Energy / Oil", "Energy / Tiny", "Other / New"]
    target["tangible_assets_sales"] = ["low", "", "high", "low"]
    target = MODULE.add_refinements(target)
    values, levels = helper.apply_factors(target, "branch_tangible_assets_sales", factors)
    assert list(values) == [3., 5., 10., 10.]
    assert list(levels) == ["branch_tangible_assets_sales", "branch", "sector", "global"]
    assert len(values) == len(target)


def test_evaluation_outcome_poison_cannot_change_calibration_or_support():
    helper = MODULE.load_helpers()
    evaluation = frame().iloc[:2].assign(calibration_ok=False, target_year=2025)
    combined = pd.concat([frame(), evaluation], ignore_index=True)
    first, support = MODULE.fit_calibration(combined, helper, minimum_rows=3, minimum_ids=3, minimum_histories=3)
    combined.loc[~combined.calibration_ok, "error"] = 1e30
    second, other_support = MODULE.fit_calibration(combined, helper, minimum_rows=3, minimum_ids=3, minimum_histories=3)
    assert first == second
    assert support == other_support


def test_branch_relative_comparison_retains_exact_rows_and_baseline_zero():
    helper = MODULE.load_helpers()
    cal = frame()
    factors, _ = MODULE.fit_calibration(cal, helper, minimum_rows=3, minimum_ids=3, minimum_histories=3)
    evaluation = cal.assign(target_year=2025)
    rows = MODULE.score_period(evaluation, factors, helper, duplicate_aware=False)
    baseline = next(row for row in rows if row["candidate"] == "branch")
    assert baseline["scoreImprovement"] == 0
    for row in rows:
        assert row["count"] == len(evaluation)
        assert row["baselineCandidate"] == "branch"
        assert row["branchScore"] == baseline["meanIntervalScore"]
        assert np.isclose(row["scoreImprovement"], 1 - row["meanIntervalScore"] / baseline["meanIntervalScore"])
        assert np.isclose(sum(row["fallbackLevelShares"].values()), 1)


def test_bins_are_copied_and_missing_is_not_asset_light():
    value = frame()
    original = value["tangible_assets_sales"].copy()
    value.loc[0, "tangible_assets_sales"] = ""
    output = MODULE.add_refinements(value)
    assert output.loc[0, "branch_tangible_assets_sales"] == ""
    assert output.loc[1, "branch_tangible_assets_sales"].endswith(" / low")
    assert list(output["tangible_assets_sales"].iloc[1:]) == list(original.iloc[1:])


def test_history_support_minimum_prevents_share_classes_creating_support():
    helper = MODULE.load_helpers()
    value = frame()
    value.loc[:2, "training_history_fingerprint"] = "duplicate"
    factors, support = MODULE.fit_calibration(value, helper, minimum_rows=3, minimum_ids=3, minimum_histories=3)
    group = factors["branch_tangible_assets_sales"]["Energy / Oil / low"]
    assert group["count"] == 3
    assert group["listings"] == 3
    assert group["histories"] == 1
    assert group["factor"] is None
    assert not group["supported"]
    assert support["branch_tangible_assets_sales"]["supportedGroups"] == 1
