"""Boundary and leakage checks for the exploratory uncertainty experiment."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

SPEC = importlib.util.spec_from_file_location(
    "segmentation", Path(__file__).parents[1] / "scripts/cash-flow-segmentation.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_quantile_preserves_mass_when_a_listing_is_duplicated():
    assert MODULE.weighted_quantile([1, 2, 9], [1, 1, 1], .8) == 9
    assert MODULE.weighted_quantile([1, 2, 9, 9], [1, 1, .5, .5], .8) == 9
    assert MODULE.weighted_quantile([1, 2, 9], [1, 1, .01], .8) == 2


def test_support_failure_and_missing_features_fall_back_without_dropping_rows():
    frame = pd.DataFrame({
        "company_id": ["a", "b", "c", "d"],
        "training_history_fingerprint": ["aa", "bb", "cc", "dd"],
        "error": [1., 2., 3., 100.], "sector": ["Energy"] * 4,
        "branch": ["Oil"] * 3 + ["Tiny"],
    })
    frame = MODULE.add_groups(frame)
    factors = MODULE.fit_factors(frame, np.ones(4), minimum_rows=3, minimum_ids=3, minimum_histories=3)
    target = MODULE.add_groups(pd.DataFrame({
        "company_id": ["x", "y", "z"], "sector": ["Energy", "Energy", "New"],
        "branch": ["Oil", "Tiny", None],
    }))
    q, levels = MODULE.apply_factors(target, "branch", factors)
    assert len(q) == 3
    assert list(levels) == ["branch", "sector", "global"]
    assert list(q) == [3., 100., 100.]


def test_calibration_never_uses_late_outcomes_or_invalid_publications():
    row = {
        "target_year": 2020, "target_publication": "2021-02-01",
        "target_end": "2020-12-31", "origin_publication": "2020-02-01",
        "train_publications": "2020-02-01|2019-02-01|2018-02-01|2017-02-01|2016-02-01",
        "train_end_dates": "2019-12-31|2018-12-31|2017-12-31|2016-12-31|2015-12-31",
    }
    assert MODULE.calibration_eligible(row)
    assert not MODULE.calibration_eligible({**row, "target_year": 2025})
    assert not MODULE.calibration_eligible({**row, "target_publication": "2021-07-01"})
    assert not MODULE.calibration_eligible({**row, "target_publication": "2020-12-01"})
    assert not MODULE.calibration_eligible({**row, "train_publications": "2020-02-01|2022-01-01|2018-02-01|2017-02-01|2016-02-01"})
    assert not MODULE.calibration_eligible({**row, "train_publications": "2020-02-01||2018-02-01|2017-02-01|2016-02-01"})


def test_property_stays_in_operating_scope_but_lenders_do_not():
    frame = pd.DataFrame({"sector": ["Financials", "Financials", "Financials", "Energy"],
                          "branch": ["Real Estate", "REITs", "Banks", "Oil"]})
    assert list(MODULE.scope_mask(frame, "operating_and_property")) == [True, True, False, True]


def test_interval_score_penalizes_width_and_keeps_negative_cash_and_misses():
    frame = pd.DataFrame({"company_id": ["a", "b"], "error": [0., 2.],
                          "signed_error": [0., -2.], "target_year": [2025, 2025],
                          "training_history_fingerprint": ["a", "b"], "actual": [-1., -3.]})
    metrics = MODULE.metrics(frame, np.array([1., 1.]), np.ones(2), np.array(["global", "global"]), "global")
    assert metrics["coverage"] == .5
    assert metrics["meanWidth"] == 2
    assert metrics["meanIntervalScore"] == 7
    assert metrics["below"] == 1


def test_unknown_numeric_features_remain_missing_and_do_not_become_zero():
    frame = MODULE.add_groups(pd.DataFrame({"sector": ["Energy", "Energy"], "branch": ["Oil", "Oil"],
                                            "tangible_assets_to_revenue": [None, 0.]}))
    assert frame["tangible_assets_sales"].iloc[0] == ""
    assert frame["tangible_assets_sales"].iloc[1] == "low"


def test_duplicate_history_weighting_retains_conflicting_outcomes():
    frame = pd.DataFrame({"company_id": ["a", "b", "c"], "target_year": [2025]*3,
                          "training_history_fingerprint": ["same", "same", "different"],
                          "actual": [10., -100., 2.]})
    weights = MODULE.pool_weights(frame, duplicate_aware=True)
    assert list(weights) == [.5, .5, 1.]
    assert len(frame) == 3
