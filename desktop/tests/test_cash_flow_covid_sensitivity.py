"""The omission changes calibration only, and recomputes listing weights."""

import importlib.util
from pathlib import Path

import pandas as pd

SPEC = importlib.util.spec_from_file_location("covid_sensitivity", Path(__file__).parents[1] / "scripts/cash-flow-covid-sensitivity.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sample():
    return pd.DataFrame({"company_id": ["a", "a", "b", "a", "b"],
                         "target_year": [2019, 2020, 2020, 2022, 2025],
                         "calibration_ok": [True, True, True, False, False],
                         "application_ok": [False, False, False, True, True],
                         "error": [1., 9., 100., 1e9, 1e10]})


def test_omission_removes_only_2020_calibration_and_reweights_remaining_listings():
    helper = MODULE.load_helper()
    before, weights = MODULE.calibration_sample(sample(), helper)
    after, new_weights = MODULE.calibration_sample(sample(), helper, True)
    assert list(before.target_year) == [2019, 2020, 2020]
    assert list(weights) == [.5, .5, 1.]
    assert list(after.target_year) == [2019]
    assert list(after.company_id) == ["a"]
    assert list(new_weights) == [1.]


def test_omission_does_not_remove_or_change_future_evaluation_rows():
    cell = sample()
    before = MODULE.evaluation_samples(cell)
    MODULE.calibration_sample(cell, MODULE.load_helper(), True)
    after = MODULE.evaluation_samples(cell)
    for (name, left), (other, right) in zip(before, after, strict=True):
        assert name == other
        pd.testing.assert_frame_equal(left, right)
    assert list(dict(after)["recent"].target_year) == [2025]


def test_future_error_poisoning_does_not_change_either_calibration_sample():
    original = sample()
    poisoned = original.copy()
    poisoned.loc[poisoned.application_ok, "error"] = -1e300
    helper = MODULE.load_helper()
    for omit in [False, True]:
        left, weights = MODULE.calibration_sample(original, helper, omit)
        right, other = MODULE.calibration_sample(poisoned, helper, omit)
        pd.testing.assert_frame_equal(left, right)
        assert list(weights) == list(other)
