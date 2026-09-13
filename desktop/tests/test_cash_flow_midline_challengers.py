"""Forecast challengers preserve chronology, signed cash and paired denominators."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SPEC = importlib.util.spec_from_file_location(
    "cash_midline_challengers", Path(__file__).parents[1] / "scripts/cash-flow-midline-challengers.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_fixed_candidates_have_independent_hand_calculated_values():
    predictions, scale, slope = MODULE.predict_candidates(np.array([[5., 4., 3., 2., 1.]]), np.array([3]))
    assert predictions["latest"][0] == 5
    assert predictions["robust_blend"][0] == 4
    assert predictions["damped_trend"][0] == pytest.approx(5.875)
    assert scale[0] == 3
    assert slope[0] == pytest.approx(1)


def test_geometric_damping_keeps_origin_level_and_has_a_finite_trend_limit():
    cash = np.tile([105., 104., 103., 102., 101.], (5, 1))
    predictions, _, _ = MODULE.predict_candidates(cash, np.arange(1, 6))
    assert predictions["damped_trend"] == pytest.approx([105.5, 105.75, 105.875, 105.9375, 105.96875])


def test_robust_blend_reduces_single_latest_spike_without_trimming_raw_history():
    cash = np.array([[100., 10., 10., 10., 10.]])
    original = cash.copy()
    predictions, scale, _ = MODULE.predict_candidates(cash, np.array([1]))
    assert predictions["robust_blend"][0] == 55
    assert scale[0] == 28
    np.testing.assert_array_equal(cash, original)


def test_negative_zero_and_near_zero_cash_are_not_clipped_or_floored():
    cash = np.array([[-5., -4., -3., -2., -1.], [0., 0., 0., 0., 0.], [1e-12] * 5])
    predictions, scale, _ = MODULE.predict_candidates(cash, np.ones(3))
    assert predictions["robust_blend"][0] == -4
    assert predictions["damped_trend"][0] == pytest.approx(-5.5)
    assert predictions["latest"][1] == scale[1] == 0
    assert scale[2] == 1e-12


@pytest.mark.parametrize("cash,horizon", [
    ([[1, 2, 3, 4]], [1]), ([[1, 2, 3, 4, np.nan]], [1]),
    ([[1, 2, 3, 4, np.inf]], [1]), ([[1, 2, 3, 4, 5]], [0]),
    ([[1, 2, 3, 4, 5]], [6]), ([[1, 2, 3, 4, 5]], [1.5]),
])
def test_invalid_histories_or_horizons_fail_closed(cash, horizon):
    with pytest.raises(ValueError):
        MODULE.predict_candidates(np.array(cash, dtype=float), np.array(horizon))


def dates():
    return {
        "origin_publication": "2021-03-01", "target_end": "2021-12-31", "target_publication": "2022-03-01",
        "train_publications": "2021-03-01|2020-03-01|2019-03-01|2018-03-01|2017-03-01",
        "train_end_dates": "2020-12-31|2019-12-31|2018-12-31|2017-12-31|2016-12-31",
    }


def test_publication_gate_prevents_future_training_but_discloses_missing_target_date():
    row = dates()
    assert MODULE.timing_issue(row) == ""
    row["target_publication"] = ""
    assert MODULE.timing_issue(row) == ""
    row["train_publications"] = row["train_publications"].replace("2020-03-01", "2022-03-01")
    assert MODULE.timing_issue(row) == "training_publication_order"
    row = dates()
    row["origin_publication"] = row["target_end"]
    assert MODULE.timing_issue(row) == "origin_not_before_target_end"


def test_target_poisoning_does_not_change_predictions_or_scale():
    frame = pd.DataFrame({"train_cash": ["5|4|3|2|1"], "horizon": [2], "actual": [7.]})
    first = MODULE.predict_candidates(np.array([list(map(float, frame.train_cash[0].split("|")))]), frame.horizon.to_numpy())
    frame["actual"] = 1e100
    second = MODULE.predict_candidates(np.array([list(map(float, frame.train_cash[0].split("|")))]), frame.horizon.to_numpy())
    for key in first[0]:
        np.testing.assert_array_equal(first[0][key], second[0][key])
    np.testing.assert_array_equal(first[1], second[1])


def test_listing_weighting_and_duplicate_sensitivity_use_the_same_paired_rows():
    frame = pd.DataFrame({"company_id": ["a", "a", "b"], "target_year": [2024, 2025, 2024],
                          "cash_history_fingerprint": ["x", "y", "x"]})
    np.testing.assert_array_equal(MODULE.pool_weights(frame, False), [.5, .5, 1])
    np.testing.assert_array_equal(MODULE.pool_weights(frame, True), [.25, .5, .5])


def test_metrics_use_one_shared_denominator_and_keep_ties():
    frame = pd.DataFrame({"company_id": ["a", "b"], "target_year": [2024, 2024],
                          "cash_history_fingerprint": ["x", "y"], "scale": [2., 4.],
                          "actual": [0., 4.], "latest": [4., 4.], "robust_blend": [2., 4.]})
    result = MODULE.metrics(frame, "robust_blend", np.ones(2))
    assert result["meanNormalizedError"] == .5
    assert result["baselineMeanNormalizedError"] == 1
    assert result["meanErrorImprovementPercent"] == 50
    assert result["betterShare"] == result["tieShare"] == .5
    assert result["worseShare"] == 0


def test_input_checksum_rejection_precedes_research_processing(tmp_path):
    path = tmp_path / "forecasts.csv.gz"
    path.write_bytes(b"example")
    receipt = {"artifacts": [{"name": path.name, "sha256": "0" * 64}]}
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="checksum"):
        MODULE.verify_inputs(tmp_path)


def test_output_guard_prevents_overwriting_research_or_source():
    with pytest.raises(ValueError):
        MODULE.check_output(Path("/tmp/research"), Path("/tmp/research"))
    with pytest.raises(ValueError):
        MODULE.check_output(Path("/tmp/research"), Path("/tmp/research/nested"))
