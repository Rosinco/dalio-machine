"""Independent checks of the fixed within-branch uncertainty refinements."""

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

REFINEMENTS = {
    "branch_tangible_assets_sales": "tangible_assets_sales",
    "branch_tbv_ebit": "tbv_ebit",
    "branch_ebit_margin": "ebit_margin",
    "branch_cash_residual_dispersion": "cash_residual_dispersion",
}


def run(directory):
    helper_path = Path(__file__).with_name("verify-cash-flow-segmentation.py")
    spec = importlib.util.spec_from_file_location("independent_scalar_verifier", helper_path)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.CANDIDATES = ["global", "sector", "branch", *REFINEMENTS]
    within = json.loads((directory / "within-branch-results.json").read_text())
    main = json.loads((directory / "results.json").read_text())
    factors = json.loads((directory / "within-branch-calibration.json").read_text())["calibrations"]
    main_factors = json.loads((directory / "group-calibration.json").read_text())["calibrations"]
    receipt = json.loads((directory / "within-branch-receipt.json").read_text())
    for artifact in receipt["artifacts"]:
        assert helper.sha256(directory / artifact["name"]) == artifact["sha256"]
    for key in ["preparedForecasts", "preparedMetadata", "helperScript"]:
        item = within["inputs"][key]
        assert helper.sha256(Path(item["path"])) == item["sha256"], key

    identity_keys = ["scope", "model", "horizon", "weighting", "period"]
    paired = defaultdict(list)
    for metric in within["metrics"]:
        identity = tuple(metric[key] for key in identity_keys)
        paired[identity].append(metric)
        assert metric["inside"] + metric["below"] + metric["above"] == metric["count"]
        helper.close(sum(metric["fallbackLevelShares"].values()), 1., "fallback partition")
    baseline_checks = 0
    for identity, rows in paired.items():
        baseline = next(row for row in main["metrics"] if tuple(row[key] for key in identity_keys) == identity and row["candidate"] == "branch")
        branch = next(row for row in rows if row["candidate"] == "branch")
        for key in ["coverage", "meanWidth", "meanIntervalScore", "fallbackShare"]:
            helper.close(branch[key], baseline[key], "independent main branch baseline")
        for row in rows:
            assert row["count"] == baseline["count"] and row["listings"] == baseline["listings"]
            helper.close(row["branchCoverage"], baseline["coverage"], "paired coverage")
            helper.close(row["branchWidth"], baseline["meanWidth"], "paired width")
            helper.close(row["branchScore"], baseline["meanIntervalScore"], "paired score")
            helper.close(row["scoreImprovement"], 1 - row["meanIntervalScore"] / baseline["meanIntervalScore"], "paired score improvement")
        baseline_checks += 1
    for cell in factors:
        original = next(row for row in main_factors if all(row[key] == cell[key] for key in identity_keys[:-1]))
        for candidate in ["global", "sector", "branch"]:
            assert cell["groups"][candidate] == original["groups"][candidate]

    columns = ["company_id", "model", "horizon", "in_operating_scope", "target_year",
               "training_history_fingerprint", "calibration_ok", "application_ok", "error",
               "branch", "sector", *REFINEMENTS.values()]
    frame = pd.read_parquet(directory / "prepared-forecasts.parquet", columns=columns)
    frame = frame.loc[frame.in_operating_scope & frame.model.eq("linear") & frame.horizon.eq(1)]
    rows = frame.to_dict("records")
    for row in rows:
        for candidate, feature in REFINEMENTS.items():
            row[candidate] = row["branch"] + " / " + row[feature] if row["branch"] and row[feature] else ""

    scalar_factors_checked, scalar_cells_checked = 0, 0
    for weighting in ["listing_balanced", "duplicate_history_downweighted"]:
        duplicate = weighting != "listing_balanced"
        training = [row for row in rows if row["calibration_ok"]]
        scalar_factors = helper.factors_for(training, helper.weights_for(training, duplicate))
        saved = next(row for row in factors if row["model"] == "linear" and row["horizon"] == 1 and row["weighting"] == weighting)
        for candidate in REFINEMENTS:
            assert set(scalar_factors[candidate]) == set(saved["groups"][candidate])
            for key, value in scalar_factors[candidate].items():
                expected = saved["groups"][candidate][key]["factor"]
                if value is None:
                    assert expected is None
                else:
                    helper.close(value, expected, "independent refinement factor")
                scalar_factors_checked += 1

        def choose(row, candidate, selected_factors=scalar_factors):
            key = "all" if candidate == "global" else row[candidate]
            value = selected_factors[candidate].get(key)
            if value is not None:
                return value, candidate
            parent = "branch" if candidate in REFINEMENTS else "sector" if candidate == "branch" else "global"
            return choose(row, parent, selected_factors)

        for period, years in [("validation", [2022, 2023]), ("recent", [2024, 2025])]:
            selected = [row for row in rows if row["application_ok"] and row["target_year"] in years]
            weights = helper.weights_for(selected, duplicate)
            total = sum(weights)
            for candidate in REFINEMENTS:
                coverage, width, score, fallback = 0., 0., 0., 0.
                for row, weight in zip(selected, weights, strict=True):
                    factor, level = choose(row, candidate)
                    coverage += weight * (row["error"] <= factor)
                    width += weight * 2 * factor
                    score += weight * (2 * factor + 10 * max(row["error"] - factor, 0))
                    fallback += weight * (level != candidate)
                expected = next(row for row in within["metrics"] if row["model"] == "linear" and row["horizon"] == 1
                                and row["weighting"] == weighting and row["period"] == period and row["candidate"] == candidate)
                for key, value in [("coverage", coverage), ("meanWidth", width), ("meanIntervalScore", score), ("fallbackShare", fallback)]:
                    helper.close(value / total, expected[key], "independent refinement metric")
                scalar_cells_checked += 1
    output = {"status": "passed", "pairedMetricRows": len(within["metrics"]),
              "mainBranchBaselinesReconciled": baseline_checks,
              "independentLinearH1RefinementFactorsChecked": scalar_factors_checked,
              "independentLinearH1SummaryCellsChecked": scalar_cells_checked,
              "resultsSha256": helper.sha256(directory / "within-branch-results.json"),
              "validatorSha256": helper.sha256(Path(__file__)), "scalarHelperSha256": helper.sha256(helper_path),
              "limits": "All candidate rows use the same verified branch baseline. Independent scalar recalculation covers four refinements for linear h1, both weighting schemes and combined validation/recent periods; other cells receive paired baseline/score/receipt checks."}
    (directory / "within-branch-independent-validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
