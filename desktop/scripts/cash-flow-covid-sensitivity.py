"""Exploratory omission of FY2020 calibration errors, with unchanged evaluation."""

import argparse
import importlib.util
import json
import math
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

CANDIDATES = ["global", "cash_dispersion", "cash_residual_dispersion"]
HELPER_PATH = Path(__file__).with_name("cash-flow-segmentation.py")
PROTOCOL = {
    "id": "exploratory-fy2020-calibration-omission-v1",
    "scope": "operating_and_property", "horizon": 1, "models": ["linear", "naive"],
    "weighting": "listing_balanced", "candidates": CANDIDATES,
    "scenarios": ["original", "omit_target_fy2020"], "targetCoverage": .8,
    "minimumCalibrationRows": 300, "minimumCalibrationListings": 100,
    "minimumCalibrationHistories": 100,
    "calibration": "Retain the existing publication-eligible targets through FY2020. In the omission scenario remove only target FY2020 rows; recompute each listing's total-one weights over its remaining calibration rows and refit the same supported group quantiles.",
    "evaluation": "Identical admissible FY2022, FY2023, FY2024 and FY2025 observations in both scenarios; pooled validation/recent and separate annual metrics. Evaluation weights are unchanged between scenarios.",
    "limitations": [
        "This removes FY2020 forecast errors only. Pandemic-era values remain in five-year histories and forecast inputs. FY2021 is already excluded from calibration by the original protocol.",
        "An omission changes sample size, listing support and weights as well as calendar composition. It is not a causal estimate of COVID's effect.",
        "FY labels do not identify an exact pandemic exposure window for every issuer's fiscal calendar.",
        "This is an explicitly exploratory sensitivity, not an endorsed deletion of difficult years or a replacement base model.",
        "Current-vintage statements, current taxonomy/membership, related listings, cash-definition issues and few independent calendar regimes remain limitations.",
    ],
}


def load_helper():
    spec = importlib.util.spec_from_file_location("private_covid_sensitivity_helper", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.CANDIDATES = CANDIDATES.copy()
    helper.PARENTS = {}
    return helper


def calibration_sample(cell, helper, omit_2020=False):
    mask = cell.calibration_ok
    if omit_2020:
        mask = mask & cell.target_year.ne(2020)
    selected = cell.loc[mask].reset_index(drop=True)
    return selected, helper.pool_weights(selected)


def evaluation_samples(cell):
    return [(period, cell.loc[cell.application_ok & cell.target_year.isin(years)].reset_index(drop=True))
            for period, years in [("validation", [2022, 2023]), ("recent", [2024, 2025]),
                                  ("FY2022", [2022]), ("FY2023", [2023]),
                                  ("FY2024", [2024]), ("FY2025", [2025])]]


def run(source_directory, output_directory):
    source_directory, output_directory = source_directory.resolve(), output_directory.resolve()
    if output_directory.exists():
        raise ValueError("Output directory already exists; earlier sensitivity artifacts must not be overwritten")
    helper = load_helper()
    main = json.loads((source_directory / "results.json").read_text())
    prior_calibration = json.loads((source_directory / "group-calibration.json").read_text())["calibrations"]
    main_receipt = json.loads((source_directory / "experiment-receipt.json").read_text())
    source_receipt = json.loads((source_directory / "within-branch-receipt.json").read_text())
    prepared = source_directory / "prepared-forecasts.parquet"
    if helper.file_hash(HELPER_PATH) != main_receipt["scriptSha256"]:
        raise ValueError("Helper differs from the original executed experiment")
    if helper.file_hash(prepared) != source_receipt["inputs"]["preparedForecasts"]["sha256"]:
        raise ValueError("Prepared source differs from its verified receipt")
    inputs = {"preparedForecasts": {"path": str(prepared), "sha256": helper.file_hash(prepared)},
              "mainResults": {"path": str(source_directory / "results.json"), "sha256": helper.file_hash(source_directory / "results.json")},
              "mainCalibration": {"path": str(source_directory / "group-calibration.json"), "sha256": helper.file_hash(source_directory / "group-calibration.json")},
              "helper": {"path": str(HELPER_PATH), "sha256": helper.file_hash(HELPER_PATH)},
              "script": {"path": str(Path(__file__).resolve()), "sha256": helper.file_hash(Path(__file__))}}
    output_directory.mkdir(parents=True)
    declared = {**PROTOCOL, "declaredAt": datetime.now(UTC).isoformat(), "inputs": inputs}
    (output_directory / "protocol.json").write_text(json.dumps(declared, indent=2) + "\n")
    shutil.copyfile(Path(__file__), output_directory / "cash-flow-covid-sensitivity.executed.py")
    # Protocol is saved before reading the financial/error rows.
    frame = pd.read_parquet(prepared)
    frame = frame.loc[frame.in_operating_scope & frame.horizon.eq(1)]
    metrics, calibration, samples, original_checks = [], [], [], 0
    for model in PROTOCOL["models"]:
        cell = frame.loc[frame.model.eq(model)]
        original, original_weights = calibration_sample(cell, helper)
        covid = original.target_year.eq(2020).to_numpy()
        original_ids = set(original.company_id)
        for scenario in PROTOCOL["scenarios"]:
            selected, weights = calibration_sample(cell, helper, scenario != "original")
            factors = helper.fit_factors(selected, weights)
            sample = {"model": model, "scenario": scenario, "rows": len(selected),
                      "listings": int(selected.company_id.nunique()),
                      "histories": int(selected.training_history_fingerprint.nunique()),
                      "calibrationListingsLost": len(original_ids - set(selected.company_id)),
                      "removedRows": len(original) - len(selected),
                      "originalFY2020Rows": int(covid.sum()),
                      "originalFY2020WeightShare": float(original_weights[covid].sum() / original_weights.sum())}
            samples.append(sample)
            calibration.append({"model": model, "scenario": scenario, "groups": factors})
            if scenario == "original":
                old = next(row for row in prior_calibration if row["scope"] == PROTOCOL["scope"]
                           and row["model"] == model and row["horizon"] == 1 and row["weighting"] == "listing_balanced")
                assert all(factors[candidate] == old["groups"][candidate] for candidate in CANDIDATES)
            for period, evaluation in evaluation_samples(cell):
                evaluation_weights = helper.pool_weights(evaluation)
                global_values, global_levels = helper.apply_factors(evaluation, "global", factors)
                baseline = helper.metrics(evaluation, global_values, evaluation_weights, global_levels, "global")
                for candidate in CANDIDATES:
                    values, levels = helper.apply_factors(evaluation, candidate, factors)
                    result = helper.metrics(evaluation, values, evaluation_weights, levels, candidate)
                    result.update(model=model, scenario=scenario, period=period, candidate=candidate,
                                  scoreImprovementVsSameScenarioGlobal=1 - result["meanIntervalScore"] / baseline["meanIntervalScore"])
                    metrics.append(result)
                    if scenario == "original":
                        old = next(row for row in main["metrics"] if row["scope"] == PROTOCOL["scope"]
                                   and row["model"] == model and row["horizon"] == 1
                                   and row["weighting"] == "listing_balanced" and row["period"] == period and row["candidate"] == candidate)
                        assert result["count"] == old["count"] and result["listings"] == old["listings"]
                        for key in ["coverage", "meanWidth", "meanIntervalScore", "fallbackShare"]:
                            assert math.isclose(result[key], old[key], rel_tol=1e-12, abs_tol=1e-12), (model, period, candidate, key)
                        original_checks += 1
        print(json.dumps({"model": model, "completed": True}), flush=True)
    for result in metrics:
        original = next(row for row in metrics if row["scenario"] == "original" and row["model"] == result["model"]
                        and row["period"] == result["period"] and row["candidate"] == result["candidate"])
        assert result["count"] == original["count"] and result["listings"] == original["listings"]
        result["coverageChangeVsOriginal"] = result["coverage"] - original["coverage"]
        result["scoreImprovementVsOriginalSameCandidate"] = 1 - result["meanIntervalScore"] / original["meanIntervalScore"]
        result["widthChangeVsOriginalSameCandidate"] = result["meanWidth"] / original["meanWidth"] - 1
    for item in inputs.values():
        assert helper.file_hash(Path(item["path"])) == item["sha256"], item["path"]
    output = {"protocol": declared, "calibrationSamples": samples, "calibration": calibration, "metrics": metrics,
              "validation": {"originalMetricCellsReconciled": original_checks, "pairedMetricRows": len(metrics),
                             "sourceHashesUnchanged": True}, "completedAt": datetime.now(UTC).isoformat()}
    (output_directory / "results.json").write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(metrics).to_csv(output_directory / "comparisons.csv", index=False)
    names = ["protocol.json", "cash-flow-covid-sensitivity.executed.py", "results.json", "comparisons.csv"]
    receipt = {"inputs": inputs, "artifacts": [{"name": name, "sha256": helper.file_hash(output_directory / name)} for name in names]}
    (output_directory / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(output["validation"], indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("--output-directory", type=Path)
    args = parser.parse_args()
    run(args.source_directory, args.output_directory or args.source_directory / "covid-sensitivity")
