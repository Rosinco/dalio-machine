"""Predeclared cash-uncertainty refinements conditional on the same branch.

Uses a private imported instance of the original experiment's helpers. Only
that instance's candidate/parent configuration changes; its source and existing
artifacts remain immutable. The input parquet already binds original source
history, fixed bins, timing gates and normalized forecast errors.
"""
import argparse
import importlib.util
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REFINEMENTS = {
    "branch_tangible_assets_sales": "tangible_assets_sales",
    "branch_tbv_ebit": "tbv_ebit",
    "branch_ebit_margin": "ebit_margin",
    "branch_cash_residual_dispersion": "cash_residual_dispersion",
}
CANDIDATES = ["branch", *REFINEMENTS]
HELPER_PATH = Path(__file__).with_name("cash-flow-segmentation.py")


def load_helpers():
    spec = importlib.util.spec_from_file_location("private_within_branch_segmentation", HELPER_PATH)
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    helper.CANDIDATES = ["global", "sector", *CANDIDATES]
    helper.PARENTS = {"branch": "sector", **dict.fromkeys(REFINEMENTS, "branch")}
    return helper


def add_refinements(frame):
    frame = frame.copy()
    required = ["branch", *REFINEMENTS.values()]
    if any(column not in frame for column in required):
        raise ValueError("Prepared fixed branch and feature bins are required")
    for candidate, feature in REFINEMENTS.items():
        branch, bin_name = frame.branch.fillna(""), frame[feature].fillna("")
        frame[candidate] = np.where(branch.ne("") & bin_name.ne(""), branch + " / " + bin_name, "")
    return frame


def fit_calibration(cell, helper, duplicate_aware=False, *, minimum_rows=300,
                    minimum_ids=100, minimum_histories=100):
    calibration = cell.loc[cell.calibration_ok].reset_index(drop=True)
    if not len(calibration):
        raise ValueError("No publication-eligible calibration rows")
    weights = helper.pool_weights(calibration, duplicate_aware)
    factors = helper.fit_factors(calibration, weights, minimum_rows, minimum_ids, minimum_histories)
    total_weight = float(weights.sum())
    support = {}
    for candidate, groups in factors.items():
        supported = [group for group in groups.values() if group["supported"]]
        missing = np.zeros(len(calibration), dtype=bool) if candidate == "global" else calibration[candidate].eq("").to_numpy()
        support[candidate] = {
            "calibrationRows": len(calibration), "calibrationListings": int(calibration.company_id.nunique()),
            "calibrationHistories": int(calibration.training_history_fingerprint.nunique()),
            "groups": len(groups), "supportedGroups": len(supported),
            "unsupportedGroups": len(groups) - len(supported),
            "missingGroupRows": int(missing.sum()),
            "missingGroupWeightShare": float(weights[missing].sum() / total_weight),
            "supportedCalibrationWeightShare": sum(group["weight"] for group in supported) / total_weight,
            "minimumSupportedListings": min((group["listings"] for group in supported), default=None),
            "minimumSupportedHistories": min((group["histories"] for group in supported), default=None),
        }
    return factors, support


def score_period(frame, factors, helper, duplicate_aware=False):
    if not len(frame):
        return []
    weights = helper.pool_weights(frame, duplicate_aware)
    baseline, base_levels = helper.apply_factors(frame, "branch", factors)
    base = helper.metrics(frame, baseline, weights, base_levels, "branch")
    results = []
    for candidate in CANDIDATES:
        values, levels = helper.apply_factors(frame, candidate, factors)
        result = helper.metrics(frame, values, weights, levels, candidate)
        result.update({
            "candidate": candidate, "baselineCandidate": "branch",
            "branchCoverage": base["coverage"], "branchWidth": base["meanWidth"],
            "branchScore": base["meanIntervalScore"],
            "scoreImprovement": 1 - result["meanIntervalScore"] / base["meanIntervalScore"] if base["meanIntervalScore"] else None,
            "widthChangeRelativeToBranch": result["meanWidth"] / base["meanWidth"] - 1 if base["meanWidth"] else None,
            "fallbackLevelShares": {str(level): float(weights[levels == level].sum() / weights.sum()) for level in np.unique(levels)},
        })
        results.append(result)
    return results


def run(directory):
    directory = Path(directory).resolve()
    helper = load_helpers()
    protocol_path = directory / "within-branch-protocol.json"
    protocol = json.loads(protocol_path.read_text())
    if protocol["candidates"] != CANDIDATES or protocol["featureGroups"] != REFINEMENTS:
        raise ValueError("Refinements do not match predeclared protocol")
    if protocol["inputHelperSha256"] != helper.file_hash(HELPER_PATH):
        raise ValueError("Root helper changed after protocol declaration")
    prepared_path = directory / "prepared-forecasts.parquet"
    metadata_path = directory / "prepared-metadata.json"
    paths = [prepared_path, metadata_path, protocol_path, HELPER_PATH, Path(__file__)]
    hashes = {str(path): helper.file_hash(path) for path in paths}
    inputs = {
        "preparedForecasts": {"path": str(prepared_path), "sha256": hashes[str(prepared_path)]},
        "preparedMetadata": {"path": str(metadata_path), "sha256": hashes[str(metadata_path)], "value": json.loads(metadata_path.read_text())},
        "protocolSha256": hashes[str(protocol_path)],
        "helperScript": {"path": str(HELPER_PATH), "sha256": hashes[str(HELPER_PATH)]},
        "scriptSha256": hashes[str(Path(__file__))],
    }
    frame = pd.read_parquet(prepared_path)
    frame = add_refinements(frame.loc[frame.in_operating_scope])
    metrics, supports, all_factors = [], [], []
    for comparison in protocol["comparisons"]:
        model, horizon = comparison["model"], comparison["horizon"]
        cell = frame.loc[frame.model.eq(model) & frame.horizon.eq(horizon)]
        for weighting in protocol["weightings"]:
            duplicate_aware = weighting == "duplicate_history_downweighted"
            factors, support = fit_calibration(
                cell, helper, duplicate_aware,
                minimum_rows=protocol["minimumCalibrationRows"],
                minimum_ids=protocol["minimumCalibrationListings"],
                minimum_histories=protocol["minimumCalibrationHistories"],
            )
            identity = {"scope": protocol["scope"], "model": model, "horizon": horizon, "weighting": weighting}
            all_factors.append({**identity, "groups": factors})
            supports.extend({**identity, "candidate": candidate, **value} for candidate, value in support.items())
            for period, bounds in [("validation", (2021, 2023)), ("recent", (2024, 2025))]:
                selected = cell.loc[cell.application_ok & cell.target_year.between(*bounds)].reset_index(drop=True)
                metrics.extend({**identity, "period": period, **value} for value in score_period(selected, factors, helper, duplicate_aware))
                for year, annual in selected.groupby("target_year", sort=True):
                    # score_period regenerates weights inside this exact annual pool.
                    metrics.extend({**identity, "period": f"FY{year}", **value} for value in score_period(annual.reset_index(drop=True), factors, helper, duplicate_aware))
            print(json.dumps({"completedCell": identity}), flush=True)
    for path in paths:
        if helper.file_hash(path) != hashes[str(path)]:
            raise ValueError(f"Source changed during within-branch experiment: {path}")
    output = {"protocol": protocol, "inputs": inputs, "metrics": metrics, "supports": supports,
              "completedAt": datetime.now(UTC).isoformat()}
    result_path = directory / "within-branch-results.json"
    calibration_path = directory / "within-branch-calibration.json"
    result_path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    calibration_path.write_text(json.dumps({"protocol": protocol, "calibrations": all_factors}, indent=2, allow_nan=False) + "\n")
    receipt = {"inputs": inputs, "artifacts": [{"name": path.name, "sha256": helper.file_hash(path), "bytes": path.stat().st_size} for path in [result_path, calibration_path]]}
    (directory / "within-branch-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"output": str(result_path), "metricRows": len(metrics), "supportRows": len(supports)}), flush=True)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
