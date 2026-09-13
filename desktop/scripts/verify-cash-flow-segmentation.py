"""Independent scalar checks of the exploratory cash uncertainty experiment."""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

CANDIDATES = ["global", "sector", "branch", "cash_dispersion",
              "cash_residual_dispersion", "tangible_assets_sales", "tbv_ebit"]
BIN_CHECKS = {
    "cash_dispersion": ("fcf_volatility", [.25, .75], ["low", "medium", "high"]),
    "cash_residual_dispersion": ("fcf_trend_residual_volatility", [.15, .5], ["low", "medium", "high"]),
    "tangible_assets_sales": ("tangible_assets_to_revenue", [.25, 1], ["low", "medium", "high"]),
    "tbv_ebit": ("tangible_book_to_ebit", [3, 10], ["under 3 years", "3–10 years", "10 years+"]),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(actual, expected, label):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise AssertionError(f"{label}: {actual} != {expected}")


def weights_for(rows, duplicate_aware):
    companies = Counter(row["company_id"] for row in rows)
    copies = Counter((row["training_history_fingerprint"], row["target_year"]) for row in rows)
    return [1 / companies[row["company_id"]] /
            (copies[row["training_history_fingerprint"], row["target_year"]] if duplicate_aware else 1)
            for row in rows]


def scalar_quantile(rows, weights):
    ordered = sorted((row["error"], weight) for row, weight in zip(rows, weights, strict=True))
    threshold = .8 * sum(weight for _, weight in ordered)
    cumulative = 0.
    for error, weight in ordered:
        cumulative += weight
        if cumulative >= threshold:
            return error
    return ordered[-1][0]


def factors_for(rows, weights):
    result = {}
    for candidate in CANDIDATES:
        groups = defaultdict(list)
        for index, row in enumerate(rows):
            key = "all" if candidate == "global" else row[candidate]
            if key:
                groups[key].append(index)
        result[candidate] = {}
        for key, indices in groups.items():
            members = [rows[index] for index in indices]
            supported = (len(members) >= 300
                         and len({row["company_id"] for row in members}) >= 100
                         and len({row["training_history_fingerprint"] for row in members}) >= 100)
            result[candidate][key] = scalar_quantile(members, [weights[index] for index in indices]) if supported else None
    return result


def choose(row, candidate, factors):
    key = "all" if candidate == "global" else row[candidate]
    factor = factors[candidate].get(key)
    if factor is not None:
        return factor, candidate
    return choose(row, "sector" if candidate == "branch" else "global", factors)


def scalar_metrics(rows, weights, candidate, factors):
    total = sum(weights)
    coverage, width, score, fallback = 0., 0., 0., 0.
    for row, weight in zip(rows, weights, strict=True):
        factor, level = choose(row, candidate, factors)
        coverage += weight * (row["error"] <= factor)
        width += weight * 2 * factor
        score += weight * (2 * factor + 10 * max(row["error"] - factor, 0))
        fallback += weight * (level != candidate)
    return {"coverage": coverage / total, "meanWidth": width / total,
            "meanIntervalScore": score / total, "fallbackShare": fallback / total}


def run(directory):
    results = json.loads((directory / "results.json").read_text())
    calibration = json.loads((directory / "group-calibration.json").read_text())["calibrations"]
    receipt = json.loads((directory / "experiment-receipt.json").read_text())
    for artifact in receipt["artifacts"]:
        assert sha256(directory / artifact["name"]) == artifact["sha256"], artifact["name"]
    input_files = {"featuresSha256": "features.jsonl.gz", "forecastsSha256": "segmentation-forecasts.csv.gz"}
    for key, name in input_files.items():
        assert sha256(directory / name) == results["inputs"][key], name

    frame = pd.read_parquet(directory / "prepared-forecasts.parquet")
    assert not frame.duplicated(["company_id", "model", "horizon", "origin_year", "target_year"]).any()
    expected_scope = frame.sector.ne("Financials") | frame.branch_label.isin(["Real Estate", "REITs"])
    assert expected_scope.equals(frame.in_operating_scope)
    for candidate, (feature, edges, labels) in BIN_CHECKS.items():
        expected = ["" if not math.isfinite(value) else labels[sum(value >= edge for edge in edges)]
                    for value in frame[feature]]
        assert expected == frame[candidate].tolist(), candidate
    assert frame.loc[frame.application_ok].origin_year.ge(2021).all()
    assert frame.loc[frame.calibration_ok].target_year.le(2020).all()
    assert frame.loc[frame.calibration_ok].target_publication.le("2021-06-30").all()
    assert not frame.loc[frame.application_ok].target_year.eq(2021).any()

    paired = defaultdict(list)
    for metric in results["metrics"]:
        identity = tuple(metric[key] for key in ["scope", "model", "horizon", "weighting", "period"])
        paired[identity].append(metric)
        assert metric["inside"] + metric["below"] + metric["above"] == metric["count"]
        close(metric["coverage"] + metric["belowShare"] + metric["aboveShare"], 1., "coverage partition")
    for identity, rows in paired.items():
        assert len({row["count"] for row in rows}) == 1, identity
        assert len({row["listings"] for row in rows}) == 1, identity
        assert len({tuple(row["targetYears"]) for row in rows}) == 1, identity
        baseline = next(row for row in rows if row["candidate"] == "global")
        for row in rows:
            close(row["scoreImprovement"], 1 - row["meanIntervalScore"] / baseline["meanIntervalScore"], "score improvement")

    checked, factor_checks, anchors = [], 0, []
    for scope in ["operating_and_property", "all_listings"]:
        scope_frame = frame.loc[frame.in_operating_scope] if scope == "operating_and_property" else frame
        for model in ["linear", "naive"]:
            cell = scope_frame.loc[scope_frame.model.eq(model) & scope_frame.horizon.eq(1)]
            training = cell.loc[cell.calibration_ok].to_dict("records")
            for weighting in ["listing_balanced", "duplicate_history_downweighted"]:
                duplicates = weighting != "listing_balanced"
                factors = factors_for(training, weights_for(training, duplicates))
                saved = next(row for row in calibration if row["scope"] == scope and row["model"] == model
                             and row["horizon"] == 1 and row["weighting"] == weighting)
                for candidate, groups in factors.items():
                    assert set(groups) == set(saved["groups"][candidate])
                    for key, factor in groups.items():
                        expected = saved["groups"][candidate][key]["factor"]
                        if factor is None:
                            assert expected is None
                        else:
                            close(factor, expected, f"{scope}/{model}/{weighting}/{candidate}/{key}")
                        factor_checks += 1
                for period, years in [("validation", [2022, 2023]), ("recent", [2024, 2025])]:
                    rows = cell.loc[cell.application_ok & cell.target_year.isin(years)].to_dict("records")
                    weights = weights_for(rows, duplicates)
                    for candidate in CANDIDATES:
                        actual = scalar_metrics(rows, weights, candidate, factors)
                        expected = next(row for row in results["metrics"] if row["scope"] == scope and row["model"] == model
                                        and row["horizon"] == 1 and row["weighting"] == weighting
                                        and row["period"] == period and row["candidate"] == candidate)
                        for key, value in actual.items():
                            close(value, expected[key], f"{scope}/{model}/{weighting}/{period}/{candidate}/{key}")
                        checked.append({"scope": scope, "model": model, "weighting": weighting,
                                        "period": period, "candidate": candidate})
                    if scope == "all_listings" and weighting == "listing_balanced":
                        anchors.append({"model": model, "period": period, "calibrationFolds": len(training),
                                        "calibrationListings": len({row["company_id"] for row in training}),
                                        "factor": factors["global"]["all"], "evaluationFolds": len(rows),
                                        "evaluationListings": len({row["company_id"] for row in rows}),
                                        **scalar_metrics(rows, weights, "global", factors)})

    # These anchors were independently recomputed from the original 4,139,652-row
    # forecast CSV before this segmentation implementation produced group results.
    prior = {"linear": [1.5779635358388735, .7750902249865622, 14.512243329424727,
                        .8293407112490722, 14.789709367504143],
             "naive": [1.4078264290186215, .7775090224986563, 14.196028063969528,
                       .8323436129293474, 14.50130011953927]}
    for anchor in anchors:
        reference = prior[anchor["model"]]
        assert anchor["calibrationFolds"] == 96351 and anchor["calibrationListings"] == 11514
        close(anchor["factor"], reference[0], "original-CSV factor")
        position = 1 if anchor["period"] == "validation" else 3
        close(anchor["coverage"], reference[position], "original-CSV coverage")
        close(anchor["meanIntervalScore"], reference[position + 1], "original-CSV score")
    output = {"status": "passed", "preparedRows": len(frame), "pairedComparisons": len(results["metrics"]),
              "pairedPools": len(paired), "independentlyCheckedFactors": factor_checks,
              "independentlyCheckedSummaryCells": len(checked), "originalCsvAnchors": anchors,
              "checkedCandidates": CANDIDATES, "checkedHorizon": 1,
              "limits": "Scalar factor and metric checks cover seven h1 candidates, two models, two scopes and both weighting schemes. All reported horizons are checked for paired counts, score arithmetic and outcome partitions. Current-vintage and survivorship limitations remain.",
              "resultsSha256": sha256(directory / "results.json"), "validatorSha256": sha256(Path(__file__))}
    (directory / "independent-validation.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    run(parser.parse_args().directory)
