"""Independent receipt, aggregation and selected-source-row checks; offline only."""

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent.parent / "test-results/cash-flow-backtest-2026-09-12")
directory = parser.parse_args().directory.resolve()
receipt = json.loads((directory / "receipt.json").read_text())
summary = json.loads((directory / "summary.json").read_text())


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (actual, expected)


checked_hashes = []
for artifact in receipt["artifacts"]:
    path = directory / artifact["name"]
    assert path.stat().st_size == artifact["bytes"]
    assert digest(path) == artifact["sha256"], path
    checked_hashes.append(artifact["name"])
assert digest(Path(summary["inputs"]["script"]["path"])) == receipt["scriptSha256"]

metrics = summary["metrics"]
assert sum(row["count"] for row in metrics) == summary["forecastRows"]
for row in metrics:
    assert sum(row[key] for key in ("below", "inside", "above")) == row["count"]
    close(row["coverage"], row["inside"] / row["count"])
    assert sum(row[key] for key in ("calibratedBelow", "calibratedInside", "calibratedAbove")) == row["calibratedCount"]
    assert sum(row[key] for key in ("rawOnCalibratedBelow", "rawOnCalibratedInside", "rawOnCalibratedAbove")) == row["calibratedCount"]
    assert row["rawOnCalibratedCount"] == row["calibratedCount"]
    assert row["calibratedCount"] <= row["count"]
    if row["calibratedCount"]:
        close(row["calibratedCoverage"], row["calibratedInside"] / row["calibratedCount"])
    if row["split"] == "test" and row["horizon"] == 5:
        assert row["calibratedCount"] == 0
    peers = [other for other in metrics if (other["split"], other["window"], other["horizon"]) == (row["split"], row["window"], row["horizon"])]
    assert len({other["count"] for other in peers}) == 1
    assert len({other["calibratedCount"] for other in peers}) == 1

selected = []
with gzip.open(directory / "forecasts.csv.gz", "rt", newline="") as stream:
    for row in csv.DictReader(stream):
        if int(row["company_id"]) > 696:
            break
        if row["company_id"] not in {"3", "197", "696"} or (row["model"], row["window"], row["split"]) != ("linear", "5", "test"):
            continue
        cash = [float(value) for value in row["train_cash"].split("|")]
        weights, x = [30, 25, 20, 15, 10], [0, -1, -2, -3, -4]
        weighted_x = sum(a * b for a, b in zip(weights, x, strict=True)) / 100
        weighted_cash = sum(a * b for a, b in zip(weights, cash, strict=True)) / 100
        slope = sum(w * (a - weighted_x) * (b - weighted_cash) for w, a, b in zip(weights, x, cash, strict=True)) / sum(w * (a - weighted_x) ** 2 for w, a in zip(weights, x, strict=True))
        intercept = weighted_cash - slope * weighted_x
        horizon, actual = int(row["horizon"]), float(row["actual"])
        mid = intercept + slope * horizon
        lower, upper = mid - abs(mid) * horizon / 10, mid + abs(mid) * horizon / 10
        scale = sum(abs(value) for value in cash) / 5
        close(float(row["prediction"]), mid)
        close(float(row["lower"]), lower)
        close(float(row["upper"]), upper)
        outcome = "below" if actual < lower else "above" if actual > upper else "inside"
        assert row["outcome"] == outcome
        close(float(row["normalized_interval_score"]), (upper - lower + 10 * max(lower - actual, actual - upper, 0)) / scale)
        if row["calibrated_lower"]:
            assert int(row["origin_year"]) >= 2021
            assert row["origin_publication"] >= "2021-06-30"
            assert row["origin_publication"] < row["target_end"]
            assert all(date and date <= row["origin_publication"] for date in row["train_publications"].split("|"))
        else:
            assert not row["calibrated_factor"]
        selected.append({key: row[key] for key in ("company_id", "origin_year", "target_year", "horizon", "prediction", "actual", "outcome")})
assert {row["company_id"] for row in selected} == {"3", "197", "696"}

common_path = directory / "matched-window-common-scale.json"
common = json.loads(common_path.read_text())
assert common["inputs"]["forecasts"]["sha256"] == next(row["sha256"] for row in receipt["artifacts"] if row["name"] == "forecasts.csv.gz")
assert digest(Path(__file__).with_name("cash-flow-backtest-common-scale.py")) == common["scriptSha256"]
assert sum(row["count"] for row in common["metrics"]) == common["forecastRows"] == 2 * common["pairedForecasts"]
for row in common["metrics"]:
    other = next(value for value in common["metrics"] if value["model"] == row["model"] and value["horizon"] == row["horizon"] and value["window"] != row["window"])
    assert row["count"] == other["count"] and row["normalizedCount"] == other["normalizedCount"]
    if row["window"] == 5:
        original = next(value for value in summary["matchedTenYearMetrics"] if value["split"] == "test" and value["model"] == row["model"] and value["horizon"] == row["horizon"] and value["window"] == 5)
        for key in ("coverage", "medianNormalizedAbsoluteError", "meanNormalizedWidth", "meanNormalizedIntervalScore"):
            close(row[key], original[key])
    if row["model"] == "naive":
        for key in ("coverage", "medianNormalizedAbsoluteError", "meanNormalizedWidth", "meanNormalizedIntervalScore"):
            close(row[key], other[key])

result = {"status": "pass", "checkedArtifactHashes": checked_hashes, "scriptSha256": receipt["scriptSha256"], "summaryGroups": len(metrics), "forecastRowsReconciledFromSummary": summary["forecastRows"], "selectedForecastRowsRecalculated": len(selected), "selectedRows": selected, "commonScale": {"sha256": digest(common_path), "rows": common["forecastRows"], "pairs": common["pairedForecasts"], "pairedCountsMatch": True, "fiveYearOriginalScoresMatch": True, "naiveWindowScoresInvariant": True}, "scope": "Independent artifact hashes, summary group arithmetic/model counts, every available linear-five-year recent-test row for ABB, SCA and Stora Enso, and common-scale group invariants; not a full CSV arithmetic rerun."}
(directory / "independent-reconciliation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({key: value for key, value in result.items() if key != "selectedRows"}))
