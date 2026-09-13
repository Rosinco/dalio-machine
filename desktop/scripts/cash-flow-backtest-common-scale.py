"""Compare matched five/ten-year forecasts using the same latest-five scale.

This is a postprocessing comparison of existing forecasts. It does not refit,
select or deploy a model, change learned cash-unit bands, or alter the original
own-window-denominator summary.
"""
import argparse
import csv
import gzip
import hashlib
import json
from array import array
from pathlib import Path
from statistics import median


def measure(row):
    history = [float(x) for x in row["train_cash"].split("|")]
    if len(history) != int(row["window"]) or len(history) < 5:
        raise ValueError("Complete requested training window is required")
    scale = sum(abs(x) for x in history[:5]) / 5
    prediction, actual = float(row["prediction"]), float(row["actual"])
    lower, upper = float(row["lower"]), float(row["upper"])
    width = upper - lower
    error = prediction - actual
    outcome = "below" if actual < lower else "above" if actual > upper else "inside"
    interval_score = width + 10 * max(lower - actual, actual - upper, 0)
    return {
        "scale": scale, "latestFive": history[:5], "outcome": outcome,
        "absoluteError": abs(error), "bias": error,
        "normalizedAbsoluteError": abs(error) / scale if scale > 0 else None,
        "normalizedBias": error / scale if scale > 0 else None,
        "normalizedWidth": width / scale if scale > 0 else None,
        "normalizedIntervalScore": interval_score / scale if scale > 0 else None,
    }


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def calculate(directory):
    forecast_path = directory / "forecasts.csv.gz"
    receipt = json.loads((directory / "receipt.json").read_text())
    forecast_hash = file_hash(forecast_path)
    expected = next(x["sha256"] for x in receipt["artifacts"] if x["name"] == forecast_path.name)
    if forecast_hash != expected:
        raise ValueError("Forecast artifact checksum mismatch")
    groups, pending = {}, {}
    rows, paired = 0, 0
    with gzip.open(forecast_path, "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != "test" or row["matched_ten_year"] != "true":
                continue
            value = measure(row)
            key = (row["model"], int(row["window"]), int(row["horizon"]))
            pair_key = (row["company_id"], row["origin_year"], row["target_year"], row["model"], row["horizon"])
            previous = pending.pop(pair_key, None)
            if previous is None:
                pending[pair_key] = (key[1], value["latestFive"], row["currency"])
            else:
                if previous[0] == key[1] or previous[1] != value["latestFive"] or previous[2] != row["currency"]:
                    raise ValueError("Matched window fold does not share its latest five cash amounts and currency")
                paired += 1
            if key not in groups:
                groups[key] = {
                    "count": 0, "below": 0, "inside": 0, "above": 0,
                    "ids": set(), "years": set(), "errors": array("d"),
                    "normalizedBiasSum": 0, "normalizedWidthSum": 0,
                    "normalizedIntervalScoreSum": 0, "currencies": {},
                }
            group = groups[key]
            group["count"] += 1
            group[value["outcome"]] += 1
            group["ids"].add(row["company_id"])
            group["years"].add(int(row["target_year"]))
            if value["normalizedAbsoluteError"] is not None:
                group["errors"].append(value["normalizedAbsoluteError"])
                for field in ["normalizedBias", "normalizedWidth", "normalizedIntervalScore"]:
                    group[field + "Sum"] += value[field]
            currency = group["currencies"].setdefault(row["currency"], {"count": 0, "absoluteError": 0, "bias": 0})
            currency["count"] += 1
            currency["absoluteError"] += value["absoluteError"]
            currency["bias"] += value["bias"]
            rows += 1
    if pending or rows != paired * 2:
        raise ValueError("Unbalanced five/ten-year comparison cohort")
    metrics = []
    for (model, window, horizon), group in sorted(groups.items()):
        count = len(group["errors"])
        metrics.append({
            "split": "test", "window": window, "model": model, "horizon": horizon,
            "count": group["count"], "companies": len(group["ids"]),
            "targetYears": sorted(group["years"]),
            "below": group["below"], "inside": group["inside"], "above": group["above"],
            "coverage": group["inside"] / group["count"], "normalizedCount": count,
            "medianNormalizedAbsoluteError": median(group["errors"]) if count else None,
            "meanNormalizedAbsoluteError": sum(group["errors"]) / count if count else None,
            "meanNormalizedBias": group["normalizedBiasSum"] / count if count else None,
            "meanNormalizedWidth": group["normalizedWidthSum"] / count if count else None,
            "meanNormalizedIntervalScore": group["normalizedIntervalScoreSum"] / count if count else None,
            "maeByCurrency": {code: {"count": c["count"], "mae": c["absoluteError"] / c["count"], "meanBias": c["bias"] / c["count"]} for code, c in group["currencies"].items()},
        })
    output = {
        "protocol": "matched-test-windows-common-latest-five-scale-v1",
        "normalization": "Mean absolute cash across the latest five training annuals, identical for paired five/ten-year forecasts. Zero common scale leaves both normalized scores missing; raw coverage remains counted.",
        "scope": "Existing fixed-model test forecasts only, on the ten-year-eligible common cohort. No forecasts refitted and no model selected. Learned intervals are not compared here.",
        "forecastRows": rows, "pairedForecasts": paired, "metrics": metrics,
        "inputs": {"forecasts": {"path": str(forecast_path.resolve()), "sha256": forecast_hash}, "sourcePackSha256": receipt["sourcePackSha256"], "backtestScriptSha256": receipt["scriptSha256"]},
        "scriptSha256": file_hash(Path(__file__)),
    }
    path = directory / "matched-window-common-scale.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"path": str(path), "forecastRows": rows, "pairedForecasts": paired, "sha256": file_hash(path)}))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    calculate(parser.parse_args().directory)
