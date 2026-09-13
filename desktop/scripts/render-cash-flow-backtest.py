"""Render the saved audit as a standalone offline HTML explorer (stdlib only)."""

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(path.read_text())


def number(value):
    return float(value) if value not in (None, "") else None


def render(directory):
    summary = read_json(directory / "summary.json")
    companies = read_json(directory / "company-summary.json")["companies"]
    # Tuple schema is shared explicitly with the template. Keep numeric precision
    # so a rounded value cannot change an inside/outside classification.
    rows = []
    with gzip.open(directory / "forecasts.csv.gz", "rt", newline="") as stream:
        for row in csv.DictReader(stream):
            if (row["model"], row["window"], row["split"]) != ("linear", "5", "test"):
                continue
            rows.append([
                row["company_id"], int(row["origin_year"]), int(row["target_year"]),
                int(row["horizon"]), number(row["prediction"]), number(row["actual"]),
                number(row["lower"]), number(row["upper"]),
                number(row["calibrated_lower"]), number(row["calibrated_upper"]),
                row["currency"], row["audit_flags"].split("|"),
                [number(x) for x in row["train_cash"].split("|")],
                number(row["intercept"]), number(row["slope"]),
                row["calibration_unavailable"], number(row["training_mean_absolute_cash"]),
            ])
    pilot = read_json(directory / "frozen-2025-pilot-companies.json")
    pilot_rows = []
    for row in pilot["companies"]:
        pilot_rows.append({k: row.get(k) for k in (
            "id", "name", "status", "reason", "currency", "latest_training_year", "target_year",
            "low", "mid", "high", "actual", "miss_direction", "training_revision_count_in_later_snapshot",
        )})
    metrics_keys = (
        "split", "window", "model", "horizon", "count", "companies", "coverage",
        "medianNormalizedAbsoluteError", "meanNormalizedWidth", "meanNormalizedIntervalScore",
        "calibratedCount", "calibratedCoverage", "calibratedMeanNormalizedWidth",
        "calibratedMeanNormalizedIntervalScore", "rawOnCalibratedCoverage",
        "rawOnCalibratedMeanNormalizedWidth", "rawOnCalibratedMeanNormalizedIntervalScore",
    )
    payload = {
        "summary": {k: summary[k] for k in ("listings", "listingsWithEligibleFolds", "forecastRows", "protocol")},
        "metrics": [{k: r.get(k) for k in metrics_keys} for r in summary["metrics"] if r["split"] == "test"],
        "matched": [{k: r.get(k) for k in metrics_keys} for r in summary["matchedTenYearMetrics"] if r["split"] == "test"],
        "matchedCommon": [
            {k: r.get(k) for k in metrics_keys}
            for r in read_json(directory / "matched-window-common-scale.json")["metrics"]
        ],
        "sameSource": [{k: r.get(k) for k in metrics_keys} for r in summary["sameSourceMetrics"] if r["split"] == "test"],
        "companies": [{k: c.get(k) for k in ("id", "name", "isin", "sector", "branch")} for c in companies],
        "rows": rows,
        "pilot": read_json(directory / "frozen-2025-pilot-summary.json"),
        "pilotRows": pilot_rows,
        "vintage": read_json(directory / "source-vintage-change-diagnostics.json"),
    }

    data = json.dumps(payload, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("cash-flow-backtest-report.html").read_text()
    output = directory / "report.html"
    output.write_text(template.replace("__AUDIT_DATA__", data))
    receipt = {
        "report": str(output.resolve()), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "bytes": output.stat().st_size, "listings": len(companies), "retrospectiveRows": len(rows),
        "frozenPilotListings": len(pilot_rows), "embeddedDataOnly": True,
        "generatorSha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "templateSha256": hashlib.sha256(Path(__file__).with_name("cash-flow-backtest-report.html").read_bytes()).hexdigest(),
    }
    (directory / "report-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    render(parser.parse_args().directory)
