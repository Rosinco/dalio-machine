"""Render a self-contained offline report from the fixed segmentation experiment."""
import argparse
import gzip
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render(directory):
    directory = Path(directory).resolve()
    results_path = directory / "results.json"
    calibration_path = directory / "group-calibration.json"
    features_path = directory / "features.jsonl.gz"
    feature_summary_path = directory / "feature-summary.json"
    template_path = Path(__file__).with_name("cash-flow-segmentation-report.html")
    inputs = [results_path, calibration_path, features_path, feature_summary_path]
    within_path = directory / "within-branch-results.json"
    if within_path.exists():
        inputs.append(within_path)
    hashes = {path.name: file_hash(path) for path in inputs}
    results = json.loads(results_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    feature_summary = json.loads(feature_summary_path.read_text())
    if results["protocol"] != calibration["protocol"]:
        raise ValueError("Results and calibration protocols differ")
    if results["inputs"]["featuresSha256"] != hashes[features_path.name]:
        raise ValueError("Feature file does not match scored inputs")
    if not results["metrics"]:
        raise ValueError("No scored comparisons available")
    company_keys = ["company_id", "company_name", "sector", "branch", "cash_sign_regime",
                    "fcf_volatility", "tangible_assets_to_revenue", "tangible_book_to_ebit",
                    "tangible_book_to_ebit_status", "net_debt_to_total_assets", "ebit_margin"]
    companies = []
    seen = set()
    with gzip.open(features_path, "rt") as stream:
        for line in stream:
            row = json.loads(line)
            if row["origin_year"] != 2024:
                continue
            if row["company_id"] in seen:
                raise ValueError("Duplicate FY2024 company features")
            seen.add(row["company_id"])
            companies.append([row.get(key) for key in company_keys])
    companies.sort(key=lambda row: (str(row[1]).casefold(), str(row[0])))
    data = {"results": results, "calibrations": calibration["calibrations"],
            "withinBranch": json.loads(within_path.read_text()) if within_path.exists() else None,
            "features": {key: feature_summary[key] for key in [
                "definitions", "aggregation", "numericAvailability", "featureRows", "listings",
                "minimumPositiveEbitMargin", "limitations"]},
            "companyKeys": company_keys, "companies": companies, "inputHashes": hashes}
    serialized = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    serialized = serialized.replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    template = template_path.read_text()
    if template.count("__REPORT_DATA__") != 1:
        raise ValueError("Template must contain exactly one report-data marker")
    output_path = directory / "report.html"
    output_path.write_text(template.replace("__REPORT_DATA__", serialized))
    if any(file_hash(path) != hashes[path.name] for path in inputs):
        raise ValueError("An input changed while the report was rendered")
    receipt = {"completedAt": datetime.now(UTC).isoformat(), "protocolId": results["protocol"]["id"],
               "rendererSha256": file_hash(Path(__file__)), "templateSha256": file_hash(template_path),
               "inputs": hashes, "comparisonRows": len(results["metrics"]),
               "groupRows": len(results["groupMetrics"]), "companyFY2024Rows": len(companies),
               "withinBranchRows": len(data["withinBranch"]["metrics"]) if data["withinBranch"] else 0,
               "report": {"name": output_path.name, "sha256": file_hash(output_path), "bytes": output_path.stat().st_size}}
    (directory / "report-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    arguments = parser.parse_args()
    render(arguments.directory)
