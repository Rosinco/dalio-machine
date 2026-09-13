"""Fixed offline midline challengers; no calibration or app defaults are changed."""

import argparse
import functools
import hashlib
import json
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

MODELS = ["latest", "robust_blend", "damped_trend"]
PROTOCOL = {
    "id": "exploratory-cash-midline-challengers-v1", "date": "2026-09-13",
    "models": MODELS, "historyYears": 5, "horizons": [1, 2, 3, 4, 5],
    "latest": "Latest signed annual provider cash, held flat.",
    "robust_blend": "0.5 * latest signed annual cash + 0.5 * median of the five signed annual cash observations; held flat.",
    "damped_trend": "Latest signed annual cash + five-year weighted slope * sum(0.5**j for j=1..h). Weights 30/25/20/15/10, newest first; level starts at observed latest cash, not fitted intercept.",
    "selection": "Two fixed candidates and coefficients recorded before this new experiment executes. Recent outcomes were already inspected in earlier research, so this is exploratory, not fresh confirmation or a prospective test. No automatic selection or deployment.",
    "normalization": "Unweighted mean absolute five-year training cash, identical for every candidate on each paired row. Zero scale remains unscored; no floor, clipping or removal of negative observations.",
    "chronology": {"older": "target FY<=2020", "validation": "target FY2021-2023", "recent": "target FY2024-2025", "annualDiagnostics": [2020, 2021, 2022, 2023, 2024, 2025]},
    "cohorts": "Reuse only window=5/model=naive rows from the hash-verified original ledger, one listing/origin/horizon each. All candidates have identical histories, targets, eligibility and weights. Original full-period, currency, intermediate-year and missing-data guards remain inherited.",
    "timing": "Known ordered training publications, each at or after its period end and no later than origin publication; origin publication strictly before target end. A missing target publication is disclosed, not changed into a missing realized value. A supplied target publication before its end is rejected. Target annual periods may already be partly elapsed at origin.",
    "rangeDistinction": "Point-error comparisons do not need the old range-calibration application cutoff. Report its eligible count separately; higher-horizon point evidence does not create empirical Years 5-10 uncertainty factors.",
    "scopes": ["operating_and_property", "all_listings"],
    "scopeDefinition": "Operating/property follows the original segmentation scope: exclude Financials except Real Estate and REITs. All-listing sensitivity retains financial provider cash; it does not authorize projecting financial firms in the app.",
    "weighting": "Each listing receives total weight one within scope/horizon/period. Duplicate sensitivity further divides by exact native-cash-history-fingerprint/target-year multiplicity. This is not definitive issuer deduplication.",
    "fingerprint": "SHA256 of reporting currency, five training fiscal end dates and canonical finite signed native cash values, newest first. Deliberately cash-only, not the earlier all-feature fingerprint.",
    "metrics": ["weighted mean, median, 90th and 99th percentile normalized absolute error", "weighted signed prediction-minus-actual bias", "paired mean error improvement", "paired better/equal/worse shares"],
    "limitations": [
        "Later-vintage provider FCF and historical classifications are not point-in-time source facts; provider cash-definition changes remain unresolved.",
        "The input contains eligible realized folds. Original unscored-company and exclusion ledgers remain the universe-coverage record; this slice does not recreate missing companies or outcomes.",
        "Related listings, repeated targets and overlapping histories are dependent. Counts and paired shares are descriptive, not independent trials or statistical significance.",
        "Few recent calendar regimes and retrospective inspection prevent claims of stable company-specific accuracy or an optimal ten-year midline.",
        "No error-range factors are refitted, no annual coverage is claimed and no DCF, terminal value, required return or investment-price accuracy is tested.",
        "A result that merits an untouched future trial is not authorization to replace the current latest-cash default.",
    ],
}
COLUMNS = [
    "company_id", "company_name", "isin", "sector", "branch", "currency", "window", "model", "horizon",
    "origin_year", "target_year", "split", "origin_publication", "target_end", "target_publication",
    "train_years", "train_cash", "train_end_dates", "train_publications", "train_source_ids", "target_source_id",
    "training_mean_absolute_cash", "slope", "prediction", "actual", "calibration_unavailable",
]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_inputs(directory):
    receipt_path, forecast_path = directory / "receipt.json", directory / "forecasts.csv.gz"
    receipt = json.loads(receipt_path.read_text())
    expected = {item["name"]: item["sha256"] for item in receipt["artifacts"]}
    actual = file_hash(forecast_path)
    if expected.get(forecast_path.name) != actual:
        raise ValueError("Original forecast ledger checksum mismatch")
    return {"forecastPath": str(forecast_path.resolve()), "forecastsSha256": actual,
            "upstreamReceiptSha256": file_hash(receipt_path), "upstreamSourcePackSha256": receipt.get("sourcePackSha256")}


def check_output(source, output):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the retained source research directory")
    if output.exists() and any(path.name not in {"protocol.json", "protocol-lock.json"} for path in output.iterdir()):
        raise ValueError("Output contains retained artifacts; choose a new output directory")


def predict_candidates(cash, horizons):
    cash, horizons = np.asarray(cash, dtype=float), np.asarray(horizons, dtype=float)
    if cash.ndim != 2 or cash.shape[1] != 5 or horizons.shape != (len(cash),) or not np.isfinite(cash).all():
        raise ValueError("Candidates require five finite signed cash observations per horizon")
    if not np.isfinite(horizons).all() or ((horizons < 1) | (horizons > 5) | (horizons != np.floor(horizons))).any():
        raise ValueError("Candidate horizons must be integers from one to five")
    weights = np.array([30., 25., 20., 15., 10.])
    offsets = np.array([0., -1., -2., -3., -4.])
    center = np.average(offsets, weights=weights)
    weighted_cash = np.average(cash, axis=1, weights=weights)
    slope = ((cash - weighted_cash[:, None]) * weights * (offsets - center)).sum(axis=1) / np.sum(weights * (offsets - center) ** 2)
    predictions = {
        "latest": cash[:, 0].copy(),
        "robust_blend": .5 * cash[:, 0] + .5 * np.median(cash, axis=1),
        "damped_trend": cash[:, 0] + slope * (1 - .5 ** horizons),
    }
    scale = np.mean(np.abs(cash), axis=1)
    if not all(np.isfinite(values).all() for values in [*predictions.values(), scale, slope]):
        raise ValueError("Candidate arithmetic exceeds finite range")
    return predictions, scale, slope


def valid_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


@functools.lru_cache(maxsize=250_000)
def training_timing(publications, ends, origin):
    pubs, dates = publications.split("|"), ends.split("|")
    return len(pubs) == len(dates) == 5 and valid_date(origin) and all(
        valid_date(pub) and valid_date(end) and end <= pub <= origin
        for pub, end in zip(pubs, dates, strict=True)
    )


def timing_issue(row):
    origin, end, pub = row["origin_publication"], row["target_end"], row["target_publication"]
    if not valid_date(origin):
        return "origin_publication_missing_or_invalid"
    if not valid_date(end) or origin >= end:
        return "origin_not_before_target_end"
    if not training_timing(row["train_publications"], row["train_end_dates"], origin):
        return "training_publication_order"
    if pub and (not valid_date(pub) or pub < end):
        return "target_publication_invalid_or_before_end"
    return ""


@functools.lru_cache(maxsize=250_000)
def history_fingerprint(currency, ends, cash):
    values = [float(value) for value in cash.split("|")]
    normalized = [0. if value == 0 else value for value in values]
    body = json.dumps([currency, ends.split("|"), normalized], separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode()).hexdigest()


def prepare_chunk(frame):
    frame = frame.copy()
    training = np.array([list(map(float, values.split("|"))) for values in frame.train_cash], dtype=float)
    predictions, scale, slope = predict_candidates(training, frame.horizon.to_numpy())
    expected_years = frame.origin_year.to_numpy()[:, None] - np.arange(5)
    years = np.array([list(map(int, values.split("|"))) for values in frame.train_years])
    if not np.array_equal(years, expected_years) or not np.array_equal(frame.target_year, frame.origin_year + frame.horizon):
        raise ValueError("Inherited training or target chronology mismatch")
    if not np.isfinite(frame.actual.to_numpy(dtype=float)).all():
        raise ValueError("Original realized fold has unavailable cash")
    for name, recomputed in [("prediction", predictions["latest"]), ("training_mean_absolute_cash", scale), ("slope", slope)]:
        if not np.allclose(frame[name].to_numpy(dtype=float), recomputed, rtol=1e-10, atol=1e-8):
            raise ValueError(f"Independent reconstruction disagrees with original {name}")
    frame["scale"] = scale
    for model, values in predictions.items():
        frame[model] = values
    frame["timing_issue"] = [timing_issue(row) for row in frame.to_dict("records")]
    frame["point_eligible"] = frame.timing_issue.eq("") & frame.scale.gt(0)
    frame["in_operating_scope"] = frame.sector.ne("Financials") | frame.branch.isin(["Real Estate", "REITs"])
    frame["same_source"] = [len(set(training.split("|") + [target])) == 1
                            for training, target in zip(frame.train_source_ids, frame.target_source_id, strict=True)]
    frame["cash_history_fingerprint"] = [history_fingerprint(currency, ends, cash)
                                         for currency, ends, cash in zip(frame.currency, frame.train_end_dates, frame.train_cash, strict=True)]
    return frame


def pool_weights(frame, duplicate_aware):
    weights = 1. / frame.groupby("company_id")["company_id"].transform("size").to_numpy()
    if duplicate_aware:
        weights /= frame.groupby(["cash_history_fingerprint", "target_year"])["company_id"].transform("size").to_numpy()
    return weights


def weighted_quantile(values, weights, quantile):
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"), len(values) - 1)
    return float(values[order[index]])


def metrics(frame, model, weights):
    actual, scale = frame.actual.to_numpy(dtype=float), frame.scale.to_numpy(dtype=float)
    signed = (frame[model].to_numpy() - actual) / scale
    error = np.abs(signed)
    baseline = np.abs(frame.latest.to_numpy() - actual) / scale
    if not len(frame) or not np.isfinite(error).all() or not np.isfinite(baseline).all() or (scale <= 0).any():
        raise ValueError("Metrics require a nonempty finite paired cohort with positive historical scale")
    def average(values):
        return float(np.average(values, weights=weights))
    mean, base = average(error), average(baseline)
    ties = np.isclose(error, baseline, rtol=1e-12, atol=1e-12)
    return {
        "count": len(frame), "listings": int(frame.company_id.nunique()),
        "histories": int(frame.cash_history_fingerprint.nunique()), "weight": float(weights.sum()),
        "targetYears": sorted(int(value) for value in frame.target_year.unique()),
        "meanNormalizedError": mean, "medianNormalizedError": weighted_quantile(error, weights, .5),
        "p90NormalizedError": weighted_quantile(error, weights, .9), "p99NormalizedError": weighted_quantile(error, weights, .99),
        "meanNormalizedBias": average(signed), "baselineMeanNormalizedError": base,
        "meanErrorImprovementPercent": (base - mean) / base * 100 if base > 0 else None,
        "pairedMeanErrorDifference": average(error - baseline),
        "betterShare": average((error < baseline) & ~ties), "tieShare": average(ties), "worseShare": average((error > baseline) & ~ties),
    }


def comparisons(frame):
    periods = {"older": frame.target_year.le(2020), "validation": frame.target_year.between(2021, 2023),
               "recent": frame.target_year.between(2024, 2025),
               **{f"FY{year}": frame.target_year.eq(year) for year in range(2020, 2026)}}
    results = []
    for scope in PROTOCOL["scopes"]:
        scope_mask = frame.in_operating_scope if scope == "operating_and_property" else pd.Series(True, index=frame.index)
        for horizon in PROTOCOL["horizons"]:
            for period, period_mask in periods.items():
                cell = frame.loc[scope_mask & period_mask & frame.horizon.eq(horizon) & frame.point_eligible]
                if cell.empty:
                    continue
                for duplicate_aware in [False, True]:
                    weights = pool_weights(cell, duplicate_aware)
                    for model in MODELS:
                        results.append({"scope": scope, "horizon": horizon, "period": period,
                                        "weighting": "duplicate_cash_history" if duplicate_aware else "listing",
                                        "model": model, **metrics(cell, model, weights),
                                        "originalRangeApplicationEligible": int(cell.calibration_unavailable.eq("").sum()),
                                        "sameSourceCount": int(cell.same_source.sum()),
                                        "missingTargetPublication": int(cell.target_publication.eq("").sum())})
    return results


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def render_report(results, audit):
    lines = ["# Fixed cash-midline challenger experiment", "", "Exploratory offline research; the app's latest-cash default and uncertainty calibration remain unchanged.", "",
             f"Paired source folds: {audit['sourceFiveYearNaiveRows']:,}; chronology/positive-scale eligible: {audit['pointEligibleRows']:,}.", "",
             "| Period | Horizon | Model | Folds | Listings | Mean normalized error | Median normalized error | Mean error improvement vs latest |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in results:
        if row["scope"] == "operating_and_property" and row["weighting"] == "listing" and row["period"] in {"validation", "recent"}:
            improvement = row["meanErrorImprovementPercent"]
            display = "unavailable" if improvement is None else f"{improvement:+.2f}%"
            lines.append(f"| {row['period']} | {row['horizon']} | {row['model']} | {row['count']:,} | {row['listings']:,} | {row['meanNormalizedError']:.4f} | {row['medianNormalizedError']:.4f} | {display} |")
    lines += ["", "All horizon/regime/scope/weighting comparisons are retained in comparisons.csv and results.json. Definitions, predeclared coefficients and limitations are in protocol.json. No calibrated ranges, terminal values or investment returns are tested."]
    return "\n".join(lines) + "\n"


def run(source, output, protocol_only=False):
    source, output = source.resolve(), output.resolve()
    check_output(source, output)
    output.mkdir(parents=True, exist_ok=True)
    protocol_path = output / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != PROTOCOL:
        raise ValueError("Retained protocol differs; use a new experiment directory")
    if not protocol_path.exists():
        write_json(protocol_path, PROTOCOL)
        write_json(output / "protocol-lock.json", {"recordedAt": datetime.now(UTC).isoformat(), "protocolSha256": file_hash(protocol_path)})
    if protocol_only:
        return {"status": "protocol recorded", "output": str(output)}
    inputs = verify_inputs(source)
    script = Path(__file__)
    (output / "cash-flow-midline-challengers.executed.py").write_bytes(script.read_bytes())
    chunks, source_rows = [], 0
    for chunk in pd.read_csv(source / "forecasts.csv.gz", usecols=COLUMNS, keep_default_na=False,
                             dtype={"company_id": str, "isin": str}, chunksize=200_000):
        source_rows += len(chunk)
        selected = chunk.loc[chunk.window.eq(5) & chunk.model.eq("naive")]
        if not selected.empty:
            chunks.append(prepare_chunk(selected))
    frame = pd.concat(chunks, ignore_index=True)
    if frame.duplicated(["company_id", "origin_year", "horizon"]).any():
        raise ValueError("Duplicate inherited paired forecast key")
    frame.to_parquet(output / "forecast-ledger.parquet", index=False)
    audit = {"sourceRows": source_rows, "sourceFiveYearNaiveRows": len(frame), "sourceListings": int(frame.company_id.nunique()),
             "pointEligibleRows": int(frame.point_eligible.sum()), "pointEligibleListings": int(frame.loc[frame.point_eligible].company_id.nunique()),
             "zeroScaleRows": int(frame.scale.eq(0).sum()), "timingExclusions": dict(Counter(frame.loc[frame.timing_issue.ne(""), "timing_issue"])),
             "baselinePredictionScaleSlopeReconciledRows": len(frame), "missingSectorRows": int(frame.sector.eq("").sum()),
             "missingTargetPublicationEligibleRows": int((frame.point_eligible & frame.target_publication.eq("")).sum()),
             "sameSourceEligibleRows": int((frame.point_eligible & frame.same_source).sum()),
             "scopeListingCounts": {scope: int(frame.loc[frame.point_eligible & (frame.in_operating_scope if scope == "operating_and_property" else True)].company_id.nunique()) for scope in PROTOCOL["scopes"]}}
    results = comparisons(frame)
    write_json(output / "results.json", {"protocolId": PROTOCOL["id"], "audit": audit, "comparisons": results})
    pd.DataFrame(results).to_csv(output / "comparisons.csv", index=False)
    (output / "report.md").write_text(render_report(results, audit))
    if verify_inputs(source) != inputs:
        raise ValueError("Source identity changed during experiment")
    artifacts = [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
                 for path in sorted(output.iterdir()) if path.is_file()]
    receipt = {"status": "PASS", "completedAt": datetime.now(UTC).isoformat(), "protocolId": PROTOCOL["id"],
               "scriptSha256": file_hash(script), "inputs": inputs, "artifacts": artifacts}
    write_json(output / "receipt.json", receipt)
    return {"status": "PASS", "audit": audit, "comparisonCells": len(results), "output": str(output)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_directory", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--protocol-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.source_directory, arguments.output_directory, arguments.protocol_only), indent=2, allow_nan=False))
