"""Fixed exploratory peer-group experiment on saved cash-flow forecast errors."""
import argparse
import functools
import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = {
    "cash_dispersion": ("fcf_volatility", [.25, .75], ["low", "medium", "high"]),
    "cash_residual_dispersion": ("fcf_trend_residual_volatility", [.15, .5], ["low", "medium", "high"]),
    "cfo_dispersion": ("cfo_volatility", [.25, .75], ["low", "medium", "high"]),
    "revenue_dispersion": ("revenue_volatility", [.1, .3], ["low", "medium", "high"]),
    "ebit_margin": ("ebit_margin", [0, .05, .2], ["negative", "0–5%", "5–20%", "20%+"]),
    "margin_dispersion": ("ebit_margin_volatility", [.03, .1], ["under 3pp", "3–10pp", "10pp+"]),
    "tangible_assets_sales": ("tangible_assets_to_revenue", [.25, 1], ["low", "medium", "high"]),
    "tangible_assets_share": ("tangible_assets_to_total_assets", [.2, .6], ["low", "medium", "high"]),
    "tbv_ebit": ("tangible_book_to_ebit", [3, 10], ["under 3 years", "3–10 years", "10 years+"]),
    "net_debt_assets": ("net_debt_to_total_assets", [0, .25, .5], ["net cash", "0–25%", "25–50%", "50%+"]),
    "working_capital_sales": ("working_capital_proxy_to_revenue", [0, .2, .5], ["negative", "0–20%", "20–50%", "50%+"]),
    "cash_gap_dispersion": ("cash_gap_volatility", [.25, .75], ["low", "medium", "high"]),
}
CANDIDATES = ["global", "sector", "branch", *NUMERIC, "cash_sign", "sector_cash_dispersion", "branch_cash_dispersion"]
PARENTS = {"branch": "sector", "sector_cash_dispersion": "sector", "branch_cash_dispersion": "branch"}
PROTOCOL = {
    "id": "exploratory-cash-uncertainty-groups-v1", "date": "2026-09-12",
    "models": ["linear", "naive"], "historyYears": 5, "horizons": [1, 2, 3, 4],
    "candidates": CANDIDATES, "numericBins": NUMERIC, "targetCoverage": .8,
    "minimumCalibrationRows": 300, "minimumCalibrationListings": 100, "minimumCalibrationHistories": 100,
    "calibrationLastTargetYear": 2020, "nominalCalibrationCutoff": "2021-06-30",
    "validationTargets": [2021, 2022, 2023], "exploratoryRecentTargets": [2024, 2025],
    "scopes": ["operating_and_property", "all_listings"],
    "scopeDefinition": "Primary excludes Financials except its Real Estate and REITs branches. All-listing sensitivity includes financial businesses with unreviewed provider cash proxies.",
    "weighting": "Within each scope/model/horizon calibration pool, each listing has total weight one divided across its eligible folds; these fixed weights enter every group quantile. Evaluation gives each listing total weight one within the full scope/model/horizon/period pool. Subgroup diagnostics retain those pool weights rather than rebalancing each group.",
    "duplicateSensitivity": "Refit and evaluate after additionally dividing weights by identical origin-history-fingerprint/target-year multiplicity. Conflicting outcomes remain included. This is duplicate-history downweighting, not definitive issuer deduplication.",
    "factor": "Weighted nearest cumulative 80th percentile of absolute error / mean absolute five-year training cash. Half-width = factor × that scale; unchanged point forecast.",
    "fallback": "Sparse or missing single-feature group -> scope global. Branch -> sector -> global. Sector×cash -> sector; branch×cash -> branch -> sector -> global. All candidates retain the same scored rows.",
    "selection": "Fixed bins and candidates documented before new group results. No threshold search, winner deployment or claimed fresh confirmation on previously inspected recent years. Exploratory comparisons require future untouched outcomes.",
    "limitations": [
        "Source values are retained later-vintage financial statements; current sector and branch labels are descriptive, not verified historical classifications.",
        "Provider FCF measurement changes between downloads remain unresolved. Same-source snapshot membership does not prove consistent accounting definitions.",
        "Small within-company samples, related share classes and repeated targets prevent interpreting observations as independent trials.",
        "Cash dispersion and normalized error share a training cash denominator; conditional usefulness is not proof of intrinsic business risk or causality.",
        "Support minima and 80% coverage are chosen research settings, not statistical guarantees. Symmetric ranges need not have equal tail probabilities.",
        "Year 1 validation has only FY2022–2023 after timing gates; Year 2 only FY2023; Years 3–4 have no admissible validation outcomes. Recent higher horizons likewise span few calendar regimes.",
        "Missing ratio values receive parent/global ranges. They are never silently treated as asset-light or zero.",
        "Tangible assets, total equity minus intangibles, current working capital and net debt are vendor aggregates. They are not reconciled PP&E, common tangible equity, operating capital or maintenance capex.",
        "This experiment evaluates uncertainty around unchanged annual cash forecasts, not stock returns, equity-distributable cash or DCF terminal values.",
    ],
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def weighted_quantile(values, weights, quantile=.8):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Quantiles need finite values and positive total weight")
    order = np.argsort(values, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = min(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"), len(values) - 1)
    return float(values[order[index]])


def valid_date(value):
    try:
        return isinstance(value, str) and date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


@functools.lru_cache(maxsize=250_000)
def training_publications_valid(publications, ends, origin):
    pubs, dates = publications.split("|"), ends.split("|")
    return len(pubs) == len(dates) == 5 and valid_date(origin) and all(
        valid_date(p) and valid_date(e) and e <= p <= origin for p, e in zip(pubs, dates, strict=True)
    )


def calibration_eligible(row):
    pub, end, origin = row["target_publication"], row["target_end"], row["origin_publication"]
    return (int(row["target_year"]) <= 2020 and valid_date(pub) and valid_date(end)
            and end <= pub <= "2021-06-30" and valid_date(origin) and origin < end
            and training_publications_valid(row["train_publications"], row["train_end_dates"], origin))


def scope_mask(frame, scope):
    if scope == "all_listings":
        return pd.Series(True, index=frame.index)
    if scope != "operating_and_property":
        raise ValueError(scope)
    return frame.sector.ne("Financials") | frame.branch.isin(["Real Estate", "REITs"])


def add_groups(frame):
    frame = frame.copy()
    for name in ["sector", "branch"]:
        if name not in frame:
            frame[name] = ""
        frame[name] = frame[name].fillna("").astype(str)
    for candidate, (column, edges, labels) in NUMERIC.items():
        values = pd.to_numeric(frame[column], errors="coerce") if column in frame else pd.Series(np.nan, index=frame.index)
        groups = np.asarray(labels, dtype=object)[np.searchsorted(edges, values.to_numpy(), side="right")]
        groups[~np.isfinite(values.to_numpy())] = ""
        frame[candidate] = groups
    signs = frame.get("cash_sign_regime", pd.Series("", index=frame.index)).fillna("").astype(str)
    frame["cash_sign"] = signs.mask(signs.eq("missing"), "")
    # Composite keys include the parent context to avoid cross-sector name aliases.
    frame["branch"] = np.where(frame.branch.ne("") & frame.sector.ne(""), frame.sector + " / " + frame.branch, "")
    frame["sector_cash_dispersion"] = np.where(frame.sector.ne("") & frame.cash_dispersion.ne(""), frame.sector + " / " + frame.cash_dispersion, "")
    frame["branch_cash_dispersion"] = np.where(frame.branch.ne("") & frame.cash_dispersion.ne(""), frame.branch + " / " + frame.cash_dispersion, "")
    return frame


def pool_weights(frame, duplicate_aware=False):
    weights = 1. / frame.groupby("company_id")["company_id"].transform("size").to_numpy()
    if duplicate_aware:
        weights /= frame.groupby(["training_history_fingerprint", "target_year"])["company_id"].transform("size").to_numpy()
    return weights


def fit_factors(frame, weights, minimum_rows=300, minimum_ids=100, minimum_histories=100):
    factors = {}
    for candidate in CANDIDATES:
        groups = {"all": np.arange(len(frame))} if candidate == "global" else frame.groupby(candidate, sort=True).indices
        factors[candidate] = {}
        for key, indices in groups.items():
            if key == "":
                continue
            selected = frame.iloc[indices]
            ids = selected.company_id.nunique()
            histories = selected.training_history_fingerprint.nunique()
            supported = len(indices) >= minimum_rows and ids >= minimum_ids and histories >= minimum_histories
            factors[candidate][str(key)] = {
                "factor": weighted_quantile(selected.error.to_numpy(), weights[indices]) if supported else None,
                "count": len(indices), "listings": int(ids), "histories": int(histories),
                "weight": float(weights[indices].sum()), "supported": bool(supported),
            }
    return factors


def apply_factors(frame, candidate, factors):
    global_factor = factors["global"]["all"]["factor"]
    if global_factor is None:
        raise ValueError("Insufficient global calibration support")
    result = np.full(len(frame), global_factor, dtype=float)
    levels = np.full(len(frame), "global", dtype=object)
    chain, current = [], candidate
    while current != "global":
        chain.append(current)
        current = PARENTS.get(current, "global")
    for level in reversed(chain):
        supported = {key: value["factor"] for key, value in factors[level].items() if value["factor"] is not None}
        q = frame[level].map(supported).to_numpy(dtype=float)
        valid = np.isfinite(q)
        result[valid], levels[valid] = q[valid], level
    return result, levels


def metrics(frame, factor, weights, levels, candidate):
    error = frame.error.to_numpy()
    signed = frame.signed_error.to_numpy()
    inside, width = error <= factor, 2 * factor
    score = width + 10 * np.maximum(error - factor, 0)
    weight = weights.sum()
    def average(values):
        return float(np.dot(values, weights) / weight)
    worst = np.argsort(score)[-max(1, int(np.ceil(.01 * len(score)))):]
    contributions = score * weights
    return {
        "count": len(frame), "listings": int(frame.company_id.nunique()),
        "histories": int(frame.training_history_fingerprint.nunique()), "weight": float(weight),
        "targetYears": sorted(int(v) for v in frame.target_year.unique()),
        "coverage": average(inside), "coverageGap": average(inside) - .8,
        "belowShare": average(signed < -factor), "aboveShare": average(signed > factor),
        "inside": int(inside.sum()), "below": int((signed < -factor).sum()), "above": int((signed > factor).sum()),
        "foldCoverage": float(inside.mean()), "meanWidth": average(width),
        "meanIntervalScore": average(score), "medianNormalizedError": weighted_quantile(error, weights, .5),
        "meanNormalizedError": average(error), "meanNormalizedBias": average(-signed),
        "fallbackShare": average(levels != candidate),
        "worstOnePercentScoreShare": float(contributions[worst].sum() / contributions.sum()) if contributions.sum() else 0.,
    }


def prepare(directory):
    forecast_path, feature_path = directory / "segmentation-forecasts.csv.gz", directory / "features.jsonl.gz"
    upstream = json.loads((directory / "feature-summary.json").read_text())
    expected = {r["name"]: r["sha256"] for r in upstream["artifacts"]}
    input_hashes = {path.name: file_hash(path) for path in [forecast_path, feature_path]}
    if any(input_hashes[name] != expected[name] for name in input_hashes):
        raise ValueError("Feature-extraction artifact checksum mismatch")
    columns = ["company_id", "company_name", "sector", "branch", "model", "horizon", "origin_year", "target_year", "split", "currency",
               "origin_publication", "train_publications", "train_end_dates", "target_publication", "target_end", "training_mean_absolute_cash",
               "prediction", "actual", "calibration_unavailable", "train_source_ids", "target_source_id"]
    frame = pd.read_csv(forecast_path, usecols=columns, dtype={"company_id": str}, keep_default_na=False)
    with gzip.open(feature_path, "rt") as stream:
        features = pd.DataFrame(json.loads(line) for line in stream)
    if features.duplicated(["company_id", "origin_year"]).any():
        raise ValueError("Duplicate origin features")
    keep = ["company_id", "origin_year", "training_history_fingerprint", "cash_sign_regime", "tangible_book_to_ebit_status",
            *set(column for column, _, _ in NUMERIC.values())]
    joined = frame.merge(features[keep], on=["company_id", "origin_year"], how="left", validate="many_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise ValueError("Forecast origin has no verified feature row")
    joined = joined.drop(columns="_merge")
    scale = pd.to_numeric(joined.training_mean_absolute_cash, errors="coerce")
    valid_scale = np.isfinite(scale) & scale.gt(0)
    invalid_scale = int((~valid_scale).sum())
    joined = joined.loc[valid_scale].copy()
    scale = scale.loc[valid_scale]
    joined["signed_error"] = (joined.actual - joined.prediction) / scale
    joined["error"] = np.abs(joined.signed_error)
    timing_columns = ["target_year", "target_publication", "target_end", "origin_publication", "train_publications", "train_end_dates"]
    joined["calibration_ok"] = [
        calibration_eligible(dict(zip(timing_columns, values, strict=True)))
        for values in joined[timing_columns].itertuples(index=False, name=None)
    ]
    # The prior source-bound audit already checks strict periods and application publication gates.
    joined["application_ok"] = joined.target_year.gt(2020) & joined.calibration_unavailable.eq("")
    joined["in_operating_scope"] = scope_mask(joined, "operating_and_property")
    joined["branch_label"] = joined.branch
    joined["same_source"] = [len(set(train.split("|") + [target])) == 1 for train, target in zip(joined.train_source_ids, joined.target_source_id, strict=True)]
    joined = add_groups(joined)
    joined = joined.loc[joined.calibration_ok | joined.application_ok].reset_index(drop=True)
    joined.to_parquet(directory / "prepared-forecasts.parquet", index=False)
    metadata = {"sourceForecastRows": len(frame), "invalidScaleRows": invalid_scale,
                "retainedRows": len(joined), "featureOrigins": len(features),
                "forecastsSha256": input_hashes[forecast_path.name], "featuresSha256": input_hashes[feature_path.name],
                "upstreamSourceInputs": upstream["inputs"], "upstreamFeatureSummarySha256": file_hash(directory / "feature-summary.json")}
    (directory / "prepared-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return joined, metadata


def run(directory):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    frame, inputs = prepare(directory)
    results, details, calibrations, coverage = [], [], [], []
    outputs = {"protocol": PROTOCOL, "inputs": inputs}
    for scope in PROTOCOL["scopes"]:
        scoped = frame.loc[frame.in_operating_scope] if scope == "operating_and_property" else frame
        for model in PROTOCOL["models"]:
            for horizon in PROTOCOL["horizons"]:
                cell = scoped.loc[scoped.model.eq(model) & scoped.horizon.eq(horizon)]
                cal = cell.loc[cell.calibration_ok].reset_index(drop=True)
                if not len(cal):
                    continue
                for weighting in ["listing_balanced", "duplicate_history_downweighted"]:
                    duplicate_aware = weighting == "duplicate_history_downweighted"
                    factors = fit_factors(cal, pool_weights(cal, duplicate_aware))
                    identity = {"scope": scope, "model": model, "horizon": horizon, "weighting": weighting}
                    calibrations.append({**identity, "groups": factors})
                    for period, mask in [("validation", cell.target_year.between(2021, 2023)), ("recent", cell.target_year.between(2024, 2025))]:
                        test = cell.loc[mask & cell.application_ok].reset_index(drop=True)
                        if not len(test):
                            continue
                        if weighting == "listing_balanced":
                            by_history = test.groupby(["training_history_fingerprint", "target_year"]).actual
                            coverage.append({**identity, "period": period, "count": len(test), "listings": int(test.company_id.nunique()),
                                             "uniqueHistoryTargets": len(by_history), "conflictingHistoryTargets": int((by_history.nunique() > 1).sum()),
                                             "sameSourceCount": int(test.same_source.sum())})
                        weights = pool_weights(test, duplicate_aware)
                        baseline, baseline_levels = apply_factors(test, "global", factors)
                        base_metrics = metrics(test, baseline, weights, baseline_levels, "global")
                        for candidate in CANDIDATES:
                            q, levels = apply_factors(test, candidate, factors)
                            scored = metrics(test, q, weights, levels, candidate)
                            scored["scoreImprovement"] = 1 - scored["meanIntervalScore"] / base_metrics["meanIntervalScore"]
                            results.append({**identity, "period": period, "candidate": candidate, **scored})
                            for year, indices in test.groupby("target_year").indices.items():
                                group = test.iloc[indices]
                                # A single target-year row per listing: annual evaluation weights are regenerated.
                                year_weights = pool_weights(group, duplicate_aware)
                                annual = metrics(group, q[indices], year_weights, levels[indices], candidate)
                                annual_base = metrics(group, baseline[indices], year_weights, baseline_levels[indices], "global")
                                annual["scoreImprovement"] = 1 - annual["meanIntervalScore"] / annual_base["meanIntervalScore"]
                                results.append({**identity, "period": f"FY{year}", "candidate": candidate, **annual})
                            if model == "linear" and horizon <= 2 and weighting == "listing_balanced" and candidate != "global":
                                for key, indices in test.groupby(candidate, dropna=False).indices.items():
                                    group = test.iloc[indices]
                                    part = metrics(group, q[indices], weights[indices], levels[indices], candidate)
                                    original = metrics(group, baseline[indices], weights[indices], baseline_levels[indices], "global")
                                    details.append({**identity, "period": period, "candidate": candidate, "group": key or "Missing → fallback",
                                                    **part, "globalCoverage": original["coverage"], "globalWidth": original["meanWidth"],
                                                    "globalScore": original["meanIntervalScore"], "scoreImprovement": 1 - part["meanIntervalScore"] / original["meanIntervalScore"]})
                    print(f"Finished {scope} / {model} / year {horizon} / {weighting}", flush=True)
    outputs.update(metrics=results, groupMetrics=details, eligibility=coverage,
                   scriptSha256=file_hash(Path(__file__)))
    (directory / "results.json").write_text(json.dumps(outputs, indent=2, allow_nan=False) + "\n")
    (directory / "group-calibration.json").write_text(json.dumps({"protocol": PROTOCOL, "calibrations": calibrations}, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(results).to_csv(directory / "comparisons.csv", index=False)
    pd.DataFrame(details).to_csv(directory / "group-diagnostics.csv", index=False)
    names = ["protocol.json", "prepared-metadata.json", "results.json", "group-calibration.json", "comparisons.csv", "group-diagnostics.csv"]
    receipt = {"scriptSha256": outputs["scriptSha256"], "inputs": inputs,
               "artifacts": [{"name": name, "sha256": file_hash(directory / name), "bytes": (directory / name).stat().st_size} for name in names]}
    (directory / "experiment-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--protocol-only", action="store_true")
    args = parser.parse_args()
    if args.protocol_only:
        args.directory.mkdir(parents=True, exist_ok=True)
        (args.directory / "protocol.json").write_text(json.dumps(PROTOCOL, indent=2) + "\n")
    else:
        run(args.directory)
