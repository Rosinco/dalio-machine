"""Copy the original dated research factors into an immutable app data bundle.

No fitting, threshold selection or crisis-year removal occurs here. The original
research outputs and content-addressed financial pack are verified before export.
"""
import argparse
import copy
import gzip
import hashlib
import json
import math
import sqlite3
from pathlib import Path

ARTIFACT_ID = "cash-uncertainty-2026-09-12-v1"
SCOPE = "operating_and_property"
WEIGHTING = "listing_balanced"
MODELS = ("linear", "naive")
HORIZONS = (1, 2, 3, 4)
SOURCE_PACK_HASH = "1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a"
SOURCE_TAXONOMY_HASH = "cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff"
SOURCE_HASHES = {
    "group-calibration.json": "bf386c3709497665ad1fe23df1f3818a87686da5792c4c75d61d80509a1595a5",
    "results.json": "7d79048bb2390ffe41f806f185da45f850df69dac48186580cffad96a62219eb",
    "experiment-receipt.json": "bf96bd1c091c898f1c5d3c349cb82871eb73b2444674b0a4dbcf5e8a0c5d22e4",
    "feature-summary.json": "6850553b6c5470b581240337e570fe65bc964c73992a17953d0ae1871b9019ca",
}
METRIC_KEYS = (
    "count", "listings", "histories", "targetYears", "weight", "coverage", "belowShare", "aboveShare",
    "meanWidth", "meanIntervalScore", "medianNormalizedError", "meanNormalizedError", "fallbackShare",
)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_verified_sources(directory):
    directory = Path(directory)
    loaded = {}
    for name, expected in SOURCE_HASHES.items():
        path = directory / name
        if not path.exists() or file_hash(path) != expected:
            raise ValueError(f"Original research source identity mismatch: {name}")
        loaded[name] = json.loads(path.read_text())
    receipt = loaded["experiment-receipt.json"]
    identities = {row["name"]: row["sha256"] for row in receipt["artifacts"]}
    for name in ("group-calibration.json", "results.json"):
        if identities.get(name) != SOURCE_HASHES[name]:
            raise ValueError(f"Receipt source identity mismatch: {name}")
    features = loaded["feature-summary.json"]
    if receipt["inputs"] != loaded["results.json"]["inputs"]:
        raise ValueError("Results do not match the original experiment inputs")
    pack_descriptor = features["inputs"]["pack"]
    pack_path = Path(pack_descriptor["path"])
    if pack_descriptor["sha256"] != SOURCE_PACK_HASH or file_hash(pack_path) != SOURCE_PACK_HASH:
        raise ValueError("Financial source identity mismatch")
    connection = sqlite3.connect(f"file:{pack_path}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        compressed = connection.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0]
        index = json.loads(gzip.decompress(compressed))
    finally:
        connection.close()
    if index["taxonomy_sha256"] != SOURCE_TAXONOMY_HASH or index["as_of"] != pack_descriptor["asOf"]:
        raise ValueError("Financial source taxonomy or vintage mismatch")
    if file_hash(pack_path) != SOURCE_PACK_HASH:
        raise ValueError("Financial source changed during read-only export")
    return {"calibration": loaded["group-calibration.json"], "results": loaded["results.json"],
            "receipt": receipt, "featureSummary": features,
            "sourceIndex": {"as_of": index["as_of"], "taxonomy_sha256": index["taxonomy_sha256"],
                            "sources": index["sources"]}}


def build_artifact(sources):
    calibration, results = sources["calibration"], sources["results"]
    protocol = calibration["protocol"]
    if protocol != results["protocol"]:
        raise ValueError("Calibration and evaluation protocols differ")
    required = {"id": "exploratory-cash-uncertainty-groups-v1", "date": "2026-09-12",
                "historyYears": 5, "targetCoverage": .8, "calibrationLastTargetYear": 2020,
                "nominalCalibrationCutoff": "2021-06-30", "minimumCalibrationRows": 300,
                "minimumCalibrationListings": 100, "minimumCalibrationHistories": 100,
                "models": list(MODELS), "horizons": list(HORIZONS)}
    if any(protocol.get(key) != value for key, value in required.items()):
        raise ValueError("Unexpected original research protocol")
    if protocol["numericBins"]["cash_dispersion"] != ["fcf_volatility", [.25, .75], ["low", "medium", "high"]]:
        raise ValueError("Unexpected cash dispersion definition")
    factors = {model: {} for model in MODELS}
    evaluations = {model: {str(h): {} for h in HORIZONS} for model in MODELS}
    for row in calibration["calibrations"]:
        if row["scope"] != SCOPE or row["weighting"] != WEIGHTING:
            continue
        model, horizon = row["model"], str(row["horizon"])
        if model not in MODELS or row["horizon"] not in HORIZONS or horizon in factors[model]:
            raise ValueError("Unexpected or duplicate calibration model/horizon")
        factors[model][horizon] = {
            "global": copy.deepcopy(row["groups"]["global"]["all"]),
            "cashDispersion": copy.deepcopy(row["groups"]["cash_dispersion"]),
        }
    for row in results["metrics"]:
        if row["scope"] != SCOPE or row["weighting"] != WEIGHTING or row["candidate"] not in ("global", "cash_dispersion"):
            continue
        if row["model"] not in MODELS or row["horizon"] not in HORIZONS:
            raise ValueError("Unexpected evaluation model/horizon")
        period = evaluations[row["model"]][str(row["horizon"])].setdefault(row["period"], {})
        key = "cashDispersion" if row["candidate"] == "cash_dispersion" else "global"
        if key in period:
            raise ValueError("Duplicate evaluation identity")
        period[key] = {name: copy.deepcopy(row[name]) for name in METRIC_KEYS}
    evidence = {}
    for horizon in HORIZONS:
        periods = evaluations["linear"][str(horizon)]
        evidence[str(horizon)] = {
            "validationTargetYears": periods.get("validation", {}).get("global", {}).get("targetYears", []),
            "exploratoryRecentTargetYears": periods.get("recent", {}).get("global", {}).get("targetYears", []),
            "status": "research" if horizon <= 2 else "limited_recent_regimes_only",
        }
    index = sources["sourceIndex"]
    features = sources["featureSummary"]
    artifact = {
        "id": ARTIFACT_ID, "schemaVersion": 1, "researchDate": "2026-09-12", "status": "research",
        "targetCoverage": .8, "latestEmpiricalHorizon": 4, "historyYears": 5,
        "scope": SCOPE, "weighting": WEIGHTING,
        "modelDefinitions": {
            "linear": {"description": "Weighted least-squares line over five signed annual cash values, extrapolated from its fitted latest intercept.",
                       "newestFirstWeights": [30, 25, 20, 15, 10]},
            "naive": {"description": "Latest annual signed cash held constant in every forecast year."},
        },
        "cashFlowBasis": {"field": "free_cash_flow", "currency": "native_reporting_currency",
                          "conversion": "Each saved raw amount divided by its own positive currency_ratio.",
                          "requiresConsistentReportingCurrency": True,
                          "quoteCurrencyFallbackTested": False},
        "calibrationTiming": "Calibration targets are FY2020 or earlier, with known publication by 2021-06-30. Training publications must be known, on or after their fiscal end and no later than origin publication, which precedes target fiscal end. Evaluation applies factors only to origins FY2021 or later published on or after the cutoff.",
        "scopeDefinition": "Operating companies and property: excludes Financials except Real Estate and REITs. Provider cash remains an unreviewed proxy.",
        "sourceAsOf": index["as_of"], "taxonomySha256": index["taxonomy_sha256"],
        "sourceFiles": [{key: row[key] for key in ("id", "as_of", "path", "sha256", "frequency")}
                        for row in index["sources"] if row["frequency"] == "annual"],
        "calibrationCutoff": "2021-06-30", "calibrationLastTargetYear": 2020,
        "minimumCalibrationRows": 300, "minimumCalibrationListings": 100, "minimumCalibrationHistories": 100,
        "covidTreatment": {"omittedTargetYears": [], "historicalShocksRemoved": False,
                           "description": "Original all-year calibration retains FY2020 and the unchanged raw histories. No COVID-omission sensitivity or normalization is used."},
        "cashDispersion": {"bounds": [.25, .75], "labels": ["low", "medium", "high"],
                           "boundConvention": "Lower-inclusive groups: low <0.25; medium >=0.25 and <0.75; high >=0.75.",
                           "definition": features["definitions"]["fcf_volatility"],
                           "requiredHistoryYears": 5, "standardDeviationDegreesOfFreedom": 0},
        "rangeDefinition": {"scale": "Mean absolute cash across all five signed annual history values, unweighted and strictly positive.",
                            "halfWidth": "The selected model/horizon factor times the historical cash scale.",
                            "bounds": "Mid cash minus/plus half-width. Signed cash and zero crossings are retained.",
                            "quantile": protocol["factor"], "weighting": protocol["weighting"],
                            "fallback": "An unavailable or unsupported cash group uses the model/horizon's global pool. A different peer pool need not imply a wider range.",
                            "ineligibleHistory": "Partial, invalid, unavailable or zero-scale histories have no empirical interval; any starter range for them is an explicit assumption.",
                            "beyondLatestEmpiricalHorizon": "No calibrated factor is provided after Year 4. Longer-horizon values require separately labelled assumptions.",
                            "probabilityMeaning": "An empirical 80% research target for annual cash, not a guarantee for a company, joint cash path or DCF valuation."},
        "calibrationByModelHorizon": factors, "evaluationByModelHorizon": evaluations,
        "horizonEvidence": evidence,
        "evaluationMeaning": "Global and cashDispersion identify complete range rules on matching scored observations. Coverage is weighted empirical coverage; widths are full widths divided by historical cash scale. These are not per-company probabilities.",
        "limitations": [
            "Later-vintage source statements and current taxonomy do not recreate point-in-time financial information or historical company classifications.",
            "Provider FCF definition differences between downloads remain unresolved; matching source lineage does not establish reconciled FCFF or distributable equity cash.",
            "The retained universe can omit former or delisted issuers. Related listings, repeated targets and origin histories are dependent.",
            "Recent FY2024-2025 outcomes were inspected in the earlier audit and remain exploratory. New untouched outcomes are needed for confirmation.",
            "Year 1 validation contains FY2022-2023 and Year 2 only FY2023. Years 3-4 have no admissible validation period; Year 4 recent results cover only FY2025.",
            "Cash dispersion includes trend and growth and shares a denominator with normalized forecast errors; this is not proof of underlying business risk or causation.",
            "Negative cash remains signed. Narrow uncertainty around losses does not indicate an attractive business or investment.",
            "Support counts and an 80% target are research choices, not probability guarantees. Annual ranges do not establish joint cash-path or DCF intervals.",
        ],
        "provenance": {"sourceProtocolId": protocol["id"], "inputArtifactSha256": dict(SOURCE_HASHES),
                       "sourcePack": {"path": f"financial-data/{SOURCE_PACK_HASH}.sqlite", "sha256": SOURCE_PACK_HASH, "asOf": index["as_of"]},
                       "sourceForecastsSha256": features["inputs"]["forecasts"]["sha256"],
                       "featureRowsSha256": results["inputs"]["featuresSha256"],
                       "experimentScriptSha256": sources["receipt"]["scriptSha256"],
                       "featureExtractorScriptSha256": features["scriptSha256"],
                       "exportMethod": "Exact factors, support records and selected evaluation fields copied without recomputation, refitting, threshold changes or crisis-year deletion."},
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact):
    if (artifact["id"] != ARTIFACT_ID or artifact["status"] != "research" or artifact["targetCoverage"] != .8
            or artifact["historyYears"] != 5 or artifact["latestEmpiricalHorizon"] != 4
            or artifact["scope"] != SCOPE or artifact["weighting"] != WEIGHTING
            or artifact["calibrationLastTargetYear"] != 2020 or artifact["calibrationCutoff"] != "2021-06-30"
            or artifact["covidTreatment"]["omittedTargetYears"] or artifact["covidTreatment"]["historicalShocksRemoved"]
            or artifact["cashDispersion"]["bounds"] != [.25, .75]
            or artifact["modelDefinitions"]["linear"]["newestFirstWeights"] != [30, 25, 20, 15, 10]
            or artifact["cashFlowBasis"]["currency"] != "native_reporting_currency"
            or artifact["cashFlowBasis"]["quoteCurrencyFallbackTested"] is not False
            or artifact["taxonomySha256"] != SOURCE_TAXONOMY_HASH):
        raise ValueError("Invalid dated research contract")
    if artifact["provenance"]["inputArtifactSha256"] != SOURCE_HASHES:
        raise ValueError("Invalid original source identities")
    factors = artifact["calibrationByModelHorizon"]
    if set(factors) != set(MODELS):
        raise ValueError("Invalid calibrated models")
    for model, horizons in factors.items():
        if set(horizons) != {str(h) for h in HORIZONS}:
            raise ValueError("Invalid calibrated horizons")
        for horizon, cell in horizons.items():
            if set(cell["cashDispersion"]) != {"low", "medium", "high"}:
                raise ValueError("Invalid cash groups")
            for support in [cell["global"], *cell["cashDispersion"].values()]:
                for key in ("count", "listings", "histories"):
                    if type(support[key]) is not int or support[key] < 0:
                        raise ValueError("Invalid support count")
                supported = support["count"] >= 300 and support["listings"] >= 100 and support["histories"] >= 100
                if support["supported"] is not supported or min(support["count"] - support["listings"], support["count"] - support["histories"]) < 0:
                    raise ValueError("Inconsistent calibration support")
                factor = support["factor"]
                if supported and (not isinstance(factor, (int, float)) or not math.isfinite(factor) or factor <= 0):
                    raise ValueError("Invalid calibrated factor")
                if not supported and factor is not None:
                    raise ValueError("Unsupported group must not invent a factor")
                if not math.isfinite(support["weight"]) or support["weight"] < 0:
                    raise ValueError("Invalid calibration weight")
            if not cell["global"]["supported"]:
                raise ValueError("Missing supported global fallback")
            periods = artifact["evaluationByModelHorizon"][model][horizon]
            if "recent" not in periods:
                raise ValueError("Missing research evaluation")
            for candidates in periods.values():
                if set(candidates) != {"global", "cashDispersion"}:
                    raise ValueError("Incomplete matching evaluation")
                if any(candidates["global"][key] != candidates["cashDispersion"][key]
                       for key in ("count", "listings", "histories", "targetYears")):
                    raise ValueError("Evaluation rules do not share the same cohort")
                for row in candidates.values():
                    if not 0 <= row["coverage"] <= 1 or row["meanWidth"] < 0 or row["meanIntervalScore"] < 0:
                        raise ValueError("Invalid evaluation metrics")
    json.dumps(artifact, allow_nan=False)


def serialize(artifact):
    return (json.dumps(artifact, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()


def write_immutable(path, content):
    path = Path(path)
    if path.exists():
        if path.read_bytes() != content:
            raise ValueError("Refusing to replace an immutable dated calibration artifact; create a new version instead")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)


if __name__ == "__main__":
    root = Path(__file__).parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=root / "test-results/cash-flow-segmentation-2026-09-12")
    parser.add_argument("--output", type=Path, default=root / "src/data/cash-uncertainty-2026-09-12.json")
    parser.add_argument("--check", action="store_true", help="Verify exact reproduction without writing")
    args = parser.parse_args()
    data = build_artifact(load_verified_sources(args.directory))
    content = serialize(data)
    if args.check:
        if not args.output.exists() or args.output.read_bytes() != content:
            raise ValueError("Dated calibration artifact does not reproduce exactly")
    else:
        write_immutable(args.output, content)
    print(json.dumps({"id": data["id"], "path": str(args.output), "bytes": len(content),
                      "sha256": hashlib.sha256(content).hexdigest(), "mode": "check" if args.check else "immutable export"}))
