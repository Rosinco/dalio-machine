"""Extract strict, origin-only financial features for offline uncertainty research.

The saved pack is a retrospective merged vintage. These features respect fiscal
origins but do not restore point-in-time accounting or historical taxonomy.
They are descriptive source proxies, not verified operating-capital measures.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from statistics import mean, median, pstdev

PROTOCOL_ID = "origin-only-five-year-segmentation-features-v1"
MINIMUM_EBIT_MARGIN = 0.01
FEATURE_DEFINITIONS = {
    "tangible_assets_to_revenue": "Median of five annual vendor tangible_assets / positive revenues; nonnegative assets required. Vendor proxy, not verified PP&E.",
    "tangible_assets_to_total_assets": "Median of five annual nonnegative tangible_assets / positive total_assets.",
    "tangible_book_to_ebit": "Median of five annual (total_equity - nonnegative intangible_assets) / operating_income; positive tangible book and EBIT/revenues >= 1% required in every year. This mixes an equity stock with pre-interest earnings and is not pure asset intensity.",
    "ebit_margin": "Median of five annual operating_income / positive revenues; losses retained.",
    "ebit_margin_volatility": "Population standard deviation of five annual EBIT/revenue margins; losses retained.",
    "fcf_volatility": "Population standard deviation of five signed FCF levels / mean absolute five-year FCF; missing for zero scale. This includes trend and growth.",
    "fcf_trend_residual_volatility": "Unweighted population standard deviation of historical residuals around the 30/25/20/15/10 newest-first weighted fitted line, divided by mean absolute five-year FCF. In-sample dispersion, not an uncertainty interval.",
    "cfo_volatility": "Population standard deviation of five signed CFO levels / mean absolute five-year CFO; missing for zero scale.",
    "revenue_volatility": "Population standard deviation of five positive annual revenue levels / their mean; includes growth, not just cyclical shocks.",
    "net_debt_to_mean_ebit": "Latest signed net_debt / mean five-year EBIT, provided mean EBIT > 0 and mean EBIT / mean positive revenue >= 1%; negative debt is retained as net cash.",
    "net_debt_to_total_assets": "Latest signed net_debt / positive latest total_assets; negative debt is retained as net cash.",
    "working_capital_proxy_to_revenue": "Median five-year (current_assets - cash_and_equivalents - current_liabilities) / positive revenues. Imperfect balance-sheet proxy; current liabilities include financing and this is not true operating working capital.",
    "cash_gap_level": "Median absolute (CFO - provider FCF) / mean absolute CFO, all five years required. Broad investing/cash-definition gap; not assumed to be capex.",
    "cash_gap_volatility": "Population standard deviation of signed (CFO - provider FCF) / mean absolute CFO, all five years required; not verified investment spending.",
    "cash_sign_regime": "Signs of all five signed historical FCF values; exact zero retained separately.",
}


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def parse_day(value):
    if not isinstance(value, str) or len(value) != 10:
        raise ValueError("invalid_period_date")
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("invalid_period_date") from exc
    if parsed.isoformat() != value:
        raise ValueError("invalid_period_date")
    return parsed


def decode_training(company, index, origin_year, *, expected_years=None,
                    expected_source_ids=None, expected_cash=None, expected_currency=None):
    """Return newest-first five native annual rows, never inspecting future payloads."""
    years = list(range(origin_year, origin_year - 5, -1))
    by_year = {}
    for packed in company.get("annual", []):
        if not isinstance(packed, list) or not packed or packed[0] not in years:
            continue
        if packed[0] in by_year:
            raise ValueError("duplicate_year")
        by_year[packed[0]] = packed
    withheld = {row.get("year") for row in company.get("withheld", []) if row.get("period") == 5}
    sources = {source["id"]: source for source in index["sources"]}
    columns = index["columns"]
    training = []
    for year in years:
        if year in withheld:
            raise ValueError("withheld")
        packed = by_year.get(year)
        if packed is None:
            raise ValueError("missing_annual")
        if len(packed) != len(columns) + 8 or packed[1] != 5:
            raise ValueError("invalid_period")
        start, end = parse_day(packed[2]), parse_day(packed[3])
        if not 330 <= (end - start).days + 1 <= 400:
            raise ValueError("not_full_annual")
        currency, ratio, source_id = packed[5:8]
        if not isinstance(currency, str) or len(currency) != 3 or not currency.isascii() or not currency.isupper() or not currency.isalpha():
            raise ValueError("currency_missing")
        if not finite(ratio) or ratio <= 0:
            raise ValueError("currency_ratio_missing")
        if source_id not in sources:
            raise ValueError("unknown_source")
        raw = packed[8:]
        if any(value == 0 for value in raw) and all(value is None or value == 0 for value in raw):
            raise ValueError("all_amounts_zero")
        values = {column: value / ratio if finite(value) and finite(value / ratio) else None
                  for column, value in zip(columns, raw, strict=True)}
        if not finite(values.get("free_cash_flow")):
            raise ValueError("cash_missing")
        if training:
            if currency != training[0]["currency"]:
                raise ValueError("training_currency_change")
            if not 1 <= (parse_day(training[-1]["start"]) - end).days <= 35:
                raise ValueError("training_gap_or_overlap")
        training.append({"year": year, "start": packed[2], "end": packed[3],
                         "publication": packed[4], "currency": currency, "currency_ratio": ratio,
                         "source_id": source_id, "source_as_of": sources[source_id]["as_of"],
                         "values": values})
    if expected_years is not None and years != expected_years:
        raise ValueError("forecast_training_years_mismatch")
    if expected_source_ids is not None and [row["source_id"] for row in training] != expected_source_ids:
        raise ValueError("forecast_training_source_ids_mismatch")
    if expected_currency is not None and training[0]["currency"] != expected_currency:
        raise ValueError("forecast_training_currency_mismatch")
    if expected_cash is not None:
        cash = [row["values"]["free_cash_flow"] for row in training]
        if len(expected_cash) != 5 or any(not finite(b) or not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-10)
                                          for a, b in zip(cash, expected_cash, strict=True)):
            raise ValueError("forecast_training_cash_mismatch")
    return training


def training_fingerprint(training, columns):
    vectors = [[row["year"], row["start"], row["end"], row["currency"],
                [None if row["values"].get(column) is None else float(row["values"][column]) for column in columns]]
               for row in training]
    return hashlib.sha256(json.dumps(vectors, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def extract_features(training):
    if len(training) != 5:
        raise ValueError("complete_five_year_window_required")
    rows = [row["values"] for row in training]
    output, valid_counts = {}, {}

    def ratio_value(numerator, denominator, *, nonnegative=False):
        if not finite(numerator) or not finite(denominator) or denominator <= 0:
            return None
        if nonnegative and numerator < 0:
            return None
        result = numerator / denominator
        return result if finite(result) else None

    def aggregate(key, values, method=median):
        available = [value for value in values if finite(value)]
        valid_counts[key] = len(available)
        value = method(available) if len(available) == 5 else None
        output[key] = value if finite(value) else None

    def levels(key, column, positive=False):
        values = [row.get(column) for row in rows]
        valid = [value for value in values if finite(value) and (not positive or value > 0)]
        valid_counts[key] = len(valid)
        scale = mean([abs(value) for value in valid]) if len(valid) == 5 else 0
        output[key] = pstdev(valid) / scale if scale > 0 else None
        return valid, scale

    aggregate("tangible_assets_to_revenue", [ratio_value(row.get("tangible_assets"), row.get("revenues"), nonnegative=True) for row in rows])
    aggregate("tangible_assets_to_total_assets", [ratio_value(row.get("tangible_assets"), row.get("total_assets"), nonnegative=True) for row in rows])
    margins = [ratio_value(row.get("operating_income"), row.get("revenues")) for row in rows]
    aggregate("ebit_margin", margins)
    aggregate("ebit_margin_volatility", margins, pstdev)
    cash, cash_scale = levels("fcf_volatility", "free_cash_flow")
    cfo, cfo_scale = levels("cfo_volatility", "cash_flow_from_operating_activities")
    levels("revenue_volatility", "revenues", positive=True)

    # The fit uses exactly the application's fixed newest-first WLS weights.
    residual_dispersion = None
    if len(cash) == 5 and cash_scale > 0:
        x = [row["year"] - training[0]["year"] for row in training]
        weights = [30, 25, 20, 15, 10]
        x_bar = sum(value * weight for value, weight in zip(x, weights, strict=True)) / 100
        y_bar = sum(value * weight for value, weight in zip(cash, weights, strict=True)) / 100
        covariance = sum(weight * (xx - x_bar) * (yy - y_bar) for xx, yy, weight in zip(x, cash, weights, strict=True))
        variance = sum(weight * (xx - x_bar) ** 2 for xx, weight in zip(x, weights, strict=True))
        slope = covariance / variance if variance else 0
        intercept = y_bar - slope * x_bar
        residual_dispersion = pstdev([yy - intercept - slope * xx for xx, yy in zip(x, cash, strict=True)]) / cash_scale
    output["fcf_trend_residual_volatility"] = residual_dispersion
    valid_counts["fcf_trend_residual_volatility"] = len(cash)
    signs = {math.copysign(1, value) if value != 0 else 0 for value in cash}
    output["cash_sign_regime"] = (
        "missing" if len(cash) != 5 else "positive" if signs == {1} else "negative" if signs == {-1}
        else "zero" if signs == {0} else "nonnegative_with_zero" if signs == {0, 1}
        else "nonpositive_with_zero" if signs == {0, -1} else "mixed"
    )

    book_ratios, book_flags = [], set()
    for row, margin in zip(rows, margins, strict=True):
        equity, intangible, ebit = row.get("total_equity"), row.get("intangible_assets"), row.get("operating_income")
        issues = []
        if not all(finite(value) for value in [equity, intangible, ebit]) or margin is None:
            issues.append("missing_or_invalid_inputs")
        else:
            if intangible < 0:
                issues.append("missing_or_invalid_inputs")
            if equity - intangible <= 0:
                issues.append("nonpositive_tangible_book")
            if ebit <= 0:
                issues.append("nonpositive_ebit")
            elif margin < MINIMUM_EBIT_MARGIN:
                issues.append("near_zero_positive_ebit")
        book_flags.update(issues)
        book_ratios.append((equity - intangible) / ebit if not issues else None)
    aggregate("tangible_book_to_ebit", book_ratios)
    flag_order = ["missing_or_invalid_inputs", "nonpositive_tangible_book", "nonpositive_ebit", "near_zero_positive_ebit"]
    output["tangible_book_to_ebit_flags"] = [flag for flag in flag_order if flag in book_flags]
    output["tangible_book_to_ebit_status"] = output["tangible_book_to_ebit_flags"][0] if book_flags else "valid"

    ebit = [row.get("operating_income") for row in rows]
    revenue = [row.get("revenues") for row in rows]
    net_debt = rows[0].get("net_debt")
    debt_status = "valid"
    if not finite(net_debt) or not all(finite(value) for value in ebit) or not all(finite(value) and value > 0 for value in revenue):
        debt_status = "missing_or_invalid_inputs"
    elif mean(ebit) <= 0:
        debt_status = "nonpositive_mean_ebit"
    elif mean(ebit) / mean(revenue) < MINIMUM_EBIT_MARGIN:
        debt_status = "near_zero_positive_mean_ebit"
    output["net_debt_to_mean_ebit_status"] = debt_status
    output["net_debt_to_mean_ebit"] = net_debt / mean(ebit) if debt_status == "valid" else None
    # Latest ratios need only the origin balance sheet; their five-year coverage
    # is intentionally absent from valid_counts.
    output["net_debt_to_total_assets"] = ratio_value(net_debt, rows[0].get("total_assets"))

    working_capital = []
    gaps = []
    for row in rows:
        working = [row.get(key) for key in ["current_assets", "cash_and_equivalents", "current_liabilities"]]
        numerator = working[0] - working[1] - working[2] if all(finite(value) and value >= 0 for value in working) else None
        working_capital.append(ratio_value(numerator, row.get("revenues")))
        operating, free = row.get("cash_flow_from_operating_activities"), row.get("free_cash_flow")
        gaps.append(operating - free if finite(operating) and finite(free) else None)
    aggregate("working_capital_proxy_to_revenue", working_capital)
    aggregate("cash_gap_level", [abs(value) / cfo_scale if finite(value) and cfo_scale > 0 else None for value in gaps])
    aggregate("cash_gap_volatility", [value / cfo_scale if finite(value) and cfo_scale > 0 else None for value in gaps], pstdev)
    output["valid_counts"] = valid_counts
    return output


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_origins(forecasts, slim_path):
    """Retain a unique source-bound origin and stream the predeclared lean subset."""
    origins, read_rows, slim_rows = {}, 0, 0
    with gzip.open(forecasts, "rt", newline="") as source, gzip.open(slim_path, "wt", newline="", compresslevel=5) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            read_rows += 1
            if row["window"] != "5":
                continue
            if row["model"] in {"linear", "naive"} and int(row["horizon"]) <= 4:
                writer.writerow(row)
                slim_rows += 1
            key = (row["company_id"], int(row["origin_year"]))
            signature = tuple(row[name] for name in ["train_years", "train_source_ids", "train_cash", "currency", "company_source_sha256"])
            if key in origins:
                if signature != origins[key]["signature"]:
                    raise ValueError(f"Inconsistent existing forecast lineage for {key}")
                continue
            origins[key] = {
                "signature": signature,
                **{name: row[name] for name in ["company_id", "company_name", "isin", "sector", "branch", "currency", "company_source_sha256"]},
                "origin_year": key[1], "train_years": [int(value) for value in row["train_years"].split("|")],
                "train_source_ids": row["train_source_ids"].split("|"),
                "train_cash": [float(value) for value in row["train_cash"].split("|")],
            }
            if len(origins) % 10000 == 0:
                print(json.dumps({"stage": "collect_origins", "sourceRowsRead": read_rows, "uniqueOrigins": len(origins)}), flush=True)
    return origins, read_rows, slim_rows


def run(pack_path, forecast_path, output_dir):
    pack_path, forecast_path, output_dir = Path(pack_path).resolve(), Path(forecast_path).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pack_hash, forecast_hash = file_hash(pack_path), file_hash(forecast_path)
    receipt = json.loads(forecast_path.with_name("receipt.json").read_text())
    expected_forecast = next(row["sha256"] for row in receipt["artifacts"] if row["name"] == forecast_path.name)
    if pack_hash != receipt["sourcePackSha256"] or forecast_hash != expected_forecast:
        raise ValueError("Source pack or forecast checksum does not match original audit receipt")
    slim_path = output_dir / "segmentation-forecasts.csv.gz"
    origins, source_rows, slim_rows = collect_origins(forecast_path, slim_path)
    print(json.dumps({"stage": "extract_features", "origins": len(origins), "slimRows": slim_rows}), flush=True)
    connection = sqlite3.connect(f"file:{pack_path}?mode=ro", uri=True)
    connection.execute("PRAGMA query_only=ON")
    index = json.loads(gzip.decompress(connection.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0]))
    feature_path = output_dir / "features.jsonl.gz"
    available, categories, sectors = Counter(), {}, Counter()
    listings, fingerprints = set(), set()
    current_id, company, count = None, None, 0
    try:
        with gzip.open(feature_path, "wt", compresslevel=5) as output:
            for (company_id, origin), row in sorted(origins.items(), key=lambda item: (int(item[0][0]), item[0][1])):
                if company_id != current_id:
                    compressed, saved_hash = connection.execute("SELECT payload, sha256 FROM companies WHERE id=?", (company_id,)).fetchone()
                    unpacked = gzip.decompress(compressed)
                    digest = hashlib.sha256(unpacked).hexdigest()
                    if digest != saved_hash or digest != index["companies"][company_id]["sha256"]:
                        raise ValueError(f"Source company checksum mismatch: {company_id}")
                    company = json.loads(unpacked)
                    current_id = company_id
                if row["company_source_sha256"] != saved_hash:
                    raise ValueError(f"Forecast company checksum mismatch: {company_id}")
                training = decode_training(company, index, origin, expected_years=row["train_years"],
                                           expected_source_ids=row["train_source_ids"], expected_cash=row["train_cash"],
                                           expected_currency=row["currency"])
                features = extract_features(training)
                result = {key: value for key, value in row.items() if key != "signature"}
                result.update({"train_end_dates": [annual["end"] for annual in training],
                               "train_publications": [annual["publication"] for annual in training],
                               "train_currency_ratios": [annual["currency_ratio"] for annual in training],
                               "train_source_as_of": [annual["source_as_of"] for annual in training],
                               "training_history_fingerprint": training_fingerprint(training, index["columns"]), **features})
                output.write(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
                listings.add(company_id)
                fingerprints.add(result["training_history_fingerprint"])
                sectors[row["sector"] or "missing"] += 1
                for feature, value in features.items():
                    if finite(value):
                        available[feature] += 1
                    elif isinstance(value, str):
                        categories.setdefault(feature, Counter())[value] += 1
    finally:
        connection.close()
    # Rehash source files after the read-only audit to make immutability reviewable.
    if file_hash(pack_path) != pack_hash or file_hash(forecast_path) != forecast_hash:
        raise ValueError("Source file changed during extraction")
    summary = {
        "protocolId": PROTOCOL_ID, "completedAt": datetime.now(UTC).isoformat(),
        "sourceForecastRows": source_rows, "slimForecastRows": slim_rows,
        "featureRows": count, "listings": len(listings), "distinctTrainingHistories": len(fingerprints),
        "numericAvailability": {key: {"available": available[key], "missing": count - available[key], "fraction": available[key] / count if count else None}
                                for key in FEATURE_DEFINITIONS if key != "cash_sign_regime"},
        "categories": categories, "sectorOriginCounts": sectors, "definitions": FEATURE_DEFINITIONS,
        "minimumPositiveEbitMargin": MINIMUM_EBIT_MARGIN,
        "aggregation": "Five valid observations required for five-year medians/dispersion; any missing or invalid value leaves the feature missing. Latest debt/assets requires only origin values. No imputations, fitted thresholds or winsorization.",
        "fingerprint": "SHA256 of the five newest-first native financial vectors, fiscal periods and currency. Listing IDs, source IDs and publication metadata excluded. Exact duplicates only; not a corporate-family map. Does not inspect future financial values.",
        "limitations": ["Retrospective merged source vintage may contain restatements.", "Sector/branch/name/ISIN are current saved taxonomy from the original forecast artifact; no historical classifications inferred.", "Cash and accounting proxies are not verified FCFF, PP&E or operating capital; financial companies need separate interpretation.", "Features have five observations, are noisy and may combine growth, acquisitions, cyclicality and accounting differences.", "Extraction is only for origins with at least one eligible existing five-year backtest outcome; it does not establish universe-wide feature availability."],
        "inputs": {"pack": {"path": str(pack_path), "sha256": pack_hash, "asOf": index["as_of"]},
                   "forecasts": {"path": str(forecast_path), "sha256": forecast_hash},
                   "taxonomySha256": index["taxonomy_sha256"]},
        "artifacts": [{"name": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)} for path in [feature_path, slim_path]],
        "scriptSha256": file_hash(Path(__file__)),
    }
    summary_path = output_dir / "feature-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"featureRows": count, "listings": len(listings), "slimForecastRows": slim_rows, "summary": str(summary_path)}), flush=True)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--forecasts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.pack, args.forecasts, args.output_dir)
