"""Read-only cash-component availability and arithmetic audit; no forecast changes."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

FIELDS = {
    "operating": "cash_flow_from_operating_activities",
    "investing": "cash_flow_from_investing_activities",
    "financing": "cash_flow_from_financing_activities",
    "netCash": "cash_flow_for_the_year",
    "providerFcf": "free_cash_flow",
}
ABS_TOL, REL_TOL = 1e-6, 1e-8
MISSING_COMPONENTS = [
    "gross_capex", "maintenance_capex", "growth_capex", "acquisition_cash",
    "disposal_proceeds", "lease_principal", "lease_interest", "cash_interest",
    "cash_taxes", "working_capital_cash_change", "restricted_cash",
    "regulatory_capital_cash_requirement",
]
DEFINITIONS = {
    "operatingPlusInvesting": "Operating cash + signed investing cash; broad cash subtotal, not owner cash or a capex-only FCF measure.",
    "providerFcfMinusOperatingAndInvesting": "Provider FCF minus (operating cash + signed investing cash); unexplained arithmetic difference, not an identified adjustment.",
    "operatingMinusProviderFcf": "Operating cash minus provider FCF; a cash-definition gap, not observed capex.",
    "componentSum": "Operating + investing + financing cash, using signed same-report amounts.",
    "netCashMinusComponentSum": "Net cash for the period minus the three-component sum; not attributed automatically to FX, reclassifications or a data error.",
}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def comparison(a, b):
    if not finite(a) or not finite(b):
        return "unavailable"
    return "matches" if abs(a - b) <= ABS_TOL + REL_TOL * max(abs(a), abs(b)) else "differs"


def placeholder(amounts):
    return any(x == 0 for x in amounts.values()) and all(x is None or x == 0 for x in amounts.values())


def component_evidence(amounts, ratio, currency):
    raw = {label: amounts.get(field) if finite(amounts.get(field)) else None for label, field in FIELDS.items()}
    converted = finite(ratio) and ratio > 0 and isinstance(currency, str) and re.fullmatch(r"[A-Z]{3}", currency)
    native = {key: value / ratio if converted and value is not None and finite(value / ratio) else None for key, value in raw.items()}
    cfo, cfi, cff, net, fcf = (native[key] for key in FIELDS)

    def add(*values):
        return sum(values) if all(finite(x) for x in values) and finite(sum(values)) else None

    def subtract(a, b):
        return a - b if finite(a) and finite(b) and finite(a - b) else None

    op_inv, total = add(cfo, cfi), add(cfo, cfi, cff)
    return {
        "raw": raw, "native": native, "nativeConversionAvailable": bool(converted),
        "derived": {
            "operatingPlusInvesting": op_inv,
            "providerFcfMinusOperatingAndInvesting": subtract(fcf, op_inv),
            "operatingMinusProviderFcf": subtract(cfo, fcf),
            "componentSum": total,
            "netCashMinusComponentSum": subtract(net, total),
        },
        "fcfComparison": comparison(fcf, op_inv), "netCashComparison": comparison(net, total),
        "definitionVerified": False,
    }


def compare_vintages(old, new, old_currency, new_currency):
    if old_currency != new_currency:
        return {"status": "currency_changed", "cause": "unconfirmed"}
    states = {key: comparison(old["native"][key], new["native"][key]) for key in FIELDS}
    if states["providerFcf"] == "unavailable":
        return {"status": "fcf_unavailable", "cause": "unconfirmed"}
    a, b = old["native"]["providerFcf"], new["native"]["providerFcf"]
    return {
        "status": "comparable", "changedFields": [key for key, state in states.items() if state == "differs"],
        "unavailableFields": [key for key, state in states.items() if state == "unavailable"],
        "fcfChanged": states["providerFcf"] == "differs",
        "fcfSignChanged": (a > 0) - (a < 0) != (b > 0) - (b < 0),
        "operatingAndInvestingUnchanged": states["operating"] == states["investing"] == "matches",
        "otherFourCashFieldsUnchanged": all(states[key] == "matches" for key in FIELDS if key != "providerFcf"),
        "cause": "unconfirmed",
    }


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as source:
        for part in iter(lambda: source.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def identity(path, root):
    return {"path": str(path.relative_to(root)), "sha256": digest(path), "bytes": path.stat().st_size}


def empty_stats():
    return {"rows": 0, "placeholderRows": 0, "missingPublicationDate": 0,
            "invalidNativeConversion": 0, "rawAvailable": Counter(), "nativeAvailable": Counter(),
            "derivedAvailable": Counter(), "fcfComparison": Counter(), "netCashComparison": Counter(),
            "nonplaceholderFcfComparison": Counter(), "nonplaceholderNetCashComparison": Counter()}


def count_row(stats, row):
    evidence = row["cash"]
    stats["rows"] += 1
    stats["placeholderRows"] += row["placeholder"]
    stats["missingPublicationDate"] += row["published"] is None
    stats["invalidNativeConversion"] += not evidence["nativeConversionAvailable"]
    for group in ["raw", "native", "derived"]:
        for key, value in evidence[group].items():
            stats[f"{group}Available"][key] += finite(value)
    for field in ["fcfComparison", "netCashComparison"]:
        stats[field][evidence[field]] += 1
        if not row["placeholder"]:
            stats[f"nonplaceholder{field[0].upper()}{field[1:]}"][evidence[field]] += 1


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with path.open("wb") as stream, gzip.GzipFile(fileobj=stream, mode="wb", mtime=0) as compressed:
        for row in rows:
            compressed.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode())


def audit_pack(pack, taxonomy, output):
    with sqlite3.connect(f"file:{pack}?mode=ro", uri=True) as connection:
        connection.execute("PRAGMA query_only=ON")
        index = json.loads(gzip.decompress(connection.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0]))
        listings = taxonomy["catalogue"]["listings"]
        if set(index["companies"]) != set(listings):
            raise ValueError("Taxonomy and financial-pack listing cohorts differ")
        sources = {item["id"]: item for item in index["sources"]}
        stats = {"allAnnualRows": empty_stats(), "latestAnnualPerListing": empty_stats(),
                 "latestDirectoryMembers": empty_stats(), "olderDirectoryMembers": empty_stats(),
                 "latestOperatingAndProperty": empty_stats(), "latestManualFinancials": empty_stats(),
                 "latestUnclassified": empty_stats()}
        by_source, latest_rows = {}, []
        membership = Counter()
        for ins_id, payload, expected_hash in connection.execute("SELECT id,payload,sha256 FROM companies ORDER BY CAST(id AS INTEGER)"):
            raw_bytes = gzip.decompress(payload)
            if hashlib.sha256(raw_bytes).hexdigest() != expected_hash or expected_hash != index["companies"][ins_id]["sha256"]:
                raise ValueError(f"Company payload identity mismatch: {ins_id}")
            company = json.loads(raw_bytes)
            listing, classification = listings[ins_id], taxonomy["classifications"][ins_id]
            member = listing["source_as_of"] == taxonomy["catalogue"]["as_of"]
            membership["latestDirectory" if member else "olderDirectory"] += 1
            rows = []
            for packed in company["annual"]:
                year, period, start, end, published, currency, ratio, source_id, *amounts = packed
                if period != 5 or len(amounts) != len(index["columns"]) or source_id not in sources:
                    raise ValueError(f"Unexpected packed annual row for {ins_id}")
                raw = dict(zip(index["columns"], amounts, strict=True))
                row = {"year": year, "period": period, "start": start, "end": end, "published": published,
                       "currency": currency, "currencyRatio": ratio, "sourceId": source_id,
                       "sourceAsOf": sources[source_id]["as_of"], "sourceSha256": sources[source_id]["sha256"],
                       "sourcePath": sources[source_id]["path"], "placeholder": placeholder(raw),
                       "cash": component_evidence(raw, ratio, currency)}
                count_row(stats["allAnnualRows"], row)
                count_row(by_source.setdefault(source_id, empty_stats()), row)
                rows.append(row)
            latest = max(rows, key=lambda x: (x["end"], x["year"]), default=None)
            manual = classification["sector_id"] == "1" and classification["branch_id"] not in ["75", "76"]
            if latest:
                count_row(stats["latestAnnualPerListing"], latest)
                count_row(stats["latestDirectoryMembers" if member else "olderDirectoryMembers"], latest)
                group = "latestManualFinancials" if manual else "latestOperatingAndProperty" if classification["sector_id"] in {str(i) for i in range(1, 11)} else "latestUnclassified"
                count_row(stats[group], latest)
            else:
                membership["noAnnualRows"] += 1
            latest_rows.append({"id": ins_id, "name": listing["name"], "isin": listing["isin"],
                                "sourceListingAsOf": listing["source_as_of"], "latestDirectoryMember": member,
                                "sectorId": classification["sector_id"], "branchId": classification["branch_id"],
                                "manualFinancials": manual, "companyPayloadSha256": expected_hash,
                                "annualRows": len(rows), "withheldRows": len(company["withheld"]), "latest": latest})
        if stats["allAnnualRows"]["rows"] != index["summary"]["annual"]:
            raise ValueError("Annual totals do not reconcile")
        write_jsonl(output / "latest-company-cash-components.jsonl.gz", latest_rows)
        return index, {"cohort": {"listings": len(listings), **membership}, "summaries": stats,
                       "byAnnualSource": by_source, "selectedExamples": [row for row in latest_rows if row["id"] in {"3", "20", "102", "197", "696"}]}


def audit_vintages(root, index, listings, output):
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq

    source_files = sorted([row for row in index["sources"] if row["frequency"] == "annual"], key=lambda x: x["as_of"])
    if len(source_files) != 2:
        raise ValueError("This dated audit requires exactly the two retained annual vintages")
    versions, schemas, summaries, input_files, vector_checks = [], {}, {}, [], {}
    for source in source_files:
        path = root / source["path"]
        record = identity(path, root)
        if record["sha256"] != source["sha256"]:
            raise ValueError(f"Annual vintage hash mismatch: {source['path']}")
        input_files.append(record)
        schemas[source["id"]] = pq.read_schema(path).names
        columns = ["ins_id", "year", "period", "currency", "currency_ratio", "report_date", *FIELDS.values()]
        frame = pd.read_parquet(path, columns=columns)
        if frame.duplicated(["ins_id", "year", "period"]).any():
            raise ValueError("Duplicate fiscal key in original annual source")
        version, stats = {}, empty_stats()
        for values in frame.itertuples(index=False, name=None):
            ins, year, period, currency, ratio, published, *amounts = values
            cash = component_evidence(dict(zip(FIELDS.values(), amounts, strict=True)), ratio, currency)
            key = (int(ins), int(year), int(period))
            version[key] = {"currency": currency, "cash": cash}
            # A five-field all-zero pattern cannot establish a whole-statement placeholder.
            count_row(stats, {"cash": cash, "placeholder": False, "published": None if pd.isna(published) else str(published)})
        versions.append(version)
        summaries[source["id"]] = stats
        # A separate vector calculation operates directly on the original parquet,
        # without component_evidence/comparison or their row dictionaries.
        ratio = frame.currency_ratio
        a = frame.free_cash_flow / ratio
        b = (frame.cash_flow_from_operating_activities + frame.cash_flow_from_investing_activities) / ratio
        currency_valid = frame.currency.astype(str).str.fullmatch("[A-Z]{3}")
        complete = np.isfinite(a) & np.isfinite(b) & np.isfinite(ratio) & (ratio > 0)
        valid = complete & currency_valid
        matched = valid & np.isclose(a, b, rtol=REL_TOL, atol=ABS_TOL)
        if int(matched.sum()) != stats["fcfComparison"]["matches"] or int(valid.sum()) != stats["fcfComparison"]["matches"] + stats["fcfComparison"]["differs"]:
            raise ValueError("Independent vector component reconciliation differs")
        vector_checks[source["id"]] = {
            "completeRowsWithNativeCurrency": int(valid.sum()), "matches": int(matched.sum()),
            "completeRowsBeforeCurrencyLabelCheck": int(complete.sum()),
            "excludedInvalidCurrencyLabels": frame.loc[complete & ~currency_valid, ["ins_id", "year", "currency"]].to_dict("records"),
            "method": "Independent pandas/NumPy calculation directly from parquet: FCF/ratio versus (CFO+CFI)/ratio, np.isclose rtol1e-8 atol1e-6.",
        }
    old, new = versions
    counter, current_counter, examples, comparison_rows = Counter(), Counter(), [], []
    for key in sorted(old.keys() & new.keys()):
        a, b = old[key], new[key]
        result = compare_vintages(a["cash"], b["cash"], a["currency"], b["currency"])
        for target in [counter, *([current_counter] if str(key[0]) in listings else [])]:
            target["overlappingFiscalKeys"] += 1
            target[result["status"]] += 1
            if result["status"] == "comparable":
                target["fcfChanged"] += result["fcfChanged"]
                target["fcfSignChanged"] += result["fcfSignChanged"]
                target["fcfChangedWithOperatingAndInvestingUnchanged"] += result["fcfChanged"] and result["operatingAndInvestingUnchanged"]
                target["fcfChangedWithOtherFourCashFieldsUnchanged"] += result["fcfChanged"] and result["otherFourCashFieldsUnchanged"]
        item = {"id": str(key[0]), "year": key[1], "period": key[2],
                "inCurrentUnion": str(key[0]) in listings, "oldCurrency": a["currency"], "newCurrency": b["currency"],
                "oldNative": a["cash"]["native"], "newNative": b["cash"]["native"], **result}
        comparison_rows.append(item)
        if result.get("fcfChanged") and result.get("otherFourCashFieldsUnchanged") and key[1] == 2024 and len(examples) < 12:
            examples.append(item)
    write_jsonl(output / "cross-vintage-cash-comparisons.jsonl.gz", comparison_rows)
    return {"inputFiles": input_files, "originalReportColumns": schemas, "perVintageAllRawRows": summaries,
            "independentVectorReconciliation": vector_checks,
            "allOverlappingKeys": dict(counter), "overlappingKeysInCurrentUnion": dict(current_counter),
            "unexplainedFY2024ExamplesByAscendingListingId": examples,
            "cause": "Unconfirmed. Changed provider FCF with unchanged aggregate components cannot be reconciled from the retained cash fields alone.",
            "placeholderPolicy": "Whole-statement placeholder flags are only computed for the retained pack, using every amount column. Raw-vintage component counts include zeros and do not claim placeholder exclusion."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borsdata-root", type=Path, default=os.getenv("MACRO_ATLAS_BORSDATA_ROOT"))
    parser.add_argument("--desktop-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.borsdata_root:
        parser.error("Supply --borsdata-root or MACRO_ATLAS_BORSDATA_ROOT")
    desktop, root = args.desktop_root.resolve(), args.borsdata_root.resolve()
    output = (args.output or desktop / "test-results/cash-component-audit-2026-09-13").resolve()
    if output.is_relative_to(root):
        raise ValueError("Output must be outside the read-only source repository")
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = desktop / "src/data/cash-uncertainty-2026-09-12.json"
    calibration = json.loads(calibration_path.read_bytes())
    pack = desktop / calibration["provenance"]["sourcePack"]["path"]
    taxonomy_path = desktop / "public/data/taxonomy.json"
    taxonomy = json.loads(taxonomy_path.read_bytes())
    inputs = [identity(path, desktop) for path in [pack, taxonomy_path, calibration_path]]
    if inputs[0]["sha256"] != calibration["provenance"]["sourcePack"]["sha256"] or inputs[1]["sha256"] != calibration["taxonomySha256"]:
        raise ValueError("Pinned calibration pack/taxonomy identities differ")
    docs = [root / "DATA.md", root / "src/borsdata_client/models.py", root / "framework/data_prep/snapshot_fetch.py",
            root / "reference/api_wiki_2026-08-10/swagger_v1.json", root / "decisions/0016-fcf-earnings-divergence-guardrail.md"]
    documentation = [identity(path, root) for path in docs]
    index, current = audit_pack(pack, taxonomy, output)
    vintages = audit_vintages(root, index, taxonomy["catalogue"]["listings"], output)
    if inputs != [identity(path, desktop) for path in [pack, taxonomy_path, calibration_path]]:
        raise ValueError("Atlas input changed during read-only audit")
    if documentation != [identity(root / item["path"], root) for item in documentation]:
        raise ValueError("Source documentation changed during read-only audit")
    for item in vintages["inputFiles"]:
        if item != identity(root / item["path"], root):
            raise ValueError("Annual source changed during read-only audit")
    result = {
        "id": "cash-component-audit-2026-09-13-v1", "status": "passed", "sourceAsOf": index["as_of"],
        "protocol": {"nativeCurrency": "Every amount divided by its own row's positive saved currency_ratio; no cross-currency totals.",
                     "unit": "native reporting currency millions", "latestPolicy": "Latest retained annual by period end then fiscal year, with no older replacement for missing components. This is source availability, not starter eligibility; stale/short/withheld contexts remain in the ledger.",
                     "cohort": "All 19,140 listings in the two-directory union, with latest-directory membership shown separately. Not distinct issuers.",
                     "absTolerance": ABS_TOL, "relTolerance": REL_TOL,
                     "comparison": "abs(a-b) <= absTolerance + relTolerance * max(abs(a),abs(b)); a diagnostic equality, never accounting-definition validation.",
                     "reportedFields": FIELDS, "derivedDefinitions": DEFINITIONS,
                     "absentSeparateReportFields": MISSING_COMPONENTS,
                     "sourceRowsEdited": False, "calibrationRefitted": False, "ownerCashDerived": False,
                     "dataDefinitionChangeCause": "unconfirmed"},
        "current": current, "vintages": vintages, "inputs": inputs, "sourceDocumentation": documentation,
        "earlierDocumentationReconciliation": {
            "document": "docs/cash-flow-backtest-data-vintages-2026-09-12.md",
            "earlierMatchingCounts": {"2025-06-21": 57598, "2026-08-10": 48754},
            "earlierCountsReproduced": False,
            "denominatorDifferenceExplained": "Frozen raw complete-row count205587 includes listing87 FY2024 currency='0'. Requiring a valid native currency gives205586. Fresh remains250862.",
            "conclusion": "The earlier matching counts are not supported by its retained generator/receipt or the independent strict native arithmetic. Use the reproducible counts in independentVectorReconciliation. This corrects audit documentation; it does not establish a cause for provider-vintage FCF changes.",
        },
        "primaryReferences": [
            {"url": "https://borsdata.se/en/info/ratios/fcf-per-share", "accessed": "2026-09-13", "finding": "Broad operating/investing description; no exact versioned API field reconciliation."},
            {"url": "https://borsdata.se/en/info/ratios/capex", "accessed": "2026-09-13", "finding": "Provider describes its investment concept as including acquisitions and divestments."},
            {"url": "https://github.com/Borsdata-Sweden/API/wiki/Reports", "accessed": "2026-09-13", "finding": "API report currency and aggregate fields; no verified explanation of the observed FCF changes."},
            {"url": "https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/", "accessed": "2026-09-13", "finding": "Investing cash includes long-term asset purchases/disposals and acquisitions/loss of business control; financing includes equity and borrowings."},
        ],
    }
    write_json(output / "cash-component-audit.json", result)
    receipt = {"status": "passed", "createdAt": datetime.now(UTC).isoformat(), "script": identity(Path(__file__), desktop),
               "inputs": inputs, "sourceDocumentation": documentation, "rawAnnualSources": vintages["inputFiles"],
               "outputs": [identity(output / name, desktop) for name in ["cash-component-audit.json", "latest-company-cash-components.jsonl.gz", "cross-vintage-cash-comparisons.jsonl.gz"]]}
    write_json(output / "receipt.json", receipt)
    print(json.dumps({"status": "passed", "cohort": current["cohort"], "latest": current["summaries"]["latestAnnualPerListing"], "vintages": vintages["allOverlappingKeys"]}, indent=2))


if __name__ == "__main__":
    main()
