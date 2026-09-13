#!/usr/bin/env python3
"""Independently inspect a completed universe audit without importing app code."""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

DESKTOP = Path(__file__).resolve().parents[1]
OUTPUT = DESKTOP / "test-results/empirical-cash-starter-2026-09-13"


def identity(path: Path) -> dict:
    return {
        "path": str(path.relative_to(DESKTOP)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "bytes": path.stat().st_size,
    }


def close(actual: float, expected: float) -> None:
    assert math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-8), (
        actual,
        expected,
    )


def main() -> None:
    receipt_path = OUTPUT / "universe-runtime-audit.json"
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "passed"
    bound_files = receipt["inputs"] + receipt["outputs"] + [receipt["script"]]
    for bound in bound_files:
        assert identity(DESKTOP / bound["path"]) == bound, bound["path"]

    calibration_path = DESKTOP / "src/data/cash-uncertainty-2026-09-12.json"
    calibration = json.loads(calibration_path.read_text())
    assert identity(calibration_path)["sha256"] == (
        "9d4664c6f433a9fa9dc7bf8256686522db1660d70d51e875e59eb85fb1690ff2"
    )
    assert calibration["covidTreatment"]["omittedTargetYears"] == []
    assert calibration["covidTreatment"]["historicalShocksRemoved"] is False
    with gzip.open(OUTPUT / "universe-starters.jsonl.gz", "rt") as stream:
        ledger = [json.loads(line) for line in stream]
    by_id = {row["id"]: row for row in ledger}
    assert len(ledger) == len(by_id) == receipt["counts"]["listings"] == 19140
    statuses = Counter(row["status"] for row in ledger)
    groups = Counter(
        row["group"] for row in ledger if row["status"] == "historical"
    )
    assert dict(groups) == receipt["groups"]
    reconciled = {
        **statuses,
        "manualFinancials": sum(row["manual"] for row in ledger),
        "valuationReady": sum(row["ready"] for row in ledger),
        "quoteBasis": sum(row["amountBasis"] == "quote" for row in ledger),
        "negativeMid": sum(
            row["mid"] is not None and row["mid"] < 0 for row in ledger
        ),
        "zeroMid": sum(row["mid"] == 0 for row in ledger),
        "zeroMidHistorical": sum(
            row["mid"] == 0 and row["status"] == "historical" for row in ledger
        ),
        "historyShort": sum(0 < len(row["annualYears"]) < 5 for row in ledger),
        "noEligibleAnnualCash": sum(not row["annualYears"] for row in ledger),
        "missingMarketValue": sum(row["marketCap"] is None for row in ledger),
    }
    for key, count in reconciled.items():
        assert count == receipt["counts"][key], key
    for row in ledger:
        for forecast in row["forecast"]:
            if row["manual"]:
                assert all(forecast[key] is None for key in ("low", "mid", "high"))
            elif forecast["low"] is not None:
                assert forecast["low"] <= forecast["mid"] <= forecast["high"]
            if row["mid"] == 0 and row["status"] == "historical":
                assert forecast["low"] < 0 < forecast["high"]

    pack = DESKTOP / calibration["provenance"]["sourcePack"]["path"]
    db = sqlite3.connect(f"file:{pack}?mode=ro", uri=True)
    index = json.loads(
        gzip.decompress(
            db.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0]
        )
    )
    cash_column = 8 + index["columns"].index("free_cash_flow")
    sources = {source["id"]: source for source in index["sources"]}
    anchors = []
    expected_status = {
        "696": "historical",
        "20": "historical",
        "3": "percentage",
        "159": "unavailable",
        "14473": "unavailable",
        "167": "unavailable",
    }
    for listing, status in expected_status.items():
        payload = db.execute(
            "SELECT payload FROM companies WHERE id=?", (listing,)
        ).fetchone()[0]
        raw = gzip.decompress(payload)
        saved = json.loads(raw)
        observed = by_id[listing]
        assert hashlib.sha256(raw).hexdigest() == observed["companyPayloadSha256"]
        # These fixed examples have contiguous, same-currency annuals. Select them
        # directly from packed rows, independently of both runtime and decoder.
        history = sorted(
            (
                row
                for row in saved["annual"]
                if row[3] <= receipt["asOf"]
                and sources[row[7]]["as_of"] <= receipt["asOf"]
                and (row[4] is None or row[4] <= receipt["asOf"])
            ),
            key=lambda row: (row[3], row[0]),
            reverse=True,
        )[:5]
        native = [row[cash_column] / row[6] for row in history]
        assert observed["annualYears"] == [row[0] for row in history]
        assert observed["status"] == status
        for actual, expected in zip(observed["signedNativeCash"], native, strict=True):
            close(actual, expected)
        anchor = {"id": listing, "nativeCash": native, "status": status}
        if native:
            scale = sum(abs(value) for value in native) / 5
            mean = sum(native) / 5
            dispersion = math.sqrt(sum((value - mean) ** 2 for value in native) / 5) / scale
            group = "low" if dispersion < 0.25 else "medium" if dispersion < 0.75 else "high"
            close(observed["scale"], scale)
            close(observed["dispersion"], dispersion)
            assert observed["group"] == group
            anchor.update(scale=scale, dispersion=dispersion, group=group)
            if status == "historical":
                close(observed["mid"], native[0])
                for forecast in observed["forecast"]:
                    year = forecast["year"]
                    factor = calibration["calibrationByModelHorizon"]["naive"][
                        str(min(year, 4))
                    ]["cashDispersion"][group]["factor"]
                    half = scale * (factor + max(0, year - 4) / 10)
                    close(forecast["halfWidth"], half)
                    close(forecast["low"], native[0] - half)
                    close(forecast["high"], native[0] + half)
                anchor["forecast"] = observed["forecast"]
        anchors.append(anchor)
    db.close()
    result = {
        "status": "passed",
        "createdAt": datetime.now(UTC).isoformat(),
        "scope": "Independent receipt-chain, ledger-count and six raw packed-source anchor review; no production decoder or valuation helper imported.",
        "receipt": identity(receipt_path),
        "verifier": identity(Path(__file__).resolve()),
        "boundFilesVerified": len(bound_files),
        "ledgerRows": len(ledger),
        "reconciledCounts": reconciled,
        "historicalGroups": dict(groups),
        "anchors": anchors,
        "limits": [
            "The parent runtime audit verifies all annual generation and ready DCF/NPV arithmetic. This independent review samples six raw-source anchors and reconciles the complete output ledger.",
            "Implementation correctness does not provide new evidence of forecast accuracy or annual coverage.",
        ],
    }
    target = OUTPUT / "universe-independent-review.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": "passed", "boundFiles": len(bound_files), "listings": len(ledger), "anchors": len(anchors), "output": str(target)}))


if __name__ == "__main__":
    main()
