"""Compact, source-bound financial records and indexed offline storage."""

from __future__ import annotations

import gzip
import hashlib
import json
import sqlite3
from pathlib import Path

from business_model import AMOUNTS, numeric, report

COLUMNS = (
    *AMOUNTS,
    "gross_income",
    "profit_before_tax",
    "net_sales",
    "total_liabilities_and_equity",
    "cash_flow_from_investing_activities",
    "cash_flow_from_financing_activities",
    "cash_flow_for_the_year",
)


def encode(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def pack_report(row, as_of, source_id, *, annual):
    try:
        normalized = report(row, as_of)
        if (normalized["period"] == 5) != annual:
            raise ValueError("Unexpected annual/quarterly period")
        packed = [
            normalized[k]
            for k in ("year", "period", "start", "end", "report_date", "currency", "currency_ratio")
        ]
        return [*packed, source_id, *(numeric(row.get(k)) for k in COLUMNS)], None
    except (ValueError, TypeError, OverflowError) as error:
        return None, {
            "year": int(row["year"]),
            "period": int(row["period"]),
            "source_id": source_id,
            "start": str(row.get("report_start_date")),
            "end": str(row.get("report_end_date")),
            "published": str(row.get("report_date")),
            "reason": str(error)[:200],
        }


def period_coverage(rows, *, annual):
    if not rows:
        return {
            "count": 0,
            "first": None,
            "last": None,
            "last_period": None,
            "end": None,
            "published": None,
            "gaps": 0,
            "unavailable": 0,
        }

    def position(r):
        return r[0] if annual else r[0] * 4 + r[1] - 1

    return {
        "count": len(rows),
        "first": rows[0][0],
        "last": rows[-1][0],
        "last_period": rows[-1][1],
        "end": rows[-1][3],
        "published": rows[-1][4],
        "gaps": position(rows[-1]) - position(rows[0]) + 1 - len(rows),
        "unavailable": sum(
            r[6] is None or r[6] <= 0 or all(v is None for v in r[8:]) for r in rows
        ),
    }


def create_database(path: Path):
    if path.exists():
        raise ValueError("Financial output already exists")
    conn = sqlite3.connect(path)
    conn.executescript("""
        PRAGMA page_size=4096;
        CREATE TABLE metadata (key TEXT PRIMARY KEY, payload BLOB NOT NULL);
        CREATE TABLE companies (id TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL);
    """)
    return conn


def store_company(conn, payload):
    data = encode(payload)
    if len(data) > 2_000_000:
        raise ValueError("A company history exceeds the supported size")
    digest = hashlib.sha256(data).hexdigest()
    conn.execute(
        "INSERT INTO companies VALUES (?, ?, ?)",
        (payload["id"], gzip.compress(data, mtime=0), digest),
    )
    return {
        "annual": period_coverage(payload["annual"], annual=True),
        "quarterly": period_coverage(payload["quarterly"], annual=False),
        "withheld": len(payload["withheld"]),
        "currencies": sorted({r[5] for r in payload["annual"] + payload["quarterly"]}),
        "sha256": digest,
    }
