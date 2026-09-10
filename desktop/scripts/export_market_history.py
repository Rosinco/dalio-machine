"""Add all-directory market history to an immutable v1 financial pack (v2 output).

Read-only Börsdata source. Uses source-local shares and prices from the same
snapshot, explicit publication-close dates, and observed FX with no static fill.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from financial_model import COLUMNS, create_database, encode, store_company
from market_model import (
    ENTRY_DAYS,
    FX_DAYS,
    METHOD,
    PRICE_LIMIT,
    market_coverage,
    market_record,
    select_fx,
    select_quote,
)
from taxonomy_model import file_record


def read_company(conn, identifier, expected):
    blob, digest = conn.execute(
        "SELECT payload,sha256 FROM companies WHERE id=?", (identifier,)
    ).fetchone()
    raw = gzip.decompress(blob)
    if digest != expected or hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError("Base financial company checksum mismatch")
    return json.loads(raw)


def export(root: Path, base: Path, output: Path):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Output must be outside the Börsdata project")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "framework"))
    import pandas as pd
    from analysis.edpb import basis_ok_mask, detect_share_basis_breaks
    from core.data import read_validated
    from core.schemas import (
        SNAPSHOT_INSTRUMENTS_SCHEMA,
        SNAPSHOT_PRICES_SCHEMA,
        SNAPSHOT_REPORTS_SCHEMA,
    )

    with base.open("rb") as handle:
        identity = hashlib.file_digest(handle, "sha256").hexdigest()
    if identity != base.stem:
        raise ValueError("Base financial filename checksum mismatch")
    src = sqlite3.connect(f"file:{base.resolve()}?mode=ro", uri=True)
    index = json.loads(
        gzip.decompress(src.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0])
    )
    if index["version"] != 1 or index["format"] != "macro-atlas-financials":
        raise ValueError("Expected an original v1 financial base")
    targets = {}
    for identifier, coverage in index["companies"].items():
        company = read_company(src, identifier, coverage["sha256"])
        for r in company["annual"]:
            targets.setdefault(r[7], {}).setdefault(int(identifier), []).append(
                dict(
                    year=r[0],
                    start=r[2],
                    end=r[3],
                    published=r[4],
                    source_id=r[7],
                    profit=r[8 + COLUMNS.index("profit_to_equity_holders")],
                    equity=r[8 + COLUMNS.index("total_equity")],
                )
            )
    markets = {k: [] for k in index["companies"]}
    market_sources = []
    schema = SNAPSHOT_PRICES_SCHEMA.remove_columns(["date"])

    def prices(path, ids):
        frame = read_validated(
            path, schema, columns=["ins_id", "date", "close"], filters=[("ins_id", "in", ids)]
        )
        if not pd.api.types.is_datetime64_any_dtype(frame.date) or frame.date.isna().any():
            raise ValueError("Invalid raw price dates")
        if frame.duplicated(["ins_id", "date"]).any():
            raise ValueError("Duplicate daily prices")
        frame = frame[frame.close.gt(0) & frame.close.lt(PRICE_LIMIT)].copy()
        frame["day"] = frame.date.astype(str).str.slice(0, 10)
        return {int(i): sorted(zip(g.day, g.close, strict=True)) for i, g in frame.groupby("ins_id")}

    for source in index["sources"]:
        if source["frequency"] != "annual":
            continue
        sid, stamp = source["id"], source["as_of"]
        folder = (root / source["path"]).parent.parent
        ipath, ppath = (
            folder / "all_instruments/all_instruments.parquet",
            folder / "all_stockprices/all_stockprices.parquet",
        )
        print(f"Preparing {stamp} shares, basis checks and observed FX", flush=True)
        ins = read_validated(ipath, SNAPSHOT_INSTRUMENTS_SCHEMA)
        instrument = ins.set_index("ins_id").to_dict("index")
        reports = read_validated(root / source["path"], SNAPSHOT_REPORTS_SCHEMA)
        if reports.duplicated(["ins_id", "year"]).any():
            raise ValueError("Ambiguous annual shares")
        breaks = detect_share_basis_breaks(reports)
        breaks["ok"] = basis_ok_mask(breaks, 1)
        basis = breaks.set_index(["ins_id", "year"]).ok.to_dict()
        shares = reports.set_index(["ins_id", "year"]).number_of_shares.to_dict()
        fx_names = {
            r["name"]: str(i)
            for i, r in instrument.items()
            if r["instrument_type"] == 6 and isinstance(r["name"], str) and "/" in r["name"]
        }
        currency_set = set(ins.stock_price_currency.dropna()) - {"SEK"}
        paths = {}
        for ccy in currency_set:
            if f"{ccy}/SEK" in fx_names:
                paths[ccy] = ("direct", [fx_names[f"{ccy}/SEK"]])
            elif "USD/SEK" in fx_names and f"USD/{ccy}" in fx_names:
                paths[ccy] = ("usd_cross", [fx_names["USD/SEK"], fx_names[f"USD/{ccy}"]])
        fx_ids = sorted({int(i) for _, legs in paths.values() for i in legs})
        fx_quotes = prices(ppath, fx_ids)
        fx_series = {}
        for ccy, (method, legs) in paths.items():
            if method == "direct":
                series = [
                    (d, v, method, legs) for d, v in fx_quotes.get(int(legs[0]), []) if d <= stamp
                ]
            else:
                denominator = dict(fx_quotes.get(int(legs[1]), []))
                series = [
                    (d, v / denominator[d], method, legs)
                    for d, v in fx_quotes.get(int(legs[0]), [])
                    if d <= stamp and d in denominator
                ]
            fx_series[ccy] = series
        market_sources.append(
            dict(
                id=sid,
                as_of=stamp,
                instruments=file_record(root, ipath),
                prices=file_record(root, ppath),
                fx_pairs={str(i): instrument[i]["name"] for i in fx_ids},
            )
        )
        company_ids = sorted(targets.get(sid, {}))
        quotes = {}
        for offset in range(0, len(company_ids), 256):
            batch = company_ids[offset : offset + 256]
            quotes = prices(ppath, batch)
            for identifier in batch:
                info = instrument.get(identifier, {})
                ccy = info.get("stock_price_currency")
                for a in targets[sid][identifier]:
                    a = {**a, "as_of": stamp}
                    quote = select_quote(quotes.get(identifier, []), a["published"], stamp)
                    fx = None
                    if quote:
                        fx = (
                            (quote[0], 1.0, "identity", [])
                            if ccy == "SEK"
                            else select_fx(fx_series.get(ccy, []), quote[0])
                        )
                    m = market_record(
                        a,
                        shares=shares.get((identifier, a["year"])),
                        currency=ccy,
                        quote=quote,
                        fx=fx,
                        basis_ok=bool(basis.get((identifier, a["year"]), False)),
                        receipt=info.get("instrument_type") in [1, 9, 10],
                    )
                    markets[str(identifier)].append(m)
            if offset % 2048 == 0:
                print(
                    f"  {stamp}: {min(offset + 256, len(company_ids))}/{len(company_ids)} listings",
                    flush=True,
                )
        del reports, instrument, ins, shares, quotes

    output.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="market-export-", dir=output))
    path = staging / "pack.sqlite"
    conn = create_database(path)
    coverage = {}
    totals = dict(
        count=0, local=0, sek=0, flagged=0, listings=len(markets), with_local=0, with_sek=0
    )
    reasons = {}
    try:
        for identifier, original in index["companies"].items():
            company = read_company(src, identifier, original["sha256"])
            rows = sorted(markets[identifier], key=lambda r: r["year"])
            if [r["year"] for r in rows] != [r[0] for r in company["annual"]]:
                raise ValueError("Market rows must cover every annual financial observation")
            company["market"] = rows
            c = market_coverage(rows)
            coverage[identifier] = {**store_company(conn, company), "market": c}
            for k in ["count", "local", "sek", "flagged"]:
                totals[k] += c[k]
            totals["with_local"] += c["local"] > 0
            totals["with_sek"] += c["sek"] > 0
            for r in rows:
                for flag in r["flags"]:
                    reasons[flag] = reasons.get(flag, 0) + 1
        result = {
            **index,
            "version": 2,
            "generated_at": datetime.now(UTC).isoformat(),
            "companies": coverage,
            "market": dict(
                version=1,
                method=METHOD,
                base_pack=identity,
                entry_days=ENTRY_DAYS,
                fx_days=FX_DAYS,
                sources=market_sources,
                summary=totals,
                basis_code=file_record(root, root / "framework/analysis/edpb.py"),
            ),
        }
        payload = encode(result)
        if len(payload) > 16_000_000:
            raise ValueError("Financial index exceeds supported metadata size")
        conn.execute("INSERT INTO metadata VALUES ('index',?)", (gzip.compress(payload, mtime=0),))
        conn.commit()
        if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity check failed")
    finally:
        conn.close()
        src.close()
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    dest = output / f"{digest}.sqlite"
    if dest.exists():
        raise ValueError("Market financial pack already exists")
    os.replace(path, dest)
    staging.rmdir()
    record = dict(
        id=digest,
        bytes=dest.stat().st_size,
        taxonomy_sha256=result["taxonomy_sha256"],
        as_of=result["as_of"],
    )
    (output / "catalog.json").write_bytes(encode({"version": 1, "packs": [record]}))
    (output / "market-coverage-audit.json").write_bytes(
        encode(dict(pack=record, summary=totals, reasons=reasons, sources=market_sources))
    )
    print(json.dumps(dict(pack=record, summary=totals, reasons=reasons)), flush=True)
    return record


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--borsdata-root", type=Path, required=True)
    p.add_argument("--financial-pack", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    export(a.borsdata_root, a.financial_pack, a.output)
