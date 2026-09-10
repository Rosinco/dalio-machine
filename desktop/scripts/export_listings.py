"""Collect all saved company identities without loading prices, KPIs or financial reports."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from listing_model import COMPANY_TYPES, COUNTRIES, listing_record, select_listings
from taxonomy_model import file_record


def export(root: Path, output: Path):
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("The catalogue must be written outside the Börsdata project")
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "framework"))
    from core.data import read_validated
    from core.schemas import COUNTRIES_SCHEMA, INSTRUMENTS_SCHEMA

    folders = [("2025-06-21", root / "data/raw_api")]
    folders += [
        (p.name, p)
        for p in (root / "data/raw_api_snapshots").iterdir()
        if p.is_dir() and re.fullmatch(r"20\d\d-\d\d-\d\d", p.name)
    ]
    folders.sort()
    snapshots, sources, countries = [], [], {}
    for stamp, folder in folders:
        instrument_path = folder / "all_instruments/all_instruments.parquet"
        country_path = folder / "all_instruments/countries.parquet"
        instruments = read_validated(
            instrument_path,
            INSTRUMENTS_SCHEMA,
            columns=[
                "ins_id",
                "name",
                "ticker",
                "isin",
                "instrument_type",
                "sector_id",
                "branch_id",
                "country_id",
                "listing_date",
                "stock_price_currency",
                "report_currency",
            ],
        )
        country_frame = read_validated(country_path, COUNTRIES_SCHEMA)
        for c in country_frame.to_dict("records"):
            code, name = COUNTRIES.get(c["name"], (None, c["name"]))
            countries[str(c["id"])] = {
                "id": str(c["id"]),
                "name": c["name"],
                "name_en": name,
                "iso2": code,
            }
        snapshots.append(
            (stamp, instruments.astype(object).where(instruments.notna(), None).to_dict("records"))
        )
        included = int(instruments.instrument_type.isin(COMPANY_TYPES).sum())
        sources.append(
            {
                "as_of": stamp,
                "instruments": file_record(root, instrument_path),
                "countries": file_record(root, country_path),
                "instrument_count": len(instruments),
                "company_count": included,
                "excluded_count": len(instruments) - included,
            }
        )
    selected = select_listings(snapshots)
    rows = {key: listing_record(row, countries) for key, row in selected.items()}
    latest = folders[-1][0]
    raw = {
        "version": 1,
        "as_of": latest,
        "exported_at": datetime.now(UTC).isoformat(),
        "countries": countries,
        "snapshots": sources,
        "included_types": list(COMPANY_TYPES),
        "listings": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(raw, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return {
        "listings": len(rows),
        "latest": sum(r["source_as_of"] == latest for r in rows.values()),
        "older_only": sum(r["source_as_of"] != latest for r in rows.values()),
        "countries": len({r["listing_country"] for r in rows.values()} - {None}),
        "unmapped_countries": sum(r["listing_country"] is None for r in rows.values()),
        "bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borsdata-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.borsdata_root, args.output)))
