"""Select five forestry companies from saved Börsdata files; no API or source writes.

Run using Börsdata's locked .venv-wsl environment; output belongs to Atlas.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

from business_model import AMOUNTS, reports

COHORT = {
    102: (
        "Holmen",
        "SE",
        ["Forest", "Wood products", "Paperboard", "Paper", "Energy"],
        "Reference company; segment mix is based on the archived study.",
    ),
    197: (
        "SCA",
        "SE",
        ["Forest", "Wood products", "Pulp", "Containerboard"],
        "Overlap in forest, wood and fibre products. History before the 2017 Essity separation represents a different group.",
    ),
    34: (
        "Billerud",
        "SE",
        ["Paperboard", "Paper"],
        "Paper and board overlap; not an equivalent forest-land ownership comparison.",
    ),
    696: (
        "Stora Enso",
        "FI",
        ["Forest", "Wood products", "Packaging materials"],
        "Overlap in forest products and board; the business mix and asset transactions require separate review.",
    ),
    699: (
        "UPM-Kymmene",
        "FI",
        ["Forest", "Pulp", "Paper", "Specialty materials", "Energy"],
        "Overlap in fibre, paper and energy; diversified segments are not a whole-company match.",
    ),
}
STUDY = Path("studies/sectors/material/skogsbolag")


def export(root: Path, snapshot: str, output: Path) -> dict:
    if output.resolve().is_relative_to(root.resolve()):
        raise ValueError("Atlas exports must be written outside the Börsdata project")
    if date.fromisoformat(snapshot).isoformat() != snapshot:
        raise ValueError("Snapshot must be an ISO calendar date")
    # Import the source project's read-only, validated projection API, not load_data().
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "framework"))
    from core.data import read_validated
    from core.schemas import INSTRUMENTS_SCHEMA, YEARLY_REPORTS_SCHEMA
    from pandas.api.types import is_numeric_dtype

    sources = []

    def source(relative: Path, label: str) -> str:
        path = root / relative
        with path.open("rb") as stream:
            sha = hashlib.file_digest(stream, "sha256").hexdigest()
        identifier = f"source-{len(sources) + 1}"
        sources.append(
            {
                "id": identifier,
                "label": label,
                "path": relative.as_posix(),
                "sha256": sha,
                "bytes": path.stat().st_size,
            }
        )
        return identifier

    def archive(relative: Path, title: str, as_of: str, basis: str) -> dict:
        identifier = source(relative, title)
        return {
            "title": title,
            "as_of": as_of,
            "basis": basis,
            "source_id": identifier,
            "text": (root / relative).read_text(encoding="utf-8"),
        }

    base = Path("data/raw_api_snapshots") / snapshot
    instrument_path = base / "all_instruments/all_instruments.parquet"
    instrument_source = source(instrument_path, f"Börsdata instruments · saved {snapshot}")
    columns = [
        "ins_id",
        "name",
        "ticker",
        "isin",
        "instrument_type",
        "sector_id",
        "branch_id",
        "country_id",
        "report_currency",
        "stock_price_currency",
    ]
    instruments = read_validated(
        root / instrument_path,
        INSTRUMENTS_SCHEMA,
        columns=columns,
        filters=[("ins_id", "in", list(COHORT))],
    )
    if set(instruments.ins_id) != set(COHORT) or len(instruments) != len(COHORT):
        raise ValueError("The selected canonical listings are incomplete or duplicated")
    frames, report_sources = {}, {}
    for kind in ("yearly", "quarterly"):
        relative = base / f"all_reports/all_{kind}_reports.parquet"
        report_sources[kind] = source(relative, f"Börsdata {kind} reports · saved {snapshot}")
        columns = [
            "ins_id",
            "year",
            "period",
            "report_start_date",
            "report_end_date",
            "report_date",
            "currency",
            "currency_ratio",
            *AMOUNTS,
        ]
        frames[kind] = read_validated(
            root / relative,
            YEARLY_REPORTS_SCHEMA,
            columns=columns,
            filters=[("ins_id", "in", list(COHORT))],
        )
        if not all(is_numeric_dtype(frames[kind][key]) for key in (*AMOUNTS, "currency_ratio")):
            raise ValueError("Financial amounts and currency ratios must be numeric")
    study = archive(
        STUDY / "README.md",
        "Forestry branch study",
        "2026-05-22",
        "Marked research-grade 2026-05-22; later amendments remain in the saved text. Older prices, rankings and study conclusions have not been refreshed.",
    )
    kpis = archive(
        STUDY / "kpis.md",
        "Forestry KPI rationale",
        "2026-05-22",
        "Companion to the May 2026 branch study. This is archived methodology, not a refreshed financial assessment.",
    )
    holmen = archive(
        STUDY / "deep_dives/holm_b/deep_dive.md",
        "Holmen deep dive",
        "2026-05-21",
        "Run 2026-05-21; financial anchor 2025-06-20 and annual reports 2005, 2015 and 2024. Prices and investment judgments in this text are historical.",
    )
    holmen_sources = archive(
        STUDY / "deep_dives/holm_b/sources.md",
        "Holmen source register",
        "2026-05-21",
        "Source register associated with the archived Holmen deep dive; external documents are not downloaded by Atlas.",
    )
    companies = {}
    for instrument in instruments.to_dict("records"):
        identifier = int(instrument["ins_id"])
        name, country, segments, overlap = COHORT[identifier]
        if (
            instrument["sector_id"] != 7
            or instrument["branch_id"] != 21
            or instrument["instrument_type"] != 0
            or instrument["country_id"] != (1 if country == "SE" else 3)
        ):
            raise ValueError(f"Canonical listing metadata changed for {identifier}")
        companies[str(identifier)] = {
            "id": str(identifier),
            "name": name,
            "listing_name": instrument["name"],
            "ticker": instrument["ticker"],
            "isin": instrument["isin"],
            "listing_country": country,
            "report_currency": instrument["report_currency"],
            "stock_currency": instrument["stock_price_currency"],
            "sector_id": "7",
            "branch_id": "21",
            "segments": segments,
            "overlap": overlap,
            "context_as_of": study["as_of"],
            "context_source_id": study["source_id"],
            "instrument_source_id": instrument_source,
            "annual_source_id": report_sources["yearly"],
            "quarterly_source_id": report_sources["quarterly"],
            "annual": reports(
                frames["yearly"].loc[frames["yearly"].ins_id == identifier].to_dict("records"),
                snapshot,
                annual=True,
            ),
            "quarterly": reports(
                frames["quarterly"]
                .loc[frames["quarterly"].ins_id == identifier]
                .to_dict("records"),
                snapshot,
                annual=False,
            ),
            "research": [holmen, holmen_sources] if identifier == 102 else [],
        }
    result = {
        "version": 1,
        "as_of": snapshot,
        "exported_at": datetime.now(UTC).isoformat(),
        "default_company": "102",
        "countries": {"SE": "Sweden", "FI": "Finland"},
        "companies": companies,
        "branch": {
            "id": "21",
            "name": "Forestry",
            "source_name": "Skogsbolag",
            "sector_id": "7",
            "sector_name": "Materials",
            "source_id": study["source_id"],
            "as_of": study["as_of"],
            "description": "Forest ownership, wood products, pulp, paper and board. This first selection covers five Nordic companies from the existing branch study.",
            "drivers": [
                {
                    "title": "Demand and construction",
                    "text": "The archived study links timber demand to construction and fibre products to industrial and consumer demand. Country GDP provides background, not a company sales forecast.",
                    "indicator": "gdp_growth_fwd5",
                },
                {
                    "title": "Energy and production costs",
                    "text": "Energy intensity and access to fibre affect the cost curve. A country's net energy import share is background; each company's energy mix needs separate review.",
                    "indicator": "energy_net_imports_pct",
                },
                {
                    "title": "Capital and the cycle",
                    "text": "The study highlights leverage, cash generation and capital spending through downturns. Country fundamentals cannot establish a company's funding cost or debt capacity.",
                    "indicator": "gov_debt_pct_gdp",
                },
            ],
        },
        "research": [study, kpis],
        "sources": sources,
        "limitations": [
            "Map coverage follows the selected stock-exchange listings. It does not locate plants, land or sales exposure, and it is not a complete sector census.",
            "Amounts are millions of each report's currency, recovered by dividing stored Börsdata values by currency_ratio. No cross-currency totals are calculated.",
            "Reported profits and return ratios can include forest revaluations, disposals and other non-recurring effects. These figures are not normalized earnings.",
            "Börsdata free cash flow is the provider series. Lease principal and interest are not deducted; it is not owner earnings.",
            "Return on capital is an annual, pre-tax proxy: EBIT / (year-end equity + net debt). It is not company-reported ROCE or an adjusted ROIC measure.",
            "Peer comparisons require a matching full-year period and use ratios. Segment overlap is qualitative context from the archived study, not a current competitor census.",
            "No new branch forecast, valuation or investment recommendation is calculated. Missing observations remain missing.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False),
        encoding="utf-8",
    )
    return {
        "output": str(output),
        "snapshot": snapshot,
        "companies": len(companies),
        "annual_rows": sum(len(c["annual"]) for c in companies.values()),
        "quarterly_rows": sum(len(c["quarterly"]) for c in companies.values()),
        "bytes": output.stat().st_size,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--borsdata-root", type=Path, required=True)
    parser.add_argument("--snapshot", default="2026-08-10")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.borsdata_root, args.snapshot, args.output)))
