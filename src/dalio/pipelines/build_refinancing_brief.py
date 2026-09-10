"""Export dated, descriptive refinancing comparisons from audited local releases."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path

from sqlalchemy import Engine, create_engine

from dalio.data_sources.eurostat_refinancing import (
    INDICATOR_APPARENT_COST,
    INDICATOR_AVG_RESIDUAL_MATURITY,
    INDICATOR_DUE_LE1Y,
    INDICATOR_FOREIGN_CURRENCY,
    INDICATOR_LT_VARIABLE_RATE,
    INDICATOR_RMD_SCOPE,
)
from dalio.storage.refinancing import load_stored_refinancing_batch

COUNTRIES = {"DE": "Germany", "FR": "France", "IT": "Italy", "ES": "Spain", "SE": "Sweden"}
METRICS = {
    "debt_pct_gdp": INDICATOR_RMD_SCOPE,
    "maturity_years": INDICATOR_AVG_RESIDUAL_MATURITY,
    "due_le1y_pct_gdp": INDICATOR_DUE_LE1Y,
    "foreign_currency_pct_gdp": INDICATOR_FOREIGN_CURRENCY,
    "apparent_cost_pct": INDICATOR_APPARENT_COST,
}


def build_brief(engine: Engine, *, as_of: date) -> dict:
    """Fail closed unless source bytes, immutable rows and current rows agree."""
    batch = load_stored_refinancing_batch(engine, as_of=as_of)
    histories = {
        (item.binding.spec.country, item.binding.spec.indicator): {
            row.date: {"value": float(row.value), "status": row.status}
            for row in item.frame.itertuples(index=False)
        }
        for item in batch
    }
    common = set.intersection(
        *(
            set(histories[(country, indicator)])
            for country in COUNTRIES
            for indicator in METRICS.values()
        )
    )
    if not common:
        raise ValueError("No common reported annual period across the five countries")
    period = max(common)
    baseline = date(period.year - 3, 1, 1)

    def value(country: str, indicator: str, on: date) -> float | None:
        return histories.get((country, indicator), {}).get(on, {}).get("value")

    countries = []
    for country, name in COUNTRIES.items():
        row = {"country": country, "name": name, "year": period.year}
        row.update(
            {field: value(country, indicator, period) for field, indicator in METRICS.items()}
        )
        row["variable_rate_pct_gdp"] = value(country, INDICATOR_LT_VARIABLE_RATE, period)
        row["due_le1y_share_pct"] = (
            100 * row["due_le1y_pct_gdp"] / row["debt_pct_gdp"] if row["debt_pct_gdp"] > 0 else None
        )
        row["statuses"] = {
            field: histories[(country, indicator)][period]["status"]
            for field, indicator in METRICS.items()
        }
        row["change_from_year"] = baseline.year
        row["three_year_changes"] = {
            field: row[field] - value(country, indicator, baseline)
            if value(country, indicator, baseline) is not None
            else None
            for field, indicator in METRICS.items()
        }
        countries.append(row)
    comparison = []
    for item in batch:
        if item.binding.spec.country != "EA21":
            continue
        spec = item.binding.spec
        history = histories[(spec.country, spec.indicator)]
        latest_on = max(history)
        prior_on = latest_on.replace(year=latest_on.year - 1)
        latest_value = history[latest_on]["value"]
        comparison.append(
            {
                "indicator": spec.indicator,
                "title": spec.title,
                "unit": spec.unit,
                "period": latest_on.isoformat(),
                "value": latest_value,
                "status": history[latest_on]["status"],
                "year_ago_change": latest_value - history[prior_on]["value"]
                if prior_on in history
                else None,
                "definition": spec.definition,
                "source_url": item.meta.source_url,
            }
        )
    return {
        "schema_version": 1,
        "as_of": as_of.isoformat(),
        "common_year": period.year,
        "complete_batch_available_at": batch[0].meta.available_at.isoformat(),
        "observation_count": sum(len(item.frame) for item in batch),
        "harmonized_ready": 31,
        "total_expected": 48,
        "national_native_planned": 17,
        "countries": countries,
        "euro_area_comparison": comparison,
        "due_share_formula": "100 * same-country same-year gov_10dd_rmd Y_LE1 PC_GDP / TOTAL PC_GDP",
        "interpretation_limits": [
            "Annual reference years describe year-end debt stocks; January 1 is a canonical year label, not the measurement day.",
            "The due-within-one-year share is calculated from rounded, matching Eurostat ratios; it is not a default probability or a funding forecast.",
            "Eurostat general-government debt includes deposits, securities and loans, consolidated at face value. It is broader than central-government marketable debt.",
            "Apparent cost is a publisher historical debt-cost rate, not the yield on a new issue or a marginal funding rate.",
            "The variable-rate series is debt with original maturity above one year; it is not a residual-maturity schedule or a measure of all near-term interest refixing.",
            "Sweden's variable-rate series is unavailable in the pinned package, not zero. Missing years are neither interpolated nor filled with zero.",
            "EA21 is a fixed-composition comparator, not a sovereign. Its non-consolidated face-value debt-securities scope differs from Eurostat Maastricht debt.",
            "ECB 1–12 month scheduled redemptions concern securities outstanding at the reference date; future issues, early redemptions and deficit financing can alter actual financing needs.",
            "These are the latest collected histories, including revisions. Their older years do not establish what was known in real time before collection.",
        ],
        "input_releases": [
            {
                "partition_id": item.binding.partition.partition_id,
                "partition_key": item.meta.partition_key,
                "source_url": item.meta.source_url,
                "country": item.binding.spec.country,
                "indicator": item.binding.spec.indicator,
                "first_date": min(item.frame["date"]).isoformat(),
                "latest_date": max(item.frame["date"]).isoformat(),
                "observations": len(item.frame),
                "missing_periods": len(item.frame.attrs["missing_period_records"]),
                "source_updated_at": item.meta.published_at.isoformat()
                if item.meta.published_at
                else None,
                "artifacts": [
                    {"role": a.role, "sha256": a.artifact_sha256, "path": str(a.artifact_path)}
                    for a in item.meta.artifacts
                ],
            }
            for item in batch
        ],
    }


def _number(value: float | None, *, signed: bool = False) -> str:
    return "Unavailable" if value is None else (f"{value:+.1f}" if signed else f"{value:.1f}")


def render_markdown(brief: dict) -> str:
    year = brief["common_year"]
    lines = [
        f"# Sovereign refinancing evidence — {brief['as_of']}",
        "",
        f"Verified {brief['observation_count']:,} observations across 31 harmonized histories. "
        "The fixed package contains 48 streams; 17 national debt-office streams remain planned.",
        "",
        f"Complete collected batch available at {brief['complete_batch_available_at']}. "
        "This is a descriptive comparison of stored evidence; no composite refinancing score is assigned.",
        "",
        f"## General-government comparison, {year}",
        "",
        "All countries use the latest common reported year. Debt and due amounts use the same Eurostat residual-maturity denominator.",
        "",
        "| Country | Debt / GDP % | Avg residual maturity, years | Due ≤1y / GDP % | Due ≤1y / debt % (calculated) | FX debt / GDP % | Apparent cost % | LT variable-rate / GDP % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    fields = [
        "debt_pct_gdp",
        "maturity_years",
        "due_le1y_pct_gdp",
        "due_le1y_share_pct",
        "foreign_currency_pct_gdp",
        "apparent_cost_pct",
        "variable_rate_pct_gdp",
    ]
    for row in brief["countries"]:
        lines.append(
            "| " + " | ".join([row["name"], *[_number(row[field]) for field in fields]]) + " |"
        )
    lines.extend(
        [
            "",
            "Calculated due share: `" + brief["due_share_formula"] + "`. "
            "This is a share of the reported stock, not the share of GDP. "
            "Native observation flags remain in the JSON and source artifacts.",
            "",
            f"## Changes from {year - 3} to {year}",
            "",
            "Exact endpoints only; no interpolation. Ratio changes are percentage points.",
            "",
            "| Country | Debt / GDP, pp | Maturity, years | Due ≤1y / GDP, pp | Apparent cost, pp |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in brief["countries"]:
        changes = row["three_year_changes"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["name"],
                    *[
                        _number(changes[field], signed=True)
                        for field in (
                            "debt_pct_gdp",
                            "maturity_years",
                            "due_le1y_pct_gdp",
                            "apparent_cost_pct",
                        )
                    ],
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Euro-area securities comparator",
            "",
            "EA21 fixed composition; non-consolidated general-government debt securities at face value. "
            "These figures are kept separate from the national table.",
            "",
            "| Measure | Reference month | Value | Unit | Change from year earlier |",
            "|---|---|---:|---|---:|",
        ]
    )
    for row in brief["euro_area_comparison"]:
        lines.append(
            f"| [{row['title']}]({row['source_url']}) | {row['period'][:7]} | "
            f"{_number(row['value'])} | {row['unit']} | {_number(row['year_ago_change'], signed=True)} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation and next questions",
            "",
            "The reported debt burden, near-term maturity share and historical cost describe different exposures. "
            "They should be read together before forming an issuer risk judgment. "
            "A short average maturity alone does not establish a funding crisis.",
            "",
            "The next national-source pass should explain Sweden's debt composition and redemption schedule, "
            "then compare contractual maturities with interest refixing, currency hedges and gross funding plans. "
            "These annual buckets cannot establish a five-year maturity wall.",
            "",
            "## Coverage and interpretation limits",
            "",
            *["- " + limit for limit in brief["interpretation_limits"]],
            "",
            "Definitions: [Eurostat government-debt metadata](https://ec.europa.eu/eurostat/cache/metadata/en/gov_10dd_sgd_esms.htm); "
            "[ECB government-finance statistics](https://www.ecb.europa.eu/stats/macroeconomic_and_sectoral/government_finance/html/index.en.html).",
            "",
            "## Retained input histories",
            "",
            "| Partition / official source | First finite period | Latest finite period | Observations | Missing native periods |",
            "|---|---|---|---:|---:|",
        ]
    )
    for row in brief["input_releases"]:
        lines.append(
            f"| [{row['partition_id']}]({row['source_url']}) | {row['first_date']} | "
            f"{row['latest_date']} | {row['observations']} | {row['missing_periods']} |"
        )
    lines.extend(
        [
            "",
            "The accompanying JSON retains all source-update clocks, input identities and source/native/missingness/catalogue hashes and paths.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    parser.add_argument("--output-dir", type=Path, default=Path("data/snapshots"))
    args = parser.parse_args(argv)
    engine = create_engine(
        "sqlite://",
        creator=lambda: sqlite3.connect(args.db.resolve().as_uri() + "?mode=ro", uri=True),
    )
    try:
        brief = build_brief(engine, as_of=args.as_of)
    finally:
        engine.dispose()
    content = (
        json.dumps(brief, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
    stem = f"refinancing_{args.as_of}_{digest[:12]}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix, body in (("json", content), ("md", render_markdown(brief))):
        path = args.output_dir / f"{stem}.{suffix}"
        try:
            with path.open("x", encoding="utf-8") as handle:
                handle.write(body)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != body:
                raise ValueError(
                    f"Dated briefing already exists with different content: {path}"
                ) from None
        print(path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
