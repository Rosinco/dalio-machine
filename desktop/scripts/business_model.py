"""Pure financial transformations shared by the export and its contract tests."""

from __future__ import annotations

import math
from datetime import date

AMOUNTS = (
    "revenues",
    "operating_income",
    "profit_to_equity_holders",
    "cash_flow_from_operating_activities",
    "free_cash_flow",
    "total_equity",
    "net_debt",
    "total_assets",
    "cash_and_equivalents",
    "current_assets",
    "current_liabilities",
    "non_current_assets",
    "non_current_liabilities",
    "intangible_assets",
    "tangible_assets",
    "financial_assets",
)
RATIOS = {
    "operating_margin": ("operating_income", "revenues"),
    "operating_cash_margin": ("cash_flow_from_operating_activities", "revenues"),
    "equity_ratio": ("total_equity", "total_assets"),
    "net_debt_to_equity": ("net_debt", "total_equity"),
}


def numeric(value):
    return (
        float(value)
        if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        else None
    )


def date_text(value):
    if value is None or str(value) in {"NaT", "nan", "None"}:
        return None
    return date.fromisoformat(str(value)[:10]).isoformat()


def report(row: dict, as_of: str) -> dict:
    year, period = int(row["year"]), int(row["period"])
    if not 2000 <= year <= 2300 or period not in range(1, 6):
        raise ValueError("Invalid report year or period")
    start, end, published = (
        date_text(row.get(k)) for k in ("report_start_date", "report_end_date", "report_date")
    )
    if not start or not end or start > end or (published and published < end):
        raise ValueError("Invalid financial period dates")
    if end > as_of or (published and published > as_of):
        raise ValueError("Report is dated after snapshot")
    currency = row.get("currency")
    if (
        not isinstance(currency, str)
        or len(currency) != 3
        or not currency.isalpha()
        or not currency.isupper()
    ):
        raise ValueError("Report currency is missing or invalid")
    ratio = numeric(row.get("currency_ratio"))
    raw = {k: numeric(row.get(k)) for k in AMOUNTS}
    # DATA.md: stored amount = reporting-currency amount * currency_ratio.
    values = {
        k: numeric(v / ratio) if v is not None and ratio is not None and ratio > 0 else None
        for k, v in raw.items()
    }
    for key, (num, den) in RATIOS.items():
        a, b = values[num], values[den]
        values[key] = numeric(100 * a / b) if a is not None and b is not None and b > 0 else None
    equity, debt, ebit = (values[k] for k in ("total_equity", "net_debt", "operating_income"))
    capital = equity + debt if equity is not None and debt is not None else None
    values["return_on_capital"] = (
        numeric(100 * ebit / capital)
        if period == 5 and capital is not None and capital > 0 and ebit is not None
        else None
    )
    return {
        "year": year,
        "period": period,
        "start": start,
        "end": end,
        "report_date": published,
        "currency": currency,
        "currency_ratio": ratio,
        "raw": raw,
        "values": values,
    }


def reports(rows: list[dict], as_of: str, *, annual: bool) -> list[dict]:
    result, seen = [], set()
    for row in rows:
        item = report(row, as_of)
        key = (item["year"], item["period"])
        if (item["period"] == 5) != annual:
            raise ValueError("Unexpected financial period type")
        if key in seen:
            raise ValueError("Duplicate financial period")
        seen.add(key)
        result.append(item)
    return sorted(result, key=lambda r: (r["year"], r["period"]))


def business_index(raw: dict) -> dict:
    # Compare exactly matching full-year periods. No latest-row or FX substitution.
    periods = [
        set((r["year"], r["start"], r["end"]) for r in c["annual"])
        for c in raw["companies"].values()
    ]
    common = sorted(set.intersection(*periods)) if periods else []
    chosen = common[-1] if common else None
    summaries = {}
    for key, company in raw["companies"].items():
        summaries[key] = {
            k: v for k, v in company.items() if k not in {"annual", "quarterly", "research"}
        }
        summaries[key].update(
            {
                "latest_annual": company["annual"][-1] if company["annual"] else None,
                "latest_quarter": company["quarterly"][-1] if company["quarterly"] else None,
                "comparison": next(
                    (r for r in company["annual"] if (r["year"], r["start"], r["end"]) == chosen),
                    None,
                ),
                "research_count": len(company["research"]),
            }
        )
    return {
        **{k: v for k, v in raw.items() if k not in {"companies", "research"}},
        "companies": summaries,
        "common_year": chosen[0] if chosen else None,
    }
