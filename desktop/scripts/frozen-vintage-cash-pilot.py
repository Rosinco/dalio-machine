"""Read-only input-vintage audit and one-origin cash-proxy prediction pilot.

This is not a reconstructed daily point-in-time database. Inputs are frozen at
2025-06-21 by default, outcomes come from the 2026-08-10 provider snapshot, and
every frozen company listing remains in the attrition/status output.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--borsdata-root",
    type=Path,
    default=os.environ.get("MACRO_ATLAS_BORSDATA_ROOT"),
    required="MACRO_ATLAS_BORSDATA_ROOT" not in os.environ,
    help="Read-only source repository; alternatively set MACRO_ATLAS_BORSDATA_ROOT.",
)
parser.add_argument(
    "--desktop-root",
    type=Path,
    default=Path(__file__).resolve().parents[1],
    help="Desktop app root containing financial-data/.",
)
parser.add_argument(
    "--output", type=Path, required=True, help="Artifact directory outside the source repository."
)
parser.add_argument(
    "--input-vintage",
    default="data/raw_api",
    help="Frozen input folder relative to --borsdata-root.",
)
parser.add_argument(
    "--outcome-vintage",
    default="data/raw_api_snapshots/2026-08-10",
    help="Later outcome folder relative to --borsdata-root.",
)
parser.add_argument("--origin", default="2025-06-21", help="Input snapshot date, YYYY-MM-DD.")
parser.add_argument(
    "--outcome-as-of", default="2026-08-10", help="Outcome snapshot date, YYYY-MM-DD."
)
parser.add_argument("--audit-date", default="2026-09-12", help="Receipt date, YYYY-MM-DD.")
args = parser.parse_args()
SOURCE = args.borsdata_root.resolve()
ROOT = args.desktop_root.resolve().parent
OUT = args.output.resolve()
ORIGIN = date.fromisoformat(args.origin).isoformat()
OUTCOME_AS_OF = date.fromisoformat(args.outcome_as_of).isoformat()
AUDIT_DATE = date.fromisoformat(args.audit_date).isoformat()
PREFIX = f"frozen-{ORIGIN[:4]}-pilot"
if not ORIGIN < OUTCOME_AS_OF <= AUDIT_DATE:
    parser.error("Require origin < outcome-as-of <= audit-date.")
if OUT.is_relative_to(SOURCE):
    parser.error("Output must be outside the read-only source repository.")
OUT.mkdir(parents=True, exist_ok=True)
DATES = ["report_start_date", "report_end_date", "report_date"]
WEIGHTS = np.array([30, 25, 20, 15, 10], dtype=float)
INCLUDED_TYPES = {0, 1, 3, 8, 9, 10}
MONEY = [
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
    "gross_income",
    "profit_before_tax",
    "net_sales",
    "total_liabilities_and_equity",
    "cash_flow_from_investing_activities",
    "cash_flow_from_financing_activities",
    "cash_flow_for_the_year",
]


def scalar(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def save(name, value):
    def clean(item):
        if isinstance(item, dict):
            return {key: clean(val) for key, val in item.items()}
        if isinstance(item, list):
            return [clean(val) for val in item]
        if isinstance(item, (np.integer, np.floating, np.bool_)):
            item = item.item()
        return None if isinstance(item, float) and not math.isfinite(item) else item

    (OUT / name).write_text(
        json.dumps(clean(value), indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    )


def near(a, b):
    return math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-6)


def read_vintage(folder, cutoff):
    annual_path = SOURCE / folder / "all_reports/all_yearly_reports.parquet"
    instruments_path = SOURCE / folder / "all_instruments/all_instruments.parquet"
    annual = pd.read_parquet(annual_path)
    instruments = pd.read_parquet(instruments_path)
    for key in DATES:
        annual[key + "_text"] = annual[key].astype(str).str.slice(0, 10)
        annual[key] = pd.to_datetime(annual[key + "_text"], errors="coerce", utc=True)
    annual["full_year"] = (
        (annual.report_end_date - annual.report_start_date).dt.days.add(1).between(330, 400)
    )
    annual["currency_valid"] = annual.currency.astype(str).str.fullmatch("[A-Z]{3}")
    annual["positive_ratio"] = (
        annual.currency_ratio.notna()
        & np.isfinite(annual.currency_ratio)
        & annual.currency_ratio.gt(0)
    )
    annual["native_fcf"] = annual.free_cash_flow / annual.currency_ratio.where(
        annual.positive_ratio
    )
    annual["cash_valid"] = (
        annual.native_fcf.notna()
        & np.isfinite(annual.native_fcf)
        & annual.native_fcf.abs().le(1e12)
    )
    annual["placeholder"] = annual[MONEY].fillna(0).eq(0).all(axis=1) & annual[MONEY].eq(0).any(
        axis=1
    )
    bound = pd.Timestamp(cutoff, tz="UTC")
    annual["ended_by_snapshot"] = annual.report_end_date.notna() & annual.report_end_date.le(bound)
    annual["published_by_snapshot"] = (
        annual.report_date.notna()
        & annual.report_date.le(bound)
        & annual.report_date.ge(annual.report_end_date)
    )
    annual["row_valid"] = (
        annual.period.eq(5)
        & annual.full_year
        & annual.currency_valid
        & annual.positive_ratio
        & annual.cash_valid
        & ~annual.placeholder
        & annual.ended_by_snapshot
        & annual.published_by_snapshot
    )
    stats = {
        "snapshot": cutoff,
        "annualPath": str(annual_path.relative_to(SOURCE)),
        "annualSha256": hashlib.sha256(annual_path.read_bytes()).hexdigest(),
        "instrumentsPath": str(instruments_path.relative_to(SOURCE)),
        "instrumentsSha256": hashlib.sha256(instruments_path.read_bytes()).hexdigest(),
        "annualRows": len(annual),
        "annualListings": int(annual.ins_id.nunique()),
        "instrumentRows": len(instruments),
        "companyListingRows": int(instruments.instrument_type.isin(INCLUDED_TYPES).sum()),
        "fiscalYearMin": int(annual.year.min()),
        "fiscalYearMax": int(annual.year.max()),
        "fiscalYearCounts": {
            str(k): int(v) for k, v in annual.year.value_counts().sort_index().items()
        },
        "missingPublication": int(annual.report_date.isna().sum()),
        "publicationAfterSnapshot": int(annual.report_date.gt(bound).sum()),
        "publicationBeforePeriodEnd": int(annual.report_date.lt(annual.report_end_date).sum()),
        "invalidStart": int(annual.report_start_date.isna().sum()),
        "invalidEnd": int(annual.report_end_date.isna().sum()),
        "nonFullPeriod": int((~annual.full_year).sum()),
        "missingCash": int(annual.free_cash_flow.isna().sum()),
        "invalidNativeCashOrRatio": int((~annual.cash_valid | ~annual.positive_ratio).sum()),
        "zeroCash": int(annual.free_cash_flow.eq(0).sum()),
        "allMonetaryZeroPlaceholder": int(annual.placeholder.sum()),
        "validPublishedNativeCashFullYears": int(annual.row_valid.sum()),
        "duplicateFiscalKeys": int(annual.duplicated(["ins_id", "year", "period"]).sum()),
    }
    return annual, instruments, stats


old, old_inst, old_stats = read_vintage(args.input_vintage, ORIGIN)
new, new_inst, new_stats = read_vintage(args.outcome_vintage, OUTCOME_AS_OF)
old_ids = set(old_inst.ins_id)
new_ids = set(new_inst.ins_id)
old_company = old_inst[old_inst.instrument_type.isin(INCLUDED_TYPES)].copy()
new_company = new_inst[new_inst.instrument_type.isin(INCLUDED_TYPES)].copy()
old_company_ids, new_company_ids = set(old_company.ins_id), set(new_company.ins_id)
new_inst_by_id = new_inst.set_index("ins_id")

comparison_columns = [
    "ins_id",
    "year",
    "period",
    "currency",
    "currency_ratio",
    "free_cash_flow",
    "native_fcf",
    "cash_flow_from_operating_activities",
    "cash_flow_from_investing_activities",
    "revenues",
    "report_start_date_text",
    "report_end_date_text",
    "report_date_text",
]
overlap = old[comparison_columns].merge(
    new[comparison_columns], on=["ins_id", "year", "period"], suffixes=("_old", "_new")
)
same_currency = overlap.currency_old.eq(overlap.currency_new)
finite_native = (
    overlap.native_fcf_old.notna()
    & overlap.native_fcf_new.notna()
    & np.isfinite(overlap.native_fcf_old)
    & np.isfinite(overlap.native_fcf_new)
)
native_changed = (
    finite_native
    & same_currency
    & ~np.isclose(overlap.native_fcf_old, overlap.native_fcf_new, rtol=1e-8, atol=1e-6)
)
raw_changed = (
    overlap.free_cash_flow_old.notna()
    & overlap.free_cash_flow_new.notna()
    & ~np.isclose(overlap.free_cash_flow_old, overlap.free_cash_flow_new, rtol=1e-8, atol=1e-6)
)
date_changed = (
    overlap[[k + "_text_old" for k in DATES]]
    .set_axis(DATES, axis=1)
    .ne(overlap[[k + "_text_new" for k in DATES]].set_axis(DATES, axis=1))
    .any(axis=1)
)
revisions = {
    "matchedFiscalKeys": len(overlap),
    "matchedListings": int(overlap.ins_id.nunique()),
    "sameCurrencyFiniteNativeCashKeys": int((same_currency & finite_native).sum()),
    "nativeCashChangedKeys": int(native_changed.sum()),
    "nativeCashChangedListings": int(overlap.loc[native_changed, "ins_id"].nunique()),
    "rawCashChangedKeys": int(raw_changed.sum()),
    "currencyCodeChangedKeys": int((~same_currency).sum()),
    "currencyRatioChangedKeys": int(
        (
            ~np.isclose(
                overlap.currency_ratio_old,
                overlap.currency_ratio_new,
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
        ).sum()
    ),
    "periodOrPublicationDateChangedKeys": int(date_changed.sum()),
    "nativeCashSignChangedKeys": int(
        (
            same_currency
            & finite_native
            & np.sign(overlap.native_fcf_old).ne(np.sign(overlap.native_fcf_new))
        ).sum()
    ),
    "note": "Differences are observed provider-vintage differences, not individually verified accounting restatements; conversion, source corrections, fiscal-period metadata and restatements can all contribute.",
}
same_native = {}
for key in [
    "cash_flow_from_operating_activities",
    "cash_flow_from_investing_activities",
    "revenues",
]:
    a = overlap[key + "_old"] / overlap.currency_ratio_old
    b = overlap[key + "_new"] / overlap.currency_ratio_new
    same_native[key] = (
        a.notna()
        & b.notna()
        & np.isfinite(a)
        & np.isfinite(b)
        & np.isclose(a, b, rtol=1e-8, atol=1e-6)
    )
delta = (overlap.native_fcf_old - overlap.native_fcf_new).abs()
scale = np.maximum(overlap.native_fcf_old.abs(), overlap.native_fcf_new.abs())
measurement_diagnostics = {
    "matchedSameCurrencyFiniteNativeFCF": int((same_currency & finite_native).sum()),
    "fcfChangedOver1PercentMaxAbs": int(
        (same_currency & finite_native & delta.gt(np.maximum(1e-6, scale * 0.01))).sum()
    ),
    "fcfChangedOver20PercentMaxAbs": int(
        (same_currency & finite_native & delta.gt(np.maximum(1e-6, scale * 0.2))).sum()
    ),
    "fcfChangedButNativeCFOandInvestingUnchanged": int(
        (
            native_changed
            & same_native["cash_flow_from_operating_activities"]
            & same_native["cash_flow_from_investing_activities"]
        ).sum()
    ),
    "fcfChangedButNativeCFOInvestingRevenueUnchanged": int(
        (
            native_changed
            & same_native["cash_flow_from_operating_activities"]
            & same_native["cash_flow_from_investing_activities"]
            & same_native["revenues"]
        ).sum()
    ),
}
save("source-vintage-change-diagnostics.json", measurement_diagnostics)

catalog = json.loads((ROOT / "desktop/financial-data/catalog.json").read_text())
pack_id = catalog["packs"][0]["id"]
pack_path = ROOT / "desktop/financial-data" / (pack_id + ".sqlite")
con = sqlite3.connect(f"file:{pack_path}?mode=ro", uri=True)
idx = json.loads(
    gzip.decompress(con.execute("SELECT payload FROM metadata WHERE key='index'").fetchone()[0])
)
pack_years, pack_sources, pack_latest, pack_pub = Counter(), Counter(), Counter(), Counter()
pack_hist_counts = Counter()
for (blob,) in con.execute("SELECT payload FROM companies"):
    company = json.loads(gzip.decompress(blob))
    annual = company["annual"]
    pack_hist_counts[len(annual)] += 1
    if annual:
        pack_latest[str(annual[-1][0])] += 1
    for r in annual:
        pack_years[str(r[0])] += 1
        pack_sources[r[7]] += 1
        pack_pub["known" if r[4] is not None else "missing"] += 1
con.close()
survivorship_path = (
    SOURCE / f"data/complementing/survivorship/observed_delistings_{OUTCOME_AS_OF}.parquet"
)
observed = pd.read_parquet(survivorship_path) if survivorship_path.exists() else pd.DataFrame()
audit = {
    "asOf": AUDIT_DATE,
    "sourceRoot": str(SOURCE),
    "availableVendorDownloadVintages": [ORIGIN, OUTCOME_AS_OF],
    "generator": {
        "path": str(Path(__file__).resolve()),
        "sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputVintage": args.input_vintage,
        "outcomeVintage": args.outcome_vintage,
    },
    "rawVintages": [old_stats, new_stats],
    "crossVintageDifferences": revisions,
    "providerMeasurementChangeDiagnostics": measurement_diagnostics,
    "universe": {
        "oldAllInstruments": len(old_ids),
        "newAllInstruments": len(new_ids),
        "departedAllInstruments": len(old_ids - new_ids),
        "newAllInstrumentIds": len(new_ids - old_ids),
        "oldCompanyListings": len(old_company_ids),
        "newCompanyListings": len(new_company_ids),
        "unionCompanyListings": len(old_company_ids | new_company_ids),
        "departedCompanyListings": len(old_company_ids - new_company_ids),
        "newCompanyListingIds": len(new_company_ids - old_company_ids),
        "oldCompanyIsinUniqueNonMissing": int(old_company["isin"].nunique()),
        "newCompanyIsinUniqueNonMissing": int(new_company["isin"].nunique()),
        "observedDepartureTablePath": str(survivorship_path.relative_to(SOURCE)),
        "observedDepartureTableRows": len(observed) if survivorship_path.exists() else None,
        "observedDepartureTableAvailable": survivorship_path.exists(),
        "observedDepartureTableColumns": list(observed.columns),
        "note": "Instrument absence is an unadjudicated snapshot observation. No delisting status, effective delisting date, delisting return or pre-2025 historical membership panel exists in these directories. Multiple listings/share classes/receipts are not independent issuers.",
    },
    "pack": {
        "path": str(pack_path),
        "id": pack_id,
        "sourceAsOf": idx["as_of"],
        "schema": [
            "metadata(key TEXT PRIMARY KEY, payload BLOB gzip JSON)",
            "companies(id TEXT PRIMARY KEY, payload BLOB gzip JSON, sha256 TEXT)",
        ],
        "reportRowFormat": [
            "year",
            "period",
            "start",
            "end",
            "published",
            "nativeCurrency",
            "currencyRatio",
            "sourceId",
        ]
        + idx["columns"],
        "summary": idx["summary"],
        "annualFiscalYearCounts": dict(pack_years),
        "annualSources": dict(pack_sources),
        "latestAnnualFiscalYearCounts": dict(pack_latest),
        "annualPublicationCounts": dict(pack_pub),
        "companyHistoryLengthCounts": {str(k): v for k, v in sorted(pack_hist_counts.items())},
        "revisionRetention": "export_financials.py sorts by source_as_of and drops duplicates on ins_id/year/period keeping last. Older rows survive only where the later snapshot lacks that fiscal key. This is a mixed latest-available-vintage panel, not a historical version ledger.",
    },
    "limitations": [
        f"FCF changes across vintages while native CFO and investing cash remain identical in {measurement_diagnostics['fcfChangedButNativeCFOandInvestingUnchanged']:,} overlapping rows. Possible provider definition changes/corrections confound the pilot: its coverage measures cross-vintage proxy agreement and cannot isolate forecasting error or business volatility.",
        f"This pilot uses input vintage {ORIGIN} and outcome vintage {OUTCOME_AS_OF}. Historical publication dates do not restore earlier report values or universe membership.",
        f"A frozen {ORIGIN} input pilot can control input-vintage leakage only from that date. It is not a comprehensive point-in-time database and not a pre-registered experiment: model choices were made later.",
        "The fresh batch endpoint has approximately 20 annual/40 quarterly history depth caps. Old-only deep rows in the current pack can reflect endpoint depth rather than reporting attrition.",
        f"Annual targets may already be partly elapsed at the {ORIGIN} origin; the experiment forecasts the full next annual reported cash proxy, not cash generated solely after the origin.",
        f"Later outcome values are from the {OUTCOME_AS_OF} snapshot and may be restated; they are not guaranteed first-published values.",
        "Overlapping rolling origins share training years and target observations. Horizons and multiple share listings are correlated; naive fold counts are not independent sample sizes.",
        "A current-vintage retrospective test should use a target-year holdout with no model/band selection on its locked years, and report issuer/target-year clustering and subgroup sample sizes.",
        "Historical provider FCF remains an unreviewed cash proxy; predictive coverage is not validation of distributable equity cash, a DCF or an investment strategy.",
    ],
}
save("source-vintage-audit.json", audit)
print(
    json.dumps(
        {
            "phase": "vintage audit complete",
            "rawVintages": [
                {
                    k: s[k]
                    for k in [
                        "snapshot",
                        "annualRows",
                        "annualListings",
                        "missingPublication",
                        "validPublishedNativeCashFullYears",
                    ]
                }
                for s in [old_stats, new_stats]
            ],
            "universe": audit["universe"],
            "differences": revisions,
        },
        indent=2,
    ),
    flush=True,
)

old_groups = {
    int(k): g.sort_values(["report_end_date", "year"], ascending=False, na_position="last")
    for k, g in old.groupby("ins_id", sort=False)
}
new_keys = {(int(r.ins_id), int(r.year), int(r.period)): r for r in new.itertuples(index=False)}


def row_problem(r, cutoff):
    if r.period != 5:
        return "not_annual"
    if pd.isna(r.report_start_date) or pd.isna(r.report_end_date):
        return "invalid_period_dates"
    if not r.full_year:
        return "non_full_annual_period"
    if pd.isna(r.report_date):
        return "missing_publication_date"
    if r.report_date > cutoff:
        return "publication_after_snapshot"
    if r.report_date < r.report_end_date:
        return "publication_before_period_end"
    if not r.currency_valid:
        return "invalid_native_currency"
    if not r.positive_ratio:
        return "missing_positive_currency_ratio"
    if not r.cash_valid:
        return "missing_or_invalid_native_cash"
    if r.placeholder:
        return "all_monetary_zero_placeholder"
    return None


records = []
origin_date = pd.Timestamp(ORIGIN, tz="UTC")
outcome_date = pd.Timestamp(OUTCOME_AS_OF, tz="UTC")
for instrument in old_company.sort_values("ins_id").itertuples(index=False):
    iid = int(instrument.ins_id)
    rec = {
        "id": str(iid),
        "name": instrument.name,
        "isin": scalar(instrument.isin),
        "instrument_type": int(instrument.instrument_type),
        "sector_id_at_origin": scalar(instrument.sector_id),
        "branch_id_at_origin": scalar(instrument.branch_id),
        "country_id_at_origin": scalar(instrument.country_id),
        "origin": ORIGIN,
        "outcome_snapshot": OUTCOME_AS_OF,
        "quote_currency_at_origin": scalar(instrument.stock_price_currency),
        "in_later_instrument_directory": iid in new_ids,
        "in_later_company_directory": iid in new_company_ids,
        "status": "training_unavailable",
        "reason": None,
        "training_years": [],
        "training_cash_native": [],
        "training_ratios": [],
        "training_publication_dates": [],
        "currency": None,
        "latest_training_year": None,
        "target_year": None,
        "target_start": None,
        "target_end": None,
        "target_publication": None,
        "training_revision_count_in_later_snapshot": None,
        "weighted_mean": None,
        "slope": None,
        "intercept": None,
        "low": None,
        "mid": None,
        "high": None,
        "actual": None,
        "hit": None,
        "actual_minus_mid": None,
        "miss_direction": None,
        "diagnostic_flags": [],
    }
    if iid in new_ids:
        current = new_inst_by_id.loc[iid]
        rec["identity_isin_changed"] = (
            scalar(instrument.isin) is not None
            and scalar(current["isin"]) is not None
            and instrument.isin != current["isin"]
        )
        rec["quote_currency_changed"] = scalar(instrument.stock_price_currency) != scalar(
            current.stock_price_currency
        )
    else:
        rec["identity_isin_changed"] = None
        rec["quote_currency_changed"] = None
        rec["diagnostic_flags"].append("absent_from_later_directory_not_adjudicated_delisting")
    g = old_groups.get(iid)
    if g is None or g.empty:
        rec["reason"] = "no_frozen_annual_history"
        records.append(rec)
        continue
    ended = g[g.ended_by_snapshot]
    if ended.empty:
        rec["reason"] = "no_annual_period_ended_by_origin"
        records.append(rec)
        continue
    rows = list(ended.itertuples(index=False))
    latest = rows[0]
    rec.update(
        currency=latest.currency,
        latest_training_year=int(latest.year),
        target_year=int(latest.year) + 1,
    )
    if (origin_date - latest.report_end_date).days > 550:
        rec["reason"] = "latest_annual_stale_over_550_days"
        records.append(rec)
        continue
    # An invalid-date higher year cannot silently disappear behind a usable year.
    if g[g.report_end_date.isna() & g.year.gt(latest.year)].shape[0]:
        rec["reason"] = "newer_annual_has_unknown_end_date"
        records.append(rec)
        continue
    if len(rows) < 5:
        rec["reason"] = "fewer_than_five_ended_annual_periods"
        records.append(rec)
        continue
    chosen = []
    for r in rows[:5]:
        problem = row_problem(r, origin_date)
        if problem:
            rec["reason"] = "training_" + problem
            break
        if r.currency != latest.currency:
            rec["reason"] = "training_native_currency_switch"
            break
        if chosen and (
            r.year != chosen[-1].year - 1
            or not 1 <= (chosen[-1].report_start_date - r.report_end_date).days <= 35
        ):
            rec["reason"] = "training_fiscal_gap_or_overlap"
            break
        chosen.append(r)
    rec["training_years"] = [int(r.year) for r in chosen]
    rec["training_cash_native"] = [float(r.native_fcf) for r in chosen]
    rec["training_ratios"] = [float(r.currency_ratio) for r in chosen]
    rec["training_publication_dates"] = [r.report_date_text for r in chosen]
    if len(chosen) != 5:
        records.append(rec)
        continue
    rec["status"] = "outcome_unavailable"
    x = np.array([r.year - latest.year for r in chosen], dtype=float)
    y = np.array(rec["training_cash_native"], dtype=float)
    sw = WEIGHTS.sum()
    sx = (WEIGHTS * x).sum()
    sy = (WEIGHTS * y).sum()
    slope = (sw * (WEIGHTS * x * y).sum() - sx * sy) / (sw * (WEIGHTS * x * x).sum() - sx * sx)
    intercept = (sy - slope * sx) / sw
    mid = intercept + slope
    rec.update(
        weighted_mean=float(sy / sw),
        slope=float(slope),
        intercept=float(intercept),
        mid=float(mid),
        low=float(mid - abs(mid) * 0.1),
        high=float(mid + abs(mid) * 0.1),
    )
    changed = 0
    available_revision_comparisons = 0
    for train in chosen:
        updated = new_keys.get((iid, int(train.year), 5))
        if updated is not None and updated.currency == train.currency and updated.cash_valid:
            available_revision_comparisons += 1
            changed += int(not near(float(train.native_fcf), float(updated.native_fcf)))
    rec["training_revision_count_in_later_snapshot"] = changed
    rec["training_revision_comparisons_available"] = available_revision_comparisons
    if changed:
        rec["diagnostic_flags"].append("training_cash_changed_in_later_vintage")
    actual = new_keys.get((iid, int(latest.year) + 1, 5))
    if actual is None:
        rec["reason"] = "next_annual_outcome_absent"
        records.append(rec)
        continue
    rec.update(
        target_start=actual.report_start_date_text,
        target_end=actual.report_end_date_text,
        target_publication=actual.report_date_text,
    )
    problem = row_problem(actual, outcome_date)
    if problem:
        rec["reason"] = "outcome_" + problem
        records.append(rec)
        continue
    if actual.report_end_date > outcome_date:
        rec["reason"] = "outcome_period_not_ended_by_snapshot"
        records.append(rec)
        continue
    if actual.report_end_date <= origin_date or actual.report_date <= origin_date:
        rec["reason"] = "target_not_forward_of_origin"
        records.append(rec)
        continue
    if actual.currency != latest.currency:
        rec["reason"] = "outcome_native_currency_switch"
        records.append(rec)
        continue
    if not 1 <= (actual.report_start_date - latest.report_end_date).days <= 35:
        rec["reason"] = "target_fiscal_gap_or_overlap"
        records.append(rec)
        continue
    if rec["identity_isin_changed"]:
        rec["reason"] = "listing_isin_changed_requires_identity_review"
        records.append(rec)
        continue
    actual_cash = float(actual.native_fcf)
    rec.update(
        status="evaluated",
        reason=None,
        actual=actual_cash,
        actual_ratio=float(actual.currency_ratio),
        actual_raw=float(actual.free_cash_flow),
        actual_minus_mid=float(actual_cash - mid),
    )
    # Boundaries allow only ordinary floating arithmetic tolerance.
    tolerance = max(1e-8, 1e-10 * abs(mid))
    hit = rec["low"] - tolerance <= actual_cash <= rec["high"] + tolerance
    rec["hit"] = bool(hit)
    rec["miss_direction"] = None if hit else "below" if actual_cash < rec["low"] else "above"
    rec["target_period_partly_elapsed_at_origin"] = bool(actual.report_start_date <= origin_date)
    rec["absolute_error"] = abs(actual_cash - mid)
    rec["symmetric_absolute_percentage_error"] = (
        0.0
        if abs(actual_cash) + abs(mid) == 0
        else 200 * abs(actual_cash - mid) / (abs(actual_cash) + abs(mid))
    )
    rec["actual_to_predicted_absolute_error_ratio"] = (
        None if abs(mid) < 1e-9 else abs(actual_cash - mid) / abs(mid)
    )
    if np.sign(actual_cash) != np.sign(mid):
        rec["diagnostic_flags"].append("actual_and_mid_differ_in_sign")
    if np.sign(actual_cash - y[0]) != np.sign(slope) and abs(slope) > 1e-9:
        rec["diagnostic_flags"].append("next_annual_change_opposes_fitted_slope")
    if abs(mid) < 1e-9:
        rec["diagnostic_flags"].append("zero_mid_has_zero_default_percentage_band")
    records.append(rec)

frame = pd.DataFrame(records)
evaluated = frame[frame.status.eq("evaluated")]
training = frame[frame.status.ne("training_unavailable")]


def aggregate(f):
    e = f[f.status.eq("evaluated")]
    t = f[f.status.ne("training_unavailable")]
    return {
        "frozenListings": len(f),
        "trainingEligible": len(t),
        "evaluated": len(e),
        "unavailableOutcomeAfterTraining": len(t) - len(e),
        "hits": int(e.hit.eq(True).sum()),
        "below": int(e.miss_direction.eq("below").sum()),
        "above": int(e.miss_direction.eq("above").sum()),
        "coveragePercent": None if e.empty else float(100 * e.hit.eq(True).mean()),
        "medianSymmetricAbsolutePercentageError": None
        if e.empty
        else float(e.symmetric_absolute_percentage_error.median()),
        "absentFromLaterDirectory": int(f.in_later_instrument_directory.eq(False).sum()),
        "evaluatedAbsentFromLaterDirectory": int(e.in_later_instrument_directory.eq(False).sum()),
        "trainingEligibleAbsentFromLaterDirectory": int(
            t.in_later_instrument_directory.eq(False).sum()
        ),
        "trainingReasons": {
            str(k): int(v)
            for k, v in f[f.status.eq("training_unavailable")].reason.value_counts().items()
        },
        "outcomeReasons": {
            str(k): int(v) for k, v in t[t.status.ne("evaluated")].reason.value_counts().items()
        },
    }


summary = {
    "format": "macro-atlas-frozen-vintage-cash-pilot",
    "version": 1,
    "created": AUDIT_DATE,
    "origin": ORIGIN,
    "outcomeSnapshot": OUTCOME_AS_OF,
    "description": "One frozen-input-origin, next-annual saved provider FCF proxy test. All frozen company listings are retained in the status ledger; valid training needs five consecutive full annual periods, known publication dates by origin, positive saved conversion ratios and one native reporting currency. It is a vintage-controlled input pilot, not a comprehensive point-in-time or pre-registered backtest.",
    "model": {
        "historyYears": 5,
        "weightsNewestFirst": [30, 25, 20, 15, 10],
        "projection": "weighted least squares on fiscal year minus latest year; first next-year payment = fitted intercept + slope",
        "band": "mid ± 10% × abs(mid)",
        "inclusivePeriodGapDays": [1, 35],
        "cashUnit": "millions in native reporting currency recovered independently in each snapshot as raw provider monetary amount / that row currency_ratio",
        "priceRequired": False,
    },
    "overall": aggregate(frame),
    "byLatestTrainingFiscalYear": {
        str(int(k)): aggregate(g)
        for k, g in frame[frame.latest_training_year.notna()].groupby("latest_training_year")
    },
    "byLaterDirectoryPresence": {
        str(bool(k)): aggregate(g) for k, g in frame.groupby("in_later_instrument_directory")
    },
    "byInstrumentType": {str(int(k)): aggregate(g) for k, g in frame.groupby("instrument_type")},
    "bySectorAtOrigin": {
        str(k): aggregate(g) for k, g in frame.groupby("sector_id_at_origin", dropna=False)
    },
    "diagnosticCounts": dict(
        Counter(
            flag for r in records if r["status"] == "evaluated" for flag in r["diagnostic_flags"]
        )
    ),
    "providerMeasurementChangeDiagnostics": measurement_diagnostics,
    "limits": audit["limitations"]
    + [
        "Only one frozen origin is evaluated. Latest fiscal-year cohorts are reported separately and should not be pooled without the date checks.",
        "No unavailable outcome is imputed as zero and no disappearance is called bankruptcy. Complete-case coverage remains subject to outcome attrition even though the full cohort is disclosed.",
        "Instrument directory uses included types 0/1/3/8/9/10; receipts/share classes can repeat economic issuers. The directory is snapshot membership, not proof of continuous trading.",
        "Rows with changed ISIN are withheld pending identity review. Cash-flow patterns and vintage differences are diagnostic flags, not proven causal business explanations.",
    ],
    "sources": audit["rawVintages"],
    "recordsFile": f"{PREFIX}-companies.json",
    "csvFile": f"{PREFIX}-companies.csv",
}
save(f"{PREFIX}-summary.json", summary)
save(
    f"{PREFIX}-companies.json",
    {
        "format": "macro-atlas-frozen-vintage-pilot-companies",
        "origin": ORIGIN,
        "outcomeSnapshot": OUTCOME_AS_OF,
        "companies": records,
    },
)
csv = frame.copy()
for key in [
    "training_years",
    "training_cash_native",
    "training_ratios",
    "training_publication_dates",
    "diagnostic_flags",
]:
    csv[key] = csv[key].map(lambda values: json.dumps(values, separators=(",", ":")))
csv.to_csv(OUT / f"{PREFIX}-companies.csv", index=False)
print(
    json.dumps(
        {
            "phase": "pilot complete",
            "overall": summary["overall"],
            "byLatestTrainingFiscalYear": summary["byLatestTrainingFiscalYear"],
            "diagnosticCounts": summary["diagnosticCounts"],
        },
        indent=2,
    ),
    flush=True,
)
