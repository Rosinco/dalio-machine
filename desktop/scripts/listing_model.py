"""Identity-only company catalogue; no financial inference or issuer deduplication."""

from __future__ import annotations

import math
import re
from datetime import date

COMPANY_TYPES = (0, 1, 3, 8, 9, 10)
COUNTRIES = {
    "Sverige": ("SE", "Sweden"),
    "Norge": ("NO", "Norway"),
    "Finland": ("FI", "Finland"),
    "Danmark": ("DK", "Denmark"),
    "USA": ("US", "United States"),
    "Kanada": ("CA", "Canada"),
    "England": ("GB", "United Kingdom"),
    "Tyskland": ("DE", "Germany"),
    "Frankrike": ("FR", "France"),
    "Spanien": ("ES", "Spain"),
    "Portugal": ("PT", "Portugal"),
    "Italien": ("IT", "Italy"),
    "Schweiz": ("CH", "Switzerland"),
    "Belgien": ("BE", "Belgium"),
    "Nederländerna": ("NL", "Netherlands"),
    "Polen": ("PL", "Poland"),
    "Estland": ("EE", "Estonia"),
    "Lettland": ("LV", "Latvia"),
    "Litauen": ("LT", "Lithuania"),
}


def optional_id(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, bool) or int(value) != float(value) or not 0 < int(value) < 10**10:
        raise ValueError("Invalid source identifier")
    return str(int(value))


def select_listings(snapshots):
    latest, dates = {}, set()
    for stamp, rows in sorted(snapshots):
        if date.fromisoformat(stamp).isoformat() != stamp or stamp in dates:
            raise ValueError("Duplicate or invalid snapshot date")
        dates.add(stamp)
        seen = set()
        for row in rows:
            key = optional_id(row["ins_id"])
            if key is None or key in seen:
                raise ValueError("Duplicate or missing instrument ID in snapshot")
            seen.add(key)
            latest[key] = {**row, "source_as_of": stamp}
    return {
        k: v
        for k, v in sorted(latest.items(), key=lambda x: int(x[0]))
        if v["instrument_type"] in COMPANY_TYPES
    }


def listing_record(row, countries):
    cid = optional_id(row["country_id"])
    country = countries.get(cid, {})
    stamp = row.get("listing_date")
    return {
        "id": optional_id(row["ins_id"]),
        "name": row.get("name"),
        "ticker": row.get("ticker"),
        "isin": row.get("isin"),
        "country_id": cid,
        "listing_country": country.get("iso2"),
        "sector_id": optional_id(row.get("sector_id")),
        "branch_id": optional_id(row.get("branch_id")),
        "instrument_type": int(row["instrument_type"]),
        "source_as_of": row["source_as_of"],
        "listing_date": str(stamp)[:10] if stamp is not None else None,
        "stock_currency": row.get("stock_price_currency"),
        "report_currency": row.get("report_currency"),
    }


def classify_listing(key, sector, branch, correction, branches):
    target = branch if branch in branches else None
    status = (
        "unclassified"
        if target is None
        else "source"
        if branches[target]["sector_id"] == sector
        else "sector_mismatch"
    )
    if correction is not None:
        for field in ("company_id", "expected_sector_id", "expected_branch_id", "branch_id"):
            if not isinstance(correction.get(field), str) or not re.fullmatch(
                r"[1-9][0-9]{0,9}", correction[field]
            ):
                raise ValueError("Invalid reviewed classification identifier")
        if correction["company_id"] != key or correction["branch_id"] not in branches:
            raise ValueError("Correction belongs to an unknown listing or target branch")
        if any(
            not isinstance(correction.get(f), str) or not correction[f].strip()
            for f in ("reason", "source")
        ):
            raise ValueError("A correction needs a reason and evidence source")
        stamp = correction.get("reviewed_at", "")
        if (
            not re.fullmatch(r"20\d\d-\d\d-\d\d", stamp)
            or date.fromisoformat(stamp).isoformat() != stamp
        ):
            raise ValueError("Invalid classification review date")
        if branch == correction["branch_id"] and sector == branches[branch]["sector_id"]:
            status = "aligned"
        elif (sector, branch) == (
            correction["expected_sector_id"],
            correction["expected_branch_id"],
        ):
            status, target = "corrected", correction["branch_id"]
        else:
            status = "needs_review"
    return {
        "company_id": key,
        "source_sector_id": sector,
        "source_branch_id": branch,
        "sector_id": branches[target]["sector_id"] if target else None,
        "branch_id": target,
        "status": status,
        "correction": correction,
    }
