"""Same-series Swedish monitoring observations, kept separate from hypotheses."""

from __future__ import annotations

import calendar
import hashlib
import math
from copy import deepcopy
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from dalio.assessments.core import content_hash, validate_snapshot

VERSION = "sweden-monitoring-v1"
SPECS = {
    "industrial_production": {
        "label": "Industrial production", "channel": "Industrial activity",
        "frequency": "monthly", "comparison": "three_month_means",
        "scenarios": ["demand_shortfall", "stronger_activity"],
        "limits": ["Mining and manufacturing exclude energy and do not represent the whole economy.",
                   "Three-month industrial momentum is neither an annual GDP forecast error nor an annualised growth rate.",
                   "Calendar and seasonal adjustment, revisions and individual large producers can affect the index."],
    },
    "industrial_orders": {
        "label": "Industrial order intake", "channel": "Industrial activity",
        "frequency": "monthly", "comparison": "three_month_means",
        "scenarios": ["demand_shortfall", "stronger_activity"],
        "limits": ["Order intake covers mining and manufacturing, including domestic and export markets; it is not realised sales.",
                   "Three-month order momentum cannot measure an annual GDP forecast error or demand in a particular company's customer markets.",
                   "Large orders, cancellations and revised seasonal adjustment can affect the comparison."],
    },
    "corporate_new_lending_rate": {
        "label": "Business lending rate: new and renegotiated agreements",
        "channel": "Business financing", "frequency": "monthly", "comparison": "three_month_rate",
        "scenarios": ["funding_strain"],
        "limits": ["The covered new and renegotiated loan agreements are not the rate paid on every outstanding corporate liability.",
                   "Borrower, loan-size and interest-fixation composition can change the aggregate rate even without like-for-like repricing.",
                   "A lower observed rate does not establish easier credit access, lending standards or company-specific financing terms."],
    },
    "policy_rate": {
        "label": "Riksbank effective policy rate", "channel": "Monetary conditions",
        "frequency": "daily", "comparison": "ninety_day_rate", "scenarios": ["funding_strain"],
        "limits": ["The policy rate is not a corporate borrowing rate or a credit-availability measure.",
                   "Pass-through depends on loan contracts, repricing dates, risk premiums and borrower circumstances."],
    },
    "yield_10y": {
        "label": "Swedish 10-year government benchmark yield", "channel": "Government funding context",
        "frequency": "daily", "comparison": "ninety_day_rate", "scenarios": ["funding_strain"],
        "limits": ["This is a secondary-market government benchmark yield, not an auction borrowing cost or company lending rate.",
                   "A constant-tenor benchmark may roll between securities; the change is not a same-bond return or proof of refinancing stress."],
    },
}
MAX_AGE_DAYS = {"daily": 10, "monthly": 75}
DAILY_ANCHOR_TOLERANCE_DAYS = 7
OBSERVED_STATUSES = {"observed", "published_statistic", "preliminary", "revised", "final", "estimated"}


def _utc(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Monitoring cutoff and source clocks need an explicit timezone")
    return result.astimezone(UTC)


def _month_offset(period: str, offset: int) -> str:
    year, month = map(int, period.split("-"))
    number = year * 12 + month - 1 + offset
    return f"{number // 12:04d}-{number % 12 + 1:02d}"


def _rows(series, *, as_of, known_at):
    if _utc(series["available_at"]) > known_at:
        raise ValueError("Monitoring source is after the known-at cutoff")
    for clock in ("retrieved_at", "published_at", "publisher_updated_at"):
        if series.get(clock) is not None and _utc(series[clock]) > known_at:
            raise ValueError("Monitoring source clock is after the cutoff")
    rows, seen = [], set()
    for original in series["observations"]:
        row = deepcopy(original)
        end = date.fromisoformat(row["date"])
        period = row["period"]
        if series["frequency"] == "monthly":
            year, month = map(int, period.split("-"))
            if period != f"{year:04d}-{month:02d}" or end != date(
                year, month, calendar.monthrange(year, month)[1]
            ):
                raise ValueError("Monthly monitoring observation is not its native period end")
        elif series["frequency"] == "daily" and period != end.isoformat():
            raise ValueError("Daily monitoring observation period mismatch")
        if period in seen:
            raise ValueError("Duplicate monitoring period")
        seen.add(period)
        if row["value"] is not None:
            if (isinstance(row["value"], bool) or not isinstance(row["value"], (int, float))
                    or not math.isfinite(row["value"])):
                raise ValueError("Monitoring values must be finite numbers or explicit nulls")
            if row["status"] not in OBSERVED_STATUSES:
                raise ValueError("Monitoring calculations require observed source values, not forecasts")
        if end <= as_of:
            rows.append(row)
    return sorted(rows, key=lambda row: row["date"])


def _cited(series, row, citations):
    ref = f"monitor:{content_hash(series)}:{row['date']}"
    point = {**deepcopy(row), "evidence_ref": ref, "unit": series["unit"]}
    record = {
        **point, "country": "SE", "indicator": series["indicator"],
        **{key: deepcopy(series.get(key)) for key in (
            "source", "series_id", "source_url", "available_at", "retrieved_at", "published_at", "publisher_updated_at",
            "definition", "adjustment", "publisher_metadata", "artifacts", "bundle_sha256")},
    }
    if ref in citations and citations[ref] != record:
        raise ValueError("Monitoring citation identity collision")
    citations[ref] = record
    return point


def _comparison(series, rows, spec, citations):
    latest = rows[-1]
    method = spec["comparison"]
    if method == "three_month_means":
        adjustment = series.get("adjustment", series.get("publisher_metadata", {}).get("adjustment"))
        if adjustment not in {"seasonally_adjusted", "calendar_and_seasonally_adjusted"}:
            return None, ["Short-term index comparison requires verified seasonal adjustment."]
        periods = [_month_offset(latest["period"], shift) for shift in range(-5, 1)]
        by_period = {row["period"]: row for row in rows}
        missing = [period for period in periods
                   if period not in by_period or by_period[period]["value"] is None]
        if missing:
            return None, ["Missing comparison months: " + ", ".join(missing) + "."]
        selected = [by_period[period] for period in periods]
        if any(row["value"] <= 0 for row in selected):
            return None, ["Volume-index momentum needs positive values in both complete windows."]
        prior_mean = math.fsum(row["value"] for row in selected[:3]) / 3
        current_mean = math.fsum(row["value"] for row in selected[3:]) / 3
        result = {"method": method, "value": (current_mean / prior_mean - 1) * 100,
                  "unit": "%", "prior_mean": prior_mean, "current_mean": current_mean,
                  "prior_periods": periods[:3], "current_periods": periods[3:],
                  "window": f"Mean {periods[3]}–{periods[5]} versus {periods[0]}–{periods[2]}"}
    elif method == "three_month_rate":
        target = _month_offset(latest["period"], -3)
        anchor = next((row for row in rows if row["period"] == target and row["value"] is not None), None)
        if anchor is None:
            return None, [f"No observed lending rate in the exact comparison month {target}."]
        selected = [anchor, latest]
        result = {"method": method, "value": latest["value"] - anchor["value"],
                  "unit": "percentage points", "anchor_date": anchor["date"],
                  "window": f"{latest['period']} versus {target}"}
    else:
        target = date.fromisoformat(latest["date"]) - timedelta(days=90)
        eligible = [row for row in rows if row["value"] is not None
                    and 0 <= (target - date.fromisoformat(row["date"])).days <= DAILY_ANCHOR_TOLERANCE_DAYS]
        if not eligible:
            return None, [f"No observed rate at or within {DAILY_ANCHOR_TOLERANCE_DAYS} days before the 90-day anchor {target}."]
        anchor = eligible[-1]
        selected = [anchor, latest]
        result = {"method": method, "value": latest["value"] - anchor["value"],
                  "unit": "percentage points", "anchor_date": anchor["date"],
                  "target_anchor_date": target.isoformat(),
                  "anchor_lag_days": (target - date.fromisoformat(anchor["date"])).days,
                  "window": f"{latest['date']} versus {anchor['date']} (90-day target {target})"}
    result["evidence_refs"] = [_cited(series, row, citations)["evidence_ref"] for row in selected]
    return result, []


def _reading(key, direction):
    if direction == "unavailable":
        return "No current directional reading: a fresh, comparable observation window is unavailable."
    if key in {"industrial_production", "industrial_orders"}:
        noun = "production" if key == "industrial_production" else "order intake"
        description = {"up": "higher", "down": "lower", "unchanged": "unchanged"}[direction]
        return (f"Average industrial {noun} is {description} over the specified three-month comparison. "
                "This describes the covered industrial activity; it does not establish whether annual GDP will miss or beat the IMF path.")
    description = {"up": "higher", "down": "lower", "unchanged": "unchanged"}[direction]
    scope = {"corporate_new_lending_rate": "rate on the covered new and renegotiated business loan agreements",
             "policy_rate": "effective policy rate", "yield_10y": "10-year government benchmark yield"}[key]
    return (f"The {scope} is {description} compared with the stated comparison point. "
            "This is one financing-price observation; credit access, borrower exposure and the cause of the change need separate evidence.")


def build_monitoring(evidence: dict, assessment: dict, *, as_of: date) -> dict:
    if type(as_of) is not date:
        raise ValueError("Monitoring as_of must be a date")
    if evidence.get("evidence_digest") != content_hash(
        {key: value for key, value in evidence.items() if key != "evidence_digest"}
    ):
        raise ValueError("Monitoring evidence hash mismatch")
    known_at = _utc(evidence["as_known_at"])
    if evidence.get("country") != "SE" or as_of > known_at.date():
        raise ValueError("Monitoring country or assessment date/cutoff is invalid")
    validate_snapshot(assessment)
    if (_utc(assessment["as_known_at"]) != known_at or assessment["as_of"] != as_of.isoformat()
            or len(assessment["countries"]) != 1 or assessment["countries"][0]["country"] != "SE"):
        raise ValueError("Parent Sweden assessment must use the same date and exact cutoff")
    parent = assessment["countries"][0]
    scenarios = deepcopy(parent["scenarios"])
    scenario_ids = {item["id"] for item in scenarios}
    by_key = {}
    for series in evidence["series"]:
        key = series["indicator"]
        if key not in SPECS or key in by_key:
            raise ValueError("Unknown or duplicate monitoring indicator")
        if series["frequency"] != SPECS[key]["frequency"]:
            raise ValueError("Monitoring frequency does not match the declared comparison rule")
        if SPECS[key]["comparison"] != "three_month_means" and series["unit"] not in {
            "percent", "%", "% per annum"
        }:
            raise ValueError("Monitoring rate unit must be percent before a percentage-point comparison")
        by_key[key] = series
    citations = deepcopy(assessment["citations"])
    signals = []
    for key, spec in SPECS.items():
        series = by_key.get(key)
        item = {"indicator": key, "label": spec["label"], "channel": spec["channel"],
                "frequency": spec["frequency"], "definition": series.get("definition", "") if series else "",
                "status": "unavailable", "latest": None, "comparison": None,
                "direction": "unavailable", "age_days": None, "evidence_refs": [],
                "freshness_limit_days": MAX_AGE_DAYS[spec["frequency"]],
                "limits": list(spec["limits"]), "gaps": [], "scenario_links": []}
        if series:
            item["source"] = {k: deepcopy(series.get(k)) for k in (
                "source", "source_url", "series_id", "available_at", "retrieved_at", "published_at", "publisher_updated_at",
                "unit", "adjustment", "publisher_metadata", "artifacts")}
            rows = _rows(series, as_of=as_of, known_at=known_at)
            if rows:
                latest = _cited(series, rows[-1], citations)
                item["latest"] = latest
                item["evidence_refs"].append(latest["evidence_ref"])
                item["age_days"] = (as_of - date.fromisoformat(latest["date"])).days
                if latest["value"] is None:
                    item["status"] = "missing_latest"
                    item["gaps"].append(f"The latest native period {latest['period']} has no reported value; it is not backfilled.")
                elif item["age_days"] > item["freshness_limit_days"]:
                    item["status"] = "stale"
                    item["gaps"].append("The latest observation is outside this pilot's editorial freshness window.")
                else:
                    comparison, gaps = _comparison(series, rows, spec, citations)
                    item["comparison"], item["gaps"] = comparison, gaps
                    item["status"] = "available" if comparison else "no_comparison"
                    if comparison:
                        item["evidence_refs"] = list(dict.fromkeys(
                            [*item["evidence_refs"], *comparison["evidence_refs"]]))
                        delta = comparison["value"]
                        item["direction"] = "up" if delta > 1e-9 else "down" if delta < -1e-9 else "unchanged"
            else:
                item["gaps"].append("No source observation is at or before the assessment date.")
        else:
            item["gaps"].append("No eligible original-response-bound input was selected for this signal.")
        item["reading"] = _reading(key, item["direction"])
        for scenario_id in spec["scenarios"]:
            if scenario_id in scenario_ids:
                scenario = next(s for s in scenarios if s["id"] == scenario_id)
                item["scenario_links"].append({
                    "scenario_id": scenario_id, "title": scenario["title"],
                    "interpretation": item["reading"],
                    "evidence_that_challenges_assumption": deepcopy(scenario["invalidators"]),
                    "reference_evidence_refs": deepcopy(scenario["evidence_refs"]),
                })
        signals.append(item)
    native = deepcopy(evidence.get("national_debt_context", []))
    if native != parent.get("national_debt_context", []):
        raise ValueError("Monitoring and parent assessment national debt evidence differ")
    protected = sorted(set(evidence.get("protected_artifact_paths", []))
                       | set(assessment.get("protected_artifact_paths", [])))
    methodology = {
        "version": VERSION, "time_basis": "UTC assessment date and exact UTC known-at cutoff.",
        "comparison_rules": deepcopy(SPECS), "max_age_days": MAX_AGE_DAYS,
        "daily_anchor_tolerance_days": DAILY_ANCHOR_TOLERANCE_DAYS,
        "freshness_policy": "Editorial applicability windows, not publisher expiry dates or predictive-confidence estimates.",
        "index_policy": "Ratio of two adjacent complete three-calendar-month arithmetic means of positive seasonally adjusted volume indices; no interpolation or annualisation.",
        "rate_policy": "Percentage-point changes: exact month three months earlier, or latest observation at/before the 90-calendar-day daily target within seven days.",
        "vintage_policy": "Comparisons within a retained source vintage are historical changes, not evidence of what was known at the earlier observation date.",
        "revision_policy": "No prior compatible monitoring capture has been selected for a change-since-last-capture comparison in this pilot.",
        "scope_policy": "No aggregate risk score, scenario probability, causal conclusion or company verdict. Industrial signals are not whole-economy GDP forecasts.",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    renderer = Path(__file__).with_name("render.py")
    methodology["renderer_sha256"] = hashlib.sha256(renderer.read_bytes()).hexdigest() if renderer.exists() else None
    methodology["sha256"] = content_hash(methodology)
    result = {
        "schema_version": 1, "country": "SE", "name": "Sweden", "as_of": as_of.isoformat(),
        "as_known_at": known_at.isoformat(), "methodology": methodology,
        "signals": signals, "scenarios": scenarios, "national_debt_context": native,
        "coverage": {"signals_expected": len(SPECS), "signals_source_bound": len(by_key),
                     "signals_with_latest_value": sum(s["latest"] is not None and s["latest"]["value"] is not None for s in signals),
                     "signals_with_comparison": sum(s["comparison"] is not None for s in signals)},
        "remaining_gaps": [
            "Bank lending standards, credit availability, company defaults and individual refinancing terms are not measured by these five signals.",
            "Government auctions and actual funding execution are not inferred from secondary-market benchmark yields or annual funding forecasts.",
            "Current energy prices/mix, household demand, investment, services and company/customer geography remain outside this fixed pilot.",
            "Historical source vintages can be revised. This first capture does not establish a forecast revision or a change since an earlier monitoring capture.",
        ],
        "input_gaps": deepcopy(evidence.get("gaps", [])), "citations": citations,
        "source_evidence": deepcopy(evidence), "country_assessment": deepcopy(assessment),
        "protected_artifact_paths": protected,
    }
    result["snapshot_sha256"] = content_hash(result)
    validate_monitoring(result)
    return result


def validate_monitoring(snapshot: dict) -> None:
    if snapshot.get("snapshot_sha256") != content_hash(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    ):
        raise ValueError("Monitoring snapshot hash mismatch")
    refs = snapshot["citations"]
    scenario_ids = {s["id"] for s in snapshot["scenarios"]}
    for signal in snapshot["signals"]:
        if not set(signal["evidence_refs"]) <= refs.keys():
            raise ValueError("Monitoring signal has unresolved citations")
        if signal["latest"] and signal["latest"]["evidence_ref"] not in refs:
            raise ValueError("Monitoring latest value has an unresolved citation")
        if signal["comparison"] and not set(signal["comparison"]["evidence_refs"]) <= refs.keys():
            raise ValueError("Monitoring comparison has unresolved citations")
        for link in signal["scenario_links"]:
            if link["scenario_id"] not in scenario_ids or not set(link["reference_evidence_refs"]) <= refs.keys():
                raise ValueError("Monitoring scenario link is unresolved")
