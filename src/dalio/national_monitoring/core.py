"""Dated native signals connected to three countries' conditional scenarios."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from datetime import date, timedelta
from pathlib import Path

from dalio.assessments.core import content_hash
from dalio.assessments.core import validate_snapshot as validate_assessment
from dalio.monitoring.acquisition import utc
from dalio.monitoring.core import MAX_AGE_DAYS, _month_offset, _rows
from dalio.national_monitoring.acquisition import INDICATORS, catalogue

COUNTRIES = ("US", "DE", "CA")
VERSION = "national-monitoring-v1"


def _cite(series, row, citations):
    ref = f"national:{content_hash(series)}:{row['date']}"
    point = {**deepcopy(row), "evidence_ref": ref, "unit": series["unit"]}
    record = {**point, **{k: deepcopy(series.get(k)) for k in (
        "country", "indicator", "source", "series_id", "source_url", "definition", "adjustment",
        "available_at", "retrieved_at", "published_at", "publisher_updated_at", "publisher_metadata",
        "artifacts", "bundle_sha256")}}
    if ref in citations and citations[ref] != record:
        raise ValueError("National citation identity collision")
    citations[ref] = record
    return point


def _compare(series, rows, spec, citations):
    latest, method = rows[-1], spec["comparison"]
    if method == "three_month_means":
        if series.get("adjustment") not in {"seasonally_adjusted", "calendar_and_seasonally_adjusted"}:
            return None, ["Short-term index comparison requires verified seasonal adjustment."]
        periods = [_month_offset(latest["period"], i) for i in range(-5, 1)]
        lookup = {r["period"]: r for r in rows}
        missing = [p for p in periods if p not in lookup or lookup[p]["value"] is None]
        if missing:
            return None, ["Missing comparison months: " + ", ".join(missing)]
        selected = [lookup[p] for p in periods]
        if any(r["value"] <= 0 for r in selected):
            return None, ["Industrial volume momentum requires positive values in both complete windows."]
        prior = math.fsum(r["value"] for r in selected[:3]) / 3
        current = math.fsum(r["value"] for r in selected[3:]) / 3
        result = {"method": method, "value": (current / prior - 1) * 100, "unit": "%",
                  "prior_mean": prior, "current_mean": current,
                  "prior_periods": periods[:3], "current_periods": periods[3:],
                  "window": f"Mean {periods[3]}–{periods[5]} versus {periods[0]}–{periods[2]}"}
    elif method == "three_month_rate":
        target = _month_offset(latest["period"], -3)
        anchor = next((r for r in rows if r["period"] == target and r["value"] is not None), None)
        if anchor is None:
            return None, [f"No observed rate in exact comparison month {target}."]
        selected = [anchor, latest]
        result = {"method": method, "value": latest["value"] - anchor["value"],
                  "unit": "percentage points", "anchor_date": anchor["date"],
                  "window": f"{latest['period']} versus {target}"}
    else:
        target = date.fromisoformat(latest["date"]) - timedelta(days=90)
        eligible = [r for r in rows if r["value"] is not None
                    and 0 <= (target - date.fromisoformat(r["date"])).days <= 7]
        if not eligible:
            return None, [f"No observed rate within seven days at/before 90-day target {target}."]
        anchor = eligible[-1]
        selected = [anchor, latest]
        result = {"method": method, "value": latest["value"] - anchor["value"],
                  "unit": "percentage points", "anchor_date": anchor["date"],
                  "target_anchor_date": target.isoformat(),
                  "anchor_lag_days": (target - date.fromisoformat(anchor["date"])).days,
                  "window": f"{latest['date']} versus {anchor['date']} (90-day target {target})"}
    result["evidence_refs"] = [_cite(series, r, citations)["evidence_ref"] for r in selected]
    return result, []


def _signal(series, spec, scenarios, citations, *, as_of, cutoff):
    frequency, key = spec["frequency"], spec["indicator"]
    method = spec["comparison"]
    if frequency not in MAX_AGE_DAYS or method not in {"three_month_means", "three_month_rate", "ninety_day_rate"}:
        raise ValueError("Unknown National native frequency/comparison contract")
    if (method == "ninety_day_rate") != (frequency == "daily"):
        raise ValueError("National comparison must preserve its native frequency")
    item = {"indicator": key, "label": spec["label"], "frequency": frequency,
            "channel": "Industrial activity" if key.startswith("industrial_") else "Financing conditions",
            "definition": spec.get("definition", ""), "limits": list(spec.get("limits", [])),
            "status": "unavailable", "latest": None, "comparison": None, "direction": "unavailable",
            "age_days": None, "freshness_limit_days": MAX_AGE_DAYS[frequency],
            "evidence_refs": [], "gaps": [], "scenario_links": []}
    if series is not None:
        if (method == "three_month_means" and series["unit"] != "index" and not series["unit"].startswith("index, ")
                and spec.get("measure_kind") != "real_industrial_value_added_volume"):
            raise ValueError("Non-index industrial volume requires an explicit native measure kind")
        if series["frequency"] != frequency or ("unit" in spec and series["unit"] != spec["unit"]):
            raise ValueError("National series unit/frequency differs from native contract")
        if method != "three_month_means" and series["unit"] not in {"percent", "per cent", "%", "% per annum"}:
            raise ValueError("National rate unit must be percent for percentage-point changes")
        item["definition"] = series["definition"]
        item["source"] = {k: deepcopy(series.get(k)) for k in (
            "source", "source_url", "series_id", "available_at", "retrieved_at", "published_at",
            "publisher_updated_at", "unit", "adjustment", "publisher_metadata", "artifacts")}
        rows = _rows(series, as_of=as_of, known_at=cutoff)
        if rows:
            latest = _cite(series, rows[-1], citations)
            item["latest"] = latest
            item["evidence_refs"] = [latest["evidence_ref"]]
            item["age_days"] = (as_of - date.fromisoformat(latest["date"])).days
            if latest["value"] is None:
                item["status"] = "missing_latest"
                item["gaps"] = ["Latest native period is missing; no older-value backfill."]
            elif item["age_days"] > item["freshness_limit_days"]:
                item["status"] = "stale"
                item["gaps"] = ["Outside this pilot's editorial freshness window; no current directional reading."]
            else:
                item["comparison"], item["gaps"] = _compare(series, rows, spec, citations)
                item["status"] = "available" if item["comparison"] else "no_comparison"
                if item["comparison"]:
                    delta = item["comparison"]["value"]
                    item["direction"] = "up" if delta > 1e-9 else "down" if delta < -1e-9 else "unchanged"
                    item["evidence_refs"] = list(dict.fromkeys(
                        item["evidence_refs"] + item["comparison"]["evidence_refs"]))
        else:
            item["gaps"] = ["No native observation at or before the assessment date."]
    else:
        item["gaps"] = ["No eligible original-response-bound input for this country/signal."]
    description = {"up": "higher", "down": "lower", "unchanged": "unchanged"}
    item["reading"] = ("No current directional reading: a complete fresh comparison is unavailable."
                       if item["direction"] == "unavailable" else
                       f"The covered {'industrial volume average' if method == 'three_month_means' else 'rate'} is "
                       f"{description[item['direction']]} over the stated comparison window. " +
                       ("This does not establish whether annual GDP will miss or beat the IMF path."
                        if method == "three_month_means" else
                        "Credit access, borrower exposure and the cause of the change require separate evidence."))
    ids = {"demand_shortfall", "stronger_activity"} if key.startswith("industrial_") else {"funding_strain"}
    for scenario in scenarios:
        if scenario["id"] in ids:
            item["scenario_links"].append({"scenario_id": scenario["id"], "title": scenario["title"],
                "interpretation": item["reading"], "reference_evidence_refs": scenario["evidence_refs"],
                "evidence_that_challenges_assumption": scenario["invalidators"]})
    return item


def build_snapshot(national: dict, assessment: dict, *, as_of: date) -> dict:
    """Build a fixed US/DE/CA panel without changing earlier supplement meanings."""
    if type(as_of) is not date:
        raise ValueError("National as_of must be a date")
    if national.get("evidence_digest") != content_hash({
            k: v for k, v in national.items() if k != "evidence_digest"}):
        raise ValueError("National evidence hash mismatch")
    cutoff = utc(national["as_known_at"])
    validate_assessment(assessment)
    if (utc(assessment["as_known_at"]) != cutoff or as_of > cutoff.date()
            or assessment["as_of"] != as_of.isoformat()):
        raise ValueError("National inputs must use the same exact cutoff and UTC assessment date")
    parents = {c["country"]: c for c in assessment["countries"]}
    if len(assessment["countries"]) != len(COUNTRIES) or set(parents) != set(COUNTRIES):
        raise ValueError("National comparison requires the three original country identities")
    expected = {(c, key) for c in COUNTRIES for key in INDICATORS}
    by_key = {}
    for original in national["series"]:
        identity = original["country"], original["indicator"]
        if identity not in expected:
            raise ValueError("Unknown National country/signal identity")
        if identity in by_key:
            raise ValueError("Duplicate National country/signal identity")
        if any(original["spec"].get(k) != original[k] for k in ("country", "indicator")):
            raise ValueError("National source specification identity mismatch")
        by_key[identity] = original
    specs = {(s["country"], s["indicator"]): s for s in catalogue(as_of)}
    if set(specs) != expected:
        raise ValueError("National catalogue identity mismatch")
    gaps = {}
    for gap in national.get("gaps", []):
        identity = gap["country"], gap["indicator"]
        if identity not in expected or identity in gaps or identity in by_key:
            raise ValueError("Unknown, duplicate or contradictory National gap identity")
        gaps[identity] = gap
    citations, countries = deepcopy(assessment["citations"]), []
    for code in COUNTRIES:
        parent, signals = parents[code], []
        for key in INDICATORS:
            series = by_key.get((code, key))
            spec = series["spec"] if series is not None else specs[code, key]
            signal = _signal(series, spec, parent["scenarios"], citations, as_of=as_of, cutoff=cutoff)
            gap = gaps.get((code, key))
            if gap:
                if gap.get("error"):
                    signal["gaps"].append(gap["error"])
                signal["source_gap"] = deepcopy(gap)
            signals.append(signal)
        countries.append({"country": code, "name": parent["name"], "signals": signals,
            "scenarios": deepcopy(parent["scenarios"]),
            "national_debt_context": deepcopy(parent.get("national_debt_context", [])),
            "coverage": {"signals_expected": len(INDICATORS),
                         "signals_source_bound": sum((code, key) in by_key for key in INDICATORS),
                         "signals_with_comparison": sum(s["comparison"] is not None for s in signals)}})
    package = Path(__file__).parents[1]
    methodology = {"version": VERSION, "common_indicators": list(INDICATORS),
        "comparison_rules": "Two adjacent complete three-month means of positive adjusted industrial volume measures (%); exact three-month rate change or daily 90-day target anchored at/before target within seven days (percentage points).",
        "freshness": MAX_AGE_DAYS, "freshness_policy": "Editorial applicability, not predictive confidence.",
        "scope_policy": "Native definitions, industrial output versus real value added, currencies, loan populations and policy instruments differ. No country ranking, composite score, causal claim, forecast revision, scenario probability or company verdict.",
        "vintage_policy": "Same-vintage historical changes; no claim that today's revised history was known at prior reference dates. No gap backfill from older captures or substituted proxy rates.",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).with_name("render.py").read_bytes()).hexdigest(),
        "shared_implementation_sha256": {
            relative: hashlib.sha256((package / relative).read_bytes()).hexdigest()
            for relative in ("monitoring/core.py", "monitoring/render.py", "assessments/render.py")}}
    result = {"schema_version": 1, "as_of": as_of.isoformat(), "as_known_at": cutoff.isoformat(),
        "methodology": methodology, "countries": countries, "citations": citations,
        "country_assessment": deepcopy(assessment), "source_evidence": {"national": deepcopy(national)},
        "input_gaps": deepcopy(national.get("gaps", [])),
        "remaining_gaps": ["Lending standards, credit access, defaults and actual government funding execution need separate evidence.",
                           "Services, household demand, energy markets and company/customer exposures are outside this fixed panel.",
                           "Listing country does not establish revenue, asset, cost or financing exposure."],
        "protected_artifact_paths": sorted({p for source in (national, assessment)
                                            for p in source.get("protected_artifact_paths", [])})}
    result["snapshot_sha256"] = content_hash(result)
    validate_snapshot(result)
    return result


def validate_snapshot(snapshot: dict) -> None:
    if snapshot.get("snapshot_sha256") != content_hash({k: v for k, v in snapshot.items() if k != "snapshot_sha256"}):
        raise ValueError("National snapshot hash mismatch")
    countries = snapshot["countries"]
    if len(countries) != len(COUNTRIES) or {c["country"] for c in countries} != set(COUNTRIES):
        raise ValueError("National country identity mismatch")
    refs = snapshot["citations"]
    for country in countries:
        if len(country["signals"]) != len(INDICATORS) or {s["indicator"] for s in country["signals"]} != set(INDICATORS):
            raise ValueError("National signal identity mismatch")
        ids = {s["id"] for s in country["scenarios"]}
        for signal in country["signals"]:
            used = list(signal["evidence_refs"])
            point = signal["latest"]
            if point and (point["evidence_ref"] not in refs or
                          any(refs[point["evidence_ref"]].get(k) != v for k, v in point.items())):
                raise ValueError("National latest point differs from its native citation")
            if point:
                used.append(point["evidence_ref"])
            if signal["comparison"]:
                used.extend(signal["comparison"]["evidence_refs"])
            for ref in used:
                if ref not in refs or refs[ref]["country"] != country["country"] or refs[ref]["indicator"] != signal["indicator"]:
                    raise ValueError("Unresolved or wrong-country National citation")
            for link in signal["scenario_links"]:
                if link["scenario_id"] not in ids or any(
                    ref not in refs or refs[ref]["country"] != country["country"]
                    for ref in link["reference_evidence_refs"]
                ):
                    raise ValueError("Unresolved National country/scenario citation")
