"""Dated native signals connected to four countries' conditional scenarios."""

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
from dalio.monitoring.core import SPECS as SE_SPECS
from dalio.nordic_monitoring.acquisition import INDICATORS, catalogue

COUNTRIES = ("SE", "NO", "DK", "FI")
VERSION = "nordic-monitoring-v1"


def _cite(series, row, citations):
    ref = f"nordic:{content_hash(series)}:{row['date']}"
    point = {**deepcopy(row), "evidence_ref": ref, "unit": series["unit"]}
    record = {**point, **{k: deepcopy(series.get(k)) for k in (
        "country", "indicator", "source", "series_id", "source_url", "definition", "adjustment",
        "available_at", "retrieved_at", "published_at", "publisher_updated_at", "publisher_metadata",
        "artifacts", "bundle_sha256")}}
    if ref in citations and citations[ref] != record:
        raise ValueError("Nordic citation identity collision")
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
            return None, ["Volume-index momentum requires positive values in both complete windows."]
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
        raise ValueError("Unknown Nordic native frequency/comparison contract")
    if (method == "ninety_day_rate") != (frequency == "daily"):
        raise ValueError("Nordic comparison must preserve its native frequency")
    item = {"indicator": key, "label": spec["label"], "frequency": frequency,
            "channel": "Industrial activity" if key.startswith("industrial_") else "Financing conditions",
            "definition": spec.get("definition", ""), "limits": list(spec.get("limits", [])),
            "status": "unavailable", "latest": None, "comparison": None, "direction": "unavailable",
            "age_days": None, "freshness_limit_days": MAX_AGE_DAYS[frequency],
            "evidence_refs": [], "gaps": [], "scenario_links": []}
    if series is not None:
        if series["frequency"] != frequency or ("unit" in spec and series["unit"] != spec["unit"]):
            raise ValueError("Nordic series unit/frequency differs from native contract")
        if method != "three_month_means" and series["unit"] not in {"percent", "per cent", "%", "% per annum"}:
            raise ValueError("Nordic rate unit must be percent for percentage-point changes")
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
                       f"The covered {'industrial index average' if method == 'three_month_means' else 'rate'} is "
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


def build_snapshot(nordic: dict, sweden: dict, assessment: dict, *, as_of: date) -> dict:
    if type(as_of) is not date:
        raise ValueError("Nordic as_of must be a date")
    for evidence in (nordic, sweden):
        if evidence.get("evidence_digest") != content_hash({k: v for k, v in evidence.items() if k != "evidence_digest"}):
            raise ValueError("Nordic/Sweden evidence hash mismatch")
    cutoff = utc(nordic["as_known_at"])
    validate_assessment(assessment)
    if (utc(sweden["as_known_at"]) != cutoff or utc(assessment["as_known_at"]) != cutoff
            or as_of > cutoff.date() or assessment["as_of"] != as_of.isoformat()):
        raise ValueError("Nordic inputs must use the same exact cutoff and UTC assessment date")
    parents = {c["country"]: c for c in assessment["countries"]}
    if len(assessment["countries"]) != 4 or set(parents) != set(COUNTRIES) or sweden.get("country") != "SE":
        raise ValueError("Nordic comparison requires the four original country identities")
    expected = {(c, key) for c in COUNTRIES[1:] for key in INDICATORS}
    by_key = {}
    for original in nordic["series"]:
        identity = original["country"], original["indicator"]
        if identity not in expected:
            raise ValueError("Unknown Nordic country/signal identity")
        if identity in by_key:
            raise ValueError("Duplicate Nordic country/signal identity")
        if any(original["spec"].get(k) != original[k] for k in ("country", "indicator")):
            raise ValueError("Nordic source specification identity mismatch")
        by_key[identity] = original
    for original in sweden["series"]:
        key = original["indicator"]
        if key not in SE_SPECS or ("SE", key) in by_key or original.get("country", "SE") != "SE":
            raise ValueError("Unknown or duplicate Sweden source identity")
        by_key["SE", key] = {**original, "country": "SE"}
    specs = {(s["country"], s["indicator"]): s for s in catalogue(as_of)}
    citations, countries = deepcopy(assessment["citations"]), []
    for code in COUNTRIES:
        parent = parents[code]
        keys = list(SE_SPECS) if code == "SE" else list(INDICATORS)
        signals = []
        for key in keys:
            series = by_key.get((code, key))
            spec = ({**SE_SPECS[key], "indicator": key} if code == "SE" else
                    series["spec"] if series is not None else specs[code, key])
            signals.append(_signal(series, spec, parent["scenarios"], citations, as_of=as_of, cutoff=cutoff))
        countries.append({"country": code, "name": parent["name"], "signals": signals,
            "scenarios": deepcopy(parent["scenarios"]),
            "national_debt_context": deepcopy(parent.get("national_debt_context", [])),
            "coverage": {"signals_expected": len(keys),
                         "signals_source_bound": sum((code, key) in by_key for key in keys),
                         "signals_with_comparison": sum(s["comparison"] is not None for s in signals)}})
    if sweden.get("national_debt_context", []) != parents["SE"].get("national_debt_context", []):
        raise ValueError("Sweden national context differs from parent assessment")
    methodology = {"version": VERSION, "common_indicators": list(INDICATORS),
        "comparison_rules": "Two adjacent complete three-month means of positive adjusted indices (%); exact three-month rate change or daily 90-day target anchored at/before target within seven days (percentage points).",
        "freshness": MAX_AGE_DAYS, "freshness_policy": "Editorial applicability, not predictive confidence.",
        "scope_policy": "Native definitions, currencies, industrial coverage, loan agreements and policy instruments differ. No country ranking, composite score, causal claim, forecast revision, scenario probability or company verdict.",
        "vintage_policy": "Same-vintage historical changes; no claim that today's revised history was known at prior reference dates. Sweden retains its earlier eligible capture with its original clock.",
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    renderer = Path(__file__).with_name("render.py")
    methodology["renderer_sha256"] = hashlib.sha256(renderer.read_bytes()).hexdigest() if renderer.exists() else None
    package = Path(__file__).parents[1]
    methodology["shared_implementation_sha256"] = {
        relative: hashlib.sha256((package / relative).read_bytes()).hexdigest()
        for relative in ("monitoring/core.py", "monitoring/render.py", "assessments/render.py")}
    protected = sorted({p for source in (nordic, sweden, assessment) for p in source.get("protected_artifact_paths", [])})
    result = {"schema_version": 1, "as_of": as_of.isoformat(), "as_known_at": cutoff.isoformat(),
        "methodology": methodology, "countries": countries, "citations": citations,
        "country_assessment": deepcopy(assessment), "source_evidence": {"nordic": nordic, "sweden": sweden},
        "input_gaps": [*deepcopy(nordic.get("gaps", [])),
                       *[{"country": "SE", **g} for g in sweden.get("gaps", [])]],
        "remaining_gaps": ["Lending standards, credit access, defaults and actual government funding execution need separate evidence.",
                           "Services, household demand, energy markets and company/customer exposures are outside this fixed panel.",
                           "Listing country does not establish revenue, asset, cost or financing exposure."],
        "protected_artifact_paths": protected}
    result["snapshot_sha256"] = content_hash(result)
    validate_snapshot(result)
    return result


def validate_snapshot(snapshot: dict) -> None:
    if snapshot.get("snapshot_sha256") != content_hash({k: v for k, v in snapshot.items() if k != "snapshot_sha256"}):
        raise ValueError("Nordic snapshot hash mismatch")
    refs = snapshot["citations"]
    for country in snapshot["countries"]:
        ids = {s["id"] for s in country["scenarios"]}
        for signal in country["signals"]:
            if not set(signal["evidence_refs"]) <= refs.keys():
                raise ValueError("Unresolved Nordic signal citation")
            point = signal["latest"]
            if point and (point["evidence_ref"] not in refs or
                          any(refs[point["evidence_ref"]].get(k) != v for k, v in point.items()) or
                          refs[point["evidence_ref"]]["country"] != country["country"]):
                raise ValueError("Nordic latest point differs from its native citation")
            if signal["comparison"] and not set(signal["comparison"]["evidence_refs"]) <= refs.keys():
                raise ValueError("Unresolved Nordic comparison citation")
            for link in signal["scenario_links"]:
                if link["scenario_id"] not in ids or not set(link["reference_evidence_refs"]) <= refs.keys():
                    raise ValueError("Unresolved Nordic country/scenario citation")
