"""Descriptive country paths and conditional research cases from verified evidence.

This consumer never changes source facts, scoring populations or probabilities.
The cases are research hypotheses, not forecasts attributed to the publishers.
"""

from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from datetime import UTC, date, datetime
from pathlib import Path

VERSION = "country-assessments-v1"
CORE = (
    "real_gdp_growth", "gov_debt_pct_gdp", "fiscal_balance_pct_gdp",
    "primary_balance_pct_gdp", "current_account_pct_gdp",
)
STRUCTURAL = ("old_age_dependency", "energy_net_imports_pct", "rd_pct_gdp")
LABELS = {
    "real_gdp_growth": "Real GDP growth",
    "gov_debt_pct_gdp": "General-government gross debt",
    "fiscal_balance_pct_gdp": "General-government fiscal balance",
    "primary_balance_pct_gdp": "General-government primary balance (Fiscal Monitor)",
    "current_account_pct_gdp": "Current-account balance",
    "old_age_dependency": "Old-age population ratio",
    "energy_net_imports_pct": "Net energy imports",
    "rd_pct_gdp": "Research and development expenditure",
}
UNITS = {
    **{key: "% of GDP" for key in CORE if key != "real_gdp_growth"},
    "real_gdp_growth": "% annual real growth",
    "old_age_dependency": "people aged 65+ per 100 people aged 15–64",
    "energy_net_imports_pct": "% of primary energy use",
    "rd_pct_gdp": "% of GDP",
}
STRUCTURAL_MAX_AGE_YEARS = 3
REFERENCES = [
    {"title": "IMF WEO definitions, estimates and revisions",
     "url": "https://www.imf.org/en/publications/weo/frequently-asked-questions"},
    {"title": "IMF: interpreting current-account deficits",
     "url": "https://www.imf.org/en/publications/fandd/issues/series/back-to-basics/current-account-deficits"},
    {"title": "ECB: monetary-policy transmission",
     "url": "https://www.ecb.europa.eu/mopo/intro/transmission/html/index.en.html"},
    {"title": "World Bank: old-age dependency definition",
     "url": "https://databank.worldbank.org/metadataglossary/world-development-indicators/series/SP.POP.DPND.OL"},
    {"title": "World Bank: net energy imports definition",
     "url": "https://databank.worldbank.org/metadataglossary/world-development-indicators/series/EG.IMP.CONS.ZS"},
]


def content_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False,
    ).encode()).hexdigest()


def _aware(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Assessment known-at and source clocks require a timezone")
    return result.astimezone(UTC)


def _point(series, year, *, country, known_at, citations, projection=False):
    if series is None:
        return None
    matches = [row for row in series["observations"] if row["year"] == year]
    if len(matches) > 1:
        raise ValueError("Country evidence contains duplicate annual cells")
    if not matches:
        return None
    row = matches[0]
    if projection != (row["status"] == "forecast_calendar_convention"):
        return None
    if _aware(series["available_at"]) > known_at:
        raise ValueError("Country evidence is unavailable at the known-at cutoff")
    if (isinstance(row["value"], bool) or not isinstance(row["value"], (int, float))
            or not math.isfinite(row["value"])):
        raise ValueError("Country evidence requires finite numerical cells")
    if row["date"] != f"{year:04d}-12-31":
        raise ValueError("Country evidence annual identity mismatch")
    metric = series["indicator"]
    ref = f"r{series['release_id']}:{country}:{metric}:{year}:{row['source']}"
    point = {"value": float(row["value"]), "year": year, "unit": UNITS[metric],
             "status": row["status"], "evidence_ref": ref, "source": row["source"],
             "source_url": series["source_url"], "release_id": series["release_id"],
             "series_id": series["series_id"], "available_at": series["available_at"]}
    citation = {**point, "country": country, "indicator": metric,
                "partition_key": series["partition_key"],
                "retrieved_at": series["retrieved_at"],
                "publisher_metadata": deepcopy(series["publisher_metadata"]),
                "artifacts": deepcopy(series["artifacts"])}
    if ref in citations and citations[ref] != citation:
        raise ValueError("Conflicting country evidence reference")
    citations[ref] = citation
    return point


def _refs(*points):
    return list(dict.fromkeys(p["evidence_ref"] for p in points if p is not None))


def _movement(delta):
    return "higher" if delta > 1e-9 else "lower" if delta < -1e-9 else "unchanged"


def _finding(identity, title, text, *points, limits=()):
    return {"id": identity, "title": title, "kind": "derived_interpretation",
            "text": text, "evidence_refs": _refs(*points), "limits": list(limits)}


def _findings(baseline, projections, structural, previous_structural):
    findings = []
    first, last = projections[0]["metrics"], projections[-1]["metrics"]
    growth, current = baseline["real_gdp_growth"], first["real_gdp_growth"]
    if growth and current:
        delta = current["value"] - growth["value"]
        direction = "positive" if current["value"] > 0 else (
            "negative" if current["value"] < 0 else "zero")
        comparison = ("unchanged from" if abs(delta) <= 1e-9 else
                      f"{abs(delta):.2f} percentage points {_movement(delta)} than")
        findings.append(_finding(
            "growth_path", "Published growth path",
            f"The retained IMF path puts {current['year']} real GDP growth at "
            f"{current['value']:.2f}%, {comparison} the {growth['year']} "
            f"estimate/outturn of {growth['value']:.2f}%. "
            f"Projected real growth is {direction}.", growth, current,
            limits=("This compares growth rates, not the change in the level of GDP.",
                    "Historical IMF entries may be estimates and revisions; the forecast is the retained publisher vintage."),
        ))
    debt, endpoint = baseline["gov_debt_pct_gdp"], last["gov_debt_pct_gdp"]
    if debt and endpoint:
        delta = endpoint["value"] - debt["value"]
        findings.append(_finding(
            "debt_path", "Published debt-ratio path",
            f"General-government gross debt is {debt['value']:.2f}% of GDP in the "
            f"{debt['year']} estimate/outturn and {endpoint['value']:.2f}% in the "
            f"{endpoint['year']} IMF projection: {abs(delta):.2f} percentage points "
            f"{_movement(delta)}.", debt, endpoint,
            limits=("The ratio can change through nominal GDP, fiscal flows, valuation and other stock-flow adjustments; it does not measure debt repayments.",
                    "This ratio alone does not establish debt sustainability, refinancing cost or market access."),
        ))
    fiscal, end_fiscal = baseline["fiscal_balance_pct_gdp"], last["fiscal_balance_pct_gdp"]
    if fiscal and end_fiscal:
        delta = end_fiscal["value"] - fiscal["value"]
        findings.append(_finding(
            "fiscal_path", "Published fiscal-balance path",
            f"The fiscal balance moves from {fiscal['value']:.2f}% of GDP in "
            f"{fiscal['year']} to {end_fiscal['value']:.2f}% in the {end_fiscal['year']} "
            f"projection, {abs(delta):.2f} percentage points {_movement(delta)}. "
            "Positive values denote net lending; negative values denote net borrowing.",
            fiscal, end_fiscal,
            limits=("Balance changes can reflect the cycle, one-offs and denominator changes; they do not isolate discretionary fiscal policy.",
                    "Fiscal Monitor primary balances remain a separate dataset; no cross-dataset interest-expense calculation is made."),
        ))
    external = baseline["current_account_pct_gdp"]
    if external:
        sign = "deficit" if external["value"] < 0 else "surplus" if external["value"] > 0 else "balance"
        findings.append(_finding(
            "external_balance", "External saving and investment balance",
            f"The {external['year']} current-account estimate/outturn is "
            f"{external['value']:.2f}% of GDP, a {sign}. This is a saving–investment "
            "balance for the economy.", external,
            limits=("Interpretation needs the composition and durability of financing, investment and external positions.",
                    "A deficit or surplus alone is not a verdict on economic health or company profitability."),
        ))
    ageing = structural["old_age_dependency"]
    if ageing and not ageing["stale"]:
        old = previous_structural.get("old_age_dependency")
        trend = (f" It was {old['value']:.2f} in {old['year']} on the same retained series."
                 if old else " A matching observation five years earlier is unavailable.")
        findings.append(_finding(
            "age_structure", "Historical age structure",
            f"In {ageing['year']}, there were {ageing['value']:.2f} people aged 65+ "
            f"per 100 people aged 15–64.{trend}", ageing, old,
            limits=("This is an age-population ratio, not pensioners per worker. Employment, migration and participation affect economic dependence.",
                    "This historical series does not provide a population forecast or quantify future pension costs."),
        ))
    energy = structural["energy_net_imports_pct"]
    if energy and not energy["stale"]:
        kind = "net energy importer" if energy["value"] > 0 else (
            "net energy exporter" if energy["value"] < 0 else "balanced net energy trader")
        findings.append(_finding(
            "energy_exposure", "Historical energy-trade structure",
            f"The {energy['year']} net energy-import ratio was {energy['value']:.2f}% "
            f"of primary energy use, identifying a {kind} on this energy-volume measure.", energy,
            limits=("The ratio uses energy equivalents rather than trade values and does not establish insulation from energy prices.",
                    "Energy mix, import origins, contractual pricing and company hedges require separate evidence."),
        ))
    research = structural["rd_pct_gdp"]
    if research and not research["stale"]:
        findings.append(_finding(
            "research_input", "Historical research spending",
            f"Research and development expenditure was {research['value']:.2f}% of "
            f"GDP in {research['year']}.", research,
            limits=("Spending is an input, not a measured innovation outcome or productivity forecast.",),
        ))
    return findings


def _case(identity, title, horizon, assumptions, pathway, signposts, invalidators,
          points, company_checks, limitations=()):
    refs = _refs(*points)
    return {
        "id": identity, "title": title, "kind": "conditional_scenario", "horizon": horizon,
        "assumptions": assumptions, "pathway": pathway,
        "signposts": [{"text": "To monitor: " + text, "evidence_refs": refs}
                      for text in signposts],
        "invalidators": invalidators, "evidence_refs": refs,
        "company_checks": company_checks,
        "limitations": [
            "This is an Observatory research hypothesis, with no calibrated probability or numerical stress forecast.",
            "Horizons are editorial research windows; effects and timing require further evidence.",
            "Listing country defines the research directory. Revenue, asset, cost and financing exposures must be verified before a company implication is assigned.",
            *limitations,
        ],
    }


def _scenarios(baseline, projections, structural, national_context=()):
    first = projections[0]["metrics"]
    growth, forecast = baseline["real_gdp_growth"], first["real_gdp_growth"]
    if growth is None or forecast is None:
        return []
    cases = []
    def growth_window(first_year, last_year):
        window = [row for row in projections if first_year <= row["year"] <= last_year]
        points = [row["metrics"]["real_gdp_growth"] for row in window
                  if row["metrics"]["real_gdp_growth"] is not None]
        values = [f"{row['year']}: " + (
            f"{row['metrics']['real_gdp_growth']['value']:.2f}%"
            if row["metrics"]["real_gdp_growth"] else "unavailable") for row in window]
        return "Retained IMF annual real GDP growth references: " + "; ".join(values) + ".", points

    demand_anchor, demand_points = growth_window(forecast["year"], forecast["year"] + 2)
    cases.append(_case(
        "demand_shortfall", "Real activity falls below the published path", "6–24 months",
        [demand_anchor, "Country real activity falls below its retained GDP path; which demand or supply component changes must be established. No specific shock size is assumed."],
        ["If weaker country activity includes weaker demand in a company's actual end markets, sales volumes and capacity use can fall.",
         "Operating leverage, pricing power and cost flexibility determine the effect on profits and investment."],
        ["new GDP releases and forecast revisions against the retained reference path",
         "household consumption, investment, industrial orders and customer demand; these series are not inputs to this first assessment"],
        ["Real activity meets or exceeds the reference path over subsequent releases.",
         "A company's end markets and volumes remain resilient despite weaker aggregate activity."],
        [growth, *demand_points],
        ["Revenue by customer/end-market country, rather than exchange listing.",
         "Customer concentration, order book, demand cyclicality and operating leverage.",
         "Capacity utilisation, pricing contracts and the variable/fixed cost split."],
        ["A country's GDP path is not an external-demand forecast. Foreign customer markets require their own matching evidence.",
         "Annual references only approximate the scenario window; unavailable years are explicit and not interpolated."],
    ))
    debt, fiscal = baseline["gov_debt_pct_gdp"], baseline["fiscal_balance_pct_gdp"]
    if debt and fiscal:
        cases.append(_case(
            "funding_strain", "Financing conditions tighten", "6–36 months",
            [f"The {debt['year']} reference has general-government gross debt of {debt['value']:.2f}% "
             f"of GDP and a fiscal balance of {fiscal['value']:.2f}% of GDP.",
             "New sovereign or private borrowing costs rise, or access to credit tightens; this assessment does not establish that either has occurred."],
            ["Higher marginal borrowing costs affect borrowers as floating rates reset or obligations refinance.",
             "Debt-service demands and tighter lending can constrain investment, spending and customer credit."],
            ["new-issue yields, lending rates, credit spreads and lending standards; not yet included in this country comparison",
             "issuer maturity schedules, fixed/floating rates, currency and liquidity buffers; a gross debt ratio cannot replace these"],
            ["Financing rates and credit availability remain stable or improve.",
             "Long fixed-rate maturities, cash buffers or matching currency cash flows materially weaken the proposed channel."],
            [debt, fiscal],
            ["Debt maturities, fixed/floating split, covenant terms and undrawn facilities.",
             "Debt and operating cash-flow currencies; hedges and refinancing counterparties.",
             "Dependence on public procurement or credit-sensitive customers."],
            ["Central-government instruments and general-government debt ratios have different perimeters.",
             "No debt/GDP path is recomputed from real GDP growth alone; nominal growth, interest and stock-flow assumptions are absent."],
        ))
        funding = cases[-1]
        for fact in national_context:
            if fact["metric"] == "gross_borrowing_requirement" and fact["status"] == "forecast":
                funding["assumptions"].append(
                    f"The original national debt-office plan separately forecasts gross "
                    f"central-government borrowing of {fact['value']:.2f} "
                    f"{'billion SEK' if fact['unit'] == 'SEK_bn' else fact['unit']} "
                    f"for {fact['year']}. This is an annual flow under its own source scope."
                )
                funding["evidence_refs"].append(fact["evidence_ref"])
    stronger_anchor, stronger_points = growth_window(forecast["year"] + 1, forecast["year"] + 3)
    if stronger_points:
        cases.append(_case(
            "stronger_activity", "Activity exceeds the published path", "1–3 years",
            [stronger_anchor, "Country real activity develops more strongly than its retained GDP path; financing and input supply permit the response."],
            ["Where stronger country activity includes stronger demand in a company's actual end markets, real sales and capacity use can rise.",
             "Wages, input costs, competition and interest rates determine how much benefit reaches individual companies."],
            ["GDP and investment revisions above the retained path",
             "orders, real household income, productivity and capacity use; additional national evidence is needed"],
            ["Growth revisions and realised orders fail to improve.",
             "Higher input costs, funding costs or capacity constraints offset stronger demand."],
            [growth, *stronger_points],
            ["Exposure to the end markets showing stronger real demand.",
             "Available capacity, incremental margins, investment needs and competitive responses.",
             "Wage/input sensitivity and ability to pass higher costs through to customers."],
            ["A country's GDP path does not establish demand in a company's foreign customer markets.",
             "Annual reference points are retained only where published; missing years remain unavailable."],
        ))
    energy = structural["energy_net_imports_pct"]
    if energy and not energy["stale"] and energy["value"] > 0:
        cases.append(_case(
            "energy_import_shock", "Imported energy becomes more costly or disrupted", "6–24 months",
            [f"The historical net-import ratio was {energy['value']:.2f}% of primary energy use in {energy['year']}.",
             "Relevant imported energy becomes more expensive or less available, and current energy dependence remains material."],
            ["Energy-intensive production and household purchasing power can face pressure.",
             "The effect varies with fuel mix, local pricing, substitution, hedges and cost pass-through."],
            ["current energy-import quantities, mix, suppliers and applicable local prices; the historical ratio is only a structural starting point",
             "company energy procurement, hedge maturity and production interruptions"],
            ["Supply substitution, lower relevant prices or effective hedges offset the assumed shock.",
             "Current energy dependence differs materially from the historical ratio."],
            [energy],
            ["Plant locations, fuel/power intensity and procurement/hedging contracts.",
             "Ability to substitute fuels or production sites and to pass through costs."],
        ))
    return cases


def _national_context(entry, *, as_of, known_at, citations):
    """Keep original debt-office units and scope outside the harmonized baseline."""
    selected = []
    for original in entry.get("national_debt_context", []):
        fact = deepcopy(original)
        if _aware(fact["available_at"]) > known_at:
            raise ValueError("National debt evidence is after the known-at cutoff")
        end = date.fromisoformat(fact["period_end"])
        if fact["status"] == "observed" and end > as_of:
            continue
        if fact["status"] == "forecast" and not as_of.year <= end.year <= as_of.year + 5:
            continue
        if fact["status"] not in {"observed", "forecast"}:
            raise ValueError("Unexpected national debt context status")
        if (isinstance(fact["value"], bool) or not isinstance(fact["value"], (int, float))
                or not math.isfinite(fact["value"])):
            raise ValueError("National debt context requires finite values")
        ref = fact["evidence_ref"]
        citation = {**fact, "country": entry["country"],
                    "indicator": "national_debt:" + fact["metric"]}
        if ref in citations:
            raise ValueError("Duplicate national debt context reference")
        citations[ref] = citation
        selected.append(fact)
    return selected


def build_snapshot(evidence: dict, *, as_of: date) -> dict:
    """Build all selected countries deterministically; retain source dates and gaps."""
    if type(as_of) is not date:
        raise ValueError("Assessment as_of must be a date")
    if evidence.get("evidence_digest") != content_hash(
        {key: value for key, value in evidence.items() if key != "evidence_digest"}
    ):
        raise ValueError("Country evidence digest mismatch")
    known_at = _aware(evidence["as_known_at"])
    if as_of > known_at.date():
        raise ValueError("Assessment date is after the known-at cutoff")
    baseline_year, end_year = as_of.year - 1, as_of.year + 5
    citations, countries, seen = {}, [], set()
    for entry in evidence["countries"]:
        country = entry["country"]
        if country in seen:
            raise ValueError("Duplicate country assessment")
        seen.add(country)
        series = {}
        for item in entry["series"]:
            key = (item["family"], item["indicator"])
            if key in series:
                raise ValueError("Duplicate source-series assessment identity")
            series[key] = item

        def point(metric, year, *, projection=False, selected_series=series,
                  selected_country=country):
            family = "imf" if metric in CORE else "wb"
            return _point(selected_series.get((family, metric)), year, country=selected_country,
                          known_at=known_at, citations=citations, projection=projection)

        baseline = {key: point(key, baseline_year) for key in CORE}
        projections = [{"year": year, "metrics": {
            key: point(key, year, projection=True) for key in CORE}}
            for year in range(as_of.year, end_year + 1)]
        gaps = [f"{item['family']}/{item['indicator']}: {item['reason']}"
                for item in entry["gaps"]]
        gaps.extend(f"{LABELS[key]}: no verified {baseline_year} estimate/outturn."
                    for key in CORE if baseline[key] is None)
        for row in projections:
            absent = [LABELS[key] for key, value in row["metrics"].items() if value is None]
            if absent:
                gaps.append(f"Missing publisher projections for {row['year']}: {', '.join(absent)}.")
        structural, previous_structural = {}, {}
        for metric in STRUCTURAL:
            item = series.get(("wb", metric))
            years = [r["year"] for r in item["observations"]
                     if r["year"] <= baseline_year and r["status"] == "published_statistic"] if item else []
            latest = point(metric, max(years)) if years else None
            if latest:
                latest["age_years"] = as_of.year - latest["year"]
                latest["stale"] = latest["age_years"] > STRUCTURAL_MAX_AGE_YEARS
                if latest["stale"]:
                    gaps.append(f"{LABELS[metric]} is dated {latest['year']}; it is outside the "
                                f"{STRUCTURAL_MAX_AGE_YEARS}-year structural interpretation window.")
                if metric == "old_age_dependency" and not latest["stale"]:
                    previous_structural[metric] = point(metric, latest["year"] - 5)
            else:
                gaps.append(f"{LABELS[metric]}: no verified historical observation.")
            structural[metric] = latest
        gaps.extend([
            "Country-specific inflation, employment, credit conditions and current funding prices are outside this first comparable assessment input set.",
            "Historical demographic, energy and research inputs are not forecasts; demographic projections and complete issuer maturity schedules remain incomplete.",
            "Sector/customer exposures and company financial/physical-asset geography have not been joined to these country cases.",
        ])
        core_available = sum(p is not None for p in baseline.values())
        projection_available = sum(p is not None for row in projections for p in row["metrics"].values())
        national_context = _national_context(entry, as_of=as_of, known_at=known_at,
                                              citations=citations)
        scenarios = _scenarios(baseline, projections, structural, national_context)
        status = ("insufficient_evidence" if not scenarios else "assessment_available"
                  if core_available == len(CORE) and projection_available == len(CORE) * len(projections)
                  else "partial")
        countries.append({
            **{key: entry[key] for key in ("country", "listing_iso2", "name", "listing_count")},
            "status": status,
            "coverage": {"core_available": core_available, "core_expected": len(CORE),
                         "projection_available": projection_available,
                         "projection_expected": len(CORE) * len(projections),
                         "structural_available": sum(p is not None for p in structural.values()),
                         "structural_expected": len(STRUCTURAL),
                         "structural_within_age_window": sum(p is not None and not p["stale"] for p in structural.values())},
            "baseline": baseline, "projections": projections, "structural": structural,
            "national_debt_context": national_context,
            "findings": _findings(baseline, projections, structural, previous_structural),
            "scenarios": scenarios, "gaps": gaps,
        })
    if not countries:
        raise ValueError("Assessment selection must contain countries")
    methodology = {
        "version": VERSION,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "renderer_sha256": hashlib.sha256(Path(__file__).with_name("render.py").read_bytes()).hexdigest(),
        "baseline": "Prior-calendar-year IMF estimate/outturn from the selected eligible release.",
        "publisher_path": "Current calendar year plus the following five years; retained IMF values labelled forecast by the collector's calendar convention, with gaps preserved.",
        "structural_max_age_years": STRUCTURAL_MAX_AGE_YEARS,
        "structural_window_policy": "Editorial eligibility window, not an official expiry or data-quality rating.",
        "comparison": "Differences are same-indicator, same-source-series arithmetic; no cross-dataset fiscal interest calculation or debt dynamics model.",
        "scenario_policy": "Conditional Observatory hypotheses. No calibrated probabilities, synthetic forecasts, company verdicts or composite risk score.",
        "source_policy": "Prefer first-hand national evidence; retained WB/IMF harmonized baselines preserve original producer/dataset metadata.",
        "national_debt_context": "Available original debt-office facts retain native units, periods, forecast status and central-government scope; they do not replace general-government ratios.",
        "confidence": "Coverage is reported separately from predictive confidence, which is not estimated.",
        "references_checked_on": "2026-09-10", "references": REFERENCES,
        "metrics": {key: {"label": LABELS[key], "unit": UNITS[key]} for key in (*CORE, *STRUCTURAL)},
    }
    methodology["sha256"] = content_hash(methodology)
    snapshot = {
        "schema_version": 1, "as_of": as_of.isoformat(), "as_known_at": known_at.isoformat(),
        "baseline_year": baseline_year, "horizon_end_year": end_year,
        "methodology": methodology, "countries": countries, "citations": citations,
        "source_evidence": deepcopy(evidence),
        "protected_artifact_paths": list(evidence.get("protected_artifact_paths", [])),
    }
    snapshot["snapshot_sha256"] = content_hash(snapshot)
    validate_snapshot(snapshot)
    return snapshot


def validate_snapshot(snapshot: dict) -> None:
    """Reject altered snapshots and unresolved numerical references before publishing."""
    if snapshot.get("snapshot_sha256") != content_hash(
        {key: value for key, value in snapshot.items() if key != "snapshot_sha256"}
    ):
        raise ValueError("Country assessment snapshot hash mismatch")
    refs = snapshot["citations"]
    for country in snapshot["countries"]:
        records = country["findings"] + country["scenarios"]
        for item in records:
            if not item["evidence_refs"] or not set(item["evidence_refs"]) <= refs.keys():
                raise ValueError("Country assessment has unresolved evidence references")
            if any(refs[key]["country"] != country["country"] for key in item["evidence_refs"]):
                raise ValueError("Country assessment cites a different country")
