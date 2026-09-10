"""Deterministic offline report for the five-signal Sweden monitoring pilot."""

from __future__ import annotations

import hashlib
import html

from dalio.assessments.render import IMF_STATUS_CONVENTION, _anchor, _link

ANNUAL_METRICS = (
    ("real_gdp_growth", "Real GDP growth"),
    ("gov_debt_pct_gdp", "General-government debt"),
    ("fiscal_balance_pct_gdp", "Fiscal balance"),
    ("primary_balance_pct_gdp", "Primary balance (Fiscal Monitor)"),
    ("current_account_pct_gdp", "Current account"),
)


def _text(value) -> str:
    result = html.escape(str(value), quote=False).replace("\r", " ").replace("\n", " ")
    for character in ("\\", "`", "*", "_", "[", "]", "|"):
        result = result.replace(character, "\\" + character)
    return result


def _words(value) -> str:
    return _text(str(value).replace("_", " "))


def _publisher(record) -> str:
    source = record.get("source") or "Source not supplied"
    if source in {"SCB", "SCB_MONITORING"}:
        label = "Statistics Sweden (SCB)"
        native_source = (record.get("publisher_metadata") or {}).get("source", "")
        if str(native_source).strip().lower() in {"the riksbank", "sveriges riksbank"}:
            label += ", on behalf of Sveriges Riksbank"
        return _text(label)
    if source == "RIKSBANK_SWEA":
        return "Sveriges Riksbank"
    return _words(source)


def _number(value, *, whole_sek=False) -> str:
    if value is None:
        return "Not available"
    if whole_sek and float(value).is_integer():
        return format(int(value), ",")
    return format(float(value), ",.6g")


def _value(point) -> str:
    if not point or point.get("value") is None:
        return "Not available"
    unit = point.get("unit", "")
    display_unit = {"SEK_bn": "billion SEK"}.get(unit, unit)
    return f"{_number(point['value'], whole_sek=unit == 'SEK')} {_text(display_unit)}".strip()


def _change(comparison) -> str:
    if not comparison:
        return "Not available"
    value = comparison["value"]
    sign = "−" if value < 0 else "+" if value > 0 else ""
    return f"{sign}{_number(abs(value))} {_text(comparison['unit'])}"


def _period(point) -> str:
    if not point:
        return "Not available"
    if point.get("period"):
        return _text(point["period"])
    if point.get("period_start") and point.get("period_end"):
        if point["period_start"] == point["period_end"]:
            return _text(point["period_end"])
        return f"{_text(point['period_start'])} to {_text(point['period_end'])}"
    if point.get("year") is not None:
        return _text(point["year"])
    return _text(point.get("date", "Not available"))


def _refs(values, citations, numbers) -> str:
    result = []
    for ref in dict.fromkeys(values):
        if ref not in citations:
            raise ValueError(f"Unresolved monitoring citation: {ref}")
        result.append(f"[{numbers[ref]}](#{_anchor(ref)})")
    return " ".join(result)


def _freshness(signal) -> str:
    status = _words(signal.get("status", "unavailable")).capitalize()
    age = signal.get("age_days")
    if age is None:
        return status
    day_unit = "day" if age == 1 else "days"
    return (f"{status}; {age} {day_unit} since reference-period end; "
            f"editorial limit {signal['freshness_limit_days']} days")


def _scenario_anchor(identifier) -> str:
    return "scenario-" + hashlib.sha256(str(identifier).encode()).hexdigest()[:16]


def _bullets(lines, label, values) -> None:
    if values:
        lines.extend(["", f"**{label}**", "", *(f"- {_text(value)}" for value in values)])


def _annual_reference(lines, snapshot, citations, numbers) -> None:
    assessment = snapshot.get("country_assessment", {})
    countries = assessment.get("countries", [])
    if not countries:
        return
    country = countries[0]
    lines.extend([
        "", "## Annual IMF reference", "", IMF_STATUS_CONVENTION, "",
        "These annual estimates and paths are the scenario reference. The industrial indices cover "
        "part of the economy; their short-term changes do not measure an annual GDP forecast error. "
        "The fiscal and primary-balance series remain separate WEO and Fiscal Monitor measures.", "",
        "| Year | " + " | ".join(label for _, label in ANNUAL_METRICS) + " | Status label |",
        "|---|" + "---|" * (len(ANNUAL_METRICS) + 1),
    ])
    rows = [(assessment.get("baseline_year", "Baseline"), country.get("baseline", {}))]
    rows.extend((row["year"], row.get("metrics", {})) for row in country.get("projections", []))
    for year, metrics in rows:
        cells, statuses = [], []
        for metric, _ in ANNUAL_METRICS:
            point = metrics.get(metric)
            refs = [point["evidence_ref"]] if point and point.get("evidence_ref") else []
            cells.append(f"{_value(point)} {_refs(refs, citations, numbers)}".strip())
            if point and point.get("status"):
                statuses.append(_words(point["status"]))
        status = "; ".join(dict.fromkeys(statuses)) or "Not available"
        lines.append(f"| {_text(year)} | " + " | ".join(cells) + f" | {status} |")


def _native_tables(lines, snapshot, citations, numbers) -> None:
    native = snapshot.get("national_debt_context", [])
    groups = (
        ("Observed national debt context", [p for p in native if p.get("status") != "forecast"]),
        ("Published national funding forecasts", [p for p in native if p.get("status") == "forecast"]),
    )
    for title, points in groups:
        lines.extend(["", f"## {title}", ""])
        if title.startswith("Observed"):
            lines.append(
                "Original debt-office records retain central-government scope. Average time to refixing "
                "measures interest-rate resets, not principal repayment dates. Reference-date and "
                "monthly-mean values remain separate; detailed derivative and accounting bases are in snapshot.json."
            )
        else:
            lines.append(
                "These are dated funding-plan forecasts, not actual borrowing or auction results. "
                "They retain native currency amounts and must not be combined with general-government GDP ratios."
            )
        lines.append("")
        if not points:
            lines.append("No eligible original debt-office values are available in this snapshot.")
            continue
        lines.extend([
            "| Native metric | Value and unit | Reference period | Source status | Scope / aggregation | Evidence |",
            "|---|---:|---|---|---|---|",
        ])
        for point in points:
            dims = point.get("dimensions", {})
            scope = "; ".join(_words(dims[key]) for key in ("scope", "aggregation", "debt_class") if dims.get(key))
            label = _words(point["metric"]).capitalize()
            refs = _refs([point["evidence_ref"]], citations, numbers)
            lines.append(f"| {label} | {_value(point)} | {_period(point)} | "
                         f"{_words(point.get('status', 'unspecified')).capitalize()} | {scope or 'See source'} | {refs} |")


def _citations(lines, citations, numbers) -> None:
    lines.extend([
        "", "## Evidence and source clocks", "",
        "Each numbered reference identifies a retained source cell. Availability is when the retained "
        "evidence was usable by this system; it is not the economic reference date. A missing publication "
        "timestamp stays unspecified. Source links identify the publisher or documented delivery service. "
        "[snapshot.json](snapshot.json) retains complete source metadata, raw-artifact bindings and precision.", "",
    ])
    for ref in sorted(citations):
        record = citations[ref]
        source = _publisher(record)
        link = _link("Publisher / delivery source", record["source_url"]) if record.get("source_url") else "Source URL not supplied"
        release = (f" Local release: {_text(record['release_id'])}." if record.get("release_id") is not None else "")
        lines.extend([
            f'<a id="{_anchor(ref)}"></a>', "",
            f"**[{numbers[ref]}] {source}** — {link}.{release} Native series: {_text(record.get('series_id', record.get('metric', 'not supplied')))}.", "",
            f"Reference period: {_period(record)}; value: {_value(record)}; "
            f"status: {_words(record.get('status', 'unspecified'))}.", "",
            f"Publication timestamp: {_text(record.get('published_at') or 'not supplied')}; "
            f"available: {_text(record.get('available_at') or 'not supplied')}; "
            f"retrieved: {_text(record.get('retrieved_at') or 'not supplied')}.",
        ])
        if record.get("source_locator"):
            lines.extend(["", f"Native location: {_text(record['source_locator'])}."])
        if record.get("publisher_updated_at"):
            lines.extend(["", f"Dataset updated: {_text(record['publisher_updated_at'])}. "
                          "This update timestamp is not a separately established first-publication time."])
        if record.get("definition"):
            lines.extend(["", f"Definition: {_text(record['definition'])}"])
        if record.get("adjustment"):
            lines.extend(["", f"Adjustment: {_words(record['adjustment'])}."])
        if record.get("bundle_sha256"):
            lines.extend(["", f"Source bundle SHA-256: {_text(record['bundle_sha256'])}."])
        if record.get("artifacts"):
            artifact_rows = []
            for artifact in record["artifacts"]:
                role = artifact.get("role", "source artifact")
                digest = artifact.get("sha256", artifact.get("artifact_sha256", "not supplied"))
                artifact_rows.append(f"{_words(role)}: {_text(digest)}")
            lines.extend(["", "Artifact SHA-256: " + "; ".join(artifact_rows) + "."])
        lines.extend(["", f"Evidence identity: {_text(ref)}.", ""])


def render_monitoring(snapshot: dict) -> str:
    """Render core-owned readings and explicit source scopes without new inference."""
    citations = snapshot.get("citations", {})
    numbers = {ref: number for number, ref in enumerate(sorted(citations), 1)}
    coverage = snapshot.get("coverage", {})
    method = snapshot.get("methodology", {})
    signals = snapshot.get("signals", [])
    lines = [
        "# Sweden — scenario monitoring", "",
        f"Assessment date (UTC): {_text(snapshot['as_of'])}. Exact known-at cutoff (UTC): {_text(snapshot['as_known_at'])}.", "",
        f"Original-response-bound signals: {coverage.get('signals_source_bound', 0)}/{coverage.get('signals_expected', 5)}; "
        f"signals with a comparable window: {coverage.get('signals_with_comparison', 0)}/{coverage.get('signals_expected', 5)}. "
        "These are coverage counts, not confidence scores.", "",
        "[Complete offline snapshot](snapshot.json). Display values use up to six significant digits; "
        "whole-SEK amounts are grouped without rounding. The snapshot preserves full precision.", "",
        "## At a glance", "",
    ]
    for signal in signals:
        latest, comparison = signal.get("latest"), signal.get("comparison")
        detail = f"Latest {_period(latest)}: {_value(latest)}."
        if comparison:
            detail += f" Change: {_change(comparison)}; {_text(comparison['window'])}."
        refs = _refs(signal.get("evidence_refs", []), citations, numbers)
        lines.append(f"- **{_text(signal['label'])}.** {_text(signal['reading'])} {detail} {refs}".rstrip())
    lines.extend([
        "", "## Observed signals", "",
        "Changes compare observations within the retained source vintage. They are not changes since "
        "an earlier monitoring capture. Rate changes are percentage points; index changes are percentages "
        "between complete three-month means, without annualisation.", "",
        "| Signal | Latest period | Value and unit | Comparison window | Change | Freshness | Evidence |",
        "|---|---|---:|---|---:|---|---|",
    ])
    for signal in signals:
        comparison = signal.get("comparison")
        refs = _refs(signal.get("evidence_refs", []), citations, numbers)
        window = _text(comparison["window"]) if comparison else "Not available"
        lines.append(f"| {_text(signal['label'])} | {_period(signal.get('latest'))} | {_value(signal.get('latest'))} | "
                     f"{window} | {_change(comparison)} | {_freshness(signal)} | {refs or '—'} |")
    lines.extend(["", _text(method.get("freshness_policy", "Freshness limits are editorial applicability windows."))])
    lines.extend(["", "## Definitions, limits and scenario checks", "",
                  "Observed changes inform research questions. They do not mechanically confirm a hypothesis, "
                  "assign a probability or establish the cause of the change."])
    for signal in signals:
        lines.extend(["", f"### {_text(signal['label'])}", ""])
        if signal.get("definition"):
            lines.append(_text(signal["definition"]))
        source = signal.get("source", {})
        if source:
            lines.extend(["", f"Source: {_publisher(source)}; "
                          f"native series: {_text(source.get('series_id') or 'not supplied')}; "
                          f"available: {_text(source.get('available_at') or 'not supplied')}."])
            if source.get("publisher_updated_at"):
                lines.extend(["", f"Dataset updated: {_text(source['publisher_updated_at'])}."])
        _bullets(lines, "Interpretation limits", signal.get("limits", []))
        _bullets(lines, "Missing or inapplicable inputs", signal.get("gaps", []))
        for link in signal.get("scenario_links", []):
            lines.extend(["", f"Related scenario hypothesis: [{_text(link['title'])}](#{_scenario_anchor(link['scenario_id'])})."])
            _bullets(lines, "Evidence that would challenge its assumptions", link.get("evidence_that_challenges_assumption", []))
            refs = _refs(link.get("reference_evidence_refs", []), citations, numbers)
            if refs:
                lines.extend(["", f"Scenario reference evidence: {refs}."])
    _native_tables(lines, snapshot, citations, numbers)
    _annual_reference(lines, snapshot, citations, numbers)
    lines.extend(["", "## Conditional scenario hypotheses", "",
                  "The existing country cases supply the hypotheses below. Monitoring observations do not "
                  "turn them into calibrated forecasts or company conclusions."])
    for scenario in snapshot.get("scenarios", []):
        lines.extend(["", f'<a id="{_scenario_anchor(scenario['id'])}"></a>', "",
                      f"### {_text(scenario['title'])}", "",
                      f"**Scenario hypothesis.** Horizon: {_text(scenario.get('horizon', 'not supplied'))}."])
        _bullets(lines, "Assumptions", scenario.get("assumptions", []))
        _bullets(lines, "Conditional transmission", scenario.get("pathway", []))
        _bullets(lines, "Required company evidence", scenario.get("company_checks", []))
        _bullets(lines, "Limitations", scenario.get("limitations", []))
        refs = _refs(scenario.get("evidence_refs", []), citations, numbers)
        if refs:
            lines.extend(["", f"Reference evidence: {refs}."])
    lines.extend(["", "## Uncollected and unresolved evidence", "",
                  "Outside this pilot does not mean absent from the project database. Reuse or collection "
                  "requires a suitable definition and verified source lineage."])
    _bullets(lines, "Remaining scope gaps", snapshot.get("remaining_gaps", []))
    for gap in snapshot.get("input_gaps", []):
        if isinstance(gap, dict):
            label = gap.get("indicator", gap.get("country", "Input"))
            reason = gap.get("reason", gap.get("error", "Unspecified gap"))
            lines.extend(["", f"- {_words(label)}: {_words(reason)}."])
        else:
            lines.extend(["", f"- {_text(gap)}"])
    lines.extend(["", "## Reproducibility", "",
                  f"Method: {_text(method.get('version', 'unspecified'))}. "
                  f"Monitoring snapshot SHA-256: {_text(snapshot.get('snapshot_sha256', 'not supplied'))}.", "",
                  f"Parent country-assessment SHA-256: {_text(snapshot.get('country_assessment', {}).get('snapshot_sha256', 'not supplied'))}."])
    for key in ("time_basis", "index_policy", "rate_policy", "vintage_policy", "revision_policy", "scope_policy"):
        if method.get(key):
            lines.extend(["", _text(method[key])])
    _citations(lines, citations, numbers)
    return "\n".join(lines).rstrip() + "\n"
