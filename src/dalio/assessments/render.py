"""Deterministic Markdown views of evidence-bound country assessments."""

from __future__ import annotations

import hashlib
import html
import re
from urllib.parse import quote, urlsplit

IMF_STATUS_CONVENTION = (
    "IMF DataMapper does not supply a native per-point status. "
    "Prior-year rows are labelled estimate/outturn; current and future years "
    "are labelled forecast using the documented calendar convention."
)


def _md(value) -> str:
    return html.escape(str(value), quote=False).replace("|", "\\|").replace("\n", " ")


def country_code(country: dict) -> str:
    code = country["country"]
    if not isinstance(code, str) or not re.fullmatch(r"[A-Z]{2}", code):
        raise ValueError("Country code is unsafe for an assessment filename")
    return "UK" if code == "GB" else code


def _link(label: str, url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc or any(ord(x) < 32 for x in url):
        raise ValueError("Unsupported source citation URL")
    encoded = quote(url, safe=":/?#[]@!$&'()*+,;=%~._-")
    return f"[{_md(label)}](<{encoded}>)"


def _anchor(ref: str) -> str:
    return "evidence-" + hashlib.sha256(ref.encode()).hexdigest()[:16]


def _citation(ref: str, snapshot: dict) -> dict:
    record = snapshot.get("citations", {}).get(ref)
    if not isinstance(record, dict):
        raise ValueError(f"Unknown assessment evidence reference: {ref}")
    return record


def _source(ref: str, snapshot: dict, prefix: str = "") -> str:
    record = _citation(ref, snapshot)
    label = f"{record.get('source', 'Source')} · local release {record.get('release_id', 'unavailable')}"
    return f"[{_md(label)}]({prefix}#{_anchor(ref)})"


def _references(refs, snapshot, prefix="") -> str:
    return "; ".join(_source(ref, snapshot, prefix) for ref in dict.fromkeys(refs))


def _all_refs(value) -> set[str]:
    refs = set()
    if isinstance(value, dict):
        if value.get("evidence_ref"):
            refs.add(value["evidence_ref"])
        refs.update(value.get("evidence_refs", []))
        for child in value.values():
            refs.update(_all_refs(child))
    elif isinstance(value, list):
        for child in value:
            refs.update(_all_refs(child))
    return refs


def _metric(metric: str, snapshot: dict | None = None) -> str:
    if snapshot:
        label = snapshot.get("methodology", {}).get("metrics", {}).get(metric, {}).get("label")
        if label:
            return _md(label)
    words = [
        word.upper() if word.lower() in {"gdp", "cpi", "ppp", "fx", "usd", "sek"} else word
        for word in metric.replace("_", " ").split()
    ]
    if words and not words[0].isupper():
        words[0] = words[0].capitalize()
    return _md(" ".join(words))


def _status(point) -> str:
    return _md(str(point.get("status", "unspecified")).replace("_", " ").capitalize())


def _value(point) -> str:
    if not point or point.get("value") is None:
        return "Not available"
    value = point["value"]
    # Six significant digits keep small nonzero values visible. The JSON retains
    # exact source precision and the original unit; no unit conversion is implied.
    unit = point.get("unit", "")
    if unit == "SEK" and float(value).is_integer():
        number = format(int(value), ",")
    else:
        number = format(value, ",.6g")
    display_unit = {"SEK_bn": "billion SEK"}.get(unit, unit)
    result = f"{number} {_md(display_unit)}".strip()
    if point.get("stale"):
        age = point.get("age_years")
        result += f"; stale{f' ({age} years old)' if age is not None else ''}"
    return result


def _point_table(points: dict, snapshot: dict) -> list[str]:
    lines = [
        "| Metric | Value and unit | Source year | Status label | Evidence |",
        "|---|---:|---:|---|---|",
    ]
    for metric, point in sorted(points.items()):
        if not point:
            lines.append(f"| {_metric(metric, snapshot)} | Not available | — | Not available | — |")
            continue
        refs = [point["evidence_ref"]] if point.get("evidence_ref") else []
        lines.append(
            f"| {_metric(metric, snapshot)} | {_value(point)} | {_md(point.get('year', '—'))} | {_status(point)} | {_references(refs, snapshot) or '—'} |"
        )
    return lines


def _bullet_section(lines: list[str], title: str, values) -> None:
    if values:
        lines.extend(["", f"**{title}**", "", *(f"- {_md(value)}" for value in values)])


def render_country(country: dict, snapshot: dict) -> str:
    code = country_code(country)
    coverage = country.get("coverage", {})
    lines = [
        f"# {_md(country['name'])} ({code})",
        "",
        "[All countries](../index.md)",
        "",
        f"Assessment date: {_md(snapshot['as_of'])}. Evidence available by: {_md(snapshot['as_known_at'])}.",
        "",
        f"Baseline year: {_md(snapshot['baseline_year'])}. Outlook through {_md(snapshot['horizon_end_year'])}. Method: {_md(snapshot.get('methodology', {}).get('version', 'unspecified'))}.",
        "",
        f"Core baseline coverage: {_md(coverage.get('core_available', 0))}/{_md(coverage.get('core_expected', 0))}. Status: {_md(country.get('status', 'unspecified'))}. Downloaded listings: {_md(country.get('listing_count', 0))}; listing country: {_md(country.get('listing_iso2', code))}.",
        "",
        "## At a glance",
        "",
    ]
    for finding in country.get("findings", [])[:3]:
        refs = _references(finding.get("evidence_refs", []), snapshot)
        lines.append(f"- {_md(finding['text'])}{f' Evidence: {refs}.' if refs else ''}")
    if not country.get("findings"):
        lines.append(
            "No findings are available for this snapshot. Coverage and remaining gaps are listed below."
        )
    lines.extend(
        [
            "",
            "## Source baseline",
            "",
            IMF_STATUS_CONVENTION,
            "",
            "Display values normally use up to six significant digits; whole-SEK amounts are grouped without rounding. snapshot.json retains full precision.",
            "",
            *_point_table(country.get("baseline", {}), snapshot),
            "",
            "## Published projections",
            "",
            "IMF values retain their source year. Current-year and future-year forecast labels follow the calendar convention above; they are not native per-point status flags.",
            "",
        ]
    )
    for projection in country.get("projections", []):
        lines.extend(
            [
                f"### {_md(projection['year'])}",
                "",
                *_point_table(projection.get("metrics", {}), snapshot),
                "",
            ]
        )
    if country.get("national_debt_context"):
        lines.extend(
            [
                "## Original national debt-office evidence",
                "",
                "These issuer records retain their native units, periods, accounting scope and aggregation. Central-government amounts and refixing statistics are shown separately from the harmonized general-government baseline.",
                "",
                "Average time to refixing measures the timing of interest-rate resets; it does not measure principal repayment dates. Monthly-mean and reference-date ATR are presented separately, with their respective derivative and accounting bases retained in snapshot.json.",
                "",
                "| Native metric | Value and unit | Reference period | Source status | Scope and aggregation | Evidence |",
                "|---|---:|---|---|---|---|",
            ]
        )
        for point in country["national_debt_context"]:
            dimensions = point.get("dimensions", {})
            period = f"{point.get('period_start', 'unavailable')} to {point.get('period_end', 'unavailable')}"
            scope = "; ".join(
                str(dimensions[key]).replace("_", " ")
                for key in ("scope", "aggregation")
                if dimensions.get(key)
            )
            label = _metric(point["metric"], snapshot)
            native_label = point.get("native_label")
            if native_label and native_label.lower() != point["metric"].replace("_", " ").lower():
                label += " — " + _md(native_label)
            lines.append(
                f"| {label} | {_value(point)} | {_md(period)} | {_status(point)} | {_md(scope) or 'See source'} | {_references([point['evidence_ref']], snapshot)} |"
            )
        lines.append("")
    if country.get("structural"):
        lines.extend(
            ["## Structural evidence", "", *_point_table(country["structural"], snapshot), ""]
        )
    lines.extend(["## Findings", ""])
    if not country.get("findings"):
        lines.extend(["No evidence-supported finding is available for this snapshot.", ""])
    for finding in country.get("findings", []):
        lines.extend(
            [
                f"### {_md(finding['title'])}",
                "",
                f"Kind: {_md(finding.get('kind', 'unspecified'))}.",
                "",
                _md(finding["text"]),
            ]
        )
        refs = _references(finding.get("evidence_refs", []), snapshot)
        if refs:
            lines.extend(["", f"Evidence: {refs}."])
        _bullet_section(lines, "Limits", finding.get("limits", []))
        lines.append("")
    lines.extend(
        [
            "## Conditional scenarios",
            "",
            "Scenario hypotheses describe conditional pathways. Probabilities are not assigned.",
            "",
        ]
    )
    for scenario in country.get("scenarios", []):
        lines.extend(
            [
                f"### {_md(scenario['title'])}",
                "",
                f"**Scenario hypothesis.** Horizon: {_md(scenario.get('horizon', 'unspecified'))}.",
            ]
        )
        _bullet_section(lines, "Assumptions", scenario.get("assumptions", []))
        pathway = scenario.get("pathway", [])
        if pathway:
            lines.extend(
                [
                    "",
                    "**Conditional pathway**",
                    "",
                    *(f"{i}. {_md(step)}" for i, step in enumerate(pathway, 1)),
                ]
            )
        if scenario.get("signposts"):
            lines.extend(["", "**Signposts to monitor**", ""])
            for signpost in scenario["signposts"]:
                text = signpost["text"]
                if not text.lower().startswith("to monitor"):
                    text = "To monitor: " + text
                refs = _references(signpost.get("evidence_refs", []), snapshot)
                lines.append(f"- {_md(text)}{f' Evidence: {refs}.' if refs else ''}")
        _bullet_section(lines, "Invalidators", scenario.get("invalidators", []))
        _bullet_section(lines, "Company evidence to verify", scenario.get("company_checks", []))
        _bullet_section(lines, "Limitations", scenario.get("limitations", []))
        refs = _references(scenario.get("evidence_refs", []), snapshot)
        if refs:
            lines.extend(["", f"Scenario inputs: {refs}."])
        lines.append("")
    _bullet_section(lines, "Remaining data gaps", country.get("gaps", []))
    lines.extend(
        [
            "",
            "## Evidence and release provenance",
            "",
            "The release numbers identify immutable local database releases. Source links open the original publisher; [snapshot.json](../snapshot.json) contains the complete evidence records.",
            "",
        ]
    )
    for ref in sorted(_all_refs(country)):
        record = _citation(ref, snapshot)
        source = record.get("source", "Source")
        link = _link(source, record["source_url"]) if record.get("source_url") else _md(source)
        period = (
            f"source year {record['year']}"
            if record.get("year") is not None
            else f"period {record.get('period_start', 'unavailable')} to {record.get('period_end', 'unavailable')}"
        )
        identity = (
            f"series `{_md(record['series_id'])}`"
            if record.get("series_id")
            else f"native metric `{_md(record.get('metric', 'unavailable'))}`"
        )
        locator = (
            f" Source location: {_md(record['source_locator'])}."
            if record.get("source_locator")
            else ""
        )
        lines.extend(
            [
                f'<a id="{_anchor(ref)}"></a>',
                "",
                f"- {link} — local release **{_md(record.get('release_id', 'unavailable'))}**; {identity}; {_md(period)}; status {_status(record)}; available {_md(record.get('available_at', 'unavailable'))}. Evidence reference: `{_md(ref)}`.{locator}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_index(snapshot: dict) -> str:
    countries = sorted(snapshot["countries"], key=lambda country: country_code(country))
    lines = [
        "# Country macro assessments",
        "",
        f"Assessment date: {_md(snapshot['as_of'])}. Evidence available by: {_md(snapshot['as_known_at'])}.",
        "",
        f"Baseline: {_md(snapshot['baseline_year'])}. Published outlook through {_md(snapshot['horizon_end_year'])}. Method: {_md(snapshot.get('methodology', {}).get('version', 'unspecified'))}.",
        "",
        "[Complete machine-readable snapshot](snapshot.json). Each country report contains source-linked findings, conditional scenarios, company checks and data gaps.",
        "",
        "## Coverage",
        "",
        "| Country | Listings | Core baseline coverage | Status |",
        "|---|---:|---:|---|",
    ]
    for country in countries:
        code = country_code(country)
        coverage = country.get("coverage", {})
        lines.append(
            f"| [{_md(country['name'])} ({code})](countries/{code}.md) | {_md(country.get('listing_count', 0))} | {_md(coverage.get('core_available', 0))}/{_md(coverage.get('core_expected', 0))} | {_md(country.get('status', 'unspecified'))} |"
        )
    available_metrics = sorted(
        {metric for country in countries for metric in country.get("baseline", {})}
    )
    preferred = []
    for fragment in ("growth", "inflation", "debt", "current_account"):
        preferred.extend(
            metric for metric in available_metrics if fragment in metric and metric not in preferred
        )
    metrics = (preferred + [metric for metric in available_metrics if metric not in preferred])[:4]
    lines.extend(
        [
            "",
            "## Comparable baseline view",
            "",
            IMF_STATUS_CONVENTION,
            "",
            "Cells retain their source year and unit. Missing values remain unavailable; company exposure must be checked against actual operations and balance sheets.",
            "",
        ]
    )
    if metrics:
        lines.extend(
            [
                "| Country | " + " | ".join(_metric(metric, snapshot) for metric in metrics) + " |",
                "|---|" + "---|" * len(metrics),
            ]
        )
        for country in countries:
            code = country_code(country)
            cells = []
            for metric in metrics:
                point = country.get("baseline", {}).get(metric)
                if not point:
                    cells.append("Not available")
                    continue
                text = f"{_value(point)} ({_md(point.get('year', '—'))}; {_status(point)})"
                if point.get("evidence_ref"):
                    _citation(point["evidence_ref"], snapshot)
                    text = f"[{text}](countries/{code}.md#{_anchor(point['evidence_ref'])})"
                cells.append(text)
            lines.append(
                f"| [{_md(country['name'])}](countries/{code}.md) | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines).rstrip() + "\n"
