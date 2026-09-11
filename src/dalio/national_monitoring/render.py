"""Offline National comparison retaining native definitions and evidence clocks."""

from __future__ import annotations

from dalio.assessments.render import _anchor
from dalio.monitoring.render import (
    _bullets,
    _change,
    _citations,
    _freshness,
    _native_tables,
    _period,
    _publisher,
    _text,
    _value,
    _words,
)
from dalio.national_monitoring.core import COUNTRIES, validate_snapshot

COMMON = (
    ("industrial_production", "Industrial production"),
    ("corporate_new_lending_rate", "Corporate lending rate"),
    ("policy_rate", "Applicable policy instrument"),
    ("yield_10y", "10-year government reference yield"),
)


def _ordered(snapshot):
    countries = {row["country"]: row for row in snapshot["countries"]}
    if len(countries) != len(snapshot["countries"]) or set(countries) != set(COUNTRIES):
        raise ValueError("National report requires three distinct country identities")
    return [countries[code] for code in COUNTRIES]


def _source_name(record) -> str:
    source = record.get("source") or "Source not supplied"
    if source in {"SCB", "SCB_MONITORING", "RIKSBANK_SWEA"}:
        return _publisher(record)
    metadata = record.get("publisher_metadata") or {}
    for key in ("producer", "publisher"):
        label = metadata.get(key)
        if isinstance(label, str) and label.strip():
            return _text(label)
    return _words(source)


def _refs(refs, snapshot, numbers, *, country, prefix="") -> str:
    result = []
    for ref in dict.fromkeys(refs):
        record = snapshot["citations"].get(ref)
        if not record or record.get("country") != country:
            raise ValueError("Unresolved or wrong-country National citation")
        result.append(f"[{numbers[ref]}]({prefix}#{_anchor(ref)})")
    return " ".join(result)


def _numbers(snapshot):
    return {ref: i for i, ref in enumerate(sorted(snapshot["citations"]), 1)}


def _signal(country, indicator):
    return next((s for s in country["signals"] if s["indicator"] == indicator), None)


def _all_signal_refs(signal):
    refs = set(signal.get("evidence_refs", []))
    if signal.get("latest"):
        refs.add(signal["latest"]["evidence_ref"])
    if signal.get("comparison"):
        refs.update(signal["comparison"]["evidence_refs"])
    for link in signal.get("scenario_links", []):
        refs.update(link.get("reference_evidence_refs", []))
    return refs


def _capture(source) -> str:
    return f"{_source_name(source)}; available {_text(source.get('available_at') or 'not supplied')}"


def _common_intro(snapshot) -> list[str]:
    return [
        f"Assessment date (UTC): {_text(snapshot['as_of'])}. Exact known-at cutoff (UTC): {_text(snapshot['as_known_at'])}.",
        "",
        "Native definitions, currencies, industrial coverage, loan populations and policy instruments "
        "differ. This is a comparison of dated observations, with no country ranking or composite score. "
        "Industrial output indices and real industrial value added differ; levels with different units, bases "
        "or coverage are not directly comparable. Rates do not measure credit access.",
        "",
        "Changes describe one retained source vintage. Each row names its actual comparison window. "
        "They are not forecast errors, revisions since an earlier capture or company conclusions.",
    ]


def render_index(snapshot: dict) -> str:
    """Compare four fixed topics without forcing the native measures to match."""
    validate_snapshot(snapshot)
    countries, numbers = _ordered(snapshot), _numbers(snapshot)
    lines = ["# National scenario monitoring", "", *_common_intro(snapshot), "",
             "[Complete offline snapshot](snapshot.json). Values use up to six significant digits; "
             "the snapshot retains full precision and source definitions.", "", "## Country reports", "",
             "| Country | Signals with verified source evidence | Complete comparison windows | Annual context |",
             "|---|---:|---:|---|"]
    for country in countries:
        code, coverage = country["country"], country["coverage"]
        lines.append(f"| [{_text(country['name'])}]({code}.md) | {coverage['signals_source_bound']}/{coverage['signals_expected']} | "
                     f"{coverage['signals_with_comparison']}/{coverage['signals_expected']} | "
                     f"[IMF reference and scenarios](context/{code}.md) |")
    lines.extend(["", "Coverage is not a predictive-confidence measure. Native industrial volume concepts "
                  "and loan populations differ; consult each country’s definitions."])
    for indicator, title in COMMON:
        lines.extend(["", f"## {title}", "",
                      "| Country / native measure | Reference period | Value and native unit | Change | Actual comparison window | Freshness | Evidence |",
                      "|---|---|---:|---:|---|---|---|"])
        for country in countries:
            code, signal = country["country"], _signal(country, indicator)
            if signal is None:
                lines.append(f"| [{_text(country['name'])}]({code}.md) | Not available | Not available | Not available | Not available | Unavailable | — |")
                continue
            comparison = signal.get("comparison")
            refs = _refs(signal.get("evidence_refs", []), snapshot, numbers, country=code, prefix=code + ".md")
            window = _text(comparison["window"]) if comparison else "Not available"
            lines.append(f"| [{_text(country['name'])}]({code}.md) — {_text(signal['label'])} | "
                         f"{_period(signal.get('latest'))} | {_value(signal.get('latest'))} | {_change(comparison)} | "
                         f"{window} | {_freshness(signal)} | {refs or '—'} |")
        lines.extend(["", "**Native definitions and capture clocks**", ""])
        for country in countries:
            signal = _signal(country, indicator)
            if signal is None:
                continue
            lines.append(f"- **{_text(country['name'])}:** {_text(signal.get('definition') or 'No eligible definition supplied.')} "
                         f"Frequency: {_words(signal['frequency'])}. {_capture(signal.get('source', {}))}.")
    lines.extend(["", "## Scope still to investigate", ""])
    lines.extend("- " + _text(gap) for gap in snapshot.get("remaining_gaps", []))
    lines.extend(["", _text(snapshot["methodology"].get("freshness_policy", "")), "",
                  _text(snapshot["methodology"].get("vintage_policy", "")), "",
                  "Germany’s applicable ECB instrument is shared euro-area policy, not an independent German "
                  "policy decision. National lending rates and sovereign yields retain their own scope.", "",
                  f"Snapshot SHA-256: {_text(snapshot['snapshot_sha256'])}."])
    return "\n".join(lines).rstrip() + "\n"


def render_country(snapshot: dict, code: str) -> str:
    """Render one country's observed signals and only its used source citations."""
    validate_snapshot(snapshot)
    if code not in COUNTRIES:
        raise ValueError("Unknown National report country")
    country = next(row for row in _ordered(snapshot) if row["country"] == code)
    numbers, coverage = _numbers(snapshot), country["coverage"]
    lines = [f"# {_text(country['name'])} — scenario monitoring", "",
             f"[National comparison](index.md) · [Annual IMF reference and scenario context](context/{code}.md)", "",
             *_common_intro(snapshot), "",
             f"Signals with verified source evidence: {coverage['signals_source_bound']}/{coverage['signals_expected']}; "
             f"complete comparison windows: {coverage['signals_with_comparison']}/{coverage['signals_expected']}. "
             "These counts describe coverage, not confidence.", "", "## At a glance", ""]
    for signal in country["signals"]:
        comparison = signal.get("comparison")
        refs = _refs(signal.get("evidence_refs", []), snapshot, numbers, country=code)
        detail = f"Latest {_period(signal.get('latest'))}: {_value(signal.get('latest'))}."
        if comparison:
            detail += f" Change: {_change(comparison)}; {_text(comparison['window'])}."
        lines.append(f"- **{_text(signal['label'])}.** {_text(signal['reading'])} {detail} {refs}".rstrip())
    lines.extend(["", "## Observed signals", "",
                  "| Native measure | Reference period | Value and unit | Change | Actual comparison window | Freshness | Evidence |",
                  "|---|---|---:|---:|---|---|---|"])
    for signal in country["signals"]:
        comparison = signal.get("comparison")
        window = _text(comparison["window"]) if comparison else "Not available"
        refs = _refs(signal.get("evidence_refs", []), snapshot, numbers, country=code)
        lines.append(f"| {_text(signal['label'])} | {_period(signal.get('latest'))} | {_value(signal.get('latest'))} | "
                     f"{_change(comparison)} | {window} | {_freshness(signal)} | {refs or '—'} |")
    lines.extend(["", _text(snapshot["methodology"].get("freshness_policy", "")), "",
                  "## Definitions, limits and scenario checks", "",
                  "The linked country assessment supplies conditional hypotheses. A change in one observed "
                  "price or industrial series does not mechanically confirm or invalidate a scenario."])
    used_refs = set()
    for signal in country["signals"]:
        used_refs.update(_all_signal_refs(signal))
        lines.extend(["", f"### {_text(signal['label'])}", "",
                      _text(signal.get("definition") or "No eligible source definition available."), "",
                      f"Frequency: {_words(signal['frequency'])}. {_capture(signal.get('source', {}))}."])
        source = signal.get("source", {})
        if source.get("series_id"):
            lines.extend(["", f"Native series: {_text(source['series_id'])}."])
        if source.get("publisher_updated_at"):
            lines.extend(["", f"Dataset updated: {_text(source['publisher_updated_at'])}. "
                          "This is separate from acquisition and any first-publication timestamp."])
        native = source.get("publisher_metadata") or {}
        if native.get("report_updated_date"):
            lines.extend(["", f"Report updated (date only): {_text(native['report_updated_date'])}; "
                          f"publisher timezone: {_text(native.get('update_timezone', 'not supplied'))}. "
                          "The source supplies no exact publication or update time."])
        _bullets(lines, "Interpretation limits", signal.get("limits", []))
        _bullets(lines, "Missing or inapplicable inputs", signal.get("gaps", []))
        gap = signal.get("source_gap", {})
        for url in gap.get("source_urls", []):
            if url.startswith("https://"):
                lines.extend(["", f"[Original documentation for this gap]({url})."])
        for link in signal.get("scenario_links", []):
            lines.extend(["", f"Related scenario hypothesis: [{_text(link['title'])}](context/{code}.md)."])
            _bullets(lines, "Evidence that would challenge its assumptions", link.get("evidence_that_challenges_assumption", []))
            refs = _refs(link.get("reference_evidence_refs", []), snapshot, numbers, country=code)
            if refs:
                lines.extend(["", f"Scenario reference evidence: {refs}."])
    native = country.get("national_debt_context", [])
    if native:
        used_refs.update(point["evidence_ref"] for point in native)
        _native_tables(lines, {"national_debt_context": native}, snapshot["citations"], numbers)
    else:
        lines.extend(["", "## National debt context", "",
                      "No original national debt-office context was selected for this country in the fixed "
                      "monitoring panel. The linked annual assessment retains the harmonized IMF debt and fiscal reference."])
    lines.extend(["", "## Uncollected and unresolved evidence", "",
                  "An input outside this panel may exist elsewhere in the project; it needs a suitable "
                  "definition and verified source lineage before use here."])
    _bullets(lines, "Remaining scope gaps", snapshot.get("remaining_gaps", []))
    for gap in snapshot.get("input_gaps", []):
        if gap.get("country") == code:
            lines.extend(["", f"- {_words(gap.get('indicator', 'Input'))}: {_words(gap.get('reason', 'unavailable'))}."])
    lines.extend(["", "## Reproducibility", "",
                  f"Method: {_text(snapshot['methodology']['version'])}. Snapshot SHA-256: {_text(snapshot['snapshot_sha256'])}.", "",
                  _text(snapshot["methodology"].get("comparison_rules", "")), "",
                  _text(snapshot["methodology"].get("vintage_policy", ""))])
    selected = {}
    for ref in used_refs:
        record = snapshot["citations"].get(ref)
        if record is None or record.get("country") != code:
            raise ValueError("Used evidence is missing or belongs to another country")
        selected[ref] = {**record, "source": _source_name(record)}
    _citations(lines, selected, numbers)
    return "\n".join(lines).rstrip() + "\n"
