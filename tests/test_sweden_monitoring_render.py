"""Readable windows, source clocks and scope in the offline Sweden report."""

import re
from copy import deepcopy

import pytest

from dalio.monitoring.render import render_monitoring


def fixture():
    citations = {}
    signals = []
    for key, label, unit, latest, change, window in (
        ("industrial_production", "Industrial production", "index, 2021=100", 110, 10,
         "Mean 2026-05–2026-07 versus 2026-02–2026-04"),
        ("industrial_orders", "Industrial order intake", "index, 2021=100", 105, 5,
         "Mean 2026-05–2026-07 versus 2026-02–2026-04"),
        ("corporate_new_lending_rate", "Business lending rate", "% per annum", 3.5, -0.4,
         "2026-07 versus 2026-04"),
        ("policy_rate", "Riksbank effective policy rate", "%", 1.75, -0.5,
         "2026-09-08 versus 2026-06-09 (90-day target 2026-06-10)"),
        ("yield_10y", "Swedish 10-year government benchmark yield", "%", 3.1, 0.2,
         "2026-09-07 versus 2026-06-08 (90-day target 2026-06-09)"),
    ):
        monthly = key not in {"policy_rate", "yield_10y"}
        point = dict(value=latest, unit=unit, period="2026-07" if monthly else "2026-09-08",
                     date="2026-07-31" if monthly else "2026-09-08", status="observed",
                     evidence_ref=key)
        source = dict(source="SCB" if monthly else "RIKSBANK_SWEA", series_id=key,
                      source_url="https://api.scb.se/example?selection[]=B+C",
                      published_at=None, publisher_updated_at="2026-09-10T08:00:00+00:00",
                      available_at="2026-09-10T23:00:00+00:00",
                      retrieved_at="2026-09-10T23:00:00+00:00", artifacts=[])
        if key == "corporate_new_lending_rate":
            source["publisher_metadata"] = {"source": "The Riksbank", "producer": "Statistics Sweden"}
        citations[key] = {**point, **source, "country": "SE", "indicator": key}
        definition = ("Mining and manufacturing B+C, excluding energy; calendar and seasonally adjusted."
                      if key.startswith("industrial") else
                      "New and renegotiated SEK term-loan agreements for non-financial corporations; "
                      "includes floating-rate loans and excludes transaction-account balances.")
        signals.append(dict(indicator=key, label=label, frequency="monthly" if monthly else "daily",
                            definition=definition, source=source, latest=point, status="available",
                            age_days=41 if monthly else 2, freshness_limit_days=75 if monthly else 10,
                            reading=f"Retained reading for {label}.", evidence_refs=[key], gaps=[],
                            limits=["A lower rate does not establish easier credit access."],
                            comparison=dict(value=change, unit="%" if key.startswith("industrial")
                                            else "percentage points", window=window, evidence_refs=[key]),
                            scenario_links=[dict(scenario_id="funding_strain", title="Funding strain",
                                evidence_that_challenges_assumption=["Financing costs fall while credit access improves."],
                                reference_evidence_refs=["annual"])]))
    annual = dict(value=2.0, unit="%", year=2026, date="2026-12-31",
                  status="forecast_calendar_convention", evidence_ref="annual", source="IMF_WEO",
                  source_url="https://www.imf.org/external/datamapper/NGDP_RPCH/SWE", series_id="NGDP_RPCH",
                  available_at="2026-09-10T20:00:00+00:00", retrieved_at="2026-09-10T20:00:00+00:00")
    citations["annual"] = annual
    native = []
    for key, value, unit, status, aggregation in (
        ("central_gov_gross_debt_sek", 1249558872203, "SEK", "observed", "reference_date"),
        ("average_time_to_refixing", 5.05, "years", "observed", "reference_date"),
        ("gross_borrowing_requirement", 655.6, "SEK_bn", "forecast", "annual"),
    ):
        point = dict(metric=key, value=value, unit=unit, status=status, evidence_ref=key, year=2026,
                     period_start="2026-01-01" if status == "forecast" else "2026-08-31",
                     period_end="2026-12-31" if status == "forecast" else "2026-08-31",
                     dimensions=dict(scope="central_government", aggregation=aggregation),
                     source="RIKSGALDEN_DEBT", series_id="native", source_url="https://www.riksgalden.se/data.pdf",
                     available_at="2026-09-10T20:59:01+00:00", published_at="2026-09-07T00:00:00+00:00",
                     source_locator="PDF page 1", release_id=3454)
        native.append(point)
        citations[key] = point
    return dict(country="SE", name="Sweden", as_of="2026-09-10", as_known_at="2026-09-10T23:00:00+00:00",
                snapshot_sha256="a" * 64, methodology=dict(version="sweden-monitoring-v1",
                    freshness_policy="Editorial applicability windows, not publisher expiry dates.",
                    vintage_policy="Within-vintage comparisons do not reconstruct earlier knowledge."),
                signals=signals, citations=citations, national_debt_context=native,
                coverage=dict(signals_expected=5, signals_source_bound=5, signals_with_comparison=5),
                input_gaps=[], remaining_gaps=["Credit standards are outside this pilot."],
                scenarios=[dict(id="funding_strain", title="Funding strain", horizon="6–36 months",
                    assumptions=["Financing costs rise."], pathway=["Debt reprices."],
                    company_checks=["Verify actual debt maturities and loan contracts."],
                    limitations=["No company exposure is established by listing country."], evidence_refs=["annual"])],
                country_assessment=dict(as_of="2026-09-10", as_known_at="2026-09-10T23:00:00+00:00",
                    snapshot_sha256="b" * 64, baseline_year=2025, countries=[dict(baseline={},
                        projections=[dict(year=2026, metrics={"real_gdp_growth": annual})])]))


def test_readout_retains_five_signals_actual_windows_units_and_core_reading():
    snapshot = fixture()
    text = render_monitoring(snapshot)
    assert text.index("## At a glance") < text.index("## Observed signals")
    for signal in snapshot["signals"]:
        assert signal["reading"] in text
        assert signal["comparison"]["window"] in text
    assert "−0.4 percentage points" in text
    assert "index, 2021=100" in text
    assert "41 days" in text and "75 days" in text
    assert "known-at" in text and "UTC" in text
    assert "floating-rate" in text and "transaction-account" in text
    assert "B+C" in text and "excluding energy" in text


def test_missing_and_stale_values_remain_visible_without_silent_backfill():
    snapshot = fixture()
    snapshot["signals"][0].update(latest=None, comparison=None, status="unavailable", age_days=None,
        reading="No current directional reading.", gaps=["No original-response-bound input."])
    snapshot["signals"][1]["latest"]["value"] = None
    snapshot["signals"][1].update(comparison=None, status="missing_latest",
        reading="No current directional reading.", gaps=["Latest July value is not backfilled."])
    snapshot["signals"][2].update(comparison=None, status="stale", age_days=120,
        reading="No current directional reading.", gaps=["Outside editorial freshness window."])
    text = render_monitoring(snapshot)
    assert "Not available" in text and "2026-07" in text
    assert "Missing latest" in text and "Stale" in text and "120 days" in text
    assert "Latest July value is not backfilled." in text
    assert "No original-response-bound input." in text


def test_native_observed_and_forecast_scopes_are_separate_and_readable():
    snapshot = fixture()
    mean = deepcopy(snapshot["national_debt_context"][1])
    mean.update(value=4.85, period_start="2026-08-01", evidence_ref="atr-monthly-mean")
    mean["dimensions"]["aggregation"] = "monthly_mean"
    snapshot["national_debt_context"].append(mean)
    snapshot["citations"][mean["evidence_ref"]] = mean
    text = render_monitoring(snapshot)
    assert "1,249,558,872,203 SEK" in text and "655.6 billion SEK" in text
    assert "## Observed national debt context" in text
    assert "## Published national funding forecasts" in text
    assert text.index("1,249,558,872,203 SEK") < text.index("## Published national funding forecasts")
    assert "interest-rate resets" in text and "principal repayment" in text
    assert "central government" in text.lower() and "reference date" in text.lower()
    assert "5.05 years | 2026-08-31 |" in text
    assert "4.85 years | 2026-08-01 to 2026-08-31 |" in text
    assert "655.6 billion SEK | 2026-01-01 to 2026-12-31 |" in text


def test_annual_status_convention_and_scenario_challenges_remain_explicit():
    text = render_monitoring(fixture())
    assert "does not supply a native per-point status" in text
    assert "calendar convention" in text and "annual GDP forecast error" in text
    assert "Scenario hypothesis" in text and "6–36 months" in text
    assert "Financing costs fall while credit access improves." in text
    assert "Credit standards are outside this pilot." in text
    assert "not publisher expiry dates" in text


def test_every_citation_link_resolves_and_sources_retain_clocks():
    snapshot = fixture()
    text = render_monitoring(snapshot)
    targets = set(re.findall(r'\]\(#(evidence-[a-f0-9]+)\)', text))
    anchors = set(re.findall(r'<a id="(evidence-[a-f0-9]+)"></a>', text))
    assert targets and targets <= anchors and len(anchors) == len(snapshot["citations"])
    assert "2026-09-10T23:00:00+00:00" in text
    assert "Publication timestamp: not supplied" in text
    assert "Dataset updated: 2026-09-10T08:00:00+00:00" in text
    assert "PDF page 1" in text and "3454" in text
    snapshot["signals"][0]["evidence_refs"] = ["not-a-citation"]
    with pytest.raises(ValueError, match="citation"):
        render_monitoring(snapshot)


def test_markdown_text_and_source_links_are_safe():
    snapshot = fixture()
    snapshot["signals"][0]["definition"] = '<script>alert(1)</script> | [claim](javascript:bad)'
    text = render_monitoring(snapshot)
    assert "<script>" not in text and "&lt;script&gt;" in text
    assert r"\|" in text and r"\[claim\]" in text
    snapshot["citations"]["industrial_production"]["source_url"] = "javascript:bad"
    with pytest.raises(ValueError, match="URL"):
        render_monitoring(snapshot)


def test_render_is_deterministic_and_does_not_mutate_snapshot():
    snapshot = fixture()
    before = deepcopy(snapshot)
    assert render_monitoring(snapshot) == render_monitoring(snapshot)
    assert snapshot == before


def test_source_names_are_readable_attribution_is_retained_and_day_is_singular():
    snapshot = fixture()
    snapshot["signals"][-1]["age_days"] = 1
    text = render_monitoring(snapshot)
    assert "Statistics Sweden (SCB)" in text and "Sveriges Riksbank" in text
    assert "Statistics Sweden (SCB), on behalf of Sveriges Riksbank" in text
    assert "1 day since reference-period end" in text
    assert not re.search(r"\b1 days since", text)
    assert "RIKSBANK SWEA" not in text
