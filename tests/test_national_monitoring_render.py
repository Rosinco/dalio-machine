"""Offline links, native definitions and explicit structural gap presentation."""

import re

import pytest

from dalio.national_monitoring.render import render_country, render_index
from tests.test_national_monitoring_core import build, inputs  # noqa: F401 - fixture


def test_report_set_links_resolve_without_nordic_country_leakage(inputs):  # noqa: F811
    snapshot = build(inputs)
    reports = {"index.md": render_index(snapshot)}
    reports.update({f"{c}.md": render_country(snapshot, c) for c in ("US", "DE", "CA")})
    for body in reports.values():
        assert "SE.md" not in body and "FI.md" not in body
        for target in re.findall(r"\]\(([^)]+)\)", body):
            target = target.strip("<>")
            if target.startswith(("https://", "http://", "#")):
                continue
            path = target.split("#")[0]
            assert path in {*reports, "snapshot.json", "context/US.md", "context/DE.md", "context/CA.md"}
    assert "Germany’s applicable ECB instrument is shared" in reports["index.md"]
    assert "no country ranking" in reports["index.md"]


def test_documented_gap_keeps_reason_source_and_conditional_scenario(inputs):  # noqa: F811
    inputs[0]["series"] = [s for s in inputs[0]["series"] if
                           (s["country"], s["indicator"]) != ("US", "corporate_new_lending_rate")]
    inputs[0]["gaps"] = [{"country": "US", "indicator": "corporate_new_lending_rate",
                          "reason": "structural_gap", "error": "Survey discontinued in 2017.",
                          "source_urls": ["https://www.federalreserve.gov/releases/e2/"]}]
    body = render_country(build(inputs), "US")
    assert "3/4" in body
    assert "Survey discontinued in 2017." in body
    assert "[Original documentation for this gap](https://www.federalreserve.gov/releases/e2/)" in body
    assert "conditional hypotheses" in body
    assert "No current directional reading" in body


def test_native_monthly_yield_remains_a_monthly_comparison(inputs):  # noqa: F811
    source = next(s for s in inputs[0]["series"] if s["country"] == "DE" and s["indicator"] == "yield_10y")
    monthly = next(s for s in inputs[0]["series"] if s["country"] == "DE" and s["indicator"] == "corporate_new_lending_rate")
    source["observations"] = monthly["observations"]
    source["frequency"] = source["spec"]["frequency"] = "monthly"
    source["spec"]["comparison"] = "three_month_rate"
    body = render_country(build(inputs), "DE")
    assert "2026-07 versus 2026-04" in body


def test_unknown_country_is_rejected(inputs):  # noqa: F811
    with pytest.raises(ValueError, match="country"):
        render_country(build(inputs), "SE")
