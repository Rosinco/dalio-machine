"""Native volume concepts, dated comparisons and country-specific missingness."""

from copy import deepcopy
from datetime import date

import pytest

from dalio.assessments.core import build_snapshot as build_assessment
from dalio.assessments.core import content_hash
from dalio.national_monitoring import core
from tests.test_country_assessments import evidence
from tests.test_sweden_monitoring import KNOWN, data, seal


@pytest.fixture
def inputs(monkeypatch):
    parent = evidence()
    parent["as_known_at"] = KNOWN
    template = parent["countries"][0]
    parent["countries"] = []
    national = {"as_known_at": KNOWN, "series": [], "gaps": [], "protected_artifact_paths": []}
    specs = []
    for code, name in (("US", "United States"), ("DE", "Germany"), ("CA", "Canada")):
        country = deepcopy(template)
        country.update(country=code, listing_iso2=code, name=name)
        for series in country["series"]:
            series["partition_key"] = series["partition_key"].replace(":SE", ":" + code)
        parent["countries"].append(country)
        for original in data()["series"]:
            if original["indicator"] == "industrial_orders":
                continue
            series = deepcopy(original)
            series["country"] = code
            key = series["indicator"]
            spec = {"country": code, "indicator": key, "label": code + " " + key,
                    "frequency": series["frequency"], "unit": series["unit"],
                    "comparison": "three_month_means" if key == "industrial_production" else
                    "three_month_rate" if key == "corporate_new_lending_rate" else "ninety_day_rate",
                    "definition": code + " original definition", "limits": ["Native scopes differ."],
                    "adjustment": series.get("adjustment")}
            series.update(spec=spec, definition=spec["definition"])
            specs.append(spec)
            national["series"].append(series)
    monkeypatch.setattr(core, "catalogue", lambda end: specs)
    return seal(national), build_assessment(seal(parent), as_of=date(2026, 9, 10)), specs


def build(inputs):
    return core.build_snapshot(seal(inputs[0]), inputs[1], as_of=date(2026, 9, 10))


def signal(snapshot, country="US", indicator="industrial_production"):
    return next(s for c in snapshot["countries"] if c["country"] == country
                for s in c["signals"] if s["indicator"] == indicator)


def test_three_country_citations_and_same_series_math(inputs):
    result = build(inputs)
    assert [c["country"] for c in result["countries"]] == ["US", "DE", "CA"]
    assert signal(result)["comparison"]["value"] == pytest.approx(10)
    assert all(result["citations"][ref]["country"] == "US" for ref in signal(result)["evidence_refs"])
    assert all(ref.startswith("national:") for ref in signal(result)["evidence_refs"])
    assert result == build(inputs)


def test_structural_gap_is_not_a_stale_rate_or_a_proxy(inputs):
    inputs[0]["series"] = [s for s in inputs[0]["series"]
                           if (s["country"], s["indicator"]) != ("US", "corporate_new_lending_rate")]
    inputs[0]["gaps"] = [{"country": "US", "indicator": "corporate_new_lending_rate",
                          "reason": "structural_gap", "error": "Survey discontinued in 2017.",
                          "artifacts": [{"role": "documentation", "sha256": "d" * 64}]}]
    result = build(inputs)
    item = signal(result, indicator="corporate_new_lending_rate")
    assert item["latest"] is None and item["comparison"] is None
    assert item["direction"] == "unavailable"
    assert "Survey discontinued in 2017." in item["gaps"]
    assert result["countries"][0]["coverage"]["signals_source_bound"] == 3
    assert signal(result, "DE", "corporate_new_lending_rate")["status"] == "available"


def test_canadian_real_industry_volume_is_not_rebased_or_called_an_index(inputs):
    source = next(s for s in inputs[0]["series"] if s["country"] == "CA"
                  and s["indicator"] == "industrial_production")
    unit = "millions of chained 2017 Canadian dollars"
    source["unit"] = source["spec"]["unit"] = unit
    source["spec"]["measure_kind"] = "real_industrial_value_added_volume"
    source["definition"] = "Canada real industrial GDP at basic prices, seasonally adjusted annual rate."
    item = signal(build(inputs), "CA")
    assert item["latest"]["unit"] == unit
    assert item["latest"]["value"] == source["observations"][-1]["value"]
    assert item["comparison"]["value"] == pytest.approx(10)
    assert "volume average" in item["reading"] and "index" not in item["reading"]
    del source["spec"]["measure_kind"]
    with pytest.raises(ValueError, match="volume|measure"):
        build(inputs)


def test_missing_latest_and_stale_rates_have_no_direction(inputs):
    inputs[0]["series"][0]["observations"][-1].update(value=None, status="not_reported")
    assert signal(build(inputs))["status"] == "missing_latest"
    inputs[0]["series"][2]["observations"].pop()
    item = signal(build(inputs), indicator="policy_rate")
    assert item["status"] == "stale" and item["direction"] == "unavailable"


def test_exact_month_and_daily_anchor_gaps_are_preserved(inputs):
    inputs[0]["series"][1]["observations"].pop(0)
    assert signal(build(inputs), indicator="corporate_new_lending_rate")["comparison"] is None
    inputs[0]["series"][3]["observations"][0].update(period="2026-06-11", date="2026-06-11")
    assert signal(build(inputs), indicator="yield_10y")["comparison"] is None


def test_wrong_country_duplicate_and_cutoff_mismatch_fail(inputs):
    inputs[0]["series"][0]["country"] = "SE"
    with pytest.raises(ValueError, match="identity"):
        build(inputs)
    inputs[0]["series"][0]["country"] = "US"
    inputs[0]["series"].append(deepcopy(inputs[0]["series"][0]))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        build(inputs)
    inputs[0]["series"].pop()
    inputs[0]["as_known_at"] = "2026-09-11T00:00:00+00:00"
    with pytest.raises(ValueError, match="cutoff"):
        build(inputs)


def test_snapshot_country_and_citation_tampering_fail_even_when_rehashed(inputs):
    result = build(inputs)
    signal(result)["latest"]["value"] += 1
    with pytest.raises(ValueError, match="hash"):
        core.validate_snapshot(result)
    result["snapshot_sha256"] = content_hash({k: v for k, v in result.items() if k != "snapshot_sha256"})
    with pytest.raises(ValueError, match="citation"):
        core.validate_snapshot(result)
