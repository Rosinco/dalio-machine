"""Country identity, native comparison windows and scenario-context boundaries."""

from copy import deepcopy
from datetime import date

import pytest

from dalio.assessments.core import build_snapshot, content_hash
from dalio.nordic_monitoring import core
from tests.test_sweden_monitoring import KNOWN, data, seal


@pytest.fixture
def inputs(monkeypatch):
    from tests.test_country_assessments import evidence
    parent = evidence()
    parent["as_known_at"] = KNOWN
    template = parent["countries"][0]
    parent["countries"] = []
    for code, name in (("SE", "Sweden"), ("NO", "Norway"), ("DK", "Denmark"), ("FI", "Finland")):
        country = deepcopy(template)
        country.update(country=code, listing_iso2=code, name=name)
        for series in country["series"]:
            series["partition_key"] = series["partition_key"].replace(":SE", ":" + code)
        parent["countries"].append(country)
    seal(parent)
    assessment = build_snapshot(parent, as_of=date(2026, 9, 10))
    se = data()
    nordic = {"as_known_at": KNOWN, "series": [], "gaps": [], "protected_artifact_paths": []}
    specs = []
    for code in ("NO", "DK", "FI"):
        for original in se["series"]:
            if original["indicator"] == "industrial_orders":
                continue
            series = deepcopy(original)
            series["country"] = code
            key = series["indicator"]
            spec = {"country": code, "indicator": key, "label": code + " " + key,
                    "frequency": series["frequency"], "unit": series["unit"],
                    "comparison": "three_month_means" if key == "industrial_production" else
                    "three_month_rate" if key == "corporate_new_lending_rate" else "ninety_day_rate",
                    "definition": code + " original definition", "limits": ["Native country scope differs."],
                    "adjustment": series.get("adjustment")}
            series.update(spec=spec, definition=spec["definition"])
            specs.append(spec)
            nordic["series"].append(series)
    monkeypatch.setattr(core, "catalogue", lambda end: specs)
    return seal(nordic), se, assessment


def build(inputs):
    n, s, p = inputs
    return core.build_snapshot(seal(n), seal(s), p, as_of=date(2026, 9, 10))


def signal(result, country="NO", indicator="industrial_production"):
    row = next(c for c in result["countries"] if c["country"] == country)
    return next(s for s in row["signals"] if s["indicator"] == indicator)


def test_four_country_comparison_keeps_se_orders_extra_and_citations_native(inputs):
    result = build(inputs)
    assert len(result["countries"]) == 4
    assert [len(c["signals"]) for c in result["countries"]] == [5, 4, 4, 4]
    item = signal(result)
    assert item["comparison"]["value"] == pytest.approx(10)
    assert all(result["citations"][r]["country"] == "NO" for r in item["evidence_refs"])
    assert item["definition"] == "NO original definition"
    assert {x["scenario_id"] for x in item["scenario_links"]} == {"demand_shortfall"}
    assert result == build(inputs)


def test_missing_latest_never_backfills_and_stale_rate_has_no_current_direction(inputs):
    inputs[0]["series"][0]["observations"][-1].update(value=None, status="not_reported")
    item = signal(build(inputs))
    assert item["status"] == "missing_latest" and item["comparison"] is None
    rate = inputs[0]["series"][2]
    rate["observations"].pop()
    item = signal(build(inputs), indicator="policy_rate")
    assert item["status"] == "stale" and item["direction"] == "unavailable"


def test_rate_units_and_forecast_observations_rejected(inputs):
    inputs[0]["series"][1]["unit"] = "basis points"
    with pytest.raises(ValueError, match="unit|contract"):
        build(inputs)
    inputs[0]["series"][1]["unit"] = "% per annum"
    inputs[0]["series"][0]["observations"][-1]["status"] = "forecast"
    with pytest.raises(ValueError, match="observed"):
        build(inputs)


def test_exact_month_and_daily_anchor_limits(inputs):
    inputs[0]["series"][1]["observations"].pop(0)
    assert signal(build(inputs), indicator="corporate_new_lending_rate")["comparison"] is None
    inputs[0]["series"][3]["observations"][0].update(period="2026-06-11", date="2026-06-11")
    assert signal(build(inputs), indicator="yield_10y")["comparison"] is None


def test_wrong_country_duplicate_unknown_and_clock_mismatch_fail(inputs):
    inputs[0]["series"][0]["country"] = "SE"
    with pytest.raises(ValueError, match="identity"):
        build(inputs)
    inputs[0]["series"][0]["country"] = "NO"
    inputs[0]["series"].append(deepcopy(inputs[0]["series"][0]))
    with pytest.raises(ValueError, match="[Dd]uplicate"):
        build(inputs)
    inputs[0]["series"].pop()
    inputs[0]["as_known_at"] = "2026-09-11T00:00:00+00:00"
    with pytest.raises(ValueError, match="cutoff"):
        build(inputs)


def test_gaps_remain_country_specific(inputs):
    inputs[0]["series"] = [s for s in inputs[0]["series"] if s["country"] != "FI"]
    result = build(inputs)
    assert signal(result, "FI")["status"] == "unavailable"
    assert signal(result, "NO")["status"] == "available"


def test_snapshot_tamper_fails_even_when_country_rows_look_plausible(inputs):
    result = build(inputs)
    signal(result)["latest"]["value"] += 1
    with pytest.raises(ValueError, match="hash"):
        core.validate_snapshot(result)
    result["snapshot_sha256"] = content_hash({k: v for k, v in result.items() if k != "snapshot_sha256"})
    with pytest.raises(ValueError, match="citation"):
        core.validate_snapshot(result)


def test_original_norway_source_contracts_reach_country_math_without_unit_or_identity_relabelling(inputs):
    from dalio.data_sources.norway_monitoring import parse_input
    from tests.test_norway_monitoring import capture
    inputs[0]["series"] = [s for s in inputs[0]["series"] if s["country"] != "NO"]
    for key in core.INDICATORS:
        spec, bodies = capture(key)
        series = parse_input(spec, bodies)
        series.update(spec=spec, available_at=KNOWN, retrieved_at=KNOWN, published_at=None,
                      source_url=spec["requests"][-1]["url"], artifacts=[])
        inputs[0]["series"].append(series)
    result = build(inputs)
    rate = signal(result, indicator="corporate_new_lending_rate")
    assert rate["latest"]["value"] == 6.3 and rate["latest"]["unit"] == "per cent"
    assert rate["comparison"]["unit"] == "percentage points"
    assert signal(result)["status"] == "available"
    assert "Petroleum-related manufacturing" in signal(result)["definition"]
    yield_ = signal(result, indicator="yield_10y")
    assert yield_["latest"]["value"] == 4.489
    assert "nearest" in yield_["definition"]
    assert result["citations"][yield_["latest"]["evidence_ref"]]["country"] == "NO"
