"""Replay original SCB captures; no network is used in these tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from dalio.data_sources.scb_monitoring import (
    SCB_MONITORING_SPECS,
    parse_scb_input,
)

FIXTURES = Path(__file__).parent / "fixtures" / "scb_monitoring"


def capture(key):
    spec = next(item for item in SCB_MONITORING_SPECS if item.key == key)
    metadata = json.loads((FIXTURES / f"{key}_metadata.json").read_bytes())
    response = json.loads((FIXTURES / f"{key}_response.json").read_bytes())
    return spec, metadata, response


def parse(spec, metadata, response, request=None):
    return parse_scb_input(
        spec.key,
        json.dumps(metadata).encode(),
        json.dumps(response).encode(),
        request if request is not None else spec.request,
    )


@pytest.mark.parametrize(
    "key,value,count,updated",
    [
        ("industrial_production", 108.9, 319, "2026-09-10T06:00:00+00:00"),
        ("industrial_orders", 99.9, 319, "2026-09-10T06:00:00+00:00"),
        ("corporate_new_lending_rate", 3.5912, 325, "2026-08-27T06:00:00+00:00"),
    ],
)
def test_original_scb_captures_have_exact_native_values(key, value, count, updated):
    result = parse(*capture(key))
    assert result["indicator"] == key
    assert result["publisher_updated_at"] == updated
    assert result["frequency"] == "monthly"
    assert len(result["observations"]) == count
    latest = result["observations"][-1]
    assert latest["date"] == latest["period_end"] == "2026-07-31"
    assert latest["period_start"] == "2026-07-01"
    assert latest["period"] == "2026-07"
    assert latest["value"] == value
    assert latest["status"] == "observed"


def test_index_scopes_adjustments_and_revisions_are_preserved():
    result = parse(*capture("industrial_production"))
    assert result["unit"] == "index"
    assert result["adjustment"] == "seasonally_adjusted"
    metadata = result["publisher_metadata"]
    assert metadata["base_period"] == "2021"
    assert metadata["native_adjustment"] == "WorkAndSes"
    assert metadata["price_type"] == "Fixed"
    assert metadata["dimensions"]["SNI2007"]["code"] == "B+C"
    assert any("2026-09-10" in note for note in metadata["notes"])
    assert "NV0402AL" in result["series_id"]


def test_nfc_rate_is_sek_new_and_renegotiated_not_household_or_all_accounts():
    result = parse(*capture("corporate_new_lending_rate"))
    metadata = result["publisher_metadata"]
    assert result["unit"] == "percent"
    assert result["adjustment"] == "not_adjusted"
    assert metadata["source"] == "The Riksbank"
    assert metadata["dimensions"]["Motpartssektor"]["code"] == "1"
    assert metadata["dimensions"]["Avtal"]["code"] == "0100"
    assert metadata["dimensions"]["Rantebindningstid"]["code"] == "1.1"
    assert metadata["denomination_currency"] == "SEK"
    assert "renegotiated" in result["definition"]
    assert "annualised" in result["definition"]
    assert result["missingness"]["not_reported"] == 74
    assert result["observations"][0]["value"] is None
    assert result["observations"][0]["native_status"] == ".."


def test_last_missing_slot_survives_and_is_distinct_from_zero():
    spec, metadata, response = capture("industrial_orders")
    response["value"][-2:] = [0, None]
    response["status"] = {str(len(response["value"]) - 1): ".."}
    result = parse(spec, metadata, response)
    assert result["observations"][-2]["value"] == 0.0
    assert result["observations"][-2]["status"] == "observed"
    assert result["observations"][-1]["period"] == "2026-07"
    assert result["observations"][-1]["status"] == "not_reported"


def test_sparse_values_and_scrambled_category_mapping_follow_declared_positions():
    spec, metadata, response = capture("industrial_orders")
    response["value"] = {str(i): value for i, value in enumerate(response["value"])}
    index = response["dimension"]["Tid"]["category"]["index"]
    response["dimension"]["Tid"]["category"]["index"] = dict(reversed(list(index.items())))
    assert parse(spec, metadata, response)["observations"][-1]["value"] == 99.9


@pytest.mark.parametrize("field", ["source_url", "method", "selection"])
def test_changed_request_is_rejected(field):
    spec, metadata, response = capture("industrial_orders")
    request = copy.deepcopy(spec.request)
    request[field] = "changed"
    with pytest.raises(ValueError, match="request"):
        parse(spec, metadata, response, request)


@pytest.mark.parametrize("change", ["unit", "base", "adjustment", "scope", "source", "table"])
def test_native_identity_or_measure_drift_is_rejected(change):
    spec, metadata, response = capture("industrial_production")
    measure = metadata["dimension"]["ContentsCode"]
    if change == "unit":
        measure["category"]["unit"]["NV0402AL"]["base"] = "percent"
    elif change == "base":
        measure["extension"]["basePeriod"]["NV0402AL"] = "2025"
    elif change == "adjustment":
        measure["extension"]["adjustment"]["NV0402AL"] = "None"
    elif change == "scope":
        response["dimension"]["SNI2007"]["category"]["index"] = {"B-D": 0}
    elif change == "source":
        response["source"] = "Some distributor"
    else:
        response["extension"]["px"]["tableid"] = "TAB4894"
    with pytest.raises(ValueError):
        parse(spec, metadata, response)


@pytest.mark.parametrize("clock", ["2026-09-10T07:00:00Z", "2026-09-10T06:00:00"])
def test_torn_or_naive_publisher_release_is_rejected(clock):
    spec, metadata, response = capture("industrial_production")
    response["updated"] = clock
    with pytest.raises(ValueError, match="clock|release|timezone"):
        parse(spec, metadata, response)


@pytest.mark.parametrize("bad", [True, "100", float("nan"), float("inf")])
def test_non_numeric_and_non_finite_values_fail_closed(bad):
    spec, metadata, response = capture("industrial_orders")
    response["value"][-1] = bad
    with pytest.raises(ValueError):
        parse(spec, metadata, response)


def test_incomplete_response_fails_even_when_array_matches_reduced_time_dimension():
    spec, metadata, response = capture("industrial_production")
    response["value"].pop()
    response["size"][-1] -= 1
    for field in ["index", "label"]:
        response["dimension"]["Tid"]["category"][field].pop("2026M07")
    with pytest.raises(ValueError, match="period|time"):
        parse(spec, metadata, response)


def test_future_observation_and_unknown_status_fail_closed():
    spec, metadata, response = capture("industrial_production")
    response["status"] = {"0": "forecast"}
    with pytest.raises(ValueError, match="status"):
        parse(spec, metadata, response)
    response.pop("status")
    metadata["updated"] = response["updated"] = "2026-06-10T06:00:00Z"
    with pytest.raises(ValueError, match="publication"):
        parse(spec, metadata, response)


def test_conflicting_status_and_duplicate_category_positions_fail_closed():
    spec, metadata, response = capture("industrial_orders")
    response["status"] = {"0": ".."}
    with pytest.raises(ValueError, match="status"):
        parse(spec, metadata, response)
    response.pop("status")
    response["dimension"]["Tid"]["category"]["index"]["2000M02"] = 0
    with pytest.raises(ValueError, match="position"):
        parse(spec, metadata, response)
