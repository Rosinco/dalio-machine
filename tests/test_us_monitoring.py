import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.us_monitoring import parse_input, specs

FIX = Path(__file__).parent / "fixtures" / "us_monitoring"
END = date(2026, 9, 11)


def example(indicator):
    spec = next(s for s in specs(END) if s["indicator"] == indicator)
    name = {
        "industrial_production": "industry",
        "corporate_new_lending_rate": "lending",
        "policy_rate": "policy",
        "yield_10y": "yield",
    }[indicator]
    return spec, {
        r["role"]: gzip.decompress((FIX / f"{name}_{r['role']}.bin.gz").read_bytes())
        for r in spec["requests"]
    }


@pytest.mark.parametrize(
    "indicator,count,last",
    [
        ("industrial_production", 1291, 102.9939),
        ("policy_rate", 424, 3.63),
        ("yield_10y", 423, 4.95),
    ],
)
def test_original_source_histories(indicator, count, last):
    result = parse_input(*example(indicator))
    assert len(result["observations"]) == count
    assert result["observations"][-1]["value"] == last
    assert result["published_at"] is None
    assert all(r["period"] and r["source_locator"] for r in result["observations"])


def test_lending_gap_is_bound_to_discontinuation_notice():
    spec, bodies = example("corporate_new_lending_rate")
    result = parse_input(spec, bodies)
    assert result == {
        "country": "US",
        "indicator": "corporate_new_lending_rate",
        "unavailable_reason": spec["unavailable_reason"],
    }
    bodies["documentation"] = bodies["documentation"].replace(b"discontinued", b"continued")
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "mutation",
    ["wrong_series", "duplicate_year", "missing_month", "wrong_base", "wrong_latest", "nan"],
)
def test_g17_fails_closed(mutation):
    spec, bodies = example("industrial_production")
    if mutation == "wrong_series":
        bodies["catalogue"] = bodies["catalogue"].replace(b"B50001", b"B99999")
    elif mutation == "duplicate_year":
        bodies["data"] += b'\n"B50001" 2026 100\n'
    elif mutation == "missing_month":
        bodies["data"] = bodies["data"].replace(b"102.5198", b"")
    elif mutation == "wrong_base":
        bodies["metadata"] = bodies["metadata"].replace(b"2017", b"2012")
    elif mutation == "wrong_latest":
        bodies["data"] = bodies["data"].replace(b"102.9939", b"900.0000")
    else:
        bodies["data"] = bodies["data"].replace(b"102.9939", b"NaN")
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "mutation", ["type", "duplicate", "future", "flag", "nonfinite", "missing"]
)
def test_effr_identity_status_and_dates(mutation):
    spec, bodies = example("policy_rate")
    raw = json.loads(bodies["data"])
    if mutation == "type":
        raw["refRates"][0]["type"] = "SOFR"
    elif mutation == "duplicate":
        raw["refRates"].append(raw["refRates"][0])
    elif mutation == "future":
        raw["refRates"][0]["effectiveDate"] = "2026-09-12"
    elif mutation == "flag":
        raw["refRates"][0]["revisionIndicator"] = "UNKNOWN"
    elif mutation == "nonfinite":
        raw["refRates"][0]["percentRate"] = float("inf")
    else:
        raw["refRates"][0].pop("percentRate")
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"DailyTreasuryYieldCurveRateData", b"DailyTreasuryRealYieldCurveRateData"),
        (b"BC_10YEAR", b"BC_5YEAR"),
        (b"2026-01-02T00:00:00", b"2027-01-02T00:00:00"),
        (b"4.95</d:BC_10YEAR>", b"NaN</d:BC_10YEAR>"),
    ],
)
def test_treasury_identity_and_values(old, new):
    spec, bodies = example("yield_10y")
    bodies["data"] = bodies["data"].replace(old, new)
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_contract_and_body_set_cannot_change():
    spec, bodies = example("policy_rate")
    spec["definition"] = "Target ceiling"
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
    spec, bodies = example("policy_rate")
    bodies["extra"] = b"other"
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
