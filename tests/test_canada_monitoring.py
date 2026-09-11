import gzip
import json
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.canada_monitoring import parse_input, specs

FIX = Path(__file__).parent / "fixtures" / "canada_monitoring"
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
        ("industrial_production", 36, 392804),
        ("corporate_new_lending_rate", 30, 4.33),
        ("policy_rate", 697, 2.25),
        ("yield_10y", 672, 3.84),
    ],
)
def test_original_source_histories(indicator, count, last):
    result = parse_input(*example(indicator))
    assert len(result["observations"]) == count
    assert result["observations"][-1]["value"] == last
    assert result["published_at"] is None


def test_industrial_gdp_is_not_rebased_or_called_index():
    result = parse_input(*example("industrial_production"))
    assert result["measure_kind"] == "real_industrial_value_added_volume"
    assert result["unit"] == "millions of chained 2017 Canadian dollars"
    assert result["comparison"] == "three_month_means"
    assert "value added" in result["definition"]
    assert result["observations"][-1]["date"] == "2026-06-30"
    assert "not a physical output index" in " ".join(result["limits"])


def test_business_loan_scope_is_explicit():
    result = parse_input(*example("corporate_new_lending_rate"))
    assert "individuals" in result["definition"]
    assert "Canadian dollars" in result["definition"]
    assert "refinancing" in result["definition"]
    assert result["observations"][-1]["date"] == "2026-06-30"


@pytest.mark.parametrize(
    "mutation",
    [
        "coordinate",
        "vector",
        "scale",
        "status",
        "security",
        "duplicate",
        "period",
        "native_clock",
        "latest_slot",
    ],
)
def test_statcan_full_evidence(mutation):
    spec, bodies = example("industrial_production")
    raw = json.loads(bodies["data"])
    obj = raw[0]["object"]
    row = obj["vectorDataPoint"][-1]
    if mutation == "coordinate":
        obj["coordinate"] = "1.1.1.1.0.0.0.0.0.0"
    elif mutation == "vector":
        obj["vectorId"] = 1
    elif mutation == "scale":
        row["scalarFactorCode"] = 0
    elif mutation == "status":
        row["statusCode"] = 99
    elif mutation == "security":
        row["securityLevelCode"] = 1
    elif mutation == "duplicate":
        obj["vectorDataPoint"].append(row)
    elif mutation == "period":
        row["refPer"] = "2026-06-02"
    elif mutation == "native_clock":
        row["releaseTime"] = "2026-09-12T08:30"
    else:
        obj["vectorDataPoint"].pop()
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize("mutation", ["geography", "seasonal", "base", "scope"])
def test_statcan_metadata_scope(mutation):
    spec, bodies = example("industrial_production")
    raw = json.loads(bodies["metadata"])
    dims = raw[0]["object"]["dimension"]
    pos = {"geography": 0, "seasonal": 1, "base": 2, "scope": 3}[mutation]
    member = next(m for m in dims[pos]["member"] if m["memberId"] == (10 if pos == 3 else 1))
    member["memberNameEn"] = "Other"
    bodies["metadata"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "mutation", ["series", "label", "duplicate", "date", "future", "nonfinite", "extra"]
)
def test_valet_identity_dates_and_status(mutation):
    spec, bodies = example("corporate_new_lending_rate")
    raw = json.loads(bodies["data"])
    key = "V122667819"
    if mutation == "series":
        raw["seriesDetail"]["V122667824"] = raw["seriesDetail"].pop(key)
    elif mutation == "label":
        raw["seriesDetail"][key]["label"] = "Outstanding loans"
    elif mutation == "duplicate":
        raw["observations"].append(raw["observations"][-1])
    elif mutation == "date":
        raw["observations"][-1]["d"] = "2026-06-02"
    elif mutation == "future":
        raw["observations"][-1]["d"] = "2026-10-01"
    elif mutation == "nonfinite":
        raw["observations"][-1][key]["v"] = "NaN"
    else:
        raw["observations"][-1][key]["flag"] = "unreviewed"
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_valet_null_is_retained():
    spec, bodies = example("corporate_new_lending_rate")
    raw = json.loads(bodies["data"])
    raw["observations"][-1]["V122667819"]["v"] = ""
    bodies["data"] = json.dumps(raw).encode()
    assert parse_input(spec, bodies)["observations"][-1]["value"] is None
