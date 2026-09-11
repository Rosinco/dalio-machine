import json
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.finland_monitoring import parse_input, specs

FIX = Path(__file__).parent / "fixtures" / "finland_monitoring"
END = date(2026, 9, 11)


def example(index):
    name = ["industry", "lending", "policy"][index]
    spec = specs(END)[index]
    return spec, {
        r["role"]: (FIX / f"{name}_{r['role']}.bin").read_bytes() for r in spec["requests"]
    }


@pytest.mark.parametrize("index,count,last", [(0, 379, 104.8), (1, 194, None), (2, 7923, 2.25)])
def test_original_full_responses(index, count, last):
    result = parse_input(*example(index))
    assert len(result["observations"]) == count
    if last is not None:
        assert result["observations"][-1]["value"] == last
    assert result["observations"][-1]["period"]
    assert result["published_at"] is None


def test_scopes_do_not_conflate_finland_and_euro_area():
    industry = parse_input(*example(0))
    lending = parse_input(*example(1))
    policy = parse_input(*example(2))
    assert "BCD" in industry["definition"] and "energy" in industry["definition"]
    assert "domestic" in lending["definition"] and "housing" in lending["definition"]
    assert "All currencies" in lending["definition"]
    assert "drawdown" in lending["definition"]
    assert policy["publisher_metadata"]["policy_area"] == "Euro area (changing composition)"
    assert "not a separate Finnish" in policy["definition"]


@pytest.mark.parametrize("mutation", ["axis", "scope", "unit", "base", "status", "source"])
def test_industry_native_identity(mutation):
    spec, bodies = example(0)
    raw = json.loads(bodies["data"])
    if mutation == "axis":
        raw["dimension"]["timeperiod_m"]["category"]["index"].pop("1995M01")
    elif mutation == "scope":
        raw["dimension"]["toimiala_78_20180201"]["category"]["label"]["BTD"] = "Manufacturing only"
    elif mutation == "unit":
        raw["dimension"]["contentscode"]["category"]["unit"]["ttvi-Kausitasoitettu"]["base"] = "USD"
    elif mutation == "base":
        raw["label"] = raw["label"].replace("2021=100", "2015=100")
    elif mutation == "source":
        raw["source"] = "Other"
    else:
        raw["status"] = {"0": "unreviewed"}
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "mutation",
    ["geography", "housing", "scale", "name", "pages", "duplicate", "period", "nonfinite"],
)
def test_lending_native_identity_and_full_pages(mutation):
    spec, bodies = example(1)
    role = "metadata" if mutation in ("geography", "housing", "scale") else "data"
    raw = json.loads(bodies[role])
    item = raw["items"][0]
    if mutation == "geography":
        item["dimensions"][7]["value"] = "U2"
    elif mutation == "housing":
        item["dimensions"][8]["value"] = "2240"
    elif mutation == "scale":
        next(x for x in item["metadatas"] if x["name"] == "UNIT_MULT")["value"] = "2"
    elif mutation == "name":
        item["name"] = "Other"
    elif mutation == "pages":
        raw["totalPages"] = 2
    elif mutation == "duplicate":
        item["observations"].append(item["observations"][0])
    elif mutation == "period":
        item["observations"][0]["period"] = "2010-06-01"
    else:
        item["observations"][0]["value"] = float("inf")
    bodies[role] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "old,new",
    [
        ("FM.D.U2.EUR.4F.KR.DFR.LEV", "FM.D.FI.EUR.4F.KR.DFR.LEV"),
        (",PCPA,", ",EUR,"),
        (",A,F,", ",X,F,"),
    ],
)
def test_ecb_native_identity_status_units(old, new):
    spec, bodies = example(2)
    bodies["data"] = bodies["data"].replace(old.encode(), new.encode())
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_null_bof_value_remains_missing():
    spec, bodies = example(1)
    raw = json.loads(bodies["data"])
    raw["items"][0]["observations"][-1]["value"] = None
    bodies["data"] = json.dumps(raw).encode()
    assert parse_input(spec, bodies)["observations"][-1]["status"] == "not_reported"
