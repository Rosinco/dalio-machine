import copy
import json
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.denmark_monitoring import parse_input, specs

FIX = Path(__file__).parent / "fixtures" / "denmark_monitoring"
END = date(2026, 9, 11)
NAMES = ["industry", "lending", "policy", "yield"]


def example(index):
    return specs(END)[index], {
        role: (FIX / f"{NAMES[index]}_{role}.bin").read_bytes() for role in ("metadata", "data")
    }


@pytest.mark.parametrize(
    "index,count,last", [(0, 67, 148.9), (1, 283, 3.383), (2, 671, 1.85), (3, 499, 2.98)]
)
def test_original_full_responses(index, count, last):
    spec, bodies = example(index)
    result = parse_input(spec, bodies)
    assert len(result["observations"]) == count
    assert result["observations"][-1]["value"] == last
    assert result["observations"][-1]["period"]
    assert result["published_at"] is None
    assert result["publisher_updated_at"].endswith("+00:00")
    assert result["missingness"]["total"] == count


def test_native_scopes_revision_and_missingness():
    industry = parse_input(*example(0))
    lending = parse_input(*example(1))
    policy = parse_input(*example(2))
    yield_ = parse_input(*example(3))
    assert "DB25" in industry["definition"]
    assert "DKK" in lending["definition"] and "repo" in lending["definition"]
    assert lending["missingness"]["not_reported"] == 129
    assert policy["frequency"] == "daily"
    assert "Certificates of deposit" in policy["label"]
    assert yield_["comparison"] == "three_month_rate"
    assert "2021M1" in yield_["publisher_metadata"]["footnote"]["text"]
    assert any("August 2026" in note for note in yield_["limits"])
    assert any(row["value"] == 0 for row in yield_["observations"])


@pytest.mark.parametrize(
    "mutation",
    [
        "selection",
        "unit",
        "time",
        "clock",
        "table",
        "source",
        "status",
        "nonfinite",
        "extra_dimension",
    ],
)
def test_reject_wrong_native_response(mutation):
    spec, bodies = example(0)
    raw = json.loads(bodies["data"])
    data = raw["dataset"]
    if mutation == "selection":
        data["dimension"]["SÆSON"]["category"]["label"]["SÆSON"] = "Original"
    elif mutation == "unit":
        data["dimension"]["ContentsCode"]["category"]["unit"]["IPOP21"]["base"] = "USD"
    elif mutation == "time":
        data["dimension"]["Tid"]["category"]["index"].pop("2021M01")
    elif mutation == "clock":
        data["updated"] = "2026-09-09T06:00:00Z"
    elif mutation == "table":
        data["extension"]["px"]["tableid"] = "OTHER"
    elif mutation == "source":
        data["source"] = "Other publisher"
    elif mutation == "status":
        data["status"] = {"0": "unknown"}
    elif mutation == "nonfinite":
        data["value"][0] = float("nan")
    else:
        data["dimension"]["rogue"] = {}
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_contract_is_fixed_and_complete():
    spec, bodies = example(1)
    spec = copy.deepcopy(spec)
    spec["definition"] = "Other rate"
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
    spec, bodies = example(1)
    bodies.pop("metadata")
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_null_retained_but_finite_with_missing_status_rejected():
    spec, bodies = example(0)
    raw = json.loads(bodies["data"])
    raw["dataset"]["value"][0] = None
    raw["dataset"]["status"] = {"0": ".."}
    bodies["data"] = json.dumps(raw).encode()
    assert parse_input(spec, bodies)["observations"][0]["value"] is None
    raw["dataset"]["value"][0] = 0
    bodies["data"] = json.dumps(raw).encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
