"""Original Norwegian source replay and native scope guards; no HTTP."""

import csv
import io
import json
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources.norway_monitoring import parse_input, specs

FIXTURES = Path(__file__).parent / "fixtures" / "norway_monitoring"
END = date(2026, 9, 10)


def capture(indicator):
    spec = next(item for item in specs(END) if item["indicator"] == indicator)
    suffixes = ("json", "json") if spec["kind"] == "ssb" else ("xml", "csv")
    bodies = {
        role: (FIXTURES / f"{indicator}_{role}.{suffix}").read_bytes()
        for role, suffix in zip(("metadata", "data"), suffixes, strict=True)
    }
    return spec, bodies


def mutate_json(bodies, role, change):
    value = json.loads(bodies[role])
    change(value)
    bodies[role] = json.dumps(value).encode()


@pytest.mark.parametrize(
    "indicator,value,period",
    [
        ("industrial_production", 107.2, "2026-07"),
        ("corporate_new_lending_rate", 6.3, "2026-07"),
        ("policy_rate", 4.25, "2026-09-09"),
        ("yield_10y", 4.489, "2026-09-09"),
    ],
)
def test_official_originals_replay_native_identity_and_latest(indicator, value, period):
    result = parse_input(*capture(indicator))
    assert result["country"] == "NO"
    assert result["indicator"] == indicator
    assert result["observations"][-1]["period"] == period
    assert result["observations"][-1]["value"] == value
    assert result["observations"][-1]["status"] == "observed"
    assert result["published_at"] is None
    assert result["publisher_metadata"]["metadata_sha256"]


def test_industry_excludes_extraction_but_not_petroleum_related_manufacturing():
    spec, bodies = capture("industrial_production")
    result = parse_input(spec, bodies)
    assert result["series_id"] == "07095/P103/Sesongjustert"
    assert result["unit"] == "index"
    assert result["adjustment"] == "seasonally_adjusted"
    assert result["publisher_metadata"]["native_adjustment"] == "WorkAndSes"
    assert result["publisher_metadata"]["base_period"] == "2021"
    assert result["publisher_updated_at"] == "2026-09-07T06:00:00+00:00"
    assert "Petroleum-related manufacturing remains included" in result["definition"]
    assert "extraction" in result["definition"]
    assert spec["capture_exclusion"]["timezone"] == "Europe/Oslo"


def test_new_corporate_repayment_loans_preserve_fee_currency_and_sample_differences():
    result = parse_input(*capture("corporate_new_lending_rate"))
    assert result["series_id"] == "10729/02/03/RenterNyeUtlan"
    assert result["unit"] == "per cent"
    assert result["publisher_metadata"]["currency"] == "NOK"
    assert result["publisher_metadata"]["dimensions"]["Sektor"]["code"] == "03"
    assert "repayment loans" in result["definition"]
    assert any("commissions" in limit for limit in result["limits"])
    assert "sample" in result["definition"]
    assert result["observations"][-1]["date"] == "2026-07-31"


def test_nb_does_not_turn_structure_generation_time_into_publication_clock():
    result = parse_input(*capture("yield_10y"))
    assert result["publisher_updated_at"] is None
    assert result["series_id"] == "GOVT_GENERIC_RATES/B.10Y.GBON"
    assert "nearest" in result["definition"]
    assert "mid-yield" in result["definition"]
    assert result["publisher_metadata"]["native_series"]["TENOR"] == "10Y"


@pytest.mark.parametrize("field", ["country", "unit", "definition", "requests", "comparison"])
def test_changed_source_spec_fails_closed(field):
    spec, bodies = capture("industrial_production")
    spec[field] = "wrong"
    with pytest.raises(ValueError, match="spec"):
        parse_input(spec, bodies)


@pytest.mark.parametrize("change", ["industry", "unit", "base", "adjustment", "time", "source"])
def test_ssb_rejects_other_scope_units_adjustment_partial_axis_and_source(change):
    spec, bodies = capture("industrial_production")

    def modify(raw):
        if change == "industry":
            raw["dimension"]["PKoder"]["category"]["index"] = {"P101": 0}
        elif change == "unit":
            raw["dimension"]["ContentsCode"]["category"]["unit"]["Sesongjustert"]["base"] = "%"
        elif change in {"base", "adjustment"}:
            name = "basePeriod" if change == "base" else "adjustment"
            raw["dimension"]["ContentsCode"]["extension"][name]["Sesongjustert"] = "wrong"
        elif change == "time":
            raw["value"].pop()
            raw["size"][-1] -= 1
            for key in ("index", "label"):
                raw["dimension"]["Tid"]["category"][key].pop("2026M07")
        else:
            raw["source"] = "Distributor"

    mutate_json(bodies, "data", modify)
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_ssb_null_sparse_cells_and_literal_zero_are_distinct():
    spec, bodies = capture("industrial_production")

    def modify(raw):
        raw["value"][-2:] = [0, None]
        raw["value"] = {str(i): v for i, v in enumerate(raw["value"]) if v is not None}
        raw["status"] = {str(raw["size"][-1] - 1): ".."}

    mutate_json(bodies, "data", modify)
    rows = parse_input(spec, bodies)["observations"]
    assert rows[-2]["value"] == 0.0
    assert rows[-2]["status"] == "observed"
    assert rows[-1]["period"] == "2026-07"
    assert rows[-1]["value"] is None
    assert rows[-1]["native_status"] == ".."


@pytest.mark.parametrize("bad", [True, "100", float("nan"), float("inf")])
def test_ssb_non_numeric_values_rejected(bad):
    spec, bodies = capture("industrial_production")
    mutate_json(bodies, "data", lambda raw: raw["value"].__setitem__(-1, bad))
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_ssb_observation_must_not_postdate_native_update():
    spec, bodies = capture("industrial_production")
    mutate_json(bodies, "data", lambda raw: raw.update(updated="2026-06-07T06:00:00Z"))
    with pytest.raises(ValueError, match="update"):
        parse_input(spec, bodies)


@pytest.mark.parametrize("change", ["scope", "duplicate", "forecast", "date", "value", "metadata"])
def test_nb_native_contract_failures_are_rejected(change):
    spec, bodies = capture("policy_rate")
    if change == "metadata":
        bodies["metadata"] = bodies["metadata"].replace(b'id="KPRA"', b'id="OTHER"')
    else:
        reader = csv.DictReader(io.StringIO(bodies["data"].decode()), delimiter=";")
        rows, fields = list(reader), reader.fieldnames
        if change == "scope":
            rows[-1]["INSTRUMENT_TYPE"] = "NOWA"
        elif change == "duplicate":
            rows.append(deepcopy(rows[-1]))
        elif change == "forecast":
            rows[-1]["CALC_METHOD"] = "E"
        elif change == "date":
            rows[-1]["TIME_PERIOD"] = "2026-09-11"
        else:
            rows[-1]["OBS_VALUE"] = "NaN"
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
        bodies["data"] = output.getvalue().encode()
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_nb_explicit_missing_cell_is_preserved():
    spec, bodies = capture("yield_10y")
    lines = bodies["data"].decode().splitlines()
    cells = lines[-1].split(";")
    cells[-1] = ""
    lines[-1] = ";".join(cells)
    bodies["data"] = ("\n".join(lines) + "\n").encode()
    assert parse_input(spec, bodies)["observations"][-1]["status"] == "not_reported"


def test_unknown_body_roles_and_duplicate_json_keys_are_rejected():
    spec, bodies = capture("industrial_production")
    bodies["extra"] = b"unexpected"
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
    bodies.pop("extra")
    bodies["data"] = bodies["data"].replace(b'"class":', b'"class":"dataset","class":', 1)
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
