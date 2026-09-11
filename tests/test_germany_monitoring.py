"""Offline replay of original German and ECB evidence and definition guards."""

from copy import deepcopy
from datetime import date
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from dalio.data_sources.germany_monitoring import parse_input, specs

FIXTURES = Path(__file__).parent / "fixtures" / "germany_monitoring"
END = date(2026, 9, 11)
G = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/data/generic}"
M = "{http://www.sdmx.org/resources/sdmxml/schemas/v2_1/message}"
PREFIX = {
    "industrial_production": "industry",
    "corporate_new_lending_rate": "loan",
    "policy_rate": "policy",
    "yield_10y": "yield",
}


def capture(indicator):
    spec = next(s for s in specs(END) if s["indicator"] == indicator)
    return spec, {
        r["role"]: (FIXTURES / f"{PREFIX[indicator]}_{r['role']}.bin").read_bytes()
        for r in spec["requests"]
    }


@pytest.mark.parametrize(
    "indicator,value,period",
    [
        ("industrial_production", 90.1, "2026-07"),
        ("corporate_new_lending_rate", 3.78, "2026-07"),
        ("policy_rate", 2.25, "2026-09-11"),
        ("yield_10y", 3.45, "2026-09-10"),
    ],
)
def test_original_series_replay(indicator, value, period):
    parsed = parse_input(*capture(indicator))
    latest = parsed["observations"][-1]
    assert (latest["period"], latest["value"]) == (period, value)
    assert parsed["country"] == "DE"
    assert latest["status"] == "observed"
    assert parsed["published_at"] is None
    assert parsed["publisher_metadata"]["raw_sha256"]


def test_scopes_clocks_and_source_sunset_are_explicit():
    industry = parse_input(*capture("industrial_production"))
    assert industry["adjustment"] == "seasonally_adjusted"
    assert (
        industry["publisher_metadata"]["native_adjustment"]
        == "X13 JDemetra+, calendar and seasonal"
    )
    assert industry["publisher_metadata"]["report_updated_date"] == "2026-09-04"
    assert industry["publisher_updated_at"] is None
    assert "except energy and construction" in industry["definition"]
    assert any("October 2026" in line for line in industry["limits"])
    loan = parse_input(*capture("corporate_new_lending_rate"))
    assert "euro-area" in loan["definition"]
    assert "renegotiated" in loan["definition"]
    assert (
        "Excluding overdrafts"
        in loan["publisher_metadata"]["native_attributes"]["BBK_COMM_GEN_ENG"]
    )
    assert loan["observations"][-1]["native_status"] == "P"
    assert loan["publisher_updated_at"] == "2026-09-02T09:57:47.945000+00:00"
    assert loan["observations"][-1]["date"] == "2026-07-31"
    policy = parse_input(*capture("policy_rate"))
    assert "Germany" in policy["publisher_metadata"]["applicable_country"]
    assert "shared euro-area" in policy["definition"]
    benchmark = parse_input(*capture("yield_10y"))
    assert "current 10 year federal bond" in benchmark["definition"]
    assert any("constant-maturity" in line for line in benchmark["limits"])
    assert benchmark["publisher_metadata"]["native_attributes"]["BBK_UNIT"] == "PROZENT"


@pytest.mark.parametrize("field", ["country", "requests", "unit", "definition", "comparison"])
def test_changed_descriptor_rejected(field):
    spec, bodies = capture("industrial_production")
    spec[field] = "other"
    with pytest.raises(ValueError, match="contract"):
        parse_input(spec, bodies)


@pytest.mark.parametrize(
    "old,new",
    [
        (b"2021=100", b"2015=100"),
        (b"X 13 JDemetra+", b"BV 4.1"),
        (b"01/07/2026", b"02/07/2026"),
        (b"90,1;-2,2", b"nan;-2,2"),
        (b"90,1;-2,2", b"99,1;-2,2"),
    ],
)
def test_changed_industry_data_rejected(old, new):
    spec, bodies = capture("industrial_production")
    assert old in bodies["data"]
    bodies["data"] = bodies["data"].replace(old, new)
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_industry_metadata_scope_and_duplicate_dates_rejected():
    spec, bodies = capture("industrial_production")
    bodies["metadata"] = bodies["metadata"].replace(
        b"except energy and construction", b"including construction"
    )
    with pytest.raises(ValueError):
        parse_input(spec, bodies)
    spec, bodies = capture("industrial_production")
    bodies["data"] += bodies["data"].splitlines(keepends=True)[-1]
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def alter_xml(indicator, change):
    spec, bodies = capture(indicator)
    root = ET.fromstring(bodies["data"])
    change(root)
    bodies["data"] = ET.tostring(root)
    return spec, bodies


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("BBK_UNIT", "EUR"),
        ("BBK_UNIT_MULT", "6"),
        ("BBK_TITLE_ENG", "Loans to households"),
        ("BBK_ID", "BBIM1.other"),
        ("TIME_FORMAT", "P1Y"),
    ],
)
def test_bundesbank_native_attributes_pinned(attribute, value):
    def change(root):
        root.find(f".//{G}Attributes/{G}Value[@id='{attribute}']").set("value", value)

    with pytest.raises(ValueError):
        parse_input(*alter_xml("corporate_new_lending_rate", change))


@pytest.mark.parametrize(
    "change",
    [
        "duplicate_dimension",
        "wrong_dimension",
        "duplicate_observation",
        "future",
        "naive_clock",
        "forecast",
    ],
)
def test_bundesbank_identity_and_temporal_guards(change):
    def mutate(root):
        series = root.find(f".//{G}Series")
        axis = series.find(G + "SeriesKey")
        obs = series.findall(G + "Obs")[-1]
        if change == "duplicate_dimension":
            axis.append(deepcopy(axis[0]))
        elif change == "wrong_dimension":
            axis[1].set("value", "FR")
        elif change == "duplicate_observation":
            series.append(deepcopy(obs))
        elif change == "future":
            obs.find(G + "ObsDimension").set("value", "2026-12")
        elif change == "naive_clock":
            root.find(M + "DataSet").set("validFromDate", "2026-09-02T11:57:47")
        elif change == "forecast":
            obs.find(f"{G}Attributes/{G}Value[@id='OBS_STATUS']").set("value", "F")

    with pytest.raises(ValueError):
        parse_input(*alter_xml("corporate_new_lending_rate", mutate))


def test_bundesbank_native_nulls_preserved_and_zero_not_missing():
    parsed = parse_input(*capture("yield_10y"))
    missing = [r for r in parsed["observations"] if r["value"] is None]
    assert missing and all(r["native_status"] == "K" for r in missing)
    assert all(r["status"] == "not_reported" for r in missing)

    def change(root):
        root.findall(f".//{G}Obs")[-1].find(G + "ObsValue").set("value", "0")

    zero = parse_input(*alter_xml("yield_10y", change))["observations"][-1]
    assert zero["value"] == 0.0 and zero["status"] == "observed"


def test_value_with_native_missing_status_is_rejected():
    def change(root):
        missing = next(o for o in root.findall(f".//{G}Obs") if o.find(G + "ObsValue") is None)
        ET.SubElement(missing, G + "ObsValue", value="1.0")

    with pytest.raises(ValueError):
        parse_input(*alter_xml("yield_10y", change))


@pytest.mark.parametrize("old,new", [(b"PCPA", b"EUR"), (b"U2", b"DE"), (b",A,F,", b",F,F,")])
def test_ecb_native_identity_unit_and_forecast_rejected(old, new):
    spec, bodies = capture("policy_rate")
    assert old in bodies["data"]
    bodies["data"] = bodies["data"].replace(old, new)
    with pytest.raises(ValueError):
        parse_input(spec, bodies)


def test_response_beyond_descriptor_date_rejected():
    _, bodies = capture("yield_10y")
    earlier = next(s for s in specs(date(2026, 9, 9)) if s["indicator"] == "yield_10y")
    with pytest.raises(ValueError):
        parse_input(earlier, bodies)


def test_date_only_report_update_after_capture_rejected():
    spec, bodies = capture("industrial_production")
    bodies["metadata"] = bodies["metadata"].replace(b"September 4, 2026", b"September 12, 2026")
    with pytest.raises(ValueError, match="report date"):
        parse_input(spec, bodies)


@pytest.mark.parametrize("tag", ["ObsDimension", "Attributes"])
def test_duplicate_native_observation_metadata_rejected(tag):
    def change(root):
        obs = root.findall(f".//{G}Obs")[-1]
        obs.append(deepcopy(obs.find(G + tag)))

    with pytest.raises(ValueError, match="observation metadata"):
        parse_input(*alter_xml("corporate_new_lending_rate", change))
