"""Offline tests for the original Bank of Finland benchmark report."""

import gzip
from copy import deepcopy
from datetime import date
from pathlib import Path

import pytest

from dalio.data_sources import finland_yield_monitoring as source

END = date(2026, 9, 10)
FIXTURE = Path(__file__).parent / "fixtures/finland_yield_monitoring/benchmark_report.html.gz"


def sample(rows=None, *, header=None, updated="10 Sep 2026"):
    rows = rows or [("10.9.2026", "3.74"), ("12.6.2026", "3.33")]
    title = header or "Yield on goverment bonds, 10 year"
    table = f"<table><tr><td></td><td>Period</td><td>Yield on goverment bonds, 5 year</td><td>{title}</td></tr>"
    for period, value in rows:
        table += f"<tr><td></td><td>{period}</td><td>2.00</td><td>{value}</td></tr>"
    return (
        "<html><body><h1>Yields on Finnish benchmark government bonds</h1>"
        "<p>Statistics on the yields on Finnish government benchmark bonds, issued by the Bank of Finland, "
        "calculated from LSEG's data on primary dealers' daily average selling prices of debt instruments "
        "as of 13.00 hours.</p>" + table + f"</table><p>Report updated {updated}</p></body></html>"
    ).encode()


def parse(raw=None, spec=None):
    return source.parse_input(spec or source.specs(END)[0], {"data": raw or sample()})


def test_original_report_all_daily_rows_and_exact_anchor():
    result = parse(gzip.decompress(FIXTURE.read_bytes()))
    rows = result["observations"]
    assert len(rows) == 174
    assert rows[0]["date"] == "2026-01-02"
    assert rows[-1]["date"] == "2026-09-10"
    assert rows[-1]["value"] == 3.74
    assert next(r["value"] for r in rows if r["date"] == "2026-06-12") == 3.33
    assert result["unit"] == "percent"
    assert result["frequency"] == "daily"
    assert result["published_at"] is None
    assert result["publisher_updated_at"] is None
    assert result["publisher_metadata"]["report_updated_date"] == "2026-09-10"
    assert result["publisher_metadata"]["update_precision"] == "date"
    assert result["publisher_metadata"]["underlying_source"] == "LSEG"
    assert "13:00" in result["definition"]
    assert "current-year" in " ".join(result["limits"])
    assert "not a separate literal unit field" in " ".join(result["limits"])
    assert all(r["period_start"] == r["period_end"] == r["date"] for r in rows)


def test_source_column_not_an_individual_security_or_five_year():
    result = parse()
    assert result["observations"][-1]["value"] == 3.74
    assert result["publisher_metadata"]["native_column"] == "Yield on goverment bonds, 10 year"
    with pytest.raises(ValueError):
        parse(sample(header="Loan period 16 Apr 2026 – 15 Sep 2036"))


@pytest.mark.parametrize(
    "value,expected", [("&nbsp;", None), ("", None), ("0.00", 0.0), ("-0.25", -0.25)]
)
def test_blank_zero_and_negative_remain_distinct(value, expected):
    result = parse(sample([("10.9.2026", value)]))
    row = result["observations"][0]
    assert row["value"] == expected
    assert row["status"] == ("not_reported" if expected is None else "observed")
    assert row["native_status"] is None
    assert len(result["missingness"]) == (1 if expected is None else 0)


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "1_2", "3.74*", "3.74 forecast", "3,74", "3.74%"]
)
def test_unexpected_value_or_status_fails_closed(value):
    with pytest.raises(ValueError):
        parse(sample([("10.9.2026", value)]))


@pytest.mark.parametrize(
    "rows",
    [
        [("10.9.2026", "3.74"), ("10.9.2026", "3.72")],
        [("11.9.2026", "3.74")],
        [("2026", "3.74")],
        [("31.2.2026", "3.74")],
    ],
)
def test_duplicate_future_non_daily_or_invalid_dates_rejected(rows):
    with pytest.raises(ValueError):
        parse(sample(rows))


def test_observation_cannot_postdate_report_update():
    with pytest.raises(ValueError):
        parse(sample(updated="9 Sep 2026"))


def test_native_report_date_cannot_postdate_descriptor_end():
    with pytest.raises(ValueError):
        parse(sample(updated="11 Sep 2026"))


@pytest.mark.parametrize(
    "old,new",
    [
        (b"Finnish benchmark", b"German benchmark"),
        (b"LSEG's", b"Unknown's"),
        (b"13.00 hours", b"16.00 hours"),
        (b"Report updated 10 Sep 2026", b"Unknown update date"),
        (b"<td>3.74</td>", b"<td colspan='2'>3.74</td>"),
        (b"<td>3.74</td>", b""),
    ],
)
def test_definition_or_table_layout_drift_rejected(old, new):
    with pytest.raises(ValueError):
        parse(sample().replace(old, new))


def test_ambiguous_two_tables_rejected():
    body = sample()
    with pytest.raises(ValueError):
        parse(body + body)


@pytest.mark.parametrize(
    "key,value", [("country", "SE"), ("unit", "basis points"), ("comparison", "three_month_rate")]
)
def test_descriptor_pins_source_semantics(key, value):
    spec = deepcopy(source.specs(END)[0])
    spec[key] = value
    with pytest.raises(ValueError):
        parse(spec=spec)


def test_exact_original_report_url_and_body_role_required():
    spec = source.specs(END)[0]
    assert spec["requests"] == [{"role": "data", "method": "GET", "url": source.REPORT_URL}]
    bad = deepcopy(spec)
    bad["requests"][0]["url"] = "https://example.com/report"
    with pytest.raises(ValueError):
        parse(spec=bad)
    with pytest.raises(ValueError):
        source.parse_input(spec, {"data": sample(), "metadata": sample()})
