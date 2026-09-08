"""Bank of Japan money-stock API semantics and parser tests."""

from __future__ import annotations

import hashlib
from datetime import date, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from dalio.data_sources.money_liquidity import (
    BOJ_BROADLY_DEFINED_LIQUIDITY,
    BOJ_M3,
    INDICATOR_BROADLY_DEFINED_LIQUIDITY,
    MONEY_LIQUIDITY_SERIES,
    SOURCE_BOJ_DATA,
    MoneyLiquiditySource,
    parse_boj_csv,
)


def _response(text: str):
    response = MagicMock()
    response.text = text
    response.status_code = 200
    response.raise_for_status.return_value = None
    return response


def _boj_csv(
    *,
    series_code: str = "MAM1NAM3M3MO",
    name: str = "M3/Average Amounts Outstanding/Money Stock",
    unit: str = "100 million yen",
    frequency: str = "MONTHLY",
    category: str = "Money Stock",
    status: str = "200",
    database: str = "MD02",
    next_position: str = "",
) -> str:
    return (
        f"STATUS,{status}\n"
        "MESSAGEID,M181000I\n"
        "MESSAGE,Successfully completed\n"
        "DATE,2026-08-12T08:50:00+09:00\n"
        "PARAMETER,FORMAT,CSV\n"
        "PARAMETER,LANG,EN\n"
        f"PARAMETER,DB,{database}\n"
        "PARAMETER,STARTDATE,\n"
        "PARAMETER,ENDDATE,\n"
        "PARAMETER,STARTPOSITION,\n"
        f"NEXTPOSITION,{next_position}\n"
        "SERIES_CODE,NAME_OF_TIME_SERIES,UNIT,FREQUENCY,CATEGORY,LAST_UPDATE,"
        "SURVEY_DATES,VALUES\n"
        f"{series_code},{name},{unit},{frequency},{category},20260812,200304,10121694\n"
        f"{series_code},{name},{unit},{frequency},{category},20260812,200305,null\n"
        f"{series_code},{name},{unit},{frequency},{category},20260812,200306,10135410\n"
    )


def test_boj_catalogue_pins_current_m3_and_broadly_defined_liquidity():
    assert BOJ_M3 in MONEY_LIQUIDITY_SERIES
    assert BOJ_BROADLY_DEFINED_LIQUIDITY in MONEY_LIQUIDITY_SERIES
    assert BOJ_M3.native_series_id == "MAM1NAM3M3MO"
    assert BOJ_BROADLY_DEFINED_LIQUIDITY.native_series_id == "MAM1NABLBLMO"
    assert BOJ_M3.indicator == "broad_money_m3_stock"
    assert BOJ_BROADLY_DEFINED_LIQUIDITY.indicator == INDICATOR_BROADLY_DEFINED_LIQUIDITY
    for spec in (BOJ_M3, BOJ_BROADLY_DEFINED_LIQUIDITY):
        assert spec.country == "JP"
        assert spec.currency == "JPY"
        assert spec.unit == "JPY 100 million"
        assert spec.native_unit == "100 million yen"
        assert spec.unit_multiplier == 100_000_000
        assert spec.frequency == "monthly"
        assert spec.adjustment == "not seasonally adjusted"
        assert spec.source_family == SOURCE_BOJ_DATA
        assert spec.expected_start == date(2003, 4, 1)
        assert spec.response_format == "boj_csv"
        assert "db=MD02" in spec.url
        assert f"code={spec.native_series_id}" in spec.url


def test_boj_source_fetches_exact_series_and_preserves_native_provenance(tmp_path):
    client = MagicMock()
    response_text = _boj_csv()
    client.get.return_value = _response(response_text)
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )

    frame = source.fetch(BOJ_M3, use_cache=False)

    client.get.assert_called_once()
    assert client.get.call_args.args[0] == BOJ_M3.url
    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2003, 4, 1), 10121694.0),
        (date(2003, 6, 1), 10135410.0),
    ]
    assert set(frame["country"]) == {"JP"}
    assert set(frame["source"]) == {SOURCE_BOJ_DATA}
    assert set(frame["series_id"]) == {BOJ_M3.native_series_id}
    assert frame.attrs["source_url"] == BOJ_M3.url
    assert frame.attrs["native_periods"] == ("200304", "200305", "200306")
    assert frame.attrs["native_period_format"] == "YYYYMM"
    assert frame.attrs["publisher_last_updated_on"] == date(2026, 8, 12)
    assert frame.attrs["api_generated_at"] == datetime.fromisoformat("2026-08-12T08:50:00+09:00")
    artifact_path = Path(frame.attrs["source_artifact_path"])
    assert artifact_path.read_bytes() == response_text.encode("utf-8")
    assert (
        frame.attrs["source_artifact_sha256"]
        == hashlib.sha256(response_text.encode("utf-8")).hexdigest()
    )
    assert frame.attrs["native_payload_artifact_path"] == str(artifact_path)
    assert frame.attrs["native_payload_sha256"] == frame.attrs["source_artifact_sha256"]
    assert frame.attrs["missing_period_records"] == (
        {
            "native_period": "200305",
            "native_position": 1,
            "missing_kind": "null_token",
            "native_token": "null",
            "reason": "publisher_null_not_zero",
            "evidence": "boj_csv_values_cell",
        },
    )
    missing_path = Path(frame.attrs["missing_provenance_artifact_path"])
    assert (
        hashlib.sha256(missing_path.read_bytes()).hexdigest()
        == frame.attrs["missing_provenance_sha256"]
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"status": "503"}, "status"),
        ({"database": "MD01"}, "database"),
        ({"next_position": "2"}, "truncated"),
        ({"series_code": "MAM1NAM2M2MO"}, "native series"),
        ({"name": "M2/Average Amounts Outstanding/Money Stock"}, "series name"),
        ({"unit": "million yen"}, "currency/unit"),
        ({"frequency": "QUARTERLY"}, "frequency"),
        ({"category": "Monetary Base"}, "category"),
    ],
)
def test_boj_parser_rejects_api_or_native_metadata_drift(kwargs, message):
    with pytest.raises(ValueError, match=message):
        parse_boj_csv(_boj_csv(**kwargs), BOJ_M3)


def test_boj_parser_rejects_malformed_period_update_and_value():
    valid = _boj_csv()
    with pytest.raises(ValueError, match="monthly period"):
        parse_boj_csv(valid.replace("200304", "2003-04", 1), BOJ_M3)
    with pytest.raises(ValueError, match="last-update"):
        parse_boj_csv(valid.replace("20260812,200304", "2026-08-12,200304", 1), BOJ_M3)
    with pytest.raises(ValueError, match="non-numeric"):
        parse_boj_csv(valid.replace("10121694", "not-a-number", 1), BOJ_M3)


def test_boj_parser_rejects_json_error_body_and_duplicate_periods():
    with pytest.raises(ValueError, match="CSV"):
        parse_boj_csv('{"STATUS":400,"MESSAGE":"Invalid parameters"}', BOJ_M3)

    duplicate = _boj_csv().replace("200306,10135410", "200304,10135410")
    with pytest.raises(ValueError, match="duplicate observation periods"):
        parse_boj_csv(duplicate, BOJ_M3)
