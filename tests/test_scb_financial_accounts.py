"""Offline coverage for the SCB government-debt-holder adapter."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest

from dalio.data_sources.scb_financial_accounts import (
    INSTRUMENT_CODES,
    ISSUER_SECTOR_CODE,
    LONG_COLUMNS,
    MEASURE_CODE,
    SCB_FINANCIAL_ACCOUNTS_TABLE_ID,
    SCB_GOVERNMENT_DEBT_HOLDERS,
    SCB_GOVERNMENT_DEBT_HOLDERS_URL,
    SOURCE_SCB_FINANCIAL_ACCOUNTS,
    ScbFinancialAccountsSource,
    parse_jsonstat,
)


def _sample_payload() -> dict:
    # Dimension and category insertion orders are intentionally unlike the
    # desired output order. JSON-stat positions, not dict order, are binding.
    return {
        "version": "2.0",
        "class": "dataset",
        "label": "Financial accounts, balances, SEK million",
        "id": ["Tid", "Motsektor", "ContentsCode", "Kontopost", "Sektor"],
        "size": [2, 2, 1, 3, 1],
        "dimension": {
            "Sektor": {
                "category": {
                    "index": {"S1311": 0},
                    "label": {"S1311": "Central government"},
                }
            },
            "Kontopost": {
                "category": {
                    "index": {"FL3200": 2, "FL3000": 0, "FL3100": 1},
                    "label": {
                        "FL3200": "Long-term debt securities",
                        "FL3000": "Debt securities",
                        "FL3100": "Short-term debt securities",
                    },
                }
            },
            "Motsektor": {
                "category": {
                    "index": {"S2": 1, "S0": 0},
                    "label": {"S2": "Rest of the world", "S0": "All sectors"},
                }
            },
            "ContentsCode": {
                "category": {
                    "index": {"FM0103AS": 0},
                    "label": {"FM0103AS": "Balances"},
                    "unit": {"FM0103AS": {"base": "SEK million", "decimals": 0}},
                }
            },
            "Tid": {
                "category": {
                    "index": {"2026K1": 1, "2025K4": 0},
                    "label": {"2026K1": "2026K1", "2025K4": "2025K4"},
                }
            },
        },
        # In declared dimension order: time, holder, measure, instrument, issuer.
        "value": [100, 40, 60, 30, 5, 25, 110, 45, 65, 35, None, 30],
    }


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def test_catalogue_exact_url_cache_and_typed_long_rows(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(json.dumps(_sample_payload()))
    source = ScbFinancialAccountsSource(client=client, cache_dir=tmp_path)

    first = source.fetch()
    second = source.fetch()

    assert client.get.call_count == 1
    assert first.equals(second)
    assert client.get.call_args.args[0] == SCB_GOVERNMENT_DEBT_HOLDERS_URL
    assert SCB_GOVERNMENT_DEBT_HOLDERS.url == SCB_GOVERNMENT_DEBT_HOLDERS_URL
    assert SCB_FINANCIAL_ACCOUNTS_TABLE_ID == "TAB1203"
    assert ISSUER_SECTOR_CODE == "S1311"
    assert INSTRUMENT_CODES == ("FL3000", "FL3100", "FL3200")
    assert MEASURE_CODE == "FM0103AS"
    assert SCB_GOVERNMENT_DEBT_HOLDERS_URL == (
        "https://api.scb.se/OV0104/v2beta/api/v2/tables/TAB1203/data"
        "?valueCodes%5BSektor%5D=S1311"
        "&valueCodes%5BKontopost%5D=FL3000%2CFL3100%2CFL3200"
        "&valueCodes%5BMotsektor%5D=%2A"
        "&valueCodes%5BContentsCode%5D=FM0103AS"
        "&valueCodes%5BTid%5D=%2A"
        "&outputFormat=json-stat2&lang=en"
    )
    assert list(first.columns) == LONG_COLUMNS
    assert first["value"].dtype == "float64"
    assert all(isinstance(value, date) for value in first["date"])
    assert set(first["country"]) == {"SE"}
    assert set(first["issuer_sector_code"]) == {"S1311"}
    assert set(first["measure_code"]) == {"FM0103AS"}
    assert set(first["measure_label"]) == {"Balances"}
    assert set(first["unit"]) == {"SEK million"}
    assert set(first["source"]) == {SOURCE_SCB_FINANCIAL_ACCOUNTS}
    assert set(first["status"]) == {"observed"}


def test_jsonstat_declared_order_is_authoritative_and_nulls_are_omitted():
    frame = parse_jsonstat(json.dumps(_sample_payload()))

    assert len(frame) == 11
    selected = frame.loc[
        (frame["date"] == date(2025, 10, 1))
        & (frame["instrument_code"] == "FL3100")
        & (frame["holder_sector_code"] == "S2")
    ].iloc[0]
    assert selected["value"] == 5.0
    assert selected["issuer_sector_label"] == "Central government"
    assert selected["instrument_label"] == "Short-term debt securities"
    assert selected["holder_sector_label"] == "Rest of the world"
    assert selected["series_id"] == "TAB1203/S1311/FL3100/S2/FM0103AS"

    missing_key = (
        (frame["date"] == date(2026, 1, 1))
        & (frame["instrument_code"] == "FL3100")
        & (frame["holder_sector_code"] == "S2")
    )
    assert not missing_key.any()


def test_sparse_jsonstat_values_preserve_flat_positions():
    payload = _sample_payload()
    payload["value"] = {"0": 100, "11": 30}

    frame = parse_jsonstat(json.dumps(payload))

    assert list(frame["date"]) == [date(2025, 10, 1), date(2026, 1, 1)]
    assert list(frame["instrument_code"]) == ["FL3000", "FL3200"]
    assert list(frame["holder_sector_code"]) == ["S0", "S2"]
    assert list(frame["value"]) == [100.0, 30.0]


@pytest.mark.parametrize("bad_value", ["NaN", "Infinity", True, "not-a-number"])
def test_non_finite_and_non_numeric_values_fail_closed(bad_value):
    payload = _sample_payload()
    payload["value"][0] = bad_value

    with pytest.raises(ValueError, match="non-finite|non-numeric"):
        parse_jsonstat(json.dumps(payload))


def test_duplicate_dimension_positions_and_size_mismatch_fail_closed():
    duplicate = _sample_payload()
    duplicate["dimension"]["Motsektor"]["category"]["index"] = {"S0": 0, "S2": 0}
    with pytest.raises(ValueError, match="duplicate category positions"):
        parse_jsonstat(json.dumps(duplicate))

    mismatched = _sample_payload()
    mismatched["value"] = mismatched["value"][:-1]
    with pytest.raises(ValueError, match="size mismatch"):
        parse_jsonstat(json.dumps(mismatched))


def test_selection_leak_invalid_json_and_empty_values_fail_closed():
    leaked = _sample_payload()
    leaked["dimension"]["Sektor"]["category"]["index"] = {"S13": 0}
    leaked["dimension"]["Sektor"]["category"]["label"] = {"S13": "General government"}
    with pytest.raises(ValueError, match="outside issuer S1311"):
        parse_jsonstat(json.dumps(leaked))

    with pytest.raises(ValueError, match="invalid JSON"):
        parse_jsonstat("not-json")

    empty = _sample_payload()
    empty["value"] = [None] * len(empty["value"])
    with pytest.raises(ValueError, match="no usable observations"):
        parse_jsonstat(json.dumps(empty))


def test_404_fails_fast_without_retry(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("missing", 404)

    with pytest.raises(ValueError, match="404"):
        ScbFinancialAccountsSource(client=client, cache_dir=tmp_path).fetch(use_cache=False)

    assert client.get.call_count == 1


def test_missing_labels_or_unit_fail_closed():
    missing_label = deepcopy(_sample_payload())
    del missing_label["dimension"]["Motsektor"]["category"]["label"]["S2"]
    with pytest.raises(ValueError, match="missing label"):
        parse_jsonstat(json.dumps(missing_label))

    missing_unit = deepcopy(_sample_payload())
    del missing_unit["dimension"]["ContentsCode"]["category"]["unit"]
    with pytest.raises(ValueError, match="missing unit"):
        parse_jsonstat(json.dumps(missing_unit))


def test_no_duplicate_long_dimension_keys():
    frame = parse_jsonstat(json.dumps(_sample_payload()))
    key = [
        "date",
        "issuer_sector_code",
        "instrument_code",
        "holder_sector_code",
        "measure_code",
    ]
    assert not frame.duplicated(key).any()
    assert pd.api.types.is_float_dtype(frame["value"])
