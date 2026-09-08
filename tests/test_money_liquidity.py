"""Money/liquidity source catalogue, parsers, and immutable-release pipeline."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.data_sources.money_liquidity import (
    BOE_M4,
    BOE_M4EX,
    BOE_M4EX_QUARTERLY,
    BOJ_BROADLY_DEFINED_LIQUIDITY,
    BOJ_M3,
    ECB_EUROSYSTEM_ASSETS,
    ECB_M3,
    FED_M2,
    FED_TOTAL_ASSETS,
    MONEY_LIQUIDITY_SERIES,
    SCB_M3,
    MoneyLiquiditySource,
    money_liquidity_catalogue_sha256,
    parse_boe_csv,
    parse_ecb_csv,
    parse_scb_jsonstat,
)
from dalio.pipelines.fetch_money_liquidity import run_pipeline
from dalio.storage.db import DataRelease, DataReleaseArtifact, Observation, make_engine
from dalio.storage.releases import load_vintage_panel, make_partition_key, release_history

_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _response(text: str, status_code: int = 200):
    response = MagicMock()
    response.text = text
    response.status_code = status_code
    response.raise_for_status.return_value = None
    return response


def _boe_payload(
    spec,
    observations: str,
    *,
    data_header: str | None = None,
    description: str | None = None,
) -> str:
    return (
        "SERIES,DESCRIPTION\n"
        f"{spec.native_series_id},{description or spec.title}\n\n"
        f"DATE,{data_header or spec.native_series_id}\n"
        f"{observations}"
    )


def _at(month: int) -> datetime:
    return datetime(2026, month, 10, 12, tzinfo=UTC)


def _small_history_spec(
    spec,
    *,
    expected_start: date,
    max_latest_lag_days: int = 366,
):
    """Relax only production volume/recency bounds for tiny pipeline fixtures."""
    return replace(
        spec,
        expected_start=expected_start,
        minimum_observations=1,
        max_latest_lag_days=max_latest_lag_days,
    )


def _scb_payload() -> dict:
    return {
        "version": "2.0",
        "class": "dataset",
        "label": "Outstanding money supply, SEK millions by monetary aggregate and month",
        "source": "The Riksbank",
        "updated": "2026-08-27T06:00:00Z",
        "role": {"time": ["Tid"], "metric": ["ContentsCode"]},
        "id": ["Penningm", "ContentsCode", "Tid"],
        "size": [1, 1, 3],
        "dimension": {
            "Penningm": {
                "category": {
                    "index": {"5LLM3a.1E.NEP.V.A": 0},
                    "label": {"5LLM3a.1E.NEP.V.A": "M3"},
                }
            },
            "ContentsCode": {
                "category": {
                    "index": {"000007WQ": 0},
                    "label": {"000007WQ": "Outstanding money supply, SEK millions"},
                    "unit": {"000007WQ": {"base": "SEK millions", "decimals": 0}},
                },
                "extension": {
                    "refperiod": {"000007WQ": "Month"},
                    "measuringType": {"000007WQ": "Stock"},
                    "priceType": {"000007WQ": "Current"},
                    "adjustment": {"000007WQ": "None"},
                },
            },
            "Tid": {"category": {"index": {"2026M03": 2, "2026M01": 0, "2026M02": 1}}},
        },
        "value": [4900000, None, 5100000],
    }


def test_catalogue_pins_exact_native_series_currency_units_and_frequency():
    assert MONEY_LIQUIDITY_SERIES == (
        FED_M2,
        FED_TOTAL_ASSETS,
        ECB_M3,
        ECB_EUROSYSTEM_ASSETS,
        SCB_M3,
        BOE_M4EX,
        BOE_M4EX_QUARTERLY,
        BOE_M4,
        BOJ_M3,
        BOJ_BROADLY_DEFINED_LIQUIDITY,
    )
    assert {
        spec.native_series_id: (
            spec.country,
            spec.currency,
            spec.unit,
            spec.unit_multiplier,
            spec.frequency,
        )
        for spec in MONEY_LIQUIDITY_SERIES
    } == {
        "M2SL": ("US", "USD", "USD billion", 1_000_000_000, "monthly"),
        "WALCL": ("US", "USD", "USD million", 1_000_000, "weekly"),
        "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E": (
            "EU",
            "EUR",
            "EUR million",
            1_000_000,
            "monthly",
        ),
        "ILM.W.U2.C.T000000.Z5.Z01": (
            "EU",
            "EUR",
            "EUR million",
            1_000_000,
            "weekly",
        ),
        "TAB6541/5LLM3a.1E.NEP.V.A/000007WQ": (
            "SE",
            "SEK",
            "SEK million",
            1_000_000,
            "monthly",
        ),
        "RPMB53Q": ("UK", "GBP", "GBP million", 1_000_000, "monthly"),
        "RPQB53Q": ("UK", "GBP", "GBP million", 1_000_000, "quarterly"),
        "LPMAUYN": ("UK", "GBP", "GBP million", 1_000_000, "monthly"),
        "MAM1NAM3M3MO": (
            "JP",
            "JPY",
            "JPY 100 million",
            100_000_000,
            "monthly",
        ),
        "MAM1NABLBLMO": (
            "JP",
            "JPY",
            "JPY 100 million",
            100_000_000,
            "monthly",
        ),
    }
    assert FED_M2.url == "https://fred.stlouisfed.org/graph/fredgraph.csv?id=M2SL"
    assert ECB_M3.url == (
        "https://data-api.ecb.europa.eu/service/data/BSI/"
        "M.U2.Y.V.M30.X.1.U2.2300.Z01.E?format=csvdata"
    )
    assert SCB_M3.url == (
        "https://api.scb.se/ov0104/v2beta/api/v2/tables/TAB6541/data?"
        "valueCodes%5BPenningm%5D=5LLM3a.1E.NEP.V.A&"
        "valueCodes%5BContentsCode%5D=000007WQ&valueCodes%5BTid%5D=%2A&"
        "outputFormat=json-stat2&lang=en"
    )
    assert BOE_M4EX.url == (
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jul/2009&Dateto=now&"
        "SeriesCodes=RPMB53Q&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    )
    assert BOE_M4.url == (
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Jun/1982&Dateto=now&"
        "SeriesCodes=LPMAUYN&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    )
    assert BOE_M4EX_QUARTERLY.url == (
        "https://www.bankofengland.co.uk/boeapps/database/"
        "_iadb-fromshowcolumns.asp?csv.x=yes&Datefrom=01/Oct/1997&Dateto=now&"
        "SeriesCodes=RPQB53Q&CSVF=TT&UsingCodes=Y&VPD=Y&VFD=N"
    )
    assert BOE_M4EX.expected_start == date(2009, 7, 1)
    assert BOE_M4EX_QUARTERLY.expected_start == date(1997, 10, 1)
    assert BOE_M4.expected_start == date(1982, 6, 1)
    assert BOE_M4EX.research_role == "primary"
    assert BOE_M4EX_QUARTERLY.research_role == "diagnostic_bridge"
    assert BOE_M4.research_role == "diagnostic"
    assert BOE_M4EX_QUARTERLY.parent_native_series_id == BOE_M4EX.native_series_id
    assert set(BOE_M4EX.non_additive_groups) & set(BOE_M4EX_QUARTERLY.non_additive_groups)
    assert all(spec.perimeter and spec.definition_notes for spec in MONEY_LIQUIDITY_SERIES)
    assert all(spec.non_additive_groups for spec in MONEY_LIQUIDITY_SERIES)
    # No stable machine-readable Riksbank total-assets series has been verified.
    assert not any(
        spec.country == "SE" and spec.indicator == "central_bank_total_assets"
        for spec in MONEY_LIQUIDITY_SERIES
    )

    digest = money_liquidity_catalogue_sha256()
    assert len(digest) == 64
    assert digest == money_liquidity_catalogue_sha256(tuple(reversed(MONEY_LIQUIDITY_SERIES)))
    assert digest != money_liquidity_catalogue_sha256(
        (replace(FED_M2, unit="USD million"), *MONEY_LIQUIDITY_SERIES[1:])
    )
    assert digest != money_liquidity_catalogue_sha256(
        tuple(
            replace(spec, research_role="diagnostic") if spec is BOE_M4EX else spec
            for spec in MONEY_LIQUIDITY_SERIES
        )
    )
    assert digest != money_liquidity_catalogue_sha256(
        tuple(
            replace(spec, perimeter="changed perimeter") if spec is BOE_M4EX else spec
            for spec in MONEY_LIQUIDITY_SERIES
        )
    )


def test_source_fetches_and_parses_fred_ecb_and_scb_without_live_http(tmp_path):
    fred_csv = (
        "observation_date,M2SL\r\n"
        "1959-01-01,286.6\r\n"
        "1959-02-01,.\r\n"
        "1959-03-01,287.7\r\n"
        "1959-04-01,\r\n"
    )
    ecb_csv = (
        "KEY,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS,TIME_FORMAT,UNIT,UNIT_MULT\n"
        "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E,M,1980-01,1097404,A,P1M,EUR,6\n"
        "BSI.M.U2.Y.V.M30.X.1.U2.2300.Z01.E,M,1980-02,1101900,P,P1M,EUR,6\n"
    )
    scb_json = json.dumps(_scb_payload())
    client = MagicMock()
    client.get.side_effect = [_response(fred_csv), _response(ecb_csv), _response(scb_json)]
    artifact_dir = tmp_path / "artifacts"
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    fred = source.fetch(FED_M2, use_cache=False)
    ecb = source.fetch(ECB_M3, use_cache=False)
    scb = source.fetch(SCB_M3, use_cache=False)

    assert client.get.call_count == 3
    assert [call.args[0] for call in client.get.call_args_list] == [
        FED_M2.url,
        ECB_M3.url,
        SCB_M3.url,
    ]
    assert list(fred.columns) == _COLUMNS
    assert list(fred[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(1959, 1, 1), 286.6),
        (date(1959, 3, 1), 287.7),
    ]
    assert list(ecb["date"]) == [date(1980, 1, 1), date(1980, 2, 1)]
    assert list(ecb["status"]) == ["observed", "provisional"]
    assert set(ecb["series_id"]) == {ECB_M3.native_series_id}
    assert list(scb[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2026, 1, 1), 4900000.0),
        (date(2026, 3, 1), 5100000.0),
    ]
    assert set(scb["series_id"]) == {SCB_M3.native_series_id}
    assert fred.attrs["source_url"] == FED_M2.url
    assert ecb.attrs["source_url"] == ECB_M3.url
    assert scb.attrs["source_url"] == SCB_M3.url
    assert ecb.attrs["native_periods"] == ("1980-01", "1980-02")
    assert ecb.attrs["native_period_format"] == "P1M"
    assert scb.attrs["native_periods"] == ("2026M01", "2026M02", "2026M03")

    fred_artifact = Path(fred.attrs["source_artifact_path"])
    assert fred_artifact.read_bytes() == fred_csv.encode("utf-8")
    assert (
        fred.attrs["source_artifact_sha256"] == hashlib.sha256(fred_csv.encode("utf-8")).hexdigest()
    )
    assert fred.attrs["native_payload_artifact_path"] == str(fred_artifact)
    assert fred.attrs["native_payload_sha256"] == fred.attrs["source_artifact_sha256"]
    assert "fed_fred" in fred_artifact.parts
    assert "ecb_data" in Path(ecb.attrs["source_artifact_path"]).parts
    assert "scb_riksbank_money" in Path(scb.attrs["source_artifact_path"]).parts

    assert [record["missing_kind"] for record in fred.attrs["missing_period_records"]] == [
        "null_token",
        "blank",
    ]
    assert [record["native_token"] for record in fred.attrs["missing_period_records"]] == [
        ".",
        "",
    ]
    assert scb.attrs["missing_period_records"] == (
        {
            "native_period": "2026M02",
            "native_position": 1,
            "missing_kind": "json_null",
            "native_token": None,
            "reason": "publisher_null_not_zero",
            "evidence": "scb_jsonstat_value_cell",
        },
    )
    for frame in (fred, ecb, scb):
        missing_path = Path(frame.attrs["missing_provenance_artifact_path"])
        missing_bytes = missing_path.read_bytes()
        assert hashlib.sha256(missing_bytes).hexdigest() == frame.attrs["missing_provenance_sha256"]
        assert missing_bytes.decode("utf-8") == frame.attrs["missing_provenance_json"]


def test_source_fetches_and_parses_boe_month_end_stocks_without_live_http(tmp_path):
    boe_csv = _boe_payload(
        BOE_M4EX,
        "31 Jul 2009,1539822\n31 Aug 2009,..\n30 Sep 2009,1536227\n",
    )
    client = MagicMock()
    client.get.return_value = _response(boe_csv)
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )

    frame = source.fetch(BOE_M4EX, use_cache=False)

    client.get.assert_called_once_with(BOE_M4EX.url, timeout=30.0)
    assert list(frame.columns) == _COLUMNS
    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2009, 7, 1), 1539822.0),
        (date(2009, 9, 1), 1536227.0),
    ]
    assert set(frame["country"]) == {"UK"}
    assert set(frame["source"]) == {BOE_M4EX.source_family}
    assert set(frame["series_id"]) == {BOE_M4EX.native_series_id}
    assert frame.attrs["source_url"] == BOE_M4EX.url
    assert frame.attrs["native_periods"] == (
        "31 Jul 2009",
        "31 Aug 2009",
        "30 Sep 2009",
    )
    assert frame.attrs["native_period_format"] == "calendar month end"
    assert frame.attrs["missing_period_records"] == (
        {
            "native_period": "31 Aug 2009",
            "native_position": 1,
            "missing_kind": "null_token",
            "native_token": "..",
            "reason": "publisher_null_not_zero",
            "evidence": "boe_csv_value_cell",
        },
    )


def test_source_artifact_archive_is_content_addressed_and_idempotent(tmp_path):
    text = "observation_date,M2SL\r\n1959-01-01,286.6\r\n"
    client = MagicMock()
    client.get.return_value = _response(text)
    artifact_dir = tmp_path / "artifacts"
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    first = source.fetch(FED_M2, use_cache=False)
    second = source.fetch(FED_M2, use_cache=True)

    client.get.assert_called_once_with(FED_M2.url, timeout=30.0)
    assert first.attrs["source_artifact_path"] == second.attrs["source_artifact_path"]
    assert first.attrs["source_artifact_sha256"] == second.attrs["source_artifact_sha256"]
    assert (
        first.attrs["missing_provenance_artifact_path"]
        == second.attrs["missing_provenance_artifact_path"]
    )
    assert Path(first.attrs["source_artifact_path"]).read_bytes() == text.encode("utf-8")
    assert len(list(artifact_dir.rglob("*.csv"))) == 1
    assert len(list(artifact_dir.rglob("*.json"))) == 1


def test_source_rejects_corrupt_content_addressed_artifact(tmp_path):
    text = "observation_date,M2SL\n1959-01-01,286.6\n"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    artifact_dir = tmp_path / "artifacts"
    destination = artifact_dir / "fed_fred" / "source-responses" / digest[:2] / f"{digest}.csv"
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"corrupt")
    client = MagicMock()
    client.get.return_value = _response(text)
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="corrupt archive"):
        source.fetch(FED_M2, use_cache=False)

    assert destination.read_bytes() == b"corrupt"
    assert not list(artifact_dir.rglob("missingness-ledgers/*.json"))


def test_source_rejects_corrupt_missingness_ledger(tmp_path):
    text = "observation_date,M2SL\n1959-01-01,286.6\n"
    client = MagicMock()
    client.get.return_value = _response(text)
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )
    first = source.fetch(FED_M2, use_cache=False)
    missing_path = Path(first.attrs["missing_provenance_artifact_path"])
    missing_path.write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="corrupt archive"):
        source.fetch(FED_M2, use_cache=True)

    assert missing_path.read_bytes() == b"corrupt"


def test_scb_sparse_absence_is_distinct_from_explicit_json_null(tmp_path):
    payload = _scb_payload()
    payload["value"] = {"0": 4900000, "2": None}
    text = json.dumps(payload)
    client = MagicMock()
    client.get.return_value = _response(text)
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=tmp_path / "artifacts",
    )

    frame = source.fetch(SCB_M3, use_cache=False)

    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(2026, 1, 1), 4900000.0)
    ]
    assert [record["missing_kind"] for record in frame.attrs["missing_period_records"]] == [
        "sparse_absent",
        "json_null",
    ]
    assert [record["native_period"] for record in frame.attrs["missing_period_records"]] == [
        "2026M02",
        "2026M03",
    ]


def test_invalid_response_is_not_admitted_to_money_artifact_archive(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("observation_date,WRONG\n1959-01-01,1\n")
    artifact_dir = tmp_path / "artifacts"
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="missing columns"):
        source.fetch(FED_M2, use_cache=False)

    assert not artifact_dir.exists()


def test_empty_response_is_returned_unbacked_for_pipeline_rejection(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("observation_date,M2SL\n")
    artifact_dir = tmp_path / "artifacts"
    source = MoneyLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    frame = source.fetch(FED_M2, use_cache=False)

    assert frame.empty
    assert "source_artifact_path" not in frame.attrs
    assert not artifact_dir.exists()


def test_boe_quarterly_m4ex_bridge_keeps_native_frequency():
    csv = _boe_payload(
        BOE_M4EX_QUARTERLY,
        "31 Dec 1997,681139\n31 Mar 1998,693372\n30 Jun 1998,702065\n",
    )

    frame = parse_boe_csv(csv, BOE_M4EX_QUARTERLY)

    assert list(frame[["date", "value"]].itertuples(index=False, name=None)) == [
        (date(1997, 10, 1), 681139.0),
        (date(1998, 1, 1), 693372.0),
        (date(1998, 4, 1), 702065.0),
    ]
    assert frame.attrs["native_periods"] == (
        "31 Dec 1997",
        "31 Mar 1998",
        "30 Jun 1998",
    )
    assert frame.attrs["native_period_format"] == "calendar quarter end"


@pytest.mark.parametrize(
    ("csv", "message"),
    [
        (
            _boe_payload(BOE_M4EX, "31 Jul 2009,1\n", data_header="LPMAUYN"),
            "exact native series",
        ),
        (
            _boe_payload(BOE_M4EX, "31 Jul 2009,1,x\n"),
            "malformed observation rows",
        ),
        (_boe_payload(BOE_M4EX, "30 Jul 2009,1\n"), "calendar month end"),
        (_boe_payload(BOE_M4EX, "31 Jul 2009,0\n"), "must be positive"),
        (_boe_payload(BOE_M4EX, "31 Jul 2009,not-a-number\n"), "non-numeric"),
    ],
)
def test_boe_parser_rejects_native_identity_period_and_value_drift(csv, message):
    with pytest.raises(ValueError, match=message):
        parse_boe_csv(csv, BOE_M4EX)


def test_boe_quarterly_parser_rejects_non_quarter_end():
    with pytest.raises(ValueError, match="quarter end"):
        parse_boe_csv(
            _boe_payload(BOE_M4EX_QUARTERLY, "31 Jan 1998,1\n"),
            BOE_M4EX_QUARTERLY,
        )


def test_boe_parser_rejects_description_drift_or_missing_preamble():
    with pytest.raises(ValueError, match="description mismatch"):
        parse_boe_csv(
            _boe_payload(
                BOE_M4EX,
                "31 Jul 2009,1\n",
                description="Changed definition",
            ),
            BOE_M4EX,
        )
    with pytest.raises(ValueError, match="description preamble"):
        parse_boe_csv("DATE,RPMB53Q\n31 Jul 2009,1\n", BOE_M4EX)


def test_ecb_weekly_periods_remain_native_weekly_and_semantics_are_validated():
    csv = (
        "KEY,FREQ,TIME_PERIOD,OBS_VALUE,OBS_STATUS,TIME_FORMAT,UNIT,UNIT_MULT\n"
        "ILM.W.U2.C.T000000.Z5.Z01,W,1998-W53,697160,A,P7D,EUR,6\n"
        "ILM.W.U2.C.T000000.Z5.Z01,W,1999-W01,699200,E,P7D,EUR,6\n"
    )

    frame = parse_ecb_csv(csv, ECB_EUROSYSTEM_ASSETS)

    # Weekly levels are not silently converted into synthetic monthly data.
    assert list(frame["date"]) == [date(1999, 1, 1), date(1999, 1, 8)]
    assert list(frame["value"]) == [697160.0, 699200.0]
    assert list(frame["status"]) == ["observed", "estimated"]
    assert frame.attrs["native_periods"] == ("1998-W53", "1999-W01")
    assert frame.attrs["native_period_format"] == "P7D"

    wrong_unit = csv.replace(",EUR,6", ",USD,6")
    with pytest.raises(ValueError, match="currency/unit"):
        parse_ecb_csv(wrong_unit, ECB_EUROSYSTEM_ASSETS)

    wrong_key = csv.replace(ECB_EUROSYSTEM_ASSETS.native_series_id, "ILM.W.WRONG")
    with pytest.raises(ValueError, match="native series"):
        parse_ecb_csv(wrong_key, ECB_EUROSYSTEM_ASSETS)

    wrong_period_format = csv.replace(",P7D,", ",P1M,")
    with pytest.raises(ValueError, match="period format"):
        parse_ecb_csv(wrong_period_format, ECB_EUROSYSTEM_ASSETS)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update(source="Statistics Sweden"), "publisher"),
        (
            lambda payload: payload["dimension"]["Penningm"]["category"]["label"].update(
                {"5LLM3a.1E.NEP.V.A": "M2"}
            ),
            "M3 label",
        ),
        (
            lambda payload: payload["dimension"]["ContentsCode"]["extension"]["refperiod"].update(
                {"000007WQ": "Quarter"}
            ),
            "reference period",
        ),
        (
            lambda payload: payload["dimension"]["ContentsCode"]["extension"][
                "measuringType"
            ].update({"000007WQ": "Flow"}),
            "measure type",
        ),
        (
            lambda payload: payload["dimension"]["ContentsCode"]["extension"]["priceType"].update(
                {"000007WQ": "Constant"}
            ),
            "price type",
        ),
        (
            lambda payload: payload["dimension"]["ContentsCode"]["extension"]["adjustment"].update(
                {"000007WQ": "Seasonal"}
            ),
            "adjustment",
        ),
    ],
)
def test_scb_parser_rejects_native_semantic_metadata_drift(mutation, message):
    payload = _scb_payload()
    mutation(payload)

    with pytest.raises(ValueError, match=message):
        parse_scb_jsonstat(json.dumps(payload), SCB_M3)


def _frame(spec, values: list[tuple[date, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "country": spec.country,
                "indicator": spec.indicator,
                "date": observed_on,
                "value": value,
                "source": spec.source_family,
                "series_id": spec.native_series_id,
            }
            for observed_on, value in values
        ],
        columns=_COLUMNS,
    )


class _MutableSource:
    def __init__(self, frames):
        self.frames = frames
        self.calls: list[tuple[str, bool]] = []

    def fetch(self, spec, use_cache=True):
        self.calls.append((spec.native_series_id, use_cache))
        result = self.frames[spec.native_series_id]
        if isinstance(result, Exception):
            raise result
        frame = result.copy()
        frame.attrs["source_url"] = spec.url
        if not frame.empty:
            artifact_dir = Path(tempfile.mkdtemp(prefix="dalio-money-test-"))
            source_bytes = json.dumps(
                {
                    "native_series_id": spec.native_series_id,
                    "observations": [
                        {
                            "date": row.date.isoformat(),
                            "value": float(row.value),
                        }
                        for row in frame.itertuples(index=False)
                    ],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            missing_payload = {
                "native_series_id": spec.native_series_id,
                "records": [],
            }
            missing_bytes = json.dumps(
                missing_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            source_path = artifact_dir / "source.json"
            missing_path = artifact_dir / "missing.json"
            source_path.write_bytes(source_bytes)
            missing_path.write_bytes(missing_bytes)
            source_hash = hashlib.sha256(source_bytes).hexdigest()
            missing_hash = hashlib.sha256(missing_bytes).hexdigest()
            frame.attrs.update(
                {
                    "source_artifact_path": str(source_path),
                    "source_artifact_sha256": source_hash,
                    "native_payload_artifact_path": str(source_path),
                    "native_payload_sha256": source_hash,
                    "missing_period_records": (),
                    "missing_provenance_json": missing_bytes.decode("utf-8"),
                    "missing_provenance_artifact_path": str(missing_path),
                    "missing_provenance_sha256": missing_hash,
                }
            )
        return frame


def test_pipeline_keeps_complete_native_releases_and_fails_closed(tmp_path):
    engine = make_engine(tmp_path / "money.db")
    fed_spec = _small_history_spec(FED_M2, expected_start=date(2026, 1, 1))
    ecb_spec = _small_history_spec(ECB_M3, expected_start=date(2026, 1, 1))
    source = _MutableSource(
        {
            FED_M2.native_series_id: _frame(
                FED_M2,
                [(date(2026, 1, 1), 22000.0), (date(2026, 2, 1), 22100.0)],
            ),
            ECB_M3.native_series_id: _frame(
                ECB_M3,
                [(date(2026, 1, 1), 17000.0)],
            ),
        }
    )

    first = run_pipeline(
        (fed_spec, ecb_spec),
        source=source,
        use_cache=False,
        engine=engine,
        retrieved_at=_at(3),
    )
    source.frames[FED_M2.native_series_id] = _frame(
        FED_M2,
        [(date(2026, 1, 1), 22050.0)],
    )
    rejected = run_pipeline((fed_spec,), source=source, engine=engine, retrieved_at=_at(4))
    second = run_pipeline(
        (fed_spec,),
        source=source,
        engine=engine,
        retrieved_at=_at(4),
        allow_contraction=True,
    )
    source.frames[FED_M2.native_series_id] = pd.DataFrame(columns=_COLUMNS)
    failed = run_pipeline((fed_spec, ecb_spec), source=source, engine=engine, retrieved_at=_at(5))

    assert source.calls[:2] == [("M2SL", False), (ECB_M3.native_series_id, False)]
    assert first["US/broad_money_m2_stock"]["inserted"] == 2
    assert first["US/broad_money_m2_stock"]["currency"] == "USD"
    assert first["US/broad_money_m2_stock"]["unit"] == "USD billion"
    assert first["US/broad_money_m2_stock"]["catalogue_semantic_sha256"] == (
        money_liquidity_catalogue_sha256()
    )
    assert "contracts complete history" in rejected["US/broad_money_m2_stock"]["error"]
    assert second["US/broad_money_m2_stock"]["removed"] == 1
    assert "empty snapshot" in failed["US/broad_money_m2_stock"]["error"]
    assert "error" not in failed["EU/broad_money_m3_stock"]

    partition = make_partition_key(
        FED_M2.source_family,
        FED_M2.native_series_id,
        FED_M2.country,
        FED_M2.indicator,
    )
    with Session(engine) as session:
        history = release_history(session, partition)
        old = load_vintage_panel(session, _at(3), partition_keys=(partition,))
        new = load_vintage_panel(session, _at(4), partition_keys=(partition,))
        current = session.execute(
            select(Observation.date, Observation.value).where(
                Observation.source == FED_M2.source_family
            )
        ).all()
        releases = session.execute(select(DataRelease)).scalars().all()
        artifacts = session.execute(select(DataReleaseArtifact)).scalars().all()

    assert len(history) == 2
    assert history[0].source_url == FED_M2.url
    assert all(
        item.vintage_label
        == f"money-liquidity-catalogue-sha256:{money_liquidity_catalogue_sha256()}"
        for item in history
    )
    assert list(old["value"]) == [22000.0, 22100.0]
    assert list(new["value"]) == [22050.0]
    assert current == [(date(2026, 1, 1), 22050.0)]
    assert len(releases) == 3  # two US vintages plus one idempotent EU release
    assert len(artifacts) == 9
    assert {artifact.role for artifact in artifacts} == {
        "source_response",
        "native_series_payload",
        "missingness_ledger",
    }


def test_pipeline_rejects_wrong_partition_but_continues_to_next_series(tmp_path):
    engine = make_engine(tmp_path / "money-invalid.db")
    fed_spec = _small_history_spec(FED_M2, expected_start=date(2026, 1, 1))
    ecb_spec = _small_history_spec(ECB_M3, expected_start=date(2026, 1, 1))
    wrong = _frame(FED_M2, [(date(2026, 1, 1), 22000.0)])
    wrong["series_id"] = "WALCL"
    source = _MutableSource(
        {
            FED_M2.native_series_id: wrong,
            ECB_M3.native_series_id: _frame(
                ECB_M3,
                [(date(2026, 1, 1), 17000.0)],
            ),
        }
    )

    summary = run_pipeline((fed_spec, ecb_spec), source=source, engine=engine, retrieved_at=_at(3))

    assert "native series" in summary["US/broad_money_m2_stock"]["error"]
    assert summary["EU/broad_money_m3_stock"]["inserted"] == 1


@pytest.mark.parametrize(
    ("values", "message"),
    [
        ([(date(2026, 2, 1), 1.0)], "expected start"),
        ([(date(2026, 1, 1), 1.0), (date(2026, 3, 1), 2.0)], "cadence"),
        ([(date(2026, 1, 1), 1.0)], "minimum observation"),
    ],
)
def test_pipeline_rejects_incomplete_native_history(tmp_path, values, message):
    spec = replace(
        FED_M2,
        expected_start=date(2026, 1, 1),
        minimum_observations=2,
        max_latest_lag_days=366,
    )
    source = _MutableSource({FED_M2.native_series_id: _frame(FED_M2, values)})

    summary = run_pipeline(
        (spec,), source=source, engine=make_engine(tmp_path / "guard.db"), retrieved_at=_at(4)
    )

    assert message in summary["US/broad_money_m2_stock"]["error"]


def test_pipeline_validates_quarterly_m4ex_bridge_cadence(tmp_path):
    spec = replace(
        BOE_M4EX_QUARTERLY,
        expected_start=date(2025, 1, 1),
        minimum_observations=2,
        max_latest_lag_days=500,
    )
    source = _MutableSource(
        {
            spec.native_series_id: _frame(
                spec,
                [(date(2025, 1, 1), 1.0), (date(2025, 7, 1), 2.0)],
            )
        }
    )

    summary = run_pipeline(
        (spec,),
        source=source,
        engine=make_engine(tmp_path / "quarter-gap.db"),
        retrieved_at=datetime(2025, 8, 1, tzinfo=UTC),
    )

    assert "gap in quarterly cadence" in next(iter(summary.values()))["error"]


def test_pipeline_rejects_stale_native_history(tmp_path):
    spec = replace(
        FED_M2,
        expected_start=date(2026, 1, 1),
        minimum_observations=1,
        max_latest_lag_days=30,
    )
    source = _MutableSource({FED_M2.native_series_id: _frame(FED_M2, [(date(2026, 1, 1), 1.0)])})

    summary = run_pipeline(
        (spec,), source=source, engine=make_engine(tmp_path / "stale.db"), retrieved_at=_at(4)
    )

    assert "latest observation is stale" in summary["US/broad_money_m2_stock"]["error"]


def test_pipeline_captures_default_clock_after_each_successful_fetch(tmp_path):
    fed_spec = _small_history_spec(FED_M2, expected_start=date(2026, 6, 1))
    ecb_spec = _small_history_spec(ECB_M3, expected_start=date(2026, 6, 1))
    source = _MutableSource(
        {
            FED_M2.native_series_id: _frame(FED_M2, [(date(2026, 6, 1), 1.0)]),
            ECB_M3.native_series_id: _frame(ECB_M3, [(date(2026, 6, 1), 2.0)]),
        }
    )
    engine = make_engine(tmp_path / "clocks.db")
    first_clock = datetime(2026, 6, 10, 12, tzinfo=UTC)
    second_clock = datetime(2026, 6, 10, 12, 5, tzinfo=UTC)

    with patch("dalio.pipelines.fetch_money_liquidity.datetime") as clock:
        clock.now.side_effect = [first_clock, second_clock]
        run_pipeline((fed_spec, ecb_spec), source=source, engine=engine)

    with Session(engine) as session:
        releases = session.execute(select(DataRelease).order_by(DataRelease.id)).scalars().all()
    assert [release.retrieved_at for release in releases] == [
        first_clock.replace(tzinfo=None),
        second_clock.replace(tzinfo=None),
    ]
    assert [release.available_at for release in releases] == [
        first_clock.replace(tzinfo=None),
        second_clock.replace(tzinfo=None),
    ]
