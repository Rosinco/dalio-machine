"""BIS Global Liquidity catalogue and strict source parsing."""

from __future__ import annotations

from dataclasses import replace
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from dalio.data_sources.bis_global_liquidity import (
    BIS_GLI_EUR,
    BIS_GLI_JPY,
    BIS_GLI_USD,
    BIS_GLOBAL_LIQUIDITY_SERIES,
    BisGlobalLiquiditySource,
    bis_global_liquidity_catalogue_sha256,
    parse_bis_global_liquidity_csv,
)


def _csv(**overrides: str) -> str:
    row = {
        "FREQ": "Q",
        "CURR_DENOM": "USD",
        "BORROWERS_CTY": "3P",
        "BORROWERS_SECTOR": "N",
        "LENDERS_SECTOR": "A",
        "L_POS_TYPE": "I",
        "L_INSTR": "B",
        "UNIT_MEASURE": "USD",
        "TITLE": (
            "USD denominated credit (bank loans & debt securities) to non-bank "
            "borrowers located outside the US"
        ),
        "UNIT_MULT": "6",
        "TIME_PERIOD": "2000-Q1",
        "OBS_VALUE": "2331313.294",
        "OBS_STATUS": "A",
        "OBS_PRE_BREAK": "",
        "OBS_CONF": "F",
    }
    row.update(overrides)
    return pd.DataFrame([row]).to_csv(index=False)


class _Response:
    status_code = 200

    def __init__(self, text: str) -> None:
        self.text = text

    def raise_for_status(self) -> None:
        return None


class _Client:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[tuple[str, float]] = []
        self.headers: dict[str, str] = {}

    def get(self, url: str, *, timeout: float = 0) -> _Response:
        self.calls.append((url, timeout))
        return _Response(self.text)


def test_catalogue_is_currency_native_and_explicitly_non_additive() -> None:
    assert len(BIS_GLOBAL_LIQUIDITY_SERIES) == 3
    assert {spec.currency for spec in BIS_GLOBAL_LIQUIDITY_SERIES} == {"USD", "EUR", "JPY"}
    assert {spec.unit for spec in BIS_GLOBAL_LIQUIDITY_SERIES} == {
        "USD million",
        "EUR million",
        "JPY million",
    }
    assert {spec.country for spec in BIS_GLOBAL_LIQUIDITY_SERIES} == {"GLOBAL"}
    assert {spec.non_additive_group for spec in BIS_GLOBAL_LIQUIDITY_SERIES} == {
        "bis_gli_offshore_credit_by_currency"
    }
    assert all(spec.measure_kind == "credit_stock" for spec in BIS_GLOBAL_LIQUIDITY_SERIES)
    assert all(spec.claim_side == "borrower_liability" for spec in BIS_GLOBAL_LIQUIDITY_SERIES)
    assert BIS_GLI_USD.native_series_id == "Q.USD.3P.N.A.I.B.USD"
    assert BIS_GLI_EUR.native_series_id == "Q.EUR.3P.N.A.I.B.EUR"
    assert BIS_GLI_JPY.native_series_id == "Q.JPY.3P.N.A.I.B.JPY"


def test_catalogue_hash_is_order_stable_and_covers_semantics() -> None:
    digest = bis_global_liquidity_catalogue_sha256()
    assert len(digest) == 64
    assert digest == bis_global_liquidity_catalogue_sha256(
        tuple(reversed(BIS_GLOBAL_LIQUIDITY_SERIES))
    )
    assert digest != bis_global_liquidity_catalogue_sha256(
        (replace(BIS_GLI_USD, claim_side="asset"), BIS_GLI_EUR, BIS_GLI_JPY)
    )


def test_parse_exact_series_preserves_status_and_native_period() -> None:
    frame = parse_bis_global_liquidity_csv(_csv(), BIS_GLI_USD)

    assert frame.to_dict("records") == [
        {
            "country": "GLOBAL",
            "indicator": "offshore_usd_credit_nonbanks_stock",
            "date": pd.Timestamp("2000-01-01").date(),
            "value": 2331313.294,
            "source": "BIS_GLI",
            "series_id": "Q.USD.3P.N.A.I.B.USD",
            "status": "observed",
        }
    ]
    assert frame.attrs["native_periods"] == ("2000-Q1",)
    assert frame.attrs["native_period_format"] == "quarterly"


def test_source_uses_exact_official_url_and_cache(tmp_path) -> None:
    client = _Client(_csv())
    artifact_dir = tmp_path / "artifacts"
    source = BisGlobalLiquiditySource(
        client=client,
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    frame = source.fetch(BIS_GLI_USD, use_cache=False)

    assert client.calls == [(BIS_GLI_USD.url, 60.0)]
    assert frame.attrs["source_url"] == BIS_GLI_USD.url
    artifact_path = frame.attrs["source_artifact_path"]
    assert str(artifact_path).startswith(str(artifact_dir))
    assert frame.attrs["source_artifact_sha256"] == frame.attrs["native_payload_sha256"]
    assert len(frame.attrs["source_artifact_sha256"]) == 64
    assert Path(artifact_path).read_text() == _csv()
    assert client.headers["User-Agent"].startswith("dalio-machine/")


def test_invalid_bis_response_is_never_archived(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    source = BisGlobalLiquiditySource(
        client=_Client(_csv(CURR_DENOM="EUR")),
        cache_dir=tmp_path / "cache",
        artifact_dir=artifact_dir,
    )

    with pytest.raises(ValueError, match="identity mismatch"):
        source.fetch(BIS_GLI_USD, use_cache=False)

    assert not list(artifact_dir.rglob("*.csv"))


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("CURR_DENOM", "EUR", "CURR_DENOM identity mismatch"),
        ("BORROWERS_CTY", "US", "BORROWERS_CTY identity mismatch"),
        ("BORROWERS_SECTOR", "P", "BORROWERS_SECTOR identity mismatch"),
        ("LENDERS_SECTOR", "B", "LENDERS_SECTOR identity mismatch"),
        ("L_POS_TYPE", "A", "L_POS_TYPE identity mismatch"),
        ("L_INSTR", "G", "L_INSTR identity mismatch"),
        ("UNIT_MEASURE", "771", "UNIT_MEASURE identity mismatch"),
        ("UNIT_MULT", "9", "UNIT_MULT identity mismatch"),
        ("OBS_CONF", "C", "OBS_CONF identity mismatch"),
    ],
)
def test_parser_rejects_dimension_or_unit_drift(
    column: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_bis_global_liquidity_csv(_csv(**{column: value}), BIS_GLI_USD)


@pytest.mark.parametrize("column", ["CURR_DENOM", "TITLE", "OBS_CONF", "UNIT_MULT"])
def test_parser_rejects_partial_null_identity_dimension(column: str) -> None:
    first = pd.read_csv(StringIO(_csv()), dtype=str)
    second = first.copy()
    second["TIME_PERIOD"] = "2000-Q2"
    second.loc[0, column] = None
    text = pd.concat([first, second], ignore_index=True).to_csv(index=False)

    with pytest.raises(ValueError, match=rf"{column} identity mismatch"):
        parse_bis_global_liquidity_csv(text, BIS_GLI_USD)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"TIME_PERIOD": "2000-05"}, "invalid quarterly period"),
        ({"OBS_VALUE": "not-a-number"}, "non-numeric observation"),
        ({"OBS_VALUE": "0"}, "must be positive"),
        ({"OBS_VALUE": "inf"}, "non-finite observation"),
        ({"OBS_STATUS": ""}, "missing observation status"),
        ({"OBS_PRE_BREAK": "10"}, "introduced pre-break values"),
    ],
)
def test_parser_rejects_ambiguous_or_invalid_observations(
    overrides: dict[str, str],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_bis_global_liquidity_csv(_csv(**overrides), BIS_GLI_USD)


def test_parser_rejects_missing_required_column() -> None:
    text = pd.read_csv(StringIO(_csv())).drop(columns="L_INSTR").to_csv(index=False)
    with pytest.raises(ValueError, match="missing columns"):
        parse_bis_global_liquidity_csv(text, BIS_GLI_USD)
