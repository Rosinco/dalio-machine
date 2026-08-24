"""OECD QNA / LFS adapter (slice 26). Mocked HTTP only."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from dalio.countries import get_country
from dalio.data_sources.oecd import (
    LFS_UNEMPLOYMENT,
    QNA_GDP_GROWTH,
    OecdSource,
    period_to_date,
)

SAMPLE_QNA = (
    "DATAFLOW,FREQ,ADJUSTMENT,REF_AREA,SECTOR,COUNTERPART_SECTOR,TRANSACTION,INSTR_ASSET,ACTIVITY,EXPENDITURE,UNIT_MEASURE,PRICE_BASE,TRANSFORMATION,TABLE_IDENTIFIER,TIME_PERIOD,OBS_VALUE\n"
    "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD(1.1),Q,Y,GBR,S1,S1,B1GQ,_Z,_Z,_Z,PC,L,GY,T0102,2026-Q1,0.912997847\n"
    "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD(1.1),Q,Y,GBR,S1,S1,B1GQ,_Z,_Z,_Z,PC,L,GY,T0102,2026-Q2,1.174400872\n"
    "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD(1.1),Q,Y,EA,S1,S1,B1GQ,_Z,_Z,_Z,PC,L,GY,T0102,2026-Q2,1.3\n"
    "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD(1.1),Q,Y,BRA,S1,S1,B1GQ,_Z,_Z,_Z,PC,L,GY,T0102,2026-Q1,\n"
    "OECD.SDD.NAD:DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD(1.1),Q,Y,ZAF,S1,S1,B1GQ,_Z,_Z,_Z,PC,L,GY,T0102,2026-Q1,0.5\n"
)
SAMPLE_LFS = (
    "DATAFLOW,REF_AREA,MEASURE,UNIT_MEASURE,TRANSFORMATION,ADJUSTMENT,SEX,AGE,ACTIVITY,FREQ,TIME_PERIOD,OBS_VALUE\n"
    "OECD.SDD.TPS:DSD_LFS@DF_IALFS_UNE_M(1.0),JPN,UNE_LF_M,PT_LF_SUB,_Z,Y,_T,Y_GE15,_Z,M,2026-06,2.5\n"
    "OECD.SDD.TPS:DSD_LFS@DF_IALFS_UNE_M(1.0),EA,UNE_LF_M,PT_LF_SUB,_Z,Y,_T,Y_GE15,_Z,M,2026-06,6.3\n"
)


def _resp(text, status_code=200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def _basket():
    return [get_country(c) for c in ("UK", "EU", "BR", "JP")]


def test_gdp_growth_flow(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp(SAMPLE_QNA)
    df = OecdSource(client=client, cache_dir=tmp_path).fetch_gdp_growth(_basket(), use_cache=False, start_year=2015)
    url = client.get.call_args[0][0]
    assert url == ("https://sdmx.oecd.org/public/rest/data/"
                   "OECD.SDD.NAD,DSD_NAMAIN1@DF_QNA_EXPENDITURE_GROWTH_OECD,1.1/"
                   "Q.Y.GBR+EA+BRA+JPN.S1.S1.B1GQ._Z._Z._Z.PC.L.GY.T0102?startPeriod=2015-Q1")
    assert set(df["country"]) == {"UK", "EU"}                              # BR empty, ZAF not requested, JP absent
    uk = df[df["country"] == "UK"].sort_values("date")
    assert list(uk["date"]) == [date(2026, 1, 1), date(2026, 4, 1)]        # first month of the quarter
    assert uk["value"].iloc[-1] == pytest.approx(1.174400872)
    assert set(df["indicator"]) == {"real_gdp_yoy"} and set(df["source"]) == {"OECD_QNA"}
    assert set(df["series_id"]) == {QNA_GDP_GROWTH.series_id}


def test_unemployment_flow_and_missing_areas(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp(SAMPLE_LFS)
    src = OecdSource(client=client, cache_dir=tmp_path)
    df = src.fetch_unemployment(_basket(), use_cache=False)
    assert "GBR+EA+BRA+JPN.UNE_LF_M.PT_LF_SUB._Z.Y._T.Y_GE15._Z.M?startPeriod=2010-01" in client.get.call_args[0][0]
    assert set(df["country"]) == {"JP", "EU"}                              # Brazil has no monthly LFS: no rows, no error
    assert df[df["country"] == "EU"]["date"].iloc[0] == date(2026, 6, 1)
    assert set(df["source"]) == {LFS_UNEMPLOYMENT.source}
    client.get.return_value = _resp("nothing", 404)                         # a key with no data at all
    with pytest.raises(ValueError, match="404"):
        src.fetch_unemployment(_basket(), use_cache=False)
    assert src.fetch(QNA_GDP_GROWTH, [], use_cache=False).empty


def test_period_to_date():
    assert period_to_date("2026-Q2") == date(2026, 4, 1)
    assert period_to_date("2026-06") == date(2026, 6, 1)
    assert period_to_date("2026") is None and period_to_date(None) is None and period_to_date("2026-Qx") is None
