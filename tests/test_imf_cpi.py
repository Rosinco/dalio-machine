"""IMF CPI adapter (slice 26). Mocked HTTP only."""
from datetime import date
from unittest.mock import MagicMock

import pytest

from dalio.countries import get_country
from dalio.data_sources.imf_cpi import (
    INDICATOR_CPI,
    SERIES_ID_CPI,
    SOURCE_IMF_CPI,
    ImfCpiSource,
)

# Real column layout; FULL_DESCRIPTION is a quoted MULTI-LINE field on every row.
SAMPLE = (
    "DATAFLOW,COUNTRY,INDEX_TYPE,COICOP_1999,TYPE_OF_TRANSFORMATION,FREQUENCY,TIME_PERIOD,OBS_VALUE,SCALE,FULL_DESCRIPTION\n"
    'IMF.STA:CPI(5.0.0),JPN,CPI,_T,YOY_PCH_PA_PT,M,2026-M05,1.9,0,"The CPI dataset\nspans two lines"\n'
    'IMF.STA:CPI(5.0.0),JPN,CPI,_T,YOY_PCH_PA_PT,M,2026-M06,1.700984780662489,0,"The CPI dataset\nspans two lines"\n'
    'IMF.STA:CPI(5.0.0),JPN,CPI,_T,IX,M,2026-M06,119.81,0,"index level — must be ignored"\n'
    'IMF.STA:CPI(5.0.0),JPN,CPI,CP01,YOY_PCH_PA_PT,M,2026-M06,3.2,0,"food division — must be ignored"\n'
    'IMF.STA:CPI(5.0.0),GBR,CPI,_T,YOY_PCH_PA_PT,M,2026-M06,2.8,0,"x"\n'
    'IMF.STA:CPI(5.0.0),GBR,CPI,_T,YOY_PCH_PA_PT,M,2026-M07,,0,"empty value — dropped"\n'
    'IMF.STA:CPI(5.0.0),DEU,CPI,_T,YOY_PCH_PA_PT,M,2026-M06,2.1,0,"not requested"\n'
)


def _resp(text, status_code=200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def _basket():
    return [get_country(c) for c in ("JP", "UK", "EU")]


def test_fetch_parses_multiline_csv_and_filters(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp(SAMPLE)
    df = ImfCpiSource(client=client, cache_dir=tmp_path).fetch(_basket(), use_cache=False, start_year=2020)
    url = client.get.call_args[0][0]
    assert url.startswith("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,CPI/")
    assert "JPN+GBR.CPI._T.YOY_PCH_PA_PT.M?startPeriod=2020" in url        # EU has no entity → not in the key
    assert set(df["country"]) == {"JP", "UK"}
    jp = df[df["country"] == "JP"].sort_values("date")
    assert list(jp["date"]) == [date(2026, 5, 1), date(2026, 6, 1)]
    assert jp["value"].iloc[-1] == pytest.approx(1.700984780662489)
    assert len(df[df["country"] == "UK"]) == 1                               # empty OBS_VALUE dropped
    assert set(df["indicator"]) == {INDICATOR_CPI}
    assert set(df["source"]) == {SOURCE_IMF_CPI} and set(df["series_id"]) == {SERIES_ID_CPI}


def test_errors_and_empties(tmp_path):
    client = MagicMock()
    src = ImfCpiSource(client=client, cache_dir=tmp_path)
    client.get.return_value = _resp("denied", 403)
    with pytest.raises(ValueError, match="403"):
        src.fetch(_basket(), use_cache=False)
    client.get.return_value = _resp("COUNTRY,TIME_PERIOD,OBS_VALUE\n")
    assert src.fetch(_basket(), use_cache=False).empty
    assert src.fetch([get_country("EU")], use_cache=False).empty              # nothing to request
    assert client.get.call_count == 2
