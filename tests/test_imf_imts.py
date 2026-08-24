"""IMF IMTS bilateral-trade adapter (slice 24). Mocked HTTP only."""
from datetime import date
from unittest.mock import MagicMock

import pytest
import requests

from dalio.countries import ISO2_TO_IMTS, get_country
from dalio.data_sources.imf_imts import (
    IMTS_ACCEPT,
    IMTS_FLOWS,
    IMTS_WORLD,
    SOURCE_IMTS,
    ImtsSource,
    ImtsSpec,
)

HEADER = ("DATAFLOW,COUNTRY,INDICATOR,COUNTERPART_COUNTRY,FREQUENCY,TIME_PERIOD,OBS_VALUE,"
          "SCALE,PRECISION,DECIMALS_DISPLAYED,TRADE_FLOW,VALUATION,UNIT,DERIVATION_TYPE,OVERLAP,STATUS")


def _row(rep, cp, year, value):
    v = "" if value is None else str(value)
    return f"IMF.STA:IMTS(1.0.0),{rep},XG_FOB_USD,{cp},A,{year},{v},6,,,XG,FOB,USD,M,OL,"


SAMPLE = "\n".join([
    HEADER,
    _row("USA", "CAN", 2024, 349397894619),
    _row("USA", "CAN", 2025, 336518346659),
    _row("USA", "DEU", 2025, 83061837748),
    _row("USA", "G001", 2024, 2065411250609),
    _row("USA", "G001", 2025, 2185205847471),
    _row("USA", "G163", 2025, 376767676136),
    _row("USA", "XYZ", 2025, 1),                    # counterpart outside the basket
    _row("USA", "USA", 2025, 5),                    # reporter == partner (never expected, dropped)
    _row("DEU", "G001", 2025, 1757153297511),
    _row("DEU", "G163", 2025, 684810118694),        # member → euro area: kept (intra for a member is fine)
    _row("DEU", "USA", 2025, None),                 # empty OBS_VALUE
    _row("G163", "USA", 2025, 523282089350),
    _row("G163", "DEU", 2025, 675755945303),        # euro-area reporter × own member: dropped
    _row("G163", "G001", 2025, 5813988559150),
]) + "\n"


@pytest.fixture
def http_client():
    return MagicMock()


def _resp(text, status_code=200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def _basket():
    return [get_country(c) for c in ("US", "DE", "CA", "EU")]


def test_fetch_maps_codes_drops_intra_union_and_unknowns(http_client, tmp_path):
    http_client.get.return_value = _resp(SAMPLE)
    src = ImtsSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch(IMTS_FLOWS[0], _basket(), use_cache=False, today=date(2026, 8, 24))
    assert set(df["source"]) == {SOURCE_IMTS}
    us = df[df["country"] == "US"].set_index(["indicator", "date"])["value"]
    assert us[("exports_to_CA", date(2024, 12, 31))] == 349397894619.0
    assert us[("exports_to_WLD", date(2025, 12, 31))] == 2185205847471.0
    assert us[("exports_to_EU", date(2025, 12, 31))] == 376767676136.0
    assert not any(i.endswith("_XYZ") or i.endswith("_US") for i in us.index.get_level_values(0))
    de = set(df[df["country"] == "DE"]["indicator"])
    assert de == {"exports_to_WLD", "exports_to_EU"}                      # empty value row dropped
    eu = set(df[df["country"] == "EU"]["indicator"])
    assert eu == {"exports_to_US", "exports_to_WLD"}                      # EU × DE intra row dropped
    assert df[df["indicator"] == "exports_to_CA"]["series_id"].iloc[0] == "XG_FOB_USD/CAN"
    url = http_client.get.call_args[0][0]
    assert url.startswith("https://api.imf.org/external/sdmx/2.1/data/IMF.STA,IMTS/")
    assert f".XG_FOB_USD.USA+DEU+CAN+G163+{IMTS_WORLD}.A?startPeriod=2020" in url
    assert "/USA+DEU+CAN+G163.XG_FOB_USD." in url


def test_headers_set_on_real_session(tmp_path):
    s = requests.Session()
    ImtsSource(client=s, cache_dir=tmp_path, user_agent="ua-test/1")
    assert s.headers["User-Agent"] == "ua-test/1"
    assert s.headers["Accept"] == IMTS_ACCEPT


def test_403_fails_fast(http_client, tmp_path):
    http_client.get.return_value = _resp("denied", status_code=403)
    src = ImtsSource(client=http_client, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="403"):
        src.fetch(IMTS_FLOWS[0], _basket(), use_cache=False)
    assert http_client.get.call_count == 1


def test_header_only_and_garbage_return_empty(http_client, tmp_path):
    src = ImtsSource(client=http_client, cache_dir=tmp_path)
    http_client.get.return_value = _resp(HEADER + "\n")
    assert src.fetch(IMTS_FLOWS[0], _basket(), use_cache=False).empty
    http_client.get.return_value = _resp("<html>nope</html>")
    assert src.fetch(IMTS_FLOWS[0], _basket(), use_cache=False).empty
    assert src.fetch(IMTS_FLOWS[0], [], use_cache=False).empty
    assert http_client.get.call_count == 2


def test_cache_hit(http_client, tmp_path):
    http_client.get.return_value = _resp(SAMPLE)
    src = ImtsSource(client=http_client, cache_dir=tmp_path)
    src.fetch(IMTS_FLOWS[0], _basket(), use_cache=True, today=date(2026, 1, 1))
    src.fetch(IMTS_FLOWS[0], _basket(), use_cache=True, today=date(2026, 1, 1))
    assert http_client.get.call_count == 1


def test_flow_bundle_and_registry_codes():
    assert [s.indicator_prefix for s in IMTS_FLOWS] == ["exports_to", "imports_from"]
    assert IMTS_FLOWS[1].imts_code == "MG_CIF_USD"
    assert ImtsSpec("exports", "X", "exports_to").indicator("WLD") == "exports_to_WLD"
    assert ISO2_TO_IMTS["EU"] == "G163" and ISO2_TO_IMTS["US"] == "USA" and len(ISO2_TO_IMTS) == 22
