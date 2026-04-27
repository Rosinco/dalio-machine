"""Tests for big-cycle qualitative loaders (Slice 17)."""
from unittest.mock import MagicMock

import pytest

from dalio.scoring.big_cycle import ISO2_TO_WB, BigCycleSource

SAMPLE_WB_GINI_JSON = """[
  {"page": 1, "pages": 1, "per_page": 200, "total": 4, "sourceid": "2"},
  [
    {"indicator": {"id": "SI.POV.GINI", "value": "Gini index"},
     "country": {"id": "US", "value": "United States"},
     "countryiso3code": "USA", "date": "2024", "value": 41.8, "decimal": 1},
    {"indicator": {"id": "SI.POV.GINI", "value": "Gini index"},
     "country": {"id": "US", "value": "United States"},
     "countryiso3code": "USA", "date": "2023", "value": 41.8, "decimal": 1},
    {"indicator": {"id": "SI.POV.GINI", "value": "Gini index"},
     "country": {"id": "US", "value": "United States"},
     "countryiso3code": "USA", "date": "2022", "value": null, "decimal": 1},
    {"indicator": {"id": "SI.POV.GINI", "value": "Gini index"},
     "country": {"id": "US", "value": "United States"},
     "countryiso3code": "USA", "date": "2021", "value": 39.7, "decimal": 1}
  ]
]"""

SAMPLE_WB_NO_DATA = """[
  {"message": [{"id": "120", "key": "Format", "value": "No data found"}]}
]"""

SAMPLE_COFER_XML = """<?xml version='1.0' encoding='UTF-8'?>
<message:StructureSpecificData xmlns:message="urn:msg" xmlns:ss="urn:ss">
  <message:DataSet>
    <Series COUNTRY="G001" INDICATOR="AFXRA" FXR_CURRENCY="CI_USD" TYPE_OF_TRANSFORMATION="SHRO_PT" FREQUENCY="Q">
      <Obs TIME_PERIOD="2025-Q4" OBS_VALUE="56.7701683044434" />
      <Obs TIME_PERIOD="2025-Q3" OBS_VALUE="56.9322395324707" />
      <Obs TIME_PERIOD="2000-Q1" OBS_VALUE="70.7965528604486" />
    </Series>
  </message:DataSet>
</message:StructureSpecificData>"""

SAMPLE_COFER_EMPTY = """<?xml version='1.0' encoding='UTF-8'?>
<message:StructureSpecificData xmlns:message="urn:msg">
  <message:DataSet />
</message:StructureSpecificData>"""


@pytest.fixture
def http_client():
    return MagicMock()


def _mock_response(text: str, status_code: int = 200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def test_iso2_to_wb_mapping_covers_basket():
    assert {"US", "CN", "EU", "UK", "JP", "SE", "IN", "BR"}.issubset(ISO2_TO_WB.keys())
    assert ISO2_TO_WB["EU"] == "EMU"
    assert ISO2_TO_WB["US"] == "USA"


def test_fetch_gini_parses_long_format(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_WB_GINI_JSON)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch_gini("US", use_cache=False)
    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]
    # 4 rows in JSON, but the null-value 2022 row is dropped → 3 obs
    assert len(df) == 3
    assert df["country"].iloc[0] == "US"
    assert df["indicator"].iloc[0] == "gini"
    assert df["source"].iloc[0] == "WORLD_BANK"
    # Sorted ascending by date
    assert df["date"].is_monotonic_increasing
    assert df["value"].iloc[-1] == pytest.approx(41.8)


def test_fetch_gini_empty_response_returns_empty_frame(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_WB_NO_DATA)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch_gini("US", use_cache=False)
    assert df.empty
    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]


def test_fetch_gini_unknown_country_raises(http_client, tmp_path):
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    with pytest.raises(KeyError, match="No World Bank code"):
        src.fetch_gini("ZZ", use_cache=False)


def test_fetch_cofer_parses_long_format(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_COFER_XML)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch_cofer_usd_share(use_cache=False)
    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]
    assert len(df) == 3
    assert df["country"].iloc[0] == "WLD"
    assert df["indicator"].iloc[0] == "cofer_usd_share"
    # Sorted ascending — Q1 2000 should come first
    assert df["date"].is_monotonic_increasing
    assert df["value"].iloc[0] == pytest.approx(70.797, abs=1e-3)
    # Latest value matches the historical decline pattern
    assert df["value"].iloc[-1] < df["value"].iloc[0]


def test_fetch_cofer_empty_xml_returns_empty_frame(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_COFER_EMPTY)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch_cofer_usd_share(use_cache=False)
    assert df.empty


def test_cache_hit_avoids_second_http_call(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_COFER_XML)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    src.fetch_cofer_usd_share(use_cache=True)
    src.fetch_cofer_usd_share(use_cache=True)
    assert http_client.get.call_count == 1


def test_404_fails_fast_no_retry(http_client, tmp_path):
    http_client.get.return_value = _mock_response("not found", status_code=404)
    src = BigCycleSource(client=http_client, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="404"):
        src.fetch_gini("US", use_cache=False)
    # No retry on 4xx
    assert http_client.get.call_count == 1
