"""Tests for the World Bank adapter (slice 18). Mocked HTTP only."""
import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from dalio.countries import COUNTRIES, get_country
from dalio.data_sources.worldbank import (
    WB_FUNDAMENTALS,
    WbIndicatorSpec,
    WorldBankSource,
)


def _page(meta: dict, rows: list[dict]) -> str:
    return json.dumps([meta, rows])


def _obs(iso3: str, year: int, value, code="NY.GDP.PCAP.PP.KD"):
    return {
        "indicator": {"id": code, "value": "x"},
        "country": {"id": iso3[:2], "value": iso3},
        "countryiso3code": iso3,
        "date": str(year),
        "value": value,
        "unit": "", "obs_status": "", "decimal": 1,
    }


PAGE_1 = _page(
    {"page": 1, "pages": 2, "per_page": 1000, "total": 5, "lastupdated": "2026-07-13"},
    [_obs("USA", 2024, 80000.0), _obs("USA", 2025, None), _obs("SWE", 2024, 60000.0)],
)
PAGE_2 = _page(
    {"page": 2, "pages": 2, "per_page": 1000, "total": 5, "lastupdated": "2026-07-13"},
    [_obs("EMU", 2024, 55000.0), _obs("WLD", 2024, 20000.0), _obs("ZZZ", 2024, 1.0)],
)
ERROR_SHAPE = json.dumps([{"message": [{"id": "120", "key": "Invalid value", "value": "bad"}]}])


@pytest.fixture
def http_client():
    return MagicMock()


def _resp(text: str, status_code: int = 200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def _basket():
    return [get_country("US"), get_country("SE"), get_country("EU")]


def test_fetch_paginates_and_maps_codes(http_client, tmp_path):
    http_client.get.side_effect = [_resp(PAGE_1), _resp(PAGE_2)]
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    spec = WbIndicatorSpec("gdp_pc_ppp", "NY.GDP.PCAP.PP.KD", include_world=True)
    df = src.fetch(spec, _basket(), use_cache=False, today=date(2026, 8, 24))

    assert http_client.get.call_count == 2
    assert list(df.columns) == ["country", "indicator", "date", "value", "source", "series_id"]
    # null 2025 US row dropped; unknown ZZZ dropped; EMU→EU; WLD kept as WLD
    assert set(df["country"]) == {"US", "SE", "EU", "WLD"}
    assert len(df) == 4
    assert df["source"].unique().tolist() == ["WORLD_BANK"]
    assert df["series_id"].unique().tolist() == ["NY.GDP.PCAP.PP.KD"]
    assert df["date"].iloc[0] == date(2024, 12, 31)


def test_url_carries_source_range_and_codes(http_client, tmp_path):
    http_client.get.return_value = _resp(_page({"page": 1, "pages": 1}, []))
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    spec = WbIndicatorSpec("rule_of_law", "GOV_WGI_RL.EST", source_id=3, start_year=1996)
    df = src.fetch(spec, _basket(), use_cache=False, today=date(2026, 8, 24))
    url = http_client.get.call_args[0][0]
    assert "/country/USA;SWE;EMU/indicator/GOV_WGI_RL.EST" in url
    assert "source=3" in url
    assert "date=1996:2026" in url
    assert "per_page=1000" in url
    assert "mrv" not in url
    assert df.empty


def test_wgi_source_label(http_client, tmp_path):
    http_client.get.return_value = _resp(_page({"page": 1, "pages": 1}, [_obs("SWE", 2023, 1.69, "GOV_WGI_RL.EST")]))
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    df = src.fetch(WbIndicatorSpec("rule_of_law", "GOV_WGI_RL.EST", source_id=3), _basket(), use_cache=False)
    assert df["source"].iloc[0] == "WORLD_BANK_WGI"


def test_world_not_requested_unless_asked(http_client, tmp_path):
    http_client.get.return_value = _resp(_page({"page": 1, "pages": 1}, [_obs("WLD", 2024, 1.0)]))
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    df = src.fetch(WbIndicatorSpec("x", "X"), _basket(), use_cache=False)
    assert "WLD" not in http_client.get.call_args[0][0]
    assert df.empty  # WLD row ignored because it was not requested


def test_error_shape_returns_empty(http_client, tmp_path):
    http_client.get.return_value = _resp(ERROR_SHAPE)
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    df = src.fetch(WbIndicatorSpec("x", "X"), _basket(), use_cache=False)
    assert df.empty
    assert list(df.columns) == ["country", "indicator", "date", "value", "source", "series_id"]


def test_404_fails_fast(http_client, tmp_path):
    http_client.get.return_value = _resp("nope", status_code=404)
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    with pytest.raises(ValueError, match="404"):
        src.fetch(WbIndicatorSpec("x", "X"), _basket(), use_cache=False)
    assert http_client.get.call_count == 1


def test_cache_hit_avoids_second_call(http_client, tmp_path):
    http_client.get.return_value = _resp(_page({"page": 1, "pages": 1}, [_obs("USA", 2024, 1.0)]))
    src = WorldBankSource(client=http_client, cache_dir=tmp_path, page_pause_seconds=0)
    spec = WbIndicatorSpec("x", "X")
    src.fetch(spec, _basket(), use_cache=True, today=date(2026, 1, 1))
    src.fetch(spec, _basket(), use_cache=True, today=date(2026, 1, 1))
    assert http_client.get.call_count == 1


def test_fundamentals_bundle_is_tracer_trio_across_three_categories():
    names = [s.indicator for s in WB_FUNDAMENTALS]
    assert names == ["gdp_pc_ppp", "old_age_dependency", "military_pct_gdp"]
    assert all(s.source_id == 2 for s in WB_FUNDAMENTALS)


def test_every_registry_country_has_wb_id_for_fetch():
    assert all(c.wb_id for c in COUNTRIES)
