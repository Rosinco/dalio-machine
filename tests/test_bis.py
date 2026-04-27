from unittest.mock import MagicMock

import pytest

from dalio.data_sources.bis import (
    ISO2_TO_BIS,
    TIER_1_DSR,
    TIER_1_TOTAL_CREDIT,
    BisSource,
    DsrSpec,
    Sector,
    TotalCreditSpec,
)

SAMPLE_TC_CSV = """FREQ,BORROWERS_CTY,TC_BORROWERS,TC_LENDERS,VALUATION,UNIT_TYPE,TC_ADJUST,TIME_PERIOD,OBS_VALUE
Q,US,P,A,M,770,A,2025-Q2,140.9
Q,US,P,A,M,770,A,2025-Q3,140.4
Q,US,P,A,M,770,A,2024-Q3,142.0
"""

SAMPLE_DSR_CSV = """FREQ,BORROWERS_CTY,BORROWERS,TIME_PERIOD,OBS_VALUE
Q,US,P,2025-Q2,14.0
Q,US,P,2025-Q3,14.1
"""

SAMPLE_EMPTY_CSV = "FREQ,BORROWERS_CTY,TIME_PERIOD,OBS_VALUE\n"


@pytest.fixture
def http_client():
    c = MagicMock()
    return c


def _mock_response(text: str, status_code: int = 200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def test_iso2_to_bis_mapping_covers_basket():
    expected = {"US", "CN", "EU", "UK", "JP", "SE", "IN", "BR"}
    assert expected.issubset(ISO2_TO_BIS.keys())
    # EU maps to XM aggregate
    assert ISO2_TO_BIS["EU"] == "XM"
    # UK maps to GB
    assert ISO2_TO_BIS["UK"] == "GB"


def test_fetch_total_credit_parses_long_format(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_TC_CSV)
    src = BisSource(client=http_client, cache_dir=tmp_path)
    spec = TotalCreditSpec("private_nonfin_pct_gdp", "US", Sector.PRIVATE_NON_FIN)
    df = src.fetch_total_credit(spec, use_cache=False)
    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]
    assert len(df) == 3
    assert df["country"].iloc[0] == "US"
    assert df["indicator"].iloc[0] == "private_nonfin_pct_gdp"
    assert df["source"].iloc[0] == "BIS_TC"
    # Latest US private debt ~140.4% per sample
    latest = df.sort_values("date").iloc[-1]
    assert latest["value"] == pytest.approx(140.4)


def test_fetch_dsr_parses_long_format(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_DSR_CSV)
    src = BisSource(client=http_client, cache_dir=tmp_path)
    spec = DsrSpec("debt_service_ratio", "US")
    df = src.fetch_dsr(spec, use_cache=False)
    assert df["source"].iloc[0] == "BIS_DSR"
    assert len(df) == 2


def test_quarter_string_parsed_to_first_month_of_quarter():
    # 2025-Q2 → 2025-04-01 (first month of Q2)
    d = BisSource._period_to_date("2025-Q2")
    assert d.year == 2025
    assert d.month == 4
    assert d.day == 1


def test_monthly_string_parsed():
    d = BisSource._period_to_date("2025-M03")
    assert d.year == 2025 and d.month == 3 and d.day == 1


def test_annual_string_parsed():
    d = BisSource._period_to_date("2025")
    assert d.year == 2025 and d.month == 1 and d.day == 1


def test_fetch_404_fails_fast(http_client, tmp_path):
    http_client.get.return_value = _mock_response("", status_code=404)
    src = BisSource(client=http_client, cache_dir=tmp_path)
    spec = TotalCreditSpec("foo", "US", "C")
    with pytest.raises(ValueError, match="not found"):
        src.fetch_total_credit(spec, use_cache=False)
    # Only one call — no retry on 4xx
    assert http_client.get.call_count == 1


def test_fetch_uses_cache_when_fresh(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_TC_CSV)
    src = BisSource(client=http_client, cache_dir=tmp_path)
    spec = TotalCreditSpec("private_nonfin_pct_gdp", "US", Sector.PRIVATE_NON_FIN)
    # First call hits the network
    src.fetch_total_credit(spec, use_cache=True)
    assert http_client.get.call_count == 1
    # Second call should hit cache
    src.fetch_total_credit(spec, use_cache=True)
    assert http_client.get.call_count == 1


def test_empty_csv_returns_empty_dataframe(http_client, tmp_path):
    http_client.get.return_value = _mock_response(SAMPLE_EMPTY_CSV)
    src = BisSource(client=http_client, cache_dir=tmp_path)
    spec = TotalCreditSpec("foo", "US", "C")
    df = src.fetch_total_credit(spec, use_cache=False)
    assert df.empty
    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]


def test_tier_1_total_credit_covers_six_countries():
    countries = {s.country for s in TIER_1_TOTAL_CREDIT}
    assert countries == {"US", "CN", "EU", "UK", "JP", "SE"}
    # 5 sectors per country
    assert len(TIER_1_TOTAL_CREDIT) == 6 * 5


def test_tier_1_dsr_covers_six_countries():
    countries = {s.country for s in TIER_1_DSR}
    assert countries == {"US", "CN", "EU", "UK", "JP", "SE"}
