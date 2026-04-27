from unittest.mock import MagicMock

import pandas as pd
import pytest

from dalio.data_sources.fred import (
    TIER_1_SERIES,
    US_SHORT_TERM_SERIES,
    FredSeriesSpec,
    FredSource,
    specs_for_countries,
)


@pytest.fixture
def mock_client():
    return MagicMock()


def test_fetch_raw_returns_long_format(mock_client):
    idx = pd.date_range("2025-01-01", periods=3, freq="MS")
    mock_client.get_series.return_value = pd.Series([1.0, 2.0, 3.0], index=idx)

    src = FredSource(client=mock_client)
    spec = FredSeriesSpec("policy_rate", "DFF", "US", frequency="M")
    df = src.fetch(spec)

    assert list(df.columns) == [
        "country", "indicator", "date", "value", "source", "series_id",
    ]
    assert len(df) == 3
    assert df["country"].iloc[0] == "US"
    assert df["indicator"].iloc[0] == "policy_rate"
    assert df["source"].iloc[0] == "FRED"
    assert df["series_id"].iloc[0] == "DFF"
    assert df["value"].tolist() == [1.0, 2.0, 3.0]


def test_fetch_yoy_transform(mock_client):
    idx = pd.date_range("2024-01-01", periods=13, freq="MS")
    values = [100.0] * 12 + [110.0]  # 10% YoY at month 13
    mock_client.get_series.return_value = pd.Series(values, index=idx)

    src = FredSource(client=mock_client)
    spec = FredSeriesSpec("cpi_yoy", "CPIAUCSL", "US", frequency="M", transform="yoy")
    df = src.fetch(spec)

    assert len(df) == 1
    assert df["value"].iloc[0] == pytest.approx(10.0)


def test_fetch_drops_na_values(mock_client):
    idx = pd.date_range("2025-01-01", periods=3, freq="MS")
    mock_client.get_series.return_value = pd.Series([1.0, float("nan"), 3.0], index=idx)

    src = FredSource(client=mock_client)
    spec = FredSeriesSpec("policy_rate", "DFF", "US", frequency="M")
    df = src.fetch(spec)

    assert len(df) == 2
    assert df["value"].tolist() == [1.0, 3.0]


def test_yoy_unsupported_frequency_raises():
    with pytest.raises(ValueError, match="not supported"):
        FredSource._yoy(pd.Series([1.0, 2.0]), "D")


def test_unknown_transform_raises(mock_client):
    idx = pd.date_range("2025-01-01", periods=2, freq="MS")
    mock_client.get_series.return_value = pd.Series([1.0, 2.0], index=idx)

    src = FredSource(client=mock_client)
    spec = FredSeriesSpec("foo", "BAR", "US", frequency="M", transform="bogus")
    with pytest.raises(ValueError, match="Unknown transform"):
        src.fetch(spec)


def test_no_api_key_raises(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="FRED_API_KEY"):
        FredSource()


def test_explicit_api_key_used(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    # Just ensure it doesn't raise — actual Fred() is constructed but never called.
    src = FredSource(api_key="fake_key_for_test")
    assert src._client is not None


# ─── Country-map tests ─────────────────────────────────────────────────────

def test_tier_1_covers_six_countries():
    countries = {s.country for s in TIER_1_SERIES}
    assert countries == {"US", "CN", "EU", "UK", "JP", "SE"}


def test_us_subset_alias_matches_filter():
    assert specs_for_countries(("US",)) == US_SHORT_TERM_SERIES


def test_specs_for_countries_filter():
    eu_jp = specs_for_countries(("EU", "JP"))
    assert {s.country for s in eu_jp} == {"EU", "JP"}
    assert all(s.country in {"EU", "JP"} for s in eu_jp)


def test_specs_for_countries_none_returns_all():
    assert specs_for_countries(None) == TIER_1_SERIES


def test_us_has_complete_indicator_set():
    us_indicators = {s.indicator for s in TIER_1_SERIES if s.country == "US"}
    assert us_indicators == {
        "policy_rate", "cpi_yoy", "unemployment_rate",
        "yield_10y", "yield_2y", "real_gdp_yoy",
    }


def test_known_data_gaps_documented():
    # JP CPI is a documented gap (FRED's OECD-MEI mirror discontinued 2021)
    jp_indicators = {s.indicator for s in TIER_1_SERIES if s.country == "JP"}
    assert "cpi_yoy" not in jp_indicators
    # CN has no FRED 10Y yield series and no quarterly GDP
    cn_indicators = {s.indicator for s in TIER_1_SERIES if s.country == "CN"}
    assert "yield_10y" not in cn_indicators


def test_retry_skips_4xx_errors(mock_client):
    """Bad-series-ID errors should fail fast (no retry waste)."""
    mock_client.get_series.side_effect = ValueError("Bad Request.  The series does not exist.")
    src = FredSource(client=mock_client)
    spec = FredSeriesSpec("foo", "BOGUS", "US", frequency="M")
    with pytest.raises(ValueError, match="does not exist"):
        src.fetch(spec)
    # Only one attempt made (no retry on 4xx)
    assert mock_client.get_series.call_count == 1
