from unittest.mock import MagicMock

import pandas as pd
import pytest

from dalio.data_sources.fred import FredSeriesSpec, FredSource


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
