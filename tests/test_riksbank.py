import json
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from dalio.data_sources.riksbank import (
    DEFAULT_UNAUTHENTICATED_PACING_SECONDS,
    NOK_COMPARABLE_FROM,
    RIKSBANK_API_BASE,
    RIKSBANK_SERIES,
    RiksbankSource,
)

FIXTURES = Path(__file__).parent / "fixtures" / "riksbank"
EXPECTED_COLUMNS = ["country", "indicator", "date", "value", "source", "series_id"]


def _response(
    fixture: str | None = None,
    *,
    status_code: int = 200,
    payload: object | None = None,
    headers: dict[str, str] | None = None,
) -> MagicMock:
    response = MagicMock()
    response.status_code = status_code
    response.headers = headers or {}
    if fixture is not None:
        response.text = (FIXTURES / fixture).read_text(encoding="utf-8")
    else:
        response.text = json.dumps(payload)
    return response


def _spec(indicator: str):
    return next(spec for spec in RIKSBANK_SERIES if spec.indicator == indicator)


def test_catalogue_contains_only_the_eight_verified_swea_series():
    assert [(spec.series_id, spec.indicator, spec.unit) for spec in RIKSBANK_SERIES] == [
        ("SECBREPOEFF", "policy_rate", "percent"),
        ("SEGVB2YC", "yield_2y", "percent"),
        ("SEGVB5YC", "yield_5y", "percent"),
        ("SEGVB10YC", "yield_10y", "percent"),
        ("SEKUSDPMI", "sek_per_usd", "SEK per USD"),
        ("SEKEURPMI", "sek_per_eur", "SEK per EUR"),
        ("SEKNOKPMI", "sek_per_nok", "SEK per NOK"),
        ("SEKGBPPMI", "sek_per_gbp", "SEK per GBP"),
    ]
    assert {spec.country for spec in RIKSBANK_SERIES} == {"SE"}
    assert {spec.frequency for spec in RIKSBANK_SERIES} == {"D"}
    assert all("KIX" not in spec.series_id for spec in RIKSBANK_SERIES)


def test_fetch_parses_official_policy_fixture_without_a_live_call(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("policy_rate.json")
    source = RiksbankSource(client=client, cache_dir=tmp_path)

    frame = source.fetch(
        _spec("policy_rate"),
        from_date=date(2025, 6, 2),
        to_date=date(2025, 6, 19),
        use_cache=False,
    )

    assert list(frame.columns) == EXPECTED_COLUMNS
    assert len(frame) == 13
    assert frame.iloc[0].to_dict() == {
        "country": "SE",
        "indicator": "policy_rate",
        "date": date(2025, 6, 2),
        "value": 2.25,
        "source": "RIKSBANK_SWEA",
        "series_id": "SECBREPOEFF",
    }
    url = f"{RIKSBANK_API_BASE}/Observations/SECBREPOEFF/2025-06-02/2025-06-19"
    assert client.get.call_args.args == (url,)
    assert client.get.call_args.kwargs["headers"] == {"Accept": "application/json"}


def test_fetch_preserves_fx_precision_and_sends_optional_api_key_header(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("sek_usd.json")
    source = RiksbankSource(client=client, cache_dir=tmp_path, api_key="test-key")

    frame = source.fetch(
        _spec("sek_per_usd"),
        from_date="2025-06-16",
        to_date="2025-06-19",
        use_cache=False,
    )

    assert list(frame["value"]) == [9.4708, 9.46404, 9.58203, 9.64192]
    assert client.get.call_args.kwargs["headers"] == {
        "Accept": "application/json",
        "Ocp-Apim-Subscription-Key": "test-key",
    }
    assert source.recommended_pacing_seconds < DEFAULT_UNAUTHENTICATED_PACING_SECONDS


def test_unauthenticated_source_advertises_safe_five_per_minute_pacing(tmp_path):
    source = RiksbankSource(client=MagicMock(), cache_dir=tmp_path)
    assert DEFAULT_UNAUTHENTICATED_PACING_SECONDS > 12
    assert source.recommended_pacing_seconds == DEFAULT_UNAUTHENTICATED_PACING_SECONDS


def test_nok_history_is_clamped_before_the_one_unit_quote_became_comparable(tmp_path):
    source = RiksbankSource(client=MagicMock(), cache_dir=tmp_path)
    nok = _spec("sek_per_nok")

    url = source.url_for(
        nok,
        from_date="2005-01-01",
        to_date="2024-01-31",
    )

    assert nok.history_start == NOK_COMPARABLE_FROM == date(2023, 11, 27)
    assert url.endswith("/Observations/SEKNOKPMI/2023-11-27/2024-01-31")


def test_empty_204_response_returns_the_standard_empty_frame(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(status_code=204, payload=None)
    source = RiksbankSource(client=client, cache_dir=tmp_path)

    frame = source.fetch(
        _spec("policy_rate"),
        from_date="2025-06-02",
        to_date="2025-06-19",
        use_cache=False,
    )

    assert frame.empty
    assert list(frame.columns) == EXPECTED_COLUMNS


def test_fresh_cache_avoids_a_second_http_request(tmp_path):
    client = MagicMock()
    client.get.return_value = _response("policy_rate.json")
    source = RiksbankSource(client=client, cache_dir=tmp_path)
    kwargs = {
        "from_date": "2025-06-02",
        "to_date": "2025-06-19",
        "use_cache": True,
    }

    source.fetch(_spec("policy_rate"), **kwargs)
    source.fetch(_spec("policy_rate"), **kwargs)

    assert client.get.call_count == 1


def test_429_honours_retry_after_before_retrying(tmp_path):
    client = MagicMock()
    client.get.side_effect = [
        _response(status_code=429, payload={"error": "limited"}, headers={"Retry-After": "7"}),
        _response("policy_rate.json"),
    ]
    source = RiksbankSource(client=client, cache_dir=tmp_path)

    with patch("dalio.data_sources.riksbank.time.sleep") as sleep:
        frame = source.fetch(
            _spec("policy_rate"),
            from_date="2025-06-02",
            to_date="2025-06-19",
            use_cache=False,
        )

    sleep.assert_called_once_with(7.0)
    assert len(frame) == 13
    assert client.get.call_count == 2


def test_permanent_4xx_fails_without_retry(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(status_code=404, payload={"error": "missing"})
    source = RiksbankSource(client=client, cache_dir=tmp_path)

    with pytest.raises(ValueError, match="404"):
        source.fetch(
            _spec("policy_rate"),
            from_date="2025-06-02",
            to_date="2025-06-19",
            use_cache=False,
        )

    assert client.get.call_count == 1


def test_malformed_payload_fails_closed(tmp_path):
    client = MagicMock()
    client.get.return_value = _response(payload={"observations": []})
    source = RiksbankSource(client=client, cache_dir=tmp_path)

    with pytest.raises(ValueError, match="JSON list"):
        source.fetch(
            _spec("policy_rate"),
            from_date="2025-06-02",
            to_date="2025-06-19",
            use_cache=False,
        )
