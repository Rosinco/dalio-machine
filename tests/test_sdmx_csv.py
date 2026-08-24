"""Shared cached fetcher (slice 26): cache TTL, permanent errors, transient retry."""
import time
from unittest.mock import MagicMock, patch

import pytest
import requests

from dalio.data_sources.sdmx_csv import CachedTextFetcher, set_default_headers


def _resp(text, status_code=200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def test_cache_hit_within_ttl_and_miss_after(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp("v1")
    f = CachedTextFetcher(client, tmp_path, cache_ttl_hours=1, label="T")
    assert f.fetch("http://x/a", use_cache=True) == "v1"
    client.get.return_value = _resp("v2")
    assert f.fetch("http://x/a", use_cache=True) == "v1"          # served from cache
    assert client.get.call_count == 1
    assert f.fetch("http://x/a", use_cache=False) == "v2"         # bypass rewrites the cache
    path = f.cache_path_for("http://x/a")
    assert path.suffix == ".csv" and path.read_text() == "v2"
    old = time.time() - 2 * 3600
    import os
    os.utime(path, (old, old))
    client.get.return_value = _resp("v3")
    assert f.fetch("http://x/a", use_cache=True) == "v3"          # expired → refetched


def test_403_and_404_fail_fast_without_retry(tmp_path):
    client = MagicMock()
    f = CachedTextFetcher(client, tmp_path, label="Prov", forbidden_hint="hint")
    client.get.return_value = _resp("no", 403)
    with pytest.raises(ValueError, match="Prov.*403.*hint"):
        f.fetch("http://x/b", use_cache=False)
    client.get.return_value = _resp("no", 404)
    with pytest.raises(ValueError, match="404"):
        f.fetch("http://x/b", use_cache=False)
    assert client.get.call_count == 2


def test_5xx_retries_then_succeeds(tmp_path):
    client = MagicMock()
    client.get.side_effect = [_resp("boom", 503), requests.ConnectionError("net"), _resp("ok")]
    f = CachedTextFetcher(client, tmp_path, label="T", suffix=".json")
    with patch("dalio.data_sources.sdmx_csv.time.sleep") as sleep:
        assert f.fetch("http://x/c", use_cache=False) == "ok"
    assert client.get.call_count == 3 and sleep.call_count == 2
    assert f.cache_path_for("http://x/c").suffix == ".json"


def test_5xx_exhausts_and_raises_last_error(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp("boom", 500)
    f = CachedTextFetcher(client, tmp_path, label="T")
    with patch("dalio.data_sources.sdmx_csv.time.sleep"), pytest.raises(RuntimeError, match="500"):
        f.fetch("http://x/d", use_cache=False, attempts=2)


def test_set_default_headers_only_on_real_sessions():
    s = requests.Session()
    set_default_headers(s, "ua/1", "text/csv")
    assert s.headers["User-Agent"] == "ua/1" and s.headers["Accept"] == "text/csv"
    set_default_headers(MagicMock(spec=[]), "ua/1")               # no headers attr → no-op
