"""Shared cached HTTP text fetcher for the keyless JSON/SDMX-CSV adapters.

Every keyless adapter (IMF DataMapper, IMF IMTS, IMF CPI, OEC, OECD) needs the
same four things: a User-Agent (Akamai fronts refuse anonymous clients), an
on-disk cache keyed by URL, fail-fast on 403/404 (permanent — never retried)
and a short exponential retry on 5xx / transport errors. Until slice 26 that
body was copy-pasted per adapter; this module is the one implementation.

`bis.py` and `worldbank.py` keep their own fetchers (older, different
signatures, both tested) — not migrated on purpose.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
from pathlib import Path
from typing import Protocol

import requests

logger = logging.getLogger(__name__)

SDMX_CSV_ACCEPT = "application/vnd.sdmx.data+csv;version=1.0.0"
DEFAULT_USER_AGENT = "dalio-machine/0.1 (+https://github.com/Rosinco/dalio-machine)"


class HttpClient(Protocol):
    def get(self, url: str, *, timeout: float = ...) -> requests.Response: ...


def set_default_headers(client: HttpClient, user_agent: str, accept: str | None = None) -> None:
    """Put the UA (and optionally an Accept) on a real `requests.Session`;
    silently skip clients without a headers mapping (mocks)."""
    headers = getattr(client, "headers", None)
    if isinstance(headers, dict | requests.structures.CaseInsensitiveDict):
        headers["User-Agent"] = user_agent
        if accept:
            headers["Accept"] = accept


class CachedTextFetcher:
    """GET a URL as text with a TTL disk cache and the retry policy above.

    `label` names the provider in log lines and error messages; `forbidden_hint`
    is appended to the 403 message (each provider fails differently).
    """

    def __init__(
        self,
        client: HttpClient,
        cache_dir: Path,
        cache_ttl_hours: float = 24.0,
        *,
        label: str,
        timeout: float = 60,
        suffix: str = ".csv",
        forbidden_hint: str = "try another network",
    ):
        self._client = client
        self._cache_dir = cache_dir
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_seconds = cache_ttl_hours * 3600
        self._label = label
        self._timeout = timeout
        self._suffix = suffix
        self._forbidden_hint = forbidden_hint

    @staticmethod
    def resolve_cache_dir(cache_dir: Path | None, env_var: str, default: str) -> Path:
        return cache_dir or Path(os.environ.get(env_var, default))

    def fetch(self, url: str, use_cache: bool = True, attempts: int = 3,
              backoff_base: float = 1.0) -> str:
        cache_path = self.cache_path_for(url)
        if use_cache and cache_path.exists():
            age = time.time() - cache_path.stat().st_mtime
            if age < self._cache_ttl_seconds:
                logger.debug("%s cache hit (age=%.0fs): %s", self._label, age, url)
                return cache_path.read_text()
        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                resp = self._client.get(url, timeout=self._timeout)
                if resp.status_code == 403:
                    raise ValueError(
                        f"{self._label} refused the request (403 — {self._forbidden_hint}): {url}"
                    )
                if resp.status_code == 404:
                    raise ValueError(f"{self._label}: resource not found or moved (404): {url}")
                if resp.status_code >= 500:
                    raise RuntimeError(f"{self._label} server error {resp.status_code}: {url}")
                resp.raise_for_status()
                cache_path.write_text(resp.text)
                return resp.text
            except (ValueError, FileNotFoundError):
                raise                                   # permanent — never retried
            except Exception as e:  # noqa: BLE001
                last_error = e
                if attempt < attempts - 1:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning("%s %s attempt %d/%d failed (%s) — retrying in %.1fs",
                                   self._label, url, attempt + 1, attempts, e, wait)
                    time.sleep(wait)
        assert last_error is not None
        raise last_error

    def cache_path_for(self, url: str) -> Path:
        h = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self._cache_dir / f"{h}{self._suffix}"
