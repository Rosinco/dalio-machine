"""OEC ECI adapter (slice 23) + pipeline wiring. Mocked HTTP only."""
import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sqlalchemy import select

from dalio.countries import get_country
from dalio.data_sources.oec import INDICATOR_ECI, OEC_SERIES_ID, SOURCE_OEC, OecSource
from dalio.pipelines import fetch_fundamentals
from dalio.scoring.fundamentals import FUNDAMENTALS
from dalio.storage.db import Observation, make_engine

SAMPLE = json.dumps({"data": [
    {"Country ID": "asjpn", "Country": "Japan", "Year": 2023, "ECI": 2.19},
    {"Country ID": "asjpn", "Country": "Japan", "Year": 1990, "ECI": 2.0},        # < start_year
    {"Country ID": "nausa", "Country": "United States", "Year": 2023, "ECI": 1.59},
    {"Country ID": "nausa", "Country": "United States", "Year": 2024, "ECI": None},  # unusable
    {"Country ID": "afago", "Country": "Angola", "Year": 2023, "ECI": -1.35},     # not in basket
    {"Country ID": "eudeu", "Country": "Germany", "Year": 2023, "ECI": 1.9},
    {"Country ID": "eufra", "Country": "France", "Year": 2023, "ECI": 1.29},
    {"Country ID": "euita", "Country": "Italy", "Year": 2023, "ECI": 1.22},
    {"Country ID": "xx", "Year": 2023, "ECI": 1.0},                               # malformed id
]})


def _resp(text, status_code=200):
    r = MagicMock()
    r.text = text
    r.status_code = status_code
    r.raise_for_status.return_value = None
    return r


def _basket():
    return [get_country(c) for c in ("US", "JP", "DE", "FR", "IT", "EU")]


def test_fetch_eci_maps_ids_and_filters(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp(SAMPLE)
    df = OecSource(client=client, cache_dir=tmp_path).fetch_eci(_basket(), use_cache=False)
    assert set(df["country"]) == {"JP", "US", "DE", "FR", "IT"}                 # EU aggregate not requested
    jp = df[df["country"] == "JP"]
    assert len(jp) == 1 and jp["date"].iloc[0] == date(2023, 12, 31) and jp["value"].iloc[0] == 2.19
    assert set(df["source"]) == {SOURCE_OEC} and set(df["series_id"]) == {OEC_SERIES_ID}
    assert set(df["indicator"]) == {INDICATOR_ECI}
    assert client.get.call_args[0][0].startswith("https://oec.world/api/olap-proxy/data.jsonrecords?cube=")


def test_fetch_eci_errors_and_empty(tmp_path):
    client = MagicMock()
    src = OecSource(client=client, cache_dir=tmp_path)
    client.get.return_value = _resp("<html>", 404)
    with pytest.raises(ValueError, match="404"):
        src.fetch_eci(_basket(), use_cache=False)
    client.get.return_value = _resp("not json")
    assert src.fetch_eci(_basket(), use_cache=False).empty
    client.get.return_value = _resp(json.dumps({"data": "nope"}))
    assert src.fetch_eci(_basket(), use_cache=False).empty
    assert src.fetch_eci([get_country("EU")], use_cache=False).empty              # nothing on_map → no call
    assert client.get.call_count == 3


def test_cache_hit(tmp_path):
    client = MagicMock()
    client.get.return_value = _resp(SAMPLE)
    src = OecSource(client=client, cache_dir=tmp_path)
    src.fetch_eci(_basket())
    src.fetch_eci(_basket())
    assert client.get.call_count == 1


def test_registry_has_economic_complexity_in_production():
    spec = next(s for s in FUNDAMENTALS if s.name == INDICATOR_ECI)
    assert spec.category == "production" and spec.uncertainty == "C" and spec.higher_is_better
    assert spec.preferred_sources == (SOURCE_OEC,)
    assert len(FUNDAMENTALS) == 16


class _FakeOec:
    def __init__(self, df):
        self._df = df

    def fetch_eci(self, countries, use_cache=True, start_year=1995):
        if isinstance(self._df, Exception):
            raise self._df
        return self._df


def test_pipeline_oec_stores_rows_and_member_mean(tmp_path, monkeypatch):
    monkeypatch.setenv("DALIO_DB_PATH", str(tmp_path / "p.db"))
    df = pd.DataFrame([
        {"country": c, "indicator": INDICATOR_ECI, "date": date(2023, 12, 31), "value": v,
         "source": SOURCE_OEC, "series_id": OEC_SERIES_ID}
        for c, v in (("US", 1.59), ("DE", 1.9), ("FR", 1.29), ("IT", 1.22))
    ])
    basket = [get_country(c) for c in ("US", "DE", "FR", "IT", "EU")]
    summary = fetch_fundamentals.run_pipeline(("oec",), countries=basket, use_cache=False,
                                              oec_source=_FakeOec(df))
    assert summary["oec/economic_complexity"]["rows"] == 4
    assert summary["oec/economic_complexity:EU"]["rows"] == 1
    with make_engine(tmp_path / "p.db").connect() as conn:
        rows = conn.execute(select(Observation.country, Observation.value, Observation.series_id)).all()
    eu = next(r for r in rows if r[0] == "EU")
    assert eu[1] == pytest.approx((1.9 + 1.29 + 1.22) / 3) and eu[2].endswith(":member-mean")
    bad = fetch_fundamentals.run_pipeline(("oec",), countries=basket, use_cache=False,
                                          oec_source=_FakeOec(RuntimeError("down")))
    assert "down" in bad["oec/economic_complexity"]["error"]
