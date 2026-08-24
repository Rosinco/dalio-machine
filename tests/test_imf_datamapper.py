"""IMF DataMapper adapter + forward-panel scoring (slice 20). Mocked HTTP only."""
import json
from datetime import date
from unittest.mock import MagicMock

import pandas as pd
import pytest
import requests

from dalio.countries import get_country
from dalio.data_sources.imf_datamapper import (
    IMF_FUNDAMENTALS,
    SOURCE_FORECAST,
    SOURCE_HISTORY,
    ImfDataMapperSource,
    ImfSpec,
    derive_interest_burden,
)
from dalio.scoring.fundamentals import (
    FUNDAMENTALS,
    IndicatorSpec,
    build_snapshot,
    load_forward_panel,
    load_history,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

SAMPLE = json.dumps({"values": {"GGXWDG_NGDP": {
    "USA": {"2023": 120.0, "2024": 122.3, "2025": 123.9, "2026": 125.8, "2031": 142.1},
    "EURO": {"2024": 88.0, "2026": 89.5},
    "XYZ": {"2024": 1.0},
    "SWE": {"1979": 40.0, "2024": 33.9, "2030": None},
}}})


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
    return [get_country("US"), get_country("EU"), get_country("SE"), get_country("CN")]


def test_fetch_filters_maps_and_splits_forecasts(http_client, tmp_path):
    http_client.get.return_value = _resp(SAMPLE)
    src = ImfDataMapperSource(client=http_client, cache_dir=tmp_path)
    df = src.fetch(ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP"), _basket(), use_cache=False,
                   today=date(2026, 8, 24))
    assert set(df["country"]) == {"US", "EU", "SE"}          # XYZ dropped, CN absent
    us = df[df.country == "US"].set_index("date")
    assert us.loc[date(2025, 12, 31), "source"] == SOURCE_HISTORY
    assert us.loc[date(2026, 12, 31), "source"] == SOURCE_FORECAST
    assert us.loc[date(2031, 12, 31), "source"] == SOURCE_FORECAST
    eu = df[df.country == "EU"]
    assert len(eu) == 2 and eu["series_id"].iloc[0] == "GGXWDG_NGDP"
    se = df[df.country == "SE"]
    assert len(se) == 1                                        # 1979 < start_year, None dropped
    url = http_client.get.call_args[0][0]
    assert url.endswith("/GGXWDG_NGDP")


def test_user_agent_set_on_real_session(tmp_path):
    s = requests.Session()
    ImfDataMapperSource(client=s, cache_dir=tmp_path, user_agent="ua-test/1")
    assert s.headers["User-Agent"] == "ua-test/1"


def test_403_fails_fast_with_hint(http_client, tmp_path):
    http_client.get.return_value = _resp("denied", status_code=403)
    src = ImfDataMapperSource(client=http_client, cache_dir=tmp_path)
    with pytest.raises(ValueError, match="403"):
        src.fetch(ImfSpec("x", "X"), _basket(), use_cache=False)
    assert http_client.get.call_count == 1


def test_malformed_and_missing_block_return_empty(http_client, tmp_path):
    src = ImfDataMapperSource(client=http_client, cache_dir=tmp_path)
    http_client.get.return_value = _resp("not json")
    assert src.fetch(ImfSpec("x", "X"), _basket(), use_cache=False).empty
    http_client.get.return_value = _resp(json.dumps({"values": {"OTHER": {}}}))
    assert src.fetch(ImfSpec("x", "X"), _basket(), use_cache=False).empty


def test_cache_hit(http_client, tmp_path):
    http_client.get.return_value = _resp(SAMPLE)
    src = ImfDataMapperSource(client=http_client, cache_dir=tmp_path)
    spec = ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP")
    src.fetch(spec, _basket(), use_cache=True, today=date(2026, 1, 1))
    src.fetch(spec, _basket(), use_cache=True, today=date(2026, 1, 1))
    assert http_client.get.call_count == 1


def test_derive_interest_burden_aligns_on_key():
    def f(ind, rows):
        return pd.DataFrame([{"country": c, "indicator": ind, "date": date(y, 12, 31), "value": v,
                              "source": s, "series_id": "X"} for c, y, v, s in rows])
    primary = f("primary_balance_pct_gdp", [("US", 2025, -3.0, SOURCE_HISTORY), ("US", 2026, -2.5, SOURCE_FORECAST),
                                            ("SE", 2025, 0.5, SOURCE_HISTORY)])
    overall = f("fiscal_balance_pct_gdp", [("US", 2025, -6.5, SOURCE_HISTORY), ("US", 2026, -6.0, SOURCE_FORECAST),
                                           ("SE", 2024, 0.0, SOURCE_HISTORY)])
    out = derive_interest_burden(primary, overall)
    assert set(out["indicator"]) == {"interest_burden_pct_gdp"}
    by = out.set_index(["country", "date"])["value"]
    assert by[("US", date(2025, 12, 31))] == pytest.approx(3.5)
    assert by[("US", date(2026, 12, 31))] == pytest.approx(3.5)
    assert ("SE", date(2025, 12, 31)) not in by.index          # no matching overall row
    assert set(out["source"]) == {SOURCE_HISTORY, SOURCE_FORECAST}
    assert derive_interest_burden(primary.iloc[0:0], overall).empty


def test_bundle_covers_registry_imf_needs():
    stored = {s.indicator for s in IMF_FUNDAMENTALS}
    assert {"real_gdp_growth", "gov_debt_pct_gdp", "fiscal_balance_pct_gdp",
            "primary_balance_pct_gdp", "current_account_pct_gdp"} == stored
    fwd = next(s for s in FUNDAMENTALS if s.name == "gdp_growth_fwd5")
    assert fwd.forward and fwd.base_indicator == "real_gdp_growth"


# ─── forward panel + snapshot integration ──────────────────────────────────


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "t.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed_growth(session, iso2, years_values, source):
    for y, v in years_values:
        session.add(Observation(country=iso2, indicator="real_gdp_growth", date=date(y, 12, 31),
                                value=v, source=source, series_id="NGDP_RPCH"))


def test_forward_panel_means_next_five_forecast_years(session_factory):
    with session_factory() as s:
        _seed_growth(s, "US", [(2024, 2.8), (2025, 1.9)], SOURCE_HISTORY)
        _seed_growth(s, "US", [(2026, 2.0), (2027, 2.1), (2028, 2.2), (2029, 2.3), (2030, 2.4), (2031, 9.9)],
                     SOURCE_FORECAST)
        _seed_growth(s, "IN", [(2026, 6.0), (2027, 6.5)], SOURCE_FORECAST)
        s.commit()
    with session_factory() as s:
        fwd = load_forward_panel(s, FUNDAMENTALS, [get_country("US"), get_country("IN"), get_country("SE")],
                                 as_of=date(2026, 8, 24))
    f = fwd.set_index("country")
    assert f.loc["US", "indicator"] == "gdp_growth_fwd5"
    assert f.loc["US", "value"] == pytest.approx((2.0 + 2.1 + 2.2 + 2.3 + 2.4) / 5)   # 2031 excluded
    assert f.loc["US", "n_years"] == 5 and f.loc["US", "date"] == date(2030, 12, 31)
    assert f.loc["US", "source"] == SOURCE_FORECAST
    assert f.loc["IN", "n_years"] == 2
    assert "SE" not in f.index


def test_snapshot_forward_cell_and_history_alias(session_factory):
    with session_factory() as s:
        _seed_growth(s, "US", [(2024, 2.8), (2025, 1.9)], SOURCE_HISTORY)
        _seed_growth(s, "US", [(2026, 2.0), (2027, 2.2)], SOURCE_FORECAST)
        _seed_growth(s, "IN", [(2025, 6.5)], SOURCE_HISTORY)
        _seed_growth(s, "IN", [(2026, 6.0), (2027, 6.4)], SOURCE_FORECAST)
        s.commit()
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 8, 24))
    us = snap["countries"]["US"]["indicators"]["gdp_growth_fwd5"]
    assert us["value"] == pytest.approx(2.1)
    assert us["is_forecast"] is True and us["forecast_horizon_years"] == 5
    assert us["source"] == SOURCE_FORECAST and us["date"] == "2027-12-31"
    assert us["trend"] is None                                 # no backward lag for a forecast
    assert us["pct"] == 0.0 and snap["countries"]["IN"]["indicators"]["gdp_growth_fwd5"]["pct"] == 100.0
    hist = snap["countries"]["US"]["history"]["gdp_growth_fwd5"]
    assert [h["year"] for h in hist] == [2024, 2025, 2026, 2027]
    assert [h["is_forecast"] for h in hist] == [False, False, True, True]


def test_history_keeps_projection_rows_for_history_sourced_specs(session_factory):
    """gov_debt lists only IMF_WEO (+ BIS_TC) — its IMF_WEO_FCST rows must still ship."""
    spec = next(s for s in FUNDAMENTALS if s.name == "gov_debt_pct_gdp")
    with session_factory() as s:
        for y, v, src in ((2024, 122.3, SOURCE_HISTORY), (2025, 123.9, SOURCE_HISTORY),
                          (2026, 125.8, SOURCE_FORECAST), (2031, 142.1, SOURCE_FORECAST)):
            s.add(Observation(country="US", indicator="gov_debt_pct_gdp", date=date(y, 12, 31),
                              value=v, source=src, series_id="GGXWDG_NGDP"))
        s.add(Observation(country="US", indicator="gov_debt_pct_gdp", date=date(2024, 10, 1),
                          value=118.0, source="BIS_TC", series_id="Q.US.G"))     # less preferred family
        s.commit()
    with session_factory() as s:
        h = load_history(s, [spec], [get_country("US")])
        latest = build_snapshot(s, as_of=date(2026, 8, 24), include_history=False)
    assert list(h["year"]) == [2024, 2025, 2026, 2031]
    assert list(h["is_forecast"]) == [False, False, True, True]
    assert (h["value"] != 118.0).all()                                   # BIS family dropped
    cell = latest["countries"]["US"]["indicators"]["gov_debt_pct_gdp"]
    assert cell["value"] == 123.9 and cell["is_forecast"] is False       # latest ≤ as_of is history


def test_history_alias_only_for_forward_specs(session_factory):
    spec = IndicatorSpec("gdp_growth_fwd5", "production", "x", "", True, "B",
                         ("IMF_WEO_FCST", "IMF_WEO"), "x" * 30, forward=True, base_indicator="real_gdp_growth")
    with session_factory() as s:
        _seed_growth(s, "US", [(2025, 1.9)], SOURCE_HISTORY)
        s.commit()
    with session_factory() as s:
        h = load_history(s, [spec], [get_country("US")])
    assert list(h["indicator"]) == ["gdp_growth_fwd5"]
