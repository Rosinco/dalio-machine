"""Bilateral trade shares (slice 24): loading, shares, both directions, snapshot block."""
from datetime import date

import pandas as pd
import pytest

from dalio.countries import get_country
from dalio.data_sources.imf_imts import SOURCE_IMTS
from dalio.scoring.trade import (
    TRADE_COLUMNS,
    exposed_players,
    exposure_to,
    load_trade,
    partner_indicator_names,
    top_partners,
    trade_block,
    trade_shares,
    world_totals,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "t.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed(session, rows):
    for c, ind, y, v in rows:
        session.add(Observation(country=c, indicator=ind, date=date(y, 12, 31), value=v,
                                source=SOURCE_IMTS, series_id="X"))


BASKET = [get_country(c) for c in ("US", "CA", "MX", "EU")]


def _seed_default(session):
    _seed(session, [
        ("US", "exports_to_CA", 2024, 349.4), ("US", "exports_to_CA", 2025, 336.5),
        ("US", "exports_to_MX", 2025, 338.0), ("US", "exports_to_EU", 2025, 376.8),
        ("US", "exports_to_WLD", 2024, 2065.4), ("US", "exports_to_WLD", 2025, 2185.2),
        ("US", "imports_from_CA", 2025, 420.0), ("US", "imports_from_WLD", 2025, 3300.0),
        ("CA", "exports_to_US", 2024, 400.0), ("CA", "exports_to_US", 2025, 410.0),  # 2025 has no WLD total
        ("CA", "exports_to_WLD", 2024, 520.0),
        ("MX", "exports_to_US", 2025, 500.0), ("MX", "exports_to_WLD", 2025, 600.0),
        ("MX", "imports_from_US", 2025, 250.0),                                        # no import total → NaN share
        ("EU", "exports_to_US", 2025, 523.3), ("EU", "exports_to_WLD", 2025, 5814.0),
    ])


def test_partner_indicator_names_cover_both_flows_and_world():
    names = partner_indicator_names(["US", "CA"])
    assert names == ["exports_to_US", "exports_to_CA", "exports_to_WLD",
                     "imports_from_US", "imports_from_CA", "imports_from_WLD"]


def test_load_trade_latest_year_with_world_total(session_factory):
    with session_factory() as s:
        _seed_default(s)
        s.commit()
    with session_factory() as s:
        raw = load_trade(s, BASKET, as_of=date(2026, 8, 24))
        older = load_trade(s, BASKET, as_of=date(2024, 12, 31))
    by = raw.set_index(["iso2", "partner"])
    assert by.loc[("US", "CA"), "year"] == 2025 and by.loc[("US", "CA"), "x_usd"] == 336.5
    assert by.loc[("US", "CA"), "m_usd"] == 420.0
    assert by.loc[("CA", "US"), "year"] == 2024 and by.loc[("CA", "US"), "x_usd"] == 400.0   # 2025 lacks a total
    assert pd.isna(by.loc[("MX", "US"), "m_usd"]) is False and pd.isna(by.loc[("MX", "WLD"), "m_usd"])
    assert set(older[older["iso2"] == "US"]["year"]) == {2024}
    assert "EU" in set(raw["iso2"])


def test_load_trade_empty(session_factory):
    with session_factory() as s:
        assert load_trade(s, BASKET, as_of=date(2026, 8, 24)).empty


def test_trade_shares_hand_calc(session_factory):
    with session_factory() as s:
        _seed_default(s)
        s.commit()
    with session_factory() as s:
        shares = trade_shares(load_trade(s, BASKET, as_of=date(2026, 8, 24)))
    assert list(shares.columns) == TRADE_COLUMNS
    assert "WLD" not in set(shares["partner"])
    by = shares.set_index(["iso2", "partner"])
    assert by.loc[("US", "CA"), "x_share"] == pytest.approx(336.5 / 2185.2 * 100)
    assert by.loc[("US", "CA"), "m_share"] == pytest.approx(420.0 / 3300.0 * 100)
    assert by.loc[("US", "EU"), "x_share"] == pytest.approx(376.8 / 2185.2 * 100)
    assert pd.isna(by.loc[("US", "EU"), "m_share"])                       # no import row
    assert by.loc[("MX", "US"), "x_share"] == pytest.approx(500 / 600 * 100)
    assert pd.isna(by.loc[("MX", "US"), "m_share"])                       # no import total
    us = shares[shares["iso2"] == "US"]
    assert list(us["partner"])[:2] == ["EU", "MX"]                        # sorted by x_share desc
    wt = world_totals(load_trade(s, BASKET, as_of=date(2026, 8, 24)) if False else pd.DataFrame(
        [{"iso2": "US", "partner": "WLD", "year": 2025, "x_usd": 1.0, "m_usd": 2.0}]))
    assert wt.loc["US", "m_usd"] == 2.0


def _shares():
    return pd.DataFrame([
        {"iso2": "US", "partner": "CA", "year": 2025, "x_share": 16.9, "m_share": 12.7, "x_usd": 1, "m_usd": 1},
        {"iso2": "US", "partner": "MX", "year": 2025, "x_share": 15.5, "m_share": 15.0, "x_usd": 1, "m_usd": 1},
        {"iso2": "US", "partner": "CN", "year": 2025, "x_share": 6.9, "m_share": 13.8, "x_usd": 1, "m_usd": 1},
        {"iso2": "US", "partner": "SE", "year": 2025, "x_share": 0.4, "m_share": 0.5, "x_usd": 1, "m_usd": 1},
        {"iso2": "US", "partner": "EU", "year": 2025, "x_share": 16.3, "m_share": None, "x_usd": 1, "m_usd": None},
        {"iso2": "CA", "partner": "US", "year": 2024, "x_share": 76.9, "m_share": 49.0, "x_usd": 1, "m_usd": 1},
        {"iso2": "MX", "partner": "US", "year": 2025, "x_share": 83.0, "m_share": None, "x_usd": 1, "m_usd": None},
        {"iso2": "CN", "partner": "US", "year": 2025, "x_share": 14.7, "m_share": 6.0, "x_usd": 1, "m_usd": 1},
        {"iso2": "SE", "partner": "US", "year": 2025, "x_share": 9.0, "m_share": 3.0, "x_usd": 1, "m_usd": 1},
        {"iso2": "SA", "partner": "US", "year": 2025, "x_share": 1.9, "m_share": 12.0, "x_usd": 1, "m_usd": 1},
    ])


def test_top_partners_by_flow_total_and_min_share():
    sh = _shares()
    assert top_partners(sh, "US", by="x_share", n=3) == [("CA", 16.9), ("EU", 16.3), ("MX", 15.5)]
    assert top_partners(sh, "US", by="m_share", n=2) == [("MX", 15.0), ("CN", 13.8)]
    tot = top_partners(sh, "US", by="total", n=5)
    assert tot[0] == ("MX", pytest.approx(30.5)) and ("EU", 16.3) in tot     # NaN m_share counts as 0
    assert ("SE", 0.4) not in top_partners(sh, "US", by="x_share", n=10)      # below min_share
    assert top_partners(sh, "US", by="x_share", n=10, min_share=0.0)[-1] == ("SE", 0.4)
    assert top_partners(sh, "ZZ", n=3) == [] and top_partners(pd.DataFrame(columns=TRADE_COLUMNS), "US") == []


def test_exposure_direction():
    sh = _shares()
    assert exposure_to(sh, "CA", "US") == 76.9          # 76.9 % of Canada's exports go to the US
    assert exposure_to(sh, "US", "CA") == 16.9
    assert exposure_to(sh, "US", "JP") is None
    assert exposed_players(sh, "US", n=3) == [("MX", 83.0), ("CA", 76.9), ("CN", 14.7)]
    assert exposed_players(sh, "US", n=10) == [("MX", 83.0), ("CA", 76.9), ("CN", 14.7), ("SE", 9.0)]  # SA < 2 %
    assert exposed_players(sh, "SE") == []
    assert exposed_players(pd.DataFrame(columns=TRADE_COLUMNS), "US") == []


def test_trade_block_json_ready():
    block = trade_block(_shares())
    assert block[4]["m_share"] is None and block[4]["m_usd"] is None
    assert block[0] == {"iso2": "US", "partner": "CA", "year": 2025, "x_share": 16.9, "m_share": 12.7,
                        "x_usd": 1.0, "m_usd": 1.0}
    assert trade_block(pd.DataFrame(columns=TRADE_COLUMNS)) is None
