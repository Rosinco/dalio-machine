"""Shared fixtures.

`synthetic_snapshot_dict` is the executable contract between the data layer
(`build_snapshot`) and the presentation layer (`load_snapshot`): it is produced
by the real builder on a seeded temp DB, then decorated with one fake pressure
chain so the loader's chain path is exercised before slice 21 exists.

Seeded indicators (of the 15): gdp_pc_ppp, old_age_dependency (6 players incl.
EU, two years); rd_pct_gdp, military_share_world, rule_of_law (+ _se)
(5 individual players, latest year only). Category coverage that results:
real_stuff 1/2 · production 2/3 · exchange 0/3 · promises 0/4 · enforcer 2/3.
Bilateral goods trade (slice 24, USD bn, 2025) for US · CN · DE · EU incl.
world totals, so the snapshot carries a real ``trade`` block.
"""
from datetime import date
from pathlib import Path

import pytest

from dalio.countries import get_country
from dalio.scoring.fundamentals import build_snapshot, write_snapshot
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

SYNTHETIC_PLAYERS = ("US", "SE", "CN", "IN", "DE", "EU")
SYNTHETIC_FILLED = 6 + 6 + 5 + 5 + 5   # gdp, dep, rd, mil, rol
# (reporter, partner) → (exports_to, imports_from) in USD bn, 2025; WLD = world total.
SYNTHETIC_TRADE = {
    ("US", "CN"): (143.5, 438.9), ("US", "DE"): (75.6, 160.0), ("US", "EU"): (376.8, 600.0),
    ("US", "WLD"): (2185.2, 3300.0),
    ("CN", "US"): (525.0, 163.0), ("CN", "DE"): (110.0, 95.0), ("CN", "EU"): (440.0, 250.0),
    ("CN", "WLD"): (3580.0, 2590.0),
    ("DE", "US"): (175.0, 95.0), ("DE", "CN"): (97.0, 160.0), ("DE", "EU"): (631.0, 600.0),
    ("DE", "WLD"): (1668.0, 1400.0),
    ("EU", "US"): (523.0, 400.0), ("EU", "CN"): (207.0, 550.0), ("EU", "WLD"): (5814.0, 5600.0),
}
SYNTHETIC_TRADE_ROWS = sum(1 for k in SYNTHETIC_TRADE if k[1] != "WLD")   # 11 partner rows


def seed_observation(session, country, indicator, year_values, source="WORLD_BANK"):
    for year, v in year_values:
        session.add(Observation(
            country=country, indicator=indicator, date=date(year, 12, 31),
            value=v, source=source, series_id="X",
        ))


def seed_synthetic(session) -> None:
    gdp = {"US": 80000, "SE": 60000, "CN": 24000, "IN": 10000, "DE": 65000, "EU": 58000}
    dep = {"US": 27, "SE": 33, "CN": 21, "IN": 10, "DE": 36, "EU": 34}
    mil = {"US": 40.0, "SE": 0.4, "CN": 12.0, "IN": 3.4, "DE": 2.6}
    rd = {"US": 3.5, "SE": 3.4, "CN": 2.4, "IN": 0.7, "DE": 3.1}
    rol = {"US": 1.4, "SE": 1.7, "CN": -0.5, "IN": 0.1, "DE": 1.6}
    for iso2, v in gdp.items():
        seed_observation(session, iso2, "gdp_pc_ppp", [(2019, v * 0.9), (2024, v)])
    for iso2, v in dep.items():
        seed_observation(session, iso2, "old_age_dependency", [(2019, v - 2), (2025, v)])
    for iso2, v in mil.items():
        seed_observation(session, iso2, "military_share_world", [(2024, v)])
    for iso2, v in rd.items():
        seed_observation(session, iso2, "rd_pct_gdp", [(2023, v)])
    for iso2, v in rol.items():
        seed_observation(session, iso2, "rule_of_law", [(2023, v)], source="WORLD_BANK_WGI")
        seed_observation(session, iso2, "rule_of_law_se", [(2023, 0.15)], source="WORLD_BANK_WGI")
    for (rep, partner), (x, m) in SYNTHETIC_TRADE.items():
        seed_observation(session, rep, f"exports_to_{partner}", [(2025, x)], source="IMF_IMTS")
        seed_observation(session, rep, f"imports_from_{partner}", [(2025, m)], source="IMF_IMTS")


@pytest.fixture
def synthetic_snapshot_dict(tmp_path) -> dict:
    engine = make_engine(tmp_path / "synthetic.db")
    init_db(engine)
    sf = make_session_factory(engine)
    with sf() as s:
        seed_synthetic(s)
        s.commit()
    with sf() as s:
        snap = build_snapshot(
            s, as_of=date(2026, 8, 24),
            countries=[get_country(c) for c in SYNTHETIC_PLAYERS],
            population=[c for c in SYNTHETIC_PLAYERS if c != "EU"],
        )
    # One fake fired rule on the US so the chain path is exercised (slice 21 will
    # produce these for real).
    snap["countries"]["US"]["pressures"] = [{
        "rule_id": "fiscal_dominance", "triggered": True, "severity": 0.6,
        "constraint": "Debt > 90 % GDP with deficit > 3 %",
        "forced_options": ["monetize", "financial repression", "slow austerity"],
        "spillovers": [{"target": "CN", "text": "real return on USD reserves falls"},
                       {"target": "SA", "text": "peg imports US inflation"}],
        "confidence": 0.8, "inputs": {"gov_debt_pct_gdp": 122.0, "fiscal_balance_pct_gdp": -6.0},
        "uncertainty": "C",
    }]
    return snap


@pytest.fixture
def synthetic_snapshot_dir(synthetic_snapshot_dict, tmp_path) -> Path:
    d = tmp_path / "snapshots"
    write_snapshot(synthetic_snapshot_dict, d / "fundamentals_latest.json")
    return d
