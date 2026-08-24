"""Shared fixtures.

`synthetic_snapshot_dict` is the executable contract between the data layer
(`build_snapshot`) and the presentation layer (`load_snapshot`): it is produced
by the real builder on a seeded temp DB, then decorated with one fake pressure
chain so the loader's chain path is exercised before slice 21 exists.
"""
from datetime import date
from pathlib import Path

import pytest

from dalio.countries import get_country
from dalio.scoring.fundamentals import build_snapshot, write_snapshot
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

SYNTHETIC_PLAYERS = ("US", "SE", "CN", "IN", "DE", "EU")


def _seed(session, country, indicator, year_values, source="WORLD_BANK"):
    for year, v in year_values:
        session.add(Observation(
            country=country, indicator=indicator, date=date(year, 12, 31),
            value=v, source=source, series_id="X",
        ))


@pytest.fixture
def synthetic_snapshot_dict(tmp_path) -> dict:
    engine = make_engine(tmp_path / "synthetic.db")
    init_db(engine)
    sf = make_session_factory(engine)
    gdp = {"US": 80000, "SE": 60000, "CN": 24000, "IN": 10000, "DE": 65000, "EU": 58000}
    dep = {"US": 27, "SE": 33, "CN": 21, "IN": 10, "DE": 36, "EU": 34}
    mil = {"US": 3.3, "SE": 2.0, "CN": 1.7, "IN": 2.4, "DE": 1.5}
    with sf() as s:
        for iso2, v in gdp.items():
            _seed(s, iso2, "gdp_pc_ppp", [(2019, v * 0.9), (2024, v)])
        for iso2, v in dep.items():
            _seed(s, iso2, "old_age_dependency", [(2019, v - 2), (2025, v)])
        for iso2, v in mil.items():
            _seed(s, iso2, "military_pct_gdp", [(2024, v)])
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
