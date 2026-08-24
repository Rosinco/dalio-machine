"""Derived World Bank series (slice 19): world shares and euro-area member means."""
from datetime import date

import pandas as pd
import pytest

from dalio.data_sources.worldbank import (
    WB_FUNDAMENTALS,
    WB_MEMBER_MEAN_INDICATORS,
    WB_WORLD_SHARES,
    derive_member_mean,
    derive_world_share,
)
from dalio.scoring.fundamentals import FUNDAMENTALS, SE_INDICATORS


def _rows(indicator, data, source="WORLD_BANK", sid="X"):
    return pd.DataFrame([
        {"country": c, "indicator": indicator, "date": date(y, 12, 31), "value": v,
         "source": source, "series_id": sid}
        for c, y, v in data
    ])


def test_world_share():
    df = _rows("exports_usd", [("US", 2023, 3000.0), ("SE", 2023, 300.0), ("WLD", 2023, 30000.0),
                               ("US", 2022, 2800.0), ("WLD", 2022, 0.0), ("SE", 2021, 250.0)])
    out = derive_world_share(df, "exports_share_world")
    assert set(out["indicator"]) == {"exports_share_world"}
    assert "WLD" not in set(out["country"])
    by = out.set_index(["country", "date"])["value"]
    assert by[("US", date(2023, 12, 31))] == pytest.approx(10.0)
    assert by[("SE", date(2023, 12, 31))] == pytest.approx(1.0)
    assert ("US", date(2022, 12, 31)) not in by.index      # world value 0 → dropped
    assert ("SE", date(2021, 12, 31)) not in by.index      # no world row → dropped
    assert out["series_id"].iloc[0] == "X÷WLD"
    assert derive_world_share(df.iloc[0:0], "x").empty


def test_member_mean_flagged_and_min_members():
    df = _rows("rule_of_law", [("DE", 2023, 1.6), ("FR", 2023, 1.3), ("IT", 2023, 0.3),
                               ("ES", 2023, 0.9), ("NL", 2023, 1.8), ("US", 2023, 1.4),
                               ("DE", 2022, 1.5), ("FR", 2022, 1.2)],
               source="WORLD_BANK_WGI", sid="GOV_WGI_RL.EST")
    out = derive_member_mean(df, ["DE", "FR", "IT", "ES", "NL"], "EU")
    assert list(out["country"].unique()) == ["EU"]
    assert len(out) == 1                                   # 2022 has only 2 members → dropped
    assert out["value"].iloc[0] == pytest.approx((1.6 + 1.3 + 0.3 + 0.9 + 1.8) / 5)
    assert out["series_id"].iloc[0] == "GOV_WGI_RL.EST:member-mean"
    assert out["source"].iloc[0] == "WORLD_BANK_WGI"
    assert derive_member_mean(df, ["ZZ"], "EU").empty


def test_registry_is_the_sixteen_of_adr_0001_and_0002():
    names = [s.name for s in FUNDAMENTALS]
    assert len(names) == 16 and len(set(names)) == 16
    by_cat = {}
    for s in FUNDAMENTALS:
        by_cat.setdefault(s.category, []).append(s.name)
    assert {k: len(v) for k, v in by_cat.items()} == {
        "real_stuff": 2, "production": 4, "exchange": 3, "promises": 4, "enforcer": 3,
    }
    assert SE_INDICATORS == ("rule_of_law_se", "political_stability_se")
    # every WB-sourced scored indicator is either fetched raw or derived
    fetched = {s.indicator for s in WB_FUNDAMENTALS}
    derived = {out for _, out in WB_WORLD_SHARES}
    wb_scored = {s.name for s in FUNDAMENTALS if "WORLD_BANK" in s.preferred_sources or "WORLD_BANK_WGI" in s.preferred_sources}
    assert wb_scored <= fetched | derived
    assert set(WB_MEMBER_MEAN_INDICATORS) <= fetched
    assert all(s.source_id == 3 for s in WB_FUNDAMENTALS if s.wb_code.startswith("GOV_WGI"))
