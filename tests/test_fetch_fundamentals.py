"""Pipeline wiring tests (slice 18): mocked adapter, temp DB via DALIO_DB_PATH."""
from datetime import date

import pandas as pd
import pytest
from sqlalchemy import select

from dalio.countries import get_country
from dalio.data_sources.worldbank import WbIndicatorSpec
from dalio.pipelines import fetch_fundamentals, score_fundamentals
from dalio.storage.db import Observation, make_engine


class _FakeWb:
    def __init__(self, frames):
        self._frames = frames
        self.calls = []

    def fetch(self, spec, countries, use_cache=True, today=None):
        self.calls.append((spec.indicator, tuple(c.iso2 for c in countries), use_cache))
        out = self._frames.get(spec.indicator)
        if isinstance(out, Exception):
            raise out
        return out


def _frame(indicator, rows):
    return pd.DataFrame([
        {"country": c, "indicator": indicator, "date": date(y, 12, 31), "value": v,
         "source": "WORLD_BANK", "series_id": "X"}
        for c, y, v in rows
    ])


@pytest.fixture
def db_env(tmp_path, monkeypatch):
    db = tmp_path / "pipe.db"
    monkeypatch.setenv("DALIO_DB_PATH", str(db))
    monkeypatch.setenv("FUNDAMENTALS_DIR", str(tmp_path / "snapshots"))
    return db


def test_run_pipeline_upserts_and_collects_errors(db_env):
    specs = (
        WbIndicatorSpec("gdp_pc_ppp", "NY.GDP.PCAP.PP.KD"),
        WbIndicatorSpec("old_age_dependency", "SP.POP.DPND.OL"),
    )
    fake = _FakeWb({
        "gdp_pc_ppp": _frame("gdp_pc_ppp", [("US", 2024, 80000.0), ("KR", 2024, 50000.0)]),
        "old_age_dependency": RuntimeError("boom"),
    })
    from dalio.data_sources.imf_datamapper import ImfSpec
    summary = fetch_fundamentals.run_pipeline(
        ("wb", "imf", "bis", "nope"), countries=[get_country("US"), get_country("KR")],
        use_cache=False, wb_source=fake, wb_specs=specs,
        imf_source=_FakeImf({"gov_debt_pct_gdp": RuntimeError("akamai 403")}),
        imf_specs=(ImfSpec("gov_debt_pct_gdp", "GGXWDG_NGDP"),),
    )
    assert summary["wb/gdp_pc_ppp"]["rows"] == 2
    assert summary["wb/gdp_pc_ppp"]["inserted"] == 2
    assert summary["wb/gdp_pc_ppp"]["countries"] == 2
    assert "error" in summary["wb/old_age_dependency"]
    assert "akamai" in summary["imf/gov_debt_pct_gdp"]["error"]
    assert "not implemented" in summary["bis/*"]["error"]
    assert "unknown source" in summary["nope/*"]["error"]
    assert fake.calls[0] == ("gdp_pc_ppp", ("US", "KR"), False)

    with make_engine(db_env).connect() as conn:
        rows = conn.execute(select(Observation.country, Observation.value)).all()
    assert sorted(rows) == [("KR", 50000.0), ("US", 80000.0)]


def test_run_pipeline_derives_world_share_and_member_mean(db_env):
    specs = (
        WbIndicatorSpec("exports_usd", "NE.EXP.GNFS.CD", include_world=True),
        WbIndicatorSpec("rule_of_law", "GOV_WGI_RL.EST", source_id=3),
    )
    frames = {
        "exports_usd": _frame("exports_usd", [("US", 2023, 3000.0), ("DE", 2023, 1800.0), ("WLD", 2023, 30000.0)]),
        "rule_of_law": pd.DataFrame([
            {"country": c, "indicator": "rule_of_law", "date": date(2023, 12, 31), "value": v,
             "source": "WORLD_BANK_WGI", "series_id": "GOV_WGI_RL.EST"}
            for c, v in (("DE", 1.6), ("FR", 1.3), ("IT", 0.3), ("ES", 0.9), ("NL", 1.8), ("US", 1.4))
        ]),
    }
    basket = [get_country(c) for c in ("US", "DE", "FR", "IT", "ES", "NL", "EU")]
    summary = fetch_fundamentals.run_pipeline(("wb",), countries=basket, use_cache=False,
                                              wb_source=_FakeWb(frames), wb_specs=specs)
    assert summary["wb/exports_share_world"]["rows"] == 2
    assert summary["wb/rule_of_law:EU"]["rows"] == 1
    with make_engine(db_env).connect() as conn:
        rows = conn.execute(select(Observation.country, Observation.indicator, Observation.value,
                                   Observation.series_id)).all()
    by = {(r[0], r[1]): (r[2], r[3]) for r in rows}
    assert by[("US", "exports_share_world")][0] == 10.0
    assert by[("US", "exports_share_world")][1].endswith("÷WLD")
    assert ("WLD", "exports_share_world") not in by
    assert by[("EU", "rule_of_law")][0] == pytest.approx((1.6 + 1.3 + 0.3 + 0.9 + 1.8) / 5)
    assert by[("EU", "rule_of_law")][1].endswith(":member-mean")


class _FakeImf:
    def __init__(self, frames):
        self._frames = frames

    def fetch(self, spec, countries, use_cache=True, today=None):
        out = self._frames.get(spec.indicator)
        if isinstance(out, Exception):
            raise out
        return out if out is not None else pd.DataFrame(
            columns=["country", "indicator", "date", "value", "source", "series_id"])


def _imf_frame(indicator, rows):
    return pd.DataFrame([
        {"country": c, "indicator": indicator, "date": date(y, 12, 31), "value": v,
         "source": src, "series_id": "X"}
        for c, y, v, src in rows
    ])


def test_imf_pipeline_replaces_forecasts_and_derives_interest(db_env):
    from dalio.data_sources.imf_datamapper import ImfSpec
    specs = (ImfSpec("fiscal_balance_pct_gdp", "GGXCNL_NGDP"), ImfSpec("primary_balance_pct_gdp", "P"))
    old = _FakeImf({
        "fiscal_balance_pct_gdp": _imf_frame("fiscal_balance_pct_gdp",
                                             [("US", 2025, -6.5, "IMF_WEO"), ("US", 2026, -6.0, "IMF_WEO_FCST"),
                                              ("US", 2027, -5.9, "IMF_WEO_FCST")]),
        "primary_balance_pct_gdp": _imf_frame("primary_balance_pct_gdp",
                                              [("US", 2025, -3.0, "IMF_WEO"), ("US", 2026, -2.5, "IMF_WEO_FCST"),
                                               ("US", 2027, -2.4, "IMF_WEO_FCST")]),
    })
    basket = [get_country("US")]
    s1 = fetch_fundamentals.run_pipeline(("imf",), countries=basket, use_cache=False,
                                         imf_source=old, imf_specs=specs)
    assert s1["imf/fiscal_balance_pct_gdp"]["rows"] == 3
    assert s1["imf/interest_burden_pct_gdp"]["rows"] == 3

    # New vintage: 2027 forecast gone, 2026 revised → old 2027 forecast row must disappear
    new = _FakeImf({
        "fiscal_balance_pct_gdp": _imf_frame("fiscal_balance_pct_gdp",
                                             [("US", 2025, -6.5, "IMF_WEO"), ("US", 2026, -6.2, "IMF_WEO_FCST")]),
        "primary_balance_pct_gdp": _imf_frame("primary_balance_pct_gdp",
                                              [("US", 2025, -3.0, "IMF_WEO"), ("US", 2026, -2.6, "IMF_WEO_FCST")]),
    })
    fetch_fundamentals.run_pipeline(("imf",), countries=basket, use_cache=False, imf_source=new, imf_specs=specs)
    with make_engine(db_env).connect() as conn:
        rows = conn.execute(select(Observation.indicator, Observation.date, Observation.value, Observation.source)).all()
    fiscal = {(r[1].year, r[3]): r[2] for r in rows if r[0] == "fiscal_balance_pct_gdp"}
    assert fiscal == {(2025, "IMF_WEO"): -6.5, (2026, "IMF_WEO_FCST"): -6.2}
    interest = {(r[1].year, r[3]): r[2] for r in rows if r[0] == "interest_burden_pct_gdp"}
    assert interest == {(2025, "IMF_WEO"): 3.5, (2026, "IMF_WEO_FCST"): pytest.approx(3.6)}


def test_score_writes_latest_and_dated_snapshot(db_env, tmp_path):
    fake = _FakeWb({"gdp_pc_ppp": _frame("gdp_pc_ppp", [("US", 2024, 80000.0), ("SE", 2024, 60000.0)])})
    fetch_fundamentals.run_pipeline(
        ("wb",), use_cache=False, wb_source=fake,
        wb_specs=(WbIndicatorSpec("gdp_pc_ppp", "NY.GDP.PCAP.PP.KD"),),
    )
    path, snap = score_fundamentals.run(as_of=date(2026, 8, 24))
    assert path == tmp_path / "snapshots" / "fundamentals_latest.json"
    assert path.exists()
    assert (tmp_path / "snapshots" / "fundamentals_2026-08-24.json").exists()
    assert snap["countries"]["US"]["indicators"]["gdp_pc_ppp"]["pct"] == 100.0
    assert snap["countries"]["SE"]["indicators"]["gdp_pc_ppp"]["pct"] == 0.0
    assert snap["coverage"]["filled"] == 2
