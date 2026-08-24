"""Pressure-chain rules (slice 21): trigger / no-trigger / borderline / missing inputs / spillovers."""
import numpy as np
import pandas as pd
import pytest

from dalio.countries import COUNTRIES
from dalio.scoring.pressure import (
    LAYER_UNCERTAINTY,
    Panel,
    evaluate,
    rule_credit_bust,
    rule_demographic_squeeze,
    rule_energy_dependence,
    rule_external_financing,
    rule_fiscal_dominance,
    rule_isolation,
    to_dict,
)

COLS = ["gov_debt_pct_gdp", "fiscal_balance_pct_gdp", "interest_burden_pct_gdp", "gdp_growth_fwd5",
        "current_account_pct_gdp", "reserves_months_imports", "debt_service_ratio",
        "old_age_dependency", "energy_net_imports_pct"]


def _panel(rows: dict[str, dict], trend=None, dsr_q90=None) -> Panel:
    df = pd.DataFrame.from_dict(rows, orient="index").reindex(columns=COLS)
    return Panel(values=df.astype(float), trend=trend or {}, dsr_q90=dsr_q90 or {},
                 countries={c.iso2: c for c in COUNTRIES})


# ─── fiscal dominance ──────────────────────────────────────────────────────


def test_fiscal_dominance_reserve_issuer_branch_and_spillovers():
    p = _panel({
        "US": {"gov_debt_pct_gdp": 124, "fiscal_balance_pct_gdp": -6.8, "interest_burden_pct_gdp": 3.6, "gdp_growth_fwd5": 2.0,
               "reserves_months_imports": 2.0},
        "CN": {"reserves_months_imports": 15.0}, "JP": {"reserves_months_imports": 16.0},
        "CH": {"reserves_months_imports": 20.0}, "SA": {"reserves_months_imports": 20.0},
        "IN": {"reserves_months_imports": 10.0}, "KR": {"reserves_months_imports": 8.0},
        "EU": {"reserves_months_imports": 30.0},   # aggregate — must not be a spillover target
    })
    r = rule_fiscal_dominance("US", p)
    assert r.triggered and 0 < r.severity <= 1
    assert "r > g" in r.constraint                      # 3.6/124 = 2.9 % > 2.0 %
    assert "monetize" in r.forced_options[0]
    targets = [s.target for s in r.spillovers]
    assert "EU" not in targets and "US" not in targets
    assert targets[:5] == ["CH", "SA", "JP", "CN", "IN"] or set(targets[:5]) == {"CH", "SA", "JP", "CN", "IN"}
    assert "SA" in targets and any(s.channel == "via FX" for s in r.spillovers if s.target == "SA")
    assert r.confidence == 1.0 * 0.7                    # forecast-derived growth input


def test_fiscal_dominance_union_and_other_branches():
    p = _panel({
        "IT": {"gov_debt_pct_gdp": 135, "fiscal_balance_pct_gdp": -3.5, "interest_burden_pct_gdp": 3.9},
        "BR": {"gov_debt_pct_gdp": 92, "fiscal_balance_pct_gdp": -8.0, "interest_burden_pct_gdp": 6.0},
        "DE": {}, "FR": {}, "EU": {},
    })
    it = rule_fiscal_dominance("IT", p)
    assert it.triggered and "ECB" in it.forced_options[-1]
    assert {"DE", "FR", "EU"} <= {s.target for s in it.spillovers}
    assert it.confidence == 1.0 * 0.9                   # no forecast growth → ×0.9 only
    br = rule_fiscal_dominance("BR", p)
    assert br.triggered and "restructure / default" in br.forced_options
    assert {s.target for s in br.spillovers} == {"domestic banks", "foreign holders"}
    assert br.confidence == pytest.approx(0.9)          # BR is medium quality → no ×0.8


def test_fiscal_dominance_not_triggered_and_missing():
    p = _panel({"SE": {"gov_debt_pct_gdp": 34, "fiscal_balance_pct_gdp": -1.0, "interest_burden_pct_gdp": 0.4},
                "CN": {"gov_debt_pct_gdp": 96, "fiscal_balance_pct_gdp": -2.9, "interest_burden_pct_gdp": 1.0},
                "RU": {"gov_debt_pct_gdp": 20}})
    assert not rule_fiscal_dominance("SE", p).triggered
    cn = rule_fiscal_dominance("CN", p)                 # borderline: deficit −2.9 ≥ −3 and interest ≤ 2.5
    assert not cn.triggered and cn.confidence == pytest.approx(0.9 * 0.8)
    ru = rule_fiscal_dominance("RU", p)
    assert not ru.triggered and ru.constraint == "insufficient inputs"


# ─── external financing ────────────────────────────────────────────────────


def test_external_financing():
    p = _panel({"TR": {"current_account_pct_gdp": -4.5, "reserves_months_imports": 3.0},
                "MX": {"current_account_pct_gdp": -1.0, "reserves_months_imports": 5.0},
                "US": {"current_account_pct_gdp": -4.0, "reserves_months_imports": 1.0},
                "ES": {"current_account_pct_gdp": -4.0, "reserves_months_imports": 1.0},
                "ID": {"current_account_pct_gdp": -4.0}})
    tr = rule_external_financing("TR", p)
    assert tr.triggered and "IMF programme" in tr.forced_options
    assert {s.target for s in tr.spillovers} == {"foreign holders", "trade partners"}
    assert not rule_external_financing("MX", p).triggered
    assert not rule_external_financing("US", p).triggered      # reserve issuer exempt
    assert not rule_external_financing("ES", p).triggered      # currency union exempt
    idn = rule_external_financing("ID", p)
    assert not idn.triggered and idn.constraint == "insufficient inputs" and idn.confidence == pytest.approx(0.5)


# ─── credit bust ───────────────────────────────────────────────────────────


def test_credit_bust_uses_calibrated_threshold_with_fallback():
    p = _panel({"SE": {"debt_service_ratio": 23.3}, "US": {"debt_service_ratio": 14.1}, "KR": {}},
               dsr_q90={"SE": 24.7})
    assert not rule_credit_bust("SE", p).triggered            # 23.3 ≤ calibrated 24.7
    p2 = _panel({"SE": {"debt_service_ratio": 23.3}})          # fallback 18 → fires
    se = rule_credit_bust("SE", p2)
    assert se.triggered and se.inputs["dsr_q90"] == 18.0
    assert se.spillovers[0].target == "SE" and "fiscal dominance" in se.spillovers[0].text
    kr = rule_credit_bust("KR", p)
    assert not kr.triggered and "BIS" in kr.constraint and kr.confidence == 0.0


# ─── demographic squeeze ───────────────────────────────────────────────────


def test_demographic_squeeze_needs_trend():
    rows = {"JP": {"old_age_dependency": 51, "gov_debt_pct_gdp": 230},
            "IT": {"old_age_dependency": 40, "gov_debt_pct_gdp": 135},
            "IN": {"old_age_dependency": 11, "gov_debt_pct_gdp": 82}}
    p = _panel(rows, trend={("JP", "old_age_dependency"): "worsening", ("IT", "old_age_dependency"): "flat"})
    jp = rule_demographic_squeeze("JP", p)
    assert jp.triggered and jp.severity == 1.0
    assert all(s.target == "JP" for s in jp.spillovers)
    assert not rule_demographic_squeeze("IT", p).triggered   # not worsening
    assert not rule_demographic_squeeze("IN", p).triggered


# ─── energy dependence ─────────────────────────────────────────────────────


def test_energy_dependence_names_exporters():
    p = _panel({"JP": {"energy_net_imports_pct": 88}, "KR": {"energy_net_imports_pct": 82},
                "SA": {"energy_net_imports_pct": -180}, "RU": {"energy_net_imports_pct": -85},
                "AU": {"energy_net_imports_pct": -170}, "US": {"energy_net_imports_pct": -5},
                "EU": {"energy_net_imports_pct": -60}})
    jp = rule_energy_dependence("JP", p)
    assert jp.triggered and jp.severity == pytest.approx(0.76)
    assert {s.target for s in jp.spillovers} == {"SA", "RU", "AU"}   # EU aggregate excluded, US not < −25
    assert not rule_energy_dependence("SA", p).triggered
    assert not rule_energy_dependence("US", p).triggered


# ─── isolation ─────────────────────────────────────────────────────────────


def test_isolation_sanctioned_and_low_stability():
    p = _panel({"RU": {}, "US": {}, "EU": {}, "TR": {}, "SE": {}})
    ru = rule_isolation("RU", p)
    assert ru.triggered and ru.severity == 1.0 and ru.constraint == "Under sanctions"
    assert {s.target for s in ru.spillovers} >= {"US", "EU", "trade partners"}
    assert ru.confidence == pytest.approx(0.8)                 # opaque statistics
    tr = rule_isolation("TR", p, pv_pct=10.0)
    assert tr.triggered and tr.severity == 0.5
    assert not rule_isolation("SE", p, pv_pct=85.0).triggered
    none = rule_isolation("SE", p, pv_pct=None)
    assert not none.triggered and none.confidence == 0.0


# ─── evaluate + serialisation ──────────────────────────────────────────────


def test_evaluate_returns_all_six_and_serialises():
    p = _panel({"US": {"gov_debt_pct_gdp": 124, "fiscal_balance_pct_gdp": -6.8, "interest_burden_pct_gdp": 3.6,
                       "energy_net_imports_pct": -5, "old_age_dependency": 28, "current_account_pct_gdp": -4,
                       "reserves_months_imports": 1.5, "debt_service_ratio": 14.1}})
    out = evaluate("US", p, pv_pct=50.0)
    assert [o.rule_id for o in out] == ["fiscal_dominance", "external_financing", "credit_bust",
                                        "demographic_squeeze", "energy_dependence", "isolation"]
    assert [o.triggered for o in out] == [True, False, False, False, False, False]
    d = to_dict(out[0])
    assert d["uncertainty"] == LAYER_UNCERTAINTY and d["triggered"] is True
    assert isinstance(d["spillovers"], list) and set(d) >= {"rule_id", "severity", "forced_options", "inputs"}
    assert all(v is None or isinstance(v, float) for v in d["inputs"].values())
    assert not any(isinstance(v, np.floating) for v in d["inputs"].values())
