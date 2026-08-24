"""Pressure chains — "what will this country be FORCED to do, and who feels it?"

Judgment encoded as six explicit rules (ADR 0001 §7). Each rule reads the
scored panel (latest values, already oriented as raw numbers — not percentiles)
plus the registry's static flags, and returns a ``Pressure``:

    inputs → trigger / severity → forced option set → spillovers

*Forced* means the option set is branched on the exit routes a country
actually has: a reserve-currency issuer can monetize, a currency-union member
cannot (the ECB decides), everyone else pays in FX. Spillovers are mechanical
templates keyed on OTHER players' flags and values — never free text. With
bilateral trade in the panel (slice 24) the "trade partners" group label
becomes the players most exposed to the country (share of THEIR exports that
go there); without it the group label stays.

Everything here is tier C: thresholds are round numbers from the sovereign-
debt / balance-of-payments literature, not calibrated. Confidence shrinks when
inputs are missing, forecast-derived, or from a low-quality statistical
system.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from dalio.countries import COUNTRIES, Country, DataQuality
from dalio.scoring.trade import exposed_players, import_share

LAYER_UNCERTAINTY = ("C — judgment encoded as rules; thresholds are round numbers from the "
                     "sovereign-debt/BoP literature, not calibrated")
FORECAST_DERIVED: frozenset[str] = frozenset({"gdp_growth_fwd5"})
DSR_FALLBACK_Q90 = 18.0


@dataclass(frozen=True)
class Spillover:
    target: str          # player iso2, or a group label (e.g. "domestic banks")
    text: str
    channel: str = ""    # "via rates" · "via FX" · "via demand" · "via supply" · "via debt"


@dataclass(frozen=True)
class Pressure:
    rule_id: str
    title: str
    triggered: bool
    severity: float                        # 0..1, normalised excess over the threshold
    constraint: str                        # one line: what binds
    forced_options: tuple[str, ...]
    spillovers: tuple[Spillover, ...]
    confidence: float                      # 0..1
    inputs: dict[str, float | None]        # the values the rule looked at
    uncertainty: str = LAYER_UNCERTAINTY


@dataclass(frozen=True)
class Panel:
    """Everything the rules may read. ``values``: index iso2, columns = raw
    indicator values (latest). ``trend``: {(iso2, indicator): direction}.
    ``dsr_q90``: per-country calibrated DSR distress threshold."""
    values: pd.DataFrame
    trend: Mapping[tuple[str, str], str | None] = field(default_factory=dict)
    dsr_q90: Mapping[str, float] = field(default_factory=dict)
    countries: Mapping[str, Country] = field(default_factory=lambda: {c.iso2: c for c in COUNTRIES})
    trade: pd.DataFrame | None = None       # bilateral shares (scoring.trade.trade_shares); None → group labels


# ─── helpers ─────────────────────────────────────────────────────────────────


def _v(panel: Panel, iso2: str, name: str) -> float | None:
    if name not in panel.values.columns or iso2 not in panel.values.index:
        return None
    x = panel.values.at[iso2, name]
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else float(x)


def _excess(value: float, threshold: float, scale: float, higher_is_worse: bool = True) -> float:
    d = (value - threshold) if higher_is_worse else (threshold - value)
    return float(min(1.0, max(0.0, d / scale)))


def _confidence(inputs: Mapping[str, float | None], required: Sequence[str], country: Country) -> float:
    present = sum(1 for k in required if inputs.get(k) is not None)
    conf = present / len(required) if required else 1.0
    if any(k in FORECAST_DERIVED and inputs.get(k) is not None for k in required):
        conf *= 0.7
    if country.data_quality in (DataQuality.LOW, DataQuality.OPAQUE):
        conf *= 0.8
    return round(conf, 3)


def _players(panel: Panel, predicate: Callable[[Country], bool]) -> list[str]:
    return [c.iso2 for c in panel.countries.values() if predicate(c)]


def _top_by(panel: Panel, name: str, n: int, exclude: str, higher: bool = True) -> list[str]:
    if name not in panel.values.columns:
        return []
    s = panel.values[name].dropna().drop(labels=[exclude], errors="ignore")
    s = s[[i for i in s.index if panel.countries.get(i) and panel.countries[i].on_map]]
    return list(s.sort_values(ascending=not higher).head(n).index)


def _trade_exposed(panel: Panel, iso2: str, n: int = 3, min_share: float = 2.0) -> list[tuple[str, float]]:
    """Players whose exports to ``iso2`` are ≥ ``min_share`` % of their own
    exports (top ``n``); empty when the panel carries no trade."""
    if panel.trade is None or panel.trade.empty:
        return []
    return [(t, s) for t, s in exposed_players(panel.trade, iso2, n, min_share) if t in panel.countries]


def _partner_spillovers(panel: Panel, iso2: str, text: str, fallback: str,
                        exclude: Sequence[str] = ()) -> tuple[Spillover, ...]:
    """Named exposed players (share of their exports that go to ``iso2``) or
    the group label when bilateral trade is absent. ``exclude`` drops targets
    the rule has already named through another channel."""
    exposed = [(t, s) for t, s in _trade_exposed(panel, iso2) if t not in exclude]
    if not exposed:
        return (Spillover("trade partners", fallback, "via demand"),)
    return tuple(Spillover(t, f"{panel.countries[t].name} sends {s:.0f} % of its exports here — {text}", "via demand")
                 for t, s in exposed)


def _not_triggered(rule_id: str, title: str, constraint: str, inputs: dict, conf: float) -> Pressure:
    return Pressure(rule_id, title, False, 0.0, constraint, (), (), conf, inputs)


# ─── rules ───────────────────────────────────────────────────────────────────


def rule_fiscal_dominance(iso2: str, panel: Panel) -> Pressure:
    c = panel.countries[iso2]
    req = ["gov_debt_pct_gdp", "fiscal_balance_pct_gdp", "interest_burden_pct_gdp", "gdp_growth_fwd5"]
    inp = {k: _v(panel, iso2, k) for k in req}
    # growth is optional (it only sharpens r > g): ×0.9 when absent, ×0.7 when
    # present because it is forecast-derived.
    conf = round(_confidence(inp, req[:3], c) * (0.9 if inp["gdp_growth_fwd5"] is None else 0.7), 3)
    debt, bal, intr, g = inp["gov_debt_pct_gdp"], inp["fiscal_balance_pct_gdp"], inp["interest_burden_pct_gdp"], inp["gdp_growth_fwd5"]
    title = "Fiscal dominance"
    if debt is None or (bal is None and intr is None):
        return _not_triggered("fiscal_dominance", title, "insufficient inputs", inp, conf)
    deficit_bad = bal is not None and bal < -3.0
    interest_bad = intr is not None and intr > 2.5
    if not (debt > 90.0 and (deficit_bad or interest_bad)):
        return _not_triggered("fiscal_dominance", title,
                              "debt ≤ 90 % of GDP, or deficit ≥ −3 % and interest ≤ 2.5 %", inp, conf)
    sev = max(_excess(debt, 90.0, 60.0),
              _excess(bal, -3.0, 5.0, higher_is_worse=False) if bal is not None else 0.0,
              _excess(intr, 2.5, 2.5) if intr is not None else 0.0)
    r_gt_g = False
    if intr is not None and g is not None and debt > 0:
        implied_rate = intr / debt * 100.0        # nominal effective rate on the debt stock
        r_gt_g = implied_rate > g                 # vs REAL forward growth: conservative flag
        if r_gt_g:
            sev = min(1.0, sev + 0.2)
    constraint = (f"Debt {debt:.0f} % of GDP with "
                  + (f"deficit {bal:.1f} %" if deficit_bad else f"interest {intr:.1f} % of GDP")
                  + (" · r > g" if r_gt_g else ""))
    if c.fx_regime == "reserve_issuer":
        options = ("monetize / inflate the stock away", "financial repression (negative real rates)",
                   "slow austerity behind a reserve-currency shield")
        holders = _top_by(panel, "reserves_months_imports", 5, iso2)
        peggers = _players(panel, lambda k: k.fx_regime == "peg" and k.iso2 != iso2)
        spill = tuple(Spillover(t, f"real return on {c.currency} reserves falls", "via rates")
                      for t in holders if t not in peggers) + \
                tuple(Spillover(t, f"{c.currency} peg imports the issuer's inflation and rates", "via FX")
                      for t in peggers)
    elif c.fx_regime == "currency_union":
        options = ("austerity within the union's fiscal rules", "internal devaluation (wages, prices)",
                   "restructuring with a central-bank backstop — the ECB decides on inflation")
        partners = _players(panel, lambda k: (k.eu_member or k.iso2 == "EU") and k.iso2 != iso2)
        spill = tuple(Spillover(t, "sovereign spread contagion inside the union", "via rates") for t in partners) + \
                (Spillover("domestic banks", "sovereign-bank loop: bank capital tied to the sovereign", "via debt"),)
    else:
        options = ("austerity", "restructure / default", "inflate with FX depreciation")
        spill = (Spillover("domestic banks", "sovereign-bank loop: bank capital tied to the sovereign", "via debt"),
                 Spillover("foreign holders", "haircut or depreciation risk on local-currency debt", "via FX"))
    return Pressure("fiscal_dominance", title, True, round(sev, 3), constraint, options, spill, conf, inp)


def rule_external_financing(iso2: str, panel: Panel) -> Pressure:
    c = panel.countries[iso2]
    req = ["current_account_pct_gdp", "reserves_months_imports"]
    inp = {k: _v(panel, iso2, k) for k in req}
    inp["fx_regime"] = None
    conf = _confidence(inp, req, c)
    ca, res = inp["current_account_pct_gdp"], inp["reserves_months_imports"]
    title = "External financing"
    if c.fx_regime in ("reserve_issuer", "currency_union"):
        return _not_triggered("external_financing", title, "reserve issuer / currency union: no own-FX sudden stop", inp, conf)
    if ca is None or res is None:
        return _not_triggered("external_financing", title, "insufficient inputs", inp, conf)
    if not (ca < -3.0 and res < 4.0):
        return _not_triggered("external_financing", title, "current account ≥ −3 % or reserves ≥ 4 months", inp, conf)
    sev = max(_excess(ca, -3.0, 5.0, higher_is_worse=False), _excess(res, 4.0, 4.0, higher_is_worse=False))
    constraint = f"Current account {ca:.1f} % of GDP with {res:.1f} months of reserves ({c.fx_regime} FX)"
    options = ("devalue", "hike rates to defend the currency", "capital controls", "IMF programme")
    spill = (Spillover("foreign holders", "convertibility / repatriation risk — jurisdiction-gate flag", "via FX"),
             *_partner_spillovers(panel, iso2, "import compression",
                                  "cheaper exports from, and lost demand in, this economy"))
    return Pressure("external_financing", title, True, round(sev, 3), constraint, options, spill, conf, inp)


def rule_credit_bust(iso2: str, panel: Panel) -> Pressure:
    c = panel.countries[iso2]
    req = ["debt_service_ratio"]
    inp = {k: _v(panel, iso2, k) for k in req}
    conf = _confidence(inp, req, c)
    dsr = inp["debt_service_ratio"]
    q90 = float(panel.dsr_q90.get(iso2, DSR_FALLBACK_Q90))
    inp["dsr_q90"] = q90
    title = "Credit bust"
    if dsr is None:
        return _not_triggered("credit_bust", title, "no debt-service ratio for this country (BIS)", inp, conf)
    if dsr <= q90:
        return _not_triggered("credit_bust", title, f"DSR {dsr:.1f} ≤ distress threshold {q90:.1f}", inp, conf)
    sev = _excess(dsr, q90, 6.0)
    constraint = f"Private debt service {dsr:.1f} % of income > distress threshold {q90:.1f}"
    options = ("bank recapitalisation → public debt", "rate cuts if inflation allows", "forbearance / repression")
    spill = (Spillover(iso2, "losses migrate to the sovereign → feeds fiscal dominance", "via debt"),
             Spillover("creditors abroad", "cross-border bank claims on this economy reprice", "via rates"))
    return Pressure("credit_bust", title, True, round(sev, 3), constraint, options, spill, conf, inp)


def rule_demographic_squeeze(iso2: str, panel: Panel) -> Pressure:
    c = panel.countries[iso2]
    req = ["old_age_dependency", "gov_debt_pct_gdp"]
    inp = {k: _v(panel, iso2, k) for k in req}
    conf = _confidence(inp, req, c)
    dep, debt = inp["old_age_dependency"], inp["gov_debt_pct_gdp"]
    trend = panel.trend.get((iso2, "old_age_dependency"))
    title = "Demographic squeeze"
    if dep is None or debt is None:
        return _not_triggered("demographic_squeeze", title, "insufficient inputs", inp, conf)
    if not (dep > 30.0 and trend == "worsening" and debt > 90.0):
        return _not_triggered("demographic_squeeze", title,
                              "dependency ≤ 30, not worsening, or debt ≤ 90 %", inp, conf)
    sev = max(_excess(dep, 30.0, 20.0), _excess(debt, 90.0, 60.0))
    constraint = f"Old-age dependency {dep:.0f} and rising, with debt {debt:.0f} % of GDP"
    options = ("raise the retirement age", "immigration", "pension cuts", "higher taxes", "inflate")
    spill = (Spillover(iso2, "structural growth drag → lowers the forward-growth input to fiscal dominance", "via demand"),
             Spillover(iso2, "savings run-down → lower current-account surplus", "via demand"))
    return Pressure("demographic_squeeze", title, True, round(sev, 3), constraint, options, spill, conf, inp)


def rule_energy_dependence(iso2: str, panel: Panel) -> Pressure:
    c = panel.countries[iso2]
    req = ["energy_net_imports_pct"]
    inp = {k: _v(panel, iso2, k) for k in req}
    conf = _confidence(inp, req, c)
    e = inp["energy_net_imports_pct"]
    title = "Energy dependence"
    if e is None:
        return _not_triggered("energy_dependence", title, "insufficient inputs", inp, conf)
    if e <= 50.0:
        return _not_triggered("energy_dependence", title, "energy net imports ≤ 50 % of use", inp, conf)
    sev = _excess(e, 50.0, 50.0)
    constraint = f"Imports {e:.0f} % of the energy it uses"
    options = ("long-term supply contracts", "subsidise / absorb terms-of-trade shocks",
               "build nuclear and renewables", "strategic stockpiles")
    # Goods-only IMTS cannot isolate energy flows: exporters are still picked by
    # their energy balance, then ORDERED by this country's import share from them.
    exporters = [i for i in panel.values.index
                 if (x := _v(panel, i, "energy_net_imports_pct")) is not None and x < -25.0 and i != iso2
                 and panel.countries.get(i) and panel.countries[i].on_map]
    m_share = {t: (import_share(panel.trade, iso2, t) if panel.trade is not None else None) for t in exporters}
    exporters.sort(key=lambda t: -(m_share[t] if m_share[t] is not None else -1.0))
    spill = tuple(Spillover(t, "exposed to this exporter's supply and pricing"
                            + (f" · {m_share[t]:.0f} % of its imports" if m_share[t] is not None else ""),
                            "via supply") for t in exporters)
    return Pressure("energy_dependence", title, True, round(sev, 3), constraint, options, spill, conf, inp)


def rule_isolation(iso2: str, panel: Panel, pv_pct: float | None = None) -> Pressure:
    c = panel.countries[iso2]
    inp: dict[str, float | None] = {"political_stability_pct": pv_pct, "sanctioned": 1.0 if c.sanctioned else 0.0}
    conf = _confidence(inp, ["political_stability_pct"], c) if not c.sanctioned else 1.0 * (0.8 if c.data_quality in (DataQuality.LOW, DataQuality.OPAQUE) else 1.0)
    title = "Isolation"
    low_stability = pv_pct is not None and pv_pct < 20.0
    if not (c.sanctioned or low_stability):
        return _not_triggered("isolation", title, "not sanctioned; political stability ≥ 20th percentile", inp, conf)
    sev = 1.0 if c.sanctioned else _excess(pv_pct, 20.0, 20.0, higher_is_worse=False)  # type: ignore[arg-type]
    constraint = "Under sanctions" if c.sanctioned else f"Political stability at the {pv_pct:.0f}th percentile"
    options = ("capital controls", "import substitution", "alternative payment rails",
               "reserve diversification into gold / CNY")
    issuers = _players(panel, lambda k: k.fx_regime == "reserve_issuer" and k.iso2 != iso2)
    spill = tuple(Spillover(t, "marginal loss of demand for reserve-issuer debt", "via rates") for t in issuers) + \
            _partner_spillovers(panel, iso2, "exports at risk of re-routing", "re-routed trade and payment flows",
                                exclude=issuers)
    return Pressure("isolation", title, True, round(sev, 3), constraint, options, spill, conf, inp)


RULES: tuple[Callable[..., Pressure], ...] = (
    rule_fiscal_dominance, rule_external_financing, rule_credit_bust,
    rule_demographic_squeeze, rule_energy_dependence, rule_isolation,
)


def evaluate(iso2: str, panel: Panel, pv_pct: float | None = None) -> tuple[Pressure, ...]:
    """All six rules for one player (triggered or not)."""
    out = []
    for rule in RULES:
        if rule is rule_isolation:
            out.append(rule(iso2, panel, pv_pct))
        else:
            out.append(rule(iso2, panel))
    return tuple(out)


def to_dict(p: Pressure) -> dict:
    return {
        "rule_id": p.rule_id, "title": p.title, "triggered": p.triggered, "severity": p.severity,
        "constraint": p.constraint, "forced_options": list(p.forced_options),
        "spillovers": [{"target": s.target, "text": s.text, "channel": s.channel} for s in p.spillovers],
        "confidence": p.confidence, "inputs": dict(p.inputs), "uncertainty": p.uncertainty,
    }
