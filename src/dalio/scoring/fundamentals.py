"""World Fundamentals Map — indicator registry, percentile scoring, snapshot.

Five categories (a first-principles ontology of the economy):

    real_stuff  · people, land, energy, capital     (what exists)
    production  · output and its growth             (what gets made)
    exchange    · trade, external balance, reserves (how it moves)
    promises    · debt, deficits, debt service      (claims on the future)
    enforcer    · rule of law, stability, force     (what makes promises real)

Dalio's 18 determinants are Pareto-pruned into ≤3 sourceable indicators per
category. Scoring is deliberately simple and inspectable:

* **Percentile rank** among the 21 individual countries (``RANKING_POPULATION``),
  oriented so higher = better. The euro-area aggregate is interpolated against
  that population, never part of it.
* **Category score** = mean of available indicator percentiles; ``None`` if
  fewer than half the category's indicators are present.
* **Views** = weight vectors over category scores (purposes never enter data).

Everything here is judgment made explicit; the numbers carry an uncertainty
tier (A measured / B model or forecast / C ordinal index) so the UI can say so.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from dalio.countries import COUNTRIES, RANKING_POPULATION, Country
from dalio.storage.db import Observation

SNAPSHOT_VERSION = 1

CATEGORIES: tuple[str, ...] = ("real_stuff", "production", "exchange", "promises", "enforcer")
CATEGORY_LABELS: dict[str, str] = {
    "real_stuff": "Real stuff",
    "production": "Production",
    "exchange": "Exchange",
    "promises": "Promises",
    "enforcer": "Enforcer",
}

Uncertainty = Literal["A", "B", "C"]
Trend = Literal["improving", "worsening", "flat"]

FORECAST_SUFFIX = "_FCST"


@dataclass(frozen=True)
class IndicatorSpec:
    name: str
    category: str
    label: str
    unit: str
    higher_is_better: bool
    uncertainty: Uncertainty
    preferred_sources: tuple[str, ...]
    description: str
    cadence: str = "A"
    first_year: int = 1960
    forward: bool = False            # value is a mean of forecast rows after as_of
    se_indicator: str | None = None  # sibling indicator carrying a standard error


# The 15 scored indicators (ADR 0001). Keep this list the single source of
# truth for what is scored — the UI reads it from the snapshot. IMF-sourced
# cells stay empty until slice 20 wires the DataMapper adapter.
FUNDAMENTALS: tuple[IndicatorSpec, ...] = (
    # ── real stuff ──
    IndicatorSpec(
        "energy_net_imports_pct", "real_stuff", "Energy net imports", "% of energy use",
        higher_is_better=False, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1960,
        description="Energy use minus production, as a share of use. Negative = net exporter. "
                    "The physical dependency that supply shocks hit first.",
    ),
    IndicatorSpec(
        "old_age_dependency", "real_stuff", "Old-age dependency", "% of working-age pop.",
        higher_is_better=False, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1960,
        description="People 65+ per 100 people aged 15–64. Higher = fewer hands per pension; "
                    "the slow variable behind growth, savings and fiscal pressure.",
    ),
    # ── production ──
    IndicatorSpec(
        "gdp_pc_ppp", "production", "GDP per capita (PPP)", "intl $ 2021",
        higher_is_better=True, uncertainty="B",
        preferred_sources=("WORLD_BANK",), first_year=1990,
        description="Output per person at purchasing-power parity. The level of prosperity; "
                    "PPP conversion is model-dependent (tier B).",
    ),
    IndicatorSpec(
        "gdp_growth_fwd5", "production", "Real growth, next 5 y", "% p.a. (IMF forecast)",
        higher_is_better=True, uncertainty="B",
        preferred_sources=("IMF_WEO_FCST",), first_year=1980, forward=True,
        description="Mean of the IMF's real-GDP growth forecasts for the next five years. "
                    "A forecast, not a fact (tier B): 1-year RMSE ~1.5 pp, worse beyond.",
    ),
    IndicatorSpec(
        "rd_pct_gdp", "production", "R&D spending", "% of GDP",
        higher_is_better=True, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1996,
        description="Gross domestic expenditure on R&D. The investment behind future "
                    "productivity; ragged in the latest 1–2 years.",
    ),
    # ── exchange ──
    IndicatorSpec(
        "exports_share_world", "exchange", "Share of world exports", "% of world",
        higher_is_better=True, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1960,
        description="Exports of goods and services ÷ world exports. Dalio's trade-share "
                    "measure of power — scale, not openness.",
    ),
    IndicatorSpec(
        "current_account_pct_gdp", "exchange", "Current account", "% of GDP",
        higher_is_better=True, uncertainty="A",
        preferred_sources=("IMF_WEO", "WORLD_BANK"), first_year=1960,
        description="Net lending to the rest of the world. Persistent deficits need financing; "
                    "surpluses export savings. Errors & omissions are large for some players.",
    ),
    IndicatorSpec(
        "reserves_months_imports", "exchange", "FX reserves", "months of imports",
        higher_is_better=True, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1960,
        description="Total reserves in months of import cover. The buffer against a sudden stop; "
                    "less relevant for reserve-currency issuers.",
    ),
    # ── promises ──
    IndicatorSpec(
        "gov_debt_pct_gdp", "promises", "Government debt", "% of GDP",
        higher_is_better=False, uncertainty="B",
        preferred_sources=("IMF_WEO", "BIS_TC"), first_year=1980,
        description="General government gross debt. The stock of promises the state must "
                    "service; BIS core-debt figures fill in until IMF lands.",
    ),
    IndicatorSpec(
        "fiscal_balance_pct_gdp", "promises", "Fiscal balance", "% of GDP",
        higher_is_better=True, uncertainty="B",
        preferred_sources=("IMF_WEO",), first_year=1980,
        description="General government net lending. The flow that grows or shrinks the debt "
                    "stock; the deficit lever in a deleveraging.",
    ),
    IndicatorSpec(
        "interest_burden_pct_gdp", "promises", "Interest burden", "% of GDP",
        higher_is_better=False, uncertainty="B",
        preferred_sources=("IMF_WEO",), first_year=1980,
        description="Primary balance minus overall balance = net interest paid. Where fiscal "
                    "dominance bites; compare with growth (r vs g).",
    ),
    IndicatorSpec(
        "debt_service_ratio", "promises", "Private debt service", "% of income",
        higher_is_better=False, uncertainty="B",
        preferred_sources=("BIS_DSR",), first_year=1999, cadence="Q",
        description="Interest + amortisation of the private non-financial sector over income "
                    "(BIS estimate, tier B). The credit-bust trigger.",
    ),
    # ── enforcer ──
    IndicatorSpec(
        "rule_of_law", "enforcer", "Rule of law", "WGI estimate (−2.5..2.5)",
        higher_is_better=True, uncertainty="C",
        preferred_sources=("WORLD_BANK_WGI",), first_year=1996, se_indicator="rule_of_law_se",
        description="Worldwide Governance Indicators: contract enforcement, property rights, "
                    "courts. Perception-based (tier C) with a published standard error; the "
                    "euro-area value is a flagged member mean.",
    ),
    IndicatorSpec(
        "political_stability", "enforcer", "Political stability", "WGI estimate (−2.5..2.5)",
        higher_is_better=True, uncertainty="C",
        preferred_sources=("WORLD_BANK_WGI",), first_year=1996,
        se_indicator="political_stability_se",
        description="WGI: likelihood of unconstitutional or violent destabilisation. "
                    "Perception-based (tier C); euro-area value is a flagged member mean.",
    ),
    IndicatorSpec(
        "military_share_world", "enforcer", "Share of world military spending", "% of world",
        higher_is_better=True, uncertainty="A",
        preferred_sources=("WORLD_BANK",), first_year=1960,
        description="Military expenditure ÷ world (SIPRI via World Bank). Capacity to enforce "
                    "externally — a power measure, not a virtue score.",
    ),
)

SE_INDICATORS: tuple[str, ...] = tuple(s.se_indicator for s in FUNDAMENTALS if s.se_indicator)

VIEWS: dict[str, dict[str, float]] = {
    "learning": {c: 0.2 for c in CATEGORIES},
    "jurisdiction": {"enforcer": 0.5, "promises": 0.3, "exchange": 0.2},
    "allocation": {"promises": 0.4, "production": 0.3, "exchange": 0.2, "enforcer": 0.1},
    "moonshot": {"real_stuff": 0.5, "exchange": 0.3, "production": 0.2},
}
VIEW_MIN_WEIGHT_COVERAGE = 0.6


@dataclass(frozen=True)
class CategoryScore:
    score: float | None
    n_available: int
    n_total: int
    distance_to_best: float | None
    best_iso2: str | None


# ─── Panel loading ───────────────────────────────────────────────────────────


def _as_of_cutoff(as_of: date, lag_years: int) -> date:
    """``as_of`` minus whole years; only Feb 29 needs clamping (→ Feb 28).

    World Bank rows are dated Dec 31, so a naive day-28 clamp would silently
    turn a 5-year lag into 6 years for any year-end ``as_of``.
    """
    if not lag_years:
        return as_of
    try:
        return as_of.replace(year=as_of.year - lag_years)
    except ValueError:  # Feb 29 → Feb 28
        return as_of.replace(year=as_of.year - lag_years, day=28)


def load_panel(
    session: Session,
    specs: Sequence[IndicatorSpec],
    countries: Sequence[Country],
    as_of: date,
    lag_years: int = 0,
) -> pd.DataFrame:
    """Latest observation per (country, indicator) at or before ``as_of``
    (minus ``lag_years``), honouring each spec's preferred-source order.

    Returns a long frame with columns ``country, indicator, value, date, source``
    — one row per (country, indicator) that has any data. ONE query.
    """
    names = [s.name for s in specs]
    sources = sorted({src for s in specs for src in s.preferred_sources})
    iso2s = [c.iso2 for c in countries]
    cutoff = _as_of_cutoff(as_of, lag_years)
    rows = session.execute(
        select(Observation.country, Observation.indicator, Observation.value,
               Observation.date, Observation.source)
        .where(
            Observation.indicator.in_(names),
            Observation.country.in_(iso2s),
            Observation.source.in_(sources),
            Observation.date <= cutoff,
        )
    ).all()
    if not rows:
        return pd.DataFrame(columns=["country", "indicator", "value", "date", "source"])
    df = pd.DataFrame(rows, columns=["country", "indicator", "value", "date", "source"])
    df = _rank_by_source_preference(df, specs)
    if df.empty:
        return pd.DataFrame(columns=["country", "indicator", "value", "date", "source"])
    df = (
        df.sort_values(["country", "indicator", "_pref", "date"], ascending=[True, True, True, False])
        .drop_duplicates(subset=["country", "indicator"], keep="first")
        .drop(columns="_pref")
        .reset_index(drop=True)
    )
    return df


def _rank_by_source_preference(df: pd.DataFrame, specs: Sequence[IndicatorSpec]) -> pd.DataFrame:
    """Attach ``_pref`` (0 = most preferred) and DROP rows whose source is not
    in that indicator's own ``preferred_sources`` — the query filters on the
    union of all specs' sources, so without this a spec could silently consume
    another spec's source."""
    pref = {s.name: {src: i for i, src in enumerate(s.preferred_sources)} for s in specs}
    ranks = [pref.get(i, {}).get(src) for i, src in zip(df["indicator"], df["source"], strict=True)]
    df = df.assign(_pref=ranks)
    return df[df["_pref"].notna()].copy()


def load_history(
    session: Session,
    specs: Sequence[IndicatorSpec],
    countries: Sequence[Country],
) -> pd.DataFrame:
    """All annual rows for the snapshot's history block (preferred source only).
    Columns: ``country, indicator, year, value, is_forecast``."""
    names = [s.name for s in specs]
    sources = sorted({src for s in specs for src in s.preferred_sources})
    iso2s = [c.iso2 for c in countries]
    rows = session.execute(
        select(Observation.country, Observation.indicator, Observation.date,
               Observation.value, Observation.source)
        .where(
            Observation.indicator.in_(names),
            Observation.country.in_(iso2s),
            Observation.source.in_(sources),
        )
    ).all()
    cols = ["country", "indicator", "year", "value", "is_forecast"]
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=["country", "indicator", "date", "value", "source"])
    df = _rank_by_source_preference(df, specs)
    if df.empty:
        return pd.DataFrame(columns=cols)
    # One source per (country, indicator): the most preferred one that has data.
    best = df.groupby(["country", "indicator"])["_pref"].transform("min")
    df = df[df["_pref"] == best]
    df["year"] = [d.year for d in df["date"]]
    df["is_forecast"] = df["source"].str.endswith(FORECAST_SUFFIX)
    # Quarterly series (BIS DSR): keep the LAST observation of each year.
    df = (
        df.sort_values(["country", "indicator", "year", "date"])
        .drop_duplicates(subset=["country", "indicator", "year"], keep="last")
    )
    return df[cols].reset_index(drop=True)


# ─── Scoring primitives ──────────────────────────────────────────────────────


def percentile_rank(
    values: pd.Series,
    higher_is_better: bool,
    population: Sequence[str] = RANKING_POPULATION,
) -> pd.Series:
    """0–100 percentile, higher = better, over ``population`` members only.

    Population members: ``(rank − 1) / (n − 1) × 100`` with average ranks for
    ties (worst = 0, best = 100). Non-population members present in ``values``
    (the euro-area aggregate) are linearly interpolated against the
    population's value→percentile curve and never affect anyone else's rank.
    ``NaN`` propagates; fewer than two population values → all ``NaN``.
    """
    v = values.astype(float)
    pop = v.reindex([p for p in population if p in v.index]).dropna()
    out = pd.Series(np.nan, index=v.index, dtype=float)
    n = len(pop)
    if n < 2:
        return out
    ranks = pop.rank(method="average", ascending=higher_is_better)
    pct = (ranks - 1.0) / (n - 1.0) * 100.0
    out.loc[pct.index] = pct

    others = [i for i in v.index if i not in pop.index and not np.isnan(v[i])]
    if others:
        # np.interp needs ascending x; y may run either direction (it is the
        # population's own value→percentile curve, which is monotone).
        order = np.argsort(pop.to_numpy(), kind="stable")
        xs = pop.to_numpy()[order]
        ys = pct.to_numpy()[order]
        for i in others:
            out[i] = float(np.interp(v[i], xs, ys))
    return out


def trend_direction(
    latest: float | None,
    lag: float | None,
    higher_is_better: bool,
    cross_std: float | None,
) -> Trend | None:
    """Improving / worsening / flat with a dead band of 0.1 × cross-sectional std."""
    if latest is None or lag is None or cross_std is None or np.isnan(cross_std):
        return None
    delta = latest - lag
    if not higher_is_better:
        delta = -delta
    band = 0.1 * cross_std
    if delta > band:
        return "improving"
    if delta < -band:
        return "worsening"
    return "flat"


def category_scores(
    pct: pd.DataFrame,
    specs: Sequence[IndicatorSpec],
    population: Sequence[str] = RANKING_POPULATION,
) -> dict[str, dict[str, CategoryScore]]:
    """``pct`` is wide: index = iso2, columns = indicator names (percentiles).
    Returns ``{iso2: {category: CategoryScore}}`` for every row of ``pct``."""
    by_cat: dict[str, list[str]] = {c: [] for c in CATEGORIES}
    for s in specs:
        by_cat[s.category].append(s.name)

    result: dict[str, dict[str, CategoryScore]] = {}
    means: dict[str, pd.Series] = {}
    for cat, names in by_cat.items():
        present = [n for n in names if n in pct.columns]
        if not present:
            means[cat] = pd.Series(np.nan, index=pct.index)
            continue
        sub = pct[present]
        n_avail = sub.notna().sum(axis=1)
        mean = sub.mean(axis=1, skipna=True)
        mean[n_avail * 2 < len(names)] = np.nan   # coverage floor: at least half
        means[cat] = mean

    for cat, names in by_cat.items():
        m = means[cat]
        pop_scores = m.reindex([p for p in population if p in m.index]).dropna()
        best_iso2 = str(pop_scores.idxmax()) if not pop_scores.empty else None
        best = float(pop_scores.max()) if not pop_scores.empty else None
        present = [n for n in names if n in pct.columns]
        for iso2 in pct.index:
            score = None if np.isnan(m[iso2]) else float(m[iso2])
            n_avail = int(pct.loc[iso2, present].notna().sum()) if present else 0
            # Non-population rows (the euro-area aggregate) are interpolated and
            # can sit above the population's best; they are never ranked, so
            # the gap floors at zero rather than going negative.
            dist = None if score is None or best is None else max(0.0, float(best - score))
            result.setdefault(str(iso2), {})[cat] = CategoryScore(
                score=score, n_available=n_avail, n_total=len(names),
                distance_to_best=dist, best_iso2=best_iso2,
            )
    return result


def view_scores(
    cat_scores: dict[str, dict[str, CategoryScore]],
    views: dict[str, dict[str, float]] = VIEWS,
) -> dict[str, dict[str, float | None]]:
    """Weighted mean of available category scores per view, weights renormalised
    over the available categories; ``None`` if < 60 % of the view's weight is
    backed by a score."""
    out: dict[str, dict[str, float | None]] = {}
    for iso2, cats in cat_scores.items():
        out[iso2] = {}
        for view, weights in views.items():
            total = sum(weights.values())
            avail = {c: w for c, w in weights.items() if cats.get(c) and cats[c].score is not None}
            covered = sum(avail.values())
            if total <= 0 or covered / total < VIEW_MIN_WEIGHT_COVERAGE:
                out[iso2][view] = None
                continue
            out[iso2][view] = float(
                sum(w * cats[c].score for c, w in avail.items()) / covered  # type: ignore[operator]
            )
    return out


# ─── Snapshot ────────────────────────────────────────────────────────────────


def _wide(panel: pd.DataFrame, col: str, index: Iterable[str]) -> pd.DataFrame:
    if panel.empty:
        return pd.DataFrame(index=list(index))
    return panel.pivot(index="country", columns="indicator", values=col).reindex(list(index))


def build_snapshot(
    session: Session,
    as_of: date | None = None,
    countries: Sequence[Country] = COUNTRIES,
    specs: Sequence[IndicatorSpec] = FUNDAMENTALS,
    population: Sequence[str] = RANKING_POPULATION,
    include_history: bool = True,
) -> dict:
    """Assemble the JSON-serialisable snapshot the app renders."""
    as_of = as_of or date.today()
    iso2s = [c.iso2 for c in countries]
    by_name = {s.name: s for s in specs}

    latest = load_panel(session, specs, countries, as_of)
    lagged = load_panel(session, specs, countries, as_of, lag_years=5)
    # Standard-error siblings (WGI) share the estimate's sources.
    se_specs = [
        IndicatorSpec(s.se_indicator, s.category, s.label, "se", True, s.uncertainty,
                      s.preferred_sources, "standard error", first_year=s.first_year)
        for s in specs if s.se_indicator
    ]
    se_panel = load_panel(session, se_specs, countries, as_of) if se_specs else pd.DataFrame(
        columns=["country", "indicator", "value", "date", "source"]
    )
    se_values = _wide(se_panel, "value", iso2s)
    values = _wide(latest, "value", iso2s)
    dates = _wide(latest, "date", iso2s)
    sources = _wide(latest, "source", iso2s)
    lag_values = _wide(lagged, "value", iso2s)
    lag_dates = _wide(lagged, "date", iso2s)

    pct = pd.DataFrame(index=values.index)
    for s in specs:
        if s.name in values.columns:
            pct[s.name] = percentile_rank(values[s.name], s.higher_is_better, population)
        else:
            pct[s.name] = np.nan

    cats = category_scores(pct, specs, population)
    views = view_scores(cats)
    history = load_history(session, specs, countries) if include_history else pd.DataFrame(
        columns=["country", "indicator", "year", "value", "is_forecast"]
    )

    def _cell(iso2: str, s: IndicatorSpec) -> dict:
        has = s.name in values.columns and not pd.isna(values.at[iso2, s.name])
        if not has:
            return {
                "value": None, "date": None, "source": None, "pct": None, "trend": None,
                "trend_5y": None, "lag_value": None, "is_forecast": False,
                "forecast_horizon_years": None, "se": None, "uncertainty": s.uncertainty,
            }
        v = float(values.at[iso2, s.name])
        lag = lag_values.at[iso2, s.name] if s.name in lag_values.columns else np.nan
        lag = None if pd.isna(lag) else float(lag)
        # A stale series (latest row older than the lag cutoff) serves the same
        # row as both "latest" and "lagged" — that is one data point, not a trend.
        if lag is not None and s.name in lag_dates.columns:
            ld = lag_dates.at[iso2, s.name]
            if ld is not None and not pd.isna(ld) and ld == dates.at[iso2, s.name]:
                lag = None
        pop_std = float(values.loc[[p for p in population if p in values.index], s.name].std())
        src = str(sources.at[iso2, s.name])
        d = dates.at[iso2, s.name]
        se = None
        if s.se_indicator and s.se_indicator in se_values.columns:
            se_v = se_values.at[iso2, s.se_indicator]
            se = None if pd.isna(se_v) else float(se_v)
        return {
            "value": v,
            "date": d.isoformat() if d is not None and not pd.isna(d) else None,
            "source": src,
            "pct": None if pd.isna(pct.at[iso2, s.name]) else float(pct.at[iso2, s.name]),
            "trend": trend_direction(v, lag, s.higher_is_better, pop_std),
            "trend_5y": None if lag is None else float(v - lag),
            "lag_value": lag,
            "is_forecast": src.endswith(FORECAST_SUFFIX),
            "forecast_horizon_years": None,
            "se": se,
            "uncertainty": s.uncertainty,
        }

    countries_block: dict[str, dict] = {}
    filled = 0
    by_indicator: dict[str, int] = dict.fromkeys(by_name, 0)
    for c in countries:
        cells = {s.name: _cell(c.iso2, s) for s in specs}
        for name, cell in cells.items():
            if cell["value"] is not None:
                filled += 1
                by_indicator[name] += 1
        hist = history[history["country"] == c.iso2] if not history.empty else history
        countries_block[c.iso2] = {
            "name": c.name,
            "iso3": c.iso3,
            "tier": int(c.tier),
            "eu_member": c.eu_member,
            "members": list(c.members),
            "on_map": c.on_map,
            "fx_regime": c.fx_regime,
            "sanctioned": c.sanctioned,
            "currency": c.currency,
            "data_quality": {"flag": str(c.data_quality), "note": c.data_quality_note},
            "indicators": cells,
            "categories": {cat: asdict(cs) for cat, cs in cats[c.iso2].items()},
            "views": views[c.iso2],
            "pressures": [],
            "cycle": None,
            "history": {
                name: [
                    {"year": int(r.year), "value": float(r.value), "is_forecast": bool(r.is_forecast)}
                    for r in hist[hist["indicator"] == name].sort_values("year").itertuples()
                ]
                for name in by_name
            } if include_history else {},
        }

    return {
        "version": SNAPSHOT_VERSION,
        "as_of": as_of.isoformat(),
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "ranking_population": list(population),
        "indicators": [
            {
                "name": s.name, "category": s.category, "label": s.label, "unit": s.unit,
                "uncertainty": s.uncertainty, "higher_is_better": s.higher_is_better,
                "description": s.description, "cadence": s.cadence,
                "sources": list(s.preferred_sources), "first_year": s.first_year,
            }
            for s in specs
        ],
        "categories": list(CATEGORIES),
        "category_labels": dict(CATEGORY_LABELS),
        "views": {k: dict(v) for k, v in VIEWS.items()},
        "countries": countries_block,
        "trade": None,
        "coverage": {
            "cells": len(countries) * len(specs),
            "filled": filled,
            "by_indicator": by_indicator,
        },
    }


def write_snapshot(snapshot: dict, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, indent=1, ensure_ascii=False))
    return path


def read_snapshot(path: Path) -> dict:
    return json.loads(path.read_text())
