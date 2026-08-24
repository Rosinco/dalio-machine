"""Pure data shaping for the Fundamentals page — no Streamlit, no plotly.

Everything the charts and tables need is computed here from a `Snapshot`, so
it is unit-testable on the synthetic fixture and the rendering layer stays
thin. Percentiles arrive already oriented (higher = better) — never re-orient.
"""
from __future__ import annotations

import html
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

from dalio.app.fundamentals.snapshot import Snapshot
from dalio.app.theme import EXPOSURE_RAMP, PCT_BIN_LABELS, PCT_RAMP, PLAYER_CENTROIDS

TIER_WEIGHT = {"A": 1.0, "B": 0.6, "C": 0.3}
DQ_MARKED = ("low", "opaque")
TREND_GLYPH = {"improving": "▲", "worsening": "▼", "flat": "▬", None: "·"}


class MapMode(StrEnum):
    COMPOSITE = "composite"
    CATEGORY = "category"
    INDICATOR = "indicator"
    CHAINS = "chains"
    EXPOSURE = "exposure"


@dataclass(frozen=True)
class MapLayer:
    locations: tuple[str, ...]
    z_bin: tuple[float, ...]
    hover: tuple[str, ...]
    opacity: tuple[float, ...]
    selected: tuple[bool, ...]
    no_data_locations: tuple[str, ...]
    no_data_hover: tuple[str, ...]
    dq_points: tuple[tuple[float, float, str], ...]   # (lat, lon, hover) for flagged players
    legend: tuple[tuple[str, str], ...]
    ramp: tuple[str, ...]
    title: str
    caption: str


# ─── Small helpers ───────────────────────────────────────────────────────────


def bin_quintile(values: pd.Series) -> pd.Series:
    """0–100 → bin 0..4 (NaN preserved). 100 lands in the top bin."""
    v = values.astype(float)
    out = np.floor(v / 20.0).clip(upper=4)
    return out.where(v.notna())


def rank_label(pct: float | None, n: int, interpolated: bool = False) -> str:
    if pct is None or (isinstance(pct, float) and np.isnan(pct)):
        return "—"
    r = int(round((100.0 - pct) / 100.0 * (n - 1))) + 1
    return f"≈{r}/{n}" if interpolated else f"{r}/{n}"


def fmt_value(value: float | None, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    return f"{value:.1f}"


def iso3_to_player(snap: Snapshot, bloc: bool = False) -> dict[str, str]:
    """ISO-3 polygon → player iso2 for click handling.

    Members mode (default): individual players win; the aggregate covers the
    remaining member polygons. Bloc mode: every member polygon → the aggregate.
    """
    mapping: dict[str, str] = {}
    aggregates = snap.players[~snap.players["on_map"]]
    individuals = snap.players[snap.players["on_map"]]
    if bloc:
        for _, agg in aggregates.iterrows():
            for iso3 in agg["members"]:
                mapping[iso3] = agg["iso2"]
        for _, p in individuals.iterrows():
            mapping.setdefault(p["iso3"], p["iso2"])
    else:
        for _, p in individuals.iterrows():
            mapping[p["iso3"]] = p["iso2"]
        for _, agg in aggregates.iterrows():
            for iso3 in agg["members"]:
                mapping.setdefault(iso3, agg["iso2"])
    return mapping


def player_locations(snap: Snapshot, iso2: str, bloc: bool = False) -> tuple[str, ...]:
    """Polygons shaded for a player (aggregates expand to members)."""
    row = snap.players.loc[snap.players["iso2"] == iso2].iloc[0]
    if row["on_map"]:
        return (row["iso3"],)
    members = tuple(row["members"])
    if bloc:
        return members
    own = set(snap.players.loc[snap.players["on_map"], "iso3"])
    return tuple(m for m in members if m not in own)


# ─── Metric selection ────────────────────────────────────────────────────────


def metric_series(snap: Snapshot, mode: MapMode, key: str | None, view: str) -> tuple[pd.Series, str]:
    """0–100 series indexed by iso2 for the map + a human label."""
    if mode == MapMode.COMPOSITE:
        s = snap.view_scores[snap.view_scores["view"] == view].set_index("iso2")["score"]
        return s.astype(float), f"Composite · {view} view"
    if mode == MapMode.CATEGORY:
        cs = snap.category_scores[snap.category_scores["category"] == key]
        return cs.set_index("iso2")["score"].astype(float), snap.category_labels.get(key, key or "")
    if mode == MapMode.INDICATOR:
        ind = snap.indicators[snap.indicators["indicator"] == key]
        meta = snap.catalog.get(key or "")
        return ind.set_index("iso2")["pct"].astype(float), (meta.label if meta else key or "")
    if mode == MapMode.CHAINS:
        counts = pd.Series({c: 0.0 for c in snap.player_codes})
        for ch in snap.chains:
            if ch.triggered:
                counts[ch.iso2] += 1
        return counts, "Pressure chains fired"
    raise ValueError(f"metric_series does not handle {mode}")


def exposure_counts(snap: Snapshot, source_iso2: str) -> dict[str, int]:
    """How many of ``source_iso2``'s fired rules name each other player."""
    out: dict[str, int] = {}
    for ch in snap.chains:
        if ch.iso2 != source_iso2 or not ch.triggered:
            continue
        for target in ch.targets:
            if target != source_iso2:
                out[target] = out.get(target, 0) + 1
    return out


# ─── Map layer ───────────────────────────────────────────────────────────────


def _hover(snap: Snapshot, iso2: str, label: str, value_text: str, pct: float | None,
           extra: str = "") -> str:
    p = snap.players.loc[snap.players["iso2"] == iso2].iloc[0]
    n = len(snap.ranking_population)
    interp = iso2 not in snap.ranking_population
    lines = [f"<b>{html.escape(str(p['name']))}</b>",
             f"{html.escape(label)}: {value_text} · rank {rank_label(pct, n, interp)}"]
    if extra:
        lines.append(extra)
    if p["dq_flag"] in DQ_MARKED:
        lines.append(f"† data quality: {p['dq_flag']}")
    return "<br>".join(lines)


def map_layer(
    snap: Snapshot,
    mode: MapMode,
    key: str | None,
    view: str,
    selected_iso2: str | None,
    bloc: bool = False,
) -> MapLayer:
    n = len(snap.ranking_population)
    if mode == MapMode.EXPOSURE:
        counts = exposure_counts(snap, selected_iso2 or "")
        series = pd.Series({c: float(counts.get(c, np.nan)) for c in snap.player_codes})
        label = f"Exposure to {snap.player_name(selected_iso2 or '')}'s pressure chains"
        z = series.clip(upper=4) - 1          # 1..4+ → 0..3
        ramp = EXPOSURE_RAMP
        legend = tuple(zip(EXPOSURE_RAMP, ("1 chain", "2", "3", "4+"), strict=True))
        caption = ("Countries shaded by the number of the selected country's fired rules "
                   "whose spillovers name them.")
    elif mode == MapMode.CHAINS:
        series, label = metric_series(snap, mode, key, view)
        z = series.clip(upper=4) - 1
        series = series.where(series > 0)
        z = z.where(series.notna())
        ramp = EXPOSURE_RAMP
        legend = tuple(zip(EXPOSURE_RAMP, ("1 chain", "2", "3", "4+"), strict=True))
        caption = "Number of pressure-chain rules currently firing (tier C, rules not data)."
    else:
        series, label = metric_series(snap, mode, key, view)
        z = bin_quintile(series)
        ramp = PCT_RAMP
        legend = tuple(zip(PCT_RAMP, PCT_BIN_LABELS, strict=True))
        caption = (f"Quintile bins of {label.lower()} among the {n} ranked countries — "
                   f"one bin is roughly four countries; the euro-area aggregate is interpolated.")

    # Per-indicator extras for hover (value, tier, as-of)
    ind_cells = None
    if mode == MapMode.INDICATOR and key:
        ind_cells = snap.indicators[snap.indicators["indicator"] == key].set_index("iso2")
        meta = snap.catalog.get(key)
        unit = meta.unit if meta else ""

    players = snap.players.copy()
    players["_sel"] = players["iso2"] == selected_iso2
    players = players.sort_values("_sel", kind="stable")          # selected last

    locations: list[str] = []
    z_bin: list[float] = []
    hover: list[str] = []
    opacity: list[float] = []
    selected: list[bool] = []
    no_data: list[str] = []
    no_data_hover: list[str] = []
    dq_points: list[tuple[float, float, str]] = []

    for _, p in players.iterrows():
        iso2 = p["iso2"]
        locs = player_locations(snap, iso2, bloc)
        if bloc and not p["on_map"]:
            pass
        elif bloc and p["eu_member"]:
            continue                                      # bloc covers members
        zval = z.get(iso2, np.nan)
        if isinstance(zval, float) and np.isnan(zval):
            for loc in locs:
                no_data.append(loc)
                no_data_hover.append(f"<b>{html.escape(str(p['name']))}</b><br>no data for {html.escape(label)}")
            continue
        if ind_cells is not None and iso2 in ind_cells.index:
            c = ind_cells.loc[iso2]
            vtxt = f"{fmt_value(c['value'], unit)} {html.escape(unit)}".strip()
            extra = f"tier {c['tier']} · as of {c['as_of']} · {c['source']}"
            pct = c["pct"]
        else:
            v = series.get(iso2, np.nan)
            vtxt = fmt_value(v)
            extra = ""
            pct = v if mode in (MapMode.COMPOSITE, MapMode.CATEGORY) else None
        h = _hover(snap, iso2, label, vtxt, pct, extra)
        is_agg_cover = (not p["on_map"]) and not bloc
        for loc in locs:
            locations.append(loc)
            z_bin.append(float(zval))
            hover.append(h)
            opacity.append(0.45 if is_agg_cover else 1.0)
            selected.append(bool(p["_sel"]))
        if p["dq_flag"] in DQ_MARKED and iso2 in PLAYER_CENTROIDS:
            lat, lon = PLAYER_CENTROIDS[iso2]
            dq_points.append((lat, lon, f"{p['name']}: official statistics {p['dq_flag']} — treat with caution"))

    return MapLayer(
        locations=tuple(locations), z_bin=tuple(z_bin), hover=tuple(hover),
        opacity=tuple(opacity), selected=tuple(selected),
        no_data_locations=tuple(no_data), no_data_hover=tuple(no_data_hover),
        dq_points=tuple(dq_points), legend=legend, ramp=tuple(ramp),
        title=label, caption=caption,
    )


# ─── Leaderboard ─────────────────────────────────────────────────────────────


def coverage_confidence(snap: Snapshot, iso2: str) -> float:
    """Mean tier weight (A 1 · B .6 · C .3) × share of indicators present, in [0, 1]."""
    ind = snap.indicators[snap.indicators["iso2"] == iso2]
    if ind.empty:
        return 0.0
    present = ind["value"].notna()
    if not present.any():
        return 0.0
    w = ind.loc[present, "tier"].map(TIER_WEIGHT).fillna(0.3).mean()
    return float(w * present.mean())


def leaderboard(snap: Snapshot, view: str) -> pd.DataFrame:
    """One row per player: name, composite for ``view``, category scores,
    fired chains, coverage confidence. Sorted by composite desc, NaN last."""
    comp = snap.view_scores[snap.view_scores["view"] == view].set_index("iso2")["score"]
    cats = snap.category_scores.pivot(index="iso2", columns="category", values="score")
    chains = pd.Series({c: 0 for c in snap.player_codes})
    for ch in snap.chains:
        if ch.triggered:
            chains[ch.iso2] += 1
    rows = []
    for _, p in snap.players.iterrows():
        iso2 = p["iso2"]
        name = f"{p['name']} (Σ)" if not p["on_map"] else str(p["name"])
        if p["dq_flag"] in DQ_MARKED:
            name += " †"
        row = {"iso2": iso2, "Player": name, "Composite": comp.get(iso2, np.nan)}
        for cat in snap.categories:
            row[snap.category_labels.get(cat, cat)] = cats.at[iso2, cat] if iso2 in cats.index and cat in cats.columns else np.nan
        row["Chains"] = int(chains.get(iso2, 0))
        row["Coverage"] = round(coverage_confidence(snap, iso2) * 100)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values("Composite", ascending=False, na_position="last", kind="stable").reset_index(drop=True)


# ─── Purpose views ───────────────────────────────────────────────────────────

VIEW_LABELS: dict[str, str] = {
    "learning": "Learning",
    "jurisdiction": "Jurisdiction",
    "allocation": "Allocation",
    "moonshot": "Moonshot",
}

VIEW_CAPTIONS: dict[str, str] = {
    "learning": ("Equal weights. Tells you where a country is strong or weak relative to the "
                 "other twenty; cannot tell you what to do about it."),
    "jurisdiction": ("Enforcer 50 · Promises 30 · Exchange 20. Tells you which jurisdictions carry "
                     "structural risk to a concentrated stock position (property rights, "
                     "convertibility, sovereign stress); cannot tell you whether the company "
                     "itself is exposed to that risk."),
    "allocation": ("Promises 40 · Production 30 · Exchange 20 · Enforcer 10. Tells you which "
                   "economies' fundamentals argue for more or less margin of safety in the "
                   "index/value sleeve; cannot time anything — pair it with the cycle map."),
    "moonshot": ("Real stuff 50 · Exchange 30 · Production 20. Tells you which countries own the "
                 "physical inputs and trade position that supply chains run through; cannot "
                 "tell you which company captures it."),
}


def view_caption(view: str) -> str:
    return VIEW_CAPTIONS.get(view, "")


def weights_line(snap: Snapshot, view: str) -> str:
    """'Jurisdiction view · Enforcer 50 · Promises 30 · Exchange 20' (mono line)."""
    w = snap.views.get(view, {})
    parts = [f"{snap.category_labels.get(c, c)} {round(v * 100)}" for c, v in
             sorted(w.items(), key=lambda kv: -kv[1]) if v > 0]
    return f"{VIEW_LABELS.get(view, view)} view · " + " · ".join(parts)


# ─── Pareto (gap to best-in-class) ───────────────────────────────────────────


def pareto_frame(snap: Snapshot, iso2: str, level: str = "indicator", top_n: int = 10) -> pd.DataFrame:
    """Where does a player's weakness concentrate?

    ``level="indicator"``: gap = 100 − percentile per scored indicator.
    ``level="category"``: gap = distance_to_best per category score.
    Sorted descending with ``share`` (of total gap), ``cum_share`` and
    ``crosses_80`` (True on the first row where cumulative share ≥ 80 %).
    Rows with no data are excluded; an all-zero gap yields an empty frame.
    """
    if level == "category":
        cs = snap.category_scores[snap.category_scores["iso2"] == iso2]
        df = pd.DataFrame({
            "key": cs["category"].map(lambda c: snap.category_labels.get(c, c)),
            "gap": cs["distance_to_best"].astype(float),
        })
    else:
        ind = snap.indicators[snap.indicators["iso2"] == iso2]
        df = pd.DataFrame({
            "key": ind["indicator"].map(lambda n: snap.catalog[n].label if n in snap.catalog else n),
            "gap": (100.0 - ind["pct"].astype(float)),
        })
    df = df[df["gap"].notna()].copy()
    total = float(df["gap"].sum())
    if df.empty or total <= 0:
        return pd.DataFrame(columns=["key", "gap", "share", "cum_share", "crosses_80"])
    df = df.sort_values("gap", ascending=False, kind="stable").reset_index(drop=True)
    df["share"] = df["gap"] / total
    df["cum_share"] = df["share"].cumsum()
    first = int((df["cum_share"] >= 0.8).idxmax()) if (df["cum_share"] >= 0.8).any() else len(df) - 1
    df["crosses_80"] = [i == first for i in range(len(df))]
    return df.head(top_n).reset_index(drop=True)


def pareto_caption(df: pd.DataFrame, name: str, level: str) -> str:
    if df.empty:
        return f"{name} has no gap to best-in-class on the scored {level}s — or no data."
    n80 = int(df.index[df["crosses_80"]][0]) + 1 if df["crosses_80"].any() else len(df)
    unit = "indicators" if level == "indicator" else "categories"
    return (f"{n80} of {len(df)} shown {unit} explain 80 % of {name}'s gap to best-in-class "
            f"(largest gap: {df.iloc[0]['key']}, {df.iloc[0]['gap']:.0f} points).")


# ─── Country table ───────────────────────────────────────────────────────────


def country_table(snap: Snapshot, iso2: str) -> pd.DataFrame:
    """Every catalogued indicator for a player, in category then catalog order;
    missing ones kept with NaN so the table shows the gap."""
    ind = snap.indicators[snap.indicators["iso2"] == iso2].set_index("indicator")
    order = {c: i for i, c in enumerate(snap.categories)}
    rows = []
    for name, meta in snap.catalog.items():
        cell = ind.loc[name] if name in ind.index else None
        rows.append({
            "category": meta.category, "indicator": name, "label": meta.label, "unit": meta.unit,
            "value": None if cell is None else cell["value"],
            "pct": None if cell is None else cell["pct"],
            "trend": None if cell is None else cell["trend"],
            "trend_5y": None if cell is None else cell["trend_5y"],
            "tier": meta.uncertainty, "as_of": None if cell is None else cell["as_of"],
            "source": None if cell is None else cell["source"],
            "se": None if cell is None else cell["se"],
            "is_forecast": False if cell is None else bool(cell["is_forecast"]),
        })
    df = pd.DataFrame(rows)
    df["_o"] = df["category"].map(order)
    return df.sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def html_dense_table(df: pd.DataFrame, category_labels: dict[str, str], n_rank: int,
                     interpolated: bool = False) -> str:
    """Dense editorial table: category header rows, mono numerals, inline
    percentile bar, tier badge, trend glyph. All text HTML-escaped."""
    esc = html.escape
    out = ['<table class="dense"><thead><tr>',
           '<th>Indicator</th><th class="num">Value</th><th>Unit</th>'
           '<th>Percentile</th><th>Trend</th><th>Tier</th><th>As of</th><th>Source</th>',
           '</tr></thead><tbody>']
    last_cat = None
    for _, r in df.iterrows():
        if r["category"] != last_cat:
            last_cat = r["category"]
            out.append(f'<tr class="cat"><td colspan="8">{esc(category_labels.get(last_cat, last_cat))}</td></tr>')
        pct = r["pct"]
        has_pct = pct is not None and not (isinstance(pct, float) and np.isnan(pct))
        bar = (f'<span class="pbar"><span style="width:{float(pct):.0f}%"></span></span> '
               f'<span class="mono">{rank_label(pct, n_rank, interpolated)}</span>') if has_pct else "—"
        val = fmt_value(r["value"], r["unit"])
        if r["se"] is not None and not (isinstance(r["se"], float) and np.isnan(r["se"])):
            val += f" ±{float(r['se']):.2f}"
        trend = TREND_GLYPH.get(r["trend"], "·")
        t5 = r["trend_5y"]
        trend_txt = trend if t5 is None or (isinstance(t5, float) and np.isnan(t5)) else f"{trend} {float(t5):+.1f}"
        tier = str(r["tier"]).lower()
        fc = ' fcast' if r["is_forecast"] else ''
        out.append(
            f'<tr><td>{esc(str(r["label"]))}</td>'
            f'<td class="num{fc}">{esc(val)}</td>'
            f'<td class="unit">{esc(str(r["unit"]))}</td>'
            f'<td>{bar}</td>'
            f'<td class="mono trend-{esc(str(r["trend"] or "none"))}">{esc(trend_txt)}</td>'
            f'<td><span class="tier tier-{esc(tier)}">{esc(str(r["tier"]))}</span></td>'
            f'<td class="mono">{esc(str(r["as_of"] or "—"))}</td>'
            f'<td class="src">{esc(str(r["source"] or "—"))}</td></tr>'
        )
    out.append("</tbody></table>")
    return "".join(out)
