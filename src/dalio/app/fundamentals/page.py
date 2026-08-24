"""Fundamentals page — Streamlit rendering only (thin).

Layout: control strip → map → leaderboard + country panel (two columns) →
pressure chains → trade partners → trajectories → Pareto.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dalio.app.fundamentals.charts import build_bubble, build_fundamentals_map, build_pareto
from dalio.app.fundamentals.dot import chain_to_dot
from dalio.app.fundamentals.snapshot import (
    Snapshot,
    SnapshotError,
    load_snapshot,
    snapshot_dir,
    snapshot_fingerprint,
)
from dalio.app.fundamentals.view_models import (
    BUBBLE_DEFAULT_X,
    BUBBLE_DEFAULT_Y,
    BUBBLE_SIZE_INDICATOR,
    VIEW_LABELS,
    MapMode,
    bubble_frame,
    bubble_indicator_options,
    country_table,
    coverage_confidence,
    forecast_boundary,
    html_dense_table,
    html_trade_table,
    iso3_to_player,
    leaderboard,
    map_layer,
    pareto_caption,
    pareto_frame,
    trade_caption,
    trade_partner_table,
    view_caption,
    weights_line,
)
from dalio.app.theme import confidence_block, legend_row

MAP_KEY = "fund_map"
LB_KEY = "fund_lb"
_ISO3_MAP_KEY = "_fund_iso3_map"
_LB_ORDER_KEY = "_fund_lb_order"
DEFAULT_VIEW = "learning"


@st.cache_data(show_spinner=False)
def _cached_snapshot(dir_str: str, fingerprint: int) -> Snapshot:
    """``fingerprint`` (file mtime) is part of the cache key, so a re-export
    invalidates automatically without restarting the app."""
    return load_snapshot(Path(dir_str))


def get_snapshot() -> Snapshot | None:
    d = snapshot_dir()
    fp = snapshot_fingerprint(d)
    if fp == 0:
        return None
    return _cached_snapshot(str(d), fp)


# ─── Selection callbacks (run before widgets are instantiated) ──────────────


def _on_map_select() -> None:
    state = st.session_state.get(MAP_KEY)
    sel = getattr(state, "selection", None)
    pts = sel.get("points", []) if isinstance(sel, dict) else getattr(sel, "points", []) or []
    if not pts:
        return
    loc = pts[0].get("location") if isinstance(pts[0], dict) else None
    target = (st.session_state.get(_ISO3_MAP_KEY) or {}).get(loc)
    if target and target != st.session_state.get("country"):
        st.session_state.country = target


def _on_lb_select() -> None:
    state = st.session_state.get(LB_KEY)
    sel = getattr(state, "selection", None)
    rows = sel.get("rows", []) if isinstance(sel, dict) else getattr(sel, "rows", []) or []
    order = st.session_state.get(_LB_ORDER_KEY) or []
    if rows and 0 <= rows[0] < len(order):
        target = order[rows[0]]
        if target != st.session_state.get("country"):
            st.session_state.country = target


def _on_spill_select(key: str) -> None:
    """Spillover pill clicked: follow the chain to that player."""
    target = st.session_state.get(key)
    if target and target != st.session_state.get("country"):
        st.session_state.country = target


# ─── Sections ────────────────────────────────────────────────────────────────


def _purpose_view(snap: Snapshot) -> str:
    """Purpose view = a weight vector over category scores (never in the data)."""
    views = [v for v in VIEW_LABELS if v in snap.views]
    chosen = st.segmented_control(
        "Purpose", views, default=DEFAULT_VIEW if DEFAULT_VIEW in views else views[0],
        format_func=lambda v: VIEW_LABELS.get(v, v), key="purpose_view",
    ) or DEFAULT_VIEW
    st.markdown(f'<div class="weights">{weights_line(snap, chosen)}</div>', unsafe_allow_html=True)
    st.caption(view_caption(chosen))
    return chosen


def _control_strip(snap: Snapshot, view: str) -> tuple[MapMode, str | None, bool]:
    c1, c2, c3 = st.columns([3, 3, 2], gap="large")
    with c1:
        mode_label = st.segmented_control(
            "Color by", ["Composite", "Category", "Indicator", "Chains", "Exposure", "Trade"],
            default="Composite", key="fund_map_mode",
        ) or "Composite"
    mode = {"Indicator": MapMode.INDICATOR, "Category": MapMode.CATEGORY,
            "Composite": MapMode.COMPOSITE, "Chains": MapMode.CHAINS,
            "Exposure": MapMode.EXPOSURE, "Trade": MapMode.TRADE}[mode_label]
    key: str | None = None
    with c2:
        if mode == MapMode.INDICATOR:
            names = list(snap.scored_catalog)
            key = st.selectbox("Indicator", names, format_func=lambda n: snap.catalog[n].label,
                               key="fund_map_indicator")
        elif mode == MapMode.CATEGORY:
            key = st.selectbox("Category", list(snap.categories),
                               format_func=lambda c: snap.category_labels.get(c, c),
                               key="fund_map_category")
        elif mode == MapMode.CHAINS:
            st.caption("How many pressure-chain rules fire per country — judgment encoded as "
                       "rules (tier C), not data.")
        elif mode == MapMode.EXPOSURE:
            st.caption("Who feels the selected country's chains: shaded by how many of its "
                       "fired rules name them as a spillover target.")
        elif mode == MapMode.TRADE:
            st.caption("Share of the selected country's goods trade (exports + imports) with each "
                       "player; arcs join it to its top-5 partners. IMF IMTS, goods only.")
        else:
            st.caption(f"Composite for the {VIEW_LABELS.get(view, view)} view — weighted mean of "
                       "the category scores, renormalised over the categories a country has; "
                       "blank when less than 60 % of the weight is backed by data.")
    with c3:
        europe = st.segmented_control("Europe", ["Members", "EU bloc"], default="Members",
                                      key="fund_eu_bloc") or "Members"
    return mode, key, europe == "EU bloc"


def _render_map(snap: Snapshot, mode: MapMode, key: str | None, bloc: bool, selected: str,
                view: str) -> None:
    layer = map_layer(snap, mode, key, view, selected, bloc)
    st.session_state[_ISO3_MAP_KEY] = iso3_to_player(snap, bloc)
    fig = build_fundamentals_map(layer)
    st.plotly_chart(fig, width="stretch", on_select=_on_map_select,
                    selection_mode=("points",), key=MAP_KEY)
    none_label = ("None fired" if mode in (MapMode.CHAINS, MapMode.EXPOSURE)
                  else "Selected · no data" if mode == MapMode.TRADE else "No data")
    items = list(layer.legend) + [("#d9d4c5", none_label)]
    st.markdown(legend_row(items), unsafe_allow_html=True)
    dq_note = " ◇ = official statistics contested." if layer.dq_points else ""
    st.caption(f"{layer.caption}{dq_note}")


def _render_leaderboard(snap: Snapshot, selected: str, view: str) -> None:
    lb = leaderboard(snap, view)
    st.session_state[_LB_ORDER_KEY] = list(lb["iso2"])
    cat_cols = [snap.category_labels.get(c, c) for c in snap.categories]
    cfg = {
        "iso2": None,
        "Player": st.column_config.TextColumn("Player", width="medium"),
        "Composite": st.column_config.ProgressColumn("Composite", min_value=0, max_value=100, format="%d"),
        **{c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%d") for c in cat_cols},
        "Chains": st.column_config.NumberColumn("Chains", format="%d"),
        "Coverage": st.column_config.ProgressColumn("Coverage", min_value=0, max_value=100, format="%d%%"),
    }
    st.dataframe(
        lb, hide_index=True, column_config=cfg, on_select=_on_lb_select,
        selection_mode="single-row", key=LB_KEY, height=min(60 + 35 * len(lb), 820),
    )
    st.caption("Σ = aggregate (interpolated, never ranked) · † = data-quality flag · "
               "Coverage = tier-weighted share of indicators present. Click a row to select.")


def _render_country_panel(snap: Snapshot, iso2: str, view: str) -> None:
    p = snap.players.loc[snap.players["iso2"] == iso2].iloc[0]
    comp = snap.view_scores[(snap.view_scores["iso2"] == iso2) & (snap.view_scores["view"] == view)]["score"]
    comp_v = None if comp.empty or pd.isna(comp.iloc[0]) else float(comp.iloc[0])
    n = len(snap.ranking_population)
    interp = iso2 not in snap.ranking_population
    st.markdown(
        f'<span class="kicker no-rule">Country card</span>'
        f'<h3 class="section-title">{p["name"]}</h3>',
        unsafe_allow_html=True,
    )
    comp_txt = "—" if comp_v is None else f"{comp_v:.0f}"
    st.markdown(
        f'<div class="pullquote">Composite <span class="num">{comp_txt}</span> / 100 '
        f'({VIEW_LABELS.get(view, view)} view) · currency <span class="num">{p["currency"] or "—"}</span> · '
        f'FX regime <span class="num">{p["fx_regime"]}</span>'
        + (' · <span class="num">sanctioned</span>' if p["sanctioned"] else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if p["dq_flag"] in ("low", "opaque"):
        st.markdown(f"<p class='section-lede'>† Data quality <b>{p['dq_flag']}</b>: "
                    f"{p['dq_note'] or 'official statistics contested.'}</p>",
                    unsafe_allow_html=True)
    cyc = snap.cycles.get(iso2)
    if cyc:
        st.markdown(
            f"<p class='section-lede'>Cycle lens (juxtaposed, never blended): long-term "
            f"<b>{cyc['long_term_label']}</b> ({cyc['long_term_confidence']:.0%}) · short-term "
            f"<b>{cyc['short_term_label']}</b> ({cyc['short_term_confidence']:.0%}).</p>",
            unsafe_allow_html=True,
        )
    st.markdown(confidence_block("Coverage confidence", coverage_confidence(snap, iso2)),
                unsafe_allow_html=True)
    table = country_table(snap, iso2)
    st.markdown(html_dense_table(table, snap.category_labels, n, interp), unsafe_allow_html=True)


def _render_pareto(snap: Snapshot, iso2: str) -> None:
    name = snap.player_name(iso2)
    st.markdown(
        '<span class="kicker">Pareto</span>'
        f'<h3 class="section-title">Where {name}\'s gap concentrates</h3>'
        '<p class="section-lede">Gap to best-in-class per scored item, largest first, with the '
        'cumulative share — the 80 % rule marks how few items explain most of the distance.</p>',
        unsafe_allow_html=True,
    )
    level_label = st.segmented_control("Pareto by", ["Indicator", "Category"], default="Indicator",
                                       key="fund_pareto_level") or "Indicator"
    level = level_label.lower()
    df = pareto_frame(snap, iso2, level=level, top_n=10)
    st.plotly_chart(build_pareto(df), width="stretch", key="fund_pareto")
    st.caption(pareto_caption(df, name, level))


@st.cache_data(show_spinner=False)
def _cached_bubble(dir_str: str, fingerprint: int, x: str, y: str, view: str, bloc: bool,
                   selected: str, log_x: bool):
    snap = _cached_snapshot(dir_str, fingerprint)
    frame = bubble_frame(snap, x, y, BUBBLE_SIZE_INDICATOR, view, bloc)
    fig = build_bubble(frame, snap.catalog[x].label, snap.catalog[y].label, selected, log_x=log_x)
    return fig, forecast_boundary(frame), len(frame)


def _render_bubble(snap: Snapshot, iso2: str, view: str, bloc: bool) -> None:
    st.markdown(
        '<span class="kicker">Trajectories</span>'
        '<h3 class="section-title">Where players have been, and where the IMF thinks they go</h3>'
        '<p class="section-lede">Gapminder-style: one bubble per player per year, sized by GDP, '
        'coloured by the current view\'s composite. Open circles are IMF projections — a forecast, '
        'not a fact. The selected player leaves a trail (dotted over the projection).</p>',
        unsafe_allow_html=True,
    )
    options = bubble_indicator_options(snap)
    if len(options) < 2 or BUBBLE_SIZE_INDICATOR not in snap.catalog:
        st.caption("Not enough history in the snapshot for trajectories yet.")
        return
    c1, c2, c3 = st.columns([3, 3, 2], gap="large")
    x_default = BUBBLE_DEFAULT_X if BUBBLE_DEFAULT_X in options else options[0]
    y_default = BUBBLE_DEFAULT_Y if BUBBLE_DEFAULT_Y in options else options[1]
    with c1:
        x = st.selectbox("Horizontal", options, index=options.index(x_default),
                         format_func=lambda n: snap.catalog[n].label, key="fund_bubble_x")
    with c2:
        y = st.selectbox("Vertical", options, index=options.index(y_default),
                         format_func=lambda n: snap.catalog[n].label, key="fund_bubble_y")
    with c3:
        log_x = st.toggle("Log horizontal axis", value=(x == "gdp_pc_ppp"), key="fund_bubble_log")
    fig, boundary, n_rows = _cached_bubble(str(snapshot_dir()), snapshot_fingerprint(snapshot_dir()),
                                           x, y, view, bloc, iso2, log_x)
    if n_rows == 0:
        st.caption("No overlapping history for that pair.")
        return
    st.plotly_chart(fig, width="stretch", key="fund_bubble", on_select="ignore")
    tail = (f" Projections start in {boundary} (open circles; bubble size held at the last actual GDP)."
            if boundary else " Pick two IMF series (debt, deficit, growth, current account) to see projections.")
    st.caption(f"{n_rows} player-years since 1990. Bubble size = {snap.catalog[BUBBLE_SIZE_INDICATOR].label}."
               f"{tail} Press ▶ to animate; drag the slider to a year.")


def _render_chains(snap: Snapshot, iso2: str) -> None:
    name = snap.player_name(iso2)
    chains = [c for c in snap.chains if c.iso2 == iso2 and c.triggered]
    st.markdown(
        '<span class="kicker">Pressure chains</span>'
        f'<h3 class="section-title">What {name} will be forced to do</h3>'
        '<p class="section-lede">Six rules over the scored panel: binding constraint → forced '
        'option set → who feels it. Judgment encoded as rules (tier C) — thresholds are round '
        'numbers from the sovereign-debt and balance-of-payments literature, not calibrated.</p>',
        unsafe_allow_html=True,
    )
    if not chains:
        st.caption(f"No rule fires for {name} on the current snapshot.")
        return
    players = dict(zip(snap.players["iso2"], snap.players["name"], strict=True))
    for ch in sorted(chains, key=lambda c: -c.severity):
        st.graphviz_chart(chain_to_dot(ch, players, snap.catalog), width="stretch")
        targets = [t for t in ch.targets if t in players and t != iso2]
        if targets:
            key = f"spill_{iso2}_{ch.rule_id}"
            st.pills("Follow the spillover to", targets, format_func=lambda t: players[t],
                     key=key, on_change=_on_spill_select, args=(key,))


def _render_trade(snap: Snapshot, iso2: str) -> None:
    name = snap.player_name(iso2)
    st.markdown(
        '<span class="kicker">Trade partners</span>'
        f'<h3 class="section-title">Who {name} trades with</h3>'
        '<p class="section-lede">Largest goods-trade partners by combined share of exports and imports. '
        'Two directions, two numbers: how much of this country\'s trade a partner is, and how much of '
        'the partner\'s own exports come here (their exposure — the direction the pressure chains use).</p>',
        unsafe_allow_html=True,
    )
    table = trade_partner_table(snap, iso2)
    if table.empty:
        st.caption(trade_caption(snap, iso2))
        return
    st.markdown(html_trade_table(table), unsafe_allow_html=True)
    st.caption(trade_caption(snap, iso2))


# ─── Page ────────────────────────────────────────────────────────────────────


def render_fundamentals_page() -> None:
    st.markdown(
        '<span class="kicker">The map</span>'
        '<h2 class="section-title">World fundamentals</h2>'
        '<p class="section-lede">Twenty-one countries and the euro area, ranked by '
        'percentile on fundamentals in five categories — real stuff, production, '
        'exchange, promises, enforcer. Ranks, not decimals: with 21 players a '
        'quintile is four countries.</p>',
        unsafe_allow_html=True,
    )
    try:
        snap = get_snapshot()
    except SnapshotError as e:
        st.error(f"Snapshot invalid: {e}")
        return
    if snap is None:
        st.info("No fundamentals snapshot yet — run `dalio-fetch-fundamentals` then `dalio-score`.")
        return

    selected: str = st.session_state.get("country", "US")
    if selected not in set(snap.player_codes):
        selected = "US"

    view = _purpose_view(snap)
    mode, key, bloc = _control_strip(snap, view)
    _render_map(snap, mode, key, bloc, selected, view)

    cov = snap.coverage
    st.caption(f"Snapshot as of {snap.as_of.isoformat()} · {cov.get('filled', 0)}/{cov.get('cells', 0)} "
               f"cells filled · {len(snap.catalog)} indicators · {len(snap.players)} players.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<span class="kicker">Leaderboard</span>'
                    '<h3 class="section-title">All players</h3>', unsafe_allow_html=True)
        _render_leaderboard(snap, selected, view)
    with right:
        _render_country_panel(snap, selected, view)

    _render_chains(snap, selected)
    _render_trade(snap, selected)
    _render_bubble(snap, selected, view, bloc)
    _render_pareto(snap, selected)
