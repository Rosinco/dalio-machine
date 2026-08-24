"""Fundamentals page — Streamlit rendering only (thin).

Layout: control strip → map → leaderboard + country panel (two columns).
P2 adds the purpose-view selector, weights, captions and the Pareto; P3 the
pressure chains; P4 the bubble.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from dalio.app.fundamentals.charts import build_fundamentals_map
from dalio.app.fundamentals.snapshot import (
    Snapshot,
    SnapshotError,
    load_snapshot,
    snapshot_dir,
    snapshot_fingerprint,
)
from dalio.app.fundamentals.view_models import (
    MapMode,
    country_table,
    coverage_confidence,
    html_dense_table,
    iso3_to_player,
    leaderboard,
    map_layer,
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


# ─── Sections ────────────────────────────────────────────────────────────────


def _control_strip(snap: Snapshot) -> tuple[MapMode, str | None, bool]:
    c1, c2, c3 = st.columns([2, 3, 2], gap="large")
    with c1:
        mode_label = st.segmented_control(
            "Color by", ["Indicator", "Category", "Composite"], default="Indicator",
            key="fund_map_mode",
        ) or "Indicator"
    mode = {"Indicator": MapMode.INDICATOR, "Category": MapMode.CATEGORY,
            "Composite": MapMode.COMPOSITE}[mode_label]
    key: str | None = None
    with c2:
        if mode == MapMode.INDICATOR:
            names = list(snap.catalog)
            key = st.selectbox("Indicator", names, format_func=lambda n: snap.catalog[n].label,
                               key="fund_map_indicator")
        elif mode == MapMode.CATEGORY:
            key = st.selectbox("Category", list(snap.categories),
                               format_func=lambda c: snap.category_labels.get(c, c),
                               key="fund_map_category")
        else:
            st.caption(f"Composite = equal-weight mean of category scores ({DEFAULT_VIEW} view). "
                       "Purpose views arrive in P2.")
    with c3:
        europe = st.segmented_control("Europe", ["Members", "EU bloc"], default="Members",
                                      key="fund_eu_bloc") or "Members"
    return mode, key, europe == "EU bloc"


def _render_map(snap: Snapshot, mode: MapMode, key: str | None, bloc: bool, selected: str) -> None:
    layer = map_layer(snap, mode, key, DEFAULT_VIEW, selected, bloc)
    st.session_state[_ISO3_MAP_KEY] = iso3_to_player(snap, bloc)
    fig = build_fundamentals_map(layer)
    st.plotly_chart(fig, width="stretch", on_select=_on_map_select,
                    selection_mode=("points",), key=MAP_KEY)
    items = list(layer.legend) + [("#d9d4c5", "No data")]
    st.markdown(legend_row(items), unsafe_allow_html=True)
    dq_note = " ◇ = official statistics contested." if layer.dq_points else ""
    st.caption(f"{layer.caption}{dq_note}")


def _render_leaderboard(snap: Snapshot, selected: str) -> None:
    lb = leaderboard(snap, DEFAULT_VIEW)
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


def _render_country_panel(snap: Snapshot, iso2: str) -> None:
    p = snap.players.loc[snap.players["iso2"] == iso2].iloc[0]
    comp = snap.view_scores.query("iso2 == @iso2 and view == @DEFAULT_VIEW")["score"]
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
        f'({DEFAULT_VIEW} view) · currency <span class="num">{p["currency"] or "—"}</span> · '
        f'FX regime <span class="num">{p["fx_regime"]}</span>'
        + (' · <span class="num">sanctioned</span>' if p["sanctioned"] else "")
        + "</div>",
        unsafe_allow_html=True,
    )
    if p["dq_flag"] in ("low", "opaque"):
        st.markdown(f"<p class='section-lede'>† Data quality <b>{p['dq_flag']}</b>: "
                    f"{p['dq_note'] or 'official statistics contested.'}</p>",
                    unsafe_allow_html=True)
    st.markdown(confidence_block("Coverage confidence", coverage_confidence(snap, iso2)),
                unsafe_allow_html=True)
    table = country_table(snap, iso2)
    st.markdown(html_dense_table(table, snap.category_labels, n, interp), unsafe_allow_html=True)


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

    mode, key, bloc = _control_strip(snap)
    _render_map(snap, mode, key, bloc, selected)

    cov = snap.coverage
    st.caption(f"Snapshot as of {snap.as_of.isoformat()} · {cov.get('filled', 0)}/{cov.get('cells', 0)} "
               f"cells filled · {len(snap.catalog)} indicators · {len(snap.players)} players.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        st.markdown('<span class="kicker">Leaderboard</span>'
                    '<h3 class="section-title">All players</h3>', unsafe_allow_html=True)
        _render_leaderboard(snap, selected)
    with right:
        _render_country_panel(snap, selected)
