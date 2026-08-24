"""Fundamentals page — Streamlit rendering only (thin).

P0: loads the snapshot (cached on file mtime) and shows as-of + coverage.
P1 adds the map, leaderboard and country table; P2 views + Pareto; P3 chains.
"""
from __future__ import annotations

import streamlit as st

from dalio.app.fundamentals.snapshot import (
    Snapshot,
    SnapshotError,
    load_snapshot,
    snapshot_dir,
    snapshot_fingerprint,
)


@st.cache_data(show_spinner=False)
def _cached_snapshot(dir_str: str, fingerprint: int) -> Snapshot:
    """``fingerprint`` (file mtime) is part of the cache key, so a re-export
    invalidates automatically without restarting the app."""
    from pathlib import Path
    return load_snapshot(Path(dir_str))


def get_snapshot() -> Snapshot | None:
    d = snapshot_dir()
    fp = snapshot_fingerprint(d)
    if fp == 0:
        return None
    return _cached_snapshot(str(d), fp)


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

    cov = snap.coverage
    n_players = len(snap.players)
    st.markdown(
        f"""
<div class="pullquote">
Snapshot as of <span class="num">{snap.as_of.isoformat()}</span> ·
<span class="num">{cov.get('filled', 0)}</span> of <span class="num">{cov.get('cells', 0)}</span> cells filled ·
<span class="num">{len(snap.catalog)}</span> indicators ·
<span class="num">{n_players}</span> players.
</div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Map, leaderboard and country table land in slice P1.")
