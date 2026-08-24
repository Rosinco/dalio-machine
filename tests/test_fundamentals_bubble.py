"""Gapminder bubble (slice P4): frame shaping and chart construction."""
import json

import numpy as np

from dalio.app.fundamentals.charts import build_bubble
from dalio.app.fundamentals.snapshot import parse_snapshot
from dalio.app.fundamentals.view_models import (
    bubble_frame,
    bubble_indicator_options,
    bubble_ranges,
    bubble_trail,
    forecast_boundary,
)


def _with_history(raw: dict) -> dict:
    """Add gdp_usd history + a forecast tail on gdp_growth_fwd5 to the synthetic snapshot."""
    raw = json.loads(json.dumps(raw))
    for iso2, c in raw["countries"].items():
        gdp = {"US": 27e12, "SE": 0.6e12, "CN": 18e12, "IN": 3.9e12, "DE": 4.5e12, "EU": 15e12}[iso2]
        c["history"]["gdp_usd"] = [{"year": y, "value": gdp * (1 + 0.02 * (y - 2019)), "is_forecast": False}
                                   for y in (2019, 2020, 2021, 2024)]          # gap 2022–2023
        c["history"]["gdp_growth_fwd5"] = (
            [{"year": y, "value": 2.0, "is_forecast": False} for y in (2019, 2024)]
            + [{"year": y, "value": 1.8, "is_forecast": True} for y in (2026, 2027)]
        )
    raw["indicators"].append({"name": "gdp_usd", "category": "production", "label": "GDP (current US$)",
                              "unit": "US$", "uncertainty": "A", "higher_is_better": True,
                              "description": "size only", "cadence": "A", "sources": ["WORLD_BANK"],
                              "scored": False, "forward": False})
    return raw


def test_options_exclude_unscored_and_history_less(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    opts = bubble_indicator_options(snap)
    assert "gdp_usd" not in opts and "gdp_pc_ppp" in opts and "gdp_growth_fwd5" in opts
    assert "fiscal_balance_pct_gdp" not in opts          # no history seeded
    assert "gdp_usd" not in snap.scored_catalog and "gdp_usd" in snap.catalog


def test_bubble_frame_join_size_ffill_and_forecast(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    f = bubble_frame(snap, "gdp_pc_ppp", "gdp_growth_fwd5")
    # gdp_pc_ppp history: 2019, 2024 — inner join with growth (2019, 2024, 2026f, 2027f) → 2019, 2024
    assert sorted(f["year"].unique()) == [2019, 2024]
    assert set(f["iso2"]) == {"US", "SE", "CN", "IN", "DE"}      # EU off-map in Members mode
    assert not f["is_forecast"].any()
    assert (f["size"] > 0).all()
    us24 = f[(f.iso2 == "US") & (f.year == 2024)].iloc[0]
    assert us24["x"] == 80000 and us24["color"] > 0 and us24["name"] == "United States"
    # a pair where the forecast years survive: growth vs growth
    g = bubble_frame(snap, "gdp_growth_fwd5", "gdp_growth_fwd5")
    assert sorted(g["year"].unique()) == [2019, 2024, 2026, 2027]   # size held forward through projections
    assert g[g.year >= 2026]["is_forecast"].all()
    us = g[g.iso2 == "US"].set_index("year")["size"]
    assert us[2027] == us[2024]                                    # last actual GDP carried forward
    assert forecast_boundary(g) == 2026 and forecast_boundary(f) is None


def test_bubble_frame_bloc_mode(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    f = bubble_frame(snap, "gdp_pc_ppp", "old_age_dependency", bloc=True)
    assert "EU" in set(f["iso2"]) and "DE" not in set(f["iso2"])


def test_bubble_ranges_and_trail(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    f = bubble_frame(snap, "gdp_pc_ppp", "old_age_dependency")
    (x0, x1), (y0, y1) = bubble_ranges(f, log_x=True)
    assert np.isfinite([x0, x1, y0, y1]).all() and x0 < x1 and y0 < y1
    assert 10 ** x0 < f["x"].min() and 10 ** x1 > f["x"].max()
    tr = bubble_trail(f, "IN")
    assert list(tr["year"]) == sorted(tr["year"]) and set(tr["iso2"]) == {"IN"}
    assert bubble_ranges(f.iloc[0:0], True) == ((0.0, 1.0), (0.0, 1.0))


def test_build_bubble_frames_symbols_labels(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    f = bubble_frame(snap, "gdp_growth_fwd5", "gdp_growth_fwd5")
    fig = build_bubble(f, "growth", "growth", selected_iso2="US", log_x=False)
    assert len(fig.frames) == 4
    labels = [s.label for s in fig.layout.sliders[0].steps]
    assert labels == ["2019", "2024", "2026f", "2027f"]
    kinds = {t.name for t in fig.data}
    assert "forecast" in kinds or "actual" in kinds
    # trail traces appended after px's own traces
    names = [t.name for t in fig.data]
    assert "trail" in names and "trail-forecast" in names
    assert fig.layout.showlegend is False
    empty = build_bubble(f.iloc[0:0], "x", "y", None)
    assert len(empty.data) == 0


def test_build_bubble_log_axis_range(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    f = bubble_frame(snap, "gdp_pc_ppp", "old_age_dependency")
    fig = build_bubble(f, "gdp", "dep", selected_iso2="SE", log_x=True)
    assert fig.layout.xaxis.type == "log"
    lo, hi = fig.layout.xaxis.range
    assert 10 ** lo < f["x"].min() <= f["x"].max() < 10 ** hi


def test_bubble_frame_unknown_indicator_is_empty(synthetic_snapshot_dict):
    snap = parse_snapshot(_with_history(synthetic_snapshot_dict))
    assert bubble_frame(snap, "nope", "gdp_pc_ppp").empty
