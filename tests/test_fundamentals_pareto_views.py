"""Purpose views + Pareto (slice P2)."""
import pandas as pd
import pytest

from dalio.app.fundamentals.charts import build_pareto
from dalio.app.fundamentals.snapshot import parse_snapshot
from dalio.app.fundamentals.view_models import (
    VIEW_CAPTIONS,
    VIEW_LABELS,
    leaderboard,
    pareto_caption,
    pareto_frame,
    view_caption,
    weights_line,
)
from dalio.scoring.fundamentals import VIEWS


def test_every_view_has_label_and_caption():
    assert set(VIEW_CAPTIONS) == set(VIEWS) == set(VIEW_LABELS)
    for v in VIEWS:
        assert "cannot" in view_caption(v)    # every caption states a limit


def test_weights_line(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    assert weights_line(snap, "jurisdiction") == "Jurisdiction view · Enforcer 50 · Promises 30 · Exchange 20"
    assert weights_line(snap, "learning").startswith("Learning view · ")


def test_leaderboard_reorders_by_view(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    learn = leaderboard(snap, "learning")
    moon = leaderboard(snap, "moonshot")
    assert set(learn["iso2"]) == set(moon["iso2"])
    # moonshot = real_stuff .5 / exchange .3 / production .2 → only real_stuff + production
    # scored → 0.7 covered → renormalised; IN (young, poor) should beat DE (old, rich) on it
    m = moon.set_index("iso2")["Composite"]
    assert m["IN"] > m["DE"]


def test_pareto_indicator_level(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    df = pareto_frame(snap, "IN", level="indicator")
    assert list(df.columns) == ["key", "gap", "share", "cum_share", "crosses_80"]
    assert df["gap"].is_monotonic_decreasing
    assert df["share"].sum() == pytest.approx(1.0)              # ≤ top_n covers everything here
    assert df["cum_share"].iloc[-1] == pytest.approx(1.0)
    assert df["crosses_80"].sum() == 1
    first80 = df.index[df["crosses_80"]][0]
    assert df.loc[first80, "cum_share"] >= 0.8
    assert (df.loc[:first80 - 1, "cum_share"] < 0.8).all() if first80 > 0 else True
    # India's largest gap is GDP per capita (pct 0 → gap 100)
    assert df.iloc[0]["key"] == "GDP per capita (PPP)" and df.iloc[0]["gap"] == 100.0
    cap = pareto_caption(df, "India", "indicator")
    assert "80 %" in cap and "GDP per capita" in cap


def test_pareto_category_level_and_best_has_no_gap(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    df = pareto_frame(snap, "SE", level="category")
    assert set(df["key"]) <= {"Real stuff", "Production", "Enforcer"}     # only scored categories
    assert (df["gap"] >= 0).all()
    # US is best on production; its production gap must be 0 (may be excluded if all zero)
    us = pareto_frame(snap, "US", level="category")
    assert "Production" not in set(us[us["gap"] > 0]["key"]) or us.set_index("key").loc["Production", "gap"] == 0


def test_build_pareto_empty_frame_has_no_traces():
    empty = pd.DataFrame(columns=["key", "gap", "share", "cum_share", "crosses_80"])
    fig = build_pareto(empty)
    assert len(fig.data) == 0


def test_build_pareto_shapes(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    df = pareto_frame(snap, "CN", level="indicator", top_n=3)
    assert len(df) == 3
    fig = build_pareto(df)
    assert len(fig.data) == 1 and fig.data[0].type == "bar" and fig.data[0].orientation == "h"
    assert list(fig.data[0].y)[::-1] == list(df["key"])            # reversed for bottom-up drawing
    assert any(s.type == "line" for s in fig.layout.shapes)        # the 80 % rule
    assert fig.layout.xaxis.range[0] == 0
