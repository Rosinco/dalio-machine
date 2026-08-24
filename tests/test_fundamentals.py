"""Tests for fundamentals scoring + snapshot (slice 18)."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from dalio.countries import COUNTRIES, RANKING_POPULATION, get_country
from dalio.scoring.fundamentals import (
    CATEGORIES,
    FUNDAMENTALS,
    VIEWS,
    CategoryScore,
    IndicatorSpec,
    build_snapshot,
    category_scores,
    load_panel,
    percentile_rank,
    read_snapshot,
    trend_direction,
    view_scores,
    write_snapshot,
)
from dalio.storage.db import Observation, init_db, make_engine, make_session_factory

# ─── Registry sanity ───────────────────────────────────────────────────────


def test_every_spec_has_known_category_and_tier():
    for s in FUNDAMENTALS:
        assert s.category in CATEGORIES
        assert s.uncertainty in ("A", "B", "C")
        assert s.preferred_sources
        assert len(s.description) > 20


def test_views_weights_sum_to_one_and_use_known_categories():
    for name, w in VIEWS.items():
        assert abs(sum(w.values()) - 1.0) < 1e-9, name
        assert set(w) <= set(CATEGORIES)


# ─── percentile_rank ───────────────────────────────────────────────────────


def test_percentile_rank_higher_is_better_spans_0_to_100():
    pop = ["A", "B", "C", "D", "E"]
    v = pd.Series([10, 20, 30, 40, 50], index=pop, dtype=float)
    p = percentile_rank(v, True, pop)
    assert p.tolist() == [0.0, 25.0, 50.0, 75.0, 100.0]


def test_percentile_rank_inverted_direction():
    pop = ["A", "B", "C"]
    v = pd.Series([10, 20, 30], index=pop, dtype=float)
    p = percentile_rank(v, False, pop)
    assert p.tolist() == [100.0, 50.0, 0.0]


def test_percentile_rank_nan_propagates_and_does_not_shift_others():
    pop = ["A", "B", "C", "D"]
    v = pd.Series([10, np.nan, 30, 40], index=pop, dtype=float)
    p = percentile_rank(v, True, pop)
    assert np.isnan(p["B"])
    assert p[["A", "C", "D"]].tolist() == [0.0, 50.0, 100.0]


def test_percentile_rank_ties_average():
    pop = ["A", "B", "C"]
    v = pd.Series([10, 10, 30], index=pop, dtype=float)
    p = percentile_rank(v, True, pop)
    assert p["A"] == p["B"] == pytest.approx(25.0)
    assert p["C"] == 100.0


def test_non_population_member_is_interpolated_not_ranked():
    pop = ["A", "B", "C"]
    v = pd.Series([10, 20, 30, 25], index=[*pop, "EU"], dtype=float)
    p = percentile_rank(v, True, pop)
    assert p[["A", "B", "C"]].tolist() == [0.0, 50.0, 100.0]   # unchanged by EU
    assert p["EU"] == pytest.approx(75.0)


def test_non_population_member_interpolated_when_lower_is_better():
    pop = ["A", "B", "C"]
    v = pd.Series([10, 20, 30, 15], index=[*pop, "EU"], dtype=float)
    p = percentile_rank(v, False, pop)
    assert p[["A", "B", "C"]].tolist() == [100.0, 50.0, 0.0]
    assert p["EU"] == pytest.approx(75.0)


def test_percentile_rank_too_few_values_is_nan():
    v = pd.Series([1.0], index=["A"])
    assert percentile_rank(v, True, ["A", "B"]).isna().all()


# ─── trend_direction ───────────────────────────────────────────────────────


def test_trend_dead_band():
    assert trend_direction(10.0, 9.0, True, cross_std=100.0) == "flat"
    assert trend_direction(10.0, 9.0, True, cross_std=5.0) == "improving"
    assert trend_direction(9.0, 10.0, True, cross_std=5.0) == "worsening"
    assert trend_direction(9.0, 10.0, False, cross_std=5.0) == "improving"
    assert trend_direction(None, 10.0, True, cross_std=5.0) is None
    assert trend_direction(10.0, 9.0, True, cross_std=None) is None


# ─── category_scores / view_scores ─────────────────────────────────────────


def _specs():
    return (
        IndicatorSpec("a1", "real_stuff", "a1", "", True, "A", ("T",), "x" * 30),
        IndicatorSpec("a2", "real_stuff", "a2", "", True, "A", ("T",), "x" * 30),
        IndicatorSpec("b1", "production", "b1", "", True, "A", ("T",), "x" * 30),
    )


def test_category_score_is_mean_with_coverage_floor():
    pct = pd.DataFrame(
        {"a1": [80.0, 20.0, np.nan], "a2": [60.0, np.nan, np.nan], "b1": [50.0, 10.0, 90.0]},
        index=["X", "Y", "Z"],
    )
    cs = category_scores(pct, _specs(), population=["X", "Y", "Z"])
    assert cs["X"]["real_stuff"].score == pytest.approx(70.0)
    assert cs["X"]["real_stuff"].n_available == 2
    assert cs["X"]["real_stuff"].n_total == 2
    # Y has 1 of 2 → exactly half → allowed
    assert cs["Y"]["real_stuff"].score == pytest.approx(20.0)
    # Z has 0 of 2 → None
    assert cs["Z"]["real_stuff"].score is None
    assert cs["Z"]["real_stuff"].distance_to_best is None
    # best-in-class + distance
    assert cs["Y"]["real_stuff"].best_iso2 == "X"
    assert cs["Y"]["real_stuff"].distance_to_best == pytest.approx(50.0)
    assert cs["Z"]["production"].best_iso2 == "Z"
    assert cs["Z"]["production"].distance_to_best == pytest.approx(0.0)
    # categories with no indicators at all still appear, empty
    assert cs["X"]["promises"] == CategoryScore(None, 0, 0, None, None)


def test_best_in_class_excludes_non_population():
    pct = pd.DataFrame({"a1": [50.0, 99.0], "a2": [50.0, 99.0], "b1": [1.0, 1.0]}, index=["X", "EU"])
    cs = category_scores(pct, _specs(), population=["X"])
    assert cs["EU"]["real_stuff"].best_iso2 == "X"


def test_view_scores_renormalise_and_floor():
    cats = {
        "X": {
            "real_stuff": CategoryScore(80.0, 2, 2, 0.0, "X"),
            "production": CategoryScore(40.0, 1, 1, 0.0, "X"),
            "exchange": CategoryScore(None, 0, 0, None, None),
            "promises": CategoryScore(None, 0, 0, None, None),
            "enforcer": CategoryScore(None, 0, 0, None, None),
        }
    }
    v = view_scores(cats, {"m": {"real_stuff": 0.5, "exchange": 0.3, "production": 0.2},
                           "j": {"enforcer": 0.5, "promises": 0.3, "exchange": 0.2},
                           "l": {c: 0.2 for c in CATEGORIES}})
    # moonshot: 0.7 of weight covered → renormalised (0.5*80 + 0.2*40)/0.7
    assert v["X"]["m"] == pytest.approx((0.5 * 80 + 0.2 * 40) / 0.7)
    # jurisdiction: nothing covered → None
    assert v["X"]["j"] is None
    # learning: 0.4 covered < 0.6 → None
    assert v["X"]["l"] is None


# ─── DB-backed: load_panel + build_snapshot ────────────────────────────────


@pytest.fixture
def session_factory(tmp_path):
    engine = make_engine(tmp_path / "t.db")
    init_db(engine)
    return make_session_factory(engine)


def _seed(session, country, indicator, year_values, source="WORLD_BANK"):
    for year, v in year_values:
        session.add(Observation(
            country=country, indicator=indicator, date=date(year, 12, 31),
            value=v, source=source, series_id="X",
        ))


def test_load_panel_latest_at_or_before_as_of_with_source_preference(session_factory):
    spec = IndicatorSpec("gdp_pc_ppp", "production", "x", "", True, "B", ("WORLD_BANK", "OTHER"), "x" * 30)
    with session_factory() as s:
        _seed(s, "US", "gdp_pc_ppp", [(2020, 1.0), (2024, 2.0), (2027, 3.0)])
        _seed(s, "US", "gdp_pc_ppp", [(2025, 99.0)], source="OTHER")   # newer but less preferred
        _seed(s, "SE", "gdp_pc_ppp", [(2019, 5.0)], source="OTHER")    # only source available
        _seed(s, "KR", "gdp_pc_ppp", [(2024, 7.0)], source="UNLISTED")  # ignored source
        s.commit()
    with session_factory() as s:
        panel = load_panel(s, [spec], [get_country("US"), get_country("SE"), get_country("KR")],
                           as_of=date(2026, 8, 24))
        lagged = load_panel(s, [spec], [get_country("US")], as_of=date(2026, 8, 24), lag_years=5)
    p = panel.set_index("country")
    assert p.loc["US", "value"] == 2.0 and p.loc["US", "source"] == "WORLD_BANK"
    assert p.loc["SE", "value"] == 5.0 and p.loc["SE", "source"] == "OTHER"
    assert "KR" not in p.index
    assert lagged.set_index("country").loc["US", "value"] == 1.0


def _seed_tracer(session):
    """Three indicators for a handful of countries; EU as aggregate; RU missing."""
    gdp = {"US": 80000, "SE": 60000, "CN": 24000, "IN": 10000, "DE": 65000, "EU": 58000}
    dep = {"US": 27, "SE": 33, "CN": 21, "IN": 10, "DE": 36, "EU": 34}
    mil = {"US": 3.3, "SE": 2.0, "CN": 1.7, "IN": 2.4, "DE": 1.5}
    for iso2, v in gdp.items():
        _seed(session, iso2, "gdp_pc_ppp", [(2019, v * 0.9), (2024, v)])
    for iso2, v in dep.items():
        _seed(session, iso2, "old_age_dependency", [(2019, v - 2), (2025, v)])
    for iso2, v in mil.items():
        _seed(session, iso2, "military_pct_gdp", [(2024, v)])


def test_build_snapshot_schema_and_scores(session_factory, tmp_path):
    with session_factory() as s:
        _seed_tracer(s)
        s.commit()
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 8, 24))

    assert snap["version"] == 1
    assert snap["as_of"] == "2026-08-24"
    assert snap["ranking_population"] == list(RANKING_POPULATION)
    assert len(snap["ranking_population"]) == 21
    assert [i["name"] for i in snap["indicators"]] == [s.name for s in FUNDAMENTALS]
    assert snap["categories"] == list(CATEGORIES)
    assert set(snap["views"]) == set(VIEWS)
    assert set(snap["countries"]) == {c.iso2 for c in COUNTRIES}

    us = snap["countries"]["US"]
    assert us["indicators"]["gdp_pc_ppp"]["pct"] == 100.0           # best of the 5 population values
    assert us["indicators"]["gdp_pc_ppp"]["date"] == "2024-12-31"
    assert us["indicators"]["gdp_pc_ppp"]["source"] == "WORLD_BANK"
    assert us["indicators"]["gdp_pc_ppp"]["uncertainty"] == "B"
    assert us["indicators"]["gdp_pc_ppp"]["trend"] in ("improving", "flat")
    assert us["indicators"]["gdp_pc_ppp"]["lag_value"] == pytest.approx(72000.0)
    assert us["indicators"]["old_age_dependency"]["trend"] == "worsening"   # rose, lower is better
    assert us["categories"]["production"]["score"] == 100.0
    assert us["categories"]["production"]["n_available"] == 1
    assert us["categories"]["production"]["best_iso2"] == "US"
    assert us["categories"]["exchange"]["score"] is None
    assert us["data_quality"]["flag"] == "high"
    assert us["fx_regime"] == "reserve_issuer"
    assert us["pressures"] == [] and us["cycle"] is None

    # dependency: lower is better → IN (10) best = 100, DE (36) worst = 0
    assert snap["countries"]["IN"]["indicators"]["old_age_dependency"]["pct"] == 100.0
    assert snap["countries"]["DE"]["indicators"]["old_age_dependency"]["pct"] == 0.0

    # EU aggregate interpolated, off-map, members listed, never best-in-class
    eu = snap["countries"]["EU"]
    assert eu["on_map"] is False and len(eu["members"]) == 20
    assert 0.0 < eu["indicators"]["gdp_pc_ppp"]["pct"] < 100.0
    assert eu["categories"]["production"]["best_iso2"] == "US"

    # RU has nothing → empty cells, None scores, opaque flag
    ru = snap["countries"]["RU"]
    assert ru["indicators"]["gdp_pc_ppp"]["value"] is None
    assert ru["categories"]["production"]["score"] is None
    assert ru["data_quality"]["flag"] == "opaque" and ru["sanctioned"] is True

    # coverage counts
    assert snap["coverage"]["cells"] == 22 * 3
    assert snap["coverage"]["filled"] == 6 + 6 + 5
    assert snap["coverage"]["by_indicator"]["military_pct_gdp"] == 5

    # history block
    assert [h["year"] for h in us["history"]["gdp_pc_ppp"]] == [2019, 2024]
    assert us["history"]["gdp_pc_ppp"][0]["is_forecast"] is False

    # round trip
    path = write_snapshot(snap, tmp_path / "snap" / "fundamentals_latest.json")
    assert read_snapshot(path) == snap


def test_learning_view_needs_60pct_of_weight(session_factory):
    """With exactly 3 of 5 equal-weight categories scored, coverage = 0.6 → allowed."""
    with session_factory() as s:
        _seed_tracer(s)
        s.commit()
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 8, 24))
    us = snap["countries"]["US"]
    assert us["views"]["learning"] is not None
    assert us["views"]["learning"] == pytest.approx(
        (us["categories"]["real_stuff"]["score"] + us["categories"]["production"]["score"]
         + us["categories"]["enforcer"]["score"]) / 3
    )
    assert us["views"]["jurisdiction"] is None   # only enforcer (0.5) covered < 0.6


# ─── Review-pass regressions (slice 18) ────────────────────────────────────


def test_as_of_cutoff_year_end_keeps_exact_five_year_lag():
    from dalio.scoring.fundamentals import _as_of_cutoff
    assert _as_of_cutoff(date(2026, 12, 31), 5) == date(2021, 12, 31)
    assert _as_of_cutoff(date(2026, 8, 24), 5) == date(2021, 8, 24)
    assert _as_of_cutoff(date(2024, 2, 29), 1) == date(2023, 2, 28)
    assert _as_of_cutoff(date(2026, 8, 24), 0) == date(2026, 8, 24)


def test_year_end_as_of_lag_picks_row_five_years_back(session_factory):
    with session_factory() as s:
        _seed(s, "US", "gdp_pc_ppp", [(y, 80.0 + (y - 2015)) for y in range(2015, 2027)])
        s.commit()
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 12, 31), include_history=False)
    cell = snap["countries"]["US"]["indicators"]["gdp_pc_ppp"]
    assert cell["value"] == 91.0
    assert cell["lag_value"] == 86.0          # 2021 row, not 2020


def test_stale_single_point_series_has_no_trend(session_factory):
    with session_factory() as s:
        _seed(s, "SA", "military_pct_gdp", [(2019, 8.0)])
        for iso2, v in (("US", 3.3), ("SE", 2.0), ("CN", 1.7), ("IN", 2.4), ("DE", 1.5)):
            _seed(s, iso2, "military_pct_gdp", [(2019, v - 1.0), (2024, v)])
        s.commit()
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 8, 24), include_history=False)
    sa = snap["countries"]["SA"]["indicators"]["military_pct_gdp"]
    assert sa["value"] == 8.0 and sa["date"] == "2019-12-31"
    assert sa["lag_value"] is None
    assert sa["trend"] is None and sa["trend_5y"] is None
    us = snap["countries"]["US"]["indicators"]["military_pct_gdp"]
    assert us["trend"] == "improving"


def test_load_panel_never_borrows_another_specs_source(session_factory):
    a = IndicatorSpec("gdp_pc_ppp", "production", "a", "", True, "B", ("WORLD_BANK",), "x" * 30)
    b = IndicatorSpec("gov_debt_pct_gdp", "promises", "b", "", False, "B", ("IMF_WEO",), "x" * 30)
    with session_factory() as s:
        _seed(s, "US", "gdp_pc_ppp", [(2024, 999.0)], source="IMF_WEO")      # wrong source for a
        _seed(s, "US", "gov_debt_pct_gdp", [(2024, 120.0)], source="IMF_WEO")
        s.commit()
    with session_factory() as s:
        panel = load_panel(s, [a, b], [get_country("US")], as_of=date(2026, 8, 24))
        from dalio.scoring.fundamentals import load_history
        hist = load_history(s, [a, b], [get_country("US")])
    assert set(panel["indicator"]) == {"gov_debt_pct_gdp"}
    assert set(hist["indicator"]) == {"gov_debt_pct_gdp"}


def test_aggregate_distance_to_best_never_negative():
    specs = (
        IndicatorSpec("a1", "real_stuff", "a1", "", True, "A", ("T",), "x" * 30),
        IndicatorSpec("a2", "real_stuff", "a2", "", True, "A", ("T",), "x" * 30),
    )
    pct = pd.DataFrame({"a1": [100.0, 0.0, 90.0], "a2": [0.0, 100.0, 90.0]}, index=["X", "Y", "EU"])
    cs = category_scores(pct, specs, population=["X", "Y"])
    assert cs["EU"]["real_stuff"].score == 90.0
    assert cs["EU"]["real_stuff"].distance_to_best == 0.0
    assert cs["EU"]["real_stuff"].best_iso2 in ("X", "Y")


def test_build_snapshot_on_empty_db(session_factory):
    with session_factory() as s:
        snap = build_snapshot(s, as_of=date(2026, 8, 24))
    assert snap["coverage"]["filled"] == 0
    assert all(c["categories"]["production"]["score"] is None for c in snap["countries"].values())
