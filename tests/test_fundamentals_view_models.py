"""View-model tests on the synthetic snapshot (slice P1)."""
import numpy as np
import pandas as pd

from dalio.app.fundamentals.snapshot import parse_snapshot
from dalio.app.fundamentals.view_models import (
    MapMode,
    bin_quintile,
    country_table,
    coverage_confidence,
    exposure_counts,
    fmt_value,
    html_dense_table,
    iso3_to_player,
    leaderboard,
    map_layer,
    metric_series,
    player_locations,
    rank_label,
)


def _snap(synthetic_snapshot_dict):
    return parse_snapshot(synthetic_snapshot_dict)


def test_bin_quintile_edges():
    s = pd.Series([0.0, 19.9, 20.0, 59.0, 80.0, 100.0, np.nan])
    out = bin_quintile(s)
    assert out.tolist()[:6] == [0, 0, 1, 2, 4, 4]
    assert np.isnan(out.iloc[6])


def test_rank_label_and_fmt():
    assert rank_label(100.0, 21) == "1/21"
    assert rank_label(0.0, 21) == "21/21"
    assert rank_label(50.0, 21) == "11/21"
    assert rank_label(62.0, 21, interpolated=True).startswith("≈")
    assert rank_label(None, 21) == "—"
    assert fmt_value(76931.2) == "76,931" and fmt_value(3.42) == "3.4" and fmt_value(None) == "—"
    assert fmt_value(float("nan")) == "—"


def test_iso3_to_player_members_vs_bloc(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    m = iso3_to_player(snap, bloc=False)
    assert m["DEU"] == "DE"            # own player wins
    assert m["AUT"] == "EU"            # covered member → aggregate
    assert m["USA"] == "US"
    b = iso3_to_player(snap, bloc=True)
    assert b["DEU"] == "EU" and b["AUT"] == "EU" and b["USA"] == "US"


def test_player_locations(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    assert player_locations(snap, "DE") == ("DEU",)
    members = player_locations(snap, "EU", bloc=False)
    assert "DEU" not in members and "AUT" in members and len(members) == 19
    assert len(player_locations(snap, "EU", bloc=True)) == 20


def test_metric_series_modes(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    s, label = metric_series(snap, MapMode.INDICATOR, "gdp_pc_ppp", "learning")
    assert s["US"] == 100.0 and label == "GDP per capita (PPP)"
    s, label = metric_series(snap, MapMode.CATEGORY, "production", "learning")
    assert s["US"] == 100.0 and label == "Production"
    s, _ = metric_series(snap, MapMode.COMPOSITE, None, "learning")
    assert 0 < s["US"] <= 100
    s, _ = metric_series(snap, MapMode.CHAINS, None, "learning")
    assert s["US"] == 1.0 and s["SE"] == 0.0


def test_map_layer_indicator_mode(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    layer = map_layer(snap, MapMode.INDICATOR, "military_share_world", "learning", selected_iso2="EU")
    # EU has no military value → its 19 covered members are no-data; selected still last among scored? EU not scored
    assert set(layer.no_data_locations) >= {"AUT", "PRT"}
    assert "DEU" not in layer.no_data_locations           # Germany scored on its own
    assert len(layer.locations) == 5 and len(layer.z_bin) == 5
    assert set(layer.opacity) == {1.0}
    # CN is dq-flagged → one diamond marker
    assert any("China" in p[2] for p in layer.dq_points)
    assert "rank" in layer.hover[0] and "tier A" in layer.hover[0]
    assert layer.ramp and len(layer.legend) == 5
    assert "quintile" in layer.caption.lower()


def test_map_layer_selected_last_and_aggregate_opacity(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    layer = map_layer(snap, MapMode.INDICATOR, "gdp_pc_ppp", "learning", selected_iso2="SE")
    assert layer.locations[-1] == "SWE" and layer.selected[-1] is True
    assert sum(layer.selected) == 1
    # EU aggregate covers 19 members at reduced opacity (Members mode)
    covered = [o for loc, o in zip(layer.locations, layer.opacity, strict=True) if loc == "AUT"]
    assert covered == [0.45]
    assert "≈" in [h for loc, h in zip(layer.locations, layer.hover, strict=True) if loc == "AUT"][0]


def test_map_layer_bloc_mode_drops_members(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    layer = map_layer(snap, MapMode.INDICATOR, "gdp_pc_ppp", "learning", selected_iso2="US", bloc=True)
    assert "DEU" in layer.locations
    de_hover = [h for loc, h in zip(layer.locations, layer.hover, strict=True) if loc == "DEU"][0]
    assert "Eurozone" in de_hover                          # shaded by the aggregate
    assert set(layer.opacity) == {1.0}
    assert len(layer.locations) == 4 + 20                  # US SE CN IN + 20 members


def test_map_layer_exposure_and_chains(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    assert exposure_counts(snap, "US") == {"CN": 1, "SA": 1}
    layer = map_layer(snap, MapMode.EXPOSURE, None, "learning", selected_iso2="US")
    assert layer.locations == ("CHN",)                     # SA is not a synthetic player
    assert layer.z_bin == (0.0,)
    layer = map_layer(snap, MapMode.CHAINS, None, "learning", selected_iso2=None)
    assert layer.locations == ("USA",) and layer.z_bin == (0.0,)
    assert "SWE" in layer.no_data_locations


def test_leaderboard_sorted_with_markers(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    lb = leaderboard(snap, "learning")
    assert list(lb.columns) == ["iso2", "Player", "Composite", "Real stuff", "Production",
                                "Exchange", "Promises", "Enforcer", "Chains", "Coverage"]
    comps = lb["Composite"].dropna().tolist()
    assert comps == sorted(comps, reverse=True)
    assert lb.iloc[-1]["iso2"] == "EU" or not np.isnan(lb.iloc[-1]["Composite"])
    eu = lb[lb.iso2 == "EU"].iloc[0]
    assert eu["Player"].startswith("Eurozone (Σ)")
    cn = lb[lb.iso2 == "CN"].iloc[0]
    assert cn["Player"].endswith("†")
    assert lb[lb.iso2 == "US"]["Chains"].iloc[0] == 1
    assert lb["Coverage"].min() >= 0 and lb["Coverage"].max() <= 100


def test_coverage_confidence(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    # US present: gdp B, dep A, rd A, mil A, rol C → mean(.6,1,1,1,.3) × 5/15
    us = coverage_confidence(snap, "US")
    assert abs(us - ((0.6 + 1 + 1 + 1 + 0.3) / 5) * (5 / 16)) < 1e-9
    eu = coverage_confidence(snap, "EU")       # gdp B + dep A → ×2/15
    assert abs(eu - ((0.6 + 1) / 2) * (2 / 16)) < 1e-9


def test_country_table_order_and_gaps(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    t = country_table(snap, "EU")
    assert len(t) == 16
    assert list(dict.fromkeys(t["category"])) == ["real_stuff", "production", "exchange", "promises", "enforcer"]
    mil = t[t.indicator == "military_share_world"].iloc[0]
    assert mil["value"] is None or pd.isna(mil["value"])
    assert mil["tier"] == "A"
    rol = country_table(snap, "US").set_index("indicator").loc["rule_of_law"]
    assert rol["se"] == 0.15


def test_html_dense_table_escapes_and_marks(synthetic_snapshot_dict):
    snap = _snap(synthetic_snapshot_dict)
    t = country_table(snap, "US")
    t.loc[0, "label"] = "<b>evil</b>"
    html_out = html_dense_table(t, snap.category_labels, 5)
    assert "&lt;b&gt;evil&lt;/b&gt;" in html_out and "<b>evil</b>" not in html_out
    assert html_out.count('<tr class="cat">') == 5
    assert 'class="tier tier-b"' in html_out and 'class="tier tier-a"' in html_out
    assert 'class="tier tier-c"' in html_out
    assert "±0.15" in html_out
    assert "1/5" in html_out                      # US gdp best of 5
    assert "▲" in html_out or "▼" in html_out
