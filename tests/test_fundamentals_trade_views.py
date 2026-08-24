"""Trade map mode, arcs, partner table (slice 24) on the synthetic snapshot."""
import pytest

from dalio.app.fundamentals.charts import build_fundamentals_map
from dalio.app.fundamentals.snapshot import parse_snapshot
from dalio.app.fundamentals.view_models import (
    TRADE_ARCS,
    MapMode,
    html_trade_table,
    map_layer,
    trade_arcs,
    trade_caption,
    trade_partner_table,
    trade_share_series,
)

US_X_TOT, US_M_TOT = 2185.2, 3300.0


@pytest.fixture
def snap(synthetic_snapshot_dict):
    return parse_snapshot(synthetic_snapshot_dict)


def test_trade_share_series_uses_both_totals(snap):
    s = trade_share_series(snap, "US")
    assert list(s.index) == ["EU", "CN", "DE"]                                  # sorted desc
    assert s["EU"] == pytest.approx((376.8 + 600.0) / (US_X_TOT + US_M_TOT) * 100)
    assert s["CN"] == pytest.approx((143.5 + 438.9) / (US_X_TOT + US_M_TOT) * 100)
    assert trade_share_series(snap, "SE").empty                                # no trade rows for SE


def test_map_layer_trade_bins_hover_and_selected(snap):
    layer = map_layer(snap, MapMode.TRADE, None, "learning", "US", bloc=False)
    z = dict(zip(layer.locations, layer.z_bin, strict=True))
    assert z["CHN"] == 3.0                    # 10.6 % → ≥ 10 bin
    assert z["DEU"] == 1.0                    # 4.3 % → 2–5 bin
    assert z["AUT"] == 3.0                    # EU aggregate cover polygon, 17.8 %
    assert "USA" in layer.no_data_locations and "SWE" in layer.no_data_locations
    us_hover = layer.no_data_hover[layer.no_data_locations.index("USA")]
    assert "selected reporter" in us_hover
    cn_hover = layer.hover[layer.locations.index("CHN")]
    assert "exports to 6.6 %" in cn_hover and "imports from 13.3 %" in cn_hover and "2025" in cn_hover
    assert layer.legend[-1][1] == "≥ 10 %" and len(layer.arcs) == 3
    assert "goods trade" in layer.caption.lower() and "services excluded" in layer.caption


def test_trade_arcs_geometry_and_width(snap):
    arcs = trade_arcs(snap, "US")
    assert [a[5].split(" ↔ ")[1].split(":")[0] for a in arcs] == ["Eurozone", "China", "Germany"]
    assert arcs[0][4] == pytest.approx(6.0)                                       # top partner: 1 + 5
    assert 1.0 < arcs[2][4] < arcs[1][4] < 6.0
    assert (arcs[0][0], arcs[0][1]) == (39.8, -98.6) and (arcs[0][2], arcs[0][3]) == (50.1, 9.0)
    assert trade_arcs(snap, "SE") == () and len(trade_arcs(snap, "US", n=1)) == 1 and TRADE_ARCS == 5


def test_partner_table_and_html(snap):
    t = trade_partner_table(snap, "CN")
    assert list(t["partner"]) == ["EU", "US", "DE"]          # 12.3+9.7 > 14.7+6.3
    row = t.set_index("partner").loc["US"]
    assert row["x_share"] == pytest.approx(525 / 3580 * 100)
    assert row["balance_usd"] == pytest.approx(525.0 - 163.0)
    assert row["their_exposure"] == pytest.approx(143.5 / US_X_TOT * 100)         # US exports to CN
    assert row["name"] == "United States" and row["year"] == 2025
    html = html_trade_table(t)
    assert html.count("<tr><td>") == 3 and "United States" in html and "class=\"num\"" in html
    assert "14.7 %" in html and "6.6 %" in html
    assert trade_partner_table(snap, "SE").empty
    assert "2025 ·" in trade_caption(snap, "CN") and "Goods only" in trade_caption(snap, "CN")


def test_html_trade_table_escapes_and_handles_missing(snap):
    t = trade_partner_table(snap, "US")
    t.loc[0, "name"] = "<b>x</b>"
    t.loc[0, "m_share"] = float("nan")
    html = html_trade_table(t)
    assert "&lt;b&gt;x&lt;/b&gt;" in html and "<b>x</b>" not in html and "—" in html


def test_map_adds_one_line_trace_per_arc(snap):
    layer = map_layer(snap, MapMode.TRADE, None, "learning", "US")
    fig = build_fundamentals_map(layer)
    lines = [t for t in fig.data if t.type == "scattergeo" and t.mode == "lines"]
    assert len(lines) == 3 and all(t.line.width >= 1.0 for t in lines)
    plain = build_fundamentals_map(map_layer(snap, MapMode.COMPOSITE, None, "learning", "US"))
    assert not [t for t in plain.data if t.type == "scattergeo" and t.mode == "lines"]


def test_no_trade_snapshot_degrades(synthetic_snapshot_dict):
    raw = dict(synthetic_snapshot_dict)
    raw["trade"] = None
    s = parse_snapshot(raw)
    layer = map_layer(s, MapMode.TRADE, None, "learning", "US")
    assert not layer.locations and layer.arcs == () and "USA" in layer.no_data_locations
    assert trade_partner_table(s, "US").empty and "No bilateral trade" in trade_caption(s, "US")
