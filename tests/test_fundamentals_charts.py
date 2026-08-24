"""Chart builders (slice P1): trace shapes, outline arrays, flat colour scales."""
from dalio.app.fundamentals.charts import (
    DEFAULT_LINE_WIDTH,
    SELECTED_LINE_WIDTH,
    _flat_colorscale,
    build_fundamentals_map,
)
from dalio.app.fundamentals.snapshot import parse_snapshot
from dalio.app.fundamentals.view_models import MapMode, map_layer
from dalio.app.theme import PCT_RAMP, RUST


def test_flat_colorscale_has_two_stops_per_bin():
    scale = _flat_colorscale(PCT_RAMP)
    assert len(scale) == 10
    assert scale[0] == [0.0, PCT_RAMP[0]] and scale[-1] == [1.0, PCT_RAMP[-1]]
    assert scale[1][0] == scale[2][0] == 0.2


def test_map_traces_and_outline(synthetic_snapshot_dict):
    snap = parse_snapshot(synthetic_snapshot_dict)
    layer = map_layer(snap, MapMode.INDICATOR, "gdp_pc_ppp", "learning", selected_iso2="IN")
    fig = build_fundamentals_map(layer)
    assert [t.type for t in fig.data] == ["choropleth", "scattergeo"]   # no no-data trace here
    scored = fig.data[0]
    assert len(scored.locations) == len(scored.z) == len(scored.marker.line.width)
    widths = list(scored.marker.line.width)
    colors = list(scored.marker.line.color)
    assert widths.count(SELECTED_LINE_WIDTH) == 1 and colors.count(RUST) == 1
    assert widths[-1] == SELECTED_LINE_WIDTH and list(scored.locations)[-1] == "IND"
    assert set(widths[:-1]) == {DEFAULT_LINE_WIDTH}
    assert scored.zmin == -0.5 and scored.zmax == 4.5
    assert fig.layout.geo.projection.type == "natural earth"

    layer2 = map_layer(snap, MapMode.INDICATOR, "military_pct_gdp", "learning", selected_iso2="US")
    fig2 = build_fundamentals_map(layer2)
    assert [t.type for t in fig2.data] == ["choropleth", "choropleth", "scattergeo"]
    assert len(fig2.data[1].locations) == 19
