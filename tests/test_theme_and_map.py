"""Theme tokens + the selected-country outline on the cycles choropleth (slice P0)."""
from dalio.app.theme import (
    EXPOSURE_RAMP,
    PCT_BIN_LABELS,
    PCT_RAMP,
    PLAYER_CENTROIDS,
    confidence_block,
    geo_layout,
    legend_row,
    plotly_base_layout,
)
from dalio.app.views import CountryMapPoint
from dalio.countries import COUNTRIES, EUROZONE_ISO3


def test_ramps_and_labels():
    assert len(PCT_RAMP) == 5 and len(PCT_BIN_LABELS) == 5
    assert len(EXPOSURE_RAMP) == 4
    assert all(c.startswith("#") and len(c) == 7 for c in PCT_RAMP + EXPOSURE_RAMP)


def test_every_player_has_a_centroid():
    assert {c.iso2 for c in COUNTRIES} <= set(PLAYER_CENTROIDS)
    for lat, lon in PLAYER_CENTROIDS.values():
        assert -90 <= lat <= 90 and -180 <= lon <= 180


def test_plotly_base_layout_keys():
    lay = plotly_base_layout(height=300)
    assert lay["height"] == 300
    assert {"margin", "paper_bgcolor", "plot_bgcolor", "font", "hoverlabel"} <= set(lay)
    assert geo_layout()["projection_type"] == "natural earth"


def test_html_helpers():
    assert 'width:50.0%' in confidence_block("x", 0.5)
    assert 'width:100.0%' in confidence_block("x", 7.0)     # clamped
    html = legend_row([("#000000", "A"), ("#ffffff", "B")])
    assert html.count("swatch") == 2 and html.startswith('<div class="legend-row">')


def _point(iso2, iso3, has_data=True):
    return CountryMapPoint(
        iso2=iso2, iso3=iso3, name=iso2, has_data=has_data, long_term_phase=2,
        long_term_label="x", short_term_stage=1, short_term_label="y", caution_level="low",
        total_debt_pct_gdp=200.0, debt_service_ratio=15.0, cpi_yoy=2.0, real_rate_10y=1.0,
        hover_text=f"<b>{iso2}</b>",
    )


def test_choropleth_outlines_selected_country_and_draws_it_last():
    from dalio.app.streamlit_app import DEFAULT_LINE_WIDTH, SELECTED_LINE_WIDTH, _build_choropleth
    from dalio.app.theme import RUST

    points = [_point("US", "USA"), _point("EU", "EMU"), _point("JP", "JPN")]
    fig = _build_choropleth(points, "phase", selected_iso2="EU")
    tr = fig.data[0]
    locs = list(tr.locations)
    widths = list(tr.marker.line.width)
    colors = list(tr.marker.line.color)
    # EU's 20 member polygons are the LAST 20 entries
    assert locs[-20:] == list(EUROZONE_ISO3)
    assert widths[-20:] == [SELECTED_LINE_WIDTH] * 20
    assert colors[-20:] == [RUST] * 20
    assert widths[:2] == [DEFAULT_LINE_WIDTH] * 2
    assert len(locs) == len(widths) == len(colors) == 22

    fig2 = _build_choropleth(points, "phase", selected_iso2=None)
    assert set(fig2.data[0].marker.line.width) == {DEFAULT_LINE_WIDTH}
