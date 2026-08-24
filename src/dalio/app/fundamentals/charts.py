"""Plotly builders for the Fundamentals page (no Streamlit imports)."""
from __future__ import annotations

import plotly.graph_objects as go

from dalio.app.fundamentals.view_models import MapLayer
from dalio.app.theme import INK, NO_DATA, PAPER, RUST, geo_layout, plotly_base_layout

SELECTED_LINE_WIDTH = 2.2
DEFAULT_LINE_WIDTH = 0.6


def _flat_colorscale(ramp: tuple[str, ...]) -> list[list]:
    """5 (or 4) flat colour segments so z-bins never blend."""
    n = len(ramp)
    scale: list[list] = []
    for i, c in enumerate(ramp):
        scale.append([i / n, c])
        scale.append([(i + 1) / n, c])
    return scale


def build_fundamentals_map(layer: MapLayer, height: int = 460) -> go.Figure:
    """Three traces: scored polygons (binned), no-data polygons, data-quality markers."""
    n_bins = len(layer.ramp)
    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locations=list(layer.locations),
        z=list(layer.z_bin),
        text=list(layer.hover),
        hoverinfo="text",
        locationmode="ISO-3",
        colorscale=_flat_colorscale(layer.ramp),
        zmin=-0.5, zmax=n_bins - 0.5,
        showscale=False,
        marker=dict(
            opacity=list(layer.opacity),
            line=dict(
                width=[SELECTED_LINE_WIDTH if s else DEFAULT_LINE_WIDTH for s in layer.selected],
                color=[RUST if s else PAPER for s in layer.selected],
            ),
        ),
        name="scored",
    ))
    if layer.no_data_locations:
        fig.add_trace(go.Choropleth(
            locations=list(layer.no_data_locations),
            z=[0.0] * len(layer.no_data_locations),
            text=list(layer.no_data_hover),
            hoverinfo="text",
            locationmode="ISO-3",
            colorscale=[[0, NO_DATA], [1, NO_DATA]],
            showscale=False,
            marker=dict(line=dict(width=DEFAULT_LINE_WIDTH, color=PAPER)),
            name="no data",
        ))
    if layer.dq_points:
        fig.add_trace(go.Scattergeo(
            lat=[p[0] for p in layer.dq_points],
            lon=[p[1] for p in layer.dq_points],
            text=[p[2] for p in layer.dq_points],
            hoverinfo="text",
            mode="markers",
            marker=dict(symbol="diamond-open", size=9, color=INK, line=dict(width=1.2, color=INK)),
            name="data quality",
        ))
    fig.update_layout(geo=geo_layout(), showlegend=False, **plotly_base_layout(height=height))
    return fig
