"""Plotly builders for the Fundamentals page (no Streamlit imports)."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from dalio.app.fundamentals.view_models import MapLayer, bubble_ranges, bubble_trail
from dalio.app.theme import (
    FONT_BODY,
    FONT_MONO,
    INK,
    INK_MUTED,
    NO_DATA,
    PAPER,
    PCT_RAMP,
    RULE,
    RUST,
    geo_layout,
    plotly_base_layout,
)

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


def build_bubble(
    frame: pd.DataFrame,
    x_label: str,
    y_label: str,
    selected_iso2: str | None,
    log_x: bool = True,
    height: int = 520,
) -> go.Figure:
    """Gapminder-style animated scatter. Forecast years are open circles and
    their slider steps carry an 'f' suffix; the selected player gets a trail
    (dotted over the forecast segment) and the only direct label."""
    if frame.empty:
        fig = go.Figure()
        fig.update_layout(**plotly_base_layout(height=height))
        return fig
    df = frame.copy()
    df["label"] = df["name"].where(df["iso2"] == selected_iso2, "")
    df["kind"] = df["is_forecast"].map({False: "actual", True: "forecast"})
    (x0, x1), (y0, y1) = bubble_ranges(df, log_x)
    fig = px.scatter(
        df, x="x", y="y", size="size", color="color", animation_frame="year", animation_group="iso2",
        hover_name="name", symbol="kind", symbol_map={"actual": "circle", "forecast": "circle-open"},
        text="label", log_x=log_x, size_max=52, range_x=[10 ** x0, 10 ** x1] if log_x else [x0, x1],
        range_y=[y0, y1], color_continuous_scale=list(PCT_RAMP), range_color=[0, 100],
        labels={"x": x_label, "y": y_label, "color": "composite"},
    )
    fig.update_traces(marker=dict(line=dict(width=2, color=PAPER), sizemin=4),
                      textposition="top center", textfont=dict(family=FONT_BODY, size=11, color=INK),
                      selector=dict(type="scatter"))
    for fr in fig.frames:
        for tr in fr.data:
            tr.marker.line = dict(width=2, color=PAPER)
    # slider labels: forecast years suffixed 'f'
    fc_years = set(df.loc[df["is_forecast"], "year"].astype(int))
    if fig.layout.sliders:
        for step in fig.layout.sliders[0].steps:
            try:
                yr = int(step.label)
            except (TypeError, ValueError):
                continue
            if yr in fc_years and yr not in set(df.loc[~df["is_forecast"], "year"].astype(int)):
                step.label = f"{yr}f"
    # trail for the selected player (appended traces persist across frames)
    if selected_iso2:
        tr = bubble_trail(df, selected_iso2)
        if len(tr) > 1:
            actual = tr[~tr["is_forecast"]]
            fig.add_trace(go.Scatter(x=actual["x"], y=actual["y"], mode="lines", name="trail",
                                     line=dict(color=INK, width=1.5), hoverinfo="skip", showlegend=False))
            fc = tr[tr["is_forecast"]]
            if not fc.empty:
                bridge = pd.concat([actual.tail(1), fc])
                fig.add_trace(go.Scatter(x=bridge["x"], y=bridge["y"], mode="lines", name="trail-forecast",
                                         line=dict(color=INK, width=1.5, dash="dot"), hoverinfo="skip",
                                         showlegend=False))
    layout = plotly_base_layout(height=height)
    layout["margin"] = dict(l=10, r=10, t=10, b=10)
    fig.update_layout(
        showlegend=False, coloraxis_showscale=False,
        xaxis=dict(title=dict(text=x_label, font=dict(size=11)), showgrid=True, gridcolor=RULE, gridwidth=0.5,
                   zeroline=False, color=INK_MUTED, tickfont=dict(family=FONT_MONO, size=10)),
        yaxis=dict(title=dict(text=y_label, font=dict(size=11)), showgrid=True, gridcolor=RULE, gridwidth=0.5,
                   zeroline=True, zerolinecolor=RULE, color=INK_MUTED, tickfont=dict(family=FONT_MONO, size=10)),
        **layout,
    )
    return fig


def build_pareto(df: pd.DataFrame, height: int | None = None) -> go.Figure:
    """Horizontal Pareto of gap-to-best-in-class: ink bars sorted descending,
    end labels 'gap · cum %', one rust rule where cumulative share crosses 80 %.
    No secondary axis (gap points and share are different quantities)."""
    n = len(df)
    h = height or max(160, 34 * n + 40)
    fig = go.Figure()
    if n == 0:
        fig.update_layout(**plotly_base_layout(height=h))
        return fig
    keys = list(df["key"])[::-1]                    # plotly draws bottom-up
    gaps = list(df["gap"])[::-1]
    labels = [f"gap {g:.0f} · cum {c:.0%}" for g, c in zip(gaps, list(df["cum_share"])[::-1], strict=True)]
    fig.add_trace(go.Bar(
        x=gaps, y=keys, orientation="h",
        marker=dict(color=INK, line=dict(color=PAPER, width=1)),
        text=labels, textposition="outside", cliponaxis=False,
        textfont=dict(family=FONT_MONO, size=11, color=INK_MUTED),
        hovertemplate="%{y}: gap %{x:.0f} points<extra></extra>",
    ))
    if df["crosses_80"].any():
        idx80 = int(df.index[df["crosses_80"]][0])
        y_pos = n - 1 - idx80                          # reversed axis index
        fig.add_shape(type="line", x0=0, x1=max(gaps) * 1.02, y0=y_pos - 0.5, y1=y_pos - 0.5,
                      line=dict(color=RUST, width=1.5, dash="dot"))
        fig.add_annotation(x=max(gaps) * 1.02, y=y_pos - 0.5, text="80 %", showarrow=False,
                           xanchor="left", yanchor="middle",
                           font=dict(family=FONT_BODY, size=10, color=RUST))
    layout = plotly_base_layout(height=h)
    layout["margin"] = dict(l=0, r=90, t=6, b=24)
    fig.update_layout(
        bargap=0.35,
        xaxis=dict(title=dict(text="gap to best-in-class (percentile points)", font=dict(size=10)),
                   range=[0, max(gaps) * 1.4], showgrid=True, gridcolor=RULE, gridwidth=0.5,
                   zeroline=False, color=INK_MUTED, tickfont=dict(family=FONT_MONO, size=10)),
        yaxis=dict(showgrid=False, color=INK, tickfont=dict(family=FONT_BODY, size=11), automargin=True),
        showlegend=False,
        **layout,
    )
    return fig
