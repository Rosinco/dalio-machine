"""Streamlit dashboard — minimal smoke-test version for slice 1.

Loads observations from SQLite and shows latest values per country/indicator
plus a single-indicator chart. Full cycle-classification UI lands in slice 1
step 5.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import select

from dalio.storage.db import Observation, make_engine, make_session_factory


def _load_observations() -> pd.DataFrame:
    engine = make_engine()
    session_factory = make_session_factory(engine)
    with session_factory() as session:
        rows = session.execute(
            select(Observation).order_by(
                Observation.country,
                Observation.indicator,
                Observation.date.desc(),
            )
        ).scalars().all()
    if not rows:
        return pd.DataFrame(
            columns=["country", "indicator", "date", "value", "source", "series_id"]
        )
    return pd.DataFrame([
        {
            "country": r.country,
            "indicator": r.indicator,
            "date": r.date,
            "value": r.value,
            "source": r.source,
            "series_id": r.series_id,
        }
        for r in rows
    ])


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title="Dalio Machine", layout="wide")
    st.title("dalio-machine")
    st.caption("Pre-alpha — slice 1 (US short-term cycle) in progress.")

    df = _load_observations()
    if df.empty:
        st.warning("No observations yet. Run `dalio-fetch-fred` first.")
        return

    st.metric("Total observations", f"{len(df):,}")

    st.subheader("Latest value per country/indicator")
    latest = (
        df.sort_values("date")
          .groupby(["country", "indicator"], as_index=False)
          .tail(1)
          .sort_values(["country", "indicator"])
    )
    st.dataframe(latest, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Indicator history")
    countries = sorted(df["country"].unique())
    country = st.selectbox("Country", countries)
    indicators = sorted(df[df["country"] == country]["indicator"].unique())
    indicator = st.selectbox("Indicator", indicators)
    sub = (
        df[(df["country"] == country) & (df["indicator"] == indicator)]
        .sort_values("date")
        .set_index("date")["value"]
    )
    st.line_chart(sub)


if __name__ == "__main__":
    main()
