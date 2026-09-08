"""Historical regime replay — walks both classifiers across a date range.

Slice 12: builds the track-record check for the lens. Without this, the
dashboard only shows "current phase" — a user has no way to verify the
classifier's calls against known events (2008, 2020, 2022). The replay
DataFrame is the substrate for the "Historical regime path" expander.

Reuses the same `classify` functions as the live dashboard. At each cursor,
an expanding-window calibration cutoff excludes observations with later
economic/reference dates, and the same threshold set is passed to both
classifiers alongside their historical feature-extraction cursor.

This is still a revised-data, economic-date replay over the current
``observations`` projection. It is not a true "known at the time" replay:
revisions and publication lags require release-vintage materialization.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta
from sqlalchemy.orm import Session

from dalio.scoring.calibration import compute_country_thresholds
from dalio.scoring.long_term import classify as classify_long_term
from dalio.scoring.short_term import classify as classify_short_term


def replay_classifications(
    session: Session,
    country: str,
    start: date,
    end: date,
    step: str = "Q",
) -> pd.DataFrame:
    """Walk both classifiers across the date range, returning one row per step.

    `step` accepts pandas-style frequency hints: "M" (month), "Q" (quarter),
    "Y" (year). Default is quarterly — matches BIS data cadence and produces
    roughly 140 rows for a 35-year window.

    Returns columns:
      date, st_stage, st_label, st_confidence,
      lt_phase, lt_label, lt_confidence

    The replay uses revised values indexed by economic/reference date; it does
    not yet reconstruct the source release vintage known at each cursor.
    """
    delta_map = {
        "M": relativedelta(months=1),
        "Q": relativedelta(months=3),
        "Y": relativedelta(years=1),
    }
    delta = delta_map.get(step.upper(), relativedelta(months=3))

    rows: list[dict] = []
    cursor = start
    while cursor <= end:
        thresholds = compute_country_thresholds(session, country, as_of=cursor)
        st = classify_short_term(
            session,
            country,
            thresholds=thresholds,
            as_of=cursor,
        )
        lt = classify_long_term(
            session,
            country,
            thresholds=thresholds,
            as_of=cursor,
        )
        rows.append(
            {
                "date": cursor,
                "st_stage": st.stage,
                "st_label": st.stage_label,
                "st_confidence": st.confidence,
                "lt_phase": lt.phase,
                "lt_label": lt.phase_label,
                "lt_confidence": lt.confidence,
            }
        )
        cursor = cursor + delta

    return pd.DataFrame(rows)
