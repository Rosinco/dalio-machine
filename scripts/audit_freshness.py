"""Audit FRED series for data freshness and detect transform bugs."""
from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv
from fredapi import Fred

from dalio.data_sources.fred import TIER_1_SERIES

load_dotenv()
f = Fred(api_key=os.environ["FRED_API_KEY"])

today = date.today()
print(f"{'C':<4} {'Indicator':<22} {'Series ID':<28} {'Latest':<12} {'Age (d)':>7}  {'Notes'}")
print("-" * 120)

for spec in TIER_1_SERIES:
    try:
        s = f.get_series(spec.series_id).dropna()
        if len(s) == 0:
            print(f"{spec.country:<4} {spec.indicator:<22} {spec.series_id:<28} EMPTY")
            continue
        latest = s.index[-1].date()
        days = (today - latest).days
        sample = s.iloc[-1]
        notes = []
        if days > 730:
            notes.append("STALE >2y")
        elif days > 365:
            notes.append("stale >1y")
        if spec.transform == "yoy" and ("657" in spec.series_id or "659" in spec.series_id):
            notes.append("BUG: 657/659 already YoY")
        note_str = "; ".join(notes) if notes else "ok"
        print(
            f"{spec.country:<4} {spec.indicator:<22} {spec.series_id:<28} "
            f"{str(latest):<12} {days:>7}  {note_str}"
        )
    except Exception as e:
        print(f"{spec.country:<4} {spec.indicator:<22} {spec.series_id:<28} ERROR: {str(e)[:40]}")
