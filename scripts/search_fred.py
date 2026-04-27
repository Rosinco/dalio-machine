"""One-off helper: search FRED for series IDs of failing country/indicator combos.

Run from project root with venv active:
    python scripts/search_fred.py
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
f = Fred(api_key=os.environ["FRED_API_KEY"])

QUERIES = [
    ("JP CPI all items", "consumer price index all items japan"),
    ("JP policy rate", "japan policy rate"),
    ("JP unemployment", "japan unemployment rate"),
    ("JP real GDP", "japan real gross domestic product"),
    ("CN unemployment", "china unemployment"),
    ("CN 10Y yield", "china long-term government bond yield"),
    ("CN real GDP", "china real gross domestic product"),
    ("UK real GDP", "united kingdom real gdp"),
    ("SE CPI", "sweden consumer price index"),
]

for label, q in QUERIES:
    print(f"=== {label} ===")
    try:
        df = f.search(q, limit=5)
        for sid, title in zip(df.index, df["title"]):
            print(f"  {sid:<30}  {title[:80]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
