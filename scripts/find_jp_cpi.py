"""Search FRED for any current Japan CPI series."""
import os
from datetime import date

from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
f = Fred(api_key=os.environ["FRED_API_KEY"])
today = date.today()

queries = [
    "japan inflation rate monthly",
    "JPN CPI",
    "japan consumer price",
    "Japan inflation",
]
seen: set[str] = set()
for q in queries:
    print(f"--- {q} ---")
    try:
        df = f.search(q, limit=20)
        for sid, title in zip(df.index, df["title"]):
            if sid in seen:
                continue
            seen.add(sid)
            try:
                s = f.get_series(sid).dropna()
                if len(s) == 0:
                    continue
                latest = s.index[-1].date()
                days = (today - latest).days
                if days < 365:  # Only show fresh series
                    print(f"  {sid:<32}  {str(latest):<12} ({days:>4}d)  {title[:70]}")
            except Exception:
                pass
    except Exception as e:
        print(f"  ERROR: {e}")
    print()
