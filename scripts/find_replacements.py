"""Find live FRED series for the 3 stalest indicators."""
from __future__ import annotations

import os
from datetime import date

from dotenv import load_dotenv
from fredapi import Fred

load_dotenv()
f = Fred(api_key=os.environ["FRED_API_KEY"])

today = date.today()


def test_candidates(label: str, candidates: list[str]) -> None:
    print(f"=== {label} ===")
    for sid in candidates:
        try:
            s = f.get_series(sid).dropna()
            if len(s) == 0:
                print(f"  {sid:<32}  empty")
                continue
            latest = s.index[-1].date()
            days = (today - latest).days
            sample = s.iloc[-1]
            print(f"  {sid:<32}  {latest!s:<12}  ({days:>5}d old)  sample={sample:>8.2f}")
        except Exception as e:
            print(f"  {sid:<32}  ERROR: {str(e)[:50]}")
    print()


# Search FRED for current candidates
def search(query: str, n: int = 8) -> None:
    print(f"--- search: {query!r} ---")
    try:
        df = f.search(query, limit=n)
        for sid, title in zip(df.index, df["title"]):
            print(f"  {sid:<28}  {title[:80]}")
    except Exception as e:
        print(f"  ERROR: {e}")
    print()


# JP CPI — need a current series
test_candidates("JP CPI candidates", [
    "JPNCPIALLAINMEI", "JPNCPIALLQINMEI",
    "CPALCY01JPM659N", "CPALCY01JPM661S",
    "CPALTT01JPQ657S", "CPALTT01JPQ659N",
    "FPCPITOTLZGJPN",
])
search("japan consumer price index 2025", 8)

# SE policy_rate — Riksbank
test_candidates("SE policy candidates", [
    "IR3TIB01SEM156N", "IRSTCI01SEQ156N",
    "INTDSRSEM193N", "INTGSTSEM193N",
    "IR3TBB01SEM156N", "IR3TBC01SEM156N",
])
search("sweden interest rate 2025", 8)

# EU unemployment — recent
test_candidates("EU unemployment candidates", [
    "LRHUTTTTEZQ156S", "UNRTUEMM",
    "LMUNRRTTEZM156S", "LRUN24TTEZM156S",
])
search("euro area unemployment rate", 8)
