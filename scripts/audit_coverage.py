"""Coverage audit for the World Fundamentals Map.

Prints a players × indicators matrix of the latest observation year per cell
(``--`` = missing, ``f`` suffix = forecast-only), fill % per indicator and per
player, and every cell older than three years. Run after each fetch and paste
the output into the slice's project_context entry.

    python scripts/audit_coverage.py [--as-of YYYY-MM-DD] [--stale-years N]
"""
from __future__ import annotations

import argparse
from datetime import date

import pandas as pd
from dotenv import load_dotenv

from dalio.countries import COUNTRIES
from dalio.scoring.fundamentals import FUNDAMENTALS, load_forward_panel, load_history, load_panel
from dalio.scoring.trade import load_trade
from dalio.storage.db import init_db, make_engine, make_session_factory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--as-of", type=date.fromisoformat, default=None)
    parser.add_argument("--stale-years", type=int, default=3)
    args = parser.parse_args()
    load_dotenv()
    as_of = args.as_of or date.today()

    engine = make_engine()
    init_db(engine)
    with make_session_factory(engine)() as s:
        panel = load_panel(s, FUNDAMENTALS, COUNTRIES, as_of)
        forward = load_forward_panel(s, FUNDAMENTALS, COUNTRIES, as_of)
        hist = load_history(s, FUNDAMENTALS, COUNTRIES)
        trade = load_trade(s, COUNTRIES, as_of)
    if not forward.empty:   # forward indicators replace their backward reading
        fwd_names = set(forward["indicator"])
        panel = pd.concat([panel[~panel["indicator"].isin(fwd_names)],
                           forward[["country", "indicator", "value", "date", "source"]]],
                          ignore_index=True)

    latest = {(r.country, r.indicator): r.date for r in panel.itertuples()}
    fcst_only = set()
    if not hist.empty:
        for (c, i), g in hist.groupby(["country", "indicator"]):
            if g["is_forecast"].all():
                fcst_only.add((c, i))

    names = [s.name for s in FUNDAMENTALS]
    short = [n[:10] for n in names]
    print(f"Coverage as of {as_of} — latest observation year per cell\n")
    print("     " + " ".join(f"{h:>10}" for h in short))
    stale: list[tuple[str, str, date]] = []
    fill_by_ind = dict.fromkeys(names, 0)
    for c in COUNTRIES:
        cells = []
        filled = 0
        for n in names:
            d = latest.get((c.iso2, n))
            if d is None:
                cells.append("--" + ("f" if (c.iso2, n) in fcst_only else ""))
            else:
                filled += 1
                fill_by_ind[n] += 1
                cells.append(str(d.year))
                if (as_of.year - d.year) > args.stale_years:
                    stale.append((c.iso2, n, d))
        print(f"{c.iso2:<4} " + " ".join(f"{x:>10}" for x in cells) + f"   {filled:>2}/{len(names)}")

    print("\nFill by indicator:")
    for n in names:
        print(f"  {n:<26} {fill_by_ind[n]:>2}/{len(COUNTRIES)}")
    total = sum(fill_by_ind.values())
    print(f"\nTotal {total}/{len(COUNTRIES) * len(names)} cells "
          f"({100 * total / (len(COUNTRIES) * len(names)):.0f} %)")
    if trade.empty:
        print("\nBilateral trade (IMTS): none stored — run `dalio-fetch-fundamentals --only imts`")
    else:
        years = trade.groupby("iso2")["year"].first()
        print(f"\nBilateral trade (IMTS): {years.size}/{len(COUNTRIES)} reporters · "
              f"latest year {int(years.min())}–{int(years.max())}")
    if stale:
        print(f"\nCells older than {args.stale_years} years:")
        for iso2, n, d in stale:
            print(f"  {iso2} {n} {d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
