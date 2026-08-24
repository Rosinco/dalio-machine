"""Audit the cycle inputs actually stored in the database — latest date, age
and source per (country, indicator) — so stale series are visible offline.

Slice 26 rewrote this from a FRED-API probe (needed a key, saw only FRED) to a
database audit: the classifiers read whatever is freshest per indicator, so
this is the table that decides what the Cycles page runs on.

    python scripts/audit_freshness.py            # all 8 cycle countries
    python scripts/audit_freshness.py JP UK      # subset
"""
from __future__ import annotations

import sys
from datetime import date

from sqlalchemy import func, select

from dalio.countries import CYCLE_COUNTRIES
from dalio.scoring.long_term import LONG_TERM_INDICATORS
from dalio.scoring.short_term import MAX_INPUT_AGE_DAYS, SHORT_TERM_INDICATORS
from dalio.storage.db import Observation, make_engine, make_session_factory

INDICATORS = tuple(dict.fromkeys(SHORT_TERM_INDICATORS + LONG_TERM_INDICATORS))


def main() -> int:
    wanted = {c.upper() for c in sys.argv[1:]} or {c.iso2 for c in CYCLE_COUNTRIES}
    today = date.today()
    engine = make_engine()
    with make_session_factory(engine)() as session:
        rows = session.execute(
            select(Observation.country, Observation.indicator, Observation.source,
                   func.max(Observation.date), func.count())
            .where(Observation.country.in_(sorted(wanted)),
                   Observation.indicator.in_(INDICATORS))
            .group_by(Observation.country, Observation.indicator, Observation.source)
        ).all()

    latest: dict[tuple[str, str], tuple[date, str, int]] = {}
    for country, indicator, source, max_date, n in rows:
        d = max_date if isinstance(max_date, date) else date.fromisoformat(str(max_date)[:10])
        if (country, indicator) not in latest or d > latest[(country, indicator)][0]:
            latest[(country, indicator)] = (d, source, n)

    print(f"{'C':<4} {'Indicator':<24} {'Latest':<12} {'Age (d)':>7}  {'Source':<11} {'Rows':>5}  Note")
    print("-" * 90)
    stale = 0
    for country in sorted(wanted):
        for indicator in INDICATORS:
            hit = latest.get((country, indicator))
            if hit is None:
                print(f"{country:<4} {indicator:<24} {'—':<12} {'—':>7}  {'—':<11} {'—':>5}  missing")
                continue
            d, source, n = hit
            days = (today - d).days
            if days > MAX_INPUT_AGE_DAYS:
                note, stale = f"STALE — guarded out (> {MAX_INPUT_AGE_DAYS} d)", stale + 1
            elif days > 365:
                note = "stale > 1 y"
            else:
                note = "ok"
            print(f"{country:<4} {indicator:<24} {d.isoformat():<12} {days:>7}  {source:<11} {n:>5}  {note}")
    print(f"\n{stale} cell(s) older than the {MAX_INPUT_AGE_DAYS}-day guard.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
