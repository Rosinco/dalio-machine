# ADR 0003 — Cycle inputs: multi-source, freshest wins, 18-month guard

**Date:** 2026-08-24 · **Status:** accepted · **Slice:** 26

## Context

The Cycles page (slices 1–17) reads its short-term inputs from FRED. FRED's
OECD-MEI mirrors for the non-US basket are discontinued or lag a year: on
2026-08-24 six of eight countries classified on ≥ 16-month-old CPI, India on a
policy rate frozen in 2022, China on an unemployment rate last updated in 2011
— and the classifier voted on all of it, because `_latest_at` had no age limit
and the UI showed no input dates. The World Fundamentals Map was built on top
of a page whose inputs had silently rotted.

Keyless SDMX-CSV sources publish the same quantities currently (verified from
this network): IMF `IMF.STA,CPI` (monthly y/y, 7/8 — no euro-area entity),
BIS `WS_CBPOL` (policy rates, 8/8), OECD `DF_QNA_EXPENDITURE_GROWTH_OECD`
(real GDP y/y, 8/8 incl. `EA`) and OECD `DF_IALFS_UNE_M` (unemployment, 5/8 —
no CN/IN/BR). OECD's own CPI flow lacks Japan; IMF CPI wins.

## Decision

1. **Add sources, replace nothing.** `dalio-fetch-cycle` stores the four feeds
   under their own source tags (`IMF_CPI`, `BIS_CBPOL`, `OECD_QNA`, `OECD_LFS`)
   next to the FRED rows, with FRED's date conventions (first of month / first
   month of quarter). US and euro-area FRED series are daily and stay the best
   available; the classifiers already take the latest observation per
   indicator regardless of source, so no source-preference table is needed.
2. **Deterministic tie-break.** `_value_at_or_before` orders by `date desc,
   source asc` so two sources on the same date never make `replay.py`
   non-deterministic.
3. **18-month guard.** `extract_features` drops any short-term input older than
   `MAX_INPUT_AGE_DAYS = 548` relative to the as-of date and reports it in
   `ShortTermFeatures.stale_inputs`. Every short-term input is monthly or
   quarterly; older is a dead series. The guard is relative to `as_of`, so
   backtests still see 2011 data as fresh in 2011. Long-term inputs (annual /
   quarterly BIS, per-country calibrated) keep no guard.
4. **Honest zero-vote label.** "Transition (insufficient data)" is reserved for
   a missing core input (GDP, CPI, policy rate); with the core present and no
   rule firing the label is "Transition (no rule fired)". Japan had been
   reported as "insufficient data" for months of fully-present, merely quiet
   inputs.
5. **Freshness is visible.** The short-term card and the stage card print one
   mono line with the as-of date of every input, the inputs dropped as stale
   and the ones missing; `scripts/audit_freshness.py` prints the same table
   from the database (it used to probe the FRED API).
6. **One fetcher.** `data_sources/sdmx_csv.py::CachedTextFetcher` replaces the
   three copy-pasted `_fetch_text` bodies (IMF DataMapper, IMTS, OEC); the new
   adapters use it. `bis.py` / `worldbank.py` keep theirs (older signatures).

## Consequences

- 2026-08-24 live: every short-term cell ≤ 3 months old except CN/IN/BR
  unemployment (no keyless monthly source; CN's 2011 row is guarded out, IN's
  2025-01 row will be once it passes 18 months). Stage calls changed for the
  countries that had been voting on dead data — recorded in `project_context.md`.
- Mixed sources within one series (FRED history + IMF/OECD recent) are the
  same national headline measures; minor level differences at the seam are
  accepted and visible in the History explorer by source.
- Monthly maintenance: `dalio-fetch-fred && dalio-fetch-bis && dalio-fetch-cycle`.
