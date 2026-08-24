# ADR 0002 — Bilateral trade as named spillovers; economic complexity as the 16th indicator

**Date:** 2026-08-24 · **Status:** accepted · **Slices:** 24 (trade), 23 (ECI)

## Context

ADR 0001 shipped the World Fundamentals Map with 15 indicators and six pressure-chain
rules whose trade-related spillovers were group labels ("trade partners"). Two post-gate
spikes were left open: bilateral trade (IMF DOTS no longer exists) and tier-C
"production" extras (PWT, Atlas ECI). Both were verified against live endpoints on
2026-08-24: `IMF.STA,IMTS` (SDMX 2.1 CSV, keyless, one call per flow for the whole
basket, world total `G001`, euro area `G163`, goods only — no energy flows) and the OEC
olap-proxy (keyless, all 21 players 1995–2024). Harvard Atlas has no keyless API; PWT 11.0
is reachable (`.dta`, no new dependency).

## Decision

1. **Bilateral trade is stored without a schema change.** Partner-suffixed indicators
   (`exports_to_CN`, `imports_from_WLD`) in the existing `observations` table, source
   `IMF_IMTS`, USD. The euro-area reporter drops rows against its own member players
   (intra-union); its world totals still include intra-area trade and the UI says so.
2. **Shares live in one snapshot block** (`trade`: iso2 · partner · year · x_share ·
   m_share · x_usd · m_usd, latest year that has a world total per reporter). Two
   directions are kept distinct: *my share of trade with you* (map, partner table) and
   *how much of your exports come to me* (`their exposure` — the direction a demand
   shock propagates).
3. **Spillovers are named, mechanically.** With trade in the panel, `external_financing`
   and `isolation` replace the "trade partners" label with the three players most exposed
   to the country (≥ 2 % of their own exports), and `energy_dependence` orders exporters
   by the dependent country's import share from them. Without trade the group labels
   remain (both paths tested). Confidence and triggers are unchanged; goods-only IMTS
   cannot isolate energy flows, so the energy rule still picks exporters by energy balance.
4. **Trade map mode + arcs + partner table**, no new visual system: rust ramp for
   "share of the selected country's goods trade" (< 2 · 2–5 · 5–10 · ≥ 10 %), five
   centroid arcs, dense mono table.
5. **Economic complexity (OEC ECI) becomes the 16th scored indicator**, category
   `production`, tier C, euro area as a flagged member mean. Pareto reasoning: it is the
   one measure of *what an economy can make* that GDP per capita, forward growth and R&D
   do not already carry. **PWT human capital and TFP are rejected**: `hc` largely duplicates
   GDP per capita in this basket; `rtfpna` is a within-country index (= 1 in 2017) and
   cross-sectionally meaningless; `ctfp` levels revise 5–10 % per vintage and behave
   oddly for resource economies. The "any indicator beyond the 15 needs a Decision Log
   line" rule of ADR 0001 is satisfied here.

## Consequences

- 352 cells (22 × 16); production has four indicators; category means and views are
  unchanged in method. Live effect on 2026-08-24: no A-tier jurisdiction change.
- Two more keyless sources to keep alive (`imts`, `oec` in `dalio-fetch-fundamentals`);
  IMTS is ~2 MB per flow, cached 24 h. OEC's endpoint is undocumented — treat a 404 as
  "moved", not as data loss (the last snapshot keeps serving).
- Rejected: a trade-concentration score (would be a 17th indicator with no user); services
  trade (no keyless bilateral source); storing partner rows in a new table (a migration for
  ~7 k rows).
