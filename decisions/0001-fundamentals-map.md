# ADR 0001 — World Fundamentals Map: five categories, percentile scoring, views-as-weights

**Status:** accepted · **Date:** 2026-08-24 · **Slice:** 18

## Context

Adam asked for "a map of the world tracking the major economic players and how they
are doing in each category, based on fundamentals — Pareto principle, Occam's razor,
common sense", presented as Gapminder-style bubbles + world map + flowcharts + stats,
serving four purposes at once (jurisdiction gate for concentrated stock positions,
allocation lens for the index/value sleeve, learning tool, moonshot-sleeve geography).
The dalio-machine already had the data plumbing (FRED/BIS/WB/IMF adapters, SQLite,
choropleth) for 8 economies and a rule-based cycle classifier; it had no
cross-country scorecard.

## Decision

1. **Five categories, first-principles ordered, Dalio's 18 determinants pruned into
   them:** `real_stuff` (people, land, energy, capital) · `production` (output, growth)
   · `exchange` (trade, external balance, reserves) · `promises` (debt, deficits, debt
   service) · `enforcer` (rule of law, stability, force). ≤ 3 sourceable, non-redundant
   indicators per category; 15 in total at slice 19 (slice 18 ships the tracer trio
   `gdp_pc_ppp`, `old_age_dependency`, `military_pct_gdp`). Explicitly rejected:
   character/civility, acts of nature, resource-allocation efficiency, infrastructure,
   PISA, GFCI, Gini-as-score, resource rents, exports-%-GDP.
2. **Players = 21 individual countries + the euro-area aggregate** (22 registry rows).
   DE/FR/IT/ES/NL are individual players *and* inside the aggregate, so the aggregate is
   `on_map=False`, excluded from the ranking population, and interpolated in.
   `dalio.countries` is the one place to add a player; the cycle basket
   (`CYCLE_COUNTRIES`, 8) is a subset with FRED wiring.
3. **Percentile rank among the 21**, direction-aware, worst = 0 / best = 100, average
   ranks for ties. Chosen over z-scores for robustness to fat tails (SA energy −178 %,
   JP debt 230 %). With n = 21 a percentile step is ~5 points — the UI shows quintile
   bins and "rank r/21", never decimals or a continuous colour bar.
4. **Category score = simple mean** of available indicator percentiles, `None` below
   half coverage. No weights: with 2–4 indicators per category weighting is noise
   dressed as precision.
5. **Purposes are views, not data:** weight vectors over category scores (`learning`
   equal · `jurisdiction` enforcer .5 / promises .3 / exchange .2 · `allocation`
   promises .4 / production .3 / exchange .2 / enforcer .1 · `moonshot` real_stuff .5 /
   exchange .3 / production .2), renormalised over available categories, `None`
   below 60 % weight coverage. **No headline composite in the default view.**
6. **Uncertainty is carried, never hidden:** every cell has a tier (A measured / B
   model-or-forecast / C ordinal), its as-of date and source; every country a static
   `data_quality` flag (CN/TR/SA low, RU opaque, IN/BR/ID/MX medium) and `fx_regime`;
   forecasts are a source tag (`*_FCST`), never imputed into history.
7. **Pressure chains (slice 21) are judgment encoded as ≤ 6 rules**, tier C, with
   mechanical spillover templates keyed on other players' flags — never free text.
8. **One snapshot JSON** (`data/snapshots/fundamentals_latest.json`, written by
   `dalio-score`) is the only coupling between data and presentation; the app never
   touches the DB for the fundamentals page.
9. **Storage unchanged:** the single `observations` table + a Python-side
   `IndicatorSpec` registry (the `SHORT_TERM_INDICATORS` precedent). The row-by-row
   `upsert_observations` was replaced by a set-based one with identical semantics.

## Consequences

- Adding a country touches one file; adding an indicator touches the registry + one
  adapter spec + `project_context.md`.
- The cycle map and classifiers are unchanged (still exactly 8 points); a has-cycle-data
  predicate keeps fundamentals-only rows from feeding the classifiers.
- World Bank `mrv=1` is banned (returns null latest rows); WGI needs `source=3` and
  ids like `GOV_WGI_RL.EST`; IMF DOTS no longer exists on the new IMF portal (IMTS spike
  post-gate).
- Rejected alternatives: Dalio's 18 as-is (unmeasurable items, overlap), a learned or
  hand-weighted composite (false precision), a new repo (duplicates adapters/tests/design).
