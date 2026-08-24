# Project Context

## Overview

dalio-machine is a macro-cycle dashboard built on Ray Dalio's economic machine framework. For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), it pulls macro indicators from FRED, BIS, IMF and World Bank, then classifies cycle stage with rule-based logic. Output is a Streamlit dashboard for **asset-allocation decision support** — informing diversified-portfolio tilts based on regime indicators, *not* market-timing signals.

Since slice 18 it also hosts the **World Fundamentals Map**: 21 countries + the euro-area aggregate scored by percentile on fundamentals across five first-principles categories (real stuff · production · exchange · promises · enforcer), with purpose-specific *views* (jurisdiction gate for concentrated stock positions, allocation lens, learning, moonshot geography) as weight vectors over the category scores. Design in `decisions/0001-fundamentals-map.md`; full plan in `~/.claude/plans/radiant-bubbling-micali.md`.

The framework is treated as a descriptive lens, not a predictive oracle. Per Dalio's own writing and academic evidence on macro-overlay strategies, this dashboard surfaces regime state with explicit confidence — it does not output buy/sell signals.

## Tech Stack

- **Runtime:** Python 3.12
- **Framework:** Streamlit (dashboard), SQLAlchemy 2.x (storage), pandas (compute)
- **Test Framework:** pytest with mocked HTTP (pytest-mock + responses)
- **Key Dependencies:** fredapi, requests, pandas, sqlalchemy, streamlit, plotly, python-dotenv

## Architecture

- **Entry Points:**
  - `dalio-fetch-fred` (CLI ETL) — `src/dalio/pipelines/fetch_fred.py:main`
  - `dalio-fetch-bis` (CLI ETL) — `src/dalio/pipelines/fetch_bis.py:main`
  - `dalio-fetch-fundamentals` (CLI ETL, World Bank; IMF/BIS families planned) — `src/dalio/pipelines/fetch_fundamentals.py:main`
  - `dalio-score` (writes `data/snapshots/fundamentals_latest.json` + dated copy) — `src/dalio/pipelines/score_fundamentals.py:main`
  - `dalio-app` (Streamlit dashboard) — `src/dalio/app/streamlit_app.py:main`
- **Module Structure:**
  - `src/dalio/countries.py` — **the one place to add a player**: 22-row registry (8 cycle countries + 14 Tier-3 fundamentals-only), derived `ISO2_TO_WB` / `ISO2_TO_BIS` / `ISO3_TO_ISO2`, `CYCLE_COUNTRIES`, `RANKING_POPULATION` (21, no aggregate), `EUROZONE_ISO3`, static flags (`fx_regime`, `sanctioned`, `data_quality`)
  - `src/dalio/data_sources/` — one adapter per provider: `fred.py`, `bis.py`, `worldbank.py` (WDI source 2 + WGI source 3, paginated multi-country, never `mrv`)
  - `src/dalio/scoring/` — cycle classifiers + allocation map + `fundamentals.py` (indicator registry, percentile/category/view scoring, snapshot) + `big_cycle.py` (Gini/COFER; delegates WB calls to `worldbank.py`)
  - `src/dalio/storage/` — SQLite schema + session helpers (single `observations` table; fundamentals rows share it)
  - `src/dalio/pipelines/` — ETL orchestration; `fetch_fred.upsert_observations` is the shared **set-based** upsert (slice 18)
  - `src/dalio/app/` — Streamlit dashboard (`views.py` view-models; cycle map iterates `CYCLE_COUNTRIES` via `has_cycle_data`)
  - `decisions/` — ADRs (pull on demand)
  - `src/dalio/indicators/` — empty placeholder (compute lives in `scoring/`)
- **Data Flow (cycles):** API (FRED/BIS) → adapter → long-format DataFrame → SQLite → feature extraction → stage classifier → Streamlit chart
- **Data Flow (fundamentals):** World Bank/IMF/BIS → adapter → SQLite → `dalio-score` → `data/snapshots/fundamentals_latest.json` → (slice P1) Streamlit fundamentals page reading the snapshot only

## Conventions

- **Naming:** snake_case for files/functions/vars; PascalCase for classes; UPPER_CASE for module-level constants. Indicator names are snake_case strings (`cpi_yoy`, `policy_rate`).
- **File Organization:** Tests mirror src tree (`tests/test_<module>.py`). One module per data source / indicator family.
- **Error Handling:** Pipelines log + collect failures per series; never let one bad series fail the whole batch. Tests use mocked HTTP — never real API calls.
- **Imports:** Absolute imports from `dalio.*`. No relative imports.
- **Time-series invariants:** All observations stored long-format `(country, indicator, date, value, source, series_id)`. UTC-naive `date` type, not `datetime`. No duplicate `(country, indicator, date, source)` rows (enforced by unique constraint).

## Country Basket

| Code | Country | Tier | FRED ID | BIS ID | Currency |
|------|---------|------|---------|--------|----------|
| US | United States | 1 | USA | US | USD |
| CN | China | 1 | CHN | CN | CNY |
| EU | Eurozone (DE/FR/IT) | 1 | EMU | XM | EUR |
| UK | United Kingdom | 1 | GBR | GB | GBP |
| JP | Japan | 1 | JPN | JP | JPY |
| SE | Sweden | 1 | SWE | SE | SEK |
| IN | India | 2 | IND | IN | INR |
| BR | Brazil | 2 | BRA | BR | BRL |

Tier drives dashboard confidence labels — Tier 2 readings are flagged as "data thinner" in the UI.

### Fundamentals players (slice 18, Tier 3 = fundamentals only)

DE · FR · IT · ES · NL (all `eu_member`, `currency_union`) · CA · RU (`sanctioned`, `data_quality=opaque`) · KR · AU · MX (medium) · ID (managed, medium) · SA (`peg`, low) · TR (managed, low) · CH. Plus the 8 cycle countries (US `reserve_issuer`; CN managed/low; IN managed/medium; BR medium). The euro-area aggregate `EU` is `on_map=False`, carries `members=EUROZONE_ISO3` (20), and is **excluded from `RANKING_POPULATION`** (21) — its percentiles are interpolated so its members are not double-counted. Adding a player = one `Country(...)` row.

## Indicator Catalogue

### World Fundamentals Map (slices 18–19; ADR 0001)

Five categories: `real_stuff` · `production` · `exchange` · `promises` · `enforcer`. Every indicator carries an uncertainty tier (A measured / B model-or-forecast / C ordinal index) and a direction. Registry: `src/dalio/scoring/fundamentals.py::FUNDAMENTALS`; raw pulls: `src/dalio/data_sources/worldbank.py::WB_FUNDAMENTALS`.

| Indicator | Category | Definition | Source | Dir | Tier |
|-----------|----------|-----------|--------|-----|------|
| `energy_net_imports_pct` | real_stuff | Energy net imports % of use (negative = exporter) | WB `EG.IMP.CONS.ZS` | lower | A |
| `old_age_dependency` | real_stuff | Population 65+ per 100 aged 15–64 | WB `SP.POP.DPND.OL` | lower | A |
| `gdp_pc_ppp` | production | GDP per capita, PPP (constant 2021 intl $) | WB `NY.GDP.PCAP.PP.KD` | higher | B |
| `gdp_growth_fwd5` | production | Mean IMF real-growth forecast, next 5 y | IMF DM `NGDP_RPCH` (slice 20) | higher | B |
| `rd_pct_gdp` | production | R&D expenditure % of GDP | WB `GB.XPD.RSDV.GD.ZS` | higher | A |
| `exports_share_world` | exchange | Exports ÷ world exports × 100 (derived) | WB `NE.EXP.GNFS.CD` ÷ `WLD` | higher | A |
| `current_account_pct_gdp` | exchange | Current account % of GDP | IMF DM `BCA_NGDPD` → WB `BN.CAB.XOKA.GD.ZS` | higher | A |
| `reserves_months_imports` | exchange | Total reserves in months of imports | WB `FI.RES.TOTL.MO` | higher | A |
| `gov_debt_pct_gdp` | promises | General government gross debt % of GDP | IMF DM `GGXWDG_NGDP` → BIS sector G | lower | B |
| `fiscal_balance_pct_gdp` | promises | General government net lending % of GDP | IMF DM `GGXCNL_NGDP` (slice 20) | higher | B |
| `interest_burden_pct_gdp` | promises | Primary balance − overall balance (derived) | IMF DM (slice 20) | lower | B |
| `debt_service_ratio` | promises | Private non-financial DSR (quarterly) | BIS WS_DSR | lower | B |
| `rule_of_law` (+`_se`) | enforcer | WGI estimate (−2.5..2.5) with standard error; EU = flagged member mean | WB src 3 `GOV_WGI_RL.EST/.SE` | higher | C |
| `political_stability` (+`_se`) | enforcer | WGI estimate with standard error; EU = flagged member mean | WB src 3 `GOV_WGI_PV.EST/.SE` | higher | C |
| `military_share_world` | enforcer | Military expenditure ÷ world × 100 (derived; SIPRI via WB) | WB `MS.MIL.XPND.CD` ÷ `WLD` | higher | A |

Also stored, not scored: `exports_usd`, `military_usd` (raw levels incl. `WLD`), `military_pct_gdp`.

Scoring: percentile rank among the 21 (worst 0 / best 100, average ties, ~5-point steps), 5-year trend with a 0.1 × cross-sectional-std dead band, category score = mean with a half-coverage floor, views (`learning`, `jurisdiction`, `allocation`, `moonshot`) = renormalised weight vectors with a 60 % floor. Forecast rows are tagged by source suffix `_FCST` (slice 20).

### Short-term debt cycle (slice 1)

| Indicator | Definition | Primary source | Notes |
|-----------|-----------|----------------|-------|
| `policy_rate` | Effective central bank policy rate (%) | FRED `DFF` (US) | Per-country mapping needed for fan-out |
| `cpi_yoy` | CPI year-over-year change (%) | FRED `CPIAUCSL` (US) | YoY transform applied |
| `unemployment_rate` | Headline unemployment (%) | FRED `UNRATE` (US) | |
| `yield_10y` | 10-year sovereign bond yield (%) | FRED `DGS10` (US) | |
| `yield_2y` | 2-year sovereign bond yield (%) | FRED `DGS2` (US) | |
| `real_gdp_yoy` | Real GDP year-over-year change (%) | FRED `GDPC1` (US) | YoY transform applied |

### Long-term debt cycle (slice 4)

| Indicator | Definition | Primary source |
|-----------|-----------|----------------|
| `total_credit_pct_gdp` | Total non-financial credit / GDP | BIS Total Credit |
| `gov_debt_pct_gdp` | Government debt / GDP | BIS / IMF |
| `hh_debt_pct_gdp` | Household debt / GDP | BIS Total Credit |
| `corp_debt_pct_gdp` | Non-financial corporate debt / GDP | BIS Total Credit |
| `debt_service_ratio` | Private non-financial DSR | BIS DSR |
| `real_rate_10y` | 10Y nominal yield − core CPI YoY | computed |
| `credit_impulse` | Δ(total credit) ÷ GDP | computed from BIS |

### Big cycle (slice 5)

| Measure | Source |
|---------|--------|
| Education | OECD PISA, World Bank tertiary attainment |
| Innovation & Tech | WIPO patents, OECD MSTI |
| Competitiveness | WEF GCI / IMD WCY |
| Economic output | IMF WEO, World Bank GDP |
| Share of world trade | UN COMTRADE / WTO |
| Military strength | SIPRI Mil Expenditure DB |
| Financial center | GFCI index (manual or scrape) |
| Reserve currency | IMF COFER |

### Currency lifecycle (slice 6)

| Indicator | Source |
|-----------|--------|
| Reserve currency share | IMF COFER (USD/EUR/JPY/GBP/CNY) |
| FX reserve trend | IMF, BIS |
| Currency vs trade-weighted index | BIS effective exchange rates |

### Wealth & values gaps (slice 6)

| Indicator | Source |
|-----------|--------|
| Top 1% income share | WID.world |
| Gini coefficient | World Bank WDI |
| Populist vote share | manual / V-Dem |

## Cycle Stage Classification

Rule-based, transparent, four stages per Dalio's roadmap (slice 1 — short-term cycle). Each rule emits one or more `StageVote(stage, weight, reason)`. Stage = highest summed weight; "Transition" if top two stages within 0.3 weight units. Confidence is saturating: `top_weight / (total_weight + 1.0)` so a single weak vote never reads as 100%.

| Stage | Trigger conditions |
|-------|--------------------|
| 1. Expansion | `real_gdp_yoy > 1.5` AND `cpi_yoy < 3` AND unemployment stable/falling. Steep yield curve adds weight. |
| 2. Inflationary peak | `cpi_yoy > 4` (strong) OR `cpi_yoy > 3` with CB tightening (moderate) OR CPI accelerating from ≥2.5% (modifier). |
| 3. Recession | `real_gdp_yoy < 0` OR `unemployment_change_3m > 0.5pp` (Sahm-rule territory). Inverted yield curve adds weight. |
| 4. Reflation | CB cutting strongly (>0.5pp/6m) OR moderately (0.25–0.5pp/6m) with `cpi_yoy < 4` OR CPI decelerating sharply. |

All thresholds in `src/dalio/scoring/short_term.py`. Vote weights and reasons are exposed in the dashboard (expandable "Rule reasoning") so every classification is inspectable.

## Current State

- **Working:** Slices 1–17 end-to-end for all 8 cycle countries. **Slice 18 (2026-08-24):** World Fundamentals Map data layer — 22-player registry, World Bank adapter, set-based upsert, percentile scoring, snapshot JSON; live run 66/66 cells (3 indicators × 22 players). No fundamentals UI yet (slice P0/P1).
- **Tests:** 246/246 passing (`pytest`), `ruff check src tests` clean.
- **App (P0, 2026-08-24):** `st.navigation` with two pages — **Cycles** (the slice 1–17 dashboard, unchanged content) and **Fundamentals** (placeholder reading the snapshot; map/leaderboard/table in P1). Shared sidebar (country over all 22 players with `bind="query-params"`, home currency) created in the entrypoint before `pg.run()`. Verified in a headless browser via Playwright: India click → sidebar + brief switch with no extra reruns; Germany click → folds to Eurozone (no loop); selection persists across pages; `st.graphviz_chart` renders a DOT string client-side with all node/edge labels (no `dot` binary, no `graphviz` package).
- **Live fundamentals (2026-08-24, tracer trio, learning view):** SA 93 · US 80 · KR 65 · RU 63 · UK/NL/AU 58 · SE 57 · … · IT 28 · JP 20. Reads as expected for these three inputs (military-%-GDP rewards SA/RU; JP's dependency ratio 51 is worst-in-class) — not a verdict until the full 15 land.
- **Live short-term cycle (2026-04-27):** US Transition (Reflation ↔ Inflationary peak) 29% / CN Expansion 42% / EU Inflationary peak 29% / UK Inflationary peak 33% / JP Transition (insufficient CPI) 0% / SE Expansion 42%.
- **Live long-term cycle (2026-04-27):**
  - **US** — Transition (Reflation/financial repression ↔ Bubble) 32% (debt 250%, fell 40pp/5y as inflation eroded ratio)
  - **China** — Top — peak debt service 60% (debt 296%, DSR 18.8% stretched)
  - **Eurozone** — Bubble 33% (debt 240%, late-cycle leverage zone)
  - **UK** — Reflation/financial repression 41% (debt 219%, fell 86pp/5y with CPI 3.4% — beautiful deleveraging)
  - **Japan** — Top — peak debt service 37% (debt 357% — extreme zone)
  - **Sweden** — Deleveraging 44% (DSR 23.3% — household distress, real-estate stress)
- **In Progress:** World Fundamentals Map — **MVP shipped (18, P0, P1, 19, P2)**. Next: 20 (IMF DataMapper: fills `gdp_growth_fwd5`, `fiscal_balance_pct_gdp`, `interest_burden_pct_gdp`, `gov_debt_pct_gdp` for all 22 — the Promises category and the Allocation/Jurisdiction views are mostly blank until then), 21 (pressure chains), P3 (chain flowcharts + exposure map), P4 (bubble). Gate 2026-09-14.
- **Known nits:** `bind="query-params"` stores the selectbox *label* (`?country=Sweden`), not the iso2, and ignores `?country=JP`; programmatic `session_state` writes (map/row clicks) don't update the URL. Cosmetic.
- **Coverage audit (2026-08-24, slice 19):** 231/330 cells (70 %). All WB/WGI indicators 22/22 (current account 21/22 — EU aggregate null at WB, IMF fills it). Thin until slice 20: `gov_debt_pct_gdp` 5/22 (BIS core debt for US/EU/UK/JP/SE), `debt_service_ratio` 7/22 (BIS DSR cycle countries), `gdp_growth_fwd5` / `fiscal_balance_pct_gdp` / `interest_burden_pct_gdp` 0/22. Stale (> 3 y): IN/ID `rd_pct_gdp` 2020, AU 2021; RU/SA `energy_net_imports_pct` 2022. Consequence for scores: Promises category is `None` for 15 of 22 players (coverage floor), so the `allocation` and `jurisdiction` views are mostly `None` until slice 20 — the `learning` view has 4 of 5 categories. Known nit: `bind="query-params"` updates the URL on widget interaction but not on the callback's programmatic `session_state` write (map click) — cosmetic, revisit in P1.
- **Known data gaps (documented):**
  - **IMF DataMapper** may return Akamai 403 from some networks (verify from Adam's network before slice 20; SDMX WEO-vintage fallback).
  - **IMF DOTS** no longer exists on the new IMF portal — bilateral trade needs `IMF.STA,IMTS` (spike, post-gate).
  - **World Bank `mrv=1`** returns null latest rows — banned; adapters fetch a range and keep non-null years.
  - **WGI** needs `source=3` and ids `GOV_WGI_{RL,PV,...}.{EST,SE}`; no euro-area aggregate (member-mean, flagged, in slice 19).
  - **JP CPI** — FRED's OECD-MEI Japan CPI mirror discontinued 2021; no current FRED series.
  - **CN 10Y yield** — not in FRED.
  - **CN GDP** — annual only (`NAEXKP01CNA657S`), already-YoY format.
  - **CN government debt / GDP** — not in BIS Total Credit dataset.
  - **EU debt service ratio** — BIS DSR has individual member states only; no euro-area aggregate.
  - **2Y yields outside US** — yield-curve slope is US-only.
  - **UK / SE / CN CPI** — ~1 year stale via FRED's OECD-MEI cadence.

## Roadmap

| Slice | Goal |
|-------|------|
| **1** ✓ done | US short-term cycle: ETL + classifier + dashboard, end-to-end |
| **2** ✓ done | Tier-1 fan-out: CN, EU, UK, JP, SE for short-term cycle |
| 3 | Tier-2: IN, BR for short-term + long-term cycles |
| **4** ✓ done | Long-term debt cycle (BIS Total Credit, DSR) for Tier-1 |
| 5 | Big-cycle power index (8 measures, multi-country) |
| 6 | Currency lifecycle + wealth/values gaps |
| **7** ✓ done | Allocation-implication module (regime → asset-class tilts) |
| **8** ✓ done | World-map UI + simplified GUI: clickable Plotly choropleth, plain-language summary cards, framework explainer, progressive-disclosure detail expanders |
| **9** ✓ done | Editorial design pass: cream parchment + deep navy ink, Source Serif 4 / Inter Tight / JetBrains Mono typography, kicker rules, pull-quote, hairline-framed cards, repaletted phase colors (sage/ochre/terracotta/oxblood/slate), repaletted Plotly map |
| **10** ✓ done | Real-yield as master tilt multiplier — `real_rate_10y` becomes a cross-regime first-class tilt input, not just a Phase 6 distinguisher |
| **11** ✓ done | Calibration audit + per-country thresholds — DSR/debt/CPI thresholds derived from each country's own quantiles when ≥10y history; defaults as fallback; dashboard expander shows side-by-side delta |
| **12** ✓ done | Historical regime backtest — `replay.py` walks both classifiers across 35 years; dashboard step chart with Lehman/COVID/CPI-peak annotations |
| **13** ✓ done | Tier-2 fan-out (IN, BR) — closes "Slice 3 pending" placeholder; world map fully populated; back-compat preserved via TIER_1 + TIER_2 union |
| **14** ✓ done | SEK-anchored allocation view — small interest-rate-parity overlay biases USD-denominated foreign assets by home-vs-target real-rate differential; sidebar "View as: SEK / USD" radio defaults to SEK |
| **15–17** ✓ done | Growth × Inflation grid · HY-spread asset signal · big-cycle qualitative panel |
| **18** ✓ done | World Fundamentals Map data layer: 22-player registry, World Bank adapter, set-based upsert, percentile/category/view scoring, snapshot JSON, `dalio-fetch-fundamentals` + `dalio-score`, ADR 0001 |
| **P0** ✓ done | Fundamentals presentation plumbing: `st.navigation` (Cycles · Fundamentals), `theme.py`, callback-based map clicks (fixes the `session_state` write-after-instantiation bug + the DE/FR/IT/ES/NL rerun loop), selected-country outline, `snapshot.py` loader + `conftest.synthetic_snapshot_dict`, graphviz spike passed |
| **P1** ✓ done | Fundamentals page: map (Indicator / Category / Composite modes, quintile bins, no-data band, data-quality diamonds, selected outline, EU Members/bloc toggle), leaderboard (ProgressColumns, row-select), dense country table (mono values, percentile bar + rank r/21, trend glyph, tier badge), selection sync map ↔ leaderboard ↔ sidebar |
| **19** ✓ done | All 15 registry indicators defined; the 10 World Bank/WGI ones live (incl. derived world shares `exports_share_world` / `military_share_world`, WGI + standard errors, flagged EU member-means); `scripts/audit_coverage.py`; `INDICATOR_EXPLANATIONS` merged from the registry. 231/330 cells — IMF-sourced cells (growth forecast, fiscal balance, interest burden, gov debt outside BIS) wait for slice 20 |
| **P2** ✓ done | Purpose-view selector (Learning · Jurisdiction · Allocation · Moonshot) driving the composite map mode, leaderboard order and country card; weights line + "tells you / cannot tell you" caption per view; Pareto of gap-to-best-in-class (indicator / category level, 80 % rule). **MVP reached 2026-08-24, three weeks before the 09-14 gate.** |
| 20 | IMF DataMapper (WEO history + forecasts), interest burden, `gdp_growth_fwd5` |
| 21 | Pressure chains (6 rules, tier C) + views/pressures/cycle blocks in snapshot |
| P3 / P4 | DOT flowcharts + spillover pills + exposure map · Gapminder bubble with forecast markers |
| 22–25 | BIS Tier-3 extension · PWT/ECI static files · IMTS bilateral-trade spike · `jurisdiction_tier` export |

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-08-24 | **ADR 0001** — World Fundamentals Map: five first-principles categories, percentile rank among 21 (aggregate interpolated), mean category score with half-coverage floor, purposes as weight-vector views (no headline composite by default), uncertainty tiers A/B/C + data-quality flags carried per cell, one snapshot JSON as the only data↔UI coupling, storage unchanged | Pareto + Occam on Dalio's 18 determinants; n = 21 makes weighting false precision; four purposes must not leak into data. See `decisions/0001-fundamentals-map.md`. |
| 2026-08-24 | `upsert_observations` rewritten set-based (one SELECT + executemany), same `(inserted, skipped)` contract | Row-by-row SELECT cost ~8 s per 20k rows; fundamentals loads are bulk. Equivalence test against the old loop in `tests/test_upsert.py`. |
| 2026-08-24 | Cycle map/classifiers gated by `has_cycle_data` (cycle indicators present), not "any observation" | Fundamentals-only rows for Tier-3 players would otherwise run the classifiers on empty features. |
| 2026-04-27 | Frame project as decision-support for allocation tilts, not market-timing | Empirical evidence: macro-overlay strategies underperform passive after costs. Dalio's own All Weather doesn't time. |
| 2026-04-27 | 8-country basket with tiering | Tier 1: full Dalio framework relevance + complete data. Tier 2: major EM with strong but slightly thinner coverage. Drives UI confidence labels. |
| 2026-04-27 | SQLite + SQLAlchemy + long-format observations table | Simple, file-based, easy to reason about. Long format makes adding indicators trivial. |
| 2026-04-27 | Slice-by-slice (US short-term first) | Tracer-bullet vertical slice. Build end-to-end before fanning out. |
| 2026-04-27 | Rule-based stage classification (not ML) | Transparency over sophistication. User must be able to inspect why a stage was assigned. |
| 2026-04-27 | Drop Vietnam from basket | Data coverage gaps in BIS Total Credit, OECD productivity, WID inequality made Tier-3 stretch. Outside Dalio framework's reserve-currency-scale focus. |

## Recent Changes

| Date | Change | Files |
|------|--------|-------|
| 2026-08-24 | Slice P2 complete — MVP. `view_models.py`: `VIEW_LABELS`, `VIEW_CAPTIONS` (each states what the view can and cannot tell), `view_caption`, `weights_line`, `pareto_frame` (indicator / category level; gap, share, cumulative share, `crosses_80`), `pareto_caption`. `charts.py::build_pareto` (horizontal ink bars, mono end labels "gap · cum %", dotted rust 80 % rule, no secondary axis). `page.py`: `_purpose_view` segmented control (`purpose_view` in session state) feeds the Composite map mode, leaderboard order and country-card composite; `_render_pareto` with level toggle. Default map mode now Composite. Browser-verified: switching to Jurisdiction reweights everything, Pareto + caption render, no exceptions. 7 new tests (246). | `src/dalio/app/fundamentals/{view_models,charts,page}.py`, `tests/test_fundamentals_pareto_views.py` |
| 2026-08-24 | Slice 19 complete: full 15-indicator registry (IMF cells empty until slice 20). `worldbank.py`: 13 raw specs (WDI + WGI src 3 incl. standard errors), `derive_world_share` (country ÷ `WLD`, series_id `…÷WLD`), `derive_member_mean` (EU = mean of DE/FR/IT/ES/NL, ≥3 members, series_id `…:member-mean`). Pipeline stores raw + derived rows. `fundamentals.py`: `SE_INDICATORS`, SE siblings loaded into each cell's `se`; `load_history` now keeps one source per (country, indicator) and the LAST observation per year (quarterly BIS DSR). `views.INDICATOR_EXPLANATIONS` merged from the registry (cycle names win). `scripts/audit_coverage.py` (22 × 15 latest-year matrix, fill %, stale cells). Live: 231/330 cells. Fixed `_cell` SE lookup. 5 new tests (239); fixture `seed_synthetic` shared by conftest + scoring tests. | `src/dalio/data_sources/worldbank.py`, `src/dalio/pipelines/fetch_fundamentals.py`, `src/dalio/scoring/fundamentals.py`, `src/dalio/app/views.py`, `scripts/audit_coverage.py`, `tests/{conftest,test_worldbank_derived,test_fetch_fundamentals,test_fundamentals*}.py` |
| 2026-08-24 | Slice P1 complete: the Fundamentals page. `app/fundamentals/view_models.py` (pure pandas: `MapMode`, `map_layer` with selected-last ordering / aggregate cover at 0.45 opacity / bloc mode / no-data + data-quality points / hover with rank r/21 and tier, `iso3_to_player`, `leaderboard`, `coverage_confidence`, `country_table`, `html_dense_table` with HTML escaping, `exposure_counts`). `charts.py::build_fundamentals_map` (3 traces: binned choropleth with flat 5-segment ramp, no-data choropleth, diamond-open data-quality markers). `page.py`: control strip (Color by · Indicator/Category selector · Europe Members/EU bloc), map with `on_select` callback keyed through a session-state iso3→player map, leaderboard `st.dataframe(on_select, single-row)` mapping the sorted order back to iso2, country card (composite pull-quote, dq note, coverage confidence bar, dense table). Dense-table CSS + segmented-control restyle appended to `_inject_design_css`; `.streamlit/config.toml` pins the theme so canvas-rendered ProgressColumns use ink, not Streamlit red. Browser-verified: Korea map click and leaderboard row click both switch sidebar + card; no exceptions. 15 new tests (234). | `src/dalio/app/fundamentals/{view_models,charts,page}.py`, `src/dalio/app/streamlit_app.py`, `.streamlit/config.toml`, `tests/test_fundamentals_{view_models,charts}.py` |
| 2026-08-24 | Slice P0 complete: presentation plumbing. New `app/theme.py` (palette moved out of `streamlit_app.py`; `PCT_RAMP`/`EXPOSURE_RAMP` validated ramps; `plotly_base_layout`/`geo_layout`; `PLAYER_CENTROIDS`; `confidence_block`/`phase_swatch`/`legend_row`). `streamlit_app.main()` now: page config → CSS → masthead → `render_shared_sidebar()` → `st.navigation([Cycles, Fundamentals])`; the old body is `render_cycles_page()`. Map clicks handled in an `on_select` callback (`_on_cycle_map_select`) instead of a script-body `session_state` write + `st.rerun()` (the pre-P0 bug); selected country outlined in rust and drawn last. New `app/fundamentals/{snapshot,page}.py`: typed `Snapshot` + `load_snapshot`/`parse_snapshot` with contract validation (`SnapshotError`), mtime fingerprint cache key, placeholder page. `tests/conftest.py::synthetic_snapshot_dict` (built by the real `build_snapshot` on a seeded temp DB + one fake chain) is the executable data↔UI contract. 13 new tests (219). Browser-verified via Playwright. | `src/dalio/app/{theme,streamlit_app}.py`, `src/dalio/app/fundamentals/{__init__,snapshot,page}.py`, `tests/{conftest,test_fundamentals_snapshot,test_theme_and_map}.py` |
| 2026-08-24 | Slice 18 complete: World Fundamentals Map data layer. Registry → 22 players with derived maps and static flags (`countries.py` is the one place to add a country; `EUROZONE_ISO3` moved here, `views` re-exports). New `data_sources/worldbank.py` (paginated multi-country WDI/WGI, never `mrv`, disk cache; `big_cycle.fetch_gini` delegates). `upsert_observations` set-based with equivalence test. New `scoring/fundamentals.py` (IndicatorSpec registry with tiers, `percentile_rank` incl. aggregate interpolation, `trend_direction`, `category_scores`, `view_scores`, `build_snapshot`/`write_snapshot`). New pipelines `dalio-fetch-fundamentals` / `dalio-score`. `map_iso3_to_country_iso2`: DEU → "DE" (registry first), other eurozone → "EU". Cycle map iterates `CYCLE_COUNTRIES` via `has_cycle_data`. Live: 3 indicators × 22 players, 66/66 cells; snapshot 395 KB. Adversarial review pass (4 lenses + skeptic verification) found and fixed before commit: year-end `as_of` made the 5-year lag a 6-year lag (day-28 clamp); stale single-point series reported `trend="flat"`; `load_panel` could borrow another spec's source; aggregate `distance_to_best` could go negative; `upsert_observations` had no rollback (partial batch swept into the next commit); cycles-page click on DE/FR/IT/ES/NL would reset the selectbox and loop (`cycle_click_target`). 55 new tests (206 total). ADR 0001 + `decisions/README.md`. | `src/dalio/{countries,app/views,app/streamlit_app,data_sources/worldbank,data_sources/bis,scoring/fundamentals,scoring/big_cycle,pipelines/fetch_fred,pipelines/fetch_fundamentals,pipelines/score_fundamentals}.py`, `tests/test_{countries,views,upsert,worldbank,fundamentals,fetch_fundamentals}.py`, `decisions/0001-fundamentals-map.md`, `pyproject.toml`, `.gitignore` |
| 2026-04-28 | Slice 17 complete: big-cycle qualitative panel. New `src/dalio/scoring/big_cycle.py` loads two slow-moving series — World Bank Gini (per country, internal-disorder proxy) and IMF COFER USD share of allocated FX reserves (global quarterly, reserve-currency lifecycle). Neither feeds the classifier or `compute_tilts`. Rendered in a deeply-collapsed "The bigger picture" expander with two charts plus a static framework outline (6 stages of internal order, 8 measures of national power), explicitly labeled "Dalio's framework — not classified by this tool". Live: US Gini 41.8 (2024); COFER USD share 56.8% (2025-Q4, down 14pp from 70.8% in 2000-Q1). 8 new tests with mocked HTTP. Plan now fully shipped (10 → 17). | `src/dalio/scoring/big_cycle.py`, `src/dalio/app/streamlit_app.py`, `tests/test_big_cycle.py` |
| 2026-04-27 | Slice 16 complete: asset-price inputs (HY credit spread). New `src/dalio/scoring/asset_signals.py` computes z-score of latest `hy_spread` against country's 20y history (US-only — `BAMLH0A0HYM2` is a US ICE BofA index). `LongTermFeatures` gains `hy_spread`/`hy_spread_z`; `_vote_bubble` adds Phase 3 vote when z < −1.5 (complacency), `_vote_top` adds Phase 4 vote when z > +2.0 (distress repricing). Dashboard shows HY-spread metric with regime caption in the long-term card. Scope reduced from original plan: CAPE deferred (FRED lacks the 10y real S&P earnings; needs Yale spreadsheet integration), gold:bonds deferred (LBMA gold series discontinued on FRED). 9 new tests. | `src/dalio/scoring/{asset_signals,long_term}.py`, `src/dalio/data_sources/fred.py`, `src/dalio/app/streamlit_app.py`, `tests/test_asset_signals.py` |
| 2026-04-27 | Slice 15 complete: refactor allocation to growth × inflation grid. New `src/dalio/scoring/grid.py` introduces `GridQuadrant` enum (5 entries — 4 strict G×I quadrants + REFLATION hybrid per refined plan), `GRID_TILTS` canonical mapping, `quadrant_for_features()`. `SHORT_TERM_TILTS` now derived from `GRID_TILTS` via `STAGE_TO_QUADRANT` (back-compat preserved 1e-6). Stagflation (G↓I↑) gets explicit tilts (long_bonds −1.7). Dashboard "Growth × Inflation grid" panel renders the country's current `(real_gdp_yoy, cpi_yoy)` point on a 2×2 with quadrant tilt-direction labels and Reflation hybrid note. 14 new tests. | `src/dalio/scoring/{grid,allocation}.py`, `src/dalio/app/streamlit_app.py`, `tests/test_grid.py` |
| 2026-04-27 | Slice 14 complete: SEK home-currency overlay. New `src/dalio/scoring/currency.py`; `compute_tilts` accepts `home_currency` and `home_real_rate_10y` kwargs (stays DB-free per refined plan); `compute_country_view` does the home-country lookup. Sidebar "View as: SEK / USD" radio. Demonstrated effect for US in SEK view: gold +0.43→+0.24, long bonds 0.00→−0.13, equities +0.09→−0.01 (SEK +2.29% real rate vs US +1.02% means SEK appreciates → USD-denominated assets less attractive). Honest scope: IRP intuition, not a full FX model. | `src/dalio/scoring/{currency,allocation}.py`, `src/dalio/app/{views,streamlit_app}.py`, `tests/test_allocation.py` |
| 2026-04-27 | Slice 13 complete: Tier-2 fan-out (IN, BR) closes the dashboard's "Slice 3 pending" placeholder. New `TIER_2_SERIES` (FRED) and `TIER_2_TOTAL_CREDIT`/`TIER_2_DSR` (BIS) added without renaming Tier-1 constants — `specs_for_countries()` unions them transparently. Documented gaps: BR monthly unemployment (FRED OECD-MEI gap), IN/BR govt-debt-to-GDP (BIS WS_TC sector G 404 for emerging markets). Live IN: Transition (Bubble ↔ Debt outpaces) 26%; live BR: Top — peak debt service 42%. World map fully populated. | `src/dalio/data_sources/{fred,bis}.py`, `src/dalio/pipelines/fetch_bis.py`, `tests/test_fred.py` |
| 2026-04-27 | Slice 12 complete: historical regime backtest. `extract_features` and `classify` in both classifiers gain optional `as_of: date` parameter; `_latest` extracted to `_latest_at`. New `src/dalio/scoring/replay.py` with `replay_classifications(session, country, start, end, step="Q")` returning a 7-column DataFrame. Dashboard expander "Historical regime path" renders a Plotly step chart of long-term phase across 35 years with Lehman/COVID/CPI-peak annotations. The lens has a track record now. | `src/dalio/scoring/{long_term,short_term,replay}.py`, `src/dalio/app/streamlit_app.py`, `tests/test_replay.py` |
| 2026-04-27 | Slice 11 complete: calibration audit + per-country thresholds. New `src/dalio/scoring/{thresholds,calibration}.py` — `Thresholds` dataclass, `compute_country_thresholds()` builds from each country's own quantiles when ≥10y history, defaults to US-derived literals otherwise. CPI thresholds get a 2.5/3.5 floor for chronically-low-inflation countries. Long-term + short-term classifiers consume `Thresholds` (back-compat preserved). Dashboard expander "How thresholds are set for [country]" shows default vs per-country side-by-side with delta. Reveals real shifts: US `debt_extreme` 280 → 254.9, JP 280 → 401, SE `dsr_distress` 18 → 24.7. | `src/dalio/scoring/{thresholds,calibration,long_term,short_term}.py`, `src/dalio/app/streamlit_app.py`, `tests/test_calibration.py` |
| 2026-04-27 | Slice 10 complete: real-yield as master tilt multiplier. `_real_yield_multiplier(real_rate)` is a third additive layer in `compute_tilts` that biases gold/commodities/real-estate up and long bonds down whenever real 10y rate goes negative — across all phases, not just Phase 6. Read from `long_term.features.real_rate_10y` (no new arg). New `AllocationView.real_yield_regime` field exposes the label. Dashboard quick-read surfaces a one-line italic note in repression / mild-repression. | `src/dalio/scoring/allocation.py`, `src/dalio/app/streamlit_app.py`, `tests/test_allocation.py` |
| 2026-04-27 | Slice 9 complete: editorial design pass via Anthropic frontend-design skill principles. Committed to a single bold direction — "macro-research editorial" (Howard Marks memo / Bridgewater Daily Observation aesthetic): cream parchment background, deep-navy ink, Source Serif 4 display, Inter Tight body, JetBrains Mono numerals. Kicker labels + hairline rules replace section dividers. Pull-quote with rust left-bar replaces the bold-marked quick-read paragraph. Three summary cards rewritten as editorial frames with kicker eyebrows ("LONG-TERM CYCLE · PHASE N OF 7"), serif titles, italic decks, hairline confidence bars, table-row tilts. Phase colors repaletted from Tailwind brights (#16a34a, #dc2626, etc.) to muted painterly editorial (sage/olive/ochre/terracotta/oxblood/slate/charcoal/amber). Plotly choropleth landcolor + ocean + borders + fonts matched to page. Streamlit element overrides flatten the default Material card-shadow look. | `src/dalio/app/streamlit_app.py` |
| 2026-04-27 | Slice 8 complete: World-map UI + simplified GUI. Interactive Plotly choropleth (built-in ISO-3 boundaries, no GeoJSON deps), Eurozone expanded to 19 members for coherent block coloring, click-to-select via `on_select="rerun"`, metric switcher (phase/stage/caution/total-debt), 3 plain-language summary cards (long-term phase, short-term stage, top-3 tilts), "How these connect" framework expander, progressive-disclosure detail expanders. Plain-language `INDICATOR_EXPLANATIONS` tooltips on every `st.metric`. Discrete colorscale with z=-1 reserved for "no data" (light grey) vs z=0 "transition" (amber) — visually distinct categories. New `views.py` module (concept explanations + view-model functions). 14 new view tests. | `src/dalio/app/views.py` (new), `src/dalio/app/streamlit_app.py`, `tests/test_views.py` (new) |
| 2026-04-27 | Slice 7 complete: allocation-tilt mapper (`compute_tilts()`) maps short-term stage + long-term phase → 8 asset-class tilts via Dalio Growth×Inflation matrix + long-term phase risk overlay. Confidence-weighted both layers; transition states blend constituent-stage tilts via vote weights. Dashboard shows tilt table with directional arrows, magnitudes, reasoning, caution level, and a "tilts not weights" disclaimer. | `src/dalio/scoring/allocation.py`, `src/dalio/app/streamlit_app.py`, `tests/test_allocation.py` |
| 2026-04-27 | Slice 4 complete: BIS adapter (Total Credit + DSR via SDMX REST API, on-disk cache, retry-on-5xx), long-term debt cycle classifier (6 phases — Phase 5 ugly vs Phase 6 beautiful deleveraging distinguished by inflation regime + DSR), dashboard now shows both cycle layers per country with sector-debt sparklines and real-rate banner. 36 BIS series (34/36 working — 2 documented gaps). | `src/dalio/data_sources/bis.py`, `src/dalio/pipelines/fetch_bis.py`, `src/dalio/scoring/long_term.py`, `src/dalio/app/streamlit_app.py`, `tests/test_bis.py`, `tests/test_long_term_classifier.py` |
| 2026-04-27 | Slice 2 complete: Tier-1 fan-out (CN/EU/UK/JP/SE), 29 FRED series across 6 countries. Added `TIER_1_SERIES` map, `specs_for_countries()` filter, retry-on-transient-error, CLI country subset, freshness-audit + replacement-search helper scripts. Documented data gaps. | `src/dalio/data_sources/fred.py`, `src/dalio/pipelines/fetch_fred.py`, `tests/test_fred.py`, `scripts/{audit_freshness,find_replacements,find_jp_cpi,search_fred}.py` |
| 2026-04-27 | Slice 1 complete: rule-based classifier with Sahm rule + saturating confidence; full Streamlit dashboard | `src/dalio/scoring/short_term.py`, `src/dalio/app/streamlit_app.py`, `tests/test_short_term_classifier.py` |
| 2026-04-27 | First real FRED fetch — 56,950 US observations stored | `data/dalio.db` (gitignored) |
| 2026-04-27 | Initial scaffold | `pyproject.toml`, `src/dalio/{countries,storage/db,data_sources/fred,pipelines/fetch_fred,app/streamlit_app}.py`, `tests/` |
