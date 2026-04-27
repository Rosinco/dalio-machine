# Project Context

## Overview

dalio-machine is a macro-cycle dashboard built on Ray Dalio's economic machine framework. For 8 economies (US, CN, EU, UK, JP, SE, IN, BR), it pulls macro indicators from FRED, BIS, IMF, OECD, World Bank, WID, and SIPRI, then classifies cycle stage with rule-based logic. Output is a Streamlit dashboard for **asset-allocation decision support** — informing diversified-portfolio tilts based on regime indicators, *not* market-timing signals.

The framework is treated as a descriptive lens, not a predictive oracle. Per Dalio's own writing and academic evidence on macro-overlay strategies, this dashboard surfaces regime state with explicit confidence — it does not output buy/sell signals.

## Tech Stack

- **Runtime:** Python 3.12
- **Framework:** Streamlit (dashboard), SQLAlchemy 2.x (storage), pandas (compute)
- **Test Framework:** pytest with mocked HTTP (pytest-mock + responses)
- **Key Dependencies:** fredapi, requests, pandas, sqlalchemy, streamlit, plotly, python-dotenv

## Architecture

- **Entry Points:**
  - `dalio-fetch-fred` (CLI ETL) — `src/dalio/pipelines/fetch_fred.py:main`
  - `dalio-app` (Streamlit dashboard) — `src/dalio/app/streamlit_app.py:main`
- **Module Structure:**
  - `src/dalio/countries.py` — 8-country tiered registry
  - `src/dalio/data_sources/` — one adapter per provider (FRED first; BIS, IMF, OECD, WB, WID, SIPRI later)
  - `src/dalio/indicators/` — cycle-layer computation (short-term, long-term, big-cycle)
  - `src/dalio/scoring/` — stage classification + asset-allocation map
  - `src/dalio/storage/` — SQLite schema + session helpers
  - `src/dalio/pipelines/` — ETL orchestration (one pipeline per data source)
  - `src/dalio/app/` — Streamlit dashboard
- **Data Flow:** API (FRED/BIS/IMF/...) → DataSource adapter → long-format DataFrame → SQLite via SQLAlchemy → indicator compute → stage classifier → Streamlit chart

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

## Indicator Catalogue

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

- **Working:** Slices 1 + 2 + 4 + 7 end-to-end for all 6 Tier-1 countries. Two cycle classifiers + allocation-tilt mapper that translates regime states → asset-class deviations from a default diversified base.
- **Tests:** 75/75 passing.
- **Live short-term cycle (2026-04-27):** US Transition (Reflation ↔ Inflationary peak) 29% / CN Expansion 42% / EU Inflationary peak 29% / UK Inflationary peak 33% / JP Transition (insufficient CPI) 0% / SE Expansion 42%.
- **Live long-term cycle (2026-04-27):**
  - **US** — Transition (Reflation/financial repression ↔ Bubble) 32% (debt 250%, fell 40pp/5y as inflation eroded ratio)
  - **China** — Top — peak debt service 60% (debt 296%, DSR 18.8% stretched)
  - **Eurozone** — Bubble 33% (debt 240%, late-cycle leverage zone)
  - **UK** — Reflation/financial repression 41% (debt 219%, fell 86pp/5y with CPI 3.4% — beautiful deleveraging)
  - **Japan** — Top — peak debt service 37% (debt 357% — extreme zone)
  - **Sweden** — Deleveraging 44% (DSR 23.3% — household distress, real-estate stress)
- **In Progress:** Slice 3 (Tier-2 fan-out: IN, BR for short-term + long-term cycles).
- **Known data gaps (documented):**
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

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-27 | Frame project as decision-support for allocation tilts, not market-timing | Empirical evidence: macro-overlay strategies underperform passive after costs. Dalio's own All Weather doesn't time. |
| 2026-04-27 | 8-country basket with tiering | Tier 1: full Dalio framework relevance + complete data. Tier 2: major EM with strong but slightly thinner coverage. Drives UI confidence labels. |
| 2026-04-27 | SQLite + SQLAlchemy + long-format observations table | Simple, file-based, easy to reason about. Long format makes adding indicators trivial. |
| 2026-04-27 | Slice-by-slice (US short-term first) | Tracer-bullet vertical slice. Build end-to-end before fanning out. |
| 2026-04-27 | Rule-based stage classification (not ML) | Transparency over sophistication. User must be able to inspect why a stage was assigned. |
| 2026-04-27 | Drop Vietnam from basket | Data coverage gaps in BIS Total Credit, OECD productivity, WID inequality made Tier-3 stretch. Outside Dalio framework's reserve-currency-scale focus. |

## Recent Changes

| Date | Change | Files |
|------|--------|-------|
| 2026-04-27 | Slice 7 complete: allocation-tilt mapper (`compute_tilts()`) maps short-term stage + long-term phase → 8 asset-class tilts via Dalio Growth×Inflation matrix + long-term phase risk overlay. Confidence-weighted both layers; transition states blend constituent-stage tilts via vote weights. Dashboard shows tilt table with directional arrows, magnitudes, reasoning, caution level, and a "tilts not weights" disclaimer. | `src/dalio/scoring/allocation.py`, `src/dalio/app/streamlit_app.py`, `tests/test_allocation.py` |
| 2026-04-27 | Slice 4 complete: BIS adapter (Total Credit + DSR via SDMX REST API, on-disk cache, retry-on-5xx), long-term debt cycle classifier (6 phases — Phase 5 ugly vs Phase 6 beautiful deleveraging distinguished by inflation regime + DSR), dashboard now shows both cycle layers per country with sector-debt sparklines and real-rate banner. 36 BIS series (34/36 working — 2 documented gaps). | `src/dalio/data_sources/bis.py`, `src/dalio/pipelines/fetch_bis.py`, `src/dalio/scoring/long_term.py`, `src/dalio/app/streamlit_app.py`, `tests/test_bis.py`, `tests/test_long_term_classifier.py` |
| 2026-04-27 | Slice 2 complete: Tier-1 fan-out (CN/EU/UK/JP/SE), 29 FRED series across 6 countries. Added `TIER_1_SERIES` map, `specs_for_countries()` filter, retry-on-transient-error, CLI country subset, freshness-audit + replacement-search helper scripts. Documented data gaps. | `src/dalio/data_sources/fred.py`, `src/dalio/pipelines/fetch_fred.py`, `tests/test_fred.py`, `scripts/{audit_freshness,find_replacements,find_jp_cpi,search_fred}.py` |
| 2026-04-27 | Slice 1 complete: rule-based classifier with Sahm rule + saturating confidence; full Streamlit dashboard | `src/dalio/scoring/short_term.py`, `src/dalio/app/streamlit_app.py`, `tests/test_short_term_classifier.py` |
| 2026-04-27 | First real FRED fetch — 56,950 US observations stored | `data/dalio.db` (gitignored) |
| 2026-04-27 | Initial scaffold | `pyproject.toml`, `src/dalio/{countries,storage/db,data_sources/fred,pipelines/fetch_fred,app/streamlit_app}.py`, `tests/` |
