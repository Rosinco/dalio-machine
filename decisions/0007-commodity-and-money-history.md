# ADR 0007 — Commodity history and the money/liquidity domain

**Date:** 2026-09-08 · **Status:** accepted · **Slice:** 33

## Context

The Observatory had no structured commodity-price history. Gold and broad
commodities appeared only as allocation outputs, while the World Bank energy
dependency measure described physical import exposure rather than prices.
Policy rates, sovereign yields, credit and Swedish exchange rates covered part
of the price of money, but not broad-money quantities or central-bank balance
sheets.

Commodity breadth is useful because a single headline index can hide regional
energy, food, fertilizer or industrial-metal shocks. More columns must not,
however, give a large commodity family extra weight merely because it has more
published benchmarks. Money is also only *commodity-like*: modern deposits are
issuer liabilities and money has no single own-currency spot price.

## Decision

1. **Archive the complete official monthly commodity release.** The World Bank
   Commodity Price Data (Pink Sheet) is the canonical first source. Each run
   discovers the current official XLSX from the World Bank landing page,
   validates both `Monthly Prices` and `Monthly Indices`, and archives the
   original bytes by SHA-256 together with a generator- and schema-versioned
   deterministic series catalogue. Corrected catalogue generators may coexist
   beside the same immutable workbook bytes.
2. **Keep publisher semantics.** Each catalogue entry retains worksheet,
   benchmark label, semantic indicator, unit, quoted currency where applicable,
   monthly frequency, price/index basis, family and a curated-panel flag.
   Publisher blanks, ellipses and spreadsheet-error cells remain missing; they
   are never zero-filled or interpolated. Nonpositive prices and indices are
   quarantined as missing while the original publisher cells remain auditable
   in the archived workbook; the catalogue quality manifest records each
   quarantined worksheet cell, month, raw benchmark label/value and reason.
   Index semantics and base years come from each
   workbook column's unit or the index-sheet metadata, not a hardcoded base.
3. **Use global scalar partitions.** Commodity observations use entity `WLD`
   and one immutable release partition per native workbook series. Prices use
   `commodity_price_*`; family indices use `commodity_index_*`. The compatible
   current projection remains available without duplicating a world price once
   per country. Stable canonical IDs use explicit aliases for known publisher
   label drift while the catalogue preserves the exact displayed benchmark.
4. **Separate archive, analytical panel and presentation.** All source columns
   are retained. A family-balanced curated flag identifies a smaller research
   panel. No commodity series enters cycle classification, allocation tilts or
   a dashboard score in this slice.
5. **Model money as its own domain.** The initial Money, Liquidity & Currency
   catalogue contains US M2, Federal Reserve total assets, euro-area M3,
   Eurosystem total assets and Swedish M3. Each stays in its publisher-native
   denomination and frequency. Weekly balance sheets are not silently
   resampled and local-currency levels are never summed through current FX. The
   ECB weekly API exposes ISO-week keys rather than exact reporting dates, so
   Friday is the stored end-of-week representative; quarter-end and holiday
   reporting exceptions remain explicit rather than guessed.
6. **Use official delivery infrastructure with explicit provenance.** US
   series are Board of Governors releases delivered by Federal Reserve Bank of
   St. Louis FRED; euro-area series come from the ECB Data Portal; Swedish M3
   comes from SCB's official table sourced from Sveriges Riksbank. ECB native
   observation statuses and SCB's dataset-update timestamp are retained. SCB's
   publisher, M3 label, monthly reference period, stock/current-price basis and
   unadjusted status are validated against native response metadata.
7. **Do not manufacture one price of money.** Later analysis may separately
   derive broad-money growth, central-bank-assets growth, real rates, currency
   strength and funding stress. Existing policy rates, yields, credit and SEK
   FX remain separate evidence. A combined liquidity or investor regime needs
   another explicit decision and point-in-time validation.
8. **Preserve revisions honestly.** With no trustworthy row-level publication
   clock, retrieval time is the conservative availability time. Complete
   subsequent source histories append new vintages; they do not rewrite the
   earlier knowable state. A Pink Sheet workbook is one atomic release group:
   every native partition and current projection commits, or the whole workbook
   rolls back without a partial-success summary. Every partition's vintage
   label binds the workbook SHA-256 to the catalogue schema/generator, so a new
   official workbook or parser-semantic version remains visible even where a
   particular series' numeric rows are unchanged.
9. **Fail closed on incomplete history.** Each pinned series has an expected
   start, minimum observation count, native cadence and maximum latest-data lag.
   Any contraction from the latest stored release is rejected unless an operator
   explicitly confirms it with `--allow-contraction`. Every release carries a
   deterministic hash of the catalogue semantics used to interpret its values.
   The Pink Sheet additionally guards worksheet column counts, the 1960-01
   start, continuous monthly coverage, recency and stored-catalogue contractions.
   Read-only inventory reports stored and qualified-ready coverage separately
   and verifies that all current partitions share one release label and clock pair.

## Initial boundaries

- The Pink Sheet is a World Bank-compiled benchmark publication; some
  underlying assessments originate with exchanges or commercial price
  providers. It is canonical here for consistency and public reproducibility,
  not because every underlying quote is generated by the World Bank.
- Spot/reference prices explain macro pressure but are not futures investment
  returns. Roll yield, collateral return, fees and implementation are absent.
- US M2, euro-area M3 and Swedish M3 have different definitions. Compare
  within-series changes or explicitly normalized measures, not raw levels.
- The May 2020 US M2 definition break and euro-area changing composition must
  remain visible in interpretation.
- ECB ISO-week keys cannot encode exceptional exact reporting days. Friday is a
  consistent representative, not a claim that every statement was dated Friday.
- A stable machine-readable Riksbank total-assets history has not been verified,
  so Swedish central-bank assets remain missing rather than proxied.
- Funding-market scarcity, cross-currency basis and BIS global currency credit
  remain later additions.

## Consequences

The Observatory gains a long, inspectable supply/inflation history and the first
quantity-side monetary histories without pretending that heterogeneous series
are directly comparable. Storage increases because rolling workbooks and
complete release snapshots are retained. The result supports later breadth,
dispersion, real-price, SEK-translation and liquidity-regime research while
keeping unvalidated signals out of portfolio advice.

## Rejected alternatives

- Treat one broad commodity index as the complete commodity cycle.
- Add only investable ETF or futures-continuation prices.
- Store multiple benchmark grades as independent votes in a macro score.
- Call central-bank assets, broad money or credit the single “money supply.”
- Convert and sum national money stocks using today's exchange rates.
- Backdate current source histories to their economic observation dates.
