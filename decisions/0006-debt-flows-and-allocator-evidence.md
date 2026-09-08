# ADR 0006 — Debt anatomy, financial transactions and allocator disclosures

**Date:** 2026-09-08
**Status:** Accepted

## Context

Headline government debt and a trade balance are not enough to reason about a
one-to-five-year macro risk. Refinancing risk depends on maturity, currency,
instrument and creditor residence. “Money moving” can mean a transaction, a
change in a marked-to-market position, a pension-system transfer or an asset
purchase; treating those as interchangeable creates false causal stories.
Large allocator reports also publish economically useful numbers in PDFs whose
rounding and non-additive exposure conventions must be preserved.

The observatory therefore needs more dimensions than the canonical numeric
observation key, while retaining the point-in-time and source-evidence rules in
ADRs 0004 and 0005.

## Decision

### 1. Sovereign-debt anatomy remains raw and source-scoped

Use the World Bank Indicators API's official QPSD source 20 for twelve
central-government series expressed as percent of GDP: total debt; original
short maturity; long-term debt due within and after one year; securities;
loans; domestic and foreign currency; domestic and external creditors; D1; and
D2A coverage.

Each country/native-series history is an independently replaceable immutable
release. QPSD is voluntary. A valid zero-row response is recorded in the run
summary as `not_reported`; it is never converted to zero, filled from another
definition or allowed to erase an earlier release.

### 2. Holder-sector data uses a typed fact table

Swedish central-government liabilities from SCB financial accounts retain the
issuer sector, instrument, counterpart holder sector, balance measure and unit
in `debt_holder_positions`. These dimensions must not be encoded into indicator
names or collapsed into the six-column observation table. Complete SCB table
responses append one release and point-in-time queries rank releases before
filtering cells, so an omitted holder is not resurrected from an older vintage.

### 3. Cross-border transactions are not inferred from positions

Use IMF BPM6 Balance of Payments financial-account transactions from 2000
onward. Keep asset acquisition, liability incurrence and the publisher's net
entry separate for direct investment, portfolio investment and other
investment, with explicit equity, debt, deposit and loan components plus
financial derivatives and the total financial account.

Positive asset acquisition means resident money placed abroad; positive
liability incurrence means foreign funding received. The IMF net series is
assets minus liabilities, so positive means net lending/outflow. The system
does not manufacture a transaction by differencing IIP, portfolio-position or
direct-investment stocks because valuation, exchange-rate and reclassification
effects would contaminate it. The euro-area aggregate is excluded from this
country basket when the official BOP selection does not publish it; member
countries are not double-counted into an invented aggregate.

### 4. Bilateral investment positions use a typed stock ledger

Use first-hand IMF PIP and DIP SDMX data for bilateral investment positions,
stored in `cross_border_positions` rather than the scalar observation table.
PIP retains reporter-resident outward portfolio assets at total-economy sectors
(`A`, `S1`/`S1`) for total, equity, long-term debt and short-term debt, with
annual and semiannual frequencies separate. DIP retains the reporter's own
annual (`DV_TYPE=O`) inward and outward total, equity and debt positions;
counterparty-derived `SCC` mirrors are excluded.

Reporter/counterpart native codes, direction, accounting basis, instrument,
frequency, source-native key, status and unscaled USD value remain explicit.
These are stocks. They must not be described as transactions during a period,
and changes may contain valuation, exchange-rate and reclassification effects.

### 5. Allocator PDFs retain both artifact and publisher semantics

Material AP-fund disclosures use `allocator_facts`, linked to an immutable data
release and a SHA-256-addressed copy of the official PDF. Each fact retains the
fund, as-of/flow period, record type, item code, amount as printed, normalized
SEK millions, exposure, basis, row role, physical PDF page, table heading,
extraction status, quality flag and note.

Pension-system and fund-reorganisation transfers are separate record types,
not investment inflows or purchases. Published totals and subtotals survive
even when rounded components do not add exactly. Publisher-non-additive
exposures are flagged rather than silently recalculated. The initial AP2, AP3
and AP4 H1 2026 transcription is model-checked against rendered pages; it is not
represented as named human semantic review.

### 6. Preserve missingness and provenance in analysis

Analysis may derive ratios or networks from these tables, but must expose the
source definition, date, vintage, unit, status and any quality flag. Derived
financial-flow graphs must label edges as transactions, positions or estimated
changes. A national total, a holder position and a fund allocation are related
evidence, not arithmetically interchangeable facts.

For every typed complete snapshot, point-in-time selection ranks eligible
releases before applying holder, instrument or other cell filters. Otherwise a
cell omitted in a newer vintage could be silently resurrected from an older
release. Where the publisher exposes no trustworthy release timestamp,
`available_at` is conservatively the retrieval time rather than the economic
reference period.

## Landed evidence at acceptance

- The first QPSD run stored 200 of 264 requested country-series histories over
  the 22-player basket. The remaining 64 combinations were valid voluntary
  non-reports, with no unresolved fetch errors; they remain missing.
- The SCB `TAB1203` release contains 8,349 quarterly positions from 1996-Q1
  through 2026-Q1: issuer `S1311`, balance measure `FM0103AS`, total/short-/long-
  term debt securities (`FL3000`/`FL3100`/`FL3200`) and all 23 counterpart
  sectors.
- The first IMF BOP transaction run refreshed 519 of 525 requested
  country-series partitions (25 series × 21 individual economies); the six
  unpublished partitions remain absent rather than filled.
- The AP2/AP3/AP4 H1 2026 reference contains 48 facts: 21 for AP2, 12 for AP3
  and 15 for AP4. Each is linked to the exact official artifact and physical
  page. This is one point-in-time release per fund, not a historical panel.
- The first official IMF PIP/DIP pull stored 127,970 bilateral position rows:
  92,529 PIP rows across all 168 requested partitions and 35,441 DIP rows
  across 123 of 126 partitions. Saudi Arabia's outward DIP total, equity and
  debt were not reported; no request failed. Complete reporter/native-series/
  frequency releases support release-first as-of queries.

## Consequences

The database can now distinguish debt quantity from refinancing and funding
composition, and gross cross-border placement from gross foreign funding. It
can also answer what the Swedish AP funds reported holding without mistaking a
2026 administrative consolidation for a market flow.

The richer schemas require source-specific loaders and downstream joins. QPSD
and PIP/DIP coverage is necessarily uneven, and a current AP allocation is not
yet a time series. Banking claims, human-verified central-bank report
conclusions, broader pension-fund history and debt cash-flow schedules remain
separate evidence blocks; none may be imputed from the new transaction series.

## Rejected alternatives

- Encode every debt-holder dimension into a very long indicator string.
- Treat an absent voluntary QPSD cell as zero or forward-fill it silently.
- Call a change in a financial position a capital flow.
- Net asset and liability transactions before storage.
- Use counterparty-derived DIP mirrors as though they were reporter-published
  observations, or treat a change in PIP/DIP stock as a reported transaction.
- Count AP1/AP6 reorganisation transfers as investment purchases.
- Correct rounded or non-additive publisher tables without preserving the
  published values.
