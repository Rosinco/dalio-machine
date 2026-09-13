# ADR 0042: Verified provider KPIs and expanded company list tools

Date: 2026-09-13

Status: Accepted by user direction; build and installation receipts are separate.

## Context

The user asked for a much broader KPI catalogue and equal expansion of list
features, using the financial data already downloaded. The first list release
provided 41 fields from the frozen research screen. Additional provider datasets
contain many more definitions, period selections and calculations, with known
metadata and currency defects that require reconciliation before display.

## Decision

Import the saved Börsdata screener and relative-history variants through a
reproducible exporter. Preserve exact provider KPI, calculation group, calculation
and source identities. Curate ambiguous labels and units against captured primary
documentation and arithmetic identities. Current per-share earnings, revenue,
book value and dividends retain report currency; prices, market capitalization
and enterprise value use established quote currency where applicable. Growth and
return variants carry percentage units even when the parent KPI is monetary.

The loader binds the derived data to the existing financial pack, taxonomy and
complete listing directory. SHA-256 checks cover compressed and decompressed
bytes. Bounded shards load only for selected columns and active conditions;
unrequested catalogue choices do not load their values. A failed or pending
request is distinct from an observed missing value and cannot satisfy a
"missing" filter. Conflicting non-null source values are withheld; a sole usable
observation can survive an empty overlapping scope. Formatted numeric text
companions must reconcile with their numeric value at their displayed precision.

Exclude NCAV KPI definitions 307–310 with documented scaling, currency and formula
issues. Withhold identified split inconsistencies and monetary values whose
currency cannot be established. Preserve genuine zero and negative values. A
condition such as P/E <= 15 includes negative P/E unless the user adds a positive
bound; it is not implicitly described as profitable or inexpensive.

Provider snapshot date is the download date, not an invented underlying quote or
report date. Relative historical positions remain positions; R12 histories are
not labelled standalone quarters. Provider calculations are descriptive context,
not new formal strategy gates or a point-in-time backtest. Missing coverage stays
visible. All saved listings remain accessible regardless of KPI availability.

Add up to 32 columns, 12 explicit numeric conditions, 20 named watchlists, 20
saved views, three sorting priorities and eight comparison selections. Offer
column presets for valuation, dividends, profitability, financial strength and
market data. Presets select columns; the existing cash observation lenses retain
their disclosed conditions. Add compact density, page size choices, value/source
details and CSV export of all filtered rows with units, currencies, dates,
status and selection identities. Native export creates a new Downloads file and
reports its actual path; failed writes are visible. Spreadsheet-like text is
escaped while real negative numbers remain numeric.

Preferences migrate in memory from `macro-atlas-company-lists-v1` into v2. Only an
explicit list action writes `macro-atlas-company-lists-v2`; the retained older app
keeps its original v1 preferences. Watchlist membership, comparison choices and
view settings persist together. The v2 default list is authoritative, so deleted
members cannot reappear through an obsolete legacy mirror.

The source-bound standard DCF, NPV, terminal and 30% purchase gauge remain frozen.
New provider price columns do not silently reprice those scenarios. Browsing,
filtering, comparing and exporting lists never mounts Value, migrates a draft,
saves a revision or changes reviewed assumptions. A deep dive remains the place
to reconcile owner cash, capital needs, valuation inputs and current price.

## Validation

Check exporter reproduction and input/output hashes, scope reconciliation,
metadata corrections, native/report currencies and documented exclusions. Verify
loader bounds, exact variant selection, missing/error states and preferences
migration. Exercise actual rendered provider values, watchlists, presets, sorting,
conditions, comparison, CSV file output and a full Windows process restart. Re-run
the existing company-list and research-screen checks. Record installed binary
identity separately from passing source/build checks and retain the previous app.
