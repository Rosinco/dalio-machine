# ADR 0029 — Company-country acquisition and original Swedish debt statistics

Date: 2026-09-10
Status: accepted

## Decision

The user authorized continuing Macro/Dalio collection for every country in the
downloaded Börsdata company universe, and explicitly requested first-hand
sources whenever available. This extends the acquisition scope of ADR 0017;
it does not enlarge a scoring population or redefine the fixed 48-stream debt
package in ADR 0020.

Keep an independently checked, hash-bound acquisition manifest for the 19
listing countries represented by 19,140 saved listings. The latest saved
snapshot contributes 17,593 listings and earlier-only records contribute
1,547. Listing country does not establish issuer domicile, revenue exposure,
physical assets or issuer uniqueness. Explicitly map Börsdata GB to internal
UK. The existing country registry, ranking population and percentile scores
remain unchanged.

Collect 14 World Bank and five IMF source series for each country. This is
19 publisher-specific series but 18 distinct metric names, because the current
account has both WB and IMF histories. World Bank WDI/WGI are official
harmonized publications; preserve their source organizations and methodology
notes instead of calling every value a national original. IMF projections
retain their WEO or Fiscal Monitor dataset and vintage. DataMapper does not
provide an observation-specific actual/estimate cutoff: the calendar-year
forecast convention is explicit, not a claim that every earlier observation is
final. Country-specific depth should use original statistical offices, central
banks and debt offices wherever available.

Sweden's first original debt tranche consists of eight January–August 2026
Riksgälden monthly statistical PDFs plus numeric sheets F10, F11 and F17 of
funding workbook 2026:1. The monthly reference date is the publisher's business
date, not automatically calendar month-end. The workbook's information cutoff,
publication date, historical anchor cells and previous-forecast columns remain
distinct. This is a selected numeric extract, not all tables or narrative
claims in those publications.

## Storage and acquisition contracts

- Preserve original response bytes, independent native re-parsing, explicit
  missingness and content-addressed evidence. Source errors cannot erase a
  previous history. Disable HTTP redirects and check delivery identity.
- Validate receipts and publisher updates against acquisition clocks. A
  historical reference or publication date cannot backdate database
  availability. Offline promotion preserves original acquisition clocks.
- Validate every selected partition before publishing the whole batch in one
  transaction. Reject history contraction and retrograde acquisition. Verify
  prior immutable rows and evidence before accepting an exact retry.
- Add append-only `national_debt_facts`: one typed cell with native units,
  dimensions, periods, status, label and source locator. Do not flatten security
  maturities, debt classes and forecasts into scalar observations. Bind raw
  response, native payload, missingness and catalogue artifacts to each release.
- Keep observed outcomes, forecasts and missing cells separate. Negative
  derivative exposures and refixing values can be valid. Reconcile the
  publisher's independently rounded totals without altering source values.
- Keep contractual maturity, average time to refixing (ATR), Macaulay duration,
  monthly mean and reference-date measurements separate. The January 2025
  change from duration to ATR is not a continuous series. General-government
  and central-government debt, original and residual maturity, face values and
  uplifted/current-FX stocks have different scopes.
- An annual gross financing/redemption forecast is not a complete outstanding
  debt cash-flow schedule. Security-level nominal maturity summaries explicitly
  name their covered instruments and exclude uncollected repayments/coupons.

The monthly Swedish stream closes one of the fixed 48 streams. The funding
workbook is an additional separately identified stream, not a second claim
against that denominator. Readiness therefore becomes 32/48: 31 harmonized
histories plus one national monthly stream, with 16 national streams remaining.
No composite refinancing risk score follows from these counts.

## Validation and retained results

The country bundle contains 361 country/source-series histories with 15,616
observations and no empty/error histories. Individual years can still be
missing. Belgium, Denmark, Estonia, Finland, Lithuania, Latvia, Norway, Poland
and Portugal previously had no macro observations in the working database.

The nine original Swedish documents contain 1,595 typed cells: 1,400 observed,
49 forecast and 146 explicit missing/structural cells. Thus 1,449 cells contain
finite values; the full cell count is not a count of finite observations.

The verified pre-expansion backup is
`data/backups/dalio-before-country-native-expansion-2026-09-10.sqlite3`.
Combined staging, live comparison and replay receipts are retained under
`data/artifacts/debt_refinancing/runs/2026-09-10-country-expansion/`.
Country sources are retained under `data/artifacts/company_country_macro/`;
original Swedish evidence is retained under
`data/artifacts/debt_refinancing/national/riksgalden/`.

Dated source-attributed analyses are
`data/snapshots/company-country-macro-2026-09-10-summary.{md,json}` and
`data/snapshots/sweden-native-debt-2026-09-10.{md,json}`. The nine-country
first-hand source discovery map is explicitly a collection backlog, not a
claim of national-series ingestion.

Combined staging and live audits both passed. Current scalar observations rose
from 329,606 to 336,949 (7,343 net additions); the immutable scalar ledger rose
from 409,184 to 424,800. There are now 3,455 releases, 2,207 artifact bindings
and 26 tables including the 1,595 native cells. Every pre-existing immutable
row and every unrelated current observation was preserved. Both databases
passed SQLite integrity and foreign-key checks. Offline replay created zero
additional releases. The integrated test suite passed all 1,307 tests, and
`ruff check src tests` is clean. The analyses were regenerated from the verified
live database; the original staged evidence and acquisition clocks remain intact.

These raw numeric collections do not create reviewed institutional narrative
claims or communications permissions. The existing independent review
contracts for those separate products are unchanged.
