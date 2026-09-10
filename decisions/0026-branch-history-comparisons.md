# ADR 0026 — Branch histories, benchmarks and saved comparisons

Accepted 2026-09-10. The user approved the proposed first branch-analysis slice:
historical bubbles, branch median and spread, synchronized charts/table, and saved
comparisons with notes and data versions. The intended desktop workflow continues
through testing, Windows build and installation.

## Presentation and interpretation

**Sectors & branches → Compare** opens a wide analysis workspace. Fiscal year is
the X-axis; the Y-axis selects an existing verified financial amount or ratio.
Up to eight listings have stable identity colours, with a shared year/focus across
bubbles, three supporting ratio histories and the selected-listing table. Clicking
a bubble selects its listing and year; the table opens the exact listing's existing
financial statements. A year slider and playback support historical exploration.

The benchmark uses every valid listing in the branch's visible country/download,
reporting-currency and fiscal-closing-month filters. Changing the coloured selection
does not change the benchmark. Each listing has equal weight; cross-listings can
repeat an issuer. This is the current saved directory applied to historical reports,
not a reconstructed historical industry universe or an investable backtest.

Median and quartiles use linear interpolation between sorted observations (R7).
The middle-50% band requires at least four observations and is absent across missing
years. The annual coverage table exposes n/filtered-listings for every year, including
zero-coverage years. Negative values and zero are preserved; unsupported ratios have
positive-denominator guards. Annual periods outside 330–400 days are excluded. Fiscal
year labels can have different ending dates; exact dates remain visible. No stale
partial median is published during loading or after a failed batch.

Monetary measures require a single reporting currency. The existing normalization
divides stored amounts by the saved positive currency ratio; missing conversion
remains missing. This slice does not introduce common-currency or inflation-adjusted
series. Supporting ratio charts retain the same currency and directory filters.

## Bubble size and deferred measures

Equal-size points are the default. The user can explicitly size bubble **area** by
total assets or revenue, in a single reporting currency. Diameter is proportional
to square root of value, using one maximum over the displayed companies and years.
No minimum-size floor distorts ratios; missing/non-positive size observations are
omitted and explained in the table. These are quantities, not good/bad assessments.

Market cap remains visibly unavailable pending verified history. Börsdata contains
historical share counts, but the source project's `DATA.md` documents inconsistent
split bases between historical prices and shares (including Johnson Controls 13331).
Multiplying retroactively split-adjusted prices by as-reported share counts would
produce false capitalization. A same-date, same-basis share/price series and currency
definition must be verified before enabling this mode. No market-cap proxy is labelled
as market cap, and no new data are written to Börsdata's sealed valuation workflow.

The existing annual pre-tax return-on-capital proxy remains labelled as such; it is
not ROIC. Investing cash flow is not CAPEX. FCF keeps Börsdata's provider definition,
excluding lease principal and interest, and is not owner earnings. Separately verified
ROIC, CAPEX, market cap, historical universes, event overlays, scenarios and physical
assets remain later work.

## Data access and persistence

The immutable v0.6 financial SQLite pack and all research/taxonomy identities remain
unchanged. A read-only annual endpoint accepts 1–32 distinct listing IDs. Rust verifies
the original company's full payload and coverage before projecting annual rows, with
an 8 MB response cap. The browser preview checks file/payload identities and uses
the same frontend annual validation and normalization. The frontend keeps only the
requested branch's annual measures, caps a branch at 100,000 reports and cancels stale
branch results. It does not transfer quarterly histories or load all 19,140 listings'
reports into the WebView. No live service is needed.

Saved comparisons use versioned, bounded JSON in the app's persistent WebView local
storage (maximum 100 saved views, 8 listing IDs/view, 10,000 characters/notes and
2 million JSON characters).
Saving is explicit, writes one atomic storage value and reports storage errors.
Invalid existing storage is retained instead of silently reset. The saved record
includes research ID, financial-pack hash, taxonomy hash, branch, selection, focus,
metric, size, filters, year range, selected year, name, notes and creation timestamp.
Reopening requires exact matching source versions and available listing identities;
unmatched saved views are visibly disabled. The records survive app restart and
version upgrades using the stable app identifier, but are not part of research/financial
pack exports; backup/import of personal comparison settings remains future work.

## Acceptance and verification

- Independently reconcile the displayed FCF median and selected figures to saved
  annual source rows, including reporting-currency and period-length filters.
- Test quantiles, missingness, zeros/negative values, size-area ratios, fiscal month,
  currency isolation, annual projection limits/checksums and saved-version binding.
- Exercise a real canvas bubble click, linked year playback, empty/out-of-scope
  observations, cancellation, branch navigation and correct company drilldown.
- Load the largest branch (Mining, 1,745 listings) with bounded annual requests.
- Preserve Swedish notes/settings across reload and native process restart; refuse
  to reopen a comparison against a different financial pack.
- Run the required Python/lint, frontend and Rust checks, complete browser/native
  offline regressions, then install and start the Windows update.

Validation passed: 1,206 core Python tests, 32 desktop Python tests, 51 frontend
tests, 14 research-store Rust tests and five financial-store Rust tests. Complete
browser and Windows flows passed without external requests or runtime errors.
The native suite verified exact 96 MB financial export/import and a real process
restart restoring both the financial pack and the saved comparison's Swedish notes,
filters, metrics, listing focus and selected year. A saved view using another pack
remained disabled.

The full browser run opened forestry annual data in 186 ms and Mining (1,745 listings)
in 1,351 ms; Windows measured 119 ms and 1,359 ms respectively on this machine.
These are local test timings, not a general performance guarantee. Company financial
switches and the existing macro, taxonomy, research-library and statement regressions
also passed. The new workspace fits a 1,100 × 760 viewport without page overflow.

The Windows executable is 11,036,672 bytes with SHA-256
`cb6873746d62543100780e8a9b6061d6c6c77259a45929d27420bd8a3cf2ad41`.
The financial pack remains
`a0c53dad1a0a726ee955d1e3a47f1d011b272b610e22652cef30e1bdc1c6d005`;
the research JSON's full-file checksum remains
`4e490049afc437a72b60c4684c282666996b5eb6cebe4dc3ead4188c03ab9889`.

Installed and started at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.7.0\Macro Atlas.exe`.
The existing `C:\Users\Adamb\OneDrive\Desktop\Macro Atlas.lnk` now targets this
version. Installed executable and financial-pack SHA-256 hashes match the tested
files exactly. Earlier version folders are retained. The complete browser suite
passed 60 flow checks; Windows passed 63, including saved comparisons after restart.
