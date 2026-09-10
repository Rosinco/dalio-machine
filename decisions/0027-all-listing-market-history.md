# ADR 0027: Dated market-cap history across the company directory

Date: 2026-09-10

Status: Accepted

## Context

The user asked to integrate the audited Börsdata market-cap equivalent for every
downloaded company. The branch explorer already supports annual financial measures,
but v0.7 had no market-cap series or market-cap bubble size. The source audit in
ADR 0026 found existing price × reported-share calculations and quality checks.

The frozen screener panel covers only part of today's 19,140-listing directory.
Its historical currency helper also has static fallbacks. Reusing that panel alone
would miss listings and obscure which currency observations supported a valuation.
Saved raw stock-price files contain both company closes and historical FX instruments.

## Decision

Version 0.8 extends the immutable financial companion to format v2. Every listing
retains market coverage, including listings with no reports or usable valuations.
Every usable annual report has one corresponding market record, with nullable
values, source identity, price/share/FX inputs, dates and explicit quality flags.
Existing annual and quarterly financial payloads remain unchanged. Original v1
financial packs and v1/v2/v3 research packages remain readable.

### Valuation definition

- Use the reported share count (millions) and first valid daily close on or within
  30 calendar days after annual-report publication, bounded by the saved download.
  Both inputs come from the same download as the selected annual financial report.
- The resulting value is in millions of the listing's quote currency. Fiscal-year
  labels describe the report; they do not imply a fiscal-year-end valuation.
- Convert to SEK using an observed direct CCY/SEK rate or a simultaneous USD cross
  (`USD/SEK ÷ USD/CCY`). Both cross legs must share a date. The FX date must be at
  or before the stock close and at most seven calendar days old. SEK uses identity.
  No fixed rates, future rates or unlimited forward fills are allowed.
- Missing publication dates, shares, closes or currency leave local values blank.
  Missing FX alone leaves the local value available and SEK blank.
- Reuse the source share-basis check: a share-count change of at least 1.5× or at
  most 1/1.5×, while both revenues and equity remain within 25%, marks a potential
  basis break. Missing/nonpositive prior accounting amounts do not veto detection.
  Years before the last detected break are withheld. This is a retrospective
  heuristic over the saved source history, not a point-in-time corporate-action audit.
- Withhold scale suspects (positive-profit P/E below 1 or positive-equity P/B below
  0.05), annual periods outside 330–400 days, and preference, receipt or unit
  instruments (source types 1, 9, 10) pending a reviewed share basis.
- Retain the raw inputs and review reasons even when the calculated value is withheld.
  Checks detect some anomalies; they do not certify every adjustment or corporate
  action. Separate ordinary share listings retain separate IDs. Values do not
  consolidate share classes into a verified issuer-level total.

The existing report source hashes bind the share input; separate instrument and
price file hashes bind quote currency, instrument type and price/FX sources. The
export also records the source basis-check code hash, method version and original
financial-pack identity. Börsdata files are read through its validated reader and
never modified. Export processes company prices in batches of 256 IDs.

### Views and storage

Company pages show local/SEK history, coverage, a dated latest fiscal observation,
and a table with exact valuation dates, shares, prices, FX and source/review notes.
Missing and withheld observations remain gaps. Unscored values use identity colours.

Branch comparisons default to market-cap bubble area in SEK when a v2 companion
is available. Equal size, assets and revenue remain selectable. Market cap is also
a Y-axis measure. SEK market-cap size with a ratio Y-axis, or market cap on both
axes, can span reporting currencies. Nominal financial monetary measures still
require a selected reporting currency. Bubble sizes use one scale across the period.
A missing size omits its bubble but does not remove an otherwise valid financial
Y-axis observation from the whole-branch median/IQR. Table/tooltips expose valuation
dates. Saved comparisons preserve the chosen size and measure with exact pack IDs.
Older saved comparisons retain their original data binding and notes.

The native store validates all company market records before publishing an import.
Both native and TypeScript validators enforce annual/source alignment, dates, units,
FX direction, arithmetic, known flags and reconciled coverage. Annual branch reads
remain bounded to 32 companies, omit quarterly payloads, and include market records.
Only the selected branch enters comparison memory. The SQLite companion remains
separate from the portable research JSON and supports existing offline import/export.

## Export and verification

The 2026-09-10 export is bound to taxonomy
`cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff`.
Its financial-pack SHA-256 is
`1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a`
and its size is 109,862,912 bytes.

| Coverage | Count |
|---|---:|
| Directory listings | 19,140 |
| Annual market records, including gaps | 268,884 |
| Usable local and SEK valuations | 159,486 |
| Listings with usable history | 16,895 |
| Listings with no usable valuation | 2,245 |
| Observations with a quality review flag | 38,648 |

Missing and quality reasons can overlap. The generated `market-coverage-audit.json`
records each reason count and exact source files. Existing financial coverage stays
at 268,884 annual and 603,720 quarterly reports across 18,943 listings, with 1,174
withheld source rows. Comparing every company with the original v1 pack confirmed
that all prior report payloads are unchanged.

Raw-source reconciliation includes Holmen FY2024: 157.668 million reported shares ×
SEK 420.4 on 2025-01-31 = SEK 66,283.6272 million. Additional cases reconcile EUR,
CAD and GBP direct FX and PLN simultaneous USD cross-rates to the saved raw files.
These are historical observations, not current quotes. The native real-pack verifier
checked all 19,140 company records in about 4.8 seconds in this development environment.

Python, TypeScript and Rust tests cover missing inputs, price/FX date windows,
scale and share-basis flags, units, tampering with otherwise valid checksums,
coverage, v1 compatibility and benchmark denominators. Browser and native flows
exercise company currency switching, review gaps, market-cap branch medians,
saved selections, offline transfers and process restart. Test artifacts and the
generated companion are intentionally gitignored. Verification passed 1,251 Python,
66 TypeScript and 22 Rust tests, the full browser suite, and the final native Windows
suite with zero external requests or runtime errors. One intermediate native test
lost its test window during import; a complete retry passed and failure-state logging
was added. The installed v0.8.0 executable matches the tested binary SHA-256
`7858e064a4b9410289c633ba86d364c26d4bfa84bf07ac873cbdc5f12bce313f`.

## Limits and later work

This implements annual publication-window valuations, not daily price history,
verified year-end market caps or a reconstructed point-in-time issuer universe.
The saved directory can repeat issuers and does not fully reconstruct delistings
or historical branch membership. Price, report and valuation dates remain visible.
Reviewed corporate-action/share-class reconciliation can improve coverage later.
ROIC, separately verified CAPEX, balance-sheet deep dives and physical asset locations
remain distinct later work; investing cash flow is not silently relabelled CAPEX.
