# ADR 0022 — Sector and company observatories

Accepted by the user's “let's try it” and observatory-menu request, 2026-09-10.

## Scope and navigation

An Observatory selector switches between Macro, Sectors & branches, and Companies.
The Explore rail offers views appropriate to the selected observatory. The initial
route is Sweden → Materials → Forestry → Holmen, with a small named Nordic peer
panel. Country, branch and company selections persist across observatory changes.
The existing macro views and import workflow stay available from the same app.

The company slice provides search, annual financial history, a reviewed segment
comparison, dated saved research and links back to Dalio context. A country map
shows coverage by the selected company listings, explicitly distinguished from
operating assets, revenue exposure or total industry size. Plant coordinates,
land polygons and new industry forecasts are outside this slice.

The user explicitly requested a later extension of the deep-dive funnel on
2026-09-10: investigate each company's balance sheet and build a sourced physical
asset inventory. Record what each resource is, its location, ownership/lease or
joint-venture status, reporting date and evidence, then connect it to map markers
or land boundaries. Reconcile assets to balance-sheet disclosures where possible;
an asset count or book value must not imply an independently verified valuation.
This is a future research stage, not a claim that the current listing map locates
company property.

## Data and provenance

Börsdata remains authoritative for company records and its existing research;
Dalio remains authoritative for macro data. The desktop exporter reads selected
instruments and columns from the saved Parquet snapshot using predicate pushdown.
It writes only desktop-generated files, never the frozen or refreshed Börsdata
roots. The first package uses the saved 2026-08-10 financial snapshot; older prose
retains its own research date. No new investment verdict is calculated.

One canonical listing is selected per company. Reporting-currency amounts are
recovered using each row's currency ratio; units, financial periods and reported
dates remain explicit. The first cohort uses same-currency primary listings.
Margins and the existing EBIT / (year-end equity + net debt) return proxy are
descriptive calculations, not normalized earning-power or investment scores.
Missing values and nonpositive denominators produce unavailable readings. Peer
comparisons use a common full financial year and dimensionless ratios; amounts
in different currencies are never added or ranked together.

Version 2 research packages add an optional business document and include its
hash in package identity. Version 1 packages and their IDs remain supported.
The business document projects into a small catalogue and per-company resources,
so startup does not load full financial histories. Native imports use the existing
immutable archive. Bundled and imported releases expose the same resources.

## Acceptance checks

- Switch observatories, navigate to Holmen, compare peers and return to Sweden.
- Preserve full-year versus quarterly periods, currencies, zero and missing values.
- Display dates and limitations of archived research; no fresh verdict inferred.
- Version 2 imports survive restart; old packages show explicit business-coverage gaps.
- Windows desktop launch, all views and imported data work without external requests.

## Implementation checkpoint — 2026-09-10

Version **0.3.0** is installed in the user's Windows LocalAppData version folder
and opened through the updated Desktop shortcut. The installed binary matches the
Windows-tested SHA-256:
`10c955784ec6a793b98ffd0f5f7d92ea8215fdb31bb677d0877b46cf7760d458`.

Verification: 1,215 Python tests (including nine exporter contracts), 28 frontend
tests, eight native archive tests, real-package resource round trips, browser and
Windows flows. Both UI runs recorded zero external requests and runtime errors.
The Windows test imported legacy and v2 packages, restarted the process, reopened
Holmen and exported all three source documents. Archived text has its own bounded
size allowance, with a regression test using prose longer than metadata fields.

The included business catalogue is 26,498 bytes; selected company data and archived
research occupy 451,170 bytes before packaging. Financial coverage is 100 annual
and 200 quarterly rows. The v2 package ID is
`b5f357da5a4395f4496103c1541598fc241ee0da4a584ae0fce389b1dd28f81c`.
Generated research and binaries remain outside version control; canonical Dalio
and Börsdata working trees were left unchanged.
