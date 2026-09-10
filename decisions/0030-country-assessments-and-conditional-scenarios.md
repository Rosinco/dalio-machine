# ADR 0030 — Country assessments and conditional scenarios

Date: 2026-09-10
Status: accepted

## Decision and scope

After the verified evidence checkpoint and handoff, the user explicitly asked
to continue into clear country assessments and scenarios that can later
support sector and company analysis. Begin the analysis pivot now, using the
19 listing countries in ADR 0029. The remaining debt, allocator and funding
packages in ADR 0017 stay recorded gaps; this decision does not declare them
complete or require completing every collection before producing useful
descriptive analysis.

This first assessment layer reads verified numeric releases without changing
the source database, scoring population or company research. It produces
versioned JSON and readable country Markdown reports, together with a
cross-country index. The desktop remains a separate consumer; these exports
do not automatically replace its bundled macro snapshot.

## Evidence contract

Select the latest eligible release per expected source partition at an exact,
timezone-aware known-at cutoff, before examining its observations. Verify
complete immutable scalar rows and every artifact against the retained v2
acquisition bundle and original response bytes. The current projection is
not used for historical selection. An eligible legacy contract becomes an
explicit evidence gap; a corrupt selected release fails the build. A future
release cannot obscure or contaminate an earlier eligible release.

The five comparable core indicators are IMF real GDP growth,
general-government gross debt, fiscal balance, Fiscal Monitor primary balance
and current-account balance. The baseline is the prior calendar year, labelled
estimate/outturn in the retained vintage. The publisher path displays the
current calendar year and the following five years, preserving the collector's
explicit forecast convention. Missing years or endpoints stay missing; no
interpolation, last-value filling or forecast-to-outcome conversion occurs.
DataMapper does not supply the per-point native actual/estimate cutoff here:
these status labels follow the collector's calendar convention, not a claimed
publisher classification for every cell.

World Bank old-age dependency, net energy imports and R&D spending provide
historical structural context under their original years and source notes.
An editorial three-year age window controls whether these enter current
interpretations; it is not an official expiry date or a quality rating. The
full histories and missingness remain in the evidence export.

Available original Swedish debt-office evidence is added separately: the latest
eligible monthly central-government debt stock and two explicitly distinct ATR
measures, plus current-vintage annual gross-borrowing/redemption forecasts.
Native units, periods, dimensions, publication clocks and table/cell locators
remain intact. National central-government currency amounts never replace IMF
general-government percent-of-GDP ratios.

## Interpretation and scenarios

The consumer computes descriptive differences within the same indicator and
source series. Growth-rate and ratio changes are percentage points. Slower
positive real growth is not output contraction, and a falling debt/GDP ratio
is not proof of debt repayment or sustainability. Nominal growth, interest,
stock-flow and currency assumptions would be needed for a debt dynamics model.
WEO fiscal and Fiscal Monitor primary balances are not subtracted to manufacture
interest expense. Current-account signs require financing and investment
context rather than automatic favourable/unfavourable labels.

Publisher forecasts, descriptive arithmetic and Observatory conditional cases
are distinct. Cases cover weaker real activity, tighter funding conditions,
stronger activity and, when supported by a recent positive historical
net-import ratio, an energy-import shock. Their research horizons are editorial
windows, not estimated effect delays. Growth cases cite the available annual
publisher references covering their scenario window and name missing years.
A country's GDP path does not establish demand in foreign customer markets.

Every case identifies:

- conditional assumptions and a potential transmission mechanism;
- its reference evidence and 6-month to 3-year research window within the
  broader current-year plus five-year publisher path;
- signposts explicitly marked for monitoring, with uncollected inputs named;
- evidence that would invalidate or weaken the case;
- required company checks: real revenue/customer, asset, cost, currency,
  financing, maturity, hedging and contractual exposures;
- evidence gaps and the absence of calibrated probabilities or numerical
  stress forecasts.

No company implication follows solely from listing country or branch. These
cases form a structured research starting point for future sector/company
joins. They do not create composite risk scores, buy/sell verdicts, reviewed
institutional narrative claims or communications permissions.

Definitions and interpretation limits are grounded in the retained publisher
metadata and the official IMF WEO FAQ/current-account explanation, ECB monetary
transmission overview, and World Bank indicator definitions linked in each
snapshot's methodology. Coverage and predictive confidence remain separate;
the latter is not estimated by this release.

## Publication and validation

Each snapshot binds the exact evidence digest, full source lineage, cited
values/periods/releases and versioned methodology/implementation hash. A
content hash identifies an immutable output directory containing `snapshot.json`,
`index.md`, and `countries/<internal-country-code>.md`. Publish the complete
directory before updating `LATEST.json`. Validate unsafe output locations and
existing immutable contents before replacement. Repeating the same inputs,
cutoffs and implementation must reproduce the same bytes.

The source database is opened with SQLite read-only mode. Builds perform no
network acquisition and create no database tables, releases or claims. Unit
tests cover historical selection, source corruption, forecast gaps, stale
structural data, scenario horizons, native debt scope, citation identity and
atomic publication. The integrated suite and a source-to-output live audit
must pass before the completed assessment slice is committed and handed off.
