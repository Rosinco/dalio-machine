# Atlas research workflow and business overview — 2026-09-13

The intended reader understands basic economics and should be able to follow a
company from discovery to a considered valuation without learning Atlas internals.
Version 0.22.0 restructures the main research screens around that sequence. It
supersedes the initial 0.21.0 guide and layout pass.

## Navigation and continuity

Fresh profiles open Companies → Lists. Desktop has visible Companies, Industries
and Macro navigation; small screens use the same choices in a selector. Entering
Companies opens Lists, while searching or opening a company name opens Financials.
The company rail groups discovery (Lists, Evidence screen) separately from analysis
(Financials, Valuation, Peers, Research notes, Macro context). A shared company bar
keeps identity, industry and Back to Lists available throughout analysis, including
Valuation. Reloading restores the last company section in atlas.preferences;
existing list settings remain separate from valuation drafts.

## Back and Forward — 0.22.1

The top-left Back and Forward buttons retrace actual visited screens, including
company sections, peer companies, branch comparisons and country Macro screens.
Alt+Left and Alt+Right provide the same actions. Destination names appear in the
button tooltips. Buttons are disabled when that direction has no history.
Back to Lists and Company profile remain explicit destination links.

Navigation restores the company, branch, country, section, selected chart controls,
list/directory search and page, open disclosures and scroll position. Branch
comparison selections and unsaved notes stay available while exploring companies.
Valuation evidence tabs and in-page chart links participate in the same history;
clicking the same chart link again scrolls there without adding a duplicate entry.
Choosing another destination after Back discards the abandoned forward path.

This is presentation history, not Undo: authored forecasts, saved studies, list
preferences and company notebooks are never rolled back. The last screen remains
in atlas.preferences; history and temporary presentation choices are session-only,
with up to 80 prior/forward entries. Restarting Atlas starts a fresh history.
Choosing another research release clears history and temporary presentation state
so a return does not cross source releases. On narrow screens the header remains
available while scrolling, with chart anchors offset below it.

## Lists: results first

The search, watchlist scope, saved view and Columns action form the primary row.
Screening filters, Column sets and View tools have explicit disclosure buttons;
secondary controls no longer consume the whole first screen. Active constraints
are visible as removable chips, including KPI ranges and numeric conditions.
Min/Max inputs remain beneath every numeric column, including at zero matches.
The global list has no misleading return to an arbitrary company. Sector Lists
retain a Branch overview link. Full filtered CSV and saved-view behavior continue.

Information buttons explain a KPI's meaning, period, calculation and unit. Clicking
a cell adds its actual value, source context and missing reason. More detailed
methodology remains expandable, with concise interpretation near the results.

## Financials: understand the business before setting a price

The default page presents reported figures in this order:

1. **Revenue and profitability:** a revenue chart, then operating-profit and
   operating-cash margins. Operating profit also appears in the summary figures.
   Revenue is sales; a margin relates an amount to those sales. Explain cash
   timing instead of assuming profit equals cash.
2. **Cash generation and investment:** operating cash/provider free cash and
   investing cash. Explain whether cash is being generated or absorbed, and why
   acquisitions or disposals can make a period unusual.
3. **Assets and financing:** total/tangible/intangible assets, then equity, net
   debt and cash. Total assets includes components; net debt already reflects
   cash under its source definition. These lines must not be added together.

Six charts are visible by default, with nearby units, plain explanations and
annual/quarterly controls. Summary figures keep their actual period and currency.
Missing years, quarters or values stay gaps; zero and negative values remain
visible. Monetary charts use one selected reporting currency, and ratios retain
their meanings. Asset book values are not sale valuations or evidence of asset
quality. Return-on-capital remains an explicitly labelled annual pre-tax proxy.

A separately verified **Capex** panel uses the already bundled Börsdata KPI 64:
latest provider calculation, five-year mean and ten annual history positions.
It reuses the existing financial/taxonomy binding, checksums, currency boundaries
and missing-value rules. The source did not supply underlying fiscal dates for
these variants, so they are not aligned with dated statement charts, and they do
not change with the statement annual/quarterly selector. Their source date is a
KPI snapshot date. The panel is available even without usable statement history.
Investing cash includes acquisitions, financial investments and disposals; it is
not a substitute capex measure. Maintenance/growth capex remains unseparated.

The detailed measure selector, income statement, balance sheet, cash-flow table,
market history and source records remain accessible below the overview. Coverage
and technical metadata are expandable. Available business descriptions reuse the
five saved profiles' segments, overlap text and original source date; other
companies receive no invented description.

## Valuation: see the result, then inspect assumptions

A compact row identifies reviewed/starter state, dated price basis and investment
size. Low/Mid/High present value and NPV cards lead, followed by cash-flow and
DCF/NPV charts. Payback details expand within each scenario. Source/model history,
researched inputs and the historical baseline sit in a labelled disclosure.
Assumption forms remain editable through Edit price & assumptions or the section
links, with required-input messages linked to the forms. The personal investment
and company-millions chart scales stay explicit. Terminal assumptions, purchase
margin, business/capital evidence and saved revisions retain their sections.

The layout does not alter financial calculations, price dates, starter defaults,
terminal credit, source packs or authored assumptions. Scenario intervals retain
no assigned company probability or guarantee.

## Research notes

Each company has a four-part notebook: business model, financial history, risks
and next questions. It saves locally as the user types, keyed separately by listing
ID under macro-atlas-company-notes-v1. Each edit records its data release and
company source date. Opening a notebook writes nothing. Unreadable/unsupported
stored notes remain untouched and disabled; storage errors remain explicit.
Notes do not mark a company as reviewed, modify its valuation, or create a formal
deep dive. Existing archived research remains below the notebook.

## Verification boundary

Focused tests cover unscrolled table/chart visibility, real desktop and narrow
screen interaction, notebook persistence and listing isolation, exact capex values
from independently decoded source shards, existing numeric filter/CSV semantics,
and preservation of valuation storage. Source-based chart tests cover negative
and missing cash, gaps and currencies. Screenshot review complements geometric
checks; it is not a complete accessibility certification. See the current desktop
handoff for final test counts and the installed executable identity.

Related behavior: [KPI ranges](company-list-kpi-ranges.md).
