# Capital-light value candidates

Saved on 28 September 2026 in the installed Macro Atlas 0.22.1 normal profile.
Open **Companies → Lists → Saved view → Capital-light value candidates**.
The view saves the columns, exact periods, thresholds and sorting; it can be
reloaded and edited with the existing list tools. No new app release is needed.
The portable [screen record](capital-light-value-2026-09-28.json) retains the
configuration, source identities, independent calculation and verification.

## Rules

Every condition must match. Missing required values do not pass. These are
chosen research thresholds, not statistically calibrated predictors.

| Test | Saved condition |
|---|---|
| Universe | Operating businesses in the newest directory, five-period evidence, no classification conflict |
| Cash and earnings history | Positive provider FCF and EBIT in all five comparable annual reports |
| Margin of safety | Half-terminal surplus ≥42.85714285714286%, approximately 42.86% |
| Adverse valuation sensitivity | Starter Low NPV / saved price ≥0% |
| Tangible-assets return proxy | Provider ROA-G, five-year mean ≥10% |
| Broader capital return | Provider ROIC, five-year mean ≥15% |
| Investment burden | Provider Capex / operating cash flow, five-year mean between 0% and 40% |
| Physical asset intensity proxy | Latest vendor tangible assets / annual revenue between 0 and 0.5× |
| Operating profitability | Median EBIT margin across five comparable annual reports ≥10% |
| Leverage | Latest saved provider net debt / EBITDA ≤2×, including net cash |
| Debt denominator guard | Latest saved provider EBITDA margin >0% |

Sort by half-terminal surplus, highest first. Full, half and zero terminal
credit are shown together, with the saved quote and annual-report dates.
A 30% discount to scenario value is 42.857% surplus relative to price.
The 50% terminal credit is a sensitivity assumption, not a probability.
Provider historical windows and annual-statement windows are different saved
sources and must not be treated as exactly aligned reporting periods.

## What the screen does not establish

The user's inflation thesis is to find businesses that can raise prices while
requiring little incremental capital. Historical profitability, low asset
intensity and low investment burden help source candidates for that inquiry.
They cannot establish pricing power or future returns.

Provider [ROA-G](https://borsdata.se/en/info/ratios/roa-g) uses profit relative to
total assets less intangible assets. It is not normalized NOPAT divided by
average tangible operating capital. Provider
[ROIC](https://borsdata.se/info/nyckeltal/roic) adds a broader return measure;
neither replaces a reconciled company-level ROTCE calculation.
Provider [Capex %](https://borsdata.se/en/info/ratios/capex) includes acquisitions
and disposals, so a low average does not establish low maintenance investment.
The vendor tangible-assets field is not verified PP&E and can be affected by
asset age, depreciation, leases and classification.

Review actual price increases versus volumes and customer retention, competition,
replacement costs, working-capital needs, leases, capitalized development,
acquisitions, and survival/refinancing risks. Growth must pay for its investment
before cash is available to owners. See also
[Berkshire's tangible-capital discussion](https://www.berkshirehathaway.com/letters/1983.html).

## Dated result and verification

The 13 September 2026 research snapshot and 10 August provider snapshot produce
four listings for three issuers: Huuuge (two listings), Kowa and Aerostar.
Their saved quote dates range from 9 February to 9 April 2026. These results do
not establish undervaluation at today's price. Reconcile the primary listing,
share count, currency and current company information before valuation work.

The source artifacts were hash-verified and an independent calculation matched
the installed app's exact four rows and ordering. The named view was saved using
the app's Save view action and read back after reload and a full process restart.
The existing saved view,
watchlists and 61 unrelated localStorage records were preserved byte for byte.
The closed-profile backup and runtime evidence are in
`desktop/test-results/saved-screen-2026-09-28/` (not published).
