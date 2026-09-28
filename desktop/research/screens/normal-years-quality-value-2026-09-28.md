# Kvalitetsbolag med rabatt – normalår

Saved screening policy requested 28 September 2026. Exclude annual reports ending
in **2020–2023 inclusive**, from both historical quality calculations and a separate
screening valuation. These are user-selected exceptions, not a claim that every
company was unusually affected in those periods.

## Period policy

Use the latest five comparable reports outside the exception years, from at most
ten consecutive annual reports. A December-year-end company with 2025 data normally
uses 2025, 2024, 2019, 2018 and 2017. Determine exclusions from actual report end
dates, rather than provider year labels. Retain all exceptional reports and values
for inspection. Existing five-report measures and starter valuations keep all years.

The latest report must end after 2023 and be no more than 550 days old at the fixed
research snapshot. Require publication dates for the five selected reports. Currency
changes, withheld latest reports and chronology breaks stop comparable history.
Missing selected observations fail the relevant filter; an older observation cannot
replace a missing selected value. CAGR uses actual time between the endpoints, about
eight years for 2017–2025, rather than pretending that five selected reports span four.

## Fixed screening choices

These thresholds were fixed before inspecting matching company names.

| Condition | Threshold |
| --- | --- |
| Business route | Operating company, latest directory, no classification conflict |
| EBIT / (year-end equity + net debt) | Median ≥20%; minimum ≥10% in five normal reports |
| Shareholder profit / (assets − intangible assets) | Median ≥8% in five normal reports |
| EBIT margin | Median ≥12%; minimum ≥8% in five normal reports |
| Revenue CAGR | ≥3%, across actual elapsed time between normal endpoints |
| Operating cash-flow CAGR | ≥0%, requiring all five selected cash flows positive |
| Tangible assets / revenue | Median between 0 and 0.5 in normal reports |
| Provider FCF | Positive in all five normal reports |
| Latest provider net debt / EBITDA | ≤1.5; separate point-in-time balance-sheet guard |
| Latest provider EBITDA margin | >0; prevents negative-denominator debt ratios passing |
| Normal-year full-terminal NPV / saved price | ≥42.857142857%, equivalent to ≥30% discount to modeled value |
| Normal-year half-terminal NPV / saved price | ≥0%; explicit terminal sensitivity |

Sort by normal-year half-terminal NPV / price, descending. Show full, half and
zero terminal cases, normal cash/terminal PV, normal cash median, original all-year
starter comparisons, and actual saved quote/report dates.

The capital-return ratios are explicitly **accounting proxies**. The first uses
pre-tax EBIT and closing capital; the second uses shareholder profit and closing
assets after deducting all reported intangibles. Neither is normalized NOPAT divided
by average tangible operating capital, nor a reconstructed provider ROIC/ROA-G.
Provider five-year ROIC/capex aggregates lack the dated component observations needed
for this exclusion policy and are not used as normal-year conditions.

## Separate screening valuation

Let C be the signed median provider FCF of the five selected normal reports.
Assume ten annual nominal cash flows equal to C, a 10% required return and zero
growth. Cash PV = sum(C / 1.10^t), t = 1…10. Continuing terminal PV =
(C / 0.10) / 1.10^10. Full value = cash PV + terminal PV; half and zero terminal
credit are explicit sensitivities. NPV = selected value − saved equity price.

For positive C under these fixed flat-cash assumptions, full value is 10 × C and
terminal PV is about 38.55% of full value. The 30% full-value discount therefore
already implies nonnegative half-terminal NPV. That second bound is a transparent
cross-check, not an independent valuation signal or an extra confidence weight.

No cash or debt is added again. Compare only a valid dated saved equity-price basis
in the same reporting currency, no more than 550 days old at the fixed snapshot.
Retain signed cash; negative projections represent
modeled funding needs, not a claim of negative realizable limited-liability equity.
No historical-error bands or probabilities are assigned to this new projection.

This does not modify existing company Value drafts, reviewed cases, empirical
all-year starter calibration, purchase rules or position-sizing policy. Older nominal
cash amounts are not inflation-adjusted or scaled up for a larger modern business.

## What requires company research

Provider FCF is not reconciled owner cash. Tangible asset intensity does not measure
maintenance investment, replacement cost, lease principal or intangible reinvestment.
Margins and accounting returns are indicators to investigate; they do not establish
pricing power, inflation protection or a durable competitive advantage. Check the
excluded years for losses, dilution and financing stress before accepting a case.

Quotes and company data are frozen snapshots, not current trading prices. Multiple
listings can represent the same issuer. Surplus percentages are not annualized returns,
expected returns or guarantees. Verify current price/share basis and survival risks
before using a candidate in an investment decision.

Provider definitions checked: [Börsdata ROIC](https://borsdata.se/info/nyckeltal/roic)
and [Börsdata capex](https://borsdata.se/en/info/ratios/capex). They support keeping
provider ratios distinct from these explicitly calculated accounting proxies.

The companion JSON record captures the installed saved view, independent arithmetic,
matching listings and persistence verification after installation.

A companion **Kvalitetsbolag – normalår, prisbevakning** view keeps all the same
quality and debt requirements but omits the two valuation bounds. It allows price
monitoring and manual company research when the strict discount screen is empty.
Its matches are not represented as undervalued. Neither view relaxes the fixed
quality or discount thresholds in response to observed company names.
