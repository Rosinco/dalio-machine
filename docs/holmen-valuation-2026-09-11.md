# Holmen: automatic valuation starting study

Prepared 2026-09-11. Model version `holmen-2026-09-11-v1`. Open **Company
observatory → Holmen → Value**; the scenarios and charts require no field entry.
This is a new numerical sensitivity study accompanying the archived May deep
dive, not a reproduction of its historical verdict.

## Price and ownership

The latest Holmen B close in the saved 2026-08-10 Börsdata download is **SEK 329
on 2026-08-07**. The issuer reported **150,434,534 outstanding A+B shares** after
buybacks and cancellation, excluding treasury shares. [Holmen share notice](https://www.holmen.com/en/Newsroom/press/press-releases/2026/changes-in-the-total-number-of-shares-in-holmen/).

`B-equivalent equity purchase basis = 329 × 150.434534 = SEK 49,492.961686m`.
This assumes equal financial rights and an unchanged outstanding-share count;
it is the cost of the modeled proportional ownership at the B-share price,
not the sum of separately priced A and B market capitalizations. The default
investment is SEK 1,000. The quote is historical and the study is not a backtest.

## Cash reference and analyst assumptions

The [January–June 2026 report](https://vp165.alertir.com/afw/files/press/holmen/202608199658-1.pdf),
published 20 August, supplies the annual and half-year cash statements on page 11.
Trailing year = FY 2025 + H1 2026 − H1 2025:

| SEK m | Trailing year to June 2026 |
|---|---:|
| Operating cash after interest and paid taxes | 3,444 |
| Gross purchases of non-current assets | 1,557 |
| Lease principal payments | 134 |
| Cash taxes paid (negative means net refund) | −10 |

Cash after gross investment and leases is `3,444 − 1,557 − 134 = 1,753`.
Replacing the unusual net tax refund with **assumed normal annual cash tax of
SEK 500m** gives `1,753 − 10 − 500 = 1,243`. This is a reference, not the forecast:
it retains actual working-capital changes and does not establish through-cycle
maintenance investment. The tax assumption approximates 22% of FY 2025 pretax
profit excluding the biological-asset gain.

The model assumes available equity cash is paid as dividends after reinvestment,
interest and lease principal. It assumes constant proportional ownership, no
new net borrowing, no buybacks and no dilution. Historical repurchases are not
cash receipts for a shareholder who keeps their shares. All forecasts are nominal
SEK and all payments occur at the end of each full forecast year after valuation.

| Analyst assumption | Low | Mid | High |
|---|---:|---:|---:|
| First annual dividend, SEK m | 1,000 | 1,500 | 2,400 |
| Difference from normalized cash reference, SEK m | −243 | +257 | +1,157 |
| Annual payment growth in years 2–10 | −1% | 2% | 4% |
| Mature growth, year 11 onward | 0% | 1% | 2% |
| Required equity return | 9% | 9% | 9% |

The low case assumes continued operating and reinvestment pressure. The mid case
requires moderate improvement; the high case requires substantial recovery in
industrial cash generation. Construction demand, paper/board demand, energy
economics and harvest disruption inform the range. No automatic GDP multiplier,
carbon-credit proceeds or scenario probability is assigned.

The explicit horizon is **20 years**. The final equity sale capitalizes year-21
dividends at `cash / (9% − mature growth)`, less an assumed 2% selling cost. The
sale represents post-horizon distributions once. It adds neither forest book
value nor a second debt/cash adjustment. When editing the annual forecasts or
required return, review the separately editable final-sale estimate as well.

## Calculated outcome for SEK 1,000

| Outcome under these assumptions | Low | Mid | High |
|---|---:|---:|---:|
| Present value including final sale, SEK | 210.76 | 401.38 | 783.87 |
| NPV after purchase, SEK | −789.24 | −598.62 | −216.13 |
| Cash-payment payback | Not within 20 years | Not within 20 years | Year 16 |
| Discounted cash-payment payback | Not within 20 years | Not within 20 years | Not within 20 years |

All three modeled values are below this dated price. That conclusion is
conditional on the forecasts and required return; it is not an automatic trade
recommendation. The interface also shows payback including sale and the share of
value attributable to that sale. It never extrapolates an unreached payback year.

## Capital and separate breakup recovery

June 2026 tangible equity is `54,182 − 482 = SEK 53,700m`. Gross debt including
leases is `4,108 + 2,623 + 91 + 95 = SEK 6,917m`; pension claims are separate.
All SEK 182m cash is assumed needed, so surplus cash is zero. The report's broader
net-financial-debt reconciliation is on page 15. [Holmen interim report](https://vp165.alertir.com/afw/files/press/holmen/202608199658-1.pdf).

Average tangible capital is a two-endpoint FY 2025 **proxy** using saved annual
equity + net financial debt − intangibles: **SEK 60,083m**. It retains forest
revaluations and the publisher's net-debt perimeter, including pension assets
and cash. Estimated normalized NOPAT is `(3,270 − 895) × (1 − 22%) = 1,852.5`,
giving about **3.08%** on that capital proxy. This is not a fully reconciled
operating-capital return.

The independent breakup sensitivity uses June book assets, disclosed realization
haircuts and all **SEK 26,887m book liabilities**, including full deferred taxes,
deducted once. Extra costs cover transaction expenses and cash burn beyond booked
provisions. Leased assets and pension surplus have no recovery credit. Detailed
asset buckets, rates and deductions are available in **Business, capital &
evidence → Inspect the separate breakup calculation**.

| Analyst recovery case | Low | Mid | High |
|---|---:|---:|---:|
| Forest realization as a fraction of book | 55% | 75% | 90% |
| Net common-equity recovery, SEK m | 9,875.8 | 27,276.5 | 40,024.9 |
| Assumed payment year | 5 | 3 | 2 |
| Present value per SEK 1,000 invested | 129.69 | 425.56 | 680.67 |

These are stress assumptions, not asset appraisals. Full booked deferred tax can
overstate tax at depressed sale prices. Actual claims, proceeds and feasibility
need transaction-specific analysis. A minority shareholder cannot compel this
disposal program. Recovery is a separate alternative, never a guaranteed floor
or an amount added to continuing-business value.

## Reuse and provenance

The machine-readable study lives in
`desktop/research/valuations/holmen-2026-09-11.json`. It records original source
locations and available file hashes, the archived deep-dive identity and explicit
analyst assumptions. Register another completed numerical study in
`desktop/src/researchedValuations.ts`; document its ownership and cash-flow
perimeter and give each revision a new immutable study ID. A deep-dive folder or
vendor FCF column alone is insufficient to generate these assumptions.

The archived May study added forest value to cash flows already supported by
that forest. This study removes that double count, distinguishes current
financial anchors from the older narrative, and does not inherit old price
targets, probabilities or position-size recommendations. Original source files
and the macro fact base are unchanged.
