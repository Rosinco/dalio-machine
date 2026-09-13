# SCA B: automatic valuation starting study

Prepared 2026-09-12 under the [value and price framework](company-valuation-framework.md),
model `sca-2026-09-12-v1`. This is a new numerical sensitivity study accompanying
SCA's archived August research. It is sufficiently sourced to inspect and compare
the stated assumptions; it is not a new investment recommendation or a reproduction
of the archived verdict. Forecast confidence is low because industrial earnings
are cyclical and reinvestment needs are uncertain.

## Analysis basis and selection

| Field | Basis |
|---|---|
| Company / Börsdata ID / ISIN | Svenska Cellulosa Aktiebolaget SCA, SCA B / `197` / `SE0000112724` |
| Ownership | Proportional common-equity interest across A and B shares; B-share purchase price |
| Study date | 2026-09-12 |
| Financial dates | FY2025, published 2026-03-04; H1 2026, published 2026-07-22 |
| Currencies and units | Nominal SEK; company amounts in SEK million; per-share prices in SEK; no FX conversion |
| Quote / downloaded snapshot | 2026-08-07 / 2026-08-10 |
| Model | Explicit common-equity distributions, 20 full years after valuation, year-end timing, separate final equity sale and alternative breakup |
| Macro observations | Issuer's July 22 report and August 6 price announcement; no mechanical GDP or policy-rate multiplier |

SCA provides the second reviewed refinement, following Holmen, with a completed
May deep dive and a substantive August rerun. Every company also has the
[standard historical cash-flow model](standard-company-valuation.md). Subsequent
deep dives refine that common workspace using reconciled company-specific
forecasts and the same versioned study contract.

The bundled research identity is
`studies/sectors/material/skogsbolag/deep_dives/sca/deep_dive_2026-08-17_omkorning.md`,
dated 2026-08-17, SHA-256
`0d4ea16b0c3d782d02295420bf4cd8bfc81b76d7db079371dd21e8266ff89269`.
The old price, valuation targets, mandatory return thresholds, probability labels,
and statements that recovery is a floor do not become inputs to this study.

## Price and ownership

The latest SCA B close in the saved Börsdata snapshot is **SEK 110.30 on
2026-08-07**. The issuer's June share table reports **702,342,489 shares**:
60,864,830 A and 641,477,659 B, with no dilution effects. Its annual report confirms
equal dividend rights and no treasury shares. Subsequent A-to-B conversions can
change voting rights and class counts without changing this denominator.
[H1 report, p. 17](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf),
[annual report, pp. 67, 73](https://www.sca.com/siteassets/investors/reports-and-presentations/annual-reports/2025/sca-annual-report-2025.pdf).

`110.30 × 702.342489 = SEK 77,468.3765367m` is the **B-equivalent equity purchase
basis**, assuming the June total remains unchanged. It is not the sum of separately
priced A and B market capitalizations. The quote is historical and differs from the
August 17 quote in the deep dive. Financial and quote dates remain separate; this
is not a point-in-time backtest or a live quote assessment.

## Business and branch transmission

SCA monetizes forest and industrial assets through timber, wood products, pulp,
packaging and energy. Forest ownership provides resource access, but does not
eliminate commodity price risk or the need to buy wood. The economic questions
are industrial cash margins, spending needed to maintain those margins and the
amount ultimately distributed to common shareholders. Biological remeasurement
is not cash available for a dividend.

| Dated evidence | Exposure and transmission | Scenario implication and review signpost |
|---|---|---|
| H1 report, July 22: weak pulp demand and adverse price/currency effects | The issuer reports foreign sales; price, FX and purchased inputs affect mill margins | Low case keeps industrial margins under pressure. Mid/high require higher realized margins; monitor segment cash profit and actual selling prices. |
| H1 report: packaging price increases expected to pass through in H2 | Contract and delivery timing delays realization | Mid assumes recovery toward FY2025 operating earnings; high requires substantially more. An announcement alone is not booked cash. |
| August 6: announced European kraftliner increase of EUR 100/t from September 1 | European packaging sales | A monitoring signpost only; no volume-times-announced-price profit is added. Review realized net prices and volumes in the next report. |
| H1 report: fuel supported energy earnings while pressuring other businesses | Costs and energy realization can move in opposite directions | Do not extrapolate peak energy conditions across every segment; high requires group cash improvement after reinvestment. |

These are company-specific observations, not fresh macro forecasts.
[SCA half-year release](https://www.sca.com/en/media/press-releases/2026/half-year-report-q2-2026/),
[kraftliner announcement](https://www.sca.com/en/media/press-releases/2026/sca-to-increase-kraftliner-prices-by-100-per-tonne/).
The study uses original issuer statements where the app's saved vendor summary
differs; it does not change the underlying vendor or macro dataset.

## Capital, cash and financing reconciliation

June common equity is `101,194 − 17 NCI = 101,177`; tangible common equity is
`101,177 − 1,275 intangibles = SEK 99,902m`. Borrowings and lease debt total
**SEK 15,536m**, assembled from loans 12,582 long and 2,524 current, and leases
256 long and 174 current. Pension provisions of 40 are separate. The issuer's
broader gross-financial-liability figure is 15,583, including pensions and other
financial marks. Its **10,859 net debt** also nets off pension surplus 4,401,
other financial assets 161 and cash 162. Pension surplus is not immediately
distributable cash. All 162 cash is assumed required, leaving **zero surplus cash**.
[H1 report, pp. 13, 16, 18](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf).

The matched FY2025 tangible-capital denominator is a two-endpoint **proxy**:
`[(114,920 − 1,025) + (112,460 − 1,301)] / 2 = 112,527`. It uses the issuer's
capital-employed perimeter, retains forest fair values, includes associated
operations and NCI, and follows its treatment of pensions and cash. It is not a
fully reconciled operating-capital measure. FY2025 EBIT excluding biological
gains is `4,432 − 1,782 = 2,650`; a uniform assumed 20.6% operating tax gives
NOPAT 2,104.1 and a **1.87% proxy capital return**. A verified FCFF/TCE cash yield
remains unavailable: the cash reference below is after financing and is not FCFF.
[Annual report, pp. 168–173](https://www.sca.com/siteassets/investors/reports-and-presentations/annual-reports/2025/sca-annual-report-2025.pdf).

The issuer reports 3.5-year average debt maturity and an undrawn SEK 6,000m
facility to 2030 at June. FY2025 notes report no financial covenants and no
pledged assets. These historical facts support liquidity analysis but do not
eliminate refinancing, contingent-claim or cash-conversion risk. A cash interest
diagnostic is trailing EBITDA excluding biological gains 3,554 divided by paid
interest 421, about 8.44x; this is not a covenant metric and ignores required
capital spending. Contractor guarantees and joint-venture support warrant review.
[H1 report, pp. 3–4, 17](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf),
[annual report, notes E4 and G2](https://www.sca.com/siteassets/investors/reports-and-presentations/annual-reports/2025/sca-annual-report-2025.pdf).

The following uses the **statutory cash-flow statement**. SCA's alternative
“operating cash flow” measure already deducts current net investment, uses other
working-capital/hedge adjustments and precedes financing and taxes. Its FY2025
3,078 and H1 2026 1,015 must not replace statutory CFO or have the same investment
deducted a second time.

| SEK m | FY2025 | H1 2025 | H1 2026 | Trailing year to June 2026 |
|---|---:|---:|---:|---:|
| Statutory cash from operations, after interest and cash tax | 4,017 | 1,791 | 1,392 | 3,618 |
| Gross cash investment in tangible/intangible assets, current and strategic | 2,815 | 1,549 | 770 | 2,036 |
| Lease principal paid | 223 | 111 | 107 | 219 |
| Cash tax paid; negative means refund | 197 | 149 | −59 | −11 |
| Tangible asset-sale receipts, excluded from cash reference | 233 | 31 | 132 | 334 |
| Business acquisitions | 0 | 0 | 0 | 0 |
| Net financial-investment receipts/payments, excluded | −23 | −27 | 0 | 4 |
| Dividends paid, reference only | 2,107 | 2,107 | 2,107 | 2,107 |

Trailing year is `FY2025 + H1 2026 − H1 2025`. Cash after gross investment and
leases is `3,618 − 2,036 − 219 = 1,363`. Replace the net tax refund with an
**assumed normal annual cash tax of 450**: `1,363 − 11 − 450 = 902`.
Tax 450 approximates 20.6% of FY2025 ex-biological-gain EBIT after financing.
This reference retains actual working-capital changes, excludes asset sales and
net financial-investment flows, and does not prove distributable capacity.
[Annual statutory statement, p. 169](https://www.sca.com/siteassets/investors/reports-and-presentations/annual-reports/2025/sca-annual-report-2025.pdf),
[interim statutory statement, p. 14](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf).

## Continuing-business assumptions

The forecast is independently authored from an explicit year-one cash bridge.
Every cell in this table is an **analyst assumption**, not issuer guidance.

| SEK m, year one | Low | Mid | High |
|---|---:|---:|---:|
| Operating EBIT excluding biological gains | 1,900 | 2,650 | 4,000 |
| Add depreciation, including leases | 2,250 | 2,250 | 2,300 |
| Deduct gross cash capital spending | 2,400 | 2,400 | 2,600 |
| Deduct operating working-capital investment | 150 | 150 | 250 |
| Deduct cash net interest | 450 | 450 | 450 |
| Deduct cash tax | 300 | 450 | 730 |
| Deduct lease principal | 220 | 220 | 220 |
| **Assumed common-equity distribution** | **630** | **1,230** | **2,050** |
| Difference from tax-normalized cash reference 902 | −272 | +328 | +1,148 |
| Annual payment growth, years 2–10 | −1% | 2% | 4% |
| Mature payment growth, years 11 onward | 0% | 1% | 2% |
| Required nominal common-equity return | 9% | 9% | 9% |

Mid starts from FY2025 ex-gain operating EBIT. Low remains below that year's
earnings, while still requiring some improvement from trailing ex-gain EBIT
1,340. High requires a substantial industrial recovery. Cash taxes approximate
20.6% of assumed EBIT less cash interest, rounded. Gross cash spending is above
the trailing 2,036 reference; it funds an assumed adequate maintenance and growth
program, including routine forestry investment. The high case increases spending
and working capital. Those amounts are sensitivities, not verified maintenance
budgets or proof that 4% dividend growth can be delivered.

All modeled available equity cash is assumed paid as dividends after interest,
taxes, reinvestment and lease principal. Proportional ownership remains constant.
There is no additional net borrowing, buyback or dilution, no acquisition-led
growth, no incremental JV capital call, no disposal program and no separate
pension-surplus payout. Existing consolidated earnings are modeled as belonging
to common shareholders after negligible NCI earnings; revise if that changes.
Incremental forest acquisitions or JV funding needed to sustain these forecasts
must reduce distributions, rather than being financed invisibly. Higher debt
repayments than refinancing proceeds would also reduce them. This explains why
the cash reference does not mechanically become the forecast or the past 2,107
dividend automatically continue.

The required return of 9% is an analyst nominal equity hurdle, held constant
across scenarios to isolate cash assumptions; it is not an estimated market WACC
or the issuer's funding rate. There are **20 explicit annual payments**. The
year-20 equity sale is `year-21 payment / (9% − mature growth) × 98%`, an assumed
2% sale cost. It capitalizes later dividends once. No forest book value, second
cash addition or debt subtraction is attached to the equity DCF. Review the
separately editable sale whenever the payment path or required return changes.
Probabilities remain unset.

## Calculated comparison

| Per SEK 1,000 invested unless indicated | Low | Mid | High |
|---|---:|---:|---:|
| Equity value including final sale | 84.83 | 210.28 | 427.77 |
| NPV after purchase | −915.17 | −789.72 | −572.23 |
| Value / price | 0.0848 | 0.2103 | 0.4278 |
| Final sale's share of present value | 17.02% | 22.01% | 27.35% |
| Cash-distribution payback | Not within 20 years | Not within 20 years | Not within 20 years |
| Discounted cash-distribution payback | Not within 20 years | Not within 20 years | Not within 20 years |
| Ordinary payback including final sale | Not within 20 years | Not within 20 years | Year 20 |
| Discounted payback including final sale | Not within 20 years | Not within 20 years | Not within 20 years |

All three continuing values are below this dated price under these assumptions.
At 7% required return, recomputing terminal capitalization consistently, values
are 108.32 / 281.32 / 604.08; at 11% they are 69.80 / 167.65 / 330.00. The gap
reflects modest assumed dividends relative to the equity purchase price. Recovery
may differ substantially, but is an alternative strategy with separate uncertainty.
Macro Atlas shows the editable annual discounted payments and cumulative NPV;
the saved calculation also records every annual value. Payback uses whole years
and does not extrapolate.

## Separate breakup sensitivity

This is an assumed orderly disposal program, requiring board/owner decisions and
buyers; an ordinary minority shareholder cannot compel it. June book assets are
the starting schedule, not bids or guaranteed proceeds. All percentages below
are analyst stress assumptions. Lower mill realization than forest realization
reflects specialized industrial assets and closure costs, not an external appraisal.

| Asset | June book SEK m | Low realization | Mid | High |
|---|---:|---:|---:|---:|
| Forest land and standing timber | 104,393 | 50% | 70% | 90% |
| Industrial property, plant and equipment | 24,489 | 10% | 30% | 50% |
| Inventories | 6,491 | 40% | 60% | 80% |
| Trade receivables | 3,690 | 80% | 90% | 98% |
| Other current receivables | 896 | 30% | 55% | 75% |
| Other non-current assets excluding pension surplus | 1,467 | 20% | 50% | 75% |
| Cash | 162 | 100% | 100% | 100% |
| Right-of-use assets | 403 | No disposal credit | No disposal credit | No disposal credit |
| Pension surplus | 4,401 | No distribution credit | No distribution credit | No distribution credit |
| Intangibles | 1,275 | No credit assumed | No credit assumed | No credit assumed |
| **Total book assets** | **147,667** | | | |

Other non-current assets are `5,868 − 4,401 pension surplus`; their condensed
disclosure mixes investments, receivables and other assets, so the realization
range is deliberately broad. Intangible disposal proceeds and access to pension
surplus are unavailable; zero credited proceeds are explicit assumptions of this
sensitivity, not claims that those assets have no value. Legal realization of
each mixed bucket still needs review.

| Claims, costs and results, SEK m | Low | Mid | High |
|---|---:|---:|---:|
| Gross assumed proceeds | 60,918.00 | 89,025.70 | 116,941.45 |
| All booked liabilities, deducted once | 46,473 | 46,473 | 46,473 |
| Book NCI reserve, separate from liabilities | 17 | 17 | 17 |
| Assumed incremental transaction and closure costs | 3,000 | 1,900 | 1,100 |
| Assumed additional external-claim reserve | 600 | 600 | 600 |
| Assumed net cash consumed before completion | 1,000 | 600 | 400 |
| **Net common-equity recovery** | **9,828.00** | **39,435.70** | **68,351.45** |
| Assumed distribution year | 5 | 3 | 2 |
| **Present value per SEK 1,000 invested, at 9%** | **82.45** | **393.08** | **742.63** |

Booked liabilities include the full **24,490 deferred-tax liability** and existing
pension/environmental provisions. No second full tax charge or booked closure
provision is deducted. Full deferred tax can overstate tax at depressed asset
sale prices; actual tax bases, exemptions and structure remain unresolved. NCI 17
is a reserve at book, not a guarantee that minority interests can be bought out
for that amount. The additional 600 claim allowance addresses unbooked external
guarantees and support obligations without summing subsidiary guarantees over
already consolidated debt. It is an assumption, not a reported claim or cash-burn
estimate. Incremental closure costs and cash burn are separately stated.

Amounts use the June balance sheet and financing notes; annual note G2 supplies
additional contingent-exposure context, including contractor guarantees 479 and
a JV repayment undertaking 115 at December 2025. The condensed June note does
not fully update each of those exposures. Failure to sell, new claims, fire,
harvesting restrictions or sustained losses can produce worse outcomes, including
zero equity recovery or funding needs. These three cases are not guaranteed
bounds, a price floor or an amount to add to continuing-business value.
[H1 report, pp. 13, 16–18](https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf),
[annual report, note G2](https://www.sca.com/siteassets/investors/reports-and-presentations/annual-reports/2025/sca-annual-report-2025.pdf).

## Assessment and next review

Realization in the continuing model comes from dividends and a future equity sale;
the recovery model requires a different corporate strategy. Neither model imports
the old study's buy/pass gate. The chief unresolved issues are sustainable mill
margins, the cash investment required for growth, forest transaction feasibility,
tax structure, other-asset realization and contingent financing commitments.

Review at the next financial report, a material price update, a change in total
shares, a new investment commitment, sustained dividends beyond cash generation,
or evidence of an actual forest sale or distribution program. A higher company
asset value alone does not establish cash realization by minority shareholders.

## Reproducible source record

Original source archives and extracted text are retained locally under
`desktop/test-results/valuation-sources/sca/`. The independent
[calculation script](../desktop/test-results/valuation-sources/sca/reconcile.py)
writes [reconciliation.json](../desktop/test-results/valuation-sources/sca/reconciliation.json),
including the reviewed facts, all assumptions, yearly payments, terminal values,
payback, sensitivity and recovery arithmetic. It does not consume vendor FCF.

| Evidence | Source identity / SHA-256 |
|---|---|
| Issuer H1 2026 PDF; retrieved September 12 | `sca-2026-h1.pdf`; `3b0034d3f1713805a1ef0a5b8da323a74b91f903d575dfba8104b20e36931a38` |
| Issuer FY2025 PDF; old download and fresh issuer copy match | `sca-2025-annual.pdf` and `sca-2025-annual-official.pdf`; `47307fff6ceff81324c577b6b069b641d46b1cf0253d3ab4a93adcc0ae961bd5` |
| Saved quote parquet, ins_id 197, latest row August 7 | `data/raw_api_snapshots/2026-08-10/all_stockprices/all_stockprices.parquet`; `a8c9f95df438070483c4822f92a6bf806e3dd37b616c34f318c95fa8fa08e547` |
| Local quote extract | `saved-price-rows.json`; final ten observations retain dates and source values |

The source repository remains read-only. The reviewed UI study is stored at
`desktop/research/valuations/sca-2026-09-12.json` and registered separately.
Its displayed source, period and assumption notes are company-specific. Source
PDFs and intermediate calculations are local evidence artifacts; the committed
study and this document retain the formulas and immutable source identities.
