# Company analysis: value and price

Adopted for Macro Atlas company analysis on 2026-09-11. Methodology version 1.
Start each new analysis with the [company analysis template](company-analysis-template.md).
The adoption decision is [ADR 0035](../decisions/0035-company-value-and-price.md).

## Purpose and structure

Estimate the value of the cash that can ultimately reach shareholders, then
compare that value with the market price for the same ownership claim. Show
the assumptions and adverse outcomes alongside the apparent discount.

| Category | Component | Meaning |
|---|---|---|
| 1. Value | 1.1 Continuing business | Present value of future cash flows from operating the business, after the reinvestment needed to deliver them. |
| 1. Value | 1.2 Liquidation or breakup scenarios | Present value of proceeds available to shareholders from selling assets or divisions after claims, costs and delays. |
| 2. Price | 2.1 Equity market price | Dated market capitalization or share price, compared with intrinsic equity value on the same basis. |

DCF is a method for estimating present value. NPV is the result of comparing
that estimated value with the purchase price; it is not a separate source of
business value. Liquidation is another cash-realization scenario. Selling a
division as a functioning business is distinct from a forced asset sale.

Do not add full going-concern and liquidation values, or automatically choose
the larger estimate. Explain which strategy is feasible, who can implement it,
and how cash would reach minority shareholders. A DCF may include a future
sale or liquidation as its terminal cash flow, counted once.

## Analysis workflow

1. Establish the company, ownership claim, valuation date, financial periods,
   currencies and source versions. Keep facts, calculations and assumptions distinct.
2. Understand the operating business: customers, competitors, moat, pace of
   change, cyclicality, capital allocation and reinvestment requirements.
3. Reconcile tangible capital, cash, debt and other claims. Normalize earnings
   and cash flows using the original financial statements where needed.
4. Connect dated macro and branch evidence to verified company exposures and
   explicit cash-flow assumptions.
5. Estimate pessimistic, base and optimistic continuing-business values. Use a
   liquidation, breakup or run-off scenario where meaningful; explain omissions.
6. Compare equity values with the dated equity price. Explain the discount,
   adverse outcomes, realization mechanism, holding horizon and unresolved evidence.

An incomplete evidence base is a valid outcome. Historical profitability or a
high value/price estimate alone does not establish an attractive investment.

## Continuing-business valuation

Use the model that fits the business and available evidence. Record the model,
forecast horizon, timing convention, normalized starting cash flow, required
return, reinvestment and terminal assumptions. Keep currencies, nominal versus
real amounts, and pre-tax versus after-tax inputs consistent.

For a conventional operating company, one route is:

```text
NOPAT = normalized operating profit - tax attributable to operating profit
FCFF = NOPAT + depreciation/amortization - capital expenditure - change in operating working capital
Operating value = sum(FCFF_t / (1 + WACC)^t) + terminal operating value / (1 + WACC)^N
Equity value = operating value + surplus cash + other non-operating asset value
               - debt - other non-equity claims
```

WACC is the required return for the operating cash flows, reflecting their
financing and risk. Use time-specific rates when the capital structure changes
materially. Reconcile leases, pensions, preferred equity and minority interests
to the cash-flow perimeter; do not subtract liabilities already captured in the
operating cash flows again. Add only available surplus cash, and exclude its
income from the operating cash flows when its value is added separately.

Alternatively, value cash flows to common equity after financing flows at the
cost of equity. Do not deduct debt again from that equity valuation. Compare an
operating enterprise valuation with an EV using the same cash and claims
adjustments; compare equity value with market capitalization or price per share.

For a constant-growth terminal value, `TV_N = FCFF_(N+1) / (WACC - g)` requires
`g < WACC`, a sustainable mature business and reinvestment consistent with growth.
Disclose how much of the valuation comes from the terminal period. A finite-life
business can instead use a run-off forecast and net closure or recovery proceeds.

Forecast free cash flow after reinvestment, not accounting profit alone.
Acquisitions needed to sustain the forecast and dilution must be addressed.
Do not assume vendor-labelled FCF is verified FCFF or distributable owner earnings.

Growth funded by additional investment creates incremental value when its
expected return exceeds its cost of capital. A shrinking business can still be
undervalued when its remaining cash distributions and net recovery justify the
price. An attractive industry or moat is evidence for model assumptions, not
an extra value amount to add to a completed DCF.

## Tangible capital and the SEK 1,000 ownership view

These measures explain operating economics and financing; they are inputs to
valuation, rather than independent amounts to add to a DCF.

```text
Tangible book equity (TBE) = common book equity - goodwill - other intangible assets
Tangible operating capital employed (TCE) = tangible operating assets
                                           - non-interest-bearing operating liabilities
ROTCE = normalized NOPAT / average TCE over the matching earnings period
Operating cash yield on TCE = verified FCFF / average TCE over the matching period
```

With a reconciled ownership and liability perimeter, a financing-side bridge is
`TCE = TBE + debt + other relevant financing claims - surplus cash - other
non-operating assets`. Document each adjustment; a simple vendor proxy may not
include them all. Required operating cash stays in TCE. Cash is part of tangible
book equity, and book equity is already net of liabilities. Borrowing and holding
cash increases both cash and debt; it does not itself increase book equity.

Use beginning/end average capital, or a more representative average when major
transactions occur. Do not multiply a multi-year historical ROTCE by today's
capital and call it today's earnings. Goodwill exclusion can help assess existing
operations, while acquisition cost and future acquisition spending still matter
to shareholder returns. A near-zero or negative TCE makes the ratio unsuitable
for ranking; it does not automatically imply a poor business.

For an illustrative SEK 1,000 purchase, `ownership fraction = 1,000 / equity
market capitalization in SEK`. Multiply same-date company TBE, TCE, gross debt,
surplus cash and matched-period earnings/FCF by that fraction, after explicit FX
conversion. Label earnings attribution separately from cash actually distributed.
This describes exposure to corporate debt, not a personal debt obligation from
ordinary fully paid shares bought without borrowing.

At positive P/TBE of 0.5, SEK 1,000 buys exposure to SEK 2,000 of tangible book
equity: `1,000 / 0.5`. That book amount is not a guaranteed liquidation recovery.
Gross debt, interest coverage, maturities, refinancing needs, covenants and
restricted cash must remain visible even where net debt is small.

## Recovery and breakup analysis

Build a sourced asset-by-asset recovery schedule. Distinguish book amounts from
estimated realizable proceeds, and owned assets from leased or pledged assets.
Consider receivables collectability, inventory obsolescence, property, financial
holdings and separately saleable intangible rights. Assets without a supportable
recovery estimate remain unknown, not assumed worthless or fully recoverable.

```text
Net recovery to common shareholders = realizable proceeds including available cash
                                     - all prior claims
                                     - sale/closure costs and taxes
                                     - cash consumed before completion
Liquidation equity value today = present value of distributions to common shareholders
```

Account for receipt/payment timing and avoid counting the same liability or cash
burn twice. Any funding shortfall must be shown; ordinary equity recovery can be
zero. Estimates are not a guaranteed price floor or a maximum-loss limit. Current
book assets can lose value, claims can increase and a minority holder may be
unable to force a sale. A breakup case additionally needs the sale feasibility,
corporate costs, tax consequences and allocation of shared liabilities.

## Macro, branch and uncertainty evidence

Follow this chain for every material headwind or tailwind:

```text
Dated observation or publisher forecast
  -> company revenue/cost/asset/financing exposure with evidence
  -> transmission mechanism and lag
  -> scenario assumption affecting cash flow or the appropriate required return
  -> monitoring signpost and evidence that would invalidate the assumption
```

Listing country alone does not establish operating exposure. A country growth
forecast does not become company revenue growth, and a policy rate is not a
company's borrowing cost. Record contracts, hedges, debt refixing and maturities
where they affect transmission. Keep company judgments downstream from the
unchanged macro evidence, as required by ADR 0004.

Explain the forecast range using competition, technological change, customer
concentration, leverage and operating cyclicality. Slow industry change may
improve predictability but is not proof of safety. Longer-horizon cash-flow
forecasts often have wider uncertainty; the discussed risk/return diagram has
risk, not time, on its horizontal axis and promises no realized return.

Pessimistic/base/optimistic cases are analyst scenarios, not confidence intervals
or guaranteed bounds. Assign probabilities only with an explicit rationale;
otherwise leave them unset. Show sensitivity to the assumptions that matter.
Avoid applying several unexplained penalties for the same risk through cash
flows, discount rates and a final valuation haircut.

## Price comparison and decision output

For each scenario, using a positive market price and the same ownership basis:

```text
NPV = estimated equity value - equity purchase price
Value / price = estimated equity value / equity purchase price
Discount to estimated value = 1 - equity purchase price / estimated equity value
```

The last measure requires positive estimated value. Keep undefined ratios
unavailable. A positive NPV means an estimated surplus after the model's required
return; it is not a guaranteed cash profit or an annualized return.

### Interactive charts and payback

The user also requested high, mid and low DCF/NPV paths with the number of years
to recover the purchase in each case. Macro Atlas's first interactive workspace
uses explicit net cash distributions to common shareholders and separately
entered final equity sale proceeds. It supports 1–50 forecast years; it does
not automatically convert company operating profit or vendor FCF into payments.

For a company with a completed numerical study, populate the workspace from
versioned source facts and explicitly documented analyst scenarios. The user
requested automatic loading for previously researched companies, starting with
[Holmen](holmen-valuation-2026-09-11.md). Show results first and let the user inspect
and edit the populated assumptions. Preserve earlier drafts and subsequent user
edits. A research folder alone does not establish a usable forecast; unsupported
inputs remain missing. Research, financial-report and quote dates retain their
own identities, including when the selected macro release is older.

Show each year's discounted payment and cumulative NPV through each year. The
high/mid/low envelope represents the entered scenarios, not a calibrated
confidence interval. Preserve scenario labels when paths cross. The cumulative
chart uses steps because the model places payments at each year-end.

Ordinary payback is the first year cumulative undiscounted cash payments recover
the purchase price. Discounted payback uses the cumulative present value of those
payments. Report each scenario separately; show `not reached within the forecast`
instead of extrapolating. If later funding reverses a crossing, flag that reversal.
With year-end cash flows, show whole years; fractional years would need an explicit
within-year timing convention. Payback disregards later cash flows and cannot
replace NPV or the full risk assessment.

Keep cash-distribution payback separate from payback including the assumed final
sale. Let users include that sale in the NPV chart explicitly. A distinct net
recovery scenario must not be added to the continuing-business value.

For example, estimated equity value of SEK 1,600 at price SEK 1,000 gives NPV
SEK 600, value/price 1.6 and a 37.5% discount to estimated value. These use different
denominators from the 60% difference relative to price. Realized return also
depends on distributions, time and the eventual sale price.

Each completed analysis should show its scenario values, separate recovery case,
dated price, gross and net debt, cash restrictions, confidence rationale and
unresolved questions. Explain how shareholders can realize value through
distributions, reinvestment, a sale or another evidenced mechanism. Distinguish
temporary share-price declines from permanent impairment. Do not rank solely by
the highest point-estimate value/price or impose an unsupported universal discount.

## Business-specific methods and data limits

The value/price principle applies broadly, but a single TCE formula does not.
Banks and insurers generally need equity cash-flow, dividend or residual-income
analysis with regulatory capital and claim obligations. Property, holding
companies, finite-resource businesses and asset-light companies can require NAV,
sum-of-parts, run-off or adjusted capital measures. Explain the selected method
and preserve explicit unavailable outputs when the usual ratio is inappropriate.

Börsdata histories and the earlier tangible-capital screen can identify research
candidates. Their source FCF definitions, intangible treatment, leases, reporting
currency conversion and price/share basis need review before valuation use.
Downloaded prices and publication-date market-cap histories are dated evidence,
not live prices. Retain fiscal end, publication, source-download and price dates,
FX dates, source paths/hashes, formula versions and analyst assumptions. Updating
financial statements must not silently refresh an old quote or investment view.

## Method references

- [Aswath Damodaran: introduction to valuation](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/background/valintro.htm): cash flows, equity/firm consistency and alternative asset-realization approaches.
- [Aswath Damodaran: economic value added](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/lectures/eva.html): capital returns, the cost of capital and consistency with DCF.
- [Howard Marks: Fewer Losers, or More Winners?](https://www.oaktreecapital.com/insights/memo/fewer-losers-or-more-winners): interpretation of the risk/return distribution diagram.
- [SEC: stocks](https://www.investor.gov/introduction-investing/investing-basics/investment-products/stocks): ordinary shareholder priority and possible zero recovery.
- [ACCA: payback and discounted payback](https://www.accaglobal.com/gb/en/student/exam-support-resources/foundation-level-study-resources/ffm/ffm-technical-articles/discounted-payback.html): cumulative cash flows, discounting, timing conventions and the limitations of payback.

These references support the method. A company valuation still needs its own
dated financial evidence and explicit forecasts.
