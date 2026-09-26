# Survival, permanent loss and portfolio shocks

Added 2026-09-26 under [ADR 0044](../decisions/0044-survival-and-portfolio-stress-review.md).

## Research workflow

Discover candidates in Lists; understand their businesses and reconcile cash;
review structural fragility and financing survival; examine explicit cash paths
and permanent impairment; compare value with a dated price; review portfolio
exposures before accepting an investment assessment. The formal Börsdata method
retains its price-blind structural gates and existing sizing policy.

In **Company → Research notes**, the company survival review and portfolio
worksheet open in separate expandable sections above the existing notebook. **Valuation** shows the saved company
review status and a direct link back. The worksheets do not set a buy verdict or
change the historical starter, Low/Mid/High, terminal credit or purchase ceiling.

## Liquidity and reverse stress

Choose a currency and equal quarterly or annual periods. Enter amounts in
company millions, with dated source evidence and a stated combined shock.

- Opening cash is unrestricted and available to this company/claim perimeter.
- Minimum liquidity is the cash needed to keep operating, not a debt deduction.
- Net cash generation is after operating spending, working capital, mandatory
  capex, interest, tax and leases, before separately entered principal and funding.
- Principal includes payments not already captured in net cash. Count each
  obligation once. Funding is incremental cash actually drawn in that period from
  committed facilities, subject to remaining capacity and stress conditions. Do
  not repeat the total undrawn facility each period or count opening cash again.
  Include repayment of new borrowing when due; hoped-for refinancing stays an
  unresolved assumption.
- Additional shock drain captures incremental cash costs not already in net cash.

At each endpoint:

`cash = previous cash + net cash - principal + available funding - shock drain`

`headroom = cash - minimum liquidity`

For an additional constant cash drain in every period, the amount reaching the
minimum-liquidity boundary is `min(headroom_i / i)`, where `i` starts at one.
If a shortfall already exists, no additional drain is needed. This is a boundary
sensitivity over the stated schedule, not a safe spending allowance. A later
recovery does not erase an earlier breach. Zero headroom leaves no cash buffer.

Assess intra-period timing, borrowing-base changes, covenant headroom, collateral
calls, customer/supplier dependence and legal/operating restrictions separately.
The worksheet does not simulate insolvency or prove committed funding drawable.
A material gap in that evidence keeps the research unresolved.

## Permanent impairment

Enter a separate annual path of net cash reaching the **existing common-equity
claim**, after financing and rescue dilution. Record the ownership and financing
assumptions; a surviving business can still leave its original shareholders with
little value. Enter final net equity proceeds, including zero where appropriate,
and the required equity return. Payments are at year end; final proceeds enter
only in the last year. The present value is their discounted sum.

Present values are measured at the liquidity opening date; annual payments
follow that origin even if liquidity periods are quarterly. Any optional NPV
compares that value with a dated price for the same existing claim and currency.
A missing origin or a price dated after the origin withholds NPV. Do not subtract debt already reflected in equity cash again,
or add impairment/recovery value to continuing-business DCF. Growth, margins,
moat and recovery can remain permanently weaker. No probability is assigned.

## Portfolio shared shock

This is a manual scenario, not a live holdings import or allocation recommendation.
Record dated exposures, a shared cause, evidence, assumptions and disjoint rows
with initial portfolio weights and assumed percentage losses. Enter zero only
when intentionally assuming no loss; missing values remain missing.

`portfolio loss (% of starting capital) = sum(weight% × loss% / 100)`

For example, two 20% positions each losing 100% imply a 40% loss of starting
capital. The remainder is unmodeled exposure held unchanged for this calculation,
not automatically cash or safe assets. Compare against the investor's explicitly
entered tolerable loss. A scenario within tolerance does not establish that every
plausible scenario is tolerable. Investor leverage, derivatives and forced-sale
funding needs require separate analysis; the worksheet assumes unlevered long
positions. Keep the existing position-size policy unless separately changed.

## Evidence, persistence and interpretation

Unsaved edits survive internal navigation in the current app session. Save the
worksheets explicitly before closing or reloading the app. Company reviews and portfolio
scenarios use separate local records from valuation drafts, revisions and the
notebook. Opening them creates nothing. Missing values, deliberate zeroes and
incomplete research can be saved. Unreadable or unsupported saved records are
preserved; storage errors stay visible. Source changes make a prior company
review stale until explicitly reassessed. Records are local to the browser/app
profile; a browser preview does not alter the installed Windows app's data.

A reviewed company status means only that the stated scenario, evidence and
assumptions have been recorded without a modeled liquidity failure. It is not a
formal structural-gate pass, a probability, a maximum loss or a promise of safety.
Historical cash-range coverage concerns the measured sample; it does not cover
all future regimes, delisted issuers or the complete DCF path. Normal and
fat-tailed distributions are modeling choices. Leave probabilities unset without
support, and keep modeled downside distinct from possible permanent loss.

## Verification of the local implementation

Verified 2026-09-26: production build succeeds; 334 unit/transport tests across
39 files pass; 74 browser checks pass (9 review, 10 navigation, 55 valuation).
Browser runs use isolated profiles under the app's content security policy and
report zero external requests and zero runtime errors. They cover missing/zero
inputs, a breach followed by recovery, reverse stress, permanent impairment,
portfolio arithmetic, stale sources, failed saves, unsupported records, internal
navigation, mobile layout and unchanged authored valuation content.

Commands from `desktop/`: `npm test`, `npm run build`, then against the production
preview `node tests/browser.mjs --resilience-only`, `--navigation-only` and
`--valuation-only`. Local reports and screenshots are under `desktop/test-results/`.
The [portable verification record](survival-review-verification-2026-09-26.json)
retains the tested implementation and report hashes.
A development-server attempt was blocked by the normal production CSP; one early
valuation run was interrupted by a concurrent rebuild. Final runs above use the
completed, stable production build. Existing bundle-size/import warnings remain.

At this verification checkpoint, the change was local and uncommitted.
No Windows package was installed and no normal-profile user records were modified.
The repository handoff records the later source commits and publication status;
Git publication does not establish native packaging or installation.
