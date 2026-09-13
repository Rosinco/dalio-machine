# Common-sense cash-flow segmentation concepts — 2026-09-12

This research note supports the separate empirical segmentation experiment. It
does not select a production forecast or revise any saved valuation. It follows
[the company valuation framework](company-valuation-framework.md): operating
economics, acquisition decisions, financing and the price paid are separate
questions. Numerical hypotheses below need out-of-sample evidence.

## Sector and branch are useful starting points

A branch gives a reasonable initial peer group because products, customers and
investment requirements may be similar. It cannot identify every company's
cash-generating mechanism. For comparison, GICS assigns one classification at
each level based on principal business activity, emphasizing revenues and also
considering earnings and market perception. A single label necessarily compresses
diversified businesses. Atlas uses its saved vendor taxonomy, not an assumed GICS
mapping. Current taxonomy applied to historical observations is a retrospective
descriptor unless the historical classification was retained.
[MSCI's classification description](https://www.msci.com/indexes/index-resources/gics).

Useful refinements answer different questions:

| Dimension | Common-sense mechanism | Evidence required |
|---|---|---|
| Customer demand and pricing | Repeat purchases, contracted prices and volatile market prices imply different exposure to surprises. | Contract length, recurring revenue, churn, customer concentration, commodity exposure and hedges. |
| Cost flexibility | Fixed commitments can amplify revenue shocks in operating profit. | Fixed/variable costs, labor flexibility, contracts and capacity utilization; an asset ratio alone does not measure these. |
| Investment needs and timing | Replacement, expansion projects and acquisitions affect available cash differently. | Gross capex, maintenance versus growth assumptions, acquisition and disposal cash. |
| Operating history | Stable positive cash and smooth margins are different from losses, reversals or rapid changes in scale. | Consistently defined annual cash, revenue and earnings known at the forecast origin. |
| Financing and liquidity | Debt maturities and restricted cash affect cash available to shareholders and adverse outcomes. | Gross debt, lease debt, interest, maturities, cash restrictions and covenants. |
| Measurement comparability | A change in accounting or vendor definition can resemble a business shock. | Source vintages, statement reconciliation and definition-change flags. |

The fixed-cost mechanism is established corporate-finance reasoning: a larger
fixed-cost share makes earnings more sensitive to revenue changes, all else
equal. This supports testing operating stability, but does not prove that a
particular balance-sheet ratio predicts our forecast errors.
[Damodaran's corporate-finance lectures, operating leverage](https://pages.stern.nyu.edu/~adamodar/pdfiles/cfovhds/cfpacket1spr24.pdf).

Banks and insurers need separate treatment: debt is part of their operating
business, regulatory capital constrains growth, and conventional working-capital
and capex definitions may be unsuitable. A generic provider cash-flow range
cannot become an economically meaningful shareholder-cash interval merely by
using a financial-sector percentile.
[Damodaran on financial-service firms](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/littlebook/financialsvccompanies.htm).

## What tangible book value / EBIT actually measures

With consistent ownership and accounting perimeters:

```text
Tangible book equity = common book equity − goodwill − other intangible assets
Tangible book equity / annual EBIT = a book-equity-to-operating-profit multiple
                                  = (tangible book equity / sales) / EBIT margin
```

Numerically the ratio is years of current EBIT relative to the book-equity
amount. It is not cash payback: EBIT precedes interest and tax, and does not
subtract the cash investment needed to maintain or grow the business. Its
inverse is also not a coherent operating return because its numerator measures
profit before financing while its denominator is equity after financing.

Two arithmetic examples show the confounding:

| Example | Tangible operating assets | Operating liabilities | Debt | Tangible book equity | EBIT | Book equity / EBIT |
|---|---:|---:|---:|---:|---:|---:|
| Unlevered operation | 1,000 | 200 | 0 | 800 | 100 | 8× |
| Same operation after a 400 debt-funded shareholder payout | 1,000 | 200 | 400 | 400 | 100 | 4× |
| Original operation during an earnings downturn | 1,000 | 200 | 0 | 800 | 20 | 40× |

The physical assets are unchanged in all three examples. Financing and the
profit margin change the ratio. It can still be tested as a descriptive predictor,
but a useful empirical result would not make it a pure asset-intensity measure.
Tiny positive EBIT makes it unstable; negative or zero EBIT and nonpositive
tangible book equity need separate states rather than an attractive rank.

Prefer separating the questions:

| Question | Reviewed measure | Meaning and qualification |
|---|---|---|
| How much fixed operating capital supports sales? | Average net PP&E / annual sales, with operating lease assets included consistently | Book fixed-asset intensity; not replacement-cost intensity. |
| How much total tangible operating capital supports sales? | Average tangible operating capital / annual sales | Includes working capital and operating liabilities with financing/non-operating items reconciled. |
| How profitable is that capital? | Normalized NOPAT / average tangible operating capital | After-tax operating return; also show an EBIT version when reliable operating taxes are unavailable. |
| What was the acquisition economics? | Normalized operating return on the full capital invested, including acquisition cost | Excluding goodwill from operating diagnostics must not erase the price paid for acquisitions. |
| How much equity cash is exposed to debt claims? | Separate liquidity, interest-coverage and debt-maturity analysis | Neither tangible book equity nor net debt alone captures refinancing risk. |

Berkshire's 1983 discussion supports distinguishing underlying returns on
unlevered tangible capital from the full economic acquisition cost. It is a
business-economics argument, not evidence that high tangible returns guarantee
predictable future cash or attractive current investment returns.
[Berkshire 1983 letter and goodwill appendix](https://www.berkshirehathaway.com/letters/1983.html).

Asset-light on the balance sheet does not mean investment-free. IAS 38 expenses
research and excludes internally generated brands and similar items from
recognized intangibles, while qualifying development spending may be capitalized.
Therefore internally built and acquired capabilities can look different in book
capital. [IFRS Foundation, IAS 38](https://www.ifrs.org/issued-standards/list-of-standards/ias-38-intangible-assets/).
IFRS 16 brings most lessee commitments into right-of-use assets and lease
liabilities; operating-asset and financing perimeters must include them
consistently. [IFRS Foundation, IFRS 16](https://www.ifrs.org/news-and-events/news/2019/01/ifrs-16-is-now-effective/).
Net PP&E is affected by depreciation, impairments and the choice of cost or
revaluation model. Old depreciated assets can look unusually capital-efficient
without having low replacement requirements.
[IAS 16, paragraphs 30–31](https://www.ifrs.org/content/dam/ifrs/publications/pdf-standards/english/2022/issued/part-a/ias-16-property-plant-and-equipment.pdf?bypass=on).

## Small feature set feasible from the saved pack

Derive these only from admissible history through each forecast origin. Report
missingness and keep currencies, periods and data definitions consistent. Use
coarse, prespecified groups or cut points learned exclusively in calibration;
avoid searching dozens of thresholds against recent outcomes.

| Feature family | Feasible proxy | Main caution |
|---|---|---|
| Branch | Saved sector and branch | Current classification is not historical classification. Pool thin branches toward sector/global estimates. |
| Cash stability | Five-year sign persistence; dispersion and adjacent changes relative to a positive historical absolute-cash scale | Near-zero scale, secular growth and definition changes can dominate the statistic. Preserve a separate unavailable state. |
| Demand and margin stability | Annual revenue changes; EBIT/revenue level and dispersion | Require positive revenues for ratios; an annual series is short, and acquisitions can alter business scale. |
| Tangible asset intensity | Average saved `tangible_assets` / revenue | Call this a vendor tangible-assets proxy, not verified PP&E. Inspect leases, property, biological assets and field mapping. |
| Tangible capital intensity | Average (`total_equity` + `net_debt` − `intangible_assets`) / revenue | An approximate financing-side proxy only; total versus common equity, operating cash, non-operating investments and other claims are not fully reconciled. Nonpositive capital needs a separate state. |
| Financing | `net_debt` / `total_assets`, preserving net-cash observations | Not gross leverage, liquidity or maturity risk; do not replace gross-debt analysis. Financial firms are separate. |
| Investment-flow pattern | Net investing cash relative to positive revenue or total absolute operating/investing cash; frequency of positive investing flows | Net investing includes asset/business transactions and investments; this is not capex intensity or maintenance capex. |

The user's tangible-book-equity/EBIT idea is a worthwhile additional comparator,
using `total_equity − intangible_assets` only as an explicitly labelled book-equity
proxy. The pack lacks the reconciliations needed to call it exact common tangible
book value. Do not subtract goodwill a second time if it is already included in
the intangible total.

Do not manufacture missing operating working capital from current assets minus
current liabilities without addressing cash, investments and short-term financing.
Those adjustments matter to operating investment; the saved aggregate columns
do not supply a fully reconciled answer.
[Damodaran on working capital](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/noncashwc.htm).

IAS 7 separates operating, investing and financing flows; investing includes
long-term asset and business acquisitions/disposals. This supports retaining
those components separately rather than treating net investing cash as a
universal capex measure. [IFRS Foundation, IAS 7](https://www.ifrs.org/issued-standards/list-of-standards/ias-7-statement-of-cash-flows/).
The first audit's observed provider FCF revisions remain an unresolved measurement
issue. Segmenting that error cannot by itself establish business predictability.

## What would make a distinction helpful?

An interpretable grouping is useful if it improves prospective coverage at a
given width, narrows intervals at comparable coverage, or improves a proper
interval score across sufficiently broad cohorts and several time periods.
Evaluate point errors independently: changing the uncertainty band does not
improve the mid forecast. Show which groups under-cover and over-cover, signed
misses, sample sizes, and performance by target year; an overall 80% hit rate can
conceal bad subgroup calibration.

Compare sector, branch and each numeric feature against one pooled baseline on
the same eligible observations. Then test whether a feature adds value within
branch, rather than merely rediscovering sector membership. Share classes and
repeated observations from one issuer must not masquerade as independent support.
Keep thin groups pooled; do not award a permanent company accuracy label from
one or two recent outcomes. Today's reuse of already-inspected test years is
exploratory evidence and needs a new untouched period for confirmation.

After the simple tests, the most useful added inputs may be cash-flow components
rather than more ratios. An original study by Barth, Cram and Nelson found that
separate receivable, payable, inventory, depreciation, amortization and other
accrual components improved future-cash-flow prediction in its sample. That
supports investigating a richer operating cash bridge, not assuming the same
improvement for today's global provider FCF data.
[Stanford publication record and study abstract](https://www.gsb.stanford.edu/faculty-research/publications/accruals-prediction-future-cash-flows).

For deep dives, record recurring/contracted revenue, pricing resets, customer
concentration, project capex, acquisitions/disposals and financing deadlines as
dated evidence. Review the largest surprises by these mechanisms and record
whether the warning was knowable at the forecast origin. A cause identified
after the outcome is an explanation to investigate, not a usable earlier predictor.
