# Standard company DCF and NPV

Adopted 12 September 2026 and extended on 13 September for Macro Atlas 0.14.0,
with chart presentation clarified in 0.14.1, separate terminal cash in 0.15.0
(ADR 0038), and editable purchase-price ranges in 0.16.0 (ADR 0039).
Current method: `empirical-cash-starter-v3`, following
[ADR 0037](../decisions/0037-empirical-company-cash-flow-starters.md). Earlier
`weighted-cash-starter-v1` and `weighted-cash-starter-v2` studies remain supported
with their original definitions. Release verification and installation status
are recorded in the desktop handoff, independently of this methodology.

Every listing in Macro Atlas has the same Companies → Value workspace. Its
standard starting point uses saved historical cash flow. A company deep dive
refines the cash forecasts, price, financing treatment and recovery assumptions
in that workspace. Reviewed studies currently exist for Holmen and SCA and open
as the default refinements. The standard historical baseline remains available
alongside them. Applying it preserves the preceding draft as a saved revision.

## Purchase price and margin of safety

The user-selected default is **30% below the Mid scenario's equity value**,
editable per company alongside the reference scenario. For each scenario:
`purchase ceiling = (cash PV + terminal PV) × (1 − margin/100)` and
`NPV at proposed price = cash PV + terminal PV − proposed price`.
Terminal value appears once. Recovery/crisis are separate alternatives.

The NPV-versus-price chart shades positive prices at or below the selected ceiling.
Lower prices increase the modeled discount; the lowest of the three scenario
ceilings is not a minimum acceptable price. Nonpositive DCF or a 100% margin
produces no positive ceiling. Missing inputs remain unavailable. Cash-only
ceilings and terminal shares expose terminal dependence, not a recovery floor.
This margin is an assumption, not a calibrated probability or extra annual return.

Total equity millions are always the initial units. Optional per-share display
requires an explicit dated share count/source for the same equity claim and
currency; equity millions divided by shares millions gives currency per share.
Available reviewed or reported-share references can be applied deliberately.
Reported shares remain an unreviewed ownership proxy. Price, currency or claim
changes invalidate automatic reference suggestions. The saved market quote is
dated, not live; absence of a quote does not prevent cash-based valuation.

Settings, cleared inputs and source descriptions persist with the working draft
and revisions. History application and explicit baseline/reviewed resets preserve
the independently authored purchase policy. The separate CSV retains partial
results and missing/error states. See [ADR 0039](../decisions/0039-editable-purchase-price-range.md)
for source guards, formulas and validation boundaries. A deep dive still needs
to establish sustainable equity cash and business quality before price matters.

## Version 3: latest cash and historical-error ranges

The default selects **five annual periods** and holds the latest eligible signed
annual cash flow flat as the provisional mid forecast. It is a simple benchmark,
not an optimal ten-year forecast. Weighted historical trend and weighted flat
mean remain editable alternatives. The saved provider `free_cash_flow` remains
an unreviewed cash proxy until a deep dive reconciles operations, investment,
leases, financing and the cash that can reach common shareholders.

For comparable operating/property histories, the default range uses the immutable
`cash-uncertainty-2026-09-12-v1` calibration. Its original research retains FY2020
and the raw COVID/recovery histories; the later COVID-omission sensitivity is not
the default. The bundle copies exact factors and support counts without refitting,
and records source hashes, dates and measured evaluation results.

```text
historical cash scale = mean(abs(CF1), ..., abs(CF5))
cash dispersion = population standard deviation(CF1, ..., CF5) / historical cash scale
group = low when dispersion < 0.25
        medium when 0.25 <= dispersion < 0.75
        high when dispersion >= 0.75
Year 1-4 half-width = selected model/horizon/group factor * historical cash scale
low/high cash = mid cash minus/plus that half-width
```

Dispersion uses all five signed annual observations and an unweighted mean
absolute scale. It includes growth and changes in business scale, not only
unpredictable shocks. Factors target **80% empirical annual coverage** and are
specific to latest-cash or the original weighted-trend model. Supported groups
need at least 300 calibration observations, 100 listings and 100 distinct origin
histories. An unsupported group uses its matching global pool; that larger peer
pool does not necessarily produce a wider numerical band.

Empirical eligibility requires exactly five consecutive comparable full years,
positive historical cash scale, native reporting-currency cash, original
30/25/20/15/10 settings and either latest-cash or weighted-trend projection.
Financial package identity, company taxonomy and each annual source's ID, hash,
path and date must match the retained calibration. Publication/period ordering,
withheld rows, freshness and existing source-quality guards also apply. Calibration
targets end in FY2020 with a nominal 30 June 2021 publication cutoff; eligible
forecast origins must follow that cutoff. New source packs require evaluation
before they can inherit this calibration's empirical label.

Partial/ten-year histories, custom weights, weighted-flat projection, unavailable
classification or incompatible source/currency/version inputs receive explicitly
labelled **percentage sensitivities** when valid mid cash exists. Missing source
cash remains missing. Financial businesses other than the saved Real Estate and
REITs branches require manually reviewed equity cash and regulatory/financing
inputs; their generic provider cash is shown as history but is not automatically
projected. Property proxies still require cash-definition review before an
economic valuation conclusion.

Historical ranges remain nonzero around a zero mid when the five-year historical
scale is positive. When no empirical range applies, a percentage of zero supplies
no useful uncertainty estimate: v3 keeps low/high inputs missing until explicit
cash scenarios are entered. Negative cash and ranges crossing zero stay signed.
A very small positive scale is disclosed for source-unit and cash-definition
review rather than silently raised to an invented floor.

Years **5-10** use an editable assumption instead of additional empirical factors:

```text
half-width(t) = Year 4 half-width
               + historical cash scale * assumed annual tail widening / 100 * (t - 4)
default assumed annual tail widening = 10%
```

The added amount is 10% of the historical cash scale per year, not ten percentage
points around the future mid and not compounded growth. Year 1-4 rows are labelled
historical; later rows are labelled assumed extensions. There are no fitted
Year 5-10 factors in the bundle. Year 1 evidence is strongest, Year 2 has fewer
validation years, and Years 3-4 cover only recent regimes. The annual 80% research
target does not establish probabilities for an individual company, the joint cash
path, low/mid/high DCF values or terminal sale proceeds.

## Separate terminal cash in 0.15.0

New generic starters add `terminalMethod: historical-median-v1` to the existing
cash-method origin. The annual forecasts and their calibration are unchanged.
Terminal sale now uses independently editable first post-horizon equity cash:
`max(0, cash_N+1) / ((required return - mature growth) / 100)`. This cash must
already be after required reinvestment and financing. It is not multiplied by
growth again. Required return must be positive and mature growth must be above
-100% and below that return. Missing/invalid assumptions remain unavailable.

The unreviewed starting point is the signed median of at least three consecutive
eligible historical annual cash observations. Low/high subtract/add an assumed
20% of its absolute amount; mature growth starts at 0%. This is a separate
terminal sensitivity, not an empirical 80% band. It does not establish sustainable
owner cash or investment needs. All shock/rebound observations remain in history.
Short histories and manual financial models need explicit terminal inputs.
Nonpositive terminal cash yields zero assumed sale and requires turnaround/run-off
review. The last forecast payment remains visible to inspect the transition.

The working table shows actual terminal cash, growth, return and sale assumptions.
A return-sensitivity table recalculates both cash PV and sustainable-cash sale
at each rate. Explicit sale proceeds stay fixed. Editing final sale selects that
manual mode. Terminal input edits, rate edits and notes no longer erase the
historical provenance of otherwise unchanged annual cash. The crisis case keeps
its own explicit sale. DCF and final-year NPV use the same resolved sale once.

Applying history settings preserves existing independent terminal assumptions
and required returns on the current generic study. Deliberately opening the
standard baseline resets assumptions after preserving the preceding draft.
Only exact untouched old defaults may upgrade automatically after successful
backup and replacement persistence. All edits, blanks, custom calibration/models,
crisis, restored revisions and reviewed studies remain intact. Missing
terminalMethod on an old v3 study reproduces its original final-forecast sale.

The evidence view also exposes same-report cash components and arithmetic
checks. Aggregate investing cash is not maintenance capex, and matching provider
FCF is not verification of distributable cash. Missing subcomponents and possible
source placeholders remain explicit. See [ADR 0038](../decisions/0038-separate-sustainable-terminal-cash.md)
and the [component audit](cash-component-audit-2026-09-13.md).

## Historical inputs and legacy weighted baselines

Choose five or ten consecutive full annual periods, newest first. Five years is
the default. Selecting a different window supplies its default weights, which
can then be edited:

| Annual period | Five-year weight | Ten-year weight |
|---|---:|---:|
| Most recent | 30% | 19% |
| Second most recent | 25% | 17% |
| Third most recent | 20% | 15% |
| Fourth most recent | 15% | 13% |
| Fifth most recent | 10% | 11% |
| Sixth most recent | — | 9% |
| Seventh most recent | — | 7% |
| Eighth most recent | — | 5% |
| Ninth most recent | — | 3% |
| Tenth most recent | — | 1% |

For five usable periods, the weighted historical cash-flow mean is
`0.30 × CF1 + 0.25 × CF2 + 0.20 × CF3 + 0.15 × CF4 + 0.10 × CF5`.
Weights must sum to 100%. This mean is retained in the evidence. The optional
trend estimates annual change; the v3 latest-cash mid does not use this weighted
mean as its starting level.

With fewer consecutive usable years, normalize their included weights explicitly:
`sum(weight × cash flow) / sum(included weights)`. The evidence table shows
original and effective weights. Stop at gaps, unsupported periods, unavailable
amounts or inconsistent currencies; do not skip an intervening year to find a
more favorable one. An unavailable latest period cannot be replaced by an older
cash flow. Entirely zero statement rows are treated as unsupported source
placeholders; an isolated zero cash flow in an otherwise populated report stays
zero. Losses and sign changes remain visible.

The source measure is the saved vendor `free_cash_flow` field. It is an
unreviewed cash-flow proxy, not verified FCFF or a reported dividend. Definitions
can omit lease principal or reflect different interest, acquisition, disposal
and reinvestment treatment. Do not infer a universal formula from operating and
net investing cash flow. Banks, insurers and other financial companies need
particular review because their cash-flow statements do not establish ordinary
industrial free cash flow or cash distributable within capital requirements.

## Optional historical trend and legacy v2 midline

The weighted-trend alternative, which was the v2 default, fits a straight line
to the included annual cash-flow proxies. Let `x = report year − latest included report year`: the
newest annual observation has `x = 0`, the preceding year `x = −1`, and so on.
Use the entered weights after the disclosed partial-window normalization.

```text
x̄ = weighted mean of x
ȳ = weighted mean of historical cash flow
slope = sum(weight × (x − x̄) × (cash flow − ȳ))
        / sum(weight × (x − x̄)²)
fitted latest level = ȳ − slope × x̄
mid cash in forecast year t = fitted latest level + slope × t
```

The fitted latest level is an estimated intercept; it need not equal the latest
observed cash flow or the weighted mean. At least two distinct positive-weight
years are needed to estimate a slope. With one such year, the slope is explicitly
assumed zero and the evidence reports that limitation. No positive included
weight leaves the forecast unavailable. Selecting **Flat weighted average**
instead holds the weighted historical mean constant while retaining the chosen
annual range settings.

Forecast years `t = 1…10` are full model years after the valuation date. A trend
uses annual steps from the fitted latest-period level; latest-cash and flat-mean
alternatives retain their respective starting levels. None estimates an interim
cash-flow bridge between the last fiscal end and valuation.
Historical fiscal periods, valuation date and source download date remain
separately labelled. A fitted historical trend is a model assumption about the
future, not management guidance or a validated predictor.

## Percentage fallback, legacy sensitivities and valuation assumptions

The final-forecast capitalization described below applies to retained pre-0.15
studies. New generic studies use the separate terminal model above.

The standard model assumes the selected historical cash proxy can reach common equity.
That assumption must be reviewed in a deep dive before interpreting the result
as a company valuation. It does not automatically deduct debt again or add book
assets to a forecast already supported by those assets.

- V2 mid annual path: the weighted fitted trend, or its selected flat alternative.
  V3 also offers the default latest-cash flat path.
- V2 default and v3 fallback percentage range: **±10% in year 1, ±20% in year 2, ±30% in year 3**,
  increasing by ten percentage points each year to ±100% in year 10.
- The initial percentage and annual percentage-point increase are editable.
  For year `t`, `spread = initial + annual increase × (t − 1)`.
- Low cash: `mid − abs(mid) × spread / 100`.
- High cash: `mid + abs(mid) × spread / 100`.
- Default forecast horizon: ten full years, with payments at year-end.
- Default nominal required equity return: 10%, an editable model assumption.
- Final equity sale: positive final annual cash divided by the required return,
  assuming unchanged post-horizon cash and zero perpetual growth. The fitted
  annual trend is not continued perpetually through this sale calculation.
  Nonpositive cash has zero assumed sale proceeds in the starter; this is not a recovery
  appraisal. Review the separately editable sale when changing cash or rates.

The percentage range can exceed 100%; it is not capped and paths may cross zero.
Using the absolute mid value preserves the low/mid/high ordering for negative
cash flows. Exact v1/v2 reproduction retains its zero absolute range at a zero
mid, without establishing zero uncertainty; v3 instead leaves those percentage
bounds missing as described above. The percentage range
widens over time when the annual increase is positive, while its cash amount also
depends on the projected mid value. Negative forecast payments illustrate funding
needs under the model; they do not establish a shareholder's legal obligation to contribute
capital. The spread is a sensitivity range, not a statistical confidence interval
or guaranteed bound. Scenario probabilities are unset.

## Separate crisis and recovery scenario

An optional crisis scenario applies explicit assumptions to the current mid
path, independently of low/mid/high and independently of liquidation recovery.
It starts **disabled**. Initial editable assumptions are a **40% cash reduction**
starting in **Year 1**, **two shock years**, **three recovery years**, **zero extra
annual cash cost**, a **10% required equity return** and **zero final equity sale**.
These are illustrative form defaults, not a forecast or an assigned probability.

During a full shock, `crisis cash = mid - abs(mid) * reduction / 100 - extra cost`.
The percentage reduction and extra cost fade evenly over the recovery years,
reaching the current mid path in the final recovery year. Zero recovery years
means immediate recovery. A negative mid becomes more negative under a shock.
Missing mid inputs remain missing. The crisis can have its own required return
and final sale; changing annual cash does not silently invent a revised terminal
value. If disruption extends beyond the forecast, the remaining effects need
explicit terminal review.

The crisis panel shows nominal cash, discounted cash and cumulative NPV, with
its own annual table and CSV rows when enabled. It does not overwrite ordinary
scenario cash, add recovery to going-concern value, or remove pandemic observations
from history. Dated company evidence should distinguish shutdowns, unusual demand,
support measures, inventory timing, investment and later rebounds. A deep-dive
normalization belongs in a separate explained revision; raw source facts remain
unchanged.

## Cash flow over time, DCF and NPV

The **Cash flow over time** chart places annual time periods on the horizontal
axis and nominal company cash flow in the selected currency's millions on the
vertical axis. It shows the usable historical observations, the fitted historical
line for an unchanged trend starter, the current low/mid/high forecast and its
shaded range. Historical fiscal-year labels and future model-year labels mark
the transition. An inspectable table retains the annual scenario amounts.

This chart shows cash before discounting and can display valid forecasts while a
price or discount-rate input is missing. Its historical series is hidden if the
user changes the forecast to a different currency; no conversion is inferred.
Cleared forecast inputs leave gaps instead of connecting a shaded range through
missing years. Manual edits and reviewed deep dives use their current annual
cash paths in the same chart.

The DCF and cumulative NPV charts remain below it and discount those same future
payments. Version 0.14.1 adds annual minimum/maximum markers and shaded scenario
ranges to these charts. This changes presentation only: entered cash, required
returns, investment scaling, valuation calculations and saved drafts retain their
existing meanings. The nominal cash-flow range itself is not a discounted value
range.

For nominal cash and annual DCF, the shaded bounds follow the minimum and maximum
of all three named scenario lines, including their exact linear crossings between
annual points. A scenario named Low need not remain the numerical minimum after
edits or discounting at different required returns. Preserve each scenario's name
and colour as lines cross; do not reorder or smooth the inputs. Annual whiskers
show the minimum and maximum at each forecast year. The inspectable DCF/NPV tables
retain both those bounds and each named scenario's calculated amount. Range widths
are not forced to increase: signed cash, discounting and crossings can narrow them.

The NPV chart accumulates each scenario's discounted payments independently and
subtracts the initial investment. Its bounds are the minimum and maximum of those
complete cumulative scenario paths at each year, not sums of annual minima and
maxima from different scenarios. Year-end steps retain the previous value until
the next payment. When the final-sale switch is enabled, each scenario's discounted
equity sale enters only its final NPV point. Annual cash-payment DCF, ordinary and
discounted payback, and the separate recovery analysis keep their existing roles.

For eligible unchanged starters, the source-basis labels retain the distinction
between historical-error cash ranges in Years 1–4 and assumed widening in Years
5–10. Edited, reviewed and percentage-assumption paths retain their own labels.
These labels explain the underlying cash assumptions; they do not establish an
80% DCF or NPV interval, a joint forecast-path probability or a terminal-sale
probability. The displayed value range is the envelope of the current scenarios.

## Dates, currencies and gaps

Use source-bound annual financial rows and a valid dated publication-window
listing price. Financial and price dates remain visible. Freshness, share-basis
and source-lineage checks determine whether a numeric comparison can be shown.
Saved quotes are historical, not live prices.

The financial pack preserves both reported-currency values and vendor-converted
amounts. Source documentation defines stored amounts as reported amounts times
`currency_ratio`, in the stock-price currency. A quote-currency baseline requires
matching annual/market source lineage for every included year. Its per-period
historical conversion ratios are shown; no new spot FX is invented. A market
value in a different currency is not silently compared with the cash forecast.

When source data is missing or unusable, the listing still has the standard
controls and the cash-flow/DCF/NPV chart sections. Explicit missing inputs replace
unsupported curves. Users can enter reviewed assumptions to complete the calculation.

## Deep dives, editing and saved revisions

The annual cash forecasts, discount rates, final sale, dated price and separate
recovery can all be edited. The history window, weights, projection and range
settings regenerate a starting path; applying them first saves the current work. Deep dives replace proxy
assumptions with reconciled source facts and company-specific forecasts.
Holmen and SCA retain their reviewed defaults and independent recovery schedules.

Drafts and revisions retain their exact company and research/financial package
identities. Starter method, history window, weights, projection, initial spread,
annual spread increase and dates survive reload. V3 additionally retains range
mode, calibration ID and assumed tail widening; optional crisis inputs and their
deliberately cleared values are saved with the draft. CSV exports distinguish
historical, assumed-tail, percentage and unavailable annual ranges from edited
cash and the separate crisis assumptions. Exports distinguish standard starters from reviewed studies and identify edits from each
starting model. Updating an application does not silently replace entered
forecasts or cleared inputs. Version 1 studies retain their five-year weighted
mean, flat paths and constant saved range when reconstructed or restored.

Only an exact untouched v1 or v2 default may upgrade automatically to v3. V1
must match its original five-year weighted-flat generator and constant 20%
range; v2 must match its original five-year weighted-trend generator and 10% plus
ten-per-year percentage range. The investment amount may differ and is retained.
The original draft must be saved as a revision before replacement. Edited titles,
notes, forecasts, cleared fields, custom settings, crisis assumptions and explicitly
restored studies prevent silent replacement. Reviewed studies remain reviewed.
For a preserved model, **Use empirical defaults** fills the controls;
**Apply history assumptions** saves the current revision before applying v3.

DCF discounts the annual forecast cash. Cumulative NPV subtracts the purchase
price from cumulative discounted receipts; its final-sale switch is explicit.
Ordinary and discounted payback are separate. A separate recovery estimate
requires asset, claim, cost and timing evidence, and is never auto-filled from
book equity or added to the continuing-business DCF.

See the [company valuation framework](company-valuation-framework.md),
[Holmen study](holmen-valuation-2026-09-11.md) and
[SCA study](sca-valuation-2026-09-12.md).

## Forecast accuracy and backtesting

The first offline [cash-flow backtest audit](cash-flow-backtest-2026-09-12.md)
measures the mid forecast separately from range coverage and width. It includes
company-level misses and source-backed case reviews. That research audit did not
replace the installed starter or deep-dive revisions. Original percentage bands remain
sensitivity assumptions. Learned research ranges have explicit calibration periods,
source-vintage limitations and sample counts; adopting selected factors as v3
starter assumptions does not establish guaranteed confidence intervals. Provider cash-definition differences must be reconciled before treating
cross-vintage errors as business unpredictability. Freeze new source snapshots and
baseline forecasts so later deep-dive adjustments can be compared with what was
actually forecast at the time.

The follow-up [uncertainty grouping experiment](cash-flow-segmentation-2026-09-12.md)
tests sector, branch and business characteristics on the same forecasts.
Historical cash dispersion improves conditional coverage and interval score more
than the tested taxonomy or asset-intensity proxies. Results remain exploratory:
coverage varies by year, sparse groups need broader fallback, and provider cash
definitions/current-vintage data limit interpretation. Tangible capital needs,
normalized returns and financing commitments remain useful separate deep-dive
dimensions; book equity/EBIT is not a pure asset-intensity measure. These findings
support the limited v3 adoption in ADR 0037; they do not establish calibrated
DCF value ranges. Preserve the original research outputs and assess future
untouched outcomes before replacing the dated calibration.
