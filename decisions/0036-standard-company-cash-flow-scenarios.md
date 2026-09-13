# ADR 0036 — Standard historical cash-flow scenarios across the company universe

Accepted 2026-09-12; extended by the user's subsequent historical-trend and
widening-range request in Macro Atlas 0.13.

## User direction

DCF and NPV charts are a standardized part of every company in the universe.
Open with editable historical starter scenarios, then refine them through deep
dives. The initial request used five annual cash flows weighted 30%, 25%, 20%,
15% and 10%, with a constant ±20% range (`weighted-cash-starter-v1`). The follow-up
requests a cash-flow-versus-time chart, a midline based on the trailing five or
ten years, and a range that widens from ±10% in year one to ±20% in year two,
±30% in year three, and so on.

## Decision

Use the common Value workspace for every listing. Calculate a versioned,
transparent weighted historical cash-flow starter on demand from its selected
financial pack. Display original and effective history weights, source dates,
currency basis, forecast and terminal assumptions. The range is a sensitivity
assumption, not a calibrated confidence interval. Preserve losses, missingness,
source quality flags and the exact listing/price basis.

Method `weighted-cash-starter-v2` defaults to weighted linear regression over
five full annual periods, using the existing 30/25/20/15/10 weights. A ten-year
window uses 19/17/15/13/11/9/7/5/3/1. Weights remain editable and sum to 100%.
The latest historical year is time zero; the fitted intercept and slope project
mid cash as `intercept + slope × forecast year`. The weighted historical mean
remains visible, and an explicit flat-mean alternative is available. One
positive-weight observation gives a disclosed zero slope. Missing history keeps
the existing contiguous-window, freshness, currency and source-quality guards.

Year `t` has an editable percentage spread of `initial + step × (t − 1)`,
defaulting to 10% plus ten percentage points per year. Low/high cash equals mid
minus/plus its absolute value times that spread. Bands may exceed 100% and cross
zero; signed values retain their labels. This assumed widening is not a
statistically calibrated confidence interval. The default ten-year, 10%-return
valuation capitalizes only positive final cash with zero post-horizon growth;
the fitted trend is not perpetuated in the terminal sale.

Add a nominal company cash-flow chart with annual time on x and cash amounts on
y. Show source history, the applicable fitted history, current forecast paths
and the annual range before discounting. Preserve missing-year gaps and currency
boundaries. DCF, cumulative NPV and payback continue to use the same editable
future cash, with separate final-sale and recovery treatment.

The saved vendor cash-flow field is a proxy requiring company review, not
verified cash distributed to shareholders or universally comparable FCFF.
Incomplete or unsupported sources leave explicit gaps and chart placeholders.
No company asset recovery is guessed from book equity. Deep dives refine the
same editable model, with reviewed sources and separately supported recovery.

Holmen's immutable study remains unchanged. SCA is the second reviewed case.
A reusable v2 study contract separates explicit forecasts, source evidence and
recovery, replacing shared Holmen-specific text. Reviewed v2 registration requires
listing ID, ISIN, archived document path and hash from the selected validated
research resources; no change to the existing package identities is required.

Every reset preserves the preceding draft as a revision. Starter method/settings
and reviewed-study origin remain distinct in local storage and CSV exports.
Data loads only for the selected listing, keeping the offline universe usable.

Keep the v1 storage contract and original generated drafts compatible. Only an
exact untouched v1 default draft may automatically receive v2 defaults, with a
saved backup first and its investment amount retained. Custom weights, edited
text or numbers, cleared inputs and deliberately restored studies keep their
selected model. An explicit settings/apply path lets users adopt v2 themselves.

Details: [standard model](../docs/standard-company-valuation.md) and
[SCA worksheet](../docs/sca-valuation-2026-09-12.md).

## Verification

Verify independently calculated weighted regressions over five/ten-year windows,
linear and uneven samples, negative values, zero crossings, one positive-weight
year, widening beyond 100%, and both origin versions. Preserve the exact complete
v1 draft and reviewed Holmen draft. Verify weighted-history arithmetic, source,
currency and date guards, partial-history disclosure, SCA numerical
reconciliation and per-company draft persistence. Exercise representative positive,
negative and absent-data companies in browser and native Windows flows, including
window/projection/weights/range editing, untouched-v1 backup/migration, preserved
edited-v1 studies, nominal cash-flow versus discounted charts, CSV provenance and
a full process restart.
