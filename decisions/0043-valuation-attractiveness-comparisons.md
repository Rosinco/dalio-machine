# ADR 0043: Transparent valuation attractiveness and terminal sensitivities

Date: 2026-09-13

Status: Implemented under user direction; installation evidence is recorded separately.

## Context

The user requested sortable list fields for NPV, DCF and terminal attractiveness,
then asked to try different terminal settings before choosing a preference.
Whole-company amounts rank company size. Adding separate DCF, NPV and terminal
scores counts the same valuation inputs repeatedly. An arbitrary terminal weight
must not appear to be an empirically established probability or quality score.

## Decision

Add a percentage comparison to Companies → Lists:

`attractiveness = 100 × ((cash PV + (terminal credit / 100) × terminal PV) / saved equity price − 1)`.

All components retain the frozen starter's equity claim, currency and dated
price. Terminal value contributes once. At 100% credit, the measure is Mid NPV /
price. DCF / price gives exactly the same order. Zero credit compares the explicit
forecast cash PV with price; it is not liquidation value. Signed results remain
signed, including values below −100% when cash requires funding. A 30% discount
to positive value requires at least 42.857142…% NPV / price, not 30%.

Offer 100%, 75%, 50%, 25% and 0% credit as explicitly selected sensitivities.
Retain 100% as the original DCF comparison, without claiming it is the correct
valuation. A separate Terminal sensitivities column preset shows 100%, 50% and
0% side by side. Fifty percent is a useful stress, not a calibrated default.
Presets sort the full-DCF comparison descending and preserve existing filters
and watchlists. No opaque score or automatic investment verdict is added.

Supporting fields show full Mid NPV / price, full Mid DCF / price, annual cash
PV / price and Low NPV / price. Existing terminal share, evidence coverage,
dated price and required cash at a 30% margin remain available. A terminal share
is exposure to the post-forecast assumption, not a quality verdict. The forecast
horizon matters: extending an otherwise identical explicit forecast moves value
from the terminal component into annual cash without improving the business.

Only validated research-artifact rows may supply the comparison. Require a
finite reconciled DCF, annual cash PV, nonnegative terminal PV, positive finite
equity price, known valuation currency and valid saved quote/source dates.
Manual financial models, unresolved classifications/conflicts, no history and
reconciliation failures have no generic ranking. Partial histories remain labelled.
Do not reject an already verified FX conversion merely because quote currency
differs from valuation currency. Missing values stay missing and sort last.

Percentage cells are currencyless comparisons; source amounts and their currency
are visible in details. Saved price dates remain visible under company names when
a ranking column or condition is active, even after removing the price-date column.
Provider quote columns cannot reprice the frozen gauge. Reviewed and edited
company valuations stay in Value and are not silently substituted into Lists.

The exact credit is part of each column's calculation identity. Numeric rules,
saved views, reloads and CSV retain that identity independently of subsequent
edits to a visible column. Existing v2 preferences need no schema migration.
Browsing does not save or migrate list preferences or valuation drafts.

## Interpretation and evidence

The full-universe sensitivity experiment is recorded in
`docs/valuation-attractiveness-sensitivity-2026-09-13.md`. It changes assumptions
on a single saved snapshot. It is neither a point-in-time investment backtest nor
evidence that one terminal credit predicts future returns better. The cash-flow
range backtest does not calibrate terminal weights or whole-DCF probabilities.

Use the full comparison with terminal stresses and historical cash/EBIT evidence
to prioritize questions. Review sustainability, reinvestment, financing, ownership
and current price during the deep dive. Do not optimize a weight simply because
it generates an appealing number of candidates.

## Validation

Independently reconcile all universe rows at five credits; exercise valid FX,
signed/zero cash, nonfinite inputs, specialist routes and missing dates. Verify
numeric sorting across currencies, retained filter assumptions, saved views,
all-row CSV and price-date disclosure. Run the real production browser and native
Windows flows, including full process restart and unchanged authored valuation
bytes. Record final executable identity and installed shortcut separately; retain
the previous release and source artifacts.
