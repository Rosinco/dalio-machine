# ADR 0037 — Historical-error company cash-flow starters and separate crisis assumptions

Accepted 2026-09-13 for Macro Atlas 0.14.0. Extends
[ADR 0036](0036-standard-company-cash-flow-scenarios.md); its v1/v2 definitions
remain the reproduction contract for saved earlier studies. Release verification
and installation status are recorded separately in the desktop handoff.

## Context

The user approved implementing the recommended setup across the whole company
universe after reviewing cash-flow backtesting, sector/branch and financial-pattern
grouping, and COVID sensitivity. The common chart/workflow remains universal;
cash definitions and empirical eligibility must fit the business and source data.

The fixed research found more useful conditional coverage from historical cash
dispersion than from the tested taxonomy or asset-intensity ratios. The latest-cash
benchmark also reduced one-year point error relative to the full trend in the
initial sample. These are exploratory findings with later-vintage statements,
provider cash-definition differences and only a few evaluation regimes. They do
not establish company probabilities or a validated ten-year valuation model.

## Decision

Introduce `empirical-cash-starter-v3`. Default to five annual history periods,
latest signed annual cash held flat, and historical-error ranges where the exact
source/method eligibility rules are met. Keep weighted trend and weighted flat
mean available as explicit alternatives. Preserve cash losses, gaps and source
dates; do not remove COVID or subsequent recovery observations automatically.

Copy the original all-year research factors into the immutable dated bundle
`cash-uncertainty-2026-09-12-v1`; do not refit them or substitute the COVID-omission
experiment. Its five-year scale is mean absolute signed annual cash, and its
dispersion is population standard deviation divided by that scale. Fixed groups
are low below 0.25, medium from 0.25 to below 0.75, and high from 0.75.

Years 1-4 use the model/horizon/group factor times the historical cash scale as
absolute half-width. Factors target 80% annual empirical coverage. A supported
group needs 300 observations, 100 listings and 100 distinct origin histories;
otherwise use the matching global pool. A broader peer pool does not necessarily
give a wider numerical interval. Sector/branch remains useful context and scope
classification; additional subdivision is not presumed to improve the range.

Require exactly five consecutive comparable full annual reports, native reporting
currency, positive scale, the original five-year weight settings and either
latest-cash or original weighted-trend projection. Bind financial package,
taxonomy, annual source ID/hash/path/date and calibration version. Retain the
publication cutoff, period ordering, withheld-data and existing quality guards.
New or incompatible source versions receive explicit percentage assumptions until
evaluated; they do not inherit the research label automatically.

Partial/ten-year histories, custom weights, weighted-flat projection and other
noncomparable settings retain editable percentage sensitivities when cash inputs
are available. Missing source cash is not replaced by an older or invented value.
Financial businesses other than the saved Real Estate and REITs branches require
reviewed distributable-equity and capital/financing inputs; industrial provider
FCF is not projected for them automatically. Property cash is still an unreviewed
proxy requiring reconciliation before an economic valuation conclusion.

A zero forecast mid can retain a nonzero historical-error band when its history
has positive scale. If only a percentage of zero is available, v3 leaves low/high
cash missing for explicit entry rather than displaying zero uncertainty. Negative
cash and zero-crossing bounds remain valid. Very small scales receive a source-unit
review explanation rather than an invented normalization floor.

After Year 4, use a separately labelled assumed extension: carry the Year 4
absolute half-width forward and add an editable percentage of historical cash
scale for each later year, initially 10% per year. This is additive and not
compounded. There are no calibrated Year 5-10 factors. The ten-year horizon,
required equity return and final equity sale remain explicit editable valuation
assumptions. Annual ranges do not establish joint-path or DCF probabilities.

Add a distinct crisis scenario based on the current mid cash path. It starts
disabled and has no assigned probability. Initial form assumptions are 40% cash
reduction, start Year 1, two shock years, three recovery years, zero extra annual
cash cost, 10% required return and zero final equity sale. During the shock,
subtract the percentage of absolute mid cash and extra cash cost; fade both
evenly during recovery. Negative mid cash can become more negative. Its separate
nominal cash/DCF/NPV output does not overwrite ordinary scenarios or add liquidation
recovery. Final sale remains an independent assumption, including when disruption
extends past the forecast.

Retain raw COVID and rebound histories. Any company-specific normalization belongs
in a dated, explained deep-dive revision, with source facts preserved. Crisis
stress assumptions are not a mechanism for deleting historical misses or assigning
a pandemic probability. A future pre-crisis coverage test requires factors fitted
using information preceding the shock, rather than the current FY2020-inclusive
calibration.

Keep original v1/v2 generators and reviewed Holmen/SCA studies intact. Only an
exact untouched earlier default can automatically move to v3 after a saved backup
succeeds; retain its investment amount. Edited text, settings, cash, cleared inputs,
crisis assumptions and deliberately restored revisions preserve the user's work.
An explicit defaults/apply action saves the preceding draft before replacing it.
Persist range mode, calibration ID, assumed tail widening and crisis inputs.
CSV output identifies historical, assumed-tail, percentage, missing/edited cash
and separate crisis rows without implying a calibrated whole-DCF distribution.

## Verification required

Verify exact factor/support copying and original input hashes, source/method
eligibility, independent signed dispersion arithmetic, model-specific ranges,
zero/negative/missing cash, Year 4-to-5 transition and tail editing. Test financial
manual-entry behavior and unsupported source/currency/version fallbacks. Reproduce
untouched and edited legacy studies, backup failures, explicit restores, reviewed
studies, cleared crisis inputs and CSV provenance. Check the crisis shock/recovery
formula and its independent final-sale/discounting behavior. Exercise representative
companies, all-universe eligibility, offline browser/native persistence and
existing valuation flows before recording release completion.

Sources and arithmetic: [standard model](../docs/standard-company-valuation.md),
[initial audit](../docs/cash-flow-backtest-2026-09-12.md),
[grouping experiment](../docs/cash-flow-segmentation-2026-09-12.md), and
[approved setup proposal](../docs/cash-flow-universe-setup-proposal-2026-09-12.md).
