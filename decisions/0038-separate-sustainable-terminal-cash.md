# ADR 0038: Separate terminal cash from annual forecast uncertainty

Date: 2026-09-13. Status: adopted for Macro Atlas 0.15.0. Verification and
installation are recorded separately in the desktop handoff.

## Problem

The generic starter capitalized each scenario's final forecast payment at 10%.
Its assumed Year 5–10 widening therefore became a permanent cash assumption.
Historical annual error coverage cannot establish the probability of that sale
or of cumulative DCF/NPV. Provider FCF also remains an unreconciled proxy.

## Decision

Keep the existing latest-cash benchmark, annual cash paths, source guards and
immutable empirical calibration. Add an independently editable terminal model:

```text
terminal sale at N = max(0, sustainable equity cash in N+1) / ((r - g) / 100)
terminal present value = terminal sale / (1 + r/100)^N
```

The numerator is already the first post-horizon annual cash after reinvestment
and financing. Do not multiply it by `(1+g)` again. Require positive r and
`-100 < g < r`. Missing inputs or invalid arithmetic remain unavailable.
Nonpositive cash produces zero assumed sale, with an explicit turnaround/run-off
review prompt; it is not a liquidation value. Explicit sale proceeds remain an
alternative, including zero for no sale. Editing sale proceeds selects that mode.

New generic starters retain cash method `empirical-cash-starter-v3` and add
`terminalMethod: historical-median-v1`. At least three consecutive eligible annual
observations seed the signed median, with low/high subtracting/adding 20% of its
absolute amount. All observations including shock/rebound years remain included.
The 20% spread and zero mature growth are unreviewed assumptions, not calibrated
terminal ranges. A median reduces sensitivity to one extreme observation but does
not establish sustainable cash, maintenance investment or growth economics.
Short histories and manual financial models require explicit terminal inputs.
The forecast endpoint and any transition to sustainable cash remain inspectable.

Each scenario's optional `terminalCash` is authoritative when present. Resolve
sale from its inputs during valuation and CSV export; a stale `terminalEquity`
display cache must never affect the calculation. Changing annual cash or range
widths does not change terminal cash. Applying history settings retains independent
terminal inputs and required returns for an existing generic terminal-model study.
Opening the standard baseline deliberately resets assumptions after saving a revision.

Show a working terminal assumption table and required-return NPV sensitivities.
Keep nominal cash provenance separate from terminal/rate/notes edits. A separate
crisis scenario always uses its own explicit sale and cannot inherit terminalCash.

## Cash measurement boundary

Show same-report operating, investing, financing, net-period cash and provider FCF
with currency/source lineage, arithmetic differences and missing components.
Operating plus investing cash is not inferred owner cash or a capex measure.
Missing inputs stay missing. Possible all-zero placeholders are identified; zero
arithmetic equality does not corroborate those reports. Neither component equality
nor provider wording resolves the material cross-vintage FCF revisions.

## Persistence and validation

Exact untouched v1/v2 and pre-terminal v3 defaults may upgrade only after both a
saved revision and replacement draft persist successfully. Preserve edited,
cleared, customized, crisis, explicitly restored and reviewed studies. Missing
terminalMethod on v3 reproduces the original definition exactly. Holmen and SCA
reviewed studies retain their own assumptions.

Validate signed/missing cash, N+1 timing, r/g boundaries, dynamic calculations,
CSV, crisis isolation, old-draft replay, rollback and browser/native persistence.
Audit every listing against the retained pre-change runtime; freeze the cohort,
source identities and new generic forecasts without changing source files.
This establishes implementation correctness, not improved forecast accuracy.

## Research boundary

Fixed robust-blend and damped-trend challengers are evaluated separately. Keep
latest cash unless results justify changing it. Whole-path probabilistic DCF/NPV
requires validated cross-year dependence, consistent cash definitions and frozen
outcomes. Do not relabel the current three scenarios as joint confidence bounds.

See [standard model](../docs/standard-company-valuation.md),
[cash-component audit](../docs/cash-component-audit-2026-09-13.md), and
[midline experiment](../docs/cash-flow-midline-challengers-2026-09-13.md).
The distinction between sustainable equity cash and its discount rate follows
[NYU's terminal-value framework](https://pages.stern.nyu.edu/~adamodar/New_Home_Page/valquestions/termvalapproaches.htm).
The historical median and 20% seed are Atlas assumptions, not recommendations
from that source.
