# Cash-flow midline challengers

Date: 13 September 2026. Offline research only; current starter defaults, source
data, uncertainty calibration, saved studies and the installed application were
not changed by this experiment.

## Result and next decision

**Keep latest annual cash as the default midline.** Neither of the two fixed
challengers improves the recent Year 1 mean and median error. A 50/50 blend of
latest cash and the five-year cash median does improve recent Years 2–3, and its
Year 2 result merits a bounded future trial. It does not yet justify changing the
whole forecast path or its calibrated uncertainty factors. The geometrically
damped trend performs worse than latest cash across every primary validation and
recent horizon tested.

These are retrospective comparisons on previously inspected later-vintage data.
Recording two new formulas before this run prevents a coefficient search inside
this experiment; it does not turn FY2024–2025 into untouched future evidence.
Provider cash-definition changes and the difference between provider FCF and
cash available to shareholders remain unresolved research boundaries.

## Fixed candidates

Each candidate receives the same five consecutive annual observations, newest
first, and the same target fiscal period. Signed cash, including losses and COVID
or rebound observations, is retained without clipping, winsorization or removal.

| Model | Forecast at horizon h |
|---|---|
| Latest cash, current benchmark | Most recent annual cash |
| Robust blend | 50% latest annual cash + 50% median of the five annual cash values; held flat |
| Damped trend | Latest annual cash + weighted historical slope × Σ(0.5ʲ), j = 1…h |

The slope uses the existing 30/25/20/15/10 weighted regression over fiscal offsets
0/−1/−2/−3/−4. The damped path starts from the observed latest value and adds
0.5, 0.75, 0.875, 0.9375 and 0.96875 times that slope at horizons one to five.
This differs from the earlier backtest's fitted intercept plus half-slope times
horizon: the new increments fade geometrically and the fitted intercept does
not reset the starting level.

For example, cash history 5/4/3/2/1 gives slope 1. At Year 3, the forecasts are
5 for latest, 4 for the robust blend and 5.875 for the damped trend. Its common
historical scale is 3. The median blend can reduce the effect of a single unusual
latest observation, but it can also pull a real structural improvement or decline
toward older conditions. That tradeoff must be measured rather than assumed useful.

The retained protocol was recorded at `2026-09-13T10:08:24.870915+00:00`, before
this experiment executed. Its SHA-256 is
`2fc4b1c0ac69bd1166c12910d0a9e67996f1b361dc6b7d6f3e8d76d7c71e7824`.
No candidate coefficient was adjusted after seeing these results.

## Paired cohorts and scoring

The generator reuses the original [backtest](cash-flow-backtest-2026-09-12.md)
ledger's window=5/model=naive rows. The original period, currency, intermediate
year, source and missing-data checks are inherited. Each listing/origin/horizon
key appears once in the new paired ledger, with all three candidate forecasts.

Across the original 4,139,652 rows, there are **671,700** five-year latest-cash
folds from **15,496 listings**. The new run independently reconciled latest cash,
the mean absolute training scale and the weighted slope for every one of those
folds. **658,433** folds from **15,221 listings** pass the additional publication
ordering and positive-scale conditions used for scoring. The primary operating
and property scope contains **13,423 listings** across all included periods.

The remaining rows stay in `forecast-ledger.parquet`: 3,024 have zero historical
scale, 7,629 fail training publication ordering, 2,608 lack a usable origin
publication and six have an origin publication not strictly before the target
period end. Forty otherwise eligible folds lack a target publication date; this
is disclosed rather than replacing an identified later outcome with zero.
Of eligible folds, 630,628 use one source identity through training and outcome.
One source identity does not establish a stable accounting definition.

All candidates use the identical denominator on each row:

```text
scale = mean(abs(CF1), ..., abs(CF5))
normalized absolute error = abs(forecast - actual) / scale
```

A zero scale stays unscored. A small positive scale is not silently raised, so
large normalized outliers can materially affect means. The report therefore
retains weighted medians, 90th/99th percentiles, signed bias and paired
better/equal/worse shares alongside mean error.

Each listing has total weight one within its scope/horizon/period pool. A separate
sensitivity further divides by the multiplicity of an identical cash-history
fingerprint and target year. This fingerprint binds native currency, training
fiscal end dates and five cash values; it is cash-only and differs from the
earlier segmentation experiment's all-feature fingerprint. It is not verified
issuer deduplication.

The fixed chronological groups are target FY≤2020, FY2021–2023 validation and
FY2024–2025 recent comparison. Separate rows retain every FY2020–2025 outcome
year. Point-error comparisons do not require the old uncertainty-calibration
application cutoff, so the validation group includes FY2021 here. In particular,
higher-horizon point forecasts can be scored without claiming that an admissible
historical range factor exists. The original range-application eligible count is
reported separately in every comparison cell.

The primary scope follows the previous segmentation definition: exclude Financials
except Real Estate and REITs. The financial-inclusive sensitivity remains a
research comparison of provider proxies, not permission to automatically value
banks or insurers from those proxies. There are no missing sector labels in this
input. Listing classifications are not verified historical classifications.

## Recent Year 1 result

The primary FY2024–2025 Year 1 comparison contains **24,440 forecasts across
13,069 listings**. All three models have exactly that cohort and weighting.

| Model | Mean normalized error | Median normalized error | 90th-percentile normalized error | Mean error change versus latest |
|---|---:|---:|---:|---:|
| Latest cash | 0.9597 | 0.4327 | 1.9792 | Reference |
| Robust blend | 0.9623 | 0.4522 | 1.9158 | 0.28% worse |
| Damped trend | 0.9979 | 0.4683 | 2.0882 | 3.99% worse |

The robust blend reduces the 90th-percentile error while worsening the mean and
median. Its paired weighted shares are 38.63% better, 43.89% worse and 17.48% equal
to latest cash. The damped trend is better on 40.27% and worse on 59.73%. These
are descriptive shares of dependent observations, not probabilities that a
candidate will beat the benchmark for a given company.

Duplicate-history downweighting retains the Year 1 direction: robust blend mean
error is **0.45% worse**, and damped trend is **3.90% worse**. Including financial
proxies also leaves both worse, by 0.14% and 2.15% respectively.

## Horizon and regime results

Positive numbers below mean reduced mean normalized absolute error relative to
latest cash, on the same operating/property rows. They are not improvements in
investment returns or annual uncertainty coverage.

| Horizon | Robust blend validation | Robust blend recent | Damped trend validation | Damped trend recent |
|---|---:|---:|---:|---:|
| 1 | +0.69% | −0.28% | −2.39% | −3.99% |
| 2 | +0.70% | +4.45% | −0.95% | −5.74% |
| 3 | +0.33% | +5.34% | −0.72% | −5.67% |
| 4 | −0.06% | +3.31% | −0.57% | −4.05% |
| 5 | −0.01% | +1.55% | −0.63% | −2.50% |

The robust blend's recent median error improves from **0.6793 to 0.6265** at
Year 2 and **0.8548 to 0.7549** at Year 3. Duplicate-history downweighting leaves
their mean improvements at 4.43% and 5.29%; the financial-inclusive sensitivity
gives 4.48% and 5.43%.

| Target fiscal year | Robust blend Year 1 | Robust blend Year 2 | Robust blend Year 3 | Damped trend Year 1 |
|---|---:|---:|---:|---:|
| 2020 | +1.74% | +2.78% | +1.30% | −3.60% |
| 2021 | +0.44% | +0.67% | −0.01% | −2.14% |
| 2022 | −0.42% | +0.28% | +0.04% | −1.27% |
| 2023 | +4.17% | +4.98% | +3.11% | −5.37% |
| 2024 | +0.53% | +6.97% | +5.72% | −4.97% |
| 2025 | −1.46% | +2.38% | +4.99% | −3.17% |

Year 2 blend improvements remain positive across these observed regimes, making
that the most useful candidate for a future test. The modest older-period gains,
weaker Year 1 result and unresolved source measurement prevent treating the blend
as a generally superior replacement. A future horizon-dependent combination
would itself be a new specification: freeze it before new outcomes, calibrate its
own error ranges separately, and preserve the current baseline for comparison.

## Boundaries and retained artifacts

This slice starts from already realized, eligible original folds. The original
company/exclusion ledgers remain the record of companies without scores; this
experiment does not restore absent issuers or missing outcomes. Publication
checks cannot undo later revisions, and a target year may already be partly
elapsed at the origin publication. Related share classes, overlapping histories
and repeated fiscal targets are dependent. No significance or company-specific
accuracy claim follows from the fold count.

The experiment measures annual provider-cash point errors only. It neither
refits the current uncertainty bundle nor establishes an 80% cash-path/NPV
interval, sustainable terminal cash, required returns, share-price accuracy or
ten-year forecast skill. See the [source-vintage audit](cash-flow-backtest-data-vintages-2026-09-12.md)
and [standard model](standard-company-valuation.md) for the remaining boundaries.

Generated outputs live under
`desktop/test-results/cash-flow-midline-challengers-2026-09-13/`:

- `protocol.json` and its timestamp/hash lock.
- `forecast-ledger.parquet`, retaining all paired source folds, candidates and eligibility.
- `results.json`, `comparisons.csv` and `report.md`, covering 540 comparison cells.
- `receipt.json` with the executed generator and input/output hashes.
- `independent-check.executed.py` and `independent-check.json`.

The input ledger hash is
`34aa56eaafc1d6e195154e86bbcd4b6bbf151052244cf7b1ca895105203bfd6b`, checked against
its original receipt before and after execution. The inherited source-pack hash is
`1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a`.
Results JSON SHA-256:
`2cce710dc82b0e9e87461a18b241e653e08dffa66eaf48b62e3b07a8156085d6`.
Paired ledger SHA-256:
`335fd9207ffb31b2a6544b3b72a39112e76be4c38a582cfcecab1f2b870a3cf7`.
These generated artifacts are gitignored and require separate retention.

## Reproduction and verification

From the desktop worktree, use the canonical virtual environment and an unused
output directory. The generator refuses to overwrite prior results or write
inside the original research directory.

```bash
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
python -m pytest desktop/tests/test_cash_flow_midline_challengers.py -q
python desktop/scripts/cash-flow-midline-challengers.py \
  desktop/test-results/cash-flow-backtest-2026-09-12 \
  --output-directory /tmp/atlas-cash-midline-reproduction --protocol-only
python desktop/scripts/cash-flow-midline-challengers.py \
  desktop/test-results/cash-flow-backtest-2026-09-12 \
  --output-directory /tmp/atlas-cash-midline-reproduction
```

All **16 focused tests** pass, covering hand-calculated forecasts, geometric
damping, signed/zero/tiny cash, target poisoning, timing, common scales, paired
weights and checksum/output guards. Ruff and scoped whitespace checks pass.
An independent checker that does not import the generator verified **40,000
scalar amounts from 10,000 seeded rows**, using normal equations and explicit
geometric sums. It also reconciled all **30 primary validation/recent summary
cells**, including counts, means and medians. No source, calibration, application
or default was modified by those checks.
