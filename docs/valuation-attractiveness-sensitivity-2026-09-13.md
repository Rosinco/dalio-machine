# Valuation attractiveness: testing terminal settings

Date: 2026-09-13. Frozen downloaded financial data and historical saved prices.

## Recommended use

Keep full DCF as the baseline comparison and display a 50% terminal stress and
cash-only comparison beside it. Fifty percent is a deliberately substantial
stress, not an estimated probability or an optimized valuation weight. Use the
existing five-year positive cash/EBIT lens to narrow research, then examine which
companies retain a surplus or the chosen 30% margin under the stress. Passing
does not establish sustainable owner cash, business quality or an investment case.

In **Companies → Lists → Column preset**, choose **Valuation attractiveness**
for the main comparison or **Terminal sensitivities** for 100%, 50% and 0%
side by side. Each column can independently select 100/75/50/25/0% in the KPI
picker. Filters and saved views retain their selected credit even if a displayed
column changes. Existing filters are retained when applying a column preset.

The transparent measure is:

`100 × ((annual cash PV + terminal credit / 100 × terminal PV) / saved equity price − 1)`.

At full credit this is NPV / price. DCF / price has identical ordering; combining
the two into a score adds no independent information. Terminal value enters once.
These are surplus percentages, not annual investment returns. A price 30% below
value corresponds to 42.857142…% surplus relative to price.

## What the experiment found

All 19,140 listings were evaluated at five credits, with an independent arithmetic
reference for 95,700 row/setting combinations. There are 13,002 eligible comparisons
and 6,138 withheld for missing or unsuitable evidence. The existing cash-consistency
lens contains 3,465 listings; 3,290 have eligible dated valuations. That lens requires
operating classification and five comparable positive provider FCF and EBIT periods.

| Terminal credit | Positive surplus, all eligible | 30% discount, all eligible | 30% discount within cash-consistency lens |
|---:|---:|---:|---:|
| 100% | 1,844 | 1,036 | 506 |
| 75% | 1,603 | 910 | 423 |
| 50% | 1,376 | 773 | 337 |
| 25% | 1,132 | 640 | 260 |
| 0% | 937 | 500 | 183 |

Half terminal credit removes 169 of 506 cash-consistency matches at the 30%
threshold, about one third. Ranking remains similar: 96 of that lens's top 100
listings overlap between full and half credit. Similar order does not imply a
similar margin of safety. Raw top ranks can also contain cash-definition, price,
financing and share-basis problems that a terminal haircut cannot repair.

Historical examples illustrate the distinction; these are not reviewed prospects:

| Listing | Saved price date | Terminal share | Full-credit surplus | Half-credit surplus | Cash-only surplus |
|---|---|---:|---:|---:|---:|
| Byggmax | 2026-01-30 | 35.06% | +108.52% | +71.97% | +35.42% |
| Bilia | 2026-02-05 | 18.16% | +28.66% | +16.98% | +5.30% |
| G5 Entertainment | 2026-02-17 | 64.01% | +56.14% | +6.17% | −43.80% |
| Orthex | 2026-03-05 | 32.86% | +7.04% | −10.54% | −28.13% |

Byggmax retains the 30% modeled margin at half credit; G5 loses it. Bilia remains
positive without terminal value but does not reach that margin in any setting.
These figures use unchanged historical quotes and unreviewed generic cash proxies,
not current prices or the separately reviewed company Value studies.

## Why no universal terminal weight is justified

Terminal value represents cash flows beyond the explicit forecast. Stable-growth
assumptions must be consistent with reinvestment and the appropriate cash-flow and
discount-rate definitions. Reducing terminal value alone is a sensitivity, not a
substitute for reviewing those fundamentals. See [Damodaran's terminal-value
framework](https://pages.stern.nyu.edu/adamodar/New_Home_Page/littlebook/terminalvalue.htm).

For a transparent arithmetic example, annual cash of 100, ten forecast years,
10% required return and zero terminal growth give annual cash PV of 614.46 and
terminal PV of 385.54, totaling 1,000. Terminal share is 38.55%. Crediting half
the terminal component cuts total value by 19.28%, not 50%. Extending the explicit
forecast moves value out of the terminal component without improving the business.
Zero credit therefore favors earlier cash and should not be a universal rejection
rule for durable or reinvesting companies.

The frozen starter holds latest signed annual cash flat and derives terminal
cash from historical median cash. All 9,061 eligible rows with matching annual
and valuation currencies reproduce these two formulas. Exactly 1,078 eligible
rows have the example's 38.55% terminal share. Within the cash-consistency lens,
sector medians cluster near 36.9–38.6%; sector alone supplies no defensible terminal
credit. Business economics, reinvestment, cash volatility, leverage, capital needs
and cash-definition reconciliation are more useful deep-dive questions.

Whole-universe rank stability partly reflects the common model and 6,282 eligible
rows with zero terminal PV. Spearman correlation for 0% versus 100% is 0.9855 over
all eligible listings but 0.8438 among full-credit-positive listings. Changing the
weight cannot remove every misleading result from the underlying cash proxy.

## Evidence boundaries and next research

This is a sensitivity experiment on one frozen snapshot, **not a point-in-time
backtest or a calibration of terminal credit**. The eligible saved prices have a
median age of 216 days; 1,085 are older than a year at the 2026-09-13 snapshot.
Listings include multiple share classes. There are 900 eligible limited-history
rows and 6,968 signed-cash cases. A price or cash input can be arithmetically valid
without being suitable for an investment decision.

The next useful model test is to examine whether normalized sustainable cash and
reinvestment assumptions predict later observed cash better across business types.
Use rolling historical origins with known publication dates and a held-out period;
retain COVID years and report normal/crisis/recovery results separately. An eventual
investment-return backtest also needs historically available prices, corporate
actions, delisted companies and realistic decision dates. Do not choose terminal
credit by maximizing matches or performance on the same data used to select it.
The existing annual cash-range backtest does not validate terminal value or assign
a probability to the whole DCF.

Reproducible local evidence is under the desktop worktree's ignored directory
`desktop/test-results/valuation-attractiveness-sensitivity-2026-09-13/`:
`audit.mjs`, `summary.json`, `report.md` and `listing-ledger.jsonl`.
Run `node test-results/valuation-attractiveness-sensitivity-2026-09-13/audit.mjs`
from `desktop/`. Compressed and decompressed gauge identities were verified.
Compressed gauge SHA-256:
`fcbcccadbdd86da4a6521a64c2be8907ceff91f23dc1aa2e7fcfd2f991ba0e2b`.
No source packs, provider prices, authored valuations or user preferences changed.
