# Macro Atlas 0.19.0 handoff — 2026-09-13

## Source closeout — 2026-09-13

The user requested commit, push and handoff. The complete company valuation,
research-screen and expanded-list work through **Macro Atlas 0.19.0** is committed:

- Desktop implementation: `814e2f7231782e8415d79a3d37f6617889e3e637` on `feat/offline-atlas`.
- Canonical methodology: `611c737d5d207e7c09e2f94c49577d7437884224` on `main`.
- Both content commits were pushed to `Rosinco/dalio-machine`; exact remote refs
  were verified. The branches remain separate. Subsequent handoff-only commits
  record this checkpoint; the final ref receipt is
  `desktop/test-results/closeout-2026-09-13/git-closeout.json` in the desktop worktree.

Closeout verification passed: **1,206 desktop repository Python tests** and
**1,690 canonical Python tests**, with `ruff check src tests` clean in both
worktrees; **280 app/transport unit tests**, **138 desktop Python tests and
19 subtests**, and the Node cash-flow backtest also passed. No app code changed
after the verified 0.19.0 executable was built. Its release evidence covers
35 production-browser list/research checks and 11 expanded-list Windows checks,
including restart and actual CSV output. The full older valuation browser/native
suites were not rerun on the final 0.19.0 binary.

Generated source packs, provider shards, executable and test receipts remain
local and gitignored. This source push does not back them up. The desktop README
records the retained-data prerequisites and gauge/KPI regeneration commands;
checked manifests preserve input identities. The original 0.19.0 release receipt
and handoff are retained under `test-results/releases/0.19.0/pre-closeout/`.
Installed documentation is refreshed without rebuilding or restarting the app;
installation and release verification receipts are refreshed after the copy.

Next practical task: use **Companies → Lists** to form a small named research
watchlist, inspect source dates and missing values, then select one company for a
project-native Börsdata deep dive. Reconcile owner cash, capital needs and dated
valuation inputs before revising its assumptions. List presets and cash bands
remain research aids; they establish neither an investment verdict nor calibrated
whole-path DCF/NPV probabilities.

## Current slice — expanded saved KPIs and list tools

The user explicitly requested equal expansion of the KPI catalogue and list
features, using the KPI data already collected. The new provider layer imports
210 KPI families and 3,379 exact period/calculation/source variants from the
2026-08-10 download. Of those, 177 families and 3,312 variants have usable values;
33 families remain inspectable as unavailable. The 41 existing fields remain,
for 251 catalogue entries. There are 36,374,384 populated provider cells across
19,140 Atlas listings. 17,593 Atlas IDs join the current instrument master and
1,547 have no current master identity; a joined ID does not imply complete KPIs.

The compressed provider index/shards total 241,270,353 bytes. The catalogue is
bundled metadata; only selected columns and active conditions load their bounded,
hash-verified shards. Financial-pack, taxonomy, listing identities, dimensions,
compressed/raw hashes and coverage reconcile before display. Missing values,
pending loads and failed verification remain distinct. An unavailable provider
layer does not disable the existing research-screen columns.

Corrections include FCF/share, net debt/assets, dividend yield, debt/equity units,
earnings/FCF percentage, quote-currency market capitalization/enterprise value,
report-currency EPS/dividends/book/revenue per share, period-specific price-return
units, million-share counts, technical price levels versus percentage positions,
and quote-currency turnover. Twenty-nine insider, buyback and short-value variants
with unknown monetary currency or scale are withheld. Formatted numeric text companions must agree
within printed rounding precision. Conflicting source scopes, uncertain monetary
currency combinations, NCAV definitions 307–310, and the documented Atrium NAV
split mismatch are withheld. Provider snapshot dates are not invented underlying
report or quote dates. R12 history is not labelled standalone quarterly cash.

Companies → Lists now offers up to 32 columns, 12 numeric conditions with eight
operators, 20 named watchlists, 20 saved views, three sorting priorities, compact
rows, up to 250 rows per page, and comparisons of up to eight listings. Five
column presets cover valuation, dividends, profitability/returns, financial
strength and price/insider activity. The picker defaults to fields with usable
saved values; an optional toggle reveals unavailable catalogue entries. CSV
exports all filtered rows with selection/source identities and five fields per
KPI (value, unit, currency, date, status). Windows export creates a new Downloads
file and shows its actual path. Values expose fuller source details on click.

Preferences migrate from v1 in memory; only explicit list actions save the atomic
v2 key `macro-atlas-company-lists-v2`. The retained 0.18 application keeps its
unchanged v1 key. Named watchlists, comparison selections, density and saved-view
sort/watchlist settings persist together. Deleted default-list membership is not
resurrected from a stale compatibility mirror. Unknown listing IDs are retained.

New provider prices do not reprice the frozen starter DCF, NPV, terminal, purchase
ceiling or the existing dated candidate presets. Browsing and exporting lists
never mounts Value, saves/migrates drafts or changes reviewed assumptions. No
formal Börsdata strategy, gate, backtest or investment recommendation was created.

## Verification and installation

Validation passes: 280 TypeScript/transport unit tests across 32 files and ten
Python exporter tests. The final production browser passes 10 expanded-list checks,
13 original company-list checks and 12 research-screen checks. The final Windows
executable passes 11 expanded-list checks, including a full process restart and
actual CSV output containing all 955 filtered Swedish listings. No external
requests or runtime errors occurred. Isolated tests preserve authored valuation
bytes and never seed normal-profile user preferences.

The final frontend and Windows builds pass. Windows packaging took 6m53s and
retains the existing missing Microsoft CRT PDB linker warning. Executable SHA-256:
`a3b35eb29e86083bdd08f6ecf1b138436f8130eca19d5b7334a6845869cde157`.
Provider manifest SHA-256:
`d1e1b005547a9be95173250b4a1bcf747c0149a69a7504dc2539b5f825f1c569`.
Independent verification of all 212 final shards passes. Complete source
regeneration reproduces every shard and the final manifest byte for byte; source
hashes match before and after. The regeneration receipt matches the manifest above.

**0.19.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.19.0\Macro Atlas.exe`.
The existing OneDrive Desktop Macro Atlas shortcut targets this executable and
working folder. Normal-profile process 5992 was running at verification; 0.18.0
is retained. Executable hashes/version metadata, financial packs, shortcut and
copied documentation match the verified release. Desktop/390px picker, table and
comparison screenshots were visually reviewed. Narrow screens use internal table
scrolling. Concurrent browser timings were not a clean performance benchmark.
An initial old-list browser click timed out under heavy parallel load; the full
final rerun passed without an application change.

Receipts under `test-results/`: `unit-tests-0.19.0.log`,
`frontend-build-0.19.0.log`, `windows-build-0.19.0.log`,
`expanded-company-list-browser-report.json`, `company-list-browser-report.json`,
`research-gauge-browser-report.json`, `windows-native-expanded-company-list-report.json`,
`windows-installation-0.19.0.json`, and
`expanded-company-list-release-verification-0.19.0.json`.
Source/package reproduction receipts are in `expanded-kpi-audit-2026-09-13/`.
The task's local preview was stopped; the installed Windows app remains running.

Source exporter: `scripts/export-expanded-kpis.py`; reproduce without rewriting
with `/home/rosinco/workspace/dalio-machine/.venv/bin/python scripts/export-expanded-kpis.py --check`.
Raw source inventory/reconciliation: `test-results/expanded-kpi-audit-2026-09-13/`.
Focused browser: `node tests/browser.mjs --expanded-company-list-only`.
Focused Windows: `scripts/test-windows.ps1 -ExpandedCompanyListOnly`.
Existing regression: `node tests/browser.mjs --company-list-only` and
`node tests/browser.mjs --research-gauge-only`.
ADR 0042 is mirrored into the canonical methodology worktree.

The complete source closeout is recorded above. The old 0.18 handoff,
executable and receipts were backed up under `test-results/releases/0.18.0/`
before this work. Release narratives below describe their checkpoints at the
time; the 0.19.0 release and source closeout above govern the current state.

## Earlier company-list release — 0.18.0

# Macro Atlas 0.18.0 handoff — 2026-09-13

## Current slice — customizable company lists

The user requested a list page where potential research candidates appear with
actual values in individually chosen KPI columns, using Börsdata's table and KPI
selector as visual examples. **Companies → Lists** now provides this workflow;
**Sectors & branches → Lists** starts with the selected branch. The existing
Screen, profile and Value pages remain available.

The list covers all **19,140 saved listings**. **Choose KPIs** offers **41 supported
metrics and data fields**, grouped by category with definitions, units, source
dates and supported calculations. Up to 16 columns can be added, edited, removed
and reordered. Annual-series columns support latest, three-report or five-report
windows with complete-history average/median/min/max or positive-path CAGR as
appropriate. A three-report CAGR generally spans two elapsed years; the formula
uses actual ending dates. No missing observation is dropped to manufacture an
average or substituted from an older period. Zero and negative cash remain signed.

Numeric sorting keeps missing last in both directions and groups monetary
amounts by currency before sorting within the currency. Up to six explicit
numeric conditions support >= and <=; monetary conditions require a selected
currency. Blank/invalid amounts and missing currencies retain an incomplete
condition that matches no listings, including after reload. Editing or removing
a displayed column does not relabel the saved definition of an active condition.

Users can star listings into a personal **Watchlist**, and save up to 20 named
views containing columns, filters and sorting. Preferences use the separate key
`macro-atlas-company-lists-v1`. Stars are shared across views, and unknown IDs
survive a temporary change in data coverage. Reading malformed/future-version
preferences does not overwrite their saved bytes. Local save errors are visible.
Browsing lists never mounts Value or saves/migrates valuation drafts/revisions.

Two explicitly described historical presets identify **3,465** operating listings
with five-period evidence and five positive FCF/EBIT observations, and **506**
which additionally have a positive saved price at or below 70% of positive Mid
starter value. These are descriptive research leads, not reviewed businesses or
buy recommendations. The latter's saved price dates range **2025-05-09 through
2026-08-06**. A candidate's price date remains visible even when its quote column
is removed. Preset membership and personal watchlist membership are separate.

The feature reuses the byte-identical 0.17 research artifact and exact financial
and taxonomy bindings. It neither imports additional raw KPIs/prices nor changes
any cash, terminal, uncertainty, purchase or reviewed-valuation assumptions.
EPS, P/E, dividends, price performance, R12 and longer histories require a separate
verified import. Use Value to edit the working scenarios and purchase margin;
list valuation columns retain the frozen standard starter and fixed 30% margin.

## Verification and installation

250 unit tests across 29 files pass. The final frontend and Windows builds pass.
The production browser passes 13 company-list checks and 12 existing research-
screen checks, with no external requests or runtime errors. The final executable
also passes 14 Windows company-list checks, including a full process restart,
with no external requests or runtime errors. All tests used isolated profiles;
normal-profile user stars or saved valuations were not seeded by test fixtures.

**0.18.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.18.0\Macro Atlas.exe`.
The existing OneDrive Desktop Macro Atlas shortcut targets that executable and
working folder; normal-profile process 28800 was running at verification. Version
0.17.0 is retained. Installed executable/version metadata, both financial packs
and copied documentation match the tested release. Final executable SHA-256:
`5dad9708a895f673d4210633ae508b91deb48d83a1aa7cfeb9dfb2b02721f3e0`.

Receipts: `unit-tests-0.18.0.log`, `frontend-build-0.18.0.log`,
`windows-build-0.18.0.log`, `company-list-browser-report.json`,
`research-gauge-browser-report.json`, `windows-native-company-list-report.json`,
`windows-installation-0.18.0.json`, and
`company-list-release-verification-0.18.0.json`, all under `test-results/`.
Desktop and 390px table/picker screenshots were visually reviewed. The build
retains the existing missing Microsoft CRT PDB linker warning; it links and runs.
The branch test caught and corrected an incompatible saved sector on explicit
branch entry; incomplete numeric conditions and missing currencies remain
restrictive after restart. No cash/terminal/valuation arithmetic was changed.

Focused commands are `node tests/browser.mjs --company-list-only` and Windows
`scripts/test-windows.ps1 -CompanyListOnly`. Shared flows check independent exact
preset membership, real rendered values, signed/null sorting, period calculations,
retained/blank filter definitions, picker focus, saved views/watchlists, corruption,
source mismatch, desktop/narrow layouts and authored valuation preservation. The
native harness additionally closes and restarts the process to check persistence.

The 0.17 executable and shared evidence were retained under
`test-results/releases/0.17.0/` before building. Methodology is ADR 0041, mirrored
to canonical `dalio-machine` main; implementation stays on desktop
`feat/offline-atlas`. Prior uncommitted work is preserved. No commit or push is
part of this slice.

---

# Macro Atlas 0.17.0 handoff — 2026-09-13

## Current slice — explainable universe research screen

The user approved a research-priority gauge using downloaded data to choose what
to investigate in a native Börsdata deep dive. **Companies → Screen** now opens
an alphabetical, paginated table for all **19,140** saved listings. Company
profiles contain a read-only Research gauge card; **Sectors & branches → Screen**
starts with the selected branch. Explicit sector, branch, country, presence,
business-route, coverage and historical-cash filters retain missing/manual cases.

The table starts with historical evidence and research questions. Inspect up to
five comparable annual periods, each cash/profit/revenue observation, positive
and valid counts, dispersion, latest financing/asset proxies, dated comparable
quarterly changes and missing reasons. Five-period evidence is a coverage state,
not a business rating. The 550-day annual-age rule is an explicit display
heuristic; COVID and recovery observations remain in the selected window.

Optional valuation context is a **frozen standard starter**, separate from
reviewed Holmen/SCA studies and all authored drafts. It shows dated derived equity
price/share/FX inputs, cash and terminal PV, Low NPV and required cash factors
P/V and P/(0.70*V). The fixed 30% screen margin is editable in the working Value
workspace. Scaling includes both annual and terminal cash, including any funding
needs; the factor is not forecast growth. Terminal is counted once. Provider FCF,
asset ratios and scenarios establish no verified owner cash or investment verdict.
Financial businesses keep their specialist capital/distributions route.

This slice uses the existing verified annual/quarterly financial companion and
its publication-window price comparisons. Newer raw prices, R12 and KPI histories
are already downloaded but are **not imported by this slice**; their ownership,
currency and date reconciliation is the next independent data task.

## Evidence and source boundary

The derived asset is `desktop/public/data/research-gauge/gauge.bin` (gzip payload
with an opaque extension to prevent HTTP auto-decompression before validation).
The application pins and verifies compressed/uncompressed hashes and exact byte
counts before exposing rows. It binds the active financial pack, directory,
company source hashes, model and calibration. Mismatches withhold both the table
and profile card. Runtime does not fetch universe histories or mount Value while
browsing; no draft, revision or purchase-setting writes occur.

- Financial pack: `1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a`.
- Gauge SHA-256: `fcbcccadbdd86da4a6521a64c2be8907ceff91f23dc1aa2e7fcfd2f991ba0e2b`.
- Gzip bytes: **11,272,132**; decompressed bytes: **77,849,515**.
- Historical coverage: **14,741** five-period; **3,584** limited/older;
  **616** reconciliation; **199** no annual history.
- Comparable quarterly revenue changes: **10,438** listings.
- Valuation readiness retains the previous counts: **6,941** positive/priced,
  **489** positive/unpriced, **6,749** nonpositive, **2,731** manual financial,
  **2,230** other missing. These are implementation/input counts, not candidates
  approved for investment or formal research gates.

`node scripts/export-research-gauge.mjs --check` reproduces the asset exactly.
The exporter independently reconciles Mid and Low calculations for all listings,
**13,882** reverse equations, and all **19,140** Mid valuations against the frozen
0.16 ledger. Input histories remain unchanged. The export receipt is
`test-results/research-gauge-export-2026-09-13/receipt.json`.

## Verification and installation

236 unit tests across 28 files and 12 dedicated production-browser checks pass.
The browser checks cover exact universe identity/arithmetic, filters/pagination,
source dates, signed/missing/manual cases, sector/branch navigation, same-company
return, desktop/390px layouts, rapid company selection, corruption and pack
mismatch recovery, no bulk-history queries, and exact authored-draft/revision
preservation. There were no external requests or runtime errors.

A shared locale collator and reuse of already validated data remove repeated
work when opening the table. An indicative final browser run measured 6.2 seconds
for the first company card and 0.29–0.70 seconds for subsequent screen opens;
these are local observations, not a cross-device performance guarantee.

Both production builds passed. The final executable passed **12 Windows research
checks**, including a full process restart and observed real IPC mismatch
injection, with no external requests or runtime errors. The existing valuation
suites also passed **55 browser** and **58 Windows** checks on this feature slice,
including reviewed/edited studies and restart persistence. An initial native
failure test used an immutable Tauri invoke property and injected nothing; the
corrected test verifies actual IPC response injection and observed commands.

Final executable SHA-256:
`c158ded6cf4839df4b22607e6d12d780d40cb371386741b807b314bec7ecbcff`.
Windows compilation retains the existing missing Microsoft CRT PDB warning;
the executable links successfully.

**0.17.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.17.0\Macro Atlas.exe`.
The existing `C:\Users\Adamb\OneDrive\Desktop\Macro Atlas.lnk` targets that
version and working folder. The normal-profile app is running. Executable/version
metadata, both financial packs and installed documentation match the tested
release. Version **0.16.0 is retained**. The first installation check passed at
`2026-09-13T12:43:50.4216720Z`; the installation receipt records final verification
after documentation was finalized.

Retained receipts include `unit-tests-0.17.0.log`, `frontend-build-0.17.0.log`,
`windows-build-0.17.0.log`, `research-gauge-browser-report.json`,
`windows-native-research-gauge-report.json`, `valuation-browser-report.json`,
`windows-native-valuation-report.json` and the indicative performance report
`research-gauge-browser-overview-timing.json`. The final release verifier is
`node scripts/verify-research-gauge-release.mjs --check`; its receipt binds
application sources, derived data, audit/test evidence and installation.

Generated assets, receipts and binaries are gitignored; retain them separately.
The 0.16 executable, handoff and shared receipts are retained under
`test-results/releases/0.16.0/`. No commit or push is part of this slice; prior
uncommitted valuation/research work is preserved. Methodology is ADR 0040, mirrored
to canonical `dalio-machine` main; app implementation stays on desktop
`feat/offline-atlas`.

---

## Previous release: Macro Atlas 0.16.0 — 2026-09-13

## Current slice — purchase-price range and editable margin of safety

The user requested a purchase-price view derived from DCF, NPV and terminal value,
and explicitly selected **30% below value, editable**. Every company's Value
workspace now has a purchase panel, defaulting to the Mid scenario and total
equity millions. Terminal PV is included once in DCF; NPV is value minus proposed
price. Each named scenario has its own ceiling. Positive prices at or below the
selected ceiling meet that assumed margin rule. There is no minimum buy price,
no assigned probability and no automatic investment verdict.

- Missing cash/terminal inputs stay unavailable. Nonpositive equity value and
  100% margin have no positive ceiling; signed DCF and NPV remain inspectable.
  Cash-only ceilings and terminal contribution expose terminal dependence.
- Value can be calculated without a market quote or stake investment. A valid
  dated study equity price supplies the initial candidate only when no candidate
  was authored. An explicit blank stays blank; an entered price can be compared
  independently. Saved prices are not live quotes.
- Per-share display needs an explicitly entered/applied dated share count,
  currency and ownership source. Reviewed Holmen/SCA references bind to their
  own studies. Generic reported shares remain an unreviewed proxy. An equity
  basis divided by an edited denominator is labelled an equivalent price, not
  an observed share quote. Extreme conversions fail closed before charting.
- Optional purchase settings preserve old draft shape, edits, deliberate blanks
  and revision behavior. Purchase-only edits prevent automatic replacement and
  do not relabel unchanged forecast cash. History/baseline/reviewed resets
  preserve the separately authored purchase policy. A dedicated CSV retains
  scenario values, settings, price/share/source basis and missing/error states.

The whole-universe audit passed: all **19,140** fresh starter economics and
complete DCF/NPV result objects exactly match retained 0.15.0. Independent scalar
checks reconciled **42,385** available intrinsic scenarios, their 30% ceilings,
cash-only ceilings and explicit-price NPVs; **38,885** dated-price NPVs also
reconciled. There are **7,430** positive Mid ceilings, **6,749** nonpositive Mid
values and **4,961** unavailable Mid values. **4,490** listings have three positive
scenario ceilings. These are generic implementation/input counts, not a screened
buy list. All drafts remained unchanged; no annual or terminal forecasts changed.

Verification passed: **217 unit tests**, **55 integrated browser checks**,
**10 dedicated purchase browser checks** and **58 Windows valuation checks**,
including full-process restart of the purchase policy and existing terminal/cash
studies. Browser/native runs recorded no external requests or runtime errors.
Desktop/mobile layouts were inspected, along with independent arithmetic,
partial CSV exports, source/ownership guards and exact draft restoration.
The final frontend and Windows builds pass. Windows build retains the existing
missing Microsoft CRT PDB warning; it does not prevent the linked executable.

**0.16.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.16.0\Macro Atlas.exe`.
The existing OneDrive desktop shortcut targets that version and working folder.
The running normal-profile app, version metadata, executable, both source packs
and release documents were verified at 2026-09-13T11:07:50.8527371Z.
Version 0.15.0 is retained; native tests used a separate profile. Final executable:
`ad52d076f602640645e50c809e482b605fb083433ee399b108405b1f5cb98f6a`.
New model receipt: `test-results/purchase-range-2026-09-13/universe-audit.json`.
Its frozen ledger/runtime bind source hashes and the exact 0.15.0 comparator.
Other receipts: `unit-tests-0.16.0.log`, `frontend-build-0.16.0.log`,
`windows-build-0.16.0.log`, `valuation-browser-report.json`,
`purchase-range-browser-report.json`, `windows-native-valuation-report.json`,
`windows-installation-0.16.0.json` and `purchase-release-verification-0.16.0.json`
The final release verifier binds these artifacts, current sources and installed
files; successful closeout is recorded in the purchase release receipt.
Run `node scripts/verify-purchase-release.mjs --check` to verify the final bound
release after it is installed. Retained audit inputs can be rechecked without
replacing the original ledger using the audit script's `--output` option.
Prior common 0.15.0 browser/native receipts are retained under
`test-results/releases/0.15.0/` before the shared report filenames are replaced.

Methodology: ADR 0039 and `docs/standard-company-valuation.md`, mirrored to the
canonical dalio-machine worktree. App code stays on desktop `feat/offline-atlas`;
canonical methodology stays on `main`. Existing uncommitted slices are preserved.
This slice does not commit or push. Next research remains cash-definition review,
company deep dives and future outcome evaluation; the purchase rule does not
improve the predictive accuracy of its input forecasts.

---

## Previous release: Macro Atlas 0.15.0 — 2026-09-13

### Terminal cash, source definitions and forecast challengers

The user approved improving the generic projections. Version 0.15.0 separates
first post-horizon sustainable equity cash from annual uncertainty endpoints.
The existing annual latest-cash benchmark, source histories and error calibration
remain unchanged. New starters add `terminalMethod: historical-median-v1` to
cash method v3. Signed historical median (at least three consecutive eligible
periods), an assumed ±20% terminal sensitivity and zero mature growth are editable
unreviewed starting points, not validated sustainable cash or probability bounds.

- Optional per-scenario terminal cash resolves sale as max(0, cash_N+1)/(r−g).
  Cash is already after reinvestment/financing; no additional growth multiplication.
  Missing/invalid inputs fail closed, and nonpositive cash receives zero assumed
  sale with an explicit turnaround/run-off review prompt. Explicit sale remains
  available; editing sale selects that mode. Crisis sale remains independent.
- Working terminal tables and required-return NPV sensitivities expose the effect.
  Terminal/rate/notes edits retain the provenance of unchanged annual cash.
  Applying history settings preserves independent terminal assumptions/rates even
  when the user opts into them from an edited old v3 draft. Opening the standard
  baseline is a deliberate reset with a saved revision.
- Source evidence shows same-report operating/investing/financing/net-period cash,
  provider FCF and arithmetic differences in a common sourced currency basis.
  Missing components, source placeholders and unsupported numeric amounts remain
  explicit. No maintenance capex, shareholder cash or cash-definition repair is
  inferred from aggregate investing cash.
- Exact untouched v1/v2 and pre-terminal v3 defaults can upgrade only after backup
  and replacement persistence. All edits, deliberate blanks, custom models or
  calibrations, crisis, restored revisions and reviewed Holmen/SCA studies remain
  intact. Old v3 drafts reproduce exactly when terminalMethod is absent.

All 19,140 listings passed unchanged annual-path comparisons and exact old/new
v3 replay. New terminal inputs are available for 14,179 listings and missing for
4,961 (2,731 manual financials; 2,230 other insufficient histories). Complete
DCF/NPV is available for 12,939 listings. Independent scalar checks reconciled
38,885 valid scenarios, including 68 partly complete companies. These are generic
baseline implementation/input counts; reviewed studies retain their own figures.
The full cohort, inputs and forecast outputs are frozen with hashes for later
comparison. Fiscal outcome mapping and provider-vintage changes require review.

The component audit covered every listing and 182,888 overlapping fiscal keys.
145,747 FCF changes occur with all four other cash totals unchanged. Cause remains
unconfirmed. A dated correction supersedes two unreproduced arithmetic counts in
the earlier vintage report without rewriting the original historical text.

The two fixed midline challengers do not justify replacing latest cash. On the
24,440 recent Year 1 operating/property forecasts, the robust median blend worsens
mean error 0.28%, and geometric damped trend worsens it 3.99%. The blend improves
recent Year 2/3 mean error by 4.45%/5.34%, making a later horizon-specific trial
worth considering. These outcomes were previously inspected later-vintage data,
not untouched future evidence. Cash definition reconciliation and validated joint
cash-path/DCF/NPV distributions remain future research. COVID/rebound years stay.

Verification passed: **182 frontend unit tests**, **33 new Python research tests**
and Ruff, **45 integrated browser checks**, **9 dedicated terminal browser checks**,
and **47 Windows valuation checks**, including full-process restart of terminal
cash/growth/return edits with exact source basis. Browser/native runs had no
external requests or runtime errors. Desktop/mobile controls were visually
checked. The full-universe audit and independent source/ledger review passed.

Final Windows executable SHA-256:
`b9705a0d2f7e4be9ec31305f92776b54d35c48e517c68de75050298f1a4be0d5`.
**0.15.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.15.0\Macro Atlas.exe`.
The existing OneDrive desktop shortcut targets this version and its working
folder. Installed binary, both source packs, version metadata and release documents
match the verified build. Installation was checked at 2026-09-13T10:31:47.2991969Z.
The normal-profile app opened successfully (PID 34004). Version 0.14.1
is retained. Native tests used an isolated profile and did not replace user data.

Receipts/artifacts under `desktop/test-results/`:

- `unit-tests-0.15.0.log`, `research-tests-0.15.0.log`, `research-lint-0.15.0.log`.
- `frontend-build-0.15.0.log`, `windows-build-0.15.0.log`.
- `terminal-starters-2026-09-13/universe-audit.json`, `independent-review.json`,
  `runtime.mjs`, `frozen-starters.jsonl.gz` (19,140 listings; not tracked by Git).
- `cash-component-audit-2026-09-13/` and
  `cash-flow-midline-challengers-2026-09-13/` contain their ledgers and receipts.
- `terminal-valuation-browser-report.json`, `valuation-browser-report.json`,
  screenshots and `valuation-browser-test-0.15.0.log`.
- `windows-native-valuation-report.json`, `windows-valuation-test-0.15.0.log`,
  `windows-installation-0.15.0.json`, and `terminal-release-verification-0.15.0.json`.
- Previous 0.14.1 receipts/pre-change starter sources are retained under
  `releases/0.14.1/`; original pre-terminal runtime remains in the 0.14.0 universe
  audit directory. Do not overwrite that retained comparison runtime.

Methodology is ADR 0038 and `docs/standard-company-valuation.md`. New research
reports are `docs/cash-component-audit-2026-09-13.md` and
`docs/cash-flow-midline-challengers-2026-09-13.md`. Shared methodology is mirrored
to canonical `dalio-machine`; app implementation stays on `feat/offline-atlas`.
Sources remain local/uncommitted, including all preceding work; no Macro Atlas
commit or push was performed in this slice.

## Previous 0.14.1 handoff — 2026-09-13

## Current release — cash-flow, DCF and NPV range charts (0.14.1)

The user requested that the cash-flow min/max range over time carry through to
DCF and NPV. This patch improves the shared company charts; it does not change
the saved forecasts, calibration factors, discount rates or valuation arithmetic.

- Nominal cash and annual DCF shading now follows the exact envelope of the three
  linear paths, including intersections between annual points. Negative values
  and deliberately cleared cash inputs retain their signs and gaps.
- DCF and cumulative NPV have annual min/max markers, explicit axes/units, range
  checkpoints and inspectable tables with min/max and all named scenario values.
- NPV starts at the negative initial investment and accumulates each scenario's
  discounted payments separately. Its min/max is calculated across those cumulative
  paths, never by adding annual extrema from different scenarios. Year-end steps
  retain the existing cash timing; an enabled final sale changes only final-year NPV.
- Range provenance distinguishes historically informed cash from assumed later
  years, reviewed inputs and edits. NPV after Year 4 identifies the mixed cash-input
  basis. These labels do not assign an 80% probability to DCF/NPV or the full path.
  Widths follow the actual calculations and are not forced to increase with time.

Verification passed: **169 unit tests**, **7 dedicated range-browser checks**,
**36 integrated browser valuation checks** and **37 Windows valuation checks**.
The final Windows run used the exact executable below and included a full process
restart with saved Holmen/SCA/starter/crisis edits. Browser and Windows runs had
no external requests or runtime errors. The final DCF/NPV cards were visually
checked at 1440px and 390px, including badge labels, axes and legends. All 14 input
identities from the previous universe audit remain unchanged, including source
packs, calibration, forecast generation and valuation arithmetic.

Final Windows executable SHA-256:
`99d3421553c76dedfa80593ee59feffd8cc06cece61a67b52e020566d02efc3f`.
**0.14.1 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.14.1\Macro Atlas.exe`.
The existing OneDrive desktop shortcut targets this version and its working
folder. Installed executable and source pack hashes match the verified build;
Windows version metadata is 0.14.1 and the normal-profile app is running
(PID 42576). Installation verification passed at
`2026-09-13T09:44:52.1637875Z`. Version 0.14.0 is retained. Tests used isolated
profiles and did not replace saved user valuations.

New implementation: `desktop/src/scenarioRanges.ts`, `scenarioRanges.test.ts`,
`ValuationCharts.tsx`, `CashFlowForecastChart.tsx`, and
`desktop/tests/valuation-range-flows.mjs`. The range flow is included in both
ordinary browser/native valuation suites and restores all prior test-profile
state before existing restart checks. Versioned 0.14.0 receipts and pre-change
chart sources are retained under `desktop/test-results/releases/0.14.0/`.

Receipts in the desktop worktree under `desktop/test-results/`:

- `unit-tests-0.14.1.log`, `frontend-build-0.14.1.log`, `windows-build-0.14.1.log`.
- `valuation-range-browser-report.json`, `valuation-browser-report.json`,
  `valuation-browser-test-0.14.1.log`, and `valuation-{dcf,npv}-range-{browser,native}-{1440,390}.png`.
- `windows-native-valuation-report.json`, `windows-valuation-test-0.14.1.log`.
- `source-identities-range-0.14.1.json` preserves the unchanged data/model identities.
- `windows-installation-0.14.1.json`, `windows-install-0.14.1.log` and the
  retained `verify-windows-installation-0.14.1.ps1` verification script.
- `valuation-range-verification-0.14.1.json` binds the changed source and test
  receipt identities to the final executable.

Source changes remain local and uncommitted in the desktop worktree; the separate Codex memories
configuration closeout was pushed only to `claude-config` as `599db13`.

## Previous release — empirical company starters and crisis scenario (0.14.0)

The user approved the universe recommendation on 2026-09-13. Version 0.14.0
implements `empirical-cash-starter-v3` across **Companies → Value**. New generic
starters use five annual observations and the latest eligible signed cash held
flat. Weighted trend and weighted flat mean remain editable alternatives.

Eligible operating/property histories receive model-specific historical error
ranges for Years 1–4, grouped by five-year cash dispersion. The immutable bundle
retains COVID/rebound errors and targets 80% annual coverage in the research
sample; it does not establish a company-specific or joint-path probability.
Years 5–10 use the Year 4 half-width plus an explicitly assumed, editable 10% of
historical cash scale per additional year. Unsupported settings/source, timing or
currency lineage use labelled percentage assumptions. Missing inputs remain
visible. Financial businesses other than the property exceptions require reviewed
equity-cash/capital inputs in the same workspace.

A separate, initially disabled **Crisis scenario** has editable shock size/timing,
recovery, extra annual cash cost, required return and final equity sale. Its cash,
DCF/NPV, table and CSV are separate from the original low/mid/high paths and have
no assigned probability. Deep dives can normalize unusual history with sourced
explanations; raw observations are retained. Provider cash-definition changes and
untouched future evaluation remain open research work, not solved by this release.

Reviewed Holmen/SCA studies and existing edits, cleared inputs, titles, notes,
custom settings and restored revisions remain intact. Only exact untouched v1/v2
defaults upgrade automatically, after both the original backup and replacement
have been saved successfully. **Adjust history weights & range → Use empirical
defaults → Apply history assumptions** allows a deliberate switch with backup.

Verification:

- 158 frontend unit tests, 22 Node backtest tests and 13 calibration-export tests
  passed; final frontend and Windows production builds passed.
- The integrated browser valuation suite passed 29 checks. A final standalone
  empirical suite passed 14 checks after a mobile chart-label spacing fix.
- The full Windows suite passed 110 checks before that spacing-only change;
  its receipts are archived under `releases/0.14.0-before-mobile-spacing/`.
  The final executable passed all 30 targeted Windows valuation checks, including
  full process restart with saved Holmen/SCA/starter/crisis edits. Both runs had
  no external requests or runtime errors.
- The default-runtime audit covered all **19,140 listings**: **8,706 historical**,
  **5,805 percentage**, **4,629 unavailable**, including **2,731 manual financials**.
  **13,214** listings had all inputs for DCF/NPV. All **39,642** ready scenarios
  reconciled against independent scalar calculations. Both older methods reproduced
  exactly for every listing (**38,280 full draft comparisons**). An independent
  review checked all counts, 18 bound file hashes and six raw packed-source anchors.
  This verifies implementation and coverage, not future forecast accuracy.

Final Windows executable SHA-256:
`81e0a280d393fc480c53384a888b2fa78ebd9db1d0362e65260708d7442593c4`.
**0.14.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.14.0\Macro Atlas.exe`.
The existing OneDrive desktop shortcut targets this version and its working
folder. The installed executable matches the final native-tested hash; both
financial packs match their content hashes, version metadata is 0.14.0, and the
normal-profile app is running (PID 25928). Installation verification passed at
`2026-09-13T07:20:53.0055135Z`. Previous version 0.13.0 is retained. User profiles
were not replaced by test data; app-flow verification used isolated profiles.
Source changes remain local and uncommitted; no push was requested.

Receipts in the desktop worktree under `desktop/test-results/`:

- `unit-tests-0.14.0.log`, `frontend-build-0.14.0.log`, `windows-build-0.14.0.log`.
- `valuation-browser-report.json`, `empirical-valuation-browser-report.json` and
  the `empirical-*-1440.png` / `empirical-*-390.png` screenshots.
- `windows-native-valuation-report.json`, `windows-valuation-test-0.14.0.log`.
- `windows-installation-0.14.0.json`, `windows-install-0.14.0.log` and the
  retained `verify-windows-installation-0.14.0.ps1` verification script.
- `empirical-cash-starter-2026-09-13/universe-runtime-audit.json`, SHA-256
  `958263c94c9936637aa1b79e535c19f10d8d82af8b4e27367e82099214b1187e`.
- `empirical-cash-starter-2026-09-13/universe-independent-review.json`, SHA-256
  `e3b620b8923678e35ed79e25c0f44807b01c743b06f5d56396856762751c2818`.
- Previous 0.13.0 receipts are retained under `releases/0.13.0/`.

See ADR 0037 and `docs/standard-company-valuation.md`. Keep desktop implementation
on `feat/offline-atlas` and shared methodology/handoff on canonical `main`.
Generated research/test artifacts are gitignored and need separate retention.

## Previous research checkpoint — cash-flow uncertainty groups

The subsequent universe-setup recommendation and FY2020 calibration sensitivity
are recorded in `docs/cash-flow-universe-setup-proposal-2026-09-12.md`. Omitting
FY2020 errors narrows recent cash-group ranges only 2.25%, with coverage moving
82.45% to 81.63%; grouping rankings remain unchanged. It does not remove pandemic
effects from historical features or justify deleting crisis years. At this research checkpoint, the proposal had not yet been adopted; the
current 0.14.0 release above records its subsequent implementation.

The follow-up tests whether sector, branch, asset intensity and cash behavior
help set uncertainty ranges. The completed experiment compares 18 fixed grouping
rules plus four within-branch refinements. Its primary recent Year 1 comparison
contains 24,440 forecasts across 13,069 operating/property listings; lenders,
insurers and other financial businesses are retained in a separate sensitivity.

Historical cash dispersion improves interval score by 6.81% versus one global
range, compared with 0.36% for sector, 0.59% for branch, 0.29% for guarded tangible
book equity/EBIT and -0.11% for tangible assets/sales. Lower interval score balances
width and misses; mid forecasts are unchanged. Recent global coverage within low/
medium/high cash-dispersion histories is 97.4% / 90.8% / 67.9%; conditional ranges
bring it to 82.6% / 82.3% / 82.5%. Cash-dispersion gains persist across FY2022–2025,
both mid models and duplicate-history downweighting, but FY2022 coverage is only
75.6%. This is exploratory evidence, not a guarantee of future 80% coverage.

Within branches, recent Year 1 score improvements are 0.019% for tangible assets/
sales, 0.156% for TBV/EBIT, 0.445% for EBIT margin and 2.032% for cash-trend residual
dispersion, relative to branch-only ranges. TBV/EBIT falls back to parent groups
in 91.6% of weighted observations, limiting conclusions. Book equity/EBIT mixes
capital intensity, margins and financing; keep operating capital requirements,
normalized returns and liquidity/maturities as separate analytical questions.

Report: `desktop/test-results/cash-flow-segmentation-2026-09-12/report.html`
in the desktop worktree. It includes interactive comparisons, conditional
coverage/width charts, within-branch results and historical company-feature search.
See `docs/cash-flow-segmentation-2026-09-12.md` for definitions, results, Windows
path, limitations and reproduction; the companion concepts note links primary
sources. Generated research artifacts remain gitignored and need separate
retention. The original backtest report is preserved.

Verification: 23 focused Python tests (plus 19 subtests), Ruff and seven offline
browser check groups passed. Independent recalculation checked the main and
supplemental factors/summaries, all 2,304 main and 170 supplemental comparison
cells, and all 34 branch baselines. Final report SHA-256:
`aa81f5752762d07b81659bca2cf51d3d88b97784fd72913b3590220a03d9d429`.
Browser checks found no external requests, runtime errors or page overflow at
1440px/390px. Receipts and screenshots are alongside the report.

At that checkpoint the installed app, starter defaults, saved drafts and source
pack were unchanged. Provider cash definitions still need reconciliation and new
forecasts/data need freezing for untouched future evaluation. Next research inputs should
separate cash components and add dated evidence on demand commitments, customer
concentration, cyclicality, reinvestment and financing events. Do not confuse
cash predictability with business quality or validate DCF prices from annual
cash coverage.

## Previous research checkpoint — cash-flow backtesting

The user requested measured forecast accuracy, company hit/miss identification,
and research into the misses. A first offline audit now covers all 19,140 saved
listings and 4,139,652 forecast rows across four fixed models, five/ten-year
histories, and horizons one to five. This is retrospective research; the installed
0.13.0 app, editable starter defaults and saved drafts remain unchanged.

Recent FY2024–2025 outcomes fall within the original five-year trend sensitivity
range in 13.34% / 17.54% / 20.91% / 25.50% / 30.88% of eligible observations at
horizons one through five. An older-error range targeting 80% coverage reaches
82.11% at Year 1, versus 12.99% for the original range on the same eligible rows,
but is about 11.6 times wider. Last reported cash has lower median one-year point
error than the weighted trend in this sample. No new winner/default is selected.

A separate June 2025 frozen-input pilot retains its complete 15,646-listing
cohort. Main FY2024→FY2025 agreement is 11.45% on 9,741 admissible outcomes;
1,666 eligible fits have unavailable outcomes. Large historical provider FCF
changes between downloads, including 146,682 rows whose operating/investing cash
are unchanged, confound interpretation of the pilot as pure forecast skill.
The cause of the provider measurement changes remains unconfirmed.

The searchable offline report and full ledgers live in the desktop worktree:
`desktop/test-results/cash-flow-backtest-2026-09-12/report.html`.
The report includes filters, forecast/actual charts, source-vintage caveats,
paired model and range comparisons, company CSV exports, two source-backed miss
reviews (Stora/SCA) and an ABB hit/miss example. Other causal explanations remain
unreviewed. Generated ledgers, reports and receipts are gitignored and need
separate retention; pushing code will not preserve them.

Verification: 22 Node tests and three Python common-scale tests passed; frozen
pilot arithmetic/date/classification checks passed for all 15,646 cohort rows.
Independent checks reconciled all forecast-group counts and hashes, 30 recent
ABB/SCA/Stora forecasts, and 379,768 paired history-window comparisons. Seven
browser check groups passed, including offline load, charts, filtering/CSV,
calibration/cohort displays, missing outcomes and desktop/mobile layout, with no
external requests or runtime errors. Ruff and whitespace checks are clean.

See `docs/cash-flow-backtest-2026-09-12.md` and
`docs/cash-flow-backtest-data-vintages-2026-09-12.md`. Reusable research scripts and
focused tests belong in the desktop branch. Do not interpret observations as
independent trials, current-vintage results as point-in-time execution, sensitivity
bands as confidence intervals, or annual cash coverage as validation of DCF prices.
Next improvement: reconcile cash definitions, freeze new forecasts/source vintages,
then test simpler mids and empirically scaled ranges on future untouched outcomes.

## Previous installed checkpoint — cash-flow trend and widening uncertainty (0.13.0)

The company Value workspace now includes **Cash flow over time**: nominal company
cash on the vertical axis and annual periods on the horizontal axis. Saved history
is shown separately from future low/mid/high paths, with a shaded scenario range
and annual range markers. It uses the current draft and remains useful when price
or discount-rate inputs are missing. DCF and NPV discount these same forecast
payments. The chart does not fit probability densities to assumed intervals.

New standard starters use `weighted-cash-starter-v2`: a weighted straight-line fit
to five or ten consecutive annual cash-flow proxies. Five-year weights remain
30/25/20/15/10, newest first; ten-year weights default to 19/17/15/13/11/9/7/5/3/1.
The latest historical year is model time zero. Forecast mid cash is fitted latest
level plus annual slope times model year. The default range is ±10% in year one,
±20% in year two, ±30% in year three, continuing to ±100% in year ten. Low/high
subtract/add that percentage of the absolute mid cash, preserving signed ordering.
The percentage range can exceed 100%; absolute cash width also changes with the
mid amount. This is an editable sensitivity model, not a calibrated confidence
interval. A one-observation trend explicitly uses zero slope.

**Adjust history weights & range** exposes history length, projection (trend or
flat weighted average), annual weights, first-year range and yearly widening.
Applying settings preserves the preceding draft. Exact untouched v1 defaults
upgrade with a saved revision; custom weights, entered forecasts, notes, titles,
restored revisions and deliberately cleared inputs remain intact. Old v1 drafts
can be reproduced exactly. Holmen and SCA retain their reviewed refinements; the
new chart shows their actual scenarios. Source lineage, currency and missing-data
checks continue to apply. CSV exports retain the new settings and edited status.

Verification so far: **131 frontend unit tests and the production build passed**.
The independent audit reconciled all **19,140 listings** at 2026-09-12 for both
five- and ten-year windows: **15,130 ready and 4,010 with unavailable inputs**.
There were no regression, signed-band, DCF/NPV, source-window or price-lineage
mismatches. All 26 independent edge cases passed, and original v1 numerical
parity passed for every listing. All seven original source/pack identities remain
unchanged. The final production browser suite passed all 98 checks; the focused
valuation run passed 24 checks. Neither recorded external requests or runtime
errors. All 105 Windows native checks passed on the exact binary below, with
no external requests or runtime errors, including a full process restart. The
first launcher was interrupted (exit 143, no final receipt); the complete rerun
finished successfully with exit 0.

**0.13.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.13.0\Macro Atlas.exe`.
The existing OneDrive desktop shortcut targets this version and its working
folder. The installed executable matches the native-tested SHA-256; both financial
packs match their content hashes. The normal-profile app is running (PID 41832).
Verification covered installed files, shortcut and process launch; the browser
screenshots use isolated test profiles. Installation receipt verified at
`2026-09-12T19:59:25.6712264Z`. The release is ready through the existing shortcut.

Local receipts in the desktop worktree under `desktop/test-results/`:

- `universe-valuation-runtime-0.13.0-2026-09-12.json`.
- `universe-valuation-edge-audit-0.13.0-2026-09-12.json`.
- `valuation-source-identities-0.13.0.json`.
- `windows-build-0.13.0.log` (build passed); Windows binary SHA-256:
  `d8172bcdde77d9633da14d75117adb4e6b820bd29114d43f1b8cf5f7332f9d63`.
- `windows-native-report.json` and `windows-test-0.13.0.log`.
- `windows-installation-0.13.0.json` and `windows-install-0.13.0.log`.
- `browser-report.json`, `valuation-browser-report.json` and
  `atrium-cash-flow-fan.png`, `stora-cash-flow-fan.png`, `abb-cash-flow-fan.png`.
- Prior 0.12.0 binary, handoff and test receipts: `releases/0.12.0/`.

Source changes remain local and uncommitted. Keep app implementation on
`feat/offline-atlas` and shared methodology/handoff on `main`; no branch merge is
part of this work. See `docs/standard-company-valuation.md` and ADR 0036.

## Previous checkpoint — standard company DCF/NPV and reviewed refinements (0.12.0)

The user expanded the task from a second researched company to a common editable
DCF/NPV starting point for the entire company universe. **Companies → select a
listing → Value** now supplies the standard historical cash-flow model. Reviewed
deep dives refine the same workspace: **Holmen and SCA** open with their reviewed
studies, and retain access to the standard baseline.

Method `weighted-cash-starter-v1` uses up to five consecutive full annual saved
cash-flow proxies, weighted **30%, 25%, 20%, 15%, 10%**, newest first. The mid
forecast is flat; low/high subtract/add **20% of the absolute mid amount**, so
negative scenarios retain their ordering. **Adjust history weights & range**
lets the user change the five weights and spread; weights must total 100%.
Applying new settings first saves the preceding draft. Annual payments, dated
price, required return and final sale remain independently editable.

The default horizon is ten full years with a 10% nominal equity return. The
default final sale capitalizes positive final annual cash with zero mature
growth; nonpositive cash has zero assumed sale proceeds. This is an explicit
sensitivity model. Saved vendor FCF is not verified FCFF or reported shareholder
cash, and the range is not a statistical confidence interval. Negative payments
illustrate hypothetical funding needs, not a shareholder's personal obligation.
Lease principal, interest, acquisitions, financing and regulatory capital need
company-specific review. The standard does not populate unsupported recovery or
subtract net debt again from the cash proxy.

The evidence view shows each source period, saved amount, currency ratio, amount
used, original/effective weight, dated price and provenance. Partial consecutive
history uses an explicitly disclosed included-weight denominator. Latest
missing/stale/short or withheld annual periods, source placeholder rows and
currency gaps remain unavailable. Price must match the latest annual year and
source ID; older prices cannot bypass current share/scale flags. Native amounts
or documented historical quote-currency amounts are used consistently, without
an invented current FX conversion. Company histories still load on demand.

The actual-model audit at valuation date **2026-09-12** checked all **19,140
listings**: **15,130 have a calculable standard starter; 4,010 retain missing or
unsupported inputs**. Every listing retains the controls and both chart sections;
missing values produce explanations rather than fabricated curves. These are
coverage counts, not investment recommendations. New standard studies capture
their own valuation date; saved study dates, source downloads, fiscal ends and
historical price dates remain independent.

SCA's new reviewed study is `sca-2026-09-12-v1`, built from original annual and
interim reports with an explicit shareholder-cash bridge, dated equity price,
20-year scenarios and separate asset recovery schedule. A reusable v2 authoring
contract separates evidence, source facts and assumptions from the shared UI.
Registration requires the listing/ISIN and exact archived research path/hash.
Holmen's immutable v1 source JSON and complete generated draft remain unchanged
through a legacy adapter. Earlier drafts and deliberate blank edits survive;
CSV exports distinguish reviewed and standard origins, dates, settings and edits.

Methodology and source worksheets:

- [Standard company valuation](../docs/standard-company-valuation.md) and
  [ADR 0036](../decisions/0036-standard-company-cash-flow-scenarios.md).
- [SCA reviewed study](../docs/sca-valuation-2026-09-12.md) and
  [Holmen reviewed study](../docs/holmen-valuation-2026-09-11.md).
- [Company valuation framework](../docs/company-valuation-framework.md).

## 0.12.0 verification and release status

**122 frontend unit tests, 95 complete production browser checks and a final
21-check focused valuation browser run passed.** The browser runs recorded no
external requests or runtime errors. The independent actual-model universe
audit also passed without runtime exceptions or unresolved calculation/parity
differences. Source data and the existing research/financial pack identities
were preserved; the final audit verified all seven protected source identities.

The final Windows binary passed **all 102 complete native checks**, with zero
external requests and zero runtime errors. Full-process restart preserved the
Holmen and SCA studies, standard weights, user edits and deliberately cleared
inputs. The fresh `windows-native-report.json` records this exact binary.
**0.12.0 is installed and verified** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.12.0\Macro Atlas.exe`.
Its checksum matches the native-tested binary, and the existing OneDrive desktop
`Macro Atlas.lnk` targets this executable with the correct working directory.
Both installed financial-pack hashes were verified. The normal-profile app
process was running (PID 23152) at installation verification;
`verifiedAt = 2026-09-12T19:12:52.8143275Z`. Verification covered installed files,
shortcut configuration and process launch.
The native run used process-only PowerShell
`-ExecutionPolicy Bypass` for its UNC test-script path; the machine execution
policy was not changed.

Final 0.12.0 binary SHA-256:
`784a5a84807235a8725af70d0208aff404b5b90ac09bbc2aecd4717eb680dcb6`.
Built executable: `src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe`.

Current local receipts under `test-results/`:

- `browser-report.json` and `valuation-browser-report.json`.
- `universe-valuation-runtime-receipt-2026-09-12.json` (actual model, source hash,
  independent parity review and September 12 coverage);
  `universe-valuation-coverage.json` (supporting source audit).
- `valuation-source-identities-0.12.0.json` (seven preserved source identities).
- `windows-native-report.json`, `windows-build-0.12.0.log` and the completed
  `windows-test-0.12.0.log`.
- `windows-installation-0.12.0.json` and `windows-install-0.12.0.log`.
- `sca-automatic-valuation.png`, `sca-research-evidence.png`, SCA reviewed/edited
  CSVs and `universe-valuation-export.csv`. Original SCA report bytes/text are
  retained under `valuation-sources/sca/`.

The release is ready to use through the existing desktop shortcut. This
checkpoint has not been committed or pushed. Keep the desktop worktree on
`feat/offline-atlas` and canonical engine on `main`; no branch merge is included.
Generated packs, source archives, receipts and binaries remain local/gitignored.

For subsequent Windows builds on this PC, activate the canonical `.venv` and use
`ATLAS_BUILD_TOOLS=/home/rosinco/.cache/macro-atlas-tools python scripts/build-windows-native.py`
from `desktop/`. The persistent tool cache was restored and validated for this
release; it avoids dependence on an ephemeral `/tmp/atlas-tools` directory.

## Previous checkpoint — automatic Holmen valuation (0.11.0)

User request: populate the workspace automatically for previously researched
companies, starting with a Holmen test. **Company observatory → Holmen → Value**
now loads `holmen-2026-09-11-v1`: original June 2026 accounts, saved 7 August
B-share close of SEK 329, corrected outstanding-share basis, explicit low/mid/high
dividend assumptions, 20-year DCF/NPV, payback and separate breakup recovery.
Source figures, normalization and assumptions are inspectable. Input forms open
via **Edit price & assumptions**; results show first.

The authored, versioned JSON is in `research/valuations/`. The registry requires
matching listing/ISIN and archived deep-dive SHA-256. Only Holmen is included so
far. Old price-only drafts are preserved as saved revisions before automatic
replacement. Entered forecasts/notes, explicitly restored drafts and deliberate
blank edits survive reload. Resetting to researched assumptions preserves the
current draft first. Study origin is retained in local storage and CSV.

The archived May deep dive is not treated as a current recommendation. Its
forest-plus-whole-business double count is removed. Cash taxes and lease
principal are reconciled before setting analyst dividend scenarios. A common
9% required return produces PV per SEK 1,000 of 210.76 / 401.38 / 783.87;
the high cash-payment case pays back in year 16, while discounted payments do
not recover the purchase within 20 years. These are conditional sensitivity
results, not source facts or a buy verdict. Capital uses explicitly labeled
book-value proxies; recovery retains all prior claims and is not a floor.

Verification: **101 automated tests, 85 complete production browser checks and
90 complete Windows checks passed**, including eleven valuation flow checks,
native CSV export and full-process restart. No external requests or runtime
errors occurred. Version **0.11.0 is installed** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.11.0\Macro Atlas.exe`; the existing
desktop shortcut targets it. Earlier installed versions are retained.

The installed 0.11.0 app is open on **Company observatory → Holmen → Value**
in the user's normal profile. Windows accessibility inspection and a screenshot
confirmed the automatic-study banner, archived previous-draft notice, SEK
49,492.96m price basis and all three expected present values. Screenshot:
`test-results/holmen-installed-user-profile.png`. The temporary browser preview
server was stopped; the installed app remains open.

Windows binary SHA-256:
`b41ed495c32442811d871e9a7a8d083911bde3460a5df60a15dd0d3d0e86e4c5`.
Use **`python scripts/build-windows-native.py`** with the canonical `.venv`
activated on this machine. The cargo-xwin path spends many minutes re-extracting
CAB files; the existing native-extracted SDK path completed the final build in
93 seconds. The Microsoft SDK's missing external debug-symbol warning does not
prevent the optimized binary from building or passing the native checks.

Receipts: `test-results/browser-report.json`, `valuation-browser-report.json`,
`holmen-study-provenance.json`, `holmen-automatic-valuation.png`,
`holmen-automatic-charts.png`. Original June report bytes and extracted text are
in `test-results/valuation-sources/`. The prior 0.10.0 binary is preserved under
`test-results/releases/0.10.0/`.

The framework, ADR, Holmen worksheet and workflow entry points are also present
in the canonical macro worktree. Source Börsdata files and macro facts were not
changed. See `docs/holmen-valuation-2026-09-11.md` in the worktree root.

## Previous 0.11.0 repository checkpoint and continuation

Pre-commit verification on this desktop worktree: **1,206 Python tests passed**;
`ruff check src tests` and the staged diff checks are clean. The Python log is
retained locally at `test-results/precommit-desktop-pytest.log`. The 101 frontend,
85 browser and 90 Windows checks above cover the unchanged app build.

This checkpoint combines the interactive valuation workspace and Holmen's
automatic researched study on `feat/offline-atlas`. The shared framework,
worksheet, ADR and canonical handoff are on `main` in the same GitHub repository,
`Rosinco/dalio-machine`. Keep these branches separate; no integration merge is
part of this checkpoint.

Implementation is paused for the user's commit, push and handoff request.
The next company-analysis step is to add reviewed numerical studies for further
completed deep dives using Holmen's registry pattern. Each case needs its own
source reconciliation, dated price, explicit forecasts and recovery assumptions;
Holmen remains the only automatic case. The installed app is ready to use.

Generated source archives, data packs, test receipts, screenshots and Windows
binaries remain local and gitignored. The versioned Holmen JSON and methodology
are included in the source checkpoint; a Git push does not back up local artifacts.

## Previous company valuation checkpoint — 0.10.0

The user requested the interactive company valuation workspace and high/mid/low
DCF and cumulative NPV charts with scenario payback. Implementation is in
`desktop/src/{ValuationWorkspace,ValuationCharts}.tsx`, `valuation.ts`,
`savedValuations.ts` and `valuation.css` on the desktop working tree. New company
analyses use ADR 0035 and `docs/company-{valuation-framework,analysis-template}.md`.
The same methodology and workflow entry points are present in the canonical tree.

Open **Companies → choose a company → Value**. Enter dated equity prices and
explicit shareholder-payment forecasts; the source FCF series is not an automatic
forecast. Each scenario includes ordinary/discounted payback, optional final sale
and separate net recovery. Capital and evidence notes accompany the model.
Drafts autosave by company and exact data versions, while saved revisions retain
old assumptions. CSV exports retain inputs and annual calculations.

Verification: **96 automated tests passed**, including 14 valuation/storage
checks. The complete production browser suite passed, including eight valuation
flows, without external requests or runtime errors. Web and Windows builds passed.
The complete Windows native suite also passed **87 checks**, including
valuation CSV export and draft/revision persistence after a full process restart,
with no external requests or runtime errors. **0.10.0 is installed** at
`C:\Users\Adamb\AppData\Local\MacroAtlas\0.10.0\Macro Atlas.exe`.
The existing `C:\Users\Adamb\OneDrive\Desktop\Macro Atlas.lnk` now targets
this version. The installed executable checksum matches the native-tested build.
Earlier installed version folders remain intact. Reopen Macro Atlas to use the
new workspace; installation did not close existing app windows.

Native receipt: `test-results/windows-native-report.json`.
Installation verification: `test-results/windows-installation-0.10.0.json`.
The 0.10.0 workspace source is included in the current 0.11.0 checkpoint.

0.10.0 Windows binary SHA-256:
`a6481b7d93c87875d1ebf2db80d14cc2506dbd2177de53bb3904edd4747ed163`.
The earlier 0.9.0 binary was retained at
`desktop/test-results/releases/0.9.0/macro-atlas.exe` with its original checksum.
The macro evidence pack, source data and existing import identities are unchanged.
No country collection or automatic company valuation was performed for this feature.

Current receipts in the desktop `test-results/` directory:
`browser-report.json`, `valuation-browser-report.json`,
`valuation-documentation-report.json`, `valuation-charts.png`,
`valuation-compact.png`, and `valuation-export.csv`.
Browser screenshots and CSV use explicitly hypothetical test inputs, not a
company investment conclusion. The Windows test uses an isolated app profile.

## Previous 0.9.0 checkpoint (historical)

Paused at the user's request before the final complete Windows native suite and
desktop installation. Version 0.9.0 is implemented and built, **not installed**.
The existing desktop shortcut has not been changed in this checkpoint.

## Implemented scope

Macro → Assessments presents annual profiles for all 19 saved listing countries.
Original-source monitoring covers Sweden, Norway, Denmark, Finland, the United
States, Germany and Canada: 28 source-bound histories and one documented US
corporate-lending gap. The old Fed survey is discontinued; no proxy fills it.

The view includes exact historical comparison windows, native history charts,
dated IMF paths, structural context, conditional scenarios and company evidence
requirements. Source units, scopes, missingness, provisional flags and acquisition
clocks remain visible. Directional changes use neutral colours. Existing ratings
retain red = weaker, yellow = mixed, green = stronger.

Country evidence has independent September 10–11 assessment/knowledge dates.
The bundled September 8 fundamentals and existing research-import identities
retain their original dates. Countries without an older scored profile can open
their new assessments, including Finland and Norway. Listing GB resolves to UK.
No company exposure, country score or forecast probability is inferred.

## Working locations and immutable inputs

- Desktop: `/home/rosinco/workspace/dalio-atlas-desktop`, `feat/offline-atlas`.
- Canonical engine: `/home/rosinco/workspace/dalio-machine`, `main`.
  US/DE/CA implementation: `34a229e` (pushed).
- Full macro handoff: canonical engine `HANDOFF.md`.
- Base assessment: `data/snapshots/country_assessments/b2524c09385af9682a462be968e0eb20ccd5e5917e0e73d43b06949f93fd5144/snapshot.json`.
- Nordic monitoring: `data/snapshots/nordic_monitoring/289d3f3eb37afbba9d6f62758d91947d412ac3f7d12a74d72bb809851dbf6804/snapshot.json`.
- US/DE/CA monitoring: `data/snapshots/national_monitoring/62c94ffb8a31495db5cdccf92bb7d1f806764c0f77742110be70fe15d4a0ef46/snapshot.json`.

The three snapshot paths above are relative to the canonical engine. Annual
profiles select the newest whole base or embedded monitoring parent while
retaining its own identity and cutoff. The active app pack is
`156bcef284718b80a273ae8d32d03b6c2c6013c1891da5516f2e29430b61fdad` under
`public/data/country-evidence/`: 19 country files, 152 annual histories,
29,175 native monitoring rows, including 415 explicit nulls. Country files load
on demand with a five-country memory cache. See README for the export command.

Only the active country pack is bundled. Two unreferenced development drafts
were moved to `test-results/country-evidence-drafts/` and remain available.

## Verification and installation

Backend: 1,690 tests passed. Frontend: 82 tests passed. All 53 desktop Python
tests passed, including eight new exporter tests; Ruff and diff checks passed.
Production web and Windows builds succeeded. The complete browser suite passed
with no external requests or runtime errors. The focused Windows financial test
passed full 109,862,912-byte import/export, checksum, reload and process-restart
checks. The final complete native suite and installation remain pending.

Earlier debugger sessions disconnected after the importer had published the
correct financial file. Captured logs establish malformed incoming CDP JSON
followed by the automation client closing its connection; an application crash
has not been established. Diagnostic receipts and failed isolated profiles are
retained. The test records progress, runtime versions and sanitized connection
events, and uses Playwright's local connection mode and public file-path upload
API. The production financial importer is unchanged. See
`test-results/windows-native-diagnostics-*.json` and
`test-results/windows-native-financial-report.json`.

The alternate uncompressed public Playwright transport passed the full-size
focused flow in 55 seconds on the same binary. Receipt:
`test-results/windows-native-diagnostics-2026-09-11T07-30-11.128Z.json`.
Both connections negotiated no compression, with no unexpected disconnects,
external requests or runtime errors. This establishes that transport's success;
it does not isolate a compression-library cause because the client path also
changed. This uncompressed transport is now the test default; `-CompressedCdp`
restores the previous connection for diagnostics. The complete native suite is
still required before installation.

Built executable:
`src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe`.
SHA-256: `8e9c1dccaa3d55113fe4a097f65cc3668b76a7c07cdab661521797388d386f93`.
Only test harness and documentation changed after this build; the production
application and financial importer are unchanged from the tested binary.

Independent pack audit:
`test-results/country-evidence-validation/desktop-pack-156bcef2-independent.json`.
Reproduce with `audit_desktop_country_pack.py` in the same directory. It checks
exact projection against the three verified snapshots, all country/index hashes,
source clocks and unchanged source/pack files. Original response-vector and
comparison audits are retained separately in the backend validation archive.

## Historical 0.9.0 continuation (completed by 0.11.0)

The steps below describe the earlier pause. The full native suite, installation
and installed-profile verification are complete as recorded at the top.

1. Run `scripts/test-windows.ps1` in Windows PowerShell without a focused-scope
   flag. Its default connection is now uncompressed. Require a fresh passing
   `test-results/windows-native-report.json` with the binary hash above and
   country-evidence pack `156bcef284718b80a273ae8d32d03b6c2c6013c1891da5516f2e29430b61fdad`.
   The existing full native report predates this build and is not its verification.
2. After that passes, run `scripts/install-windows.ps1`, verify the installed
   checksum and desktop shortcut, and record the installation receipt. The script
   preserves older version folders and validates both financial companions.
3. Update the desktop and canonical-engine handoffs, commit and push the release
   checkpoint, then resume the country expansion below.

Twelve listing countries still have annual profiles without this detailed
monitoring panel: BE, CH, EE, ES, FR, IT, LT, LV, NL, PL, PT and UK. Continue in
bounded country groups, auditing retained data first and preferring original
statistical offices and central banks. Preserve native definitions and explicit
gaps; this is not yet a comparable country ranking.

Destatis announces the selected German CSV updates will end in October 2026.
Establish its original-source successor while preserving the industrial scope,
adjustment and clock definitions before the next collection depends on it.

Refresh this first country-evidence version by rebuilding/installing the app.
It is independent of user-importable `.atlas.json` research packages. The full
local source archive, generated packs, reports, binaries and financial SQLite
companions are gitignored: pushing code does not back up those artifacts.
Preserve them with the backend database. Sector transmission and verified company
balance-sheet/physical-asset exposure remain subsequent work.
