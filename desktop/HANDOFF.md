# Macro Atlas 0.11.0 handoff — 2026-09-11

## Current checkpoint — automatic Holmen valuation

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

## Repository checkpoint and continuation

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
