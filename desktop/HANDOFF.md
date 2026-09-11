# Macro Atlas 0.9.0 handoff — 2026-09-11

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

## Continuation

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
