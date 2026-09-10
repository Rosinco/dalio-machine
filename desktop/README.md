# Macro Atlas desktop

An offline Windows viewer for the existing Dalio fundamentals snapshot. The app
bundles its map, scripts, charts and source data. Starting the packaged executable
requires no terminal, Python environment, WSL, account or network connection.
Windows WebView2 must be installed; it is present on Adam's PC. The Tauri installer
configuration can also bundle its offline installer when distributing to another PC.

## Version 0.1.1

Assessment colours consistently mean **red = weaker, yellow = mixed/middle,
green = stronger**. Historical maps follow each scored indicator's existing
`higher_is_better` direction: high debt or dependency is red; low debt or dependency
is green. Bands are relative to the covered countries in the selected year, rather
than absolute safe/danger thresholds. Missing observations stay grey. Unscored
quantities and trade use blue; blue/purple chart lines identify countries or series.
The radar's red inner rings represent weak scores and green outer rings strong
scores. None of these presentation changes recalculate the saved fundamentals.

- World map: 177 Natural Earth overview features; research for 21 countries and a
  separately selectable euro-area aggregate.
- Country search and map selection, with an explicit unavailable state elsewhere.
- Five-category fundamentals radar and country comparison. The map uses fixed
  20-point score bands; no default overall score or crisis probability.
- Annual history and forecast lines, with missing years left as gaps. A historical
  map displays raw observations for the selected year; it excludes forecasts and
  does not carry values forward. Its colour breaks adapt to that year's range and
  follow the indicator's saved direction, without calculating historical ranks.
- Goods-trade map, doughnut and paired bar chart. Overlapping euro-area partner
  totals are excluded. The residual preserves the total-export denominator.
- Read-only pressure diagrams for existing triggered rules, labelled as judgment.
- Indicators, observation dates, source names, quality tiers, and snapshot SHA-256.
- CSV history export; Windows writes a new timestamped file to Downloads.

The app reads the current saved vintage. Historical charts are **not** point-in-time
replays. Source metadata in this snapshot is insufficient for a per-point release
ledger, so the UI does not invent it. The five-year GDP-growth headline and annual
growth history are explicitly distinguished. Company/sector layers and geographic
asset inventories are later slices; no company sites or forecasts are fabricated.

## Data flow and scale

The canonical Python pipelines remain authoritative. `export_snapshot.py` validates
with Dalio's existing parser and creates a compact index, individual country files,
and a history-map file. Startup loads summaries; selecting a country loads only its
history; the full small annual panel loads only in History mode. The source SQLite
database is never opened or modified by the exporter or viewer. Original data and
all statistical calculations remain unchanged.

Large Börsdata Parquet tables are not included or loaded by this release. A future
bounded query layer can supply company/peer slices using DuckDB or the existing
Python adapters. Do not load the 101-million-row screener file into the webview.

## Build and refresh

From this `desktop/` folder:

```sh
npm ci
/home/rosinco/workspace/dalio-machine/.venv/bin/python scripts/export_snapshot.py \
  --source /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json
python3 scripts/verify_pack.py \
  --source /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json
npm test
npm run build
```

Generated research data is deliberately gitignored. The checked Natural Earth map
has its source, original checksum and processing notes in `public/maps/NOTICE.txt`.
Refreshing the app currently means exporting a new snapshot and rebuilding; there
is no automatic update or online refresh button.

Native Windows development uses `npm run tauri build`. The optional NSIS installer
configuration includes offline WebView2 setup. For this WSL build,
`scripts/build-windows.sh` uses Rust, cargo-xwin, LLVM and cached MSVC/Windows SDK
libraries to produce `src-tauri/target/x86_64-pc-windows-msvc/release/macro-atlas.exe`.
Set `ATLAS_LLVM_BIN`, `ATLAS_LLVM_LIB` and `ATLAS_XWIN_CACHE` to override tool paths.
The build on this PC uses `python3 scripts/build-windows-native.py` after extracting
the same official SDK packages with native `msiextract`. Its tool root defaults to
`/tmp/atlas-tools` (`ATLAS_BUILD_TOOLS` overrides it). The Windows SDK and Universal
CRT are separate packages: `ATLAS_SDK_VERSION` defaults to `10.0.26100.0` and
`ATLAS_UCRT_VERSION` to `10.0.10240.0`. The script validates its include/library
directories and statically links the C runtime. These cached tool paths are build
prerequisites, not a portable toolchain installer.
The cross-compiled executable embeds the whole `dist/` folder. Build tools are not
needed to run it.

After verification, run `scripts/install-windows.ps1` in Windows PowerShell. It
copies the app into the current user's LocalAppData and creates **Macro Atlas** on
their actual desktop, including OneDrive desktops. It updates earlier Macro Atlas
shortcuts, keeps prior version folders, and refuses to replace a different
executable in the same version folder or an unrelated desktop shortcut. `-Start`
closes older Atlas viewer windows and opens the installed version.

## Verification

`npm test` checks colour direction (including debt, negative values, missing values,
equal observations and unscored quantities), zero/missing scores, historical gaps, forecast boundaries,
single-vintage trade, overlapping aggregates and trade denominators.
`scripts/verify_pack.py` checks every exported country against the original snapshot
and verifies map coverage. `npm run test:browser` runs Playwright against
`npm run preview -- --port 1420`, blocks all external app resources, exercises the
main flows and writes screenshots and timings to `test-results/`.

`scripts/test-windows.ps1` launches the built executable in Windows, checks the
native WebView2 map, charts, comparison and modes, audits startup requests, and
captures a screenshot. It also exports one history CSV to Downloads and verifies
its contents. It closes its test process and debug connection afterwards.
Pass `-Executable` to check an installed copy. The test never disables the PC's
network connection or closes other applications.
It needs Playwright's installed JavaScript dependencies and a Windows Node runtime;
the helper can use the Node runtime in an existing VS Code installation.

The browser runner uses `/opt/google/chrome/chrome`; this is a development test
dependency, not a requirement for the Windows application. Measured timings refer
to that test environment and are not a universal performance guarantee.

MapLibre (BSD-3-Clause), React (MIT), React Flow (MIT), ECharts (Apache-2.0), Tauri
(MIT/Apache-2.0), and Lucide (ISC) are the main open-source components. See the
generated release notices for the installed dependency versions and license texts.

The application does not publish the underlying research data or give it a new
license. Its data source terms remain separate from its software dependencies.
