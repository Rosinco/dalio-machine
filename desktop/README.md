# Macro Atlas desktop

An offline Windows viewer for the existing Dalio fundamentals snapshot. The app
bundles its map, scripts, charts and source data. Starting the packaged executable
requires no terminal, Python environment, WSL, account or network connection.
Windows WebView2 must be installed; it is present on Adam's PC. The Tauri installer
configuration can also bundle its offline installer when distributing to another PC.

## Version 0.2.0

- **Research library:** import `.atlas.json` files, select earlier releases and
  export a portable copy to Downloads. Refreshing data no longer requires a rebuild.
  Included vintages: 2026-09-08 (fundamentals + liquidity) and 2026-08-24
  (fundamentals only). Each panel follows the active release.
- **Score explanations:** select a category below the radar to see raw observations,
  percentile direction, equal effective weights, contributions and source dates.
  The half-coverage rule is preserved. Comparisons use the nearest earlier distinct
  fundamentals snapshot and require matching ranking populations and indicator
  definitions. Sweden's included category scores are unchanged between these vintages.
- **Liquidity:** national broad money and dated growth history, explicitly labelled
  euro-area context for member countries, and separate global offshore-credit,
  US MMF and US repo panels. Quantities stay blue; no aggregate liquidity/risk score
  or inferred deposit flows are introduced. Source ledger, coverage, formulas and
  known-at timestamps remain accessible.

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

The app reads the selected saved vintage. Historical charts are **not** point-in-time
replays. Source metadata in this snapshot is insufficient for a per-point release
ledger, so the UI does not invent it. The five-year GDP-growth headline and annual
growth history are explicitly distinguished. Company/sector layers and geographic
asset inventories are later slices; no company sites or forecasts are fabricated.

## Data flow and scale

The canonical Python pipelines remain authoritative. `export_research.py` wraps the
exact UTF-8 fundamentals and optional liquidity JSON with their SHA-256 hashes. Its
bundle mode also uses `export_snapshot.py`, which validates
with Dalio's existing parser and creates a compact index, individual country files,
and a history-map file. Startup loads summaries; selecting a country loads only its
history; the full small annual panel loads only in History mode. The source SQLite
database is never opened or modified by the exporter or viewer. Original data and
all statistical calculations remain unchanged.

Native imported packages live in
`%LOCALAPPDATA%\local.research.macro-atlas\research-v1\<package-id>.atlas.json`,
outside the executable's version directory. The package ID hashes the two source
hashes, so duplicate imports are idempotent. The Rust `research-store` crate checks
format, bounds, required renderable fields and checksums before publishing a complete
file with a non-overwriting hard link. It only exposes fixed resource names and
country codes. A damaged archive entry is reported without hiding usable entries.
Checksums establish file integrity, not publisher authenticity. Browser previews
use IndexedDB; those imports are separate from the installed Windows library.

The local import limit is 32 MiB per package. Native imported resources are projected
from that bounded package on demand; this release is not a large analytical database.
Included assets retain country-level lazy loading. The source ledger's referenced
provider artifact files are not copied into the package. Their contents are not
independently verified by the offline viewer.

Large Börsdata Parquet tables are not included or loaded by this release. A future
bounded query layer can supply company/peer slices using DuckDB or the existing
Python adapters. Do not load the 101-million-row screener file into the webview.

## Build and refresh

From this `desktop/` folder:

```sh
npm ci
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
python scripts/export_research.py \
  --fundamentals /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json \
  --liquidity /home/rosinco/workspace/dalio-machine/data/snapshots/liquidity_latest.json \
  --previous /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_2026-08-24.json \
  --bundle public/data
python scripts/verify_pack.py \
  --source /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json
npm test
npm run build
```

Generated research data is deliberately gitignored. The checked Natural Earth map
has its source, original checksum and processing notes in `public/maps/NOTICE.txt`.
To update research in an already installed app, export a single file and import it
through **Library → Import research file**:

```sh
python scripts/export_research.py \
  --fundamentals /home/rosinco/workspace/dalio-machine/data/snapshots/fundamentals_latest.json \
  --liquidity /home/rosinco/workspace/dalio-machine/data/snapshots/liquidity_latest.json \
  --output /tmp/Macro-Atlas-Research.atlas.json
```

Omit `--liquidity` for a fundamentals-only package. Each document retains its own
date; packaging does not make old observations newer. The exporter reads saved
files and never refreshes data from the network. Updated map geometry, new data
schemas or application features still require a new app build.

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
It also checks score arithmetic and comparison eligibility, import checksums and
validation, liquidity geography and gaps. The standalone Rust archive has filesystem
tests runnable with `cargo test --manifest-path research-store/Cargo.toml`.
`cargo run --manifest-path research-store/Cargo.toml --example verify -- FILE...`
validates real exported packages and checks exact country/liquidity round trips.
`scripts/verify_pack.py` checks every exported country against the original snapshot
and verifies map coverage. `npm run test:browser` runs Playwright against
`npm run preview -- --port 1420`, blocks all external app resources, exercises the
main flows and writes screenshots and timings to `test-results/`.

`scripts/test-windows.ps1` launches the built executable in Windows, checks the
native WebView2 map, charts, comparison and modes, audits startup requests, and
captures a screenshot. It also exports one history CSV to Downloads and verifies
its contents, imports a real older package, rejects damaged input, checks duplicate
imports, restarts the executable to verify persistence, and exports a portable
research file. It uses an isolated WebView profile and temporary native archive
(`ATLAS_RESEARCH_DIR`, a development/test override) and closes its own processes.
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
