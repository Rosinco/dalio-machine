# ADR 0021 — Offline desktop atlas, first viewer slice

Accepted by user instruction, 2026-09-10. The user asked to proceed with an offline
desktop map and expandable macro/sector/company interface after reviewing
MapLibre + Tauri + React/ECharts and flowchart capabilities.

Add `desktop/` inside the Dalio repository. Keep the existing canonical pipelines,
fundamentals scoring and source snapshot as the authority. The viewer uses exported
JSON, with zero database writes or source API access. No scores are recalculated by
the GUI; history is clearly tied to the selected vintage, and no future values enter history mode.

The first slice provides world selection, Sweden and the existing country panel,
fundamentals/history/trade views, radar/line/pie/bar charts, rule diagrams, source
metadata and CSV exports. Sector/branch analyses, Börsdata company data, DuckDB
queries, ownership polygons and new forecasts remain future work. These are
explicitly described as future layers in the product, not populated with examples.

Natural Earth country outlines are bundled and recorded with provenance. All code,
styles, map assets and research files are local. The content security policy permits
local assets and Tauri IPC only. A portable Windows executable embeds the release;
the installed Windows WebView2 runtime supplies the rendering engine.

This adds a presentation target to ADR 0001's snapshot coupling, without duplicating
the data/reasoning engine. No API credentials, raw Börsdata archives or source
databases are packaged. Detailed implementation and refresh instructions are in
`desktop/README.md`.

## Second slice: local research library and evidence drilldowns

Authorized by the user's “continue” after the v0.2 proposal, 2026-09-10.
Research can now be imported without reinstalling the viewer. A bounded, versioned
package preserves exact source JSON and checksums; the native archive is immutable
and independent of app version directories. Earlier releases remain selectable.
The included latest and previous fundamentals vintages are 2026-09-08 and 2026-08-24;
the latest package also includes the saved `liquidity-diagnostics-v1` report.

Category drilldowns explain the existing equal-percentile weighting and compare
eligible earlier snapshots. They display the original score and flag arithmetic
discrepancies rather than silently changing it. The liquidity panel preserves
national versus currency-area scope, native-currency measures, observation dates,
known-at evidence and source ledger references. Descriptive quantities use neutral
series colours; repo stocks/transactions and MMF clearing categories are kept
separate. No new liquidity aggregate, risk score or causal cash-flow diagram is made.

The app remains a consumer of Dalio exports. Source ETL, sector/company layers,
reviewed industry forecasts and large Börsdata queries remain separate future work.
