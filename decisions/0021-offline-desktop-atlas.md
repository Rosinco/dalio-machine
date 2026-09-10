# ADR 0021 — Offline desktop atlas, first viewer slice

Accepted by user instruction, 2026-09-10. The user asked to proceed with an offline
desktop map and expandable macro/sector/company interface after reviewing
MapLibre + Tauri + React/ECharts and flowchart capabilities.

Add `desktop/` inside the Dalio repository. Keep the existing canonical pipelines,
fundamentals scoring and source snapshot as the authority. The viewer uses exported
JSON, with zero database writes or source API access. No scores are recalculated by
the GUI; history is clearly latest-vintage, and no future values enter history mode.

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
