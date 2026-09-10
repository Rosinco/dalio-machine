# ADR 0025 — Financial histories across the company directory

Accepted 2026-09-10: the user approved the coverage audit, general company financial
pages and a storage approach supporting offline selection of individual companies.

## Scope

- Read saved annual and quarterly report files through Börsdata's validated reader.
  Join instrument IDs to the complete Atlas directory. No live API or source writes.
- Select the newest saved row per instrument/fiscal-year/period, retaining baseline
  periods that are absent from the newer download. Never replace an invalid newer
  row with a silently older value. Keep each report's snapshot/source reference.
- Quarantine invalid period/currency/publication metadata with visible counts and
  reasons. Missing conversion produces missing financial values; zero remains zero.
- Recover report-currency millions by dividing stored amounts by currency_ratio.
  Income statement, balance sheet and cash-flow tables use the available source
  lines. No valuation, normalized earnings, automated investment score or forecast.
- Report coverage, latest periods, gaps and source vintages remain separate from
  saved deep-dive research. Existing archived five-company peer/research views stay.
- Test, build, install and open the Windows update through the desktop shortcut.

## Storage and compatibility

Financial histories live in an immutable SQLite companion pack, indexed by listing
ID with one compressed JSON record per listing. Queries read only the selected
record. Python writes this derived file; Rust's bundled SQLite reads it in Windows.
The Vite development preview uses Node's SQLite reader for the same file and data.
No Python, terminal or online service is required by the installed application.

DuckDB 1.5.2 was evaluated against the existing single-row-group report layout;
five annual-file point lookups took about 561 ms in this environment. SQLite's
indexed company lookup fits this release's access pattern and has a straightforward
bundled Windows build. Large analytical scans and cross-company SQL remain future
work; this decision does not change Börsdata's own storage or query engine.

Each pack has a whole-file SHA-256 identity, company payload checksums, source file
hashes and an exact taxonomy-document binding. It is attached only to that matching
research release. Existing v1/v2/v3 research packages and IDs remain unchanged;
unrelated or older releases cannot inherit a newer financial pack. The research
JSON and companion pack are exported separately, with this distinction visible
in the Library. Native pack import is bounded, chunked, validated and immutable.

## Acceptance

- Reconcile every source row into overlap, non-company, usable or withheld counts.
- Every directory listing shows accurate financial coverage, including no reports.
- Financial dates, currencies, zeroes, missingness and source changes remain explicit.
- Company tables/charts load by ID and persist after restart, with bounded memory.
- Corrupt/mismatched packs fail without changing an existing usable data file.
- Browser and Windows test identity, history, all statements and legacy research;
  native tests additionally cover companion export/import and process restart.

## Saved export and verification checkpoint

The two downloads contribute 1,426,836 report rows: 14,220 belong to instrument
IDs outside the company directory, 538,838 are superseded older periods, 872,604
are usable (268,884 annual / 603,720 quarterly), and 1,174 are withheld. Coverage
reconciles to all 19,140 listing IDs: 18,943 with reports and 197 without.
The pack is 96,309,248 bytes, with SHA-256
`a0c53dad1a0a726ee955d1e3a47f1d011b272b610e22652cef30e1bdc1c6d005`.
It binds to taxonomy SHA-256
`cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff`.
The research package remains
`c819eeaee53ef6725c3b7c280021a6a93d8e74ecdcbd4eb6d76374a85e3d1559`.
All four source Parquet files still match the export's provenance hashes.

Validation: 1,206 core Python tests, 32 desktop Python tests, 43 frontend tests,
14 research-store Rust tests and four financial-store Rust tests. The real pack
passes every record's checksum, source, fiscal-period and coverage checks.
Browser regression checks pass without external resources or runtime errors;
measured company selections were 124–181 ms in the complete run. These timings
include local UI/search/rendering on this machine, not a universal guarantee.

The Windows build bundles SQLite and uses LLVM's case-insensitive virtual filesystem
for extracted Windows SDK headers. See the [LLVM VFS format](https://llvm.org/doxygen/VirtualFileSystem_8h_source.html)
and [Clang driver options](https://clang.llvm.org/docs/UsersManual.html#clang-cl).
Native tests copy the pack into isolated local Windows storage before launching,
matching the installed app's access pattern; WSL filesystem latency is excluded.

Native Windows verification also passed: company histories/statements, exact
96 MB export and chunked import, rejection of malformed input, and a full process
restart with the imported pack. The app requested no external resources and
reported no JavaScript runtime errors. The built executable SHA-256 is
`7946f3601269cf2f9987041113fa1d70000cc653818b189cfc0d992cf174ef62`.

Installed and opened from `%LOCALAPPDATA%\MacroAtlas\0.6.0\Macro Atlas.exe`;
the existing OneDrive desktop shortcut now targets this version. The installed
binary and companion were rehashed against the tested originals. Older version
folders remain available.
