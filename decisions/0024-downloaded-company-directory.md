# ADR 0024 — All downloaded company listings

Accepted 2026-09-10 following the user's instruction to assign all downloaded
Börsdata companies to their existing countries, sectors and branches.

## Scope

Atlas 0.5 imports identity/classification metadata from all saved instrument
snapshots, including the immutable 2025-06-21 baseline. The latest record for each
instrument ID wins. Records absent from the newest download remain accessible,
labelled with their older source date; absence does not prove delisting. Company
instrument types 0/1/3/8/9/10 are retained, including alternate shares, preference
shares, depositary receipts and other company listings. Index, FX, commodity and
crypto records stay outside the company directory; source counts reconcile both
sets. Instrument IDs remain distinct; names/ISINs are not issuer deduplication keys.

Country mappings use saved provider country IDs/names and an explicit ISO/display
crosswalk. The provider's country locates the listing, not corporate domicile,
operating exposure or assets. Unknown metadata remains visible. A supplied branch
places a listing under that branch's parent in the navigation tree. Conflicting
source sector IDs are retained and flagged, never silently rewritten.

The directory has bounded search results and paginated company lists, country and
snapshot-presence filters, map counts, and an identity panel for every listing.
Financial profiles and archived research retain separate coverage; the existing
five-company report/peer slice remains available. No bulk financial-table load,
company assessment, new deep dive, forecast or asset mapping is part of this slice.

## Format and corrections

Taxonomy document v2 adds the company catalogue and complete source/effective
classification records. Its existing v3 package hash covers the whole directory;
no package identity change is required. Taxonomy v1 and older packages remain
supported. Business research remains a separately dated, hash-bound document.
Reviewed corrections apply to any included listing and preserve the original IDs,
reason, evidence and review date. Changed source assignments retain the correction
for review; unclassified rows are not assigned a guessed branch or country.

## Acceptance

- Every downloaded company instrument reconciles to an accessible directory row.
- The newest metadata for an ID wins; older-only and separate listings remain clear.
- All branches and listing countries have accurate, filter-aware company counts.
- Selecting a directory entry opens its own identity; rich profiles keep their charts.
- Source conflicts, missing categories and missing country mappings remain explicit.
- Browser/native package validation, offline use, import/export, and restart pass.
- Build/install/open the Windows update and retain both source projects unchanged.

## Verified 0.5.0 release

- 19,140 listings across 10 sectors, 94 populated branches and 19 listing countries;
  17,593 from 2026-08-10, 1,547 retained only from 2025-06-21. All company countries
  and branches map; 11 original sector conflicts remain flagged. Latest snapshot:
  17,830 instruments = 17,593 company listings + 237 other instruments. Baseline:
  15,886 = 15,646 + 240, before choosing each ID's newest record.
- Four metadata source files still match the saved hashes. Financial history stays
  at five profiles, 100 annual and 200 quarterly reports; source projects are read-only.
- 1,227 Python tests, 37 frontend unit tests and 14 Rust archive tests passed.
  Browser and Windows WebView2 flows passed with zero external requests/runtime
  errors, bounded search and pagination, source-conflict flags, identity persistence,
  v1/v2/v3 imports, byte-exact archive restart and portable/CSV exports.
- Current v3 package ID:
  `c819eeaee53ef6725c3b7c280021a6a93d8e74ecdcbd4eb6d76374a85e3d1559`.
  Taxonomy v2 SHA-256:
  `cc54a95110c5ab068434fdab8a548082ee26b48b67fbe2cde5ec330b578b37ff`.
- Tested Windows executable SHA-256:
  `a735559bd76b5bdeab64080cb2653d41729aa4f55ec26c03135530600d65fc1d`.
- Installed and opened the tested executable at
  `C:\Users\Adamb\AppData\Local\MacroAtlas\0.5.0\Macro Atlas.exe`;
  `C:\Users\Adamb\OneDrive\Desktop\Macro Atlas.lnk` points to this version.
  The installed binary and Downloads research copy match their tested build/bundle
  byte for byte. Previous app versions remain available.
