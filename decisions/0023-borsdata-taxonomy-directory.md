# ADR 0023 — Börsdata sector and branch directory

Accepted by the user's “continue” on 2026-09-10, following the proposal to use
Börsdata's categories as the default and retain documented exceptions.

## Slice

Atlas 0.4 exposes the saved 10-sector / 94-branch hierarchy using the existing
`data/complementing/taxonomy/branch_crosswalk.csv`. IDs are keys; Swedish and English
names are display attributes (Börsdata ADR 0034). No source folders are renamed.
A searchable directory distinguishes source branch-study status, saved deep-dive
artifacts in the project, and financial profiles actually included in Atlas.
An existing folder alone never means a completed or current investment assessment.
Shared study groups are explicit and deduplicated in global inventory totals.

Country, branch and company selection remain coherent. Selecting a branch without
included company data clears that map's company coverage; it cannot show Holmen as
if it belonged to the new branch. Company selection opens its effective branch.
The full taxonomy is independent of the current five-company financial slice.

## Source preservation and corrections

`desktop/research/classification-corrections.json` is an explicit, initially empty
reviewed overlay. Each correction binds an instrument ID, expected original sector
and branch, destination branch, reason, source and review date. It does not rewrite
Börsdata files or sealed research. Export retains the original and corrected values.
A changed source assignment preserves the correction but flags it for review instead
of silently carrying it onto a different classification. A source assignment that
now agrees with the correction is recorded as aligned. Segment tags remain separate
from primary classification and do not change peer-study membership.

The directory is exported from saved files only. Branch-study and corpus status
come from the maintained crosswalk. Deep-dive counts describe discovered document
folders, not inferred approval, completion, freshness or investment quality. Prose
outside the existing business package is inventoried, not bulk imported.

## Portable packages

An optional taxonomy document uses package schema v3. Identity includes all four
document hashes; v1/v2 IDs stay unchanged and older envelopes cannot smuggle in an
unhashed taxonomy field. Taxonomy classifications bind to the exact business
document hash and are cross-checked against its company IDs and original categories.
One fixed `taxonomy` resource supplies the small directory. Old packages explicitly
report that the full hierarchy is absent and keep their existing company views.

## Acceptance

- Browse/search all sectors and branches in either source language.
- Filter research coverage without turning missing research into a negative score.
- Show source names, dates, original assignments and correction state.
- Switch branches, countries and companies without stale or misplaced profiles.
- Preserve v1/v2 packages and verify v3 imports after browser and Windows restart.
- Build, install and open the offline Windows application; keep source repos intact.

## Verification — 2026-09-10

The real saved export contains 10 sectors, 94 branches, 17 graduated branch-study
assignments and 89 unique deep-dive folders (shared studies deduplicated). Its
66,687-byte directory retains all five original company assignments; the correction
overlay is empty. No Börsdata or canonical Dalio files were modified.

- 1,222 Python tests passed, including the seven taxonomy exporter contracts; Ruff
  passes for backend code/tests and the changed desktop Python files.
- 33 frontend and 11 native archive tests passed. Browser and Windows flows cover
  bilingual/diacritic search, coverage filters, branch gaps, shared studies, a
  synthetic correction in an isolated library, and v1/v2/v3 import compatibility.
- Real package projections match saved country/liquidity/company/taxonomy resources.
  The final Windows app passed process-restart persistence and portable/CSV export
  checks with zero external requests and zero runtime errors. Its measured readiness
  was 334 ms in the local native test (not a general performance guarantee).
- Current package ID:
  `8c3e2596a8c08b07b67db4237acca16e3c9755acbe494f926af66beb4562ac40`.
  Taxonomy SHA-256:
  `6aac8c7cb80a77e476349c13fed30485447238adfffbcfe4bff2ec43da968a49`.
  Tested Windows executable SHA-256:
  `94b40a586a5baefbb561d24ca2e0f71afbb1e51297fbcb2e9aec3a4643eca19a`.

The five financial profiles and archived peer cohort are unchanged. Directory
coverage does not bulk-import the other studies, add forecasts, or map physical assets.

Installed and opened at `%LOCALAPPDATA%\MacroAtlas\0.4.0\Macro Atlas.exe`, with
the existing OneDrive Desktop shortcut updated. The installed binary matches the
tested checksum; the portable Downloads copy matches the package byte-for-byte.
