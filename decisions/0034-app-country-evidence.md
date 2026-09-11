# ADR 0034: Independently dated country evidence in Macro Atlas

Date: 2026-09-11
Status: Implemented and built in desktop v0.9.0; paused before the complete native Windows suite and installation.

## Decision

Bundle verified annual country assessments and original-source monitoring in a
separate `macro-atlas-country-evidence-v1` pack. Preserve the existing research
release and financial companion contracts. Their dates, hashes, historical scores,
imports and company identities retain their existing meaning.

One reviewed 19-country assessment plus multiple reviewed monitoring snapshots
are accepted. Select the newest whole monitoring country profile by its exact
known-at cutoff; reject same-clock conflicting snapshots. Missing/ineligible
signals remain missing. Retain documentation and artifact provenance for explicit
structural gaps. Never substitute another country or a convenient proxy.

Choose the newest annual country assessment from the base and embedded monitoring
parents. Keep its original snapshot identity, cutoff, methodology and references;
monitoring scenario references remain bound to their own parent citations.
Retain original observations, missingness, status conventions and source clocks.
Projection validates source snapshot hashes, native latest points and country
citations; frontend validation checks index/file hashes and evidence identity.

The atomic index references immutable per-country files. Original input snapshots
are read-only. The app loads the small catalogue first and country files on demand,
with a five-country cache. Values, locators and provenance are available offline;
raw response files remain in the macro archive. Source URLs are copyable without
initiating network requests. This first version refreshes through an app rebuild,
not the existing research import interface.

## Presentation

Macro country search includes all assessment countries, including those absent
from the older scoring panel. `GB` listing-country aliases resolve to macro `UK`.
The Assessments sidebar shows neutral current signal cards, exact comparison
windows, history charts, dated IMF paths, structural reference years, conditional
scenarios and source details. Separate UTC dates remain visible when changing an
older fundamentals release. Unscored countries receive no inferred map score.

Chart gaps remain gaps; forecasts are separate dashed series using the retained
collector status convention. Monitoring does not produce good/bad colour ratings,
country rankings, probabilities, GDP nowcasts or company verdicts. Industry/rate
scopes remain explicit; lower rates alone do not establish easier credit access.
National central-government debt measures retain scope, units and observed versus
forecast status; refixing and principal maturity are distinct. Listing country
does not establish revenue, asset, cost or financing exposure.

## Verification

Focused Python tests cover source integrity, original clocks, embedded parents,
whole-country selection, structural gaps and immutable repeat exports. TypeScript
checks cover missing/zero/forecast series, safe URLs, all bundled country files
and corruption rejection. Shared browser/native cases exercise actual values and
windows, Finland/Norway/Belgium access, GB/UK identity, research-release independence
and compact layouts. Existing research, taxonomy, financial and comparison tests
remain required before packaging.
