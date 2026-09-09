# ADR 0018 — Checked 2025 Riksbank communication metadata inventory

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30I

## Context

ADR 0017 bounds the remaining Version 1 communications work to metadata while
no named-human rights clearance exists. The next central-bank cohort therefore
needs a closed first-party event denominator and representation-specific link
observations, not downloaded media, transcripts or inferred captions.

The Bank of England inventory also began as a single-cohort command with
unqualified `latest` aliases. Reusing that command for another checked cohort
must preserve the existing BoE interface and bytes while preventing one
institution's publication from silently replacing another's latest view.

## Decision

### 1. Close the 2025 monetary-policy press-conference denominator

`riksbank_2025_monetary_policy_press_conferences` contains the eight Sveriges
Riksbank monetary-policy-decision press conferences listed in the official 2025
Riksbanken Play archive:

| event key | date |
|---|---|
| `riksbank_mpr_2025_01_29` | 2025-01-29 |
| `riksbank_mpr_2025_03_20` | 2025-03-20 |
| `riksbank_mpr_2025_05_08` | 2025-05-08 |
| `riksbank_mpr_2025_06_18` | 2025-06-18 |
| `riksbank_mpr_2025_08_20` | 2025-08-20 |
| `riksbank_mpr_2025_09_23` | 2025-09-23 |
| `riksbank_mpr_2025_11_05` | 2025-11-05 |
| `riksbank_mpr_2025_12_18` | 2025-12-18 |

The denominator evidence is the two official category-28 Riksbanken Play
listing pages for 2025, beginning at
<https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/?category=28&year=2025&page=1>.
The continuation is
<https://www.riksbank.se/sv/press-och-publicerat/riksbanken-play/?category=28&year=2025&page=2>.
The inclusion rule selects the eight entries titled as monetary-policy-decision
press conferences. It excludes the Payments Report and Financial Stability
Report press conferences, speeches and seminars. Reports, updates, minutes,
votes and releases remain representations distinct from the selected replay
pages and slides.

The checked input is
`data/reference/communication_riksbank_2025_events.json`. Its event replay pages
and the monetary-policy decision pages used as slide-link evidence are all on
the Riksbank's first-party domain.

### 2. Keep all four representation states explicit

Every event records one observation for each Swedish representation:

| representation | completeness basis | 2025 result |
|---|---|---:|
| official transcript | exact first-party artifact URL | 0/8 verified |
| official replay page | exact first-party replay-page URL | 8/8 |
| official press-conference slides | exact first-party PDF URL | 8/8 |
| exact caption track | exact track URL and provenance | 0/8 verified |

The replay locator is the Riksbanken Play HTML page, with Sveriges Riksbank as
host and publisher and `official_archive_mixed` provenance. It is not described
as a direct video artifact, and no embedded-player, `qcnl.tv`, vendor or
platform-media locator is retained.

Each slide observation points directly to the official Swedish
`bilder-fran-presstraffen` PDF. Slides are publisher-authored text with
`official_published_slides` origin and remain distinct from both a transcript
and the replay. No exact first-party Swedish transcript and no exact Swedish
caption track, caption producer, origin or time coverage was verified. Replay
or slide availability never fills either missing representation.

### 3. Extend the source-policy catalogue without rewriting prior snapshots

The current static catalogue contains 22 source policies for 19 organizations:
five central banks, seven major banks and seven commodity companies. The two
new policies separately describe the Riksbank's Swedish press-conference
material and subtitle discovery surfaces. Older Fed/ECB and BoE manifests keep
resolving their own immutable catalogue snapshots; catalogue growth does not
repin or reinterpret them.

The Riksbank checked loader requires the exact manifest ID, source-policy
bindings, first-party domains, four observations per event, compatible
representation semantics, conservative clocks and the pinned semantic hash.
Unknown, modified, cross-cohort or catalogue-mismatched inputs fail closed.

### 4. Publish through a checked multi-cohort registry

`dalio-communication-metadata-inventory` dispatches only through an immutable
registry of checked cohort IDs, default manifest paths and pinned loaders. With
no `--cohort`, it remains exactly the BoE command and continues to refresh
`communication_metadata_latest.{json,md}`. The Riksbank invocation is:

```bash
dalio-communication-metadata-inventory \
  --cohort riksbank_2025_monetary_policy_press_conferences
```

It refreshes only the cohort-qualified aliases
`communication_metadata_riksbank_2025_monetary_policy_press_conferences_latest.{json,md}`.
It cannot move the legacy BoE aliases. Both cohorts retain generic immutable,
hash-bound names of the form
`communication_metadata_YYYY-MM-DD_<inventory-hash-prefix>.{json,md}`. An
unknown cohort or a manifest whose ID does not match the selected cohort fails
before output publication.

The checked Riksbank semantic manifest SHA-256 is
`34f610043e933f74ca71ae56a9d29129fc92e2dd1ea83cc932293581ac4a2468`; the
checked manifest file SHA-256 is
`db4ecb23519f7a70e3e58563bf4b9dc895fd0ff78c1ef4ca1f44e6dc5f601b1d`.
The inventory self-hash is
`43fd289a37e28f6f8345b43d52c056cb562f87412f5f2294fcf930dda2705f3a`,
so its immutable pair is
`communication_metadata_2026-09-09_43fd289a37e28f6f.{json,md}`. The generated
JSON and Markdown file SHA-256 values are respectively
`8485f8cd23d740539837921e3699704c8675d1ab908a27a495b7ee966d07f491` and
`35d38c1d7982f603c24f3c3c11fc57c0e2024d4b3212451d37abb612137f9556`.

### 5. Keep acquisition, persistence and interpretation closed

The checked publisher is offline and database-free. It reads only the selected
manifest and writes deterministic JSON/Markdown metadata views. Every
representation remains `rights_review_required` /
`manual_review_required`, automation is false and
`content_capture_authorized=false`.

This slice stores no replay-page, slide, transcript or caption content; archives
no source-page bytes; and writes no communication events, artifacts, scopes,
captures, extractions or segments to SQLite. It adds no transcription,
translation, summary, claim, signal, scenario or portfolio conclusion.

## Consequences

The Observatory can state a complete, reproducible 8/8 Riksbank event and
four-representation metadata result without mistaking links for a corpus. The
8/8 replay-page and 8/8 slide coverage are useful acquisition-planning facts;
the 0/8 transcript and 0/8 caption results remain honest unresolved states.

The checked publisher can safely add later cohorts without changing BoE's
established default or letting unrelated `latest` aliases collide. Each future
cohort still needs its own closed denominator, first-party evidence, pinned
loader and qualified alias. Content analysis remains unavailable until an
explicitly permitted acquisition and fidelity-review path exists.

## Rejected alternatives

- Treat a Riksbanken Play page as captured video content.
- Retain or reconstruct a `qcnl.tv` or other vendor media locator.
- Count official slides as a transcript or caption substitute.
- Infer a transcript or caption track from replay availability.
- Reuse the unqualified BoE `latest` aliases for Riksbank output.
- Accept an arbitrary manifest through a generic unchecked loader.
- Download replay pages, PDFs, media or captions while rights remain pending.
- Insert link metadata into the live communications ledger in this slice.
- Generate conclusions or investor guidance from representation availability.
