# ADR 0015 — Closed 2025 BoE representation-specific metadata inventory

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30G

## Context

The fixed Fed/ECB pilot proves that an event denominator and one selected text
representation can be checked before acquisition. It is intentionally not a
general manifest language, and changing its scope would invalidate its pinned
semantic and review-packet hashes.

Historical communication research also needs to distinguish representation
families. A press-conference page can link a transcript, identify a video and
say nothing verifiable about a caption track. Counting the event or the video
once for all three would overstate coverage and flatten different hosts,
publishers, transcribers, rights and fidelity questions.

No named-human rights clearance is supplied. Personal-use intent and a request
to continue without human review do not turn link discovery into acquisition
authority. This slice must therefore add only a closed denominator and link
metadata; source bytes, extraction and interpretation remain out of scope.

## Decision

### 1. Add one separate institution/year cohort

`boe_2025_mpr_press_conferences` covers calendar 2025 and only the four Bank of
England Monetary Policy Report publication dates in the official 2025 MPC
calendar:

| event key | date | official event page |
|---|---|---|
| `boe_mpr_2025_02_06` | 2025-02-06 | <https://www.bankofengland.co.uk/monetary-policy-report/2025/february-2025> |
| `boe_mpr_2025_05_08` | 2025-05-08 | <https://www.bankofengland.co.uk/monetary-policy-report/2025/may-2025> |
| `boe_mpr_2025_08_07` | 2025-08-07 | <https://www.bankofengland.co.uk/monetary-policy-report/2025/august-2025> |
| `boe_mpr_2025_11_06` | 2025-11-06 | <https://www.bankofengland.co.uk/monetary-policy-report/2025/november-2025> |

The denominator evidence is the Bank's official calendar:
<https://www.bankofengland.co.uk/news/2024/september/monetary-policy-committee-dates-for-2025>.
MPC announcements without a Monetary Policy Report, Financial Stability Report
press conferences, and research, technical or other conferences are excluded.

The existing Fed/ECB manifest, rights packet and hashes remain unchanged. The
new cohort lives in the sibling checked manifest
`data/reference/communication_boe_2025_events.json`.

### 2. Measure transcript, video and caption coverage separately

Every event carries one observation for each declared representation:

| representation | completeness basis | 2025 result |
|---|---|---:|
| official English press-conference transcript | exact first-party artifact URL | 4/4 |
| official-page video locator | external platform link or exact platform ID observed on the official page | 4/4 |
| exact English caption track | exact track URL plus artifact-specific provenance | 0/4 verified |

All four official pages link a Bank-hosted transcript PDF. February, August and
November also expose exact external-platform page URLs. The May page exposes YouTube
media ID `DAEab7yDUmE`, but its outbound video `href` was empty at the observation
clock. The manifest retains the platform ID and an `embedded_platform_id_only`
state; it does not construct or claim an exact URL.

No exact caption-track URL, caption origin, producer, language declaration or
time coverage was verified for any event. All four caption observations are
`not_verified` with no locator. This is unresolved metadata, not evidence that
captions are absent. Video coverage never fills caption coverage.

### 3. Keep platform host, publisher and transcriber provenance distinct

The Bank of England is the publisher recorded for the selected event material.
The transcript PDFs are hosted by the Bank. The observed video locators point
to YouTube, so the platform host is recorded separately and is never described
as a Bank first-party domain. Video has no transcriber. Transcript transcriber
identity remains not disclosed. Caption origin and producer remain unresolved
until an exact track is inspected under a permitted path.

The checked manifest binds both existing BoE source policies from the static
communications catalogue. Its evidence pages and direct transcript locators
must use an official allowlisted domain. External media locators have a separate
YouTube-only validation rule and the URL must resolve to the exact media ID
observed on the official page.

### 4. Make the manifest strict and the summary reproducible

`institution_year_manifest.py` validates exact JSON fields, duplicate-free
keys, one complete calendar year, event/denominator reconciliation, complete
per-event representation observations, source-policy bindings, role/material/
scope/completeness compatibility, conservative UTC clocks and safe URL/provenance
combinations. It rejects content, reviewer and rights-decision fields. The
checked BoE semantic SHA-256 is
`4bba6c8415de46718a5ae6906d0d09e9041f7be1903dbeef7be4f40223717eb2`.
The reusable contract permits a locator-free `not_verified` state for any
declared representation. An exact YouTube caption track is expressible only
with a URL bound to its video ID and language plus an artifact-specific caption
producer and classified official/automatic origin; none is asserted in this
BoE cohort.

`dalio-communication-metadata-inventory` reads only that checked manifest. It
performs no network or database I/O and writes deterministic fixed and
hash-addressed JSON/Markdown summaries. The fixed aliases are
`communication_metadata_latest.{json,md}` and the immutable pair is
`communication_metadata_2026-09-09_97c6116f1c0b94a7.{json,md}`. The inventory reports event,
observation, locator and exact-URL counts separately. Its canonical self-hash is
`97c6116f1c0b94a7d9f6b01c8bebbdff1bc13b84b0e760665b4af24b870cc230`.

### 5. Keep acquisition and interpretation closed

Every representation remains `rights_review_required` /
`manual_review_required`; `automated_collection_allowed` remains false and the
inventory states `content_capture_authorized=false`. This slice does not insert
communication rows into SQLite or archive the temporary source pages used to
observe links. It adds no transcript or caption bytes, content hashes,
extractions, speakers, claims, summaries, sentiment, scores, forecasts, causal
inference, risk cases or portfolio conclusions.

## Consequences

The Observatory gains a third central bank and a reusable one-institution/year
metadata contract without changing the 2025 Fed/ECB pilot. It can now state
precisely that BoE 2025 has a closed 4/4 event denominator, 4/4 exact transcript
links, 4/4 official-page video locators but only 3/4 exact external-platform
page URLs, and 0/4
verified exact caption tracks.

This improves historical coverage planning, not communication evidence for
analysis. The next cohort can reuse the same representation-specific method,
but each institution/year still requires a new closed denominator and fresh
first-party link observations. Content analysis remains unavailable until a
separate, explicitly permitted acquisition and fidelity-review path exists.

## Rejected alternatives

- Expand or mutate the hash-pinned Fed/ECB pilot.
- Treat all eight 2025 MPC announcements as MPR press conferences.
- Count a video locator as proof of an exact caption track.
- Invent a May YouTube URL from the observed platform ID.
- Describe YouTube as a first-party Bank of England host.
- Infer caption origin, transcriber, language, availability or time coverage.
- Download transcript, video or caption bytes while rights remain unresolved.
- Insert metadata into the live communications ledger in this slice.
- Generate summaries, signals or investment conclusions from link metadata.
