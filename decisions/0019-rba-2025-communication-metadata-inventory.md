# ADR 0019 — Checked 2025 RBA communication metadata inventory

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30J

## Context

ADR 0017 bounds the first remaining Version 1 communications package to the
2025 Riksbank and Reserve Bank of Australia central-bank cohorts. ADR 0018
completed the Riksbank half as metadata only. RBA must follow the same strict
boundary: close a first-party event denominator, record each representation's
locator and provenance separately, and do not infer content availability or
rights from another representation.

RBA media-conference pages publish the English transcript inline and link to
externally hosted video. An exact transcript-page URL is therefore not a
transcript byte capture, an officially linked video is not an RBA-hosted media
artifact, and neither establishes an exact caption track.

## Decision

### 1. Close the official eight-event 2025 denominator

`rba_2025_monetary_policy_media_conferences` contains every media conference
listed on the RBA's official 2025 monetary-policy media-conference index:

| event key | date |
|---|---|
| `rba_mp_2025_02_18` | 2025-02-18 |
| `rba_mp_2025_04_01` | 2025-04-01 |
| `rba_mp_2025_05_20` | 2025-05-20 |
| `rba_mp_2025_07_08` | 2025-07-08 |
| `rba_mp_2025_08_12` | 2025-08-12 |
| `rba_mp_2025_09_30` | 2025-09-30 |
| `rba_mp_2025_11_04` | 2025-11-04 |
| `rba_mp_2025_12_09` | 2025-12-09 |

The first-party denominator is
<https://www.rba.gov.au/monetary-policy/media-conferences/2025/>. Monetary
policy decision releases, Board meeting minutes, Statements on Monetary Policy,
unrelated speeches, parliamentary testimony and podcasts are excluded. The
checked input is `data/reference/communication_rba_2025_events.json`.

No slide representation is included or claimed for this cohort. RBA MP3 audio
files are explicitly outside Version 1, as is acquisition of all external
video and caption content.

### 2. Measure transcript pages, external video and captions independently

Every event records one observation for each declared representation:

| representation | completeness basis | 2025 result |
|---|---|---:|
| official English transcript page | exact first-party inline-HTML page URL | 8/8 |
| officially linked external video | exact external-platform page URL observed on the official page | 8/8 |
| exact English caption track | exact track URL and artifact-specific provenance | 0/8 verified |

The transcript locator is the exact `rba.gov.au` event page on which the RBA
publishes the transcript inline. Its host and publisher are the Reserve Bank of
Australia, while transcriber identity is not disclosed. The manifest retains
no HTML or transcript bytes.

Each same first-party page links one exact `youtu.be` video landing URL. The
RBA remains the media publisher, but YouTube is recorded separately as the
external platform host. The observed URL and platform media ID are locator
metadata only; no video bytes are acquired and external hosting is not called
first-party.

No exact caption-track URL, caption producer, origin, language declaration or
time coverage was verified for any event. All eight caption observations remain
`not_verified` with no locator. Video availability never fills caption
coverage.

### 3. Append source policies while preserving all earlier snapshots

The current catalogue contains 24 source policies for 19 organizations: five
central banks, seven major banks and seven commodity companies. Two new
event-specific RBA policies separate media-conference transcript/video
discovery from caption discovery; the pre-existing RBA speech policy remains
distinct.

The current catalogue SHA-256 is
`4553747ae280a669d2652c413341f0b55740419dd6ebfefd5af11bf1a55ce0b7`.
The original
`67d7049f1c63648c7b2d99dfee9eab290e2aca6469e9b872d0c605daaf716dc6`
and Riksbank-era
`ad499a7129238d70be8c7243c62252b3a1f11628a69d2df4ecf14c41d33ef1b0`
snapshots remain registered and exact-validated, so prior Fed/ECB, BoE and
Riksbank records retain their own source-policy semantics and clocks.

The RBA checked loader requires the exact manifest ID, current catalogue
snapshot, first-party denominator and evidence pages, eight events, all three
representation observations per event, compatible locators/provenance and the
pinned semantic hash. Unknown, modified, cross-cohort or catalogue-mismatched
inputs fail closed.

### 4. Publish RBA through its checked cohort registration

The offline multi-cohort publisher selects RBA only with:

```bash
dalio-communication-metadata-inventory \
  --cohort rba_2025_monetary_policy_media_conferences
```

It refreshes only
`communication_metadata_rba_2025_monetary_policy_media_conferences_latest.{json,md}`.
The unqualified BoE aliases and the cohort-qualified Riksbank aliases retain
their existing bytes. All cohorts use generic immutable, hash-bound names of
the form
`communication_metadata_YYYY-MM-DD_<inventory-hash-prefix>.{json,md}`.

The checked RBA semantic manifest SHA-256 is
`00d6016fdf83288ceb28483760f1adac6415a9d18eea3e45bc5a21278f85aeab`;
the checked manifest file SHA-256 is
`2d7d681eb986618a972cc6a3567ef098fa417276133ab7cb66318b47bb60655c`.
The inventory self-hash is
`82d5024c65e927e8007e970d9eaede1e25a194b31caa314a9927b77a01ab1309`,
so its immutable pair is
`communication_metadata_2026-09-09_82d5024c65e927e8.{json,md}`. The generated
JSON and Markdown file SHA-256 values are respectively
`9028b5c26f720139869360aa1ab158035ab64786f6b4c2fce9f621d841bab4c9` and
`913233696151c9b26e6dfd173aa6a74ed52d6d850111152c220943dab18b0490`.

### 5. Keep rights, content, persistence and analysis closed

Every representation remains `rights_review_required` /
`manual_review_required`; automation is false and
`content_capture_authorized=false`. The publisher reads only the selected
checked manifest and performs no network or database I/O.

This slice downloads or stores no transcript page, video, caption, MP3 or other
source content and inserts no communication event, artifact, scope, capture,
extraction or segment into SQLite. It creates no human rights decision,
transcription, summary, claim, score, scenario or portfolio conclusion.

## Consequences

The bounded 2025 Riksbank-plus-RBA central-bank communications package is now
complete at 16/16 denominator events. Its results remain
representation-specific metadata: Riksbank contributes replay-page and slide
locators with transcripts/captions unresolved, while RBA contributes inline
transcript-page and external-video locators with captions unresolved.

The multi-cohort publisher now protects three independent latest namespaces
without changing BoE's default. Under the Version 1 completion gate, the next
bounded collection package is core-economy debt maturity and refinancing
structure, not communication-content acquisition.

## Rejected alternatives

- Treat inline transcript HTML as captured or analysis-ready text.
- Describe YouTube as a first-party RBA host.
- Count an officially linked video as evidence of an exact caption track.
- Add a slide representation without checked first-party evidence.
- Expand Version 1 to RBA MP3 audio or external media acquisition.
- Move the BoE or Riksbank latest aliases while publishing RBA.
- Reinterpret older manifests through the new current catalogue snapshot.
- Insert metadata into the live communications ledger in this slice.
- Generate conclusions or investor guidance from link availability.
