# ADR 0013 — Closed 2025 Fed–ECB communications metadata pilot

**Date:** 2026-09-09 · **Status:** accepted · **Slice:** 30E

## Context

ADR 0012 established a rights-gated communications ledger, but its source
catalogue is discovery policy rather than a claim that any archive has been
enumerated, reviewed or collected. Before building a broad communications
corpus, the project needs one small pilot that proves it can define a closed
event universe, distinguish event coverage from artifact availability and put
an exact representation through a named human rights review.

The Federal Reserve and ECB are a useful paired test. Both publish a regular
monetary-policy press-conference history, but their selected text forms differ.
The Federal Reserve publishes a final official PDF transcript. The ECB publishes
one official HTML page containing both its prepared monetary-policy statement
and the ensuing questions and answers. Their video players and caption paths
also introduce separate hosts, formats and rights questions that are unnecessary
for the first text-only gate.

## Decision

### 1. Freeze one calendar-year pilot and two closed denominators

The pilot identity is `fed_ecb_2025_policy_press_conferences`. Its reference
window is 2025-01-01 through 2025-12-31 inclusive, and its evidence is recorded
as known on 2026-09-09. A materialized manifest must use an exact UTC
`as_known_at`; the ADR date is not converted into an invented midnight clock.

The denominator is defined independently for each organization:

| organization_id | source_id | inclusion rule | official denominator evidence | expected events |
|---|---|---|---|---:|
| `federal_reserve` | `fed_fomc_press_conferences_en` | Every regular 2025 FOMC meeting for which the official calendar links a press conference | <https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm> | 8 |
| `ecb` | `ecb_monetary_policy_press_conferences_en` | Every regular 2025 Governing Council monetary-policy decision press conference listed in the official webcast archive | <https://www.ecb.europa.eu/press/tvservices/webcast/html/index.en.html> | 8 |

The Federal Reserve's contemporary schedule notice corroborates the scheduled
meeting dates and post-meeting news-conference convention:
<https://www.federalreserve.gov/newsevents/pressreleases/monetary20240809a.htm>.
The ECB statement and decision archives are independent cross-checks:
<https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/index.en.html>
and <https://www.ecb.europa.eu/press/govcdec/mopo/html/index.en.html>.

The separate ECB strategy-assessment press conference on 2025-06-30 is outside
the denominator because it was not one of the regular decision press
conferences:
<https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/html/strategy-assessment-press-conference.en.html>.
FOMC minutes, delayed meeting transcripts, ECB monetary-policy accounts,
speeches and other press conferences are also outside this pilot.

Denominator count and representation completeness are different measures. The
denominator asks how many in-scope events occurred. Completeness asks whether
each of those events has the one required representation candidate described
below. A discovered URL makes the metadata inventory 16/16; it does not make a
corpus complete, rights-cleared, archived, ingested or analysis-ready.

### 2. Require exactly one first-party English text candidate per event

Each organization has one singular `representation_spec`, and every event has
one singular `representation` and `candidate` matching it:

| organization_id | representation_key | required form | MIME type | provenance | language | section_coverage |
|---|---|---|---|---|---|---|
| `federal_reserve` | `official_transcript_en` | Final official press-conference transcript | `application/pdf` | `official_published_transcript` | `en` | `full_transcript` |
| `ecb` | `official_statement_with_q_and_a_en` | Official monetary-policy statement with Q&A | `text/html` | `official_published_transcript` | `en` | `prepared_remarks`, `q_and_a` |

The event keys are stable lowercase snake case. The 16 exact candidates are:

#### Federal Reserve

| event_key | event date | exact official event page | exact final transcript candidate |
|---|---|---|---|
| `fomc_2025_01_29` | 2025-01-29 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250129.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250129.pdf> |
| `fomc_2025_03_19` | 2025-03-19 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250319.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250319.pdf> |
| `fomc_2025_05_07` | 2025-05-07 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250507.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250507.pdf> |
| `fomc_2025_06_18` | 2025-06-18 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250618.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250618.pdf> |
| `fomc_2025_07_30` | 2025-07-30 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250730.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250730.pdf> |
| `fomc_2025_09_17` | 2025-09-17 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20250917.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20250917.pdf> |
| `fomc_2025_10_29` | 2025-10-29 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20251029.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20251029.pdf> |
| `fomc_2025_12_10` | 2025-12-10 | <https://www.federalreserve.gov/monetarypolicy/fomcpresconf20251210.htm> | <https://www.federalreserve.gov/mediacenter/files/FOMCpresconf20251210.pdf> |

Each PDF identifies itself as the final transcript of Chair Powell's press
conference. The event pages remain locator/status metadata and are not a second
required text representation.

#### European Central Bank

| event_key | event date and location | official source title | candidate URL |
|---|---|---|---|
| `ecb_2025_01_30` | 2025-01-30, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250130~1f418aa0f4.en.html> |
| `ecb_2025_03_06` | 2025-03-06, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250306~4307bd0941.en.html> |
| `ecb_2025_04_17` | 2025-04-17, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250417~091c625eb6.en.html> |
| `ecb_2025_06_05` | 2025-06-05, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250605~f00a36ef2b.en.html> |
| `ecb_2025_07_24` | 2025-07-24, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250724~a66e730494.en.html> |
| `ecb_2025_09_11` | 2025-09-11, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is250911~a13675b834.en.html> |
| `ecb_2025_10_30` | 2025-10-30, Florence | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is251030~4f74dde15e.en.html> |
| `ecb_2025_12_18` | 2025-12-18, Frankfurt | Monetary policy statement (with Q&A) | <https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2025/html/ecb.is251218~3a10402adb.en.html> |

The ECB HTML is one published representation with
`section_coverage = [prepared_remarks, q_and_a]`. Those sections may become
distinct reviewed segments after capture, but they are not two candidates and
must never create duplicate stored bytes. Questions also remain external-speaker
speech; inclusion in an ECB-hosted transcript does not turn them into ECB claims.
The companion ECB PDFs stop before the Q&A and therefore do not satisfy this
pilot's required coverage.

At the Slice 30E checkpoint, the manifest's `q_and_a_transcript` /
`questions_and_answers` pair was only the candidate's provisional primary
classification under the then-current one-role artifact schema. It did not
classify the prepared remarks as Q&A. ADR 0014 subsequently defines how one
artifact version and one byte-capture lineage carry separately addressable
prepared-remarks and Q&A section scopes without duplication or provenance loss.

At the metadata checkpoint all 16 representations have
`availability_status = available`. That status means only that the exact
first-party candidate was exposed by the official archive when checked; it is
not acquisition authorization and says nothing about future URL stability.

### 3. Preserve conservative point-in-time clocks and mutable-source lineage

An event date or covered meeting period is not an artifact-publication clock.
Candidate metadata keeps `published_at`, `available_at`, `retrieved_at` and
`metadata_known_at` separate; the representation keeps `checked_at`; and the
manifest keeps `as_known_at`.

- `published_at` remains null unless the publisher explicitly supplies a
  defensible artifact-publication time.
- When no defensible public-availability time is exposed, `available_at` is no
  earlier than the first exact UTC observation/retrieval time. It is never
  backdated to the event date or inferred from a PDF's printed date.
- Candidate `retrieved_at` identifies when its metadata was actually observed;
  it does not prove content bytes were saved. A future content capture records
  its own exact byte-retrieval clock.
- `metadata_known_at` records when corrected titles, attribution, rights facts
  or other metadata became known to the Observatory. It cannot precede the
  evidence used to establish it.

The word `FINAL` on a Federal Reserve PDF is publisher status, not a guarantee
that bytes at its URL can never change. ECB pages are also mutable: the
2025-12-18 page explicitly records a correction to verbatim comments. A future
authorized retrieval therefore appends a content capture with its retrieval
clock, SHA-256 and predecessor/successor lineage. It never overwrites an earlier
capture or treats a new hash at the same URL as the same bytes.

### 4. Capture rights evidence, but leave every review pending

Rights evidence is structured at the denominator/representation-policy level as
`{status, basis_url, summary, questions}`. It records what the current official
policy says and never authorizes acquisition by itself. The evidence check date
for this ADR is 2026-09-09.

| organization_id | status | basis_url | summary of current official evidence | open questions for named human review |
|---|---|---|---|---|
| `federal_reserve` | `pending` | <https://www.federalreserve.gov/disclaimer.htm> | Unless otherwise indicated, Board-site information is described as public domain and may be copied and distributed without permission; the Board asks to be cited. Non-Board material, seals and logos are exceptions. | Confirm that each exact final PDF and its transcript text is Board material with no artifact-specific exception; record permitted storage, extraction and internal use; confirm required attribution. |
| `ecb` | `pending` | <https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html> | The website is © ECB. Subject to stated exceptions, directly obtained information may be used if kept accurate, attributed to the ECB and modifications are disclosed. Each selected 2025 HTML page also says reproduction is permitted when the source is acknowledged. | Confirm that the general terms and page-level notice cover the exact statement and Q&A representation, including non-ECB questioners; record permitted storage, extraction and internal use plus attribution and modification duties. |

Supporting official policy/access evidence is retained as link metadata:

| organization_id | evidence kind | exact URL | fact recorded as of 2026-09-09 |
|---|---|---|---|
| `federal_reserve` | policy index | <https://www.federalreserve.gov/policies.htm> | Official website/privacy policy index. |
| `federal_reserve` | website and external-link policy | <https://www.federalreserve.gov/website-linking-policies.htm> | Identifies `federalreserve.gov` as the Board's primary source and warns that external sites have separate terms. |
| `federal_reserve` | robots observation | <https://www.federalreserve.gov/robots.txt> | Returned HTTP 404 when checked; absence of a robots file is not rights permission or acquisition authorization. |
| `federal_reserve` | accessibility | <https://www.federalreserve.gov/accessibility.htm> | Records the Board's accessibility policy and media/PDF context; it does not establish caption ownership or reuse rights. |
| `ecb` | site-use policy index | <https://www.ecb.europa.eu/services/using-our-site/html/index.en.html> | Official entry point for ECB site-use policies. |
| `ecb` | robots policy | <https://www.ecb.europa.eu/robots.txt> | Specifies a five-second crawl delay and disallows the legacy `/press/tvservices/webcast/shared/video/` path; ordinary published statement paths are not listed as disallowed. Robots policy still is not copyright permission. |
| `ecb` | third-party media/cookies | <https://www.ecb.europa.eu/services/data-protection/privacy-statements/html/ecb.privacy_statement_cookiepolicy.en.html> | ECB video features can depend on third-party providers, including YouTube, with their own policies and terms. |
| `ecb` | accessibility | <https://www.ecb.europa.eu/services/html/accessibility-statement.en.html> | Generally notes that some media lacks transcripts or captions; it is not event-level caption evidence. |

Before any acquisition, one named human must review each exact candidate
representation, the then-current policy and all artifact-specific notices. The
review must record reviewer identity, decision, review time, basis URL, summary,
unresolved questions and the exact manifest/representation identity. A source-
level conclusion cannot silently authorize another material type, host,
language, edition or changed candidate.

Until that review occurs, every candidate remains
`rights_review.status = pending`, the existing source policy remains
`rights_review_required` and `manual_review_required`, and automated acquisition
must fail closed.

### 5. Keep bytes, ingestion and interpretation at zero

This slice stores link and policy metadata only. Its state is explicit:

| checkpoint | count/status |
|---|---:|
| In-scope denominator events | 16 |
| Events with the required first-party candidate URL | 16 |
| Named-human representation/rights reviews | 0 |
| Candidates authorized for capture | 0 |
| Communication content bytes archived | 0 |
| Communication content captures ingested into SQLite | 0 |
| Extracted or segmented transcripts | 0 |
| Claims, sentiment labels, topic/risk scores or analysis outputs | 0 |

Neither discovery nor `availability_status = available` may trigger network
collection or a database write. A later acquisition change must consume an
approved, hash-bound review packet, preflight the complete authorized batch and
then use the immutable content-capture lineage established by ADR 0012.

### 6. Defer video, audio and subtitles as separate representations

Federal Reserve event pages embed a media player, but no first-party standalone
MP4 or VTT/SRT candidate is part of this pilot. ECB webcast links use an
ECB-hosted player wrapper with a YouTube identifier; its HD-footage page links
to Dropbox, and its robots policy disallows the legacy direct-video directory.
Those facts do not make video or captions missing from the text denominator.

Video, audio, publisher captions, automatic captions and locally generated
speech-to-text require their own representation specifications, exact media
hosts, producer/transcriber attribution, rights evidence, clocks and human
review. They remain deferred rather than being used to fill, verify or amend the
selected official text.

## Consequences

The project gains a small, reproducible 8+8 metadata universe in which every
event has one exact first-party English text candidate and no ambiguity about
what 16/16 means. The differing Fed PDF and ECB mixed-section HTML forms test
the representation model without widening the rights boundary.

The conservative gate deliberately leaves the useful content unavailable to
analysis. A named human review and a separate authorized acquisition slice are
still required before bytes, extraction, claims or investment-facing synthesis
can exist.

ADR 0014 subsequently supplied the one-artifact/one-byte-lineage, ordered multi-section storage
mapping required here. It removes the structural metadata blocker only; the
pending rights state and zero-content boundary remain unchanged.

## Rejected alternatives

- Treat the open-ended source catalogue as a complete historical denominator.
- Count every discovered landing page, PDF, video and caption as another event.
- Model the ECB statement and Q&A as two stored artifacts with duplicate bytes.
- Treat the ECB prepared-statement-only PDF as a complete Q&A transcript.
- Infer publication or availability from an event date or a PDF's printed date.
- Treat `FINAL`, an unchanged URL or a page-level title as immutable content.
- Interpret public-domain or attribution language as automated legal clearance.
- Use robots policy as copyright permission.
- Download text or media before the exact representation is reviewed by a named
  human.
- Add videos, subtitles, machine transcription, claims, sentiment, scores or
  macro/portfolio analysis to this metadata pilot.
