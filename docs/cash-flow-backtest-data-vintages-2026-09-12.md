# Cash-flow backtest: source vintages and frozen-input pilot

Date: 2026-09-12. Scope: read-only data audit and a separate one-origin research pilot. No application defaults, saved valuations or vendor data were changed.

The available data supports a retrospective test of the retained historical cash-flow proxy. It does **not** support a complete historical point-in-time backtest. A separate pilot freezes inputs at the actual 2025-06-21 download, but compares them with a later provider vintage whose FCF values often differ materially. That pilot measures **cross-vintage proxy agreement**; it cannot isolate forecasting error from changes in the supplied measure.

## Available sources and retention

The source repository is supplied through `--borsdata-root` or `MACRO_ATLAS_BORSDATA_ROOT`. This audit used `/mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client` without modifying it.

| Source | Annual rows | Listings with annual rows | Fiscal-year span | Missing publication dates |
| --- | ---: | ---: | --- | ---: |
| `data/raw_api/all_reports/all_yearly_reports.parquet`, frozen 2025-06-21 | 205,821 | 14,915 | 2000–2025 | 6,536 |
| `data/raw_api_snapshots/2026-08-10/all_reports/all_yearly_reports.parquet` | 251,085 | 17,828 | 2000–2026 | 15,120 |

Annual source SHA-256 identities, in table order:

```text
3ddb13ef1540d97fb06048d4466dde2d25b5c35c4335d58c5378322597cf3efc
df421594766e357df06197a63cb447315945700520d37c303b9210dfb3d4f84c
```

The source snapshot registry, ADR 0063 and `fetch_meta.yaml` identify those two downloads. The fresh batch endpoint has an approximately 20-annual/40-quarter history cap. Older rows surviving only in the frozen archive can therefore reflect endpoint depth rather than company failure. Original company-report PDFs are available for selected researched companies, but do not form a complete versioned universe.

The active desktop pack is `desktop/financial-data/1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a.sqlite`, as of 2026-08-10. Its tables are:

```sql
CREATE TABLE metadata (key TEXT PRIMARY KEY, payload BLOB NOT NULL);
CREATE TABLE companies (
  id TEXT PRIMARY KEY, payload BLOB NOT NULL, sha256 TEXT NOT NULL
);
SELECT payload FROM metadata WHERE key = 'index';
SELECT payload FROM companies WHERE id = '696';
```

`payload` is gzip-compressed JSON, so plain SQL JSON functions require decompression first. In Python, use `sqlite3.connect('file:...sqlite?mode=ro', uri=True)`, then `json.loads(gzip.decompress(blob))`. A company payload contains `annual`, `quarterly`, `withheld` and `market`. Annual arrays contain `[year, period, start, end, published, currency, currency_ratio, source_id, ...index.columns]`.

The pack has 19,140 listings, 268,884 annual rows, 603,720 quarterly rows and 1,174 withheld rows. Annual rows comprise 22,907 from the frozen download and 245,977 from the fresh download; 11,544 annual rows lack a publication date. [The exporter](../desktop/scripts/export_financials.py) sorts downloads by `source_as_of` and keeps the last `ins_id/year/period` row. It reports **538,838 superseded annual and quarterly rows**. The pack therefore contains one retained version per fiscal key, not a version ledger. Publication timestamps attached to later-revised numbers cannot recreate the earlier information set.

The older local pack `a0c53dad1a0a726ee955d1e3a47f1d011b272b610e22652cef30e1bdc1c6d005.sqlite` uses format version 1 and the same four annual/quarterly source hashes. It provides no additional vendor download vintage. The inspected paired canonical workspace does not supply another company-financial vintage.

## Measurement differences are material

There are 182,888 overlapping annual fiscal keys across the raw downloads. Of 182,100 with the same native reporting currency and finite converted FCF:

- 160,845 have changed FCF; 155,149 differ by more than 1% of the larger absolute amount, and 92,375 by more than 20%.
- 19,833 change sign.
- **146,682 change FCF while native operating cash flow and net investing cash flow are unchanged.** In 136,340, revenue is also unchanged.
- Across all matched fiscal keys, 37,702 saved conversion ratios and 108,166 period/publication-date records differ.

Native amounts are recovered separately within each download as `stored amount / that row's positive currency_ratio`; these comparisons do not mix raw quote-currency amounts with reporting-currency amounts. Numeric equality uses absolute tolerance 0.000001 million and relative tolerance 0.00000001. Materiality thresholds use the larger absolute old/new amount, with a 0.000001-million minimum.

These facts suggest investigating a systematic provider definition or calculation change, alongside corrections and restatements. **The cause remains unconfirmed.** The current source client maps the API field `free_Cash_Flow` directly to `free_cash_flow`; the snapshot fetcher stores its model output. No verified accounting reconciliation establishes why the field changed across the universe.

FCF is not universally operating cash plus net investing cash in either archive: that equality holds within tolerance for 57,598 of 205,587 complete frozen rows, and 48,754 of 250,862 complete fresh rows. Do not infer missing FCF from that sum or silently substitute another definition. Stora, SCA and ABB's 2022–2024 FCF values match across the two downloads, although some fiscal-end metadata shifts by one day; measurement stability is heterogeneous.

**Correction, 2026-09-13:** The equality counts in the preceding paragraph are retained as historical text and **superseded** by the [cash-component audit](cash-component-audit-2026-09-13.md). Independent row-level and direct-parquet NumPy calculations reproduce **43,293 matches among 205,586 complete frozen rows with valid native currency**, and **37,290 among 250,862 fresh rows**, at the stated tolerance. Frozen listing 87, FY2024, has currency label `0`; excluding that invalid label explains the one-row denominator difference. The earlier matching counts were not reproduced, and their discrepancy remains unexplained. This documentation correction does not establish why the provider's FCF values changed between downloads.

For the retrospective experiment, disclose same-source and mixed-source folds separately. Source-ID equality reduces a known source of contamination; it proves neither constant provider definitions nor historical availability. Do not choose exclusions according to whether they improve coverage.

## Universe and attrition

The frozen and fresh instrument directories contain 15,886 and 17,830 instruments. Applying Macro Atlas's existing included types `[0, 1, 3, 8, 9, 10]` yields 15,646 and 17,593 company listings, whose union is 19,140. There are 1,547 departed company-listing IDs and 3,494 added IDs. The corresponding all-instrument departures table has 1,551 rows at `data/complementing/survivorship/observed_delistings_2026-08-10.parquet`.

These directories contain listing dates and identities, but no complete effective-delisting date, delisting reason, payout or historical membership panel. An absent listing may have migrated, merged, changed identity or ceased trading; absence alone does not establish bankruptcy. The source documentation explicitly acknowledges survivorship in the older historical universe. The union preserves observed departures between the two downloads but cannot restore companies already missing in June 2025.

Listings are not independent issuers. The frozen company directory contains 10,833 distinct non-null ISIN strings; multiple venues and receipts can repeat a claim, while different share classes may have different ISINs. Neither listing count nor ISIN count is a verified independent-company sample size.

## Frozen-input pilot protocol and results

The pilot retains **every one of the 15,646 frozen company listings** in its output ledger, including failures. It needs five consecutive full annual periods, 330–400 inclusive days each, ending and published by 2025-06-21. Adjacent periods must have a 1–35-day start-minus-prior-end gap; a shared boundary day is an overlap. The latest annual end must be no more than 550 days old. Missing publication/cash/currency inputs, placeholder statements and gaps are not filled with zeros or skipped. All training years use one native reporting currency and their own positive saved conversion ratios.

The fit uses weights 30/25/20/15/10, newest first, and fiscal-year offsets 0/−1/−2/−3/−4. The next-year forecast is the weighted least-squares intercept plus slope; the requested range is `mid ± 10% × abs(mid)`. No market price is required.

The target is the immediately following annual fiscal label in the 2026-08-10 snapshot. Its period and publication must be after the origin and no later than the outcome snapshot. It must have the same native currency, a full nonoverlapping period and usable cash. Changed ISINs are withheld for identity review. Missing outcomes remain unscored. A target annual period can already be partly elapsed at the June origin; this predicts its **full annual reported proxy**, not cash earned entirely after the origin.

| Cohort | Eligible five-year fits | Evaluated targets | Inside range | Below / above | Observed coverage |
| --- | ---: | ---: | ---: | ---: | ---: |
| FY2024 origin → FY2025 target | 11,407 | 9,741 | 1,115 | 4,333 / 4,293 | **11.4465%** |
| FY2025 origin → FY2026 target | 472 | 409 | 61 | 155 / 193 | 14.9144% |
| All scored cohorts | 12,403 | 10,150 | 1,176 | 4,488 / 4,486 | 11.5862% |

The last row's fit count also includes 524 FY2023-origin fits whose next-year targets are not scored; its evaluated rows are the two forward cohorts above. The main FY2024 cohort retains 1,666 non-evaluated fits: 1,019 targets are absent, 503 are not forward of the June origin, and the remainder fail identity, currency, period, publication or cash checks. Among its eligible fits, **916 listings are absent from the later directory**, and none supplies a scored later outcome. Complete-case coverage therefore remains subject to attrition.

The intervals were explicit sensitivity bands, not calibrated confidence intervals. Their observed coverage is low, but the cross-vintage differences prevent attributing all misses to the forecasting rule. The experiment was designed in September 2026, after outcomes existed; freezing input files does not make it a pre-registered historical decision experiment.

Selected native-million examples have unchanged five-year training FCF across vintages:

| Listing | Currency | Forecast mid | Range | Later saved target | Result |
| --- | --- | ---: | --- | ---: | --- |
| ABB, 3 | USD | 2,996.40 | 2,696.76–3,296.04 | 3,080 | Inside |
| SCA B, 197 | SEK | 1,638.26 | 1,474.43–1,802.08 | 1,412 | Below |
| Stora Enso R, 696 | EUR | −654.09 | −719.49 to −588.68 | 705 | Above, sign reversal |

These are numeric diagnostics, not verified causal explanations. The source proxy can include acquisitions/disposals, working-capital changes or uneven investment, and can omit lease/financing adjustments. Original-report reconciliation is needed before attributing an individual miss to those causes or treating the proxy as shareholder cash.

## Reproduction and retained artifacts

[The generator](../desktop/scripts/frozen-vintage-cash-pilot.py) reads parquet and SQLite only. It rejects output directories inside the source repository. It accepts alternative input/outcome folders and dates; filenames use the input year. [The independent validator](../desktop/scripts/validate-frozen-vintage-cash-pilot.mjs) recomputes the Python normal-equation fit using centered covariance in Node, checks signed bands, currency conversion, dates, missing outcomes and all summary counts.

```bash
source /home/rosinco/workspace/dalio-machine/.venv/bin/activate
export MACRO_ATLAS_BORSDATA_ROOT=/mnt/c/Users/Adamb/borsdata_project/GitClone/Modern-Borsdata-Client
python desktop/scripts/frozen-vintage-cash-pilot.py \
  --output desktop/test-results/cash-flow-backtest-2026-09-12
node desktop/scripts/validate-frozen-vintage-cash-pilot.mjs \
  --output desktop/test-results/cash-flow-backtest-2026-09-12
```

The output directory contains `source-vintage-audit.json`, `source-vintage-change-diagnostics.json`, `frozen-2025-pilot-summary.json`, the complete `frozen-2025-pilot-companies.json` and `.csv` ledgers, and `frozen-2025-pilot-validation.json`. Source hashes and generator identity accompany the audit; the validator hashes its inputs and its own code. JSON uses null for unavailable values. CSV encodes history/diagnostic arrays as JSON cells.

The current-vintage rolling experiment and learned-range calibration remain separate artifacts. Their overlapping training windows, repeated targets, share classes and short held-out time span prevent treating fold counts as independent observations. Preserve the target-year holdout and calibration-information cutoff; do not tune on the inspected test results and then call those same results a fresh test.

For stronger future evidence, retain each new raw download and first-seen report version, reconcile the cash measure, follow frozen-cohort departures, and evaluate a locked specification on genuinely later outcomes. Better measured coverage alone does not validate equity-distributable cash, a DCF or an investment strategy.
