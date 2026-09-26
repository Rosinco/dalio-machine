# Terminal-stress candidate review — 2026-09-13

The first issuer checks weaken three apparent valuation discounts and identify a fourth company as a transaction case. Ipsos is the first proposed business-research priority; SFPI merits a conditional balance-sheet and recovery review. Neither has an established 30% margin of safety from this work. These are bounded source checks, not completed deep dives or investment decisions.

## Screen and interpretation

The frozen universe contains 19,140 listings. The existing cash-consistency lens retains 3,465, of which 3,290 have usable saved valuation and price inputs. Requiring a 30% discount to value with half the terminal value counted yields **337 listings in 262 provisional research groups**.

Membership requires the operating-business route, no classification conflict, available comparable history, five positive annual provider FCF and EBIT observations, and complete valuation inputs. Grouping uses saved ISIN or matching issuer names and cash/earnings histories; it is administrative grouping, not verified ownership consolidation. All original listing IDs and their prices remain available.

```text
V50 = forecast cash PV + 0.50 × terminal PV
Displayed surplus = 100 × (V50 / saved equity price − 1)
30% discount to value requires price ≤ 0.70 × V50
                      or displayed surplus ≥ 42.857…%
```

DCF already includes terminal value; NPV subtracts price. These are related calculations, not three independent signals. The 50% terminal credit is a stress assumption, not a calibrated probability.

Across the 337 matches, 213 have at least one year with provider FCF above operating cash, 131 have positive net investing cash in some year, and 217 have saved net debt/EBIT above 3×. These overlapping flags require explanation; they do not establish data errors. The pack lacks separate lease-principal and maintenance-capex amounts. Four companies received issuer checks; no conclusion about the other 333 follows from this sample.

## Cash-definition sensitivities

All cash and equity amounts below are reporting-currency millions. The comparison holds each saved equity price fixed. It assumes flat annual cash for ten years at 10%, retains the original terminal PV, and counts half of it. Changing only annual cash isolates one sensitivity; it does **not** establish normalized owner cash or a corrected fair value.

| Company | Saved price date | Original surplus | Cash input checked | Surplus after annual-only change |
|---|---|---:|---|---:|
| Ipsos | 2026-02-24 | +52.89% | EUR 242.984 → issuer FCF 181.3 | +24.05% |
| Byggmax | 2026-01-30 | +71.97% | SEK 730 → 305 after lease principal | −6.87% |
| SFPI | 2026-02-09 | +57.96% | EUR 31.651 → 24.388 after capex and lease principal | +29.48% |

Each falls below the +42.86% threshold even with its old terminal PV retained. For reproduction, the annuity factor is `6.14456710570468`; apply it to the revised annual cash, then add half the original terminal PV:

| Company | Saved equity price | Original terminal PV | Revised V50 | 70% purchase ceiling |
|---|---:|---:|---:|---:|
| Ipsos | 1,314.304124 | 1,032.874328 | 1,630.447180 | 1,141.313026 |
| Byggmax | 3,312.312500 | 2,421.211858 | 3,084.698896 | 2,159.289227 |
| SFPI | 156.749180 | 106.224887 | 202.966146 | 142.076302 |

**Ipsos.** Issuer FY2025 FCF from operations was EUR 181.3m, versus 216.0m in 2024. A statement bridge gives `302.035 − 83.088 + 3.769 − 36.832 − 3.803 − 1.960 = 180.121m`: operating cash less capex, plus disposals, less lease principal and financing-classified interest. The remaining EUR 1.179m difference to issuer FCF is unresolved; the provider's detailed FCF construction is also unverified. Acquisitions and minority purchases consumed another EUR 178.623m and need separate capital-allocation analysis. [Issuer FY2025 release](https://www.ipsos.com/sites/default/files/ct/newsroom/documents/2026-02/Press%20Release%20-%202025%20Annual%20Results%20-%20240226%20-%20EN%20-%20FINAL.pdf), [audited accounts](https://www.ipsos.com/sites/default/files/Consolidated%20accounts%20including%20the%20statutory%20auditors%E2%80%99%20report.pdf)

H1 2026 FCF was EUR 44.3m versus 39.7m; organic revenue grew 0.8%. Research should test organic growth, client retention, AI exposure, acquisition returns and sustainable technology investment. The annual-only sensitivity leaves a 19.39% discount to assumed value; assuming 181.3m also supports terminal cash reduces that discount to 10.19%. [Issuer H1 results](https://www.ipsos.com/sites/default/files/documents/2026-07/Press%20Release%20-%202026%20Half-Year%20Results%20-%20EN%20-%202207%20-%20Final.pdf)

**Byggmax.** FY2025 operating cash of SEK 809m plus net investing cash of −79m gives the provider's 730m. Financing-classified lease principal of 425m reduces this to 305m. Using capex directly gives `809 − 83 − 425 = 301m`; lease interest already in operating cash must not be deducted again. LTM June 2026 cash after capex and lease principal was 289m. Even using the issuer's newer June 30 price of SEK 46.80, the annual-only sensitivity yields +12.43% surplus, below the threshold. June is not a September quote. Rebuild cash after leases before using the original rank to prioritize this business. [Annual report, pp. 78 and 95](https://attachment.news.eu.nasdaq.com/aa92d58a3032c2176a15cc70b7f8685ff), [Q2 report](https://attachment.news.eu.nasdaq.com/a72eecbab6d9438d160a37c7c2d0986a2)

**SFPI.** FY2025 cash after capex and lease principal was `41.685 − 10.591 − 6.706 = EUR 24.388m`. Including financing-classified net financial income of 1.234m raises the sensitivity surplus to 34.32%, still below the threshold. Reported net financial surplus of EUR 91.3m excludes IFRS16; cash availability, leases, minorities and required liquidity need reconciliation before adding surplus cash to a valuation. [Annual report, pp. 107 and 122](https://live.euronext.com/sites/default/files/company_press_releases/attachments/2026/04/30/cpr03_lesechos_16165_1405077_GROUPE_SFPI_RFA_2025.pdf)

The provider labels the FY2025 report with February 9, preceding the issuer's April 15 results release. That metadata does not establish historical information availability. SFPI remains a conditional balance-sheet/recovery research idea. [Issuer results publication](https://www.webdisclosure.com/press-release/groupe-sfpi-epa-sfpi-sfpi-group-publication-des-resultats-2025-uxyEmEOep74)

**Atkore.** The frozen screen shows +57.23% surplus against a November 20, 2025 price. On August 3, 2026, Atkore announced an agreement for Prysmian to acquire it for USD 95 cash per share, subject to closing conditions. That is an offer, not a current quote or completed payment. Nine-month operating cash of −90.335m less capex of 40.399m produced FCF of −130.734m. Route this to transaction-specific research; the old standalone screen cannot answer the acquisition question. [Issuer acquisition announcement](https://investors.atkore.com/investors/news/news-details/2026/Atkore-Inc--to-be-Acquired-by-Prysmian-for-95-00-per-Share-in-Cash/default.aspx), [Q3 results](https://investors.atkore.com/investors/news/news-details/2026/Atkore-Inc--Announces-Third-Quarter-2026-Results/default.aspx)

## Research queue and application implications

1. **Ipsos:** reconstruct several years of issuer-consistent cash and investigate business durability and reinvestment. This is research priority, not strongest verified upside.
2. **SFPI:** reconcile usable surplus cash and shareholder claims, then investigate the recovery assumptions.
3. **Byggmax:** resolve the cash model before treating apparent cheapness as a reason for a deep dive.
4. **Atkore:** separate transaction review.

For the whole universe, retain transparent terminal sensitivities but distinguish **history coverage** from **cash definition reviewed**. A proposed next list field is cash-review status, supported by source date and unresolved items. It has not been implemented in this research pass. A practical sequence is cash reconciliation → capital claims → current business and corporate actions → dated price/share basis → reviewed valuation.

Match enterprise cash to enterprise value and equity cash to equity value. Avoid charging lease payments and subtracting the entire lease liability without consistent treatment, or adding cash while retaining its interest income without adjustment. A wider uncertainty band or terminal haircut cannot repair these definition errors.

## Evidence and completion boundary

No September market quote was verified. The calculations use explicitly dated saved prices and are not a point-in-time backtest. Financial packs, app valuation inputs, user scenarios and preferences were unchanged. No new investment shortlist was silently saved into the application. Existing 0.20.0 implementation changes remain separate and uncommitted.

Local evidence is retained under `desktop/test-results/terminal-stress-shortlist-2026-09-13/`: `screen.py`, `screen.json`, `screen.csv`, four company notes, cash-sensitivity JSON files, source PDFs, `review-queue.csv` and `verification.json`. These generated artifacts are ignored by Git; this document preserves the key calculations and public sources in both repository worktrees.

The screen independently reconciled all 337 company payloads and five-year FCF/CFO histories against the pinned financial pack. Frozen gauge SHA-256: `fcbcccadbdd86da4a6521a64c2be8907ceff91f23dc1aa2e7fcfd2f991ba0e2b`. Financial pack SHA-256: `1f1534d48fc30c1e3769cb7f1e9060c85fb1dc63ec39b06ada40fb6e11d2c69a`.
