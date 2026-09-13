import { buildResearchedValuation, holmenStudy, normalizeResearchedStudy, scaStudy, type ResearchedStudy } from './researchedValuations';
import type { buildStarterValuation } from './starterValuations';
import { scenarioKeys, validAmount, validDay, type ValuationDraft } from './valuation';

export type PurchaseShareReference = {
  sharesMillions: number; date: string; source: string; currency: string;
  kind: 'reviewed' | 'reported';
};
type Starter = ReturnType<typeof buildStarterValuation>;
const positive = (value: unknown): value is number => validAmount(value) && value > 0;
const near = (a: number, b: number) => Math.abs(a - b) <= 1e-8 * Math.max(1, Math.abs(a), Math.abs(b));
const samePrice = (draft: ValuationDraft, original: ValuationDraft) => positive(draft.marketCap)
  && draft.marketCap === original.marketCap && draft.currency === original.currency
  && draft.priceDate === original.priceDate && draft.priceSource === original.priceSource;
const sameOwnershipText = (draft: ValuationDraft, original: ValuationDraft) => draft.notes.financing === original.notes.financing
  && scenarioKeys.every(key => draft.scenarios[key].rationale === original.scenarios[key].rationale);

// Explicit, dated references. These constants do not parse narrative text or
// reverse-engineer a denominator from equity value and a share price.
const reviewedReferences = {
  '102': {
    study: holmenStudy, id: 'holmen-2026-09-11-v1', isin: 'SE0011090018',
    sharesMillions: 150.434534, date: '2026-05-29', priceDate: '2026-08-07', close: 329,
    deepDiveHash: '933def46787d053c74c6ed4f79920399786782a23ac3bbe102873fe930abcc7e',
    shareSourceId: 'shares', shareSourceDate: '2026-05-29',
    shareSourceUrl: 'https://www.holmen.com/en/Newsroom/press/press-releases/2026/changes-in-the-total-number-of-shares-in-holmen/',
    shareSourceHash: null,
    description: 'Holmen B-equivalent common equity: 150.434534 million outstanding A+B shares, excluding treasury shares, from the 2026-05-29 share announcement. Equal economic rights and unchanged shares are assumptions; this is not a sum of separately priced classes.',
  },
  '197': {
    study: scaStudy, id: 'sca-2026-09-12-v1', isin: 'SE0000112724',
    sharesMillions: 702.342489, date: '2026-06-30', priceDate: '2026-08-07', close: 110.3,
    deepDiveHash: '0d4ea16b0c3d782d02295420bf4cd8bfc81b76d7db079371dd21e8266ff89269',
    shareSourceId: 'interim', shareSourceDate: '2026-07-22',
    shareSourceUrl: 'https://www.sca.com/siteassets/media/press-releases-and-reports/documents/2026/20260722-half-year-report-q2-2026-en-0-5400552.pdf',
    shareSourceHash: '3b0034d3f1713805a1ef0a5b8da323a74b91f903d575dfba8104b20e36931a38',
    description: 'SCA B-equivalent common equity: 702.342489 million outstanding A+B shares at 2026-06-30, disclosed in the half-year report published 2026-07-22, page 17. Equal economic rights and unchanged shares are assumptions; this is not a sum of separately priced classes.',
  },
} as const;
const priceFileHash = 'a8c9f95df438070483c4822f92a6bf806e3dd37b616c34f318c95fa8fa08e547';
const priceFilePath = 'data/raw_api_snapshots/2026-08-10/all_stockprices/all_stockprices.parquet';

/** A suggestion only. Persist the four share-basis fields only after user action. */
export function getPurchaseShareReference(companyId: string, draft: ValuationDraft, starter: Starter, researched: ResearchedStudy | null = null): PurchaseShareReference | null {
  if (!validDay(draft.valuationDate) || !validDay(draft.priceDate) || draft.priceDate > draft.valuationDate) return null;
  if (draft.researchOrigin) {
    const reference = reviewedReferences[companyId as keyof typeof reviewedReferences];
    if (!reference || !researched) return null;
    try {
      const study = normalizeResearchedStudy(researched), known = normalizeResearchedStudy(reference.study);
      const shareSource = study.sources.find(source => source.id === reference.shareSourceId);
      const priceSource = study.sources.find(source => source.id === 'prices');
      // Keep the reviewed reference bound to its source study and ownership
      // statement, even if another candidate reuses an existing study ID.
      if (study.company !== companyId || study.id !== reference.id || study.isin !== reference.isin || study.currency !== 'SEK'
        || study.deepDive.sha256 !== reference.deepDiveHash || study.asOf !== known.asOf
        || draft.researchOrigin.id !== study.id || draft.researchOrigin.asOf !== study.asOf
        || study.ownership !== known.ownership || JSON.stringify(study.price) !== JSON.stringify(known.price)
        || shareSource?.date !== reference.shareSourceDate || shareSource.url !== reference.shareSourceUrl
        || reference.shareSourceHash !== null && shareSource.sha256 !== reference.shareSourceHash
        || priceSource?.sha256 !== priceFileHash || priceSource.path !== priceFilePath || priceSource.date !== '2026-08-10'
        || study.price.date !== reference.priceDate || study.price.marketCap !== reference.close * reference.sharesMillions
        || reference.shareSourceDate > draft.valuationDate || priceSource.date > draft.valuationDate) return null;
      const original = buildResearchedValuation(reference.study).draft;
      if (!samePrice(draft, original) || !sameOwnershipText(draft, original)) return null;
      return { sharesMillions: reference.sharesMillions, date: reference.date, currency: 'SEK', kind: 'reviewed',
        source: `${reference.description} Reviewed study ${reference.id}. ${reference.shareSourceUrl}${reference.shareSourceHash ? ` SHA-256 ${reference.shareSourceHash}.` : ''} Price reference: ${reference.close} SEK on ${reference.priceDate}; saved 2026-08-10, SHA-256 ${priceFileHash}.` };
    } catch {
      // Unsupported or malformed research never inherits generic pack shares.
      return null;
    }
  }
  const price = starter.evidence.price;
  if (!price || !samePrice(draft, starter.draft) || !sameOwnershipText(draft, starter.draft)
    || draft.currency !== starter.evidence.currency || price.priceDate !== draft.priceDate
    || price.marketCap !== draft.marketCap || !positive(price.shares) || !positive(price.close)
    || !validDay(price.reportDate) || !validDay(price.reportEnd) || !validDay(price.sourceAsOf)
    || price.reportDate < price.reportEnd || price.reportDate > price.priceDate || price.sourceAsOf < price.priceDate
    || price.sourceAsOf > draft.valuationDate || !price.sourcePath?.trim() || !price.sourceHash || !/^[a-f0-9]{64}$/.test(price.sourceHash)) return null;
  let factor = 1;
  if (price.method === 'local') {
    if (price.currency !== draft.currency) return null;
  } else if (price.method === 'sek') {
    if (draft.currency !== 'SEK' || !positive(price.fxRate) || !validDay(price.fxDate) || price.fxDate > price.priceDate
      || (Date.parse(price.priceDate) - Date.parse(price.fxDate)) / 86400000 > 7) return null;
    factor = price.fxRate;
  } else return null;
  if (!near(price.marketCap, price.shares * price.close * factor)) return null;
  return { sharesMillions: price.shares, date: price.reportDate, currency: draft.currency, kind: 'reported',
    source: `Listing ${companyId}: ${price.shares} million reported shares in the annual report published ${price.reportDate} (${price.reportStart}–${price.reportEnd}). Reported-share proxy, unreviewed; unchanged shares and matching common-equity rights require review. Saved close ${price.close} ${price.currency} on ${price.priceDate}${price.method === 'sek' ? `; converted using ${price.fxRate} SEK per ${price.currency} dated ${price.fxDate}` : ''}. Source ${price.sourceId}, saved ${price.sourceAsOf}; ${price.sourcePath}, SHA-256 ${price.sourceHash}.` };
}
