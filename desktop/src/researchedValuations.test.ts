import { describe, expect, it } from 'vitest';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { blankValuation, calculateValuation } from './valuation';
import { buildResearchedValuation, holmenStudy, researchedStudyFor, shouldStartResearchedStudy } from './researchedValuations';
import { reviewedStudy, type ReviewedStudy } from './researchedStudy';
import ValuationResearchEvidence from './ValuationResearchEvidence';

function euroStudy(): ReviewedStudy {
  return {
    version: 2, id: 'example-2026-09-10-v2', company: 'example', isin: 'FI0000000000', name: 'Example', asOf: '2026-09-10', currency: 'EUR',
    deepDive: { date: '2026-03-12', path: 'studies/example.md', sha256: 'a'.repeat(64) },
    sources: [{ id: 'annual', title: 'Example FY 2024 report', date: '2025-02-28', location: 'Page 42', url: 'https://example.com/annual.pdf' }],
    price: { marketCap: 100, date: '2026-09-01', narrative: 'EUR 10 close × 10 million single-class common shares; no currency conversion.' },
    years: 3, ownership: 'Constant common-share ownership. Nominal EUR after-financing distributions.',
    capital: { tangibleEquity: 40, averageTCE: null, nopat: null, grossDebt: 12, surplusCash: null },
    notes: { business: 'Finite runoff.', macro: 'Export receipts.', financing: 'TCE remains unreconciled.', recovery: 'No appraisal available.', decision: 'Scenario sensitivity only.' },
    scenarios: {
      low: { cashFlows: [4, 3, 2], discountRate: 8, terminalEquity: 20, rationale: 'Low receipts.' },
      mid: { cashFlows: [7, 5, 3], discountRate: 8, terminalEquity: 30, rationale: 'Mid receipts.' },
      high: { cashFlows: [10, 8, 6], discountRate: 8, terminalEquity: 40, rationale: 'High receipts.' },
    },
    evidence: [{ heading: 'Reviewed annual bridge', blocks: [
      { kind: 'paragraph', classification: 'calculation', text: 'FY 2024 shareholder cash: EUR 9m less EUR 2m reinvestment = EUR 7m.', sourceIds: ['annual'] },
      { kind: 'table', classification: 'source', columns: ['Reported item', 'FY 2024'], rows: [{ label: 'Cash received', values: [9] }, { label: 'Matched average TCE', values: [null] }], sourceIds: ['annual'], precision: 1 },
    ] }],
    recovery: { status: 'unavailable', reason: 'Asset realizations and prior claims are not yet reconciled.', sourceIds: ['annual'] },
  };
}

describe('researched company valuation', () => {
  it('preserves the entire published Holmen draft so existing drafts do not become falsely edited', async () => {
    // Captured from the pre-adapter implementation, including provenance strings.
    const serialized = JSON.stringify(buildResearchedValuation(holmenStudy).draft);
    const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(serialized));
    expect(Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('')).toBe('bc14e3a76eb71c19bb1169c7ab23a5373b7be7fff73b4dd2f1750dcf7c938b53');
  });
  it('requires the reviewed listing identity and archived research source', () => {
    const entry = { id: '102', isin: 'SE0011090018' };
    const sources = [{ sha256: holmenStudy.deepDive.sha256 }];
    expect(researchedStudyFor(entry, sources)?.id).toBe(holmenStudy.id);
    expect(researchedStudyFor({ ...entry, id: '197' }, sources)).toBeNull();
    expect(researchedStudyFor({ ...entry, isin: 'SE0011090000' }, sources)).toBeNull();
    expect(researchedStudyFor(entry, [])).toBeNull();
    expect(researchedStudyFor(entry, [{ sha256: 'f'.repeat(64) }])).toBeNull();
  });
  it('opens a complete, dated Holmen analysis without waiting for async vendor history', () => {
    const { draft, bridge, recovery } = buildResearchedValuation(holmenStudy);
    expect(draft.marketCap).toBeCloseTo(49492.961686, 6);
    expect(draft.priceDate).toBe('2026-08-07');
    expect(draft.valuationDate).toBe('2026-09-11');
    expect(draft.priceSource).toMatch(/B-equivalent/);
    expect(draft.researchOrigin?.id).toBe(holmenStudy.id);
    expect(bridge.operatingCash).toBe(3444);
    expect(bridge.capex).toBe(1557);
    expect(bridge.leasePrincipal).toBe(134);
    expect(bridge.cashTaxes).toBe(-10);
    expect(bridge.normalizedAvailableCash).toBe(1243);
    expect(Object.values(draft.capital).every(Number.isFinite)).toBe(true);
    expect(draft.capital.tangibleEquity).toBe(53700);
    expect(draft.capital.grossDebt).toBe(6917);
    expect(draft.capital.averageTCE).toBe(60083);
    expect(draft.capital.nopat).toBe(1852.5);
    expect(draft.capital.surplusCash).toBe(0);
    expect(holmenStudy.recovery.assets.reduce((sum, asset) => sum + asset.book, 0)).toBe(81068 - 482);
    // Reconcile the publisher's broader net-financial-debt perimeter, including
    // pensions and financial receivables, rather than equating it with borrowings.
    expect(draft.capital.grossDebt! + 5 - 29 - 211 - 18 - 182).toBe(6482);
    expect(calculateValuation(draft).ready).toBe(true);
    for (const key of ['low', 'mid', 'high'] as const) {
      expect(recovery[key].claims).toBe(26887);
      expect(recovery[key].net).toBe(draft.scenarios[key].recoveryEquity);
      expect(draft.scenarios[key].recoveryYear).toBeGreaterThan(0);
    }
  });
  it('uses after-financing cash, a common required return and one terminal equity sale', () => {
    const { draft } = buildResearchedValuation(holmenStudy), result = calculateValuation(draft);
    expect(result.scenarios.low.stakeValue).toBeCloseTo(210.75968198, 6);
    expect(result.scenarios.mid.stakeValue).toBeCloseTo(401.37954124, 6);
    expect(result.scenarios.high.stakeValue).toBeCloseTo(783.87018594, 6);
    expect(result.scenarios.high.payback.year).toBe(16);
    expect(result.scenarios.high.discountedPayback.year).toBeNull();
    expect(result.scenarios.mid.payback.year).toBeNull();
    for (const key of ['low', 'mid', 'high'] as const) {
      const s = draft.scenarios[key];
      expect(s.discountRate).toBe(9);
      expect(s.terminalEquity).toBeCloseTo(s.cashFlows[19]! * (1 + holmenStudy.scenarios[key].matureGrowth / 100) / (0.09 - holmenStudy.scenarios[key].matureGrowth / 100) * 0.98);
      expect(result.scenarios[key].npv).toBeLessThan(0);
    }
    const before = result.scenarios.mid.value;
    draft.capital.grossDebt = 99999; draft.scenarios.mid.recoveryEquity = 99999;
    expect(calculateValuation(draft).scenarios.mid.value).toBe(before);
  });
  it('migrates untouched drafts, including a selected historical price, but preserves entered forecasts', () => {
    expect(shouldStartResearchedStudy(null)).toBe(true);
    const d = blankValuation('Holmen');
    d.marketCap = 54246; d.priceSource = 'A price selected in the old workspace';
    expect(shouldStartResearchedStudy(d)).toBe(true);
    d.scenarios.mid.cashFlows[0] = 0;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    d.scenarios.mid.cashFlows[0] = null; d.scenarios.mid.discountRate = 0;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    d.scenarios.mid.discountRate = null; d.researchAutofillDisabled = true;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    const seeded = buildResearchedValuation(holmenStudy).draft;
    seeded.scenarios.mid.cashFlows[0] = null;
    expect(shouldStartResearchedStudy(seeded)).toBe(false);
    const clearedStarter = blankValuation('Edited historical starter');
    clearedStarter.starterOrigin = { id: 'weighted-cash-starter-v1', asOf: '2026-09-12', weights: [30, 25, 20, 15, 10], spreadPercent: 20 };
    expect(shouldStartResearchedStudy(clearedStarter)).toBe(false);
  });
  it('returns fresh editable drafts without mutating the researched baseline', () => {
    const first = buildResearchedValuation(holmenStudy).draft;
    first.scenarios.mid.cashFlows[0] = 500; first.notes.macro = 'My view';
    const next = buildResearchedValuation(holmenStudy).draft;
    expect(next.scenarios.mid.cashFlows[0]).toBe(1500);
    expect(next.notes.macro).not.toBe('My view');
  });
  it('loads explicit distributions and dated ownership in the study currency without fabricating missing capital or recovery', () => {
    const study = euroStudy(), { draft } = buildResearchedValuation(study);
    expect(draft.currency).toBe('EUR');
    expect(draft.valuationDate).toBe('2026-09-10');
    expect(draft.priceDate).toBe('2026-09-01');
    expect(draft.priceSource).toBe(study.price.narrative);
    expect(draft.marketCap).toBe(100);
    expect(draft.capital).toEqual(study.capital);
    expect(draft.scenarios.mid.cashFlows).toEqual([7, 5, 3]);
    expect(draft.scenarios.mid.terminalEquity).toBe(30);
    expect(draft.scenarios.mid.recoveryEquity).toBeNull();
    expect(draft.scenarios.mid.recoveryYear).toBeNull();
    expect(draft.scenarios.mid.rationale).toContain(study.ownership);
    expect(calculateValuation(draft).scenarios.mid.value).toBeCloseTo(7 / 1.08 + 5 / 1.08 ** 2 + 33 / 1.08 ** 3);
    draft.scenarios.mid.cashFlows[0] = 999; draft.capital.grossDebt = 999;
    expect(study.scenarios.mid.cashFlows[0]).toBe(7);
    expect(study.capital.grossDebt).toBe(12);
  });
  it('requires both the archive path and digest for a v2 study', () => {
    const study = euroStudy(), entry = { id: study.company, isin: study.isin };
    expect(researchedStudyFor(entry, [study.deepDive], [study])).toBe(study);
    expect(researchedStudyFor(entry, [{ sha256: study.deepDive.sha256 }], [study])).toBeNull();
    expect(researchedStudyFor(entry, [{ ...study.deepDive, path: 'another.md' }], [study])).toBeNull();
    expect(researchedStudyFor({ ...entry, isin: 'wrong' }, [study.deepDive], [study])).toBeNull();
  });
  it('renders company-specific periods and source-linked calculations without legacy-company text', () => {
    const html = renderToStaticMarkup(createElement(ValuationResearchEvidence, { study: euroStudy(), edited: false }));
    expect(html).toContain('How the Example scenarios were built');
    expect(html).toContain('EUR millions');
    expect(html).toContain('FY 2024');
    expect(html).toContain('href="#research-example-2026-09-10-v2-source-annual"');
    expect(html).toContain('Unavailable');
    expect(html).toContain('Asset realizations and prior claims are not yet reconciled.');
    expect(html).not.toMatch(/Holmen|SEK|H1 2026|biological|forest|B-equivalent|57,370/);
  });
  it('keeps an incomplete forecast missing and rejects malformed reviewed source bridges', () => {
    const study = euroStudy();
    study.scenarios.mid.cashFlows[1] = null;
    expect(buildResearchedValuation(study).draft.scenarios.mid.cashFlows).toEqual([7, null, 3]);
    expect(calculateValuation(buildResearchedValuation(study).draft).ready).toBe(false);
    expect(reviewedStudy(study)).toEqual(study);
    const wrongLength = structuredClone(study); wrongLength.scenarios.mid.cashFlows.pop();
    expect(() => reviewedStudy(wrongLength)).toThrow(/cashFlows/);
    const missingSource = structuredClone(study); missingSource.evidence[0].blocks[0].sourceIds = ['missing'];
    expect(() => reviewedStudy(missingSource)).toThrow(/source/);
    const missingCapital = structuredClone(study); delete (missingCapital.capital as Partial<ReviewedStudy['capital']>).averageTCE;
    expect(() => reviewedStudy(missingCapital)).toThrow(/averageTCE/);
  });
  it('calculates a separate asset recovery, exposes funding shortfalls, and propagates unknown proceeds', () => {
    const study = euroStudy();
    study.recovery = { status: 'available', explanation: 'Illustrative asset sale.', sourceIds: ['annual'],
      assets: [{ label: 'Property', book: 50, proceeds: { low: 10, mid: 30, high: null } }],
      scenarios: {
        low: { claims: 15, costs: 2, cashBurn: 3, year: 2 },
        mid: { claims: 15, costs: 2, cashBurn: 3, year: 2 },
        high: { claims: 15, costs: 2, cashBurn: 3, year: 2 },
      }, limitations: 'Minority owners cannot force a sale.' };
    const { draft, recovery } = buildResearchedValuation(study);
    expect(recovery!.low.shortfall).toBe(10);
    expect(draft.scenarios.low.recoveryEquity).toBe(0);
    expect(draft.scenarios.mid.recoveryEquity).toBe(10);
    expect(draft.scenarios.high.recoveryEquity).toBeNull();
    expect(recovery!.high.gross).toBeNull();
    expect(calculateValuation(draft).scenarios.mid.value).toBeCloseTo(7 / 1.08 + 5 / 1.08 ** 2 + 33 / 1.08 ** 3);
    const html = renderToStaticMarkup(createElement(ValuationResearchEvidence, { study, edited: false }));
    expect(html).toContain('Funding shortfall before common equity');
  });
});
