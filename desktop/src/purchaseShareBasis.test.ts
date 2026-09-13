import { describe, expect, it } from 'vitest';
import { getPurchaseShareReference } from './purchaseShareBasis';
import { buildResearchedValuation, holmenStudy, scaStudy } from './researchedValuations';
import { buildStarterValuation } from './starterValuations';
import type { FinancialCompany } from './financialData';

function genericStarter() {
  const annual = { year: 2025, period: 5, start: '2025-01-01', end: '2025-12-31', report_date: '2026-02-01', currency: 'EUR', currency_ratio: 1,
    source_id: 'annual-source', source_as_of: '2026-08-10', raw: { free_cash_flow: 50, revenues: 1000 }, values: { free_cash_flow: 50, revenues: 1000 } };
  const history: FinancialCompany = { id: '999', annual: [annual], quarterly: [], withheld: [], market: [{ year: 2025, source_id: 'annual-source', currency: 'EUR', shares: 10, price: 100, price_date: '2026-02-01', local: 1000, sek: 11000, fx_rate: 11, fx_date: '2026-02-01', fx_method: 'direct', fx_instruments: ['1'], flags: [] }] };
  const starter = buildStarterValuation({ id: '999', display_name: 'Example', report_currency: 'EUR', stock_currency: 'EUR', source_as_of: '2026-08-10', sector_id: '2', branch_id: '13' }, history, null, '2026-09-13');
  Object.assign(starter.evidence.price!, { sourcePath: 'saved/prices.parquet', sourceHash: 'a'.repeat(64) });
  return starter;
}

describe('source-bound optional purchase share reference', () => {
  it('uses Holmen reviewed outstanding shares instead of a conflicting generic count', () => {
    const starter = genericStarter(), draft = buildResearchedValuation(holmenStudy).draft;
    starter.evidence.price!.shares = 162.512;
    const result = getPurchaseShareReference('102', draft, starter, holmenStudy);
    expect(result).toMatchObject({ sharesMillions: 150.434534, date: '2026-05-29', currency: 'SEK', kind: 'reviewed' });
    expect(result!.source).toContain('outstanding A+B');
    expect(result!.source).toContain('treasury');
    expect(draft.marketCap! / result!.sharesMillions!).toBeCloseTo(329, 10);
  });

  it('uses explicit source-bound SCA mapping without parsing its narrative', () => {
    const draft = buildResearchedValuation(scaStudy).draft;
    const result = getPurchaseShareReference('197', draft, genericStarter(), scaStudy);
    expect(result).toMatchObject({ sharesMillions: 702.342489, date: '2026-06-30', currency: 'SEK', kind: 'reviewed' });
    expect(result!.source).toContain('2026-07-22');
    expect(draft.marketCap! / result!.sharesMillions!).toBeCloseTo(110.3, 10);
  });

  it.each(['marketCap', 'priceDate', 'priceSource', 'currency'] as const)('withholds reviewed reference after editing %s', key => {
    const draft = buildResearchedValuation(scaStudy).draft;
    if (key === 'marketCap') draft.marketCap = draft.marketCap! + 1;
    else draft[key] = key === 'currency' ? 'USD' : key === 'priceDate' ? '2026-08-06' : 'Manually changed ownership basis';
    expect(getPurchaseShareReference('197', draft, genericStarter(), scaStudy)).toBeNull();
  });

  it('rejects unknown research identity, wrong company, missing study and changed share evidence', () => {
    const draft = buildResearchedValuation(scaStudy).draft, starter = genericStarter();
    expect(getPurchaseShareReference('102', draft, starter, scaStudy)).toBeNull();
    expect(getPurchaseShareReference('197', draft, starter)).toBeNull();
    const unknown = structuredClone(draft); unknown.researchOrigin!.id = 'another-study';
    expect(getPurchaseShareReference('197', unknown, starter, scaStudy)).toBeNull();
    const changed = structuredClone(scaStudy); changed.sources.find(s => s.id === 'interim')!.sha256 = '0'.repeat(64);
    expect(getPurchaseShareReference('197', draft, starter, changed)).toBeNull();
  });

  it('withholds when ownership-bearing draft text is edited, while cash/return/investment edits retain the dated reference', () => {
    const draft = buildResearchedValuation(holmenStudy).draft;
    draft.scenarios.mid.cashFlows[0] = 123; draft.scenarios.mid.discountRate = 12; draft.investment = 4000;
    expect(getPurchaseShareReference('102', draft, genericStarter(), holmenStudy)).not.toBeNull();
    draft.notes.financing += ' New share issuance changes ownership.';
    expect(getPurchaseShareReference('102', draft, genericStarter(), holmenStudy)).toBeNull();
  });

  it('offers a generic reported-share proxy only against its exact original price basis', () => {
    const starter = genericStarter(), draft = structuredClone(starter.draft);
    const before = JSON.stringify({ draft, starter });
    const result = getPurchaseShareReference('999', draft, starter);
    expect(result).toMatchObject({ sharesMillions: 10, date: '2026-02-01', currency: 'EUR', kind: 'reported' });
    expect(result!.source).toContain('unreviewed');
    expect(result!.source).toContain('saved/prices.parquet');
    expect(JSON.stringify({ draft, starter })).toBe(before);
  });

  it.each(['marketCap', 'priceDate', 'priceSource', 'currency'] as const)('does not infer shares from an edited generic %s', key => {
    const starter = genericStarter(), draft = structuredClone(starter.draft);
    if (key === 'marketCap') draft.marketCap = 2000;
    else draft[key] = key === 'currency' ? 'SEK' : key === 'priceDate' ? '2026-02-02' : 'Other source';
    expect(getPurchaseShareReference('999', draft, starter)).toBeNull();
  });

  it('keeps missing, inconsistent, future or unbound generic references unavailable', () => {
    for (const change of [
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price = null; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.shares = null; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.shares = 0; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.sourceHash = null; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.sourcePath = null; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.shares = 20; },
      (s: ReturnType<typeof genericStarter>) => { s.evidence.price!.currency = 'USD'; },
      (s: ReturnType<typeof genericStarter>) => { s.draft.valuationDate = '2026-01-01'; },
    ]) {
      const starter = genericStarter(); change(starter);
      expect(getPurchaseShareReference('999', structuredClone(starter.draft), starter)).toBeNull();
    }
  });

  it('requires validated dated conversion when the original price was converted to SEK', () => {
    const starter = genericStarter();
    Object.assign(starter.evidence.price!, { method: 'sek', marketCap: 11000 });
    starter.evidence.currency = starter.draft.currency = 'SEK';
    starter.draft.marketCap = 11000;
    expect(getPurchaseShareReference('999', starter.draft, starter)?.sharesMillions).toBe(10);
    starter.evidence.price!.fxRate = null;
    expect(getPurchaseShareReference('999', starter.draft, starter)).toBeNull();
  });

  it('never falls back to generic shares for an unrecognized reviewed draft', () => {
    const starter = genericStarter(), draft = structuredClone(starter.draft);
    draft.researchOrigin = { id: 'unknown-reviewed-study', asOf: '2026-09-13' };
    expect(getPurchaseShareReference('999', draft, starter)).toBeNull();
  });
});
