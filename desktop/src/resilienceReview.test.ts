import { describe, expect, it } from 'vitest';
import { blankResilienceReview, calculateLiquidity, calculateImpairment, parseAnnualCash, getReviewStatus, loadResilienceReview, saveResilienceReview, resilienceReviewKey } from './resilienceReview';

const basis = { companyId: '123', companyName: 'Example', releaseId: 'release', financialId: 'financial', sourceAsOf: '2026-09-26' };
function complete() {
  const r = blankResilienceReview(basis);
  Object.assign(r, { currency: 'SEK', evidenceDate: '2026-09-01', evidenceSources: 'Annual report, note 12', shockNarrative: 'Demand falls and refinancing closes together.', covenantConstraints: 'Facility confirmed drawable in this scenario; covenant headroom reviewed.', conclusionRationale: 'Endpoints and impairment reviewed.', conclusion: 'reviewed-for-stated-scenario' });
  Object.assign(r.liquidity, { startDate: '2026-10-01', openingCash: 100, minimumCash: 20, rows: [{ operatingCash: -10, principalDue: 20, committedFunding: 0, shockDrain: 5 }, { operatingCash: 0, principalDue: 10, committedFunding: 0, shockDrain: 5 }] });
  Object.assign(r.impairment, { annualCash: '10, 0', discountRate: 0, finalProceeds: 5, ownership: 'Cash belongs to the pre-rescue shareholders after 50% dilution.', assumptions: 'Permanent closure of one plant; no recovery to prior margins.', evidenceDate: '2026-09-01', evidenceSources: 'Closure budget and capital plan.' });
  return r;
}

describe('company resilience arithmetic and review boundaries', () => {
  it('accumulates combined shocks, principal and scenario-available funding before finding a constant extra drain', () => {
    const result = calculateLiquidity(complete().liquidity);
    expect(result.error).toBeNull();
    expect(result.rows.map(r => r.cash)).toEqual([65, 50]);
    expect(result.rows.map(r => r.headroom)).toEqual([45, 30]);
    expect(result.additionalDrainPerPeriod).toBe(15);
    expect(result.firstBreach).toBeNull();
  });
  it('retains an earlier breach even if a later funding inflow restores liquidity', () => {
    const r = complete();
    r.liquidity.rows[0].shockDrain = 100;
    r.liquidity.rows[1].committedFunding = 200;
    const result = calculateLiquidity(r.liquidity);
    expect(result.firstBreach).toBe(1);
    expect(result.rows[1].headroom).toBeGreaterThan(0);
    expect(result.additionalDrainPerPeriod).toBe(0);
    expect(getReviewStatus(r, basis).code).toBe('material-failure');
    expect(getReviewStatus(r, basis).reviewed).toBe(false);
  });
  it('distinguishes zero buffer, an opening breach and missing or invalid inputs', () => {
    const r = complete();
    r.liquidity.openingCash = 20;
    r.liquidity.rows = [{ operatingCash: 0, principalDue: 0, committedFunding: 0, shockDrain: 0 }];
    expect(calculateLiquidity(r.liquidity)).toMatchObject({ firstBreach: null, additionalDrainPerPeriod: 0 });
    r.liquidity.openingCash = 19;
    expect(calculateLiquidity(r.liquidity).firstBreach).toBe(0);
    r.liquidity.rows[0].operatingCash = null;
    expect(calculateLiquidity(r.liquidity).rows).toEqual([]);
    r.liquidity.rows[0].operatingCash = 0;
    r.liquidity.rows[0].principalDue = -1;
    expect(calculateLiquidity(r.liquidity).error).toBeTruthy();
  });
  it('allows future cash inflows to fund endpoint drains when opening exactly at the minimum', () => {
    const r = complete(); r.liquidity.openingCash = 20;
    r.liquidity.rows = [0, 1].map(() => ({ operatingCash: 10, principalDue: 0, committedFunding: 0, shockDrain: 0 }));
    expect(calculateLiquidity(r.liquidity)).toMatchObject({ firstBreach: null, additionalDrainPerPeriod: 10 });
  });
  it('prices only the authored existing shareholder claim and final proceeds once', () => {
    const r = complete();
    Object.assign(r.impairment, { annualCash: '11, -12.1', discountRate: 10, finalProceeds: 24.2, purchaseEquity: 5, priceDate: '2026-09-25', priceSource: 'Quote for the same existing ownership claim' });
    const result = calculateImpairment(r.impairment, r.liquidity.startDate);
    expect(result.value).toBeCloseTo(20);
    expect(result.npv).toBeCloseTo(15);
    r.impairment.annualCash = '0'; r.impairment.finalProceeds = 0;
    expect(calculateImpairment(r.impairment).value).toBe(0);
    r.impairment.priceDate = '';
    expect(calculateImpairment(r.impairment)).toMatchObject({ npv: null });
    expect(calculateImpairment(r.impairment).priceError).toBeTruthy();
  });
  it('withholds NPV for missing valuation dates or prices from after the valuation origin', () => {
    const r = complete(); Object.assign(r.impairment, { purchaseEquity: 5, priceDate: '2026-10-02', priceSource: 'Quote' });
    expect(calculateImpairment(r.impairment).npv).toBeNull();
    expect(calculateImpairment(r.impairment, '2026-10-01')).toMatchObject({ value: 15, npv: null });
    expect(getReviewStatus(r, basis).reviewed).toBe(false);
  });
  it('preserves blanks and rejects malformed annual sequences rather than making zero cash', () => {
    for (const text of ['', '1,,2', '1,', '1,no', 'Infinity', '1 2']) expect(parseAnnualCash(text).error).toBeTruthy();
    expect(parseAnnualCash('0, -2.5\n3').cash).toEqual([0, -2.5, 3]);
  });
  it('withholds reviewed status for missing evidence, unacknowledged failures and stale source bases', () => {
    const r = complete();
    expect(getReviewStatus(r, basis).reviewed).toBe(true);
    r.impairment.ownership = '';
    expect(getReviewStatus(r, basis).reviewed).toBe(false);
    r.impairment.ownership = 'Existing claim after dilution';
    r.evidenceSources = '';
    expect(getReviewStatus(r, basis).code).toBe('unresolved');
    r.evidenceSources = 'Report';
    expect(getReviewStatus(r, { ...basis, financialId: 'new' })).toMatchObject({ code: 'stale', stale: true, reviewed: false });
    expect(getReviewStatus(r, { ...basis, sourceAsOf: '2026-10-01' }).stale).toBe(true);
    r.conclusion = 'material-failure';
    expect(getReviewStatus(r, { ...basis, financialId: 'new' }).detail).toContain('material failure');
  });
});

describe('isolated explicit persistence', () => {
  function storage() { const data = new Map<string, string>(); return { data, getItem: (k: string) => data.get(k) ?? null, setItem: (k: string, v: string) => { data.set(k, v); } }; }
  it('reads without writing and keeps other company and valuation records untouched', () => {
    const s = storage(); s.data.set('macro-atlas-valuations-v1', 'existing');
    expect(loadResilienceReview(s, basis.companyId)).toBeNull();
    expect(s.data.size).toBe(1);
    saveResilienceReview(s, complete(), null);
    expect(loadResilienceReview(s, basis.companyId)?.companyId).toBe('123');
    expect(s.data.get('macro-atlas-valuations-v1')).toBe('existing');
  });
  it('preserves malformed, future and unknown-field records byte for byte', () => {
    const s = storage(), key = resilienceReviewKey(basis.companyId);
    for (const raw of ['{bad', JSON.stringify({ ...complete(), version: 2 }), JSON.stringify({ ...complete(), future: true })]) {
      s.data.set(key, raw);
      expect(() => loadResilienceReview(s, basis.companyId)).toThrow(/preserved/i);
      expect(() => saveResilienceReview(s, complete(), raw)).toThrow(/preserved/i);
      expect(s.data.get(key)).toBe(raw);
    }
  });
  it('rejects conflicting concurrent edits and surfaces storage failures', () => {
    const s = storage(), key = resilienceReviewKey(basis.companyId);
    saveResilienceReview(s, complete(), null);
    expect(() => saveResilienceReview(s, complete(), null)).toThrow(/changed/i);
    expect(s.data.has(key)).toBe(true);
    expect(() => saveResilienceReview({ getItem: () => null, setItem: () => { throw Error('quota'); } }, complete(), null)).toThrow('quota');
  });
});
