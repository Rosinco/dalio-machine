import { describe, expect, it } from 'vitest';
import type { ResearchGaugeRow } from './researchGaugeModel';
import { calculateValuationAttractiveness } from './valuationAttractiveness';

function row(): ResearchGaugeRow {
  const series = { values: [100], count: 1, positive: 1, negative: 0, zero: 0, latest: 100, median: 100, dispersion: null };
  const period = { year: 2025, period: 5, start: '2025-01-01', end: '2025-12-31', published: '2026-02-01', currency: 'SEK', sourceId: 'annual', sourceAsOf: '2026-08-10' };
  return {
    id: '1', name: 'Example', ticker: 'EX', isin: null, country: 'SE', sectorId: '3', sectorName: 'Industrials', branchId: '20', branchName: 'Tools', sourceAsOf: '2026-08-10', presence: 'latest', sourceCompanySha256: 'a'.repeat(64), route: 'operating', classificationConflict: false, readiness: 'limited_history', issues: [],
    annual: { currency: 'SEK', latest: period, periods: [period], excluded: { outsideWindow: 0, placeholder: 0, missingPublication: 0, withheld: 0 }, cash: series, operatingCash: series, ebit: series, revenue: series, margins: series, latestValues: { revenues: 100, ebit: 100, cash: 100, operatingCash: 100, financingCash: null, cashBalance: null, netDebt: null, equity: null, assets: null, netDebtToAssetsPercent: null, equityToAssetsPercent: null, tangibleAssetsToRevenue: null, intangibleAssetsToAssetsPercent: null, cashComponentDifference: null }, reason: 'One comparable annual report.' },
    quarter: { latest: null, comparison: null, revenueChangePercent: null, ebitMarginChangePoints: null, cashChange: null, reason: 'No comparable quarter.' },
    valuation: { status: 'positive-priced', currency: 'SEK', value: 1000, cashPV: 400, terminalPV: 600, terminalShare: .6, candidateEquity: 700, priceDate: '2026-02-15', priceAgeDays: 210, priceBasis: { sourceId: 'annual', sourceAsOf: '2026-08-10', reportDate: '2026-02-01', reportEnd: '2025-12-31', shares: 7, close: 100, currency: 'SEK', method: 'local', fxRate: null, fxDate: null }, ceiling: 700, lowValue: 650, lowNPV: -50, reverseCashFactor: .7, reverseCashFactor30: 1, hasSignedCash: false, annualHistoryCount: 1, rangeStatus: 'percentage', reason: null },
  };
}

const empty = { surplusPercent: null, midNpvPercent: null, dcfPriceRatio: null, cashCoveragePercent: null, lowNpvPercent: null, adjustedValue: null, adjustedNpv: null };

describe('dated starter valuation attractiveness', () => {
  it('includes terminal value once and maps a 30% discount to value to a 42.86% NPV/price comparison', () => {
    const result = calculateValuationAttractiveness(row());
    expect(result.surplusPercent).toBeCloseTo(42.857142857);
    expect(result.midNpvPercent).toBe(result.surplusPercent);
    expect(result.dcfPriceRatio).toBeCloseTo(10 / 7);
    expect(result.cashCoveragePercent).toBeCloseTo(400 / 7);
    expect(result.lowNpvPercent).toBeCloseTo(-50 / 7);
    expect(result).toMatchObject({ adjustedValue: 1000, adjustedNpv: 300, terminalCreditPercent: 100, reason: null });
  });

  it('changes only adjusted value and surplus at 0%, 50%, and 100% terminal credit', () => {
    const full = calculateValuationAttractiveness(row());
    for (const [credit, value, surplus] of [[0, 400, -300 / 7], [50, 700, 0], [100, 1000, 300 / 7]]) {
      const result = calculateValuationAttractiveness(row(), credit);
      expect(result.adjustedValue).toBe(value);
      expect(result.adjustedNpv).toBe(value - 700);
      expect(result.surplusPercent).toBeCloseTo(surplus);
      for (const key of ['midNpvPercent', 'dcfPriceRatio', 'cashCoveragePercent', 'lowNpvPercent'] as const) expect(result[key]).toBe(full[key]);
    }
    expect(calculateValuationAttractiveness(row(), 12.5).adjustedValue).toBe(475);
  });

  it('keeps ratios invariant under equity-size and consistent valuation-currency scaling', () => {
    const original = calculateValuationAttractiveness(row(), 50);
    for (const factor of [.001, 10, 1_000_000]) {
      const r = row(); r.valuation.currency = 'USD';
      for (const key of ['value', 'cashPV', 'terminalPV', 'candidateEquity', 'lowValue'] as const) r.valuation[key]! *= factor;
      const scaled = calculateValuationAttractiveness(r, 50);
      for (const key of ['surplusPercent', 'midNpvPercent', 'dcfPriceRatio', 'cashCoveragePercent', 'lowNpvPercent'] as const) expect(scaled[key]).toBeCloseTo(original[key]!);
      expect(scaled.adjustedValue).toBeCloseTo(original.adjustedValue! * factor);
    }
  });

  it('preserves negative funding cash, terminal dependence above 100%, and values below zero', () => {
    const r = row(); Object.assign(r.valuation, { cashPV: -100, terminalPV: 200, value: 100, terminalShare: 2, candidateEquity: 50, hasSignedCash: true });
    expect(calculateValuationAttractiveness(r, 50)).toMatchObject({ adjustedValue: 0, adjustedNpv: -50, surplusPercent: -100, midNpvPercent: 100, cashCoveragePercent: -200, reason: null });
    expect(calculateValuationAttractiveness(r, 0).surplusPercent).toBe(-300);
    Object.assign(r.valuation, { status: 'nonpositive', cashPV: -200, terminalPV: 100, value: -100, terminalShare: null });
    expect(calculateValuationAttractiveness(r)).toMatchObject({ surplusPercent: -300, dcfPriceRatio: -2, reason: null });
    r.valuation.cashPV = -100; r.valuation.value = 0;
    expect(calculateValuationAttractiveness(r)).toMatchObject({ surplusPercent: -100, dcfPriceRatio: 0, reason: null });
  });

  it('distinguishes zero value, no terminal value, and a terminal-only valuation', () => {
    const r = row(); Object.assign(r.valuation, { cashPV: 0, terminalPV: 1000, value: 1000 });
    expect(calculateValuationAttractiveness(r, 0)).toMatchObject({ surplusPercent: -100, cashCoveragePercent: 0, adjustedValue: 0 });
    Object.assign(r.valuation, { cashPV: 1000, terminalPV: 0 });
    expect(calculateValuationAttractiveness(r, 0).surplusPercent).toBe(calculateValuationAttractiveness(r, 100).surplusPercent);
    Object.assign(r.valuation, { cashPV: 0, value: 0, status: 'nonpositive' });
    expect(calculateValuationAttractiveness(r)).toMatchObject({ surplusPercent: -100, cashCoveragePercent: 0, adjustedValue: 0 });
  });

  it('withholds every comparison for missing, zero, negative or nonfinite purchase price', () => {
    for (const candidateEquity of [null, 0, -1, NaN, Infinity, -Infinity]) {
      const r = row(); r.valuation.candidateEquity = candidateEquity;
      expect(calculateValuationAttractiveness(r)).toMatchObject({ ...empty, reason: expect.stringMatching(/price/i) });
    }
  });

  it('does not replace invalid terminal credit with the default', () => {
    for (const credit of [-1, 100.01, NaN, Infinity, -Infinity, null, '50']) {
      const result = calculateValuationAttractiveness(row(), credit as number);
      expect(result).toMatchObject({ ...empty, reason: expect.stringMatching(/credit/i) });
      expect(result.terminalCreditPercent).toBe(credit);
    }
  });

  it('requires complete, reconciling finite components and nonnegative terminal value', () => {
    for (const patch of [{ value: null }, { cashPV: null }, { terminalPV: null }, { value: 1001 }, { terminalPV: -1, value: 399 }, { value: Infinity }, { cashPV: NaN }, { terminalPV: Infinity }, { value: Number.MAX_VALUE, cashPV: Number.MAX_VALUE, terminalPV: Number.MAX_VALUE }]) {
      const r = row(); Object.assign(r.valuation, patch);
      expect(calculateValuationAttractiveness(r)).toMatchObject({ ...empty, reason: expect.any(String) });
    }
    const rounded = row(); Object.assign(rounded.valuation, { value: .3, cashPV: .1, terminalPV: .2 });
    expect(calculateValuationAttractiveness(rounded).reason).toBeNull();
  });

  it('withholds only the low comparison when the optional low scenario is missing or out of range', () => {
    for (const lowValue of [null, NaN, Infinity, -Number.MAX_VALUE]) {
      const r = row(); r.valuation.lowValue = lowValue; if (lowValue === -Number.MAX_VALUE) r.valuation.candidateEquity = .1;
      const result = calculateValuationAttractiveness(r);
      expect(result.lowNpvPercent).toBeNull();
      expect(result.surplusPercent).not.toBeNull();
      expect(result.reason).toBeNull();
    }
    const r = row(); r.valuation.lowValue = 0;
    expect(calculateValuationAttractiveness(r).lowNpvPercent).toBe(-100);
  });

  it('retains finite amounts while withholding ratios that overflow for tiny positive prices', () => {
    const r = row(); r.valuation.candidateEquity = Number.MIN_VALUE;
    const result = calculateValuationAttractiveness(r);
    expect(result).toMatchObject({ surplusPercent: null, midNpvPercent: null, dcfPriceRatio: null, cashCoveragePercent: null, lowNpvPercent: null, adjustedValue: 1000, adjustedNpv: 1000, reason: expect.stringMatching(/numerical range/) });
  });

  it('requires a real saved source date without substituting today or comparing quote and valuation currencies', () => {
    const mutations: ((r: ResearchGaugeRow) => void)[] = [
      r => { r.valuation.priceDate = null; }, r => { r.valuation.priceDate = '2026-02-30'; },
      r => { r.valuation.priceDate = '2026-08-11'; }, r => { r.valuation.priceBasis = null; },
      r => { r.valuation.priceBasis!.sourceId = ' '; }, r => { r.valuation.priceBasis!.sourceAsOf = '2026-02-30'; },
      r => { r.valuation.currency = ''; }, r => { r.valuation.currency = 'sek'; },
    ];
    for (const mutate of mutations) { const r = row(); mutate(r); expect(calculateValuationAttractiveness(r)).toMatchObject({ ...empty, reason: expect.any(String) }); }
    const converted = row(); Object.assign(converted.valuation.priceBasis!, { currency: 'USD', method: 'sek', fxRate: 10, fxDate: '2026-02-15' });
    expect(calculateValuationAttractiveness(converted).surplusPercent).toBeCloseTo(300 / 7);
    converted.valuation.priceDate = '2020-02-15';
    expect(calculateValuationAttractiveness(converted).reason).toBeNull();
  });

  it('withholds specialist, unresolved and absent-history rankings while allowing property and limited history', () => {
    const mutations: ((r: ResearchGaugeRow) => void)[] = [
      r => { r.route = 'financial'; }, r => { r.valuation.status = 'manual-financial'; },
      r => { r.route = 'unclassified'; }, r => { r.classificationConflict = true; },
      r => { r.readiness = 'reconcile_data'; }, r => { r.readiness = 'no_history'; },
    ];
    for (const mutate of mutations) { const r = row(); mutate(r); expect(calculateValuationAttractiveness(r)).toMatchObject({ ...empty, reason: expect.any(String) }); }
    const property = row(); property.route = 'property';
    expect(calculateValuationAttractiveness(property).reason).toBeNull();
    expect(calculateValuationAttractiveness(row()).reason).toBeNull();
  });

  it('does not mutate the source row or its frozen valuation inputs', () => {
    const r = row(), before = structuredClone(r);
    calculateValuationAttractiveness(r, 0); calculateValuationAttractiveness(r, 50); calculateValuationAttractiveness(r);
    expect(r).toEqual(before);
  });
});
