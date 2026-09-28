import { describe, expect, it } from 'vitest';
import type { ResearchGaugeRow } from './researchGaugeModel';
import { calculateNormalYearFiveYearValuation, calculateNormalYearValuation, NORMAL_YEAR_FIVE_YEAR_ASSUMPTIONS, normalYearValues, selectNormalYearPeriods } from './normalYearScreen';

function row(): ResearchGaugeRow {
  const periods = Array.from({ length: 10 }, (_, i) => 2025 - i).map(year => ({ year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, published: `${year + 1}-02-01`, currency: 'SEK', sourceId: 'annual', sourceAsOf: '2026-08-10' }));
  return { id: '1', route: 'operating', classificationConflict: false, presence: 'latest', sourceAsOf: '2026-08-10', annual: { latest: periods[0] },
    screeningAnnual: { asOf: '2026-09-13', periods, cash: [100, 120, 500, -500, null, 0, 80, 90, 110, 999], operatingCash: periods.map(() => 140), ebit: periods.map(() => 30), revenue: periods.map(() => 200), equity: periods.map(() => 100), netDebt: periods.map(() => -20), assets: periods.map(() => 300), intangibleAssets: periods.map(() => 50), tangibleAssets: periods.map(() => 80), profit: periods.map(() => 25), reason: null },
    valuation: { currency: 'SEK', candidateEquity: 700, priceDate: '2026-02-09', priceBasis: { sourceId: 'annual', sourceAsOf: '2026-08-10', currency: 'SEK', shares: 7, close: 100, method: 'local', fxRate: null, fxDate: null } },
  } as unknown as ResearchGaugeRow;
}

describe('explicit 2020–2023 exception screening', () => {
  it('selects five normal fiscal end years and preserves both positive and negative exception facts', () => {
    const r = row(), before = structuredClone(r), selected = selectNormalYearPeriods(r);
    expect(selected.reason).toBeNull();
    expect(selected.periods.map(p => p.year)).toEqual([2025, 2024, 2019, 2018, 2017]);
    expect(selected.excluded.map(p => p.year)).toEqual([2023, 2022, 2021, 2020]);
    expect(normalYearValues(r, 'fcf', selected)).toEqual([100, 120, 80, 90, 110]);
    expect(r).toEqual(before);
  });
  it('does not substitute an older normal period for a missing required observation', () => {
    const r = row(); r.screeningAnnual!.cash[6] = null;
    expect(normalYearValues(r, 'fcf', selectNormalYearPeriods(r))).toEqual([100, 120, null, 90, 110]);
    expect(calculateNormalYearValuation(r).reason).toMatch(/missing/i);
  });
  it('uses actual fiscal end dates for the exception boundary', () => {
    const r = row();
    r.screeningAnnual!.periods.forEach(p => { p.start = `${p.year - 1}-04-01`; p.end = `${p.year}-03-31`; p.published = `${p.year}-05-01`; });
    expect(selectNormalYearPeriods(r).periods.map(p => p.end)).toEqual(['2025-03-31', '2024-03-31', '2019-03-31', '2018-03-31', '2017-03-31']);
    // Provider fiscal labels may differ from the actual calendar year of the end.
    r.screeningAnnual!.periods.forEach(p => p.year--);
    expect(selectNormalYearPeriods(r).periods.map(p => p.end)).toEqual(['2025-03-31', '2024-03-31', '2019-03-31', '2018-03-31', '2017-03-31']);
  });
  it('fails closed on insufficient, stale, unpublished or incompatible required evidence', () => {
    for (const mutate of [
      (r: ResearchGaugeRow) => { delete r.screeningAnnual; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods = r.screeningAnnual!.periods.slice(0, 8); },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.asOf = '2027-08-01'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[6].published = null; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[3].currency = 'USD'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[3].end = '2023-01-05'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[0].published = '2026-10-01'; },
    ]) { const r = row(); mutate(r); expect(selectNormalYearPeriods(r).reason).not.toBeNull(); }
  });
  it('calculates capital proxies from positive same-report denominators without inventing ROIC', () => {
    const r = row();
    expect(normalYearValues(r, 'normal_roce', selectNormalYearPeriods(r))).toEqual(Array(5).fill(37.5));
    expect(normalYearValues(r, 'normal_rota', selectNormalYearPeriods(r))).toEqual(Array(5).fill(10));
    expect(normalYearValues(r, 'tangible_assets_revenue', selectNormalYearPeriods(r))).toEqual(Array(5).fill(.4));
    r.screeningAnnual!.equity[0] = 20; r.screeningAnnual!.intangibleAssets[1] = 300;
    expect(normalYearValues(r, 'normal_roce', selectNormalYearPeriods(r))![0]).toBeNull();
    expect(normalYearValues(r, 'normal_rota', selectNormalYearPeriods(r))![1]).toBeNull();
  });
});

describe('separate normal-year valuation sensitivity', () => {
  it('uses a flat median, ten years, 10% discount and 0% perpetual growth with terminal once', () => {
    const r = row(), before = structuredClone(r), result = calculateNormalYearValuation(r);
    const cashPV = Array.from({ length: 10 }, (_, i) => 100 / 1.1 ** (i + 1)).reduce((sum, cash) => sum + cash, 0);
    const terminalPV = 100 / .1 / 1.1 ** 10;
    expect(result).toMatchObject({ reason: null, normalCash: 100, currency: 'SEK' });
    expect(result.cashPV).toBeCloseTo(cashPV); expect(result.terminalPV).toBeCloseTo(terminalPV);
    expect(result.value).toBeCloseTo(1000); expect(result.surplusPercent).toBeCloseTo(100 * (1000 / 700 - 1));
    expect(calculateNormalYearValuation(r, 50).value).toBeCloseTo(cashPV + terminalPV / 2);
    expect(calculateNormalYearValuation(r, 0).value).toBeCloseTo(cashPV);
    expect(r).toEqual(before);
  });
  it('retains zero and signed funding, including a continuing cash liability', () => {
    const r = row(); r.screeningAnnual!.cash.fill(-100);
    expect(calculateNormalYearValuation(r).value).toBeCloseTo(-1000);
    expect(calculateNormalYearValuation(r).terminalPV).toBeLessThan(0);
    r.screeningAnnual!.cash.fill(0);
    expect(calculateNormalYearValuation(r)).toMatchObject({ normalCash: 0, cashPV: 0, terminalPV: 0, value: 0, surplusPercent: -100 });
  });
  it('withholds unmatched currency, missing or inconsistent prices and unsuitable business routes', () => {
    for (const mutate of [
      (r: ResearchGaugeRow) => { r.valuation.currency = 'USD'; },
      (r: ResearchGaugeRow) => { r.valuation.candidateEquity = null; },
      (r: ResearchGaugeRow) => { r.valuation.priceBasis = null; },
      (r: ResearchGaugeRow) => { r.valuation.priceBasis!.close = 200; },
      (r: ResearchGaugeRow) => { r.valuation.priceDate = '2026-09-15'; },
      (r: ResearchGaugeRow) => { r.valuation.priceDate = '2024-01-01'; },
      (r: ResearchGaugeRow) => { r.route = 'financial'; },
      (r: ResearchGaugeRow) => { r.classificationConflict = true; },
    ]) { const r = row(); mutate(r); expect(calculateNormalYearValuation(r).reason).not.toBeNull(); }
    expect(calculateNormalYearValuation(row(), 25).reason).not.toBeNull();
  });
  it('keeps normal cash values in report currency when a compatible saved price is unavailable', () => {
    const r = row(); r.valuation.candidateEquity = null; r.valuation.priceBasis = null;
    expect(calculateNormalYearValuation(r)).toMatchObject({ normalCash: 100, currency: 'SEK', surplusPercent: null });
    expect(calculateNormalYearValuation(r).value).toBeCloseTo(1000);
    r.valuation.currency = 'USD';
    expect(calculateNormalYearValuation(r).cashPV).toBeGreaterThan(0);
    expect(calculateNormalYearValuation(r).terminalPV).toBeGreaterThan(0);
  });
});

describe('separate five-year normal cash valuation', () => {
  it('discounts only the five annual median cash payments and assigns no terminal value', () => {
    const result = calculateNormalYearFiveYearValuation(row());
    const fiveYearFactor = (1 - 1.1 ** -5) / .1;
    expect(result).toMatchObject({ normalCash: 100, terminalPV: 0, currency: 'SEK', reason: null });
    expect(result.cashPV).toBeCloseTo(100 * fiveYearFactor, 10);
    expect(result.value).toBe(result.cashPV);
    expect(result.surplusPercent).toBeCloseTo(100 * (100 * fiveYearFactor / 700 - 1), 10);
    expect(result.surplusPercent).toBeLessThan(0);
    expect(NORMAL_YEAR_FIVE_YEAR_ASSUMPTIONS).toMatch(/years 1–5/i);
    expect(NORMAL_YEAR_FIVE_YEAR_ASSUMPTIONS).toMatch(/no terminal/i);
  });

  it('preserves each original ten-year result and all source and authored fields exactly', () => {
    const r = row(), before = structuredClone(r);
    const cashPV = Array.from({ length: 10 }, (_, i) => 100 / 1.1 ** (i + 1)).reduce((sum, cash) => sum + cash, 0);
    const terminalPV = (100 / .1) / 1.1 ** 10;
    calculateNormalYearFiveYearValuation(r);
    for (const credit of [0, 50, 100]) {
      const value = cashPV + terminalPV * credit / 100;
      expect(calculateNormalYearValuation(r, credit)).toEqual({ normalCash: 100, cashPV, terminalPV, value, surplusPercent: 100 * (value / 700 - 1), currency: 'SEK', reason: null });
    }
    expect(r).toEqual(before);
  });

  it('retains zero and negative five-year cash without continuing payments after year five', () => {
    const r = row(); r.screeningAnnual!.cash.fill(-100);
    const negative = calculateNormalYearFiveYearValuation(r);
    expect(negative.value).toBeCloseTo(-379.0786769408446, 10);
    expect(negative.terminalPV).toBe(0);
    expect(negative.surplusPercent).toBeLessThan(-100);
    r.screeningAnnual!.cash.fill(0);
    expect(calculateNormalYearFiveYearValuation(r)).toEqual({ normalCash: 0, cashPV: 0, terminalPV: 0, value: 0, surplusPercent: -100, currency: 'SEK', reason: null });
  });

  it('uses the same selected normal reports without substituting missing cash', () => {
    const r = row(), initial = calculateNormalYearFiveYearValuation(r);
    for (const i of [2, 3, 4, 5]) r.screeningAnnual!.cash[i] = 1e6;
    expect(calculateNormalYearFiveYearValuation(r)).toEqual(initial);
    r.screeningAnnual!.cash[6] = null;
    expect(calculateNormalYearFiveYearValuation(r)).toMatchObject({ normalCash: null, cashPV: null, value: null, surplusPercent: null });
    expect(calculateNormalYearFiveYearValuation(r).reason).toMatch(/missing/i);
  });

  it('keeps five-year report-currency PV when the saved price comparison is invalid', () => {
    for (const mutate of [
      (r: ResearchGaugeRow) => { r.valuation.candidateEquity = null; },
      (r: ResearchGaugeRow) => { r.valuation.currency = 'USD'; },
      (r: ResearchGaugeRow) => { r.valuation.priceBasis = null; },
      (r: ResearchGaugeRow) => { r.valuation.priceBasis!.close = 200; },
      (r: ResearchGaugeRow) => { r.valuation.priceDate = '2024-01-01'; },
      (r: ResearchGaugeRow) => { r.valuation.priceDate = '2026-09-15'; },
    ]) {
      const r = row(); mutate(r);
      const result = calculateNormalYearFiveYearValuation(r);
      expect(result).toMatchObject({ normalCash: 100, currency: 'SEK', terminalPV: 0, surplusPercent: null });
      expect(result.cashPV).toBeCloseTo(379.0786769408446, 10);
      expect(result.value).toBe(result.cashPV);
      expect(result.reason).not.toBeNull();
    }
  });

  it('withholds unsupported business routes and stale or incompatible evidence', () => {
    for (const mutate of [
      (r: ResearchGaugeRow) => { r.route = 'financial'; },
      (r: ResearchGaugeRow) => { r.classificationConflict = true; },
      (r: ResearchGaugeRow) => { r.presence = 'older'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.asOf = '2027-08-01'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[6].currency = 'USD'; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.periods[6].published = null; },
    ]) {
      const r = row(); mutate(r);
      expect(calculateNormalYearFiveYearValuation(r)).toMatchObject({ cashPV: null, value: null, surplusPercent: null });
      expect(calculateNormalYearFiveYearValuation(r).reason).not.toBeNull();
    }
  });

  it('does not inherit an overflowing ten-year valuation when five-year cash is finite', () => {
    const r = row(); r.screeningAnnual!.cash.fill(4e307);
    expect(calculateNormalYearValuation(r).value).toBeNull();
    const fiveYear = calculateNormalYearFiveYearValuation(r);
    expect(fiveYear.reason).toBeNull();
    expect(Number.isFinite(fiveYear.value)).toBe(true);
    expect(Number.isFinite(fiveYear.surplusPercent)).toBe(true);
    expect(fiveYear.terminalPV).toBe(0);
  });

  it('retains the same value ordering as ten-year flat-cash comparisons without implying equal NPV amounts', () => {
    const observations = [[90, 700], [120, 900], [100, 700]].map(([cash, price]) => {
      const r = row(); r.screeningAnnual!.cash.fill(cash);
      r.valuation.candidateEquity = price; r.valuation.priceBasis!.shares = 1; r.valuation.priceBasis!.close = price;
      return { cash, price, five: calculateNormalYearFiveYearValuation(r).surplusPercent!, ten: calculateNormalYearValuation(r, 50).surplusPercent! };
    });
    const byFive = [...observations].sort((a, b) => b.five - a.five), byTen = [...observations].sort((a, b) => b.ten - a.ten);
    expect(byFive.map(row => [row.cash, row.price])).toEqual(byTen.map(row => [row.cash, row.price]));
    expect(observations.every(row => row.five < row.ten)).toBe(true);
  });
});
