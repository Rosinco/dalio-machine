import { describe, expect, it } from 'vitest';
import { blankValuation, calculateValuation, type ScenarioResult, type ValuationDraft } from './valuation';
import { buildValuationRangeRows, scenarioRangeGeometry } from './scenarioRanges';

function fixture(): ValuationDraft {
  const draft = blankValuation('Range example', 'SEK', '2026-09-13');
  draft.years = 2; draft.marketCap = 1000; draft.investment = 250;
  draft.priceDate = draft.valuationDate; draft.priceSource = 'Hypothetical equity price';
  for (const scenario of Object.values(draft.scenarios)) {
    scenario.cashFlows = [100, 200]; scenario.discountRate = 0;
  }
  return draft;
}

describe('valuation ranges from complete named scenario results', () => {
  it('preserves signed DCF, each scenario return, and investment scaling when names cross', () => {
    const draft = fixture();
    draft.scenarios.low.cashFlows = [-100, -100]; draft.scenarios.low.discountRate = 20;
    draft.scenarios.mid.cashFlows = [-80, -80];
    draft.scenarios.high.cashFlows = [-70, -70];
    const results = calculateValuation(draft).scenarios;
    const rows = buildValuationRangeRows(draft, results, 'dcf')!;
    expect(rows.map(row => row.year)).toEqual([1, 2]);
    expect(rows[0].low).toBeCloseTo(-100 / 1.2 / 4);
    expect(rows[0].min).toBeCloseTo(rows[0].low);
    expect(rows[1].low).toBeCloseTo(-100 / 1.2 ** 2 / 4);
    expect(rows[1].mid).toBe(-20);
    expect(rows[1].high).toBe(-17.5);
    expect(rows[1].min).toBe(-20);
    expect(rows[1].max).toBeCloseTo(rows[1].low);
  });

  it('forms NPV bounds from cumulative scenarios instead of summing unrelated annual extrema', () => {
    const draft = fixture();
    draft.scenarios.low.cashFlows = [0, 200];
    draft.scenarios.mid.cashFlows = [100, 100];
    draft.scenarios.high.cashFlows = [200, 0];
    const rows = buildValuationRangeRows(draft, calculateValuation(draft).scenarios, 'npv')!;
    expect(rows).toEqual([
      { year: 0, low: -250, mid: -250, high: -250, min: -250, max: -250 },
      { year: 1, low: -250, mid: -225, high: -200, min: -250, max: -200 },
      { year: 2, low: -200, mid: -200, high: -200, min: -200, max: -200 },
    ]);
  });

  it('adds each discounted terminal sale only to final NPV and never to annual DCF', () => {
    const draft = fixture();
    draft.scenarios.low.terminalEquity = 400; draft.scenarios.low.discountRate = 100;
    draft.scenarios.mid.terminalEquity = 100;
    draft.scenarios.high.terminalEquity = 0;
    const results = calculateValuation(draft).scenarios;
    const withoutSale = buildValuationRangeRows(draft, results, 'npv')!;
    const withSale = buildValuationRangeRows(draft, results, 'npv', true)!;
    expect(withSale.slice(0, 2)).toEqual(withoutSale.slice(0, 2));
    expect(withSale[2].low - withoutSale[2].low).toBe(400 / 2 ** 2 / 4);
    expect(withSale[2].mid - withoutSale[2].mid).toBe(100 / 4);
    expect(withSale[2].high).toBe(withoutSale[2].high);
    expect(buildValuationRangeRows(draft, results, 'dcf', true)).toEqual(buildValuationRangeRows(draft, results, 'dcf'));
    expect(withSale[2].mid).toBe(results.mid.npv! / 4);
  });

  it('uses supplied results without regenerating forecasts or mutating inputs', () => {
    const draft = fixture(), results = calculateValuation(draft).scenarios;
    results.low.discounted = [700, 800];
    const beforeDraft = structuredClone(draft), beforeResults = structuredClone(results);
    expect(buildValuationRangeRows(draft, results, 'dcf')?.map(row => row.low)).toEqual([175, 200]);
    expect(draft).toEqual(beforeDraft); expect(results).toEqual(beforeResults);
  });

  it('rejects missing, errored, nonfinite and incorrectly sized selected paths', () => {
    const draft = fixture();
    const changes: ((results: Record<string, ScenarioResult>) => void)[] = [
      results => { delete results.low; },
      results => { results.low.error = 'Missing assumptions'; },
      results => { results.low.discounted = []; },
      results => { results.low.discounted.push(0); },
      results => { results.low.discounted[0] = NaN; },
      results => { results.low.discounted[1] = Infinity; },
      results => { delete results.low.discounted[0]; },
    ];
    for (const change of changes) {
      const results = calculateValuation(draft).scenarios;
      change(results);
      expect(buildValuationRangeRows(draft, results, 'dcf')).toBeNull();
    }
    const results = calculateValuation(draft).scenarios;
    results.mid.cumulativeNPVWithSale.pop();
    expect(buildValuationRangeRows(draft, results, 'npv', true)).toBeNull();
    expect(buildValuationRangeRows(draft, results, 'npv')).not.toBeNull();
  });

  it('rejects invalid scale, horizon and scaled overflow while retaining actual zero cash', () => {
    const invalid: ((draft: ValuationDraft) => void)[] = [
      draft => { draft.marketCap = null; }, draft => { draft.marketCap = 0; },
      draft => { draft.marketCap = -1; }, draft => { draft.marketCap = Infinity; },
      draft => { draft.investment = null; }, draft => { draft.investment = 0; },
      draft => { draft.investment = -1; }, draft => { draft.investment = NaN; },
      draft => { draft.years = 0; }, draft => { draft.years = 51; },
      draft => { draft.years = 1.5; },
    ];
    for (const change of invalid) {
      const draft = fixture(), results = calculateValuation(draft).scenarios;
      change(draft);
      expect(buildValuationRangeRows(draft, results, 'dcf')).toBeNull();
    }
    const draft = fixture(); draft.investment = 2000;
    const results = calculateValuation(draft).scenarios;
    results.mid.discounted[0] = Number.MAX_VALUE;
    expect(buildValuationRangeRows(draft, results, 'dcf')).toBeNull();
    results.mid.discounted[0] = 0;
    expect(buildValuationRangeRows(draft, results, 'dcf')?.[0].mid).toBe(0);
  });
});

describe('scenario envelope geometry', () => {
  it('narrows to the real linear envelope at crossings without reassigning scenarios', () => {
    const paths = [[0, 10], [5, 5], [10, 0]];
    expect(scenarioRangeGeometry(paths)).toEqual([{
      lower: [[0, 0], [0.5, 5], [1, 0]],
      upper: [[0, 10], [0.5, 5], [1, 10]],
    }]);
    expect(paths).toEqual([[0, 10], [5, 5], [10, 0]]);
  });

  it('handles distinct crossing times and unequal negative paths without using a zero baseline', () => {
    const geometry = scenarioRangeGeometry([[-10, -2], [-6, -6], [-3, -9]])[0];
    expect(geometry.lower[0]).toEqual([0, -10]);
    expect(geometry.upper[0]).toEqual([0, -3]);
    expect(geometry.lower.at(-1)).toEqual([1, -9]);
    expect(geometry.upper.at(-1)).toEqual([1, -2]);
    for (const point of [...geometry.lower, ...geometry.upper]) expect(point[1]).toBeLessThan(0);
    // At t = 1/2 all three paths meet at -6.
    expect(geometry.upper).toContainEqual([0.5, -6]);
    expect(geometry.lower).toContainEqual([0.5, -6]);
    const distinct = scenarioRangeGeometry([[0, 12], [4, 4], [10, 0]])[0];
    const crossings = [0, 1 / 3, 5 / 11, 3 / 5, 1];
    expect(distinct.upper).toHaveLength(crossings.length);
    crossings.forEach((x, index) => expect(distinct.upper[index][0]).toBeCloseTo(x, 14));
  });

  it('breaks at missing and nonfinite values, preserving isolated complete years', () => {
    const geometry = scenarioRangeGeometry([[1, null, 3, 4, Infinity, 6], [2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8]]);
    expect(geometry).toEqual([
      { lower: [[0, 1]], upper: [[0, 3]] },
      { lower: [[2, 3], [3, 4]], upper: [[2, 5], [3, 6]] },
      { lower: [[5, 6]], upper: [[5, 8]] },
    ]);
    expect(scenarioRangeGeometry([[1, NaN, 3], [2, 3]])).toEqual([{ lower: [[0, 1]], upper: [[0, 2]] }]);
    expect(scenarioRangeGeometry([])).toEqual([]);
    expect(scenarioRangeGeometry([[], [1]])).toEqual([]);
  });

  it('keeps each prior NPV bound until the next year-end, including a final sale jump', () => {
    expect(scenarioRangeGeometry([[-100, -90, 10], [-100, -80, -20], [-100, -70, -10]], true)).toEqual([{
      lower: [[0, -100], [1, -100], [1, -90], [2, -90], [2, -20]],
      upper: [[0, -100], [1, -100], [1, -70], [2, -70], [2, 10]],
    }]);
  });

  it('retains zero-width ranges and returns finite geometry for large signed values', () => {
    expect(scenarioRangeGeometry([[0, 0], [0, 0]])).toEqual([{ lower: [[0, 0], [1, 0]], upper: [[0, 0], [1, 0]] }]);
    const geometry = scenarioRangeGeometry([[-1e308, 1e308], [1e308, -1e308], [0, 0]])[0];
    expect(geometry.lower).toEqual([[0, -1e308], [0.5, 0], [1, -1e308]]);
    expect(geometry.upper).toEqual([[0, 1e308], [0.5, 0], [1, 1e308]]);
  });
});
