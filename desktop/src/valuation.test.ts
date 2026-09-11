import { describe, expect, it } from 'vitest';
import { blankValuation, calculateScenario, calculateValuation, generateCashFlows, type ValuationDraft } from './valuation';
import { compatibleValuation, decodeValuation, loadValuationDraft, loadValuations, saveValuation, saveValuationDraft, type SavedValuation } from './savedValuations';

function draft(): ValuationDraft {
  const d = blankValuation('Example', 'SEK', '2026-09-11');
  d.marketCap = 1000; d.priceDate = d.valuationDate; d.priceSource = 'Hypothetical equity price';
  d.years = 3;
  for (const s of Object.values(d.scenarios)) { s.cashFlows = [400, 400, 400]; s.discountRate = 10; }
  return d;
}
describe('shareholder cash flow valuation and payback', () => {
  it('discounts dated annual payments, subtracts the price once, and measures end-year payback', () => {
    const d = draft(), r = calculateScenario(d, 'mid');
    expect(r.error).toBeNull();
    expect(r.discounted).toEqual([400 / 1.1, 400 / 1.1 ** 2, 400 / 1.1 ** 3]);
    expect(r.value).toBeCloseTo(994.7407964);
    expect(r.npv).toBeCloseTo(-5.2592036);
    expect(r.cumulativeNPV[0]).toBe(-1000);
    expect(r.cumulativeNPV.at(-1)).toBeCloseTo(r.npv!);
    expect(r.payback.year).toBe(3); expect(r.discountedPayback.year).toBeNull();
  });
  it('keeps a modeled terminal sale out of cash-distribution payback and shows it separately', () => {
    const d = draft(); d.scenarios.mid.cashFlows = [0, 0, 0]; d.scenarios.mid.terminalEquity = 2000;
    const r = calculateScenario(d, 'mid');
    expect(r.value).toBeCloseTo(2000 / 1.1 ** 3);
    expect(r.payback.year).toBeNull(); expect(r.discountedPayback.year).toBeNull();
    expect(r.paybackWithSale.year).toBe(3); expect(r.discountedPaybackWithSale.year).toBe(3);
    expect(r.cumulativeNPV).toEqual([-1000, -1000, -1000, -1000]);
    expect(r.cumulativeNPVWithSale.at(-1)).toBeCloseTo(r.npv!);
    expect(r.terminalShare).toBe(1);
  });
  it('keeps recovery as an alternative net-equity scenario, including zero recovery', () => {
    const d = draft(), before = calculateScenario(d, 'mid').value;
    d.scenarios.mid.recoveryEquity = 800; d.scenarios.mid.recoveryYear = 2;
    let r = calculateScenario(d, 'mid');
    expect(r.recovery!.value).toBeCloseTo(800 / 1.1 ** 2);
    expect(r.recovery!.npv).toBeCloseTo(800 / 1.1 ** 2 - 1000);
    expect(r.value).toBe(before); expect(r.recovery!.payback).toBeNull();
    d.scenarios.mid.recoveryEquity = 0;
    r = calculateScenario(d, 'mid'); expect(r.recovery!.value).toBe(0); expect(r.recovery!.npv).toBe(-1000);
  });
  it('preserves zero and negative cash flows and warns when payback subsequently reverses', () => {
    const d = draft(); d.scenarios.mid.cashFlows = [1000, -500, 600]; d.scenarios.mid.discountRate = 0;
    const r = calculateScenario(d, 'mid');
    expect(r.value).toBe(1100); expect(r.payback).toEqual({ year: 1, reversed: true });
    expect(r.discountedPayback).toEqual(r.payback);
    expect(r.cumulativeNPV).toEqual([-1000, 0, -500, 100]);
  });
  it('does not infer cash flows, price or return assumptions from missing data', () => {
    expect(calculateScenario(blankValuation('Example', 'SEK', '2026-09-11'), 'mid').value).toBeNull();
    const d = draft(); d.scenarios.mid.cashFlows[1] = null;
    expect(calculateScenario(d, 'mid').error).toMatch(/every forecast year/);
    d.scenarios.mid.cashFlows[1] = 0; expect(calculateScenario(d, 'mid').value).not.toBeNull();
    d.marketCap = 0; expect(calculateScenario(d, 'mid').error).toMatch(/positive equity/);
  });
  it('rejects invalid dates, rates, horizons and numeric overflow instead of producing a chart', () => {
    const cases: ((d: ValuationDraft) => void)[] = [
      d => { d.priceDate = '2026-02-30'; }, d => { d.priceDate = '2026-09-12'; },
      d => { d.years = 0; }, d => { d.scenarios.mid.discountRate = -100; },
      d => { d.scenarios.mid.cashFlows[0] = Infinity; }, d => { d.marketCap = 1e-300; },
    ];
    for (const change of cases) { const d = draft(); change(d); expect(calculateScenario(d, 'mid').error).not.toBeNull(); }
  });
  it('scales company values to the selected investment without applying corporate debt twice', () => {
    const d = draft(); d.investment = 2000;
    d.capital.grossDebt = 500; d.capital.surplusCash = 100;
    let r = calculateScenario(d, 'mid'); expect(r.stakeValue).toBeCloseTo(r.value! * 2);
    d.capital.grossDebt = 900; expect(calculateScenario(d, 'mid').value).toBe(r.value);
    d.marketCap = 2000; r = calculateScenario(d, 'mid'); expect(r.valuePrice).toBeCloseTo(r.value! / 2000);
  });
  it('preserves named paths when scenarios cross; their range is not a confidence interval', () => {
    const d = draft(); d.scenarios.low.cashFlows = [700, 100, 100];
    const r = calculateValuation(d); expect(r.crossing).toBe(true);
    expect(r.scenarios.low.discounted[0]).toBeCloseTo(700 / 1.1);
  });
  it('generates an explicit editable cash-flow path, including shrinking businesses', () => {
    expect(generateCashFlows(100, -50, 3)).toEqual([100, 50, 25]);
    expect(generateCashFlows(0, 0, 3)).toEqual([0, 0, 0]);
    expect(() => generateCashFlows(100, -101, 3)).toThrow();
  });
});

const saved = (): SavedValuation => ({ format: 'macro-atlas-valuation', version: 1, id: 'example-1', created: '2026-09-11T12:00:00.000Z',
  company: '102', release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), draft: draft() });
function storage(initial: string | null = null) { let value = initial; return { getItem: () => value, setItem: (_key: string, v: string) => { value = v; } }; }
describe('saved valuation studies', () => {
  it('retains every scenario, Swedish notes and the exact company/data identities through reopening', () => {
    const s = storage(), v = saved(); v.draft.notes.macro = 'Räntor, lån och utländsk efterfrågan';
    saveValuation(s, v); expect(loadValuations(s)).toEqual([v]);
    expect(decodeValuation(JSON.stringify(v))).toEqual(v);
    expect(compatibleValuation(v, '102', v.release, v.financial, v.taxonomy)).toBe(true);
    expect(compatibleValuation(v, '696', v.release, v.financial, v.taxonomy)).toBe(false);
    expect(compatibleValuation(v, '102', 'd'.repeat(64), v.financial, v.taxonomy)).toBe(false);
  });
  it('allows incomplete drafts without inventing values', () => {
    const v = saved(), s = storage(); v.draft = blankValuation('Unfinished', 'SEK', '2026-09-11');
    saveValuation(s, v); expect(loadValuations(s)[0].draft.marketCap).toBeNull();
  });
  it('rejects malformed or duplicate imports without overwriting existing work', () => {
    const s = storage(), v = saved(); saveValuation(s, v);
    expect(() => saveValuation(s, v)).toThrow();
    expect(() => decodeValuation(JSON.stringify({ ...v, version: 2 }))).toThrow();
    expect(() => decodeValuation(JSON.stringify({ ...v, draft: { ...v.draft, years: 50000 } }))).toThrow();
    expect(() => decodeValuation(JSON.stringify({ ...v, draft: { ...v.draft, scenarios: { low: {} } } }))).toThrow();
    expect(() => decodeValuation(JSON.stringify({ ...v, draft: { ...v.draft, researchOrigin: { id: 'holmen', asOf: '2026-02-30' } } }))).toThrow();
    expect(loadValuations(s)).toEqual([v]);
    const bad = storage('{broken'); expect(() => saveValuation(bad, v)).toThrow(); expect(bad.getItem()).toBe('{broken');
  });
  it('reports persistence failures and preserves zero-recovery assumptions', () => {
    const v = saved(); v.draft.scenarios.mid.recoveryEquity = 0; v.draft.scenarios.mid.recoveryYear = 1;
    expect(decodeValuation(JSON.stringify(v)).draft.scenarios.mid.recoveryEquity).toBe(0);
    expect(() => saveValuation({ getItem: () => null, setItem: () => { throw new Error('Storage full'); } }, v)).toThrow('Storage full');
  });
  it('autosaves drafts independently for each company and source version', () => {
    const data = new Map<string, string>();
    const s = { getItem: (key: string) => data.get(key) ?? null, setItem: (key: string, value: string) => { data.set(key, value); } };
    const v = saved(); saveValuationDraft(s, v);
    const other = { ...saved(), company: '696' }; other.draft.marketCap = 2000; saveValuationDraft(s, other);
    expect(loadValuationDraft(s, v)?.draft.marketCap).toBe(1000);
    expect(loadValuationDraft(s, other)?.draft.marketCap).toBe(2000);
    expect(loadValuationDraft(s, { ...v, financial: 'd'.repeat(64) })).toBeNull();
    const raw = [...data.entries()].find(([key]) => key.includes(':102:'))!;
    data.set(raw[0], '{broken');
    expect(() => saveValuationDraft(s, v)).toThrow(); expect(data.get(raw[0])).toBe('{broken');
  });
});
