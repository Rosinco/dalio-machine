import { describe, expect, it } from 'vitest';
import { blankValuation, calculateScenario } from './valuation';
import { resolveTerminalSale, seedTerminalCash } from './terminalValue';
import { buildCrisisScenario, defaultCrisisAssumptions } from './crisisScenario';
import { decodeValuation } from './savedValuations';
import { shouldStartResearchedStudy } from './researchedValuations';

function study() {
  const d = blankValuation('Example', 'SEK', '2026-09-13');
  Object.assign(d, { marketCap: 1000, priceDate: d.valuationDate, priceSource: 'Test ownership basis' });
  Object.assign(d.scenarios.mid, { cashFlows: Array(10).fill(100), discountRate: 10, terminalEquity: 99999, terminalCash: { cashFlow: 100, growthRate: 0 } });
  return d;
}
describe('separate sustainable terminal cash', () => {
  it('values post-horizon cash once and ignores a stale manual-sale cache', () => {
    const d = study(), r = calculateScenario(d, 'mid');
    expect(r.error).toBeNull();
    expect(r.value).toBeCloseTo(1000);
    expect(r.terminalPV).toBeCloseTo(1000 / 1.1 ** 10);
    expect(r.cumulativeNPVWithSale[9]).toBe(r.cumulativeNPV[9]);
    expect(r.cumulativeNPVWithSale[10] - r.cumulativeNPV[10]).toBeCloseTo(r.terminalPV!);
    d.scenarios.mid.cashFlows[9] = 10000;
    expect(calculateScenario(d, 'mid').terminalPV).toBe(r.terminalPV);
  });
  it('uses first post-horizon cash directly and responds to required return and growth', () => {
    const s = study().scenarios.mid;
    s.terminalCash!.growthRate = 2;
    expect(resolveTerminalSale(s).value).toBeCloseTo(1250);
    s.discountRate = 12;
    expect(resolveTerminalSale(s).value).toBeCloseTo(1000);
  });
  it('retains deliberate missing inputs and rejects invalid perpetual-growth arithmetic', () => {
    for (const patch of [{ cashFlow: null }, { growthRate: null }, { growthRate: 10 }, { growthRate: 11 }, { growthRate: -100 }]) {
      const d = study(); Object.assign(d.scenarios.mid.terminalCash!, patch);
      expect(calculateScenario(d, 'mid').error).toMatch(/terminal|sustainable|growth/i);
    }
    const s = study().scenarios.mid; s.terminalCash!.cashFlow = -20;
    expect(resolveTerminalSale(s).value).toBe(0);
    delete s.terminalCash; s.terminalEquity = 0;
    expect(resolveTerminalSale(s).value).toBe(0);
  });
  it('seeds signed medians with a separate assumed spread without removing shock years', () => {
    expect(seedTerminalCash([705, -181, -561, 840, 1027])).toEqual({ median: 705, low: 564, mid: 705, high: 846 });
    expect(seedTerminalCash([-10, -20, -30])).toEqual({ median: -20, low: -24, mid: -20, high: -16 });
    expect(seedTerminalCash([10, 20])).toEqual({ median: null, low: null, mid: null, high: null });
    expect(seedTerminalCash([10, null, 30])).toEqual({ median: null, low: null, mid: null, high: null });
    expect(seedTerminalCash([10, 20, 30, 40]).median).toBe(25);
    expect(seedTerminalCash([1e12, 1e12, 1e12])).toEqual({ median: 1e12, low: 8e11, mid: 1e12, high: null });
  });
  it('keeps crisis sale independent of the mid sustainable-cash model', () => {
    const d = study(); d.crisis = { ...defaultCrisisAssumptions(), enabled: true, terminalEquity: 50 };
    expect(buildCrisisScenario(d).result.terminalPV).toBeCloseTo(50 / 1.1 ** 10);
  });
  it('round-trips terminal inputs including deliberate blanks and rejects corrupt metadata', () => {
    const saved = { format: 'macro-atlas-valuation', version: 1, id: 'terminal-test', company: '696', created: '2026-09-13T00:00:00.000Z', release: 'a'.repeat(64), financial: null, taxonomy: null, draft: study() };
    saved.draft.scenarios.mid.terminalCash!.cashFlow = null;
    expect(decodeValuation(JSON.stringify(saved)).draft).toEqual(saved.draft);
    const bad = structuredClone(saved); (bad.draft.scenarios.mid.terminalCash as unknown as { cashFlow: string }).cashFlow = '100';
    expect(() => decodeValuation(JSON.stringify(bad))).toThrow(/unsupported valuation/);
    const extra = structuredClone(saved); Object.assign(extra.draft.scenarios.mid.terminalCash!, { guessed: true });
    expect(() => decodeValuation(JSON.stringify(extra))).toThrow(/unsupported valuation/);
  });
  it('preserves a terminal-only edit on an otherwise blank study', () => {
    const d = blankValuation('Example');
    d.scenarios.mid.terminalCash = { cashFlow: null, growthRate: 0 };
    expect(shouldStartResearchedStudy(d)).toBe(false);
  });
});
