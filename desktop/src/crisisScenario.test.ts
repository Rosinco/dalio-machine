import { describe, expect, it } from 'vitest';
import { blankValuation, calculateScenario } from './valuation';
import { buildCrisisScenario, defaultCrisisAssumptions } from './crisisScenario';

describe('separate crisis scenario', () => {
  it('reduces signed cash through the shock and recovers to the contemporaneous mid path', () => {
    const draft = blankValuation('Example');
    draft.years = 6; draft.scenarios.mid.cashFlows = [100, 100, -100, 100, 100, 100];
    draft.crisis = { ...defaultCrisisAssumptions(), enabled: true, shockPercent: 50, startYear: 2, durationYears: 2, recoveryYears: 2, extraAnnualCashCost: 20 };
    const before = structuredClone(draft);
    expect(buildCrisisScenario(draft).cashFlows).toEqual([100, 30, -170, 65, 100, 100]);
    expect(draft).toEqual(before);
  });
  it('leaves unavailable mid cash unavailable and rejects invalid inputs or overflow', () => {
    const draft = blankValuation('Example'); draft.crisis = defaultCrisisAssumptions();
    expect(buildCrisisScenario(draft).cashFlows.every(v => v === null)).toBe(true);
    for (const patch of [{ shockPercent: null }, { durationYears: 1.5 }, { startYear: 11 }, { extraAnnualCashCost: -1 }]) {
      expect(() => buildCrisisScenario({ ...draft, crisis: { ...draft.crisis!, ...patch } })).toThrow();
    }
    draft.scenarios.mid.cashFlows = Array(10).fill(-1e12);
    expect(() => buildCrisisScenario(draft)).toThrow(/amount range/);
  });
  it('calculates crisis DCF separately without changing baseline cash, rates or final sale', () => {
    const draft = blankValuation('Example', 'SEK', '2026-09-13');
    draft.years = 2; draft.marketCap = 100; draft.priceDate = draft.valuationDate; draft.priceSource = 'Reviewed price';
    draft.scenarios.mid = { ...draft.scenarios.mid, cashFlows: [100, 100], discountRate: 10, terminalEquity: 500 };
    draft.crisis = { ...defaultCrisisAssumptions(), enabled: true, shockPercent: 50, durationYears: 1, recoveryYears: 0, discountRate: 0, terminalEquity: 30 };
    const crisis = buildCrisisScenario(draft);
    expect(crisis.result.value).toBe(180); expect(crisis.result.npv).toBe(80);
    expect(calculateScenario(draft, 'mid').value).toBeCloseTo(100/1.1 + 600/1.21);
  });
});
