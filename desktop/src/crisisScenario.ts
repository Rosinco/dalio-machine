import { calculateScenario, validAmount, type CrisisAssumptions, type ValuationDraft } from './valuation';

export const defaultCrisisAssumptions = (): CrisisAssumptions => ({ enabled: false, shockPercent: 40, startYear: 1, durationYears: 2, recoveryYears: 3, extraAnnualCashCost: 0, discountRate: 10, terminalEquity: 0, rationale: 'Illustrative shock and recovery. Review cash effects, investment needs and financing against company evidence. No probability is assigned; no final sale is assumed.' });

export function buildCrisisScenario(draft: ValuationDraft) {
  const c = draft.crisis ?? defaultCrisisAssumptions();
  if (!validAmount(c.shockPercent) || c.shockPercent < 0 || c.shockPercent > 300) throw new Error('Enter a crisis cash reduction from 0% to 300%.');
  if (!Number.isInteger(c.startYear) || c.startYear! < 1 || c.startYear! > draft.years) throw new Error('Start the crisis within the forecast horizon.');
  if (!Number.isInteger(c.durationYears) || c.durationYears! < 1 || c.durationYears! > 50 || !Number.isInteger(c.recoveryYears) || c.recoveryYears! < 0 || c.recoveryYears! > 50) throw new Error('Enter 1–50 shock years and 0–50 recovery years.');
  if (!validAmount(c.extraAnnualCashCost) || c.extraAnnualCashCost < 0) throw new Error('Enter a nonnegative additional annual cash cost.');
  const cashFlows = Array.from({ length: draft.years }, (_, i) => {
    const mid = draft.scenarios.mid.cashFlows[i];
    if (!validAmount(mid)) return null;
    const elapsed = i + 1 - c.startYear!;
    const intensity = elapsed < 0 ? 0 : elapsed < c.durationYears! ? 1 : c.recoveryYears === 0 ? 0 : Math.max(0, 1 - (elapsed - c.durationYears! + 1) / c.recoveryYears!);
    const cash = mid - intensity * (Math.abs(mid) * c.shockPercent! / 100 + c.extraAnnualCashCost!);
    if (!validAmount(cash)) throw new Error('The crisis cash flows exceed the supported amount range.');
    return cash;
  });
  const scenario = { ...draft.scenarios.mid, terminalCash: undefined, cashFlows, discountRate: c.discountRate, terminalEquity: c.terminalEquity, recoveryEquity: null, recoveryYear: null, rationale: c.rationale };
  const result = calculateScenario({ ...draft, scenarios: { ...draft.scenarios, mid: scenario } }, 'mid');
  return { cashFlows, result };
}
