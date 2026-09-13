import { validAmount, type Scenario } from './valuation';

/** Cash is the FIRST post-horizon year's equity cash, already after reinvestment. */
export function resolveTerminalSale(s: Pick<Scenario, 'terminalCash' | 'terminalEquity' | 'discountRate'>): { value: number | null; error: string | null } {
  if (!s.terminalCash) return validAmount(s.terminalEquity) && s.terminalEquity >= 0
    ? { value: s.terminalEquity, error: null }
    : { value: null, error: 'Enter nonnegative final equity sale proceeds, or 0 for no sale.' };
  const { cashFlow, growthRate } = s.terminalCash, r = s.discountRate;
  if (!validAmount(cashFlow)) return { value: null, error: 'Enter sustainable equity cash for the first year after the forecast, after required reinvestment.' };
  if (!validAmount(r) || r <= 0 || r > 100 || !validAmount(growthRate) || growthRate <= -100 || growthRate >= r)
    return { value: null, error: 'Terminal cash needs a positive required return and mature growth above -100% and below that return.' };
  const value = Math.max(0, cashFlow) / ((r - growthRate) / 100);
  return validAmount(value) ? { value, error: null } : { value: null, error: 'The terminal value exceeds the supported amount range.' };
}

/** A disclosed unreviewed seed, not a fitted terminal cash forecast or confidence band. */
export function seedTerminalCash(cash: (number | null)[]) {
  if (cash.length < 3 || !cash.every(validAmount)) return { median: null, low: null, mid: null, high: null };
  const sorted = [...cash].sort((a, b) => a - b), middle = Math.floor(sorted.length / 2);
  const median = sorted.length % 2 ? sorted[middle] : sorted[middle - 1] / 2 + sorted[middle] / 2;
  const low = median - Math.abs(median) * .2, high = median + Math.abs(median) * .2;
  return { median, low: validAmount(low) ? low : null, mid: median, high: validAmount(high) ? high : null };
}
