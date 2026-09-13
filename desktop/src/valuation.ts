import { resolveTerminalSale } from './terminalValue';

export const scenarioKeys = ['low', 'mid', 'high'] as const;
export type ScenarioKey = typeof scenarioKeys[number];
export const scenarioNames = { low: 'Low', mid: 'Mid', high: 'High' };
export const scenarioColors = { low: '#a87840', mid: '#326d94', high: '#7966ad' };
export type Scenario = { cashFlows: (number | null)[]; discountRate: number | null; terminalEquity: number | null; terminalCash?: { cashFlow: number | null; growthRate: number | null }; recoveryEquity: number | null; recoveryYear: number | null; rationale: string };
export const capitalLabels = { tangibleEquity: 'Tangible book equity', averageTCE: 'Average tangible capital employed', nopat: 'Normalized annual NOPAT', grossDebt: 'Gross corporate debt', surplusCash: 'Available surplus cash' };
export type StarterOrigin = { asOf: string; weights: number[]; spreadPercent: number } & (
  { id: 'weighted-cash-starter-v1' } |
  { id: 'weighted-cash-starter-v2'; historyYears: 5 | 10; spreadStepPercent: number; projection: 'trend' | 'flat' } |
  { id: 'empirical-cash-starter-v3'; historyYears: 5 | 10; spreadStepPercent: number; projection: 'latest' | 'trend' | 'flat'; rangeMode: 'historical' | 'percentage'; tailWideningPercent: number; calibrationId: string; terminalMethod?: 'historical-median-v1' }
);
export type CrisisAssumptions = { enabled: boolean; shockPercent: number | null; startYear: number | null; durationYears: number | null; recoveryYears: number | null; extraAnnualCashCost: number | null; discountRate: number | null; terminalEquity: number | null; rationale: string };
export type PurchaseRangeSettings = { marginOfSafetyPercent: number | null; referenceScenario: ScenarioKey; candidateEquity?: number | null; unit: 'equity' | 'share'; shareBasis?: { sharesMillions: number | null; date: string; source: string; currency: string } };
export type ValuationDraft = {
  purchaseRange?: PurchaseRangeSettings;
  crisis?: CrisisAssumptions;
  researchOrigin?: { id: string; asOf: string };
  starterOrigin?: StarterOrigin;
  researchAutofillDisabled?: boolean;
  title: string; currency: string; valuationDate: string; priceDate: string; priceSource: string;
  marketCap: number | null; investment: number | null; years: number; scenarios: Record<ScenarioKey, Scenario>;
  capital: Record<keyof typeof capitalLabels, number | null>;
  notes: { business: string; macro: string; financing: string; recovery: string; decision: string };
};
export type Payback = { year: number | null; reversed: boolean };
export type Recovery = { value: number; npv: number; valuePrice: number; payback: number | null; discountedPayback: number | null };
export type ScenarioResult = {
  error: string | null; recoveryError: string | null; recovery: Recovery | null;
  discounted: number[]; cumulativeCash: number[]; cumulativeNPV: number[]; cumulativeNPVWithSale: number[];
  value: number | null; cashPV: number | null; terminalPV: number | null; terminalShare: number | null;
  npv: number | null; valuePrice: number | null; discountToValue: number | null; stakeValue: number | null;
  payback: Payback; discountedPayback: Payback; paybackWithSale: Payback; discountedPaybackWithSale: Payback;
};
export const validDay = (v: unknown): v is string => typeof v === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(v) && v >= '1900-01-01' && v <= '2300-12-31' && Number.isFinite(Date.parse(v)) && new Date(v).toISOString().slice(0, 10) === v;
export const validAmount = (v: unknown): v is number => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= 1e12;
export function blankValuation(company: string, currency = 'SEK', day = new Date().toISOString().slice(0, 10)): ValuationDraft {
  return { title: `${company} — value and price`, currency, valuationDate: day, priceDate: '', priceSource: '', marketCap: null, investment: 1000, years: 10,
    scenarios: Object.fromEntries(scenarioKeys.map(key => [key, { cashFlows: Array(10).fill(null), discountRate: null, terminalEquity: 0, recoveryEquity: null, recoveryYear: null, rationale: '' }])) as Record<ScenarioKey, Scenario>,
    capital: { tangibleEquity: null, averageTCE: null, nopat: null, grossDebt: null, surplusCash: null },
    notes: { business: '', macro: '', financing: '', recovery: '', decision: '' } };
}
export function generateCashFlows(first: number, growthPercent: number, years: number): number[] {
  if (!validAmount(first) || !Number.isFinite(growthPercent) || growthPercent < -100 || growthPercent > 1000 || !Number.isInteger(years) || years < 1 || years > 50) throw new Error('Enter a finite starting cash flow, growth from -100% to 1,000%, and 1–50 years.');
  const values = Array.from({ length: years }, (_, i) => first * (1 + growthPercent / 100) ** i);
  if (!values.every(validAmount)) throw new Error('The generated cash flows exceed the supported amount range.');
  return values;
}
function payback(cumulative: number[], cost: number): Payback {
  const tolerance = 1e-10 * cost;
  const first = cumulative.findIndex((v, i) => i > 0 && v >= cost - tolerance);
  return { year: first < 0 ? null : first, reversed: first >= 0 && cumulative.slice(first + 1).some(v => v < cost - tolerance) };
}
function empty(error: string, recovery: Recovery | null = null, recoveryError: string | null = null): ScenarioResult {
  const noPayback = { year: null, reversed: false };
  return { error, recovery, recoveryError, discounted: [], cumulativeCash: [], cumulativeNPV: [], cumulativeNPVWithSale: [], value: null, cashPV: null, terminalPV: null, terminalShare: null,
    npv: null, valuePrice: null, discountToValue: null, stakeValue: null, payback: noPayback, discountedPayback: noPayback, paybackWithSale: noPayback, discountedPaybackWithSale: noPayback };
}
/** Intrinsic cash value is independent of the observed purchase price and stake size. */
export function calculateCashValue(d: ValuationDraft, key: ScenarioKey): { error: string | null; discounted: number[]; cashPV: number | null; terminalPV: number | null; value: number | null } {
  const fail = (error: string) => ({ error, discounted: [], cashPV: null, terminalPV: null, value: null });
  if (!validDay(d.valuationDate)) return fail('Enter a valid valuation date.');
  if (!/^[A-Z]{3}$/.test(d.currency)) return fail('Enter one three-letter currency for every amount.');
  if (!Number.isInteger(d.years) || d.years < 1 || d.years > 50) return fail('Choose a forecast of 1–50 years.');
  const s = d.scenarios[key], r = s.discountRate;
  if (!validAmount(r) || r < 0 || r > 100) return fail('Enter a required equity return from 0% to 100%.');
  const cash = s.cashFlows.slice(0, d.years);
  if (cash.length !== d.years || !cash.every(validAmount)) return fail('Enter a cash distribution for every forecast year; use 0 for no payment.');
  const terminal = resolveTerminalSale(s);
  if (terminal.error || terminal.value === null) return fail(terminal.error!);
  const factor = 1 + r / 100, discounted = cash.map((v, i) => v / factor ** (i + 1));
  const cashPV = discounted.reduce((a, b) => a + b, 0), terminalPV = terminal.value / factor ** d.years, value = cashPV + terminalPV;
  if (![...discounted, cashPV, terminalPV, value].every(Number.isFinite)) return fail('These inputs exceed the supported numerical range.');
  return { error: null, discounted, cashPV, terminalPV, value };
}
export function calculateScenario(d: ValuationDraft, key: ScenarioKey): ScenarioResult {
  if (!validAmount(d.marketCap) || d.marketCap < 1e-9 || !validAmount(d.investment) || d.investment < 1e-9) return empty('Enter a positive equity market value and investment amount.');
  if (!validDay(d.valuationDate) || !validDay(d.priceDate) || d.priceDate > d.valuationDate) return empty('Enter valid valuation and price dates; the price cannot be later than the valuation date.');
  if (!d.priceSource.trim()) return empty('Record the price source and share/ownership basis.');
  if (!/^[A-Z]{3}$/.test(d.currency)) return empty('Enter one three-letter currency for every amount.');
  if (!Number.isInteger(d.years) || d.years < 1 || d.years > 50) return empty('Choose a forecast of 1–50 years.');
  const s = d.scenarios[key], r = s.discountRate;
  if (!validAmount(r) || r < 0 || r > 100) return empty('Enter a required equity return from 0% to 100%.');
  const factor = 1 + r / 100;
  let recovery: Recovery | null = null, recoveryError: string | null = null;
  if (s.recoveryEquity !== null || s.recoveryYear !== null) {
    if (!validAmount(s.recoveryEquity) || s.recoveryEquity < 0 || !Number.isInteger(s.recoveryYear) || s.recoveryYear! < 0 || s.recoveryYear! > 50) recoveryError = 'Recovery needs nonnegative net equity proceeds and a payment year from 0 to 50.';
    else {
      const value = s.recoveryEquity / factor ** s.recoveryYear!;
      recovery = { value, npv: value - d.marketCap, valuePrice: value / d.marketCap, payback: s.recoveryEquity >= d.marketCap ? s.recoveryYear : null, discountedPayback: value >= d.marketCap ? s.recoveryYear : null };
    }
  }
  const priced = calculateCashValue(d, key);
  if (priced.error) return empty(priced.error, recovery, recoveryError);
  const cash = s.cashFlows.slice(0, d.years) as number[], discounted = priced.discounted;
  const terminal = resolveTerminalSale(s);
  const cumulativeCash = [0], cumulativePV = [0];
  cash.forEach((value, i) => { cumulativeCash.push(cumulativeCash[i] + value); cumulativePV.push(cumulativePV[i] + discounted[i]); });
  const terminalPV = priced.terminalPV!, cashPV = priced.cashPV!, value = priced.value!;
  const cashWithSale = [...cumulativeCash], pvWithSale = [...cumulativePV];
  cashWithSale[d.years] += terminal.value!; pvWithSale[d.years] += terminalPV;
  const npv = value - d.marketCap, stakeValue = value / d.marketCap * d.investment;
  if (![...discounted, ...cumulativeCash, ...cumulativePV, value, npv, stakeValue].every(Number.isFinite)) return empty('These inputs exceed the supported numerical range.', recovery, recoveryError);
  return { error: null, recovery, recoveryError, discounted, cumulativeCash, cumulativeNPV: cumulativePV.map(v => v - d.marketCap!), cumulativeNPVWithSale: pvWithSale.map(v => v - d.marketCap!),
    value, cashPV, terminalPV, terminalShare: value > 0 ? terminalPV / value : null, npv, valuePrice: value / d.marketCap, discountToValue: value > 0 ? 1 - d.marketCap / value : null, stakeValue,
    payback: payback(cumulativeCash, d.marketCap), discountedPayback: payback(cumulativePV, d.marketCap), paybackWithSale: payback(cashWithSale, d.marketCap), discountedPaybackWithSale: payback(pvWithSale, d.marketCap) };
}
export function calculateValuation(d: ValuationDraft) {
  const scenarios = Object.fromEntries(scenarioKeys.map(key => [key, calculateScenario(d, key)])) as Record<ScenarioKey, ScenarioResult>;
  const ready = scenarioKeys.every(key => !scenarios[key].error);
  const crossing = ready && Array.from({ length: d.years }, (_, i) => i).some(i => scenarios.low.discounted[i] > scenarios.mid.discounted[i] || scenarios.mid.discounted[i] > scenarios.high.discounted[i])
    || ready && (scenarios.low.value! > scenarios.mid.value! || scenarios.mid.value! > scenarios.high.value!);
  return { scenarios, ready, crossing };
}
