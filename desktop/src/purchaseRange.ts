import { calculateCashValue, scenarioKeys, validAmount, validDay, type PurchaseRangeSettings, type ScenarioKey, type ValuationDraft } from './valuation';

export const defaultPurchaseRangeSettings: PurchaseRangeSettings = Object.freeze({ marginOfSafetyPercent: 30, referenceScenario: 'mid', unit: 'equity' });

export type PurchaseScenarioResult = {
  value: number | null; cashPV: number | null; terminalPV: number | null;
  ceiling: number | null; cashOnlyCeiling: number | null; npvAtCandidate: number | null;
  terminalShare: number | null; error: string | null;
};
export type PurchaseRangeResult = {
  marginOfSafetyPercent: number | null; referenceScenario: ScenarioKey | null; candidateEquity: number | null;
  marginError: string | null; candidateError: string | null; referenceError: string | null;
  scenarios: Record<ScenarioKey, PurchaseScenarioResult>; selected: PurchaseScenarioResult | null;
  selectedCeiling: number | null; qualifies: boolean | null; scenarioCeilingRange: { min: number; max: number } | null;
};
export type PurchaseDisplayBasis = { unit: 'equity' | 'share' | null; divisor: number | null; error: string | null };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const positive = (value: unknown): value is number => validAmount(value) && value > 0;
const finiteOrNull = (value: number): number | null => finite(value) ? value : null;

function unavailableScenario(error: string): PurchaseScenarioResult {
  return { value: null, cashPV: null, terminalPV: null, ceiling: null, cashOnlyCeiling: null, npvAtCandidate: null, terminalShare: null, error };
}

/** A ceiling is a positive purchase price; signed valuations are retained separately. */
function ceiling(value: number, margin: number | null): number | null {
  if (value <= 0 || margin === null || margin >= 100) return null;
  const result = value * (1 - margin / 100);
  return finite(result) && result > 0 ? result : null;
}

/** Value already includes discounted terminal sale; NPV is a price comparison, never another value component. */
export function calculatePurchaseRange(draft: ValuationDraft, settings: PurchaseRangeSettings = draft.purchaseRange ?? defaultPurchaseRangeSettings): PurchaseRangeResult {
  const marginError = validAmount(settings.marginOfSafetyPercent) && settings.marginOfSafetyPercent >= 0 && settings.marginOfSafetyPercent <= 100
    ? null : 'Enter a margin of safety from 0% to 100%.';
  const marginOfSafetyPercent = marginError === null ? settings.marginOfSafetyPercent : null;
  const referenceScenario = scenarioKeys.includes(settings.referenceScenario) ? settings.referenceScenario : null;
  const referenceError = referenceScenario === null ? 'Choose a named reference scenario.' : null;
  const explicitCandidate = Object.hasOwn(settings, 'candidateEquity');
  const quoteAvailable = positive(draft.marketCap) && validDay(draft.valuationDate) && validDay(draft.priceDate)
    && draft.priceDate <= draft.valuationDate && typeof draft.priceSource === 'string' && draft.priceSource.trim().length > 0;
  const rawCandidate = explicitCandidate ? settings.candidateEquity : quoteAvailable ? draft.marketCap : null;
  const candidateEquity = positive(rawCandidate) ? rawCandidate : null;
  const candidateError = candidateEquity !== null ? null : explicitCandidate
    ? 'Enter a positive candidate equity purchase price within the supported amount range.'
    : 'A usable dated equity purchase basis is unavailable. Enter a candidate price to compare NPV and the purchase ceiling.';
  const scenarios = Object.fromEntries(scenarioKeys.map(key => {
    const calculated = calculateCashValue(draft, key);
    if (calculated.error !== null) return [key, unavailableScenario(calculated.error)];
    const { value, cashPV, terminalPV } = calculated;
    if (!finite(value) || !finite(cashPV) || !finite(terminalPV)) return [key, unavailableScenario('The scenario value is unavailable or outside the finite numerical range.')];
    const result: PurchaseScenarioResult = {
      value, cashPV, terminalPV, ceiling: ceiling(value, marginOfSafetyPercent), cashOnlyCeiling: ceiling(cashPV, marginOfSafetyPercent),
      npvAtCandidate: candidateEquity === null ? null : finiteOrNull(value - candidateEquity),
      terminalShare: value > 0 ? finiteOrNull(terminalPV / value) : null, error: null,
    };
    return [key, result];
  })) as Record<ScenarioKey, PurchaseScenarioResult>;
  const selected = referenceScenario === null ? null : scenarios[referenceScenario];
  const selectedCeiling = selected?.ceiling ?? null;
  const ceilings = scenarioKeys.map(key => scenarios[key].ceiling);
  const scenarioCeilingRange = ceilings.every(finite) ? { min: Math.min(...ceilings), max: Math.max(...ceilings) } : null;
  // Allow only machine-rounding tolerance at an otherwise exact price boundary.
  const tolerance = candidateEquity !== null && selectedCeiling !== null ? 8 * Number.EPSILON * Math.max(candidateEquity, selectedCeiling) : 0;
  const qualifies = marginError || candidateError || referenceError || !selected || selected.error ? null
    : selectedCeiling === null ? false : candidateEquity! <= selectedCeiling + tolerance;
  return { marginOfSafetyPercent, referenceScenario, candidateEquity, marginError, candidateError, referenceError,
    scenarios, selected, selectedCeiling, qualifies, scenarioCeilingRange };
}

/** Share display needs an explicit reviewed ownership divisor; no historical quote or market-cap ratio supplies one. */
export function validatePurchaseDisplayBasis(draft: ValuationDraft, settings: PurchaseRangeSettings = draft.purchaseRange ?? defaultPurchaseRangeSettings): PurchaseDisplayBasis {
  const unit = settings.unit === 'equity' || settings.unit === 'share' ? settings.unit : null;
  if (unit === null) return { unit, divisor: null, error: 'Choose equity value or a per-share display.' };
  if (!/^[A-Z]{3}$/.test(draft.currency)) return { unit, divisor: null, error: 'Enter one three-letter valuation currency.' };
  if (unit === 'equity') return { unit, divisor: 1, error: null };
  const basis = settings.shareBasis;
  if (!basis || !positive(basis.sharesMillions)) return { unit, divisor: null, error: 'Enter a positive reviewed share count in millions for the valued equity claim.' };
  if (!validDay(draft.valuationDate) || !validDay(basis.date) || basis.date > draft.valuationDate)
    return { unit, divisor: null, error: 'The share basis needs a valid date no later than the valuation date.' };
  if (typeof basis.source !== 'string' || !basis.source.trim()) return { unit, divisor: null, error: 'Record the source and ownership basis for the share count.' };
  if (basis.currency !== draft.currency) return { unit, divisor: null, error: 'The share-price display currency must match the equity valuation currency.' };
  return { unit, divisor: basis.sharesMillions, error: null };
}

export function toPurchaseDisplayAmount(value: number | null, basis: PurchaseDisplayBasis): number | null {
  if (!finite(value) || basis.error !== null || !finite(basis.divisor) || basis.divisor <= 0) return null;
  return finiteOrNull(value / basis.divisor);
}
