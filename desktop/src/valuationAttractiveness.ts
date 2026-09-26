import type { ResearchGaugeRow } from './researchGaugeModel';
import { validDay } from './valuation';

export type ValuationAttractiveness = {
  surplusPercent: number | null;
  midNpvPercent: number | null;
  dcfPriceRatio: number | null;
  cashCoveragePercent: number | null;
  lowNpvPercent: number | null;
  adjustedValue: number | null;
  adjustedNpv: number | null;
  terminalCreditPercent: number;
  reason: string | null;
};

const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const observed = (value: number): number | null => finite(value) ? value : null;
const currency = (value: unknown): value is string => typeof value === 'string' && /^[A-Z]{3}$/.test(value);

/** A dated starter sensitivity, not an expected return, probability or business-quality score.
 * Terminal value enters once. The credit changes only the adjusted comparison;
 * supporting DCF, NPV and cash coverage always retain the frozen full-DCF basis.
 * The caller supplies a row from the validated, source-bound research artifact.
 */
export function calculateValuationAttractiveness(row: ResearchGaugeRow, terminalCreditPercent = 100): ValuationAttractiveness {
  const unavailable = (reason: string): ValuationAttractiveness => ({ surplusPercent: null, midNpvPercent: null, dcfPriceRatio: null, cashCoveragePercent: null, lowNpvPercent: null, adjustedValue: null, adjustedNpv: null, terminalCreditPercent, reason });
  if (!finite(terminalCreditPercent) || terminalCreditPercent < 0 || terminalCreditPercent > 100) return unavailable('Enter terminal credit from 0% to 100%; an invalid sensitivity is not replaced with the default.');
  if (row.route === 'financial' || row.valuation.status === 'manual-financial') return unavailable('Financial companies need a reviewed equity-cash model before a comparable starter ranking is available.');
  if (row.route === 'unclassified' || row.classificationConflict) return unavailable('Resolve the company classification before using the starter valuation ranking.');
  if (row.readiness === 'reconcile_data' || row.readiness === 'no_history') return unavailable('Comparable annual evidence must be available before using the starter valuation ranking.');

  const v = row.valuation;
  if (!currency(v.currency)) return unavailable('The starter valuation currency is unavailable.');
  if (!finite(v.value) || !finite(v.cashPV) || !finite(v.terminalPV) || v.terminalPV < 0) return unavailable('A complete finite starter DCF, annual cash PV and nonnegative terminal PV are required.');
  const combined = v.cashPV + v.terminalPV;
  const tolerance = 16 * Number.EPSILON * Math.max(1, Math.abs(v.value), Math.abs(v.cashPV), Math.abs(v.terminalPV));
  if (!finite(combined) || Math.abs(combined - v.value) > tolerance) return unavailable('Starter DCF does not reconcile with annual cash PV plus terminal PV.');
  if (!finite(v.candidateEquity) || v.candidateEquity <= 0) return unavailable('A strictly positive finite saved equity purchase price is required.');
  const basis = v.priceBasis;
  if (!validDay(v.priceDate) || !basis || typeof basis.sourceId !== 'string' || !basis.sourceId.trim() || !validDay(basis.sourceAsOf) || v.priceDate > basis.sourceAsOf) return unavailable('A valid dated saved equity-price source is required; no current quote or replacement date is inferred.');

  const adjustedValue = observed(v.cashPV + (terminalCreditPercent / 100) * v.terminalPV);
  if (adjustedValue === null) return unavailable('The terminal-credit sensitivity exceeds the finite numerical range.');
  const dcfPriceRatio = observed(v.value / v.candidateEquity);
  const surplusPercent = observed(100 * (adjustedValue / v.candidateEquity - 1));
  const midNpvPercent = dcfPriceRatio === null ? null : observed(100 * (dcfPriceRatio - 1));
  const cashCoveragePercent = observed(100 * (v.cashPV / v.candidateEquity));
  const lowNpvPercent = finite(v.lowValue) ? observed(100 * (v.lowValue / v.candidateEquity - 1)) : null;
  const adjustedNpv = observed(adjustedValue - v.candidateEquity);
  const reason = [surplusPercent, midNpvPercent, dcfPriceRatio, cashCoveragePercent, adjustedNpv].some(value => value === null)
    ? 'Some comparisons exceed the finite numerical range; unavailable results are withheld.' : null;
  return { surplusPercent, midNpvPercent, dcfPriceRatio, cashCoveragePercent, lowNpvPercent, adjustedValue, adjustedNpv, terminalCreditPercent, reason };
}
