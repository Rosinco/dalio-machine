import type { ValuationDraft } from './valuation';
import { cashCalibrationId } from './cashUncertainty';

/** Call with an exact reproduction of the saved method, before saving a new draft. */
export function isUntouchedLegacyStarter(prior: ValuationDraft | null | undefined, reproduced: ValuationDraft): boolean {
  const origin = prior?.starterOrigin;
  if (!prior || prior.researchAutofillDisabled || prior.crisis || prior.purchaseRange || !origin) return false;
  if (JSON.stringify(origin.weights) !== JSON.stringify([30,25,20,15,10])) return false;
  if (origin.id === 'empirical-cash-starter-v3') {
    if (origin.terminalMethod || origin.calibrationId !== cashCalibrationId || origin.historyYears !== 5 || origin.projection !== 'latest' || origin.rangeMode !== 'historical' || origin.tailWideningPercent !== 10 || origin.spreadPercent !== 10 || origin.spreadStepPercent !== 10) return false;
  } else if (origin.id === 'weighted-cash-starter-v1' ? origin.spreadPercent !== 20 : origin.historyYears !== 5 || origin.projection !== 'trend' || origin.spreadPercent !== 10 || origin.spreadStepPercent !== 10) return false;
  return JSON.stringify({ ...prior, investment: 1000 }) === JSON.stringify({ ...reproduced, investment: 1000 });
}
