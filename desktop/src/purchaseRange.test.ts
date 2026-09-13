import { describe, expect, it } from 'vitest';
import { blankValuation, type PurchaseRangeSettings, type ValuationDraft } from './valuation';
import { calculatePurchaseRange, defaultPurchaseRangeSettings, toPurchaseDisplayAmount, validatePurchaseDisplayBasis } from './purchaseRange';

function study(): ValuationDraft {
  const draft = blankValuation('Purchase example', 'SEK', '2026-09-13');
  draft.marketCap = 700;
  draft.priceDate = '2026-09-11';
  draft.priceSource = 'Reviewed hypothetical equity purchase basis';
  for (const scenario of Object.values(draft.scenarios)) {
    scenario.cashFlows = Array(10).fill(100);
    scenario.discountRate = 10;
    scenario.terminalEquity = 1000;
  }
  return draft;
}

function settings(patch: Partial<PurchaseRangeSettings> = {}): PurchaseRangeSettings {
  return { ...defaultPurchaseRangeSettings, ...patch };
}

describe('scenario purchase ceilings', () => {
  it('discounts terminal sale once and applies the default 30% discount to value', () => {
    const result = calculatePurchaseRange(study());
    const mid = result.scenarios.mid;
    expect(mid.value).toBeCloseTo(1000);
    expect(mid.cashPV).toBeCloseTo(100 * (1 - 1 / 1.1 ** 10) / .1);
    expect(mid.terminalPV).toBeCloseTo(1000 / 1.1 ** 10);
    expect(mid.ceiling).toBeCloseTo(700);
    expect(mid.cashOnlyCeiling).toBeCloseTo(mid.cashPV! * .7);
    expect(mid.npvAtCandidate).toBeCloseTo(300);
    expect(result.referenceScenario).toBe('mid');
    expect(result.selectedCeiling).toBeCloseTo(700);
    expect(result.qualifies).toBe(true);
  });

  it('keeps named scenarios when different required returns reverse their numeric order', () => {
    const draft = study();
    draft.years = 1;
    Object.assign(draft.scenarios.low, { cashFlows: [150], discountRate: 0, terminalEquity: 0 });
    Object.assign(draft.scenarios.mid, { cashFlows: [200], discountRate: 100, terminalEquity: 0 });
    Object.assign(draft.scenarios.high, { cashFlows: [180], discountRate: 50, terminalEquity: 0 });
    const result = calculatePurchaseRange(draft, settings({ referenceScenario: 'low', candidateEquity: 100 }));
    expect(result.scenarios.low.value).toBe(150);
    expect(result.scenarios.mid.value).toBe(100);
    expect(result.scenarios.high.value).toBe(120);
    expect(result.scenarioCeilingRange).toEqual({ min: 70, max: 105 });
    expect(result.selectedCeiling).toBe(105);
    expect(result.qualifies).toBe(true);
  });

  it('retains signed values and NPVs without inventing a positive price for negative value', () => {
    const draft = study();
    Object.assign(draft.scenarios.low, { cashFlows: Array(10).fill(-100), terminalEquity: 0 });
    const result = calculatePurchaseRange(draft, settings({ referenceScenario: 'low', candidateEquity: 50 }));
    expect(result.scenarios.low.value).toBeLessThan(0);
    expect(result.scenarios.low.npvAtCandidate).toBeCloseTo(result.scenarios.low.value! - 50);
    expect(result.scenarios.low.ceiling).toBeNull();
    expect(result.scenarios.low.cashOnlyCeiling).toBeNull();
    expect(result.scenarioCeilingRange).toBeNull();
    expect(result.qualifies).toBe(false);
    expect(result.scenarios.mid.ceiling).toBeCloseTo(700);
  });

  it('treats actual zero cash and 100% margin as no positive qualifying price', () => {
    const draft = study();
    Object.assign(draft.scenarios.low, { cashFlows: Array(10).fill(0), terminalEquity: 0 });
    const zero = calculatePurchaseRange(draft, settings({ referenceScenario: 'low' }));
    expect(zero.scenarios.low.value).toBe(0);
    expect(zero.scenarios.low.ceiling).toBeNull();
    expect(zero.qualifies).toBe(false);
    const full = calculatePurchaseRange(study(), settings({ marginOfSafetyPercent: 100 }));
    expect(full.scenarios.mid.value).toBeCloseTo(1000);
    expect(full.scenarios.mid.npvAtCandidate).toBeCloseTo(300);
    expect(full.selectedCeiling).toBeNull();
    expect(full.scenarioCeilingRange).toBeNull();
    expect(full.qualifies).toBe(false);
  });

  it('withholds incomplete scenario bounds without erasing other valid scenarios', () => {
    const draft = study();
    draft.scenarios.low.cashFlows[2] = null;
    const result = calculatePurchaseRange(draft);
    expect(result.scenarios.low.error).toBeTruthy();
    expect(result.scenarios.low.value).toBeNull();
    expect(result.scenarios.low.ceiling).toBeNull();
    expect(result.scenarioCeilingRange).toBeNull();
    expect(result.scenarios.mid.ceiling).toBeCloseTo(700);
    expect(result.qualifies).toBe(true);
    expect(calculatePurchaseRange(draft, settings({ referenceScenario: 'low' })).qualifies).toBeNull();
  });

  it('rejects an invalid sustainable terminal model without falling back to explicit sale', () => {
    const draft = study();
    draft.scenarios.mid.terminalCash = { cashFlow: 100, growthRate: 10 };
    const result = calculatePurchaseRange(draft);
    expect(result.selected?.error).toMatch(/growth|terminal/i);
    expect(result.selectedCeiling).toBeNull();
    expect(result.qualifies).toBeNull();
  });

  it('shows terminal contribution above 100% when negative annual cash offsets terminal value', () => {
    const draft = study();
    draft.years = 1;
    Object.assign(draft.scenarios.mid, { cashFlows: [-100], discountRate: 0, terminalEquity: 200 });
    const result = calculatePurchaseRange(draft, settings({ candidateEquity: 60 }));
    expect(result.scenarios.mid.value).toBe(100);
    expect(result.scenarios.mid.terminalShare).toBe(2);
    expect(result.scenarios.mid.cashOnlyCeiling).toBeNull();
    expect(result.selectedCeiling).toBe(70);
  });

  it('calculates ceilings without a quote or investment and accepts an explicit candidate', () => {
    const draft = study();
    draft.marketCap = null; draft.investment = null; draft.priceDate = ''; draft.priceSource = '';
    const absent = calculatePurchaseRange(draft);
    expect(absent.selectedCeiling).toBeCloseTo(700);
    expect(absent.candidateEquity).toBeNull();
    expect(absent.scenarios.mid.npvAtCandidate).toBeNull();
    expect(absent.qualifies).toBeNull();
    const explicit = calculatePurchaseRange(draft, settings({ candidateEquity: 500 }));
    expect(explicit.selectedCeiling).toBeCloseTo(700);
    expect(explicit.scenarios.mid.npvAtCandidate).toBeCloseTo(500);
    expect(explicit.qualifies).toBe(true);
  });

  it('does not reuse a historical quote when its candidate input was explicitly cleared', () => {
    const draft = study();
    draft.purchaseRange = settings({ candidateEquity: null });
    const result = calculatePurchaseRange(draft);
    expect(result.candidateEquity).toBeNull();
    expect(result.candidateError).toBeTruthy();
    expect(result.scenarios.mid.npvAtCandidate).toBeNull();
    expect(result.selectedCeiling).toBeCloseTo(700);
  });

  it('requires valid historical quote metadata only for the absent-field candidate fallback', () => {
    for (const patch of [{ priceDate: '' }, { priceDate: '2026-09-14' }, { priceDate: '2026-02-30' }, { priceSource: ' ' }, { marketCap: 0 }]) {
      const draft = Object.assign(study(), patch);
      expect(calculatePurchaseRange(draft).candidateEquity).toBeNull();
      expect(calculatePurchaseRange(draft).selectedCeiling).toBeCloseTo(700);
      expect(calculatePurchaseRange(draft, settings({ candidateEquity: 650 })).candidateEquity).toBe(650);
    }
  });

  it('keeps valuation unchanged when margin or candidate price changes and preserves inputs', () => {
    const draft = study();
    const before = structuredClone(draft);
    const first = calculatePurchaseRange(draft, settings({ marginOfSafetyPercent: 0, candidateEquity: 1100 }));
    const second = calculatePurchaseRange(draft, settings({ marginOfSafetyPercent: 50, candidateEquity: 400 }));
    expect(first.selected?.value).toBe(second.selected?.value);
    expect(first.selectedCeiling).toBeCloseTo(1000);
    expect(second.selectedCeiling).toBeCloseTo(500);
    expect(first.selected?.npvAtCandidate).toBeCloseTo(-100);
    expect(second.selected?.npvAtCandidate).toBeCloseTo(600);
    expect(first.qualifies).toBe(false);
    expect(second.qualifies).toBe(true);
    expect(draft).toEqual(before);
  });

  it('rejects missing, negative, excessive and nonfinite margins or candidates', () => {
    for (const marginOfSafetyPercent of [null, -1, 101, NaN, Infinity]) {
      const result = calculatePurchaseRange(study(), settings({ marginOfSafetyPercent }));
      expect(result.marginError).toBeTruthy();
      expect(result.selectedCeiling).toBeNull();
      expect(result.selected?.value).toBeCloseTo(1000);
      expect(result.qualifies).toBeNull();
    }
    for (const candidateEquity of [null, 0, -1, 1e12 + 1, NaN, Infinity]) {
      const result = calculatePurchaseRange(study(), settings({ candidateEquity }));
      expect(result.candidateError).toBeTruthy();
      expect(result.qualifies).toBeNull();
      expect(result.scenarios.mid.npvAtCandidate).toBeNull();
    }
  });

  it('rejects an unknown reference without silently selecting another named scenario', () => {
    const result = calculatePurchaseRange(study(), settings({ referenceScenario: 'unknown' as 'mid' }));
    expect(result.referenceError).toBeTruthy();
    expect(result.referenceScenario).toBeNull();
    expect(result.selected).toBeNull();
    expect(result.selectedCeiling).toBeNull();
    expect(result.qualifies).toBeNull();
  });
});

describe('explicit purchase display basis', () => {
  const shares = () => settings({ unit: 'share', shareBasis: { sharesMillions: 100, date: '2026-09-11', source: 'Reviewed common shares and ownership claim', currency: 'SEK' } });

  it('keeps canonical equity millions and converts only through supplied shares in millions', () => {
    const draft = study();
    expect(validatePurchaseDisplayBasis(draft, settings())).toEqual({ unit: 'equity', divisor: 1, error: null });
    const basis = validatePurchaseDisplayBasis(draft, shares());
    expect(basis).toEqual({ unit: 'share', divisor: 100, error: null });
    expect(toPurchaseDisplayAmount(700, basis)).toBe(7);
    expect(toPurchaseDisplayAmount(-100, basis)).toBe(-1);
    expect(toPurchaseDisplayAmount(null, basis)).toBeNull();
    draft.marketCap = 50000;
    expect(validatePurchaseDisplayBasis(draft, shares())).toEqual(basis);
  });

  it('requires explicit consistent dated source-bound share inputs without guessing', () => {
    expect(validatePurchaseDisplayBasis(study(), settings({ unit: 'share' })).error).toBeTruthy();
    for (const patch of [{ sharesMillions: null }, { sharesMillions: 0 }, { sharesMillions: -1 }, { sharesMillions: 1e12 + 1 }, { sharesMillions: NaN }, { date: '' }, { date: '2026-09-14' }, { date: '2026-02-30' }, { source: ' ' }, { currency: 'USD' }]) {
      const configured = shares();
      Object.assign(configured.shareBasis!, patch);
      const basis = validatePurchaseDisplayBasis(study(), configured);
      expect(basis.divisor).toBeNull();
      expect(basis.error).toBeTruthy();
      expect(toPurchaseDisplayAmount(700, basis)).toBeNull();
    }
  });

  it('withholds nonfinite converted amounts and invalid unit settings', () => {
    const basis = validatePurchaseDisplayBasis(study(), shares());
    expect(toPurchaseDisplayAmount(Infinity, basis)).toBeNull();
    expect(toPurchaseDisplayAmount(Number.MAX_VALUE, { unit: 'share', divisor: 1e-12, error: null })).toBeNull();
    expect(validatePurchaseDisplayBasis(study(), settings({ unit: 'unknown' as 'equity' })).error).toBeTruthy();
  });
});
