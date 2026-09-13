import { describe, expect, it } from 'vitest';
import { blankValuation, type ValuationDraft } from './valuation';
import { isUntouchedLegacyStarter } from './starterMigration';

const legacy = (): ValuationDraft => ({ ...blankValuation('Example', 'SEK', '2026-09-12'), starterOrigin: { id: 'weighted-cash-starter-v2', asOf: '2026-09-12', historyYears: 5, projection: 'trend', weights: [30,25,20,15,10], spreadPercent: 10, spreadStepPercent: 10 } });
describe('upgrade eligibility for existing starters', () => {
  it('allows exact default models and investment-only changes', () => {
    const old = legacy(); expect(isUntouchedLegacyStarter(old, structuredClone(old))).toBe(true);
    expect(isUntouchedLegacyStarter({ ...old, investment: 500 }, old)).toBe(true);
    const v1: ValuationDraft = { ...old, starterOrigin: { id: 'weighted-cash-starter-v1', asOf: '2026-09-12', weights: [30,25,20,15,10], spreadPercent: 20 } };
    expect(isUntouchedLegacyStarter(v1, structuredClone(v1))).toBe(true);
  });
  it('preserves edits, restored revisions, custom settings and additional saved fields', () => {
    const old = legacy();
    for (const patch of [{ title: 'My study' }, { researchAutofillDisabled: true }, { notes: { ...old.notes, financing: 'Reviewed debt' } }, { scenarios: { ...old.scenarios, mid: { ...old.scenarios.mid, cashFlows: Array(10).fill(123) } } }]) {
      expect(isUntouchedLegacyStarter({ ...old, ...patch }, old)).toBe(false);
    }
    const custom = { ...old, starterOrigin: { ...old.starterOrigin!, spreadPercent: 20 } } as ValuationDraft;
    expect(isUntouchedLegacyStarter(custom, structuredClone(custom))).toBe(false);
    expect(isUntouchedLegacyStarter(null, old)).toBe(false);
  });
  it('upgrades only untouched prior empirical defaults while retaining terminal edits and custom models', () => {
    const old: ValuationDraft = { ...legacy(), starterOrigin: { id: 'empirical-cash-starter-v3', asOf: '2026-09-12', historyYears: 5, projection: 'latest', weights: [30,25,20,15,10], spreadPercent: 10, spreadStepPercent: 10, rangeMode: 'historical', tailWideningPercent: 10, calibrationId: 'cash-uncertainty-2026-09-12-v1' } };
    expect(isUntouchedLegacyStarter(old, structuredClone(old))).toBe(true);
    const edited = structuredClone(old); edited.scenarios.mid.terminalCash = { cashFlow: null, growthRate: 0 };
    expect(isUntouchedLegacyStarter(edited, old)).toBe(false);
    for (const patch of [{ projection: 'trend' }, { tailWideningPercent: 20 }, { terminalMethod: 'historical-median-v1' }, { calibrationId: 'custom-calibration' }]) {
      const custom = { ...old, starterOrigin: { ...old.starterOrigin, ...patch } } as ValuationDraft;
      expect(isUntouchedLegacyStarter(custom, structuredClone(custom))).toBe(false);
    }
  });
});
