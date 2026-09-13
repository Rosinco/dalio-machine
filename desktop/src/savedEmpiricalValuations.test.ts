import { describe, expect, it } from 'vitest';
import { defaultCrisisAssumptions } from './crisisScenario';
import { cashCalibrationId } from './cashUncertainty';
import { decodeValuation, loadValuationDraft, loadValuations, saveValuation, saveValuationDraft, type SavedValuation } from './savedValuations';
import { blankValuation, type StarterOrigin } from './valuation';

function store() {
  const values = new Map<string, string>();
  let writes = 0;
  return { values, writes: () => writes, getItem: (key: string) => values.get(key) ?? null,
    setItem: (key: string, value: string) => { writes++; values.set(key, value); } };
}

function study(): SavedValuation {
  const draft = blankValuation('Persistence fixture', 'SEK', '2026-09-13');
  draft.starterOrigin = { id: 'empirical-cash-starter-v3', asOf: '2026-09-13', historyYears: 5,
    weights: [30, 25, 20, 15, 10], spreadPercent: 10, spreadStepPercent: 10,
    projection: 'latest', rangeMode: 'historical', tailWideningPercent: 25, calibrationId: cashCalibrationId };
  draft.crisis = { ...defaultCrisisAssumptions(), enabled: true, shockPercent: 75, startYear: 2,
    durationYears: 3, recoveryYears: 4, extraAnnualCashCost: 5, discountRate: 12, terminalEquity: 100,
    rationale: 'A separate, explicitly assumed funding and recovery path.' };
  draft.scenarios.mid.cashFlows = [100, -20, 0, 40, 50, 60, 70, 80, 90, 100];
  return { format: 'macro-atlas-valuation', version: 1, id: 'empirical-persistence', created: '2026-09-13T07:00:00.000Z',
    company: '200', release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), draft };
}

describe('saved empirical starters and separate crisis assumptions', () => {
  it('round-trips every v3 projection and range mode with its exact calibration and tail assumptions', () => {
    for (const projection of ['latest', 'trend', 'flat'] as const) for (const rangeMode of ['historical', 'percentage'] as const) {
      const original = study();
      if (original.draft.starterOrigin?.id !== 'empirical-cash-starter-v3') throw new Error('Invalid test fixture');
      original.draft.starterOrigin = { ...original.draft.starterOrigin, projection, rangeMode };
      const storage = store();
      saveValuationDraft(storage, original);
      saveValuation(storage, original);
      expect(loadValuationDraft(storage, original)).toEqual(original);
      expect(loadValuations(storage)).toEqual([original]);
      expect(decodeValuation(JSON.stringify(original))).toEqual(original);
      expect(loadValuationDraft(storage, original)?.draft.starterOrigin).toMatchObject({ projection, rangeMode, tailWideningPercent: 25, calibrationId: cashCalibrationId });
    }
  });

  it('preserves deliberately cleared crisis values and disabled scenarios without restoring defaults', () => {
    for (const enabled of [true, false]) {
      const original = study();
      original.draft.crisis = { enabled, shockPercent: null, startYear: null, durationYears: null, recoveryYears: null,
        extraAnnualCashCost: null, discountRate: null, terminalEquity: null, rationale: '' };
      original.draft.scenarios.mid.cashFlows[0] = null;
      const storage = store();
      saveValuationDraft(storage, original);
      expect(loadValuationDraft(storage, original)?.draft.crisis).toEqual(original.draft.crisis);
      expect(decodeValuation(JSON.stringify(original))).toEqual(original);
      expect(loadValuationDraft(storage, original)?.draft.scenarios.mid.cashFlows[0]).toBeNull();
    }
    const absent = study(); delete absent.draft.crisis;
    expect(decodeValuation(JSON.stringify(absent)).draft).not.toHaveProperty('crisis');
  });

  it('retains the prior working draft and revision when malformed crisis input is rejected', () => {
    const invalidPatches = [
      { enabled: 'true' }, { enabled: null }, { shockPercent: -1 }, { shockPercent: 301 },
      { shockPercent: '50' }, { startYear: 0 }, { startYear: 51 }, { startYear: 1.5 },
      { durationYears: 0 }, { durationYears: 51 }, { durationYears: 1.5 },
      { recoveryYears: -1 }, { recoveryYears: 51 }, { recoveryYears: 0.5 },
      { extraAnnualCashCost: -1 }, { extraAnnualCashCost: 1e13 }, { discountRate: -1 }, { discountRate: 101 },
      { terminalEquity: -1 }, { terminalEquity: 1e13 }, { rationale: null }, { rationale: 'x'.repeat(5001) },
    ];
    for (const patch of invalidPatches) {
      const prior = study(), storage = store();
      saveValuationDraft(storage, prior); saveValuation(storage, prior);
      const originalBytes = [...storage.values], originalWrites = storage.writes();
      const malformed = structuredClone(prior);
      Object.assign(malformed.draft.crisis!, patch);
      expect(() => saveValuationDraft(storage, malformed), JSON.stringify(patch)).toThrow(/retained/i);
      expect(() => saveValuation(storage, { ...malformed, id: 'invalid-revision' }), JSON.stringify(patch)).toThrow(/retained/i);
      expect(() => decodeValuation(JSON.stringify(malformed)), JSON.stringify(patch)).toThrow(/retained/i);
      expect(storage.writes()).toBe(originalWrites);
      expect([...storage.values]).toEqual(originalBytes);
      expect(loadValuationDraft(storage, prior)).toEqual(prior);
      expect(loadValuations(storage)).toEqual([prior]);
    }
  });

  it('rejects malformed crisis objects and nonfinite values before serialization can erase them', () => {
    const prior = study(), storage = store();
    saveValuationDraft(storage, prior);
    const originalBytes = [...storage.values];
    for (const crisis of [null, [], {}, { ...prior.draft.crisis, durationYears: undefined }]) {
      const malformed = { ...prior, draft: { ...prior.draft, crisis } } as unknown as SavedValuation;
      expect(() => saveValuationDraft(storage, malformed)).toThrow();
      expect(() => decodeValuation(JSON.stringify(malformed))).toThrow();
    }
    for (const value of [NaN, Infinity, -Infinity]) {
      const malformed = structuredClone(prior); malformed.draft.crisis!.shockPercent = value;
      expect(() => saveValuationDraft(storage, malformed)).toThrow();
    }
    // JSON.stringify converts NaN/Infinity to null; test malformed imported
    // numeric tokens directly so that this fixture does not lose its meaning.
    const encoded = JSON.stringify(prior);
    expect(() => decodeValuation(encoded.replace('"shockPercent":75', '"shockPercent":1e309'))).toThrow();
    expect(() => decodeValuation(encoded.replace('"shockPercent":75', '"shockPercent":NaN'))).toThrow();
    expect([...storage.values]).toEqual(originalBytes);
    expect(loadValuationDraft(storage, prior)).toEqual(prior);
  });

  it('rejects invalid v3 provenance while preserving supported unknown calibration versions verbatim', () => {
    const invalidPatches = [
      { calibrationId: '' }, { calibrationId: 'bad id' }, { calibrationId: 'bad_id' },
      { calibrationId: 'x'.repeat(151) }, { calibrationId: null }, { rangeMode: 'confidence' },
      { projection: 'damped' }, { tailWideningPercent: -1 }, { tailWideningPercent: 101 },
      { tailWideningPercent: null }, { spreadStepPercent: -1 }, { historyYears: 4 },
      { weights: [100] }, { weights: [30, 25, 20, 15, -10] },
    ];
    const original = study(), storage = store(); saveValuationDraft(storage, original);
    const originalBytes = [...storage.values];
    for (const patch of invalidPatches) {
      const malformed = structuredClone(original); Object.assign(malformed.draft.starterOrigin!, patch);
      expect(() => saveValuationDraft(storage, malformed), JSON.stringify(patch)).toThrow();
      expect(() => decodeValuation(JSON.stringify(malformed)), JSON.stringify(patch)).toThrow();
    }
    expect([...storage.values]).toEqual(originalBytes);
    const future = structuredClone(original);
    if (future.draft.starterOrigin?.id !== 'empirical-cash-starter-v3') throw new Error('Invalid test fixture');
    future.draft.starterOrigin.calibrationId = 'retained-calibration-version-2';
    expect(decodeValuation(JSON.stringify(future)).draft.starterOrigin).toEqual(future.draft.starterOrigin);
  });

  it('round-trips legacy v1/v2 and researched studies with optional independent crisis inputs', () => {
    const origins: (StarterOrigin | null)[] = [
      { id: 'weighted-cash-starter-v1', asOf: '2026-09-12', weights: [30, 25, 20, 15, 10], spreadPercent: 20 },
      { id: 'weighted-cash-starter-v2', asOf: '2026-09-12', weights: [30, 25, 20, 15, 10], spreadPercent: 10,
        historyYears: 5, projection: 'trend', spreadStepPercent: 10 },
      null,
    ];
    for (const origin of origins) for (const includeCrisis of [true, false]) {
      const original = study();
      if (origin) original.draft.starterOrigin = origin;
      else { delete original.draft.starterOrigin; original.draft.researchOrigin = { id: 'reviewed-company-2026-09-12', asOf: '2026-09-12' }; }
      if (!includeCrisis) delete original.draft.crisis;
      original.draft.researchAutofillDisabled = true;
      const storage = store(); saveValuationDraft(storage, original); saveValuation(storage, original);
      expect(loadValuationDraft(storage, original)).toEqual(original);
      expect(loadValuations(storage)).toEqual([original]);
      expect(decodeValuation(JSON.stringify(original))).toEqual(original);
    }
  });
});
