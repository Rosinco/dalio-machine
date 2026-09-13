import { describe, expect, it } from 'vitest';
import { blankValuation } from './valuation';
import { decodeValuation, validateValuation } from './savedValuations';
import { shouldStartResearchedStudy } from './researchedValuations';
import { isUntouchedLegacyStarter } from './starterMigration';

const saved = () => ({ format: 'macro-atlas-valuation', version: 1, id: 'purchase-test', company: '696', created: '2026-09-13T00:00:00.000Z', release: 'a'.repeat(64), financial: null, taxonomy: null, draft: blankValuation('Test', 'SEK', '2026-09-13') });
describe('saved purchase policy', () => {
  it('leaves existing drafts unchanged until a policy is edited', () => {
    const old = saved();
    expect(decodeValuation(JSON.stringify(old))).toEqual(old);
    expect(old.draft).not.toHaveProperty('purchaseRange');
  });
  it('retains authored settings, deliberate blanks and incomplete dated share bases', () => {
    const v = saved();
    v.draft.purchaseRange = { marginOfSafetyPercent: null, referenceScenario: 'mid', candidateEquity: null, unit: 'share', shareBasis: { sharesMillions: null, date: '', source: '', currency: 'SEK' } };
    expect(decodeValuation(JSON.stringify(v))).toEqual(v);
    expect(shouldStartResearchedStudy(v.draft)).toBe(false);
    v.draft.starterOrigin = { id: 'weighted-cash-starter-v1', asOf: '2026-09-13', weights: [30,25,20,15,10], spreadPercent: 20 };
    expect(isUntouchedLegacyStarter(v.draft, structuredClone(v.draft))).toBe(false);
  });
  it('rejects malformed policy inputs without repairing them into defaults', () => {
    for (const patch of [{ marginOfSafetyPercent: '30' }, { referenceScenario: 'average' }, { candidateEquity: Infinity }, { unit: 'dollars' }, { shareBasis: { sharesMillions: '2', date: '', source: '', currency: 'SEK' } }]) {
      const v = saved(); v.draft.purchaseRange = { marginOfSafetyPercent: 30, referenceScenario: 'mid', unit: 'equity', ...patch } as never;
      expect(() => validateValuation(v)).toThrow(/unsupported valuation/);
      if ('candidateEquity' in patch) continue; // JSON serializes infinity as null.
      expect(() => decodeValuation(JSON.stringify(v))).toThrow(/unsupported valuation/);
    }
  });
});
