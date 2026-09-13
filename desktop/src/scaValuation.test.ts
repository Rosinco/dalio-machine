import { describe, expect, it } from 'vitest';
import scaJson from '../research/valuations/sca-2026-09-12.json';
import { buildResearchedValuation, holmenStudy, researchedStudyFor, reviewedStudy } from './researchedValuations';
import { calculateValuation, scenarioKeys } from './valuation';

const sca = reviewedStudy(scaJson);

describe('reviewed SCA valuation', () => {
  it('binds the SCA B listing to the exact archived review in the selected directory', () => {
    const entry = { id: '197', isin: 'SE0000112724' };
    const evidence = [{ path: sca.deepDive.path, sha256: sca.deepDive.sha256 }];
    expect(researchedStudyFor(entry, evidence)?.id).toBe('sca-2026-09-12-v1');
    expect(researchedStudyFor({ ...entry, isin: 'SE0000171886' }, evidence)).toBeNull();
    expect(researchedStudyFor(entry, [{ ...evidence[0], path: 'another-company/deep_dive.md' }])).toBeNull();
    expect(researchedStudyFor(entry, [{ ...evidence[0], sha256: holmenStudy.deepDive.sha256 }])).toBeNull();
    expect(researchedStudyFor(entry, [])).toBeNull();
  });

  it('retains original-report common equity and the separate gross/net debt perimeters', () => {
    const { draft } = buildResearchedValuation(sca);
    expect(draft.marketCap).toBeCloseTo(77468.3765367, 7);
    expect(draft.priceDate).toBe('2026-08-07');
    expect(draft.valuationDate).toBe('2026-09-12');
    expect(draft.priceSource).toContain('702.342489');
    expect(draft.capital).toEqual({ tangibleEquity: 99902, averageTCE: 112527, nopat: 2104.1, grossDebt: 15536, surplusCash: 0 });
    expect(draft.notes.financing).toContain('10,859');
    expect(draft.notes.financing).toContain('4,401');
    expect(draft.capital.tangibleEquity).toBe(101194 - 17 - 1275);
    expect(draft.capital.grossDebt).toBe(12582 + 2524 + 256 + 174);
    expect(draft.capital.averageTCE).toBe(((114920 - 1025) + (112460 - 1301)) / 2);
    expect(draft.capital.nopat).toBeCloseTo((4432 - 1782) * 0.794, 6);
  });

  it('reconciles the explicit first-year cash bridges and independently calculated present values', () => {
    const { draft } = buildResearchedValuation(sca), result = calculateValuation(draft);
    expect(result.ready).toBe(true);
    expect(draft.scenarios.low.cashFlows[0]).toBe(1900 + 2250 - 2400 - 150 - 450 - 300 - 220);
    expect(draft.scenarios.mid.cashFlows[0]).toBe(2650 + 2250 - 2400 - 150 - 450 - 450 - 220);
    expect(draft.scenarios.high.cashFlows[0]).toBe(4000 + 2300 - 2600 - 250 - 450 - 730 - 220);
    // Independently reconciled from the source worksheet; not read back from
    // the production calculator or assumed to equal current vendor FCF.
    expect(result.scenarios.low.stakeValue).toBeCloseTo(84.8295322415963, 7);
    expect(result.scenarios.mid.stakeValue).toBeCloseTo(210.275208780506, 7);
    expect(result.scenarios.high.stakeValue).toBeCloseTo(427.765499120273, 7);
    for (const key of scenarioKeys) {
      expect(result.scenarios[key].payback.year).toBeNull();
      expect(result.scenarios[key].discountedPayback.year).toBeNull();
      expect(draft.scenarios[key].discountRate).toBe(9);
    }
  });

  it('deducts claims, NCI, additional costs and cash burn once from the separate recovery case', () => {
    const { draft, recovery } = buildResearchedValuation(sca);
    if (sca.recovery.status !== 'available') throw new Error('SCA recovery is missing');
    expect(sca.recovery.assets.reduce((sum, asset) => sum + asset.book!, 0)).toBe(147667);
    const outcomes = { low: 9828, mid: 39435.7, high: 68351.45 };
    for (const key of scenarioKeys) {
      expect(recovery![key].claims).toBe(46473 + 17);
      expect(recovery![key].net).toBeCloseTo(outcomes[key], 6);
      expect(draft.scenarios[key].recoveryEquity).toBeCloseTo(outcomes[key], 6);
    }
    const value = calculateValuation(draft).scenarios.mid.value;
    draft.capital.grossDebt = 999999;
    draft.scenarios.mid.recoveryEquity = 999999;
    expect(calculateValuation(draft).scenarios.mid.value).toBe(value);
  });

  it('keeps edits isolated from both researched baselines', () => {
    const draft = buildResearchedValuation(sca).draft;
    draft.scenarios.mid.cashFlows[0] = null;
    draft.capital.surplusCash = 100;
    draft.notes.macro = 'My scenario';
    const next = buildResearchedValuation(sca).draft;
    expect(next.scenarios.mid.cashFlows[0]).toBe(1230);
    expect(next.capital.surplusCash).toBe(0);
    expect(next.notes.macro).not.toBe('My scenario');
    expect(calculateValuation(buildResearchedValuation(holmenStudy).draft).scenarios.mid.stakeValue).toBeCloseTo(401.37954124, 6);
  });
});
