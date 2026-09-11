import { describe, expect, it } from 'vitest';
import { blankValuation, calculateValuation } from './valuation';
import { buildResearchedValuation, holmenStudy, researchedStudyFor, shouldStartResearchedStudy } from './researchedValuations';

describe('researched company valuation', () => {
  it('requires the reviewed listing identity and archived research source', () => {
    const entry = { id: '102', isin: 'SE0011090018' };
    const sources = [{ sha256: holmenStudy.deepDive.sha256 }];
    expect(researchedStudyFor(entry, sources)?.id).toBe(holmenStudy.id);
    expect(researchedStudyFor({ ...entry, id: '197' }, sources)).toBeNull();
    expect(researchedStudyFor({ ...entry, isin: 'SE0011090000' }, sources)).toBeNull();
    expect(researchedStudyFor(entry, [])).toBeNull();
    expect(researchedStudyFor(entry, [{ sha256: 'f'.repeat(64) }])).toBeNull();
  });
  it('opens a complete, dated Holmen analysis without waiting for async vendor history', () => {
    const { draft, bridge, recovery } = buildResearchedValuation(holmenStudy);
    expect(draft.marketCap).toBeCloseTo(49492.961686, 6);
    expect(draft.priceDate).toBe('2026-08-07');
    expect(draft.valuationDate).toBe('2026-09-11');
    expect(draft.priceSource).toMatch(/B-equivalent/);
    expect(draft.researchOrigin?.id).toBe(holmenStudy.id);
    expect(bridge.operatingCash).toBe(3444);
    expect(bridge.capex).toBe(1557);
    expect(bridge.leasePrincipal).toBe(134);
    expect(bridge.cashTaxes).toBe(-10);
    expect(bridge.normalizedAvailableCash).toBe(1243);
    expect(Object.values(draft.capital).every(Number.isFinite)).toBe(true);
    expect(draft.capital.tangibleEquity).toBe(53700);
    expect(draft.capital.grossDebt).toBe(6917);
    expect(draft.capital.averageTCE).toBe(60083);
    expect(draft.capital.nopat).toBe(1852.5);
    expect(draft.capital.surplusCash).toBe(0);
    expect(holmenStudy.recovery.assets.reduce((sum, asset) => sum + asset.book, 0)).toBe(81068 - 482);
    // Reconcile the publisher's broader net-financial-debt perimeter, including
    // pensions and financial receivables, rather than equating it with borrowings.
    expect(draft.capital.grossDebt! + 5 - 29 - 211 - 18 - 182).toBe(6482);
    expect(calculateValuation(draft).ready).toBe(true);
    for (const key of ['low', 'mid', 'high'] as const) {
      expect(recovery[key].claims).toBe(26887);
      expect(recovery[key].net).toBe(draft.scenarios[key].recoveryEquity);
      expect(draft.scenarios[key].recoveryYear).toBeGreaterThan(0);
    }
  });
  it('uses after-financing cash, a common required return and one terminal equity sale', () => {
    const { draft } = buildResearchedValuation(holmenStudy), result = calculateValuation(draft);
    expect(result.scenarios.low.stakeValue).toBeCloseTo(210.75968198, 6);
    expect(result.scenarios.mid.stakeValue).toBeCloseTo(401.37954124, 6);
    expect(result.scenarios.high.stakeValue).toBeCloseTo(783.87018594, 6);
    expect(result.scenarios.high.payback.year).toBe(16);
    expect(result.scenarios.high.discountedPayback.year).toBeNull();
    expect(result.scenarios.mid.payback.year).toBeNull();
    for (const key of ['low', 'mid', 'high'] as const) {
      const s = draft.scenarios[key];
      expect(s.discountRate).toBe(9);
      expect(s.terminalEquity).toBeCloseTo(s.cashFlows[19]! * (1 + holmenStudy.scenarios[key].matureGrowth / 100) / (0.09 - holmenStudy.scenarios[key].matureGrowth / 100) * 0.98);
      expect(result.scenarios[key].npv).toBeLessThan(0);
    }
    const before = result.scenarios.mid.value;
    draft.capital.grossDebt = 99999; draft.scenarios.mid.recoveryEquity = 99999;
    expect(calculateValuation(draft).scenarios.mid.value).toBe(before);
  });
  it('migrates untouched drafts, including a selected historical price, but preserves entered forecasts', () => {
    expect(shouldStartResearchedStudy(null)).toBe(true);
    const d = blankValuation('Holmen');
    d.marketCap = 54246; d.priceSource = 'A price selected in the old workspace';
    expect(shouldStartResearchedStudy(d)).toBe(true);
    d.scenarios.mid.cashFlows[0] = 0;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    d.scenarios.mid.cashFlows[0] = null; d.scenarios.mid.discountRate = 0;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    d.scenarios.mid.discountRate = null; d.researchAutofillDisabled = true;
    expect(shouldStartResearchedStudy(d)).toBe(false);
    const seeded = buildResearchedValuation(holmenStudy).draft;
    seeded.scenarios.mid.cashFlows[0] = null;
    expect(shouldStartResearchedStudy(seeded)).toBe(false);
  });
  it('returns fresh editable drafts without mutating the researched baseline', () => {
    const first = buildResearchedValuation(holmenStudy).draft;
    first.scenarios.mid.cashFlows[0] = 500; first.notes.macro = 'My view';
    const next = buildResearchedValuation(holmenStudy).draft;
    expect(next.scenarios.mid.cashFlows[0]).toBe(1500);
    expect(next.notes.macro).not.toBe('My view');
  });
});
