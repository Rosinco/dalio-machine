import { describe, expect, it } from 'vitest';
import { rankNormalQualityValue, type NormalQualityValueObservation } from './normalQualityValueRanking';

const observation = (id: string, changes: Partial<NormalQualityValueObservation> = {}): NormalQualityValueObservation => ({
  id, eligible: true, discount: 0, roce: 20, marginFloor: 8, cfoGrowth: 0, netDebtEbitda: 1, ...changes,
});
const resultRows = (observations: readonly NormalQualityValueObservation[]) => [...rankNormalQualityValue(observations).byId].sort(([a], [b]) => a.localeCompare(b));

describe('normal-year five-year NPV 60/40 quality and value ranking', () => {
  it('balances value and quality continuously instead of only breaking equal-discount ties', () => {
    const result = rankNormalQualityValue([
      observation('cheapest', { discount: 100, roce: 10, marginFloor: 5, cfoGrowth: 1, netDebtEbitda: 2 }),
      observation('balanced', { discount: 50, roce: 30, marginFloor: 15, cfoGrowth: 3, netDebtEbitda: 0 }),
      observation('expensive', { discount: -20, roce: 20, marginFloor: 10, cfoGrowth: 2, netDebtEbitda: 1 }),
    ]);
    expect(result.cohortSize).toBe(3);
    expect(result.byId.get('cheapest')).toMatchObject({ score: 60, quality: 0, discountRank: 100, reason: null });
    expect(result.byId.get('balanced')).toMatchObject({ score: 70, quality: 100, discountRank: 50, reason: null });
    expect(result.byId.get('expensive')).toMatchObject({ score: 20, quality: 50, discountRank: 0, reason: null });
  });

  it('uses exact tied midranks and preserves negative valuation comparisons', () => {
    const result = rankNormalQualityValue([-100, -20, -20, -1].map((discount, i) => observation(String(i), { discount })));
    expect([...result.byId.values()].map(row => row.discountRank)).toEqual([0, 50, 50, 100]);
    expect([...result.byId.values()].map(row => row.quality)).toEqual([50, 50, 50, 50]);
  });

  it('averages four quality percentiles without percentile-ranking their average again', () => {
    const result = rankNormalQualityValue([
      observation('a', { roce: 30, marginFloor: 30, cfoGrowth: 0, netDebtEbitda: 2 }),
      observation('b', { roce: 20, marginFloor: 20, cfoGrowth: 10, netDebtEbitda: 0 }),
      observation('c', { roce: 10, marginFloor: 10, cfoGrowth: 20, netDebtEbitda: 1 }),
    ]);
    expect(result.byId.get('b')).toMatchObject({ components: { roce: 50, marginFloor: 50, cfoGrowth: 50, netDebtEbitda: 100 }, quality: 62.5, score: 55 });
    expect(result.byId.get('a')?.quality).toBe(50);
    expect(result.byId.get('c')?.quality).toBe(37.5);
  });

  it('keeps mathematically equal composite scores exactly tied for a subsequent discount sort', () => {
    const result = rankNormalQualityValue([
      observation('a', { discount: 0, roce: 3, marginFloor: 3, cfoGrowth: 3, netDebtEbitda: 0 }),
      observation('b', { discount: 1, roce: 2, marginFloor: 2, cfoGrowth: 0, netDebtEbitda: 1 }),
      observation('c', { discount: 2, roce: 1, marginFloor: 1, cfoGrowth: 2, netDebtEbitda: 2 }),
      observation('d', { discount: 3, roce: 0, marginFloor: 0, cfoGrowth: 1, netDebtEbitda: 3 }),
    ]);
    // A: 6*0 + (6+6+6+6); B: 6*2 + (4+4+0+4). Both numerator 24.
    const a = result.byId.get('a')!, b = result.byId.get('b')!;
    expect(a.score).toBe(b.score);
    const tied = [{ id: 'a', rank: a }, { id: 'b', rank: b }].sort((a, b) => b.rank.score! - a.rank.score! || b.rank.discountRank! - a.rank.discountRank!);
    expect(tied.map(row => row.id)).toEqual(['b', 'a']);
  });

  it('treats all net-cash and zero-debt ratios equally and favours less positive debt', () => {
    const result = rankNormalQualityValue([-100, -1, 0, 1].map((netDebtEbitda, i) => observation(String(i), { netDebtEbitda })));
    expect([...result.byId.values()].map(row => row.components?.netDebtEbitda)).toEqual([100 * 2 / 3, 100 * 2 / 3, 100 * 2 / 3, 0]);
  });

  it('limits outliers to their relative rank rather than their accounting ratio magnitude', () => {
    const normal = [observation('a', { discount: 0, roce: 10 }), observation('b', { discount: 10, roce: 20 }), observation('c', { discount: 20, roce: 30 })];
    const extreme = normal.map(row => row.id === 'c' ? { ...row, discount: Number.MAX_VALUE, roce: Number.MAX_VALUE } : row);
    expect(resultRows(extreme)).toEqual(resultRows(normal));
  });

  it.each(['discount', 'roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda'] as const)('withholds every score when %s is missing or non-finite and excludes it from the denominator', metric => {
    for (const value of [null, NaN, Infinity, -Infinity]) {
      const result = rankNormalQualityValue([observation('valid'), observation('invalid', { [metric]: value })]);
      expect(result.cohortSize).toBe(1);
      expect(result.byId.get('valid')).toMatchObject({ score: 50, quality: 50, discountRank: 50 });
      expect(result.byId.get('invalid')).toMatchObject({ score: null, quality: null, discountRank: null, components: null });
      expect(result.byId.get('invalid')?.reason).toMatch(/missing|non-finite/i);
    }
  });

  it('uses the supplied fixed eligible cohort, so display selection does not rescale a score', () => {
    const observations = [observation('a', { discount: 0 }), observation('b', { discount: 10 }), observation('c', { discount: 20 }), observation('outside', { eligible: false, discount: 1e9 })];
    const result = rankNormalQualityValue(observations);
    const displayed = ['b'].map(id => result.byId.get(id));
    expect(result.cohortSize).toBe(3);
    expect(displayed[0]?.discountRank).toBe(50);
    expect(result.byId.get('outside')).toMatchObject({ score: null, quality: null, discountRank: null, components: null });
    expect(result.byId.get('outside')?.reason).toContain('fixed');
    expect(resultRows(observations.slice(0, 3))).toEqual(resultRows(observations).filter(([id]) => id !== 'outside'));
  });

  it('keeps distinct listing IDs as separate cohort observations even with identical inputs', () => {
    const result = rankNormalQualityValue([observation('venue-a'), observation('venue-b'), observation('other', { discount: 10 })]);
    expect(result.cohortSize).toBe(3);
    expect(result.byId.get('venue-a')?.discountRank).toBe(25);
    expect(result.byId.get('venue-b')?.discountRank).toBe(25);
    expect(result.byId.get('other')?.discountRank).toBe(100);
  });

  it('withholds the whole cohort for ambiguous repeated listing IDs', () => {
    const result = rankNormalQualityValue([observation('same'), observation('same', { discount: 100 }), observation('other')]);
    expect(result.cohortSize).toBe(0);
    for (const row of result.byId.values()) {
      expect(row).toMatchObject({ score: null, quality: null, discountRank: null, components: null });
      expect(row.reason).toMatch(/duplicate/i);
    }
  });

  it('supports an empty or all-ineligible cohort, and uses a neutral singleton score even for zero inputs', () => {
    expect(rankNormalQualityValue([])).toEqual({ cohortSize: 0, byId: new Map() });
    expect(rankNormalQualityValue([observation('x', { eligible: false })]).cohortSize).toBe(0);
    const singleton = rankNormalQualityValue([observation('zero', { discount: 0, roce: 0, marginFloor: 0, cfoGrowth: 0, netDebtEbitda: 0 })]);
    expect(singleton.byId.get('zero')).toEqual({ score: 50, quality: 50, discountRank: 50, components: { roce: 50, marginFloor: 50, cfoGrowth: 50, netDebtEbitda: 50 }, reason: null });
  });

  it('does not mutate inputs and remains deterministic when the input order is reversed', () => {
    const observations = [observation('a', { discount: 10, netDebtEbitda: -3 }), observation('b', { discount: -20, roce: 100 }), observation('c', { discount: 10, cfoGrowth: 5 })];
    const before = structuredClone(observations);
    const frozen = Object.freeze(observations.map(row => Object.freeze(row)));
    expect(resultRows(frozen)).toEqual(resultRows([...frozen].reverse()));
    expect(observations).toEqual(before);
    for (const [, row] of resultRows(frozen)) expect(row.score).toBeGreaterThanOrEqual(0);
    for (const [, row] of resultRows(frozen)) expect(row.score).toBeLessThanOrEqual(100);
  });
});
