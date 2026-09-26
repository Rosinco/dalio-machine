import { describe, expect, it, vi } from 'vitest';
import { blankPortfolioStress, calculatePortfolioStress, loadPortfolioStress, portfolioStressStorageKey, restorePortfolioStressSession, savePortfolioStress, type PortfolioStressDraft } from './portfolioStress';

const completed = (): PortfolioStressDraft => ({
  ...blankPortfolioStress(), exposureDate: '2026-09-26', exposureBasis: 'Starting equity weights, source: dated account statement.',
  shock: 'A shared financing freeze.', evidence: 'Both positions rely on the same short-term lender; sources reviewed.',
  assumptions: 'Both equity positions become worthless in this combined shock.', tolerancePercent: '30',
  rows: [
    { id: 'a', name: 'Company A', driver: 'Shared lender', weightPercent: '20', lossPercent: '100', notes: '' },
    { id: 'b', name: 'Company B', driver: 'Shared lender', weightPercent: '20', lossPercent: '100', notes: '' },
  ],
});
const storageWith = (raw: string | null = null) => {
  const values = new Map<string, string>(raw === null ? [] : [[portfolioStressStorageKey, raw]]);
  return { values, getItem: vi.fn((key: string) => values.get(key) ?? null), setItem: vi.fn((key: string, value: string) => { values.set(key, value); }) };
};
const savedAt = '2026-09-26T12:00:00.000Z';

describe('manual common-cause portfolio stress', () => {
  it('adds starting-capital losses: two 20% positions losing 100% cost 40%, without compounding', () => {
    const result = calculatePortfolioStress(completed());
    expect(result.errors).toEqual([]);
    expect(result.result).toEqual({ enteredWeightPercent: 40, residualWeightPercent: 60, lossPercent: 40, remainingCapitalPercent: 60, tolerancePercent: 30, toleranceDifferencePercent: 10, toleranceComparison: 'above' });
  });

  it('weights partial losses and describes a tolerance comparison without a probability', () => {
    const draft = completed();
    draft.rows[0].lossPercent = '50'; draft.rows[1].weightPercent = '80'; draft.rows[1].lossPercent = '25';
    expect(calculatePortfolioStress(draft).result).toMatchObject({ lossPercent: 30, remainingCapitalPercent: 70, residualWeightPercent: 0, toleranceComparison: 'equal', toleranceDifferencePercent: 0 });
    draft.tolerancePercent = '40';
    expect(calculatePortfolioStress(draft).result).toMatchObject({ toleranceComparison: 'below', toleranceDifferencePercent: -10 });
  });

  it('withholds outputs for missing context, blank rows, impossible dates, or invalid percentages', () => {
    const blank = blankPortfolioStress();
    expect(blank.rows[0].weightPercent).toBe('');
    expect(blank.tolerancePercent).toBe('');
    expect(calculatePortfolioStress(blank).result).toBeNull();
    for (const field of ['exposureBasis', 'shock', 'evidence', 'assumptions', 'tolerancePercent'] as const) {
      expect(calculatePortfolioStress({ ...completed(), [field]: '' }).result).toBeNull();
    }
    expect(calculatePortfolioStress({ ...completed(), exposureDate: '2026-02-30' }).result).toBeNull();
    expect(calculatePortfolioStress({ ...completed(), rows: [] }).result).toBeNull();
    for (const value of ['', '-1', '101', 'Infinity', 'NaN', '0x10', '10%']) {
      for (const field of ['weightPercent', 'lossPercent'] as const) {
        const draft = completed(); draft.rows[1][field] = value;
        expect(calculatePortfolioStress(draft).result).toBeNull();
      }
    }
    const incomplete = completed(); incomplete.rows.push({ ...blank.rows[0], id: 'blank' });
    expect(calculatePortfolioStress(incomplete).result).toBeNull();
  });

  it('rejects overallocated weights while accepting zero losses and fractional allocations totaling 100%', () => {
    const draft = completed(); draft.rows[1].weightPercent = '81';
    expect(calculatePortfolioStress(draft).errors).toContain('Entered starting weights exceed 100%. Review overlaps and the exposure basis.');
    expect(calculatePortfolioStress(draft).result).toBeNull();
    draft.rows[0].weightPercent = '33.3'; draft.rows[1].weightPercent = '66.7';
    draft.rows.forEach(row => { row.lossPercent = '0'; }); draft.tolerancePercent = '0';
    expect(calculatePortfolioStress(draft).result).toMatchObject({ enteredWeightPercent: 100, lossPercent: 0, remainingCapitalPercent: 100, toleranceComparison: 'equal' });
  });
});

describe('isolated explicit-save portfolio storage', () => {
  it('restores unsaved session work on remount without writes, while retaining the original conflict basis', () => {
    const storage = storageWith();
    const original = loadPortfolioStress(storage);
    const unsaved = { ...original, draft: completed(), dirty: true };
    expect(restorePortfolioStressSession(loadPortfolioStress(storage), unsaved)).toEqual({ ...unsaved, storageChanged: false });
    expect(storage.setItem).not.toHaveBeenCalled();
    const newerRaw = savePortfolioStress(storage, blankPortfolioStress(), null, savedAt);
    const restored = restorePortfolioStressSession(loadPortfolioStress(storage), unsaved);
    expect(restored).toEqual({ ...unsaved, storageChanged: true });
    expect(restored.raw).toBeNull();
    expect(() => savePortfolioStress(storage, restored.draft, restored.raw, savedAt)).toThrow(/changed/);
    expect(storage.values.get(portfolioStressStorageKey)).toBe(newerRaw);
    expect(restorePortfolioStressSession(loadPortfolioStress(storage), { ...unsaved, dirty: false }).raw).toBe(newerRaw);
  });

  it('never writes while loading and preserves intentionally incomplete draft fields on explicit save', () => {
    const storage = storageWith();
    const first = loadPortfolioStress(storage);
    expect(storage.setItem).not.toHaveBeenCalled();
    const draft = blankPortfolioStress(); draft.rows[0].weightPercent = '101';
    const raw = savePortfolioStress(storage, draft, first.raw, savedAt);
    expect(storage.setItem).toHaveBeenCalledExactlyOnceWith(portfolioStressStorageKey, raw);
    expect(loadPortfolioStress(storage)).toEqual({ raw, savedAt, draft });
  });

  it('retains malformed and unknown-version records without replacing their bytes', () => {
    for (const raw of ['{broken', JSON.stringify({ version: 2, draft: completed(), savedAt }), JSON.stringify({ version: 1, draft: { ...completed(), unknownField: 'preserve' }, savedAt })]) {
      const storage = storageWith(raw);
      expect(() => loadPortfolioStress(storage)).toThrow(/preserved/);
      expect(() => savePortfolioStress(storage, completed(), raw, savedAt)).toThrow(/preserved/);
      expect(storage.values.get(portfolioStressStorageKey)).toBe(raw);
      expect(storage.setItem).not.toHaveBeenCalled();
    }
  });

  it('detects intervening saves and does not replace newer work', () => {
    const storage = storageWith();
    const first = loadPortfolioStress(storage);
    const newer = savePortfolioStress(storage, completed(), null, savedAt);
    expect(() => savePortfolioStress(storage, blankPortfolioStress(), first.raw, savedAt)).toThrow(/changed/);
    expect(storage.values.get(portfolioStressStorageKey)).toBe(newer);
    expect(storage.setItem).toHaveBeenCalledTimes(1);
  });

  it('reports storage failures without falling back to overwriting another key', () => {
    const storage = storageWith();
    storage.setItem.mockImplementation(() => { throw new Error('Quota exceeded'); });
    expect(() => savePortfolioStress(storage, completed(), null, savedAt)).toThrow('Quota exceeded');
    expect(storage.values.size).toBe(0);
    storage.getItem.mockImplementation(() => { throw new Error('Storage unavailable'); });
    expect(() => loadPortfolioStress(storage)).toThrow('Storage unavailable');
  });
});
