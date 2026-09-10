import { describe, expect, it } from 'vitest';
import fixture from '../tests/fixtures/market.json';
import legacy from '../tests/fixtures/financial.json';
import { decodeFinancialCompany, validateFinancialIndex, type FinancialIndex } from './financialData';
import { benchmark, defaultSettings, needsCurrency, observation, projectAnnual } from './branchComparison';
import { loadComparisons, saveComparison, type SavedComparison } from './savedComparisons';

const sample = () => structuredClone({ ...fixture.index, id: 'b'.repeat(64), bytes: 1000 }) as FinancialIndex;
describe('dated all-company market histories', () => {
  it('validates local units, SEK conversion, date and provenance', () => {
    const index = sample(); validateFinancialIndex(index, 'a'.repeat(64));
    const c = decodeFinancialCompany(fixture.companies['102'], index, '102');
    expect(c.market[0]).toMatchObject({ year: 2025, local: 200, sek: 2400, price_date: '2026-02-02', fx_date: '2026-01-30' });
  });
  it('keeps v1 financial packs usable without inventing a market history', () => {
    const index = { ...legacy.index, id: 'b'.repeat(64), bytes: 1000 } as FinancialIndex;
    validateFinancialIndex(index, 'a'.repeat(64));
    expect(decodeFinancialCompany(legacy.companies['102'], index, '102').market).toEqual([]);
  });
  it.each([{ local: 300 }, { sek: 300 }, { price_date: '2026-01-30' }, { fx_date: '2026-02-03' }, { fx_date: '2026-01-01' }, { source_id: 'unknown' }, { flags: ['unknown'] }, { flags: ['share_basis'] }, { fx_instruments: ['123'] }, { fx_method: 'static_fallback' }])('rejects an inconsistent market record: %j', patch => {
    const c = structuredClone(fixture.companies['102']); Object.assign(c.market[0], patch);
    expect(() => decodeFinancialCompany(c, sample(), '102')).toThrow();
  });
  it('checks market coverage against all annual reports and source clocks', () => {
    const i = sample(); i.companies['102'].market!.count++;
    expect(() => validateFinancialIndex(i, 'a'.repeat(64))).toThrow();
    const j = sample(); j.market!.sources[0].as_of = '2025-06-21';
    expect(() => validateFinancialIndex(j, 'a'.repeat(64))).toThrow();
  });
  it('sizes bubbles in SEK across reporting currencies while preserving the Y-axis denominator', () => {
    const c = decodeFinancialCompany(fixture.companies['102'], sample(), '102');
    const data = projectAnnual(c.annual, c.market), s = { ...defaultSettings(2025), size: 'market_cap' as const };
    expect(needsCurrency(s)).toBe(false);
    expect(observation(data[0], s).size).toBe(2400);
    expect(benchmark({ '102': data }, ['102'], s).at(-1)?.n).toBe(1);
    const missing = { ...data[0], market: { ...c.market[0], sek: null, fx_rate: null, fx_date: null, fx_method: null, fx_instruments: [], flags: ['missing_fx'] } };
    expect(observation(missing, s).value).not.toBe(null);
    expect(observation(missing, s).size).toBe(null);
    expect(observation(missing, s).sizeReason).toMatch(/FX/);
  });
  it('saves the market-cap size selection with the exact pack identity', () => {
    let raw: string | null = null;
    const storage = { getItem: () => raw, setItem: (_key: string, value: string) => { raw = value; } };
    const view: SavedComparison = { id: 'market-test', title: 'Börsvärde', notes: 'År och kursdatum', created: '2026-09-10T12:00:00Z', release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), branch: '21', settings: { ...defaultSettings(2025), size: 'market_cap', selected: ['102'], focus: '102' } };
    saveComparison(storage, view); expect(loadComparisons(storage)).toEqual([view]);
  });
});
