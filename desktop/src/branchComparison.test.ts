import { describe, expect, it } from 'vitest';
import { benchmark, bubbleDiameter, defaultSettings, observation, type BranchReport } from './branchComparison';
import { loadComparisons, saveComparison, compatibleComparison, type SavedComparison } from './savedComparisons';

const report = (value: number | null, extra: Partial<BranchReport> = {}): BranchReport => ({ year: 2025, start: '2025-01-01', end: '2025-12-31', report_date: '2026-02-15', source_as_of: '2026-08-10', currency: 'SEK', values: { operating_margin: value, revenues: value, total_assets: value }, ...extra });
const settings = { ...defaultSettings(2025), from: 2024, to: 2026 };
describe('branch comparisons use an explicit, comparable annual cohort', () => {
  it('computes an unweighted median and interpolated middle 50% from the whole cohort', () => {
    const rows = { a: [report(0)], b: [report(10)], c: [report(20)], d: [report(100)], missing: [report(null)] };
    const result = benchmark(rows, Object.keys(rows), settings);
    expect(result[1]).toEqual({ year: 2025, n: 4, total: 5, median: 15, q1: 7.5, q3: 40 });
    expect(result[0]).toEqual({ year: 2024, n: 0, total: 5, median: null, q1: null, q3: null });
    expect(result[2].n).toBe(0);
  });
  it('preserves negative values and zero; hides the band with fewer than four observations', () => {
    const r = benchmark({ a: [report(-8)], b: [report(0)] }, ['a', 'b'], settings)[1];
    expect(r.median).toBe(-4); expect(r.q1).toBe(null); expect(r.q3).toBe(null);
  });
  it('never combines monetary currencies or short/extended financial years', () => {
    const rows = { a: [report(100)], b: [report(200, { currency: 'EUR' })], c: [report(500, { start: '2025-07-01' })], d: [report(800, { start: '2024-01-01' })] };
    expect(benchmark(rows, Object.keys(rows), { ...settings, metric: 'revenues', currency: 'SEK' })[1].median).toBe(100);
    expect(observation(rows.b[0], { ...settings, currency: 'SEK' }).reason).toMatch(/currency/);
    expect(observation(rows.c[0], settings).reason).toMatch(/330–400/);
    expect(() => benchmark(rows, Object.keys(rows), { ...settings, metric: 'revenues', currency: 'all' })).toThrow(/currency/);
  });
  it('filters fiscal closing month and keeps a missing metric separate from a missing bubble size', () => {
    expect(observation(report(10), { ...settings, month: 6 }).value).toBe(null);
    const r = observation(report(10, { values: { operating_margin: 10, total_assets: null } }), { ...settings, size: 'total_assets', currency: 'SEK' });
    expect(r.value).toBe(10); expect(r.size).toBe(null);
    expect(observation(undefined, settings).reason).toMatch(/No saved/);
  });
  it('encodes size as area, with one scale over the entire displayed period', () => {
    expect(bubbleDiameter(100, 400) / bubbleDiameter(400, 400)).toBeCloseTo(.5);
    expect(bubbleDiameter(0, 400)).toBe(0); expect(bubbleDiameter(null, 400)).toBe(0);
  });
});

const saved = (): SavedComparison => ({ id: 'abc-123', title: 'Skog – genom cykeln', notes: 'Årsredovisningar: jämför markvärden.', created: '2026-09-10T12:00:00.000Z', release: 'a'.repeat(64), financial: 'b'.repeat(64), taxonomy: 'c'.repeat(64), branch: '21', settings: { ...settings, selected: ['102', '197'], focus: '102' } });
function storage(initial: string | null = null) { let value = initial; return { getItem: () => value, setItem: (_key: string, v: string) => { value = v; } }; }
describe('offline saved comparisons', () => {
  it('preserves Swedish notes and the exact data version across reopening', () => {
    const s = storage(), view = saved(); saveComparison(s, view);
    expect(loadComparisons(s)).toEqual([view]);
    expect(compatibleComparison(view, view.release, view.financial, view.taxonomy, ['102', '197'])).toBe(true);
    expect(compatibleComparison(view, 'd'.repeat(64), view.financial, view.taxonomy, ['102', '197'])).toBe(false);
    expect(compatibleComparison(view, view.release, 'd'.repeat(64), view.taxonomy, ['102', '197'])).toBe(false);
    expect(compatibleComparison(view, view.release, view.financial, view.taxonomy, ['102'])).toBe(false);
  });
  it('rejects invalid settings without overwriting previous saved work', () => {
    const s = storage(); saveComparison(s, saved());
    for (const settingsPatch of [{ selected: ['102', '102'] }, { from: 2030 }, { metric: 'invented' }, { metric: 'revenues', currency: 'all' }, { year: 1999 }]) {
      expect(() => saveComparison(s, { ...saved(), settings: { ...saved().settings, ...settingsPatch } } as SavedComparison)).toThrow();
      expect(loadComparisons(s)).toEqual([saved()]);
    }
    const bad = storage('{broken'); expect(() => saveComparison(bad, saved())).toThrow(); expect(bad.getItem()).toBe('{broken');
  });
  it('reports storage failure instead of claiming a save', () => {
    expect(() => saveComparison({ getItem: () => null, setItem: () => { throw new Error('Disk full'); } }, saved())).toThrow('Disk full');
  });
});
