import { describe, expect, it } from 'vitest';
import fixture from '../tests/fixtures/financial.json';
import { decodeFinancialCompany, validateFinancialIndex, type FinancialIndex } from './financialData';
import { financialSeries } from './business';

function sample() { return structuredClone({ ...fixture.index, id: 'b'.repeat(64), bytes: 1000 }) as FinancialIndex; }
describe('source-bound financial histories', () => {
  it('checks taxonomy binding and reconciled coverage', () => {
    const i = sample();
    expect(() => validateFinancialIndex(i, 'a'.repeat(64))).not.toThrow();
    expect(() => validateFinancialIndex(i, 'b'.repeat(64))).toThrow();
    i.summary.annual++;
    expect(() => validateFinancialIndex(i, 'a'.repeat(64))).toThrow();
  });
  it('recovers report-currency amounts, preserving raw values, zeroes and sources', () => {
    const r = decodeFinancialCompany(fixture.companies['102'], sample(), '102').annual[0];
    expect(r.raw.revenues).toBe(200); expect(r.values.revenues).toBe(100);
    expect(r.values.gross_income).toBe(30); expect(r.values.cash_flow_from_operating_activities).toBe(0);
    expect(r.values.operating_margin).toBe(10); expect(r.values.return_on_capital).toBe(20);
    expect(r.source_as_of).toBe('2026-08-10'); expect(r.currency).toBe('EUR');
  });
  it('leaves missing FX values unavailable', () => {
    const c = structuredClone(fixture.companies['102']), i = sample();
    c.annual[0][6] = null; i.companies['102'].annual.unavailable = 1;
    const r = decodeFinancialCompany(c, i, '102').annual[0];
    expect(r.raw.revenues).toBe(200); expect(Object.values(r.values).every(v => v === null)).toBe(true);
  });
  it('rejects wrong identities, unavailable source references and invalid dates', () => {
    for (const [column, value] of [[7, 'unknown'], [4, '2699-01-01'], [3, '2025-02-30'], [5, 'invalid']] as const) {
      const c = structuredClone(fixture.companies['102']); c.annual[0][column] = value;
      expect(() => decodeFinancialCompany(c, sample(), '102')).toThrow();
    }
    expect(() => decodeFinancialCompany(fixture.companies['102'], sample(), '20')).toThrow();
  });
  it('rejects incorrect coverage and duplicate periods', () => {
    const i = sample(); i.companies['102'].annual.gaps = 3;
    expect(() => decodeFinancialCompany(fixture.companies['102'], i, '102')).toThrow();
    const c = structuredClone(fixture.companies['102']); c.annual.push(c.annual[0]);
    expect(() => decodeFinancialCompany(c, sample(), '102')).toThrow();
  });
  it('keeps missing fiscal periods and changes in monetary currency blank', () => {
    const r = decodeFinancialCompany(fixture.companies['102'], sample(), '102').annual[0];
    const series = financialSeries([{ ...r, year: 2023 }, { ...r, year: 2025, currency: 'SEK' }], 'revenues', 'EUR');
    expect(series.labels).toEqual(['2023', '2024', '2025']); expect(series.values).toEqual([100, null, null]);
  });
});
