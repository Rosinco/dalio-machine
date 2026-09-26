import { describe, expect, it } from 'vitest';
import type { CompanyKpiUnit, CompanyListCell } from './companyListModel';
import { companyListRangeActive, companyListRangeError, companyListRangeNumber, matchesCompanyListRange, parseCompanyListRange } from './companyListRanges';

const cell = (value: number | string | null, unit: CompanyKpiUnit = 'percent', currency: string | null = null, status?: CompanyListCell['status']): CompanyListCell => ({ value, unit, currency, display: String(value), detail: 'Fixture', date: null, ...(status ? { status } : {}) });

describe('numeric company column ranges', () => {
  it('accepts signed decimal points or Swedish decimal commas without grouping or coercion', () => {
    for (const [input, expected] of [['0', 0], ['-0', -0], ['+12', 12], [' -12,5 ', -12.5], ['.25', .25], [',25', .25], ['-.5', -.5], ['+0.001', .001], ['1,000', 1]] as const) expect(companyListRangeNumber(input)).toBe(expected);
    for (const input of ['', ' ', '-', '+', '.', ',', '1.', '1,', '1 000', '1\u00a0000', '1,234.56', '1.234,56', '1,234,567', '1_000', '1e3', '0x10', '12%', 'NaN', 'Infinity', '3x', '9'.repeat(400), `0.${'0'.repeat(400)}1`]) expect(companyListRangeNumber(input)).toBeNull();
  });

  it('treats zero and negative bounds inclusively, with either side independently unbounded', () => {
    const negativeToZero = { min: '-10,5', max: '0' };
    for (const value of [-10.5, -1, 0]) expect(matchesCompanyListRange(cell(value), negativeToZero)).toBe(true);
    for (const value of [-10.5001, .0001]) expect(matchesCompanyListRange(cell(value), negativeToZero)).toBe(false);
    expect(matchesCompanyListRange(cell(-1000), { min: '', max: '-2' })).toBe(true);
    expect(matchesCompanyListRange(cell(-1), { min: '', max: '-2' })).toBe(false);
    expect(matchesCompanyListRange(cell(1000), { min: '0', max: '' })).toBe(true);
    expect(matchesCompanyListRange(cell(-.01), { min: '0', max: '' })).toBe(false);
    expect(matchesCompanyListRange(cell(5), { min: '5', max: '5' })).toBe(true);
  });

  it('leaves missing, failed and loading observations alone when no bound is active', () => {
    for (const value of [null, 0, NaN]) for (const status of ['available', 'missing', 'loading', 'error'] as const) {
      for (const range of [undefined, null, { min: '', max: '' }, { min: ' ', max: '\t', currency: 'SEK' }]) {
        expect(companyListRangeActive(range)).toBe(false);
        expect(companyListRangeError(range, 'money')).toBeNull();
        expect(matchesCompanyListRange(cell(value, 'money', null, status), range)).toBe(true);
      }
    }
  });

  it('excludes missing, nonfinite, loading and error cells only with an active range', () => {
    for (const value of [null, NaN, Infinity, -Infinity, '10']) expect(matchesCompanyListRange(cell(value), { min: '0', max: '' })).toBe(false);
    for (const status of ['missing', 'loading', 'error'] as const) expect(matchesCompanyListRange(cell(10, 'percent', null, status), { min: '0', max: '' })).toBe(false);
    expect(matchesCompanyListRange(cell(0, 'percent', null, 'available'), { min: '0', max: '' })).toBe(true);
  });

  it('keeps malformed, partially typed and inverted active ranges closed until corrected', () => {
    for (const range of [{ min: '-', max: '' }, { min: '', max: '1,' }, { min: '1,2.3', max: '' }, { min: '10', max: '5' }, { min: 'Infinity', max: '50' }]) {
      expect(companyListRangeActive(range)).toBe(true);
      expect(companyListRangeError(range, 'percent')).not.toBeNull();
      for (const value of [-100, 0, 10, 100, null]) expect(matchesCompanyListRange(cell(value), range)).toBe(false);
    }
    expect(companyListRangeError({ min: '10', max: '5' }, 'percent')).toMatch(/Minimum.*maximum/);
  });

  it('requires an explicitly matching currency for money and per-share prices', () => {
    for (const unit of ['money', 'price'] as const) {
      for (const currency of [undefined, '', 'sek', 'SE', ' SEK ']) {
        const range = { min: '1', max: '10', currency };
        expect(companyListRangeError(range, unit)).toMatch(/currency/);
        expect(matchesCompanyListRange(cell(5, unit, 'SEK'), range)).toBe(false);
      }
      expect(matchesCompanyListRange(cell(5, unit, 'SEK'), { min: '1', max: '10', currency: 'SEK' })).toBe(true);
      expect(matchesCompanyListRange(cell(5, unit, 'USD'), { min: '1', max: '10', currency: 'SEK' })).toBe(false);
      expect(matchesCompanyListRange(cell(5, unit, null), { min: '1', max: '10', currency: 'SEK' })).toBe(false);
    }
    for (const unit of ['percent', 'points', 'multiple', 'count', 'number', 'shares_millions'] as const) expect(matchesCompanyListRange(cell(5, unit), { min: '1', max: '10' })).toBe(true);
  });

  it('preserves exact editable strings and invalid choices through parsing, without mutating the source', () => {
    const input = { min: '  -12,50 ', max: '1,', currency: 'sek' }, before = structuredClone(input);
    const parsed = parseCompanyListRange(input);
    expect(parsed).toEqual(input); expect(parsed).not.toBe(input); expect(input).toEqual(before);
    expect(parseCompanyListRange(undefined)).toBeUndefined(); expect(parseCompanyListRange(null)).toBeUndefined();
    expect(parseCompanyListRange({})).toEqual({ min: '', max: '' });
    for (const invalid of ['5', 5, [], { min: 5, max: '' }, { min: null, max: '' }]) {
      const range = parseCompanyListRange(invalid);
      expect(companyListRangeActive(range)).toBe(true);
      expect(matchesCompanyListRange(cell(100), range)).toBe(false);
    }
    expect(companyListRangeError(parseCompanyListRange({ min: '1', max: '', currency: 5 }), 'money')).toMatch(/currency/);
  });
});
