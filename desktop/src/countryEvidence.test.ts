import { describe, expect, it } from 'vitest';
import { evidenceCode, evidenceDate, evidenceLines, evidenceValue, nativeStatus, safeSource, validateCountryEvidence, type EvidenceHistory } from './countryEvidence';

const series = (rows: any[], frequency = 'annual') => ({ frequency, observations: rows } as EvidenceHistory);
describe('independent country evidence', () => {
  it('preserves annual gaps, zero values and calendar-convention forecasts', () => {
    const lines = evidenceLines(series([{ year: 2023, value: 0, status: 'estimate_or_outturn' }, { year: 2025, value: -2, status: 'estimate_or_outturn' }, { year: 2026, value: 1, status: 'forecast_calendar_convention' }]));
    expect(lines.labels).toEqual(['2023', '2024', '2025', '2026']); expect(lines.historical).toEqual([0, null, -2, null]); expect(lines.forecast).toEqual([null, null, null, 1]);
  });
  it('retains native missing months and daily observations without forward filling', () => {
    expect(evidenceLines(series([{ period: '2026-01', value: 3, status: 'observed' }, { period: '2026-03', value: null, status: 'not_reported' }], 'monthly')).historical).toEqual([3, null, null]);
    expect(evidenceLines(series([{ period: '2026-09-04', value: 2, status: 'observed' }, { period: '2026-09-07', value: 2, status: 'observed' }], 'daily')).labels).toEqual(['2026-09-04', '2026-09-07']);
  });
  it('keeps reference-date and monthly-mean debt context distinct', () => {
    expect(evidenceDate({ value: 5, status: 'observed', year: 2026, period_start: '2026-08-01', period_end: '2026-08-31' })).toBe('2026-08-01 – 2026-08-31');
    expect(evidenceValue({ value: 0, status: 'observed', unit: 'percent' })).toBe('0 percent'); expect(evidenceValue(null)).toBe('Not available');
    expect(evidenceCode('GB')).toBe('UK');
  });
  it('rejects unsafe source URLs and wrong country identities', () => {
    expect(safeSource('javascript:alert(1)')).toBe(false); expect(safeSource('https://user:password@example.test')).toBe(false); expect(safeSource('https://www.scb.se/')).toBe(true);
    expect(() => validateCountryEvidence({ version: 1, country: 'FI', name: 'Finland' }, 'SE')).toThrow(/identity/);
  });
  it('shows native provisional flags without guessing unknown publisher meanings', () => {
    const point = { value: 3.78, status: 'observed', native_status: 'P' };
    expect(nativeStatus(point, 'BUNDESBANK_MONITORING')).toBe('Native P · provisional');
    expect(nativeStatus(point, 'OTHER')).toBe('Native status: P');
    expect(nativeStatus({ ...point, native_status: null })).toBe('');
    expect(evidenceValue({ value: 655.6, unit: 'SEK_bn', status: 'forecast' })).toBe('655.6 billion SEK');
  });

});
