import { describe, expect, it } from 'vitest';
import { atYear, quintile, tradeSlices, latestTrade, format, historyLines } from './model';

describe('evidence semantics', () => {
  it('keeps missing scores distinct from genuine zero', () => {
    expect(quintile(null)).toBeNull(); expect(quintile(NaN)).toBeNull();
    expect(quintile(0)).toBe(0); expect(quintile(100)).toBe(4);
    expect(format(null)).toBe('Not available'); expect(format(0)).toBe('0');
  });
  it('does not carry a prior observation into an unavailable year', () => {
    expect(atYear([{ year: 2022, value: 15, is_forecast: false }], 2023)).toBeUndefined();
  });
  it('never treats forecast values as historical observations', () => {
    expect(atYear([{ year: 2030, value: 2, is_forecast: true }], 2030)).toBeUndefined();
  });
  it('breaks chart lines across absent years and keeps forecasts separate', () => {
    const lines = historyLines([
      { year: 2020, value: 5, is_forecast: false },
      { year: 2022, value: 7, is_forecast: false },
      { year: 2023, value: 8, is_forecast: true },
    ], 1960);
    expect(lines.historical).toEqual([[2020, 5], [2021, null], [2022, 7]]);
    expect(lines.forecast).toEqual([[2022, 7], [2023, 8]]);
  });
  it('selects one trade vintage per reporter and preserves zero', () => {
    const rows = [2023, 2024].map(year => ({ iso2: 'SE', partner: 'DE', year, x_share: 0, m_share: 0, x_usd: 0, m_usd: 0 }));
    expect(latestTrade(rows, 'SE')).toEqual([rows[1]]);
  });
  it('retains the rest-of-world denominator in a top-partners chart', () => {
    expect(tradeSlices([{ iso2: 'SE', partner: 'DE', year: 2024, x_share: 20, m_share: 0, x_usd: 1, m_usd: 0 }], {}).map(s => s.value)).toEqual([20, 80]);
  });
  it('excludes the overlapping euro-area aggregate from partner shares', () => {
    const rows = ['DE', 'EU'].map(partner => ({ iso2: 'SE', partner, year: 2024, x_share: 20, m_share: 0, x_usd: 1, m_usd: 0 }));
    expect(latestTrade(rows, 'SE').map(r => r.partner)).toEqual(['DE']);
  });
  it('refuses to draw an invalid whole from overlapping shares', () => {
    expect(tradeSlices(['DE', 'US'].map(partner => ({ iso2: 'SE', partner, year: 2024, x_share: 70, m_share: 0, x_usd: 1, m_usd: 0 })), {})).toEqual([]);
  });
});
