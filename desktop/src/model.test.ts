import { describe, expect, it } from 'vitest';
import { atYear, quintile, tradeSlices, latestTrade, format, historyLines, historyColor, historyPalette } from './model';

describe('colour meaning', () => {
  const debt = { scored: true, higher_is_better: false };
  const output = { scored: true, higher_is_better: true };
  it('shows low debt as green and high debt as red', () => {
    expect(historyColor(20, [20, 150], debt)).toBe('#3b946c');
    expect(historyColor(150, [20, 150], debt)).toBe('#cf5757');
    expect(historyPalette(debt)[0]).toBe('#3b946c');
    expect(historyPalette(debt)[4]).toBe('#cf5757');
  });
  it('shows stronger output as green and weaker output as red', () => {
    expect(historyColor(10000, [10000, 70000], output)).toBe('#cf5757');
    expect(historyColor(70000, [10000, 70000], output)).toBe('#3b946c');
  });
  it('preserves the direction across negative and positive fiscal balances', () => {
    expect(historyColor(-12, [-12, 4], output)).toBe('#cf5757');
    expect(historyColor(4, [-12, 4], output)).toBe('#3b946c');
  });
  it('uses yellow for the middle and for an undifferentiated panel', () => {
    expect(historyColor(50, [0, 100], debt)).toBe('#e5ca61');
    expect(historyColor(50, [50, 50], output)).toBe('#e5ca61');
    expect(historyColor(50, [50, 50], debt)).toBe('#e5ca61');
  });
  it('uses a blue quantity scale for unscored or unspecified measures', () => {
    expect(historyColor(100, [0, 100], { ...output, scored: false })).toBe('#245782');
    expect(historyColor(100, [0, 100], undefined)).toBe('#245782');
  });
  it('keeps missing data grey while preserving a genuine zero', () => {
    expect(historyColor(null, [0, 100], debt)).toBe('#e2e5de');
    expect(historyColor(NaN, [0, 100], debt)).toBe('#e2e5de');
    expect(historyColor(0, [0, 100], debt)).toBe('#3b946c');
  });
});

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
