import { describe, expect, it } from 'vitest';
import { countryMoney, datedHistory, type LiquidityReport } from './liquidity';

describe('liquidity geography and history', () => {
  const report = { broad_money: [{ country: 'SE' }, { country: 'EU' }, { country: 'US' }] } as LiquidityReport;
  it('uses national readings and labels currency-area context explicitly', () => {
    expect(countryMoney(report, 'SE', 'SEK').reading?.country).toBe('SE');
    expect(countryMoney(report, 'DE', 'EUR')).toEqual({ reading: report.broad_money[1], scope: 'currency-area' });
    expect(countryMoney(report, 'CA', 'CAD').reading).toBeUndefined();
  });
  it('leaves missing monthly and quarterly periods blank, preserving zero', () => {
    const points = [
      { date: '2026-01-01', period: '2026-01', annual_log_growth_pct: 0 },
      { date: '2026-07-01', period: '2026-07', annual_log_growth_pct: 3 },
    ];
    expect(datedHistory(points, 'monthly').values).toEqual([0, null, null, null, null, null, 3]);
    expect(datedHistory(points, 'quarterly')).toEqual({ periods: ['2026-01', '2026-04', '2026-07'], values: [0, null, 3] });
    expect(datedHistory([], 'monthly').values).toEqual([]);
  });
});
