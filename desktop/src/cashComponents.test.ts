import { describe, expect, it } from 'vitest';
import { reconcileCashComponents } from './cashComponents';

describe('same-period cash component arithmetic', () => {
  it('retains distinct provider cash and aggregate investing definitions', () => {
    const r = reconcileCashComponents({ operating: 100, investing: -80, financing: -10, netCash: 12, providerFcf: 40 });
    expect(r).toEqual({ operatingPlusInvesting: 20, providerDifference: 20, componentSum: 10, netDifference: 2, providerComparison: 'differs', netComparison: 'differs' });
  });
  it('does not infer missing investment or net-cash values from other fields', () => {
    const r = reconcileCashComponents({ operating: 100, investing: null, financing: 0, netCash: 12, providerFcf: 40 });
    expect(r.operatingPlusInvesting).toBeNull();
    expect(r.providerDifference).toBeNull();
    expect(r.providerComparison).toBe('unavailable');
    expect(r.componentSum).toBeNull();
  });
  it('retains reported zero and allows stated floating-point tolerance', () => {
    const r = reconcileCashComponents({ operating: 1, investing: -1, financing: 0, netCash: 0, providerFcf: 1e-7 });
    expect(r.operatingPlusInvesting).toBe(0);
    expect(r.providerComparison).toBe('matches');
    expect(r.netComparison).toBe('matches');
  });
  it('withholds derived amounts outside the supported numerical range', () => {
    const r = reconcileCashComponents({ operating: 1e12, investing: 1e12, financing: 0, netCash: 1e12, providerFcf: 1e12 });
    expect(r.operatingPlusInvesting).toBeNull();
    expect(r.netComparison).toBe('unavailable');
  });
});
