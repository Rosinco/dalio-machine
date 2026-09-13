import { describe, expect, it } from 'vitest';
import { cashCalibration, cashDispersionGroup, classifyCashHistory, historicalCashFactor } from './cashUncertainty';

describe('source-bound historical cash uncertainty', () => {
  it('classifies signed population dispersion with exact fixed boundaries', () => {
    expect(cashDispersionGroup(0.249999)).toBe('low');
    expect(cashDispersionGroup(0.25)).toBe('medium');
    expect(cashDispersionGroup(0.75)).toBe('high');
    expect(classifyCashHistory([-10, -10, -10, -10, -10])).toEqual({ scale: 10, dispersion: 0, group: 'low' });
    const signed = classifyCashHistory([-10, -5, 0, 5, 10]);
    expect(signed.scale).toBe(6);
    expect(signed.dispersion).toBeCloseTo(Math.sqrt(50) / 6);
    expect(signed.group).toBe('high');
  });
  it('does not invent empirical groups for partial, missing or zero-scale histories', () => {
    for (const values of [[1, 2, 3], [0, 0, 0, 0, 0], [1, 2, null, 4, 5], [1, 2, Infinity, 4, 5]]) {
      expect(classifyCashHistory(values).group).toBeNull();
    }
    expect(cashDispersionGroup(NaN)).toBeNull();
    expect(cashDispersionGroup(-1)).toBeNull();
  });
  it('uses exported model and horizon factors without inventing year-five support', () => {
    const factor = historicalCashFactor('naive', 1, 'low');
    expect(factor?.factor).toBe(cashCalibration.calibrationByModelHorizon.naive['1'].cashDispersion.low.factor);
    expect(factor?.group).toBe('low');
    expect(factor?.support.listings).toBeGreaterThanOrEqual(100);
    expect(historicalCashFactor('linear', 4, 'high')).not.toBeNull();
    expect(historicalCashFactor('naive', 5, 'low')).toBeNull();
    expect(historicalCashFactor('naive', 0, 'low')).toBeNull();
  });
  it('falls back to the same model and horizon global factor when a group lacks independent support', () => {
    const calibration = structuredClone(cashCalibration);
    calibration.calibrationByModelHorizon.naive['1'].cashDispersion.low.histories = 99;
    const result = historicalCashFactor('naive', 1, 'low', calibration);
    expect(result?.group).toBe('global');
    expect(result?.factor).toBe(calibration.calibrationByModelHorizon.naive['1'].global.factor);
    calibration.calibrationByModelHorizon.naive['1'].global.factor = null;
    expect(historicalCashFactor('naive', 1, 'low', calibration)).toBeNull();
  });
});
