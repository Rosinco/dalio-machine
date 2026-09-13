import savedCalibration from './data/cash-uncertainty-2026-09-12.json';

export type CashDispersionGroup = 'low' | 'medium' | 'high';
export type CashMidModel = 'naive' | 'linear';
type FactorRecord = { factor: number | null; count: number; listings: number; histories: number; weight: number; supported: boolean };
export type CashCalibration = {
  id: string; targetCoverage: number; latestEmpiricalHorizon: number; historyYears: number;
  taxonomySha256: string; calibrationCutoff: string; calibrationLastTargetYear: number;
  minimumCalibrationRows: number; minimumCalibrationListings: number; minimumCalibrationHistories: number;
  calibrationByModelHorizon: Record<CashMidModel, Record<string, { global: FactorRecord; cashDispersion: Record<CashDispersionGroup, FactorRecord> }>>;
  sourceFiles: { id: string; as_of: string; path: string; sha256: string; frequency: string }[];
  provenance: { sourcePack: { sha256: string; path: string; asOf: string } };
};
export const cashCalibration = savedCalibration as unknown as CashCalibration;
export const cashCalibrationId = cashCalibration.id;
export type HistoricalCashFactor = { factor: number; group: CashDispersionGroup | 'global'; support: { count: number; listings: number; histories: number } };

export function cashDispersionGroup(value: number): CashDispersionGroup | null {
  return !Number.isFinite(value) || value < 0 ? null : value < 0.25 ? 'low' : value < 0.75 ? 'medium' : 'high';
}

export function classifyCashHistory(cash: (number | null)[]): { scale: number | null; dispersion: number | null; group: CashDispersionGroup | null } {
  if (cash.length !== 5 || cash.some(value => typeof value !== 'number' || !Number.isFinite(value))) return { scale: null, dispersion: null, group: null };
  const values = cash as number[], scale = values.reduce((sum, value) => sum + Math.abs(value), 0) / 5;
  if (!Number.isFinite(scale) || scale <= 0) return { scale: scale === 0 ? 0 : null, dispersion: null, group: null };
  const average = values.reduce((sum, value) => sum + value, 0) / 5;
  const dispersion = Math.sqrt(values.reduce((sum, value) => sum + (value - average) ** 2, 0) / 5) / scale;
  return { scale, dispersion: Number.isFinite(dispersion) ? dispersion : null, group: cashDispersionGroup(dispersion) };
}

export function historicalCashFactor(model: CashMidModel, year: number, group: CashDispersionGroup, calibration: CashCalibration = cashCalibration): HistoricalCashFactor | null {
  if (!Number.isInteger(year) || year < 1 || year > calibration.latestEmpiricalHorizon) return null;
  const cell = calibration.calibrationByModelHorizon[model]?.[String(year)];
  if (!cell) return null;
  const supported = (record: FactorRecord | undefined): record is FactorRecord & { factor: number } => !!record && record.supported
    && typeof record.factor === 'number' && Number.isFinite(record.factor) && record.factor >= 0
    && record.count >= calibration.minimumCalibrationRows && record.listings >= calibration.minimumCalibrationListings && record.histories >= calibration.minimumCalibrationHistories;
  const selected = supported(cell.cashDispersion[group]) ? { record: cell.cashDispersion[group], group } : supported(cell.global) ? { record: cell.global, group: 'global' as const } : null;
  return selected ? { factor: selected.record.factor!, group: selected.group, support: { count: selected.record.count, listings: selected.record.listings, histories: selected.record.histories } } : null;
}
