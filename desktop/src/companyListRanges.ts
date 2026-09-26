import type { CompanyKpiUnit, CompanyListCell } from './companyListModel';
import { matchesNumericCondition } from './companyListFeatures';

export type CompanyListRange = { min: string; max: string; currency?: string };
export type CompanyListRangeCell = Pick<CompanyListCell, 'value' | 'unit' | 'currency' | 'status'>;

/** Keep editable text, including invalid/partial entries, distinct from blank bounds. */
export function parseCompanyListRange(input: unknown): CompanyListRange | undefined {
  if (input === undefined || input === null) return undefined;
  if (typeof input !== 'object' || Array.isArray(input)) return { min: 'Invalid range', max: '' };
  const value = input as Record<string, unknown>;
  const bound = (raw: unknown): string => raw === undefined ? '' : typeof raw === 'string' ? raw : 'Invalid bound';
  return {
    min: bound(value.min), max: bound(value.max),
    ...(value.currency === undefined ? {} : { currency: typeof value.currency === 'string' ? value.currency : 'Invalid currency' }),
  };
}

/** Currency alone does not restrict a column whose numeric bounds are both blank. */
export function companyListRangeActive(range: CompanyListRange | undefined | null): boolean {
  return !!range && (typeof range.min !== 'string' || typeof range.max !== 'string' || range.min.trim() !== '' || range.max.trim() !== '');
}

/** Comma and point each mean a decimal separator, never a thousands separator. */
export function companyListRangeNumber(text: string): number | null {
  if (typeof text !== 'string') return null;
  const value = text.trim();
  if (!/^[+-]?(?:\d+(?:[.,]\d+)?|[.,]\d+)$/.test(value)) return null;
  const result = Number(value.replace(',', '.'));
  // Do not round a nonzero bound outside the representable range down to zero.
  return Number.isFinite(result) && (result !== 0 || !/[1-9]/.test(value)) ? result : null;
}

export function companyListRangeError(range: CompanyListRange | undefined | null, unit: CompanyKpiUnit): string | null {
  if (!companyListRangeActive(range)) return null;
  if (unit === 'text' || unit === 'date') return 'Min and max apply only to numeric KPIs.';
  if (typeof range!.min !== 'string' || typeof range!.max !== 'string') return 'Enter a number or leave the bound blank.';
  const minText = range!.min.trim(), maxText = range!.max.trim();
  const min = companyListRangeNumber(minText), max = companyListRangeNumber(maxText);
  if (minText && min === null) return 'Enter a valid minimum, using a decimal point or comma without grouping separators.';
  if (maxText && max === null) return 'Enter a valid maximum, using a decimal point or comma without grouping separators.';
  if (min !== null && max !== null && min > max) return 'Minimum must be less than or equal to maximum.';
  if ((unit === 'money' || unit === 'price') && (typeof range!.currency !== 'string' || !/^[A-Z]{3}$/.test(range!.currency))) return 'Choose a currency for this monetary range.';
  return null;
}

/** Inclusive column bounds compose with the existing numeric rules, without consuming their limit. */
export function matchesCompanyListRange(cell: CompanyListRangeCell, range: CompanyListRange | undefined | null): boolean {
  if (!companyListRangeActive(range)) return true;
  if (companyListRangeError(range, cell.unit)) return false;
  const min = companyListRangeNumber(range!.min), max = companyListRangeNumber(range!.max);
  return min !== null && max !== null
    ? matchesNumericCondition(cell, { operator: 'between', value: min, valueTo: max, currency: range!.currency })
    : min !== null
      ? matchesNumericCondition(cell, { operator: 'gte', value: min, currency: range!.currency })
      : matchesNumericCondition(cell, { operator: 'lte', value: max, currency: range!.currency });
}
