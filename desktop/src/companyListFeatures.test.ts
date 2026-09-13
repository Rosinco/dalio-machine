import { describe, expect, it } from 'vitest';
import type { CompanyListCell, CompanyListColumn } from './companyListModel';
import { activeCompanyWatchlist, companyListCsv, companyListCsvField, defaultCompanyListFeatures, matchesNumericCondition, parseCompanyListFeatures, parseCompanySorts, sortCompanyListRows, toggleCompanyComparison, updateCompanyWatchlist } from './companyListFeatures';

const column = (id: string): CompanyListColumn => ({ id, kpiId: 'fcf', window: 'latest', calculation: 'latest' });
const cell = (value: number | null, currency: string | null = null): CompanyListCell => ({ value, display: value === null ? '—' : String(value), detail: 'Source: annual report. Signed value.', unit: currency ? 'money' : 'percent', currency, date: '2026-08-10' });

describe('named company lists and comparison preferences', () => {
  it('migrates default membership and preserves IDs absent from the active pack', () => {
    const migrated = parseCompanyListFeatures(undefined, ['1', '99999999999999999999', '1']);
    expect(migrated.watchlists).toEqual([{ id: 'default', name: 'Watchlist', listingIds: ['1', '99999999999999999999'] }]);
    const edited = updateCompanyWatchlist(migrated, 'default', '1', false);
    expect(parseCompanyListFeatures(JSON.parse(JSON.stringify(edited)), ['1', '7']).watchlists[0].listingIds).toEqual(['99999999999999999999']);
    expect(migrated.watchlists[0].listingIds).toContain('1');
  });

  it('keeps memberships separate and chooses the active named list', () => {
    const features = defaultCompanyListFeatures(['1']);
    features.watchlists.push({ id: 'ideas', name: 'Ideas', listingIds: ['2'] }); features.activeWatchlistId = 'ideas';
    const changed = updateCompanyWatchlist(features, 'ideas', '3', true);
    expect(activeCompanyWatchlist(changed).listingIds).toEqual(['2', '3']);
    expect(changed.watchlists[0].listingIds).toEqual(['1']); expect(features.watchlists[1].listingIds).toEqual(['2']);
    expect(updateCompanyWatchlist(changed, 'ideas', '3', true).watchlists[1].listingIds).toEqual(['2', '3']);
    expect(updateCompanyWatchlist(changed, 'ideas', '<invalid>', true)).toBe(changed);
  });

  it('repairs invalid references, bounds lists and comparisons, and retains view-specific sort references', () => {
    const parsed = parseCompanyListFeatures({ version: 1, watchlists: [{ id: 'ideas', name: '  Ideas\n ', listingIds: ['1', '1', '-2', '3'] }, { id: 'ideas', name: 'Duplicate', listingIds: ['4'] }], activeWatchlistId: 'missing', comparisonIds: Array.from({ length: 20 }, (_, i) => String(i)), secondarySorts: [{ columnId: 'obsolete', direction: 'desc' }, { columnId: 'revenue', direction: 'desc' }], density: 'compact', viewFeatures: { history: { secondarySorts: [{ columnId: 'history-only', direction: 'desc' }], density: 'compact', activeWatchlistId: 'ideas' } } }, ['42'], [column('revenue')]);
    expect(parsed.watchlists).toEqual([{ id: 'default', name: 'Watchlist', listingIds: ['42'] }, { id: 'ideas', name: 'Ideas', listingIds: ['1', '3'] }]);
    expect(parsed.activeWatchlistId).toBe('default'); expect(parsed.comparisonIds).toHaveLength(8);
    expect(parsed.secondarySorts).toEqual([{ columnId: 'revenue', direction: 'desc' }]);
    expect(parsed.viewFeatures.history.secondarySorts[0].columnId).toBe('history-only');
    const many = parseCompanyListFeatures({ version: 1, watchlists: Array.from({ length: 80 }, (_, i) => ({ id: `list-${i}`, name: `List ${i}`, listingIds: [] })) });
    expect(many.watchlists).toHaveLength(20); expect(many.watchlists[0].id).toBe('default');
  });

  it('round-trips all feature preferences without mutating the input', () => {
    const features = defaultCompanyListFeatures(['1']); features.secondarySorts = [{ columnId: 'cash', direction: 'desc' }]; features.density = 'compact';
    features.viewFeatures['saved-1'] = { secondarySorts: [{ columnId: 'cash', direction: 'asc' }], density: 'comfortable', activeWatchlistId: 'default' };
    const snapshot = structuredClone(features);
    expect(parseCompanyListFeatures(JSON.parse(JSON.stringify(features)), ['9'], [column('cash')])).toEqual(features);
    expect(features).toEqual(snapshot);
  });

  it('bounds comparison selection without replacing an existing company and supports removal', () => {
    let features = defaultCompanyListFeatures();
    for (let id = 1; id <= 8; id++) features = toggleCompanyComparison(features, String(id));
    expect(toggleCompanyComparison(features, '9')).toBe(features);
    expect(toggleCompanyComparison(features, 'bad')).toBe(features);
    expect(toggleCompanyComparison(features, '3').comparisonIds).toEqual(['1', '2', '4', '5', '6', '7', '8']);
    expect(features.comparisonIds).toContain('3');
  });
});

describe('numeric list conditions', () => {
  it('handles strict and inclusive bounds, equality, signs and zero', () => {
    expect(matchesNumericCondition(cell(0), { operator: 'gte', value: 0 })).toBe(true);
    expect(matchesNumericCondition(cell(0), { operator: 'gt', value: 0 })).toBe(false);
    expect(matchesNumericCondition(cell(-1), { operator: 'lt', value: 0 })).toBe(true);
    expect(matchesNumericCondition(cell(-1), { operator: 'lte', value: -1 })).toBe(true);
    expect(matchesNumericCondition(cell(-1), { operator: 'eq', value: -1 })).toBe(true);
    expect(matchesNumericCondition(cell(-1), { operator: 'between', value: -1, valueTo: 0 })).toBe(true);
  });

  it('does not interpret blanks or reversed ranges as valid thresholds', () => {
    for (const value of [null, NaN, Infinity]) expect(matchesNumericCondition(cell(20), { operator: 'gte', value })).toBe(false);
    for (const valueTo of [undefined, null, NaN, 9]) expect(matchesNumericCondition(cell(20), { operator: 'between', value: 10, valueTo })).toBe(false);
    expect(matchesNumericCondition(cell(null), { operator: 'lte', value: 0 })).toBe(false);
  });

  it('requires explicit currency for amount and price thresholds, including upper bounds', () => {
    for (const unit of ['money', 'price'] as const) {
      const money = { ...cell(20, 'SEK'), unit };
      expect(matchesNumericCondition(money, { operator: 'between', value: 10, valueTo: 30 })).toBe(false);
      expect(matchesNumericCondition(money, { operator: 'between', value: 10, valueTo: 30, currency: 'USD' })).toBe(false);
      expect(matchesNumericCondition(money, { operator: 'between', value: 10, valueTo: 30, currency: 'SEK' })).toBe(true);
    }
  });

  it('distinguishes an observed zero, absent source data, and a pending or failed request', () => {
    expect(matchesNumericCondition(cell(0), { operator: 'present', value: null })).toBe(true);
    expect(matchesNumericCondition(cell(null), { operator: 'missing', value: null })).toBe(true);
    expect(matchesNumericCondition(cell(null), { operator: 'present', value: null })).toBe(false);
    for (const status of ['loading', 'error'] as const) for (const operator of ['present', 'missing', 'gte'] as const) expect(matchesNumericCondition({ ...cell(null), status }, { operator, value: 0 })).toBe(false);
    expect(matchesNumericCondition({ ...cell(5), status: 'missing' }, { operator: 'present', value: null })).toBe(false);
  });
});

describe('multiple sort keys and CSV exports', () => {
  it('uses secondary and tertiary keys, deterministic ties, and reads each sort cell only once', () => {
    const rows = [{ id: '2', name: 'Second', values: [10, 2, 1] }, { id: '1', name: 'First', values: [10, 3, 2] }, { id: '3', name: 'Third', values: [10, 3, 1] }], columns = ['a', 'b', 'c'].map(column);
    let calls = 0;
    const sorted = sortCompanyListRows(rows, columns, [{ columnId: 'a', direction: 'asc' }, { columnId: 'b', direction: 'desc' }, { columnId: 'c', direction: 'asc' }], (row, col) => { calls++; return cell(row.values[columns.findIndex(c => c.id === col.id)]); });
    expect(sorted.map(row => row.id)).toEqual(['3', '1', '2']); expect(calls).toBe(9); expect(rows[0].id).toBe('2');
    expect(parseCompanySorts([{ columnId: 'a', direction: 'asc' }, { columnId: 'a', direction: 'desc' }, { columnId: 'deleted', direction: 'desc' }], columns)).toEqual([{ columnId: 'a', direction: 'asc' }]);
  });

  it('keeps missing last and currency groups stable in both sort directions', () => {
    const rows = [{ id: '1', name: 'Missing', value: null, currency: 'SEK' }, { id: '2', name: 'USD', value: 10, currency: 'USD' }, { id: '3', name: 'SEK', value: 20, currency: 'SEK' }];
    for (const direction of ['asc', 'desc'] as const) expect(sortCompanyListRows(rows, [column('a')], [{ columnId: 'a', direction }], row => cell(row.value, row.currency)).map(row => row.id)).toEqual(['3', '2', '1']);
  });

  it('escapes spreadsheet formulas in text and leaves actual signed numbers numeric', () => {
    for (const text of ['=SUM(A1:A2)', '+evil', '-company', '@lookup', '  =formula', '\t=bad', '\ntext']) expect(companyListCsvField(text)).toContain(`"'${text}`);
    expect(companyListCsvField(-12.5)).toBe('-12.5'); expect(companyListCsvField(0)).toBe('0');
    expect(companyListCsvField('A, "B"\nC')).toBe('"A, ""B""\nC"');
    expect(companyListCsvField(null)).toBe('');
  });

  it('exports all supplied rows with dates, units and source selection headers, including explicit unavailable states', () => {
    const rows = Array.from({ length: 75 }, (_, i) => ({ id: String(i), name: i === 0 ? '=Malicious' : `Company ${i}`, ticker: 'EX', isin: 'SE0001', country: 'SE', sourceAsOf: '2026-08-10' }));
    const output = companyListCsv(rows, [column('fcf')], row => row.id === '1' ? { ...cell(null, 'SEK'), detail: 'Request failed', status: 'error' } : cell(-12.5, 'SEK'), () => 'FCF', () => 'Atlas snapshot 2026-08-10');
    expect(output.startsWith('\uFEFF')).toBe(true); expect(output.split('\r\n')).toHaveLength(77);
    expect(output).toContain('"FCF [fcf | latest | latest | Atlas snapshot 2026-08-10] · value"');
    expect(output).toContain('"FCF [fcf | latest | latest | Atlas snapshot 2026-08-10] · status"');
    expect(output).toContain('"\'=Malicious"'); expect(output).toContain(',-12.5,"money","SEK","2026-08-10","available"');
    expect(output).toContain(',,"money","SEK","2026-08-10","error"');
    expect(output).not.toContain('Source: annual report');
    expect(output).toContain('"Company 74"');
  });

  it('keeps a full 19,140-listing, 32-column numeric export below the native delivery limit', () => {
    const rows = Array.from({ length: 19_140 }, (_, i) => ({ id: String(i + 1), name: `Company ${i + 1}`, ticker: 'EX', country: 'SE', sourceAsOf: '2026-08-10' }));
    const columns = Array.from({ length: 32 }, (_, i) => column(`metric-${i}`));
    const observation = { ...cell(-123456789.12345, 'SEK'), detail: 'Long source explanation. '.repeat(50) };
    const output = companyListCsv(rows, columns, () => observation, () => 'Provider FCF', () => 'Provider snapshot 2026-08-10');
    expect(new TextEncoder().encode(output).byteLength).toBeLessThan(64 * 1024 * 1024);
    expect(output).not.toContain('Long source explanation');
    expect(output.split('\r\n')).toHaveLength(19_142);
  });
});
