import type { CompanyListCell, CompanyListColumn, CompanyListSort } from './companyListModel';

export const MAX_COMPANY_WATCHLISTS = 20;
export const MAX_COMPANY_COMPARISONS = 8;
export const MAX_COMPANY_SORTS = 3;
const MAX_WATCHLIST_MEMBERS = 25_000;
export const COMPANY_LIST_FEATURES_STORAGE_KEY = 'macro-atlas-company-list-features-v1';

export type CompanyListNumericOperator = 'gte' | 'lte' | 'gt' | 'lt' | 'eq' | 'between' | 'present' | 'missing';
export type CompanyListNumericCondition = { operator: CompanyListNumericOperator; value: number | null; valueTo?: number | null; currency?: string };
export type CompanyListWatchlist = { id: string; name: string; listingIds: string[] };
export type CompanyListViewFeatures = { secondarySorts: CompanyListSort[]; density: 'compact' | 'comfortable'; activeWatchlistId: string };
export type CompanyListFeatures = CompanyListViewFeatures & { version: 1; watchlists: CompanyListWatchlist[]; comparisonIds: string[]; viewFeatures: Record<string, CompanyListViewFeatures> };
type Identity = { id: string; name: string; ticker?: string | null; isin?: string | null; country?: string | null; sourceAsOf?: string };
type StatusCell = Pick<CompanyListCell, 'value' | 'unit' | 'currency'> & { status?: 'loading' | 'error' | 'available' | 'missing' };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};
const validId = (value: unknown): value is string => typeof value === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(value);
const listingId = (value: unknown): value is string => typeof value === 'string' && /^[0-9]{1,20}$/.test(value);
const cleanName = (value: unknown): string => typeof value === 'string' ? value.replace(/[\u0000-\u001f\u007f]/g, '').trim().slice(0, 80) : '';
const ids = (value: unknown, limit = MAX_WATCHLIST_MEMBERS): string[] => Array.isArray(value) ? [...new Set(value.slice(0, limit * 2).filter(listingId))].slice(0, limit) : [];
const collator = new Intl.Collator('en', { sensitivity: 'base', numeric: true });

export function defaultCompanyListFeatures(legacyWatchlistIds: readonly string[] = []): CompanyListFeatures {
  return { version: 1, watchlists: [{ id: 'default', name: 'Watchlist', listingIds: ids(legacyWatchlistIds) }], activeWatchlistId: 'default', comparisonIds: [], secondarySorts: [], density: 'comfortable', viewFeatures: {} };
}

/** Unknown instrument IDs survive pack changes. A saved default list is authoritative after migration. */
export function parseCompanyListFeatures(input: unknown, legacyWatchlistIds: readonly string[] = [], columns: readonly Pick<CompanyListColumn, 'id'>[] = []): CompanyListFeatures {
  const raw = object(input), fallback = defaultCompanyListFeatures(legacyWatchlistIds);
  if (raw.version !== 1) return fallback;
  const watchlists: CompanyListWatchlist[] = [], seen = new Set<string>();
  if (Array.isArray(raw.watchlists)) for (const candidate of raw.watchlists.slice(0, MAX_COMPANY_WATCHLISTS * 2)) {
    const list = object(candidate), name = cleanName(list.name);
    if (!validId(list.id) || !name || seen.has(list.id)) continue;
    seen.add(list.id); watchlists.push({ id: list.id, name, listingIds: ids(list.listingIds) });
    if (watchlists.length === MAX_COMPANY_WATCHLISTS) break;
  }
  if (!seen.has('default')) watchlists.unshift(fallback.watchlists[0]);
  watchlists.splice(MAX_COMPANY_WATCHLISTS);
  const available = new Set(watchlists.map(list => list.id));
  const parseView = (value: unknown, selectedColumns?: readonly Pick<CompanyListColumn, 'id'>[]): CompanyListViewFeatures => {
    const view = object(value);
    return { secondarySorts: parseCompanySorts(view.secondarySorts, selectedColumns, MAX_COMPANY_SORTS - 1), density: view.density === 'compact' ? 'compact' : 'comfortable', activeWatchlistId: typeof view.activeWatchlistId === 'string' && available.has(view.activeWatchlistId) ? view.activeWatchlistId : 'default' };
  };
  const viewFeatures: Record<string, CompanyListViewFeatures> = {};
  for (const [key, value] of Object.entries(object(raw.viewFeatures)).slice(0, 20)) if (validId(key) && key !== '__proto__' && key !== 'constructor' && key !== 'prototype') viewFeatures[key] = parseView(value);
  return { version: 1, watchlists, comparisonIds: ids(raw.comparisonIds, MAX_COMPANY_COMPARISONS), ...parseView(raw, columns), viewFeatures };
}

export function activeCompanyWatchlist(features: CompanyListFeatures): CompanyListWatchlist {
  return features.watchlists.find(list => list.id === features.activeWatchlistId) ?? features.watchlists.find(list => list.id === 'default') ?? { id: 'default', name: 'Watchlist', listingIds: [] };
}

/** Does not modify the source object or mirror named-list membership into the legacy default list. */
export function updateCompanyWatchlist(features: CompanyListFeatures, watchlistId: string, companyId: string, included: boolean): CompanyListFeatures {
  if (!listingId(companyId)) return features;
  return { ...features, watchlists: features.watchlists.map(list => list.id !== watchlistId ? list : { ...list, listingIds: included ? list.listingIds.includes(companyId) || list.listingIds.length >= MAX_WATCHLIST_MEMBERS ? list.listingIds : [...list.listingIds, companyId] : list.listingIds.filter(id => id !== companyId) }) };
}

export function toggleCompanyComparison(features: CompanyListFeatures, companyId: string): CompanyListFeatures {
  if (!listingId(companyId)) return features;
  if (features.comparisonIds.includes(companyId)) return { ...features, comparisonIds: features.comparisonIds.filter(id => id !== companyId) };
  return features.comparisonIds.length >= MAX_COMPANY_COMPARISONS ? features : { ...features, comparisonIds: [...features.comparisonIds, companyId] };
}

export function parseCompanySorts(input: unknown, columns?: readonly Pick<CompanyListColumn, 'id'>[], limit = MAX_COMPANY_SORTS): CompanyListSort[] {
  const sorts: CompanyListSort[] = [], seen = new Set<string>();
  if (Array.isArray(input)) for (const candidate of input.slice(0, 12)) {
    const sort = object(candidate);
    if (!validId(sort.columnId) || seen.has(sort.columnId) || columns && sort.columnId !== 'name' && !columns.some(column => column.id === sort.columnId)) continue;
    seen.add(sort.columnId); sorts.push({ columnId: sort.columnId, direction: sort.direction === 'desc' ? 'desc' : 'asc' });
    if (sorts.length === limit) break;
  }
  return sorts;
}

/** Presence is observed numeric data, never a failed or pending shard request. Bound filters require explicit matching currency for amounts. */
export function matchesNumericCondition(cell: StatusCell, rule: CompanyListNumericCondition): boolean {
  if (cell.status === 'loading' || cell.status === 'error') return false;
  const observed = cell.status !== 'missing' && finite(cell.value);
  if (rule.operator === 'present') return observed;
  if (rule.operator === 'missing') return !observed;
  if (!observed || !finite(cell.value) || !finite(rule.value)) return false;
  if ((cell.unit === 'money' || cell.unit === 'price') && (!rule.currency || !/^[A-Z]{3}$/.test(rule.currency) || cell.currency !== rule.currency)) return false;
  switch (rule.operator) {
    case 'gte': return cell.value >= rule.value;
    case 'lte': return cell.value <= rule.value;
    case 'gt': return cell.value > rule.value;
    case 'lt': return cell.value < rule.value;
    case 'eq': return cell.value === rule.value;
    case 'between': return finite(rule.valueTo) && rule.value <= rule.valueTo && cell.value >= rule.value && cell.value <= rule.valueTo;
    default: return false;
  }
}

function compareCells(a: CompanyListCell, b: CompanyListCell, direction: CompanyListSort['direction']): number {
  const missing = (cell: CompanyListCell) => cell.value === null || typeof cell.value === 'number' && !finite(cell.value) || (cell as StatusCell).status === 'loading' || (cell as StatusCell).status === 'error' || (cell as StatusCell).status === 'missing';
  const am = missing(a), bm = missing(b);
  if (am || bm) return am === bm ? 0 : am ? 1 : -1;
  const currency = collator.compare(a.currency ?? '', b.currency ?? '');
  if (currency) return currency;
  const value = typeof a.value === 'number' && typeof b.value === 'number' ? a.value - b.value : collator.compare(String(a.value), String(b.value));
  return direction === 'asc' ? value : -value;
}

/** Precompute each active sort cell once. Currency groups and missing-last semantics apply independently at each sort key. */
export function sortCompanyListRows<Row extends Identity>(rows: readonly Row[], columns: readonly CompanyListColumn[], requestedSorts: readonly CompanyListSort[], getCell: (row: Row, column: CompanyListColumn) => CompanyListCell): Row[] {
  const sorts = parseCompanySorts(requestedSorts, columns), byId = new Map(columns.map(column => [column.id, column]));
  const keyed = rows.map(row => ({ row, cells: sorts.map(sort => sort.columnId === 'name' ? null : getCell(row, byId.get(sort.columnId)!)) }));
  keyed.sort((a, b) => {
    for (let index = 0; index < sorts.length; index++) {
      const sort = sorts[index], order = sort.columnId === 'name' ? collator.compare(a.row.name, b.row.name) * (sort.direction === 'asc' ? 1 : -1) : compareCells(a.cells[index]!, b.cells[index]!, sort.direction);
      if (order) return order;
    }
    return collator.compare(a.row.name, b.row.name) || collator.compare(a.row.id, b.row.id);
  });
  return keyed.map(item => item.row);
}

/** Quote text as text, including spreadsheet formula prefixes; actual finite negative numbers remain numbers. */
export function companyListCsvField(value: unknown): string {
  if (finite(value)) return String(value);
  if (value === null || value === undefined) return '';
  const raw = String(value), protectedText = /^[\s\u0000-\u0020]*[=+\-@]/.test(raw) || /^[\t\r\n]/.test(raw) ? `'${raw}` : raw;
  return `"${protectedText.replace(/"/g, '""')}"`;
}

/** Export every supplied row. Selection/source metadata appears once per header; observations keep their own dates and units. */
export function companyListCsv<Row extends Identity>(rows: readonly Row[], columns: readonly CompanyListColumn[], getCell: (row: Row, column: CompanyListColumn) => CompanyListCell, labelColumn: (column: CompanyListColumn) => string = column => column.kpiId, contextColumn: (column: CompanyListColumn) => string = () => ''): string {
  const headers = ['Listing ID', 'Company', 'Ticker', 'ISIN', 'Listing country', 'Directory saved', ...columns.flatMap(column => {
    const context = contextColumn(column), prefix = `${labelColumn(column)} [${column.kpiId} | ${column.window} | ${column.calculation}${context ? ` | ${context}` : ''}]`;
    return ['value', 'unit', 'currency', 'date', 'status'].map(kind => `${prefix} · ${kind}`);
  })];
  const result = [headers.map(companyListCsvField).join(',')];
  for (const row of rows) {
    const values: unknown[] = [row.id, row.name, row.ticker, row.isin, row.country, row.sourceAsOf];
    for (const column of columns) {
      const cell = getCell(row, column), status = (cell as StatusCell).status ?? (cell.value === null ? 'missing' : 'available');
      values.push(status === 'available' ? cell.value : null, cell.unit, cell.currency, cell.date, status);
    }
    result.push(values.map(companyListCsvField).join(','));
  }
  return `\uFEFF${result.join('\r\n')}\r\n`;
}
