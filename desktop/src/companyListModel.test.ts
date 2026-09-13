import { describe, expect, it } from 'vitest';
import { companyListCell, compareCompanyListRows, defaultCompanyListPreferences, matchesCompanyListFilters, parseCompanyListPreferences, type CompanyListColumn } from './companyListModel';
import { researchGaugeSeries, type ResearchGaugeRow } from './researchGaugeModel';

const column = (kpiId: string, window: CompanyListColumn['window'] = 'latest', calculation: CompanyListColumn['calculation'] = 'latest'): CompanyListColumn => ({ id: 'test', kpiId, window, calculation });
function row(): ResearchGaugeRow {
  const periods = [2025, 2024, 2023, 2022, 2021].map(year => ({ year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, published: `${year + 1}-02-01`, currency: 'SEK', sourceId: 'annual', sourceAsOf: '2026-08-10' }));
  return { id: '1', name: 'Example', ticker: 'EX', isin: null, country: 'SE', sectorId: '3', sectorName: 'Industrials', branchId: '20', branchName: 'Tools', sourceAsOf: '2026-08-10', presence: 'latest', sourceCompanySha256: 'a'.repeat(64), route: 'operating', classificationConflict: false, readiness: 'history_available',
    annual: { currency: 'SEK', latest: periods[0], periods, excluded: { outsideWindow: 0, placeholder: 0, missingPublication: 0, withheld: 0 }, cash: researchGaugeSeries([40, 20, 10, 5, 2]), operatingCash: researchGaugeSeries([50, 30, 20, 10, 5]), ebit: researchGaugeSeries([80, 60, 40, 20, 10]), revenue: researchGaugeSeries([200, 150, 100, 75, 50]), margins: researchGaugeSeries([40, 40, 40, 26.67, 20]), latestValues: { revenues: 200, ebit: 80, cash: 40, operatingCash: 50, financingCash: -5, cashBalance: 10, netDebt: -20, equity: 100, assets: 200, netDebtToAssetsPercent: -10, equityToAssetsPercent: 50, tangibleAssetsToRevenue: 1.2, intangibleAssetsToAssetsPercent: 15, cashComponentDifference: 0 }, reason: null },
    quarter: { latest: null, comparison: null, revenueChangePercent: null, ebitMarginChangePoints: null, cashChange: null, reason: 'No quarter comparison.' },
    valuation: { status: 'positive-priced', currency: 'SEK', value: 1000, cashPV: 400, terminalPV: 600, terminalShare: .6, candidateEquity: 700, priceDate: '2026-02-15', priceAgeDays: 210, priceBasis: { sourceId: 'market', sourceAsOf: '2026-08-10', reportDate: '2026-02-01', reportEnd: '2025-12-31', shares: 7, close: 100, currency: 'SEK', method: 'local', fxRate: null, fxDate: null }, ceiling: 700, lowValue: 650, lowNPV: -50, reverseCashFactor: .7, reverseCashFactor30: 1, hasSignedCash: false, annualHistoryCount: 5, rangeStatus: 'historical', reason: null }, issues: [] };
}

describe('custom company list KPIs', () => {
  it('uses exactly the selected latest annual observations and preserves signs and zero', () => {
    const r = row(); r.annual.cash = researchGaugeSeries([0, -20, 50, 100, 200]);
    expect(companyListCell(r, column('fcf')).value).toBe(0);
    expect(companyListCell(r, column('fcf', '3', 'average')).value).toBe(10);
    expect(companyListCell(r, column('fcf', '3', 'median')).value).toBe(0);
    expect(companyListCell(r, column('fcf', '3', 'min')).value).toBe(-20);
    expect(companyListCell(r, column('fcf', '3', 'max')).value).toBe(50);
  });
  it('does not backfill a missing latest value or drop missing values from a requested aggregate', () => {
    const r = row(); r.annual.cash = researchGaugeSeries([null, 20, 10, 5, 2]);
    expect(companyListCell(r, column('fcf')).value).toBeNull();
    expect(companyListCell(r, column('fcf', '3', 'average')).detail).toMatch(/missing/i);
    r.annual.periods = r.annual.periods.slice(0, 2);
    expect(companyListCell(r, column('revenue', '3', 'average')).detail).toMatch(/3.*report/i);
  });
  it('rejects incompatible annual periods even if the value array is complete', () => {
    for (const mutate of [(r: ResearchGaugeRow) => { r.annual.periods[1].currency = 'USD'; }, (r: ResearchGaugeRow) => { r.annual.periods[1].year = 2022; }, (r: ResearchGaugeRow) => { r.annual.periods[1].end = '2025-01-05'; }]) {
      const r = row(); mutate(r); expect(companyListCell(r, column('fcf', '3', 'average')).value).toBeNull();
    }
  });
  it('calculates CAGR over the actual end-date interval and withholds nonpositive paths', () => {
    const r = row(), cell = companyListCell(r, column('fcf', '3', 'growth'));
    const years = (Date.parse('2025-12-31') - Date.parse('2023-12-31')) / (365.25 * 86400000);
    expect(cell.value).toBeCloseTo(100 * (4 ** (1 / years) - 1));
    expect(cell.unit).toBe('percent'); expect(cell.detail).toMatch(/CAGR/);
    for (const values of [[40, 0, 10], [40, -5, 10], [-40, -20, -10]]) { r.annual.cash = researchGaugeSeries(values); expect(companyListCell(r, column('fcf', '3', 'growth')).value).toBeNull(); }
  });
  it('derives FCF margin from same-period cash and strictly positive revenue', () => {
    const r = row(); expect(companyListCell(r, column('fcf_margin')).value).toBe(20);
    r.annual.cash.values[0] = -10; expect(companyListCell(r, column('fcf_margin')).value).toBe(-5);
    r.annual.revenue.values[0] = 0; expect(companyListCell(r, column('fcf_margin')).value).toBeNull();
  });
  it('keeps stock quote and whole-equity NPV in their own currencies and exposes source dates', () => {
    const r = row(); r.valuation.priceBasis!.currency = 'USD';
    expect(companyListCell(r, column('stock_close'))).toMatchObject({ value: 100, currency: 'USD', unit: 'price', date: '2026-02-15' });
    expect(companyListCell(r, column('mid_npv'))).toMatchObject({ value: 300, currency: 'SEK', unit: 'money' });
    expect(companyListCell(r, column('low_npv')).value).toBe(-50);
    expect(companyListCell(r, column('terminal_share')).value).toBe(60);
    r.valuation.candidateEquity = null; expect(companyListCell(r, column('mid_npv')).value).toBeNull();
  });
  it('does not invent unsupported history or EPS/PE, and leaves specialist valuations missing', () => {
    const r = row(); expect(companyListCell(r, column('net_debt', '3', 'average')).value).toBeNull();
    expect(companyListCell(r, column('pe')).value).toBeNull();
    r.valuation.status = 'manual-financial'; r.valuation.value = null; r.valuation.reason = 'Reviewed capital model required.';
    expect(companyListCell(r, column('mid_dcf')).detail).toMatch(/Reviewed capital/);
  });
  it('sorts numerically, keeps missing last both ways and groups currencies before amounts', () => {
    const a = row(), b = row(), missing = row(); a.id = '2'; a.annual.cash.values[0] = 2; b.id = '3'; b.annual.cash.values[0] = 100; missing.annual.cash.values[0] = null;
    const c = column('fcf');
    expect(compareCompanyListRows(a, b, c, 'asc')).toBeLessThan(0); expect(compareCompanyListRows(a, b, c, 'desc')).toBeGreaterThan(0);
    for (const d of ['asc', 'desc'] as const) { expect(compareCompanyListRows(missing, a, c, d)).toBeGreaterThan(0); }
    b.annual.currency = 'USD'; b.annual.periods.forEach(p => { p.currency = 'USD'; });
    expect(compareCompanyListRows(a, b, c, 'desc')).toBeLessThan(0);
    a.annual.cash.values[0] = null; a.name = 'A'; expect(compareCompanyListRows(a, missing, c, 'desc')).toBeLessThan(0);
  });
});

describe('company list preferences and explicit filters', () => {
  it('round-trips selected columns, filters, named views, and personal research list ids', () => {
    const state = defaultCompanyListPreferences(); state.watchlistIds = ['1', '2']; state.features.watchlists[0].listingIds = ['1', '2']; state.filters.query = 'Tools';
    state.savedViews = [{ id: 'view-1', name: 'Cash ideas', columns: [column('fcf', '3', 'median')], filters: state.filters, sort: { columnId: 'test', direction: 'desc' } }];
    expect(parseCompanyListPreferences(JSON.stringify(state))).toEqual(state);
  });
  it('bounds malformed saved state and repairs references without accepting invented KPIs', () => {
    const state = defaultCompanyListPreferences();
    expect(parseCompanyListPreferences('{bad')).toEqual(state);
    expect(parseCompanyListPreferences({ ...state, version: 9 })).toEqual(state);
    const parsed = parseCompanyListPreferences({ ...state, version: 1, columns: [column('pe')], watchlistIds: ['1', '1', '2', '<bad>'], savedViews: [{ id: 'x', name: 'Bad', columns: [], filters: {}, sort: { columnId: 'lost', direction: 'desc' } }] });
    expect(parsed.columns).toEqual(state.columns); expect(parsed.watchlistIds).toEqual(['1', '2']); expect(parsed.savedViews[0].sort.columnId).toBe('name');
    const many = parseCompanyListPreferences({ ...state, columns: Array.from({ length: 40 }, (_, i) => ({ ...column('fcf'), id: `c${i}` })) });
    expect(many.columns).toHaveLength(32);
  });
  it('candidate presets require complete positive observations and current comparable evidence', () => {
    const r = row(), filters = { ...defaultCompanyListPreferences().filters, preset: 'cash_and_margin' as const };
    expect(matchesCompanyListFilters(r, filters, new Set())).toBe(true);
    r.valuation.candidateEquity = 701; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    r.valuation.candidateEquity = 700; r.annual.ebit = researchGaugeSeries([80, 60, null, 20, 10]); expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    r.annual.ebit = researchGaugeSeries([80, 60, 40, 20, 10]); r.classificationConflict = true; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    r.classificationConflict = false; r.readiness = 'limited_history'; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
  });
  it('requires explicitly matching currency for amount filters and never treats missing values as zero', () => {
    const r = row(), filters = defaultCompanyListPreferences().filters;
    filters.numericRules = [{ column: column('fcf'), operator: 'gte', value: 20 }]; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    const reopened = parseCompanyListPreferences(JSON.stringify({ ...defaultCompanyListPreferences(), filters }));
    expect(reopened.filters.numericRules).toEqual(filters.numericRules); expect(matchesCompanyListFilters(r, reopened.filters, new Set())).toBe(false);
    filters.numericRules[0].currency = 'SEK'; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(true);
    filters.numericRules[0].currency = 'USD'; expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    filters.numericRules = [{ column: column('fcf_margin'), operator: 'gte', value: 0 }]; r.annual.cash.values[0] = null;
    expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
  });
  it('retains incomplete numeric thresholds through restart without converting blanks to zero', () => {
    const r = row(), state = defaultCompanyListPreferences();
    state.filters.numericRules = [{ column: column('ebit_margin'), operator: 'gte', value: null }];
    const reopened = parseCompanyListPreferences(JSON.stringify(state));
    expect(reopened.filters.numericRules[0].value).toBeNull(); expect(matchesCompanyListFilters(r, reopened.filters, new Set())).toBe(false);
    const malformed = parseCompanyListPreferences({ ...state, filters: { ...state.filters, numericRules: [{ ...state.filters.numericRules[0], value: '' }] } });
    expect(malformed.filters.numericRules[0].value).toBeNull(); expect(matchesCompanyListFilters(r, malformed.filters, new Set())).toBe(false);
  });
  it('applies personal list membership and metadata search without changing the source row', () => {
    const r = row(), before = structuredClone(r), filters = defaultCompanyListPreferences().filters; filters.watchlistOnly = true;
    expect(matchesCompanyListFilters(r, filters, new Set())).toBe(false);
    expect(matchesCompanyListFilters(r, filters, new Set(['1']))).toBe(true);
    filters.query = 'ex'; expect(matchesCompanyListFilters(r, filters, new Set(['1']))).toBe(true);
    filters.branchId = '99'; expect(matchesCompanyListFilters(r, filters, new Set(['1']))).toBe(false); expect(r).toEqual(before);
  });
});
