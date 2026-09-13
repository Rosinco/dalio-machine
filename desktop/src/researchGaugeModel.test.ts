import { describe, expect, it } from 'vitest';
import { buildResearchGaugeRow, researchGaugeSeries } from './researchGaugeModel';
import type { FinancialCompany, FinancialIndex, SourcedReport } from './financialData';
import type { CompanyEntry } from './listingCatalogue';
import type { Taxonomy } from './taxonomy';

const asOf = '2026-09-13';
const entry = { id: '200', display_name: 'Example', report_currency: 'EUR', stock_currency: 'EUR', source_as_of: '2026-08-10', sector_id: '3', branch_id: '20', ticker: 'EX', isin: null, listing_country: 'SE' } as CompanyEntry;
const index = { id: 'a'.repeat(64), as_of: '2026-08-10', taxonomy_sha256: 'b'.repeat(64), sources: [], companies: { '200': { sha256: 'c'.repeat(64) } } } as unknown as FinancialIndex;
const taxonomy = { catalogue: { as_of: '2026-08-10' }, sectors: { '3': { name_en: 'Industrials' }, '1': { name_en: 'Financials' } }, branches: { '20': { name_en: 'Tools' }, '75': { name_en: 'Property' } },
  classifications: { '200': { company_id: '200', sector_id: '3', branch_id: '20', status: 'source' } } } as unknown as Taxonomy;
function report(year: number, cash: number | null = 10, changes: Partial<SourcedReport> = {}): SourcedReport {
  const values = { free_cash_flow: cash, revenues: 100, operating_income: 20, cash_flow_from_operating_activities: 15, cash_flow_from_investing_activities: -4,
    cash_flow_from_financing_activities: -3, net_debt: -10, total_assets: 200, total_equity: 100, tangible_assets: 60, intangible_assets: 20, cash_and_equivalents: 30 };
  return { year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, report_date: `${year + 1}-02-15`, currency: 'EUR', currency_ratio: 2,
    source_id: 'annual-2026-08-10', source_as_of: '2026-08-10', values, raw: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, v === null ? null : 2 * v])), ...changes };
}
function history(): FinancialCompany { return { id: '200', annual: [2021, 2022, 2023, 2024, 2025].map(y => report(y)), quarterly: [], withheld: [], market: [] }; }
const build = (h = history(), t = taxonomy) => buildResearchGaugeRow(entry, h, index, t, asOf);

describe('offline research evidence model', () => {
  it('keeps signed cash, zero and missing observations distinct, including a missing latest value', () => {
    const s = researchGaugeSeries([null, -10, 0, 10, 20]);
    expect(s).toMatchObject({ count: 4, positive: 2, negative: 1, zero: 1, latest: null, median: 5 });
    expect(s.dispersion).toBeCloseTo(Math.sqrt(125) / 10);
    expect(researchGaugeSeries([0, 0]).dispersion).toBeNull();
    expect(researchGaugeSeries([5]).dispersion).toBeNull();
  });
  it('retains native signed financing/asset proxies without substituting liabilities or appraising book value', () => {
    const row = build();
    expect(row.readiness).toBe('history_available');
    expect(row.annual.periods.map(p => p.year)).toEqual([2025, 2024, 2023, 2022, 2021]);
    expect(row.annual.latestValues).toMatchObject({ netDebt: -10, netDebtToAssetsPercent: -5, equityToAssetsPercent: 50, tangibleAssetsToRevenue: .6, intangibleAssetsToAssetsPercent: 10, cashComponentDifference: -1 });
    expect(row.valuation.value).toBeGreaterThan(0);
    expect(row.valuation.status).toBe('positive-unpriced');
    expect(row.valuation.reverseCashFactor).toBeNull();
  });
  it('does not backfill a missing current cash cell from older profitable years', () => {
    const h = history(); h.annual[4] = report(2025, null);
    const row = build(h);
    expect(row.annual.cash).toMatchObject({ latest: null, count: 4, positive: 4 });
    expect(row.annual.ebit.count).toBe(5);
    expect(row.annual.latestValues.cash).toBeNull();
    expect(row.readiness).toBe('limited_history');
    expect(row.valuation.value).toBeNull();
  });
  it('does not project or silently substitute an older year for a latest source placeholder', () => {
    const h = history(); h.annual[4] = report(2025, 0, { values: { free_cash_flow: 0 }, raw: { free_cash_flow: 0, revenues: null } });
    const row = build(h);
    expect(row.annual.latest?.year).toBe(2025);
    expect(row.annual.periods).toEqual([]); expect(row.annual.latestValues.revenues).toBeNull();
    expect(row.annual.excluded.placeholder).toBe(1); expect(row.readiness).toBe('reconcile_data');
    expect(row.valuation.value).toBeNull(); expect(row.annual.reason).toMatch(/placeholder/);
  });
  it('stops at gaps, overlapping periods and changed currencies instead of joining incompatible years', () => {
    for (const changed of [report(2023), report(2024, 10, { currency: 'SEK' }), report(2024, 10, { end: '2025-01-05' })]) {
      const h = history(); h.annual[3] = changed;
      const row = build(h); expect(row.annual.periods).toHaveLength(1); expect(row.readiness).toBe('limited_history');
    }
    const h = history(); h.annual[4].currency_ratio = null;
    expect(build(h).annual.periods).toHaveLength(0);
  });
  it('retains historical observations with unknown publication dates but marks timing and readiness unresolved', () => {
    const h = history(); h.annual[4].report_date = null;
    const row = build(h); expect(row.annual.cash.count).toBe(5); expect(row.annual.latest?.published).toBeNull();
    expect(row.readiness).toBe('limited_history'); expect(row.issues.join(' ')).toMatch(/publication date/);
  });
  it('gives financial and property companies distinct routes without creating financial cash forecasts', () => {
    const t = structuredClone(taxonomy); t.classifications['200'].sector_id = '1';
    const bank = build(history(), t); expect(bank.route).toBe('financial'); expect(bank.valuation.status).toBe('manual-financial');
    expect(bank.valuation.value).toBeNull(); expect(bank.valuation.lowValue).toBeNull(); expect(bank.annual.cash.count).toBe(5);
    t.classifications['200'].branch_id = '75';
    const property = build(history(), t); expect(property.route).toBe('property'); expect(property.valuation.value).toBeGreaterThan(0);
    t.classifications['200'].status = 'needs_review'; expect(build(history(), t).readiness).toBe('reconcile_data');
  });
  it('computes only comparable standalone quarter YoY with positive revenue denominators', () => {
    const h = history();
    h.quarterly = [2025, 2026].map(y => report(y, y === 2026 ? -5 : 5, { period: 2, start: `${y}-04-01`, end: `${y}-06-30`, report_date: `${y}-07-15` }));
    h.quarterly[1].values.revenues = 120; h.quarterly[1].values.operating_income = 30;
    let row = build(h); expect(row.quarter).toMatchObject({ revenueChangePercent: 20, ebitMarginChangePoints: 5, cashChange: -10, reason: null });
    h.quarterly[0].values.revenues = 0;
    row = build(h); expect(row.quarter.revenueChangePercent).toBeNull(); expect(row.quarter.ebitMarginChangePoints).toBeNull(); expect(row.quarter.cashChange).toBe(-10);
    h.quarterly[1].start = '2026-01-01'; row = build(h);
    expect(row.quarter.cashChange).toBeNull(); expect(row.quarter.reason).toMatch(/standalone/);
    h.quarterly[1].start = '2026-04-01'; h.quarterly[0].start = '2024-04-01'; h.quarterly[0].end = '2024-06-30';
    expect(build(h).quarter.cashChange).toBeNull();
  });
  it('withholds stale substitution for newer rejected annual or quarterly reports', () => {
    const h = history(); h.withheld = [{ year: 2026, period: 5, source_id: 'annual-2026-08-10', start: '', end: '', published: '', reason: 'Invalid source dates' }];
    const row = build(h); expect(row.annual.periods).toEqual([]); expect(row.annual.excluded.withheld).toBe(1); expect(row.readiness).toBe('reconcile_data');
  });
  it('preserves negative modeled values and treats missing history as missing', () => {
    const h = history(); h.annual = h.annual.map(r => report(r.year, -10));
    const row = build(h); expect(row.valuation.value).toBeLessThan(0); expect(row.valuation.terminalPV).toBe(0); expect(row.valuation.ceiling).toBeNull(); expect(row.valuation.hasSignedCash).toBe(true);
    h.annual = []; const empty = build(h); expect(empty.readiness).toBe('no_history'); expect(empty.annual.cash.count).toBe(0); expect(empty.annual.cash.latest).toBeNull();
  });
  it('rejects an identity mismatch without mutating the supplied downloaded history', () => {
    const h = history(), before = structuredClone(h); build(h); expect(h).toEqual(before);
    h.id = '201'; expect(() => build(h)).toThrow(/identity/);
    expect(() => buildResearchGaugeRow(entry, history(), index, taxonomy, '2026-01-01')).toThrow(/date mismatch/);
  });
});
