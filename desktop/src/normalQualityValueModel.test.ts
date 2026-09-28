import { describe, expect, it } from 'vitest';
import type { ResearchGaugeRow } from './researchGaugeModel';
import { researchGaugeSeries } from './researchGaugeModel';
import type { ExpandedKpiContext } from './expandedKpis';
import { expandedVariant } from './expandedKpis';
import { companyListCell, defaultCompanyListPreferences, matchesCompanyListFilters, parseCompanyListPreferences } from './companyListModel';
import { buildNormalQualityValueContext } from './normalQualityValueModel';
import { NORMAL_RANKING_DEPENDENCIES } from './normalQualityValuePolicy';

function row(id = '1'): ResearchGaugeRow {
  const periods = Array.from({ length: 10 }, (_, i) => 2025 - i).map(year => ({ year, period: 5, start: `${year}-01-01`, end: `${year}-12-31`, published: `${year + 1}-02-01`, currency: 'SEK', sourceId: 'annual', sourceAsOf: '2026-08-10' }));
  return { id, name: `Example ${id}`, route: 'operating', classificationConflict: false, presence: 'latest', sourceAsOf: '2026-08-10', annual: { latest: periods[0], periods: periods.slice(0, 5), currency: 'SEK', cash: researchGaugeSeries([40, 40, 40, 40, 40]), operatingCash: researchGaugeSeries([50, 30, -999, 999, null]), ebit: researchGaugeSeries([30, 30, 30, 30, 30]), revenue: researchGaugeSeries([200, 150, -999, 999, null]), margins: researchGaugeSeries([15, 20, null, null, null]) },
    screeningAnnual: { asOf: '2026-09-13', periods, cash: periods.map(() => 40), operatingCash: [50, 30, -999, 999, null, 0, 20, 10, 5, 1], ebit: periods.map(() => 30), revenue: [200, 150, -999, 999, null, 0, 100, 75, 50, 25], equity: periods.map(() => 100), netDebt: periods.map(() => -20), assets: periods.map(() => 300), intangibleAssets: periods.map(() => 50), tangibleAssets: periods.map(() => 20), profit: periods.map(() => 25), reason: null },
    valuation: { currency: 'SEK', candidateEquity: 100, priceDate: '2026-02-09', priceBasis: { sourceId: 'annual', sourceAsOf: '2026-08-10', currency: 'SEK', shares: 1, close: 100, method: 'local', fxRate: null, fxDate: null } },
  } as unknown as ResearchGaugeRow;
}
function provider(ids = ['1', '2'], debt = 1): ExpandedKpiContext {
  return { index: { ids, positions: new Map(ids.map((id, i) => [id, i])), reportCurrencies: ids.map(() => 'SEK'), quoteCurrencies: ids.map(() => 'SEK') }, ready: true, error: '', loading: new Set(), errors: new Map(), variants: new Map(NORMAL_RANKING_DEPENDENCIES.map(column => {
    const variant = expandedVariant(column)!;
    return [variant.id, { variant, values: ids.map(() => column.kpiId === 'provider_42' ? debt : 20) }];
  })) };
}
const scoreColumn = { id: 'rank', kpiId: 'normal_quality_value_score', window: 'latest', calculation: 'latest' } as const;

describe('normal quality and value ranking integration', () => {
  it('uses the fixed quality policy, keeps exception facts and withholds unpriced rows without changing eligibility', () => {
    const a = row(), b = row('2'); b.valuation.candidateEquity = null;
    const before = structuredClone([a, b]), context = buildNormalQualityValueContext([a, b], provider());
    expect(context).toMatchObject({ status: 'available', eligibleCount: 2, cohortSize: 1 });
    expect(companyListCell(a, scoreColumn, provider(), context)).toMatchObject({ value: 50, unit: 'number', status: 'available' });
    expect(companyListCell(b, scoreColumn, provider(), context)).toMatchObject({ value: null, status: 'missing' });
    expect([a, b]).toEqual(before);
    expect(companyListCell(a, scoreColumn, provider(), context).detail).toMatch(/60% five-year cash-only NPV.*40% quality/);
  });
  it('ranks the explicit five-year cash-only NPV and preserves the ten-year comparisons', () => {
    const a = row(), b = row('2'); b.valuation.candidateEquity = 200; b.valuation.priceBasis!.close = 200;
    const p = provider(), context = buildNormalQualityValueContext([a, b], p);
    const npv5 = { id: 'npv5', kpiId: 'normal_npv_5y_percent', window: 'latest', calculation: 'latest' } as const;
    const expected5 = 100 * (40 * [1, 2, 3, 4, 5].reduce((s, t) => s + 1 / 1.1 ** t, 0) / 100 - 1);
    expect(companyListCell(a, npv5).value).toBeCloseTo(expected5, 10);
    expect(context.observations.get(a.id)?.discount).toBe(companyListCell(a, npv5).value);
    expect(companyListCell(a, scoreColumn, p, context).value).toBe(80);
    expect(companyListCell(b, scoreColumn, p, context).value).toBe(20);
    expect(companyListCell(a, npv5).detail).toMatch(/no terminal value/i);
    const oldHalf = companyListCell(a, { id: 'old-half', kpiId: 'normal_npv_percent', window: 'latest', calculation: 'terminal_50' });
    expect(oldHalf.value).toBeCloseTo(100 * ((40 * Array.from({ length: 10 }, (_, i) => 1 / 1.1 ** (i + 1)).reduce((s, x) => s + x, 0) + 200 / 1.1 ** 10) / 100 - 1), 10);
    const settings = defaultCompanyListPreferences(); settings.columns = [scoreColumn, npv5];
    settings.savedViews = [{ id: 'npv-five', name: 'Five-year cash', columns: settings.columns, filters: settings.filters, sort: settings.sort }];
    expect(parseCompanyListPreferences(JSON.stringify(settings))).toEqual(settings);
  });
  it('does not calculate a changing partial cohort while a required provider source is missing, loading or failed', () => {
    const a = row(), loaded = provider();
    expect(buildNormalQualityValueContext([a], undefined).status).toBe('loading');
    const variant = expandedVariant(NORMAL_RANKING_DEPENDENCIES[0])!;
    loaded.variants = new Map(); loaded.loading = new Set([variant.id]);
    expect(buildNormalQualityValueContext([a], loaded).status).toBe('loading');
    loaded.loading = new Set(); loaded.errors = new Map([[variant.id, 'test error']]);
    expect(buildNormalQualityValueContext([a], loaded).status).toBe('error');
    expect(companyListCell(a, scoreColumn).value).toBeNull();
  });
  it('applies every existing quality guard before forming the reference group', () => {
    for (const mutate of [
      (r: ResearchGaugeRow) => { r.route = 'financial'; },
      (r: ResearchGaugeRow) => { r.presence = 'older'; },
      (r: ResearchGaugeRow) => { r.classificationConflict = true; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.cash[6] = 0; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.profit.fill(1); },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.tangibleAssets.fill(1000); },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.ebit[0] = 5; },
      (r: ResearchGaugeRow) => { r.screeningAnnual!.operatingCash[6] = null; },
    ]) { const a = row(); mutate(a); expect(buildNormalQualityValueContext([a], provider()).eligibleCount).toBe(0); }
    expect(buildNormalQualityValueContext([row()], provider(['1'], 1.6)).eligibleCount).toBe(0);
    const p = provider(['1']); const margin = expandedVariant(NORMAL_RANKING_DEPENDENCIES[1])!;
    p.variants = new Map([...p.variants, [margin.id, { variant: margin, values: [0] }]]);
    expect(buildNormalQualityValueContext([row()], p).eligibleCount).toBe(0);
  });
  it('supports saved ranking columns, numeric rules and stable scores when display filters change', () => {
    const a = row(), b = row('2'); b.valuation.candidateEquity = 200; b.valuation.priceBasis!.close = 200;
    const p = provider(), context = buildNormalQualityValueContext([a, b], p);
    const settings = defaultCompanyListPreferences(); settings.columns = [scoreColumn]; settings.sort = { columnId: 'rank', direction: 'desc' };
    settings.filters.numericRules = [{ column: scoreColumn, operator: 'gte', value: 50 }];
    settings.savedViews = [{ id: 'saved-rank', name: 'Quality and discount', columns: settings.columns, filters: settings.filters, sort: settings.sort }];
    expect(parseCompanyListPreferences(JSON.stringify(settings))).toEqual(settings);
    expect(matchesCompanyListFilters(a, settings.filters, new Set(), p, context)).toBe(true);
    expect(matchesCompanyListFilters(b, settings.filters, new Set(), p, context)).toBe(false);
    const value = companyListCell(a, scoreColumn, p, context).value;
    settings.filters.query = a.name;
    expect(matchesCompanyListFilters(a, settings.filters, new Set(), p, context)).toBe(true);
    expect(companyListCell(a, scoreColumn, p, context).value).toBe(value);
  });
});
