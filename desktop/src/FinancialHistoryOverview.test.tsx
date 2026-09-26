import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import fixture from '../tests/fixtures/financial.json';
import { decodeFinancialCompany, type FinancialCompany, type FinancialIndex, type SourcedReport } from './financialData';
import type { CompanyEntry } from './listingCatalogue';
import FinancialHistory from './FinancialHistory';

type ChartProps = { label: string; labels: string[]; unit: string; series: { name: string; values: (number | null)[] }[] };
const captured = vi.hoisted(() => [] as ChartProps[]);
vi.mock('./FinancialChart', () => ({ default: (props: ChartProps) => { captured.push(props); return null; } }));
vi.mock('./MarketHistory', () => ({ default: () => null }));
vi.mock('./ProviderCapexPanel', () => ({ default: () => null }));

const entry: CompanyEntry = { id: '102', name: 'Fixture Company', display_name: 'Fixture Company', ticker: 'TEST', isin: null, country_id: 'SE', listing_country: 'SE', sector_id: null, branch_id: null, instrument_type: 0, source_as_of: '2026-08-10', listing_date: null, stock_currency: 'SEK', report_currency: 'EUR', profile: null, search: 'fixture company' };
function sample() {
  const index = structuredClone({ ...fixture.index, id: 'b'.repeat(64), bytes: 1000 }) as FinancialIndex;
  return { index, original: decodeFinancialCompany(fixture.companies['102'], index, '102').annual[0] };
}
function report(original: SourcedReport, year: number, currency: string, values: Record<string, number | null>, period = 5): SourcedReport {
  return { ...original, year, period, currency, values: { ...original.values, ...values } };
}
function render(index: FinancialIndex, annual: SourcedReport[], quarterly: SourcedReport[] = []) {
  const company: FinancialCompany = { id: entry.id, annual, quarterly, withheld: [], market: [] };
  return renderToStaticMarkup(createElement(FinancialHistory, { entry, index, company }));
}
const overview = () => captured.filter(chart => chart.label.endsWith('business overview'));
const chart = (title: string) => overview().find(chart => chart.label.includes(` ${title} `))!;

describe('business overview preserves reported cash and balance-sheet meaning', () => {
  beforeEach(() => captured.splice(0));
  it('orders the six business charts and preserves signed cash, missing FCF, currency gaps and fiscal gaps', () => {
    const { index, original } = sample();
    index.companies['102'].currencies = ['EUR', 'SEK'];
    const annual = [
      report(original, 2023, 'EUR', { revenues: 100, cash_flow_from_operating_activities: 0, free_cash_flow: null, cash_flow_from_investing_activities: -9, total_assets: 100, tangible_assets: 70, intangible_assets: 40, net_debt: -4, cash_and_equivalents: 12 }),
      report(original, 2024, 'SEK', { revenues: 999, cash_flow_from_operating_activities: 888, free_cash_flow: 777, cash_flow_from_investing_activities: 666 }),
      report(original, 2026, 'EUR', { revenues: 130, cash_flow_from_operating_activities: 30, free_cash_flow: 5, cash_flow_from_investing_activities: 23, total_assets: 150, tangible_assets: 80, intangible_assets: 45, net_debt: -7, cash_and_equivalents: 21 }),
    ];
    const html = render(index, annual);
    expect(overview().map(chart => chart.label)).toEqual(['Revenue', 'Profit & operating cash margins', 'Operating & free cash flow', 'Investing cash flow', 'Asset base', 'Equity, net debt & cash'].map(title => `Fixture Company ${title} annual business overview`));
    expect(overview().every(chart => chart.labels.join(',') === '2023,2024,2025,2026')).toBe(true);
    expect(chart('Revenue').series[0].values).toEqual([100, null, null, 130]);
    expect(chart('Operating & free cash flow').series.map(series => series.values)).toEqual([[0, null, null, 30], [null, null, null, 5]]);
    expect(chart('Investing cash flow').series[0].values).toEqual([-9, null, null, 23]);
    expect(chart('Asset base').series.map(series => series.values)).toEqual([[100, null, null, 150], [70, null, null, 80], [40, null, null, 45]]);
    expect(chart('Equity, net debt & cash').series[1].values).toEqual([-4, null, null, -7]);
    expect(chart('Profit & operating cash margins').unit).toBe('%');
    expect(chart('Investing cash flow').unit).toBe('EUR million');
    expect(overview().flatMap(chart => chart.series).some(series => /capex/i.test(series.name))).toBe(false);
    expect(html).toContain('Maintenance and growth capex are not separated');
    expect(html).toContain('Total assets already includes its components');
    expect(html).toContain('Net debt already reflects the cash');
  });

  it('uses standalone quarters when annual reports are absent and retains an unavailable investing-cash series', () => {
    const { index, original } = sample();
    index.companies['102'].annual.count = 0; index.companies['102'].quarterly.count = 2;
    const quarterly = [
      report(original, 2026, 'EUR', { cash_flow_from_operating_activities: 7, free_cash_flow: 4, cash_flow_from_investing_activities: null }, 1),
      report(original, 2026, 'EUR', { cash_flow_from_operating_activities: -3, free_cash_flow: 2, cash_flow_from_investing_activities: null }, 3),
    ];
    const html = render(index, [], quarterly);
    expect(overview().every(chart => chart.label.includes(' quarterly business overview'))).toBe(true);
    expect(chart('Operating & free cash flow').labels).toEqual(['2026 Q1', '2026 Q2', '2026 Q3']);
    expect(chart('Operating & free cash flow').series.map(series => series.values)).toEqual([[7, null, -3], [4, null, 2]]);
    expect(chart('Investing cash flow').series[0].values).toEqual([null, null, null]);
    expect(html).toContain('Standalone quarters, not trailing 12 months');
  });
});
