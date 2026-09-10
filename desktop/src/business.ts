import { finite } from './model';

export type Observatory = 'macro' | 'sectors' | 'companies';
export type FinancialReport = {
  year: number; period: number; start: string; end: string; report_date: string | null;
  currency: string; currency_ratio: number | null; raw: Record<string, number | null>; values: Record<string, number | null>;
};
export type SavedStudy = { title: string; as_of: string; basis: string; source_id: string; text: string };
export type BusinessSource = { id: string; label: string; path: string; sha256: string; bytes: number };
export type Company = {
  id: string; name: string; listing_name: string; ticker: string; isin: string; listing_country: string;
  report_currency: string; stock_currency: string; sector_id: string; branch_id: string; segments: string[];
  overlap: string; context_as_of: string; context_source_id: string; instrument_source_id: string; annual_source_id: string; quarterly_source_id: string;
  annual: FinancialReport[]; quarterly: FinancialReport[]; research: SavedStudy[];
};
export type CompanySummary = Omit<Company, 'annual' | 'quarterly' | 'research'> & {
  latest_annual: FinancialReport | null; latest_quarter: FinancialReport | null; comparison: FinancialReport | null; research_count: number;
};
export type BusinessDocument = {
  version: number; as_of: string; exported_at: string; default_company: string; countries: Record<string, string>;
  branch: { id: string; name: string; source_name: string; sector_id: string; sector_name: string; source_id: string; as_of: string; description: string; drivers: { title: string; text: string; indicator: string }[] };
  companies: Record<string, Company>; sources: BusinessSource[]; limitations: string[]; research: SavedStudy[];
};
export type BusinessIndex = Omit<BusinessDocument, 'companies' | 'research'> & { companies: Record<string, CompanySummary>; common_year: number | null };
export const amountLabels: Record<string, string> = {
  revenues: 'Revenue', operating_income: 'Operating profit', profit_to_equity_holders: 'Net profit',
  cash_flow_from_operating_activities: 'Operating cash flow', free_cash_flow: 'Börsdata FCF',
  total_equity: 'Equity', net_debt: 'Net debt', total_assets: 'Total assets', cash_and_equivalents: 'Cash & equivalents',
  current_assets: 'Current assets', current_liabilities: 'Current liabilities', non_current_assets: 'Non-current assets',
  non_current_liabilities: 'Non-current liabilities', intangible_assets: 'Intangible assets', tangible_assets: 'Tangible assets', financial_assets: 'Financial assets',
};
export const ratioLabels: Record<string, string> = { operating_margin: 'Operating margin', operating_cash_margin: 'Operating cash margin', return_on_capital: 'Return on capital (proxy)', equity_ratio: 'Equity / assets', net_debt_to_equity: 'Net debt / equity' };
export const metricLabels = { ...amountLabels, ...ratioLabels };
export const formulas: Record<string, string> = {
  operating_margin: '100 × operating profit / revenue', operating_cash_margin: '100 × operating cash flow / revenue',
  return_on_capital: '100 × annual operating profit / (year-end equity + net debt). Annual pre-tax proxy; not adjusted ROIC.',
  equity_ratio: '100 × equity / total assets', net_debt_to_equity: '100 × net debt / equity',
};
export const periodLabel = (report: FinancialReport) => report.period === 5 ? `FY ${report.year}` : `Q${report.period} ${report.year}`;

// Render gaps for absent periods and changes of reporting currency. No invented zeroes.
export function financialSeries(reports: FinancialReport[], metric: string, currency: string) {
  if (!reports.length) return { labels: [], values: [] };
  const annual = reports[0].period === 5;
  const position = (r: FinancialReport) => annual ? r.year : r.year * 4 + r.period - 1;
  const byPeriod = new Map(reports.map(r => [position(r), r]));
  const labels: string[] = [], values: (number | null)[] = [];
  for (let i = position(reports[0]); i <= position(reports.at(-1)!); i++) {
    labels.push(annual ? String(i) : `${Math.floor(i / 4)} Q${i % 4 + 1}`);
    const row = byPeriod.get(i);
    values.push(row && (metric in ratioLabels || row.currency === currency) && finite(row.values[metric]) ? row.values[metric] : null);
  }
  return { labels, values };
}

export function businessIndex(raw: BusinessDocument): BusinessIndex {
  const periodKey = (r: FinancialReport) => `${r.year}|${r.start}|${r.end}`;
  const groups = Object.values(raw.companies).map(c => new Set(c.annual.map(periodKey)));
  const common = groups.length ? [...groups[0]].filter(k => groups.every(g => g.has(k))).sort().at(-1) : undefined;
  const { research: _research, companies, ...meta } = raw;
  return { ...meta, common_year: common ? Number(common.split('|')[0]) : null, companies: Object.fromEntries(Object.entries(companies).map(([id, c]) => {
    const { annual, quarterly, research, ...summary } = c;
    return [id, { ...summary, latest_annual: annual.at(-1) ?? null, latest_quarter: quarterly.at(-1) ?? null, comparison: annual.find(r => periodKey(r) === common) ?? null, research_count: research.length }];
  })) };
}

export function validateBusiness(raw: any): asserts raw is BusinessDocument {
  const check = (ok: unknown, message = 'The business research panel is invalid.') => { if (!ok) throw new Error(message); };
  const obj = (x: any) => x !== null && typeof x === 'object' && !Array.isArray(x);
  const texts = (x: any, keys: string[]) => check(obj(x) && keys.every(k => typeof x[k] === 'string'));
  const list = (x: any, max: number): any[] => { check(Array.isArray(x) && x.length <= max); return x; };
  const date = (x: any) => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x && x >= '2000-01-01' && x <= '2300-12-31';
  const numeric = (x: any) => x === null || finite(x);
  const id = (x: any) => typeof x === 'string' && /^[1-9][0-9]{0,9}$/.test(x);
  check(raw?.version === 1 && date(raw.as_of), 'Unsupported business version or snapshot date.');
  check(typeof raw.exported_at === 'string' && /^\d{4}-\d{2}-\d{2}T.+(?:Z|\+00:00)$/.test(raw.exported_at) && Number.isFinite(Date.parse(raw.exported_at)));
  check(obj(raw.countries) && Object.keys(raw.countries).length <= 300 && Object.entries(raw.countries).every(([k, v]) => /^[A-Z]{2}$/.test(k) && typeof v === 'string'));
  const sources = new Set<string>();
  for (const s of list(raw.sources, 1000)) { texts(s, ['id', 'label', 'path', 'sha256']); check(!sources.has(s.id) && /^[a-f0-9]{64}$/.test(s.sha256) && Number.isSafeInteger(s.bytes) && s.bytes >= 0); sources.add(s.id); }
  const studies = (items: any) => { for (const r of list(items, 100)) { texts(r, ['title', 'basis', 'source_id', 'text']); check(date(r.as_of) && sources.has(r.source_id) && new TextEncoder().encode(r.text).length <= 1000000); } };
  studies(raw.research);
  check(list(raw.limitations, 100).every(x => typeof x === 'string'));
  texts(raw.branch, ['id', 'name', 'source_name', 'sector_id', 'sector_name', 'source_id', 'description']);
  check(id(raw.branch.id) && id(raw.branch.sector_id) && date(raw.branch.as_of) && sources.has(raw.branch.source_id));
  for (const d of list(raw.branch.drivers, 100)) texts(d, ['title', 'text', 'indicator']);
  check(obj(raw.companies) && Object.keys(raw.companies).length > 0 && Object.keys(raw.companies).length <= 300 && raw.companies[raw.default_company]);
  for (const [key, c] of Object.entries(raw.companies) as [string, any][]) {
    texts(c, ['id', 'name', 'listing_name', 'ticker', 'isin', 'listing_country', 'report_currency', 'stock_currency', 'sector_id', 'branch_id', 'overlap', 'context_source_id', 'instrument_source_id', 'annual_source_id', 'quarterly_source_id']);
    check(id(key) && c.id === key && raw.countries[c.listing_country] && c.branch_id === raw.branch.id && c.sector_id === raw.branch.sector_id && date(c.context_as_of));
    check(/^[A-Z]{3}$/.test(c.report_currency) && /^[A-Z]{3}$/.test(c.stock_currency));
    check(['context_source_id', 'instrument_source_id', 'annual_source_id', 'quarterly_source_id'].every(k => sources.has(c[k])));
    check(list(c.segments, 100).every(x => typeof x === 'string')); studies(c.research);
    for (const kind of ['annual', 'quarterly']) {
      let previous = 0;
      for (const r of list(c[kind], kind === 'annual' ? 400 : 2000)) {
        check(obj(r) && Number.isInteger(r.year) && r.year >= 2000 && r.year <= 2300 && Number.isInteger(r.period) && (kind === 'annual' ? r.period === 5 : r.period >= 1 && r.period <= 4));
        check(r.year * 5 + r.period > previous, 'Duplicate or unordered business financial periods.'); previous = r.year * 5 + r.period;
        check(date(r.start) && date(r.end) && r.start <= r.end && r.end <= raw.as_of && (r.report_date === null || date(r.report_date) && r.report_date >= r.end && r.report_date <= raw.as_of), 'Invalid business financial period dates.');
        check(typeof r.currency === 'string' && /^[A-Z]{3}$/.test(r.currency) && numeric(r.currency_ratio));
        check(obj(r.raw) && Object.keys(amountLabels).every(k => numeric(r.raw[k])) && obj(r.values) && Object.keys(metricLabels).every(k => numeric(r.values[k])));
        check(kind === 'annual' || r.values.return_on_capital === null, 'Quarterly return on capital must be unavailable.');
        check(r.currency_ratio !== null && r.currency_ratio > 0 || Object.values(r.values).every(x => x === null), 'Missing currency conversion must leave financial values unavailable.');
      }
    }
  }
}
