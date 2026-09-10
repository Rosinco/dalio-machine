import { amountLabels, type FinancialReport } from './business';
import { finite } from './model';
import { decodeMarketRows, validateMarketIndex, type MarketCoverage, type MarketMetadata, type MarketReport } from './marketData';

export const statementLabels: Record<string, string> = { ...amountLabels,
  gross_income: 'Gross profit', profit_before_tax: 'Profit before tax', net_sales: 'Net sales',
  total_liabilities_and_equity: 'Total liabilities & equity', cash_flow_from_investing_activities: 'Investing cash flow',
  cash_flow_from_financing_activities: 'Financing cash flow', cash_flow_for_the_year: 'Net cash flow for the period',
};
export const financialColumns = Object.keys(statementLabels);
export type Coverage = { count: number; first: number | null; last: number | null; last_period: number | null; end: string | null; published: string | null; gaps: number; unavailable: number };
export type CompanyCoverage = { annual: Coverage; quarterly: Coverage; withheld: number; currencies: string[]; sha256: string; market?: MarketCoverage };
export type FinancialSource = { id: string; as_of: string; frequency: 'annual' | 'quarterly'; path: string; sha256: string; bytes: number; rows: number; outside_directory: number; usable: number; withheld: number };
export type FinancialIndex = {
  id: string; bytes: number; format: string; version: number; taxonomy_sha256: string; as_of: string; generated_at: string;
  columns: string[]; sources: FinancialSource[]; companies: Record<string, CompanyCoverage>;
  market?: MarketMetadata;
  summary: { listings: number; with_reports: number; annual: number; quarterly: number; withheld: number; source_rows: number; outside_directory: number; superseded: number };
};
export type SourcedReport = FinancialReport & { source_id: string; source_as_of: string };
export type WithheldReport = { year: number; period: number; source_id: string; start: string; end: string; published: string; reason: string };
export type FinancialCompany = { id: string; annual: SourcedReport[]; quarterly: SourcedReport[]; withheld: WithheldReport[]; market: MarketReport[] };
const check = (ok: unknown, message = 'Invalid financial history pack.') => { if (!ok) throw new Error(message); };
const object = (x: any) => x !== null && typeof x === 'object' && !Array.isArray(x);
const day = (x: any): x is string => typeof x === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(x) && x >= '1800-01-01' && x <= '2300-12-31' && Number.isFinite(Date.parse(x)) && new Date(x).toISOString().slice(0, 10) === x;
const integer = (x: any, max = 10000000) => Number.isSafeInteger(x) && x >= 0 && x <= max;
const hash = (x: any) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const companyId = (x: any) => typeof x === 'string' && /^[1-9][0-9]{0,9}$/.test(x);
const currency = (x: any) => typeof x === 'string' && /^[A-Z]{3}$/.test(x);
const numeric = (x: any) => x === null || finite(x);
export const hasFinancialHistory = (c?: CompanyCoverage | null) => !!c && c.annual.count + c.quarterly.count > 0;

export function validateFinancialIndex(v: any, taxonomy: string): asserts v is FinancialIndex {
  check(v?.format === 'macro-atlas-financials' && [1, 2].includes(v.version) && hash(v.id) && hash(taxonomy) && v.taxonomy_sha256 === taxonomy, 'Financial history belongs to a different company directory or uses an unsupported format.');
  check(integer(v.bytes, 512 * 1024 * 1024) && v.bytes > 0 && day(v.as_of));
  check(typeof v.generated_at === 'string' && /^\d{4}-\d{2}-\d{2}T.+(?:Z|\+00:00)$/.test(v.generated_at) && Number.isFinite(Date.parse(v.generated_at)));
  check(Array.isArray(v.columns) && v.columns.join(',') === financialColumns.join(','), 'Unsupported financial statement fields.');
  check(object(v.companies) && Object.keys(v.companies).length <= 100000 && object(v.summary));
  let annual = 0, quarterly = 0, withheld = 0, withReports = 0;
  for (const [id, c] of Object.entries(v.companies) as [string, any][]) {
    check(companyId(id) && object(c) && hash(c.sha256) && integer(c.withheld, 10000));
    check(Array.isArray(c.currencies) && c.currencies.length <= 100 && c.currencies.every(currency) && new Set(c.currencies).size === c.currencies.length);
    for (const freq of ['annual', 'quarterly']) {
      const f = c[freq]; check(object(f) && integer(f.count, 2000) && integer(f.gaps, 2000) && integer(f.unavailable, f.count));
      if (!f.count) check(['first', 'last', 'last_period', 'end', 'published'].every(k => f[k] === null) && f.gaps === 0);
      else check(integer(f.first, 2300) && f.first >= 2000 && integer(f.last, 2300) && f.last >= f.first && (freq === 'annual' ? f.last_period === 5 : integer(f.last_period, 4) && f.last_period >= 1) && day(f.end) && f.end <= v.as_of && (f.published === null || day(f.published) && f.published >= f.end && f.published <= v.as_of));
    }
    annual += c.annual.count; quarterly += c.quarterly.count; withheld += c.withheld; withReports += Number(hasFinancialHistory(c));
  }
  const s = v.summary;
  check(['listings', 'with_reports', 'annual', 'quarterly', 'withheld', 'source_rows', 'outside_directory', 'superseded'].every(k => integer(s[k])));
  check(s.listings === Object.keys(v.companies).length && s.with_reports === withReports && s.annual === annual && s.quarterly === quarterly && s.withheld === withheld && s.source_rows === s.outside_directory + s.superseded + annual + quarterly + withheld, 'Financial coverage counts do not reconcile.');
  check(Array.isArray(v.sources) && v.sources.length > 0 && v.sources.length <= 1000);
  const ids = new Set<string>(); let sourceRows = 0, outside = 0, usableRows = 0, withheldRows = 0;
  for (const source of v.sources) {
    check(object(source) && typeof source.id === 'string' && source.id.length > 0 && source.id.length <= 100 && !ids.has(source.id) && day(source.as_of) && source.as_of <= v.as_of && ['annual', 'quarterly'].includes(source.frequency) && hash(source.sha256));
    check(typeof source.path === 'string' && source.path.length <= 2000 && source.path.split('/').every((p: string) => p && p !== '.' && p !== '..') && !/[\\:]/.test(source.path));
    check(integer(source.bytes, 2000000000) && ['rows', 'outside_directory', 'usable', 'withheld'].every(k => integer(source[k])) && source.outside_directory + source.usable + source.withheld <= source.rows);
    ids.add(source.id); sourceRows += source.rows; outside += source.outside_directory; usableRows += source.usable; withheldRows += source.withheld;
  }
  check(sourceRows === s.source_rows && outside === s.outside_directory && usableRows === annual + quarterly && withheldRows === withheld, 'Financial source totals do not reconcile.');
  validateMarketIndex(v);
}

export function decodeFinancialRows(rows: any, index: FinancialIndex, id: string, freq: 'annual' | 'quarterly'): SourcedReport[] {
  const coverage = index.companies[id];
  check(companyId(id) && coverage, 'Financial history does not match the selected listing.');
  const sources = new Map(index.sources.map(s => [s.id, s])), currencies = new Set<string>();
  check(Array.isArray(rows) && rows.length === coverage[freq].count && rows.length <= 2000);
  let previous = 0;
  const reports = rows.map((r: any) => {
    check(Array.isArray(r) && r.length === 8 + financialColumns.length);
    const [year, period, start, end, published, code, fx, sourceId] = r, source = sources.get(sourceId);
    check(integer(year, 2300) && year >= 2000 && (freq === 'annual' ? period === 5 : integer(period, 4) && period >= 1) && year * 5 + period > previous, 'Duplicate or unordered financial periods.'); previous = year * 5 + period;
    check(source?.frequency === freq && day(start) && day(end) && start <= end && end <= source.as_of && (published === null || day(published) && published >= end && published <= source.as_of), 'Invalid report dates or financial source.');
    check(currency(code) && numeric(fx) && r.slice(8).every(numeric)); currencies.add(code);
    const raw: Record<string, number | null> = {}, values: Record<string, number | null> = {};
    financialColumns.forEach((key, i) => { raw[key] = r[8 + i]; const amount = fx !== null && fx > 0 && raw[key] !== null ? raw[key]! / fx : null; values[key] = finite(amount) ? amount : null; });
    const ratio = (numerator: number | null, denominator: number | null) => { const n = numerator !== null && denominator !== null && denominator > 0 ? 100 * numerator / denominator : null; return finite(n) ? n : null; };
    values.operating_margin = ratio(values.operating_income, values.revenues);
    values.operating_cash_margin = ratio(values.cash_flow_from_operating_activities, values.revenues);
    values.equity_ratio = ratio(values.total_equity, values.total_assets);
    values.net_debt_to_equity = ratio(values.net_debt, values.total_equity);
    values.return_on_capital = period === 5 ? ratio(values.operating_income, values.total_equity !== null && values.net_debt !== null ? values.total_equity + values.net_debt : null) : null;
    return { year, period, start, end, report_date: published, currency: code, currency_ratio: fx, source_id: sourceId, source_as_of: source!.as_of, raw, values };
  });
  const first = reports[0], last = reports.at(-1), c = coverage[freq];
  const position = (r: SourcedReport) => freq === 'annual' ? r.year : r.year * 4 + r.period - 1;
  check(c.first === (first?.year ?? null) && c.last === (last?.year ?? null) && c.last_period === (last?.period ?? null) && c.end === (last?.end ?? null) && c.published === (last?.report_date ?? null) && c.gaps === (last ? position(last) - position(first) + 1 - reports.length : 0) && c.unavailable === reports.filter((r: SourcedReport) => r.currency_ratio === null || r.currency_ratio <= 0 || Object.values(r.raw).every(n => n === null)).length, 'Financial reports disagree with their coverage index.');
  check([...currencies].every(c => coverage.currencies.includes(c)), 'Unexpected reporting currency.');
  return reports;
}

export function decodeFinancialCompany(v: any, index: FinancialIndex, id: string): FinancialCompany {
  const coverage = index.companies[id];
  check(companyId(id) && coverage && object(v) && v.id === id, 'Financial history does not match the selected listing.');
  const annual = decodeFinancialRows(v.annual, index, id, 'annual'), quarterly = decodeFinancialRows(v.quarterly, index, id, 'quarterly');
  const currencies = new Set([...annual, ...quarterly].map(r => r.currency));
  const sources = new Map(index.sources.map(s => [s.id, s]));
  check([...currencies].sort().join(',') === [...coverage.currencies].sort().join(','));
  check(Array.isArray(v.withheld) && v.withheld.length === coverage.withheld && v.withheld.every((r: any) => object(r) && Number.isSafeInteger(r.year) && Number.isSafeInteger(r.period) && sources.has(r.source_id) && ['start', 'end', 'published', 'reason'].every(k => typeof r[k] === 'string' && r[k].length <= 2000)));
  return { id, annual, quarterly, withheld: v.withheld, market: decodeMarketRows(v.market, index, id, annual) };
}
