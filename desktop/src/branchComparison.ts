import { amountLabels, formulas, ratioLabels } from './business';
import type { SourcedReport } from './financialData';
import type { Presence } from './listingCatalogue';
import { finite } from './model';

export const branchMetrics = {
  operating_margin: ratioLabels.operating_margin, operating_cash_margin: ratioLabels.operating_cash_margin,
  equity_ratio: ratioLabels.equity_ratio, net_debt_to_equity: ratioLabels.net_debt_to_equity,
  return_on_capital: ratioLabels.return_on_capital, revenues: amountLabels.revenues,
  operating_income: amountLabels.operating_income, free_cash_flow: amountLabels.free_cash_flow,
  net_debt: amountLabels.net_debt, total_assets: amountLabels.total_assets,
  cash_flow_from_investing_activities: 'Investing cash flow',
};
export type BranchMetric = keyof typeof branchMetrics;
export type BubbleSize = 'equal' | 'revenues' | 'total_assets';
export type BranchSettings = {
  metric: BranchMetric; size: BubbleSize; currency: string; country: string; presence: Presence;
  month: number; from: number; to: number; year: number; selected: string[]; focus: string | null;
};
export type BranchReport = Pick<SourcedReport, 'year' | 'start' | 'end' | 'report_date' | 'source_as_of' | 'currency'> & { values: Record<string, number | null> };
export type BranchData = Record<string, BranchReport[]>;
export type Benchmark = { year: number; n: number; total: number; median: number | null; q1: number | null; q3: number | null };
export const comparisonColors = ['#245782', '#8062a3', '#448aac', '#aaa1c9', '#394971', '#678da0', '#8c7385', '#55677a'];
export const monetary = (metric: BranchMetric) => !(metric in ratioLabels);
export const needsCurrency = (s: Pick<BranchSettings, 'metric' | 'size'>) => monetary(s.metric) || s.size !== 'equal';
export const metricUnit = (metric: BranchMetric, currency: string) => monetary(metric) ? `${currency} million` : '%';
export const metricDefinition = (metric: BranchMetric) => formulas[metric] ?? (metric === 'free_cash_flow' ? "Börsdata FCF excludes lease principal and interest; it is not owner earnings." : metric === 'cash_flow_from_investing_activities' ? 'Reported investing cash flow, including transactions. This is not a CAPEX measure.' : 'Reported nominal amount in the selected reporting currency; no inflation adjustment.');
export const defaultSettings = (lastYear: number): BranchSettings => ({ metric: 'operating_margin', size: 'equal', currency: 'all', country: 'all', presence: 'all', month: 0, from: Math.max(2000, lastYear - 10), to: lastYear, year: lastYear, selected: [], focus: null });
export function projectAnnual(reports: SourcedReport[]): BranchReport[] {
  return reports.map(r => ({ year: r.year, start: r.start, end: r.end, report_date: r.report_date, source_as_of: r.source_as_of, currency: r.currency,
    values: Object.fromEntries(Object.keys(branchMetrics).map(k => [k, r.values[k]])),
  }));
}
export function observation(r: BranchReport | undefined, s: BranchSettings) {
  let reason = '';
  if (!r) reason = 'No saved annual report';
  else {
    const days = (Date.parse(r.end) - Date.parse(r.start)) / 86400000 + 1;
    if (days < 330 || days > 400) reason = 'Period outside 330–400 days';
    else if (s.month && Number(r.end.slice(5, 7)) !== s.month) reason = 'Different fiscal closing month';
    else if (s.currency !== 'all' && r.currency !== s.currency) reason = 'Different reporting currency';
    else if (!finite(r.values[s.metric])) reason = 'Metric or currency conversion unavailable';
  }
  const value = reason ? null : r!.values[s.metric];
  const rawSize = s.size === 'equal' ? 1 : r?.values[s.size];
  const size = !reason && finite(rawSize) && rawSize > 0 ? rawSize : null;
  return { value, size, reason, sizeReason: value !== null && size === null ? 'Bubble size unavailable or non-positive' : '' };
}
function quantile(sorted: number[], p: number) {
  const at = (sorted.length - 1) * p, low = Math.floor(at), high = Math.ceil(at);
  // Weighted sum avoids overflowing the difference between opposite extreme values.
  return sorted[low] * (1 - (at - low)) + sorted[high] * (at - low);
}
export function benchmark(data: BranchData, ids: string[], settings: BranchSettings): Benchmark[] {
  if (needsCurrency(settings) && settings.currency === 'all') throw new Error('Select a reporting currency for monetary comparisons.');
  const years = new Map<number, number[]>();
  for (const id of ids) for (const r of data[id] ?? []) {
    if (r.year < settings.from || r.year > settings.to) continue;
    const { value } = observation(r, settings);
    if (value !== null) { const values = years.get(r.year) ?? []; values.push(value); years.set(r.year, values); }
  }
  return Array.from({ length: settings.to - settings.from + 1 }, (_, i) => {
    const year = settings.from + i, values = (years.get(year) ?? []).sort((a, b) => a - b);
    return { year, n: values.length, total: ids.length, median: values.length ? quantile(values, .5) : null,
      q1: values.length >= 4 ? quantile(values, .25) : null, q3: values.length >= 4 ? quantile(values, .75) : null };
  });
}
export function bubbleDiameter(size: number | null, maximum: number) {
  return finite(size) && size > 0 && maximum > 0 ? 42 * Math.sqrt(size / maximum) : 0;
}
