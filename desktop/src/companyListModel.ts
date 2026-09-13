import type { ResearchGaugeAnnualValues, ResearchGaugePeriod, ResearchGaugeReadiness, ResearchGaugeRoute, ResearchGaugeRow } from './researchGaugeModel';
import { normalizeSearch } from './listingCatalogue';
import { EXPANDED_KPIS, expandedKpiCell, expandedVariant, expandedUnit, providerWindowLabel, providerCalculationLabel } from './expandedKpis';
import type { ExpandedKpiContext } from './expandedKpis';
import { defaultCompanyListFeatures, parseCompanyListFeatures, matchesNumericCondition } from './companyListFeatures';
import type { CompanyListFeatures, CompanyListNumericCondition, CompanyListNumericOperator } from './companyListFeatures';

export type CompanyListWindow = 'latest' | '3' | '5' | `provider:${string}`;
export type CompanyListCalculation = 'latest' | 'average' | 'median' | 'min' | 'max' | 'growth' | `provider:${string}`;
export type CompanyKpiUnit = 'text' | 'date' | 'money' | 'price' | 'percent' | 'points' | 'multiple' | 'count' | 'number' | 'shares_millions';
export type CompanyKpi = { id: string; label: string; category: string; description: string; formula: string; unit: CompanyKpiUnit; windows: CompanyListWindow[]; calculations: CompanyListCalculation[]; supportedVariants?: CompanyListVariant[]; coverage?: number; source?: string; snapshot?: string; searchTerms?: string[] };
export type CompanyListVariant = { window: CompanyListWindow; calculation: CompanyListCalculation; windowLabel: string; calculationLabel: string };
export type CompanyListColumn = { id: string; kpiId: string; window: CompanyListWindow; calculation: CompanyListCalculation };
export type CompanyListCell = { value: number | string | null; display: string; detail: string; unit: CompanyKpiUnit; currency: string | null; date: string | null; status?: 'loading' | 'error' | 'available' | 'missing' };
export type CompanyListNumericRule = CompanyListNumericCondition & { column: CompanyListColumn };
export type CompanyListFilters = { query: string; sectorId: string; branchId: string; country: string; route: ResearchGaugeRoute | 'all'; readiness: ResearchGaugeReadiness | 'all'; presence: 'all' | 'latest' | 'older'; watchlistOnly: boolean; preset: 'all' | 'cash_consistency' | 'cash_and_margin'; numericRules: CompanyListNumericRule[] };
export type CompanyListSort = { columnId: string; direction: 'asc' | 'desc' };
export type CompanyListSavedView = { id: string; name: string; columns: CompanyListColumn[]; filters: CompanyListFilters; sort: CompanyListSort };
export type CompanyListPreferences = { version: 2; features: CompanyListFeatures; columns: CompanyListColumn[]; filters: CompanyListFilters; sort: CompanyListSort; watchlistIds: string[]; savedViews: CompanyListSavedView[] };
export const LEGACY_COMPANY_LIST_STORAGE_KEY = 'macro-atlas-company-lists-v1';
export const COMPANY_LIST_STORAGE_KEY = 'macro-atlas-company-lists-v2';
export const COMPANY_LIST_MAX_COLUMNS = 32;
export const COMPANY_LIST_MAX_VIEWS = 20;
const latest: CompanyListWindow[] = ['latest'];
const history: CompanyListWindow[] = ['latest', '3', '5'];
const aggregates: CompanyListCalculation[] = ['latest', 'average', 'median', 'min', 'max'];
const kpi = (id: string, label: string, category: string, unit: CompanyKpiUnit, description: string, formula: string, windows = latest, calculations: CompanyListCalculation[] = ['latest']): CompanyKpi => ({ id, label, category, unit, description, formula, windows, calculations });

/** Only fields present in the verified research artifact, or same-period arithmetic on them. */
export const COMPANY_KPIS: CompanyKpi[] = [
  kpi('country', 'Listing country', 'Company', 'text', 'Saved listing country; it does not identify operating exposure.', 'Saved directory listing country'),
  kpi('sector', 'Sector', 'Company', 'text', 'Saved sector classification; conflicts remain disclosed.', 'Saved directory sector'),
  kpi('branch', 'Branch', 'Company', 'text', 'Saved branch classification; conflicts remain disclosed.', 'Saved directory branch'),
  kpi('stock_close', 'Saved stock close', 'Market', 'price', 'Historical publication-window quote, with its original quote currency and date. Not a current quote.', 'Saved close per share'),
  kpi('price_date', 'Saved price date', 'Market', 'date', 'Date of the publication-window quote already bundled in Atlas.', 'Saved quote date'),
  kpi('price_age', 'Price age at snapshot', 'Market', 'count', 'Days between the saved quote and the fixed research snapshot, not today.', 'Snapshot date minus quote date, in days'),
  kpi('fcf', 'Provider FCF', 'Cash flow', 'money', 'Provider free cash flow, not reconciled owner cash. Negative amounts remain funding needs.', 'Saved annual free cash flow', history, [...aggregates, 'growth']),
  kpi('cfo', 'Operating cash flow', 'Cash flow', 'money', 'Saved cash flow from operating activities.', 'Saved annual operating cash flow', history, [...aggregates, 'growth']),
  kpi('fcf_margin', 'Provider FCF margin', 'Cash flow', 'percent', 'Same-report FCF divided by strictly positive revenue. Historical averages average period margins.', '100 × provider FCF / revenue', history, aggregates),
  kpi('financing_cash', 'Financing cash flow', 'Cash flow', 'money', 'Signed saved financing cash flow; a negative amount need not mean business deterioration.', 'Latest annual financing cash flow'),
  kpi('cash_component_difference', 'FCF component difference', 'Cash flow', 'money', 'Definition reconciliation aid; investing cash is not assumed to equal maintenance investment.', 'Provider FCF − operating cash flow − investing cash flow'),
  kpi('revenue', 'Revenue', 'Profitability', 'money', 'Saved annual revenue in reporting-currency millions.', 'Saved annual revenue', history, [...aggregates, 'growth']),
  kpi('ebit', 'EBIT', 'Profitability', 'money', 'Saved annual operating income, including negative observations.', 'Saved annual operating income', history, [...aggregates, 'growth']),
  kpi('ebit_margin', 'EBIT margin', 'Profitability', 'percent', 'Same-report operating income divided by strictly positive revenue.', '100 × EBIT / revenue', history, aggregates),
  kpi('net_debt', 'Net debt', 'Financial position', 'money', 'Provider net debt; negative values are retained. Aggregate liabilities are not substituted.', 'Latest annual provider net debt'),
  kpi('cash_balance', 'Cash and equivalents', 'Financial position', 'money', 'Saved annual balance-sheet cash and equivalents.', 'Latest annual cash and equivalents'),
  kpi('assets', 'Total assets', 'Financial position', 'money', 'Reported asset book value, not a market or liquidation appraisal.', 'Latest annual total assets'),
  kpi('equity', 'Total equity', 'Financial position', 'money', 'Reported total equity, not verified tangible common equity.', 'Latest annual total equity'),
  kpi('net_debt_assets', 'Net debt / assets', 'Financial position', 'percent', 'Requires strictly positive total assets. Retains net cash as negative net debt.', '100 × net debt / total assets'),
  kpi('equity_assets', 'Equity / assets', 'Financial position', 'percent', 'Requires strictly positive total assets; book ratio does not establish solvency.', '100 × total equity / total assets'),
  kpi('tangible_assets_revenue', 'Tangible assets / revenue', 'Financial position', 'multiple', 'Descriptive asset intensity, not maintenance investment or a measure of competitive advantage.', 'Tangible assets / strictly positive annual revenue'),
  kpi('intangible_assets_assets', 'Intangible assets / assets', 'Financial position', 'percent', 'Reported intangible asset share; accounting treatment varies.', '100 × intangible assets / strictly positive total assets'),
  kpi('quarter_revenue_change', 'Quarter revenue YoY', 'Recent quarter', 'percent', 'Latest compatible standalone quarter against the same fiscal quarter one year earlier.', '100 × (latest revenue / prior-year quarter revenue − 1)'),
  kpi('quarter_margin_change', 'Quarter EBIT margin YoY', 'Recent quarter', 'points', 'Change in EBIT margin between compatible same-quarter reports.', 'Latest quarter EBIT margin − prior-year quarter EBIT margin'),
  kpi('quarter_cash_change', 'Quarter FCF change YoY', 'Recent quarter', 'money', 'Signed FCF difference between compatible same-quarter reports, in that quarter pair’s currency.', 'Latest quarter provider FCF − prior-year quarter provider FCF'),
  kpi('mid_dcf', 'Starter Mid DCF', 'Starter valuation', 'money', 'Frozen standard starter whole-equity value; reviewed studies and local edits are separate.', 'Mid annual cash PV + Mid terminal PV'),
  kpi('mid_npv', 'Starter Mid NPV', 'Starter valuation', 'money', 'Mid value less the dated saved derived equity price. Terminal contributes once through value.', 'Mid DCF − saved derived equity price'),
  kpi('low_npv', 'Starter Low NPV', 'Starter valuation', 'money', 'Low scenario value less the dated saved derived equity price. No scenario probability is assigned.', 'Low DCF − saved derived equity price'),
  kpi('low_dcf', 'Starter Low DCF', 'Starter valuation', 'money', 'Low whole-equity scenario value; a sensitivity, not a guaranteed floor.', 'Low annual cash PV + Low terminal PV'),
  kpi('purchase_ceiling', 'Starter 30% ceiling', 'Starter valuation', 'money', 'Fixed screen policy of 30% below positive Mid whole-equity value; editable in working Value.', '0.70 × positive Mid DCF'),
  kpi('saved_equity_price', 'Saved derived equity price', 'Starter valuation', 'money', 'Historical equity-price proxy uses reported shares; share classes, splits and ownership require review.', 'Reported shares in millions × saved close × applicable recorded FX'),
  kpi('cash_pv', 'Starter cash PV', 'Starter valuation', 'money', 'Mid present value of explicit annual cash only.', 'Sum of discounted Mid annual cash flows'),
  kpi('terminal_pv', 'Starter terminal PV', 'Starter valuation', 'money', 'Mid discounted terminal value, already included in DCF.', 'Discounted Mid terminal value'),
  kpi('terminal_share', 'Starter terminal share', 'Starter valuation', 'percent', 'Terminal dependence; signed annual cash can make this exceed 100%.', '100 × Mid terminal PV / Mid DCF'),
  kpi('cash_factor', 'Required cash at saved price', 'Starter valuation', 'multiple', 'Uniformly scales annual and terminal cash at fixed assumptions; not an expected growth rate. Negative funding cash scales too.', 'Positive saved equity price / positive Mid DCF'),
  kpi('cash_factor_30', 'Required cash at 30% margin', 'Starter valuation', 'multiple', 'Uniform cash scaling for the fixed 30% margin, at unchanged rates, growth and horizon.', 'Positive saved equity price / (0.70 × positive Mid DCF)'),
  kpi('positive_fcf', 'Positive FCF observations', 'Evidence', 'count', 'Count of strictly positive annual FCF amounts. Requested periods must all be observed.', 'Count(provider FCF > 0) in selected report window', history),
  kpi('positive_ebit', 'Positive EBIT observations', 'Evidence', 'count', 'Count of strictly positive annual EBIT amounts. Requested periods must all be observed.', 'Count(EBIT > 0) in selected report window', history),
  kpi('annual_periods', 'Comparable annual reports', 'Evidence', 'count', 'Number of compatible annual periods retained, at most five; individual fields can still be missing.', 'Count of retained compatible annual periods'),
  kpi('annual_date', 'Latest annual period end', 'Evidence', 'date', 'Latest saved annual report end, including periods whose figures require reconciliation.', 'Latest saved annual period end'),
  ...EXPANDED_KPIS,
  kpi('coverage', 'Annual evidence coverage', 'Evidence', 'text', 'Coverage describes data availability and timing, not investment quality.', 'Saved annual evidence readiness'),
];
const catalogue = new Map(COMPANY_KPIS.map(item => [item.id, item]));
const counts = new Set(['positive_fcf', 'positive_ebit']);
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const calculated = (value: number): number | null => finite(value) ? value : null;
const collator = new Intl.Collator('en', { sensitivity: 'base', numeric: true });
const formatter = new Intl.NumberFormat('en-US', { maximumFractionDigits: 2 });
const coverageLabels: Record<ResearchGaugeReadiness, string> = { history_available: 'Five-period evidence', limited_history: 'Limited or older evidence', reconcile_data: 'Reconcile data', no_history: 'No annual history' };
const periodDays = (later: string, earlier: string) => (Date.parse(later) - Date.parse(earlier)) / 86400000;

function validColumn(input: unknown): input is CompanyListColumn {
  if (!input || typeof input !== 'object') return false;
  const c = input as CompanyListColumn, metric = catalogue.get(c.kpiId);
  if (metric?.supportedVariants) return validId(c.id) && metric.supportedVariants.some(v => v.window === c.window && v.calculation === c.calculation);
  return validId(c.id) && !!metric && metric.windows.includes(c.window) && metric.calculations.includes(c.calculation)
    && (counts.has(c.kpiId) || c.window === 'latest' ? c.calculation === 'latest' : c.calculation !== 'latest');
}
export function companyListWindowLabel(column: CompanyListColumn): string {
  const provider = expandedVariant(column as CompanyListColumn);
  if (provider) return providerWindowLabel(provider);
  if (column.window !== 'latest') return `${column.window} annual reports`;
  const category = catalogue.get(column.kpiId)?.category;
  return category === 'Company' ? 'Saved directory' : category === 'Market' ? 'Saved quote' : category === 'Starter valuation' ? 'Standard starter snapshot' : category === 'Recent quarter' ? 'Saved quarter comparison' : 'Latest annual report';
}
export function companyListCalculationLabel(column: CompanyListColumn): string {
  const provider = expandedVariant(column as CompanyListColumn);
  return provider ? providerCalculationLabel(provider) : column.calculation === 'growth' ? 'CAGR' : column.calculation === 'latest' ? counts.has(column.kpiId) ? 'Positive observations' : 'Saved value' : column.calculation;
}
export function companyListUnit(column: CompanyListColumn): CompanyKpiUnit {
  const provider = expandedVariant(column);
  return provider ? expandedUnit(provider.unit, provider.currencyBasis) : column.calculation === 'growth' ? 'percent' : catalogue.get(column.kpiId)?.unit ?? 'text';
}
export function companyListVariants(metric: CompanyKpi): CompanyListVariant[] {
  return metric.supportedVariants ?? metric.windows.flatMap(window => metric.calculations.filter(calculation => counts.has(metric.id) || window === 'latest' ? calculation === 'latest' : calculation !== 'latest').map(calculation => {
    const column = { id: 'variant', kpiId: metric.id, window, calculation };
    return { window, calculation, windowLabel: companyListWindowLabel(column), calculationLabel: companyListCalculationLabel(column) };
  }));
}
export function columnLabel(column: CompanyListColumn): string {
  const metric = catalogue.get(column.kpiId); if (!metric) return 'Unavailable KPI';
  if (metric.supportedVariants) return `${metric.label} · ${companyListWindowLabel(column)} · ${companyListCalculationLabel(column)}`;
  if (column.window === 'latest') return metric.label;
  const calc = column.calculation === 'growth' ? 'CAGR' : column.calculation === 'latest' ? 'count' : column.calculation;
  return `${metric.label} · ${column.window} reports ${calc}`;
}
function formatValue(value: number | string | null, unit: CompanyKpiUnit, currency: string | null): string {
  if (value === null) return '—'; if (typeof value === 'string') return value;
  const text = formatter.format(value);
  return unit === 'money' ? `${text} ${currency} m` : unit === 'price' ? `${text} ${currency}` : unit === 'percent' ? `${text}%` : unit === 'points' ? `${text} pp` : unit === 'multiple' ? `${text}×` : text;
}
function compatiblePeriods(row: ResearchGaugeRow, n: number): { periods: ResearchGaugePeriod[]; reason: string | null } {
  const periods = row.annual.periods.slice(0, n);
  if (periods.length !== n) return { periods, reason: `Requires ${n} comparable annual reports; ${periods.length} are available. ${row.annual.reason ?? ''}` };
  if (!row.annual.currency) return { periods, reason: 'Reporting currency is unavailable.' };
  for (let i = 0; i < periods.length; i++) {
    const p = periods[i], newer = periods[i - 1], length = periodDays(p.end, p.start) + 1;
    if (!finite(length) || length < 330 || length > 400 || p.currency !== row.annual.currency || newer && (p.year !== newer.year - 1 || periodDays(newer.start, p.end) < 1 || periodDays(newer.start, p.end) > 35)) return { periods, reason: 'Incompatible annual dates, reporting currency, or a gap prevents this calculation.' };
  }
  return { periods, reason: null };
}
const sourceDetail = (periods: ResearchGaugePeriod[]) => periods.map(p => `${p.start}–${p.end}; published ${p.published ?? 'date unavailable'}; source ${p.sourceId}, saved ${p.sourceAsOf}`).join(' | ');
const annualKeys: Record<string, keyof ResearchGaugeAnnualValues> = { financing_cash: 'financingCash', cash_component_difference: 'cashComponentDifference', net_debt: 'netDebt', cash_balance: 'cashBalance', assets: 'assets', equity: 'equity', net_debt_assets: 'netDebtToAssetsPercent', equity_assets: 'equityToAssetsPercent', tangible_assets_revenue: 'tangibleAssetsToRevenue', intangible_assets_assets: 'intangibleAssetsToAssetsPercent' };

export function companyListCell(row: ResearchGaugeRow, column: CompanyListColumn, expanded?: ExpandedKpiContext): CompanyListCell {
  if (column.kpiId.startsWith('provider_')) return expandedKpiCell(row, column, expanded);
  const metric = catalogue.get(column.kpiId), unit = column.calculation === 'growth' ? 'percent' : metric?.unit ?? 'text';
  let value: number | string | null = null, currency: string | null = null, date: string | null = null, detail = metric ? `${metric.description} Formula: ${metric.formula}.` : 'This KPI is unavailable in the verified data.';
  const finish = (reason?: string): CompanyListCell => ({ value, display: formatValue(value, unit, currency), detail: `${detail}${reason ? ` ${reason}` : ''}`, unit, currency, date });
  if (!metric || !validColumn(column)) return finish('This period or calculation is unsupported; no alternative is substituted.');
  const id = metric.id, v = row.valuation;
  if (metric.category === 'Company') {
    value = id === 'country' ? row.country : id === 'sector' ? row.sectorName : row.branchName; date = row.sourceAsOf;
    return finish(value === null ? 'Saved directory field unavailable.' : row.classificationConflict && id !== 'country' ? 'Saved classification has an unresolved conflict.' : `Directory saved ${row.sourceAsOf}.`);
  }
  if (metric.category === 'Market') {
    date = v.priceDate; currency = id === 'stock_close' ? v.priceBasis?.currency ?? null : null;
    value = id === 'stock_close' ? date && currency && finite(v.priceBasis?.close) ? v.priceBasis.close : null : id === 'price_date' ? date : v.priceAgeDays;
    return finish(`${value === null ? v.reason ?? 'Dated saved quote or currency unavailable.' : `Quote dated ${date}.`} Source ${v.priceBasis?.sourceId ?? 'unavailable'}, saved ${v.priceBasis?.sourceAsOf ?? 'unavailable'}.`);
  }
  if (metric.category === 'Starter valuation') {
    currency = unit === 'money' ? v.currency || null : null;
    date = ['mid_npv', 'low_npv', 'saved_equity_price', 'cash_factor', 'cash_factor_30'].includes(id) ? v.priceDate : null;
    const values: Record<string, number | null> = { mid_dcf: v.value, mid_npv: finite(v.value) && finite(v.candidateEquity) ? calculated(v.value - v.candidateEquity) : null, low_npv: v.lowNPV, low_dcf: v.lowValue, purchase_ceiling: v.ceiling, saved_equity_price: v.candidateEquity, cash_pv: v.cashPV, terminal_pv: v.terminalPV, terminal_share: finite(v.terminalShare) ? calculated(v.terminalShare * 100) : null, cash_factor: v.reverseCashFactor, cash_factor_30: v.reverseCashFactor30 };
    value = values[id] ?? null; if (unit === 'money' && !currency) value = null;
    return finish(`${value === null ? v.reason ?? 'Required starter scenario or saved price input unavailable.' : 'Frozen standard starter; no forecast probability or purchase verdict.'} Saved price ${v.priceDate ?? 'unavailable'}; source ${v.priceBasis?.sourceId ?? 'price unavailable'}. ${v.hasSignedCash ? 'Negative cash includes funding needs, which scale with the cash factor.' : ''}`);
  }
  if (metric.category === 'Recent quarter') {
    const q = row.quarter; date = q.latest?.end ?? null; currency = unit === 'money' ? q.latest?.currency ?? null : null;
    value = id === 'quarter_revenue_change' ? q.revenueChangePercent : id === 'quarter_margin_change' ? q.ebitMarginChangePoints : q.cashChange;
    if (q.reason || !q.latest || !q.comparison || (unit === 'money' && (!currency || currency !== q.comparison.currency))) value = null;
    return finish(`${value === null ? q.reason ?? 'Required comparable quarter values unavailable.' : ''} ${sourceDetail([q.latest, q.comparison].filter((p): p is ResearchGaugePeriod => p !== null))}`);
  }
  date = row.annual.periods[0]?.end ?? row.annual.latest?.end ?? null;
  if (id === 'annual_periods' || id === 'annual_date' || id === 'coverage') {
    value = id === 'annual_periods' ? row.annual.periods.length : id === 'annual_date' ? row.annual.latest?.end ?? null : coverageLabels[row.readiness];
    return finish(`${row.annual.reason ?? ''} Source ${row.annual.latest?.sourceId ?? 'unavailable'}, saved ${row.annual.latest?.sourceAsOf ?? row.sourceAsOf}.`);
  }
  const n = column.window === 'latest' ? 1 : Number(column.window), checked = compatiblePeriods(row, n);
  currency = unit === 'money' ? row.annual.currency : null;
  if (checked.reason) return finish(checked.reason);
  detail += ` Reports: ${sourceDetail(checked.periods)}.`;
  const series: Record<string, (number | null)[]> = { fcf: row.annual.cash.values, cfo: row.annual.operatingCash.values, ebit: row.annual.ebit.values, revenue: row.annual.revenue.values, ebit_margin: row.annual.margins.values,
    fcf_margin: row.annual.cash.values.map((cash, i) => finite(cash) && finite(row.annual.revenue.values[i]) && row.annual.revenue.values[i]! > 0 ? calculated(100 * cash / row.annual.revenue.values[i]!) : null),
    positive_fcf: row.annual.cash.values, positive_ebit: row.annual.ebit.values };
  if (annualKeys[id]) { value = row.annual.latestValues[annualKeys[id]]; return finish(value === null ? 'Required latest annual field or positive denominator is missing.' : undefined); }
  const values = series[id]?.slice(0, n) ?? [];
  if (values.length !== n || values.some(x => !finite(x))) return finish(`Missing observations in the requested ${n}-report window; incomplete averages and older substitutions are withheld.`);
  const observed = values as number[];
  if (counts.has(id)) { value = observed.filter(x => x > 0).length; return { ...finish(`${value} of ${n} observed reports are strictly positive.`), display: `${value} / ${n}` }; }
  if (column.calculation === 'growth') {
    if (observed.some(x => x <= 0)) return finish('CAGR requires strictly positive observations throughout the selected window; zero, negative and sign-changing paths are not assigned growth rates.');
    const years = periodDays(checked.periods[0].end, checked.periods.at(-1)!.end) / 365.25;
    value = years > 0 ? calculated(100 * ((observed[0] / observed.at(-1)!) ** (1 / years) - 1)) : null;
    return finish(`CAGR = 100 × ((latest / oldest)^(1 / elapsed years) − 1), across ${n} reports and ${formatter.format(years)} elapsed years. All selected values are observed.`);
  }
  const ordered = [...observed].sort((a, b) => a - b);
  value = column.calculation === 'latest' ? observed[0] : column.calculation === 'average' ? calculated(observed.reduce((sum, x) => sum + x / n, 0)) : column.calculation === 'median' ? ordered[Math.floor(n / 2)] : column.calculation === 'min' ? ordered[0] : ordered.at(-1)!;
  return finish(value === null ? 'Calculated amount is outside the finite numeric range.' : undefined);
}

/** Monetary sorts group currency ascending, then amount in the chosen direction. Missing always last. */
export function compareCompanyListRows(a: ResearchGaugeRow, b: ResearchGaugeRow, column: CompanyListColumn | null, direction: 'asc' | 'desc'): number {
  const tie = () => collator.compare(a.name, b.name) || collator.compare(a.id, b.id);
  if (!column) return (direction === 'asc' ? 1 : -1) * tie();
  return compareCompanyListCells(companyListCell(a, column), companyListCell(b, column), direction) || tie();
}
/** Reuse precomputed cells when sorting large result sets; zero lets the caller apply a stable name/id tie. */
export function compareCompanyListCells(av: CompanyListCell, bv: CompanyListCell, direction: 'asc' | 'desc'): number {
  if (av.value === null || bv.value === null) return av.value === null && bv.value === null ? 0 : av.value === null ? 1 : -1;
  const currencyOrder = collator.compare(av.currency ?? '', bv.currency ?? ''); if (currencyOrder) return currencyOrder;
  const order = typeof av.value === 'number' && typeof bv.value === 'number' ? av.value - bv.value : collator.compare(String(av.value), String(bv.value));
  return direction === 'asc' ? order : -order;
}

export function matchesCompanyListFilters(row: ResearchGaugeRow, filters: CompanyListFilters, watchlistIds: ReadonlySet<string>, expanded?: ExpandedKpiContext): boolean {
  if (filters.watchlistOnly && !watchlistIds.has(row.id)) return false;
  if (filters.query && !normalizeSearch(`${row.name} ${row.ticker ?? ''} ${row.isin ?? ''} ${row.id}`).includes(normalizeSearch(filters.query))) return false;
  for (const [selected, actual] of [[filters.sectorId, row.sectorId], [filters.branchId, row.branchId], [filters.country, row.country], [filters.route, row.route], [filters.readiness, row.readiness], [filters.presence, row.presence]]) if (selected !== 'all' && selected !== (actual ?? 'unassigned')) return false;
  if (filters.preset !== 'all') {
    if (row.route !== 'operating' || row.classificationConflict || row.readiness !== 'history_available' || row.annual.periods.length !== 5 || [row.annual.cash, row.annual.ebit].some(series => series.values.length !== 5 || series.values.some(x => !finite(x) || x <= 0))) return false;
    const v = row.valuation;
    if (filters.preset === 'cash_and_margin' && (v.status !== 'positive-priced' || !finite(v.value) || v.value <= 0 || !finite(v.candidateEquity) || v.candidateEquity <= 0 || !finite(v.ceiling) || v.ceiling <= 0 || v.candidateEquity > v.ceiling)) return false;
  }
  return filters.numericRules.every(rule => matchesNumericCondition(companyListCell(row, rule.column, expanded), rule));
}

export const DEFAULT_COMPANY_LIST_COLUMNS: CompanyListColumn[] = ['country', 'branch', 'stock_close', 'positive_fcf', 'positive_ebit', 'ebit_margin', 'quarter_revenue_change', 'cash_factor_30', 'terminal_share'].map(kpiId => ({ id: kpiId, kpiId, window: counts.has(kpiId) ? '5' : 'latest', calculation: 'latest' }));
const defaultFilters = (): CompanyListFilters => ({ query: '', sectorId: 'all', branchId: 'all', country: 'all', route: 'all', readiness: 'all', presence: 'all', watchlistOnly: false, preset: 'all', numericRules: [] });
export function defaultCompanyListPreferences(): CompanyListPreferences { return { version: 2, features: defaultCompanyListFeatures(), columns: DEFAULT_COMPANY_LIST_COLUMNS.map(c => ({ ...c })), filters: defaultFilters(), sort: { columnId: 'name', direction: 'asc' }, watchlistIds: [], savedViews: [] }; }
const object = (input: unknown): Record<string, unknown> => input !== null && typeof input === 'object' && !Array.isArray(input) ? input as Record<string, unknown> : {};
const validId = (input: unknown): input is string => typeof input === 'string' && /^[A-Za-z0-9_-]{1,80}$/.test(input);
const text = (input: unknown, fallback: string, limit: number) => typeof input === 'string' ? input.replace(/[\u0000-\u001f\u007f]/g, '').slice(0, limit) : fallback;
function parseColumns(input: unknown): CompanyListColumn[] {
  const seen = new Set<string>(), result: CompanyListColumn[] = [];
  if (Array.isArray(input)) for (const candidate of input.slice(0, 128)) if (validColumn(candidate) && !seen.has(candidate.id)) { seen.add(candidate.id); result.push({ id: candidate.id, kpiId: candidate.kpiId, window: candidate.window, calculation: candidate.calculation }); if (result.length === COMPANY_LIST_MAX_COLUMNS) break; }
  return result.length ? result : DEFAULT_COMPANY_LIST_COLUMNS.map(c => ({ ...c }));
}
function parseFilters(input: unknown): CompanyListFilters {
  const f = object(input), defaults = defaultFilters();
  const choice = <T extends string>(value: unknown, options: readonly T[], fallback: T): T => options.includes(value as T) ? value as T : fallback;
  const numericRules: CompanyListNumericRule[] = [];
  if (Array.isArray(f.numericRules)) for (const raw of f.numericRules.slice(0, 24)) {
    const r = object(raw); if (!validColumn(r.column) || !['gte', 'lte', 'gt', 'lt', 'eq', 'between', 'present', 'missing'].includes(String(r.operator))) continue;
    const unit = companyListUnit(r.column); if (unit === 'text' || unit === 'date') continue;
    const currency = typeof r.currency === 'string' && /^[A-Z]{3}$/.test(r.currency) ? r.currency : undefined;
    numericRules.push({ column: { ...r.column }, operator: r.operator as CompanyListNumericOperator, value: finite(r.value) ? r.value : null, ...(r.operator === 'between' ? { valueTo: finite(r.valueTo) ? r.valueTo : null } : {}), ...(currency ? { currency } : {}) }); if (numericRules.length === 12) break;
  }
  return { ...defaults, query: text(f.query, '', 200), sectorId: text(f.sectorId, 'all', 80), branchId: text(f.branchId, 'all', 80), country: text(f.country, 'all', 80), route: choice(f.route, ['all', 'operating', 'property', 'financial', 'unclassified'], 'all'), readiness: choice(f.readiness, ['all', 'history_available', 'limited_history', 'reconcile_data', 'no_history'], 'all'), presence: choice(f.presence, ['all', 'latest', 'older'], 'all'), watchlistOnly: f.watchlistOnly === true, preset: choice(f.preset, ['all', 'cash_consistency', 'cash_and_margin'], 'all'), numericRules };
}
function parseSort(input: unknown, columns: CompanyListColumn[]): CompanyListSort {
  const sort = object(input); return { columnId: sort.columnId === 'name' || columns.some(c => c.id === sort.columnId) ? sort.columnId as string : 'name', direction: sort.direction === 'desc' ? 'desc' : 'asc' };
}
/** Parse local preferences without touching storage. Unknown listing IDs survive data-pack changes. */
export function parseCompanyListPreferences(input: unknown): CompanyListPreferences {
  if (typeof input === 'string') { if (input.length > 16_000_000) return defaultCompanyListPreferences(); try { input = JSON.parse(input); } catch { return defaultCompanyListPreferences(); } }
  const saved = object(input); if (saved.version !== 1 && saved.version !== 2) return defaultCompanyListPreferences();
  const columns = parseColumns(saved.columns), seenViews = new Set<string>(), savedViews: CompanyListSavedView[] = [];
  if (Array.isArray(saved.savedViews)) for (const candidate of saved.savedViews.slice(0, 100)) {
    const view = object(candidate), name = text(view.name, '', 80).trim(); if (!validId(view.id) || !name || seenViews.has(view.id)) continue;
    seenViews.add(view.id); const viewColumns = parseColumns(view.columns);
    savedViews.push({ id: view.id, name, columns: viewColumns, filters: parseFilters(view.filters), sort: parseSort(view.sort, viewColumns) }); if (savedViews.length === COMPANY_LIST_MAX_VIEWS) break;
  }
  const watchlistIds = Array.isArray(saved.watchlistIds) ? [...new Set(saved.watchlistIds.slice(0, 50000).filter((id): id is string => typeof id === 'string' && /^[0-9]{1,20}$/.test(id)))].slice(0, 25000) : [];
  const features = parseCompanyListFeatures(saved.version === 2 ? saved.features : null, watchlistIds, columns);
  return { version: 2, features, columns, filters: parseFilters(saved.filters), sort: parseSort(saved.sort, columns), watchlistIds: features.watchlists.find(list => list.id === 'default')?.listingIds ?? [], savedViews };
}
