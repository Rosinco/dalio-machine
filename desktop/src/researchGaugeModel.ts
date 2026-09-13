import type { FinancialCompany, FinancialIndex, SourcedReport } from './financialData';
import type { CompanyEntry } from './listingCatalogue';
import type { Classification, Taxonomy } from './taxonomy';
import { buildStarterValuation } from './starterValuations';
import { calculatePurchaseRange, defaultPurchaseRangeSettings } from './purchaseRange';
import { validAmount, validDay } from './valuation';

export type ResearchGaugeRoute = 'operating' | 'property' | 'financial' | 'unclassified';
export type ResearchGaugeReadiness = 'history_available' | 'limited_history' | 'reconcile_data' | 'no_history';
export type ResearchGaugePeriod = { year: number; period: number; start: string; end: string; published: string | null; currency: string; sourceId: string; sourceAsOf: string };
export type ResearchGaugeSeries = { values: (number | null)[]; count: number; positive: number; negative: number; zero: number; latest: number | null; median: number | null; dispersion: number | null };
export type ResearchGaugeAnnualValues = {
  revenues: number | null; ebit: number | null; cash: number | null; operatingCash: number | null;
  financingCash: number | null; cashBalance: number | null; netDebt: number | null; equity: number | null; assets: number | null;
  netDebtToAssetsPercent: number | null; equityToAssetsPercent: number | null;
  tangibleAssetsToRevenue: number | null; intangibleAssetsToAssetsPercent: number | null;
  cashComponentDifference: number | null;
};
export type ResearchGaugeRow = {
  id: string; name: string; ticker: string | null; isin: string | null; country: string | null;
  sectorId: string | null; sectorName: string | null; branchId: string | null; branchName: string | null;
  sourceAsOf: string; presence: 'latest' | 'older'; sourceCompanySha256: string;
  route: ResearchGaugeRoute; classificationConflict: boolean; readiness: ResearchGaugeReadiness;
  annual: {
    currency: string | null; latest: ResearchGaugePeriod | null; periods: ResearchGaugePeriod[];
    excluded: { outsideWindow: number; placeholder: number; missingPublication: number; withheld: number };
    cash: ResearchGaugeSeries; operatingCash: ResearchGaugeSeries; ebit: ResearchGaugeSeries; revenue: ResearchGaugeSeries; margins: ResearchGaugeSeries;
    latestValues: ResearchGaugeAnnualValues; reason: string | null;
  };
  quarter: { latest: ResearchGaugePeriod | null; comparison: ResearchGaugePeriod | null; revenueChangePercent: number | null; ebitMarginChangePoints: number | null; cashChange: number | null; reason: string | null };
  valuation: {
    status: 'positive-priced' | 'positive-unpriced' | 'nonpositive' | 'manual-financial' | 'missing';
    currency: string; value: number | null; cashPV: number | null; terminalPV: number | null; terminalShare: number | null;
    candidateEquity: number | null; priceDate: string | null; priceAgeDays: number | null;
    priceBasis: { sourceId: string; sourceAsOf: string; reportDate: string | null; reportEnd: string; shares: number | null; close: number | null; currency: string | null; method: 'local' | 'sek'; fxRate: number | null; fxDate: string | null } | null;
    ceiling: number | null; lowValue: number | null; lowNPV: number | null;
    reverseCashFactor: number | null; reverseCashFactor30: number | null; hasSignedCash: boolean;
    annualHistoryCount: number; rangeStatus: 'historical' | 'percentage' | 'unavailable'; reason: string | null;
  };
  issues: string[];
};
export type ResearchGaugeArtifact = {
  format: 'macro-atlas-research-gauge'; version: 1; asOf: string; financialPackId: string; taxonomySha256: string;
  calibrationId: string; model: 'research-gauge-v1'; rows: ResearchGaugeRow[];
};
export type ResearchGaugeManifest = {
  format: 'macro-atlas-research-gauge-manifest'; version: 1; asOf: string; model: 'research-gauge-v1';
  financialPackId: string; financialPackBytes: number; taxonomySha256: string; calibrationId: string;
  rows: number; artifact: { path: string; sha256: string; bytes: number; uncompressedSha256: string; uncompressedBytes: number };
};

const days = (later: string, earlier: string) => (Date.parse(later) - Date.parse(earlier)) / 86400000;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const number = (value: unknown) => validAmount(value) ? value : null;
const calculated = (value: number) => finite(value) ? value : null;
const value = (report: SourcedReport | null, key: string) => number(report?.values[key]);
const divide = (a: number | null, b: number | null, scale = 1) => a !== null && b !== null && b > 0 ? calculated(scale * a / b) : null;
const placeholder = (report: SourcedReport) => Object.values(report.raw).some(x => x === 0) && Object.values(report.raw).every(x => x === null || x === 0);
const period = (r: SourcedReport): ResearchGaugePeriod => ({ year: r.year, period: r.period, start: r.start, end: r.end, published: r.report_date, currency: r.currency, sourceId: r.source_id, sourceAsOf: r.source_as_of });
const known = (r: SourcedReport, asOf: string) => r.end <= asOf && r.source_as_of <= asOf && (r.report_date === null || r.report_date <= asOf);
const newestFirst = (a: SourcedReport, b: SourcedReport) => b.end.localeCompare(a.end) || b.year - a.year || b.period - a.period;

/** Signed population dispersion / mean absolute value. No scale means no ratio, not zero risk. */
export function researchGaugeSeries(values: (number | null)[]): ResearchGaugeSeries {
  const included = values.filter(finite), ordered = [...included].sort((a, b) => a - b), count = included.length;
  const mean = count ? included.reduce((sum, x) => sum + x, 0) / count : null;
  const scale = count ? included.reduce((sum, x) => sum + Math.abs(x), 0) / count : null;
  const dispersion = count >= 2 && scale !== null && scale > 0 ? calculated(Math.sqrt(included.reduce((sum, x) => sum + (x - mean!) ** 2, 0) / count) / scale) : null;
  const median = count ? calculated((ordered[Math.floor((count - 1) / 2)] + ordered[Math.floor(count / 2)]) / 2) : null;
  return { values: [...values], count, positive: included.filter(x => x > 0).length, negative: included.filter(x => x < 0).length, zero: included.filter(x => x === 0).length, latest: values[0] ?? null, median, dispersion };
}

export function buildResearchGaugeRow(entry: CompanyEntry, history: FinancialCompany, index: FinancialIndex, taxonomy: Taxonomy, asOf: string): ResearchGaugeRow {
  if (!validDay(asOf) || asOf < index.as_of || history.id !== entry.id || !index.companies[entry.id]) throw new Error('Research gauge source identity or date mismatch.');
  const classification: Classification | undefined = taxonomy.classifications[entry.id];
  if (!classification || classification.company_id !== entry.id) throw new Error('Research gauge classification source unavailable.');
  const sectorId = classification.sector_id, branchId = classification.branch_id;
  const conflict = ['sector_mismatch', 'needs_review'].includes(classification.status);
  const route: ResearchGaugeRoute = !sectorId || !branchId ? 'unclassified' : ['75', '76'].includes(branchId) ? 'property' : sectorId === '1' ? 'financial' : 'operating';
  const reports = history.annual.filter(r => known(r, asOf)).sort(newestFirst), latest = reports[0] ?? null;
  const annualReports: SourcedReport[] = [], issues: string[] = [];
  let annualReason: string | null = null;
  const newerWithheld = history.withheld.some(r => r.period === 5 && (!latest || r.year >= latest.year || (r.end && r.end >= latest.end)));
  if (!latest) annualReason = 'No retained annual report is available.';
  else if (newerWithheld) annualReason = 'A same or newer annual report was withheld; older figures are not substituted.';
  else for (const report of reports.slice(0, 5)) {
    const previous = annualReports.at(-1), length = days(report.end, report.start) + 1;
    if (length < 330 || length > 400) { annualReason = 'A report is outside 330–400 days; the comparable annual window stops before it.'; break; }
    if (placeholder(report)) { annualReason = 'All saved monetary fields in a report are zero or missing; the annual window stops before this possible placeholder.'; break; }
    if (!(report.currency_ratio !== null && report.currency_ratio > 0) || report.currency !== latest!.currency) { annualReason = 'Missing currency conversion or changed reporting currency prevents a comparable annual window.'; break; }
    if (previous && (report.year !== previous.year - 1 || days(previous.start, report.end) < 1 || days(previous.start, report.end) > 35)) { annualReason = 'An annual gap or overlapping period stops the comparable history window.'; break; }
    annualReports.push(report);
  }
  if (annualReason) issues.push(annualReason);
  if (latest && days(asOf, latest.end) > 550) issues.push('Latest annual period ended more than 550 days before this snapshot; historical evidence is stale.');
  if (annualReports.some(r => r.report_date === null)) issues.push('At least one included annual publication date is unavailable; historical timing is not established.');
  if (conflict) issues.push('The saved sector/branch classification has an unresolved conflict.');
  if (route === 'financial') issues.push('Financial business: use a reviewed equity-capital and distributions model; automatic cash valuation is unavailable.');
  if (route === 'property') issues.push('Property business: review recurring property cash, maintenance investment and financing before interpreting the starter valuation.');
  if (route === 'unclassified') issues.push('Business classification is unavailable; select the appropriate research method after review.');
  const latestUsable = annualReports[0] ?? null, assets = value(latestUsable, 'total_assets'), revenues = value(latestUsable, 'revenues');
  const fcf = value(latestUsable, 'free_cash_flow'), cfo = value(latestUsable, 'cash_flow_from_operating_activities'), cfi = value(latestUsable, 'cash_flow_from_investing_activities');
  const latestValues: ResearchGaugeAnnualValues = {
    revenues, ebit: value(latestUsable, 'operating_income'), cash: fcf, operatingCash: cfo,
    financingCash: value(latestUsable, 'cash_flow_from_financing_activities'), cashBalance: value(latestUsable, 'cash_and_equivalents'),
    netDebt: value(latestUsable, 'net_debt'), equity: value(latestUsable, 'total_equity'), assets,
    netDebtToAssetsPercent: divide(value(latestUsable, 'net_debt'), assets, 100), equityToAssetsPercent: divide(value(latestUsable, 'total_equity'), assets, 100),
    tangibleAssetsToRevenue: divide(value(latestUsable, 'tangible_assets'), revenues), intangibleAssetsToAssetsPercent: divide(value(latestUsable, 'intangible_assets'), assets, 100),
    cashComponentDifference: fcf !== null && cfo !== null && cfi !== null ? calculated(fcf - cfo - cfi) : null,
  };
  const series = (key: string) => researchGaugeSeries(annualReports.map(r => value(r, key)));
  const annual: ResearchGaugeRow['annual'] = {
    currency: latest?.currency ?? null, latest: latest ? period(latest) : null, periods: annualReports.map(period),
    excluded: { outsideWindow: history.annual.length - annualReports.length, placeholder: reports.filter(placeholder).length, missingPublication: reports.filter(r => r.report_date === null).length, withheld: history.withheld.filter(r => r.period === 5).length },
    cash: series('free_cash_flow'), operatingCash: series('cash_flow_from_operating_activities'), ebit: series('operating_income'), revenue: series('revenues'),
    margins: researchGaugeSeries(annualReports.map(r => divide(value(r, 'operating_income'), value(r, 'revenues'), 100))), latestValues, reason: annualReason,
  };
  const quarters = history.quarterly.filter(r => known(r, asOf)).sort(newestFirst), q = quarters[0] ?? null;
  const previousQuarter = q ? quarters.find(r => r.year === q.year - 1 && r.period === q.period) ?? null : null;
  const quarter: ResearchGaugeRow['quarter'] = { latest: q ? period(q) : null, comparison: previousQuarter ? period(previousQuarter) : null, revenueChangePercent: null, ebitMarginChangePoints: null, cashChange: null, reason: null };
  const quarterLength = (r: SourcedReport) => days(r.end, r.start) + 1;
  if (!q) quarter.reason = 'No retained quarterly report is available.';
  else if (history.withheld.some(r => r.period !== 5 && (r.year * 4 + r.period >= q.year * 4 + q.period))) quarter.reason = 'A same or newer quarterly period was withheld; no older comparison is substituted.';
  else if (!previousQuarter) quarter.reason = 'The same quarter in the preceding fiscal year is unavailable.';
  else if ([q, previousQuarter].some(r => placeholder(r) || r.currency_ratio === null || r.currency_ratio <= 0 || !r.report_date || quarterLength(r) < 60 || quarterLength(r) > 120)
    || q.currency !== previousQuarter.currency || Math.abs(quarterLength(q) - quarterLength(previousQuarter)) > 15
    || days(q.end, previousQuarter.end) < 330 || days(q.end, previousQuarter.end) > 400) quarter.reason = 'The quarter pair has a placeholder, missing publication/conversion, different currency or incompatible standalone period length.';
  else {
    const revenue = value(q, 'revenues'), previousRevenue = value(previousQuarter, 'revenues');
    quarter.revenueChangePercent = revenue !== null && previousRevenue !== null ? divide(revenue - previousRevenue, previousRevenue, 100) : null;
    const margin = divide(value(q, 'operating_income'), revenue, 100), previousMargin = divide(value(previousQuarter, 'operating_income'), previousRevenue, 100);
    quarter.ebitMarginChangePoints = margin !== null && previousMargin !== null ? calculated(margin - previousMargin) : null;
    const cash = value(q, 'free_cash_flow'), previousCash = value(previousQuarter, 'free_cash_flow');
    quarter.cashChange = cash !== null && previousCash !== null ? calculated(cash - previousCash) : null;
    if ([quarter.revenueChangePercent, quarter.ebitMarginChangePoints, quarter.cashChange].every(x => x === null)) quarter.reason = 'The comparable quarters lack the required monetary fields or positive revenue denominators.';
  }
  const starter = buildStarterValuation({ ...entry, ...classification }, history, index, asOf);
  const purchase = calculatePurchaseRange(starter.draft, defaultPurchaseRangeSettings), mid = purchase.scenarios.mid, low = purchase.scenarios.low, price = starter.evidence.price;
  const hasSignedCash = starter.draft.scenarios.mid.cashFlows.some(x => x !== null && x < 0) || (starter.draft.scenarios.mid.terminalCash?.cashFlow ?? 0) < 0;
  const reverse = (denominator: number | null) => purchase.candidateEquity !== null && denominator !== null && denominator > 0 ? calculated(purchase.candidateEquity / denominator) : null;
  const valuation: ResearchGaugeRow['valuation'] = {
    status: route === 'financial' ? 'manual-financial' : mid.value === null ? 'missing' : mid.value <= 0 ? 'nonpositive' : purchase.candidateEquity === null ? 'positive-unpriced' : 'positive-priced',
    currency: starter.draft.currency, value: mid.value, cashPV: mid.cashPV, terminalPV: mid.terminalPV, terminalShare: mid.terminalShare,
    candidateEquity: purchase.candidateEquity, priceDate: purchase.candidateEquity !== null ? starter.draft.priceDate : null,
    priceAgeDays: purchase.candidateEquity !== null ? days(asOf, starter.draft.priceDate) : null,
    priceBasis: price ? { sourceId: price.sourceId, sourceAsOf: price.sourceAsOf, reportDate: price.reportDate, reportEnd: price.reportEnd, shares: price.shares, close: price.close, currency: price.currency, method: price.method, fxRate: price.fxRate, fxDate: price.fxDate } : null,
    ceiling: mid.ceiling, lowValue: low.value, lowNPV: low.npvAtCandidate, reverseCashFactor: reverse(mid.value), reverseCashFactor30: reverse(mid.ceiling), hasSignedCash,
    annualHistoryCount: starter.evidence.annual.length, rangeStatus: starter.evidence.uncertainty.status,
    reason: mid.error ?? purchase.candidateError ?? (mid.value !== null && mid.value <= 0 ? 'Mid starter value is nonpositive; a positive purchase ceiling or required cash multiple is unavailable.' : null),
  };
  if (hasSignedCash) issues.push('Starter cash includes negative amounts; uniformly scaling it also scales modeled funding needs.');
  if (mid.terminalShare !== null && mid.terminalShare > 1) issues.push('Terminal contribution exceeds 100% because negative annual cash offsets terminal value.');
  const readiness: ResearchGaugeReadiness = !latest ? 'no_history' : !annualReports.length || conflict ? 'reconcile_data'
    : annualReports.length < 5 || days(asOf, latest.end) > 550 || annualReports.some(r => r.report_date === null) || [annual.cash, annual.operatingCash, annual.ebit, annual.revenue].some(s => s.count < 5) ? 'limited_history' : 'history_available';
  return { id: entry.id, name: entry.display_name, ticker: entry.ticker, isin: entry.isin, country: entry.listing_country,
    sectorId, sectorName: sectorId ? taxonomy.sectors[sectorId]?.name_en ?? null : null, branchId, branchName: branchId ? taxonomy.branches[branchId]?.name_en ?? null : null,
    sourceAsOf: entry.source_as_of, presence: entry.source_as_of === taxonomy.catalogue?.as_of ? 'latest' : 'older', sourceCompanySha256: index.companies[entry.id].sha256,
    route, classificationConflict: conflict, readiness, annual, quarter, valuation, issues };
}
