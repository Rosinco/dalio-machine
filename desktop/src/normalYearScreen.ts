import type { ResearchGaugePeriod, ResearchGaugeRow } from './researchGaugeModel';
import { validDay } from './valuation';

export const NORMAL_YEAR_WINDOW = 'normal_2020_2023:5' as const;
export const NORMAL_YEAR_LABEL = '5 annual reports excluding 2020–2023 fiscal ends';
export const NORMAL_YEAR_ASSUMPTIONS = 'Separate screening assumption: median provider FCF from five normal fiscal periods, held constant in nominal terms for 10 years; 10% annual discount rate and 0% perpetual growth. Provider FCF is not reconciled owner cash. No cash or debt is added to this equity-cash proxy. Signed funding and continuing cash liabilities are retained, without a limited-liability floor or recovery estimate. No calibrated Low scenario, probability or expected-return claim. The original starter and user valuations are unchanged.';
export const NORMAL_YEAR_FIVE_YEAR_ASSUMPTIONS = 'Separate five-year cash-recovery sensitivity: median provider FCF from five normal fiscal periods, held constant in nominal terms for years 1–5 and discounted at 10% annually. No terminal value, sale proceeds or cash flows after year 5 are included. This compares five-year cash PV with the dated saved whole-equity price; it is not the total fair value of a continuing business. Provider FCF is not reconciled owner cash. No cash or debt is added, and signed funding needs are retained without a recovery estimate. No expected-return or probability claim. The original ten-year screening, starter and user valuations are unchanged.';
export type NormalYearSelection = { indices: number[]; periods: ResearchGaugePeriod[]; excluded: ResearchGaugePeriod[]; currency: string | null; reason: string | null };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const days = (later: string, earlier: string) => (Date.parse(later) - Date.parse(earlier)) / 86400000;
const excludedYear = (period: ResearchGaugePeriod) => period.end.slice(0, 4) >= '2020' && period.end.slice(0, 4) <= '2023';
const ratio = (numerator: number | null | undefined, denominator: number | null | undefined, scale = 1) => finite(numerator) && finite(denominator) && denominator > 0 && finite(scale * numerator / denominator) ? scale * numerator / denominator : null;

/** Select periods before inspecting values, so a missing observation cannot backfill an older year. */
export function selectNormalYearPeriods(row: ResearchGaugeRow): NormalYearSelection {
  const history = row.screeningAnnual;
  const result: NormalYearSelection = { indices: [], periods: [], excluded: [], currency: null, reason: null };
  const unavailable = (reason: string) => ({ ...result, reason });
  if (!history || !Array.isArray(history.periods) || !validDay(history.asOf)) return unavailable('Extended annual screening history is unavailable in this release. No five-report fallback is substituted.');
  if (history.periods.length > 10 || !history.periods.length) return unavailable(`Requires five comparable annual reports outside 2020–2023. ${history.reason ?? 'No comparable annual history is available.'}`);
  const latest = history.periods[0];
  if (!validDay(latest.end) || latest.end.slice(0, 4) <= '2023' || row.annual.latest?.end !== latest.end) return unavailable('A latest annual report ending after 2023 is required; an older normal period cannot replace the latest evidence.');
  if (days(history.asOf, latest.end) > 550 || latest.end > history.asOf) return unavailable('The latest annual period must end within 550 days of the screening snapshot.');
  result.currency = latest.currency;
  if (!/^[A-Z]{3}$/.test(latest.currency)) return unavailable('Reporting currency is unavailable.');
  for (let i = 0; i < history.periods.length && result.indices.length < 5; i++) {
    const p = history.periods[i], newer = history.periods[i - 1];
    if (!validDay(p.start) || !validDay(p.end) || !validDay(p.sourceAsOf) || p.end > p.sourceAsOf || p.sourceAsOf > history.asOf || p.currency !== latest.currency || p.period !== 5 || days(p.end, p.start) < 329 || days(p.end, p.start) > 399
      || newer && (p.year !== newer.year - 1 || days(newer.start, p.end) < 1 || days(newer.start, p.end) > 35)) return unavailable('Incompatible fiscal dates, reporting currency, source timing or an annual gap prevents normal-year selection. Exception years do not bridge invalid source history.');
    if (excludedYear(p)) result.excluded.push(p);
    else {
      result.indices.push(i); result.periods.push(p);
      if (!validDay(p.published) || p.published < p.end || p.published > p.sourceAsOf) return unavailable('Each selected normal annual report requires a known valid publication date.');
    }
  }
  if (result.indices.length !== 5) return unavailable(`Requires five comparable annual reports outside 2020–2023; ${result.indices.length} are available. ${history.reason ?? ''}`);
  return result;
}

/** All inputs are same-report observations in reporting-currency millions. */
export function normalYearValues(row: ResearchGaugeRow, kpiId: string, selection = selectNormalYearPeriods(row)): (number | null)[] | null {
  if (selection.reason || !row.screeningAnnual) return null;
  const h = row.screeningAnnual;
  const values = selection.indices.map(i => {
    if (kpiId === 'fcf' || kpiId === 'positive_fcf') return h.cash[i];
    if (kpiId === 'cfo') return h.operatingCash[i];
    if (kpiId === 'ebit' || kpiId === 'positive_ebit') return h.ebit[i];
    if (kpiId === 'revenue') return h.revenue[i];
    if (kpiId === 'ebit_margin') return ratio(h.ebit[i], h.revenue[i], 100);
    if (kpiId === 'fcf_margin') return ratio(h.cash[i], h.revenue[i], 100);
    if (kpiId === 'tangible_assets_revenue') return ratio(h.tangibleAssets[i], h.revenue[i]);
    if (kpiId === 'normal_roce') return ratio(h.ebit[i], finite(h.equity[i]) && finite(h.netDebt[i]) ? h.equity[i]! + h.netDebt[i]! : null, 100);
    if (kpiId === 'normal_rota') return ratio(h.profit[i], finite(h.assets[i]) && finite(h.intangibleAssets[i]) ? h.assets[i]! - h.intangibleAssets[i]! : null, 100);
    return null;
  });
  return values.map(value => finite(value) ? value : null);
}

export function normalYearDetail(row: ResearchGaugeRow, selection = selectNormalYearPeriods(row)): string {
  const source = (p: ResearchGaugePeriod) => `${p.start}–${p.end}; published ${p.published ?? 'unavailable'}; source ${p.sourceId}, saved ${p.sourceAsOf}`;
  const h = row.screeningAnnual;
  const skipped = selection.excluded.map(p => {
    const i = h!.periods.indexOf(p), show = (value: number | null) => finite(value) ? String(value) : 'missing';
    return `${source(p)}; provider FCF ${show(h!.cash[i])}, CFO ${show(h!.operatingCash[i])}, EBIT ${show(h!.ebit[i])}, revenue ${show(h!.revenue[i])} ${p.currency} m`;
  });
  return `${NORMAL_YEAR_LABEL}. Exclusions use the calendar year of each actual fiscal period end; a fiscal period ending in 2024 can include activity during 2023. Selected: ${selection.periods.map(source).join(' | ') || 'none'}. Excluded raw observations, retained for stress review: ${skipped.join(' | ') || 'none'}. Screening snapshot ${h?.asOf ?? 'unavailable'}.${selection.reason ? ` ${selection.reason}` : ''}`;
}

export type NormalYearValuation = { normalCash: number | null; cashPV: number | null; terminalPV: number | null; value: number | null; surplusPercent: number | null; currency: string | null; reason: string | null };
export function calculateNormalYearValuation(row: ResearchGaugeRow, terminalCreditPercent = 100): NormalYearValuation {
  return calculateNormalYearHorizon(row, 10, terminalCreditPercent);
}

/** Five annual cash payments only; the normal five-report historical selection
 * is unchanged. A zero terminal excludes all value after year five rather than
 * asserting that a continuing business has no remaining economic value.
 */
export function calculateNormalYearFiveYearValuation(row: ResearchGaugeRow): NormalYearValuation {
  return calculateNormalYearHorizon(row, 5, 0);
}

function calculateNormalYearHorizon(row: ResearchGaugeRow, years: 5 | 10, terminalCreditPercent: number): NormalYearValuation {
  const selection = selectNormalYearPeriods(row);
  const unavailable = (reason: string): NormalYearValuation => ({ normalCash: null, cashPV: null, terminalPV: null, value: null, surplusPercent: null, currency: selection.currency, reason });
  if (![0, 50, 100].includes(terminalCreditPercent)) return unavailable('Only explicit 100%, 50% and 0% terminal-credit sensitivities are supported.');
  if (row.route !== 'operating' || row.classificationConflict || row.presence !== 'latest') return unavailable('Normal-year screening valuation requires an operating company in the latest directory with no classification conflict.');
  if (selection.reason) return unavailable(selection.reason);
  const cash = normalYearValues(row, 'fcf', selection);
  if (!cash || cash.some(value => !finite(value))) return unavailable('Missing provider FCF in one or more of the five selected normal reports; older observations are not substituted.');
  const normalCash = [...cash as number[]].sort((a, b) => a - b)[2];
  const cashPV = Array.from({ length: years }, (_, i) => normalCash / 1.1 ** (i + 1)).reduce((sum, cash) => sum + cash, 0);
  const terminalPV = years === 5 ? 0 : (normalCash / .1) / 1.1 ** 10;
  const value = cashPV + terminalPV * terminalCreditPercent / 100;
  if (![cashPV, terminalPV, value].every(finite)) return unavailable('The normal-year sensitivity exceeds the finite numerical range.');
  const unpriced = (reason: string): NormalYearValuation => ({ normalCash, cashPV, terminalPV, value, surplusPercent: null, currency: selection.currency, reason });
  const v = row.valuation, basis = v.priceBasis;
  if (v.currency !== selection.currency) return unpriced('The saved equity-price currency does not match the normal annual cash currency; no new conversion is inferred.');
  if (!finite(v.candidateEquity) || v.candidateEquity <= 0 || !basis || !validDay(v.priceDate) || !validDay(basis.sourceAsOf) || v.priceDate > basis.sourceAsOf || basis.sourceAsOf > row.screeningAnnual!.asOf || !basis.sourceId?.trim()) return unpriced('A positive, dated and source-bound saved equity price is required.');
  if (days(row.screeningAnnual!.asOf, v.priceDate) > 550) return unpriced('The saved quote is older than 550 days at the screening snapshot; normal cash values remain available without a price comparison.');
  if (!finite(basis.shares) || basis.shares <= 0 || !finite(basis.close) || basis.close <= 0) return unpriced('The saved price requires finite positive reported shares and a close; no replacement share basis is inferred.');
  let sameCurrencyPrice = basis.shares * basis.close;
  if (basis.method === 'local') {
    if (basis.currency !== selection.currency) return unpriced('The local saved quote currency does not match the normal annual cash currency.');
  } else if (basis.method === 'sek') {
    if (selection.currency !== 'SEK' || !finite(basis.fxRate) || basis.fxRate <= 0 || !validDay(basis.fxDate) || basis.fxDate > v.priceDate) return unpriced('The saved SEK conversion is unavailable or incompatible with normal annual cash.');
    sameCurrencyPrice *= basis.fxRate;
  } else return unpriced('The saved equity-price conversion method is unsupported.');
  if (!finite(sameCurrencyPrice) || Math.abs(sameCurrencyPrice - v.candidateEquity) > 1e-9 * Math.max(1, Math.abs(sameCurrencyPrice), Math.abs(v.candidateEquity))) return unpriced('The saved equity price does not reconcile with its recorded shares, quote and conversion.');
  const surplusPercent = 100 * (value / v.candidateEquity - 1);
  if (!finite(surplusPercent)) return unpriced('The normal-year price comparison exceeds the finite numerical range.');
  return { normalCash, cashPV, terminalPV, value, surplusPercent, currency: selection.currency, reason: null };
}
