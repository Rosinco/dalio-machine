import type { CompanyListColumn } from './companyListModel';
import { NORMAL_YEAR_WINDOW } from './normalYearScreen';

export const NORMAL_RANKING_POLICY_ID = 'normal-quality-watch-npv5-60-40-v2';
export const NORMAL_RANKING_KPIS = new Set(['normal_quality_value_score', 'normal_discount_rank', 'normal_quality_rank']);
export const NORMAL_RANKING_DESCRIPTION = '60% five-year cash-only NPV points + 40% quality points. NPV uses only years 1–5 at a 10% discount rate, with no terminal value, relative to the dated saved equity price. Quality is the equal mean of percentile ranks for median pre-tax capital return, minimum EBIT margin, operating cash-flow CAGR and lower net debt / EBITDA; each contributes 10% of the total. Net cash receives the same debt rank as zero debt. Relative listing ranks, not a fair-value adjustment, expected return, probability or proof of pricing power. Missing inputs stay unranked.';
export const NORMAL_RANKING_REFERENCE = 'Fixed reference: latest operating listings with five comparable normal reports outside 2020–2023; capital-return median >=20% and minimum >=10%; tangible-asset return median >=8%; EBIT margin median >=12% and minimum >=8%; revenue CAGR >=3%; CFO CAGR >=0%; median tangible assets/revenue 0–0.5; five positive FCF reports; latest net debt/EBITDA <=1.5 and EBITDA margin >0; valid normal-year price comparison. Searches, display filters and valuation bounds do not change this reference. Multiple listings of a business can affect percentiles.';
export const NORMAL_RANKING_DEPENDENCIES: CompanyListColumn[] = [
  { id: 'rank-debt', kpiId: 'provider_42', window: 'provider:screener:last', calculation: 'provider:latest' },
  { id: 'rank-ebitda-margin', kpiId: 'provider_32', window: 'provider:screener:last', calculation: 'provider:latest' },
];
export const NORMAL_RANKING_QUALITY_BOUNDS: { key: string; column: CompanyListColumn; min?: number; max?: number }[] = [
  ['roce', 'normal_roce', 'median', 20],
  ['roceFloor', 'normal_roce', 'min', 10],
  ['rota', 'normal_rota', 'median', 8],
  ['margin', 'ebit_margin', 'median', 12],
  ['marginFloor', 'ebit_margin', 'min', 8],
  ['revenueGrowth', 'revenue', 'growth', 3],
  ['cfoGrowth', 'cfo', 'growth', 0],
  ['intensity', 'tangible_assets_revenue', 'median', 0, .5],
  ['positiveCash', 'positive_fcf', 'latest', 5],
].map(([key, kpiId, calculation, min, max]) => ({ key: String(key), column: { id: `rank-${key}`, kpiId: String(kpiId), window: NORMAL_YEAR_WINDOW, calculation: calculation as CompanyListColumn['calculation'] }, min: min as number, max: max as number | undefined }));
