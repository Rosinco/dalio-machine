import type { ResearchGaugeRow } from './researchGaugeModel';
import type { ExpandedKpiContext } from './expandedKpis';
import { expandedVariant } from './expandedKpis';
import { companyListCell } from './companyListModel';
import { selectNormalYearPeriods } from './normalYearScreen';
import { rankNormalQualityValue } from './normalQualityValueRanking';
import type { NormalQualityValueObservation, NormalQualityValueRanking } from './normalQualityValueRanking';
import { NORMAL_RANKING_DEPENDENCIES, NORMAL_RANKING_QUALITY_BOUNDS } from './normalQualityValuePolicy';

export type NormalQualityValueContext = NormalQualityValueRanking & {
  status: 'loading' | 'error' | 'available'; reason: string | null; eligibleCount: number;
  observations: ReadonlyMap<string, NormalQualityValueObservation>;
};
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** All source rows form the reference before any user search, range, or watchlist selection. */
export function buildNormalQualityValueContext(rows: readonly ResearchGaugeRow[], expanded?: ExpandedKpiContext): NormalQualityValueContext {
  const unavailable = (status: 'loading' | 'error', reason: string): NormalQualityValueContext => ({ status, reason, eligibleCount: 0, cohortSize: 0, byId: new Map(), observations: new Map() });
  if (!expanded) return unavailable('loading', 'Opening the fixed quality-ranking reference sources.');
  if (expanded.error) return unavailable('error', expanded.error);
  for (const column of NORMAL_RANKING_DEPENDENCIES) {
    const variant = expandedVariant(column)!;
    if (expanded.errors.has(variant.id)) return unavailable('error', expanded.errors.get(variant.id)!);
  }
  if (!expanded.index || NORMAL_RANKING_DEPENDENCIES.some(column => !expanded.variants.has(expandedVariant(column)!.id))) return unavailable('loading', 'Opening all required debt and margin observations before ranking; a partial reference is not used.');
  const observations = rows.map(row => {
    const result: NormalQualityValueObservation = { id: row.id, eligible: false, discount: null, roce: null, marginFloor: null, cfoGrowth: null, netDebtEbitda: null };
    if (row.route !== 'operating' || row.classificationConflict || row.presence !== 'latest' || selectNormalYearPeriods(row).reason) return result;
    const values: Record<string, number> = {};
    for (const rule of NORMAL_RANKING_QUALITY_BOUNDS) {
      const cell = companyListCell(row, rule.column, expanded), value = cell.value;
      if (!finite(value) || rule.min !== undefined && value < rule.min || rule.max !== undefined && value > rule.max) return result;
      values[rule.key] = value;
    }
    const debt = companyListCell(row, NORMAL_RANKING_DEPENDENCIES[0], expanded).value;
    const margin = companyListCell(row, NORMAL_RANKING_DEPENDENCIES[1], expanded).value;
    if (!finite(debt) || debt > 1.5 || !finite(margin) || margin <= 0) return result;
    const discount = companyListCell(row, { id: 'rank-discount', kpiId: 'normal_npv_5y_percent', window: 'latest', calculation: 'latest' }).value;
    return { id: row.id, eligible: true, discount: finite(discount) ? discount : null, roce: values.roce, marginFloor: values.marginFloor, cfoGrowth: values.cfoGrowth, netDebtEbitda: debt };
  });
  return { ...rankNormalQualityValue(observations), status: 'available', reason: null, eligibleCount: observations.filter(row => row.eligible).length, observations: new Map(observations.map(row => [row.id, row])) };
}
