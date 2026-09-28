/** Inputs come from the fixed normal-year quality-watch screen, before display
 * filters or valuation thresholds. `discount` is its five-year cash-only NPV / saved
 * price observation; ranking it is not a claim that NPV / price is discount %.
 * Every distinct listing is one observation, including multiple listing venues.
 */
export type NormalQualityValueObservation = {
  id: string;
  eligible: boolean;
  discount: number | null;
  roce: number | null;
  marginFloor: number | null;
  cfoGrowth: number | null;
  netDebtEbitda: number | null;
};

export type NormalQualityValueComponents = {
  roce: number;
  marginFloor: number;
  cfoGrowth: number;
  netDebtEbitda: number;
};

export type NormalQualityValueRank = {
  score: number | null;
  quality: number | null;
  discountRank: number | null;
  components: NormalQualityValueComponents | null;
  reason: string | null;
};

export type NormalQualityValueRanking = {
  byId: Map<string, NormalQualityValueRank>;
  cohortSize: number;
};

const metrics = ['discount', 'roce', 'marginFloor', 'cfoGrowth', 'netDebtEbitda'] as const;
type Metric = typeof metrics[number];
type CompleteObservation = Omit<NormalQualityValueObservation, Metric> & Record<Metric, number>;
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);
const unavailable = (reason: string): NormalQualityValueRank => ({ score: null, quality: null, discountRank: null, components: null, reason });

/** Doubled worst-to-best midrank positions, kept as exact integers until the
 * final division so equal weighted ranks remain exact ties. Values are never
 * winsorized or made positive.
 */
function doubledMidranks(observations: readonly CompleteObservation[], metric: Metric): Map<string, number> {
  const ascending = metric !== 'netDebtEbitda';
  const ordered = observations.map(row => ({ id: row.id, value: metric === 'netDebtEbitda' ? Math.max(0, row[metric]) : row[metric] }))
    .sort((a, b) => ascending ? a.value - b.value : b.value - a.value);
  const ranks = new Map<string, number>();
  for (let start = 0; start < ordered.length;) {
    let end = start + 1;
    while (end < ordered.length && ordered[end].value === ordered[start].value) end++;
    const rank = 2 * start + end - start - 1;
    for (let i = start; i < end; i++) ranks.set(ordered[i].id, rank);
    start = end;
  }
  return ranks;
}

/** Explicit research ordering: 60% five-year NPV percentile + 40% the mean of four
 * quality percentiles. It is not a probability, expected return or fair value.
 * Missing components exclude the listing; weights are never redistributed.
 * The caller must pass the whole fixed reference cohort, not displayed rows.
 */
export function rankNormalQualityValue(observations: readonly NormalQualityValueObservation[]): NormalQualityValueRanking {
  const byId = new Map<string, NormalQualityValueRank>();
  const ids = new Set<string>();
  let duplicate = false;
  for (const row of observations) {
    if (ids.has(row.id)) duplicate = true;
    ids.add(row.id);
  }
  if (duplicate) {
    for (const row of observations) byId.set(row.id, unavailable('Duplicate listing IDs make the reference cohort ambiguous; ranking is withheld for every listing.'));
    return { byId, cohortSize: 0 };
  }

  const reference: CompleteObservation[] = [];
  for (const row of observations) {
    if (!row.eligible) {
      byId.set(row.id, unavailable('Outside the fixed normal-year quality reference cohort; display filters do not change ranking eligibility.'));
      continue;
    }
    const missing = metrics.filter(metric => !finite(row[metric]));
    if (missing.length) {
      byId.set(row.id, unavailable(`Missing or non-finite ranking inputs: ${missing.join(', ')}. No component is imputed or reweighted.`));
      continue;
    }
    reference.push(row as CompleteObservation);
  }

  const ranks = Object.fromEntries(metrics.map(metric => [metric, doubledMidranks(reference, metric)])) as Record<Metric, Map<string, number>>;
  const percentile = (position: number) => reference.length === 1 ? 50 : 100 * position / (2 * (reference.length - 1));
  for (const row of reference) {
    const positions = {
      roce: ranks.roce.get(row.id)!, marginFloor: ranks.marginFloor.get(row.id)!,
      cfoGrowth: ranks.cfoGrowth.get(row.id)!, netDebtEbitda: ranks.netDebtEbitda.get(row.id)!,
    };
    const components: NormalQualityValueComponents = {
      roce: percentile(positions.roce),
      marginFloor: percentile(positions.marginFloor),
      cfoGrowth: percentile(positions.cfoGrowth),
      netDebtEbitda: percentile(positions.netDebtEbitda),
    };
    const qualityPositionSum = positions.roce + positions.marginFloor + positions.cfoGrowth + positions.netDebtEbitda;
    const discountPosition = ranks.discount.get(row.id)!;
    const quality = reference.length === 1 ? 50 : 100 * qualityPositionSum / (8 * (reference.length - 1));
    const discountRank = percentile(discountPosition);
    const score = reference.length === 1 ? 50 : 100 * (6 * discountPosition + qualityPositionSum) / (20 * (reference.length - 1));
    byId.set(row.id, { score, quality, discountRank, components, reason: null });
  }
  return { byId, cohortSize: reference.length };
}
