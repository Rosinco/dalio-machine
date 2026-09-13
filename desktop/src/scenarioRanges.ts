import { scenarioKeys, validAmount, type ScenarioResult, type ValuationDraft } from './valuation';

export type ValuationRangeRow = { year: number; low: number; mid: number; high: number; min: number; max: number };
export type ScenarioRangePoint = [number, number];
export type ScenarioRangeSpan = { upper: ScenarioRangePoint[]; lower: ScenarioRangePoint[] };
const finite = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value);

/** Scale existing scenario calculations to the investment without mixing their cash paths or return assumptions. */
export function buildValuationRangeRows(draft: ValuationDraft, results: Record<string, ScenarioResult>, kind: 'dcf' | 'npv', sale = false): ValuationRangeRow[] | null {
  if (!Number.isInteger(draft.years) || draft.years < 1 || draft.years > 50
    || !validAmount(draft.marketCap) || draft.marketCap < 1e-9
    || !validAmount(draft.investment) || draft.investment < 1e-9) return null;
  const scale = draft.investment / draft.marketCap;
  if (!finite(scale) || scale <= 0) return null;
  const cumulative = kind === 'npv', count = draft.years + (cumulative ? 1 : 0);
  const paths: number[][] = [];
  for (const key of scenarioKeys) {
    const result = results[key];
    if (!result || result.error !== null) return null;
    const values = cumulative ? sale ? result.cumulativeNPVWithSale : result.cumulativeNPV : result.discounted;
    if (!Array.isArray(values) || values.length !== count) return null;
    paths.push(values);
  }
  const rows: ValuationRangeRow[] = [];
  for (let index = 0; index < count; index++) {
    if (!paths.every(path => finite(path[index]))) return null;
    const values = paths.map(path => path[index] * scale);
    if (!values.every(finite)) return null;
    const [low, mid, high] = values;
    rows.push({ year: index + (cumulative ? 0 : 1), low, mid, high, min: Math.min(...values), max: Math.max(...values) });
  }
  return rows;
}

function interpolate(left: number, right: number, fraction: number): number {
  if (fraction === 0) return left;
  if (fraction === 1) return right;
  const delta = right - left;
  return finite(delta) ? left + delta * fraction : left * (1 - fraction) + right * fraction;
}

/** Strictly interior crossing; normalize differences to avoid overflow with large opposite signs. */
function crossing(leftA: number, rightA: number, leftB: number, rightB: number): number | null {
  let left = leftA - leftB, right = rightA - rightB;
  if (!finite(left) || !finite(right)) {
    const scale = Math.max(Math.abs(leftA), Math.abs(rightA), Math.abs(leftB), Math.abs(rightB));
    left = leftA / scale - leftB / scale;
    right = rightA / scale - rightB / scale;
  }
  if (!(left < 0 && right > 0 || left > 0 && right < 0)) return null;
  const scale = Math.max(Math.abs(left), Math.abs(right));
  const fromLeft = Math.abs(left) / scale, fromRight = Math.abs(right) / scale;
  const fraction = fromLeft / (fromLeft + fromRight);
  return fraction > 0 && fraction < 1 ? fraction : null;
}

/** Exact envelope of the displayed linear or end-of-period step paths; incomplete years split the geometry. */
export function scenarioRangeGeometry(paths: readonly (readonly (number | null)[])[], step = false): ScenarioRangeSpan[] {
  if (!paths.length) return [];
  const count = Math.min(...paths.map(path => path.length));
  const spans: ScenarioRangeSpan[] = [];
  let span: ScenarioRangeSpan | null = null;
  const append = (target: ScenarioRangeSpan, x: number, values: number[]) => {
    target.upper.push([x, Math.max(...values)]);
    target.lower.push([x, Math.min(...values)]);
  };
  for (let index = 0; index < count; index++) {
    const current = paths.map(path => path[index]);
    if (!current.every(finite)) { span = null; continue; }
    if (!span) {
      span = { upper: [], lower: [] }; spans.push(span);
      append(span, index, current);
      continue;
    }
    const previous = paths.map(path => path[index - 1] as number);
    if (step) {
      append(span, index, previous);
      append(span, index, current);
      continue;
    }
    const fractions = new Set<number>([1]);
    for (let a = 0; a < paths.length; a++) for (let b = a + 1; b < paths.length; b++) {
      const fraction = crossing(previous[a], current[a], previous[b], current[b]);
      if (fraction !== null) fractions.add(fraction);
    }
    for (const fraction of [...fractions].sort((a, b) => a - b)) {
      append(span, index - 1 + fraction, previous.map((value, path) => interpolate(value, current[path], fraction)));
    }
  }
  return spans;
}
