export type ResilienceBasis = { companyId: string; companyName: string; releaseId: string; financialId: string | null; sourceAsOf: string };
export const reviewConclusions = ['unassessed', 'unresolved', 'material-failure', 'reviewed-for-stated-scenario'] as const;
export type ReviewConclusion = typeof reviewConclusions[number];
export type LiquidityRow = { operatingCash: number | null; principalDue: number | null; committedFunding: number | null; shockDrain: number | null };
export type LiquidityAssumptions = { frequency: 'quarterly' | 'yearly'; startDate: string; openingCash: number | null; minimumCash: number | null; rows: LiquidityRow[] };
export type ImpairmentAssumptions = { annualCash: string; discountRate: number | null; finalProceeds: number | null; ownership: string; assumptions: string; evidenceDate: string; evidenceSources: string; purchaseEquity: number | null; priceDate: string; priceSource: string };
export type ResilienceReview = ResilienceBasis & { format: 'macro-atlas-resilience-review'; version: 1; updatedAt: string; currency: string; evidenceDate: string; evidenceSources: string; shockNarrative: string; covenantConstraints: string; conclusion: ReviewConclusion; conclusionRationale: string; liquidity: LiquidityAssumptions; impairment: ImpairmentAssumptions };
export type ReviewStatus = { code: ReviewConclusion | 'stale'; label: string; detail: string; stale: boolean; reviewed: boolean };
type ReadStorage = Pick<Storage, 'getItem'>;
type WriteStorage = Pick<Storage, 'getItem' | 'setItem'>;
const finite = (n: unknown): n is number => typeof n === 'number' && Number.isFinite(n) && Math.abs(n) <= 1e15;
const nonnegative = (n: unknown): n is number => finite(n) && n >= 0;
const present = (s: string) => s.trim().length > 0;
export const validReviewDate = (s: string) => /^\d{4}-\d{2}-\d{2}$/.test(s) && Number.isFinite(Date.parse(s)) && new Date(s).toISOString().slice(0, 10) === s;
export const blankLiquidityRow = (): LiquidityRow => ({ operatingCash: null, principalDue: null, committedFunding: null, shockDrain: null });
export function blankResilienceReview(basis: ResilienceBasis): ResilienceReview {
  return { ...basis, format: 'macro-atlas-resilience-review', version: 1, updatedAt: '', currency: '', evidenceDate: '', evidenceSources: '', shockNarrative: '', covenantConstraints: '', conclusion: 'unassessed', conclusionRationale: '',
    liquidity: { frequency: 'quarterly', startDate: '', openingCash: null, minimumCash: null, rows: Array.from({ length: 4 }, blankLiquidityRow) },
    impairment: { annualCash: '', discountRate: null, finalProceeds: null, ownership: '', assumptions: '', evidenceDate: '', evidenceSources: '', purchaseEquity: null, priceDate: '', priceSource: '' } };
}

export function calculateLiquidity(a: LiquidityAssumptions) {
  type Point = { period: number; netChange: number; cash: number; headroom: number };
  const fail = (error: string) => ({ error, rows: [] as Point[], firstBreach: null as number | null, additionalDrainPerPeriod: null as number | null });
  if (!validReviewDate(a.startDate)) return fail('Enter the opening date for equal-length periods.');
  if (!nonnegative(a.openingCash) || !nonnegative(a.minimumCash)) return fail('Enter unrestricted opening cash and minimum operating liquidity, including explicit zero where justified.');
  if (!a.rows.length || a.rows.length > 40) return fail('Use 1–40 equal quarterly or yearly periods.');
  if (!a.rows.every(r => finite(r.operatingCash) && nonnegative(r.principalDue) && nonnegative(r.committedFunding) && nonnegative(r.shockDrain))) return fail('Complete all four amounts for each period. Cash from operations may be negative; principal, available funding and extra drain must be nonnegative. Blanks are not zero.');
  let cash = a.openingCash, firstBreach: number | null = cash < a.minimumCash ? 0 : null;
  let additionalDrainPerPeriod = cash < a.minimumCash ? 0 : Infinity;
  const rows: Point[] = [];
  for (let i = 0; i < a.rows.length; i++) {
    const r = a.rows[i], netChange = r.operatingCash! - r.principalDue! + r.committedFunding! - r.shockDrain!;
    cash += netChange;
    const headroom = cash - a.minimumCash;
    if (![cash, netChange, headroom].every(Number.isFinite)) return fail('The liquidity calculation exceeds the supported numerical range.');
    rows.push({ period: i + 1, netChange, cash, headroom });
    if (headroom < 0 && firstBreach === null) firstBreach = i + 1;
    additionalDrainPerPeriod = Math.min(additionalDrainPerPeriod, Math.max(0, headroom / (i + 1)));
  }
  return { error: null, rows, firstBreach, additionalDrainPerPeriod };
}

export function parseAnnualCash(input: string): { cash: number[]; error: string | null } {
  const fail = () => ({ cash: [], error: 'Enter 1–50 annual amounts separated by commas or newlines. Every year needs an explicit number, including zero; use a decimal point.' });
  if (!input.trim()) return fail();
  const parts = input.replace(/\r\n/g, '\n').split(/[,\n]/);
  if (!parts.length || parts.length > 50 || parts.some(s => !/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(s.trim()))) return fail();
  const cash = parts.map(s => Number(s.trim()));
  return cash.every(finite) ? { cash, error: null } : fail();
}

export function calculateImpairment(a: ImpairmentAssumptions, asOf?: string) {
  const fail = (error: string) => ({ error, priceError: null as string | null, value: null as number | null, cashPV: null as number | null, finalPV: null as number | null, npv: null as number | null, years: 0 });
  const parsed = parseAnnualCash(a.annualCash);
  if (parsed.error) return fail(parsed.error);
  if (!nonnegative(a.discountRate) || a.discountRate > 100 || !nonnegative(a.finalProceeds)) return fail('Enter a required annual return from 0–100% and final net equity proceeds, including explicit zero.');
  const factor = 1 + a.discountRate / 100;
  const cashPV = parsed.cash.reduce((sum, cash, i) => sum + cash / factor ** (i + 1), 0);
  const finalPV = a.finalProceeds / factor ** parsed.cash.length, value = cashPV + finalPV;
  if (![cashPV, finalPV, value].every(Number.isFinite)) return fail('The impairment calculation exceeds the supported numerical range.');
  const priceError = a.purchaseEquity === null ? null : !nonnegative(a.purchaseEquity) || !validReviewDate(a.priceDate) || !present(a.priceSource) ? 'A purchase amount needs a nonnegative price for the same ownership claim, its date and its source.'
    : !asOf || !validReviewDate(asOf) ? 'Enter the opening valuation date before calculating NPV.'
      : a.priceDate > asOf ? 'The purchase price date is after the opening valuation date. Align the valuation and price dates before calculating NPV.' : null;
  return { error: null, priceError, value, cashPV, finalPV, npv: a.purchaseEquity === null || priceError ? null : value - a.purchaseEquity, years: parsed.cash.length };
}

/** Read-only status for Research and Value; a completed review is never a buy or safety verdict. */
export function getReviewStatus(record: ResilienceReview | null, basis: Pick<ResilienceBasis, 'releaseId' | 'financialId'> & { sourceAsOf?: string }): ReviewStatus {
  const result = (code: ReviewStatus['code'], label: string, detail: string): ReviewStatus => ({ code, label, detail, stale: code === 'stale', reviewed: code === 'reviewed-for-stated-scenario' });
  if (!record) return result('unassessed', 'Resilience unassessed', 'No saved company survival and permanent-loss review.');
  const liquidity = calculateLiquidity(record.liquidity), impairment = calculateImpairment(record.impairment, record.liquidity.startDate);
  const stale = record.releaseId !== basis.releaseId || record.financialId !== basis.financialId || basis.sourceAsOf !== undefined && record.sourceAsOf !== basis.sourceAsOf;
  if (stale) return result('stale', 'Review uses an earlier source basis', `Reassess the saved review against the current research and financial sources. ${record.conclusion === 'material-failure' || liquidity.firstBreach !== null ? 'The previous review still identifies a material failure' + (liquidity.firstBreach === null ? '.' : liquidity.firstBreach === 0 ? ': liquidity is below minimum at opening.' : `: liquidity breaches the minimum in period ${liquidity.firstBreach}.`) : 'Its calculations and authored conclusion remain preserved.'}`);
  if (record.conclusion === 'material-failure' || liquidity.firstBreach !== null) return result('material-failure', 'Material failure identified', liquidity.firstBreach !== null ? `Liquidity falls below the stated minimum ${liquidity.firstBreach === 0 ? 'at opening' : `in period ${liquidity.firstBreach}`}. Later inflows do not resolve that earlier breach.` : 'The authored review identifies a material failure. Resolve and reassess the assumptions before changing this conclusion.');
  const evidenceMissing = !/^[A-Z]{3}$/.test(record.currency) || !validReviewDate(record.evidenceDate) || !present(record.evidenceSources) || !present(record.shockNarrative) || !present(record.covenantConstraints) || !present(record.conclusionRationale)
    || !validReviewDate(record.impairment.evidenceDate) || !present(record.impairment.evidenceSources) || !present(record.impairment.ownership) || !present(record.impairment.assumptions);
  if (record.conclusion === 'unassessed') return result('unassessed', 'Resilience unassessed', 'Enter company evidence and assumptions, then record a conclusion for the stated scenarios.');
  if (record.conclusion === 'unresolved' || liquidity.error || impairment.error || impairment.priceError || evidenceMissing) return result('unresolved', 'Resilience review unresolved', liquidity.error ?? impairment.error ?? impairment.priceError ?? (evidenceMissing ? 'Complete dated source evidence, ownership, shock and covenant assumptions, and the conclusion rationale.' : 'The authored conclusion leaves questions unresolved.'));
  return result('reviewed-for-stated-scenario', 'Reviewed for the stated scenarios', 'The saved review has complete inputs and evidence fields, with no modeled endpoint liquidity breach. This is an authored scenario review, not an investment endorsement or a probability of survival.');
}

export const resilienceReviewKey = (companyId: string) => `macro-atlas-resilience-review-v1:${companyId}`;
const preserved = () => new Error('This company resilience record is unreadable or unsupported. Existing saved bytes are preserved; editing is disabled.');
const object = (v: unknown): v is Record<string, unknown> => !!v && typeof v === 'object' && !Array.isArray(v);
const exact = (v: unknown, keys: string[]): v is Record<string, unknown> => object(v) && Object.keys(v).length === keys.length && keys.every(k => Object.hasOwn(v, k));
const text = (v: unknown, max = 10000) => typeof v === 'string' && v.length <= max;
const amount = (v: unknown) => v === null || finite(v);
function validateRecord(v: unknown, companyId: string): asserts v is ResilienceReview {
  if (!exact(v, ['companyId', 'companyName', 'releaseId', 'financialId', 'sourceAsOf', 'format', 'version', 'updatedAt', 'currency', 'evidenceDate', 'evidenceSources', 'shockNarrative', 'covenantConstraints', 'conclusion', 'conclusionRationale', 'liquidity', 'impairment'])) throw preserved();
  if (v.format !== 'macro-atlas-resilience-review' || v.version !== 1 || v.companyId !== companyId || !text(v.companyId, 100) || !text(v.companyName, 500) || !text(v.releaseId, 500) || !(v.financialId === null || text(v.financialId, 500)) || !text(v.sourceAsOf, 30) || !text(v.updatedAt, 50)) throw preserved();
  if (!['currency', 'evidenceDate', 'evidenceSources', 'shockNarrative', 'covenantConstraints', 'conclusionRationale'].every(k => text(v[k])) || !reviewConclusions.includes(v.conclusion as ReviewConclusion)) throw preserved();
  const l = v.liquidity, a = v.impairment;
  if (!exact(l, ['frequency', 'startDate', 'openingCash', 'minimumCash', 'rows']) || !['quarterly', 'yearly'].includes(l.frequency as string) || !text(l.startDate, 10) || !amount(l.openingCash) || !amount(l.minimumCash) || !Array.isArray(l.rows) || l.rows.length < 1 || l.rows.length > 40 || !l.rows.every(r => exact(r, ['operatingCash', 'principalDue', 'committedFunding', 'shockDrain']) && Object.values(r).every(amount))) throw preserved();
  if (!exact(a, ['annualCash', 'discountRate', 'finalProceeds', 'ownership', 'assumptions', 'evidenceDate', 'evidenceSources', 'purchaseEquity', 'priceDate', 'priceSource']) || !['annualCash', 'ownership', 'assumptions', 'evidenceDate', 'evidenceSources', 'priceDate', 'priceSource'].every(k => text(a[k])) || !['discountRate', 'finalProceeds', 'purchaseEquity'].every(k => amount(a[k]))) throw preserved();
}
/** Throws for unsupported data; never writes, migrates, deletes or replaces it. */
export function loadResilienceReview(storage: ReadStorage, companyId: string): ResilienceReview | null {
  const raw = storage.getItem(resilienceReviewKey(companyId));
  if (raw === null) return null;
  try { if (raw.length > 150000) throw preserved(); const record: unknown = JSON.parse(raw); validateRecord(record, companyId); return record; }
  catch { throw preserved(); }
}
/** Explicit save, guarded against unsupported records and changes from another tab. */
export function saveResilienceReview(storage: WriteStorage, record: ResilienceReview, expectedRaw: string | null) {
  validateRecord(record, record.companyId);
  const existing = storage.getItem(resilienceReviewKey(record.companyId));
  loadResilienceReview(storage, record.companyId);
  if (existing !== expectedRaw) throw new Error('The saved resilience review changed in another view. Your edits remain on screen; reopen the company before replacing saved work.');
  const raw = JSON.stringify(record);
  if (raw.length > 150000) throw new Error('The resilience review is too large to save. Your edits remain on screen.');
  storage.setItem(resilienceReviewKey(record.companyId), raw);
  return raw;
}
