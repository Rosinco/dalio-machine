import { capitalLabels, scenarioKeys, validAmount, validDay, type ValuationDraft } from './valuation';

export type SavedValuation = { format: 'macro-atlas-valuation'; version: 1; id: string; created: string; company: string; release: string; financial: string | null; taxonomy: string | null; draft: ValuationDraft };
type Storage = Pick<globalThis.Storage, 'getItem' | 'setItem'>;
const key = 'macro-atlas-valuations-v1';
const check = (ok: unknown) => { if (!ok) throw new Error('Invalid or unsupported valuation study. Existing saved work has been retained.'); };
const hash = (x: unknown) => typeof x === 'string' && /^[a-f0-9]{64}$/.test(x);
const text = (x: unknown, max: number) => typeof x === 'string' && x.length <= max;
const obj = (x: any) => x !== null && typeof x === 'object' && !Array.isArray(x);
const amount = (x: unknown) => x === null || validAmount(x);
export function validateValuation(v: any): asserts v is SavedValuation {
  check(obj(v) && v.format === 'macro-atlas-valuation' && v.version === 1);
  check(typeof v.id === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(v.id) && typeof v.company === 'string' && /^[1-9][0-9]{0,9}$/.test(v.company));
  check(typeof v.created === 'string' && /^\d{4}-\d{2}-\d{2}T.+Z$/.test(v.created) && Number.isFinite(Date.parse(v.created)));
  check(hash(v.release) && (v.financial === null || hash(v.financial)) && (v.taxonomy === null || hash(v.taxonomy)));
  const d = v.draft;
  if (d?.purchaseRange !== undefined) {
    const p = d.purchaseRange;
    check(obj(p) && amount(p.marginOfSafetyPercent) && scenarioKeys.includes(p.referenceScenario) && ['equity', 'share'].includes(p.unit));
    check(p.candidateEquity === undefined || amount(p.candidateEquity));
    if (p.shareBasis !== undefined) check(obj(p.shareBasis) && amount(p.shareBasis.sharesMillions) && text(p.shareBasis.date, 10) && text(p.shareBasis.source, 5000) && text(p.shareBasis.currency, 3));
  }
  check(d?.researchOrigin === undefined || obj(d.researchOrigin) && text(d.researchOrigin.id, 100) && /^[a-zA-Z0-9-]+$/.test(d.researchOrigin.id) && validDay(d.researchOrigin.asOf));
  if (d?.starterOrigin !== undefined) {
    const origin = d.starterOrigin;
    check(obj(origin) && ['weighted-cash-starter-v1', 'weighted-cash-starter-v2', 'empirical-cash-starter-v3'].includes(origin.id) && validDay(origin.asOf));
    const historyYears = origin.id === 'weighted-cash-starter-v1' ? 5 : origin.historyYears;
    check([5, 10].includes(historyYears) && Array.isArray(origin.weights) && origin.weights.length === historyYears && origin.weights.every((n: unknown) => validAmount(n) && n >= 0 && n <= 100)
      && Math.abs(origin.weights.reduce((sum: number, n: number) => sum + n, 0) - 100) <= 1e-6
      && validAmount(origin.spreadPercent) && origin.spreadPercent >= 0 && origin.spreadPercent <= 100);
    if (origin.id === 'weighted-cash-starter-v2') check(['trend', 'flat'].includes(origin.projection) && validAmount(origin.spreadStepPercent) && origin.spreadStepPercent >= 0 && origin.spreadStepPercent <= 100);
    if (origin.id === 'empirical-cash-starter-v3') check(['latest', 'trend', 'flat'].includes(origin.projection) && ['historical', 'percentage'].includes(origin.rangeMode)
      && validAmount(origin.spreadStepPercent) && origin.spreadStepPercent >= 0 && origin.spreadStepPercent <= 100
      && validAmount(origin.tailWideningPercent) && origin.tailWideningPercent >= 0 && origin.tailWideningPercent <= 100
      && text(origin.calibrationId, 150) && /^[a-zA-Z0-9-]+$/.test(origin.calibrationId));
    check(origin.terminalMethod === undefined || origin.id === 'empirical-cash-starter-v3' && origin.terminalMethod === 'historical-median-v1');
  }
  if (d?.crisis !== undefined) {
    const c = d.crisis;
    check(obj(c) && typeof c.enabled === 'boolean' && text(c.rationale, 5000));
    for (const k of ['shockPercent', 'startYear', 'durationYears', 'recoveryYears', 'extraAnnualCashCost', 'discountRate', 'terminalEquity']) check(amount(c[k]));
    check(c.shockPercent === null || c.shockPercent >= 0 && c.shockPercent <= 300);
    for (const k of ['startYear', 'durationYears', 'recoveryYears']) check(c[k] === null || Number.isInteger(c[k]) && c[k] >= (k === 'recoveryYears' ? 0 : 1) && c[k] <= 50);
    check(c.extraAnnualCashCost === null || c.extraAnnualCashCost >= 0);
    check(c.discountRate === null || c.discountRate >= 0 && c.discountRate <= 100);
    check(c.terminalEquity === null || c.terminalEquity >= 0);
  }
  check(!d?.researchOrigin || !d?.starterOrigin);
  check(d?.researchAutofillDisabled === undefined || typeof d.researchAutofillDisabled === 'boolean');
  check(obj(d) && text(d.title, 160) && d.title.trim() && typeof d.currency === 'string' && /^[A-Z]{3}$/.test(d.currency));
  check(validDay(d.valuationDate) && (d.priceDate === '' || validDay(d.priceDate)) && text(d.priceSource, 5000));
  check(amount(d.marketCap) && amount(d.investment) && Number.isInteger(d.years) && d.years >= 1 && d.years <= 50);
  check(obj(d.scenarios) && Object.keys(d.scenarios).length === 3);
  for (const key of scenarioKeys) {
    const s = d.scenarios[key];
    check(obj(s) && Array.isArray(s.cashFlows) && s.cashFlows.length >= d.years && s.cashFlows.length <= 50 && s.cashFlows.every(amount));
    check(amount(s.discountRate) && amount(s.terminalEquity) && amount(s.recoveryEquity));
    if (s.terminalCash !== undefined) check(obj(s.terminalCash) && Object.keys(s.terminalCash).length === 2 && amount(s.terminalCash.cashFlow) && amount(s.terminalCash.growthRate));
    check(s.recoveryYear === null || Number.isInteger(s.recoveryYear) && s.recoveryYear >= 0 && s.recoveryYear <= 50);
    check(text(s.rationale, 5000));
  }
  check(obj(d.capital) && Object.keys(d.capital).length === Object.keys(capitalLabels).length && Object.keys(capitalLabels).every(k => amount(d.capital[k])));
  check(obj(d.notes) && Object.keys(d.notes).length === 5 && ['business', 'macro', 'financing', 'recovery', 'decision'].every(k => text(d.notes[k], 10000)));
}
export function decodeValuation(raw: string): SavedValuation {
  check(raw.length <= 150_000); const v = JSON.parse(raw); validateValuation(v); return v;
}
export function loadValuations(storage: Storage): SavedValuation[] {
  const raw = storage.getItem(key); if (raw === null) return [];
  check(raw.length <= 4_000_000); const data = JSON.parse(raw);
  check(data?.version === 1 && Array.isArray(data.items) && data.items.length <= 100);
  data.items.forEach(validateValuation); check(new Set(data.items.map((v: SavedValuation) => v.id)).size === data.items.length);
  return data.items;
}
export function saveValuation(storage: Storage, study: SavedValuation) {
  validateValuation(study); const existing = loadValuations(storage);
  check(existing.length < 100 && !existing.some(v => v.id === study.id));
  const raw = JSON.stringify({ version: 1, items: [study, ...existing] }); check(raw.length <= 4_000_000);
  storage.setItem(key, raw);
}
export function compatibleValuation(v: SavedValuation, company: string, release: string, financial: string | null, taxonomy: string | null) {
  return v.company === company && v.release === release && v.financial === financial && v.taxonomy === taxonomy;
}
function draftKey(v: Pick<SavedValuation, 'company' | 'release' | 'financial' | 'taxonomy'>) {
  return `macro-atlas-valuation-draft-v1:${v.company}:${v.release}:${v.financial ?? 'none'}:${v.taxonomy ?? 'none'}`;
}
export function loadValuationDraft(storage: Storage, basis: Pick<SavedValuation, 'company' | 'release' | 'financial' | 'taxonomy'>) {
  const raw = storage.getItem(draftKey(basis)); if (raw === null) return null;
  const v = decodeValuation(raw);
  check(compatibleValuation(v, basis.company, basis.release, basis.financial, basis.taxonomy));
  return v;
}
export function saveValuationDraft(storage: Storage, v: SavedValuation) {
  validateValuation(v); loadValuationDraft(storage, v);
  const raw = JSON.stringify(v); check(raw.length <= 150_000); storage.setItem(draftKey(v), raw);
}
