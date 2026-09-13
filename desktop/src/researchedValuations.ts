import { blankValuation, scenarioKeys, type ScenarioKey, type ValuationDraft } from './valuation';
import { adaptHolmenStudy, holmenStudy, type HolmenBridge, type LegacyHolmenStudy } from './legacyHolmenStudy';
import { reviewedStudy, type ReviewedStudy } from './researchedStudy';
import sca from '../research/valuations/sca-2026-09-12.json';

export { holmenStudy, reviewedStudy };
export type { ReviewedStudy } from './researchedStudy';
export type ResearchedStudy = LegacyHolmenStudy | ReviewedStudy;
export const scaStudy = reviewedStudy(sca);
// Additional deep dives join this registry only after their numerical assumptions
// and source bridge have been reviewed. Folder existence is not a valuation.
const studies: ResearchedStudy[] = [holmenStudy, scaStudy];
export function researchedStudyFor(entry: { id: string; isin: string | null }, sources: { sha256: string; path?: string }[], candidates = studies): ResearchedStudy | null {
  return candidates.find(s => s.company === entry.id && s.isin === entry.isin && sources.some(source => source.sha256 === s.deepDive.sha256 && (s.version === 1 || source.path === s.deepDive.path))) ?? null;
}

export function shouldStartResearchedStudy(d: ValuationDraft | null): boolean {
  if (!d) return true;
  if (d.researchOrigin || d.starterOrigin || d.crisis || d.purchaseRange || d.researchAutofillDisabled || Object.values(d.notes).some(v => v.trim()) || Object.values(d.capital).some(v => v !== null)) return false;
  return scenarioKeys.every(k => {
    const s = d.scenarios[k];
    return !s.terminalCash && s.discountRate === null && s.cashFlows.every(v => v === null) && s.terminalEquity === 0 && s.recoveryEquity === null && s.recoveryYear === null && !s.rationale.trim();
  });
}

export function normalizeResearchedStudy(study: ResearchedStudy): ReviewedStudy {
  return study.version === 1 ? adaptHolmenStudy(study).study : reviewedStudy(study);
}

type RecoveryAmounts = { gross: number | null; claims: number | null; costs: number | null; cashBurn: number | null; net: number | null; shortfall: number | null };
type BuiltResearchedValuation = { draft: ValuationDraft; study: ReviewedStudy; evidence: ReviewedStudy['evidence']; recovery: Record<ScenarioKey, RecoveryAmounts> | null; bridge?: HolmenBridge };
type LegacyBuiltValuation = BuiltResearchedValuation & { bridge: HolmenBridge; recovery: Record<ScenarioKey, { [K in keyof RecoveryAmounts]: number }> };

export function buildResearchedValuation(study: LegacyHolmenStudy): LegacyBuiltValuation;
export function buildResearchedValuation(study: ResearchedStudy): BuiltResearchedValuation;
export function buildResearchedValuation(input: ResearchedStudy): BuiltResearchedValuation {
  const legacy = input.version === 1 ? adaptHolmenStudy(input) : null;
  const study = legacy?.study ?? reviewedStudy(input);
  const draft = blankValuation(study.name, study.currency, study.asOf);
  draft.title = `${study.name} — researched value and price`;
  draft.years = study.years;
  draft.marketCap = study.price.marketCap;
  draft.priceDate = study.price.date;
  draft.priceSource = study.price.narrative;
  draft.researchOrigin = { id: study.id, asOf: study.asOf };
  draft.notes = { ...study.notes };
  draft.capital = { ...study.capital };
  const recovery = study.recovery.status === 'available' ? {} as Record<ScenarioKey, RecoveryAmounts> : null;
  for (const key of scenarioKeys) {
    let net: number | null = null, year: number | null = null;
    if (study.recovery.status === 'available' && recovery) {
      const s = study.recovery.scenarios[key], proceeds = study.recovery.assets.map(asset => asset.proceeds[key]);
      const gross = proceeds.some(value => value === null) ? null : (proceeds as number[]).reduce((sum, value) => sum + value, 0);
      const remaining = gross === null || s.claims === null || s.costs === null || s.cashBurn === null ? null : gross - s.claims - s.costs - s.cashBurn;
      net = remaining === null ? null : Math.max(0, remaining);
      year = s.year;
      recovery[key] = { gross, claims: s.claims, costs: s.costs, cashBurn: s.cashBurn, net, shortfall: remaining === null ? null : Math.max(0, -remaining) };
    }
    const s = study.scenarios[key];
    draft.scenarios[key] = {
      cashFlows: [...s.cashFlows], discountRate: s.discountRate, terminalEquity: s.terminalEquity, recoveryEquity: net, recoveryYear: year,
      rationale: legacy?.rationales[key] ?? `${s.rationale}\n${study.ownership}\nResearch version ${study.id}.`,
    };
  }
  return { draft, study, evidence: study.evidence, recovery, ...(legacy ? { bridge: legacy.bridge } : {}) };
}
