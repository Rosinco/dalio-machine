import holmen from '../research/valuations/holmen-2026-09-11.json';
import { blankValuation, scenarioKeys, type ScenarioKey, type ValuationDraft } from './valuation';

export type ResearchedStudy = typeof holmen;
export const holmenStudy: ResearchedStudy = holmen;
// Additional deep dives join this registry only after their numerical assumptions
// and source bridge have been reviewed. Folder existence is not a valuation.
const studies: ResearchedStudy[] = [holmenStudy];
export function researchedStudyFor(entry: { id: string; isin: string | null }, sources: { sha256: string }[]): ResearchedStudy | null {
  return studies.find(s => s.company === entry.id && s.isin === entry.isin && sources.some(source => source.sha256 === s.deepDive.sha256)) ?? null;
}

export function shouldStartResearchedStudy(d: ValuationDraft | null): boolean {
  if (!d) return true;
  if (d.researchOrigin || d.researchAutofillDisabled || Object.values(d.notes).some(v => v.trim()) || Object.values(d.capital).some(v => v !== null)) return false;
  return scenarioKeys.every(k => {
    const s = d.scenarios[k];
    return s.discountRate === null && s.cashFlows.every(v => v === null) && s.terminalEquity === 0 && s.recoveryEquity === null && s.recoveryYear === null && !s.rationale.trim();
  });
}

export function buildResearchedValuation(study: ResearchedStudy) {
  const { cashHistory: h, capital: c, assumptions: a } = study;
  const trailing = (key: keyof typeof h.annual2025) => h.annual2025[key] + h.half2026[key] - h.half2025[key];
  const bridge = {
    operatingCash: trailing('operatingCash'), capex: trailing('capex'), leasePrincipal: trailing('leasePrincipal'), cashTaxes: trailing('cashTaxes'),
    normalizedAvailableCash: trailing('operatingCash') - trailing('capex') - trailing('leasePrincipal') + trailing('cashTaxes') - a.normalizedAnnualCashTax,
  };
  const draft = blankValuation(study.name, 'SEK', study.asOf);
  draft.title = `${study.name} — researched value and price`;
  draft.years = a.years;
  draft.marketCap = study.price.close * study.price.sharesM;
  draft.priceDate = study.price.date;
  draft.priceSource = `B-equivalent common-equity basis: ${study.price.close} SEK B-share close on ${study.price.date} × ${study.price.sharesM} million outstanding A+B shares, excluding treasury shares (published ${study.price.sharesDate}). Assumes equal economic rights and unchanged outstanding shares since that report. This is not the sum of separately priced A and B market capitalizations. Saved Börsdata download ${study.sources.find(s => s.id === 'prices')!.date}; no FX conversion. Price and share sources are in the researched study ${study.id}.`;
  draft.researchOrigin = { id: study.id, asOf: study.asOf };
  draft.notes = { ...study.notes };
  draft.capital = {
    tangibleEquity: c.equity - c.intangibles,
    averageTCE: ((c.equity2024 + c.netDebt2024 - c.intangibles2024) + (c.equity2025 + c.netDebt2025 - c.intangibles2025)) / 2,
    nopat: (c.ebit2025 - c.biologicalGain2025) * (1 - a.normalizedOperatingTaxPercent / 100),
    grossDebt: c.borrowingLong + c.borrowingShort + c.leaseDebtLong + c.leaseDebtShort,
    surplusCash: a.surplusCash,
  };
  const recovery = {} as Record<ScenarioKey, { gross: number; claims: number; costs: number; net: number }>;
  for (const key of scenarioKeys) {
    const s = study.scenarios[key], cashFlows = [s.firstPayment];
    for (let year = 2; year <= a.years; year++) cashFlows.push(cashFlows.at(-1)! * (1 + (year <= a.growthPhaseYears ? s.growth : s.matureGrowth) / 100));
    const gross = study.recovery.assets.reduce((sum, asset) => sum + asset.book * asset.rates[key], 0);
    recovery[key] = { gross, claims: study.recovery.claims, costs: s.recoveryCosts, net: Math.max(0, gross - study.recovery.claims - s.recoveryCosts) };
    if (s.matureGrowth >= a.requiredReturn) throw new Error('Research terminal growth must be below the required return.');
    const terminalEquity = cashFlows.at(-1)! * (1 + s.matureGrowth / 100) / ((a.requiredReturn - s.matureGrowth) / 100) * (1 - a.saleCostPercent / 100);
    draft.scenarios[key] = { cashFlows, discountRate: a.requiredReturn, terminalEquity, recoveryEquity: recovery[key].net, recoveryYear: s.recoveryYear,
      rationale: `${s.rationale}\n${a.ownership}\nRequired return: ${a.requiredReturn}%. Final sale at year ${a.years}: next-year dividend / (required return - ${s.matureGrowth}% mature growth), less ${a.saleCostPercent}% assumed selling costs. This is already equity value. Research version ${study.id}.` };
  }
  return { draft, bridge, recovery };
}
