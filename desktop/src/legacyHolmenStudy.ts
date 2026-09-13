import holmen from '../research/valuations/holmen-2026-09-11.json';
import { format } from './model';
import { scenarioKeys, scenarioNames, type ScenarioKey } from './valuation';
import { type EvidenceBlock, type ReviewedStudy } from './researchedStudy';

export type LegacyHolmenStudy = Omit<typeof holmen, 'version'> & { version: 1 };
export const holmenStudy = holmen as LegacyHolmenStudy;
export type HolmenBridge = Record<'operatingCash' | 'capex' | 'leasePrincipal' | 'cashTaxes' | 'normalizedAvailableCash', number>;

// Keep the published v1 source file, arithmetic, IDs and draft strings stable.
// Only this adapter knows the source-specific accounting periods and model.
export function adaptHolmenStudy(study: LegacyHolmenStudy) {
  const { cashHistory: h, capital: c, assumptions: a } = study;
  const trailing = (key: keyof typeof h.annual2025) => h.annual2025[key] + h.half2026[key] - h.half2025[key];
  const bridge: HolmenBridge = {
    operatingCash: trailing('operatingCash'), capex: trailing('capex'), leasePrincipal: trailing('leasePrincipal'), cashTaxes: trailing('cashTaxes'),
    normalizedAvailableCash: trailing('operatingCash') - trailing('capex') - trailing('leasePrincipal') + trailing('cashTaxes') - a.normalizedAnnualCashTax,
  };
  const capital = {
    tangibleEquity: c.equity - c.intangibles,
    averageTCE: ((c.equity2024 + c.netDebt2024 - c.intangibles2024) + (c.equity2025 + c.netDebt2025 - c.intangibles2025)) / 2,
    nopat: (c.ebit2025 - c.biologicalGain2025) * (1 - a.normalizedOperatingTaxPercent / 100),
    grossDebt: c.borrowingLong + c.borrowingShort + c.leaseDebtLong + c.leaseDebtShort,
    surplusCash: a.surplusCash,
  };
  const scenarios = {} as ReviewedStudy['scenarios'], rationales = {} as Record<ScenarioKey, string>;
  for (const key of scenarioKeys) {
    const s = study.scenarios[key], cashFlows = [s.firstPayment];
    for (let year = 2; year <= a.years; year++) cashFlows.push(cashFlows.at(-1)! * (1 + (year <= a.growthPhaseYears ? s.growth : s.matureGrowth) / 100));
    if (s.matureGrowth >= a.requiredReturn) throw new Error('Research terminal growth must be below the required return.');
    const terminalEquity = cashFlows.at(-1)! * (1 + s.matureGrowth / 100) / ((a.requiredReturn - s.matureGrowth) / 100) * (1 - a.saleCostPercent / 100);
    scenarios[key] = { cashFlows, discountRate: a.requiredReturn, terminalEquity, rationale: s.rationale };
    rationales[key] = `${s.rationale}\n${a.ownership}\nRequired return: ${a.requiredReturn}%. Final sale at year ${a.years}: next-year dividend / (required return - ${s.matureGrowth}% mature growth), less ${a.saleCostPercent}% assumed selling costs. This is already equity value. Research version ${study.id}.`;
  }
  const paragraph = (classification: EvidenceBlock['classification'], text: string, sourceIds: string[] = []): EvidenceBlock => ({ kind: 'paragraph', classification, text, sourceIds });
  const historyLabels = { operatingCash: 'Operating cash after interest and paid tax', capex: 'Gross purchases of non-current assets', leasePrincipal: 'Lease principal payments', cashTaxes: 'Cash taxes paid (negative = refund)' };
  const adapted: ReviewedStudy = {
    version: 2, id: study.id, company: study.company, isin: study.isin, name: study.name, asOf: study.asOf, currency: 'SEK',
    deepDive: { ...study.deepDive }, sources: study.sources.map(source => ({ ...source })),
    price: {
      marketCap: study.price.close * study.price.sharesM, date: study.price.date,
      narrative: `B-equivalent common-equity basis: ${study.price.close} SEK B-share close on ${study.price.date} × ${study.price.sharesM} million outstanding A+B shares, excluding treasury shares (published ${study.price.sharesDate}). Assumes equal economic rights and unchanged outstanding shares since that report. This is not the sum of separately priced A and B market capitalizations. Saved Börsdata download ${study.sources.find(s => s.id === 'prices')!.date}; no FX conversion. Price and share sources are in the researched study ${study.id}.`,
    },
    years: a.years, ownership: a.ownership, capital, notes: { ...study.notes }, scenarios,
    evidence: [
      { heading: 'Reported cash → normalized reference → forecast dividends', blocks: [
        paragraph('calculation', 'Source: January–June 2026 report, page 11. Trailing year = FY 2025 + H1 2026 − H1 2025. Gross purchases exclude disposal proceeds; lease principal is deducted separately.', ['interim']),
        { kind: 'table', classification: 'calculation', columns: ['Reported item', 'FY 2025', 'H1 2025', 'H1 2026', 'Trailing year'], precision: 0, sourceIds: ['interim'],
          rows: (Object.keys(historyLabels) as (keyof typeof historyLabels)[]).map(key => ({ label: historyLabels[key], values: [h.annual2025[key], h.half2025[key], h.half2026[key], bridge[key]] })) },
        paragraph('calculation', `Cash after gross investment and lease principal: ${format(bridge.operatingCash, 0)} − ${format(bridge.capex, 0)} − ${format(bridge.leasePrincipal, 0)} = ${format(bridge.operatingCash - bridge.capex - bridge.leasePrincipal, 0)}. Replace trailing paid tax of ${format(bridge.cashTaxes, 0)} with assumed normal cash tax of ${format(a.normalizedAnnualCashTax, 0)}: ${format(bridge.normalizedAvailableCash, 0)} SEK m available-cash reference.`, ['interim']),
        paragraph('assumption', `Normal tax of SEK ${format(a.normalizedAnnualCashTax, 0)}m is an analyst estimate, near ${a.normalizedOperatingTaxPercent}% of FY 2025 profit before tax after removing the biological-asset gain. This reference retains actual trailing working-capital movements and interest; it is not a fully normalized through-cycle cash forecast.`, ['interim']),
        { kind: 'table', classification: 'assumption', columns: ['Analyst choice', ...scenarioKeys.map(key => scenarioNames[key])], precision: 0, sourceIds: [], rows: [
          { label: 'Year-one dividends', values: scenarioKeys.map(key => study.scenarios[key].firstPayment) },
          { label: 'Change from cash reference', values: scenarioKeys.map(key => study.scenarios[key].firstPayment - bridge.normalizedAvailableCash) },
          { label: `Annual growth, years 2–${a.growthPhaseYears}`, values: scenarioKeys.map(key => `${study.scenarios[key].growth}%`) },
          { label: `Mature growth, year ${a.growthPhaseYears + 1} onward`, values: scenarioKeys.map(key => `${study.scenarios[key].matureGrowth}%`) },
          { label: 'Required equity return', values: scenarioKeys.map(() => `${a.requiredReturn}%`) },
        ] },
        paragraph('assumption', `${a.ownership} Changes from the reference represent each scenario's assumed operating and reinvestment outcome; reported FCF and historical buybacks are not inserted as shareholder receipts.`),
        paragraph('calculation', `Final net equity sale in year ${a.years} = year ${a.years + 1} dividend ÷ (required return − mature growth) × ${100 - a.saleCostPercent}%. This capitalizes post-horizon cash once. Forest book value is not added. The ${a.saleCostPercent}% selling-cost allowance is an assumption.`),
      ] },
      { heading: 'Capital and financing bridge', blocks: [
        paragraph('calculation', `Tangible equity on ${c.date}: ${format(c.equity, 0)} − ${format(c.intangibles, 0)} = ${format(capital.tangibleEquity, 0)}. Gross debt including leases: ${format(c.borrowingLong + c.borrowingShort, 0)} borrowings + ${format(c.leaseDebtLong + c.leaseDebtShort, 0)} leases = ${format(capital.grossDebt, 0)}. Pension obligations of ${c.pensionObligations} are separate. All ${c.cash} of cash is assumed required, leaving ${a.surplusCash} surplus.`, ['interim']),
        paragraph('calculation', `FY 2025 average tangible-capital proxy = [(${format(c.equity2024, 0)} + ${format(c.netDebt2024, 0)} − ${format(c.intangibles2024, 0)}) + (${format(c.equity2025, 0)} + ${format(c.netDebt2025, 0)} − ${format(c.intangibles2025, 0)})] ÷ 2 = ${format(capital.averageTCE, 0)}. Estimated normalized NOPAT = (${format(c.ebit2025, 0)} − ${format(c.biologicalGain2025, 0)}) × (1 − ${a.normalizedOperatingTaxPercent}%) = ${format(capital.nopat, 1)}. The denominator retains forest revaluations and follows reported net-debt accounting; it is a proxy, with its limits recorded in the financing notes.`, ['interim', 'annual']),
      ] },
      { heading: 'Research context', blocks: [paragraph('source', `The ${study.deepDive.date} deep dive is available in the company's Research view. Its historical price targets and asset-floor calculation are superseded for this study.`)] },
    ],
    recovery: {
      status: 'available', explanation: 'June 2026 book assets are source facts. Estimated proceeds apply the analyst stress recovery percentages below; additional costs and timing are assumptions. Intangibles are excluded. Full book claims include deferred tax and existing provisions.', sourceIds: ['interim'],
      assets: study.recovery.assets.map(asset => ({ label: asset.label, book: asset.book, proceeds: { low: asset.book * asset.rates.low, mid: asset.book * asset.rates.mid, high: asset.book * asset.rates.high } })),
      scenarios: Object.fromEntries(scenarioKeys.map(key => [key, { claims: study.recovery.claims, costs: study.scenarios[key].recoveryCosts, cashBurn: 0, year: study.scenarios[key].recoveryYear }])) as Record<ScenarioKey, { claims: number; costs: number; cashBurn: number; year: number }>,
      limitations: `Deferred tax of ${format(study.recovery.deferredTaxIncluded, 0)} is already in claims. No additional income tax is deducted. Additional costs include cash burn beyond existing provisions; the separate cash-burn row is zero to avoid counting it again. Net recovery cannot be below zero for common equity; it is neither added to the DCF nor treated as a price floor. A minority shareholder cannot compel these sales.`,
    },
  };
  // Preserve the original, directly inspectable recovery percentages as evidence.
  adapted.evidence.push({ heading: 'Recovery assumptions', blocks: [{ kind: 'table', classification: 'assumption', columns: ['Asset', ...scenarioKeys.map(key => `${scenarioNames[key]} recovery %`)], sourceIds: ['interim'], rows: study.recovery.assets.map(asset => ({ label: asset.label, values: scenarioKeys.map(key => `${format(asset.rates[key] * 100, 0)}%`) })) }] });
  return { study: adapted, bridge, rationales };
}
