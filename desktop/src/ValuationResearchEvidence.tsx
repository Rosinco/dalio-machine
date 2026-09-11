import { format } from './model';
import { buildResearchedValuation, type ResearchedStudy } from './researchedValuations';
import { scenarioKeys, scenarioNames } from './valuation';

export default function ValuationResearchEvidence({ study, edited }: { study: ResearchedStudy; edited: boolean }) {
  const { bridge, recovery, draft } = buildResearchedValuation(study), a = study.assumptions, c = study.capital;
  const historyLabels = { operatingCash: 'Operating cash after interest and paid tax', capex: 'Gross purchases of non-current assets', leasePrincipal: 'Lease principal payments', cashTaxes: 'Cash taxes paid (negative = refund)' };
  return <section className="valuation-card valuation-research-evidence" data-research-study={study.id}>
    <div className="eyebrow">RESEARCHED STARTING ASSUMPTIONS · {study.asOf}</div><h2>How the Holmen scenarios were built</h2>
    <p>{edited ? 'Your working study differs from these starting assumptions. The tables below retain the original research calculations; current results use your edited fields.' : 'These source figures and analyst choices populate the working study. All amounts below are SEK millions unless a different unit is shown.'}</p>
    <h3>Reported cash → normalized reference → forecast dividends</h3>
    <p>Source: January–June 2026 report, page 11. Trailing year = FY 2025 + H1 2026 − H1 2025. Gross purchases exclude disposal proceeds; lease principal is deducted separately.</p>
    <div className="table-scroll"><table><thead><tr><th>Reported item</th><th>FY 2025</th><th>H1 2025</th><th>H1 2026</th><th>Trailing year</th></tr></thead><tbody>{(Object.keys(historyLabels) as (keyof typeof historyLabels)[]).map(k => <tr key={k}><th>{historyLabels[k]}</th>{[study.cashHistory.annual2025[k], study.cashHistory.half2025[k], study.cashHistory.half2026[k], bridge[k]].map((v, i) => <td key={i}>{format(v, 0)}</td>)}</tr>)}</tbody></table></div>
    <p className="valuation-source">Cash after gross investment and lease principal: {format(bridge.operatingCash, 0)} − {format(bridge.capex, 0)} − {format(bridge.leasePrincipal, 0)} = <strong>{format(bridge.operatingCash - bridge.capex - bridge.leasePrincipal, 0)}</strong>.<br />Replace trailing paid tax of {format(bridge.cashTaxes, 0)} with assumed normal cash tax of {format(a.normalizedAnnualCashTax, 0)}: <strong>{format(bridge.normalizedAvailableCash, 0)} SEK m</strong> available-cash reference.</p>
    <p>Normal tax of SEK {format(a.normalizedAnnualCashTax, 0)}m is an analyst estimate, near 22% of FY 2025 profit before tax after removing the biological-asset gain. This reference retains actual trailing working-capital movements and interest; it is not a fully normalized through-cycle cash forecast.</p>
    <div className="table-scroll"><table><thead><tr><th>Analyst choice</th>{scenarioKeys.map(k => <th key={k}>{scenarioNames[k]}</th>)}</tr></thead><tbody>
      <tr><th>Year-one dividends</th>{scenarioKeys.map(k => <td key={k}>{format(study.scenarios[k].firstPayment, 0)}</td>)}</tr>
      <tr><th>Change from cash reference</th>{scenarioKeys.map(k => <td key={k}>{format(study.scenarios[k].firstPayment - bridge.normalizedAvailableCash, 0)}</td>)}</tr>
      <tr><th>Annual growth, years 2–10</th>{scenarioKeys.map(k => <td key={k}>{study.scenarios[k].growth}%</td>)}</tr>
      <tr><th>Mature growth, year 11 onward</th>{scenarioKeys.map(k => <td key={k}>{study.scenarios[k].matureGrowth}%</td>)}</tr>
      <tr><th>Required equity return</th>{scenarioKeys.map(k => <td key={k}>{a.requiredReturn}%</td>)}</tr>
    </tbody></table></div>
    <p>{a.ownership} Changes from the reference represent each scenario's assumed operating and reinvestment outcome; reported FCF and historical buybacks are not inserted as shareholder receipts.</p>
    <p>Final net equity sale in year {a.years} = year {a.years + 1} dividend ÷ (required return − mature growth) × {100 - a.saleCostPercent}%. This capitalizes post-horizon cash once. Forest book value is not added. The 2% selling-cost allowance is an assumption.</p>
    <h3>Capital and financing bridge</h3>
    <p>Tangible equity on {c.date}: {format(c.equity, 0)} − {format(c.intangibles, 0)} = <strong>{format(draft.capital.tangibleEquity, 0)}</strong>. Gross debt including leases: {format(c.borrowingLong + c.borrowingShort, 0)} borrowings + {format(c.leaseDebtLong + c.leaseDebtShort, 0)} leases = <strong>{format(draft.capital.grossDebt, 0)}</strong>. Pension obligations of {c.pensionObligations} are separate. All {c.cash} of cash is assumed required, leaving <strong>0 surplus</strong>.</p>
    <p>FY 2025 average tangible-capital proxy = [(57,370 + 3,397 − 498) + (55,405 + 4,979 − 487)] ÷ 2 = <strong>{format(draft.capital.averageTCE, 0)}</strong>. Estimated normalized NOPAT = ({format(c.ebit2025, 0)} − {format(c.biologicalGain2025, 0)}) × (1 − {a.normalizedOperatingTaxPercent}%) = <strong>{format(draft.capital.nopat, 1)}</strong>. The denominator retains forest revaluations and follows reported net-debt accounting; it is a proxy, with its limits recorded in the financing notes.</p>
    <details className="valuation-recovery-bridge"><summary>Inspect the separate breakup calculation</summary>
      <p>June 2026 book assets are source facts. Recovery percentages, additional costs and timing are analyst stress assumptions. Intangibles are excluded. Full book claims include deferred tax and existing provisions.</p>
      <div className="table-scroll"><table><thead><tr><th>Asset</th><th>Book value</th>{scenarioKeys.map(k => <th key={k}>{scenarioNames[k]} recovery %</th>)}</tr></thead><tbody>{study.recovery.assets.map(asset => <tr key={asset.label}><th>{asset.label}</th><td>{format(asset.book, 0)}</td>{scenarioKeys.map(k => <td key={k}>{format(asset.rates[k] * 100, 0)}%</td>)}</tr>)}
        <tr><th>Estimated gross proceeds</th><td />{scenarioKeys.map(k => <td key={k}>{format(recovery[k].gross, 1)}</td>)}</tr>
        <tr><th>All prior book claims</th><td />{scenarioKeys.map(k => <td key={k}>−{format(recovery[k].claims, 0)}</td>)}</tr>
        <tr><th>Additional costs and cash burn</th><td />{scenarioKeys.map(k => <td key={k}>−{format(recovery[k].costs, 0)}</td>)}</tr>
        <tr><th>Net common-equity recovery</th><td />{scenarioKeys.map(k => <td key={k}>{format(recovery[k].net, 1)}</td>)}</tr>
        <tr><th>Realization year</th><td />{scenarioKeys.map(k => <td key={k}>{study.scenarios[k].recoveryYear}</td>)}</tr>
      </tbody></table></div>
      <p>Deferred tax of {format(study.recovery.deferredTaxIncluded, 0)} is already in claims. No additional income tax is deducted. Additional costs exclude those covered by booked provisions. Net recovery cannot be below zero for common equity; it is neither added to the DCF nor treated as a price floor.</p>
    </details>
    <h3>Sources and dates</h3>
    <ul className="valuation-source-list">{study.sources.map(source => <li key={source.id}><strong>{source.url ? <a href={source.url} target="_blank" rel="noreferrer">{source.title}</a> : source.title}</strong><span>{source.date} · {source.location}</span>{source.path && <span>{source.path}</span>}{source.sha256 && <small>SHA-256 {source.sha256}</small>}</li>)}</ul>
    <p>The {study.deepDive.date} deep dive is available in the company's Research view. Its historical price targets and asset-floor calculation are superseded for this study. Prepared {study.asOf}; source dates remain independent of the selected macro release. Research version: {study.id}.</p>
  </section>;
}
