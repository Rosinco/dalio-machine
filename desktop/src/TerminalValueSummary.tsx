import { format } from './model';
import { calculateScenario, scenarioKeys, scenarioNames, type ValuationDraft } from './valuation';
import { resolveTerminalSale } from './terminalValue';

export default function TerminalValueSummary({ draft: d }: { draft: ValuationDraft }) {
  const midRate = d.scenarios.mid.discountRate;
  const rates = midRate === null ? [] : [...new Set([midRate - 2, midRate, midRate + 2].filter(r => r >= 0 && r <= 100))];
  const money = (v: number | null) => v === null ? 'Unavailable' : `${format(v, 2)} ${d.currency} m`;
  return <section className="valuation-card" data-terminal-summary="true"><h2>Terminal value assumptions</h2>
    <p>Final sale depends on a separate long-term cash assumption or explicit sale proceeds. Annual cash uncertainty does not determine a probability range for that sale.</p>
    <div className="table-scroll"><table aria-label="Terminal cash and sale assumptions"><thead><tr><th>Scenario</th><th>Method</th><th>Year {d.years + 1} cash after reinvestment</th><th>Mature growth</th><th>Required return</th><th>Sale at year {d.years}</th></tr></thead><tbody>{scenarioKeys.map(k => {
      const s = d.scenarios[k];
      return <tr key={k}><th>{scenarioNames[k]}</th><td>{s.terminalCash ? 'Separate sustainable cash' : 'Explicit sale'}</td><td>{s.terminalCash ? money(s.terminalCash.cashFlow) : 'Not specified'}</td><td>{s.terminalCash?.growthRate == null ? 'Not specified' : `${format(s.terminalCash.growthRate, 2)}%`}</td><td>{s.discountRate === null ? 'Unavailable' : `${format(s.discountRate, 2)}%`}</td><td>{money(resolveTerminalSale(s).value)}</td></tr>;
    })}</tbody></table></div>
    {d.starterOrigin?.id === 'empirical-cash-starter-v3' && d.starterOrigin.terminalMethod && <p>The historical starting point uses signed median provider cash, with an assumed ±20% terminal-cash sensitivity and 0% mature growth. Inputs remain unreviewed until a company deep dive reconciles reinvestment and financing. The table shows your current assumptions, including edits.</p>}
    <details><summary>Inspect required-return sensitivity</summary><p>Each row applies the same required return to all three scenarios. Sustainable-cash sale values are recalculated; explicit sale proceeds stay fixed. Other inputs stay as entered. These rates are sensitivities, not estimated company risk.</p>
      <div className="table-scroll"><table aria-label="Required-return NPV sensitivity"><thead><tr><th>Required return</th>{scenarioKeys.map(k => <th key={k}>{scenarioNames[k]} NPV · your {format(d.investment, 0)} {d.currency}</th>)}</tr></thead><tbody>{rates.map(r => <tr key={r}><th>{format(r, 2)}%</th>{scenarioKeys.map(k => {
        const result = calculateScenario({ ...d, scenarios: { ...d.scenarios, [k]: { ...d.scenarios[k], discountRate: r } } }, k);
        return <td key={k}>{result.error ? 'Unavailable' : `${format(result.npv! / d.marketCap! * d.investment!, 2)} ${d.currency}`}</td>;
      })}</tr>)}</tbody></table></div>
    </details>
  </section>;
}
