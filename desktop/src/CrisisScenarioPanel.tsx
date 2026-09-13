import { Chart } from './Charts';
import { format } from './model';
import { buildCrisisScenario, defaultCrisisAssumptions } from './crisisScenario';
import type { CrisisAssumptions, ValuationDraft } from './valuation';

export default function CrisisScenarioPanel({ draft, onChange }: { draft: ValuationDraft; onChange: (crisis: CrisisAssumptions) => void }) {
  const c = draft.crisis ?? defaultCrisisAssumptions();
  let calculated: ReturnType<typeof buildCrisisScenario> | null = null, error = '';
  if (c.enabled) try { calculated = buildCrisisScenario(draft); } catch (e) { error = e instanceof Error ? e.message : String(e); }
  const input = (key: Exclude<keyof CrisisAssumptions, 'enabled' | 'rationale'>, label: string, min: number, max?: number, step = 'any') => <label>{label}<input type="number" aria-label={label} value={c[key] ?? ''} min={min} max={max} step={step} onChange={e => onChange({ ...c, [key]: e.target.value === '' || !Number.isFinite(e.target.valueAsNumber) ? null : e.target.valueAsNumber })} /></label>;
  const unit = `${draft.currency} m`, years = Array.from({ length: draft.years }, (_, i) => `Year ${i + 1}`);
  const money = (v: number | null) => v === null ? 'Unavailable' : `${format(v, 2)} ${unit}`;
  const axes = { tooltip: { trigger: 'axis' }, legend: { bottom: 0, textStyle: { color: '#60736b' } }, grid: { left: 75, right: 25, top: 35, bottom: 65 }, xAxis: { type: 'category', data: years, axisLabel: { hideOverlap: true } }, yAxis: { type: 'value', name: unit, splitLine: { lineStyle: { color: '#e0e7df', type: 'dashed' } } } };
  return <section className="valuation-card valuation-crisis" data-crisis-enabled={c.enabled} aria-label="Separate crisis scenario">
    <div className="eyebrow">SHOCK & RECOVERY · SEPARATE ASSUMPTIONS</div><h2>Crisis scenario</h2>
    <p>Explore a temporary cash shock and recovery alongside the current mid forecast. This case has no assigned probability. Historical COVID and recovery years stay in the source history.</p>
    <label className="valuation-crisis-toggle"><input type="checkbox" aria-label="Enable crisis scenario" checked={c.enabled} onChange={e => onChange({ ...c, enabled: e.target.checked })} />Enable crisis scenario</label>
    {c.enabled && <>
      <div className="valuation-crisis-inputs">
        {input('shockPercent', 'Crisis cash reduction (%)', 0, 300)}
        {input('startYear', 'Crisis start year', 1, draft.years, '1')}
        {input('durationYears', 'Crisis duration (years)', 1, 50, '1')}
        {input('recoveryYears', 'Crisis recovery (years)', 0, 50, '1')}
        {input('extraAnnualCashCost', 'Crisis extra annual cash cost', 0)}
        {input('discountRate', 'Crisis required return', 0, 100)}
        {input('terminalEquity', 'Crisis final equity sale', 0)}
      </div>
      <p className="valuation-method">Cash cost and final sale are in {unit}; required return is annual %. During the shock, cash = mid − reduction × |mid| − extra cash cost. The reduction and extra cost fade evenly over the recovery period, reaching the current mid path in its final year. Zero recovery years means immediate recovery. Negative cash can become more negative.</p>
      {c.startYear !== null && c.durationYears !== null && c.recoveryYears !== null && c.startYear + c.durationYears + c.recoveryYears - 1 > draft.years && <p className="valuation-caution">The shock or recovery extends beyond this forecast. Review the final sale against the remaining disruption.</p>}
      <label>Shock, recovery, investment and financing evidence<textarea aria-label="Crisis assumptions and evidence" maxLength={5000} rows={3} value={c.rationale} onChange={e => onChange({ ...c, rationale: e.target.value })} /></label>
      {error && <p role="alert" className="valuation-caution">{error}</p>}
      {calculated && <>
        <Chart label="Crisis cash flow compared with mid forecast" height={290} option={{ ...axes, series: [
          { type: 'line', name: 'Mid cash', data: draft.scenarios.mid.cashFlows.slice(0, draft.years), connectNulls: false, lineStyle: { color: '#326d94', type: 'dashed' }, itemStyle: { color: '#326d94' } },
          { type: 'line', name: 'Crisis cash', data: calculated.cashFlows, connectNulls: false, lineStyle: { color: '#b55d44', width: 3 }, itemStyle: { color: '#b55d44' } },
        ] }} />
        <div className="valuation-crisis-results"><p>Cash present value<strong data-crisis-result="cashPV">{money(calculated.result.cashPV)}</strong></p><p>Value including final sale<strong data-crisis-result="value">{money(calculated.result.value)}</strong></p><p>NPV after purchase price<strong data-crisis-result="npv">{money(calculated.result.npv)}</strong></p></div>
        <p className="valuation-method">Final sale is an independent assumption, initially zero. Annual cash shocks do not automatically revise sale proceeds. These calculations do not alter the low/mid/high scenarios or add liquidation recovery.</p>
        {calculated.result.error ? <p className="valuation-caution">{calculated.result.error}</p> : <Chart label="Crisis discounted cash flow and cumulative NPV" height={270} option={{ ...axes, series: [
          { type: 'bar', name: 'Discounted cash', data: calculated.result.discounted, itemStyle: { color: '#b55d4490' } },
          { type: 'line', name: 'Cumulative NPV with final sale', data: calculated.result.cumulativeNPVWithSale.slice(1), lineStyle: { color: '#b55d44', width: 2 }, itemStyle: { color: '#b55d44' } },
        ] }} />}
        <details><summary>Inspect crisis cash by year</summary><div className="table-scroll"><table aria-label="Crisis annual cash flows"><thead><tr><th>Year</th><th>Mid · {unit}</th><th>Crisis · {unit}</th></tr></thead><tbody>{calculated.cashFlows.map((v, i) => <tr key={i}><td>Year {i + 1}</td><td>{money(draft.scenarios.mid.cashFlows[i] ?? null)}</td><td>{money(v)}</td></tr>)}</tbody></table></div></details>
      </>}
    </>}
  </section>;
}
