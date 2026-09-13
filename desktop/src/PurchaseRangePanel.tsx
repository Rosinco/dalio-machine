import { useState } from 'react';
import { Chart } from './Charts';
import { format } from './model';
import { calculatePurchaseRange, defaultPurchaseRangeSettings, toPurchaseDisplayAmount, validatePurchaseDisplayBasis } from './purchaseRange';
import type { PurchaseShareReference } from './purchaseShareBasis';
import { scenarioColors, scenarioKeys, scenarioNames, validAmount, validDay, type PurchaseRangeSettings, type ValuationDraft } from './valuation';

type Props = { draft: ValuationDraft; companyId: string; financialVersion: string | null; researchVersion: string; shareReference: PurchaseShareReference | null; onChange: (settings: PurchaseRangeSettings) => void };
const number = (e: React.ChangeEvent<HTMLInputElement>) => e.target.value === '' || !Number.isFinite(e.target.valueAsNumber) ? null : e.target.valueAsNumber;

export default function PurchaseRangePanel({ draft: d, companyId, financialVersion, researchVersion, shareReference, onChange }: Props) {
  const settings = d.purchaseRange ?? defaultPurchaseRangeSettings, result = calculatePurchaseRange(d, settings), basis = validatePurchaseDisplayBasis(d, settings);
  const amounts = [result.candidateEquity, d.marketCap, ...scenarioKeys.flatMap(k => { const s = result.scenarios[k]; return [s.value, s.cashPV, s.terminalPV, s.ceiling, s.cashOnlyCeiling, s.npvAtCandidate]; })];
  const display = basis.divisor !== null && amounts.some(v => v !== null && !Number.isFinite(v / basis.divisor!))
    ? { ...basis, divisor: null, error: 'The chosen share count puts displayed amounts outside the supported numerical range. Review the share basis or use total equity units.' } : basis;
  const [message, setMessage] = useState('');
  const change = (patch: Partial<PurchaseRangeSettings>) => { setMessage(''); onChange({ ...settings, ...patch }); };
  const share = settings.shareBasis ?? { sharesMillions: null, date: '', source: '', currency: d.currency };
  const changeShares = (patch: Partial<typeof share>) => change({ shareBasis: { ...share, ...patch } });
  const divisor = display.error ? null : display.divisor;
  const units = settings.unit === 'share' ? `${d.currency} / share` : `${d.currency} m equity`;
  const valueInUnits = (v: number | null) => toPurchaseDisplayAmount(v, display);
  const money = (v: number | null) => valueInUnits(v) === null ? 'Unavailable' : `${format(valueInUnits(v), 2)} ${units}`;
  const datedPrice = validAmount(d.marketCap) && d.marketCap > 0 && validDay(d.valuationDate) && validDay(d.priceDate) && d.priceDate <= d.valuationDate && d.priceSource.trim() ? d.marketCap : null;
  const candidateInput = Object.hasOwn(settings, 'candidateEquity') ? settings.candidateEquity ?? null : result.candidateEquity;
  const known = scenarioKeys.filter(k => result.scenarios[k].value !== null);
  const maximum = Math.max(1e-9, ...known.map(k => Math.abs(result.scenarios[k].value!)), result.candidateEquity ?? 0, datedPrice ?? 0) * 1.1;
  const ceiling = result.selectedCeiling;
  const qualifyingRange = result.marginError || result.referenceError || !result.selected || result.selected.error ? '' : String(ceiling !== null && ceiling > 0);
  const xValues = [...new Set([0, maximum, ...Array.from({ length: 21 }, (_, i) => maximum * i / 20), ...known.map(k => result.scenarios[k].value!).filter(v => v > 0), ...scenarioKeys.map(k => result.scenarios[k].ceiling).filter((v): v is number => v !== null), ...(result.candidateEquity === null ? [] : [result.candidateEquity])])].sort((a,b) => a-b);
  const chartFinite = divisor !== null && xValues.every(p => Number.isFinite(p/divisor) && known.every(k => Number.isFinite((result.scenarios[k].value!-p)/divisor)));
  const selectedName = scenarioNames[settings.referenceScenario];
  const exportRange = async () => {
    try {
      const rows: (string | number | null)[][] = [['company_id', 'scenario', 'valuation_date', 'currency', 'equity_value_m', 'cash_pv_m', 'terminal_pv_m', 'terminal_share_pct', 'margin_of_safety_pct', 'purchase_ceiling_m', 'cash_only_ceiling_m', 'candidate_equity_m', 'npv_at_candidate_m', 'reference_scenario', 'display_units', 'shares_m', 'share_basis_date', 'share_basis_source', 'price_ceiling_display', 'npv_at_candidate_display', 'market_price_date', 'reference_market_cap_m', 'price_source', 'financial_version', 'research_version', 'study_origin', 'policy_error', 'scenario_error', 'display_error']];
      for (const key of scenarioKeys) {
        const s = result.scenarios[key];
        rows.push([companyId, key, d.valuationDate, d.currency, s.value, s.cashPV, s.terminalPV, s.terminalShare === null ? null : s.terminalShare * 100, settings.marginOfSafetyPercent, s.ceiling, s.cashOnlyCeiling, result.candidateEquity, s.npvAtCandidate, settings.referenceScenario, units, settings.shareBasis?.sharesMillions ?? null, settings.shareBasis?.date ?? '', settings.shareBasis?.source ?? '', valueInUnits(s.ceiling), valueInUnits(s.npvAtCandidate), d.priceDate, datedPrice, d.priceSource, financialVersion, researchVersion, d.researchOrigin?.id ?? d.starterOrigin?.id ?? 'manual', result.marginError ?? result.referenceError ?? result.candidateError ?? '', s.error, display.error]);
      }
      const csv = rows.map(row => row.map(v => typeof v === 'number' ? String(v) : `"${String(v ?? '').replace(/^[=+@\-\t\r]/, "'$&").replaceAll('"', '""')}"`).join(',')).join('\r\n');
      const filename = `Macro-Atlas-Purchase-${companyId}-${d.valuationDate}.csv`;
      if ('__TAURI_INTERNALS__' in window) {
        const { invoke } = await import('@tauri-apps/api/core'); setMessage(`Saved to ${await invoke<string>('export_csv', { filename, contents: csv })}`);
      } else {
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' })), a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); setMessage('Purchase range exported as CSV.');
      }
    } catch (error) { setMessage(`Could not export purchase range: ${String(error)}`); }
  };
  return <section className="valuation-card purchase-range" data-purchase-range="true" data-reference-ceiling={ceiling ?? ''} data-qualifying-range={qualifyingRange} data-purchase-qualifies={result.qualifies === null ? '' : String(result.qualifies)}>
    <div className="valuation-chart-heading"><div><div className="eyebrow">PRICE FROM YOUR VALUATION</div><h2>Purchase price range</h2></div><button onClick={exportRange}>Export purchase range</button></div>
    <p>A purchase is attractive under the selected assumptions when it meets your margin of safety. DCF includes forecast cash and terminal sale once; NPV is that value less the proposed price.</p>
    <div className="purchase-controls">
      <label>Margin of safety · %<input type="number" min={0} max={100} step="any" aria-label="Purchase margin of safety (%)" value={settings.marginOfSafetyPercent ?? ''} onChange={e => change({ marginOfSafetyPercent: number(e) })} /></label>
      <label>Reference scenario<select aria-label="Purchase reference scenario" value={settings.referenceScenario} onChange={e => change({ referenceScenario: e.target.value as PurchaseRangeSettings['referenceScenario'] })}>{scenarioKeys.map(k => <option key={k} value={k}>{scenarioNames[k]}</option>)}</select></label>
      <label>Price units<select aria-label="Purchase price units" value={settings.unit} onChange={e => change({ unit: e.target.value as 'equity' | 'share' })}><option value="equity">Total equity value · millions</option><option value="share">Price per share</option></select></label>
    </div>
    <div className="purchase-ceiling" data-purchase-ceiling-card="true"><small>{selectedName.toUpperCase()} SCENARIO · {settings.marginOfSafetyPercent === null ? 'MARGIN MISSING' : `${format(settings.marginOfSafetyPercent, 1)}% MARGIN`}</small>
      <strong>{display.error ? 'Complete the price display basis' : result.marginError || result.referenceError || result.selected?.error ? 'Complete the valuation and margin inputs' : ceiling === null ? 'No positive purchase price meets this rule' : `At or below ${money(ceiling)}`}</strong>
      <span>Lower positive prices provide a larger modeled discount. There is no valuation-based minimum purchase price.</span>
    </div>
    {[result.marginError, result.referenceError, result.selected?.error, display.error].filter(Boolean).map((error, i) => <p className="valuation-caution" key={i}>{error}</p>)}
    <div className="purchase-controls">
      <label>Proposed purchase price · {units}<input type="number" min={0} step="any" aria-label="Proposed purchase price" value={valueInUnits(candidateInput) ?? ''} disabled={divisor === null} onChange={e => { const v = number(e); change({ candidateEquity: v === null ? null : v * divisor! }); }} /></label>
      <div className="purchase-candidate-result"><small>NPV AT YOUR PROPOSED PRICE</small><strong data-purchase-selected-npv="true">{money(result.selected?.npvAtCandidate ?? null)}</strong><span>{result.qualifies === null ? result.candidateError ?? 'Enter a positive proposed price to compare.' : result.qualifies ? `Meets the ${selectedName} scenario margin rule.` : `Does not meet the ${selectedName} scenario margin rule.`}</span></div>
      <button onClick={() => { const next = { ...settings }; delete next.candidateEquity; onChange(next); }} disabled={datedPrice === null}>Use saved purchase price</button>
    </div>
    <p className="valuation-source">{datedPrice === null ? 'No usable dated purchase quote is available. Cash-based values and purchase ceilings can still be calculated.' : settings.unit === 'share' ? `Saved total-equity basis from ${d.priceDate}, converted using the chosen share count dated ${share.date || 'not entered'}: ${money(datedPrice)}. This is an equity equivalent under that denominator, not an independently observed per-share quote.` : `Saved purchase basis: ${money(datedPrice)} on ${d.priceDate}. This is the study's dated price, not a live market quote.`}</p>
    <details className="purchase-share-basis" open={settings.unit === 'share'}><summary>Share count and ownership basis</summary>
      <p>Per-share prices require a dated share count for the same common-equity claim as the cash flows. Review share classes, receipts, treasury shares and dilution. A changed denominator changes per-share figures; total equity values stay unchanged.</p>
      {shareReference && <p><button onClick={() => { const { sharesMillions, date, source, currency } = shareReference; change({ shareBasis: { sharesMillions, date, source, currency }, unit: 'share' }); }}>Use saved share reference</button> {shareReference.kind === 'reviewed' ? 'Reviewed study ownership basis' : 'Reported share-count proxy · ownership still requires review'}: {format(shareReference.sharesMillions, 6)} million as of {shareReference.date}.</p>}
      <div className="purchase-controls"><label>Shares · millions<input type="number" step="any" min={0} aria-label="Purchase share count (millions)" value={share.sharesMillions ?? ''} onChange={e => changeShares({ sharesMillions: number(e) })} /></label><label>Share-basis date<input type="date" max={d.valuationDate} aria-label="Purchase share basis date" value={share.date} onChange={e => changeShares({ date: e.target.value })} /></label><label>Valuation currency<input maxLength={3} aria-label="Purchase share basis currency" value={share.currency} onChange={e => changeShares({ currency: e.target.value.toUpperCase() })} /></label></div>
      <label>Source and ownership claim<textarea aria-label="Purchase share basis source" rows={3} maxLength={5000} value={share.source} onChange={e => changeShares({ source: e.target.value })} /></label>
    </details>
    {known.length > 0 && divisor !== null && !chartFinite && <p className="valuation-caution">The chart exceeds the supported numerical range. Review the valuation amounts and price basis.</p>}
    {known.length > 0 && chartFinite && divisor !== null && <Chart label="NPV across purchase prices" height={340} option={{
      tooltip: { trigger: 'axis', formatter: (params: any) => { const points = Array.isArray(params) ? params : [params]; return [`Purchase price: ${format(points[0]?.value?.[0], 2)} ${units}`, ...points.filter((p: any) => p.seriesName !== 'Purchase ceiling').map((p: any) => `${p.seriesName}: ${format(p.value[1], 2)} ${units} NPV`)].join('\n'); } },
      legend: { bottom: 0, selectedMode: false, data: known.map(k => scenarioNames[k]), textStyle: { color: '#60736b', fontSize: 11 } },
      grid: { left: 83, right: 25, top: 36, bottom: 86 },
      xAxis: { type: 'value', min: 0, max: maximum / divisor, name: `Proposed price · ${units}`, nameLocation: 'middle', nameGap: 32, nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10, formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v/1000,1)}k` : format(v,1) } },
      yAxis: { type: 'value', name: `NPV · ${units}`, nameTextStyle: { fontSize: 10 }, axisLabel: { fontSize: 10, formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v/1000,1)}k` : format(v,1) }, splitLine: { lineStyle: { color: '#e0e7df', type: 'dashed' } } },
      series: known.map((key, index) => ({ type: 'line', name: scenarioNames[key], data: xValues.map(p => [p/divisor, (result.scenarios[key].value!-p)/divisor]), showSymbol: false, lineStyle: { width: key === settings.referenceScenario ? 3 : 1.5, color: scenarioColors[key] }, itemStyle: { color: scenarioColors[key] }, ...(index === 0 ? {
        markArea: ceiling === null ? undefined : { silent: true, itemStyle: { color: '#65a98622' }, data: [[{ xAxis: 0 }, { xAxis: ceiling/divisor }]] },
        markLine: { silent: true, symbol: 'none', label: { fontSize: 10, position: 'insideEndTop' }, data: [{ yAxis: 0, name: 'Break-even NPV', label: { formatter: 'NPV = 0' } }, ...(ceiling === null ? [] : [{ xAxis: ceiling/divisor, name: 'Purchase ceiling', label: { formatter: 'Ceiling' }, lineStyle: { color: '#337854', type: 'dashed' } }]), ...(datedPrice === null ? [] : [{ xAxis: datedPrice/divisor, label: { formatter: settings.unit === 'share' ? 'Saved equity equivalent' : 'Saved price', position: 'insideStartTop' }, lineStyle: { color: '#8d7a53', type: 'dotted' } }])] }
      } : {}) }))
    }} />}
    <p className="chart-caption">The shaded positive-price region meets the selected scenario's margin rule. Each line crosses zero at its own DCF value. Required returns, cash forecasts and terminal assumptions come from the working study. The existing final-sale display checkbox does not change this total-value calculation.</p>
    <div className="table-scroll"><table aria-label="Scenario purchase ceilings"><thead><tr><th>Scenario</th><th>DCF value</th><th>Purchase ceiling</th><th>NPV at proposed price</th><th>PV of terminal sale</th><th>Terminal share of value</th><th>Cash-only ceiling</th></tr></thead><tbody>{scenarioKeys.map(key => {
      const s = result.scenarios[key];
      return <tr key={key} data-purchase-scenario={key} data-equity-value={s.value ?? ''} data-price-ceiling={s.ceiling ?? ''} data-cash-only-ceiling={s.cashOnlyCeiling ?? ''} data-terminal-pv={s.terminalPV ?? ''} data-candidate-npv={s.npvAtCandidate ?? ''}><th>{scenarioNames[key]}</th><td>{money(s.value)}</td><td>{s.error || result.marginError ? 'Unavailable' : s.ceiling === null ? 'No positive price' : money(s.ceiling)}</td><td>{money(s.npvAtCandidate)}</td><td>{money(s.terminalPV)}</td><td>{s.terminalShare === null ? 'Unavailable' : `${format(s.terminalShare*100,1)}%`}</td><td>{s.error || result.marginError ? 'Unavailable' : s.cashOnlyCeiling === null ? 'No positive price' : money(s.cashOnlyCeiling)}</td></tr>;
    })}</tbody></table></div>
    {result.scenarioCeilingRange && <p>Range across the three scenario ceilings: {money(result.scenarioCeilingRange.min)} to {money(result.scenarioCeilingRange.max)}. This compares assumptions; only the selected rule determines the highlighted purchase region.</p>}
    <p className="valuation-method">The 30% default is your editable margin assumption, not a probability or an extra annual return. Cash-only ceilings show dependence on the final sale and are not recovery floors. {d.starterOrigin ? 'These company forecasts are unreviewed starter scenarios until refined in a deep dive.' : d.researchOrigin ? 'The values use this reviewed study and any working edits.' : 'The values use your entered scenarios.'} Price alone does not establish business quality or make an unsupported forecast reliable.</p>
    {message && <p role="status" className="purchase-export-status">{message}</p>}
  </section>;
}
