import { Chart } from './Charts';
import { format } from './model';
import { scenarioRangeGeometry } from './scenarioRanges';
import type { StarterEvidence } from './starterValuations';
import { scenarioColors, scenarioKeys, scenarioNames, validAmount, type ValuationDraft } from './valuation';

/** Undiscounted company cash: usable even when price or required return is missing. */
export default function CashFlowForecastChart({ draft, evidence, matchesStarter }: { draft: ValuationDraft; evidence: StarterEvidence; matchesStarter: boolean }) {
  const currencyMatches = draft.currency === evidence.currency;
  const history = currencyMatches ? [...evidence.annual].filter(row => validAmount(row.cashFlow)).reverse() : [];
  const forecast = Array.from({ length: draft.years }, (_, i) => ({ year: i + 1, values: scenarioKeys.map(key => draft.scenarios[key].cashFlows[i] ?? null) }));
  const complete = forecast.filter(row => row.values.every(validAmount));
  const hasCash = history.length > 0 || forecast.some(row => row.values.some(validAmount));
  const labels = [...history.map(row => `FY ${row.year}`), ...forecast.map(row => `Year ${row.year}`)];
  const current = draft.starterOrigin?.id === 'empirical-cash-starter-v3';
  const trend = matchesStarter && currencyMatches && draft.starterOrigin?.id !== 'weighted-cash-starter-v1' && evidence.projection === 'trend' ? evidence.trend : null;
  const fitted = trend && validAmount(trend.intercept) && validAmount(trend.slope);
  const lastYear = history.at(-1)?.year;
  const historical = [...history.map(row => row.cashFlow), ...forecast.map(() => null)];
  const fit = history.map(row => fitted && lastYear !== undefined ? trend!.intercept! + trend!.slope! * (row.year - lastYear) : null);
  const start = history.length && matchesStarter && currencyMatches ? fitted ? trend!.intercept : current && evidence.projection === 'latest' ? history.at(-1)!.cashFlow : current && evidence.projection === 'flat' ? evidence.anchor : null : null;
  const paths = scenarioKeys.map(key => [...history.map((_, i) => i === history.length - 1 ? start : null), ...forecast.map(row => draft.scenarios[key].cashFlows[row.year - 1] ?? null)]);
  // Exact scenario crossings share the same envelope geometry as DCF.
  // Complete spans remain separate across deliberately cleared cash inputs.
  const spans = scenarioRangeGeometry(paths);
  const completeIndices = labels.map((_, i) => i).filter(i => i >= history.length && paths.every(path => validAmount(path[i])));
  const bounds = (i: number, high: boolean) => (high ? Math.max : Math.min)(...paths.map(path => path[i] as number));
  const unit = `${draft.currency} m`;
  const amount = (value: number | null) => value === null ? 'Unavailable' : format(value, 2);
  const rangeKind = (year: number) => !matchesStarter ? 'edited' : evidence.forecast[year - 1]?.rangeBasis ?? (current ? 'unavailable' : 'percentage');
  const rangeLabel = (year: number) => ({ historical: 'Historical errors', 'assumed-tail': 'Assumed later years', percentage: 'Assumed percentage', unavailable: 'Unavailable', edited: 'Edited scenario' })[rangeKind(year)];
  return <section className="valuation-card cash-flow-forecast" data-history-count={history.length} data-forecast-count={complete.length} data-range-mode={evidence.rangeMode} data-range-status={matchesStarter ? evidence.uncertainty.status : 'edited'} data-range-group={matchesStarter ? evidence.uncertainty.group ?? '' : ''} data-calibration-id={evidence.uncertainty.calibrationId ?? ''}>
    <div className="valuation-chart-heading"><div><div className="eyebrow">HISTORY → FUTURE CASH</div><h2>Cash flow over time</h2></div><span className="cash-flow-unit">Company cash · {unit}</span></div>
    <p>Annual cash flow before discounting. The mid line follows your current forecast; the shaded span runs from the lowest to the highest scenario in each year.</p>
    {matchesStarter && current && <div className="cash-flow-range-key" aria-label="Historical and assumed uncertainty by forecast year">
      {[1, Math.min(4, draft.years), draft.years].filter((year, i, all) => all.indexOf(year) === i).map(year => <span key={year}>Year {year}<strong>{evidence.forecast[year - 1]?.halfWidth == null ? 'Unavailable' : `±${format(evidence.forecast[year - 1].halfWidth!, 1)} ${unit}`}</strong><small>{rangeLabel(year)}</small></span>)}
      <small>{evidence.uncertainty.status === 'historical' ? `${evidence.uncertainty.group ?? 'Pooled'} historical cash variability · 80% research target` : evidence.uncertainty.reason ?? 'User-assumed sensitivity'}</small>
    </div>}
    {matchesStarter && draft.starterOrigin?.id === 'weighted-cash-starter-v2' && <div className="cash-flow-range-key" aria-label="Assumed uncertainty by forecast year">
      {[1, Math.min(5, draft.years), draft.years].filter((year, i, all) => all.indexOf(year) === i).map(year => <span key={year}>Year {year}<strong>±{format(evidence.spreadPercent + (year - 1) * evidence.spreadStepPercent, 0)}%</strong></span>)}
      <small>{evidence.spreadStepPercent === 0 ? 'Constant range' : `Widens by ${format(evidence.spreadStepPercent, 1)} percentage points per year`}</small>
    </div>}
    {hasCash ? <Chart label="Cash flow over time: history and forecast scenarios" height={380} option={{
      tooltip: { trigger: 'axis', valueFormatter: (value: number) => `${format(value, 2)} ${unit}` },
      legend: { bottom: 0, data: [...(history.length ? ['Historical cash flow'] : []), ...(fitted ? ['Historical trend'] : []), ...scenarioKeys.map(key => scenarioNames[key])], textStyle: { color: '#60736b', fontSize: 11 } },
      grid: { left: 82, right: 28, top: 42, bottom: 105 },
      xAxis: { type: 'category', boundaryGap: false, data: labels, name: 'Time · annual periods', nameLocation: 'middle', nameGap: 34, nameTextStyle: { color: '#677b72', fontSize: 11 }, axisLabel: { color: '#677b72', fontSize: 10, hideOverlap: true }, axisLine: { lineStyle: { color: '#b4c5b7' } } },
      yAxis: { type: 'value', name: `Cash flow · ${unit}`, nameTextStyle: { color: '#677b72', fontSize: 11 }, axisLabel: { color: '#677b72', fontSize: 10, formatter: (value: number) => Math.abs(value) >= 1000 ? `${format(value / 1000, 1)}k` : format(value, 0) }, splitLine: { lineStyle: { color: '#e0e7df', type: 'dashed' } } },
      series: [
        { type: 'custom', silent: true, clip: true, data: [0], z: 0, tooltip: { show: false }, renderItem: (_params: any, api: any) => {
          const coord = ([x, y]: [number, number]) => {
            const left = Math.floor(x), right = Math.ceil(x);
            const a = api.coord([left, y]), b = api.coord([right, y]);
            return [a[0] + (b[0] - a[0]) * (x - left), a[1]];
          };
          return { type: 'group', children: [
            ...spans.filter(span => span.upper.length > 1).map(span => ({ type: 'polygon', shape: { points: [...span.upper.map(coord), ...[...span.lower].reverse().map(coord)] }, style: { fill: '#326d9420' } })),
            ...completeIndices.map(i => { const low = api.coord([i, bounds(i, false)]), high = api.coord([i, bounds(i, true)]); return { type: 'line', shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style: { stroke: '#326d9433', lineWidth: 1 } }; }),
          ] };
        } },
        { type: 'line', name: 'Historical cash flow', data: historical, connectNulls: false, showSymbol: true, symbolSize: 5, lineStyle: { width: 2, color: '#8a9691' }, itemStyle: { color: '#8a9691' }, ...(history.length ? { markLine: { silent: true, symbol: 'none', label: { formatter: 'Forecast →', position: 'insideEndTop', color: '#75867c', fontSize: 10 }, lineStyle: { color: '#a8b9ae', type: 'dashed' }, data: [{ xAxis: history.length - 1 }] } } : {}) },
        ...(fitted ? [{ type: 'line', name: 'Historical trend', data: [...fit, ...forecast.map(() => null)], showSymbol: false, connectNulls: false, lineStyle: { width: 2, type: 'dashed', color: '#326d9480' }, itemStyle: { color: '#326d9480' } }] : []),
        ...scenarioKeys.map((key, i) => ({ type: 'line', name: scenarioNames[key], data: paths[i], connectNulls: false, showSymbol: draft.years <= 20, symbolSize: key === 'mid' ? 5 : 3, lineStyle: { width: key === 'mid' ? 3 : 1.5, color: scenarioColors[key] }, itemStyle: { color: scenarioColors[key] } })),
      ],
    }} /> : <div className="valuation-empty">Cash-flow history and forecast amounts are unavailable. Enter annual cash-flow assumptions to draw the forecast.</div>}
    <p className="chart-caption">Historical values are saved cash-flow proxies; future values are editable assumptions. Forecast years are full model years after {draft.valuationDate}. {current && matchesStarter && evidence.uncertainty.status === 'historical' ? 'Years 1–4 use historical forecast errors with an 80% coverage target. Year 5 onward uses assumed widening; evidence is strongest near term. The target is not a guarantee or a probability for the full cash path or DCF.' : 'The displayed scenario span has no assigned confidence level.'} DCF and NPV below discount these same future amounts. COVID and rebound years remain in the history.</p>
    {!currencyMatches && <p className="valuation-caution">Historical cash flow is available in {evidence.currency}; it is hidden while the forecast uses {draft.currency}. No currency conversion is inferred.</p>}
    <details className="cash-flow-range-table"><summary>Inspect annual cash-flow ranges</summary><div className="table-scroll"><table aria-label={`Annual cash-flow scenarios in ${unit}`}><thead><tr><th>Forecast year</th>{scenarioKeys.map(key => <th key={key}>{scenarioNames[key]} · {unit}</th>)}<th>Range basis</th></tr></thead><tbody>{forecast.map(row => <tr key={row.year} data-year={row.year} data-range-kind={rangeKind(row.year)}><td>Year {row.year}</td>{row.values.map((value, i) => <td key={i}>{amount(value)}</td>)}<td>{rangeLabel(row.year)}</td></tr>)}</tbody></table></div></details>
  </section>;
}
