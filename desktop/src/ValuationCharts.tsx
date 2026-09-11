import { Chart } from './Charts';
import { format } from './model';
import { scenarioColors, scenarioKeys, scenarioNames, type ScenarioResult, type ValuationDraft } from './valuation';

export default function ValuationCharts({ draft, results, sale, onSale }: { draft: ValuationDraft; results: Record<string, ScenarioResult>; sale: boolean; onSale: (v: boolean) => void }) {
  const scale = draft.investment! / draft.marketCap!;
  const options = (cumulative: boolean) => {
    const count = draft.years + (cumulative ? 1 : 0);
    const paths = scenarioKeys.map(key => ({ key, values: (cumulative ? sale ? results[key].cumulativeNPVWithSale : results[key].cumulativeNPV : results[key].discounted).map(v => v * scale) }));
    // Preserve named scenarios when they cross; shade their actual min/max envelope.
    const lows = Array.from({ length: count }, (_, i) => Math.min(...paths.map(p => p.values[i])));
    const highs = Array.from({ length: count }, (_, i) => Math.max(...paths.map(p => p.values[i])));
    return { tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${format(v, 2)} ${draft.currency}` },
      legend: { bottom: 0, data: scenarioKeys.map(k => scenarioNames[k]), textStyle: { color: '#60736b', fontSize: 11 } },
      grid: { left: 72, right: 25, top: 25, bottom: 65 },
      xAxis: { type: 'category', boundaryGap: false, data: Array.from({ length: count }, (_, i) => `Year ${i + (cumulative ? 0 : 1)}`), axisLabel: { fontSize: 10, color: '#677b72' } },
      yAxis: { type: 'value', axisLabel: { fontSize: 10, color: '#677b72', formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v / 1000, 1)}k` : format(v, 0) }, splitLine: { lineStyle: { color: '#e0e7df', type: 'dashed' } } },
      series: [{ type: 'custom', silent: true, data: [0], z: 0, tooltip: { show: false }, renderItem: (_params: any, api: any) => {
        const path = (values: number[]) => values.flatMap((v, i) => cumulative && i > 0 ? [api.coord([i, values[i - 1]]), api.coord([i, v])] : [api.coord([i, v])]);
        return { type: 'polygon', shape: { points: [...path(highs), ...path(lows).reverse()] }, style: { fill: '#326d9414' } };
      } },
        ...paths.map(({ key, values }) => ({ type: 'line', name: scenarioNames[key], data: values, step: cumulative ? 'end' : false, showSymbol: count < 16, symbolSize: 4,
          lineStyle: { width: key === 'mid' ? 3 : 2, color: scenarioColors[key] }, itemStyle: { color: scenarioColors[key] },
          ...(key === 'mid' && cumulative ? { markLine: { silent: true, symbol: 'none', label: { formatter: 'Cost recovered', position: 'insideEndTop', color: '#71827a', fontSize: 10 }, lineStyle: { color: '#93a398', type: 'dashed' }, data: [{ yAxis: 0 }] } } : {}) }))] };
  };
  return <div className="valuation-charts">
    <section className="valuation-card"><h2>Discounted cash flow</h2><p>Each year's forecast cash payment, expressed in today's {draft.currency}, for an investment of {format(draft.investment, 0)} {draft.currency}.</p>
      <Chart label="Discounted cash flow: low, mid and high scenarios" height={300} option={options(false)} />
      <p className="chart-caption">DCFₜ = cash paymentₜ / (1 + required equity return)ᵗ. Final sale proceeds are shown separately in the value table.</p>
    </section>
    <section className="valuation-card"><div className="valuation-chart-heading"><h2>Cumulative NPV</h2><label className="valuation-checkbox"><input type="checkbox" checked={sale} onChange={e => onSale(e.target.checked)} />Include final sale</label></div>
      <p>Discounted cash received through each year, less your initial {format(draft.investment, 0)} {draft.currency}. Crossing zero marks discounted payback.</p>
      <Chart label="Cumulative NPV: low, mid and high scenarios" height={300} option={options(true)} />
      <p className="chart-caption">{sale ? `Includes the assumed equity sale at the end of year ${draft.years}; any resulting recovery depends on that sale.` : 'Cash distributions only. The full valuation can also include the separately assumed final sale.'} The shaded span covers these three scenarios; it is not a probability interval.</p>
    </section>
  </div>;
}
