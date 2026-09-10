import { Chart, type ChartPoint } from './Charts';
import { benchmark, branchMetrics, bubbleDiameter, comparisonColors, metricUnit, observation, type Benchmark, type BranchData, type BranchMetric, type BranchSettings } from './branchComparison';
import type { CompanyEntry } from './listingCatalogue';
import { finite, format } from './model';

const yearMarker = (year: number) => ({ symbol: 'none', silent: true, label: { show: false }, lineStyle: { color: '#798797', width: 1, type: 'dashed' }, data: [{ xAxis: year }] });
const axes = (s: BranchSettings, small = false) => ({
  grid: { left: small ? 48 : 70, right: 22, top: 20, bottom: small ? 28 : 43 },
  xAxis: { type: 'value', min: s.from - (s.from === s.to ? .5 : .25), max: s.to + (s.from === s.to ? .5 : .25), minInterval: 1, axisLabel: { fontSize: 10, formatter: (v: number) => Number.isInteger(v) ? String(v) : '' }, splitLine: { show: false }, axisLine: { lineStyle: { color: '#d2dcd5' } } },
  yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 10, formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v / 1000, 1)}k` : format(v, 1) }, splitLine: { lineStyle: { color: '#e2e7e1', type: 'dashed' } } },
});
function band(stats: Benchmark[]) {
  const segments = stats.slice(1).flatMap((r, i) => {
    const previous = stats[i];
    return [r.q1, r.q3, previous.q1, previous.q3].every(finite) ? [[previous.year, previous.q1, previous.q3, r.year, r.q1, r.q3]] : [];
  });
  return { type: 'custom', name: 'Middle 50% of branch listings', silent: true, clip: true, z: 0, data: segments, encode: { x: [0, 3], y: [1, 2, 4, 5] },
    renderItem: (_params: any, api: any) => ({ type: 'polygon', shape: { points: [[api.value(0), api.value(1)], [api.value(0), api.value(2)], [api.value(3), api.value(5)], [api.value(3), api.value(4)]].map(point => api.coord(point)) }, style: { fill: '#bdc8d0', opacity: .32 } }),
  };
}
export default function BranchCharts({ data, companies, settings: s, stats, onPoint }: { data: BranchData; companies: CompanyEntry[]; settings: BranchSettings; stats: Benchmark[]; onPoint: (point: ChartPoint) => void }) {
  const selected = s.selected.map(id => companies.find(c => c.id === id)).filter((c): c is CompanyEntry => !!c);
  const maximum = Math.max(0, ...selected.flatMap(c => (data[c.id] ?? []).filter(r => r.year >= s.from && r.year <= s.to).map(r => observation(r, s).size ?? 0)));
  const unit = metricUnit(s.metric, s.currency);
  const pointCount = selected.reduce((n, c) => n + (data[c.id] ?? []).filter(r => r.year >= s.from && r.year <= s.to && observation(r, s).size !== null).length, 0);
  const series = selected.map(c => ({ type: 'scatter', name: c.display_name, z: 3, symbolSize: (v: number[]) => s.size === 'equal' ? 12 : bubbleDiameter(v[2], maximum),
    itemStyle: { color: comparisonColors[s.selected.indexOf(c.id)] },
    data: (data[c.id] ?? []).filter(r => r.year >= s.from && r.year <= s.to).flatMap(r => {
      const o = observation(r, s);
      return o.value === null || o.size === null ? [] : [{ id: c.id, year: r.year, value: [r.year, o.value, o.size], start: r.start, end: r.end, snapshot: r.source_as_of,
        itemStyle: { opacity: r.year === s.year ? 1 : c.id === s.focus ? .72 : .32, borderColor: '#ffffff', borderWidth: r.year === s.year ? 2 : .5 } }];
    }),
  }));
  const pointTooltip = (p: any) => p.seriesName === 'Branch median' ? `FY ${p.data.year}\nBranch median: ${format(p.value[1], 2)} ${unit}\n${p.data.n} valid listings`
    : `${p.seriesName} · FY ${p.data.year}\n${branchMetrics[s.metric]}: ${format(p.value[1], 2)} ${unit}\n${s.size === 'equal' ? 'Equal-size point' : `${branchMetrics[s.size]}: ${format(p.value[2], 2)} ${s.currency} million`}\nPeriod: ${p.data.start} → ${p.data.end}\nSource snapshot: ${p.data.snapshot}`;
  return <>
    <section className="branch-chart-card bubble-history" data-bubble-points={pointCount} data-size-mode={s.size}>
      <div className="comparison-section-title"><div><div className="eyebrow">THROUGH THE CYCLE</div><h2>{branchMetrics[s.metric]}</h2></div><span>{unit} · fiscal years</span></div>
      <Chart height={340} label="Branch bubble history" onPoint={onPoint} option={{ ...axes(s), tooltip: { trigger: 'item', formatter: pointTooltip }, series: [band(stats),
        { name: 'Branch median', type: 'line', z: 1, showSymbol: true, symbolSize: 4, connectNulls: false, lineStyle: { color: '#677784', width: 1.5 }, itemStyle: { color: '#677784' }, markLine: yearMarker(s.year), data: stats.map(r => ({ year: r.year, n: r.n, value: [r.year, r.median] })) },
        ...series,
      ] }} />
      <p className="chart-caption">Grey line: median of all valid listings in the filtered branch. Shading: middle 50% (at least 4 observations per year). Colours identify listings. Click a point to select its listing and year.</p>
      <p className="chart-caption">{s.size === 'equal' ? 'Points have equal size; they do not represent market cap.' : `Bubble area represents ${branchMetrics[s.size].toLowerCase()} in ${s.currency}, using the same scale across the displayed period. Missing or non-positive sizes are omitted. This is not market cap.`}</p>
      {!pointCount && <p className="comparison-empty">No selected listing has a comparable observation with a valid bubble size in this period. The table explains missing values.</p>}
    </section>
    <div className="comparison-small-charts">
      {(['operating_cash_margin', 'return_on_capital', 'equity_ratio'] as BranchMetric[]).map(metric => {
        const local = { ...s, metric, size: 'equal' as const };
        const miniStats = benchmark(data, companies.map(c => c.id), local);
        return <section className="branch-chart-card" key={metric}><h3>{branchMetrics[metric]}</h3><Chart label={`Linked ${branchMetrics[metric]} history`} height={175} onPoint={onPoint} option={{ ...axes(s, true), tooltip: { trigger: 'item', formatter: (p: any) => `${p.seriesName} · FY ${p.data.year}\n${format(p.value[1], 2)}%` }, series: [
          { name: 'Branch median', type: 'line', connectNulls: false, showSymbol: false, lineStyle: { color: '#9ba5ad', type: 'dashed', width: 1 }, markLine: yearMarker(s.year), data: miniStats.map(r => ({ year: r.year, value: [r.year, r.median] })) },
          ...selected.map(c => ({ name: c.display_name, type: 'line', connectNulls: false, showSymbol: true, symbolSize: 4, lineStyle: { color: comparisonColors[s.selected.indexOf(c.id)], width: c.id === s.focus ? 2.5 : 1, opacity: c.id === s.focus ? 1 : .45 }, itemStyle: { color: comparisonColors[s.selected.indexOf(c.id)] }, data: stats.map(({ year }) => ({ id: c.id, year, value: [year, observation(data[c.id]?.find(r => r.year === year), local).value] })) })),
        ] }} /></section>;
      })}
    </div>
  </>;
}
