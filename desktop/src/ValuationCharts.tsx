import { Chart } from './Charts';
import { format } from './model';
import { buildValuationRangeRows, scenarioRangeGeometry } from './scenarioRanges';
import type { StarterEvidence } from './starterValuations';
import { scenarioColors, scenarioKeys, scenarioNames, type ScenarioResult, type ValuationDraft } from './valuation';

type Props = {
  draft: ValuationDraft; results: Record<string, ScenarioResult>; sale: boolean; onSale: (v: boolean) => void;
  evidence?: StarterEvidence; matchesStarter?: boolean; matchesResearch?: boolean;
};
const basisLabels: Record<string, string> = {
  initial: 'Initial investment', historical: 'Historically informed cash', 'assumed-tail': 'Assumed later-year cash',
  'historical-and-assumed': 'Historical + assumed cash inputs', percentage: 'Assumed percentage',
  reviewed: 'Reviewed scenarios', edited: 'Edited scenarios', assumed: 'Scenario assumptions', unavailable: 'Unavailable',
};

export default function ValuationCharts({ draft, results, sale, onSale, evidence, matchesStarter = false, matchesResearch = false }: Props) {
  const money = (value: number) => `${format(value, 2)} ${draft.currency}`;
  const cashBasis = (year: number, cumulative: boolean) => {
    if (year === 0) return 'initial';
    if (matchesResearch && draft.researchOrigin) return 'reviewed';
    if (!matchesStarter || !draft.starterOrigin || !evidence) return 'edited';
    const sources = (cumulative ? evidence.forecast.slice(0, year) : evidence.forecast.slice(year - 1, year)).map(row => row.rangeBasis);
    if (!sources.length) return 'assumed';
    if (sources.every(source => source === 'historical')) return 'historical';
    if (sources.every(source => source === 'assumed-tail')) return 'assumed-tail';
    if (sources.includes('historical')) return 'historical-and-assumed';
    if (sources.every(source => source === 'percentage')) return 'percentage';
    return 'assumed';
  };
  const panel = (kind: 'dcf' | 'npv') => {
    const cumulative = kind === 'npv';
    const rows = buildValuationRangeRows(draft, results, kind, sale);
    if (!rows) return <section key={kind} className="valuation-card valuation-empty">Complete all three scenarios to show the {cumulative ? 'cumulative NPV' : 'discounted cash-flow'} range.</section>;
    const paths = scenarioKeys.map(key => rows.map(row => row[key]));
    const geometry = scenarioRangeGeometry(paths, cumulative);
    const title = cumulative ? 'Cumulative NPV' : 'Discounted cash flow';
    const includesSale = (year: number) => cumulative && sale && year === draft.years;
    const basis = (year: number) => `${basisLabels[cashBasis(year, cumulative)]}${includesSale(year) ? ' + assumed sale' : ''}`;
    const checkpoints = rows.filter(row => [cumulative ? 0 : 1, Math.min(4, draft.years), draft.years].includes(row.year));
    return <section key={kind} className="valuation-card valuation-range-chart" data-valuation-chart={kind}>
      <div className="valuation-chart-heading"><h2>{title}</h2>{cumulative && <label className="valuation-checkbox"><input type="checkbox" checked={sale} onChange={e => onSale(e.target.checked)} />Include final sale</label>}</div>
      <p>{cumulative ? `Discounted cash received through each year, less your initial ${format(draft.investment, 0)} ${draft.currency}. Crossing zero marks discounted payback.` : `Each year's forecast cash payment, expressed in today's ${draft.currency}, for an investment of ${format(draft.investment, 0)} ${draft.currency}.`}</p>
      <div className="cash-flow-range-key valuation-range-key" aria-label={`${title} minimum and maximum by year`}>
        {checkpoints.map(row => <span key={row.year}>Year {row.year}<strong>{money(row.min)} to {money(row.max)}</strong><small>{basis(row.year)}</small></span>)}
      </div>
      <Chart label={`${title}: low, mid and high scenarios`} height={340} option={{
        tooltip: { trigger: 'axis', formatter: (params: any) => {
          const point = (Array.isArray(params) ? params : [params]).find((p: any) => p.seriesType === 'line');
          const row = point && rows[point.dataIndex];
          if (!row) return '';
          return [`Year ${row.year}`, `Min–max: ${money(row.min)} to ${money(row.max)}`, ...scenarioKeys.map(key => `${scenarioNames[key]}: ${money(row[key])}`), basis(row.year)].join('\n');
        } },
        legend: { bottom: 0, selectedMode: false, data: scenarioKeys.map(key => scenarioNames[key]), textStyle: { color: '#60736b', fontSize: 11 } },
        grid: { left: 78, right: 24, top: 40, bottom: 85 },
        xAxis: { type: 'category', boundaryGap: false, data: rows.map(row => `Year ${row.year}`), name: 'Time · years after valuation', nameLocation: 'middle', nameGap: 32, nameTextStyle: { color: '#677b72', fontSize: 11 }, axisLabel: { fontSize: 10, color: '#677b72', hideOverlap: true } },
        yAxis: { type: 'value', name: `${cumulative ? 'Cumulative NPV' : 'Discounted cash'} · ${draft.currency}`, nameTextStyle: { color: '#677b72', fontSize: 11 }, axisLabel: { fontSize: 10, color: '#677b72', formatter: (value: number) => Math.abs(value) >= 1e6 ? `${format(value / 1e6, 1)}m` : Math.abs(value) >= 1000 ? `${format(value / 1000, 1)}k` : format(value, 0) }, splitLine: { lineStyle: { color: '#e0e7df', type: 'dashed' } } },
        series: [
          { type: 'custom', name: 'Scenario min–max', silent: true, clip: true, data: [0], z: 0, tooltip: { show: false }, renderItem: (_params: any, api: any) => {
            // Category axes describe annual points. Interpolate crossing positions
            // in pixels so category rounding cannot widen the displayed envelope.
            const coord = ([x, y]: [number, number]) => {
              const left = Math.floor(x), right = Math.ceil(x);
              const a = api.coord([left, y]), b = api.coord([right, y]);
              return [a[0] + (b[0] - a[0]) * (x - left), a[1]];
            };
            return { type: 'group', children: [
              ...geometry.filter(span => span.upper.length > 1).map(span => ({ type: 'polygon', shape: { points: [...span.upper.map(coord), ...[...span.lower].reverse().map(coord)] }, style: { fill: '#326d9420' } })),
              ...rows.map((row, i) => {
                const low = api.coord([i, row.min]), high = api.coord([i, row.max]);
                return { type: 'group', children: [
                  { type: 'line', shape: { x1: low[0], y1: low[1], x2: high[0], y2: high[1] }, style: { stroke: '#326d9460', lineWidth: 1 } },
                  ...[low, high].map(point => ({ type: 'line', shape: { x1: point[0] - 3, y1: point[1], x2: point[0] + 3, y2: point[1] }, style: { stroke: '#326d9480', lineWidth: 1 } })),
                ] };
              }),
            ] };
          } },
          ...scenarioKeys.map((key, i) => ({ type: 'line', name: scenarioNames[key], data: paths[i], step: cumulative ? 'end' : false, connectNulls: false, showSymbol: rows.length < 21, symbolSize: key === 'mid' ? 5 : 3,
            lineStyle: { width: key === 'mid' ? 3 : 1.5, color: scenarioColors[key] }, itemStyle: { color: scenarioColors[key] },
            ...(key === 'mid' ? { markLine: { silent: true, symbol: 'none', label: { formatter: cumulative ? 'Cost recovered' : 'Zero cash', position: 'insideEndTop', color: '#71827a', fontSize: 10 }, lineStyle: { color: '#93a398', type: 'dashed' }, data: [{ yAxis: 0 }] } } : {}) })),
        ],
      }} />
      <p className="chart-caption">{cumulative ? 'NPVₜ = −initial investment + discounted cash received through year t. Each named scenario is accumulated separately; the range covers their cumulative outcomes.' : 'DCFₜ = cash paymentₜ / (1 + required equity return)ᵗ, scaled to your investment. Each scenario uses its own required return; discounting can narrow the later-year span.'}</p>
      <p className="chart-caption">{cumulative ? sale ? `Includes the assumed equity sale at the end of year ${draft.years}; any resulting recovery depends on that sale.` : 'Cash distributions only. The full valuation can also include the separately assumed final sale.' : 'Final sale proceeds are shown separately in the value table.'} Shading and annual markers show the minimum and maximum across all three scenarios, including crossings and negative values. These are scenario bounds with no assigned confidence level.</p>
      <details className="cash-flow-range-table"><summary>{cumulative ? 'Inspect cumulative NPV ranges' : 'Inspect annual DCF ranges'}</summary>
        <div className="table-scroll"><table aria-label={`${title} ranges in ${draft.currency} for your investment`}><thead><tr><th>Year</th>{['min', ...scenarioKeys, 'max'].map(key => <th key={key}>{key === 'min' ? 'Minimum' : key === 'max' ? 'Maximum' : scenarioNames[key as typeof scenarioKeys[number]]} · {draft.currency}</th>)}<th>Cash basis</th></tr></thead>
          <tbody>{rows.map(row => <tr key={row.year} data-range-year={row.year} data-includes-sale={includesSale(row.year)}>
            <td>Year {row.year}</td>{(['min', ...scenarioKeys, 'max'] as const).map(key => <td key={key} data-range-value={key} data-value={row[key]}>{money(row[key])}</td>)}<td data-range-basis={cashBasis(row.year, cumulative)}>{basis(row.year)}</td>
          </tr>)}</tbody></table></div>
      </details>
    </section>;
  };
  return <div className="valuation-charts">{panel('dcf')}{panel('npv')}</div>;
}
