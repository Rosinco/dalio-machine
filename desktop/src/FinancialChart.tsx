import { Chart } from './Charts';
import { finite, format } from './model';

export default function FinancialChart({ labels, series, label, unit }: { labels: string[]; series: { name: string; values: (number | null)[]; color: string }[]; label: string; unit: string }) {
  if (!series.some(s => s.values.some(finite))) return <div className="empty">No comparable saved observations for this chart.</div>;
  return <Chart label={label} height={240} option={{ tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${format(v, 2)} ${unit}` }, legend: { bottom: 0, textStyle: { color: '#637467', fontSize: 9 } }, grid: { left: 55, right: 15, top: 20, bottom: 70 }, xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 9, color: '#697b6c' }, axisLine: { lineStyle: { color: '#dce4d8' } }, axisTick: { show: false } }, yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 9, color: '#697b6c', formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v / 1000, 1)}k` : format(v, 1) }, splitLine: { lineStyle: { color: '#e4e9df', type: 'dashed' } } }, dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 28, height: 14, showDetail: false, borderColor: '#dce4d8' }], series: series.map(s => ({ type: 'line', name: s.name, data: s.values, connectNulls: false, showSymbol: false, lineStyle: { width: 2, color: s.color }, itemStyle: { color: s.color } })) }} />;
}
