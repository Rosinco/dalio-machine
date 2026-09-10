import { useEffect, useRef } from 'react';
import * as echarts from 'echarts/core';
import { LineChart, RadarChart, BarChart, PieChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, RadarComponent, LegendComponent, DataZoomComponent, MarkAreaComponent, GraphicComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import type { EChartsCoreOption } from 'echarts/core';
import type { AtlasIndex, Country, Indicator, Point } from './types';
import { categories, finite, format, historyLines, seriesColors } from './model';

echarts.use([LineChart, RadarChart, BarChart, PieChart, GridComponent, TooltipComponent, RadarComponent, LegendComponent, DataZoomComponent, MarkAreaComponent, GraphicComponent, CanvasRenderer]);

export function Chart({ option, height = 230, label }: { option: EChartsCoreOption; height?: number; label: string }) {
  const element = useRef<HTMLDivElement>(null);
  const instance = useRef<echarts.EChartsType | null>(null);
  useEffect(() => {
    if (!element.current) return;
    const chart = echarts.init(element.current, undefined, { renderer: 'canvas' });
    instance.current = chart;
    const observer = new ResizeObserver(() => chart.resize());
    observer.observe(element.current);
    return () => { observer.disconnect(); chart.dispose(); instance.current = null; };
  }, []);
  useEffect(() => { instance.current?.setOption({ animation: false, textStyle: { fontFamily: 'Segoe UI, sans-serif' }, ...option, tooltip: { ...(typeof option.tooltip === 'object' && option.tooltip !== null ? option.tooltip : {}), renderMode: 'richText' } }, true); }, [option]);
  return <div ref={element} role="img" aria-label={label} style={{ height, width: '100%' }} />;
}

export function Radar({ country, comparison, index }: { country: Country; comparison?: Country; index: AtlasIndex }) {
  const complete = (c: Country) => index.categories.every(k => finite(c.categories[k]?.score));
  if (!complete(country)) return <div className="empty">A radar needs all five categories. Available scores remain listed below.</div>;
  const data = [{ name: country.name, value: index.categories.map(k => Math.round(country.categories[k].score!)), lineStyle: { width: 2, color: seriesColors.selected }, itemStyle: { color: seriesColors.selected }, areaStyle: { color: seriesColors.selected, opacity: .12 } }];
  if (comparison && complete(comparison)) data.push({ name: comparison.name, value: index.categories.map(k => Math.round(comparison.categories[k].score!)), lineStyle: { width: 1.5, color: seriesColors.comparison }, itemStyle: { color: seriesColors.comparison }, areaStyle: { color: seriesColors.comparison, opacity: .06 } });
  return <Chart label={`${country.name} fundamentals radar; five category scores out of 100`} option={{
    tooltip: { trigger: 'item' },
    radar: { center: ['50%', '49%'], radius: '65%', splitNumber: 5, indicator: index.categories.map(k => ({ name: categories[k].short, max: 100 })), axisName: { color: '#63706a', fontSize: 10 }, splitLine: { lineStyle: { color: '#dfe5dc' } }, splitArea: { areaStyle: { color: ['#f8e5e5', '#faebdf', '#faf3d7', '#edf3df', '#e1f0e8'] } }, axisLine: { lineStyle: { color: '#dfe5dc' } } },
    series: [{ type: 'radar', symbol: 'circle', symbolSize: 4, data }],
  }} />;
}

export function HistoryChart({ points, comparison, name, otherName, meta, startYear }: { points: Point[]; comparison?: Point[]; name: string; otherName?: string; meta: Indicator; startYear: number }) {
  const series = (items: Point[], label: string, color: string) => {
    const lines = historyLines(items, startYear);
    return [
      { name: label, type: 'line', showSymbol: false, connectNulls: false, lineStyle: { color, width: 2 }, itemStyle: { color }, data: lines.historical },
      { name: `${label} · forecast`, type: 'line', showSymbol: false, connectNulls: false, lineStyle: { color, width: 2, type: 'dashed' }, itemStyle: { color }, data: lines.forecast },
    ];
  };
  if (!points.some(p => p.year >= startYear && finite(p.value))) return <div className="empty">No saved history in this period.</div>;
  return <Chart label={`${meta.label} history for ${name}. Dashed lines are forecasts.`} option={{
    tooltip: { trigger: 'axis', valueFormatter: (v: number) => format(v, 2) },
    grid: { left: 52, right: 18, top: 20, bottom: 34 },
    xAxis: { type: 'value', min: 'dataMin', max: 'dataMax', minInterval: 1, axisLabel: { formatter: '{value}', color: '#7a837b', fontSize: 10 }, axisLine: { lineStyle: { color: '#dfe3da' } }, splitLine: { show: false } },
    yAxis: { type: 'value', scale: true, axisLabel: { color: '#7a837b', fontSize: 10, formatter: (v: number) => Math.abs(v) >= 1000000 ? `${format(v / 1e9, 1)}B` : Math.abs(v) >= 10000 ? `${format(v / 1000, 0)}k` : format(v) }, splitLine: { lineStyle: { color: '#e9ece5', type: 'dashed' } } },
    series: [...series(points, name, seriesColors.selected), ...(comparison && otherName ? series(comparison, otherName, seriesColors.comparison) : [])],
  }} />;
}
