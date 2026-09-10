import { useState, type ReactNode } from 'react';
import { Chart } from './Charts';
import { finite, format, seriesColors } from './model';
import { countryMoney, datedHistory, type LiquidityPoint, type LiquidityReport, type Trace } from './liquidity';

const words = (value?: string) => value?.replaceAll('_', ' ') || 'Unavailable';
const movement: Record<string, string> = {
  expanding_accelerating: 'Expanding, with growth accelerating', expanding_decelerating: 'Expanding, with growth slowing',
  contracting_accelerating: 'Contracting, with growth improving', contracting_decelerating: 'Contracting, with growth weakening',
};

export default function LiquidityPanel({ report, code, currency, name }: { report: LiquidityReport; code: string; currency: string; name: string }) {
  const [offshoreCurrency, setOffshoreCurrency] = useState('USD');
  const { reading: money, scope } = countryMoney(report, code, currency);
  const pair = report.central_bank_divergence.find(r => r.country === (scope === 'currency-area' ? 'EU' : code));
  const offshore = report.offshore_credit.find(r => r.currency === offshoreCurrency) ?? report.offshore_credit[0];
  return <div className="liquidity-panel" data-liquidity-country={code}>
    <div className="section-title"><h3>Money & liquidity</h3><span className="micro">{report.as_of}</span></div>
    <p className="section-note">Money growth, funding conditions and their supporting evidence. Blue identifies quantities; it carries no good/bad rating.</p>
    <section className="national-money">
      <div className="eyebrow">{scope === 'currency-area' ? 'EURO-AREA CONTEXT' : `${name.toUpperCase()} · BROAD MONEY`}</div>
      {scope === 'currency-area' && <p className="scope-note">This reading covers the whole euro area{code !== 'EU' && `, including ${name}`}. It is not a national money-supply series.</p>}
      {!money ? <p className="empty">No national broad-money diagnostic is included for {name} in this release. Global context is available below.</p> : <>
        <h4>{money.title}</h4><p className="section-note">{money.period || 'Period unavailable'} · {words(money.availability_status)}</p>
        <div className="liquidity-metrics"><Reading label="Annual log growth" value={money.annual_log_growth_pct} unit="%" /><Reading label="Change over 3 months" value={money.acceleration_3m_pp} unit="pp" /></div>
        <p className="movement-note">{movement[money.movement] || words(money.movement)}</p>
        <p className="body-note">Outstanding amount: <strong>{format(money.latest_value, 0)} {money.unit}</strong>.</p>
        <GrowthHistory points={money.history ?? []} cadence="monthly" label={`${scope === 'currency-area' ? 'Euro area' : name} broad-money annual log growth`} />
        <p className="chart-caption">Annual log growth: 100 × ln(value / value one year earlier). Acceleration is its change over three months. Gaps remain blank; history uses this saved vintage.</p>
        <TraceDetails trace={money} report={report} />
      </>}
      {pair && <div className="liquidity-subsection"><h4>Money & central-bank balance sheet</h4><p className="section-note">{pair.country === 'EU' ? 'Euro area' : name} · {pair.period} · {words(pair.availability_status)}</p><div className="liquidity-metrics"><Reading label="Money growth" value={pair.money_annual_log_growth_pct} unit="%" /><Reading label="Central-bank assets growth" value={pair.central_bank_assets_annual_log_growth_pct} unit="%" /></div><p className="body-note">Growth gap: <strong>{format(pair.money_minus_assets_growth_gap_pp, 2)} percentage points</strong>.</p><p className="chart-caption">{pair.interpretation_limit}</p><TraceDetails trace={pair} report={report} /></div>}
    </section>
    <div className="eyebrow">GLOBAL FUNDING CONTEXT</div><p className="section-note">The following panels keep their own geographical scope and observation dates.</p>
    <Disclosure title="Broad-money breadth">
      <p className="section-note">{report.money_summary.ready} of {report.money_summary.expected} currency areas available · {report.money_summary.common_period || 'Different observation periods'}</p>
      <div className="liquidity-metrics"><Reading label="Median annual log growth" value={report.money_summary.median_annual_log_growth_pct} unit="%" /><Reading label="Positive growth" value={report.money_summary.positive_growth_breadth} unit={`of ${report.money_summary.ready}`} digits={0} /></div>
      <table className="data-table"><thead><tr><th>Area</th><th>Period</th><th>Growth</th><th>3m change</th></tr></thead><tbody>{report.broad_money.map(r => <tr key={r.country}><td>{r.country} · {r.currency}</td><td>{r.period || '—'}</td><td>{format(r.annual_log_growth_pct, 2)}%</td><td>{format(r.acceleration_3m_pp, 2)} pp</td></tr>)}</tbody></table>
      <p className="chart-caption">{report.money_summary.interpretation_limit}</p>
    </Disclosure>
    <Disclosure title="Offshore credit by currency">
      <p className="section-note">Credit to non-bank borrowers outside the currency’s home area. Each currency is measured independently.</p>
      <select className="metric-select" aria-label="Offshore credit currency" value={offshore?.currency ?? ''} onChange={e => setOffshoreCurrency(e.target.value)}>{report.offshore_credit.map(r => <option key={r.currency} value={r.currency}>{r.currency} · {r.title}</option>)}</select>
      {offshore ? <><p className="section-note">{offshore.period} · {words(offshore.availability_status)}</p><div className="liquidity-metrics"><Reading label="Annual log growth" value={offshore.annual_log_growth_pct} unit="%" /><Reading label="Change over one quarter" value={offshore.acceleration_1q_pp} unit="pp" /></div><GrowthHistory points={offshore.history ?? []} cadence="quarterly" label={`${offshore.currency} offshore-credit annual log growth`} /><p className="chart-caption">{offshore.interpretation_limit || 'A currency-specific credit measure; amounts in different currencies are not added together.'}</p><TraceDetails trace={offshore} report={report} /></> : <p className="empty">No offshore-credit diagnostic is included.</p>}
    </Disclosure>
    <Disclosure title="US money-market funds">
      <p className="section-note">United States · {report.mmf.period || 'Period unavailable'} · {words(report.mmf.availability_status)}</p>
      <div className="liquidity-metrics"><Reading label="MMF annual log growth" value={report.mmf.mmf_annual_log_growth_pct} unit="%" /><Reading label="Growth gap versus M2" value={report.mmf.mmf_minus_m2_growth_gap_pp} unit="pp" /></div>
      <p className="chart-caption">{report.mmf.interpretation_limit}</p>
      <h4>Share of fund assets</h4><table className="data-table"><thead><tr><th>Asset category</th><th>Share</th></tr></thead><tbody>{report.mmf.asset_allocation.map(r => <tr key={r.title}><td>{r.title.replace('Money Market Mutual Fund Investments in ', '')}</td><td>{format(r.share_of_total_pct, 2)}%</td></tr>)}</tbody></table>
      <h4>Published repo counterparties & clearing categories</h4><p className="section-note">{report.mmf.published_repo_counterparty_categories.period || 'Period unavailable'} · {words(report.mmf.published_repo_counterparty_categories.availability_status)}</p>
      <p className="chart-caption">{report.mmf.published_repo_counterparty_categories.interpretation_limit}</p>
      <table className="data-table"><thead><tr><th>Published category</th><th>Ratio to repo</th></tr></thead><tbody>{report.mmf.published_repo_counterparty_categories.ratios.map(r => <tr key={r.title}><td>{r.title.replace('Money Market Mutual Fund Investments in Repurchase Agreements ', '')}</td><td>{format(r.share_of_repo_pct, 2)}%</td></tr>)}</tbody></table>
      <TraceDetails trace={report.mmf} report={report} />
    </Disclosure>
    <Disclosure title="US repo pricing & activity">
      <p className="section-note">United States · {report.repo.period_date || 'Date unavailable'} · {words(report.repo.availability_status)}</p>
      <div className="liquidity-metrics"><Reading label="Rate dispersion · 5d median" value={report.repo.fragmentation_5d_median_bp} unit="bp" /><Reading label="Dispersion · robust z" value={report.repo.fragmentation_robust_z} unit="" /></div>
      <p className="body-note">Effective federal funds rate: {format(report.repo.effr_pct, 2)}%.</p>
      <table className="data-table"><thead><tr><th>Venue</th><th>Rate</th><th>5d premium to EFFR</th></tr></thead><tbody>{report.repo.venues.map(r => <tr key={r.venue}><td>{words(r.venue)}<small>{r.status}</small></td><td>{format(r.rate_pct, 2)}%</td><td>{format(r.effr_premium_5d_median_bp, 2)} bp</td></tr>)}</tbody></table>
      <p className="chart-caption">{report.repo.interpretation_limit}</p>
      <h4>Activity and outstanding amounts</h4><p className="chart-caption">Outstanding stocks and transaction volumes use separate measures and dates. They are shown individually and are not summed.</p>
      {report.repo.volume_context.map(r => <div className="volume-reading" key={r.title}><strong>{r.title}</strong><span>{format(r.latest_value, 0)} {r.unit}</span><small>{words(r.measure_kind)} · {r.latest_date} · {r.status}</small></div>)}
      <TraceDetails trace={report.repo} report={report} />
    </Disclosure>
    <Disclosure title="Coverage, sources & calculation rules">
      <p className="body-note">The report was known at {report.as_known_at}. All required inputs were available by {report.complete_snapshot_available_at || 'an unrecorded time'}.</p>
      <p className="chart-caption">Historical values use the saved source vintage. Earlier chart dates do not imply that these values were available at that time.</p>
      <table className="data-table"><thead><tr><th>Diagnostic family</th><th>Available</th></tr></thead><tbody>{Object.entries(report.coverage).map(([key, row]) => <tr key={key}><td>{words(key)}</td><td>{row.ready} / {row.expected}</td></tr>)}</tbody></table>
      {report.horizons.map(h => <p className="body-note" key={h.horizon}><strong>{h.horizon}</strong><br />{h.supported_context}</p>)}
      <h4>Calculation rules</h4>{Object.entries(report.formulas).map(([key, formula]) => <div className="formula-row" key={key}><strong>{words(key)}</strong><code>{formula}</code></div>)}
      <dl className="release-details"><dt>Methodology</dt><dd>{report.methodology_version}</dd><dt>Methodology SHA-256</dt><dd className="hash">{report.methodology_sha256}</dd><dt>Snapshot SHA-256</dt><dd className="hash">{report.snapshot_sha256}</dd></dl>
      <p className="chart-caption">The package includes the saved source ledger. Original provider files referenced by that ledger are not bundled; their contents cannot be rechecked here offline.</p>
      <TraceDetails trace={{ availability_status: 'saved', input_release_ids: report.input_releases.map(r => r.release_id) }} report={report} />
    </Disclosure>
  </div>;
}

function Reading({ label, value, unit, digits = 2 }: { label: string; value: number | null; unit: string; digits?: number }) {
  return <div className="liquidity-reading"><small>{label}</small><strong>{format(value, digits)}<span>{finite(value) ? unit : ''}</span></strong></div>;
}
function Disclosure({ title, children }: { title: string; children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return <details className="liquidity-disclosure" onToggle={e => setOpen(e.currentTarget.open)}><summary>{title}</summary>{open && <div className="disclosure-body">{children}</div>}</details>;
}
function GrowthHistory({ points, cadence, label }: { points: LiquidityPoint[]; cadence: 'monthly' | 'quarterly'; label: string }) {
  const data = datedHistory(points, cadence);
  if (!data.values.some(finite)) return <div className="empty">No saved growth history is available.</div>;
  return <Chart label={label} height={240} option={{ tooltip: { trigger: 'axis', renderMode: 'richText', valueFormatter: (value: number) => `${format(value, 2)}%` }, grid: { left: 44, right: 15, top: 24, bottom: 65 }, xAxis: { type: 'category', data: data.periods, axisLabel: { fontSize: 10 }, boundaryGap: false }, yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 10, formatter: '{value}%' }, splitLine: { lineStyle: { color: '#e5e9e1', type: 'dashed' } } }, dataZoom: [{ type: 'slider', start: 65, end: 100, height: 17, bottom: 8, showDetail: false, borderColor: '#dfe5dc' }], series: [{ type: 'line', name: label, showSymbol: false, connectNulls: false, lineStyle: { width: 2, color: seriesColors.selected }, itemStyle: { color: seriesColors.selected }, data: data.values }] }} />;
}
function TraceDetails({ trace, report }: { trace: Trace; report: LiquidityReport }) {
  const sources = report.input_releases.filter(r => trace.input_release_ids.includes(r.release_id));
  return <details className="source-trace"><summary>Source records & observation status</summary><p className="chart-caption">Status: {words(trace.availability_status)}{trace.input_statuses?.length ? ` · ${trace.input_statuses.join(', ')}` : ''}</p>{sources.map(r => <div className="source-record" key={r.release_id}><strong>{words(r.source_family)}</strong><small>Release {r.release_id} · {r.partition_key}</small><span>Published {r.published_at || 'date unavailable'}<br />Available {r.available_at}</span><span className="source-address">{r.source_url || 'No source URL recorded'}</span><code>{r.content_sha256}</code></div>)}{!sources.length && <p className="section-note">No matching source release is recorded.</p>}</details>;
}
