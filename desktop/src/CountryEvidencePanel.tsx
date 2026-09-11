import { useEffect, useState } from 'react';
import { Activity, ArrowDown, ArrowUp, ArrowRight, FileCheck2 } from 'lucide-react';
import { Chart } from './Charts';
import { decodeCountryEvidence, evidenceCode, evidenceDate, evidenceLabel, evidenceLines, evidenceValue, nativeStatus, safeSource, validateEvidenceIndex, type CountryEvidence, type EvidenceEnvelope, type EvidenceHistory, type EvidenceIndex, type EvidencePoint } from './countryEvidence';
import './countryEvidence.css';

const ROOT = '/data/country-evidence/';
let catalogue: Promise<EvidenceIndex> | undefined;
const cache = new Map<string, CountryEvidence>();
export function useEvidenceIndex() {
  const [data, setData] = useState<EvidenceIndex>(); const [error, setError] = useState('');
  useEffect(() => {
    let active = true;
    catalogue ??= fetch(ROOT + 'index.json').then(r => { if (!r.ok) throw new Error('No country evidence pack is bundled.'); return r.json(); }).then(validateEvidenceIndex);
    catalogue.then(value => { if (active) setData(value); }).catch(e => { if (active) setError(String(e)); });
    return () => { active = false; };
  }, []);
  return { data, error };
}
function useCountryEvidence(index: EvidenceIndex | undefined, code: string) {
  code = evidenceCode(code);
  const key = `${index?.id}:${code}`;
  const [state, setState] = useState<{ key: string; data?: CountryEvidence; error?: string }>({ key });
  useEffect(() => {
    let active = true;
    if (!index?.countries[code]) { setState({ key }); return; }
    const saved = cache.get(key);
    if (saved) { setState({ key, data: saved }); return; }
    setState({ key });
    const controller = new AbortController();
    fetch(ROOT + index.countries[code].file, { signal: controller.signal }).then(r => { if (!r.ok) throw new Error('The saved country evidence file is missing.'); return r.text(); }).then(text => decodeCountryEvidence(text, index, code)).then(data => {
      if (!active) return;
      if (cache.size >= 5) cache.delete(cache.keys().next().value!);
      cache.set(key, data); setState({ key, data });
    }).catch(e => { if (active) setState({ key, error: String(e) }); });
    return () => { active = false; controller.abort(); };
  }, [index, code, key]);
  return state.key === key ? state : { key };
}
function SourceDetails({ source }: { source: Record<string, any> }) {
  const [raw, setRaw] = useState(false), [copied, setCopied] = useState(false);
  const metadata = source.publisher_metadata ?? {};
  const name = source.source === 'SCB_MONITORING' && metadata.source === 'The Riksbank' ? 'Statistics Sweden, on behalf of Sveriges Riksbank' : metadata.producer ?? metadata.publisher ?? source.source ?? 'Retained source';
  return <div className="ce-source"><strong>{evidenceLabel(name)}</strong>
    {source.value !== undefined && <p>{evidenceDate(source as EvidencePoint)} · {evidenceValue(source as EvidencePoint)} · {evidenceLabel(source.status ?? '')} {nativeStatus(source as EvidencePoint, source.source)}</p>}
    <dl><dt>Native series</dt><dd>{source.series_id ?? 'See source record'}</dd><dt>Available to this system</dt><dd>{source.available_at ?? 'Not supplied'}</dd><dt>Retrieved</dt><dd>{source.retrieved_at ?? 'Not supplied'}</dd><dt>First publication</dt><dd>{source.published_at ? metadata.publication_precision === 'date' ? `${source.published_at.slice(0, 10)} (publisher supplies date only)` : source.published_at : 'Not supplied'}</dd>
      {(source.publisher_updated_at || metadata.report_updated_date) && <><dt>Dataset/report updated</dt><dd>{source.publisher_updated_at ?? `${metadata.report_updated_date} (date only; ${metadata.update_timezone ?? 'timezone not supplied'})`}</dd></>}
    </dl>
    {source.source_locator && <p className="chart-caption">Native location: {source.source_locator}</p>}
    {safeSource(source.source_url) && <><button className="ce-link" onClick={() => navigator.clipboard.writeText(source.source_url).then(() => setCopied(true)).catch(() => setCopied(false))}>{copied ? 'Source URL copied' : 'Copy original source URL'}</button><p className="ce-url">{source.source_url}</p></>}
    <button className="ce-link" onClick={() => setRaw(!raw)}>{raw ? 'Hide' : 'Show'} complete source record</button>
    {raw && <pre>{JSON.stringify(source, null, 2)}</pre>}
  </div>;
}
function Sources({ envelope, refs }: { envelope: EvidenceEnvelope; refs: string[] }) {
  const [open, setOpen] = useState(false); const ids = [...new Set(refs)];
  if (!ids.length) return null;
  return <div className="ce-source-group"><button className="ce-link" aria-expanded={open} onClick={() => setOpen(!open)}><FileCheck2 size={13} />{open ? 'Hide sources' : `Inspect sources · ${ids.length}`}</button>{open && ids.map(ref => <SourceDetails key={ref} source={envelope.citations[ref]} />)}</div>;
}
function BulletList({ values }: { values?: string[] }) { return values?.length ? <ul>{values.map((text, i) => <li key={i}>{text}</li>)}</ul> : null; }
function NativeChart({ history, annual = false }: { history: EvidenceHistory; annual?: boolean }) {
  const [start, setStart] = useState(annual ? 1990 : 2024), [table, setTable] = useState(false);
  const points = evidenceLines(history, start);
  const any = [...points.historical, ...points.forecast].some(v => v !== null);
  const rows = history.observations.filter(p => Number((p.period ?? p.date ?? String(p.year)).slice(0, 4)) >= start);
  return <div className="ce-history" data-history-indicator={history.indicator}>
    <div className="ce-history-toolbar"><span>{history.unit} · {history.frequency}</span><select aria-label={annual ? 'Annual evidence history period' : 'Monitoring history period'} value={start} onChange={e => setStart(Number(e.target.value))}>{(annual ? [1800, 1990, 2010, 2020] : [1800, 2010, 2020, 2024, 2026]).map(year => <option key={year} value={year}>{year === 1800 ? 'All saved history' : `Since ${year}`}</option>)}</select></div>
    {any ? <Chart height={230} label={`${history.label ?? history.indicator} verified history; ${history.unit}`} option={{ color: ['#315f89', '#7963a4'], tooltip: { trigger: 'axis', valueFormatter: (v: number) => v == null ? 'Not available' : v.toLocaleString('en-GB', { maximumSignificantDigits: 6 }) }, grid: { left: 58, right: 15, top: 18, bottom: 38 }, xAxis: { type: 'category', data: points.labels, boundaryGap: false, axisLabel: { fontSize: 10, color: '#6b7780' } }, yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 10, formatter: (v: number) => v.toLocaleString('en-GB', { notation: 'compact', maximumSignificantDigits: 3 }) }, splitLine: { lineStyle: { color: '#e4e8e8' } } }, series: [{ name: 'Observed / estimate', type: 'line', data: points.historical, showSymbol: false, connectNulls: false, lineStyle: { width: 2 } }, { name: 'Retained forecast', type: 'line', data: points.forecast, showSymbol: false, connectNulls: false, lineStyle: { width: 2, type: 'dashed' } }] }} /> : <p className="empty">No numeric source observations in this window.</p>}
    <p className="chart-caption">Blue: observations/estimates{annual && ' · purple dashed: retained forecast'}. Missing periods stay blank. This is saved-vintage history, not a known-at-the-time backtest.</p>
    <button className="ce-link" onClick={() => setTable(!table)}>{table ? 'Hide' : 'Inspect'} native observations</button>
    {table && <><div className="ce-table-scroll"><table><thead><tr><th>Native period</th><th>Value</th><th>Source status</th></tr></thead><tbody>{rows.slice(-36).reverse().map(row => <tr key={row.period ?? row.date ?? row.year}><td>{evidenceDate(row)}</td><td>{evidenceValue(row, history.unit)}</td><td>{evidenceLabel(row.status)}{nativeStatus(row, history.source) && <small>{nativeStatus(row, history.source)}</small>}</td></tr>)}</tbody></table></div><p className="chart-caption">Latest {Math.min(36, rows.length)} of {rows.length} saved observations in this window. Full history and provenance are included in the offline country file.</p><SourceDetails source={history} /></>}
  </div>;
}
function Monitoring({ envelope }: { envelope: EvidenceEnvelope }) {
  const signals = envelope.profile.signals as any[];
  const [selected, setSelected] = useState(signals[0]?.indicator ?? '');
  const signal = signals.find(s => s.indicator === selected) ?? signals[0];
  const history = envelope.histories.find(s => s.indicator === signal?.indicator);
  return <section className="ce-monitoring"><div className="section-title"><h3><Activity size={16} /> Current monitoring</h3><span className="ce-neutral">{envelope.as_of}</span></div>
    <p className="section-note">Higher or lower describes the recorded change. It is not automatically good or bad, and a lower rate does not establish easier credit access.</p>
    <div className="ce-signals">{signals.map(s => <button key={s.indicator} data-signal={s.indicator} className={s.indicator === signal?.indicator ? 'selected' : ''} onClick={() => setSelected(s.indicator)} aria-pressed={s.indicator === signal?.indicator}>
      <span>{s.label}</span><strong>{evidenceValue(s.latest)}</strong><small>{evidenceDate(s.latest)} · {evidenceLabel(s.status)}</small>{nativeStatus(s.latest, s.source?.source) && <small className="ce-native-status">{nativeStatus(s.latest, s.source?.source)}</small>}<div className="ce-change">{s.direction === 'up' ? <ArrowUp size={14} /> : s.direction === 'down' ? <ArrowDown size={14} /> : <ArrowRight size={14} />}{s.comparison ? `${s.comparison.value > 0 ? '+' : ''}${s.comparison.value.toLocaleString('en-GB', { maximumSignificantDigits: 6 })} ${s.comparison.unit}` : 'No complete comparison'}</div><small>{s.comparison?.window ?? 'Missing or inapplicable window'}</small>
    </button>)}</div>
    {signal && <div className="ce-signal-detail"><h4>{signal.label}</h4><p>{signal.reading}</p><p className="section-note">{signal.definition}</p><p className="ce-dates">Reference-period age: {signal.age_days ?? 'unavailable'} days · editorial limit {signal.freshness_limit_days ?? 'not supplied'} days. This freshness rule is not predictive confidence.</p>
      {history ? <NativeChart key={`${envelope.snapshot_sha256}:${signal.indicator}`} history={{ ...history, label: signal.label }} /> : <p className="empty">No eligible source history for this signal. Older or substitute values are not used.</p>}
      <BulletList values={signal.gaps} />{signal.source_gap && <details><summary>Source documentation for this gap</summary><p>{signal.source_gap.error ?? signal.source_gap.definition}</p>{signal.source_gap.source_urls?.map((url: string) => <SourceDetails key={url} source={{ ...signal.source_gap, source_url: url }} />)}</details>}<details><summary>Interpretation limits and scenario checks</summary><BulletList values={signal.limits} />{signal.scenario_links?.map((link: any) => <div key={link.scenario_id}><h4>{link.title}</h4><p>{link.interpretation}</p><strong>Evidence that would challenge the assumptions</strong><BulletList values={link.evidence_that_challenges_assumption} /><Sources envelope={envelope} refs={link.reference_evidence_refs ?? []} /></div>)}</details>
      <Sources envelope={envelope} refs={signal.evidence_refs ?? []} />
    </div>}
    {!!envelope.input_gaps?.length && <details><summary>Uncollected or ineligible inputs</summary>{envelope.input_gaps.map((gap, i) => <div className="ce-gap" key={i}><p>{evidenceLabel(gap.indicator ?? 'Input')}: {evidenceLabel(gap.reason ?? 'Not available')}</p><pre>{JSON.stringify(gap, null, 2)}</pre></div>)}</details>}
    <details><summary>Remaining evidence gaps</summary><BulletList values={envelope.remaining_gaps} /></details>
  </section>;
}
function Annual({ envelope }: { envelope: EvidenceEnvelope }) {
  const { profile } = envelope;
  const [metric, setMetric] = useState('real_gdp_growth');
  const history = envelope.histories.find(s => s.indicator === metric) ?? envelope.histories[0];
  const metadata = envelope.methodology.metrics ?? {};
  const metrics = Object.keys(profile.baseline);
  return <>
    <section><div className="section-title"><h3>Annual history & IMF outlook</h3><span className="ce-neutral">{envelope.as_of}</span></div>
      <select className="metric-select" aria-label="Country evidence annual indicator" value={history?.indicator ?? ''} onChange={e => setMetric(e.target.value)}>{envelope.histories.map(series => <option key={series.indicator} value={series.indicator}>{series.label}</option>)}</select>
      {history && <NativeChart key={history.indicator} history={history} annual />}
      <p className="section-note">IMF historical values may be estimates/outturns. Current/future-year forecasts use the collector's calendar convention, not native per-point finality flags. World Bank structural inputs keep their historical reference years. General-government ratios and central-government amounts have different scopes.</p>
      <details className="ce-path"><summary>Baseline and dated IMF projection path</summary><div className="ce-table-scroll"><table><thead><tr><th>Year</th>{metrics.map(m => <th key={m}>{metadata[m]?.label ?? evidenceLabel(m)}</th>)}</tr></thead><tbody>{[{ year: envelope.baseline_year, metrics: profile.baseline }, ...profile.projections].map((row: any) => <tr key={row.year}><td>{row.year}</td>{metrics.map(m => <td key={m}>{evidenceValue(row.metrics[m])}</td>)}</tr>)}</tbody></table></div><Sources envelope={envelope} refs={[...Object.values(profile.baseline), ...profile.projections.flatMap((r: any) => Object.values(r.metrics))].flatMap((p: any) => p?.evidence_ref ? [p.evidence_ref] : [])} /></details>
      <div className="ce-structural"><h4>Dated structural context</h4>{Object.entries(profile.structural ?? {}).map(([m, p]: [string, any]) => <div key={m}><span>{metadata[m]?.label ?? evidenceLabel(m)}</span><strong>{evidenceValue(p)}</strong><small>{p?.year ?? 'Not available'}{p?.stale ? ' · outside editorial age window' : ' · historical input'}</small></div>)}</div>
    </section>
    <section><h3>What the saved evidence says</h3>{profile.findings?.map((finding: any) => <details key={finding.id}><summary>{finding.title}</summary><p>{finding.text}</p><BulletList values={finding.limits} /><Sources envelope={envelope} refs={finding.evidence_refs ?? []} /></details>)}</section>
    <section className="ce-scenarios"><div className="section-title"><h3>Conditional scenarios</h3><span className="ce-neutral">Hypotheses</span></div><p className="section-note">These are research cases with explicit assumptions. They carry no probability, automatic company verdict or numerical stress forecast.</p>{profile.scenarios.map((scenario: any) => <details key={scenario.id} data-scenario={scenario.id}><summary>{scenario.title}<small>{scenario.horizon}</small></summary><h4>Assumptions</h4><BulletList values={scenario.assumptions} /><h4>How it could affect a company</h4><ol className="ce-pathway">{scenario.pathway?.map((text: string, i: number) => <li key={i}>{text}</li>)}</ol><h4>What to monitor</h4><BulletList values={scenario.signposts?.map((s: any) => s.text)} /><h4>What would challenge the case</h4><BulletList values={scenario.invalidators} /><h4>Company evidence still required</h4><BulletList values={scenario.company_checks} /><BulletList values={scenario.limitations} /><Sources envelope={envelope} refs={scenario.evidence_refs ?? []} /></details>)}</section>
  </>;
}
function NativeDebt({ points, envelope }: { points: any[]; envelope: EvidenceEnvelope }) {
  if (!points.length) return null;
  return <section><h3>Original national debt context</h3><p className="section-note">Central-government scope. Average time to refixing concerns interest-rate resets, not principal repayment. Reference-date values and monthly means remain distinct.</p>{[false, true].map(forecast => <div key={String(forecast)}><h4>{forecast ? 'Dated funding-plan forecasts' : 'Observed debt-office records'}</h4>{points.filter(p => (p.status === 'forecast') === forecast).map(p => <div className="ce-native" key={p.evidence_ref}><span>{evidenceLabel(p.metric)}</span><strong>{evidenceValue(p)}</strong><small>{evidenceDate(p)} · {evidenceLabel(p.status)} · {evidenceLabel(p.dimensions?.aggregation ?? '')}</small><Sources envelope={envelope} refs={[p.evidence_ref]} /></div>)}</div>)}</section>;
}
export default function CountryEvidencePanel({ index, code, error }: { index?: EvidenceIndex; code: string; error?: string }) {
  const state = useCountryEvidence(index, code);
  if (error || state.error) return <div className="ce-panel validation-error" role="alert">{error || state.error}</div>;
  if (!index) return <div className="empty">Opening bundled country evidence…</div>;
  if (!index.countries[evidenceCode(code)]) return <div className="empty">No annual country assessment is included for this economy. Existing fundamentals and liquidity remain available in their own tabs.</div>;
  if (!state.data) return <div className="empty">Verifying the saved country evidence…</div>;
  const data = state.data, debt = data.monitoring ?? data.assessment;
  return <div className="ce-panel" data-country-evidence={data.country} data-evidence-ready="true" data-evidence-pack={index.id}>
    <div className="ce-intro"><span className="eyebrow">SAVED MACRO EVIDENCE</span><h3>{data.name} · assessment & monitoring</h3><p>Annual assessment: {data.assessment.as_of}{data.monitoring && <> · Monitoring: {data.monitoring.as_of}</>}. This separate pack keeps its own dates when you switch older fundamentals releases.</p><p className="ce-dates">Date basis: UTC. Annual known-at: {data.assessment.as_known_at}{data.monitoring && <><br />Monitoring known-at: {data.monitoring.as_known_at}</>}</p></div>
    {data.monitoring ? <Monitoring key={data.monitoring.snapshot_sha256} envelope={data.monitoring} /> : <div className="ce-gap"><h4>Monitoring has not been collected for this country</h4><p>The annual profile is available below. No Swedish or euro-area national data is substituted.</p></div>}
    <Annual key={data.assessment.snapshot_sha256 + data.country} envelope={data.assessment} />
    <NativeDebt points={debt.profile.national_debt_context ?? []} envelope={debt} />
    <section><h3>Pack provenance</h3><p className="section-note">Values and source metadata are bundled offline. Original response files remain in the macro project's evidence archive; opening publisher websites requires internet. Listing country is not company revenue, asset or financing exposure.</p><dl className="release-details"><dt>Country pack</dt><dd className="hash">{index.id}</dd><dt>Annual source snapshot</dt><dd className="hash">{data.assessment.snapshot_sha256}</dd>{data.monitoring && <><dt>Monitoring source snapshot</dt><dd className="hash">{data.monitoring.snapshot_sha256}</dd></>}</dl></section>
  </div>;
}
