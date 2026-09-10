import { useEffect, useMemo, useRef, useState } from 'react';
import { ArrowRight, BookOpen, Building2, ChevronRight, Globe2, Info, Layers3, Search, TrendingUp, Trees } from 'lucide-react';
import WorldMap, { type Paint } from './WorldMap';
import { Chart } from './Charts';
import { amountLabels, financialSeries, formulas, metricLabels, periodLabel, ratioLabels, type BusinessIndex, type Company, type CompanySummary, type Observatory, type SavedStudy } from './business';
import { finite, format, missingColor, quantityPalette, seriesColors } from './model';
import type { AtlasIndex, ResearchRelease } from './types';
import { useReleaseResource } from './useResearchResource';

type View = 'overview' | 'companies' | 'financials' | 'peers' | 'context' | 'research';
const sectorViews = [{ id: 'overview', label: 'Branch overview', short: 'Branch', icon: Trees }, { id: 'companies', label: 'Branch companies', short: 'Companies', icon: Building2 }, { id: 'context', label: 'Branch macro context', short: 'Context', icon: Globe2 }, { id: 'research', label: 'Branch research', short: 'Research', icon: BookOpen }] as const;
const companyViews = [{ id: 'financials', label: 'Company financials', short: 'Financials', icon: TrendingUp }, { id: 'peers', label: 'Company peers', short: 'Peers', icon: Building2 }, { id: 'context', label: 'Company macro context', short: 'Context', icon: Globe2 }, { id: 'research', label: 'Company research', short: 'Research', icon: BookOpen }] as const;

export function BusinessSearch({ index, onCompany, onCountry }: { index: BusinessIndex | null; onCompany: (id: string) => void; onCountry: (code: string, name?: string) => void }) {
  const [query, setQuery] = useState(''), [open, setOpen] = useState(false);
  const container = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const dismiss = (e: MouseEvent) => { if (!container.current?.contains(e.target as Node)) setOpen(false); };
    const escape = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', dismiss); document.addEventListener('keydown', escape);
    return () => { document.removeEventListener('mousedown', dismiss); document.removeEventListener('keydown', escape); };
  }, []);
  const term = query.toLowerCase();
  const companies = Object.values(index?.companies ?? {}).filter(c => `${c.name} ${c.ticker} ${c.isin}`.toLowerCase().includes(term));
  const countries = Object.entries(index?.countries ?? {}).filter(([code, name]) => `${code} ${name}`.toLowerCase().includes(term));
  const choose = (action: () => void) => { action(); setQuery(''); setOpen(false); };
  return <div className="search" ref={container}><Search size={16} /><input aria-label="Search companies or countries" placeholder="Find Holmen, a peer or a country…" value={query} onFocus={() => setOpen(true)} onChange={e => { setQuery(e.target.value); setOpen(true); }} onKeyDown={e => { if (e.key === 'Enter') { if (companies[0]) choose(() => onCompany(companies[0].id)); else if (countries[0]) choose(() => onCountry(...countries[0])); } }} /><span className="search-hint">{Object.keys(index?.companies ?? {}).length} companies</span>
    {open && <div className="search-results">{companies.map(c => <button key={c.id} onClick={() => choose(() => onCompany(c.id))}><span className="country-code">{c.listing_country}</span>{c.name}<span className="result-note">{c.ticker}</span></button>)}{countries.map(([code, name]) => <button key={code} onClick={() => choose(() => onCountry(code, name))}><span className="country-code">{code}</span>{name}<span className="result-note">Country coverage</span></button>)}{!companies.length && !countries.length && <p>No matching company or country in this release.</p>}</div>}
  </div>;
}

export default function BusinessWorkspace({ observatory, index, ready, error, macro, release, code, selectedName, companyId, onCountry, onCompany, onMacro, onSector, onLibrary }: {
  observatory: Exclude<Observatory, 'macro'>; index: BusinessIndex | null; ready: boolean; error: string; macro: AtlasIndex; release: ResearchRelease;
  code: string; selectedName: string; companyId: string; onCountry: (code: string, name?: string) => void; onCompany: (id: string) => void;
  onMacro: () => void; onSector: () => void; onLibrary: () => void;
}) {
  const [view, setView] = useState<View>(observatory === 'companies' ? 'financials' : 'overview');
  useEffect(() => setView(observatory === 'companies' ? 'financials' : 'overview'), [observatory, companyId]);
  useEffect(() => { document.querySelector('.business-content')?.scrollTo({ top: 0 }); }, [view, code, companyId]);
  const summary = index?.companies[companyId];
  const selected = summary?.listing_country === code ? summary : undefined;
  const detail = useReleaseResource<Company>(selected && observatory === 'companies' ? release : undefined, `company:${companyId}`);
  const studies = useReleaseResource<SavedStudy[]>(release, 'business-research', observatory === 'sectors' && view === 'research');
  const names = useMemo(() => ({ ...Object.fromEntries(Object.entries(macro.countries).map(([k, c]) => [k, c.name])), ...index?.countries }), [macro, index]);
  const companies = Object.values(index?.companies ?? {});
  const local = companies.filter(c => c.listing_country === code);
  const countryName = names[code] || selectedName || code;
  const paint: Paint = useMemo(() => Object.fromEntries(Object.entries(index?.countries ?? {}).map(([code]) => {
    const n = Object.values(index!.companies).filter(c => c.listing_country === code).length;
    return [code, { color: n ? quantityPalette[Math.min(n, 4)] : missingColor, label: `${n} selected listings · coverage only · assets not mapped` }];
  })), [index]);
  const views = observatory === 'companies' ? companyViews : sectorViews;
  // An unselected country opens its coverage list; it never inherits another country's company.
  const company = detail.data;
  return <div className="workspace business-workspace" data-observatory={observatory}>
    <nav className="rail" aria-label={`${observatory === 'sectors' ? 'Sector' : 'Company'} views`}><div className="rail-label">EXPLORE</div>{views.map(item => <button key={item.id} className={view === item.id ? 'active' : ''} aria-label={item.label} aria-pressed={view === item.id} onClick={() => setView(item.id)}><item.icon size={21} strokeWidth={1.5} /><span>{item.short}</span></button>)}<div className="rail-spacer" /><button onClick={onLibrary} aria-label="About this release"><Layers3 size={20} /><span>Library</span></button><span className="rail-version">V0.3.0</span></nav>
    <main className="map-panel">
      <div className="map-heading"><div><div className="eyebrow">{observatory === 'sectors' ? 'SECTORS & BRANCHES' : 'COMPANY OBSERVATORY'}</div><h1>{observatory === 'sectors' ? 'The forestry landscape' : selected?.name ?? 'Explore companies'}</h1><p>{observatory === 'sectors' ? 'Connect a branch, its businesses and the wider economy.' : 'Explore saved financial reports and company research.'}</p></div></div>
      <div className="map-filter business-map-filter"><Trees size={15} /><span>{index?.branch.sector_name ?? 'Materials'}</span><ChevronRight size={13} /><strong>{index?.branch.name ?? 'Forestry'}</strong><span>FIRST BRANCH</span></div>
      <WorldMap paint={paint} selected={code} onSelect={onCountry} names={names} />
      <div className="map-bottom business-map-bottom"><div className="legend" data-colour-direction="neutral"><div><strong>Research coverage by listing country</strong><span>Blue shows selected listings · no good/bad rating</span></div></div><div className="country-coverage">{Object.entries(index?.countries ?? {}).map(([country, name]) => <button key={country} aria-pressed={country === code} onClick={() => onCountry(country, name)}><i style={{ background: paint[country]?.color }} />{name}<strong>{companies.filter(c => c.listing_country === country).length}</strong></button>)}</div><div className="map-footnote"><Info size={13} />Plants, owned land and other physical resources have not been mapped. Grey countries have no selected listings in this release.</div></div>
    </main>
    <aside className="sidebar business-sidebar" aria-label={observatory === 'companies' ? 'Company details' : 'Branch details'} data-company={observatory === 'companies' && selected ? companyId : ''} data-business-ready={ready && !!index && (!selected || observatory !== 'companies' || detail.ready && !!company)}>
      {!index ? <div className="uncovered"><Layers3 size={30} /><h2>{error ? 'Research could not open' : !ready ? 'Opening research…' : 'No company research in this release'}</h2><p>{error || (!ready ? 'Loading the saved company catalogue.' : 'This release contains macro research. Choose a release with company coverage from the library.')}</p><button className="primary" onClick={onLibrary}>Open research library <ArrowRight size={15} /></button></div> : <>
        <div className="business-heading"><div className="breadcrumbs"><button onClick={onMacro}>{countryName}</button><ChevronRight size={11} /><button onClick={onSector}>{index.branch.sector_name}</button><ChevronRight size={11} /><button onClick={onSector}>{index.branch.name}</button></div><div className="eyebrow">{observatory === 'sectors' ? 'BRANCH PROFILE' : selected ? `${selected.ticker} · ${selected.listing_country} LISTING` : 'COUNTRY COVERAGE'}</div><h2>{observatory === 'sectors' ? index.branch.name : selected?.name ?? countryName}</h2><p>Börsdata snapshot {index.as_of} <span>·</span> {observatory === 'companies' && selected ? `${selected.report_currency} reporting currency` : `${local.length} selected listings in ${countryName}`}</p></div>
        <div className="business-content">
          {observatory === 'sectors' && view === 'overview' && <>
            <section><h3>Materials → Forestry</h3><p className="business-note">{index.branch.description}</p><div className="value-chain" aria-label="Forestry value chain"><span>Forest & fibre</span><ChevronRight size={14} /><span>Wood, pulp & mills</span><ChevronRight size={14} /><span>Construction, paper & packaging</span></div><p className="chart-caption">Qualitative value chain from the archived branch study · {index.branch.as_of}.</p></section>
            <section><div className="section-title"><h3>Explore {countryName}'s selected companies</h3><span className="micro">{local.length} LISTINGS</span></div><CompanyList companies={local} onCompany={onCompany} /><p className="business-note">This is a starting cohort from your research. Listings locate the selected shares; they do not describe where a company's assets or customers are.</p></section>
            <section><h3>Connect the research</h3><div className="context-actions"><button onClick={() => setView('context')}>Macro context <ArrowRight size={14} /></button><button onClick={() => setView('research')}>Read archived branch research <ArrowRight size={14} /></button></div><p className="business-note">Branch outlook: no refreshed forecast in this release. Saved research and macro observations retain their own dates.</p></section>
          </>}
          {observatory === 'sectors' && view === 'companies' && <><section><h3>Listings in {countryName}</h3><CompanyList companies={local} onCompany={onCompany} /></section><section><h3>The Nordic comparison cohort</h3><CompanyList companies={companies} onCompany={onCompany} /><p className="business-note">One canonical listing per company. Segment differences are shown in the peer view.</p></section></>}
          {observatory === 'companies' && !selected && <section><h3>Choose a company in {countryName}</h3><CompanyList companies={local} onCompany={onCompany} />{!local.length && <p className="business-note">Company coverage grows as saved research is added. Select Sweden or Finland to explore this first cohort.</p>}</section>}
          {observatory === 'companies' && selected && view === 'financials' && (detail.error ? <p role="alert">{detail.error}</p> : company ? <Financials company={company} index={index} /> : <div className="empty">Opening company history…</div>)}
          {observatory === 'companies' && selected && view === 'peers' && <PeerComparison index={index} selected={selected} onCompany={onCompany} />}
          {view === 'context' && (observatory === 'sectors' || selected) && <MacroContext index={index} macro={macro} code={code} countryName={countryName} onMacro={onMacro} />}
          {view === 'research' && (observatory === 'sectors' || selected) && <>
            <section><h3>Archived research</h3><p className="archive-notice">These are saved analyses with their original dates and financial anchors. Older prices, rankings and investment judgments have not been updated by the newer financial snapshot.</p><StudyList studies={observatory === 'sectors' ? studies.data : company?.research} error={observatory === 'sectors' ? studies.error : detail.error} index={index} /></section>
            <section><h3>Sources and interpretation</h3>{index.limitations.map(note => <p key={note} className="business-note">{note}</p>)}<SourceRegister index={index} /></section>
          </>}
          <div className="sidebar-end"><Trees size={14} />Atlas <span>Saved evidence, connected.</span></div>
        </div>
      </>}
    </aside>
  </div>;
}

function CompanyList({ companies, onCompany }: { companies: CompanySummary[]; onCompany: (id: string) => void }) {
  return <div className="company-list">{companies.map(c => <button key={c.id} onClick={() => onCompany(c.id)} aria-label={`Open ${c.name}`}><span className="company-monogram">{c.name.slice(0, 1)}</span><span><strong>{c.name}</strong><small>{c.ticker} · {c.listing_country} · {c.report_currency}</small><span>{c.segments.join(' · ')}</span></span><ArrowRight size={15} /></button>)}</div>;
}

function Financials({ company, index }: { company: Company; index: BusinessIndex }) {
  const [frequency, setFrequency] = useState<'annual' | 'quarterly'>('annual');
  const [metric, setMetric] = useState('revenues');
  const reports = company[frequency], latest = reports.at(-1);
  const series = financialSeries(reports, metric, company.report_currency);
  const unit = metric in ratioLabels ? '%' : `${company.report_currency} million`;
  const latestAnnual = company.annual.at(-1), latestQuarter = company.quarterly.at(-1);
  return <>
    <section><div className="period-overview"><div><small>LATEST FULL YEAR</small><strong>{latestAnnual ? periodLabel(latestAnnual) : 'Unavailable'}</strong><span>Reported {latestAnnual?.report_date ?? 'date unavailable'}</span></div><div><small>LATEST SAVED QUARTER</small><strong>{latestQuarter ? periodLabel(latestQuarter) : 'Unavailable'}</strong><span>Reported {latestQuarter?.report_date ?? 'date unavailable'}</span></div></div><p className="chart-caption">Saved {index.as_of}. Availability differs by company; the snapshot date is not the financial period.</p></section>
    <section><div className="section-title"><h3>Financial history</h3><div className="frequency-buttons" role="group" aria-label="Financial reporting frequency"><button aria-pressed={frequency === 'annual'} onClick={() => setFrequency('annual')}>Annual</button><button aria-pressed={frequency === 'quarterly'} onClick={() => setFrequency('quarterly')}>Quarterly</button></div></div><p className="business-note">{latest ? `${periodLabel(latest)} · ${latest.start} to ${latest.end} · ${latest.currency} million` : 'No saved reports for this frequency.'}</p>
      <div className="financial-cards">{['revenues', 'operating_income', 'free_cash_flow', 'net_debt'].map(key => <div key={key} data-financial={key}><small>{amountLabels[key]}</small><strong>{format(latest?.values[key], 0)}</strong><span>{latest?.currency ?? company.report_currency} million</span></div>)}</div>
      <select className="metric-select" aria-label="Company financial chart" value={metric} onChange={e => setMetric(e.target.value)}>{Object.entries(metricLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
      <FinancialChart labels={series.labels} series={[{ name: metricLabels[metric], values: series.values, color: seriesColors.selected }]} label={`${company.name} ${metricLabels[metric]} ${frequency} history`} unit={unit} />
      <p className="chart-caption">{unit}. {formulas[metric] ?? 'Reported amounts; no inflation adjustment or normalization.'} Gaps stay blank. {frequency === 'quarterly' && 'Quarterly flows are separate periods; stocks are period-end balances. The return proxy is annual only.'}</p>
      <details className="financial-table"><summary>View financial figures</summary><div className="table-scroll"><table><thead><tr><th>Period</th><th>Currency</th><th>{metricLabels[metric]}</th><th>Reported</th></tr></thead><tbody>{[...reports].reverse().map(r => <tr key={`${r.year}-${r.period}`}><td>{periodLabel(r)}</td><td>{metric in ratioLabels ? '%' : `${r.currency} m`}</td><td>{format(r.values[metric], 2)}</td><td>{r.report_date ?? 'Unknown'}</td></tr>)}</tbody></table></div></details>
    </section>
    <section><h3>Profitability through the cycle</h3><FinancialChart labels={financialSeries(company.annual, 'operating_margin', company.report_currency).labels} series={['operating_margin', 'return_on_capital'].map((key, i) => ({ name: ratioLabels[key], values: financialSeries(company.annual, key, company.report_currency).values, color: i ? seriesColors.comparison : seriesColors.selected }))} label={`${company.name} annual operating margin and return on capital proxy`} unit="%" /><p className="business-note">Reported earnings can include forest revaluations and asset transactions. The annual return proxy uses year-end capital; neither line measures normalized earning power.</p></section>
    <section><h3>Reading these figures</h3><p className="business-note">Börsdata FCF retains the provider's definition and does not deduct lease principal and interest. It is not owner earnings.</p><p className="business-note">{company.overlap}</p><p className="chart-caption">Segment context from archived branch research · {company.context_as_of}. Financial sources: {index.sources.find(s => s.id === company.annual_source_id)?.label}; {index.sources.find(s => s.id === company.quarterly_source_id)?.label}.</p></section>
  </>;
}

function FinancialChart({ labels, series, label, unit }: { labels: string[]; series: { name: string; values: (number | null)[]; color: string }[]; label: string; unit: string }) {
  if (!series.some(s => s.values.some(finite))) return <div className="empty">No comparable saved observations for this chart.</div>;
  return <Chart label={label} height={240} option={{ tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${format(v, 2)} ${unit}` }, legend: { bottom: 0, textStyle: { color: '#637467', fontSize: 9 } }, grid: { left: 55, right: 15, top: 20, bottom: 70 }, xAxis: { type: 'category', data: labels, axisLabel: { fontSize: 9, color: '#697b6c' }, axisLine: { lineStyle: { color: '#dce4d8' } }, axisTick: { show: false } }, yAxis: { type: 'value', scale: true, axisLabel: { fontSize: 9, color: '#697b6c', formatter: (v: number) => Math.abs(v) >= 1000 ? `${format(v / 1000, 1)}k` : format(v, 1) }, splitLine: { lineStyle: { color: '#e4e9df', type: 'dashed' } } }, dataZoom: [{ type: 'inside' }, { type: 'slider', bottom: 28, height: 14, showDetail: false, borderColor: '#dce4d8' }], series: series.map(s => ({ type: 'line', name: s.name, data: s.values, connectNulls: false, showSymbol: false, lineStyle: { width: 2, color: s.color }, itemStyle: { color: s.color } })) }} />;
}

function PeerComparison({ index, selected, onCompany }: { index: BusinessIndex; selected: CompanySummary; onCompany: (id: string) => void }) {
  const [metric, setMetric] = useState('operating_margin');
  const companies = Object.values(index.companies);
  return <><section><h3>Compare the same financial period</h3><p className="business-note">{index.common_year ? `FY ${index.common_year} · matching full-year dates across ${companies.length} companies. Ratios are in percent.` : 'No matching full financial year across this cohort. Comparison values remain unavailable.'}</p><select className="metric-select" aria-label="Peer comparison metric" value={metric} onChange={e => setMetric(e.target.value)}>{Object.entries(ratioLabels).map(([key, label]) => <option key={key} value={key}>{label}</option>)}</select>
    {index.common_year && <Chart label={`${ratioLabels[metric]} for forestry peers, FY ${index.common_year}`} height={220} option={{ tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${format(v, 2)}%` }, grid: { left: 105, right: 35, top: 18, bottom: 30 }, xAxis: { type: 'value', axisLabel: { fontSize: 9, formatter: '{value}%' }, splitLine: { lineStyle: { color: '#e3e8de' } } }, yAxis: { type: 'category', inverse: true, data: companies.map(c => c.name), axisLine: { show: false }, axisTick: { show: false }, axisLabel: { fontSize: 10 } }, series: [{ type: 'bar', barMaxWidth: 12, data: companies.map(c => ({ value: c.comparison?.values[metric] ?? null, itemStyle: { color: c.id === selected.id ? seriesColors.selected : seriesColors.comparison } })) }] }} />}
    <p className="chart-caption">{formulas[metric]} Blue: selected company · purple: peers. These are descriptive ratios; no investment rating or winner is assigned.</p><div className="table-scroll"><table className="peer-table"><thead><tr><th>Company</th><th>{ratioLabels[metric]}</th><th>Report currency</th></tr></thead><tbody>{companies.map(c => <tr key={c.id} data-peer={c.id} className={c.id === selected.id ? 'selected' : ''}><td><button onClick={() => onCompany(c.id)}>{c.name}</button></td><td>{format(c.comparison?.values[metric], 2)}{finite(c.comparison?.values[metric]) && '%'}</td><td>{c.comparison?.currency ?? c.report_currency}</td></tr>)}</tbody></table></div></section>
    <section><h3>Where the businesses overlap</h3><p className="business-note">Whole-company figures combine different product mixes. Compare the relevant segments before drawing a conclusion about competitive strength.</p>{companies.map(c => <div className="peer-overlap" key={c.id}><strong>{c.name}</strong><span>{c.segments.join(' · ')}</span><p>{c.overlap}</p></div>)}<p className="chart-caption">Qualitative comparison from saved branch research · {index.branch.as_of}. The newer financial snapshot does not refresh these descriptions.</p></section></>;
}

function MacroContext({ index, macro, code, countryName, onMacro }: { index: BusinessIndex; macro: AtlasIndex; code: string; countryName: string; onMacro: () => void }) {
  const country = macro.countries[code];
  return <><section><h3>{countryName}: macro context</h3><p className="business-note">Macro release {macro.as_of} · branch research {index.branch.as_of} · company data {index.as_of}.</p><p className="business-note">The connections below are questions from the saved research. A listing country is only one part of a multinational company's exposure; these observations do not establish a company tailwind or headwind.</p>{!country && <p className="archive-notice">No {countryName} country profile is included in this Dalio release. No other country's data has been substituted.</p>}</section>
    {index.branch.drivers.map(driver => { const meta = macro.indicators.find(i => i.name === driver.indicator), cell = country?.indicators[driver.indicator]; return <section key={driver.title}><h3>{driver.title}</h3><p className="business-note">{driver.text}</p>{country && meta && cell ? <div className="macro-observation"><small>{meta.label}{cell.is_forecast ? ' · Forecast' : ' · Observation'}</small><strong>{format(cell.value, 2)} <span>{meta.unit}</span></strong><p>{cell.source?.replaceAll('_', ' ') ?? 'Source unavailable'} · {cell.date ?? 'Observation date unavailable'}</p></div> : <p className="chart-caption">This macro observation is unavailable for {countryName} in the selected release.</p>}</section>; })}
    <section><button className="primary" onClick={onMacro}>Open {countryName} in Macro <ArrowRight size={15} /></button><p className="business-note">No refreshed branch forecast is included. Forecast evidence, scenarios and exposure analysis are a later research stage.</p></section></>;
}

function StudyList({ studies, error, index }: { studies: SavedStudy[] | null | undefined; error: string; index: BusinessIndex }) {
  if (error) return <p role="alert">{error}</p>;
  if (!studies) return <div className="empty">Opening archived research…</div>;
  if (!studies.length) return <p className="business-note">No individual deep dive has been included for this company. Segment context remains available in the peer view.</p>;
  return <div className="study-list">{studies.map(study => <details key={study.source_id}><summary><BookOpen size={15} /><span>{study.title}<small>Archived · {study.as_of}</small></span></summary><p className="archive-notice">{study.basis}</p><p className="chart-caption">{index.sources.find(s => s.id === study.source_id)?.path}</p><pre>{study.text}</pre></details>)}</div>;
}

function SourceRegister({ index }: { index: BusinessIndex }) {
  return <details className="business-sources"><summary>Saved source register</summary>{index.sources.map(source => <dl key={source.id}><dt>{source.label}</dt><dd>{source.path}</dd><dd className="hash">SHA-256 {source.sha256}</dd></dl>)}<p className="chart-caption">Source hashes identify the saved input files. The portable package contains selected report rows and archived text, not the complete source Parquet files.</p></details>;
}
