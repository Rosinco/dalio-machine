import { useEffect, useMemo, useState } from 'react';
import { ArrowRight, BookOpen, Building2, ChevronRight, Globe2, Info, Layers3, ListTree, TrendingUp, Trees } from 'lucide-react';
import WorldMap, { type Paint } from './WorldMap';
import { Chart } from './Charts';
import { amountLabels, financialSeries, formulas, metricLabels, periodLabel, ratioLabels, type BusinessIndex, type Company, type CompanySummary, type Observatory, type SavedStudy } from './business';
import { finite, format, missingColor, quantityPalette, seriesColors } from './model';
import type { AtlasIndex, ResearchRelease } from './types';
import { useReleaseResource } from './useResearchResource';
import { companyBranch, type Taxonomy } from './taxonomy';
import { companyEntries, countBranches, listingCountries, presenceMatches, type Presence } from './listingCatalogue';
import { BranchCoverage, BranchInventory, ClassificationDetails, TaxonomyDirectory } from './TaxonomyDirectory';
import { CompanyList, ListingDetails } from './CompanyDirectory';
export { DirectorySearch as BusinessSearch } from './CompanyDirectory';

type View = 'browse' | 'overview' | 'companies' | 'financials' | 'peers' | 'context' | 'research';
const sectorViews = [{ id: 'browse', label: 'Browse sectors and branches', short: 'Browse', icon: ListTree }, { id: 'overview', label: 'Branch overview', short: 'Branch', icon: Trees }, { id: 'companies', label: 'Branch companies', short: 'Companies', icon: Building2 }, { id: 'context', label: 'Branch macro context', short: 'Context', icon: Globe2 }, { id: 'research', label: 'Branch research', short: 'Research', icon: BookOpen }] as const;
const companyViews = [{ id: 'financials', label: 'Company financials', short: 'Profile', icon: TrendingUp }, { id: 'peers', label: 'Company peers', short: 'Peers', icon: Building2 }, { id: 'context', label: 'Company macro context', short: 'Context', icon: Globe2 }, { id: 'research', label: 'Company research', short: 'Research', icon: BookOpen }] as const;

export default function BusinessWorkspace({ observatory, index, taxonomy, ready, error, macro, release, code, selectedName, companyId, branchId, onBranch, onCountry, onCompany, onMacro, onSector, onLibrary }: {
  observatory: Exclude<Observatory, 'macro'>; index: BusinessIndex | null; ready: boolean; error: string; macro: AtlasIndex; release: ResearchRelease;
  taxonomy: Taxonomy | null; branchId: string; onBranch: (id: string) => void;
  code: string; selectedName: string; companyId: string; onCountry: (code: string, name?: string) => void; onCompany: (id: string) => void;
  onMacro: () => void; onSector: () => void; onLibrary: () => void;
}) {
  const [view, setView] = useState<View>(observatory === 'companies' ? 'financials' : 'browse');
  const [presence, setPresence] = useState<Presence>('all'), [allCountries, setAllCountries] = useState(false);
  useEffect(() => setView(observatory === 'companies' ? 'financials' : 'browse'), [observatory, companyId]);
  useEffect(() => { document.querySelector('.business-content')?.scrollTo({ top: 0 }); }, [view, code, companyId, branchId]);
  const entries = useMemo(() => companyEntries(index, taxonomy), [index, taxonomy]);
  const entry = entries.find(e => e.id === companyId && (e.listing_country ?? 'ZZ') === code);
  const selected = entry?.profile;
  const latest = taxonomy?.catalogue?.as_of ?? index?.as_of ?? '';
  const scoped = useMemo(() => entries.filter(e => presenceMatches(e, presence, latest)), [entries, presence, latest]);
  const counts = useMemo(() => countBranches(scoped, taxonomy), [scoped, taxonomy]);
  const activeBranchId = observatory === 'companies' && entry ? companyBranch(entry, taxonomy) ?? 'unassigned' : branchId === 'unassigned' ? branchId : taxonomy ? (taxonomy.branches[branchId] ? branchId : index?.branch.id ?? '21') : index?.branch.id ?? branchId;
  const branch = taxonomy?.branches[activeBranchId];
  const branchName = branch?.name_en ?? (activeBranchId === 'unassigned' ? 'Unassigned branch' : index?.branch.name ?? 'Branches');
  const sectorName = branch ? taxonomy!.sectors[branch.sector_id].name_en : activeBranchId === 'unassigned' ? 'Classification review' : index?.branch.sector_name ?? 'Sectors';
  const hasBranchResearch = !!index && activeBranchId === index.branch.id;
  const detail = useReleaseResource<Company>(selected && observatory === 'companies' ? release : undefined, `company:${companyId}`);
  const studies = useReleaseResource<SavedStudy[]>(release, 'business-research', observatory === 'sectors' && hasBranchResearch && view === 'research');
  const countries = useMemo(() => listingCountries(index, taxonomy), [index, taxonomy]);
  const names = useMemo(() => ({ ...Object.fromEntries(Object.entries(macro.countries).map(([k, c]) => [k, c.name])), ...countries }), [macro, countries]);
  const companies = useMemo(() => observatory === 'sectors' ? scoped.filter(e => (companyBranch(e, taxonomy) ?? 'unassigned') === activeBranchId) : scoped, [scoped, taxonomy, activeBranchId, observatory]);
  const local = useMemo(() => companies.filter(c => (c.listing_country ?? 'ZZ') === code), [companies, code]);
  const peers = useMemo(() => scoped.filter(e => companyBranch(e, taxonomy) === (entry ? companyBranch(entry, taxonomy) : null)), [scoped, taxonomy, entry]);
  const countryName = names[code] || selectedName || code;
  const countryCounts = useMemo(() => { const result: Record<string, number> = {}; for (const e of companies) { const c = e.listing_country ?? 'ZZ'; result[c] = (result[c] ?? 0) + 1; } return result; }, [companies]);
  const paint: Paint = useMemo(() => { const maximum = Math.max(1, ...Object.values(countryCounts)); return Object.fromEntries(Object.entries(countryCounts).filter(([c]) => c !== 'ZZ').map(([c, n]) => [c, { color: n ? quantityPalette[Math.min(4, Math.floor(4 * Math.log2(n + 1) / Math.log2(maximum + 1)))] : missingColor, label: `${n.toLocaleString('en-US')} company listings · directory coverage · assets not mapped` }])); }, [countryCounts]);
  const views = observatory === 'companies' ? companyViews : sectorViews;
  const company = detail.data;
  const chooseBranch = (id: string) => { onBranch(id); setView('overview'); };
  const browse = () => { onSector(); setView('browse'); };
  const canShow = !!index || !!taxonomy?.catalogue;
  const hasDetails = !selected || observatory !== 'companies' || detail.ready && !!company;
  return <div className="workspace business-workspace" data-observatory={observatory} data-listing-count={entries.length}>
    <nav className="rail" aria-label={`${observatory === 'sectors' ? 'Sector' : 'Company'} views`}><div className="rail-label">EXPLORE</div>{views.map(item => <button key={item.id} className={view === item.id ? 'active' : ''} aria-label={item.label} aria-pressed={view === item.id} onClick={() => setView(item.id)}><item.icon size={21} strokeWidth={1.5} /><span>{item.short}</span></button>)}<div className="rail-spacer" /><button onClick={onLibrary} aria-label="About this release"><Layers3 size={20} /><span>Library</span></button><span className="rail-version">V0.5.0</span></nav>
    <main className="map-panel">
      <div className="map-heading business-map-heading"><div><div className="eyebrow">{observatory === 'sectors' ? 'SECTORS & BRANCHES' : 'COMPANY OBSERVATORY'}</div><h1>{observatory === 'sectors' ? branchName : entry?.display_name ?? 'Explore companies'}</h1><p>{observatory === 'sectors' ? 'Connect a branch, its businesses and the wider economy.' : 'Explore your saved company directory and financial research.'}</p></div></div>
      <button className="map-filter business-map-filter branch-breadcrumb" aria-label="Choose a sector or branch" onClick={browse}><Trees size={15} /><span>{sectorName}</span><ChevronRight size={13} /><strong>{branchName}</strong><span>BROWSE</span></button>
      <WorldMap paint={paint} selected={code} onSelect={onCountry} names={names} />
      <div className="map-bottom business-map-bottom"><div className="legend" data-colour-direction="neutral"><div><strong>Company listings by Börsdata country</strong><span>Blue shows directory coverage · no good/bad rating</span></div></div>
        <div className="listing-map-controls"><label>Listing country<select aria-label="Company listing country" value={code} onChange={e => onCountry(e.target.value, names[e.target.value])}>{Object.entries({ ...countries, [code]: countryName, ...(countryCounts.ZZ ? { ZZ: 'Country unavailable' } : {}) }).sort((a, b) => a[1].localeCompare(b[1])).map(([c, name]) => <option key={c} value={c}>{name} · {(countryCounts[c] ?? 0).toLocaleString('en-US')}</option>)}</select></label>{taxonomy?.catalogue && <label>Downloaded records<select aria-label="Company snapshot coverage" value={presence} onChange={e => setPresence(e.target.value as Presence)}><option value="all">All saved listings</option><option value="latest">In newest download · {latest}</option><option value="older">Older download only</option></select></label>}<span className="coverage-total" data-country-listing-count={countryCounts[code] ?? 0}>{(countryCounts[code] ?? 0).toLocaleString('en-US')} in {countryName}</span></div>
        <div className="map-footnote"><Info size={13} />Plants, owned land and other physical resources have not been mapped. Grey countries have no listings in this selection.</div>
      </div>
    </main>
    <aside className="sidebar business-sidebar" aria-label={observatory === 'companies' ? 'Company details' : 'Branch details'} data-branch={activeBranchId} data-company={observatory === 'companies' && entry ? companyId : ''} data-business-ready={ready && !error && canShow && hasDetails}>
      {error || !ready ? <div className="uncovered"><Layers3 size={30} /><h2>{error ? 'Research could not open' : 'Opening research…'}</h2><p>{error || 'Loading the saved company and branch catalogue.'}</p></div>
      : observatory === 'sectors' && view === 'browse' && taxonomy ? <div className="business-content directory-content"><TaxonomyDirectory taxonomy={taxonomy} index={index} counts={counts} listingTotal={scoped.length} branchId={activeBranchId} onBranch={chooseBranch} /></div>
      : !canShow ? <div className="uncovered"><Layers3 size={30} /><h2>No company research in this release</h2><p>Choose a release with a company directory or financial profiles.</p><button className="primary" onClick={onLibrary}>Open research library <ArrowRight size={15} /></button></div>
      : observatory === 'sectors' && view === 'browse' ? <div className="uncovered"><ListTree size={30} /><h2>No full branch directory in this release</h2><p>The saved {index?.branch.name} cohort is still available. Choose a newer release to browse the complete directory.</p><button className="primary" onClick={() => setView('overview')}>Open saved branch <ArrowRight size={15} /></button></div>
      : <>
        <div className="business-heading"><div className="breadcrumbs"><button onClick={onMacro}>{countryName}</button><ChevronRight size={11} /><button onClick={browse}>{sectorName}</button><ChevronRight size={11} /><button onClick={browse}>{branchName}</button></div><div className="eyebrow">{observatory === 'sectors' ? 'BRANCH PROFILE' : entry ? `${entry.ticker ?? entry.id} · ${entry.listing_country ?? 'UNMAPPED'} LISTING` : 'COUNTRY DIRECTORY'}</div><h2>{observatory === 'sectors' ? branchName : entry?.display_name ?? countryName}</h2>{observatory === 'sectors' && branch && <span className="branch-local-name">{branch.name_sv} · Börsdata branch {branch.id}</span>}<p>{observatory === 'companies' && entry ? `Börsdata snapshot ${entry.source_as_of} · ${entry.report_currency ?? 'Unavailable'} reporting currency` : `Directory inventory ${taxonomy?.as_of ?? index?.as_of} · ${local.length.toLocaleString('en-US')} listings in ${countryName}`}</p></div>
        {observatory === 'companies' && entry && <ClassificationDetails taxonomy={taxonomy} company={entry} />}
        <div className="business-content">
          {observatory === 'sectors' && view === 'overview' && <>
            {taxonomy && branch && <BranchCoverage taxonomy={taxonomy} branch={branch} profiles={counts[branch.id]?.profiles ?? 0} listings={companies.length} />}
            {hasBranchResearch && index && <section><h3>{sectorName} → {branchName}</h3><p className="business-note">{index.branch.description}</p><div className="value-chain" aria-label="Forestry value chain"><span>Forest & fibre</span><ChevronRight size={14} /><span>Wood, pulp & mills</span><ChevronRight size={14} /><span>Construction, paper & packaging</span></div><p className="chart-caption">Qualitative value chain from the archived branch study · {index.branch.as_of}.</p></section>}
            <section><div className="section-title"><h3>Companies listed in {countryName}</h3><span className="micro">{local.length.toLocaleString('en-US')} LISTINGS</span></div><CompanyList companies={local} taxonomy={taxonomy} onCompany={onCompany} /><p className="business-note">Each Börsdata ID has its own directory entry. Separate listings can represent the same business.</p></section>
            <section><h3>Connect the research</h3><div className="context-actions"><button onClick={() => setView('companies')}>Browse all listings in this branch <ArrowRight size={14} /></button><button onClick={() => setView('context')}>Macro context <ArrowRight size={14} /></button><button onClick={() => setView('research')}>{hasBranchResearch ? 'Read archived branch research' : 'View project research inventory'} <ArrowRight size={14} /></button></div><p className="business-note">Saved research retains its own dates. No refreshed branch forecast is included.</p></section>
          </>}
          {observatory === 'sectors' && view === 'companies' && <section><h3>{branchName}: company listings</h3><label className="business-note"><input type="checkbox" aria-label="Show all listing countries" checked={allCountries} onChange={e => setAllCountries(e.target.checked)} /> Show all listing countries</label><CompanyList companies={allCountries ? companies : local} taxonomy={taxonomy} onCompany={onCompany} /></section>}
          {observatory === 'companies' && !entry && <section><h3>Choose a company in {countryName}</h3><CompanyList companies={local} taxonomy={taxonomy} onCompany={onCompany} /></section>}
          {observatory === 'companies' && entry && view === 'financials' && (selected && index ? detail.error ? <p role="alert">{detail.error}</p> : company ? <Financials company={company} index={index} /> : <div className="empty">Opening company history…</div> : taxonomy?.catalogue ? <ListingDetails entry={entry} taxonomy={taxonomy} onSector={browse} onMacro={onMacro} /> : null)}
          {observatory === 'companies' && entry && view === 'peers' && (selected && index ? <PeerComparison index={index} selected={selected} onCompany={onCompany} /> : <section><h3>Other listings in this branch</h3><p className="business-note">These share a Börsdata branch assignment. Detailed competitor relationships and comparable financial analysis can be added as the research grows.</p><CompanyList companies={peers} taxonomy={taxonomy} onCompany={onCompany} /></section>)}
          {view === 'context' && (observatory === 'sectors' || entry) && (hasBranchResearch && index ? <MacroContext index={index} macro={macro} code={code} countryName={countryName} onMacro={onMacro} /> : <section><h3>Branch context is not included yet</h3><p className="business-note">This release has no saved macro exposure analysis for {branchName}. Open the country observatory to inspect the available macro data.</p><button className="primary" onClick={onMacro}>Open {countryName} in Macro <ArrowRight size={15} /></button></section>)}
          {view === 'research' && (observatory === 'sectors' || entry) && <>
            {index && (observatory === 'companies' && selected || observatory === 'sectors' && hasBranchResearch) && <><section><h3>Archived research</h3><p className="archive-notice">These are saved analyses with their original dates and financial anchors. Older prices, rankings and investment judgments have not been updated by the newer financial snapshot.</p><StudyList studies={observatory === 'sectors' ? studies.data : company?.research} error={observatory === 'sectors' ? studies.error : detail.error} index={index} /></section><section><h3>Sources and interpretation</h3>{index.limitations.map(note => <p key={note} className="business-note">{note}</p>)}<SourceRegister index={index} /></section></>}
            {taxonomy && branch && (observatory === 'sectors' || !selected) && <BranchInventory taxonomy={taxonomy} branch={branch} />}
          </>}
          <div className="sidebar-end"><Trees size={14} />Atlas <span>Saved evidence, connected.</span></div>
        </div>
      </>}
    </aside>
  </div>;
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
