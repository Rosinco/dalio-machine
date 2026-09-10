import { useFinancialIndex } from './financialService';
import './financial.css';
import { lazy, Suspense, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDownToLine, ArrowRight, BookOpen, Check, ChevronDown, Compass, Globe2, Info, Layers3, Search, Share2, TrendingUp, X } from 'lucide-react';
import WorldMap, { type Paint } from './WorldMap';
import { Chart, HistoryChart, Radar } from './Charts';
import { assessmentPalette, atYear, categories, finite, format, historyColor, historyPalette, indicatorDirection, latestTrade, missingColor, quantityPalette, quintile, seriesColors, sourceUrl, tradePalette, tradeSlices } from './model';
import type { AtlasIndex, Category, Country, HistoryPanel, Indicator, Mode, ResearchRelease } from './types';
import { listResearch, previousRelease, resource } from './research';
import type { LiquidityReport } from './liquidity';
import ResearchLibrary from './ResearchLibrary';
import ScoreDetails from './ScoreDetails';
import LiquidityPanel from './LiquidityPanel';
import BusinessWorkspace, { BusinessSearch } from './BusinessWorkspace';
import type { BusinessIndex, Observatory } from './business';
import { useReleaseResource } from './useResearchResource';
import { companyBranch, type Taxonomy } from './taxonomy';
import { companyEntry, listingCountries } from './listingCatalogue';
import './style.css';
import './research.css';
import './business.css';
import './taxonomy.css';
import './listings.css';

const PressureFlow = lazy(() => import('./PressureFlow'));
const modes = [{ id: 'fundamentals', label: 'Fundamentals', icon: Globe2 }, { id: 'history', label: 'History & outlook', icon: TrendingUp }, { id: 'trade', label: 'Trade connections', icon: Share2 }] as const;
const bands = ['Weakest', 'Weaker', 'Middle', 'Stronger', 'Strongest'];
const preferences = (() => { try { return JSON.parse(localStorage.getItem('atlas.preferences') ?? '{}'); } catch { return {}; } })();

function useCountry(code: string, index: AtlasIndex | null, release: ResearchRelease | undefined) {
  const result = useReleaseResource<Country>(index?.countries[code] ? release : undefined, `country:${code}`);
  return { country: result.data ?? index?.countries[code], ready: result.ready && !result.error, error: result.error };
}

export default function App() {
  const [index, setIndex] = useState<AtlasIndex | null>(null);
  const [error, setError] = useState('');
  const [releases, setReleases] = useState<ResearchRelease[]>([]);
  const [release, setRelease] = useState<ResearchRelease>();
  const [unreadable, setUnreadable] = useState(0);
  const switchSequence = useRef(0);
  const [code, setCode] = useState<string>(preferences.code ?? 'SE');
  const [unknownName, setUnknownName] = useState('');
  const [mode, setMode] = useState<Mode>('fundamentals');
  const [observatory, setObservatory] = useState<Observatory>(['macro', 'sectors', 'companies'].includes(preferences.observatory) ? preferences.observatory : 'macro');
  const [companyId, setCompanyId] = useState<string>(/^[1-9][0-9]{0,9}$/.test(preferences.companyId ?? '') ? preferences.companyId : '102');
  const [branchId, setBranchId] = useState<string>(/^(?:[1-9][0-9]{0,9}|unassigned)$/.test(preferences.branchId ?? '') ? preferences.branchId : '21');
  const [category, setCategory] = useState<Category>('production');
  const [metric, setMetric] = useState('gov_debt_pct_gdp');
  const [compare, setCompare] = useState<string>('');
  const [year, setYear] = useState(2025);
  const [startYear, setStartYear] = useState(1990);
  const [tab, setTab] = useState('overview');
  const [query, setQuery] = useState('');
  const [searchOpen, setSearchOpen] = useState(false);
  const [libraryOpen, setLibraryOpen] = useState(false);
  const [message, setMessage] = useState('');
  const [pressureIndex, setPressureIndex] = useState(0);
  const searchRef = useRef<HTMLDivElement>(null);
  const detail = useCountry(code, index, release);
  const other = useCountry(compare, index, release);
  const historyResult = useReleaseResource<HistoryPanel>(release, 'history', mode === 'history');
  const histories = historyResult.data;
  const historyError = historyResult.error;
  const liquidity = useReleaseResource<LiquidityReport | null>(release, 'liquidity', tab === 'liquidity');
  const priorRelease = release ? previousRelease(releases, release) : undefined;
  const prior = useReleaseResource<AtlasIndex>(priorRelease, 'index');
  const country = detail.country;
  const business = useReleaseResource<BusinessIndex>(release, 'business-index', observatory !== 'macro');
  const taxonomy = useReleaseResource<Taxonomy>(release, 'taxonomy', observatory !== 'macro');
  const [financialRevision, setFinancialRevision] = useState(0);
  const financial = useFinancialIndex(release?.taxonomy_sha256, observatory !== 'macro' || libraryOpen, financialRevision);
  useEffect(() => {
    const selected = companyEntry(companyId, business.data, taxonomy.data);
    if (observatory === 'companies' && selected && taxonomy.ready) setBranchId(companyBranch(selected, taxonomy.data) ?? 'unassigned');
  }, [observatory, companyId, business.data, taxonomy.data, taxonomy.ready]);

  const openRelease = async (next: ResearchRelease) => {
    const sequence = ++switchSequence.current;
    const nextIndex = await resource<AtlasIndex>(next, 'index');
    if (nextIndex.version !== 1 || !nextIndex.manifest?.sha256 || !nextIndex.countries?.SE) throw new Error('Unsupported or incomplete research release.');
    if (sequence !== switchSequence.current) return;
    setIndex(nextIndex); setRelease(next); setError(''); setPressureIndex(0);
    setCode(current => /^[A-Z]{2}$/.test(current) ? current : 'SE');
    setCompare(current => nextIndex.countries[current] ? current : '');
    setMetric(current => nextIndex.indicators.some(i => i.name === current) ? current : nextIndex.indicators[0].name);
    try { localStorage.setItem('atlas.release', next.id); } catch { /* The release still opens without persistent preferences. */ }
  };
  const refreshLibrary = async (selected: ResearchRelease) => {
    const saved = await listResearch(); setReleases(saved.releases); setUnreadable(saved.unreadable);
    await openRelease(saved.releases.find(r => r.id === selected.id) ?? selected);
  };
  useEffect(() => {
    let cancelled = false;
    listResearch().then(async saved => {
      if (cancelled) return;
      setReleases(saved.releases); setUnreadable(saved.unreadable);
      let preferred: string | null = null;
      try { preferred = localStorage.getItem('atlas.release'); } catch { /* Use the included release. */ }
      const selected = saved.releases.find(r => r.id === preferred) ?? saved.releases.find(r => r.id === saved.catalogue.default_id) ?? saved.releases[0];
      if (!selected) throw new Error('No research release is available.');
      await openRelease(selected);
      if (preferred && !saved.releases.some(r => r.id === preferred)) setMessage('The previously selected release is unavailable. The included release has been opened.');
    }).catch(e => { if (!cancelled) setError(String(e)); });
    return () => { cancelled = true; switchSequence.current++; };
  }, []);
  useEffect(() => { try { localStorage.setItem('atlas.preferences', JSON.stringify({ code, observatory, companyId, branchId })); } catch { /* View still works without persistent settings. */ } }, [code, observatory, companyId, branchId]);
  useEffect(() => { setPressureIndex(0); }, [code]);
  useEffect(() => { if (compare === code) setCompare(''); }, [compare, code]);
  useEffect(() => { document.querySelector('.sidebar-content')?.scrollTo({ top: 0 }); }, [code, tab]);
  useEffect(() => { if (!message) return; const t = setTimeout(() => setMessage(''), 5000); return () => clearTimeout(t); }, [message]);
  useEffect(() => {
    const dismiss = (e: MouseEvent) => { if (!searchRef.current?.contains(e.target as Node)) setSearchOpen(false); };
    const keys = (e: KeyboardEvent) => { if (e.key === 'Escape') { setLibraryOpen(false); setSearchOpen(false); } };
    document.addEventListener('mousedown', dismiss); document.addEventListener('keydown', keys);
    return () => { document.removeEventListener('mousedown', dismiss); document.removeEventListener('keydown', keys); };
  }, []);

  const countries = useMemo(() => Object.entries(index?.countries ?? {}).sort((a, b) => a[1].name.localeCompare(b[1].name)), [index]);
  const names = useMemo(() => Object.fromEntries(countries.map(([k, c]) => [k, c.name])), [countries]);
  const meta = index?.indicators.find(i => i.name === metric);
  const rows = useMemo(() => latestTrade(index?.trade ?? [], code), [index, code]);
  const slices = useMemo(() => tradeSlices(rows, index?.countries ?? {}), [rows, index]);
  const historyRange = useMemo(() => {
    const values = countries.filter(([, c]) => c.on_map).flatMap(([key]) => { const p = atYear(histories?.[key]?.[metric], year); return p && finite(p.value) ? [p.value] : []; });
    return values.length ? [Math.min(...values), Math.max(...values)] : [0, 1];
  }, [countries, histories, metric, year]);
  const historyBounds = useMemo(() => {
    let first = Infinity, last = -Infinity;
    for (const country of Object.values(histories ?? {})) for (const points of Object.values(country)) for (const point of points) {
      if (!point.is_forecast && finite(point.value)) { first = Math.min(first, point.year); last = Math.max(last, point.year); }
    }
    return Number.isFinite(first) ? [first, last] : [1960, Number(index?.as_of.slice(0,4) ?? 2026) - 1];
  }, [histories, index?.as_of]);
  useEffect(() => { if (histories) setYear(current => Math.max(historyBounds[0], Math.min(historyBounds[1], current))); }, [historyBounds, histories]);
  const direction = indicatorDirection(meta);
  const directionLabel = direction === 'lower' ? 'Lower values = stronger' : direction === 'higher' ? 'Higher values = stronger' : 'Amount only · no good/bad rating';
  const mapColours = mode === 'fundamentals' ? assessmentPalette : mode === 'history' ? historyPalette(meta) : quantityPalette;
  const paint: Paint = useMemo(() => {
    const out: Paint = {};
    countries.filter(([, c]) => c.on_map).forEach(([k, c]) => {
      if (mode === 'fundamentals') {
        const score = c.categories[category]?.score; const q = quintile(score);
        out[k] = { color: q === null ? missingColor : assessmentPalette[q], label: q === null ? 'Not enough data' : `${categories[category].short} · ${bands[q]} band · ${format(score, 0)}/100` };
      } else if (mode === 'history') {
        const value = atYear(histories?.[k]?.[metric], year)?.value;
        out[k] = { color: historyColor(value, historyRange, meta), label: finite(value) ? `${format(value)} ${metric === 'gdp_growth_fwd5' ? '% annual growth' : meta?.unit ?? ''} · ${year} · ${directionLabel}` : `No historical observation for ${year}` };
      } else {
        const value = rows.find(r => r.partner === k)?.x_share;
        out[k] = { color: k === code ? seriesColors.comparison : finite(value) ? quantityPalette[value >= 20 ? 4 : value >= 10 ? 3 : value >= 5 ? 2 : value >= 1 ? 1 : 0] : missingColor, label: k === code ? 'Selected exporter' : finite(value) ? `${format(value)}% of ${country?.name}'s goods exports · ${rows[0]?.year}` : 'No partner value in this release' };
      }
    });
    return out;
  }, [countries, mode, category, histories, metric, year, historyRange, meta, directionLabel, rows, code, country]);

  const selectCountry = (next: string, name = '') => { setCode(next); setUnknownName(name); setQuery(''); setSearchOpen(false); };
  const selectCompany = (id: string) => { const company = companyEntry(id, business.data, taxonomy.data); if (!company) return; setCompanyId(id); setBranchId(companyBranch(company, taxonomy.data) ?? 'unassigned'); selectCountry(company.listing_country ?? 'ZZ', listingCountries(business.data, taxonomy.data)[company.listing_country ?? ''] ?? 'Country unavailable'); setObservatory('companies'); };
  const changeMode = (next: Mode) => { setMode(next); setTab(next === 'trade' ? 'trade' : 'overview'); };
  const exportHistory = async () => {
    if (!country?.history || !meta) return;
    const quote = (s: unknown) => `"${String(s ?? '').replaceAll('"', '""')}"`;
    const csv = [['country', 'indicator', 'year', 'value', 'is_forecast', 'snapshot_as_of', 'source_snapshot_sha256'],
      ...(country.history[metric] ?? []).map(p => [country.name, metric, p.year, p.value, p.is_forecast, index!.as_of, index!.manifest.sha256])].map(r => r.map(quote).join(',')).join('\r\n');
    const filename = `Macro-Atlas-${code}-${metric}.csv`;
    try {
      if ('__TAURI_INTERNALS__' in window) {
        const { invoke } = await import('@tauri-apps/api/core');
        const path = await invoke<string>('export_csv', { filename, contents: csv }); setMessage(`Saved to ${path}`);
      } else {
        const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv;charset=utf-8' }));
        const a = document.createElement('a'); a.href = url; a.download = filename; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); setMessage('History exported as CSV.');
      }
    } catch (e) { setMessage(`Export failed: ${String(e)}`); }
  };

  if (!index || !release) return <div className="startup"><Compass size={42} /><h1>Macro Atlas</h1><p>{error || 'Opening your saved world…'}</p>{error && <button onClick={() => location.reload()}>Try again</button>}</div>;
  const cell = country?.indicators[metric];
  const points = country?.history?.[metric] ?? [];
  const totalIndicators = country ? Object.values(country.indicators).filter(c => finite(c.value)).length : 0;
  const matches = countries.filter(([k, c]) => `${c.name} ${k} ${c.iso3}`.toLowerCase().includes(query.toLowerCase()));
  const pressure = country?.pressures[pressureIndex];

  return <div className="app" data-active-release={release.id} data-active-observatory={observatory}>
    <header className="topbar">
      <div className="brand"><div className="brand-mark"><Compass size={25} strokeWidth={1.3} /></div><div><strong>ATLAS<span> / </span></strong><select className="observatory-select" aria-label="Observatory" value={observatory} onChange={e => { setObservatory(e.target.value as Observatory); setSearchOpen(false); setQuery(''); }}><option value="macro">Macro observatory</option><option value="sectors">Sectors & branches</option><option value="companies">Company observatory</option></select></div></div>
      {observatory === 'macro' ? <div className="search" ref={searchRef}><Search size={16} /><input aria-label="Search countries" placeholder="Find a country…" value={query} onFocus={() => setSearchOpen(true)} onChange={e => { setQuery(e.target.value); setSearchOpen(true); }} onKeyDown={e => { if (e.key === 'Enter' && matches[0]) selectCountry(matches[0][0]); }} /><span className="search-hint">{countries.length} economies</span>
        {searchOpen && <div className="search-results">{matches.map(([k, c]) => <button key={k} onClick={() => selectCountry(k)}><span className="country-code">{k}</span>{c.name}<span className="result-note">{c.on_map ? c.currency : 'Aggregate'}</span></button>)}{!matches.length && <p>No matching country in this data release.</p>}</div>}
      </div> : <BusinessSearch index={business.data} taxonomy={taxonomy.data} financial={financial.data} onCompany={selectCompany} onCountry={selectCountry} />}
      <div className="release"><span className="status-dot" />Offline ready <span className="release-divider">|</span><button className="release-picker" aria-label="Choose research release" onClick={() => setLibraryOpen(true)}>Data release {index.as_of}<ChevronDown size={12} /></button></div>
      <button className="header-icon" aria-label="Open data library" onClick={() => setLibraryOpen(true)}><BookOpen size={19} /></button>
    </header>
    {observatory === 'macro' ? <div className="workspace">
      <nav className="rail" aria-label="Map modes"><div className="rail-label">EXPLORE</div>{modes.map(m => <button key={m.id} className={mode === m.id ? 'active' : ''} aria-label={m.label} aria-pressed={mode === m.id} onClick={() => changeMode(m.id)}><m.icon size={21} strokeWidth={1.5} /><span>{m.id === 'fundamentals' ? 'World' : m.id === 'history' ? 'History' : 'Trade'}</span></button>)}<div className="rail-spacer" /><button onClick={() => setLibraryOpen(true)} aria-label="About this release"><Layers3 size={20} strokeWidth={1.5} /><span>Library</span></button><span className="rail-version">V0.8.0</span></nav>
      <main className="map-panel">
        <div className="map-heading"><div><div className="eyebrow">THE WORLD, IN CONTEXT</div><h1>{mode === 'fundamentals' ? 'World fundamentals' : mode === 'history' ? 'History & outlook' : 'Trade connections'}</h1><p>{mode === 'fundamentals' ? 'Explore the forces shaping each economy.' : mode === 'history' ? 'Follow the data through time, from one saved release.' : `Where ${country?.name ?? 'an economy'} sells its goods.`}</p></div><span className="coverage-pill">{countries.filter(([, c]) => c.on_map).length} countries <span>+ {countries.filter(([, c]) => !c.on_map).map(([, c]) => c.name).join(", ")}</span></span></div>
        <div className="map-filter"><span>{mode === 'fundamentals' ? 'COLOUR BY' : mode === 'history' ? 'INDICATOR' : 'MEASURE'}</span>{mode === 'fundamentals' ? <select aria-label="Map category" value={category} onChange={e => setCategory(e.target.value as Category)}>{index.categories.map(k => <option value={k} key={k}>{categories[k].label}</option>)}</select> : mode === 'history' ? <select aria-label="Map historical indicator" value={metric} onChange={e => setMetric(e.target.value)}>{index.indicators.map(i => <option key={i.name} value={i.name}>{i.name === 'gdp_growth_fwd5' ? 'GDP growth · annual' : i.label}</option>)}</select> : <strong>Share of selected country’s goods exports</strong>}<ChevronDown size={14} /></div>
        <WorldMap paint={paint} selected={code} onSelect={selectCountry} names={names} />
        <div className="map-bottom">
          {mode === 'history' && <div className="time-control"><span className="time-label">OBSERVATION YEAR</span><strong>{year}</strong><input aria-label="Historical year" type="range" min={historyBounds[0]} max={historyBounds[1]} value={year} onChange={e => setYear(Number(e.target.value))} /><span>{historyBounds[0]}–{historyBounds[1]}</span></div>}
          <div className="legend" data-colour-direction={mode === 'fundamentals' ? 'higher' : mode === 'history' ? direction : 'neutral'}>
            <div><strong>{mode === 'fundamentals' ? 'Relative fundamentals' : mode === 'history' ? `${metric === 'gdp_growth_fwd5' ? '% annual growth' : meta?.unit ?? ''} · ${year}` : 'Share of goods exports'}</strong><span>{mode === 'fundamentals' ? 'Red: weaker · yellow: mixed · green: stronger' : mode === 'history' ? directionLabel : `Blue: share only · purple: exporter · ${rows[0]?.year ?? 'no data'}`}</span></div>
            <div className="legend-scale">{mapColours.map((color, i) => <div key={color}><i style={{ background: color }} /><span>{mode === 'fundamentals' ? bands[i] : mode === 'trade' ? ['<1%', '1–5%', '5–10%', '10–20%', '20%+'][i] : format(historyRange[0] + i * (historyRange[1] - historyRange[0]) / 5, 1)}</span></div>)}</div><div className="no-data"><i />No data</div>
          </div>
          <div className="map-footnote"><Info size={12} />{mode === 'fundamentals' ? `Scores use the saved comparison group of ${index.ranking_population.length} countries. The euro area is shown separately.` : mode === 'history' ? historyError || (!histories ? 'Loading saved annual histories…' : `${direction === 'neutral' ? 'Blue shows quantity only.' : 'Colours follow Dalio’s direction, relative to covered countries this year.'} Selected release vintage; missing years stay blank.`) : 'The euro-area partner aggregate is excluded to avoid overlap with its member countries.'}</div>
        </div>
      </main>
      <aside className="sidebar" aria-label="Country details" data-country={code} data-ready={detail.ready}>
        {!country ? <div className="uncovered"><span className="country-badge">{code}</span><h2>{unknownName || code}</h2><p>This country has no data in the current release.</p><p>The map is global; research coverage grows as countries are added to Dalio.</p><button className="primary" onClick={() => selectCountry('SE')}>Explore Sweden <ArrowRight size={15} /></button></div> : <>
          <div className="country-header"><div className="country-badge">{code === 'SE' ? <span className="swedish-flag" /> : code}</div><div><div className="eyebrow">{country.on_map ? 'COUNTRY PROFILE' : 'REGIONAL AGGREGATE'}</div><h2>{country.name}</h2><p>{country.currency || 'Multiple currencies'} <span>·</span> {totalIndicators} indicators available</p></div></div>
          <div className="compare-row"><span>COMPARE WITH</span><select aria-label="Comparison country" value={compare} onChange={e => setCompare(e.target.value)}><option value="">Add a comparison</option>{countries.filter(([k]) => k !== code).map(([k, c]) => <option value={k} key={k}>{c.name}</option>)}</select></div>
          <div className="tabs" role="tablist">{[['overview', 'Overview'], ['indicators', 'Indicators'], ['liquidity', 'Liquidity'], ['trade', 'Trade'], ['evidence', 'Evidence']].map(([k, label]) => <button key={k} role="tab" aria-selected={tab === k || tab === 'score' && k === 'overview'} onClick={() => setTab(k)}>{label}</button>)}</div>
          <div className="sidebar-content">
            {tab === 'overview' && <>
              {mode !== 'history' && <section><div className="section-title"><h3>Fundamentals at a glance</h3><span className="micro">0–100</span></div><p className="section-note">Category scores · snapshot {index.as_of}</p><Radar country={country} comparison={other.country} index={index} /><CountryKey country={country.name} comparison={other.country?.name} /><p className="chart-caption">Inner red rings: weaker · middle yellow: mixed · outer green: stronger.</p>
                <div className="category-list">{index.categories.map(k => { const s = country.categories[k]; const q = quintile(s?.score); return <button key={k} className={category === k && mode === 'fundamentals' ? 'selected' : ''} onClick={() => { setCategory(k); setMode('fundamentals'); setTab('score'); }}><span>{categories[k].label}<small>{s?.n_available ?? 0}/{s?.n_total ?? 0} indicators</small></span><div className="mini-track"><i style={{ width: `${s?.score ?? 0}%`, background: q === null ? missingColor : assessmentPalette[q] }} /></div><strong>{finite(s?.score) ? format(s.score, 0) : '—'}</strong></button>; })}</div><p className="chart-caption">Select a category to see its calculation and changes since the previous release.</p>
              </section>}
              <section><div className="section-title"><h3>Through time</h3><button className="icon-button" aria-label="Export selected history as CSV" disabled={!detail.ready} onClick={exportHistory}><ArrowDownToLine size={15} /></button></div><select className="metric-select" aria-label="Chart indicator" value={metric} onChange={e => setMetric(e.target.value)}>{index.indicators.map(i => <option key={i.name} value={i.name}>{i.name === 'gdp_growth_fwd5' ? 'GDP growth · annual history & forecast' : i.label}</option>)}</select>
                <div className="metric-readout"><strong>{format(mode === 'history' ? atYear(points, year)?.value : cell?.value, 2)}</strong><span>{mode === 'history' && metric === 'gdp_growth_fwd5' ? '% annual growth' : meta?.unit}<small>{mode === 'history' ? `Historical observation · ${year}` : cell?.date ? `${cell.is_forecast ? 'Forecast · ' : ''}${cell.date}` : 'No current value'}</small></span></div>
                <div className="period-buttons">{[1960, 1990, 2010].map(y => <button key={y} aria-pressed={startYear === y} onClick={() => setStartYear(y)}>{y === 1960 ? 'All years' : `Since ${y}`}</button>)}</div>
                {detail.error ? <div className="empty">{detail.error}</div> : !detail.ready ? <div className="empty">Opening saved history…</div> : meta && <HistoryChart points={points} comparison={other.country?.history?.[metric]} name={country.name} otherName={other.country?.name} meta={meta} startYear={startYear} />}
                <CountryKey country={country.name} comparison={other.country?.name} />
                <p className="chart-caption">{directionLabel}{direction !== 'neutral' && ' in Dalio’s framework'}. Line colours identify countries. Solid: history · dashed: forecasts. {metric === 'gdp_growth_fwd5' && 'The headline is a five-year outlook; the line shows annual growth. '}History uses the saved vintage.</p>
              </section>
              {country.cycle && <section><div className="section-title"><h3>Cycle context</h3></div><div className="cycle-grid"><div><small>SHORT CYCLE</small><strong>{country.cycle.short_term_label}</strong><span>Rule-match confidence {format(country.cycle.short_term_confidence * 100, 0)}%</span></div><div><small>LONG CYCLE</small><strong>{country.cycle.long_term_label}</strong><span>Rule-match confidence {format(country.cycle.long_term_confidence * 100, 0)}%</span></div></div></section>}
              <section><div className="section-title"><h3>Pressure pathways</h3><span className="micro">Rule output</span></div>{!pressure ? <p className="section-note">No pressure rules triggered for this economy in the saved snapshot.</p> : <><select className="metric-select" aria-label="Pressure pathway" value={pressureIndex} onChange={e => setPressureIndex(Number(e.target.value))}>{country.pressures.map((p, i) => <option key={p.rule_id} value={i}>{p.title}</option>)}</select><Suspense fallback={<div className="empty">Opening diagram…</div>}><PressureFlow pressure={pressure} /></Suspense><p className="chart-caption">{pressure.uncertainty}. These are modelled pathways.</p></>}</section>
            </>}
            {tab === 'score' && <ScoreDetails index={index} country={country} category={category} previous={prior.data} previousError={prior.error} previousRelease={priorRelease} onBack={() => setTab('overview')} onIndicator={name => { setMetric(name); setMode('history'); setTab('overview'); }} />}
            {tab === 'liquidity' && (liquidity.error ? <p className="validation-error">{liquidity.error}</p> : !liquidity.ready ? <div className="empty">Opening saved liquidity diagnostics…</div> : liquidity.data ? <LiquidityPanel key={`${release.id}:${code}`} report={liquidity.data} code={code} currency={country.currency} name={country.name} /> : <div className="empty">No liquidity report is included in this release. Choose a newer research release from the library.</div>)}
            {tab === 'indicators' && <><div className="section-title"><h3>The underlying indicators</h3><span className="micro">{totalIndicators} available</span></div><p className="section-note">Select an indicator to explore its history. Dates and evidence tiers belong to each observation.</p>{index.categories.map(k => <section key={k}><h4>{categories[k].label}</h4>{index.indicators.filter(i => i.category === k && i.scored).map(i => <IndicatorRow key={i.name} meta={i} country={country} onSelect={() => { setMetric(i.name); setTab('overview'); }} />)}</section>)}</>}
            {tab === 'trade' && <>
              <section><div className="section-title"><h3>Goods export destinations</h3><span className="micro">{rows[0]?.year ?? 'No data'}</span></div><p className="section-note">Top five country partners and all other destinations.</p>{slices.length ? <Chart height={245} label={`${country.name} goods exports by destination`} option={{ color: tradePalette, tooltip: { trigger: 'item', formatter: '{b}: {c}%' }, series: [{ type: 'pie', radius: ['52%', '74%'], center: ['50%', '49%'], itemStyle: { borderColor: '#fafbf7', borderWidth: 3 }, label: { show: false }, data: slices.map(s => ({ ...s, value: Number(s.value.toFixed(2)) })) }], graphic: [{ type: 'text', left: 'center', top: '44%', style: { text: 'GOODS\nEXPORTS', textAlign: 'center', fill: '#68776a', font: '11px Segoe UI', lineHeight: 18 } }] }} /> : <div className="empty">No compatible trade breakdown in this release.</div>}
                <div className="trade-partners">{slices.map((s, i) => <button key={s.name} disabled={!s.code} onClick={() => selectCountry(s.code)}><i style={{ background: tradePalette[i] }} /><span>{s.name}</span><strong>{format(s.value)}%</strong>{s.code && <ArrowRight size={13} />}</button>)}</div>
                <p className="chart-caption">Shares retain the original total-export denominator. Euro-area aggregates are excluded to avoid counting member countries twice. “Other” includes destinations outside this country panel.</p>
              </section>
              <section><div className="section-title"><h3>Exports & imports</h3><span className="micro">US$ billion</span></div>{rows.length > 0 && <Chart height={260} label="Goods exports and imports for five leading export partners" option={{ color: [seriesColors.selected, seriesColors.comparison], tooltip: { trigger: 'axis' }, grid: { left: 88, right: 20, top: 18, bottom: 35 }, legend: { bottom: 0, textStyle: { fontSize: 10 } }, xAxis: { type: 'value', axisLabel: { fontSize: 10 }, splitLine: { lineStyle: { color: '#e5e9e1' } } }, yAxis: { type: 'category', inverse: true, data: rows.slice(0, 5).map(r => names[r.partner] ?? r.partner), axisLabel: { fontSize: 10 }, axisTick: { show: false }, axisLine: { show: false } }, series: [{ name: 'Exports', type: 'bar', data: rows.slice(0, 5).map(r => finite(r.x_usd) ? r.x_usd / 1e9 : null), barMaxWidth: 9 }, { name: 'Imports', type: 'bar', data: rows.slice(0, 5).map(r => finite(r.m_usd) ? r.m_usd / 1e9 : null), barMaxWidth: 9 }] }} />}</section>
            </>}
            {tab === 'evidence' && <>
              <section><div className="section-title"><h3>Evidence & methodology</h3></div><p className="section-note">This profile displays the saved Dalio release from {index.as_of}. Each series can have an older observation date.</p><div className="evidence-card"><small>COUNTRY DATA QUALITY</small><strong>{country.data_quality.flag}</strong><p>{country.data_quality.note || 'No additional country-level note in this release.'}</p></div><h4>How to read the scores</h4><p className="body-note">Indicators are ranked within the {index.ranking_population.length}-country comparison group with direction adjusted so higher is stronger. Category scores combine available indicator percentiles. They describe relative fundamentals; they are not probabilities of a crisis.</p><p className="body-note">The default view has no overall score. Missing observations remain missing. The euro-area aggregate is displayed separately and is excluded from the ranking population.</p><div className="tier-list"><span><b>A</b>Measured</span><span><b>B</b>Model or forecast</span><span><b>C</b>Ordinal or judgment</span></div></section>
              <section><h4>Sources in this profile</h4>{[...new Set(Object.values(country.indicators).map(c => c.source).filter((s): s is string => !!s))].map(s => <div className="source-row" key={s}><span>{s.replaceAll('_', ' ')}</span>{sourceUrl(s) && <button onClick={() => { navigator.clipboard.writeText(sourceUrl(s)!).then(() => setMessage('Source URL copied. Opening the provider website requires internet.')).catch(() => setMessage(sourceUrl(s)!)); }}>Copy URL</button>}</div>)}<p className="chart-caption">Source labels come from the saved snapshot. Historical points do not include individual release dates in this export.</p></section>
              <section><h4>Saved release</h4><dl className="release-details"><dt>Snapshot date</dt><dd>{index.as_of}</dd><dt>Generated</dt><dd>{index.generated_at}</dd><dt>Source</dt><dd>{index.manifest.source_file}</dd><dt>SHA-256</dt><dd className="hash">{index.manifest.sha256}</dd></dl></section>
            </>}
            <div className="sidebar-end"><Compass size={14} />Macro Atlas <span>Saved evidence, connected.</span></div>
          </div>
        </>}
      </aside>
    </div> : <BusinessWorkspace key={release.id} observatory={observatory} index={business.data} taxonomy={taxonomy.data} financial={financial.data} financialReady={financial.ready} financialError={financial.error} ready={business.ready && taxonomy.ready} error={business.error || taxonomy.error} macro={index} release={release} code={code} selectedName={unknownName} companyId={companyId} branchId={branchId} onBranch={setBranchId} onCountry={selectCountry} onCompany={selectCompany} onMacro={() => { setObservatory('macro'); changeMode('fundamentals'); }} onSector={() => { const selected = companyEntry(companyId, business.data, taxonomy.data); if (observatory === 'companies' && selected) setBranchId(companyBranch(selected, taxonomy.data) ?? 'unassigned'); setObservatory('sectors'); }} onLibrary={() => setLibraryOpen(true)} />}
    {message && <div className="toast" role="status"><Check size={16} />{message}<button aria-label="Dismiss message" onClick={() => setMessage('')}><X size={14} /></button></div>}
    {libraryOpen && <ResearchLibrary financial={financial.data} financialReady={financial.ready} financialError={financial.error} onFinancialImported={() => setFinancialRevision(n => n + 1)} releases={releases} active={release} unreadable={unreadable} onUse={openRelease} onImported={refreshLibrary} onClose={() => setLibraryOpen(false)} />}
  </div>;
}

function IndicatorRow({ meta, country, onSelect }: { meta: Indicator; country: Country; onSelect: () => void }) {
  const cell = country.indicators[meta.name];
  return <button className="indicator-row" onClick={onSelect}><div><strong>{meta.label}</strong><small>{cell?.source?.replaceAll('_', ' ')} · {cell?.date || 'Date unavailable'} {cell?.is_forecast && '· Forecast'}</small><p>{meta.description}</p></div><div><strong>{format(cell?.value, 2)}</strong><small>{meta.unit}</small><span className="tier">{cell?.uncertainty ?? meta.uncertainty}</span></div></button>;
}

function CountryKey({ country, comparison }: { country: string; comparison?: string }) {
  return <div className="chart-key"><span><i style={{ background: seriesColors.selected }} />{country}</span>{comparison && <span><i style={{ background: seriesColors.comparison }} />{comparison}</span>}</div>;
}
