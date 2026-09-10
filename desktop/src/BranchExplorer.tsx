import { useEffect, useMemo, useState } from 'react';
import { ArrowLeft, ArrowRight, Pause, Play, Save, X } from 'lucide-react';
import BranchCharts from './BranchCharts';
import { benchmark, branchMetrics, comparisonColors, defaultSettings, metricDefinition, metricUnit, needsCurrency, observation, type BranchMetric, type BranchSettings, type BubbleSize } from './branchComparison';
import { compatibleComparison, loadComparisons, removeComparison, saveComparison, type SavedComparison } from './savedComparisons';
import type { FinancialIndex } from './financialData';
import { useBranchAnnual } from './financialService';
import { normalizeSearch, presenceMatches, type CompanyEntry, type Presence } from './listingCatalogue';
import { format } from './model';
import type { ResearchRelease } from './types';
import type { Taxonomy } from './taxonomy';
import './comparison.css';

const monthNames = ['All closing months', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];
export default function BranchExplorer({ entries, financial, financialReady, financialError, release, taxonomy, branchId, branchName, onBranch, onBrowse, onCompany }: {
  entries: CompanyEntry[]; financial: FinancialIndex | null; release: ResearchRelease; taxonomy: Taxonomy | null;
  financialReady: boolean; financialError: string;
  branchId: string; branchName: string; onBranch: (id: string) => void; onBrowse: () => void; onCompany: (id: string) => void;
}) {
  const ids = useMemo(() => entries.map(c => c.id), [entries]);
  const annual = useBranchAnnual(financial, ids);
  const last = Math.min(Number(financial?.as_of.slice(0, 4) ?? 2026) - 1, Math.max(2000, ...entries.map(c => c.financial?.annual.last ?? 2000)));
  const [s, setSettings] = useState<BranchSettings>(() => {
    const selected = entries.filter(c => c.financial?.annual.count).slice(0, 5).map(c => c.id);
    return { ...defaultSettings(last), size: financial?.market ? 'market_cap' : 'equal', selected, focus: selected[0] ?? null };
  });
  const [sizeChosen, setSizeChosen] = useState(false);
  useEffect(() => {
    if (financial?.market && !sizeChosen) setSettings(previous => ({ ...previous, size: 'market_cap' }));
  }, [financial?.id, financial?.market, sizeChosen]);
  const [query, setQuery] = useState(''), [playing, setPlaying] = useState(false);
  const [title, setTitle] = useState(`${branchName} through the cycle`), [notes, setNotes] = useState('');
  const [saved, setSaved] = useState<SavedComparison[]>([]), [message, setMessage] = useState(''), [saveError, setSaveError] = useState('');
  useEffect(() => { try { setSaved(loadComparisons(localStorage)); } catch (e) { setSaveError(String(e)); } }, []);
  useEffect(() => {
    if (!playing) return;
    if (s.year >= s.to) { setPlaying(false); return; }
    const timer = window.setTimeout(() => setSettings(previous => ({ ...previous, year: previous.year + 1 })), 1000);
    return () => window.clearTimeout(timer);
  }, [playing, s.year, s.to]);
  const latest = taxonomy?.catalogue?.as_of ?? financial?.as_of ?? '';
  const eligible = useMemo(() => entries.filter(c => presenceMatches(c, s.presence, latest) && (s.country === 'all' || (c.listing_country ?? 'ZZ') === s.country)), [entries, s.presence, s.country, latest]);
  const eligibleIds = useMemo(() => eligible.map(c => c.id), [eligible]);
  const currencies = useMemo(() => [...new Set(entries.flatMap(c => c.financial?.currencies ?? []))].sort(), [entries]);
  const countries = [...new Set(entries.map(c => c.listing_country ?? 'ZZ'))].sort();
  const countryName = (code: string) => Object.values(taxonomy?.catalogue?.countries ?? {}).find(c => c.iso2 === code)?.name_en ?? code;
  const preferredCurrency = currencies.includes('SEK') ? 'SEK' : currencies.includes('EUR') ? 'EUR' : currencies[0] ?? 'SEK';
  const update = (patch: Partial<BranchSettings>) => {
    if (patch.size !== undefined) setSizeChosen(true);
    setPlaying(false); setMessage('');
    setSettings(previous => {
      const next = { ...previous, ...patch };
      if (needsCurrency(next) && next.currency === 'all') next.currency = preferredCurrency;
      if (next.from > next.to) { if (patch.from !== undefined) next.to = next.from; else next.from = next.to; }
      next.year = Math.max(next.from, Math.min(next.to, next.year));
      return next;
    });
  };
  const toggle = (id: string) => {
    const selected = s.selected.includes(id) ? s.selected.filter(v => v !== id) : [...s.selected, id].slice(0, 8);
    update({ selected, focus: selected.includes(s.focus ?? '') ? s.focus : selected[0] ?? null });
  };
  const stats = useMemo(() => annual.data ? benchmark(annual.data, eligibleIds, s) : [], [annual.data, eligibleIds, s]);
  const current = stats.find(r => r.year === s.year);
  const yearLimits = { min: Math.min(2000, s.from), max: Math.max(last, ...entries.map(c => c.financial?.annual.last ?? 2000), s.to) };
  const years = Array.from({ length: yearLimits.max - yearLimits.min + 1 }, (_, i) => yearLimits.min + i);
  const found = eligible.filter(c => c.search.includes(normalizeSearch(query)) && !s.selected.includes(c.id));
  const persist = () => {
    if (!financial || !release.taxonomy_sha256) return;
    try {
      const view: SavedComparison = { id: crypto.randomUUID(), title: title.trim(), notes, created: new Date().toISOString(), release: release.id, financial: financial.id, taxonomy: release.taxonomy_sha256, branch: branchId, settings: s };
      saveComparison(localStorage, view); setSaved(loadComparisons(localStorage)); setSaveError(''); setMessage(`Saved “${view.title}” with this data version.`);
    } catch (e) { setSaveError(String(e instanceof Error ? e.message : e)); setMessage(''); }
  };
  const openSaved = (view: SavedComparison) => {
    if (!financial || !compatibleComparison(view, release.id, financial.id, release.taxonomy_sha256 ?? '', ids)) return;
    setSizeChosen(true);
    setPlaying(false); setSettings(structuredClone(view.settings)); setTitle(view.title); setNotes(view.notes); setMessage(`Opened “${view.title}” from ${view.created.slice(0, 10)}.`);
  };
  return <main className="branch-explorer" aria-label="Branch comparison workspace" data-comparison-branch={branchId} data-comparison-ready={annual.ready && !!annual.data && !annual.error} data-comparison-year={s.year} data-comparison-focus={s.focus ?? ''}>
    <header className="comparison-heading"><div><button className="comparison-back" onClick={onBrowse}><ArrowLeft size={14} />Sector directory</button><div className="eyebrow">BRANCH OBSERVATORY / COMPARE</div><h1>{branchName} through the cycle</h1><p>Follow businesses over time and compare them with the saved branch universe.</p></div>
      <label>Branch<select aria-label="Comparison branch" value={branchId} onChange={e => onBranch(e.target.value)}>{Object.values(taxonomy?.branches ?? {}).sort((a, b) => a.name_en.localeCompare(b.name_en)).map(b => <option value={b.id} key={b.id}>{b.name_en}</option>)}{branchId === 'unassigned' && <option value="unassigned">Unassigned branch</option>}</select></label>
    </header>
    {!financialReady ? <div className="comparison-empty" role="status">Opening saved financial coverage…</div> : financialError ? <div className="comparison-empty" role="alert">{financialError}</div> : !financial ? <div className="comparison-empty">No matching financial history pack is available for this release. Open the Library to select a release with saved company histories.</div>
    : <>
      <div className="comparison-filters">
        <label>Y-axis<select aria-label="Branch comparison metric" value={s.metric} onChange={e => update({ metric: e.target.value as BranchMetric })}>{Object.entries(branchMetrics).map(([id, label]) => <option value={id} key={id} disabled={id === 'market_cap' && !financial.market}>{label}</option>)}<option disabled>ROIC · verified series not yet included</option><option disabled>CAPEX · separate series not yet included</option></select></label>
        <label>Bubble area<select aria-label="Branch bubble size" value={s.size} onChange={e => update({ size: e.target.value as BubbleSize })}><option value="equal">Equal-size points</option><option value="total_assets">Total assets</option><option value="revenues">Revenue</option><option value="market_cap" disabled={!financial.market}>Derived market cap · SEK</option></select></label>
        <label>Reporting currency<select aria-label="Comparison reporting currency" value={s.currency} onChange={e => update({ currency: e.target.value })}><option value="all" disabled={needsCurrency(s)}>All currencies · ratios / SEK market cap</option>{currencies.map(c => <option value={c} key={c}>{c}</option>)}</select></label>
        <label>Listing country<select aria-label="Comparison listing country" value={s.country} onChange={e => update({ country: e.target.value })}><option value="all">All listing countries</option>{countries.map(c => <option value={c} key={c}>{countryName(c)}</option>)}</select></label>
        <label>Directory coverage<select aria-label="Comparison snapshot coverage" value={s.presence} onChange={e => update({ presence: e.target.value as Presence })}><option value="all">All saved listings</option><option value="latest">In newest download · {latest}</option><option value="older">Older download only</option></select></label>
        <label>Fiscal year ends<select aria-label="Comparison closing month" value={s.month} onChange={e => update({ month: +e.target.value })}>{monthNames.map((name, i) => <option value={i} key={i}>{name}</option>)}</select></label>
      </div>
      <div className="comparison-body"><div className="comparison-plots">
        <div className="comparison-time"><label>From<select aria-label="Comparison first year" value={s.from} onChange={e => update({ from: +e.target.value })}>{years.map(y => <option key={y}>{y}</option>)}</select></label><label>To<select aria-label="Comparison last year" value={s.to} onChange={e => update({ to: +e.target.value })}>{years.map(y => <option key={y}>{y}</option>)}</select></label><button aria-label={playing ? 'Pause comparison years' : 'Play comparison years'} aria-pressed={playing} disabled={s.from === s.to || !annual.data} onClick={() => { if (!playing && s.year === s.to) setSettings({ ...s, year: s.from }); setPlaying(!playing); }}>{playing ? <Pause size={16} /> : <Play size={16} />}</button><input aria-label="Comparison selected year" type="range" min={s.from} max={s.to} value={s.year} onChange={e => update({ year: +e.target.value })} /><strong>FY {s.year}</strong></div>
        {!annual.ready ? <div className="comparison-empty" role="status">Opening annual histories · {annual.loaded.toLocaleString('en-US')} of {entries.filter(c => c.financial?.annual.count).length.toLocaleString('en-US')} listings checked…</div>
        : annual.error ? <div className="comparison-empty" role="alert">{annual.error} No partial branch median is displayed.</div>
        : annual.data && <>
          <div className="comparison-readout" data-benchmark-n={current?.n ?? 0}><div><small>VALID LISTINGS · FY {s.year}</small><strong>{current?.n ?? 0}<span> / {eligible.length.toLocaleString('en-US')}</span></strong></div><div><small>BRANCH MEDIAN</small><strong>{format(current?.median, 2)}<span> {metricUnit(s.metric, s.currency)}</span></strong></div><div><small>MIDDLE 50%</small><strong>{current?.q1 !== null && current?.q1 !== undefined ? `${format(current.q1, 2)} → ${format(current.q3, 2)}` : '—'}</strong></div></div>
          <BranchCharts data={annual.data} companies={eligible} settings={s} stats={stats} onPoint={({ id, year }) => update({ year, ...(id && s.selected.includes(id) ? { focus: id } : {}) })} />
          <section className="branch-chart-card comparison-table"><div className="comparison-section-title"><h2>Selected listings · FY {s.year}</h2><span>{metricUnit(s.metric, s.currency)}</span></div><div className="table-scroll"><table><thead><tr><th>Listing</th><th>{branchMetrics[s.metric]}</th><th>Bubble area</th><th>Financial period</th><th>Source snapshot</th><th /></tr></thead><tbody>{s.selected.map((id, i) => {
            const c = entries.find(c => c.id === id)!, r = annual.data![id]?.find(r => r.year === s.year), o = observation(r, s), inScope = eligibleIds.includes(id);
            return <tr key={id} data-comparison-listing={id} className={s.focus === id ? 'focused' : ''}><td><button className="comparison-listing-link" aria-pressed={s.focus === id} onClick={() => update({ focus: id })}><i style={{ background: comparisonColors[i] }} />{c.display_name}</button><small>{c.ticker ?? id} · {c.listing_country ?? 'ZZ'}</small></td><td data-comparison-value={inScope ? o.value ?? '' : ''}>{format(inScope ? o.value : null, 2)}<small>{!inScope ? 'Outside current directory filters' : o.reason || (s.metric === 'market_cap' && r?.market?.price_date ? `Valued ${r.market.price_date}` : '')}</small></td><td>{s.size === 'equal' ? 'Equal size' : format(inScope ? o.size : null, 2)}{s.size !== 'equal' && <small>{o.sizeReason || `${s.size === 'market_cap' ? 'SEK' : s.currency} million`}{s.size === 'market_cap' && r?.market?.price_date ? ` · valued ${r.market.price_date}` : ''}</small>}</td><td>{r ? `${r.start} → ${r.end}` : '—'}<small>{r?.currency ?? ''}</small></td><td>{r?.source_as_of ?? '—'}<small>Published {r?.report_date ?? 'unknown'}</small></td><td><button aria-label={`Open financials for ${c.display_name}`} onClick={() => onCompany(id)}><ArrowRight size={16} /></button></td></tr>;
          })}</tbody></table></div>{!s.selected.length && <p className="comparison-empty">Choose listings from the comparison panel. The branch benchmark remains available.</p>}</section>
          <details className="comparison-method"><summary>Definitions, annual coverage and source versions</summary><p>{metricDefinition(s.metric)}</p><p>Market cap uses reported shares × the first valid close within 30 days after publication, with observed FX at or up to 7 days before that close. Review flags withhold uncertain values. Valuation dates can fall in the following calendar year. Each listing is separate.</p><p>Annual periods must span 330–400 days. The X-axis uses Börsdata fiscal-year labels; year-end dates can differ. Use the closing-month filter and inspect exact dates in the table.</p><p>The benchmark gives one vote to each valid listing in this saved branch directory. Cross-listings may repeat an issuer. Historical branch memberships and delisted businesses are not fully reconstructed, so this is not a survivorship-free historical universe.</p><p>Older years may use restated reports. This view does not reconstruct what investors knew at the time. No forecasts, inflation adjustments or investment ratings are inferred. The return-on-capital measure is an annual pre-tax proxy using year-end capital, not ROIC. Investing cash flow is not CAPEX.</p><p>Börsdata FCF retains its provider definition and excludes lease principal and interest; it is not owner earnings. Percentages require a positive denominator. Missing observations stay blank.</p><p>Report sources saved {financial.sources.map(source => source.as_of).filter((v, i, a) => a.indexOf(v) === i).join(' / ')}. Median and interpolated quartiles use every valid listing within the visible filters; quartiles are shown only when at least four observations are available.</p><p className="hash">Research {release.id}<br />Financial data {financial.id}<br />Company directory {release.taxonomy_sha256}</p><div className="table-scroll"><table><thead><tr><th>Fiscal year</th><th>Valid / filtered listings</th><th>Median</th><th>25th percentile</th><th>75th percentile</th></tr></thead><tbody>{stats.map(r => <tr key={r.year} data-benchmark-year={r.year}><td><button onClick={() => update({ year: r.year })}>{r.year}</button></td><td>{r.n} / {r.total}</td><td>{format(r.median, 2)}</td><td>{format(r.q1, 2)}</td><td>{format(r.q3, 2)}</td></tr>)}</tbody></table></div></details>
        </>}
      </div><aside className="comparison-panel" aria-label="Comparison selection and saved views">
        <section><div className="eyebrow">YOUR COMPARISON</div><h2>{s.selected.length} of 8 listings</h2><p>Selection controls the coloured series. The grey benchmark uses the whole filtered branch.</p><div className="comparison-chips">{s.selected.map((id, i) => {
          const c = entries.find(c => c.id === id)!;
          return <div key={id} className={s.focus === id ? 'focused' : ''}><button aria-label={`Focus ${c.display_name}`} onClick={() => update({ focus: id })}><i style={{ background: comparisonColors[i] }} /><span>{c.display_name}<small>{eligibleIds.includes(id) ? c.ticker ?? id : 'Outside current filters'}</small></span></button><button aria-label={`Remove ${c.display_name} from comparison`} onClick={() => toggle(id)}><X size={13} /></button></div>;
        })}</div><label>Find a listing<input aria-label="Find branch comparison listings" placeholder="Name, ticker or Börsdata ID" value={query} onChange={e => setQuery(e.target.value)} /></label><div className="comparison-candidates">{found.slice(0, 15).map(c => <button key={c.id} data-add-comparison={c.id} disabled={s.selected.length >= 8} onClick={() => toggle(c.id)}><span>{c.display_name}<small>{c.ticker ?? c.id} · {c.listing_country ?? 'ZZ'} · {c.financial?.annual.count ?? 0} annual reports</small></span><span>+</span></button>)}</div><p>{found.length.toLocaleString('en-US')} other listings match these filters. Search to narrow the list.</p></section>
        <section><div className="eyebrow">KEEP YOUR RESEARCH</div><h2>Save this comparison</h2><label>Name<input aria-label="Saved comparison name" maxLength={120} value={title} onChange={e => { setTitle(e.target.value); setMessage(''); }} /></label><label>Research notes<textarea aria-label="Comparison research notes" maxLength={10000} rows={4} value={notes} onChange={e => { setNotes(e.target.value); setMessage(''); }} placeholder="Questions, observations and sources to investigate…" /></label><button className="primary" disabled={!title.trim() || !annual.data || !!annual.error} onClick={persist}><Save size={14} />Save new comparison</button><p>Save keeps the listings, measures, filters, years and notes with this exact data version. Available offline after restarting Atlas.</p>{message && <p role="status" className="comparison-save-status">{message}</p>}{saveError && <p role="alert">{saveError}</p>}</section>
        <section><h2>Saved in this branch</h2><div className="saved-comparisons">{saved.filter(v => v.branch === branchId).map(v => {
          const matches = compatibleComparison(v, release.id, financial.id, release.taxonomy_sha256 ?? '', ids);
          return <div key={v.id} data-saved-comparison={v.id}><button disabled={!matches} onClick={() => openSaved(v)}><strong>{v.title}</strong><span>{v.created.slice(0, 10)} · {v.settings.selected.length} listings · FY {v.settings.from}–{v.settings.to}</span><small>{matches ? 'Open with matching data' : 'Requires its original release, financial pack and listings'}</small></button><button aria-label={`Delete saved comparison ${v.title}`} onClick={() => { try { removeComparison(localStorage, v.id); setSaved(loadComparisons(localStorage)); } catch (e) { setSaveError(String(e)); } }}><X size={13} /></button></div>;
        })}</div>{!saved.some(v => v.branch === branchId) && <p>No saved comparisons in this branch yet.</p>}</section>
      </aside></div>
    </>}
  </main>;
}
